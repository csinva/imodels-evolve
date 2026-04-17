"""
Interpretable regressor autoresearch script.
Defines a scikit-learn compatible interpretable regressor and evaluates it
on interpretability tests and TabArena regression datasets (same suite used
for baselines in run_baselines.py).

Usage: uv run model.py
"""

import csv
import os
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class CompactSparseRuleLinearRegressor(BaseEstimator, RegressorMixin):
    """Sparse linear backbone with a tiny additive set of single-feature stump rules."""

    def __init__(
        self,
        max_linear_features=5,
        max_rules=4,
        candidate_quantiles=(0.15, 0.3, 0.5, 0.7, 0.85),
        top_rule_features=10,
        ridge_lambda=5e-2,
        rule_shrinkage=0.9,
        min_rel_gain=0.005,
        min_leaf_frac=0.08,
        negligible_feature_eps=5e-3,
    ):
        self.max_linear_features = max_linear_features
        self.max_rules = max_rules
        self.candidate_quantiles = candidate_quantiles
        self.top_rule_features = top_rule_features
        self.ridge_lambda = ridge_lambda
        self.rule_shrinkage = rule_shrinkage
        self.min_rel_gain = min_rel_gain
        self.min_leaf_frac = min_leaf_frac
        self.negligible_feature_eps = negligible_feature_eps

    def _safe_abs_corr(self, a, b):
        a_std = float(np.std(a))
        b_std = float(np.std(b))
        if a_std < 1e-12 or b_std < 1e-12:
            return 0.0
        c = np.corrcoef(a, b)[0, 1]
        if not np.isfinite(c):
            return 0.0
        return float(abs(c))

    def _fit_sparse_ridge_linear(self, X, y, feature_ids):
        if len(feature_ids) == 0:
            return float(np.mean(y)), np.zeros(X.shape[1], dtype=float)

        Xs = X[:, feature_ids]
        x_mean = np.mean(Xs, axis=0)
        x_std = np.std(Xs, axis=0)
        x_std[x_std < 1e-12] = 1.0
        Z = (Xs - x_mean) / x_std

        y_mean = float(np.mean(y))
        y_c = y - y_mean

        gram = Z.T @ Z + self.ridge_lambda * np.eye(Z.shape[1], dtype=float)
        coef_std = np.linalg.solve(gram, Z.T @ y_c)
        coef_sel = coef_std / x_std
        intercept = float(y_mean - np.dot(coef_sel, x_mean))

        coef_full = np.zeros(X.shape[1], dtype=float)
        coef_full[np.asarray(feature_ids, dtype=int)] = coef_sel
        return intercept, coef_full

    def _predict_rules(self, X, rules):
        if len(rules) == 0:
            return np.zeros(X.shape[0], dtype=float)
        out = np.zeros(X.shape[0], dtype=float)
        for rule in rules:
            feat = rule["feature"]
            thr = rule["threshold"]
            left_val = rule["left_value"]
            right_val = rule["right_value"]
            out += np.where(X[:, feat] <= thr, left_val, right_val)
        return out

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        corr = np.array([self._safe_abs_corr(X[:, j], y) for j in range(n_features)], dtype=float)
        ranked = list(np.argsort(corr)[::-1].astype(int))

        linear_feats = ranked[: min(self.max_linear_features, n_features)]
        self.intercept_, self.linear_coef_ = self._fit_sparse_ridge_linear(X, y, linear_feats)

        pred = self.intercept_ + X @ self.linear_coef_
        candidate_rule_feats = ranked[: min(self.top_rule_features, n_features)]
        min_leaf = max(6, int(self.min_leaf_frac * n_samples))

        rules = []
        for _ in range(self.max_rules):
            residual = y - pred
            base_sse = float(np.dot(residual, residual))
            min_gain_abs = self.min_rel_gain * (base_sse + 1e-12)
            best = None

            for feat in candidate_rule_feats:
                xj = X[:, feat]
                thr_vals = np.unique(np.round(np.quantile(xj, self.candidate_quantiles), 8))
                for thr in thr_vals:
                    left = xj <= thr
                    n_left = int(np.sum(left))
                    n_right = n_samples - n_left
                    if n_left < min_leaf or n_right < min_leaf:
                        continue
                    left_val = float(np.mean(residual[left])) * self.rule_shrinkage
                    right_val = float(np.mean(residual[~left])) * self.rule_shrinkage
                    update = np.where(left, left_val, right_val)
                    new_residual = residual - update
                    sse = float(np.dot(new_residual, new_residual))
                    gain = base_sse - sse
                    if gain > min_gain_abs and (best is None or sse < best["sse"]):
                        best = {
                            "feature": int(feat),
                            "threshold": float(thr),
                            "left_value": float(left_val),
                            "right_value": float(right_val),
                            "update": update,
                            "sse": sse,
                        }

            if best is None:
                break
            rules.append({
                "feature": best["feature"],
                "threshold": best["threshold"],
                "left_value": best["left_value"],
                "right_value": best["right_value"],
            })
            pred = pred + best["update"]

        self.rules_ = rules

        self.feature_importances_ = np.zeros(n_features, dtype=float)
        linear_contrib = np.abs(self.linear_coef_ * X).mean(axis=0)
        self.feature_importances_ += linear_contrib
        for rule in self.rules_:
            feat = rule["feature"]
            contrib = np.mean(np.abs(np.where(X[:, feat] <= rule["threshold"], rule["left_value"], rule["right_value"])))
            self.feature_importances_[feat] += float(contrib)
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "linear_coef_", "rules_"])
        X = np.asarray(X, dtype=float)
        return self.intercept_ + X @ self.linear_coef_ + self._predict_rules(X, self.rules_)

    def __str__(self):
        check_is_fitted(self, ["intercept_", "linear_coef_", "rules_", "feature_importances_"])
        active_linear = [j for j in range(self.n_features_in_) if abs(self.linear_coef_[j]) > 1e-10]
        active_rule_feats = sorted({r["feature"] for r in self.rules_})
        active_all = sorted(set(active_linear) | set(active_rule_feats))

        lines = [
            "Compact Sparse Rule-Linear Regressor",
            "Model form: y = intercept + sparse linear terms + additive one-feature stump rules",
            f"Intercept: {self.intercept_:+.5f}",
            f"Active linear features ({len(active_linear)}): "
            + (", ".join(f"x{j}" for j in active_linear) if active_linear else "none"),
            f"Rule features ({len(active_rule_feats)}): "
            + (", ".join(f"x{j}" for j in active_rule_feats) if active_rule_feats else "none"),
            "",
            "Linear coefficients:",
        ]
        if active_linear:
            for j in active_linear:
                lines.append(f"  x{j}: {self.linear_coef_[j]:+.5f}")
        else:
            lines.append("  (no linear terms)")

        lines.append("")
        lines.append("Rules (add this contribution):")
        if self.rules_:
            for i, rule in enumerate(self.rules_, 1):
                lines.append(
                    f"  rule{i}: if x{rule['feature']} <= {rule['threshold']:.5f} add {rule['left_value']:+.5f}, "
                    f"else add {rule['right_value']:+.5f}"
                )
        else:
            lines.append("  (no rules)")

        equation_parts = [f"{self.intercept_:+.5f}"]
        for j in active_linear:
            equation_parts.append(f"{self.linear_coef_[j]:+.5f}*x{j}")
        for i, rule in enumerate(self.rules_, 1):
            equation_parts.append(
                f"rule{i}(x{rule['feature']})"
            )

        order = np.argsort(self.feature_importances_)[::-1]
        top = [f"x{j}:{self.feature_importances_[j]:.4f}" for j in order[: min(10, self.n_features_in_)]]
        negligible = [f"x{j}" for j in range(self.n_features_in_) if self.feature_importances_[j] <= self.negligible_feature_eps]

        lines.extend([
            "",
            "Compact equation:",
            "  y = " + " ".join(equation_parts),
            "",
            "Most influential features (mean absolute contribution):",
            "  " + ", ".join(top),
            "Features with negligible effect:",
            "  " + (", ".join(negligible) if negligible else "none"),
            "",
            "Simulation recipe: compute intercept + linear terms, then add each rule contribution.",
            "Total terms to evaluate: "
            + str(1 + len(active_linear) + len(self.rules_))
            + " (intercept + linear + rules).",
        ])
        if active_all:
            lines.append("Active feature set for simulation: " + ", ".join(f"x{j}" for j in active_all))
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
CompactSparseRuleLinearRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "CompactRuleLinear_v1"
model_description = "Sparse ridge linear backbone plus a tiny residual additive stump-rule set with calculator-friendly simulation string"
model_defs = [(model_shorthand_name, CompactSparseRuleLinearRegressor())]


# ---------------------------------------------------------------------------
# Evaluation (do not edit anything below this line)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()

    # Interpretability tests
    interp_results = run_all_interp_tests(model_defs)
    n_passed = sum(r["passed"] for r in interp_results)
    total = len(interp_results)

    # prediction performance (RMSE)
    dataset_rmses = evaluate_all_regressors(model_defs)

    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        git_hash = ""

    # --- Upsert interpretability_results.csv ---
    model_name = model_defs[0][0]
    interp_csv = os.path.join(RESULTS_DIR, "interpretability_results.csv")
    interp_fields = ["model", "test", "suite", "passed", "ground_truth", "response"]

    def _suite(test_name):
        if test_name.startswith("insight_"): return "insight"
        if test_name.startswith("hard_"):    return "hard"
        return "standard"

    # Load existing rows, dropping old rows for this model
    existing_interp = []
    if os.path.exists(interp_csv):
        with open(interp_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("model") != model_name:
                    existing_interp.append(row)

    new_interp = [{
        "model": r["model"],
        "test": r["test"],
        "suite": _suite(r["test"]),
        "passed": r["passed"],
        "ground_truth": r.get("ground_truth", ""),
        "response": r.get("response", ""),
    } for r in interp_results]

    with open(interp_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=interp_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing_interp + new_interp)
    print(f"Interpretability results saved → {interp_csv}")

    # --- Upsert performance_results.csv and recompute ranks ---
    perf_csv = os.path.join(RESULTS_DIR, "performance_results.csv")
    perf_fields = ["dataset", "model", "rmse", "rank"]

    # Load existing rows, dropping old rows for this model
    existing_perf = []
    if os.path.exists(perf_csv):
        with open(perf_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("model") != model_name:
                    existing_perf.append(row)

    # Add new rows (without rank for now)
    for ds_name, model_rmses in dataset_rmses.items():
        rmse_val = model_rmses.get(model_name, float("nan"))
        existing_perf.append({
            "dataset": ds_name,
            "model": model_name,
            "rmse": "" if np.isnan(rmse_val) else f"{rmse_val:.6f}",
            "rank": "",
        })

    # Recompute ranks per dataset
    by_dataset = defaultdict(list)
    for row in existing_perf:
        by_dataset[row["dataset"]].append(row)

    for ds_name, rows in by_dataset.items():
        valid = [(r, float(r["rmse"])) for r in rows if r["rmse"] not in ("", None)]
        valid.sort(key=lambda x: x[1])
        for rank_idx, (r, _) in enumerate(valid, 1):
            r["rank"] = rank_idx
        # Leave rank empty for rows with no RMSE
        for r in rows:
            if r["rmse"] in ("", None):
                r["rank"] = ""

    with open(perf_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=perf_fields)
        writer.writeheader()
        for ds_name in by_dataset:
            for row in by_dataset[ds_name]:
                writer.writerow(row)
    print(f"Performance results saved → {perf_csv}")

    # --- Compute mean_rank from the updated performance_results.csv ---
    # Build dataset_rmses dict with all models from the CSV for ranking
    all_dataset_rmses = defaultdict(dict)
    for row in existing_perf:
        rmse_str = row.get("rmse", "")
        if rmse_str not in ("", None):
            all_dataset_rmses[row["dataset"]][row["model"]] = float(rmse_str)
        else:
            all_dataset_rmses[row["dataset"]][row["model"]] = float("nan")
    avg_rank, _ = compute_rank_scores(dict(all_dataset_rmses))
    mean_rank = avg_rank.get(model_shorthand_name, float("nan"))

    upsert_overall_results([{
        "commit":                             git_hash,
        "mean_rank":                          f"{mean_rank:.2f}" if not np.isnan(mean_rank) else "nan",
        "frac_interpretability_tests_passed": f"{n_passed / total:.4f}" if total > 0 else "nan",
        "status":                             "",
        "model_name":                         model_shorthand_name,
        "description":                        model_description,
    }], RESULTS_DIR)

    # --- Plot ---
    overall_csv = os.path.join(RESULTS_DIR, "overall_results.csv")
    plot_interp_vs_performance(
        overall_csv,
        os.path.join(RESULTS_DIR, "interpretability_vs_performance.png"),
    )

    print()
    print("---")
    print(f"tests_passed:  {n_passed}/{total}" + (f" ({n_passed/total:.2%})" if total > 0 else ""))
    print(f"mean_rank:     {mean_rank:.2f}" if not np.isnan(mean_rank) else "mean_rank:     nan")
    print(f"total_seconds: {time.time() - t0:.1f}s")
