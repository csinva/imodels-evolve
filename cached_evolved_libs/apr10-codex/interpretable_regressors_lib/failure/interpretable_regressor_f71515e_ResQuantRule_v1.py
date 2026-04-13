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


class ResidualQuantileRuleRegressor(BaseEstimator, RegressorMixin):
    """
    Small residual rule ensemble:
    - Add up to `max_rules` one-feature quantile threshold rules.
    - Each rule predicts one value on each side of the threshold.
    - Refit rule weights jointly with ridge to stabilize coefficients.
    """

    def __init__(
        self,
        max_rules=8,
        screening_features=14,
        quantiles=(0.2, 0.35, 0.5, 0.65, 0.8),
        learning_rate=0.7,
        min_leaf_frac=0.08,
        min_rel_rss_gain=0.006,
        l2=2e-2,
        coef_keep_ratio=0.08,
    ):
        self.max_rules = max_rules
        self.screening_features = screening_features
        self.quantiles = quantiles
        self.learning_rate = learning_rate
        self.min_leaf_frac = min_leaf_frac
        self.min_rel_rss_gain = min_rel_rss_gain
        self.l2 = l2
        self.coef_keep_ratio = coef_keep_ratio

    @staticmethod
    def _corr_abs(x, y):
        xc = x - np.mean(x)
        yc = y - np.mean(y)
        denom = (np.linalg.norm(xc) + 1e-12) * (np.linalg.norm(yc) + 1e-12)
        return float(abs(np.dot(xc, yc) / denom))

    def _fit_ridge(self, M, y):
        n, d = M.shape
        D = np.column_stack([np.ones(n, dtype=float), M])
        reg = float(self.l2) * np.eye(d + 1)
        reg[0, 0] = 0.0
        A = D.T @ D + reg
        b = D.T @ y
        beta = np.linalg.solve(A, b)
        return float(beta[0]), np.asarray(beta[1:], dtype=float)

    def _prune_weights(self, w):
        if w.size == 0:
            return w
        m = float(np.max(np.abs(w)))
        if m <= 1e-12:
            return np.zeros_like(w)
        keep = np.abs(w) >= float(self.coef_keep_ratio) * m
        if np.sum(keep) == 0:
            keep[np.argmax(np.abs(w))] = True
        out = w.copy()
        out[~keep] = 0.0
        return out

    @staticmethod
    def _rule_response(xj, threshold, left_value, right_value):
        return np.where(xj <= threshold, left_value, right_value)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape

        intercept = float(np.mean(y))
        pred = np.full(n, intercept, dtype=float)
        rules = []
        rule_cols = []
        current_rss = float(np.sum((y - pred) ** 2))
        min_leaf = max(2, int(np.ceil(float(self.min_leaf_frac) * n)))
        q_list = tuple(float(q) for q in self.quantiles)

        for _ in range(max(0, int(self.max_rules))):
            residual = y - pred
            screening = np.array([self._corr_abs(X[:, j], residual) for j in range(p)], dtype=float)
            k = min(max(1, int(self.screening_features)), p)
            feat_order = np.argsort(-screening)[:k]

            best = None
            best_rss = current_rss
            for j in feat_order:
                xj = X[:, int(j)]
                thresholds = np.unique(np.quantile(xj, q_list))
                for threshold in thresholds:
                    left_mask = xj <= threshold
                    n_left = int(np.sum(left_mask))
                    n_right = n - n_left
                    if n_left < min_leaf or n_right < min_leaf:
                        continue
                    left_val = float(np.mean(residual[left_mask]))
                    right_val = float(np.mean(residual[~left_mask]))
                    response = self._rule_response(xj, threshold, left_val, right_val)
                    trial_pred = pred + float(self.learning_rate) * response
                    trial_rss = float(np.sum((y - trial_pred) ** 2))
                    if trial_rss < best_rss:
                        best_rss = trial_rss
                        best = {
                            "feature": int(j),
                            "threshold": float(threshold),
                            "left": left_val,
                            "right": right_val,
                            "response": response,
                        }

            if best is None:
                break

            rel_gain = (current_rss - best_rss) / (current_rss + 1e-12)
            if rel_gain < float(self.min_rel_rss_gain):
                break

            rules.append({
                "feature": best["feature"],
                "threshold": best["threshold"],
                "left": best["left"],
                "right": best["right"],
            })
            col = float(self.learning_rate) * best["response"]
            rule_cols.append(col)
            pred = pred + col
            current_rss = best_rss

        if len(rule_cols) == 0:
            self.intercept_ = intercept
            self.rule_weights_ = np.zeros(0, dtype=float)
            self.rules_ = []
            self.n_features_in_ = p
            self.is_fitted_ = True
            return self

        M = np.column_stack(rule_cols)
        ri, rw = self._fit_ridge(M, y)
        rw = self._prune_weights(rw)
        keep = np.abs(rw) > 0
        self.rules_ = [r for r, k in zip(rules, keep) if k]
        self.rule_weights_ = rw[keep]
        self.intercept_ = float(ri)
        self.n_features_in_ = p
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        X = np.asarray(X, dtype=float)
        yhat = np.full(X.shape[0], self.intercept_, dtype=float)
        for w, rule in zip(self.rule_weights_, self.rules_):
            response = self._rule_response(
                X[:, rule["feature"]],
                rule["threshold"],
                rule["left"],
                rule["right"],
            )
            yhat += float(w) * response
        return yhat

    def __str__(self):
        check_is_fitted(self, "is_fitted_")
        lines = ["Residual Quantile Rule Regressor:"]
        lines.append(f"  Base value: {self.intercept_:+.4f}")
        if len(self.rules_) == 0:
            lines.append("  Rules: none")
            return "\n".join(lines)
        lines.append("  Prediction = base + sum(weight_k * rule_k(x))")
        for i, (w, r) in enumerate(zip(self.rule_weights_, self.rules_), 1):
            lines.append(
                f"  Rule {i}: weight={w:+.4f}, if x{r['feature']} <= {r['threshold']:.4f} "
                f"then {r['left']:+.4f} else {r['right']:+.4f}"
            )
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
ResidualQuantileRuleRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "ResQuantRule_v1"
model_description = "Residual quantile-threshold rule ensemble with screened one-feature stumps and ridge-refit rule weights"
model_defs = [(model_shorthand_name, ResidualQuantileRuleRegressor())]


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

    std_tests = {t.__name__ for t in ALL_TESTS}
    hard_tests = {t.__name__ for t in HARD_TESTS}
    insight_tests = {t.__name__ for t in INSIGHT_TESTS}
    std_passed = sum(r["passed"] for r in interp_results if r["test"] in std_tests)
    hard_passed = sum(r["passed"] for r in interp_results if r["test"] in hard_tests)
    insight_passed = sum(r["passed"] for r in interp_results if r["test"] in insight_tests)
    print(f"[std {std_passed}/{len(std_tests)}  hard {hard_passed}/{len(hard_tests)}  insight {insight_passed}/{len(insight_tests)}]")
    print(f"total_seconds: {time.time() - t0:.1f}s")
