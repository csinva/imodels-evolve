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


class SparseLinearRuleGateRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse linear backbone with a small number of threshold gate rules.

    Model form:
      y = intercept + sum_j w_j * x_j + sum_m a_m * I(condition_m)

    where each condition_m is either (x_j > t) or (x_j <= t).
    """

    def __init__(
        self,
        screen_features=10,
        max_rules=2,
        thresholds_per_feature=5,
        ridge_lambda=5e-2,
        rule_lambda=1e-4,
        min_rel_gain=2e-3,
    ):
        self.screen_features = screen_features
        self.max_rules = max_rules
        self.thresholds_per_feature = thresholds_per_feature
        self.ridge_lambda = ridge_lambda
        self.rule_lambda = rule_lambda
        self.min_rel_gain = min_rel_gain

    @staticmethod
    def _safe_scale(X):
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
        sigma = np.where(sigma < 1e-8, 1.0, sigma)
        return mu, sigma

    @staticmethod
    def _ridge_joint_fit(X_lin, Z, y, lam_lin, lam_rule):
        n = y.shape[0]
        p = X_lin.shape[1]
        r = Z.shape[1] if Z is not None else 0

        if r == 0:
            D = np.hstack([np.ones((n, 1), dtype=float), X_lin])
        else:
            D = np.hstack([np.ones((n, 1), dtype=float), X_lin, Z])

        q = D.shape[1]
        reg = np.zeros(q, dtype=float)
        reg[1 : 1 + p] = max(lam_lin, 0.0)
        if r > 0:
            reg[1 + p :] = max(lam_rule, 0.0)

        beta = np.linalg.solve(D.T @ D + np.diag(reg), D.T @ y)
        intercept = float(beta[0])
        w_lin = np.asarray(beta[1 : 1 + p], dtype=float)
        w_rule = np.asarray(beta[1 + p :], dtype=float) if r > 0 else np.zeros(0, dtype=float)
        pred = D @ beta
        return intercept, w_lin, w_rule, pred

    def _rule_col(self, X, rule):
        xj = X[:, rule["j"]]
        if rule["op"] == ">":
            return (xj > rule["thr"]).astype(float)
        return (xj <= rule["thr"]).astype(float)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape
        self.n_features_in_ = p

        y0 = y - np.mean(y)
        corr = np.abs(X.T @ y0)
        keep = min(p, max(1, int(self.screen_features)))
        feat_idx = np.argsort(corr)[-keep:]
        feat_idx = np.sort(feat_idx.astype(int))
        self.active_features_ = feat_idx

        Xk_raw = X[:, feat_idx]
        mu, sigma = self._safe_scale(Xk_raw)
        Xk = (Xk_raw - mu) / sigma

        selected_rules = []
        intercept, w_std, _, pred = self._ridge_joint_fit(
            Xk,
            np.zeros((n, 0), dtype=float),
            y,
            self.ridge_lambda,
            self.rule_lambda,
        )
        current_mse = float(np.mean((y - pred) ** 2))

        candidate_rules = []
        for j in feat_idx:
            xj = X[:, j]
            q = np.linspace(0.15, 0.85, max(2, int(self.thresholds_per_feature)))
            thrs = np.unique(np.quantile(xj, q))
            for thr in thrs:
                candidate_rules.append({"j": int(j), "op": ">", "thr": float(thr)})
                candidate_rules.append({"j": int(j), "op": "<=", "thr": float(thr)})

        used = set()
        for _ in range(max(0, int(self.max_rules))):
            residual = y - pred
            best_rule = None
            best_score = -np.inf

            for ridx, rule in enumerate(candidate_rules):
                if ridx in used:
                    continue
                z = self._rule_col(X, rule)
                zc = z - np.mean(z)
                denom = float(np.sqrt(np.dot(zc, zc)) + 1e-12)
                if denom <= 1e-10:
                    continue
                score = float(abs(np.dot(zc, residual)) / denom)
                if score > best_score:
                    best_score = score
                    best_rule = (ridx, rule)

            if best_rule is None:
                break

            trial_rules = selected_rules + [best_rule[1]]
            Z = np.column_stack([self._rule_col(X, r) for r in trial_rules])
            t_intercept, t_w_std, t_w_rule, t_pred = self._ridge_joint_fit(
                Xk,
                Z,
                y,
                self.ridge_lambda,
                self.rule_lambda,
            )
            t_mse = float(np.mean((y - t_pred) ** 2))
            rel_gain = (current_mse - t_mse) / max(current_mse, 1e-12)
            if rel_gain < self.min_rel_gain:
                break

            selected_rules = trial_rules
            used.add(best_rule[0])
            intercept, w_std, w_rule, pred = t_intercept, t_w_std, t_w_rule, t_pred
            current_mse = t_mse

        w_raw = w_std / sigma
        intercept_raw = intercept - float(np.dot(w_raw, mu))

        if selected_rules:
            active_rule_mask = np.abs(w_rule) > 5e-4
            selected_rules = [r for r, keep_rule in zip(selected_rules, active_rule_mask) if keep_rule]
            w_rule = w_rule[active_rule_mask]

        self.feature_mu_ = mu
        self.feature_sigma_ = sigma
        self.intercept_ = float(intercept_raw)
        self.linear_coef_active_ = np.asarray(w_raw, dtype=float)
        self.rules_ = selected_rules
        self.rule_coef_ = np.asarray(w_rule if selected_rules else np.zeros(0), dtype=float)
        self.train_mse_ = float(current_mse)
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "active_features_", "linear_coef_active_", "rules_", "rule_coef_"])
        X = np.asarray(X, dtype=float)
        yhat = np.full(X.shape[0], self.intercept_, dtype=float)

        if self.active_features_.size > 0:
            yhat += X[:, self.active_features_] @ self.linear_coef_active_

        for a, rule in zip(self.rule_coef_, self.rules_):
            yhat += a * self._rule_col(X, rule)
        return yhat

    def __str__(self):
        check_is_fitted(self, ["intercept_", "active_features_", "linear_coef_active_", "rules_", "rule_coef_"])
        lines = [
            "Sparse Linear + Rule Gates Regressor",
            "Prediction equation on raw features:",
        ]

        eq_terms = [f"{self.intercept_:.6f}"]
        linear_pairs = sorted(
            zip(self.active_features_, self.linear_coef_active_),
            key=lambda x: -abs(x[1]),
        )
        for j, c in linear_pairs:
            eq_terms.append(f"{c:+.6f}*x{j}")

        rule_pairs = sorted(
            zip(self.rule_coef_, self.rules_),
            key=lambda x: -abs(x[0]),
        )
        for a, r in rule_pairs:
            eq_terms.append(f"{a:+.6f}*I(x{r['j']} {r['op']} {r['thr']:.4f})")

        lines.append("  y = " + " ".join(eq_terms))
        lines.append("")
        lines.append("Linear terms (sorted by |coefficient|):")
        for i, (j, c) in enumerate(linear_pairs, 1):
            lines.append(f"  {i:2d}. {c:+.6f} * x{j}")

        lines.append("")
        lines.append("Rule gate terms (sorted by |coefficient|):")
        if rule_pairs:
            for i, (a, r) in enumerate(rule_pairs, 1):
                lines.append(f"  {i:2d}. {a:+.6f} * I(x{r['j']} {r['op']} {r['thr']:.4f})")
        else:
            lines.append("  none")

        used = set(int(j) for j in self.active_features_)
        used.update(int(r["j"]) for r in self.rules_)
        inactive = [f"x{i}" for i in range(self.n_features_in_) if i not in used]
        lines.append("")
        lines.append("Features with no direct effect:")
        lines.append("  " + (", ".join(inactive) if inactive else "none"))
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparseLinearRuleGateRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SparseLinearRuleGateV1"
model_description = "Sparse screened linear backbone with at most two learned threshold indicator gate terms jointly ridge-refit for compact nonlinear corrections"
model_defs = [(model_shorthand_name, SparseLinearRuleGateRegressor())]


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
