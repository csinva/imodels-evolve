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


class GatedSparseLinearTreeRegressor(BaseEstimator, RegressorMixin):
    """
    Single-split model tree with sparse linear equations in each region.

    y = f_left(x) if x_gate <= threshold else f_right(x)

    Each leaf equation is a compact linear model over a shared sparse feature set,
    chosen by correlation screening and fit with ridge-stabilized least squares.
    """

    def __init__(
        self,
        top_features=10,
        top_gate_features=6,
        threshold_quantiles=(0.2, 0.35, 0.5, 0.65, 0.8),
        l2=1e-3,
        min_leaf_frac=0.15,
        min_split_gain=0.01,
        coef_keep_ratio=0.08,
    ):
        self.top_features = top_features
        self.top_gate_features = top_gate_features
        self.threshold_quantiles = threshold_quantiles
        self.l2 = l2
        self.min_leaf_frac = min_leaf_frac
        self.min_split_gain = min_split_gain
        self.coef_keep_ratio = coef_keep_ratio

    @staticmethod
    def _corr_abs(x, y):
        xc = x - np.mean(x)
        yc = y - np.mean(y)
        denom = (np.linalg.norm(xc) + 1e-12) * (np.linalg.norm(yc) + 1e-12)
        return float(abs(np.dot(xc, yc) / denom))

    def _fit_ridge_subset(self, X, y, feat_idx):
        n = X.shape[0]
        if len(feat_idx) == 0:
            intercept = float(np.mean(y)) if n > 0 else 0.0
            return intercept, np.zeros(0, dtype=float)

        Z = X[:, feat_idx]
        D = np.column_stack([np.ones(n, dtype=float), Z])
        reg = float(self.l2) * np.eye(D.shape[1])
        reg[0, 0] = 0.0  # do not penalize intercept
        A = D.T @ D + reg
        b = D.T @ y
        beta = np.linalg.solve(A, b)
        return float(beta[0]), np.asarray(beta[1:], dtype=float)

    def _predict_linear(self, X, feat_idx, intercept, coef):
        if len(feat_idx) == 0:
            return np.full(X.shape[0], intercept, dtype=float)
        return intercept + X[:, feat_idx] @ coef

    def _prune_coefs(self, coef):
        if coef.size == 0:
            return coef
        m = float(np.max(np.abs(coef)))
        if m <= 1e-12:
            return np.zeros_like(coef)
        keep = np.abs(coef) >= float(self.coef_keep_ratio) * m
        if np.sum(keep) == 0:
            keep[np.argmax(np.abs(coef))] = True
        out = coef.copy()
        out[~keep] = 0.0
        return out

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape

        corr = np.array([self._corr_abs(X[:, j], y) for j in range(p)], dtype=float)
        k_feat = min(max(1, int(self.top_features)), p)
        feat_idx = np.argsort(-corr)[:k_feat].astype(int)

        base_intercept, base_coef = self._fit_ridge_subset(X, y, feat_idx)
        base_coef = self._prune_coefs(base_coef)
        base_pred = self._predict_linear(X, feat_idx, base_intercept, base_coef)
        base_rss = float(np.sum((y - base_pred) ** 2))

        k_gate = min(max(1, int(self.top_gate_features)), p)
        gate_pool = np.argsort(-corr)[:k_gate].astype(int)
        min_leaf = max(3, int(np.ceil(float(self.min_leaf_frac) * n)))

        best = None
        for gj in gate_pool:
            col = X[:, gj]
            thr_vals = sorted({float(np.quantile(col, q)) for q in self.threshold_quantiles})
            for thr in thr_vals:
                left = col <= thr
                n_left = int(np.sum(left))
                n_right = n - n_left
                if n_left < min_leaf or n_right < min_leaf:
                    continue

                li, lc = self._fit_ridge_subset(X[left], y[left], feat_idx)
                ri, rc = self._fit_ridge_subset(X[~left], y[~left], feat_idx)

                lc = self._prune_coefs(lc)
                rc = self._prune_coefs(rc)

                pred = np.empty(n, dtype=float)
                pred[left] = self._predict_linear(X[left], feat_idx, li, lc)
                pred[~left] = self._predict_linear(X[~left], feat_idx, ri, rc)
                rss = float(np.sum((y - pred) ** 2))

                if (best is None) or (rss < best["rss"]):
                    best = {
                        "gate_j": int(gj),
                        "thr": float(thr),
                        "left_i": float(li),
                        "left_c": lc,
                        "right_i": float(ri),
                        "right_c": rc,
                        "rss": rss,
                    }

        use_split = False
        if best is not None:
            rel_gain = (base_rss - best["rss"]) / (base_rss + 1e-12)
            use_split = rel_gain >= float(self.min_split_gain)

        self.feature_idx_ = feat_idx
        self.global_intercept_ = float(base_intercept)
        self.global_coef_ = np.asarray(base_coef, dtype=float)

        self.use_split_ = bool(use_split)
        if self.use_split_:
            self.gate_feature_ = int(best["gate_j"])
            self.threshold_ = float(best["thr"])
            self.left_intercept_ = float(best["left_i"])
            self.left_coef_ = np.asarray(best["left_c"], dtype=float)
            self.right_intercept_ = float(best["right_i"])
            self.right_coef_ = np.asarray(best["right_c"], dtype=float)
            self.rel_gain_ = float((base_rss - best["rss"]) / (base_rss + 1e-12))
        else:
            self.gate_feature_ = -1
            self.threshold_ = 0.0
            self.left_intercept_ = float(base_intercept)
            self.left_coef_ = np.asarray(base_coef, dtype=float)
            self.right_intercept_ = float(base_intercept)
            self.right_coef_ = np.asarray(base_coef, dtype=float)
            self.rel_gain_ = 0.0

        self.n_features_in_ = p
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        X = np.asarray(X, dtype=float)

        if not self.use_split_:
            return self._predict_linear(X, self.feature_idx_, self.global_intercept_, self.global_coef_)

        mask = X[:, self.gate_feature_] <= self.threshold_
        yhat = np.empty(X.shape[0], dtype=float)
        yhat[mask] = self._predict_linear(X[mask], self.feature_idx_, self.left_intercept_, self.left_coef_)
        yhat[~mask] = self._predict_linear(X[~mask], self.feature_idx_, self.right_intercept_, self.right_coef_)
        return yhat

    def _equation_text(self, intercept, coef):
        pieces = [f"{intercept:+.4f}"]
        for j, c in zip(self.feature_idx_, coef):
            if abs(c) <= 1e-12:
                continue
            pieces.append(f"{c:+.4f}*x{int(j)}")
        return " ".join(pieces)

    def __str__(self):
        check_is_fitted(self, "is_fitted_")

        lines = [
            "Gated Sparse Linear Tree Regressor:",
            "  Structure: one threshold split with a sparse linear equation per side",
            f"  Shared candidate features: {', '.join(f'x{int(j)}' for j in self.feature_idx_)}",
        ]

        if not self.use_split_:
            lines.append("  Split selected: no (global sparse linear equation)"
            )
            lines.append(f"  y = {self._equation_text(self.global_intercept_, self.global_coef_)}")
            return "\n".join(lines)

        lines.append(
            f"  Split selected: yes (gain={self.rel_gain_:.2%}) on x{self.gate_feature_} at threshold {self.threshold_:+.4f}"
        )
        lines.append(f"  If x{self.gate_feature_} <= {self.threshold_:+.4f}:  y = {self._equation_text(self.left_intercept_, self.left_coef_)}")
        lines.append(f"  If x{self.gate_feature_} >  {self.threshold_:+.4f}:  y = {self._equation_text(self.right_intercept_, self.right_coef_)}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
GatedSparseLinearTreeRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "GateLinTree_v1"
model_description = "Single-threshold sparse model tree with shared screened features and ridge-fit linear equations in each branch"
model_defs = [(model_shorthand_name, GatedSparseLinearTreeRegressor())]


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
