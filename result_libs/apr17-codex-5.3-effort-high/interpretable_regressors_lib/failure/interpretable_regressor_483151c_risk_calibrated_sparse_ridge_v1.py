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


class RiskCalibratedSparseRidgeRegressor(BaseEstimator, RegressorMixin):
    """
    Validation-calibrated sparse ridge in raw-feature equation form.

    Procedure:
    1) Fit a dense ridge model (alpha chosen on validation set).
    2) Build a coefficient-magnitude ordering.
    3) Refit ridge on top-k features for several k values.
    4) Pick k via validation objective = MSE + complexity_penalty * (k / p),
       while always considering the dense fallback (k=p) to preserve performance.
    """

    def __init__(
        self,
        alpha_grid=(0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0),
        holdout_frac=0.2,
        min_keep=1,
        max_keep=12,
        complexity_penalty=0.008,
        max_candidates=14,
        random_state=0,
    ):
        self.alpha_grid = alpha_grid
        self.holdout_frac = holdout_frac
        self.min_keep = min_keep
        self.max_keep = max_keep
        self.complexity_penalty = complexity_penalty
        self.max_candidates = max_candidates
        self.random_state = random_state

    @staticmethod
    def _safe_std(X):
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std = np.where(std > 1e-12, std, 1.0)
        return mean.astype(float), std.astype(float)

    @staticmethod
    def _ridge_fit(D, y, alpha):
        p = D.shape[1]
        A = D.T @ D + float(alpha) * np.eye(p, dtype=float)
        A[0, 0] -= float(alpha)
        b = D.T @ y
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(A) @ b

    @staticmethod
    def _mse(y_true, y_pred):
        d = y_true - y_pred
        return float(np.mean(d * d))

    def _fit_best_alpha(self, Xtr, ytr, Xval, yval):
        mean, std = self._safe_std(Xtr)
        Xtrz = (Xtr - mean) / std
        Xvalz = (Xval - mean) / std
        Dtr = np.column_stack([np.ones(len(Xtrz), dtype=float), Xtrz])
        Dval = np.column_stack([np.ones(len(Xvalz), dtype=float), Xvalz])

        best = None
        for alpha in self.alpha_grid:
            w = self._ridge_fit(Dtr, ytr, float(alpha))
            val_mse = self._mse(yval, Dval @ w)
            if best is None or val_mse < best["val_mse"]:
                coef_z = np.asarray(w[1:], dtype=float)
                coef_raw = coef_z / std
                intercept_raw = float(w[0]) - float(np.dot(coef_raw, mean))
                best = {
                    "alpha": float(alpha),
                    "val_mse": float(val_mse),
                    "coef": coef_raw,
                    "intercept": float(intercept_raw),
                }
        return best

    def _candidate_ks(self, p):
        lo = max(1, int(self.min_keep))
        hi = min(p, int(self.max_keep))
        ks = {p, lo, hi}
        if hi >= lo:
            lin = np.linspace(lo, hi, num=max(3, int(self.max_candidates)))
            ks.update(int(round(v)) for v in lin)
        # Include geometric spacing to probe very compact models.
        g = 1
        while g < p:
            if lo <= g <= hi:
                ks.add(int(g))
            g *= 2
        return sorted(k for k in ks if 1 <= k <= p)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape
        self.n_features_in_ = p

        rng = np.random.RandomState(int(self.random_state))
        n_val = max(20, int(float(self.holdout_frac) * n))
        if n - n_val < 15:
            n_val = max(1, n // 5)
        perm = rng.permutation(n)
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]
        if len(tr_idx) < 12:
            tr_idx = perm
            val_idx = perm[: max(1, min(10, n // 4))]

        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xval, yval = X[val_idx], y[val_idx]

        dense_fit = self._fit_best_alpha(Xtr, ytr, Xval, yval)
        order = np.argsort(-np.abs(dense_fit["coef"]))

        best = None
        for k in self._candidate_ks(p):
            active = np.sort(order[:k].astype(int))
            part = self._fit_best_alpha(Xtr[:, active], ytr, Xval[:, active], yval)
            complexity = float(self.complexity_penalty) * (len(active) / max(1, p))
            objective = part["val_mse"] + complexity
            if best is None or objective < best["objective"]:
                best = {
                    "objective": float(objective),
                    "val_mse": float(part["val_mse"]),
                    "alpha": float(part["alpha"]),
                    "features": active,
                    "intercept": float(part["intercept"]),
                    "coef": np.asarray(part["coef"], dtype=float),
                }

        # Dense fallback if sparse objective did not earn enough real validation gain.
        dense_gain = (dense_fit["val_mse"] - best["val_mse"]) / max(dense_fit["val_mse"], 1e-12)
        use_dense = dense_gain < 0.002

        if use_dense:
            self.active_features_ = np.arange(p, dtype=int)
            self.intercept_ = float(dense_fit["intercept"])
            self.coef_ = np.asarray(dense_fit["coef"], dtype=float)
            self.alpha_ = float(dense_fit["alpha"])
            self.validation_mse_ = float(dense_fit["val_mse"])
            self.objective_ = float(dense_fit["val_mse"]) + float(self.complexity_penalty)
        else:
            self.active_features_ = np.asarray(best["features"], dtype=int)
            self.intercept_ = float(best["intercept"])
            self.coef_ = np.asarray(best["coef"], dtype=float)
            self.alpha_ = float(best["alpha"])
            self.validation_mse_ = float(best["val_mse"])
            self.objective_ = float(best["objective"])

        self.inactive_features_ = np.asarray([j for j in range(p) if j not in set(self.active_features_)], dtype=int)

        # Refit chosen structure on full data with selected features only.
        final = self._fit_best_alpha(X[:, self.active_features_], y, X[:, self.active_features_], y)
        self.intercept_ = float(final["intercept"])
        self.coef_ = np.asarray(final["coef"], dtype=float)
        self.alpha_ = float(final["alpha"])
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "coef_", "active_features_"])
        X = np.asarray(X, dtype=float)
        out = self.intercept_ + X[:, self.active_features_] @ self.coef_
        return np.asarray(out, dtype=float)

    def __str__(self):
        check_is_fitted(self, ["intercept_", "coef_", "active_features_", "inactive_features_", "alpha_"])
        lines = [
            "Risk-Calibrated Sparse Ridge Regressor",
            f"ridge_alpha={self.alpha_:.6g}",
            "Prediction uses this exact equation:",
        ]

        equation = f"y = {self.intercept_:+.6f}"
        for j, c in zip(self.active_features_, self.coef_):
            equation += f" {float(c):+.6f}*x{int(j)}"
        lines.append(equation)

        lines.append("Active feature coefficients (largest absolute effect first):")
        ranked = sorted(zip(self.active_features_, self.coef_), key=lambda t: -abs(float(t[1])))
        for j, c in ranked:
            lines.append(f"  x{int(j)}: {float(c):+.6f}")

        if len(self.inactive_features_) > 0:
            lines.append(
                "Features with negligible or zero effect in this fitted equation: "
                + ", ".join(f"x{int(j)}" for j in self.inactive_features_)
            )

        ops = 1 + len(self.active_features_)
        lines.append(f"Approx arithmetic operations: {ops}")
        lines.append(f"Validation objective: {self.objective_:.6f}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
RiskCalibratedSparseRidgeRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "RiskCalibratedSparseRidgeV1"
model_description = "Validation-calibrated top-k sparse ridge in exact raw-feature equation form with dense fallback when sparsity does not earn real validation gain"
model_defs = [(model_shorthand_name, RiskCalibratedSparseRidgeRegressor())]

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
