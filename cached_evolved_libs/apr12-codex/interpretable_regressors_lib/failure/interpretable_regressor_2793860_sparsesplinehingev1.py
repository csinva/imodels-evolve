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


class SparseAdaptiveSplineRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse linear backbone with a few additive hinge terms max(0, x_j - t).
    """

    def __init__(
        self,
        l1_alpha=0.06,
        max_linear_terms=8,
        cd_iters=60,
        max_hinges=2,
        hinge_quantiles=(0.15, 0.35, 0.5, 0.65, 0.85),
        candidate_features=8,
        min_hinge_gain=2e-3,
        coef_tol=1e-6,
    ):
        self.l1_alpha = l1_alpha
        self.max_linear_terms = max_linear_terms
        self.cd_iters = cd_iters
        self.max_hinges = max_hinges
        self.hinge_quantiles = hinge_quantiles
        self.candidate_features = candidate_features
        self.min_hinge_gain = min_hinge_gain
        self.coef_tol = coef_tol

    @staticmethod
    def _soft_threshold(value, lam):
        if value > lam:
            return value - lam
        if value < -lam:
            return value + lam
        return 0.0

    def _impute(self, X):
        X = np.asarray(X, dtype=float).copy()
        mask = ~np.isfinite(X)
        if mask.any():
            X[mask] = np.take(self.feature_medians_, np.where(mask)[1])
        return X

    def _fit_sparse_linear(self, Xs, y_center):
        n, p = Xs.shape
        w = np.zeros(p, dtype=float)
        x_sq = np.mean(Xs * Xs, axis=0) + 1e-12
        y_hat = np.zeros(n, dtype=float)

        lam = float(self.l1_alpha) * float(np.std(y_center)) * np.sqrt(np.log(p + 1.0) / max(n, 1))

        for _ in range(self.cd_iters):
            for j in range(p):
                old = w[j]
                r_j = y_center - y_hat + old * Xs[:, j]
                rho = float(np.mean(Xs[:, j] * r_j))
                new = self._soft_threshold(rho, lam) / x_sq[j]
                if new != old:
                    y_hat += (new - old) * Xs[:, j]
                    w[j] = new

        keep = np.where(np.abs(w) > self.coef_tol)[0]
        if keep.size > self.max_linear_terms:
            top = np.argsort(np.abs(w))[::-1][: self.max_linear_terms]
            keep = np.sort(top)
        w_sparse = np.zeros_like(w)
        w_sparse[keep] = w[keep]
        return w_sparse

    @staticmethod
    def _hinge_feature(x, threshold):
        return np.maximum(0.0, x - threshold)

    def _design_matrix(self, X, active_linear, hinges):
        cols = [np.ones(X.shape[0], dtype=float)]
        for j in active_linear:
            cols.append(X[:, j])
        for h in hinges:
            cols.append(self._hinge_feature(X[:, h["j"]], h["threshold"]))
        return np.column_stack(cols)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_features_in_ = X.shape[1]

        self.feature_medians_ = np.nanmedian(X, axis=0)
        X = self._impute(X)

        self.x_mean_ = np.mean(X, axis=0)
        self.x_scale_ = np.std(X, axis=0) + 1e-12
        Xs = (X - self.x_mean_) / self.x_scale_
        y_center = y - float(np.mean(y))

        w_sparse = self._fit_sparse_linear(Xs, y_center)
        active_linear = np.where(np.abs(w_sparse) > self.coef_tol)[0].tolist()
        if not active_linear:
            active_linear = [int(np.argmax(np.abs(np.mean(Xs * y_center[:, None], axis=0))))]

        # Convert from standardized coefficients to original-space equation.
        linear_coef = np.zeros(self.n_features_in_, dtype=float)
        linear_coef[active_linear] = w_sparse[active_linear] / self.x_scale_[active_linear]
        intercept = float(np.mean(y)) - float(np.sum(linear_coef * self.x_mean_))

        base_pred = intercept + X @ linear_coef
        hinges = []
        current_pred = base_pred.copy()
        prev_mse = float(np.mean((y - current_pred) ** 2))
        corr = np.abs(np.mean(X * y_center[:, None], axis=0))
        feat_order = np.argsort(corr)[::-1]
        pool = list(dict.fromkeys(active_linear + feat_order[: self.candidate_features].tolist()))
        all_candidates = []
        for j in pool:
            xj = X[:, j]
            thresholds = {0.0}
            for q in self.hinge_quantiles:
                thresholds.add(float(np.quantile(xj, q)))
            for thr in sorted(thresholds):
                z = self._hinge_feature(xj, thr)
                if float(np.std(z)) > 1e-8:
                    all_candidates.append({"j": int(j), "threshold": float(thr)})

        for _ in range(self.max_hinges):
            residual = y - current_pred
            best = None
            best_score = 0.0
            for cand in all_candidates:
                z = self._hinge_feature(X[:, cand["j"]], cand["threshold"])
                zc = z - float(np.mean(z))
                denom = float(np.sqrt(np.mean(zc * zc))) + 1e-12
                score = abs(float(np.mean(zc * residual))) / denom
                if score > best_score:
                    best = cand
                    best_score = score
            if best is None:
                break

            trial_hinges = hinges + [best]
            D = self._design_matrix(X, active_linear, trial_hinges)
            coef = np.linalg.lstsq(D, y, rcond=None)[0]
            trial_pred = D @ coef
            mse = float(np.mean((y - trial_pred) ** 2))
            if (prev_mse - mse) < self.min_hinge_gain:
                break

            hinges = trial_hinges
            current_pred = trial_pred
            prev_mse = mse
            all_candidates = [c for c in all_candidates if c != best]

        D_final = self._design_matrix(X, active_linear, hinges)
        coef_final = np.linalg.lstsq(D_final, y, rcond=None)[0]

        self.intercept_ = float(coef_final[0])
        self.active_linear_idx_ = list(active_linear)
        self.linear_coef_ = np.array(coef_final[1: 1 + len(active_linear)], dtype=float)
        self.hinges_ = []
        self.hinge_coef_ = np.array(coef_final[1 + len(active_linear):], dtype=float)

        for h, c in zip(hinges, self.hinge_coef_):
            if abs(float(c)) > self.coef_tol:
                hh = dict(h)
                hh["coef"] = float(c)
                self.hinges_.append(hh)

        # Recompute coefficients if some hinges were dropped by tolerance.
        if len(self.hinges_) != len(hinges):
            D = self._design_matrix(X, active_linear, self.hinges_)
            coef = np.linalg.lstsq(D, y, rcond=None)[0]
            self.intercept_ = float(coef[0])
            self.linear_coef_ = np.array(coef[1: 1 + len(active_linear)], dtype=float)
            self.hinge_coef_ = np.array(coef[1 + len(active_linear):], dtype=float)
            for i, c in enumerate(self.hinge_coef_):
                self.hinges_[i]["coef"] = float(c)

        fi = np.zeros(self.n_features_in_, dtype=float)
        for j, c in zip(self.active_linear_idx_, self.linear_coef_):
            fi[j] += abs(float(c))
        for h in self.hinges_:
            fi[h["j"]] += abs(float(h["coef"]))
        self.feature_importance_ = fi
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "active_linear_idx_", "linear_coef_", "hinges_", "feature_importance_"])
        X = self._impute(X)
        y_hat = np.full(X.shape[0], self.intercept_, dtype=float)
        for j, c in zip(self.active_linear_idx_, self.linear_coef_):
            y_hat += c * X[:, j]
        for h in self.hinges_:
            y_hat += h["coef"] * self._hinge_feature(X[:, h["j"]], h["threshold"])
        return y_hat

    def __str__(self):
        check_is_fitted(self, ["intercept_", "active_linear_idx_", "linear_coef_", "hinges_", "feature_importance_"])
        lines = [
            "SparseAdaptiveSplineRegressor",
            "Prediction equation:",
            f"y = {self.intercept_:+.5f}",
        ]
        for j, c in sorted(
            zip(self.active_linear_idx_, self.linear_coef_),
            key=lambda t: abs(float(t[1])),
            reverse=True,
        ):
            lines.append(f"  {c:+.5f} * x{j}")
        for h in sorted(self.hinges_, key=lambda r: abs(float(r["coef"])), reverse=True):
            lines.append(
                f"  {h['coef']:+.5f} * max(0, x{h['j']} - {h['threshold']:.5f})"
            )

        lines.append("")
        lines.append(
            f"Model size: {len(self.active_linear_idx_)} linear terms + {len(self.hinges_)} hinge terms"
        )
        lines.append("Feature importance (sum of absolute coefficients):")
        for j in np.argsort(self.feature_importance_)[::-1]:
            if self.feature_importance_[j] > self.coef_tol:
                lines.append(f"  x{j}: {self.feature_importance_[j]:.5f}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparseAdaptiveSplineRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SparseSplineHingeV1"
model_description = "Sparse L1 linear backbone with up to two learned additive ReLU hinge terms for compact piecewise-linear effects"
model_defs = [(model_shorthand_name, SparseAdaptiveSplineRegressor())]


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
