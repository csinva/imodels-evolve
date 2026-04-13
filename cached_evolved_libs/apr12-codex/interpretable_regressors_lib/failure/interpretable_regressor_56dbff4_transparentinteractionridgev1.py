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


class TransparentInteractionRidgeRegressor(BaseEstimator, RegressorMixin):
    """
    Dense linear ridge regressor with one optional pairwise interaction term.

    Steps:
      1) Fit dense ridge in standardized feature space with GCV alpha selection.
      2) Screen pairwise products among top-magnitude linear features.
      3) Add at most one interaction if it improves training MSE enough.

    Final model is explicitly represented in raw input features:
      y = intercept + sum_j w_j * x_j + w_ij * (x_i * x_j)
    """

    def __init__(
        self,
        alpha_grid=(1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0),
        interaction_top_features=6,
        interaction_min_gain=1e-4,
        coef_decimals=5,
    ):
        self.alpha_grid = alpha_grid
        self.interaction_top_features = interaction_top_features
        self.interaction_min_gain = interaction_min_gain
        self.coef_decimals = coef_decimals

    def _impute(self, X):
        X = np.asarray(X, dtype=float).copy()
        bad = ~np.isfinite(X)
        if bad.any():
            X[bad] = np.take(self.feature_medians_, np.where(bad)[1])
        return X

    @staticmethod
    def _ridge_from_svd(U, s, Vt, yc, alpha):
        ut_y = U.T @ yc
        shrink = s / (s * s + float(alpha))
        return Vt.T @ (shrink * ut_y)

    def _fit_linear_gcv(self, Xs, yc):
        n = Xs.shape[0]
        U, s, Vt = np.linalg.svd(Xs, full_matrices=False)
        n_float = float(n)

        best = None
        for alpha in self.alpha_grid:
            beta = self._ridge_from_svd(U, s, Vt, yc, alpha)
            resid = yc - (Xs @ beta)
            rss = float(np.dot(resid, resid))
            df = float(np.sum((s * s) / (s * s + float(alpha))))
            denom = max((1.0 - df / n_float) ** 2, 1e-12)
            gcv = (rss / n_float) / denom
            if best is None or gcv < best["gcv"]:
                best = {"alpha": float(alpha), "beta": beta, "gcv": gcv}
        return best

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape
        self.n_features_in_ = p

        self.feature_medians_ = np.nanmedian(X, axis=0)
        self.feature_medians_ = np.where(np.isfinite(self.feature_medians_), self.feature_medians_, 0.0)
        X = self._impute(X)

        self.x_mean_ = np.mean(X, axis=0)
        self.x_scale_ = np.std(X, axis=0)
        self.x_scale_[self.x_scale_ < 1e-12] = 1.0
        Xs = (X - self.x_mean_) / self.x_scale_

        y_mean = float(np.mean(y))
        yc = y - y_mean

        # Step 1: dense ridge backbone
        linear_fit = self._fit_linear_gcv(Xs, yc)
        beta_std = linear_fit["beta"]
        pred_centered = Xs @ beta_std
        resid = yc - pred_centered

        coef_raw = beta_std / self.x_scale_
        intercept_raw = y_mean - float(np.dot(coef_raw, self.x_mean_))

        # Step 2: one interaction term from top linear features
        k = min(max(int(self.interaction_top_features), 2), p)
        top_idx = np.argsort(np.abs(coef_raw))[::-1][:k]

        best_pair = None
        best_score = 0.0
        for a in range(len(top_idx)):
            i = int(top_idx[a])
            xi = X[:, i]
            for b in range(a + 1, len(top_idx)):
                j = int(top_idx[b])
                xj = X[:, j]
                z = xi * xj
                zc = z - float(np.mean(z))
                denom = float(np.dot(zc, zc))
                if denom < 1e-12:
                    continue
                corr_num = float(np.dot(zc, resid))
                score = abs(corr_num) / (np.sqrt(denom * float(np.dot(resid, resid))) + 1e-12)
                if score > best_score:
                    best_score = score
                    best_pair = (i, j, z, zc)

        self.interaction_ = None
        base_mse = float(np.mean((y - (intercept_raw + X @ coef_raw)) ** 2))

        if best_pair is not None:
            i, j, z, zc = best_pair
            denom = float(np.dot(zc, zc))
            gamma = float(np.dot(zc, yc - (Xs @ beta_std)) / max(denom, 1e-12))

            centered_inter = z - float(np.mean(z))
            beta2 = beta_std + 0.0
            intercept2 = y_mean - float(np.dot(beta2 / self.x_scale_, self.x_mean_))
            pred2 = intercept2 + X @ (beta2 / self.x_scale_) + gamma * centered_inter
            mse2 = float(np.mean((y - pred2) ** 2))

            if base_mse - mse2 >= float(self.interaction_min_gain):
                self.interaction_ = {
                    "i": int(i),
                    "j": int(j),
                    "gamma": float(gamma),
                    "z_mean": float(np.mean(z)),
                }
                base_mse = mse2

        q = int(self.coef_decimals)
        self.alpha_ = float(linear_fit["alpha"])
        self.coef_ = np.round(coef_raw, q)
        self.intercept_ = float(np.round(intercept_raw, q))

        if self.interaction_ is not None:
            self.interaction_["gamma"] = float(np.round(self.interaction_["gamma"], q))
            self.interaction_["z_mean"] = float(np.round(self.interaction_["z_mean"], q))

        imp = np.abs(self.coef_)
        if self.interaction_ is not None:
            g = abs(self.interaction_["gamma"])
            imp[self.interaction_["i"]] += 0.5 * g
            imp[self.interaction_["j"]] += 0.5 * g

        self.feature_importance_ = imp
        self.feature_rank_ = np.argsort(self.feature_importance_)[::-1]
        self.fitted_mse_ = base_mse
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "coef_"])
        X = self._impute(X)
        y = self.intercept_ + X @ self.coef_
        if getattr(self, "interaction_", None) is not None:
            i = self.interaction_["i"]
            j = self.interaction_["j"]
            g = self.interaction_["gamma"]
            zm = self.interaction_["z_mean"]
            y = y + g * ((X[:, i] * X[:, j]) - zm)
        return y

    def __str__(self):
        check_is_fitted(self, ["intercept_", "coef_", "feature_rank_"])

        terms = [f"{self.intercept_:+0.5f}"]
        for j, c in enumerate(self.coef_):
            if abs(float(c)) > 0.0:
                terms.append(f"{c:+0.5f}*x{j}")

        if self.interaction_ is not None:
            i = self.interaction_["i"]
            j = self.interaction_["j"]
            g = self.interaction_["gamma"]
            zm = self.interaction_["z_mean"]
            terms.append(f"{g:+0.5f}*(x{i}*x{j} {(-zm):+0.5f})")

        lines = [
            "TransparentInteractionRidgeRegressor",
            f"Selected alpha (GCV): {self.alpha_:.6g}",
            "",
            "Prediction equation:",
            "  y = " + " ".join(terms),
            "",
            "Feature coefficients (sorted by |effect|):",
        ]
        for j in self.feature_rank_[: min(12, self.n_features_in_)]:
            lines.append(
                f"  x{int(j)}: coef={self.coef_[int(j)]:+0.5f}  |coef|={abs(float(self.coef_[int(j)])):0.5f}"
            )

        if self.interaction_ is not None:
            lines.extend(
                [
                    "",
                    "Interaction term:",
                    f"  features: x{self.interaction_['i']} * x{self.interaction_['j']}",
                    f"  coefficient: {self.interaction_['gamma']:+0.5f}",
                    f"  centered at mean(x{self.interaction_['i']}*x{self.interaction_['j']}) = {self.interaction_['z_mean']:+0.5f}",
                ]
            )
        lines.append(f"Training MSE: {self.fitted_mse_:.6f}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys

_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
TransparentInteractionRidgeRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "TransparentInteractionRidgeV1"
model_description = "Dense GCV-selected ridge equation with one optional centered pairwise interaction term chosen from top linear features by residual correlation"
model_defs = [(model_shorthand_name, TransparentInteractionRidgeRegressor())]

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------

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
