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


class ValidatedSparseRidgeHingeRegressor(BaseEstimator, RegressorMixin):
    """
    Validation-selected sparse ridge equation with an optional single hinge term.

    Pipeline:
    1) median-impute and standardize,
    2) choose ridge alpha by holdout validation,
    3) choose a sparse subset of strongest features if validation error stays close,
    4) optionally add one hinge residual correction if validation error improves.
    """

    def __init__(
        self,
        alpha_min_exp=-3.0,
        alpha_max_exp=3.0,
        alpha_grid_size=19,
        max_active_features=12,
        sparse_within_frac=0.015,
        use_sparse_within_full_frac=0.02,
        val_frac=0.2,
        min_val_samples=40,
        hinge_quantiles=(0.2, 0.4, 0.6, 0.8),
        max_hinge_features=6,
        hinge_l2=0.2,
        hinge_shrink=0.8,
        hinge_min_gain_frac=0.005,
        coef_tol=1e-6,
        random_state=42,
    ):
        self.alpha_min_exp = alpha_min_exp
        self.alpha_max_exp = alpha_max_exp
        self.alpha_grid_size = alpha_grid_size
        self.max_active_features = max_active_features
        self.sparse_within_frac = sparse_within_frac
        self.use_sparse_within_full_frac = use_sparse_within_full_frac
        self.val_frac = val_frac
        self.min_val_samples = min_val_samples
        self.hinge_quantiles = hinge_quantiles
        self.max_hinge_features = max_hinge_features
        self.hinge_l2 = hinge_l2
        self.hinge_shrink = hinge_shrink
        self.hinge_min_gain_frac = hinge_min_gain_frac
        self.coef_tol = coef_tol
        self.random_state = random_state

    def _impute(self, X):
        X = np.asarray(X, dtype=float).copy()
        bad = ~np.isfinite(X)
        if bad.any():
            X[bad] = np.take(self.feature_medians_, np.where(bad)[1])
        return X

    @staticmethod
    def _fit_subset_ridge(X, y, active, alpha):
        if len(active) == 0:
            return {
                "active": np.array([], dtype=int),
                "coef": np.zeros(0, dtype=float),
                "intercept": float(np.mean(y)),
            }

        Xa = X[:, active]
        mu = np.mean(Xa, axis=0)
        sigma = np.std(Xa, axis=0)
        sigma[sigma < 1e-12] = 1.0
        Z = (Xa - mu) / sigma

        y_mean = float(np.mean(y))
        yc = y - y_mean

        k = Z.shape[1]
        gram = Z.T @ Z
        rhs = Z.T @ yc
        beta_std = np.linalg.solve(gram + float(alpha) * np.eye(k), rhs)
        coef = beta_std / sigma
        intercept = y_mean - float(np.dot(mu, coef))

        return {
            "active": np.asarray(active, dtype=int),
            "coef": np.asarray(coef, dtype=float),
            "intercept": float(intercept),
        }

    @staticmethod
    def _predict_from_params(X, params):
        yhat = np.full(X.shape[0], params["intercept"], dtype=float)
        if len(params["active"]) > 0:
            yhat += X[:, params["active"]] @ params["coef"]
        return yhat

    def _make_val_split(self, n):
        if n < max(25, self.min_val_samples + 10):
            return None, None
        rng = np.random.RandomState(self.random_state)
        idx = rng.permutation(n)
        val_n = max(int(self.val_frac * n), int(self.min_val_samples))
        val_n = min(val_n, n // 2)
        if val_n < 20:
            return None, None
        val_idx = idx[:val_n]
        tr_idx = idx[val_n:]
        return tr_idx, val_idx

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape
        self.n_features_in_ = p

        self.feature_medians_ = np.nanmedian(X, axis=0)
        X = self._impute(X)

        tr_idx, val_idx = self._make_val_split(n)
        if tr_idx is None:
            tr_idx = np.arange(n)
            val_idx = np.arange(n)

        Xtr = X[tr_idx]
        ytr = y[tr_idx]
        Xval = X[val_idx]
        yval = y[val_idx]

        alphas = np.logspace(float(self.alpha_min_exp), float(self.alpha_max_exp), int(self.alpha_grid_size))

        all_features = np.arange(p)
        best_alpha = float(alphas[0])
        best_full = None
        best_full_mse = np.inf
        for alpha in alphas:
            cand = self._fit_subset_ridge(Xtr, ytr, all_features, alpha)
            pred_val = self._predict_from_params(Xval, cand)
            mse = float(np.mean((yval - pred_val) ** 2))
            if mse < best_full_mse:
                best_full_mse = mse
                best_full = cand
                best_alpha = float(alpha)

        self.alpha_ = best_alpha

        # Rank by standardized-effect magnitude from the best full model on train split.
        coef_full = np.zeros(p, dtype=float)
        coef_full[best_full["active"]] = best_full["coef"]
        std_xtr = np.std(Xtr, axis=0)
        contribution = np.abs(coef_full) * std_xtr
        order = np.argsort(contribution)[::-1]

        max_k = min(int(self.max_active_features), p)
        sparse_candidates = []
        for k in range(1, max_k + 1):
            active = np.sort(order[:k])
            cand = self._fit_subset_ridge(Xtr, ytr, active, self.alpha_)
            pred_val = self._predict_from_params(Xval, cand)
            mse = float(np.mean((yval - pred_val) ** 2))
            sparse_candidates.append((mse, cand))

        sparse_best_mse, sparse_best = min(sparse_candidates, key=lambda t: t[0])
        sparse_cutoff = sparse_best_mse * (1.0 + float(self.sparse_within_frac))
        sparse_small = None
        sparse_small_mse = np.inf
        for mse, cand in sparse_candidates:
            if mse <= sparse_cutoff:
                k = len(cand["active"])
                if k < sparse_small_mse:
                    sparse_small_mse = float(k)
                    sparse_small = (mse, cand)

        chosen = best_full
        chosen_val_mse = best_full_mse
        self.used_sparse_ = False
        if sparse_small is not None:
            sparse_mse, sparse_cand = sparse_small
            if sparse_mse <= best_full_mse * (1.0 + float(self.use_sparse_within_full_frac)):
                chosen = sparse_cand
                chosen_val_mse = sparse_mse
                self.used_sparse_ = True

        # Refit chosen active set on all data.
        refit = self._fit_subset_ridge(X, y, chosen["active"], self.alpha_)
        self.selected_features_ = [int(j) for j in refit["active"].tolist()]
        self.coef_ = np.asarray(refit["coef"], dtype=float)
        self.intercept_ = float(refit["intercept"])
        self.linear_val_mse_ = float(chosen_val_mse)

        # Optional one-term hinge residual correction chosen on validation.
        self.has_hinge_ = False
        self.hinge_feature_ = -1
        self.hinge_threshold_ = 0.0
        self.hinge_sign_ = 1.0
        self.hinge_coef_ = 0.0

        pred_tr = self.predict(Xtr)
        pred_val = self.predict(Xval)
        resid_tr = ytr - pred_tr
        base_val_mse = float(np.mean((yval - pred_val) ** 2))
        yval_var = float(np.var(yval)) + 1e-12

        candidate_feats = order[: min(int(self.max_hinge_features), p)]
        best_hinge = None

        for j in candidate_feats:
            xtr_j = Xtr[:, j]
            xval_j = Xval[:, j]
            thresholds = np.unique(np.quantile(xtr_j, self.hinge_quantiles))
            for thr in thresholds:
                for sgn in (1.0, -1.0):
                    phi_tr = np.maximum(0.0, sgn * (xtr_j - float(thr)))
                    denom = float(np.dot(phi_tr, phi_tr) + float(self.hinge_l2) * len(phi_tr))
                    if denom <= 1e-12:
                        continue
                    coef_h = float(np.dot(phi_tr, resid_tr) / denom)
                    coef_h *= float(self.hinge_shrink)

                    phi_val = np.maximum(0.0, sgn * (xval_j - float(thr)))
                    pred_h_val = pred_val + coef_h * phi_val
                    mse_h = float(np.mean((yval - pred_h_val) ** 2))
                    if best_hinge is None or mse_h < best_hinge["mse"]:
                        best_hinge = {
                            "mse": mse_h,
                            "feature": int(j),
                            "threshold": float(thr),
                            "sign": float(sgn),
                            "coef": float(coef_h),
                        }

        if best_hinge is not None:
            gain = base_val_mse - best_hinge["mse"]
            if gain > float(self.hinge_min_gain_frac) * yval_var and abs(best_hinge["coef"]) > float(self.coef_tol):
                self.has_hinge_ = True
                self.hinge_feature_ = int(best_hinge["feature"])
                self.hinge_threshold_ = float(best_hinge["threshold"])
                self.hinge_sign_ = float(best_hinge["sign"])
                self.hinge_coef_ = float(best_hinge["coef"])

        if len(self.selected_features_) > 0:
            drop_idx = np.where(np.abs(self.coef_) <= float(self.coef_tol))[0]
            if drop_idx.size > 0:
                keep = np.where(np.abs(self.coef_) > float(self.coef_tol))[0]
                self.selected_features_ = [self.selected_features_[int(i)] for i in keep]
                self.coef_ = self.coef_[keep]

        self.fitted_mse_ = float(np.mean((y - self.predict(X)) ** 2))

        feature_imp = np.zeros(p, dtype=float)
        for j, c in zip(self.selected_features_, self.coef_):
            feature_imp[int(j)] = abs(float(c))
        if self.has_hinge_:
            feature_imp[self.hinge_feature_] += abs(self.hinge_coef_)
        self.feature_importance_ = feature_imp
        self.feature_rank_ = np.argsort(self.feature_importance_)[::-1]
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "coef_", "selected_features_"])
        X = self._impute(X)

        yhat = np.full(X.shape[0], self.intercept_, dtype=float)
        if len(self.selected_features_) > 0:
            yhat += X[:, self.selected_features_] @ self.coef_

        if self.has_hinge_:
            xh = X[:, self.hinge_feature_]
            yhat += self.hinge_coef_ * np.maximum(0.0, self.hinge_sign_ * (xh - self.hinge_threshold_))
        return yhat

    def __str__(self):
        check_is_fitted(self, ["intercept_", "coef_", "selected_features_"])
        lines = [
            "ValidatedSparseRidgeHingeRegressor",
            f"Ridge alpha: {self.alpha_:.6g}",
            "Exact prediction rule:",
        ]

        if len(self.selected_features_) == 0:
            lines.append(f"  y = {self.intercept_:+.6f}")
        else:
            ordered = np.argsort(np.abs(self.coef_))[::-1]
            terms = [f"({self.coef_[i]:+.6f})*x{self.selected_features_[i]}" for i in ordered]
            lines.append(f"  y = {self.intercept_:+.6f} " + " ".join(terms))

        if self.has_hinge_:
            if self.hinge_sign_ > 0:
                hinge_expr = f"max(0, x{self.hinge_feature_} - {self.hinge_threshold_:+.6f})"
            else:
                hinge_expr = f"max(0, {self.hinge_threshold_:+.6f} - x{self.hinge_feature_})"
            lines.append(f"      + ({self.hinge_coef_:+.6f})*{hinge_expr}")
        else:
            lines.append("      + 0.0  (no hinge correction selected)")

        lines.append("")
        lines.append("Feature importance (sorted):")
        for j in self.feature_rank_[: min(15, self.n_features_in_)]:
            lines.append(f"  x{int(j)}: {self.feature_importance_[int(j)]:.6f}")

        unused = [f"x{j}" for j in range(self.n_features_in_) if self.feature_importance_[j] <= 1e-8]
        if unused:
            lines.append("Features with negligible effect: " + ", ".join(unused))
        lines.append(f"Training MSE: {self.fitted_mse_:.6f}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
ValidatedSparseRidgeHingeRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "ValSparseRidgeHingeV1"
model_description = "Holdout-selected sparse ridge equation with optional one-term hinge residual correction when validation gain is significant"
model_defs = [(model_shorthand_name, ValidatedSparseRidgeHingeRegressor())]

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
