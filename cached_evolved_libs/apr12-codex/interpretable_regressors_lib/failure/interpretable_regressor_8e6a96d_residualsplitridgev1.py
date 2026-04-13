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


class ResidualSplitRidgeRegressor(BaseEstimator, RegressorMixin):
    """
    Dense ridge backbone + optional one-split residual correction.

    Design goals:
      - Keep strong predictive performance via dense RidgeCV backbone.
      - Add one compact split rule only when validation error improves.
      - Keep explanation concise by exposing only dominant terms.

    Prediction form:
      y = ridge(x) + I[x_s <= t] * (aL + bL^T x_top) + I[x_s > t] * (aR + bR^T x_top)
    """

    def __init__(
        self,
        ridge_alphas=(1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0, 100.0),
        split_top_features=8,
        split_quantiles=(0.2, 0.4, 0.6, 0.8),
        residual_top_features=5,
        min_leaf_frac=0.15,
        min_val_gain=0.005,
        residual_shrinkage=0.85,
        display_max_terms=12,
        display_coef_threshold=0.015,
        random_state=42,
    ):
        self.ridge_alphas = ridge_alphas
        self.split_top_features = split_top_features
        self.split_quantiles = split_quantiles
        self.residual_top_features = residual_top_features
        self.min_leaf_frac = min_leaf_frac
        self.min_val_gain = min_val_gain
        self.residual_shrinkage = residual_shrinkage
        self.display_max_terms = display_max_terms
        self.display_coef_threshold = display_coef_threshold
        self.random_state = random_state

    def _impute(self, X):
        X = np.asarray(X, dtype=float).copy()
        bad = ~np.isfinite(X)
        if bad.any():
            X[bad] = np.take(self.feature_medians_, np.where(bad)[1])
        return X

    @staticmethod
    def _safe_std(x):
        s = np.std(x, axis=0)
        s[s < 1e-12] = 1.0
        return s

    def _fit_dense_ridge(self, X, y):
        from sklearn.linear_model import RidgeCV

        x_mean = np.mean(X, axis=0)
        x_std = self._safe_std(X)
        Xs = (X - x_mean) / x_std

        reg = RidgeCV(alphas=np.asarray(self.ridge_alphas, dtype=float), cv=3)
        reg.fit(Xs, y)

        coef_raw = reg.coef_ / x_std
        intercept_raw = float(reg.intercept_ - np.dot(coef_raw, x_mean))
        return {
            "coef": coef_raw.astype(float),
            "intercept": intercept_raw,
            "alpha": float(reg.alpha_),
        }

    @staticmethod
    def _fit_leaf_linear(X_sub, r_sub, feat_idx):
        if X_sub.shape[0] == 0:
            return {"intercept": 0.0, "coef": np.zeros(len(feat_idx), dtype=float)}

        Z = X_sub[:, feat_idx]
        z_mean = np.mean(Z, axis=0)
        z_std = np.std(Z, axis=0)
        z_std[z_std < 1e-12] = 1.0
        Zs = (Z - z_mean) / z_std

        A = np.concatenate([np.ones((Zs.shape[0], 1)), Zs], axis=1)
        beta, _, _, _ = np.linalg.lstsq(A, r_sub, rcond=None)
        intercept = float(beta[0] - np.dot(beta[1:] / z_std, z_mean))
        coef = (beta[1:] / z_std).astype(float)
        return {"intercept": intercept, "coef": coef}

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape
        self.n_features_in_ = p

        self.feature_medians_ = np.nanmedian(X, axis=0)
        self.feature_medians_ = np.where(np.isfinite(self.feature_medians_), self.feature_medians_, 0.0)
        X = self._impute(X)

        # Deterministic holdout split for model selection without heavy CV.
        rng = np.random.RandomState(self.random_state)
        idx = np.arange(n)
        rng.shuffle(idx)
        n_val = max(40, int(0.2 * n))
        val_idx = idx[:n_val]
        tr_idx = idx[n_val:]
        if len(tr_idx) < max(20, int(0.5 * n)):
            tr_idx = idx
            val_idx = idx[: min(80, n)]

        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xva, yva = X[val_idx], y[val_idx]

        base = self._fit_dense_ridge(Xtr, ytr)
        self.coef_ = base["coef"]
        self.intercept_ = base["intercept"]
        self.alpha_ = base["alpha"]

        pred_tr = self.intercept_ + Xtr @ self.coef_
        pred_va = self.intercept_ + Xva @ self.coef_
        resid_tr = ytr - pred_tr

        base_val_mse = float(np.mean((yva - pred_va) ** 2))

        # Use strongest backbone features for compact residual correction.
        abs_coef = np.abs(self.coef_)
        k_res = min(max(1, int(self.residual_top_features)), p)
        top_res_idx = np.argsort(abs_coef)[::-1][:k_res]

        # Choose split features by residual nonlinearity signal.
        nl_scores = np.zeros(p, dtype=float)
        r_centered = resid_tr - np.mean(resid_tr)
        r_norm = np.sqrt(np.dot(r_centered, r_centered)) + 1e-12
        for j in range(p):
            xj = Xtr[:, j]
            xj_c = xj - np.mean(xj)
            denom = np.sqrt(np.dot(xj_c, xj_c)) + 1e-12
            corr_lin = abs(float(np.dot(xj_c, r_centered)) / (denom * r_norm))
            xj_sq_c = (xj * xj) - np.mean(xj * xj)
            denom2 = np.sqrt(np.dot(xj_sq_c, xj_sq_c)) + 1e-12
            corr_sq = abs(float(np.dot(xj_sq_c, r_centered)) / (denom2 * r_norm))
            nl_scores[j] = corr_lin + 0.6 * corr_sq

        k_split = min(max(1, int(self.split_top_features)), p)
        split_candidates = np.argsort(nl_scores)[::-1][:k_split]

        min_leaf = max(20, int(float(self.min_leaf_frac) * len(tr_idx)))
        best = None

        for feat in split_candidates:
            thresholds = np.unique(np.quantile(Xtr[:, feat], self.split_quantiles))
            for thr in thresholds:
                left_tr = Xtr[:, feat] <= thr
                right_tr = ~left_tr
                if int(left_tr.sum()) < min_leaf or int(right_tr.sum()) < min_leaf:
                    continue

                leaf_left = self._fit_leaf_linear(Xtr[left_tr], resid_tr[left_tr], top_res_idx)
                leaf_right = self._fit_leaf_linear(Xtr[right_tr], resid_tr[right_tr], top_res_idx)

                left_va = Xva[:, feat] <= thr
                right_va = ~left_va

                delta_va = np.zeros(len(Xva), dtype=float)
                if left_va.any():
                    delta_va[left_va] = (
                        leaf_left["intercept"] + Xva[left_va][:, top_res_idx] @ leaf_left["coef"]
                    )
                if right_va.any():
                    delta_va[right_va] = (
                        leaf_right["intercept"] + Xva[right_va][:, top_res_idx] @ leaf_right["coef"]
                    )

                pred_va_split = pred_va + float(self.residual_shrinkage) * delta_va
                val_mse = float(np.mean((yva - pred_va_split) ** 2))

                if best is None or val_mse < best["val_mse"]:
                    best = {
                        "feature": int(feat),
                        "threshold": float(thr),
                        "val_mse": val_mse,
                        "top_res_idx": top_res_idx.copy(),
                        "left": leaf_left,
                        "right": leaf_right,
                    }

        self.use_split_ = False
        if best is not None and base_val_mse - best["val_mse"] >= float(self.min_val_gain):
            self.use_split_ = True
            self.split_feature_ = best["feature"]
            self.split_threshold_ = best["threshold"]
            self.split_features_ = best["top_res_idx"]
            self.left_leaf_ = best["left"]
            self.right_leaf_ = best["right"]

        imp = np.abs(self.coef_).copy()
        if self.use_split_:
            for local_coef in [self.left_leaf_["coef"], self.right_leaf_["coef"]]:
                for j_loc, feat_j in enumerate(self.split_features_):
                    imp[int(feat_j)] += 0.5 * abs(float(local_coef[j_loc]))
            imp[int(self.split_feature_)] += 0.25

        self.feature_importance_ = imp
        self.feature_rank_ = np.argsort(imp)[::-1]

        # Compact display support only; prediction uses full coefficients.
        if np.max(np.abs(self.coef_)) > 0:
            rel = np.abs(self.coef_) / (np.max(np.abs(self.coef_)) + 1e-12)
            keep = np.where(rel >= float(self.display_coef_threshold))[0]
        else:
            keep = np.array([], dtype=int)
        if keep.size == 0:
            keep = np.array([int(np.argmax(np.abs(self.coef_)))], dtype=int)
        if keep.size > int(self.display_max_terms):
            order = np.argsort(np.abs(self.coef_[keep]))[::-1]
            keep = keep[order[: int(self.display_max_terms)]]
        self.display_terms_ = np.array(sorted(set(int(k) for k in keep.tolist())), dtype=int)

        return self

    def predict(self, X):
        check_is_fitted(self, ["coef_", "intercept_"])
        X = self._impute(X)
        y = self.intercept_ + X @ self.coef_
        if getattr(self, "use_split_", False):
            left = X[:, self.split_feature_] <= self.split_threshold_
            delta = np.zeros(X.shape[0], dtype=float)
            if left.any():
                delta[left] = self.left_leaf_["intercept"] + X[left][:, self.split_features_] @ self.left_leaf_["coef"]
            if (~left).any():
                delta[~left] = self.right_leaf_["intercept"] + X[~left][:, self.split_features_] @ self.right_leaf_["coef"]
            y = y + float(self.residual_shrinkage) * delta
        return y

    def _linear_eq_str(self):
        terms = [f"{self.intercept_:+.4f}"]
        for j in self.display_terms_:
            c = float(self.coef_[j])
            terms.append(f"{c:+.4f}*x{int(j)}")
        return "y = " + " ".join(terms)

    @staticmethod
    def _leaf_eq_str(leaf, feat_idx):
        terms = [f"{leaf['intercept']:+.4f}"]
        for k, j in enumerate(feat_idx):
            c = float(leaf["coef"][k])
            if abs(c) > 1e-6:
                terms.append(f"{c:+.4f}*x{int(j)}")
        return " ".join(terms)

    def __str__(self):
        check_is_fitted(self, ["feature_rank_", "display_terms_"])
        lines = [
            "ResidualSplitRidgeRegressor",
            f"Backbone: dense RidgeCV (alpha={self.alpha_:.4g})",
            "Main equation (dominant terms shown):",
            "  " + self._linear_eq_str(),
        ]

        hidden = self.n_features_in_ - len(self.display_terms_)
        if hidden > 0:
            lines.append(f"  (+ {hidden} additional small linear terms omitted for readability)")

        if getattr(self, "use_split_", False):
            lines.extend(
                [
                    "",
                    f"Residual split rule: if x{self.split_feature_} <= {self.split_threshold_:.4f}",
                    "  add delta_left = " + self._leaf_eq_str(self.left_leaf_, self.split_features_),
                    "else",
                    "  add delta_right = " + self._leaf_eq_str(self.right_leaf_, self.split_features_),
                    f"Final prediction: y = backbone + {self.residual_shrinkage:.2f} * delta",
                ]
            )
        else:
            lines.append("No residual split was selected (backbone already sufficient on validation).")

        lines.extend(["", "Feature importance (top 12):"])
        for j in self.feature_rank_[: min(12, self.n_features_in_)]:
            lines.append(f"  x{int(j)}: {self.feature_importance_[int(j)]:.4f}")

        near_zero = [f"x{j}" for j, v in enumerate(np.abs(self.coef_)) if v < 1e-5]
        if near_zero:
            lines.append("Near-zero backbone coefficients: " + ", ".join(near_zero[:20]))
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys

_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
ResidualSplitRidgeRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "ResidualSplitRidgeV1"
model_description = "Dense RidgeCV backbone with one validation-gated threshold split that adds compact residual linear leaf corrections"
model_defs = [(model_shorthand_name, ResidualSplitRidgeRegressor())]


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
