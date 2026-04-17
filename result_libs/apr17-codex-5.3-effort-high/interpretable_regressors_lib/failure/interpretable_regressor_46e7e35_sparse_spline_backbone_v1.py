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


class SparseSplineBackboneV1(BaseEstimator, RegressorMixin):
    """
    Validation-gated sparse ridge backbone with optional hinge-spline corrections.

    1) Select ridge alpha by K-fold CV on standardized features.
    2) Build a sparse linear backbone using coefficient-mass feature selection.
    3) Optionally add a tiny set of one-sided hinge terms on dominant features
       if a holdout split shows meaningful MSE gain.
    """

    def __init__(
        self,
        alpha_grid=None,
        cv_folds=3,
        max_active_features=10,
        keep_mass=0.97,
        max_spline_features=2,
        spline_quantiles=(0.25, 0.75),
        sparse_tolerance=0.03,
        min_spline_gain=5e-4,
        seed=42,
        display_precision=5,
    ):
        self.alpha_grid = alpha_grid
        self.cv_folds = cv_folds
        self.max_active_features = max_active_features
        self.keep_mass = keep_mass
        self.max_spline_features = max_spline_features
        self.spline_quantiles = spline_quantiles
        self.sparse_tolerance = sparse_tolerance
        self.min_spline_gain = min_spline_gain
        self.seed = seed
        self.display_precision = display_precision

    @staticmethod
    def _safe_standardize(X):
        mean = np.mean(X, axis=0)
        scale = np.std(X, axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        return mean.astype(float), scale.astype(float)

    @staticmethod
    def _ridge_closed_form(Xs, y, alpha):
        n = Xs.shape[0]
        D = np.column_stack([np.ones(n, dtype=float), Xs])
        reg = np.eye(D.shape[1], dtype=float)
        reg[0, 0] = 0.0
        A = D.T @ D + float(alpha) * reg
        b = D.T @ y
        try:
            theta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            theta = np.linalg.pinv(A) @ b
        return np.asarray(theta, dtype=float)

    def _fit_raw_ridge(self, X, y, alpha):
        mean, scale = self._safe_standardize(X)
        Xs = (X - mean) / scale
        theta = self._ridge_closed_form(Xs, y, alpha)
        coef_raw = theta[1:] / scale
        intercept_raw = float(theta[0] - np.dot(coef_raw, mean))
        return intercept_raw, np.asarray(coef_raw, dtype=float)

    @staticmethod
    def _predict_raw_linear(X, intercept, coef):
        return float(intercept) + np.asarray(X, dtype=float) @ np.asarray(coef, dtype=float)

    def _select_alpha_cv(self, X, y):
        alphas = (
            np.asarray(self.alpha_grid, dtype=float)
            if self.alpha_grid is not None
            else np.logspace(-5, 2, 13)
        )
        n = X.shape[0]
        if n < 18:
            return float(alphas[0]), float("nan")

        n_folds = int(max(2, min(int(self.cv_folds), 5)))
        rng = np.random.RandomState(self.seed)
        idx = np.arange(n)
        rng.shuffle(idx)
        folds = np.array_split(idx, n_folds)

        best_alpha = float(alphas[len(alphas) // 2])
        best_score = float("inf")
        for alpha in alphas:
            fold_mses = []
            for fold_id in range(n_folds):
                val_idx = folds[fold_id]
                tr_idx = np.concatenate([folds[k] for k in range(n_folds) if k != fold_id])
                if len(tr_idx) < 5 or len(val_idx) < 3:
                    continue
                intercept, coef = self._fit_raw_ridge(X[tr_idx], y[tr_idx], float(alpha))
                preds = self._predict_raw_linear(X[val_idx], intercept, coef)
                fold_mses.append(float(np.mean((y[val_idx] - preds) ** 2)))
            if not fold_mses:
                continue
            mse = float(np.mean(fold_mses))
            if mse < best_score:
                best_score = mse
                best_alpha = float(alpha)
        return best_alpha, best_score

    def _select_active_features(self, coef_raw):
        p = len(coef_raw)
        if p == 0:
            return np.array([], dtype=int)
        abs_coef = np.abs(np.asarray(coef_raw, dtype=float))
        order = np.argsort(-abs_coef)
        total = float(np.sum(abs_coef))
        if total <= 1e-12:
            k = int(max(1, min(self.max_active_features, p)))
            return order[:k].astype(int)

        cum = np.cumsum(abs_coef[order]) / total
        mass_target = float(np.clip(self.keep_mass, 0.5, 0.999))
        k_mass = int(np.searchsorted(cum, mass_target) + 1)
        k = int(max(1, min(self.max_active_features, max(2, k_mass), p)))
        return order[:k].astype(int)

    @staticmethod
    def _design_hinges(X_feature, knots):
        x = np.asarray(X_feature, dtype=float).ravel()
        cols = [np.maximum(0.0, x - float(k)) for k in knots]
        if not cols:
            return np.zeros((len(x), 0), dtype=float)
        return np.column_stack(cols)

    @staticmethod
    def _build_fold_split(n, seed):
        rng = np.random.RandomState(seed)
        idx = np.arange(n)
        rng.shuffle(idx)
        n_val = max(10, int(0.25 * n))
        return idx[n_val:], idx[:n_val]

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape
        self.n_features_in_ = p

        alpha, cv_mse = self._select_alpha_cv(X, y)
        dense_intercept, dense_coef = self._fit_raw_ridge(X, y, float(alpha))
        order = np.argsort(-np.abs(dense_coef))

        active_sparse = self._select_active_features(dense_coef)
        tr_idx, val_idx = self._build_fold_split(n, self.seed)
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        dense_tr_intercept, dense_tr_coef = self._fit_raw_ridge(X_tr, y_tr, float(alpha))
        dense_val_pred = self._predict_raw_linear(X_val, dense_tr_intercept, dense_tr_coef)
        dense_val_mse = float(np.mean((y_val - dense_val_pred) ** 2))

        sparse_tr_intercept, sparse_tr_coef = self._fit_raw_ridge(X_tr[:, active_sparse], y_tr, float(alpha))
        sparse_val_pred = self._predict_raw_linear(X_val[:, active_sparse], sparse_tr_intercept, sparse_tr_coef)
        sparse_val_mse = float(np.mean((y_val - sparse_val_pred) ** 2))

        use_sparse = sparse_val_mse <= dense_val_mse * (1.0 + float(self.sparse_tolerance))
        if use_sparse:
            linear_active = np.asarray(active_sparse, dtype=int)
            linear_intercept, linear_coef = self._fit_raw_ridge(X[:, linear_active], y, float(alpha))
            base_intercept, base_coef = self._fit_raw_ridge(X_tr[:, linear_active], y_tr, float(alpha))
            base_val_pred = self._predict_raw_linear(X_val[:, linear_active], base_intercept, base_coef)
            base_val_mse = float(np.mean((y_val - base_val_pred) ** 2))
        else:
            linear_active = np.arange(p, dtype=int)
            linear_intercept = float(dense_intercept)
            linear_coef = np.asarray(dense_coef, dtype=float)
            base_intercept, base_coef = dense_tr_intercept, dense_tr_coef
            base_val_mse = dense_val_mse

        spline_feature_order = np.argsort(-np.abs(base_coef))
        max_spline = int(max(0, min(self.max_spline_features, len(spline_feature_order))))
        spline_features = np.asarray(
            [int(linear_active[i]) for i in spline_feature_order[:max_spline]],
            dtype=int,
        ) if len(linear_active) > 0 and max_spline > 0 else np.array([], dtype=int)

        knots_per_feature = {}
        for feat in spline_features:
            vals = X_tr[:, int(feat)]
            knots = np.unique(np.quantile(vals, np.asarray(self.spline_quantiles, dtype=float)))
            knots = [float(k) for k in knots]
            if knots:
                knots_per_feature[int(feat)] = knots

        def _augmented_design(X_block):
            cols = [np.ones(X_block.shape[0], dtype=float)]
            for i, feat in enumerate(linear_active):
                cols.append(X_block[:, int(feat)])
                for knot in knots_per_feature.get(int(feat), []):
                    cols.append(np.maximum(0.0, X_block[:, int(feat)] - float(knot)))
            return np.column_stack(cols)

        if knots_per_feature:
            D_tr = _augmented_design(X_tr)
            D_val = _augmented_design(X_val)
            reg = np.eye(D_tr.shape[1], dtype=float)
            reg[0, 0] = 0.0
            A = D_tr.T @ D_tr + float(alpha) * reg
            b = D_tr.T @ y_tr
            try:
                theta_aug = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                theta_aug = np.linalg.pinv(A) @ b
            aug_val_pred = D_val @ theta_aug
            aug_val_mse = float(np.mean((y_val - aug_val_pred) ** 2))
            use_splines = (base_val_mse - aug_val_mse) > float(self.min_spline_gain)
        else:
            use_splines = False

        if use_splines:
            D_all = _augmented_design(X)
            reg = np.eye(D_all.shape[1], dtype=float)
            reg[0, 0] = 0.0
            A = D_all.T @ D_all + float(alpha) * reg
            b = D_all.T @ y
            try:
                theta_all = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                theta_all = np.linalg.pinv(A) @ b
            intercept_final = float(theta_all[0])
            linear_final = np.asarray(theta_all[1:1 + len(linear_active)], dtype=float)
            spline_coefs = {}
            pos = 1 + len(linear_active)
            for feat in linear_active:
                feat = int(feat)
                ks = knots_per_feature.get(feat, [])
                if ks:
                    spline_coefs[feat] = np.asarray(theta_all[pos:pos + len(ks)], dtype=float)
                    pos += len(ks)
        else:
            intercept_final = float(linear_intercept)
            linear_final = np.asarray(linear_coef, dtype=float)
            knots_per_feature = {}
            spline_coefs = {}

        active_set = set(int(j) for j in linear_active.tolist())
        negligible = np.array([j for j in range(p) if j not in active_set], dtype=int)

        self.use_sparse_backbone_ = bool(use_sparse)
        self.use_splines_ = bool(use_splines)
        self.active_features_ = np.asarray(linear_active, dtype=int)
        self.negligible_features_ = negligible
        self.linear_intercept_ = float(intercept_final)
        self.linear_coef_ = np.asarray(linear_final, dtype=float)
        self.knots_per_feature_ = {int(k): list(v) for k, v in knots_per_feature.items()}
        self.spline_coef_per_feature_ = {int(k): np.asarray(v, dtype=float) for k, v in spline_coefs.items()}
        self.alpha_ = float(alpha)
        self.alpha_cv_mse_ = float(cv_mse)
        self.validation_mse_dense_ = float(dense_val_mse)
        self.validation_mse_sparse_ = float(sparse_val_mse)
        self.validation_mse_backbone_ = float(base_val_mse)
        self.sorted_features_ = order.astype(int)
        self.dominant_feature_ = int(order[0]) if p > 0 else 0
        return self

    def predict(self, X):
        check_is_fitted(self, ["active_features_", "linear_intercept_", "linear_coef_"])
        X = np.asarray(X, dtype=float)
        yhat = self._predict_raw_linear(X[:, self.active_features_], self.linear_intercept_, self.linear_coef_)
        if not self.use_splines_:
            return yhat
        out = np.asarray(yhat, dtype=float).copy()
        for feat in self.active_features_:
            feat = int(feat)
            knots = self.knots_per_feature_.get(feat, [])
            coefs = self.spline_coef_per_feature_.get(feat, [])
            if len(knots) == 0:
                continue
            xj = X[:, feat]
            for k, c in zip(knots, coefs):
                out += float(c) * np.maximum(0.0, xj - float(k))
        return out

    def _format_linear_equation(self):
        prec = int(self.display_precision)
        terms = [f"{float(self.linear_intercept_):+.{prec}f}"]
        for local_i, feat in enumerate(self.active_features_):
            terms.append(f"{float(self.linear_coef_[local_i]):+.{prec}f}*x{int(feat)}")
        return " ".join(terms)

    def __str__(self):
        check_is_fitted(self, ["active_features_", "linear_intercept_", "linear_coef_"])
        prec = int(self.display_precision)
        lines = [
            "Sparse Spline Backbone Regressor",
            "Prediction: sparse ridge linear backbone with optional feature-wise hinge spline corrections.",
            f"Active features: {', '.join(f'x{int(j)}' for j in self.active_features_)}",
        ]
        if len(self.negligible_features_) > 0:
            lines.append("Negligible features: " + ", ".join(f"x{int(j)}" for j in self.negligible_features_))

        lines.extend(
            [
                "",
                "Linear backbone equation:",
                "  y = " + self._format_linear_equation(),
            ]
        )

        if self.use_splines_:
            lines.extend(
                [
                    "",
                    "Spline corrections (add these terms to the linear backbone):",
                ]
            )
            for feat in self.active_features_:
                feat = int(feat)
                knots = self.knots_per_feature_.get(feat, [])
                coefs = self.spline_coef_per_feature_.get(feat, [])
                if len(knots) == 0:
                    continue
                for knot, coef in zip(knots, coefs):
                    lines.append(f"  {float(coef):+.{prec}f}*max(0, x{feat}-{float(knot):.{prec}f})")
            lines.extend(
                [
                    "",
                    "Simulation: compute the linear backbone, then add each hinge correction term.",
                ]
            )
        else:
            lines.extend(
                [
                    "",
                    "No spline corrections kept (linear backbone is final prediction).",
                    "",
                    "Simulation: multiply active features by coefficients, sum, and add intercept.",
                ]
            )

        lines.append("")
        lines.append(f"Dominant global feature: x{self.dominant_feature_}")
        lines.append(f"Selected ridge alpha: {self.alpha_:.6g}")
        lines.append(
            f"Backbone mode: {'sparse' if self.use_sparse_backbone_ else 'dense'} "
            f"(holdout MSE dense={self.validation_mse_dense_:.5f}, sparse={self.validation_mse_sparse_:.5f})"
        )
        return "\n".join(lines)


import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparseSplineBackboneV1.__module__ = "interpretable_regressor"

model_shorthand_name = "SparseSplineBackboneV1"
model_description = "K-fold-selected ridge backbone with validation-gated coefficient-mass sparsification and optional dominant-feature hinge spline corrections for a compact simulatable equation"
model_defs = [(model_shorthand_name, SparseSplineBackboneV1())]

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
