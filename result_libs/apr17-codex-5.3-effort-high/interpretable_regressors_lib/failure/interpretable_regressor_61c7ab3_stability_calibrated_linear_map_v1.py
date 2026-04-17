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


class StabilityCalibratedLinearMapV1(BaseEstimator, RegressorMixin):
    """
    Bootstrap-stability screened linear model with lightweight score calibration.

    Steps:
    1) Choose ridge alpha by cross-validation over a small grid.
    2) Estimate coefficient sign stability with bootstrap subsamples.
    3) Keep a compact active set (stable + high-mass coefficients).
    4) Refit linear map on active features.
    5) Optionally calibrate the linear score with two hinge terms on the score axis.

    Prediction remains a closed-form arithmetic equation.
    """

    def __init__(
        self,
        alpha_grid=None,
        cv_folds=3,
        n_bootstrap=8,
        bootstrap_frac=0.75,
        stability_threshold=0.75,
        keep_mass=0.98,
        max_active_features=12,
        min_active_features=2,
        calibration_quantiles=(0.33, 0.67),
        min_calibration_gain=3e-4,
        seed=42,
        display_precision=5,
    ):
        self.alpha_grid = alpha_grid
        self.cv_folds = cv_folds
        self.n_bootstrap = n_bootstrap
        self.bootstrap_frac = bootstrap_frac
        self.stability_threshold = stability_threshold
        self.keep_mass = keep_mass
        self.max_active_features = max_active_features
        self.min_active_features = min_active_features
        self.calibration_quantiles = calibration_quantiles
        self.min_calibration_gain = min_calibration_gain
        self.seed = seed
        self.display_precision = display_precision

    @staticmethod
    def _standardize(X):
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
        mean, scale = self._standardize(X)
        Xs = (X - mean) / scale
        theta = self._ridge_closed_form(Xs, y, alpha)
        coef_raw = theta[1:] / scale
        intercept_raw = float(theta[0] - np.dot(coef_raw, mean))
        return intercept_raw, np.asarray(coef_raw, dtype=float)

    @staticmethod
    def _predict_linear(X, intercept, coef):
        return float(intercept) + np.asarray(X, dtype=float) @ np.asarray(coef, dtype=float)

    def _select_alpha_cv(self, X, y):
        alphas = np.asarray(self.alpha_grid, dtype=float) if self.alpha_grid is not None else np.logspace(-5, 2, 13)
        n = X.shape[0]
        if n < 20:
            return float(alphas[0]), float("nan")

        n_folds = int(max(2, min(int(self.cv_folds), 5)))
        rng = np.random.RandomState(self.seed)
        idx = np.arange(n)
        rng.shuffle(idx)
        folds = np.array_split(idx, n_folds)

        best_alpha = float(alphas[len(alphas) // 2])
        best_mse = float("inf")
        for alpha in alphas:
            fold_mses = []
            for k in range(n_folds):
                val_idx = folds[k]
                tr_idx = np.concatenate([folds[j] for j in range(n_folds) if j != k])
                if len(tr_idx) < 8 or len(val_idx) < 4:
                    continue
                intercept, coef = self._fit_raw_ridge(X[tr_idx], y[tr_idx], float(alpha))
                pred = self._predict_linear(X[val_idx], intercept, coef)
                fold_mses.append(float(np.mean((y[val_idx] - pred) ** 2)))
            if not fold_mses:
                continue
            mse = float(np.mean(fold_mses))
            if mse < best_mse:
                best_mse = mse
                best_alpha = float(alpha)
        return best_alpha, best_mse

    def _bootstrap_stability(self, X, y, alpha):
        n, p = X.shape
        rng = np.random.RandomState(self.seed)
        B = int(max(1, self.n_bootstrap))
        frac = float(np.clip(self.bootstrap_frac, 0.4, 1.0))
        m = max(8, int(frac * n))

        sign_votes = np.zeros((B, p), dtype=float)
        mag = np.zeros((B, p), dtype=float)
        for b in range(B):
            idx = rng.choice(n, size=m, replace=True)
            _, coef = self._fit_raw_ridge(X[idx], y[idx], float(alpha))
            sign_votes[b] = np.sign(coef)
            mag[b] = np.abs(coef)

        sign_consistency = np.abs(np.mean(sign_votes, axis=0))
        mean_mag = np.mean(mag, axis=0)
        return sign_consistency.astype(float), mean_mag.astype(float)

    def _select_active(self, coef_dense, sign_consistency, mean_mag):
        p = len(coef_dense)
        if p == 0:
            return np.array([], dtype=int)

        abs_coef = np.abs(np.asarray(coef_dense, dtype=float))
        order = np.argsort(-abs_coef)

        stable = np.where(sign_consistency >= float(self.stability_threshold))[0]
        stable = stable[np.argsort(-mean_mag[stable])] if len(stable) else stable

        total = float(np.sum(abs_coef))
        if total <= 1e-12:
            mass_keep = np.array([], dtype=int)
        else:
            cum = np.cumsum(abs_coef[order]) / total
            k_mass = int(np.searchsorted(cum, float(np.clip(self.keep_mass, 0.7, 0.999))) + 1)
            mass_keep = order[:k_mass]

        selected = list(stable.tolist())
        for j in mass_keep:
            if int(j) not in selected:
                selected.append(int(j))

        if len(selected) < int(self.min_active_features):
            for j in order:
                if int(j) not in selected:
                    selected.append(int(j))
                if len(selected) >= int(self.min_active_features):
                    break

        max_k = int(max(self.min_active_features, min(self.max_active_features, p)))
        selected = np.asarray(selected[:max_k], dtype=int)
        if len(selected) == 0:
            selected = order[: int(min(max_k, p))].astype(int)
        return selected

    @staticmethod
    def _build_split(n, seed):
        rng = np.random.RandomState(seed)
        idx = np.arange(n)
        rng.shuffle(idx)
        n_val = max(12, int(0.25 * n))
        return idx[n_val:], idx[:n_val]

    @staticmethod
    def _fit_score_calibrator(score, y, alpha, quantiles):
        qvals = np.unique(np.quantile(score, np.asarray(quantiles, dtype=float)))
        qvals = [float(v) for v in qvals]
        cols = [np.ones_like(score), score]
        for q in qvals:
            cols.append(np.maximum(0.0, score - q))
        D = np.column_stack(cols)

        reg = np.eye(D.shape[1], dtype=float)
        reg[0, 0] = 0.0
        A = D.T @ D + float(alpha) * reg
        b = D.T @ y
        try:
            theta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            theta = np.linalg.pinv(A) @ b
        return np.asarray(theta, dtype=float), qvals

    @staticmethod
    def _apply_score_calibrator(score, theta, qvals):
        out = float(theta[0]) + float(theta[1]) * score
        pos = 2
        for q in qvals:
            out += float(theta[pos]) * np.maximum(0.0, score - float(q))
            pos += 1
        return out

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape
        self.n_features_in_ = p

        alpha, cv_mse = self._select_alpha_cv(X, y)
        dense_intercept, dense_coef = self._fit_raw_ridge(X, y, float(alpha))
        sign_consistency, mean_mag = self._bootstrap_stability(X, y, float(alpha))

        active = self._select_active(dense_coef, sign_consistency, mean_mag)
        sparse_intercept, sparse_coef = self._fit_raw_ridge(X[:, active], y, float(alpha))

        tr_idx, val_idx = self._build_split(n, self.seed)
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        tr_intercept, tr_coef = self._fit_raw_ridge(X_tr[:, active], y_tr, float(alpha))
        val_score = self._predict_linear(X_val[:, active], tr_intercept, tr_coef)
        base_val_mse = float(np.mean((y_val - val_score) ** 2))

        theta_cal, qvals = self._fit_score_calibrator(
            val_score,
            y_val,
            alpha=max(1e-8, float(alpha) * 0.5),
            quantiles=self.calibration_quantiles,
        )
        val_pred_cal = self._apply_score_calibrator(val_score, theta_cal, qvals)
        cal_val_mse = float(np.mean((y_val - val_pred_cal) ** 2))
        use_calibration = (base_val_mse - cal_val_mse) > float(self.min_calibration_gain)

        full_score = self._predict_linear(X[:, active], sparse_intercept, sparse_coef)
        if use_calibration:
            theta_all, qvals_all = self._fit_score_calibrator(
                full_score,
                y,
                alpha=max(1e-8, float(alpha) * 0.5),
                quantiles=self.calibration_quantiles,
            )
            self.calibration_theta_ = theta_all.astype(float)
            self.calibration_knots_ = [float(v) for v in qvals_all]
        else:
            self.calibration_theta_ = np.array([0.0, 1.0], dtype=float)
            self.calibration_knots_ = []

        active_set = set(int(j) for j in active.tolist())
        negligible = np.array([j for j in range(p) if j not in active_set], dtype=int)

        self.linear_intercept_ = float(sparse_intercept)
        self.linear_coef_ = np.asarray(sparse_coef, dtype=float)
        self.active_features_ = np.asarray(active, dtype=int)
        self.negligible_features_ = negligible
        self.alpha_ = float(alpha)
        self.alpha_cv_mse_ = float(cv_mse)
        self.base_val_mse_ = float(base_val_mse)
        self.calibrated_val_mse_ = float(cal_val_mse)
        self.use_calibration_ = bool(use_calibration)
        self.sign_consistency_ = np.asarray(sign_consistency, dtype=float)
        self.mean_abs_coef_bootstrap_ = np.asarray(mean_mag, dtype=float)
        self.dominant_feature_ = int(np.argsort(-np.abs(dense_coef))[0]) if p > 0 else 0
        return self

    def _linear_score(self, X):
        return self._predict_linear(X[:, self.active_features_], self.linear_intercept_, self.linear_coef_)

    def predict(self, X):
        check_is_fitted(self, ["active_features_", "linear_intercept_", "linear_coef_", "calibration_theta_"])
        X = np.asarray(X, dtype=float)
        score = self._linear_score(X)
        if not self.use_calibration_:
            return score
        return self._apply_score_calibrator(score, self.calibration_theta_, self.calibration_knots_)

    def _format_linear(self):
        prec = int(self.display_precision)
        terms = [f"{self.linear_intercept_:+.{prec}f}"]
        for local_i, feat in enumerate(self.active_features_):
            terms.append(f"{float(self.linear_coef_[local_i]):+.{prec}f}*x{int(feat)}")
        return " ".join(terms)

    def __str__(self):
        check_is_fitted(self, ["active_features_", "linear_intercept_", "linear_coef_"])
        prec = int(self.display_precision)
        lines = [
            "Stability Calibrated Linear Map Regressor",
            "Prediction is computed in two steps: a sparse linear score, then an optional 1D hinge calibration on that score.",
            f"Active features: {', '.join(f'x{int(j)}' for j in self.active_features_)}",
        ]
        if len(self.negligible_features_) > 0:
            lines.append("Negligible features: " + ", ".join(f"x{int(j)}" for j in self.negligible_features_))

        lines.extend([
            "",
            "Step 1 (linear score):",
            "  s = " + self._format_linear(),
        ])

        if self.use_calibration_:
            theta = self.calibration_theta_
            lines.extend([
                "",
                "Step 2 (score calibration):",
                f"  y = {float(theta[0]):+.{prec}f} {float(theta[1]):+.{prec}f}*s",
            ])
            for i, knot in enumerate(self.calibration_knots_):
                coef = float(theta[2 + i])
                lines.append(f"      {coef:+.{prec}f}*max(0, s-{float(knot):.{prec}f})")
            lines.append("")
            lines.append("Simulation: compute s from Step 1, then plug s into Step 2.")
        else:
            lines.extend([
                "",
                "No score calibration kept (final prediction is y = s).",
                "Simulation: multiply active features by coefficients, sum, add intercept.",
            ])

        lines.extend([
            "",
            f"Dominant global feature: x{self.dominant_feature_}",
            f"Selected ridge alpha: {self.alpha_:.6g}",
            f"Validation MSE (linear={self.base_val_mse_:.6f}, calibrated={self.calibrated_val_mse_:.6f})",
        ])
        return "\n".join(lines)


import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
StabilityCalibratedLinearMapV1.__module__ = "interpretable_regressor"

model_shorthand_name = "StabilityCalibratedLinearMapV1"
model_description = "Bootstrap sign-stability screened sparse ridge equation with optional two-knot hinge calibration on the 1D linear score, kept only when validation gain is meaningful"
model_defs = [(model_shorthand_name, StabilityCalibratedLinearMapV1())]

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
