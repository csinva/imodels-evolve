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


class CalibratedSparseSingleIndexV1(BaseEstimator, RegressorMixin):
    """
    Sparse single-index ridge with optional two-knot score calibration.

    1) Fit a global ridge and keep a compact active subset by coefficient mass.
    2) Refit ridge on active features only for a short explicit score equation.
    3) Optionally calibrate that 1D score with a two-knot hinge map selected
       on validation, yielding lightweight nonlinearity without losing traceability.
    """

    def __init__(
        self,
        alpha_grid=None,
        max_active_features=4,
        keep_mass=0.94,
        n_knot_candidates=6,
        min_calibration_gain=1e-4,
        seed=42,
        display_precision=5,
    ):
        self.alpha_grid = alpha_grid
        self.max_active_features = max_active_features
        self.keep_mass = keep_mass
        self.n_knot_candidates = n_knot_candidates
        self.min_calibration_gain = min_calibration_gain
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

    def _predict_raw_linear(self, X, intercept, coef):
        return float(intercept) + np.asarray(X, dtype=float) @ np.asarray(coef, dtype=float)

    @staticmethod
    def _design_calibration(z, k1, k2):
        z = np.asarray(z, dtype=float).ravel()
        return np.column_stack(
            [
                np.ones_like(z),
                z,
                np.maximum(0.0, z - float(k1)),
                np.maximum(0.0, z - float(k2)),
            ]
        )

    @staticmethod
    def _fit_lstsq(B, y):
        B = np.asarray(B, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        try:
            beta, *_ = np.linalg.lstsq(B, y, rcond=None)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(B) @ y
        return np.asarray(beta, dtype=float)

    def _select_alpha_holdout(self, X, y):
        alphas = (
            np.asarray(self.alpha_grid, dtype=float)
            if self.alpha_grid is not None
            else np.logspace(-5, 2, 12)
        )
        n = X.shape[0]
        if n < 12:
            return float(alphas[0]), float("nan")

        rng = np.random.RandomState(self.seed)
        idx = np.arange(n)
        rng.shuffle(idx)
        n_val = max(8, int(0.25 * n))
        val_idx = idx[:n_val]
        tr_idx = idx[n_val:]

        best_alpha = float(alphas[0])
        best_mse = float("inf")
        for alpha in alphas:
            intercept, coef = self._fit_raw_ridge(X[tr_idx], y[tr_idx], float(alpha))
            preds = self._predict_raw_linear(X[val_idx], intercept, coef)
            mse = float(np.mean((y[val_idx] - preds) ** 2))
            if mse < best_mse:
                best_mse = mse
                best_alpha = float(alpha)
        return best_alpha, best_mse

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

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape
        self.n_features_in_ = p

        alpha, alpha_val_mse = self._select_alpha_holdout(X, y)
        global_intercept, global_coef = self._fit_raw_ridge(X, y, float(alpha))
        order = np.argsort(-np.abs(global_coef))
        active = self._select_active_features(global_coef)
        active_set = set(active.tolist())
        negligible = np.array([j for j in range(p) if j not in active_set], dtype=int)

        # Final sparse linear score on full data.
        sparse_intercept, sparse_coef = self._fit_raw_ridge(X[:, active], y, float(alpha))
        score_all = self._predict_raw_linear(X[:, active], sparse_intercept, sparse_coef)

        # Holdout selection for optional nonlinear score calibration.
        rng = np.random.RandomState(self.seed)
        idx = np.arange(n)
        rng.shuffle(idx)
        n_val = max(8, int(0.25 * n))
        val_idx = idx[:n_val]
        tr_idx = idx[n_val:]

        X_tr = X[tr_idx]
        y_tr = y[tr_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]

        tr_intercept, tr_coef = self._fit_raw_ridge(X_tr[:, active], y_tr, float(alpha))
        score_tr = self._predict_raw_linear(X_tr[:, active], tr_intercept, tr_coef)
        score_val = self._predict_raw_linear(X_val[:, active], tr_intercept, tr_coef)
        base_val_pred = score_val
        base_val_mse = float(np.mean((y_val - base_val_pred) ** 2))

        q = np.linspace(0.15, 0.85, int(max(3, self.n_knot_candidates)))
        cand = np.unique(np.quantile(score_tr, q))
        best_cal = {
            "mse": base_val_mse,
            "k1": None,
            "k2": None,
            "beta": np.array([0.0, 1.0, 0.0, 0.0], dtype=float),
        }
        if len(cand) >= 2:
            for i in range(len(cand)):
                for j in range(i + 1, len(cand)):
                    k1 = float(cand[i])
                    k2 = float(cand[j])
                    B_tr = self._design_calibration(score_tr, k1, k2)
                    beta = self._fit_lstsq(B_tr, y_tr)
                    B_val = self._design_calibration(score_val, k1, k2)
                    pred_val = B_val @ beta
                    mse = float(np.mean((y_val - pred_val) ** 2))
                    if mse < best_cal["mse"]:
                        best_cal = {"mse": mse, "k1": k1, "k2": k2, "beta": beta}

        use_calibration = (
            best_cal["k1"] is not None
            and (base_val_mse - best_cal["mse"]) > float(self.min_calibration_gain)
        )

        self.use_calibration_ = bool(use_calibration)
        if self.use_calibration_:
            self.knot1_ = float(best_cal["k1"])
            self.knot2_ = float(best_cal["k2"])
            B_all = self._design_calibration(score_all, self.knot1_, self.knot2_)
            self.calibration_beta_ = self._fit_lstsq(B_all, y)
        else:
            self.knot1_ = 0.0
            self.knot2_ = 0.0
            self.calibration_beta_ = np.array([0.0, 1.0, 0.0, 0.0], dtype=float)

        self.active_features_ = np.asarray(active, dtype=int)
        self.negligible_features_ = negligible
        self.global_intercept_ = float(global_intercept)
        self.global_coef_ = np.asarray(global_coef, dtype=float)
        self.sorted_features_ = order.astype(int)
        self.score_intercept_ = float(sparse_intercept)
        self.score_coef_active_ = np.asarray(sparse_coef, dtype=float)
        self.alpha_ = float(alpha)
        self.alpha_holdout_mse_ = float(alpha_val_mse)
        self.validation_mse_linear_ = float(base_val_mse)
        self.validation_mse_calibrated_ = float(best_cal["mse"])
        self.dominant_feature_ = int(order[0]) if p > 0 else 0
        return self

    def predict(self, X):
        check_is_fitted(self, ["active_features_", "score_intercept_", "score_coef_active_", "calibration_beta_"])
        X = np.asarray(X, dtype=float)
        score = self._predict_raw_linear(X[:, self.active_features_], self.score_intercept_, self.score_coef_active_)
        if not self.use_calibration_:
            return score
        B = self._design_calibration(score, self.knot1_, self.knot2_)
        return B @ self.calibration_beta_

    def _format_linear_score(self):
        prec = int(self.display_precision)
        terms = [f"{float(self.score_intercept_):+.{prec}f}"]
        for local_i, feat in enumerate(self.active_features_):
            terms.append(f"{float(self.score_coef_active_[local_i]):+.{prec}f}*x{int(feat)}")
        return " ".join(terms)

    def __str__(self):
        check_is_fitted(self, ["active_features_", "score_intercept_", "score_coef_active_", "calibration_beta_"])
        prec = int(self.display_precision)
        lines = [
            "Calibrated Sparse Single-Index Regressor",
            "Prediction: short sparse linear score, optionally passed through a tiny two-knot hinge calibrator.",
            f"Active features: {', '.join(f'x{int(j)}' for j in self.active_features_)}",
        ]
        if len(self.negligible_features_) > 0:
            lines.append("Negligible features: " + ", ".join(f"x{int(j)}" for j in self.negligible_features_))

        lines.extend(
            [
                "",
                "Score equation:",
                "  s = " + self._format_linear_score(),
            ]
        )

        if self.use_calibration_:
            b0, b1, b2, b3 = [float(v) for v in self.calibration_beta_]
            lines.extend(
                [
                    "",
                    "Calibrated prediction:",
                    f"  y = {b0:+.{prec}f} {b1:+.{prec}f}*s {b2:+.{prec}f}*max(0, s-{self.knot1_:.{prec}f}) {b3:+.{prec}f}*max(0, s-{self.knot2_:.{prec}f})",
                    "",
                    "Simulation: compute s from the sparse linear score, then apply the hinge terms at the two score knots.",
                ]
            )
        else:
            lines.extend(
                [
                    "",
                    "No nonlinear calibration kept (linear score is final prediction):",
                    "  y = s",
                    "",
                    "Simulation: multiply active features by coefficients, sum, and add intercept.",
                ]
            )

        lines.append("")
        lines.append(f"Dominant global feature: x{self.dominant_feature_}")
        lines.append(f"Selected ridge alpha: {self.alpha_:.6g}")
        return "\n".join(lines)


import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
CalibratedSparseSingleIndexV1.__module__ = "interpretable_regressor"

model_shorthand_name = "CalibratedSparseSingleIndexV1"
model_description = "Sparse coefficient-mass ridge score on raw features with optional validation-gated two-knot hinge calibration over the 1D score for lightweight nonlinear correction"
model_defs = [(model_shorthand_name, CalibratedSparseSingleIndexV1())]

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
