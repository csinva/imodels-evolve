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


class SparseLatentSplineRidgeV1(BaseEstimator, RegressorMixin):
    """
    Sparse ridge backbone with one latent spline channel.

    The model is intentionally compact and simulatable:
    y = b + sum_j w_j * x_j + a2 * z^2 + ah * max(0, z - t)
    where z is a weighted sum of a few dominant features.
    """

    def __init__(
        self,
        alpha_grid=None,
        cv_folds=3,
        k_candidates=(3, 5, 8, 12),
        latent_dims=4,
        val_fraction=0.2,
        min_relative_gain=0.004,
        random_state=42,
        display_precision=5,
    ):
        self.alpha_grid = alpha_grid
        self.cv_folds = cv_folds
        self.k_candidates = k_candidates
        self.latent_dims = latent_dims
        self.val_fraction = val_fraction
        self.min_relative_gain = min_relative_gain
        self.random_state = random_state
        self.display_precision = display_precision

    @staticmethod
    def _safe_scale(X):
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
        sigma = np.where(sigma > 1e-12, sigma, 1.0)
        return mu.astype(float), sigma.astype(float)

    @staticmethod
    def _solve_ridge(D, y, alpha):
        reg = np.eye(D.shape[1], dtype=float)
        reg[0, 0] = 0.0
        A = D.T @ D + float(alpha) * reg
        b = D.T @ y
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(A) @ b

    def _fit_linear(self, Xs, y, alpha):
        n = Xs.shape[0]
        D = np.column_stack([np.ones(n, dtype=float), Xs])
        theta = self._solve_ridge(D, y, alpha)
        return theta

    def _predict_linear(self, Xs, theta):
        return float(theta[0]) + Xs @ np.asarray(theta[1:], dtype=float)

    def _select_alpha_cv(self, Xs, y):
        alphas = np.asarray(self.alpha_grid, dtype=float) if self.alpha_grid is not None else np.logspace(-5, 2, 12)
        n = Xs.shape[0]
        if n < 25:
            return float(alphas[0]), float("nan")

        n_folds = int(max(2, min(int(self.cv_folds), 5)))
        rng = np.random.RandomState(self.random_state)
        idx = np.arange(n)
        rng.shuffle(idx)
        folds = np.array_split(idx, n_folds)

        best_alpha = float(alphas[len(alphas) // 2])
        best_mse = float("inf")
        for alpha in alphas:
            mses = []
            for k in range(n_folds):
                va = folds[k]
                tr = np.concatenate([folds[j] for j in range(n_folds) if j != k])
                if len(tr) < 8 or len(va) < 4:
                    continue
                theta = self._fit_linear(Xs[tr], y[tr], float(alpha))
                pred = self._predict_linear(Xs[va], theta)
                mses.append(float(np.mean((y[va] - pred) ** 2)))
            if not mses:
                continue
            mse = float(np.mean(mses))
            if mse < best_mse:
                best_mse = mse
                best_alpha = float(alpha)
        return best_alpha, best_mse

    @staticmethod
    def _split_indices(n, val_fraction, seed):
        rng = np.random.RandomState(seed)
        idx = np.arange(n)
        rng.shuffle(idx)
        n_val = max(16, int(float(val_fraction) * n))
        n_val = min(max(1, n - 8), n_val)
        return idx[n_val:], idx[:n_val]

    @staticmethod
    def _latent_score(Xs_sub, latent_weights):
        return Xs_sub @ latent_weights

    @staticmethod
    def _build_nonlinear_terms(z, knot):
        quad = z ** 2
        hinge = np.maximum(0.0, z - float(knot))
        return quad, hinge

    def _fit_with_terms(self, Xs_sub, Xs_latent, y, alpha, add_quad, add_hinge, knot, latent_weights):
        z = self._latent_score(Xs_latent, latent_weights)
        cols = [np.ones(Xs_sub.shape[0], dtype=float), Xs_sub]
        if add_quad or add_hinge:
            quad, hinge = self._build_nonlinear_terms(z, knot)
            if add_quad:
                cols.append(quad)
            if add_hinge:
                cols.append(hinge)
        D = np.column_stack(cols)

        reg = np.full(D.shape[1], float(alpha), dtype=float)
        reg[0] = 0.0
        if D.shape[1] > 1 + Xs_sub.shape[1]:
            reg[1 + Xs_sub.shape[1] :] = max(1e-8, float(alpha) * 0.5)

        A = D.T @ D + np.diag(reg)
        b = D.T @ y
        try:
            theta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            theta = np.linalg.pinv(A) @ b

        pred = D @ theta
        return theta.astype(float), pred.astype(float)

    def _predict_with_terms(self, Xs_sub, Xs_latent, theta, add_quad, add_hinge, knot, latent_weights):
        z = self._latent_score(Xs_latent, latent_weights)
        cols = [np.ones(Xs_sub.shape[0], dtype=float), Xs_sub]
        if add_quad or add_hinge:
            quad, hinge = self._build_nonlinear_terms(z, knot)
            if add_quad:
                cols.append(quad)
            if add_hinge:
                cols.append(hinge)
        D = np.column_stack(cols)
        return D @ theta

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape
        self.n_features_in_ = p

        mu, sigma = self._safe_scale(X)
        Xs = (X - mu) / sigma

        alpha, cv_mse = self._select_alpha_cv(Xs, y)
        theta_dense = self._fit_linear(Xs, y, alpha)
        dense_coef_std = theta_dense[1:]

        tr_idx, val_idx = self._split_indices(n, self.val_fraction, self.random_state)
        X_tr = Xs[tr_idx]
        y_tr = y[tr_idx]
        X_val = Xs[val_idx]
        y_val = y[val_idx]

        order = np.argsort(-np.abs(dense_coef_std))
        k_pool = [int(k) for k in self.k_candidates]
        k_pool.append(p)
        k_pool = sorted(set(max(1, min(p, k)) for k in k_pool))

        best = None
        best_val_mse = float("inf")

        for k in k_pool:
            idx = order[:k]
            theta_tr = self._fit_linear(X_tr[:, idx], y_tr, alpha)
            val_pred_base = self._predict_linear(X_val[:, idx], theta_tr)
            base_mse = float(np.mean((y_val - val_pred_base) ** 2))

            latent_k = int(max(1, min(self.latent_dims, k)))
            latent_local = np.argsort(-np.abs(theta_tr[1:]))[:latent_k]
            latent_weights = theta_tr[1:][latent_local].copy()
            norm = float(np.linalg.norm(latent_weights))
            if norm <= 1e-12:
                latent_weights = np.ones(latent_k, dtype=float) / np.sqrt(latent_k)
            else:
                latent_weights = latent_weights / norm

            idx_latent = idx[latent_local]
            z_tr = self._latent_score(X_tr[:, idx_latent], latent_weights)
            knots = np.unique(np.quantile(z_tr, [0.25, 0.5, 0.75]))

            local_best = {
                "k": int(k),
                "idx": idx.copy(),
                "idx_latent": idx_latent.copy(),
                "latent_weights": latent_weights.copy(),
                "add_quad": False,
                "add_hinge": False,
                "knot": 0.0,
                "theta": theta_tr.copy(),
                "val_mse": base_mse,
                "base_mse": base_mse,
            }

            for add_quad in (False, True):
                for add_hinge in (False, True):
                    if not add_quad and not add_hinge:
                        continue
                    for knot in knots:
                        theta_nl, _ = self._fit_with_terms(
                            X_tr[:, idx], X_tr[:, idx_latent], y_tr, alpha, add_quad, add_hinge, float(knot), latent_weights
                        )
                        pred_val = self._predict_with_terms(
                            X_val[:, idx], X_val[:, idx_latent], theta_nl, add_quad, add_hinge, float(knot), latent_weights
                        )
                        mse_val = float(np.mean((y_val - pred_val) ** 2))
                        if mse_val < local_best["val_mse"]:
                            local_best.update(
                                {
                                    "add_quad": bool(add_quad),
                                    "add_hinge": bool(add_hinge),
                                    "knot": float(knot),
                                    "theta": theta_nl.copy(),
                                    "val_mse": mse_val,
                                }
                            )

            if local_best["val_mse"] < best_val_mse:
                best_val_mse = local_best["val_mse"]
                best = local_best

        if best is None:
            raise RuntimeError("Model selection failed")

        rel_gain = (best["base_mse"] - best["val_mse"]) / max(best["base_mse"], 1e-12)
        use_nonlinear = bool((best["add_quad"] or best["add_hinge"]) and (rel_gain >= float(self.min_relative_gain)))

        idx = best["idx"]
        idx_latent = best["idx_latent"]
        latent_weights = best["latent_weights"]
        add_quad = bool(best["add_quad"]) if use_nonlinear else False
        add_hinge = bool(best["add_hinge"]) if use_nonlinear else False
        knot = float(best["knot"]) if use_nonlinear else 0.0

        theta_final, pred_train = self._fit_with_terms(
            Xs[:, idx], Xs[:, idx_latent], y, alpha, add_quad, add_hinge, knot, latent_weights
        )

        coef_std = np.zeros(p, dtype=float)
        coef_std[idx] = theta_final[1 : 1 + len(idx)]
        coef_raw = coef_std / sigma
        intercept_raw = float(theta_final[0] - np.dot(coef_raw, mu))

        self.feature_mean_ = mu
        self.feature_scale_ = sigma
        self.alpha_ = float(alpha)
        self.alpha_cv_mse_ = float(cv_mse)
        self.selected_features_ = np.asarray(idx, dtype=int)
        self.latent_features_ = np.asarray(idx_latent, dtype=int)
        self.latent_weights_ = np.asarray(latent_weights, dtype=float)
        self.linear_coef_raw_ = coef_raw.astype(float)
        self.intercept_raw_ = intercept_raw
        self.add_quad_ = bool(add_quad)
        self.add_hinge_ = bool(add_hinge)
        self.knot_std_ = float(knot)

        pos = 1 + len(idx)
        self.quad_coef_ = float(theta_final[pos]) if add_quad else 0.0
        pos += 1 if add_quad else 0
        self.hinge_coef_ = float(theta_final[pos]) if add_hinge else 0.0

        self.validation_mse_ = float(best["val_mse"])
        self.validation_base_mse_ = float(best["base_mse"])
        self.train_mse_ = float(np.mean((y - pred_train) ** 2))
        self.nonlinear_kept_ = bool(use_nonlinear)
        self.dominant_feature_ = int(order[0]) if p > 0 else 0

        return self

    def _core_predict(self, X):
        linear = float(self.intercept_raw_) + np.asarray(X, dtype=float) @ self.linear_coef_raw_
        if not self.nonlinear_kept_:
            return linear

        X = np.asarray(X, dtype=float)
        Xs = (X - self.feature_mean_) / self.feature_scale_
        z = Xs[:, self.latent_features_] @ self.latent_weights_
        out = linear.copy()
        if self.add_quad_:
            out = out + float(self.quad_coef_) * (z ** 2)
        if self.add_hinge_:
            out = out + float(self.hinge_coef_) * np.maximum(0.0, z - float(self.knot_std_))
        return out

    def predict(self, X):
        check_is_fitted(self, ["linear_coef_raw_", "intercept_raw_", "selected_features_"])
        X = np.asarray(X, dtype=float)
        return self._core_predict(X)

    def _format_linear(self):
        prec = int(self.display_precision)
        pieces = [f"{self.intercept_raw_:+.{prec}f}"]
        for j in self.selected_features_:
            pieces.append(f"{float(self.linear_coef_raw_[j]):+.{prec}f}*x{int(j)}")
        return " ".join(pieces)

    def _format_latent(self):
        prec = int(self.display_precision)
        parts = []
        for i, j in enumerate(self.latent_features_):
            parts.append(f"{float(self.latent_weights_[i]):+.{prec}f}*z_x{int(j)}")
        return " ".join(parts)

    def __str__(self):
        check_is_fitted(self, ["selected_features_", "linear_coef_raw_", "intercept_raw_"])
        prec = int(self.display_precision)

        negligible = [j for j in range(self.n_features_in_) if j not in set(self.selected_features_.tolist())]

        lines = [
            "Sparse Latent Spline Ridge Regressor",
            "Prediction is a sparse linear equation plus an optional one-channel latent spline correction.",
            f"Active linear features: {', '.join(f'x{int(j)}' for j in self.selected_features_)}",
        ]
        if negligible:
            lines.append("Negligible features: " + ", ".join(f"x{int(j)}" for j in negligible))

        lines.extend([
            "",
            "Step 1 (sparse linear backbone in raw features):",
            "  y_linear = " + self._format_linear(),
        ])

        if self.nonlinear_kept_:
            lines.extend([
                "",
                "Step 2 (latent channel on standardized active features):",
                "  z = " + self._format_latent(),
            ])
            if self.add_quad_ and self.add_hinge_:
                lines.append(
                    f"  y = y_linear {self.quad_coef_:+.{prec}f}*z^2 {self.hinge_coef_:+.{prec}f}*max(0, z-{self.knot_std_:.{prec}f})"
                )
            elif self.add_quad_:
                lines.append(f"  y = y_linear {self.quad_coef_:+.{prec}f}*z^2")
            elif self.add_hinge_:
                lines.append(f"  y = y_linear {self.hinge_coef_:+.{prec}f}*max(0, z-{self.knot_std_:.{prec}f})")
            lines.append("Simulation: compute Step 1, then Step 2.")
        else:
            lines.extend([
                "",
                "No latent spline kept (final prediction is y = y_linear).",
                "Simulation: multiply active features by coefficients, sum, add intercept.",
            ])

        lines.extend([
            "",
            f"Dominant global feature: x{self.dominant_feature_}",
            f"Selected ridge alpha: {self.alpha_:.6g}",
            f"Validation MSE (base={self.validation_base_mse_:.6f}, selected={self.validation_mse_:.6f})",
        ])
        return "\n".join(lines)


import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparseLatentSplineRidgeV1.__module__ = "interpretable_regressor"

model_shorthand_name = "SparseLatentSplineRidgeV1"
model_description = "CV ridge with validation-selected sparse feature subset plus one latent projection spline correction (quadratic and/or hinge), rendered as an explicit simulatable equation"
model_defs = [(model_shorthand_name, SparseLatentSplineRidgeV1())]
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
