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


class DualKnotSparseAdditiveRidgeV1(BaseEstimator, RegressorMixin):
    """
    From-scratch sparse additive ridge in raw-feature space.

    Steps:
    1) Fit dense ridge with GCV-selected alpha.
    2) Keep a compact linear backbone by coefficient-mass screening.
    3) Add at most two one-sided hinge terms on dominant features/quantiles
       only when they reduce training MSE meaningfully.
    """

    def __init__(
        self,
        alpha_grid=None,
        min_features=2,
        max_features=8,
        mass_keep=0.94,
        hinge_top_features=3,
        max_hinges=2,
        min_relative_gain=0.012,
        eps=1e-12,
    ):
        self.alpha_grid = alpha_grid
        self.min_features = min_features
        self.max_features = max_features
        self.mass_keep = mass_keep
        self.hinge_top_features = hinge_top_features
        self.max_hinges = max_hinges
        self.min_relative_gain = min_relative_gain
        self.eps = eps

    @staticmethod
    def _safe_scale(X):
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
        sigma = np.where(sigma > 1e-12, sigma, 1.0)
        return mu.astype(float), sigma.astype(float)

    @staticmethod
    def _ridge_fit(D, y, alpha):
        p = D.shape[1]
        reg = np.eye(p, dtype=float) * float(alpha)
        reg[0, 0] = 0.0
        A = D.T @ D + reg
        b = D.T @ y
        try:
            theta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            theta = np.linalg.pinv(A) @ b
        return np.asarray(theta, dtype=float)

    def _gcv_alpha(self, Xs, y):
        n, p = Xs.shape
        D = np.column_stack([np.ones(n, dtype=float), Xs])
        alphas = (
            np.asarray(self.alpha_grid, dtype=float)
            if self.alpha_grid is not None
            else np.logspace(-5, 3, 18)
        )
        U, s, Vt = np.linalg.svd(D, full_matrices=False)
        Uy = U.T @ y
        reg_mask = np.ones_like(s)
        if reg_mask.size > 0:
            reg_mask[0] = 0.0

        best_alpha = float(alphas[0])
        best_score = None
        for alpha in alphas:
            denom = s**2 + float(alpha) * reg_mask
            f = np.where(denom > 0, (s**2) / denom, 0.0)
            yhat = U @ (f * Uy)
            resid = y - yhat
            rss = float(np.dot(resid, resid))
            df = float(np.sum(f))
            denom_gcv = max((n - df) ** 2, self.eps)
            gcv = rss / denom_gcv
            if (best_score is None) or (gcv < best_score):
                best_score = gcv
                best_alpha = float(alpha)
        return best_alpha

    @staticmethod
    def _hinge(x, thr, side):
        if side > 0:
            return np.maximum(0.0, x - thr)
        return np.maximum(0.0, thr - x)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape
        self.n_features_in_ = p

        mu, sigma = self._safe_scale(X)
        Xs = (X - mu) / sigma

        alpha0 = self._gcv_alpha(Xs, y)
        D0 = np.column_stack([np.ones(n, dtype=float), Xs])
        theta0 = self._ridge_fit(D0, y, alpha=alpha0)
        b0 = float(theta0[0])
        w0 = np.asarray(theta0[1:], dtype=float)

        mass = np.abs(w0)
        order = np.argsort(-mass)
        total_mass = float(np.sum(mass)) + self.eps
        running = 0.0
        active = []
        for j in order:
            active.append(int(j))
            running += float(mass[j])
            if (running / total_mass) >= float(self.mass_keep):
                break
        active = active[: int(self.max_features)]
        if len(active) < int(self.min_features):
            needed = int(self.min_features) - len(active)
            for j in order:
                jj = int(j)
                if jj not in active:
                    active.append(jj)
                    needed -= 1
                    if needed <= 0:
                        break
        active = np.array(sorted(set(active)), dtype=int)

        if active.size == 0:
            active = np.array([int(order[0])], dtype=int)

        X_lin = Xs[:, active]
        D_lin = np.column_stack([np.ones(n, dtype=float), X_lin])
        alpha_lin = self._gcv_alpha(X_lin, y)
        theta_lin = self._ridge_fit(D_lin, y, alpha=alpha_lin)
        pred_lin = D_lin @ theta_lin
        mse_lin = float(np.mean((y - pred_lin) ** 2))

        candidate_feats = [int(active[i]) for i in np.argsort(-np.abs(theta_lin[1:]))[: int(self.hinge_top_features)]]
        candidates = []
        for j in candidate_feats:
            col = X[:, j]
            q25, q50, q75 = np.quantile(col, [0.25, 0.5, 0.75])
            for thr in (float(q25), float(q50), float(q75)):
                candidates.append((j, thr, 1))
                candidates.append((j, thr, -1))

        selected = []
        current_pred = pred_lin.copy()
        current_mse = mse_lin
        hinge_cols = []
        used_keys = set()

        for _ in range(int(self.max_hinges)):
            best = None
            best_gain = 0.0
            for j, thr, side in candidates:
                key = (int(j), round(float(thr), 8), int(side))
                if key in used_keys:
                    continue
                z = self._hinge(X[:, j], thr, side)
                if float(np.std(z)) < 1e-10:
                    continue
                resid = y - current_pred
                denom = float(np.dot(z, z) + alpha_lin)
                if denom <= 0:
                    continue
                gamma = float(np.dot(z, resid) / denom)
                trial_pred = current_pred + gamma * z
                mse_trial = float(np.mean((y - trial_pred) ** 2))
                rel_gain = (current_mse - mse_trial) / max(current_mse, self.eps)
                if rel_gain > best_gain:
                    best_gain = rel_gain
                    best = (j, thr, side, gamma, z)

            if best is None or best_gain < float(self.min_relative_gain):
                break

            j, thr, side, gamma, z = best
            selected.append((int(j), float(thr), int(side)))
            hinge_cols.append(np.asarray(z, dtype=float))
            current_pred = current_pred + float(gamma) * z
            current_mse = float(np.mean((y - current_pred) ** 2))
            used_keys.add((int(j), round(float(thr), 8), int(side)))

        if hinge_cols:
            H = np.column_stack(hinge_cols)
            D = np.column_stack([np.ones(n, dtype=float), X_lin, H])
        else:
            H = np.zeros((n, 0), dtype=float)
            D = np.column_stack([np.ones(n, dtype=float), X_lin])

        alpha_final = self._gcv_alpha(D[:, 1:], y)
        theta = self._ridge_fit(D, y, alpha=alpha_final)

        b_std = float(theta[0])
        w_lin_std = np.asarray(theta[1 : 1 + active.size], dtype=float)
        w_hinge = np.asarray(theta[1 + active.size :], dtype=float)

        coef_raw = np.zeros(p, dtype=float)
        coef_raw[active] = w_lin_std / sigma[active]
        intercept_raw = float(b_std - np.dot(coef_raw, mu))

        self.intercept_ = intercept_raw
        self.coef_ = coef_raw
        self.active_features_ = active
        self.alpha_linear_ = float(alpha_lin)
        self.alpha_final_ = float(alpha_final)
        self.train_mse_linear_ = float(mse_lin)

        hinge_terms = []
        for k, (j, thr, side) in enumerate(selected):
            c = float(w_hinge[k])
            if abs(c) <= 1e-10:
                continue
            hinge_terms.append((int(j), float(thr), int(side), c))
        self.hinge_terms_ = hinge_terms

        pred_final = self.predict(X)
        self.train_mse_final_ = float(np.mean((y - pred_final) ** 2))

        mass_raw = np.abs(self.coef_)
        total = float(np.sum(mass_raw)) + self.eps
        norm = mass_raw / total
        self.meaningful_features_ = np.where(norm >= 0.06)[0].astype(int)
        if self.meaningful_features_.size == 0:
            self.meaningful_features_ = np.array([int(np.argmax(mass_raw))], dtype=int)
        self.negligible_features_ = np.where(norm < 0.01)[0].astype(int)
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "coef_", "hinge_terms_"])
        X = np.asarray(X, dtype=float)
        yhat = self.intercept_ + X @ self.coef_
        for j, thr, side, c in self.hinge_terms_:
            if side > 0:
                yhat = yhat + c * np.maximum(0.0, X[:, j] - thr)
            else:
                yhat = yhat + c * np.maximum(0.0, thr - X[:, j])
        return yhat

    def __str__(self):
        check_is_fitted(self, ["intercept_", "coef_", "hinge_terms_"])
        lines = [
            "Dual-Knot Sparse Additive Ridge Regressor",
            "Exact prediction equation in raw features:",
        ]

        terms = [f"{self.intercept_:+.6f}"]
        for j in np.where(np.abs(self.coef_) > 1e-12)[0]:
            terms.append(f"{float(self.coef_[j]):+.6f}*x{int(j)}")
        for j, thr, side, c in self.hinge_terms_:
            if side > 0:
                terms.append(f"{c:+.6f}*max(0, x{j} - ({thr:+.6f}))")
            else:
                terms.append(f"{c:+.6f}*max(0, ({thr:+.6f}) - x{j})")
        lines.append("  y = " + " ".join(terms))

        lines.append("")
        lines.append("Linear coefficients (sorted by absolute magnitude):")
        for j in np.argsort(-np.abs(self.coef_)):
            lines.append(f"  x{int(j)}: {float(self.coef_[j]):+.6f}")

        lines.append("")
        if self.hinge_terms_:
            lines.append("Hinge terms (piecewise corrections):")
            for idx, (j, thr, side, c) in enumerate(self.hinge_terms_, 1):
                side_txt = "x > threshold" if side > 0 else "x < threshold"
                lines.append(
                    f"  h{idx}: feature=x{j}, threshold={thr:+.6f}, side={side_txt}, coefficient={c:+.6f}"
                )
        else:
            lines.append("Hinge terms: none (pure sparse linear map)")

        lines.append("")
        lines.append(f"GCV alpha (linear screen): {self.alpha_linear_:.6g}")
        lines.append(f"GCV alpha (final refit): {self.alpha_final_:.6g}")
        lines.append(f"Train MSE linear screen: {self.train_mse_linear_:.6f}")
        lines.append(f"Train MSE final model: {self.train_mse_final_:.6f}")

        if len(self.meaningful_features_) > 0:
            lines.append("Meaningful features: " + ", ".join(f"x{int(i)}" for i in self.meaningful_features_))
        if len(self.negligible_features_) > 0:
            lines.append("Negligible features: " + ", ".join(f"x{int(i)}" for i in self.negligible_features_))
        return "\n".join(lines)


import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
DualKnotSparseAdditiveRidgeV1.__module__ = "interpretable_regressor"

model_shorthand_name = "DualKnotSparseAdditiveRidgeV1"
model_description = "From-scratch GCV ridge with coefficient-mass sparse linear backbone plus at most two quantile-knot one-sided hinge corrections accepted only with meaningful fit gain"
model_defs = [(model_shorthand_name, DualKnotSparseAdditiveRidgeV1())]

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
