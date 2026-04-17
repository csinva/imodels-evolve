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


class BiRegionSparseLinearRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse linear regressor with an optional single raw-feature threshold split.
    If a split materially improves validation error, two sparse linear equations
    are fit (left/right region); otherwise a single global sparse equation is used.
    """

    def __init__(
        self,
        alpha_grid=None,
        max_active_features=8,
        coef_rel_threshold=0.1,
        val_frac=0.2,
        threshold_quantiles=(0.2, 0.35, 0.5, 0.65, 0.8),
        max_split_features=3,
        min_leaf_frac=0.15,
        min_split_gain=0.01,
        random_state=42,
    ):
        self.alpha_grid = alpha_grid
        self.max_active_features = max_active_features
        self.coef_rel_threshold = coef_rel_threshold
        self.val_frac = val_frac
        self.threshold_quantiles = threshold_quantiles
        self.max_split_features = max_split_features
        self.min_leaf_frac = min_leaf_frac
        self.min_split_gain = min_split_gain
        self.random_state = random_state

    @staticmethod
    def _safe_scale(X):
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
        sigma = np.where(sigma < 1e-8, 1.0, sigma)
        return mu, sigma

    @staticmethod
    def _ridge_fit(X, y, alpha):
        n, p = X.shape
        D = np.hstack([np.ones((n, 1), dtype=float), X])
        reg = np.zeros(p + 1, dtype=float)
        reg[1:] = max(float(alpha), 0.0)
        A = D.T @ D + np.diag(reg)
        b = D.T @ y
        try:
            beta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(A) @ b
        return float(beta[0]), np.asarray(beta[1:], dtype=float)

    @staticmethod
    def _mse(y_true, y_pred):
        return float(np.mean((y_true - y_pred) ** 2))

    def _select_alpha_gcv(self, X, y):
        n = X.shape[0]
        if self.alpha_grid is None:
            grid = np.logspace(-4, 2, 11)
        else:
            grid = np.asarray(self.alpha_grid, dtype=float)
        grid = np.maximum(grid, 1e-10)

        y_centered = y - np.mean(y)
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        Uy = U.T @ y_centered
        s2 = s ** 2

        best_alpha = float(grid[0])
        best_gcv = float("inf")
        for alpha in grid:
            shrink = s2 / (s2 + alpha)
            yhat_centered = U @ (shrink * Uy)
            resid = y_centered - yhat_centered
            mse = float(np.mean(resid ** 2))
            df = float(np.sum(shrink))
            denom = max((1.0 - df / max(n, 1)) ** 2, 1e-8)
            gcv = mse / denom
            if gcv < best_gcv:
                best_gcv = gcv
                best_alpha = float(alpha)
        return best_alpha, best_gcv

    def _sparsify(self, coef_std):
        p = len(coef_std)
        abs_w = np.abs(coef_std)
        order = np.argsort(abs_w)[::-1]
        if p == 0:
            return np.array([], dtype=int)
        top = abs_w[order[0]]
        thresh = max(float(self.coef_rel_threshold) * top, 1e-10)
        keep = [int(j) for j in order if abs_w[j] >= thresh]
        keep = keep[: max(1, int(self.max_active_features))]
        if not keep:
            keep = [int(order[0])]
        return np.array(sorted(set(keep)), dtype=int)

    def _fit_sparse_linear(self, X_std, y, alpha):
        b0, w0 = self._ridge_fit(X_std, y, alpha)
        active = self._sparsify(w0)
        Xa = X_std[:, active] if len(active) > 0 else np.zeros((len(X_std), 0))
        b, w_small = self._ridge_fit(Xa, y, alpha)
        w_full = np.zeros(X_std.shape[1], dtype=float)
        if len(active) > 0:
            w_full[active] = w_small
        return float(b), np.asarray(w_full, dtype=float), np.asarray(active, dtype=int)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape
        self.n_features_in_ = p

        mu, sigma = self._safe_scale(X)
        Xs = (X - mu) / sigma

        rng = np.random.RandomState(self.random_state)
        perm = rng.permutation(n)
        n_val = max(20, int(self.val_frac * n))
        if n - n_val < 20:
            n_val = max(1, n // 5)
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]
        if len(tr_idx) < 20:
            tr_idx = perm
            val_idx = perm

        Xtr = Xs[tr_idx]
        ytr = y[tr_idx]
        Xval = Xs[val_idx]
        yval = y[val_idx]

        alpha, gcv_score = self._select_alpha_gcv(Xtr, ytr)
        b_std_global, w_std_global, active_global = self._fit_sparse_linear(Xtr, ytr, alpha)
        pred_val_lin = b_std_global + Xval @ w_std_global
        mse_lin = self._mse(yval, pred_val_lin)

        best_split = None
        best_mse = mse_lin
        min_leaf = max(10, int(float(self.min_leaf_frac) * len(Xtr)))

        feat_order = np.argsort(np.abs(w_std_global))[::-1]
        cand_features = [int(j) for j in feat_order if abs(float(w_std_global[j])) > 1e-12]
        cand_features = cand_features[: max(1, int(self.max_split_features))]

        for j in cand_features:
            xj_tr = Xtr[:, j]
            xj_val = Xval[:, j]
            knots = np.unique(np.quantile(xj_tr, np.asarray(self.threshold_quantiles, dtype=float)))
            for knot in knots:
                left_tr = xj_tr <= float(knot)
                right_tr = ~left_tr
                left_val = xj_val <= float(knot)
                right_val = ~left_val
                if left_tr.sum() < min_leaf or right_tr.sum() < min_leaf:
                    continue
                if left_val.sum() == 0 or right_val.sum() == 0:
                    continue

                b_l, w_l, active_l = self._fit_sparse_linear(Xtr[left_tr], ytr[left_tr], alpha)
                b_r, w_r, active_r = self._fit_sparse_linear(Xtr[right_tr], ytr[right_tr], alpha)

                pred = np.empty_like(yval, dtype=float)
                pred[left_val] = b_l + Xval[left_val] @ w_l
                pred[right_val] = b_r + Xval[right_val] @ w_r
                mse = self._mse(yval, pred)

                if mse < best_mse:
                    best_mse = mse
                    best_split = {
                        "feature": int(j),
                        "knot_std": float(knot),
                        "b_left_std": float(b_l),
                        "w_left_std": np.asarray(w_l, dtype=float),
                        "active_left": np.asarray(active_l, dtype=int),
                        "b_right_std": float(b_r),
                        "w_right_std": np.asarray(w_r, dtype=float),
                        "active_right": np.asarray(active_r, dtype=int),
                    }

        rel_gain = (mse_lin - best_mse) / max(mse_lin, 1e-12)
        self.use_split_ = best_split is not None and rel_gain >= float(self.min_split_gain)

        coef_raw_global = w_std_global / sigma
        intercept_raw_global = float(b_std_global - np.dot(coef_raw_global, mu))
        self.intercept_ = intercept_raw_global
        self.coef_ = coef_raw_global
        self.active_features_ = np.asarray(active_global, dtype=int)

        if self.use_split_:
            j = int(best_split["feature"])
            thr_raw = float(mu[j] + sigma[j] * best_split["knot_std"])

            coef_left_raw = best_split["w_left_std"] / sigma
            coef_right_raw = best_split["w_right_std"] / sigma
            b_left_raw = float(best_split["b_left_std"] - np.dot(coef_left_raw, mu))
            b_right_raw = float(best_split["b_right_std"] - np.dot(coef_right_raw, mu))

            self.split_feature_ = j
            self.split_threshold_ = thr_raw
            self.intercept_left_ = b_left_raw
            self.intercept_right_ = b_right_raw
            self.coef_left_ = coef_left_raw
            self.coef_right_ = coef_right_raw
            self.active_left_ = np.asarray(best_split["active_left"], dtype=int)
            self.active_right_ = np.asarray(best_split["active_right"], dtype=int)

        self.alpha_ = alpha
        self.gcv_score_ = gcv_score
        self.val_mse_linear_ = mse_lin
        self.val_mse_best_ = best_mse
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "coef_", "active_features_"])
        X = np.asarray(X, dtype=float)
        if not getattr(self, "use_split_", False):
            return self.intercept_ + X @ self.coef_

        j = int(self.split_feature_)
        left = X[:, j] <= float(self.split_threshold_)
        out = np.empty(X.shape[0], dtype=float)
        if left.any():
            out[left] = float(self.intercept_left_) + X[left] @ self.coef_left_
        if (~left).any():
            out[~left] = float(self.intercept_right_) + X[~left] @ self.coef_right_
        return out

    @staticmethod
    def _linear_terms(coef, max_terms=8):
        active = [j for j in range(len(coef)) if abs(float(coef[j])) > 1e-10]
        active = sorted(active, key=lambda j: -abs(float(coef[j])))[: int(max_terms)]
        return [f"{float(coef[j]):+.6f}*x{j}" for j in active], active

    def __str__(self):
        check_is_fitted(self, ["intercept_", "coef_", "active_features_"])
        terms_g, active_g = self._linear_terms(self.coef_)
        expr_g = f"{float(self.intercept_):.6f}"
        if terms_g:
            expr_g += " " + " ".join(terms_g)

        lines = [
            "Bi-Region Sparse Linear Regressor",
            f"Chosen ridge alpha: {self.alpha_:.6g}",
            "Raw-feature equations:",
        ]
        if not getattr(self, "use_split_", False):
            lines.append(f"y = {expr_g}")
            lines.append(f"Active features: {', '.join(f'x{j}' for j in active_g) if active_g else '(none)'}")
            return "\n".join(lines)

        terms_l, active_l = self._linear_terms(self.coef_left_)
        terms_r, active_r = self._linear_terms(self.coef_right_)
        expr_l = f"{float(self.intercept_left_):.6f}"
        expr_r = f"{float(self.intercept_right_):.6f}"
        if terms_l:
            expr_l += " " + " ".join(terms_l)
        if terms_r:
            expr_r += " " + " ".join(terms_r)

        lines.extend([
            f"if x{int(self.split_feature_)} <= {float(self.split_threshold_):.6f}:",
            f"  y = {expr_l}",
            "else:",
            f"  y = {expr_r}",
            f"Left active features: {', '.join(f'x{j}' for j in active_l) if active_l else '(none)'}",
            f"Right active features: {', '.join(f'x{j}' for j in active_r) if active_r else '(none)'}",
            f"Fallback global equation: y = {expr_g}",
            f"Global active features: {', '.join(f'x{j}' for j in active_g) if active_g else '(none)'}",
        ])
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
BiRegionSparseLinearRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "BiRegionSparseLinearV1"
model_description = "Sparse ridge equation with optional single-feature threshold split into two region-specific sparse linear equations accepted only when validation gain is meaningful"
model_defs = [(model_shorthand_name, BiRegionSparseLinearRegressor())]


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
