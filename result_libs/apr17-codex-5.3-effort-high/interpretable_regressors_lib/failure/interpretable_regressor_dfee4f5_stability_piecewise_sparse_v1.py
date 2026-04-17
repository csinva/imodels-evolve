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


class StabilityPiecewiseSparseRegressor(BaseEstimator, RegressorMixin):
    """
    Sign-stability screened sparse linear model with an optional single split.

    The model remains compact and explicitly simulatable:
      - global sparse linear equation, or
      - one-threshold piecewise sparse linear equation.
    """

    def __init__(
        self,
        screening_lambda=1.0,
        linear_lambda=0.25,
        piece_lambda=0.35,
        n_bootstraps=9,
        bootstrap_frac=0.72,
        max_linear_features=6,
        top_split_features=3,
        min_stability=0.4,
        min_leaf=24,
        min_relative_gain=0.01,
        random_state=42,
    ):
        self.screening_lambda = screening_lambda
        self.linear_lambda = linear_lambda
        self.piece_lambda = piece_lambda
        self.n_bootstraps = n_bootstraps
        self.bootstrap_frac = bootstrap_frac
        self.max_linear_features = max_linear_features
        self.top_split_features = top_split_features
        self.min_stability = min_stability
        self.min_leaf = min_leaf
        self.min_relative_gain = min_relative_gain
        self.random_state = random_state

    @staticmethod
    def _safe_scale(X):
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
        sigma = np.where(sigma < 1e-8, 1.0, sigma)
        return mu, sigma

    @staticmethod
    def _solve_ridge(X, y, lam):
        n, p = X.shape
        D = np.hstack([np.ones((n, 1), dtype=float), X])
        reg = np.zeros(p + 1, dtype=float)
        if p > 0:
            reg[1:] = max(float(lam), 0.0)
        A = D.T @ D + np.diag(reg)
        b = D.T @ y
        try:
            beta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(A) @ b
        pred = D @ beta
        return float(beta[0]), np.asarray(beta[1:], dtype=float), pred

    @staticmethod
    def _to_raw_params(active_idx, b_std, w_std, mu, sigma, n_features):
        coef_raw = np.zeros(n_features, dtype=float)
        if len(active_idx) > 0:
            coef_raw[np.asarray(active_idx, dtype=int)] = np.asarray(w_std, dtype=float) / sigma[np.asarray(active_idx, dtype=int)]
        intercept_raw = float(b_std - np.dot(coef_raw, mu))
        return intercept_raw, coef_raw

    def _screen_features(self, Xs, y):
        n, p = Xs.shape
        rng = np.random.RandomState(self.random_state)
        boot_size = max(16, int(float(self.bootstrap_frac) * n))
        weights = []
        for _ in range(max(1, int(self.n_bootstraps))):
            idx = rng.randint(0, n, size=boot_size)
            _, w, _ = self._solve_ridge(Xs[idx], y[idx], self.screening_lambda)
            weights.append(w)
        W = np.vstack(weights)

        mag = np.median(np.abs(W), axis=0)
        sign_stab = np.abs(np.mean(np.sign(W), axis=0))
        score = mag * sign_stab
        order = np.argsort(score)[::-1]

        selected = []
        for j in order:
            if len(selected) >= int(self.max_linear_features):
                break
            if score[j] > 1e-10 and sign_stab[j] >= float(self.min_stability):
                selected.append(int(j))

        if not selected:
            selected = [int(order[0])]
        return np.array(sorted(set(selected)), dtype=int), score, sign_stab

    def _fit_piece(self, X_sub, y_sub, lam):
        if len(y_sub) < 2:
            return None
        b, w, pred = self._solve_ridge(X_sub, y_sub, lam)
        mse = float(np.mean((y_sub - pred) ** 2))
        return b, w, mse

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape
        self.n_features_in_ = p

        mu, sigma = self._safe_scale(X)
        Xs = (X - mu) / sigma

        active_idx, screen_score, sign_stability = self._screen_features(Xs, y)
        Xa = Xs[:, active_idx]

        rng = np.random.RandomState(self.random_state + 17)
        perm = rng.permutation(n)
        n_val = max(20, n // 5)
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]
        if len(tr_idx) < 20:
            tr_idx = perm
            val_idx = perm

        Xtr = Xa[tr_idx]
        Xval = Xa[val_idx]
        ytr = y[tr_idx]
        yval = y[val_idx]

        b_base, w_base, _ = self._solve_ridge(Xtr, ytr, self.linear_lambda)
        pred_val_base = b_base + Xval @ w_base
        mse_base = float(np.mean((yval - pred_val_base) ** 2))

        top_order_local = np.argsort(screen_score[active_idx])[::-1]
        top_local = top_order_local[: min(int(self.top_split_features), len(active_idx))]

        best_piece = None
        best_mse = mse_base
        min_leaf = max(int(self.min_leaf), max(8, len(tr_idx) // 30))
        q_levels = [0.3, 0.5, 0.7]

        for local_k in top_local:
            split_local = int(local_k)
            xj_tr = Xtr[:, split_local]
            xj_val = Xval[:, split_local]
            thrs = np.unique(np.quantile(xj_tr, q_levels))
            for thr in thrs:
                thr = float(thr)
                left_tr = xj_tr <= thr
                right_tr = ~left_tr
                left_val = xj_val <= thr
                right_val = ~left_val
                if (left_tr.sum() < min_leaf) or (right_tr.sum() < min_leaf):
                    continue
                if (left_val.sum() < 2) or (right_val.sum() < 2):
                    continue

                left_fit = self._fit_piece(Xtr[left_tr], ytr[left_tr], self.piece_lambda)
                right_fit = self._fit_piece(Xtr[right_tr], ytr[right_tr], self.piece_lambda)
                if left_fit is None or right_fit is None:
                    continue
                b_l, w_l, _ = left_fit
                b_r, w_r, _ = right_fit

                pred_val = np.empty_like(yval)
                pred_val[left_val] = b_l + Xval[left_val] @ w_l
                pred_val[right_val] = b_r + Xval[right_val] @ w_r
                mse_val = float(np.mean((yval - pred_val) ** 2))

                if mse_val < best_mse:
                    best_mse = mse_val
                    best_piece = {
                        "split_local": split_local,
                        "split_global": int(active_idx[split_local]),
                        "thr_std": thr,
                    }

        rel_gain = (mse_base - best_mse) / max(mse_base, 1e-12)
        use_piecewise = (best_piece is not None) and (rel_gain >= float(self.min_relative_gain))

        if use_piecewise:
            split_local = best_piece["split_local"]
            split_global = best_piece["split_global"]
            thr_std = float(best_piece["thr_std"])
            gate = Xa[:, split_local] <= thr_std

            left_fit = self._fit_piece(Xa[gate], y[gate], self.piece_lambda)
            right_fit = self._fit_piece(Xa[~gate], y[~gate], self.piece_lambda)
            if left_fit is None or right_fit is None:
                use_piecewise = False
            else:
                b_l_std, w_l_std, _ = left_fit
                b_r_std, w_r_std, _ = right_fit
                pred_train = np.empty_like(y)
                pred_train[gate] = b_l_std + Xa[gate] @ w_l_std
                pred_train[~gate] = b_r_std + Xa[~gate] @ w_r_std

                b_l_raw, coef_l_raw = self._to_raw_params(active_idx, b_l_std, w_l_std, mu, sigma, p)
                b_r_raw, coef_r_raw = self._to_raw_params(active_idx, b_r_std, w_r_std, mu, sigma, p)

                self.use_piecewise_ = True
                self.split_feature_ = int(split_global)
                self.split_threshold_ = float(mu[split_global] + sigma[split_global] * thr_std)
                self.intercept_left_ = b_l_raw
                self.coef_left_ = coef_l_raw
                self.intercept_right_ = b_r_raw
                self.coef_right_ = coef_r_raw

        if not use_piecewise:
            b_std, w_std, pred_train = self._solve_ridge(Xa, y, self.linear_lambda)
            b_raw, coef_raw = self._to_raw_params(active_idx, b_std, w_std, mu, sigma, p)
            self.use_piecewise_ = False
            self.intercept_ = b_raw
            self.coef_ = coef_raw

        self.feature_mu_ = mu
        self.feature_sigma_ = sigma
        self.active_features_ = active_idx
        self.screen_score_ = screen_score
        self.sign_stability_ = sign_stability
        self.val_mse_base_ = mse_base
        self.val_mse_best_ = best_mse
        self.train_mse_ = float(np.mean((y - pred_train) ** 2))
        return self

    def predict(self, X):
        check_is_fitted(self, ["use_piecewise_", "active_features_"])
        X = np.asarray(X, dtype=float)
        if not self.use_piecewise_:
            return self.intercept_ + X @ self.coef_

        yhat = np.empty(X.shape[0], dtype=float)
        left = X[:, int(self.split_feature_)] <= float(self.split_threshold_)
        yhat[left] = self.intercept_left_ + X[left] @ self.coef_left_
        yhat[~left] = self.intercept_right_ + X[~left] @ self.coef_right_
        return yhat

    @staticmethod
    def _equation_terms(intercept, coef, max_terms=6):
        terms = [f"{float(intercept):.6f}"]
        active = [j for j in range(len(coef)) if abs(float(coef[j])) > 1e-10]
        active = sorted(active, key=lambda j: -abs(float(coef[j])))[: int(max_terms)]
        for j in active:
            terms.append(f"{float(coef[j]):+.6f}*x{j}")
        return " ".join(terms), active

    def __str__(self):
        check_is_fitted(self, ["use_piecewise_", "active_features_"])
        lines = [
            "Stability Piecewise Sparse Regressor",
            "Equation in raw features (omitted coefficients are 0):",
        ]

        if not self.use_piecewise_:
            eq, active = self._equation_terms(self.intercept_, self.coef_)
            lines.append(f"y = {eq}")
            lines.append(f"Active features: {', '.join(f'x{j}' for j in active) if active else '(none)'}")
            return "\n".join(lines)

        eq_l, active_l = self._equation_terms(self.intercept_left_, self.coef_left_)
        eq_r, active_r = self._equation_terms(self.intercept_right_, self.coef_right_)
        lines.append(f"If x{self.split_feature_} <= {self.split_threshold_:.6f}:")
        lines.append(f"  y = {eq_l}")
        lines.append(f"Else (x{self.split_feature_} > {self.split_threshold_:.6f}):")
        lines.append(f"  y = {eq_r}")
        used = sorted(set(active_l) | set(active_r))
        lines.append(f"Active features: {', '.join(f'x{j}' for j in used) if used else '(none)'}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
StabilityPiecewiseSparseRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "StabilityPiecewiseSparseV1"
model_description = "Sign-stability screened sparse linear regressor with optional single-threshold piecewise sparse equations selected by validation gain"
model_defs = [(model_shorthand_name, StabilityPiecewiseSparseRegressor())]


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
