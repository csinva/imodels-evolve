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
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class SplitLinearModelTreeRegressor(BaseEstimator, RegressorMixin):
    """Single-split sparse linear model tree with compact branch equations."""

    def __init__(
        self,
        alpha_grid=(1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0),
        max_terms_global=8,
        max_terms_leaf=5,
        top_split_features=8,
        split_quantiles=(0.2, 0.35, 0.5, 0.65, 0.8),
        min_leaf_frac=0.18,
        min_split_gain=1e-4,
        split_penalty=1e-3,
        n_splits=4,
        coef_prune_rel=0.02,
        random_state=0,
    ):
        self.alpha_grid = alpha_grid
        self.max_terms_global = max_terms_global
        self.max_terms_leaf = max_terms_leaf
        self.top_split_features = top_split_features
        self.split_quantiles = split_quantiles
        self.min_leaf_frac = min_leaf_frac
        self.min_split_gain = min_split_gain
        self.split_penalty = split_penalty
        self.n_splits = n_splits
        self.coef_prune_rel = coef_prune_rel
        self.random_state = random_state

    @staticmethod
    def _safe_standardize(X):
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
        sigma = np.where(sigma > 1e-12, sigma, 1.0)
        return (X - mu) / sigma, mu, sigma

    @staticmethod
    def _ridge_solve(X, y, alpha):
        lhs = X.T @ X + float(alpha) * np.eye(X.shape[1])
        rhs = X.T @ y
        try:
            return np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(lhs) @ rhs

    def _cv_alpha(self, Xz, yc, rng_seed):
        n = Xz.shape[0]
        if n < 3 or Xz.shape[1] == 0:
            return 1.0
        n_splits = min(max(2, int(self.n_splits)), n - 1)
        if n_splits < 2:
            return 1.0
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=rng_seed)
        alpha_choices = [float(a) for a in self.alpha_grid] or [1.0]

        best_alpha = alpha_choices[0]
        best_mse = np.inf
        for alpha in alpha_choices:
            mses = []
            for tr, va in kf.split(Xz):
                Xtr, ytr = Xz[tr], yc[tr]
                Xva, yva = Xz[va], yc[va]

                beta = self._ridge_solve(Xtr, ytr, alpha)
                pred = Xva @ beta
                mses.append(float(np.mean((yva - pred) ** 2)))
            mse = float(np.mean(mses))
            if mse < best_mse:
                best_mse = mse
                best_alpha = alpha
        return float(best_alpha)

    def _top_corr_features(self, X, y, max_terms):
        p = X.shape[1]
        max_terms = int(max(1, min(max_terms, p)))
        yc = y - float(np.mean(y))
        ys = float(np.std(yc)) + 1e-12
        scores = np.zeros(p, dtype=float)
        for j in range(p):
            xj = X[:, j]
            xj = xj - float(np.mean(xj))
            xs = float(np.std(xj)) + 1e-12
            scores[j] = abs(float(np.mean((xj / xs) * (yc / ys))))
        order = np.argsort(scores)[::-1]
        return np.sort(order[:max_terms]).astype(int), scores

    def _fit_sparse_linear(self, X, y, max_terms, seed_shift=0):
        n, p = X.shape
        chosen, corr_scores = self._top_corr_features(X, y, max_terms=max_terms)
        Xs = X[:, chosen]
        Xz, mu, sigma = self._safe_standardize(Xs)
        y_mean = float(np.mean(y))
        yc = y - y_mean
        alpha = self._cv_alpha(Xz, yc, rng_seed=int(self.random_state) + int(seed_shift))
        beta_z = self._ridge_solve(Xz, yc, alpha)

        coef_local = beta_z / sigma
        intercept = y_mean - float(np.dot(coef_local, mu))

        coef_full = np.zeros(p, dtype=float)
        coef_full[chosen] = coef_local

        max_abs = float(np.max(np.abs(coef_full))) if coef_full.size else 0.0
        if max_abs > 0:
            thr = float(self.coef_prune_rel) * max_abs
            coef_full[np.abs(coef_full) < thr] = 0.0

        pred = intercept + X @ coef_full
        mse = float(np.mean((y - pred) ** 2))
        active = np.flatnonzero(np.abs(coef_full) > 1e-12).astype(int)
        return intercept, coef_full, alpha, mse, active, corr_scores

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y have incompatible shapes")
        if X.shape[1] == 0:
            raise ValueError("No features provided")

        self.n_features_in_ = X.shape[1]
        p = self.n_features_in_
        (
            self.global_intercept_,
            self.global_coef_,
            self.global_alpha_,
            base_mse,
            global_active,
            corr_scores,
        ) = self._fit_sparse_linear(X, y, max_terms=self.max_terms_global, seed_shift=0)

        max_split_features = max(1, min(int(self.top_split_features), p))
        split_candidates = np.argsort(corr_scores)[::-1][:max_split_features]

        n = X.shape[0]
        min_leaf = max(12, int(float(self.min_leaf_frac) * n))
        best = None

        for j in split_candidates:
            xj = X[:, int(j)]
            thresholds = []
            for q in self.split_quantiles:
                try:
                    t = float(np.quantile(xj, q))
                    if np.isfinite(t):
                        thresholds.append(t)
                except Exception:
                    continue
            thresholds = sorted(set(thresholds))
            for t in thresholds:
                left_mask = xj <= t
                n_left = int(np.sum(left_mask))
                n_right = n - n_left
                if n_left < min_leaf or n_right < min_leaf:
                    continue

                Xl, yl = X[left_mask], y[left_mask]
                Xr, yr = X[~left_mask], y[~left_mask]
                li, lc, la, _, left_active, _ = self._fit_sparse_linear(
                    Xl, yl, max_terms=self.max_terms_leaf, seed_shift=11
                )
                ri, rc, ra, _, right_active, _ = self._fit_sparse_linear(
                    Xr, yr, max_terms=self.max_terms_leaf, seed_shift=29
                )

                pred = np.empty(n, dtype=float)
                pred[left_mask] = li + Xl @ lc
                pred[~left_mask] = ri + Xr @ rc
                mse = float(np.mean((y - pred) ** 2))
                gain = base_mse - mse

                complexity = left_active.size + right_active.size + 1
                penalized = mse + float(self.split_penalty) * complexity
                candidate = (
                    penalized,
                    mse,
                    gain,
                    int(j),
                    float(t),
                    li,
                    lc,
                    la,
                    ri,
                    rc,
                    ra,
                    left_active,
                    right_active,
                )
                if best is None or candidate[0] < best[0]:
                    best = candidate

        self.has_split_ = False
        if best is not None and best[2] > float(self.min_split_gain) * max(base_mse, 1e-8):
            (
                _,
                self.split_mse_,
                self.split_gain_,
                self.split_feature_,
                self.split_threshold_,
                self.left_intercept_,
                self.left_coef_,
                self.left_alpha_,
                self.right_intercept_,
                self.right_coef_,
                self.right_alpha_,
                left_active,
                right_active,
            ) = best
            self.has_split_ = True
            self.active_features_ = np.sort(
                np.unique(np.concatenate([left_active, right_active]))
            ).astype(int)
            importances = np.abs(self.left_coef_) + np.abs(self.right_coef_)
        else:
            self.split_mse_ = float(base_mse)
            self.split_gain_ = 0.0
            self.active_features_ = global_active
            importances = np.abs(self.global_coef_)

        max_imp = float(np.max(importances)) if importances.size else 0.0
        self.feature_importances_ = importances / max_imp if max_imp > 0 else importances
        return self

    def predict(self, X):
        check_is_fitted(self, ["global_intercept_", "global_coef_", "has_split_"])
        X = np.asarray(X, dtype=float)
        if not self.has_split_:
            return self.global_intercept_ + X @ self.global_coef_
        mask = X[:, int(self.split_feature_)] <= float(self.split_threshold_)
        pred = np.empty(X.shape[0], dtype=float)
        pred[mask] = self.left_intercept_ + X[mask] @ self.left_coef_
        pred[~mask] = self.right_intercept_ + X[~mask] @ self.right_coef_
        return pred

    def _equation_lines(self, intercept, coef):
        lines = [f"y = {float(intercept):+.6f}"]
        active = np.flatnonzero(np.abs(coef) > 1e-12)
        for j in active:
            lines.append(f"    {float(coef[j]):+.6f} * x{int(j)}")
        if active.size == 0:
            lines.append("    +0.000000 * (no active feature terms)")
        return lines

    def __str__(self):
        check_is_fitted(self, ["global_intercept_", "global_coef_", "has_split_", "active_features_"])
        lines = ["Split Linear Model Tree (single rule, sparse branch equations)"]

        if self.has_split_:
            j = int(self.split_feature_)
            t = float(self.split_threshold_)
            lines.append(f"Rule: if x{j} <= {t:.6f}, use LEFT equation; else use RIGHT equation.")
            lines.append("")
            lines.append("LEFT branch equation:")
            lines.extend(self._equation_lines(self.left_intercept_, self.left_coef_))
            lines.append("")
            lines.append("RIGHT branch equation:")
            lines.extend(self._equation_lines(self.right_intercept_, self.right_coef_))
            lines.append("")
            lines.append(f"split_gain_mse = {float(self.split_gain_):.6f}")
            lines.append(f"left_alpha = {float(self.left_alpha_):.6g}")
            lines.append(f"right_alpha = {float(self.right_alpha_):.6g}")
        else:
            lines.append("No split selected. Global sparse linear equation:")
            lines.extend(self._equation_lines(self.global_intercept_, self.global_coef_))
            lines.append("")
            lines.append(f"global_alpha = {float(self.global_alpha_):.6g}")

        lines.append("")
        lines.append(
            "active_features = "
            + (", ".join(f"x{int(j)}" for j in self.active_features_) if self.active_features_.size else "none")
        )
        lines.append("All unlisted features have zero contribution in every branch.")
        lines.append("feature_importance_by_abs_coefficient:")
        for j, s in enumerate(self.feature_importances_):
            if s > 1e-8:
                lines.append(f"  x{j}: {float(s):.4f}")
        lines.append("")
        lines.append("Simulation recipe: choose branch from the rule, then sum listed terms.")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys

_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])

# Update the model shorthand name and description below to reflect the class above and any changes you make.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
SplitLinearModelTreeRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "SplitLinearModelTree_v1"
model_description = "Single-split sparse linear model tree: one correlation-screened threshold rule with branch-specific ridge-CV equations in raw feature space"
model_defs = [(model_shorthand_name, SplitLinearModelTreeRegressor())]


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

    # --- Recompute global rank summary from updated performance_results.csv ---
    # Build dataset -> {model: rmse}
    perf_table = defaultdict(dict)
    with open(perf_csv, newline="") as f:
        for row in csv.DictReader(f):
            ds = row["dataset"]
            m = row["model"]
            rmse_s = row.get("rmse", "")
            if rmse_s in ("", None):
                perf_table[ds][m] = float("nan")
            else:
                try:
                    perf_table[ds][m] = float(rmse_s)
                except ValueError:
                    perf_table[ds][m] = float("nan")

    avg_rank, _ = compute_rank_scores(perf_table)
    mean_rank = avg_rank.get(model_name, float("nan"))

    # --- Upsert overall_results.csv ---
    overall_rows = [{
        "commit": git_hash,
        "mean_rank": f"{mean_rank:.2f}" if np.isfinite(mean_rank) else "",
        "frac_interpretability_tests_passed": f"{(n_passed / total):.4f}" if total else "",
        "status": "",  # fill manually after reviewing
        "model_name": model_name,
        "description": model_description,
    }]
    upsert_overall_results(overall_rows, RESULTS_DIR)

    # --- Plot update ---
    overall_csv = os.path.join(RESULTS_DIR, "overall_results.csv")
    plot_interp_vs_performance(
        overall_csv,
        os.path.join(RESULTS_DIR, "interpretability_vs_performance.png"),
    )

    # Print compact summary
    std_names = {t.__name__ for t in ALL_TESTS}
    hard_names = {t.__name__ for t in HARD_TESTS}
    ins_names = {t.__name__ for t in INSIGHT_TESTS}
    n_std = sum(r["passed"] for r in interp_results if r["test"] in std_names)
    n_hard = sum(r["passed"] for r in interp_results if r["test"] in hard_names)
    n_ins = sum(r["passed"] for r in interp_results if r["test"] in ins_names)

    print("\n---")
    print(f"tests_passed:  {n_passed}/{total} ({(n_passed/total):.2%})  "
          f"[std {n_std}/{len(std_names)}  hard {n_hard}/{len(hard_names)}  insight {n_ins}/{len(ins_names)}]")
    print(f"total_seconds: {time.time() - t0:.1f}s")
