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


class SparseMeanHingeAdditiveRegressor(BaseEstimator, RegressorMixin):
    """Sparse additive piecewise-linear regressor with mean-knot hinges."""

    def __init__(
        self,
        alpha_grid=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
        max_active_features=14,
        min_rel_feature_strength=0.03,
        n_splits=4,
        coef_prune_rel=0.03,
        random_state=0,
    ):
        self.alpha_grid = alpha_grid
        self.max_active_features = max_active_features
        self.min_rel_feature_strength = min_rel_feature_strength
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

    def _build_basis(self, Xz):
        n, p = Xz.shape
        B = np.zeros((n, 3 * p), dtype=float)
        B[:, :p] = Xz
        B[:, p : 2 * p] = np.maximum(Xz, 0.0)
        B[:, 2 * p : 3 * p] = np.maximum(-Xz, 0.0)
        return B

    def _cv_alpha(self, Xz, yc):
        n = Xz.shape[0]
        n_splits = min(max(2, int(self.n_splits)), n)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
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

    def _group_strength(self, coef, n_features):
        a = coef[:n_features]
        b = coef[n_features : 2 * n_features]
        c = coef[2 * n_features : 3 * n_features]
        return np.sqrt(a * a + b * b + c * c)

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
        Xz, self.x_mean_, self.x_scale_ = self._safe_standardize(X)
        self.y_mean_ = float(np.mean(y))
        yc = y - self.y_mean_

        B = self._build_basis(Xz)
        self.alpha_ = self._cv_alpha(B, yc)
        coef_full = self._ridge_solve(B, yc, self.alpha_)
        strengths = self._group_strength(coef_full, self.n_features_in_)
        order = np.argsort(strengths)[::-1]
        top_k = max(1, min(int(self.max_active_features), self.n_features_in_))
        keep = order[:top_k]
        max_strength = float(np.max(strengths)) if strengths.size else 0.0
        if max_strength > 0:
            rel = float(self.min_rel_feature_strength)
            keep = keep[strengths[keep] >= rel * max_strength]
        if keep.size == 0:
            keep = order[:1]
        self.active_features_ = np.sort(keep).astype(int)

        cols = np.concatenate(
            [
                self.active_features_,
                self.active_features_ + self.n_features_in_,
                self.active_features_ + 2 * self.n_features_in_,
            ]
        ).astype(int)
        B_small = B[:, cols]
        self.alpha_refit_ = self._cv_alpha(B_small, yc)
        coef_small = self._ridge_solve(B_small, yc, self.alpha_refit_)

        p = self.n_features_in_
        self.raw_intercept_ = float(self.y_mean_)
        self.raw_linear_coef_ = np.zeros(p, dtype=float)
        self.raw_pos_coef_ = np.zeros(p, dtype=float)
        self.raw_neg_coef_ = np.zeros(p, dtype=float)
        self.knots_ = self.x_mean_.copy()

        n_active = self.active_features_.size
        a = coef_small[:n_active]
        b = coef_small[n_active : 2 * n_active]
        c = coef_small[2 * n_active : 3 * n_active]
        for idx, j in enumerate(self.active_features_):
            jj = int(j)
            scale = float(self.x_scale_[jj])
            mu = float(self.x_mean_[jj])
            aj = float(a[idx])
            bj = float(b[idx])
            cj = float(c[idx])

            self.raw_linear_coef_[jj] += aj / scale
            self.raw_intercept_ += -(aj * mu / scale)
            self.raw_pos_coef_[jj] += bj / scale
            self.raw_neg_coef_[jj] += cj / scale

        max_abs = float(
            np.max(
                np.concatenate(
                    [
                        np.abs(self.raw_linear_coef_),
                        np.abs(self.raw_pos_coef_),
                        np.abs(self.raw_neg_coef_),
                    ]
                )
            )
        )
        if max_abs > 0:
            thr = float(self.coef_prune_rel) * max_abs
            self.raw_linear_coef_[np.abs(self.raw_linear_coef_) < thr] = 0.0
            self.raw_pos_coef_[np.abs(self.raw_pos_coef_) < thr] = 0.0
            self.raw_neg_coef_[np.abs(self.raw_neg_coef_) < thr] = 0.0

        feat_scores = (
            np.abs(self.raw_linear_coef_)
            + np.abs(self.raw_pos_coef_)
            + np.abs(self.raw_neg_coef_)
        )
        max_score = float(np.max(feat_scores)) if feat_scores.size else 0.0
        self.feature_importances_ = feat_scores / max_score if max_score > 0 else feat_scores
        self.active_features_ = np.flatnonzero(feat_scores > 1e-12).astype(int)
        return self

    def predict(self, X):
        check_is_fitted(
            self,
            ["raw_intercept_", "raw_linear_coef_", "raw_pos_coef_", "raw_neg_coef_", "knots_"],
        )
        X = np.asarray(X, dtype=float)
        pred = np.full(X.shape[0], float(self.raw_intercept_), dtype=float)
        pred += X @ self.raw_linear_coef_
        pred += np.maximum(0.0, X - self.knots_) @ self.raw_pos_coef_
        pred += np.maximum(0.0, self.knots_ - X) @ self.raw_neg_coef_
        return pred

    def __str__(self):
        check_is_fitted(
            self,
            [
                "raw_intercept_",
                "raw_linear_coef_",
                "active_features_",
                "feature_importances_",
                "alpha_",
                "alpha_refit_",
                "raw_pos_coef_",
                "raw_neg_coef_",
                "knots_",
            ],
        )

        lines = [
            "Sparse Mean-Hinge Additive Regressor",
            "Exact prediction equation on raw features:",
            f"y = {self.raw_intercept_:+.6f}",
        ]
        for j in range(self.n_features_in_):
            a = float(self.raw_linear_coef_[j])
            b = float(self.raw_pos_coef_[j])
            c = float(self.raw_neg_coef_[j])
            if abs(a) > 1e-12:
                lines.append(f"    {a:+.6f} * x{j}")
            t = float(self.knots_[j])
            if abs(b) > 1e-12:
                lines.append(f"    {b:+.6f} * max(0, x{j} - {t:.6f})")
            if abs(c) > 1e-12:
                lines.append(f"    {c:+.6f} * max(0, {t:.6f} - x{j})")

        lines.append("")
        lines.append(f"ridge_alpha_initial = {self.alpha_:.6g}")
        lines.append(f"ridge_alpha_refit = {self.alpha_refit_:.6g}")
        lines.append(
            "active_features = "
            + (", ".join(f"x{int(j)}" for j in self.active_features_) if self.active_features_.size else "none")
        )
        lines.append("All unlisted features have zero contribution.")
        lines.append("feature_importance_by_abs_contribution:")
        for j, s in enumerate(self.feature_importances_):
            if s > 1e-8:
                lines.append(f"  x{j}: {float(s):.4f}")
        lines.append("")
        lines.append("Simulation recipe: sum every listed term exactly as written.")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys

_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])

# Update the model shorthand name and description below to reflect the class above and any changes you make.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
SparseMeanHingeAdditiveRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "SparseMeanHingeAdditive_v1"
model_description = "From-scratch additive piecewise-linear model with per-feature mean-knot hinges, ridge-CV fitting, group sparsification, and exact raw-feature equation output"
model_defs = [(model_shorthand_name, SparseMeanHingeAdditiveRegressor())]


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
