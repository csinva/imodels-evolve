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


class SparseTailSplineRidgeRegressor(BaseEstimator, RegressorMixin):
    """
    Interpretable additive regression with three terms per feature:
      linear z_j, upper-tail hinge max(0, z_j-q75_j), lower-tail hinge max(0, q25_j-z_j).
    A custom GCV ridge fit is applied to all terms, then feature groups are pruned
    with an explicit complexity penalty and refit.
    """

    def __init__(
        self,
        alpha_grid_size=21,
        alpha_min_log10=-6.0,
        alpha_max_log10=3.0,
        top_group_candidates=(4, 8, 12, 20),
        complexity_penalty=0.008,
        max_display_groups=8,
    ):
        self.alpha_grid_size = alpha_grid_size
        self.alpha_min_log10 = alpha_min_log10
        self.alpha_max_log10 = alpha_max_log10
        self.top_group_candidates = top_group_candidates
        self.complexity_penalty = complexity_penalty
        self.max_display_groups = max_display_groups

    @staticmethod
    def _standardize(X):
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
        sigma = np.where(sigma < 1e-8, 1.0, sigma)
        Z = (X - mu) / sigma
        return Z, mu, sigma

    @staticmethod
    def _ridge_gcv(Xd, y, alpha_min_log10, alpha_max_log10, alpha_grid_size):
        n, p = Xd.shape
        if n == 0:
            return {"coef": np.zeros(p), "intercept": 0.0, "alpha": 1.0, "pred": np.zeros(0), "gcv": 0.0}
        if p == 0:
            y_mean = float(np.mean(y))
            pred = np.full(n, y_mean, dtype=float)
            mse = float(np.mean((y - pred) ** 2))
            return {"coef": np.zeros(0), "intercept": y_mean, "alpha": 1.0, "pred": pred, "gcv": mse}

        x_mean = np.mean(Xd, axis=0)
        y_mean = float(np.mean(y))
        Xc = Xd - x_mean
        yc = y - y_mean

        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        s2 = s * s
        uty = U.T @ yc
        alphas = np.logspace(
            float(alpha_min_log10),
            float(alpha_max_log10),
            num=int(max(5, alpha_grid_size)),
        )

        best = None
        for alpha in alphas:
            shrink = s / (s2 + alpha)
            coef = Vt.T @ (shrink * uty)
            intercept = y_mean - float(x_mean @ coef)
            pred = Xd @ coef + intercept
            mse = float(np.mean((y - pred) ** 2))
            df = float(np.sum(s2 / (s2 + alpha)))
            denom = max(1e-8, 1.0 - df / max(n, 1))
            gcv = mse / (denom * denom)
            cur = {
                "coef": coef,
                "intercept": intercept,
                "alpha": float(alpha),
                "pred": pred,
                "gcv": gcv,
            }
            if best is None or cur["gcv"] < best["gcv"]:
                best = cur
        return best

    @staticmethod
    def _build_basis(Z, q25, q75, feature_ids):
        n = Z.shape[0]
        k = len(feature_ids)
        B = np.zeros((n, 3 * k), dtype=float)
        for i, j in enumerate(feature_ids):
            z = Z[:, j]
            B[:, 3 * i] = z
            B[:, 3 * i + 1] = np.maximum(0.0, z - q75[j])
            B[:, 3 * i + 2] = np.maximum(0.0, q25[j] - z)
        return B

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape
        self.n_features_in_ = int(p)
        self.feature_names_ = [f"x{j}" for j in range(p)]

        if n == 0:
            self.mu_ = np.zeros(p, dtype=float)
            self.sigma_ = np.ones(p, dtype=float)
            self.q25_ = np.zeros(p, dtype=float)
            self.q75_ = np.zeros(p, dtype=float)
            self.selected_features_ = np.array([], dtype=int)
            self.coef_groups_ = np.zeros((0, 3), dtype=float)
            self.intercept_ = float(np.mean(y)) if y.size else 0.0
            self.alpha_ = 1.0
            self.training_mse_ = 0.0
            self.is_fitted_ = True
            return self

        Z, mu, sigma = self._standardize(X)
        q25 = np.percentile(Z, 25, axis=0)
        q75 = np.percentile(Z, 75, axis=0)
        self.mu_ = mu
        self.sigma_ = sigma
        self.q25_ = q25
        self.q75_ = q75

        all_features = np.arange(p, dtype=int)
        B_all = self._build_basis(Z, q25, q75, all_features)
        base = self._ridge_gcv(
            B_all,
            y,
            self.alpha_min_log10,
            self.alpha_max_log10,
            self.alpha_grid_size,
        )
        coef_all = base["coef"].reshape(p, 3)
        group_strength = np.sqrt(np.sum(coef_all * coef_all, axis=1))

        k_candidates = []
        for k in self.top_group_candidates:
            kk = int(k)
            if kk > 0:
                k_candidates.append(min(p, kk))
        if p > 0:
            k_candidates.append(p)
        k_candidates = sorted(set(k_candidates))

        ranked = np.argsort(group_strength)[::-1]
        best = None
        for k in k_candidates:
            feat_ids = ranked[:k]
            B = self._build_basis(Z, q25, q75, feat_ids)
            cur = self._ridge_gcv(
                B,
                y,
                self.alpha_min_log10,
                self.alpha_max_log10,
                self.alpha_grid_size,
            )
            mse = float(np.mean((y - cur["pred"]) ** 2))
            score = mse * (1.0 + float(self.complexity_penalty) * k)
            cand = {
                "feat_ids": feat_ids.copy(),
                "coef": cur["coef"].reshape(k, 3),
                "intercept": float(cur["intercept"]),
                "alpha": float(cur["alpha"]),
                "mse": mse,
                "score": score,
            }
            if best is None or cand["score"] < best["score"]:
                best = cand

        self.selected_features_ = best["feat_ids"]
        self.coef_groups_ = best["coef"]
        self.intercept_ = best["intercept"]
        self.alpha_ = best["alpha"]
        self.training_mse_ = best["mse"]
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        X = np.asarray(X, dtype=float)
        Z = (X - self.mu_) / self.sigma_
        if self.selected_features_.size == 0:
            return np.full(X.shape[0], self.intercept_, dtype=float)
        B = self._build_basis(Z, self.q25_, self.q75_, self.selected_features_)
        return B @ self.coef_groups_.reshape(-1) + self.intercept_

    def __str__(self):
        check_is_fitted(self, "is_fitted_")
        lines = ["Sparse Tail-Spline Ridge Regressor:"]
        lines.append("  prediction = intercept + sum_j [a_j*z_j + b_j*max(0,z_j-q75_j) + c_j*max(0,q25_j-z_j)]")
        lines.append("  where z_j = (x_j - mean_j)/std_j")
        lines.append(f"  selected feature groups: {len(self.selected_features_)}")
        lines.append(f"  ridge alpha (GCV): {self.alpha_:.4g}")
        lines.append(f"  training MSE: {self.training_mse_:.6f}")
        lines.append(f"  intercept: {self.intercept_:+.6f}")

        max_groups = int(max(1, self.max_display_groups))
        strengths = np.sqrt(np.sum(self.coef_groups_ * self.coef_groups_, axis=1))
        order = np.argsort(strengths)[::-1][:max_groups]
        lines.append("  strongest feature contributions:")
        for idx in order:
            j = int(self.selected_features_[idx])
            a, b, c = self.coef_groups_[idx]
            lines.append(
                f"    x{j}: {a:+.4f}*z + {b:+.4f}*max(0,z-{self.q75_[j]:+.3f}) + {c:+.4f}*max(0,{self.q25_[j]:+.3f}-z)"
            )
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparseTailSplineRidgeRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "SparseTailSplineRidge_v1"
model_description = "Custom additive tail-spline basis (linear + upper/lower hinges per feature) with GCV ridge and complexity-aware feature-group pruning"
model_defs = [(model_shorthand_name, SparseTailSplineRidgeRegressor())]


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

    std_tests = {t.__name__ for t in ALL_TESTS}
    hard_tests = {t.__name__ for t in HARD_TESTS}
    insight_tests = {t.__name__ for t in INSIGHT_TESTS}
    std_passed = sum(r["passed"] for r in interp_results if r["test"] in std_tests)
    hard_passed = sum(r["passed"] for r in interp_results if r["test"] in hard_tests)
    insight_passed = sum(r["passed"] for r in interp_results if r["test"] in insight_tests)
    print(f"[std {std_passed}/{len(std_tests)}  hard {hard_passed}/{len(hard_tests)}  insight {insight_passed}/{len(insight_tests)}]")
    print(f"total_seconds: {time.time() - t0:.1f}s")
