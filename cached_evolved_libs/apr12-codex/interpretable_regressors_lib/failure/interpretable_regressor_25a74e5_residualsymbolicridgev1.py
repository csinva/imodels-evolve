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


class ResidualSymbolicRidgeRegressor(BaseEstimator, RegressorMixin):
    """
    Ridge backbone plus a tiny symbolic residual basis.

    Model:
      y = b + sum_j w_j * x_j + sum_k v_k * g_k(x)
    where g_k are greedily selected from hinge / square / pairwise interaction
    candidates on high-impact features.
    """

    def __init__(
        self,
        alphas=(0.03, 0.1, 0.3, 1.0, 3.0, 10.0),
        top_base_features=4,
        max_extra_terms=2,
        min_extra_score=0.015,
        min_col_std=1e-8,
        coef_tol=8e-4,
    ):
        self.alphas = alphas
        self.top_base_features = top_base_features
        self.max_extra_terms = max_extra_terms
        self.min_extra_score = min_extra_score
        self.min_col_std = min_col_std
        self.coef_tol = coef_tol

    def _impute(self, X):
        X = np.asarray(X, dtype=float).copy()
        bad = ~np.isfinite(X)
        if bad.any():
            X[bad] = np.take(self.feature_medians_, np.where(bad)[1])
        return X

    @staticmethod
    def _ridge_fit(A, y, alpha):
        n, p = A.shape
        M = np.column_stack([np.ones(n), A])
        reg = float(alpha) * np.eye(p + 1)
        reg[0, 0] = 0.0
        beta = np.linalg.solve(M.T @ M + reg, M.T @ y)
        return float(beta[0]), beta[1:].astype(float)

    def _fit_best_ridge(self, A, y):
        best = None
        for alpha in self.alphas:
            b, w = self._ridge_fit(A, y, alpha)
            pred = b + A @ w
            mse = float(np.mean((y - pred) ** 2))
            if (best is None) or (mse < best["mse"]):
                best = {"alpha": float(alpha), "intercept": b, "coef": w, "mse": mse}
        return best

    def _build_term(self, X, spec):
        t = spec["type"]
        if t == "hinge_pos":
            j, thr = spec["j"], spec["thr"]
            return np.maximum(0.0, X[:, j] - thr)
        if t == "hinge_neg":
            j, thr = spec["j"], spec["thr"]
            return np.maximum(0.0, thr - X[:, j])
        if t == "square":
            j, mu = spec["j"], spec["mu"]
            return (X[:, j] - mu) ** 2
        if t == "interaction":
            j, k = spec["j"], spec["k"]
            return (X[:, j] - self.feature_means_[j]) * (X[:, k] - self.feature_means_[k])
        raise ValueError(f"Unknown term type: {t}")

    def _term_name(self, spec):
        t = spec["type"]
        if t == "hinge_pos":
            return f"max(0, x{spec['j']} - {spec['thr']:+.5f})"
        if t == "hinge_neg":
            return f"max(0, {spec['thr']:+.5f} - x{spec['j']})"
        if t == "square":
            return f"(x{spec['j']} - {spec['mu']:+.5f})^2"
        if t == "interaction":
            return f"(x{spec['j']} - {self.feature_means_[spec['j']]:+.5f})*(x{spec['k']} - {self.feature_means_[spec['k']]:+.5f})"
        return "unknown_term"

    def _candidate_specs(self, X, base_order):
        specs = []
        q_grid = (0.35, 0.5, 0.65)
        for j in base_order:
            xj = X[:, j]
            for q in q_grid:
                thr = float(np.quantile(xj, q))
                specs.append({"type": "hinge_pos", "j": int(j), "thr": thr})
                specs.append({"type": "hinge_neg", "j": int(j), "thr": thr})
            specs.append({"type": "square", "j": int(j), "mu": float(np.mean(xj))})
        for a in range(len(base_order)):
            for b in range(a + 1, len(base_order)):
                j, k = int(base_order[a]), int(base_order[b])
                specs.append({"type": "interaction", "j": j, "k": k})
        return specs

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.feature_medians_ = np.nanmedian(X, axis=0)
        X = self._impute(X)

        self.feature_means_ = np.mean(X, axis=0)
        self.feature_stds_ = np.std(X, axis=0)
        self.feature_stds_[self.feature_stds_ < self.min_col_std] = 1.0

        base_fit = self._fit_best_ridge(X, y)
        self.base_alpha_ = base_fit["alpha"]
        self.base_intercept_ = base_fit["intercept"]
        self.base_coef_ = base_fit["coef"]

        residual = y - (self.base_intercept_ + X @ self.base_coef_)
        top = np.argsort(np.abs(self.base_coef_))[::-1]
        top = top[: min(int(self.top_base_features), self.n_features_in_)]
        specs = self._candidate_specs(X, top)

        scored = []
        for spec in specs:
            col = self._build_term(X, spec)
            col = col - np.mean(col)
            col_std = float(np.std(col))
            if col_std < self.min_col_std:
                continue
            score = abs(float(col @ residual)) / (float(np.linalg.norm(col)) * float(np.linalg.norm(residual)) + 1e-12)
            scored.append((score, spec, col))

        scored.sort(key=lambda x: x[0], reverse=True)
        selected_specs = []
        selected_cols = []
        for score, spec, col in scored:
            if score < float(self.min_extra_score):
                break
            keep = True
            for prev in selected_cols:
                denom = float(np.linalg.norm(col) * np.linalg.norm(prev)) + 1e-12
                if abs(float(col @ prev) / denom) > 0.95:
                    keep = False
                    break
            if keep:
                selected_specs.append(spec)
                selected_cols.append(col)
            if len(selected_specs) >= int(self.max_extra_terms):
                break

        self.extra_specs_ = selected_specs
        if self.extra_specs_:
            extra_mat = np.column_stack([self._build_term(X, s) for s in self.extra_specs_])
            self.extra_means_ = np.mean(extra_mat, axis=0)
            self.extra_stds_ = np.std(extra_mat, axis=0)
            self.extra_stds_[self.extra_stds_ < self.min_col_std] = 1.0
            extra_z = (extra_mat - self.extra_means_) / self.extra_stds_
            A = np.hstack([X, extra_z])
        else:
            self.extra_means_ = np.zeros(0, dtype=float)
            self.extra_stds_ = np.zeros(0, dtype=float)
            A = X

        full_fit = self._fit_best_ridge(A, y)
        self.alpha_ = full_fit["alpha"]
        coef_full = full_fit["coef"]
        self.intercept_ = float(full_fit["intercept"])
        self.linear_coef_ = coef_full[: self.n_features_in_].astype(float)
        if self.extra_specs_:
            self.extra_coef_ = (coef_full[self.n_features_in_ :] / self.extra_stds_).astype(float)
            self.intercept_ -= float(np.sum((coef_full[self.n_features_in_ :] * self.extra_means_) / self.extra_stds_))
        else:
            self.extra_coef_ = np.zeros(0, dtype=float)

        self.linear_coef_[np.abs(self.linear_coef_) < self.coef_tol] = 0.0
        if self.extra_coef_.size:
            self.extra_coef_[np.abs(self.extra_coef_) < self.coef_tol] = 0.0

        self.feature_importance_ = np.abs(self.linear_coef_).copy()
        for c, s in zip(self.extra_coef_, self.extra_specs_):
            if s["type"] in ("hinge_pos", "hinge_neg", "square"):
                self.feature_importance_[s["j"]] += abs(float(c))
            elif s["type"] == "interaction":
                self.feature_importance_[s["j"]] += 0.5 * abs(float(c))
                self.feature_importance_[s["k"]] += 0.5 * abs(float(c))
        self.selected_feature_order_ = np.argsort(self.feature_importance_)[::-1]
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "linear_coef_", "extra_specs_", "extra_coef_"])
        X = self._impute(X)
        pred = self.intercept_ + X @ self.linear_coef_
        for c, s in zip(self.extra_coef_, self.extra_specs_):
            if abs(float(c)) < self.coef_tol:
                continue
            pred += float(c) * self._build_term(X, s)
        return pred

    def __str__(self):
        check_is_fitted(self, ["intercept_", "linear_coef_", "extra_specs_", "extra_coef_"])
        lines = [
            "ResidualSymbolicRidgeRegressor",
            f"Selected ridge alpha: {self.alpha_:.4g}",
            "Exact prediction equation:",
            f"  y = {self.intercept_:+.6f}",
        ]

        lin_idx = np.argsort(np.abs(self.linear_coef_))[::-1]
        for j in lin_idx:
            c = float(self.linear_coef_[j])
            if abs(c) >= self.coef_tol:
                lines.append(f"      {c:+.6f} * x{j}")

        for c, s in sorted(
            [(float(c), s) for c, s in zip(self.extra_coef_, self.extra_specs_) if abs(float(c)) >= self.coef_tol],
            key=lambda t: -abs(t[0]),
        ):
            lines.append(f"      {c:+.6f} * {self._term_name(s)}")

        if np.sum(np.abs(self.linear_coef_) >= self.coef_tol) == 0 and np.sum(np.abs(self.extra_coef_) >= self.coef_tol) == 0:
            lines.append("      (all terms near zero; nearly constant model)")

        lines.append("")
        lines.append("Feature summary (sorted by total attributed effect):")
        for j in self.selected_feature_order_[: min(12, self.n_features_in_)]:
            lines.append(
                f"  x{int(j)}: linear={self.linear_coef_[int(j)]:+.6f}, importance={self.feature_importance_[int(j)]:.6f}"
            )
        inactive = [f"x{j}" for j in range(self.n_features_in_) if self.feature_importance_[j] < self.coef_tol]
        if inactive:
            lines.append("Features with near-zero effect: " + ", ".join(inactive))
        lines.append("To predict, sum intercept plus every listed term exactly as written.")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
ResidualSymbolicRidgeRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "ResidualSymbolicRidgeV1"
model_description = "Ridge linear backbone with up to two greedily selected symbolic residual terms (hinge/square/interaction) and explicit exact equation"
model_defs = [(model_shorthand_name, ResidualSymbolicRidgeRegressor())]

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------

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
