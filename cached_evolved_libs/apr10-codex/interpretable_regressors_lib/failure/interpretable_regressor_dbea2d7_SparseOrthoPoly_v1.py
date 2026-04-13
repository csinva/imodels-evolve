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


class SparseOrthoPolyAdditiveRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse additive regressor with orthogonal polynomial blocks per feature.

    For each selected feature j, the model adds:
      c1 * z + c2 * (z^2 - 1) + c3 * (z^3 - 3z), where z is standardized x_j.
    Feature blocks are selected greedily by residual reduction with complexity
    penalty, then jointly refit with ridge stabilization.
    """

    def __init__(
        self,
        top_features=14,
        max_groups=5,
        degree=3,
        l2=5e-3,
        complexity_penalty=1.5e-3,
        min_gain=1e-4,
    ):
        self.top_features = top_features
        self.max_groups = max_groups
        self.degree = degree
        self.l2 = l2
        self.complexity_penalty = complexity_penalty
        self.min_gain = min_gain

    @staticmethod
    def _safe_scale(col):
        s = np.std(col)
        if (not np.isfinite(s)) or s < 1e-12:
            return 1.0
        return float(s)

    @staticmethod
    def _corr(x, y):
        xc = x - np.mean(x)
        yc = y - np.mean(y)
        denom = (np.linalg.norm(xc) + 1e-12) * (np.linalg.norm(yc) + 1e-12)
        return float(np.dot(xc, yc) / denom)

    def _basis_block(self, z):
        cols = []
        b1 = z.copy()
        cols.append(b1)
        if int(self.degree) >= 2:
            cols.append(z * z - 1.0)
        if int(self.degree) >= 3:
            cols.append(z * z * z - 3.0 * z)

        normed = []
        for c in cols:
            c = np.asarray(c, dtype=float) - float(np.mean(c))
            nrm = float(np.linalg.norm(c))
            if nrm < 1e-12:
                continue
            normed.append(c / nrm)
        if not normed:
            return np.zeros((z.shape[0], 0), dtype=float)
        return np.column_stack(normed)

    def _fit_block(self, B, r):
        if B.shape[1] == 0:
            return None
        A = B.T @ B + float(self.l2) * np.eye(B.shape[1])
        b = B.T @ r
        coef = np.linalg.solve(A, b)
        fit = B @ coef
        gain = float(np.dot(fit, fit))
        return coef, gain

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape

        self.x_mean_ = np.mean(X, axis=0)
        self.x_scale_ = np.array([self._safe_scale(X[:, j]) for j in range(p)], dtype=float)
        Xs = (X - self.x_mean_) / self.x_scale_

        self.intercept_ = float(np.mean(y))
        yc = y - self.intercept_
        y_var = float(np.var(yc) + 1e-12)

        corr = np.array([self._corr(Xs[:, j], yc) for j in range(p)], dtype=float)
        pool_size = min(max(1, int(self.top_features)), p)
        feature_pool = [int(j) for j in np.argsort(-np.abs(corr))[:pool_size]]

        blocks = {j: self._basis_block(Xs[:, j]) for j in feature_pool}

        selected = []
        selected_set = set()
        resid = yc.copy()

        for _ in range(int(self.max_groups)):
            best = None
            for j in feature_pool:
                if j in selected_set:
                    continue
                B = blocks[j]
                res = self._fit_block(B, resid)
                if res is None:
                    continue
                coef, gain = res
                penalized_gain = gain - float(self.complexity_penalty) * B.shape[1] * y_var
                if (best is None) or (penalized_gain > best["penalized_gain"]):
                    best = {
                        "j": j,
                        "coef": coef,
                        "gain": gain,
                        "penalized_gain": penalized_gain,
                    }

            if best is None:
                break
            if best["gain"] < float(self.min_gain) or best["penalized_gain"] <= 0.0:
                break

            j = int(best["j"])
            selected.append(j)
            selected_set.add(j)
            resid -= blocks[j] @ best["coef"]

        self.groups_ = []
        if selected:
            design = np.column_stack([blocks[j] for j in selected])
            A = design.T @ design + float(self.l2) * np.eye(design.shape[1])
            b = design.T @ yc
            coef_all = np.linalg.solve(A, b)

            start = 0
            for j in selected:
                d = blocks[j].shape[1]
                c = np.asarray(coef_all[start:start + d], dtype=float)
                start += d
                self.groups_.append({"j": int(j), "coef": c})

            norms = [float(np.linalg.norm(g["coef"])) for g in self.groups_]
            max_norm = max(norms) if norms else 0.0
            keep_thresh = max(1e-8, 0.08 * max_norm)
            self.groups_ = [g for g in self.groups_ if float(np.linalg.norm(g["coef"])) >= keep_thresh]

        self.n_features_in_ = p
        self.is_fitted_ = True
        return self

    def _predict_group(self, z, coef):
        B = self._basis_block(z)
        if B.shape[1] == 0:
            return np.zeros(z.shape[0], dtype=float)
        d = min(B.shape[1], coef.shape[0])
        return B[:, :d] @ coef[:d]

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        X = np.asarray(X, dtype=float)
        Xs = (X - self.x_mean_) / self.x_scale_
        yhat = np.full(X.shape[0], self.intercept_, dtype=float)
        for g in self.groups_:
            yhat += self._predict_group(Xs[:, g["j"]], g["coef"])
        return yhat

    def __str__(self):
        check_is_fitted(self, "is_fitted_")
        lines = [
            "Sparse Orthogonal Polynomial Additive Regressor:",
            "  y = intercept + sum_j [c1*z + c2*(z^2-1) + c3*(z^3-3z)]",
            f"  intercept: {self.intercept_:+.6f}",
            f"  active_features: {len(self.groups_)}",
            "  Feature effects:",
        ]
        if not self.groups_:
            lines.append("    none")
            return "\n".join(lines)

        for i, g in enumerate(sorted(self.groups_, key=lambda t: -np.linalg.norm(t["coef"])), start=1):
            j = int(g["j"])
            c = g["coef"]
            c1 = float(c[0]) if c.shape[0] >= 1 else 0.0
            c2 = float(c[1]) if c.shape[0] >= 2 else 0.0
            c3 = float(c[2]) if c.shape[0] >= 3 else 0.0
            delta_low = (-c1) + c3 * 2.0
            delta_high = c1 - c3 * 2.0
            trend = "raises" if (delta_high - delta_low) >= 0 else "lowers"
            lines.append(
                f"    {i}. x{j}: {c1:+.4f}*z + {c2:+.4f}*(z^2-1) + {c3:+.4f}*(z^3-3z) "
                f"({trend} predictions as x{j} increases)"
            )
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparseOrthoPolyAdditiveRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SparseOrthoPoly_v1"
model_description = "Sparse additive regressor selecting feature-wise orthogonal polynomial blocks (up to cubic) with penalized greedy group selection and joint ridge refit"
model_defs = [(model_shorthand_name, SparseOrthoPolyAdditiveRegressor())]


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
