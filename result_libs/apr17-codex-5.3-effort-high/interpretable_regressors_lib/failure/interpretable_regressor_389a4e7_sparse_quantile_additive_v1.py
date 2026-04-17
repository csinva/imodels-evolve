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


class SparseQuantileAdditiveRegressor(BaseEstimator, RegressorMixin):
    """
    From-scratch sparse additive regressor:
    - selects a small set of features by absolute correlation with y
    - learns one piecewise-constant contribution function per selected feature
      using cyclic backfitting over quantile bins
    """

    def __init__(
        self,
        max_active_features=6,
        n_bins=6,
        n_backfit_rounds=14,
        min_bin_weight=25.0,
    ):
        self.max_active_features = max_active_features
        self.n_bins = n_bins
        self.n_backfit_rounds = n_backfit_rounds
        self.min_bin_weight = min_bin_weight

    @staticmethod
    def _safe_std(v):
        s = float(np.std(v))
        return s if s > 1e-12 else 1.0

    def _feature_scores(self, X, y):
        yc = y - float(np.mean(y))
        ysd = self._safe_std(yc)
        scores = np.zeros(X.shape[1], dtype=float)
        for j in range(X.shape[1]):
            xj = X[:, j]
            xjc = xj - float(np.mean(xj))
            xsd = self._safe_std(xjc)
            corr = float(np.dot(xjc, yc) / (len(y) * xsd * ysd))
            scores[j] = abs(corr)
        return scores

    def _build_edges(self, x):
        n_bins = int(max(2, self.n_bins))
        q = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(x, q)
        edges = np.unique(edges.astype(float))
        if len(edges) < 2:
            c = float(np.mean(x))
            edges = np.array([c - 1.0, c + 1.0], dtype=float)
        return edges

    @staticmethod
    def _digitize(x, edges):
        # bins are [edge[k], edge[k+1]) except final bin includes the right edge
        return np.searchsorted(edges[1:-1], x, side="right").astype(int)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape
        self.n_features_in_ = p
        self.intercept_ = float(np.mean(y))

        k = int(max(1, min(self.max_active_features, p)))
        scores = self._feature_scores(X, y)
        active = np.argsort(-scores)[:k].astype(int)
        active = np.sort(active)
        inactive = np.array([j for j in range(p) if j not in set(active.tolist())], dtype=int)

        self.active_features_ = active
        self.negligible_features_ = inactive
        self.feature_scores_ = scores

        self.bin_edges_ = {}
        self.bin_values_ = {}
        train_bins = {}
        for j in self.active_features_:
            edges = self._build_edges(X[:, j])
            bins = self._digitize(X[:, j], edges)
            nb = len(edges) - 1
            self.bin_edges_[int(j)] = edges
            self.bin_values_[int(j)] = np.zeros(nb, dtype=float)
            train_bins[int(j)] = bins

        pred = np.full(n, self.intercept_, dtype=float)
        max_rounds = int(max(1, self.n_backfit_rounds))
        shrink_weight = float(max(1.0, self.min_bin_weight))

        for _ in range(max_rounds):
            for j in self.active_features_:
                jj = int(j)
                bins = train_bins[jj]
                old_vals = self.bin_values_[jj]
                old_contrib = old_vals[bins]
                residual = y - (pred - old_contrib)

                nb = len(old_vals)
                sums = np.bincount(bins, weights=residual, minlength=nb).astype(float)
                cnts = np.bincount(bins, minlength=nb).astype(float)
                means = np.divide(sums, np.maximum(cnts, 1.0))
                weights = cnts / (cnts + shrink_weight)
                new_vals = weights * means

                # Keep each per-feature contribution centered so intercept remains global level.
                center = float(np.sum(new_vals * cnts) / max(np.sum(cnts), 1.0))
                new_vals = new_vals - center

                self.bin_values_[jj] = new_vals
                pred = pred - old_contrib + new_vals[bins]

        self.train_rmse_ = float(np.sqrt(np.mean((y - pred) ** 2)))
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "active_features_", "bin_edges_", "bin_values_"])
        X = np.asarray(X, dtype=float)
        out = np.full(X.shape[0], self.intercept_, dtype=float)
        for j in self.active_features_:
            jj = int(j)
            bins = self._digitize(X[:, jj], self.bin_edges_[jj])
            out += self.bin_values_[jj][bins]
        return out

    def _piecewise_lines(self, j):
        edges = self.bin_edges_[j]
        vals = self.bin_values_[j]
        lines = []
        for k, v in enumerate(vals):
            lo = float(edges[k])
            hi = float(edges[k + 1])
            if k == 0:
                cond = f"x{j} <= {hi:.4f}"
            elif k == len(vals) - 1:
                cond = f"x{j} > {lo:.4f}"
            else:
                cond = f"{lo:.4f} < x{j} <= {hi:.4f}"
            lines.append(f"    if {cond}: add {float(v):+.6f}")
        return lines

    def __str__(self):
        check_is_fitted(self, ["intercept_", "active_features_", "bin_edges_", "bin_values_"])
        lines = [
            "Sparse Quantile Additive Regressor",
            "Exact prediction recipe:",
            f"  y = {self.intercept_:+.6f} + sum_j g_j(xj) over active features",
            f"Active features ({len(self.active_features_)}): " + ", ".join(f"x{int(j)}" for j in self.active_features_),
        ]
        if len(self.negligible_features_) > 0:
            lines.append(
                "Features with negligible/zero effect in this fitted model: "
                + ", ".join(f"x{int(j)}" for j in self.negligible_features_)
            )
        lines.append("Each g_j(xj) is a piecewise-constant lookup:")
        for j in self.active_features_:
            jj = int(j)
            lines.append(f"  g_{jj}(x{jj}):")
            lines.extend(self._piecewise_lines(jj))
        lines.append(
            f"Computation count: 1 intercept + {len(self.active_features_)} feature lookups/additions "
            f"(<= {1 + len(self.active_features_)} core operations)."
        )
        lines.append(f"Training RMSE: {self.train_rmse_:.6f}")
        return "\n".join(lines)


import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparseQuantileAdditiveRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "SparseQuantileAdditiveV1"
model_description = "From-scratch sparse additive backfitting model using quantile-bin piecewise feature contributions with explicit per-feature lookup rules"
model_defs = [(model_shorthand_name, SparseQuantileAdditiveRegressor())]

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
