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


class SparseBackfitBinsRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse additive quantile-binning regressor with smoothed backfitting.

    Each selected feature gets a small 1D shape function over quantile bins.
    Shapes are learned by cyclic residual backfitting, lightly smoothed across
    neighboring bins, then linearly reweighted in a final ridge refit.
    """

    def __init__(
        self,
        max_features=6,
        screening_features=14,
        n_bins=8,
        n_backfit_rounds=5,
        smooth_strength=0.35,
        shrinkage=0.9,
        refit_l2=0.15,
        min_bin_count=8,
    ):
        self.max_features = max_features
        self.screening_features = screening_features
        self.n_bins = n_bins
        self.n_backfit_rounds = n_backfit_rounds
        self.smooth_strength = smooth_strength
        self.shrinkage = shrinkage
        self.refit_l2 = refit_l2
        self.min_bin_count = min_bin_count

    @staticmethod
    def _safe_corr(x, y):
        xc = x - np.mean(x)
        yc = y - np.mean(y)
        denom = (np.linalg.norm(xc) + 1e-12) * (np.linalg.norm(yc) + 1e-12)
        return float(abs(np.dot(xc, yc) / denom))

    @staticmethod
    def _ridge_fit(M, y, l2):
        n = M.shape[0]
        D = np.column_stack([np.ones(n, dtype=float), M])
        reg = float(l2) * np.eye(D.shape[1], dtype=float)
        reg[0, 0] = 0.0
        beta = np.linalg.solve(D.T @ D + reg, D.T @ y)
        return float(beta[0]), np.asarray(beta[1:], dtype=float)

    def _build_edges(self, x):
        q = max(3, int(self.n_bins) + 1)
        grid = np.linspace(0.0, 1.0, q)
        edges = np.quantile(x, grid)
        edges = np.asarray(edges, dtype=float)

        # Enforce strictly increasing edges for stable digitization.
        for i in range(1, edges.size):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + 1e-9
        return edges

    @staticmethod
    def _digitize(x, edges):
        # Map into [0, n_bins-1] using interior cut points.
        ids = np.searchsorted(edges[1:-1], x, side="right")
        return np.asarray(ids, dtype=int)

    def _smooth_shape(self, vals):
        lam = float(self.smooth_strength)
        if vals.size <= 2 or lam <= 1e-12:
            return vals
        out = vals.copy()
        # One tridiagonal-style Jacobi step is enough for light smoothing.
        for i in range(1, vals.size - 1):
            out[i] = (vals[i] + lam * (vals[i - 1] + vals[i + 1])) / (1.0 + 2.0 * lam)
        return out

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape

        k_screen = min(max(1, int(self.screening_features)), p)
        corrs = np.array([self._safe_corr(X[:, j], y) for j in range(p)], dtype=float)
        screened = list(np.argsort(-corrs)[:k_screen])

        edges_by_feat = {}
        bins_by_feat = {}
        nbin_by_feat = {}
        for j in screened:
            edges = self._build_edges(X[:, j])
            bid = self._digitize(X[:, j], edges)
            edges_by_feat[int(j)] = edges
            bins_by_feat[int(j)] = bid
            nbin_by_feat[int(j)] = int(edges.size - 1)

        shape_by_feat = {
            int(j): np.zeros(nbin_by_feat[int(j)], dtype=float)
            for j in screened
        }

        intercept = float(np.mean(y))
        pred = np.full(n, intercept, dtype=float)

        for _ in range(max(1, int(self.n_backfit_rounds))):
            for j in screened:
                j = int(j)
                bid = bins_by_feat[j]
                cur = shape_by_feat[j]

                pred_wo_j = pred - cur[bid]
                resid = y - pred_wo_j

                new_shape = np.zeros_like(cur)
                counts = np.bincount(bid, minlength=cur.size)
                sums = np.bincount(bid, weights=resid, minlength=cur.size)

                for b in range(cur.size):
                    if counts[b] >= int(self.min_bin_count):
                        new_shape[b] = sums[b] / counts[b]
                    else:
                        new_shape[b] = 0.0

                new_shape = self._smooth_shape(new_shape)
                weighted_mean = float(np.dot(new_shape, counts) / (np.sum(counts) + 1e-12))
                new_shape -= weighted_mean
                new_shape *= float(self.shrinkage)

                shape_by_feat[j] = new_shape
                pred = pred_wo_j + new_shape[bid]

        # Rank feature importance by contribution variance and keep only a few.
        amps = []
        for j in screened:
            j = int(j)
            bid = bins_by_feat[j]
            contrib = shape_by_feat[j][bid]
            amps.append((j, float(np.std(contrib))))
        amps.sort(key=lambda t: -t[1])

        k_keep = min(max(1, int(self.max_features)), len(amps))
        kept = [j for j, _ in amps[:k_keep] if _ > 1e-10]
        if len(kept) == 0:
            kept = [int(amps[0][0])]

        M = np.column_stack([shape_by_feat[j][bins_by_feat[j]] for j in kept])
        refit_intercept, refit_coef = self._ridge_fit(M, y, self.refit_l2)

        # Tiny coefficients are dropped for cleaner explanations.
        keep_cols = [i for i, w in enumerate(refit_coef) if abs(float(w)) >= 0.05]
        if len(keep_cols) == 0:
            self.intercept_ = float(np.mean(y))
            self.kept_features_ = []
            self.edges_ = {}
            self.shapes_ = {}
            self.weights_ = np.zeros(0, dtype=float)
            self.n_features_in_ = p
            self.is_fitted_ = True
            return self

        kept_final = [kept[i] for i in keep_cols]
        weights_final = np.asarray([refit_coef[i] for i in keep_cols], dtype=float)

        self.intercept_ = refit_intercept
        self.kept_features_ = [int(j) for j in kept_final]
        self.edges_ = {int(j): edges_by_feat[int(j)] for j in kept_final}
        self.shapes_ = {int(j): shape_by_feat[int(j)] for j in kept_final}
        self.weights_ = weights_final
        self.n_features_in_ = p
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        yhat = np.full(n, self.intercept_, dtype=float)

        for w, j in zip(self.weights_, self.kept_features_):
            edges = self.edges_[j]
            bid = self._digitize(X[:, j], edges)
            yhat += float(w) * self.shapes_[j][bid]
        return yhat

    def __str__(self):
        check_is_fitted(self, "is_fitted_")
        lines = ["Sparse Backfit Bins Regressor:"]
        lines.append(f"  Base value: {self.intercept_:+.4f}")

        if len(self.kept_features_) == 0:
            lines.append("  Terms: none")
            return "\n".join(lines)

        lines.append("  Prediction = base + sum_j w_j * g_j(x_j)")
        for w, j in zip(self.weights_, self.kept_features_):
            edges = self.edges_[j]
            shape = self.shapes_[j]
            lines.append(f"  Feature x{j}: weight {w:+.3f}")
            for b in range(shape.size):
                lo = edges[b]
                hi = edges[b + 1]
                lines.append(f"    if {lo:.3g} <= x{j} < {hi:.3g}: g={shape[b]:+.3f}")
        return "\n".join(lines)



# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparseBackfitBinsRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SparseBackfitBins_v1"
model_description = "Sparse additive quantile-bin backfitting model with neighbor-smoothed feature shapes and ridge reweighting"
model_defs = [(model_shorthand_name, SparseBackfitBinsRegressor())]


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
