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


class SparseSmoothBackfitRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse smooth additive regressor using histogram backfitting.
    For each selected feature j, learn a 1D piecewise-constant shape f_j(x_j):
      y_hat = intercept + sum_j f_j(x_j)
    """

    def __init__(
        self,
        max_features=10,
        n_bins=12,
        n_backfit_rounds=6,
        learning_rate=0.7,
        smooth_steps=2,
        min_bin_count=5,
        feature_keep_quantile=0.7,
        display_top_k=4,
    ):
        self.max_features = max_features
        self.n_bins = n_bins
        self.n_backfit_rounds = n_backfit_rounds
        self.learning_rate = learning_rate
        self.smooth_steps = smooth_steps
        self.min_bin_count = min_bin_count
        self.feature_keep_quantile = feature_keep_quantile
        self.display_top_k = display_top_k

    @staticmethod
    def _robust_scale(X):
        med = np.median(X, axis=0)
        q25 = np.quantile(X, 0.25, axis=0)
        q75 = np.quantile(X, 0.75, axis=0)
        iqr = np.where(np.abs(q75 - q25) < 1e-8, 1.0, (q75 - q25))
        return (X - med) / iqr, med, iqr

    def _select_features(self, Z, y):
        n, p = Z.shape
        if p == 0:
            return np.zeros(0, dtype=int)
        y0 = y - np.mean(y)
        y_norm = float(np.sqrt(np.dot(y0, y0)) + 1e-12)
        corr = np.zeros(p, dtype=float)
        for j in range(p):
            zj = Z[:, j] - np.mean(Z[:, j])
            z_norm = float(np.sqrt(np.dot(zj, zj)) + 1e-12)
            corr[j] = abs(float(np.dot(zj, y0))) / (z_norm * y_norm)
        thresh = float(np.quantile(corr, float(np.clip(self.feature_keep_quantile, 0.0, 1.0))))
        kept = np.where(corr >= thresh)[0]
        if kept.size == 0:
            kept = np.array([int(np.argmax(corr))], dtype=int)
        k = int(max(1, min(int(self.max_features), kept.size)))
        top = kept[np.argsort(corr[kept])[-k:]]
        return np.sort(top.astype(int))

    def _make_edges(self, z):
        n_bins = int(max(3, self.n_bins))
        q = np.linspace(0.0, 1.0, n_bins + 1)
        e = np.quantile(z, q)
        e[0] = -np.inf
        e[-1] = np.inf
        for i in range(1, len(e)):
            if e[i] <= e[i - 1]:
                e[i] = np.nextafter(e[i - 1], np.inf)
        return e

    def _digitize(self, z, edges):
        return np.searchsorted(edges, z, side="right") - 1

    def _smooth_values(self, vals, counts):
        out = vals.copy()
        min_count = int(max(1, self.min_bin_count))
        for _ in range(int(max(0, self.smooth_steps))):
            prev = out.copy()
            for b in range(out.size):
                c0 = counts[b]
                if c0 >= min_count:
                    continue
                left = prev[b - 1] if b > 0 else prev[b]
                right = prev[b + 1] if b + 1 < out.size else prev[b]
                out[b] = 0.5 * (left + right)
            if out.size >= 3:
                mid = out.copy()
                out[1:-1] = 0.25 * mid[:-2] + 0.5 * mid[1:-1] + 0.25 * mid[2:]
        return out

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape
        self.n_features_in_ = int(p)

        if n == 0 or p == 0:
            self.intercept_ = float(np.mean(y)) if y.size > 0 else 0.0
            self.selected_features_ = np.zeros(0, dtype=int)
            self.effects_ = {}
            self.edges_ = {}
            self.medians_ = np.zeros(p, dtype=float)
            self.iqrs_ = np.ones(p, dtype=float)
            self.training_mse_ = float(np.mean((y - self.intercept_) ** 2)) if y.size > 0 else 0.0
            self.is_fitted_ = True
            return self

        Z, med, iqr = self._robust_scale(X)
        selected = self._select_features(Z, y)

        edges = {}
        bin_ids = {}
        counts = {}
        effects = {}
        for j in selected:
            e = self._make_edges(Z[:, j])
            b = self._digitize(Z[:, j], e)
            nb = len(e) - 1
            c = np.bincount(b, minlength=nb).astype(float)
            edges[int(j)] = e
            bin_ids[int(j)] = b
            counts[int(j)] = c
            effects[int(j)] = np.zeros(nb, dtype=float)

        intercept = float(np.mean(y))
        pred = np.full(n, intercept, dtype=float)

        for _ in range(int(max(1, self.n_backfit_rounds))):
            for j in selected:
                ji = int(j)
                b = bin_ids[ji]
                c = counts[ji]
                old = effects[ji]
                pred -= old[b]
                residual = y - pred
                numer = np.bincount(b, weights=residual, minlength=old.size)
                raw = np.divide(numer, np.maximum(c, 1.0))
                smoothed = self._smooth_values(raw, c)
                # weighted centering keeps additive effects identifiable
                centered = smoothed - float(np.dot(smoothed, c) / np.maximum(np.sum(c), 1.0))
                new = (1.0 - float(self.learning_rate)) * old + float(self.learning_rate) * centered
                effects[ji] = new
                pred += new[b]

        self.intercept_ = intercept
        self.selected_features_ = selected
        self.effects_ = effects
        self.edges_ = edges
        self.medians_ = med
        self.iqrs_ = iqr
        self.training_mse_ = float(np.mean((y - pred) ** 2))
        self.total_effect_l1_ = float(sum(np.sum(np.abs(v)) for v in effects.values()))
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        X = np.asarray(X, dtype=float)
        Z = (X - self.medians_) / self.iqrs_
        y = np.full(X.shape[0], self.intercept_, dtype=float)
        for j in self.selected_features_:
            ji = int(j)
            e = self.edges_[ji]
            vals = self.effects_[ji]
            b = np.searchsorted(e, Z[:, ji], side="right") - 1
            b = np.clip(b, 0, vals.size - 1)
            y += vals[b]
        return y

    def __str__(self):
        check_is_fitted(self, "is_fitted_")
        lines = ["Sparse Smooth Backfit Regressor:"]
        lines.append("  prediction = intercept + sum of selected 1D feature shape functions")
        lines.append(f"  intercept: {self.intercept_:+.4f}")
        lines.append(f"  selected features: {len(self.selected_features_)}/{self.n_features_in_}")
        lines.append(f"  training MSE: {self.training_mse_:.6f}")
        if len(self.selected_features_) == 0:
            lines.append("  no feature effects")
            return "\n".join(lines)

        ranked = sorted(
            [int(j) for j in self.selected_features_],
            key=lambda j: float(np.max(np.abs(self.effects_[j]))),
            reverse=True,
        )
        top_k = int(max(1, self.display_top_k))
        shown = ranked[:top_k]
        lines.append(f"  top feature effects shown: {len(shown)}")
        for j in shown:
            e = self.edges_[j]
            v = self.effects_[j]
            amp = float(np.max(v) - np.min(v))
            lines.append(
                f"    x{j}: robust z=(x{j}-{self.medians_[j]:.4f})/{self.iqrs_[j]:.4f}, "
                f"shape amplitude={amp:.4f}"
            )
            qs = [0.1, 0.5, 0.9]
            cuts = np.quantile(e[1:-1], qs) if len(e) > 3 else np.array([0.0, 0.0, 0.0])
            for c in cuts:
                bi = int(np.clip(np.searchsorted(e, c, side="right") - 1, 0, len(v) - 1))
                lines.append(f"      near z={c:+.3f}: contribution {v[bi]:+.4f}")
        if len(ranked) > len(shown):
            lines.append(f"  ... plus {len(ranked) - len(shown)} smaller feature effects")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparseSmoothBackfitRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SparseSmoothBackfit_v1"
model_description = "Sparse additive histogram-backfitting regressor with robust scaling, smoothed bin effects, and shrinkage updates"
model_defs = [(model_shorthand_name, SparseSmoothBackfitRegressor())]


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
