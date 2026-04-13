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


class MonotoneBinningAdditiveRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse additive model with one monotone binned shape per selected feature.
    Shapes are learned greedily on residuals and then linearly re-calibrated.
    """

    def __init__(
        self,
        top_features=12,
        max_terms=5,
        num_bins=6,
        min_gain=1e-4,
    ):
        self.top_features = top_features
        self.max_terms = max_terms
        self.num_bins = num_bins
        self.min_gain = min_gain

    @staticmethod
    def _safe_scale(x):
        s = np.std(x)
        if (not np.isfinite(s)) or s < 1e-12:
            return 1.0
        return float(s)

    @staticmethod
    def _centered_corr(a, b):
        ac = a - np.mean(a)
        bc = b - np.mean(b)
        denom = (np.linalg.norm(ac) + 1e-12) * (np.linalg.norm(bc) + 1e-12)
        return float(np.dot(ac, bc) / denom)

    def _pava(self, y, w, increasing=True):
        yv = np.asarray(y, dtype=float)
        wv = np.asarray(w, dtype=float)
        if not increasing:
            yv = -yv

        means = []
        weights = []
        lengths = []
        for i in range(len(yv)):
            means.append(float(yv[i]))
            weights.append(float(max(wv[i], 1e-12)))
            lengths.append(1)
            while len(means) >= 2 and means[-2] > means[-1]:
                m1, m2 = means[-2], means[-1]
                w1, w2 = weights[-2], weights[-1]
                l1, l2 = lengths[-2], lengths[-1]
                new_w = w1 + w2
                new_m = (w1 * m1 + w2 * m2) / new_w
                means[-2] = new_m
                weights[-2] = new_w
                lengths[-2] = l1 + l2
                means.pop()
                weights.pop()
                lengths.pop()

        out = np.empty(len(yv), dtype=float)
        idx = 0
        for m, l in zip(means, lengths):
            out[idx: idx + l] = m
            idx += l
        if not increasing:
            out = -out
        return out

    def _fit_feature_shape(self, z, resid, direction):
        quantiles = np.linspace(0.0, 1.0, int(self.num_bins) + 1)
        edges = np.quantile(z, quantiles)
        inner = np.unique(np.asarray(edges[1:-1], dtype=float))
        bin_ids = np.searchsorted(inner, z, side="right")
        n_bins = int(len(inner) + 1)
        if n_bins < 2:
            return None

        counts = np.bincount(bin_ids, minlength=n_bins).astype(float)
        sums = np.bincount(bin_ids, weights=resid, minlength=n_bins).astype(float)
        means = sums / np.maximum(counts, 1.0)
        shape_vals = self._pava(means, counts, increasing=(direction >= 0))
        contrib = shape_vals[bin_ids]
        contrib = contrib - np.mean(contrib)
        denom = float(np.dot(contrib, contrib))
        if denom < 1e-12:
            return None

        coef = float(np.dot(resid, contrib) / (denom + 1e-12))
        term_out = coef * contrib
        return {
            "inner_edges": inner,
            "shape_vals": shape_vals,
            "coef": coef,
            "term_out": term_out,
            "direction": 1 if direction >= 0 else -1,
        }

    def _apply_term(self, z, term):
        bin_ids = np.searchsorted(term["inner_edges"], z, side="right")
        contrib = term["shape_vals"][bin_ids]
        contrib = contrib - np.mean(contrib)
        return float(term["coef"]) * contrib

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        _, p = X.shape

        self.x_mean_ = np.mean(X, axis=0)
        self.x_scale_ = np.array([self._safe_scale(X[:, j]) for j in range(p)], dtype=float)
        Xs = (X - self.x_mean_) / self.x_scale_

        corr = np.array([self._centered_corr(Xs[:, j], y) for j in range(p)], dtype=float)
        top_k = min(int(self.top_features), p)
        feature_pool = [int(j) for j in np.argsort(-np.abs(corr))[:top_k]]

        self.intercept_ = float(np.mean(y))
        pred = np.full_like(y, self.intercept_, dtype=float)
        selected_terms = []
        used = set()
        best_rss = float(np.sum((y - pred) ** 2))

        for _ in range(int(self.max_terms)):
            resid = y - pred
            best = None
            for j in feature_pool:
                if j in used:
                    continue
                term = self._fit_feature_shape(Xs[:, j], resid, direction=np.sign(corr[j]))
                if term is None:
                    continue
                trial_pred = pred + term["term_out"]
                trial_rss = float(np.sum((y - trial_pred) ** 2))
                gain = best_rss - trial_rss
                if (best is None) or (gain > best["gain"]):
                    best = {"gain": gain, "j": j, "term": term, "pred": trial_pred, "rss": trial_rss}

            if (best is None) or (best["gain"] < float(self.min_gain)):
                break

            selected_terms.append({"feature": best["j"], **best["term"]})
            used.add(best["j"])
            pred = best["pred"]
            best_rss = best["rss"]

        if selected_terms:
            cols = [np.ones(Xs.shape[0], dtype=float)]
            for term in selected_terms:
                z = Xs[:, term["feature"]]
                raw = term["shape_vals"][np.searchsorted(term["inner_edges"], z, side="right")]
                cols.append(raw - np.mean(raw))
            design = np.column_stack(cols)
            beta, *_ = np.linalg.lstsq(design, y, rcond=None)
            self.intercept_ = float(beta[0])
            for k, term in enumerate(selected_terms, start=1):
                term["coef"] = float(beta[k])

        self.terms_ = selected_terms
        self.n_features_in_ = p
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        X = np.asarray(X, dtype=float)
        Xs = (X - self.x_mean_) / self.x_scale_

        yhat = np.full(X.shape[0], self.intercept_, dtype=float)
        for term in self.terms_:
            z = Xs[:, term["feature"]]
            bin_ids = np.searchsorted(term["inner_edges"], z, side="right")
            contrib = term["shape_vals"][bin_ids]
            contrib = contrib - np.mean(contrib)
            yhat += float(term["coef"]) * contrib
        return yhat

    def __str__(self):
        check_is_fitted(self, "is_fitted_")
        lines = [
            "Monotone Binning Additive Regressor:",
            "  y = intercept + sum_j coef_j * monotone_bin_shape_j(z_j)",
            f"  intercept: {self.intercept_:+.6f}",
            f"  num_terms: {len(self.terms_)}",
            "  Terms:",
        ]
        if len(self.terms_) == 0:
            lines.append("    none")
        else:
            for i, t in enumerate(self.terms_, start=1):
                dir_txt = "increasing" if t["direction"] >= 0 else "decreasing"
                knots = ", ".join(f"{v:.2f}" for v in t["inner_edges"][:4])
                if len(t["inner_edges"]) > 4:
                    knots += ", ..."
                lines.append(
                    f"    {i}. {float(t['coef']):+.6f} * shape(z{t['feature']}; {dir_txt}; knots=[{knots}])"
                )
        used = sorted({t["feature"] for t in self.terms_})
        lines.append("  Active features: " + (", ".join(f"x{j}" for j in used) if used else "none"))
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
MonotoneBinningAdditiveRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "MonoBinAdd_v1"
model_description = "Sparse additive model with one monotone quantile-binned shape per selected feature, fit greedily on residuals then OLS-refit"
model_defs = [(model_shorthand_name, MonotoneBinningAdditiveRegressor())]


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
