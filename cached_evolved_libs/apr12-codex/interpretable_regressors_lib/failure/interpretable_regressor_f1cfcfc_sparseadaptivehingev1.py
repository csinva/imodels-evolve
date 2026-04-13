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


class SparseAdaptiveHingeRegressor(BaseEstimator, RegressorMixin):
    """
    Compact additive equation selected greedily from linear and hinge basis terms.
    """

    def __init__(
        self,
        max_terms=8,
        hinge_quantiles=(0.2, 0.4, 0.6, 0.8),
        min_term_std=1e-6,
        min_mse_gain=5e-5,
        max_candidates=700,
        coef_tol=1e-6,
    ):
        self.max_terms = max_terms
        self.hinge_quantiles = hinge_quantiles
        self.min_term_std = min_term_std
        self.min_mse_gain = min_mse_gain
        self.max_candidates = max_candidates
        self.coef_tol = coef_tol

    def _impute(self, X):
        X = np.asarray(X, dtype=float).copy()
        mask = ~np.isfinite(X)
        if mask.any():
            X[mask] = np.take(self.feature_medians_, np.where(mask)[1])
        return X

    def _build_candidates(self, X):
        n, p = X.shape
        mats = []
        metas = []
        for j in range(p):
            xj = X[:, j]
            mats.append(xj)
            metas.append({"kind": "linear", "j": int(j)})
            qs = np.unique(np.quantile(xj, self.hinge_quantiles))
            for thr in qs:
                thr = float(thr)
                mats.append(np.maximum(0.0, xj - thr))
                metas.append({"kind": "hinge_pos", "j": int(j), "threshold": thr})
                mats.append(np.maximum(0.0, thr - xj))
                metas.append({"kind": "hinge_neg", "j": int(j), "threshold": thr})

        if not mats:
            return np.empty((n, 0), dtype=float), [], np.array([], dtype=float), np.array([], dtype=float)

        Z = np.column_stack(mats).astype(float)
        means = np.mean(Z, axis=0)
        Z = Z - means
        scales = np.sqrt(np.mean(Z ** 2, axis=0))
        keep = scales > self.min_term_std
        if keep.sum() == 0:
            return np.empty((n, 0), dtype=float), [], np.array([], dtype=float), np.array([], dtype=float)

        Z = Z[:, keep]
        means = means[keep]
        scales = scales[keep]
        metas = [metas[i] for i in np.where(keep)[0]]

        return Z, metas, means, scales

    def _eval_term(self, X, meta):
        xj = X[:, meta["j"]]
        if meta["kind"] == "linear":
            return xj
        if meta["kind"] == "hinge_pos":
            return np.maximum(0.0, xj - meta["threshold"])
        return np.maximum(0.0, meta["threshold"] - xj)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_features_in_ = X.shape[1]

        self.feature_medians_ = np.nanmedian(X, axis=0)
        X = self._impute(X)

        y_mean = float(np.mean(y))
        y_center = y - y_mean
        self.intercept_ = y_mean

        Z_raw, meta_raw, z_means_raw, z_scales_raw = self._build_candidates(X)
        if Z_raw.shape[1] == 0:
            self.selected_terms_ = []
            self.selected_coef_ = np.array([], dtype=float)
            self.feature_importance_ = np.zeros(self.n_features_in_, dtype=float)
            return self

        if Z_raw.shape[1] > self.max_candidates:
            scores = np.abs(Z_raw.T @ y_center) / (z_scales_raw + 1e-12)
            keep_idx = np.argsort(scores)[::-1][: self.max_candidates]
            keep_idx = np.sort(keep_idx)
            Z = Z_raw[:, keep_idx]
            metas = [meta_raw[i] for i in keep_idx]
            z_means = z_means_raw[keep_idx]
        else:
            Z = Z_raw
            metas = meta_raw
            z_means = z_means_raw

        selected = []
        coef_sel = np.array([], dtype=float)
        y_hat = np.zeros_like(y_center)
        residual = y_center - y_hat
        prev_mse = float(np.mean(residual ** 2))

        for _ in range(self.max_terms):
            best_idx = None
            best_score = -1.0
            for idx in range(Z.shape[1]):
                if idx in selected:
                    continue
                zj = Z[:, idx]
                score = abs(float(zj @ residual)) / (float(np.sqrt(np.mean(zj ** 2))) + 1e-12)
                if score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx is None:
                break

            trial_sel = selected + [best_idx]
            Zs = Z[:, trial_sel]
            coef = np.linalg.lstsq(Zs, y_center, rcond=None)[0]
            y_trial = Zs @ coef
            mse_trial = float(np.mean((y_center - y_trial) ** 2))
            if (prev_mse - mse_trial) < self.min_mse_gain:
                break
            selected = trial_sel
            coef_sel = coef
            y_hat = y_trial
            prev_mse = mse_trial
            residual = y_center - y_hat

        if len(selected) == 0:
            self.selected_terms_ = []
            self.selected_coef_ = np.array([], dtype=float)
            self.feature_importance_ = np.zeros(self.n_features_in_, dtype=float)
            return self

        kept = [i for i, c in enumerate(coef_sel) if abs(float(c)) > self.coef_tol]
        selected = [selected[i] for i in kept]
        coef_sel = np.array([coef_sel[i] for i in kept], dtype=float)

        if len(selected) == 0:
            self.selected_terms_ = []
            self.selected_coef_ = np.array([], dtype=float)
            self.feature_importance_ = np.zeros(self.n_features_in_, dtype=float)
            return self

        Zs = Z[:, selected]
        coef_sel = np.linalg.lstsq(Zs, y_center, rcond=None)[0]

        self.selected_terms_ = []
        self.selected_coef_ = coef_sel.astype(float)
        for local_i, global_idx in enumerate(selected):
            meta = dict(metas[global_idx])
            meta["mean"] = float(z_means[global_idx])
            meta["coef"] = float(self.selected_coef_[local_i])
            self.selected_terms_.append(meta)

        fi = np.zeros(self.n_features_in_, dtype=float)
        for term in self.selected_terms_:
            fi[term["j"]] += abs(term["coef"])
        self.feature_importance_ = fi
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "selected_terms_", "selected_coef_", "feature_importance_"])
        X = self._impute(X)
        y_hat = np.full(X.shape[0], self.intercept_, dtype=float)
        for term in self.selected_terms_:
            z = self._eval_term(X, term) - term["mean"]
            y_hat += term["coef"] * z
        return y_hat

    def __str__(self):
        check_is_fitted(self, ["intercept_", "selected_terms_", "selected_coef_", "feature_importance_"])
        lines = [
            "SparseAdaptiveHingeRegressor",
            "Prediction rule (single compact additive equation):",
            f"y = {self.intercept_:+.5f}",
        ]
        ordered = sorted(self.selected_terms_, key=lambda t: abs(float(t["coef"])), reverse=True)
        for term in ordered:
            if term["kind"] == "linear":
                basis = f"(x{term['j']} - {term['mean']:.5f})"
            elif term["kind"] == "hinge_pos":
                basis = f"(max(0, x{term['j']} - {term['threshold']:.5f}) - {term['mean']:.5f})"
            else:
                basis = f"(max(0, {term['threshold']:.5f} - x{term['j']}) - {term['mean']:.5f})"
            lines.append(f"  {term['coef']:+.5f} * {basis}")

        lines.append("")
        lines.append(f"Active terms: {len(self.selected_terms_)}")
        lines.append("Feature importance (sum of |active coefficients| by feature):")
        for j in np.argsort(self.feature_importance_)[::-1]:
            if self.feature_importance_[j] > 0:
                lines.append(f"  x{j}: {self.feature_importance_[j]:.5f}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparseAdaptiveHingeRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SparseAdaptiveHingeV1"
model_description = "Forward-selected compact additive regressor using sparse linear and hinge basis terms with an explicit arithmetic equation"
model_defs = [(model_shorthand_name, SparseAdaptiveHingeRegressor())]


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
