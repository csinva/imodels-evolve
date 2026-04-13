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


class AdaptiveCalibratedSparseRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse additive regressor with one human-readable calibration per selected feature.

    Workflow:
    1) robust-standardize every feature (median + IQR),
    2) for each feature choose one calibration from:
       - identity: z
       - signed_log: sign(z) * log1p(|z|)
       - saturation: z / (1 + |z|)
       by highest absolute correlation with y,
    3) greedily add calibrated features that improve RSS, with a ridge refit.
    """

    def __init__(
        self,
        screening_features=16,
        max_terms=7,
        l2=2e-3,
        min_rel_rss_gain=0.004,
        coef_keep_ratio=0.07,
    ):
        self.screening_features = screening_features
        self.max_terms = max_terms
        self.l2 = l2
        self.min_rel_rss_gain = min_rel_rss_gain
        self.coef_keep_ratio = coef_keep_ratio

    @staticmethod
    def _corr_abs(x, y):
        xc = x - np.mean(x)
        yc = y - np.mean(y)
        denom = (np.linalg.norm(xc) + 1e-12) * (np.linalg.norm(yc) + 1e-12)
        return float(abs(np.dot(xc, yc) / denom))

    @staticmethod
    def _identity(z):
        return z

    @staticmethod
    def _signed_log(z):
        return np.sign(z) * np.log1p(np.abs(z))

    @staticmethod
    def _saturation(z):
        return z / (1.0 + np.abs(z))

    def _robust_scale(self, X):
        med = np.median(X, axis=0)
        q25 = np.percentile(X, 25, axis=0)
        q75 = np.percentile(X, 75, axis=0)
        iqr = q75 - q25
        scale = np.where(iqr > 1e-12, iqr, 1.0)
        Z = (X - med) / scale
        return Z, med, scale

    def _fit_ridge_subset(self, X, y, col_idx):
        n = X.shape[0]
        if len(col_idx) == 0:
            intercept = float(np.mean(y)) if n > 0 else 0.0
            return intercept, np.zeros(0, dtype=float)

        Z = X[:, col_idx]
        D = np.column_stack([np.ones(n, dtype=float), Z])
        reg = float(self.l2) * np.eye(D.shape[1])
        reg[0, 0] = 0.0  # do not penalize intercept
        A = D.T @ D + reg
        b = D.T @ y
        beta = np.linalg.solve(A, b)
        return float(beta[0]), np.asarray(beta[1:], dtype=float)

    def _predict_linear(self, X, col_idx, intercept, coef):
        if len(col_idx) == 0:
            return np.full(X.shape[0], intercept, dtype=float)
        return intercept + X[:, col_idx] @ coef

    def _prune_coefs(self, coef):
        if coef.size == 0:
            return coef
        m = float(np.max(np.abs(coef)))
        if m <= 1e-12:
            return np.zeros_like(coef)
        keep = np.abs(coef) >= float(self.coef_keep_ratio) * m
        if np.sum(keep) == 0:
            keep[np.argmax(np.abs(coef))] = True
        out = coef.copy()
        out[~keep] = 0.0
        return out

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape

        Z, med, scale = self._robust_scale(X)
        transform_bank = [
            ("identity", self._identity),
            ("signed_log", self._signed_log),
            ("saturation", self._saturation),
        ]

        screening = np.array([self._corr_abs(Z[:, j], y) for j in range(p)], dtype=float)
        k_screen = min(max(1, int(self.screening_features)), p)
        screened_feats = np.argsort(-screening)[:k_screen].astype(int)

        feat_to_transform = {}
        calibrated_columns = []
        for j in screened_feats:
            zj = Z[:, j]
            best_name = "identity"
            best_col = zj
            best_score = -1.0
            for tname, tfun in transform_bank:
                c = tfun(zj)
                score = self._corr_abs(c, y)
                if score > best_score:
                    best_score = score
                    best_name = tname
                    best_col = c
            feat_to_transform[int(j)] = best_name
            calibrated_columns.append(best_col)

        if len(calibrated_columns) == 0:
            Phi = np.zeros((n, 0), dtype=float)
        else:
            Phi = np.column_stack(calibrated_columns)

        chosen = []
        intercept, coef = self._fit_ridge_subset(Phi, y, chosen)
        pred = self._predict_linear(Phi, chosen, intercept, coef)
        rss = float(np.sum((y - pred) ** 2))

        max_terms = min(max(1, int(self.max_terms)), Phi.shape[1] if Phi.ndim == 2 else 0)
        candidates = list(range(Phi.shape[1]))
        while len(chosen) < max_terms and len(candidates) > 0:
            best_col_idx = None
            best_rss = rss
            best_fit = None
            for c in candidates:
                trial = chosen + [c]
                ti, tc = self._fit_ridge_subset(Phi, y, trial)
                tp = self._predict_linear(Phi, trial, ti, tc)
                trss = float(np.sum((y - tp) ** 2))
                if trss < best_rss:
                    best_rss = trss
                    best_col_idx = c
                    best_fit = (ti, tc)

            if best_col_idx is None:
                break

            rel_gain = (rss - best_rss) / (rss + 1e-12)
            if rel_gain < float(self.min_rel_rss_gain):
                break

            chosen.append(best_col_idx)
            candidates.remove(best_col_idx)
            intercept, coef = best_fit
            rss = best_rss

        coef = self._prune_coefs(np.asarray(coef, dtype=float))
        if len(chosen) > 0:
            keep_mask = np.abs(coef) > 0.0
            chosen = [c for c, keep in zip(chosen, keep_mask) if keep]
            coef = coef[keep_mask]

        selected_features = [int(screened_feats[c]) for c in chosen]
        selected_transforms = [feat_to_transform[j] for j in selected_features]

        self.feature_median_ = med
        self.feature_scale_ = scale
        self.selected_features_ = np.asarray(selected_features, dtype=int)
        self.selected_transforms_ = selected_transforms
        self.intercept_ = float(intercept)
        self.coef_ = np.asarray(coef, dtype=float)
        self.n_features_in_ = p
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        X = np.asarray(X, dtype=float)
        Z = (X - self.feature_median_) / self.feature_scale_
        pieces = []
        for j, tname in zip(self.selected_features_, self.selected_transforms_):
            zj = Z[:, int(j)]
            if tname == "identity":
                pieces.append(self._identity(zj))
            elif tname == "signed_log":
                pieces.append(self._signed_log(zj))
            else:
                pieces.append(self._saturation(zj))
        if len(pieces) == 0:
            return np.full(X.shape[0], self.intercept_, dtype=float)
        Phi = np.column_stack(pieces)
        return self.intercept_ + Phi @ self.coef_

    def _equation_text(self):
        pieces = [f"{self.intercept_:+.4f}"]
        for j, tname, c in zip(self.selected_features_, self.selected_transforms_, self.coef_):
            if abs(c) <= 1e-12:
                continue
            if tname == "identity":
                term = f"z(x{int(j)})"
            elif tname == "signed_log":
                term = f"sign(z(x{int(j)}))*log1p(|z(x{int(j)})|)"
            else:
                term = f"z(x{int(j)})/(1+|z(x{int(j)})|)"
            pieces.append(f"{c:+.4f}*{term}")
        return " ".join(pieces)

    def __str__(self):
        check_is_fitted(self, "is_fitted_")
        lines = ["Adaptive Calibrated Sparse Regressor:"]
        lines.append("  Standardization: z(xj) = (xj - median_j) / IQR_j")
        if len(self.selected_features_) == 0:
            lines.append("  Selected terms: none")
            lines.append(f"  y = {self.intercept_:+.4f}")
            return "\n".join(lines)
        lines.append("  Selected calibrated terms:")
        for j, tname, c in zip(self.selected_features_, self.selected_transforms_, self.coef_):
            lines.append(f"    - {c:+.4f} * {tname}(x{int(j)})")
        lines.append(f"  y = {self._equation_text()}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
AdaptiveCalibratedSparseRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "AdaptiveCalSparse_v1"
model_description = "Sparse additive model with robust scaling and per-feature adaptive calibration (identity, signed-log, saturation) plus greedy ridge selection"
model_defs = [(model_shorthand_name, AdaptiveCalibratedSparseRegressor())]


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
