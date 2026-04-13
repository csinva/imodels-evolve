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


class SparsePiecewiseAnchorRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse additive regressor with one-piece corrections.

    Steps:
      1) robustly scale features with median/IQR
      2) fit a sparse linear backbone on screened features
      3) add a few one-sided hinge corrections selected on residuals
      4) refit + quantize for a compact equation
    """

    def __init__(
        self,
        screening_features=14,
        max_linear_terms=5,
        max_hinge_terms=3,
        max_total_terms=7,
        alpha_grid=(0.005, 0.02, 0.08, 0.3, 1.0),
        min_abs_coef=0.05,
        coef_quant_step=0.05,
    ):
        self.screening_features = screening_features
        self.max_linear_terms = max_linear_terms
        self.max_hinge_terms = max_hinge_terms
        self.max_total_terms = max_total_terms
        self.alpha_grid = alpha_grid
        self.min_abs_coef = min_abs_coef
        self.coef_quant_step = coef_quant_step

    @staticmethod
    def _safe_corr(x, y):
        xc = x - np.mean(x)
        yc = y - np.mean(y)
        denom = (np.linalg.norm(xc) + 1e-12) * (np.linalg.norm(yc) + 1e-12)
        return float(np.dot(xc, yc) / denom)

    @staticmethod
    def _ridge_fit(M, y, alpha):
        n = M.shape[0]
        D = np.column_stack([np.ones(n, dtype=float), M])
        reg = float(alpha) * np.eye(D.shape[1], dtype=float)
        reg[0, 0] = 0.0
        beta = np.linalg.solve(D.T @ D + reg, D.T @ y)
        return float(beta[0]), np.asarray(beta[1:], dtype=float)

    @staticmethod
    def _robust_scale(X):
        med = np.median(X, axis=0)
        q25 = np.quantile(X, 0.25, axis=0)
        q75 = np.quantile(X, 0.75, axis=0)
        iqr = np.where((q75 - q25) > 1e-9, q75 - q25, 1.0)
        return (X - med) / iqr, med, iqr

    @staticmethod
    def _quantize(v, step):
        if step <= 0:
            return float(v)
        return float(np.round(v / step) * step)

    def _score_fit(self, y_true, y_pred, n_params):
        mse = float(np.mean((y_true - y_pred) ** 2))
        return mse + 0.02 * float(n_params) / max(1, len(y_true))

    @staticmethod
    def _hinge(z, t):
        # Centering keeps terms near-orthogonal to the intercept.
        h = np.maximum(0.0, z - t)
        return h - float(np.mean(h))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        _, p = X.shape

        Z, med, scale = self._robust_scale(X)
        self.feature_medians_ = med
        self.feature_scales_ = scale

        k = min(max(1, int(self.screening_features)), p)
        corrs = np.array([abs(self._safe_corr(Z[:, j], y)) for j in range(p)], dtype=float)
        screened = [int(i) for i in np.argsort(-corrs)[:k]]

        M_screen = Z[:, screened]
        best_linear = (float("inf"), float(np.mean(y)), np.zeros(M_screen.shape[1], dtype=float))
        for alpha in self.alpha_grid:
            inter, coef = self._ridge_fit(M_screen, y, alpha)
            score = self._score_fit(y, inter + M_screen @ coef, np.sum(np.abs(coef) > 1e-8) + 1)
            if score < best_linear[0]:
                best_linear = (score, inter, coef)
        _, lin_inter, lin_coef = best_linear

        lin_strength = np.abs(lin_coef)
        lin_order = np.argsort(-lin_strength)
        lin_keep = []
        for idx in lin_order:
            if len(lin_keep) >= max(1, int(self.max_linear_terms)):
                break
            if lin_strength[int(idx)] > 1e-10:
                lin_keep.append(int(idx))
        if not lin_keep:
            lin_keep = [int(lin_order[0])]

        terms = []
        cols = []
        for idx in lin_keep:
            feat = screened[int(idx)]
            terms.append(("linear", int(feat), 0.0))
            cols.append(Z[:, int(feat)])

        base_pred = lin_inter + np.column_stack(cols) @ np.array([lin_coef[i] for i in lin_keep], dtype=float)
        residual = y - base_pred

        hinge_candidates = []
        used_feats = set()
        hinge_thresholds = (-0.75, 0.0, 0.75)
        for idx in lin_keep:
            feat = screened[int(idx)]
            z = Z[:, int(feat)]
            for t in hinge_thresholds:
                h = self._hinge(z, float(t))
                c = abs(self._safe_corr(h, residual))
                hinge_candidates.append((c, int(feat), float(t), h))

        hinge_candidates.sort(key=lambda z: -z[0])
        for c, feat, t, h in hinge_candidates:
            if len(used_feats) >= max(0, int(self.max_hinge_terms)):
                break
            if feat in used_feats or c < 0.03:
                continue
            used_feats.add(feat)
            terms.append(("hinge", int(feat), float(t)))
            cols.append(h)

        M = np.column_stack(cols)
        best_all = (float("inf"), float(np.mean(y)), np.zeros(M.shape[1], dtype=float))
        for alpha in self.alpha_grid:
            inter, coef = self._ridge_fit(M, y, alpha)
            score = self._score_fit(y, inter + M @ coef, np.sum(np.abs(coef) > 1e-8) + 1)
            if score < best_all[0]:
                best_all = (score, inter, coef)
        _, _, coef = best_all

        strength = np.abs(coef)
        order = np.argsort(-strength)
        keep = [int(i) for i in order if strength[int(i)] >= float(self.min_abs_coef)]
        keep = keep[: max(1, int(self.max_total_terms))]
        if not keep:
            keep = [int(order[0])]

        final_terms = [terms[i] for i in keep]
        final_cols = np.column_stack([cols[i] for i in keep])
        inter_final, coef_final = self._ridge_fit(final_cols, y, alpha=0.02)

        qcoef = np.array([self._quantize(w, self.coef_quant_step) for w in coef_final], dtype=float)
        qcoef[np.abs(qcoef) < float(self.min_abs_coef)] = 0.0
        nz = [i for i, w in enumerate(qcoef) if abs(float(w)) > 0.0]
        if not nz:
            j = int(np.argmax(np.abs(coef_final)))
            nz = [j]
            qcoef = np.zeros_like(coef_final, dtype=float)
            qcoef[j] = self._quantize(float(coef_final[j]), self.coef_quant_step)
            if abs(qcoef[j]) < float(self.min_abs_coef):
                qcoef[j] = float(np.sign(coef_final[j]) * self.min_abs_coef)

        self.intercept_ = float(inter_final)
        self.terms_ = [final_terms[i] for i in nz]
        self.weights_ = np.array([qcoef[i] for i in nz], dtype=float)
        self.n_features_in_ = p
        self.is_fitted_ = True
        return self

    def _eval_term(self, Z, term):
        kind, feat, t = term
        z = Z[:, int(feat)]
        if kind == "linear":
            return z
        return self._hinge(z, float(t))

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        X = np.asarray(X, dtype=float)
        Z = (X - self.feature_medians_) / self.feature_scales_
        pred = np.full(X.shape[0], self.intercept_, dtype=float)
        for w, term in zip(self.weights_, self.terms_):
            pred += float(w) * self._eval_term(Z, term)
        return pred

    def __str__(self):
        check_is_fitted(self, "is_fitted_")
        lines = ["Sparse Piecewise Anchor Regressor:"]
        lines.append("  z_j = (x_j - median_j) / IQR_j")
        lines.append(f"  intercept: {self.intercept_:+.4f}")
        lines.append("  prediction = intercept + sum_k w_k * term_k")
        order = np.argsort(-np.abs(self.weights_))
        for rank, idx in enumerate(order, 1):
            w = float(self.weights_[idx])
            kind, feat, t = self.terms_[int(idx)]
            if kind == "linear":
                expr = f"z{feat}"
            else:
                expr = f"max(0, z{feat} - {t:+.2f}) - mean_train[max(0, z{feat} - {t:+.2f})]"
            lines.append(f"  {rank}. {w:+.4f} * {expr}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparsePiecewiseAnchorRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SparsePwAnchor_v1"
model_description = "Sparse robust-scaled additive model with screened linear terms plus one centered hinge correction per selected feature and quantized coefficients"
model_defs = [(model_shorthand_name, SparsePiecewiseAnchorRegressor())]


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
