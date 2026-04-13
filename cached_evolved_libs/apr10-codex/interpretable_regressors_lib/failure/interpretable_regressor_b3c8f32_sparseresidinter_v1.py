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


class SparseResidualInteractionRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse robust-linear regressor with a tiny residual interaction layer.

    Stage 1:
      - Robustly standardize features using median and IQR.
      - Fit a sparse ridge-style linear equation on screened features.
    Stage 2:
      - Add at most a couple pairwise interactions chosen by correlation
        with residuals.
      - Refit and quantize coefficients to keep the final equation compact.
    """

    def __init__(
        self,
        screening_features=12,
        max_linear_terms=6,
        max_interactions=2,
        max_total_terms=8,
        alpha_grid=(0.01, 0.05, 0.2, 0.8, 2.0),
        min_abs_coef=0.04,
        coef_quant_step=0.05,
    ):
        self.screening_features = screening_features
        self.max_linear_terms = max_linear_terms
        self.max_interactions = max_interactions
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
        iqr = q75 - q25
        scale = np.where(iqr > 1e-9, iqr, 1.0)
        Z = (X - med) / scale
        return Z, med, scale

    @staticmethod
    def _quantize(v, step):
        if step <= 0:
            return float(v)
        return float(np.round(v / step) * step)

    def _score_fit(self, y_true, y_pred, n_params):
        resid = y_true - y_pred
        mse = float(np.mean(resid ** 2))
        # Lightweight information criterion to avoid overfitting.
        return mse + 0.01 * float(n_params) / max(1, len(y_true))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape

        Z, med, scale = self._robust_scale(X)
        self.feature_medians_ = med
        self.feature_scales_ = scale

        k = min(max(1, int(self.screening_features)), p)
        corrs = np.array([abs(self._safe_corr(Z[:, j], y)) for j in range(p)], dtype=float)
        screened = [int(i) for i in np.argsort(-corrs)[:k]]

        M_lin = Z[:, screened]
        best = (float("inf"), float(np.mean(y)), np.zeros(M_lin.shape[1], dtype=float))
        for alpha in self.alpha_grid:
            inter, coef = self._ridge_fit(M_lin, y, alpha)
            yhat = inter + M_lin @ coef
            score = self._score_fit(y, yhat, np.sum(np.abs(coef) > 1e-8) + 1)
            if score < best[0]:
                best = (score, inter, coef)
        _, base_intercept, base_coef = best

        lin_strength = np.abs(base_coef)
        lin_order = np.argsort(-lin_strength)
        lin_keep_k = min(max(1, int(self.max_linear_terms)), len(lin_order))
        lin_keep = [int(i) for i in lin_order[:lin_keep_k] if lin_strength[int(i)] > 1e-9]
        if len(lin_keep) == 0:
            lin_keep = [int(lin_order[0])]

        terms = [("linear", screened[i], None) for i in lin_keep]
        cols = [Z[:, screened[i]] for i in lin_keep]

        residual = y - (base_intercept + np.column_stack(cols) @ base_coef[lin_keep])
        pool = [screened[i] for i in lin_keep[: min(4, len(lin_keep))]]
        interaction_candidates = []
        for a in range(len(pool)):
            for b in range(a + 1, len(pool)):
                j1 = int(pool[a])
                j2 = int(pool[b])
                prod = Z[:, j1] * Z[:, j2]
                c = abs(self._safe_corr(prod, residual))
                interaction_candidates.append((c, j1, j2, prod))
        interaction_candidates.sort(key=lambda z: -z[0])
        for c, j1, j2, prod in interaction_candidates[: max(0, int(self.max_interactions))]:
            if c < 0.04:
                continue
            cols.append(prod)
            terms.append(("interaction", j1, j2))

        M = np.column_stack(cols)
        best2 = (float("inf"), float(np.mean(y)), np.zeros(M.shape[1], dtype=float))
        for alpha in self.alpha_grid:
            inter, coef = self._ridge_fit(M, y, alpha)
            yhat = inter + M @ coef
            score = self._score_fit(y, yhat, np.sum(np.abs(coef) > 1e-8) + 1)
            if score < best2[0]:
                best2 = (score, inter, coef)
        _, intercept, coef = best2

        strength = np.abs(coef)
        order = np.argsort(-strength)
        keep = [int(i) for i in order if strength[int(i)] >= float(self.min_abs_coef)]
        keep = keep[: max(1, int(self.max_total_terms))]
        if len(keep) == 0:
            keep = [int(order[0])]

        final_terms = [terms[i] for i in keep]
        final_cols = np.column_stack([cols[i] for i in keep])
        inter_final, coef_final = self._ridge_fit(final_cols, y, alpha=0.05)

        qcoef = np.array([self._quantize(w, self.coef_quant_step) for w in coef_final], dtype=float)
        qcoef[np.abs(qcoef) < float(self.min_abs_coef)] = 0.0
        nz = [i for i, w in enumerate(qcoef) if abs(float(w)) > 0]
        if len(nz) == 0:
            nz = [int(np.argmax(np.abs(coef_final)))]
            qcoef = np.zeros_like(qcoef)
            qcoef[nz[0]] = self._quantize(coef_final[nz[0]], self.coef_quant_step)
            if abs(qcoef[nz[0]]) < float(self.min_abs_coef):
                qcoef[nz[0]] = float(np.sign(coef_final[nz[0]]) * self.min_abs_coef)

        self.intercept_ = float(inter_final)
        self.terms_ = [final_terms[i] for i in nz]
        self.weights_ = np.asarray([qcoef[i] for i in nz], dtype=float)
        self.n_features_in_ = p
        self.is_fitted_ = True
        return self

    def _eval_term(self, Z, term):
        kind, j, aux = term
        if kind == "linear":
            return Z[:, int(j)]
        return Z[:, int(j)] * Z[:, int(aux)]

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        X = np.asarray(X, dtype=float)
        Z = (X - self.feature_medians_) / self.feature_scales_
        out = np.full(X.shape[0], self.intercept_, dtype=float)
        for w, term in zip(self.weights_, self.terms_):
            out += float(w) * self._eval_term(Z, term)
        return out

    def __str__(self):
        check_is_fitted(self, "is_fitted_")
        lines = ["Sparse Residual Interaction Regressor:"]
        lines.append("  z_j = (x_j - median_j) / IQR_j")
        lines.append("  Prediction = intercept + sum_k w_k * term_k")
        lines.append(f"  intercept: {self.intercept_:+.4f}")
        abs_order = np.argsort(-np.abs(self.weights_))
        for rank, idx in enumerate(abs_order, 1):
            w = float(self.weights_[idx])
            kind, j, aux = self.terms_[int(idx)]
            if kind == "linear":
                expr = f"z{j}"
            else:
                expr = f"z{j} * z{aux}"
            lines.append(f"  {rank}. {w:+.4f} * {expr}")
        lines.append("  Feature medians/IQRs are learned from training data.")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparseResidualInteractionRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SparseResidInter_v1"
model_description = "Robustly standardized sparse linear equation with a tiny residual-selected pairwise interaction layer and quantized coefficients"
model_defs = [(model_shorthand_name, SparseResidualInteractionRegressor())]


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
