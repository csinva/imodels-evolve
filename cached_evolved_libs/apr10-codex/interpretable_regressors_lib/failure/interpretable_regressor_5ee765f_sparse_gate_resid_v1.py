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


class SparseGateResidualRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse linear equation with one explicit gate correction.

    Model form:
      y_hat = b + sum_j w_j z_j + a * I(z_g > t) + c * I(z_g > t) * z_g
    where z_j uses robust median/IQR scaling.
    """

    def __init__(
        self,
        screening_features=16,
        max_linear_terms=6,
        alpha_grid=(0.002, 0.01, 0.05, 0.2, 0.8),
        gate_thresholds=(-0.5, 0.0, 0.5),
        min_abs_coef=0.05,
        coef_quant_step=0.05,
    ):
        self.screening_features = screening_features
        self.max_linear_terms = max_linear_terms
        self.alpha_grid = alpha_grid
        self.gate_thresholds = gate_thresholds
        self.min_abs_coef = min_abs_coef
        self.coef_quant_step = coef_quant_step

    @staticmethod
    def _safe_corr(x, y):
        xc = x - np.mean(x)
        yc = y - np.mean(y)
        denom = (np.linalg.norm(xc) + 1e-12) * (np.linalg.norm(yc) + 1e-12)
        return float(np.dot(xc, yc) / denom)

    @staticmethod
    def _robust_scale(X):
        med = np.median(X, axis=0)
        q25 = np.quantile(X, 0.25, axis=0)
        q75 = np.quantile(X, 0.75, axis=0)
        iqr = np.where((q75 - q25) > 1e-9, q75 - q25, 1.0)
        return (X - med) / iqr, med, iqr

    @staticmethod
    def _ridge_fit(D, y, alpha):
        reg = float(alpha) * np.eye(D.shape[1], dtype=float)
        reg[0, 0] = 0.0
        beta = np.linalg.solve(D.T @ D + reg, D.T @ y)
        return np.asarray(beta, dtype=float)

    @staticmethod
    def _quantize(v, step):
        if step <= 0:
            return float(v)
        return float(np.round(v / step) * step)

    def _bic_like(self, y_true, y_pred, n_params):
        n = max(1, y_true.shape[0])
        mse = float(np.mean((y_true - y_pred) ** 2))
        return np.log(mse + 1e-12) + 0.5 * float(n_params) * np.log(n) / n

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

        M = Z[:, screened]
        D_lin = np.column_stack([np.ones(n, dtype=float), M])
        best_lin = None
        for alpha in self.alpha_grid:
            beta = self._ridge_fit(D_lin, y, alpha=float(alpha))
            score = self._bic_like(y, D_lin @ beta, n_params=1 + M.shape[1])
            if best_lin is None or score < best_lin[0]:
                best_lin = (score, beta)

        beta_lin = best_lin[1]
        lin_weights = beta_lin[1:]
        order = np.argsort(-np.abs(lin_weights))
        keep_local = [int(i) for i in order[: max(1, int(self.max_linear_terms))]]
        if not keep_local:
            keep_local = [int(order[0])]
        lin_features = [screened[i] for i in keep_local]

        base_cols = [Z[:, f] for f in lin_features]
        base_terms = [("linear", int(f), 0.0) for f in lin_features]
        D_base = np.column_stack([np.ones(n, dtype=float)] + base_cols)

        best = None
        for alpha in self.alpha_grid:
            beta = self._ridge_fit(D_base, y, alpha=float(alpha))
            pred = D_base @ beta
            score = self._bic_like(y, pred, n_params=D_base.shape[1])
            if best is None or score < best[0]:
                best = (score, beta, None, None)

        for gate_feat in lin_features:
            zg = Z[:, int(gate_feat)]
            for t in self.gate_thresholds:
                gate = (zg > float(t)).astype(float)
                gate_center = gate - np.mean(gate)
                gate_slope = gate_center * zg
                D = np.column_stack([D_base, gate_center, gate_slope])
                for alpha in self.alpha_grid:
                    beta = self._ridge_fit(D, y, alpha=float(alpha))
                    pred = D @ beta
                    score = self._bic_like(y, pred, n_params=D.shape[1])
                    if best is None or score < best[0]:
                        best = (score, beta, int(gate_feat), float(t))

        _, beta, gate_feat, gate_t = best
        intercept = float(beta[0])
        lin_coef = np.asarray(beta[1:1 + len(base_terms)], dtype=float)
        rem = np.asarray(beta[1 + len(base_terms):], dtype=float)

        terms = list(base_terms)
        coefs = list(lin_coef)
        if gate_feat is not None and rem.shape[0] == 2:
            terms.extend([("gate", gate_feat, gate_t), ("gate_slope", gate_feat, gate_t)])
            coefs.extend([float(rem[0]), float(rem[1])])

        qcoefs = np.array([self._quantize(c, self.coef_quant_step) for c in coefs], dtype=float)
        keep = [i for i, c in enumerate(qcoefs) if abs(float(c)) >= float(self.min_abs_coef)]
        if not keep:
            j = int(np.argmax(np.abs(coefs)))
            keep = [j]
            qcoefs = np.zeros_like(qcoefs, dtype=float)
            qcoefs[j] = self._quantize(float(coefs[j]), self.coef_quant_step)
            if abs(qcoefs[j]) < float(self.min_abs_coef):
                qcoefs[j] = float(np.sign(coefs[j]) * self.min_abs_coef)

        self.intercept_ = intercept
        self.terms_ = [terms[i] for i in keep]
        self.weights_ = np.array([qcoefs[i] for i in keep], dtype=float)
        self.n_features_in_ = p
        self.is_fitted_ = True
        return self

    def _eval_term(self, Z, term):
        kind, feat, t = term
        zf = Z[:, int(feat)]
        if kind == "linear":
            return zf
        gate = (zf > float(t)).astype(float)
        gate_center = gate - np.mean(gate)
        if kind == "gate":
            return gate_center
        return gate_center * zf

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
        lines = ["Sparse Gate Residual Regressor:"]
        lines.append("  z_j = (x_j - median_j) / IQR_j")
        lines.append(f"  intercept: {self.intercept_:+.4f}")
        lines.append("  prediction = intercept + sum_k w_k * term_k")
        order = np.argsort(-np.abs(self.weights_))
        for rank, idx in enumerate(order, 1):
            w = float(self.weights_[idx])
            kind, feat, t = self.terms_[int(idx)]
            if kind == "linear":
                expr = f"z{feat}"
            elif kind == "gate":
                expr = f"I(z{feat} > {t:+.2f}) - mean_train[I(z{feat} > {t:+.2f})]"
            else:
                expr = f"(I(z{feat} > {t:+.2f}) - mean_train[I(z{feat} > {t:+.2f})]) * z{feat}"
            lines.append(f"  {rank}. {w:+.4f} * {expr}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparseGateResidualRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SparseGateResid_v1"
model_description = "Sparse robust linear equation with one learned gate indicator and gated slope correction on a selected feature, with quantized coefficients"
model_defs = [(model_shorthand_name, SparseGateResidualRegressor())]


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
