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


class SparseAdditiveBasisPursuitRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse additive regressor built from raw-feature basis terms.

    The model learns a compact equation with forward-selected terms from:
    - linear terms x_j
    - one-sided hinge terms max(0, x_j - k), max(0, k - x_j)

    All terms are selected greedily using residual correlation, then jointly
    refit with a small ridge penalty. The final predictor is an explicit raw-
    feature equation that is easy to simulate by hand.
    """

    def __init__(
        self,
        max_terms=8,
        max_nonlinear_features=8,
        knot_quantiles=(0.2, 0.4, 0.6, 0.8),
        ridge_alpha=1e-3,
        min_rel_improvement=2e-4,
        coef_prune_threshold=0.025,
    ):
        self.max_terms = max_terms
        self.max_nonlinear_features = max_nonlinear_features
        self.knot_quantiles = knot_quantiles
        self.ridge_alpha = ridge_alpha
        self.min_rel_improvement = min_rel_improvement
        self.coef_prune_threshold = coef_prune_threshold

    @staticmethod
    def _safe_std(v):
        s = float(np.std(v))
        return s if s > 1e-12 else 1.0

    @staticmethod
    def _solve_ridge(A, y, alpha):
        p = A.shape[1]
        lhs = A.T @ A + float(alpha) * np.eye(p)
        rhs = A.T @ y
        return np.linalg.solve(lhs + 1e-10 * np.eye(p), rhs)

    @staticmethod
    def _term_label(term):
        kind = term["kind"]
        j = int(term["j"])
        if kind == "linear":
            return f"x{j}"
        knot = float(term["knot"])
        if kind == "hinge_pos":
            return f"max(0, x{j} - {knot:.6g})"
        return f"max(0, {knot:.6g} - x{j})"

    @staticmethod
    def _eval_term(term, X):
        xj = X[:, int(term["j"])]
        kind = term["kind"]
        if kind == "linear":
            return xj
        knot = float(term["knot"])
        if kind == "hinge_pos":
            return np.maximum(0.0, xj - knot)
        return np.maximum(0.0, knot - xj)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape
        self.n_features_in_ = p

        y_mean = float(np.mean(y))
        yc = y - y_mean

        # Screen features by standardized correlation with target.
        x_std = np.array([self._safe_std(X[:, j]) for j in range(p)], dtype=float)
        corr_scores = np.abs((X - np.mean(X, axis=0)) / x_std).T @ np.abs(yc)
        corr_order = np.argsort(-corr_scores)
        top_nonlin = corr_order[: min(int(self.max_nonlinear_features), p)]

        # Candidate term pool.
        terms = []
        basis_cols = []

        for j in range(p):
            terms.append({"kind": "linear", "j": int(j), "knot": None})
            basis_cols.append(X[:, j])

        for j in top_nonlin:
            xj = X[:, int(j)]
            for q in self.knot_quantiles:
                knot = float(np.quantile(xj, float(q)))
                b_pos = np.maximum(0.0, xj - knot)
                b_neg = np.maximum(0.0, knot - xj)
                if np.std(b_pos) > 1e-10:
                    terms.append({"kind": "hinge_pos", "j": int(j), "knot": knot})
                    basis_cols.append(b_pos)
                if np.std(b_neg) > 1e-10:
                    terms.append({"kind": "hinge_neg", "j": int(j), "knot": knot})
                    basis_cols.append(b_neg)

        B_raw = np.column_stack(basis_cols) if basis_cols else np.zeros((n, 0), dtype=float)
        m = B_raw.shape[1]
        if m == 0:
            self.intercept_ = y_mean
            self.terms_ = []
            self.coefs_ = np.zeros(0, dtype=float)
            self.active_features_ = np.array([], dtype=int)
            self.meaningful_features_ = np.array([], dtype=int)
            self.negligible_features_ = np.arange(p, dtype=int)
            return self

        b_mean = np.mean(B_raw, axis=0)
        b_std = np.std(B_raw, axis=0)
        b_std = np.where(b_std > 1e-12, b_std, 1.0)
        B = (B_raw - b_mean) / b_std

        active = []
        beta = np.zeros(0, dtype=float)
        current_mse = float(np.mean(yc * yc))
        residual = yc.copy()
        available = set(range(m))

        for _ in range(min(int(self.max_terms), m)):
            if not available:
                break
            idxs = np.array(sorted(available), dtype=int)
            # Correlation screening with current residual.
            scores = np.abs(B[:, idxs].T @ residual)
            chosen = int(idxs[int(np.argmax(scores))])

            trial_active = active + [chosen]
            A = B[:, trial_active]
            trial_beta = self._solve_ridge(A, yc, self.ridge_alpha)
            trial_resid = yc - A @ trial_beta
            trial_mse = float(np.mean(trial_resid * trial_resid))

            rel_impr = (current_mse - trial_mse) / max(current_mse, 1e-12)
            if rel_impr < float(self.min_rel_improvement):
                break

            active = trial_active
            beta = trial_beta
            residual = trial_resid
            current_mse = trial_mse
            available.remove(chosen)

        if not active:
            best = int(np.argmax(np.abs(B.T @ yc)))
            active = [best]
            beta = self._solve_ridge(B[:, active], yc, self.ridge_alpha)

        # Convert normalized-basis coefficients back to raw-basis coefficients.
        active = np.array(active, dtype=int)
        raw_coefs = beta / b_std[active]
        intercept = y_mean - float(np.dot(beta, b_mean[active] / b_std[active]))

        # Magnitude-based pruning with mandatory best term retention.
        abs_c = np.abs(raw_coefs)
        max_c = float(np.max(abs_c)) if len(abs_c) > 0 else 0.0
        keep = abs_c >= float(self.coef_prune_threshold) * max(max_c, 1e-12)
        if np.any(keep):
            active = active[keep]
            raw_coefs = raw_coefs[keep]

        # Refit on kept raw terms for numerical consistency.
        if len(active) > 0:
            A_raw = B_raw[:, active]
            beta_raw = self._solve_ridge(A_raw, yc, self.ridge_alpha)
            raw_coefs = beta_raw
            intercept = y_mean
        else:
            raw_coefs = np.zeros(0, dtype=float)
            active = np.zeros(0, dtype=int)
            intercept = y_mean

        self.intercept_ = float(intercept)
        self.terms_ = [terms[int(i)] for i in active.tolist()]
        self.coefs_ = np.asarray(raw_coefs, dtype=float)

        active_features = sorted({int(t["j"]) for t in self.terms_})
        self.active_features_ = np.array(active_features, dtype=int)

        feat_strength = np.zeros(p, dtype=float)
        for c, t in zip(self.coefs_, self.terms_):
            feat_strength[int(t["j"])] += abs(float(c))
        thr = 0.05 * max(float(np.max(feat_strength)) if p > 0 else 0.0, 1e-12)
        self.meaningful_features_ = np.where(feat_strength >= thr)[0].astype(int)
        self.negligible_features_ = np.where(feat_strength < thr)[0].astype(int)
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "terms_", "coefs_"])
        X = np.asarray(X, dtype=float)
        pred = np.full(X.shape[0], self.intercept_, dtype=float)
        for c, t in zip(self.coefs_, self.terms_):
            pred += float(c) * self._eval_term(t, X)
        return pred

    def __str__(self):
        check_is_fitted(self, ["intercept_", "terms_", "coefs_"])
        lines = [
            "Sparse Additive Basis Pursuit Regressor",
            "Prediction uses this exact raw-feature equation:",
        ]

        eq = f"y = {self.intercept_:+.6f}"
        for c, t in zip(self.coefs_, self.terms_):
            eq += f" {float(c):+.6f}*({self._term_label(t)})"
        lines.append(eq)

        lines.append("Active terms (largest absolute coefficient first):")
        order = np.argsort(-np.abs(self.coefs_)) if len(self.coefs_) else np.array([], dtype=int)
        for idx in order:
            lines.append(f"  {float(self.coefs_[idx]):+.6f} * {self._term_label(self.terms_[idx])}")

        if len(self.active_features_) > 0:
            lines.append("Meaningfully used features: " + ", ".join(f"x{int(j)}" for j in self.active_features_))
        if len(self.negligible_features_) > 0:
            lines.append(
                "Features with negligible or no effect in this fitted equation: "
                + ", ".join(f"x{int(j)}" for j in self.negligible_features_)
            )

        approx_ops = 1 + sum(2 if t["kind"].startswith("hinge") else 1 for t in self.terms_)
        lines.append(f"Approx arithmetic operations: {approx_ops}")
        return "\n".join(lines)


import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparseAdditiveBasisPursuitRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "SparseAdditiveBasisPursuitV1"
model_description = "Forward-selected sparse additive raw-feature equation with linear and quantile-knot hinge basis terms, jointly ridge-refit for compact interpretable nonlinearity"
model_defs = [(model_shorthand_name, SparseAdditiveBasisPursuitRegressor())]

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
