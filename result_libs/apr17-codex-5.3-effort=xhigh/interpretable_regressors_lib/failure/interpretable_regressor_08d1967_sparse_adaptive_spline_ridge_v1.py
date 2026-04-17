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
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class SparseAdaptiveSplineRidgeRegressor(BaseEstimator, RegressorMixin):
    """Greedy sparse additive spline regressor with CV-ridge refit."""

    def __init__(
        self,
        alpha_grid=(1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
        max_terms=16,
        min_improvement=1e-4,
        n_splits=5,
        random_state=0,
        tiny_coef_threshold=1e-4,
    ):
        self.alpha_grid = alpha_grid
        self.max_terms = max_terms
        self.min_improvement = min_improvement
        self.n_splits = n_splits
        self.random_state = random_state
        self.tiny_coef_threshold = tiny_coef_threshold

    @staticmethod
    def _safe_standardize(X):
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
        sigma = np.where(sigma > 1e-12, sigma, 1.0)
        return (X - mu) / sigma, mu, sigma

    @staticmethod
    def _ridge_solve(X, y, alpha):
        if X.shape[1] == 0:
            return np.zeros(0, dtype=float)
        gram = X.T @ X
        rhs = X.T @ y
        eye = np.eye(gram.shape[0], dtype=float)
        try:
            return np.linalg.solve(gram + float(alpha) * eye, rhs)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(gram + float(alpha) * eye) @ rhs

    def _fit_linear_ridge(self, X, y, alpha):
        if X.shape[1] == 0:
            intercept = float(np.mean(y))
            return intercept, np.zeros(0, dtype=float)
        x_mu = np.mean(X, axis=0)
        Xc = X - x_mu
        y_mu = float(np.mean(y))
        yc = y - y_mu
        coef = self._ridge_solve(Xc, yc, alpha)
        intercept = float(y_mu - np.dot(x_mu, coef))
        return intercept, coef

    def _cv_alpha(self, X, y):
        n = X.shape[0]
        if n < 3:
            return 1.0

        n_splits = min(max(2, int(self.n_splits)), n - 1)
        if n_splits < 2:
            return 1.0

        alpha_choices = [float(a) for a in self.alpha_grid] or [1.0]
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=int(self.random_state))

        best_alpha = alpha_choices[0]
        best_mse = np.inf
        for alpha in alpha_choices:
            mses = []
            for tr, va in kf.split(X):
                intercept, coef = self._fit_linear_ridge(X[tr], y[tr], alpha)
                pred = intercept + X[va] @ coef
                mses.append(float(np.mean((y[va] - pred) ** 2)))
            mse = float(np.mean(mses))
            if mse < best_mse:
                best_mse = mse
                best_alpha = alpha
        return float(best_alpha)

    @staticmethod
    def _build_terms(X):
        n, p = X.shape
        terms = []
        cols = []
        eps = 1e-9
        for j in range(p):
            xj = X[:, j]
            terms.append({"kind": "linear", "feature": j, "knot": 0.0})
            cols.append(xj)

            for q in (0.25, 0.5, 0.75):
                knot = float(np.quantile(xj, q))
                hj = np.maximum(0.0, xj - knot)
                if np.std(hj) > eps:
                    terms.append({"kind": "hinge_pos", "feature": j, "knot": knot})
                    cols.append(hj)
                hneg = np.maximum(0.0, knot - xj)
                if np.std(hneg) > eps:
                    terms.append({"kind": "hinge_neg", "feature": j, "knot": knot})
                    cols.append(hneg)

        if not cols:
            return terms, np.zeros((n, 0), dtype=float)
        Z = np.column_stack(cols).astype(float)
        return terms, Z

    @staticmethod
    def _term_values(X, term):
        j = int(term["feature"])
        xj = X[:, j]
        kind = term["kind"]
        if kind == "linear":
            return xj
        knot = float(term["knot"])
        if kind == "hinge_pos":
            return np.maximum(0.0, xj - knot)
        if kind == "hinge_neg":
            return np.maximum(0.0, knot - xj)
        raise ValueError(f"Unknown term kind: {kind}")

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y have incompatible shapes")
        if X.shape[1] == 0:
            raise ValueError("No features provided")

        self.n_features_in_ = X.shape[1]
        self.candidate_terms_, Z = self._build_terms(X)
        if Z.shape[1] == 0:
            self.intercept_ = float(np.mean(y))
            self.coef_ = np.zeros(0, dtype=float)
            self.selected_terms_ = []
            self.alpha_ = 1.0
            self.feature_importances_ = np.zeros(self.n_features_in_, dtype=float)
            self.active_features_ = np.zeros(0, dtype=int)
            return self

        Zs, self.z_mu_, self.z_sigma_ = self._safe_standardize(Z)
        y_center = y - float(np.mean(y))

        selected = []
        remaining = list(range(Zs.shape[1]))
        current_pred = np.full_like(y, fill_value=float(np.mean(y)), dtype=float)
        prev_mse = float(np.mean((y - current_pred) ** 2))
        max_terms = max(1, min(int(self.max_terms), Zs.shape[1]))

        for _ in range(max_terms):
            residual = y_center - (current_pred - float(np.mean(y)))
            corr = np.abs(Zs[:, remaining].T @ residual)
            best_local = int(np.argmax(corr))
            best_idx = int(remaining[best_local])
            selected.append(best_idx)
            remaining.remove(best_idx)

            X_sel = Z[:, selected]
            alpha = self._cv_alpha(X_sel, y)
            intercept, coef = self._fit_linear_ridge(X_sel, y, alpha)
            pred = intercept + X_sel @ coef
            mse = float(np.mean((y - pred) ** 2))
            gain = prev_mse - mse

            current_pred = pred
            prev_mse = mse
            if gain < float(self.min_improvement):
                break

        X_sel = Z[:, selected] if selected else np.zeros((X.shape[0], 0), dtype=float)
        self.alpha_ = self._cv_alpha(X_sel, y)
        self.intercept_, self.coef_ = self._fit_linear_ridge(X_sel, y, self.alpha_)

        keep = np.abs(self.coef_) > float(self.tiny_coef_threshold)
        if np.any(keep):
            kept_indices = [selected[i] for i in np.flatnonzero(keep)]
            X_kept = Z[:, kept_indices]
            self.alpha_ = self._cv_alpha(X_kept, y)
            self.intercept_, self.coef_ = self._fit_linear_ridge(X_kept, y, self.alpha_)
            selected = kept_indices
        else:
            selected = []
            self.coef_ = np.zeros(0, dtype=float)
            self.intercept_ = float(np.mean(y))

        self.selected_terms_ = [self.candidate_terms_[k] for k in selected]

        feat_imp = np.zeros(self.n_features_in_, dtype=float)
        for term, coef in zip(self.selected_terms_, self.coef_):
            feat_imp[int(term["feature"])] += abs(float(coef))
        max_imp = float(np.max(feat_imp)) if feat_imp.size else 0.0
        self.feature_importances_ = feat_imp / max_imp if max_imp > 0 else feat_imp
        self.active_features_ = np.flatnonzero(feat_imp > 0).astype(int)
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "coef_", "selected_terms_"])
        X = np.asarray(X, dtype=float)
        pred = np.full(X.shape[0], float(self.intercept_), dtype=float)
        for coef, term in zip(self.coef_, self.selected_terms_):
            pred += float(coef) * self._term_values(X, term)
        return pred

    def __str__(self):
        check_is_fitted(self, ["intercept_", "coef_", "feature_importances_", "active_features_", "selected_terms_"])

        lines = ["Sparse Adaptive Spline Ridge Regressor"]
        lines.append("Prediction equation in raw input features:")
        lines.append(f"y = {float(self.intercept_):+.6f}")
        for c, term in zip(self.coef_, self.selected_terms_):
            j = int(term["feature"])
            kind = term["kind"]
            if kind == "linear":
                expr = f"x{j}"
            elif kind == "hinge_pos":
                expr = f"max(0, x{j} - {float(term['knot']):.6f})"
            else:
                expr = f"max(0, {float(term['knot']):.6f} - x{j})"
            lines.append(f"  {float(c):+.6f} * {expr}")

        lines.append("")
        lines.append(
            "Active features used by equation: "
            + (", ".join(f"x{int(j)}" for j in self.active_features_) if self.active_features_.size else "none")
        )
        lines.append(f"Number of active terms: {len(self.selected_terms_)}")
        lines.append(f"Chosen ridge alpha from CV: {float(self.alpha_):.6g}")
        lines.append("To simulate any sample: compute each term value exactly, multiply by coefficient, then add intercept.")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys

_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])


# Update the model shorthand name and description below to reflect the class above and any changes you make.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
SparseAdaptiveSplineRidgeRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "SparseAdaptiveSplineRidge_v1"
model_description = "From-scratch greedy sparse additive model with linear and quantile hinge basis terms, CV ridge refits, and a compact raw-feature equation"
model_defs = [(model_shorthand_name, SparseAdaptiveSplineRidgeRegressor())]
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

    # --- Recompute global rank summary from updated performance_results.csv ---
    # Build dataset -> {model: rmse}
    perf_table = defaultdict(dict)
    with open(perf_csv, newline="") as f:
        for row in csv.DictReader(f):
            ds = row["dataset"]
            m = row["model"]
            rmse_s = row.get("rmse", "")
            if rmse_s in ("", None):
                perf_table[ds][m] = float("nan")
            else:
                try:
                    perf_table[ds][m] = float(rmse_s)
                except ValueError:
                    perf_table[ds][m] = float("nan")

    avg_rank, _ = compute_rank_scores(perf_table)
    mean_rank = avg_rank.get(model_name, float("nan"))

    # --- Upsert overall_results.csv ---
    overall_rows = [{
        "commit": git_hash,
        "mean_rank": f"{mean_rank:.2f}" if np.isfinite(mean_rank) else "",
        "frac_interpretability_tests_passed": f"{(n_passed / total):.4f}" if total else "",
        "status": "",  # fill manually after reviewing
        "model_name": model_name,
        "description": model_description,
    }]
    upsert_overall_results(overall_rows, RESULTS_DIR)

    # --- Plot update ---
    overall_csv = os.path.join(RESULTS_DIR, "overall_results.csv")
    plot_interp_vs_performance(
        overall_csv,
        os.path.join(RESULTS_DIR, "interpretability_vs_performance.png"),
    )

    # Print compact summary
    std_names = {t.__name__ for t in ALL_TESTS}
    hard_names = {t.__name__ for t in HARD_TESTS}
    ins_names = {t.__name__ for t in INSIGHT_TESTS}
    n_std = sum(r["passed"] for r in interp_results if r["test"] in std_names)
    n_hard = sum(r["passed"] for r in interp_results if r["test"] in hard_names)
    n_ins = sum(r["passed"] for r in interp_results if r["test"] in ins_names)

    print("\n---")
    print(f"tests_passed:  {n_passed}/{total} ({(n_passed/total):.2%})  "
          f"[std {n_std}/{len(std_names)}  hard {n_hard}/{len(hard_names)}  insight {n_ins}/{len(ins_names)}]")
    print(f"total_seconds: {time.time() - t0:.1f}s")
