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


class AdaptiveSplineRidgeRegressor(BaseEstimator, RegressorMixin):
    """
    Custom interpretable regressor:
    - starts with a linear ridge backbone,
    - adds a small set of residual-selected basis terms (hinges and interactions),
    - refits all kept terms jointly with GCV-selected ridge.
    """

    def __init__(
        self,
        alpha_grid=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
        max_extra_terms=8,
        min_rel_gain=0.015,
        coef_prune_tol=1e-3,
        coef_display_tol=0.02,
    ):
        self.alpha_grid = alpha_grid
        self.max_extra_terms = max_extra_terms
        self.min_rel_gain = min_rel_gain
        self.coef_prune_tol = coef_prune_tol
        self.coef_display_tol = coef_display_tol

    @staticmethod
    def _sanitize_alpha_grid(alpha_grid):
        a = np.asarray(alpha_grid, dtype=float)
        a = a[np.isfinite(a) & (a > 0)]
        if a.size == 0:
            return np.asarray([1.0], dtype=float)
        return np.unique(a)

    @staticmethod
    def _fit_ridge_with_gcv(Z, y, alpha_grid):
        y_mean = float(np.mean(y))
        Z_mean = np.mean(Z, axis=0)
        Zc = Z - Z_mean
        yc = y - y_mean
        if Zc.shape[1] == 0:
            return y_mean, np.zeros(0, dtype=float), 1.0

        U, s, Vt = np.linalg.svd(Zc, full_matrices=False)
        Uy = U.T @ yc
        n = float(max(1, Z.shape[0]))
        best = None
        for alpha in alpha_grid:
            filt = s / (s * s + float(alpha))
            coef = Vt.T @ (filt * Uy)
            pred = y_mean + Zc @ coef
            mse = float(np.mean((y - pred) ** 2))
            dof = float(np.sum((s * s) / (s * s + float(alpha))))
            denom = max(1e-8, 1.0 - dof / n)
            gcv = mse / (denom * denom)
            if (best is None) or (gcv < best["gcv"]):
                best = {"gcv": gcv, "alpha": float(alpha), "coef": coef}
        coef = best["coef"]
        intercept = float(y_mean - np.dot(Z_mean, coef))
        return intercept, coef, float(best["alpha"])

    @staticmethod
    def _candidate_terms(X):
        n, p = X.shape
        terms = []
        for j in range(p):
            xj = X[:, j]
            med = float(np.median(xj))
            q25 = float(np.quantile(xj, 0.25))
            q75 = float(np.quantile(xj, 0.75))
            terms.append(
                (
                    np.maximum(0.0, xj - med),
                    {"kind": "hinge_pos", "j": int(j), "knot": med},
                )
            )
            terms.append(
                (
                    np.maximum(0.0, med - xj),
                    {"kind": "hinge_neg", "j": int(j), "knot": med},
                )
            )
            terms.append(
                (
                    np.maximum(0.0, xj - q75),
                    {"kind": "tail_pos", "j": int(j), "knot": q75},
                )
            )
            terms.append(
                (
                    np.maximum(0.0, q25 - xj),
                    {"kind": "tail_neg", "j": int(j), "knot": q25},
                )
            )
        top = min(6, p)
        if top >= 2:
            v = np.var(X, axis=0)
            order = np.argsort(-v)[:top]
            for a in range(top):
                j = int(order[a])
                for b in range(a + 1, top):
                    k = int(order[b])
                    terms.append(
                        (
                            X[:, j] * X[:, k],
                            {"kind": "interaction", "j": j, "k": k},
                        )
                    )
        return terms

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape
        self.n_features_in_ = int(p)
        alpha_grid = self._sanitize_alpha_grid(self.alpha_grid)

        if p == 0:
            self.intercept_ = float(np.mean(y))
            self.linear_coef_ = np.zeros(0, dtype=float)
            self.extra_terms_ = []
            self.extra_coef_ = np.zeros(0, dtype=float)
            self.alpha_ = 1.0
            self.training_mse_ = float(np.mean((y - self.intercept_) ** 2))
            self.is_fitted_ = True
            return self

        intercept_lin, coef_lin, alpha_lin = self._fit_ridge_with_gcv(X, y, alpha_grid)
        pred_lin = intercept_lin + X @ coef_lin
        residual = y - pred_lin
        base_mse = float(np.mean(residual * residual))

        selected_cols = []
        selected_info = []
        used_keys = set()
        max_extra = int(max(0, self.max_extra_terms))
        min_gain = float(max(0.0, self.min_rel_gain))
        cur_pred = pred_lin.copy()
        cur_mse = base_mse

        for _ in range(max_extra):
            best = None
            for col, info in self._candidate_terms(X):
                key = tuple(sorted(info.items()))
                if key in used_keys:
                    continue
                v = float(np.var(col))
                if v < 1e-10:
                    continue
                col_c = col - np.mean(col)
                score = abs(float(np.dot(col_c, y - cur_pred))) / (float(np.linalg.norm(col_c)) + 1e-12)
                if (best is None) or (score > best["score"]):
                    best = {"score": score, "col": col, "info": info, "key": key}
            if best is None:
                break
            trial_cols = selected_cols + [best["col"]]
            Z_trial = np.column_stack([X] + trial_cols)
            intercept_t, coef_t, _ = self._fit_ridge_with_gcv(Z_trial, y, alpha_grid)
            pred_t = intercept_t + Z_trial @ coef_t
            mse_t = float(np.mean((y - pred_t) ** 2))
            rel_gain = (cur_mse - mse_t) / max(1e-12, cur_mse)
            if rel_gain < min_gain:
                break
            selected_cols = trial_cols
            selected_info.append(best["info"])
            used_keys.add(best["key"])
            cur_pred = pred_t
            cur_mse = mse_t

        Z = X if not selected_cols else np.column_stack([X] + selected_cols)
        intercept, coef_full, alpha = self._fit_ridge_with_gcv(Z, y, alpha_grid)

        coef_full = coef_full.copy()
        prune_tol = float(max(0.0, self.coef_prune_tol))
        small = np.abs(coef_full) < prune_tol
        coef_full[small] = 0.0

        pred = intercept + Z @ coef_full
        self.intercept_ = float(intercept)
        self.alpha_ = float(alpha)
        self.linear_coef_ = coef_full[:p]
        self.extra_coef_ = coef_full[p:]
        self.extra_terms_ = selected_info
        self.training_mse_ = float(np.mean((y - pred) ** 2))
        self.n_active_linear_ = int(np.sum(np.abs(self.linear_coef_) > 0))
        self.n_active_extra_ = int(np.sum(np.abs(self.extra_coef_) > 0))
        self.is_fitted_ = True
        return self

    def _eval_extra_terms(self, X):
        if not self.extra_terms_:
            return np.zeros((X.shape[0], 0), dtype=float)
        cols = []
        for info in self.extra_terms_:
            kind = info["kind"]
            j = int(info["j"])
            if kind == "hinge_pos":
                cols.append(np.maximum(0.0, X[:, j] - float(info["knot"])))
            elif kind == "hinge_neg":
                cols.append(np.maximum(0.0, float(info["knot"]) - X[:, j]))
            elif kind == "tail_pos":
                cols.append(np.maximum(0.0, X[:, j] - float(info["knot"])))
            elif kind == "tail_neg":
                cols.append(np.maximum(0.0, float(info["knot"]) - X[:, j]))
            elif kind == "interaction":
                k = int(info["k"])
                cols.append(X[:, j] * X[:, k])
            else:
                cols.append(np.zeros(X.shape[0], dtype=float))
        return np.column_stack(cols)

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        X = np.asarray(X, dtype=float)
        y = self.intercept_ + X @ self.linear_coef_
        if self.extra_terms_:
            y = y + self._eval_extra_terms(X) @ self.extra_coef_
        return y

    @staticmethod
    def _term_to_text(info):
        kind = info["kind"]
        j = int(info["j"])
        if kind == "hinge_pos":
            return f"max(0, x{j} - {float(info['knot']):.3f})"
        if kind == "hinge_neg":
            return f"max(0, {float(info['knot']):.3f} - x{j})"
        if kind == "tail_pos":
            return f"max(0, x{j} - {float(info['knot']):.3f})"
        if kind == "tail_neg":
            return f"max(0, {float(info['knot']):.3f} - x{j})"
        if kind == "interaction":
            return f"(x{j} * x{int(info['k'])})"
        return f"term_x{j}"

    def __str__(self):
        check_is_fitted(self, "is_fitted_")
        lines = ["Adaptive Spline-Ridge Regressor:"]
        lines.append("  prediction = intercept + linear terms + selected nonlinear terms")
        lines.append(f"  ridge alpha (GCV): {self.alpha_:.6f}")
        lines.append(f"  intercept: {self.intercept_:+.4f}")
        lines.append(f"  training MSE: {self.training_mse_:.6f}")
        lines.append(
            f"  active linear: {self.n_active_linear_}/{self.n_features_in_}, active nonlinear: {self.n_active_extra_}/{len(self.extra_terms_)}"
        )

        tol = float(max(0.0, self.coef_display_tol))
        linear_terms = []
        for j, c in enumerate(self.linear_coef_):
            if abs(float(c)) >= tol:
                linear_terms.append((abs(float(c)), int(j), float(c)))
        linear_terms.sort(reverse=True)

        extra_terms = []
        for i, c in enumerate(self.extra_coef_):
            if abs(float(c)) >= tol:
                extra_terms.append(
                    (abs(float(c)), i, float(c), self._term_to_text(self.extra_terms_[i]))
                )
        extra_terms.sort(reverse=True)

        lines.append(f"  shown terms (|coef| >= {tol:.3f}):")
        if not linear_terms and not extra_terms:
            lines.append("    none above display threshold")
            return "\n".join(lines)

        if linear_terms:
            lines.append("  linear coefficients:")
            for _, j, c in linear_terms[:20]:
                lines.append(f"    {c:+.4f} * x{j}")
        if extra_terms:
            lines.append("  nonlinear coefficients:")
            for _, _, c, text in extra_terms[:16]:
                lines.append(f"    {c:+.4f} * {text}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
AdaptiveSplineRidgeRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "AdaptiveSplineRidge_v1"
model_description = "Linear ridge backbone plus greedily added residual-selected hinge/interaction basis terms, jointly refit by GCV ridge"
model_defs = [(model_shorthand_name, AdaptiveSplineRidgeRegressor())]


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
