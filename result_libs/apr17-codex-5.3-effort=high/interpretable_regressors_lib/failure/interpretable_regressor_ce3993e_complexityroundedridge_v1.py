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


class ComplexityRegularizedRoundedRidgeRegressor(BaseEstimator, RegressorMixin):
    """
    Custom interpretable regressor:
      1) fit ridge on standardized features with holdout-selected alpha
      2) jointly tune mild coefficient pruning + decimal rounding
      3) choose the simplest arithmetic form with near-best validation RMSE
    """

    def __init__(
        self,
        alphas=(0.0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 2e-1, 1.0, 5.0, 20.0),
        val_fraction=0.2,
        prune_rel_grid=(0.0, 0.0005, 0.001, 0.0025, 0.005, 0.01),
        round_decimals_grid=(6, 5, 4, 3, 2, 1),
        complexity_penalty=0.15,
        min_abs_coef_rel=0.0005,
        coef_tol=1e-10,
        meaningful_rel=0.08,
        random_state=0,
    ):
        self.alphas = alphas
        self.val_fraction = val_fraction
        self.prune_rel_grid = prune_rel_grid
        self.round_decimals_grid = round_decimals_grid
        self.complexity_penalty = complexity_penalty
        self.min_abs_coef_rel = min_abs_coef_rel
        self.coef_tol = coef_tol
        self.meaningful_rel = meaningful_rel
        self.random_state = random_state

    @staticmethod
    def _ridge_with_intercept(D, y, ridge):
        n = D.shape[0]
        if D.shape[1] == 0:
            return float(np.mean(y)), np.zeros(0, dtype=float)
        A = np.column_stack([np.ones(n, dtype=float), D])
        reg = np.eye(A.shape[1], dtype=float)
        reg[0, 0] = 0.0
        lhs = A.T @ A + max(float(ridge), 0.0) * reg
        rhs = A.T @ y
        try:
            sol = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        return float(sol[0]), np.asarray(sol[1:], dtype=float)

    @staticmethod
    def _rmse(y_true, y_pred):
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    def _inner_split(self, n):
        if n <= 40:
            idx = np.arange(n, dtype=int)
            return idx, idx
        rng = np.random.RandomState(self.random_state)
        order = rng.permutation(n)
        n_val = int(round(float(self.val_fraction) * n))
        n_val = min(max(n_val, 20), n - 20)
        val_idx = order[:n_val]
        tr_idx = order[n_val:]
        if tr_idx.size == 0:
            tr_idx = val_idx
        return tr_idx, val_idx

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        self.x_mean_ = X.mean(axis=0)
        self.x_scale_ = X.std(axis=0)
        self.x_scale_[self.x_scale_ < 1e-12] = 1.0
        Xs = (X - self.x_mean_) / self.x_scale_

        idx_tr, idx_va = self._inner_split(n_samples)
        Xtr_std = Xs[idx_tr]
        Xva_raw = X[idx_va]
        ytr = y[idx_tr]
        yva = y[idx_va]

        best_cfg = None
        best_obj = np.inf
        best_rmse = np.inf

        for alpha in self.alphas:
            inter_dense, coef_dense = self._ridge_with_intercept(Xtr_std, ytr, alpha)
            max_abs = float(np.max(np.abs(coef_dense))) if coef_dense.size else 0.0

            for rel_thr in self.prune_rel_grid:
                if max_abs <= 0:
                    keep_idx = np.arange(n_features, dtype=int)
                else:
                    keep_mask = np.abs(coef_dense) >= float(rel_thr) * max_abs
                    keep_idx = np.where(keep_mask)[0]
                    if keep_idx.size == 0:
                        keep_idx = np.array([int(np.argmax(np.abs(coef_dense)))], dtype=int)

                inter_sub, coef_sub = self._ridge_with_intercept(Xtr_std[:, keep_idx], ytr, alpha)
                coef_std_full = np.zeros(n_features, dtype=float)
                coef_std_full[keep_idx] = coef_sub

                coef_raw = coef_std_full / self.x_scale_
                inter_raw = float(inter_sub - np.sum(coef_std_full * self.x_mean_ / self.x_scale_))

                for decimals in self.round_decimals_grid:
                    coef_round = np.round(coef_raw, int(decimals))
                    inter_round = float(np.round(inter_raw, int(decimals)))
                    round_eps = 0.5 * (10.0 ** (-int(decimals)))
                    coef_round[np.abs(coef_round) < round_eps] = 0.0

                    if np.all(np.abs(coef_round) <= self.coef_tol):
                        j_star = int(np.argmax(np.abs(coef_raw)))
                        coef_round[j_star] = float(np.round(coef_raw[j_star], int(decimals)))
                        if abs(coef_round[j_star]) <= self.coef_tol:
                            coef_round[j_star] = float(coef_raw[j_star])

                    pred_va = inter_round + Xva_raw @ coef_round
                    rmse_va = self._rmse(yva, pred_va)

                    n_active = int(np.sum(np.abs(coef_round) > self.coef_tol))
                    decimal_complexity = max(0, 6 - int(decimals))
                    complexity = n_active + 0.4 * decimal_complexity
                    obj = rmse_va * (
                        1.0
                        + float(self.complexity_penalty) * complexity / max(float(n_features), 1.0)
                    )

                    if (obj < best_obj - 1e-12) or (
                        abs(obj - best_obj) <= 1e-12 and rmse_va < best_rmse
                    ):
                        best_obj = obj
                        best_rmse = rmse_va
                        best_cfg = {
                            "alpha": float(alpha),
                            "rel_thr": float(rel_thr),
                            "decimals": int(decimals),
                        }

        if best_cfg is None:
            self.intercept_ = float(np.mean(y))
            self.coef_ = np.zeros(n_features, dtype=float)
            self.feature_importance_ = np.zeros(n_features, dtype=float)
            self.selected_features_ = []
            self.alpha_selected_ = 0.0
            self.prune_rel_selected_ = 0.0
            self.round_decimals_selected_ = 6
            self.operations_ = 0
            return self

        self.alpha_selected_ = float(best_cfg["alpha"])
        self.prune_rel_selected_ = float(best_cfg["rel_thr"])
        self.round_decimals_selected_ = int(best_cfg["decimals"])

        # Refit on all data with selected alpha/pruning, then apply selected rounding.
        inter_dense_all, coef_dense_all = self._ridge_with_intercept(Xs, y, self.alpha_selected_)
        max_abs_all = float(np.max(np.abs(coef_dense_all))) if coef_dense_all.size else 0.0
        if max_abs_all <= 0:
            keep_idx = np.arange(n_features, dtype=int)
        else:
            keep_mask = np.abs(coef_dense_all) >= self.prune_rel_selected_ * max_abs_all
            keep_idx = np.where(keep_mask)[0]
            if keep_idx.size == 0:
                keep_idx = np.array([int(np.argmax(np.abs(coef_dense_all)))], dtype=int)

        inter_sub_all, coef_sub_all = self._ridge_with_intercept(Xs[:, keep_idx], y, self.alpha_selected_)
        coef_std_all = np.zeros(n_features, dtype=float)
        coef_std_all[keep_idx] = coef_sub_all

        coef_raw_all = coef_std_all / self.x_scale_
        inter_raw_all = float(inter_sub_all - np.sum(coef_std_all * self.x_mean_ / self.x_scale_))

        decimals = int(self.round_decimals_selected_)
        coef_final = np.round(coef_raw_all, decimals)
        inter_final = float(np.round(inter_raw_all, decimals))

        max_abs_final = float(np.max(np.abs(coef_final))) if coef_final.size else 0.0
        if max_abs_final > 0:
            tiny = np.abs(coef_final) < float(self.min_abs_coef_rel) * max_abs_final
            coef_final[tiny] = 0.0

        round_eps = 0.5 * (10.0 ** (-decimals))
        coef_final[np.abs(coef_final) < round_eps] = 0.0
        if np.all(np.abs(coef_final) <= self.coef_tol):
            j_star = int(np.argmax(np.abs(coef_raw_all)))
            coef_final[j_star] = float(np.round(coef_raw_all[j_star], decimals))
            if abs(coef_final[j_star]) <= self.coef_tol:
                coef_final[j_star] = float(coef_raw_all[j_star])

        # Small intercept correction after rounding keeps predictions calibrated.
        resid_bias = float(np.mean(y - (inter_final + X @ coef_final)))
        inter_final = float(np.round(inter_final + resid_bias, decimals))

        self.intercept_ = inter_final
        self.coef_ = np.asarray(coef_final, dtype=float)
        self.feature_importance_ = np.abs(self.coef_)
        self.selected_features_ = sorted(int(i) for i in np.where(self.feature_importance_ > self.coef_tol)[0])
        n_terms = len(self.selected_features_)
        self.operations_ = n_terms * 2 + max(n_terms - 1, 0)
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "coef_", "feature_importance_"])
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.intercept_ + X @ self.coef_

    def __str__(self):
        check_is_fitted(self, ["intercept_", "coef_", "feature_importance_"])
        decimals = int(getattr(self, "round_decimals_selected_", 6))
        terms = [(j, float(c)) for j, c in enumerate(self.coef_) if abs(float(c)) > self.coef_tol]
        terms.sort(key=lambda t: abs(t[1]), reverse=True)

        if not terms:
            equation = f"{self.intercept_:+.{decimals}f}"
        else:
            rhs_parts = [f"{self.intercept_:+.{decimals}f}"] + [
                f"({c:+.{decimals}f})*x{j}" for j, c in terms
            ]
            equation = " + ".join(rhs_parts)

        lines = [
            "Complexity-Regularized Rounded Ridge Regressor",
            "Exact prediction equation:",
            "  y = " + equation,
            "",
            "Computation recipe:",
            "  1) Start with the intercept.",
            "  2) For each listed feature term, multiply coefficient * feature value.",
            "  3) Add all terms.",
            "",
            "Active linear terms (sorted by |coefficient|):",
        ]
        if terms:
            for i, (j, c) in enumerate(terms, 1):
                lines.append(
                    f"  {i:2d}. x{j}: coef={c:+.{decimals}f} "
                    f"(+1 change in x{j} changes prediction by {c:+.{decimals}f})"
                )
        else:
            lines.append("  (none)")

        lines.append("")
        active = [int(i) for i in self.selected_features_]
        lines.append("Active features: " + (", ".join(f"x{i}" for i in active) if active else "none"))
        if self.n_features_in_ <= 30 and len(active) < self.n_features_in_:
            active_set = set(active)
            zero_feats = [f"x{i}" for i in range(self.n_features_in_) if i not in active_set]
            lines.append("Zero-contribution features: " + ", ".join(zero_feats))

        if self.feature_importance_.size > 0:
            max_imp = float(np.max(self.feature_importance_))
            if max_imp > 0:
                thr = float(self.meaningful_rel) * max_imp
                meaningful = [f"x{i}" for i in range(self.n_features_in_) if self.feature_importance_[i] >= thr]
                lines.append(
                    "Meaningful features (>= "
                    f"{float(self.meaningful_rel):.2f} * max importance): "
                    + (", ".join(meaningful) if meaningful else "none")
                )

        lines.append(f"Selected ridge alpha: {self.alpha_selected_:.6g}")
        lines.append(f"Selected prune ratio: {self.prune_rel_selected_:.6g}")
        lines.append(f"Selected coefficient decimals: {self.round_decimals_selected_}")
        lines.append(f"Approximate arithmetic operations: {self.operations_}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
ComplexityRegularizedRoundedRidgeRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "ComplexityRoundedRidge_v1"
model_description = "Holdout-selected ridge with mild pruning and decimal-rounded coefficients, tuned for RMSE and arithmetic simplicity"
model_defs = [(model_shorthand_name, ComplexityRegularizedRoundedRidgeRegressor())]


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
