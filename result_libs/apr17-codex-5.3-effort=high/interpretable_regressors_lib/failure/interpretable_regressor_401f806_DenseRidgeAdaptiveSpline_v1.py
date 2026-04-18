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


class DenseRidgeAdaptiveSplineRegressor(BaseEstimator, RegressorMixin):
    """
    Custom interpretable regressor:
      1) Dense ridge backbone selected by holdout validation
      2) Mild magnitude pruning (keeps compactness without forcing ultra-sparsity)
      3) Forward-selected residual one-dimensional nonlinear terms (hinge/abs/quad)
    """

    def __init__(
        self,
        alphas=(1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 2e-1, 1.0, 5.0, 20.0),
        val_fraction=0.2,
        prune_rel_grid=(0.0, 0.001, 0.0025, 0.005, 0.01, 0.02),
        max_nonlinear_terms=3,
        screen_features=12,
        nonlinear_quantiles=(0.15, 0.35, 0.5, 0.65, 0.85),
        min_nonlinear_gain=0.005,
        refit_ridge=1e-6,
        min_abs_coef_rel=0.0015,
        coef_tol=1e-10,
        meaningful_rel=0.08,
        random_state=0,
    ):
        self.alphas = alphas
        self.val_fraction = val_fraction
        self.prune_rel_grid = prune_rel_grid
        self.max_nonlinear_terms = max_nonlinear_terms
        self.screen_features = screen_features
        self.nonlinear_quantiles = nonlinear_quantiles
        self.min_nonlinear_gain = min_nonlinear_gain
        self.refit_ridge = refit_ridge
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

    @staticmethod
    def _nonlinear_column(xj, kind, knot):
        knot = float(knot)
        if kind == "hinge_pos":
            return np.maximum(0.0, xj - knot)
        if kind == "hinge_neg":
            return np.maximum(0.0, knot - xj)
        if kind == "abs":
            return np.abs(xj - knot)
        if kind == "quad":
            d = xj - knot
            return d * d
        raise ValueError(f"Unknown nonlinear kind: {kind}")

    def _inner_split(self, n):
        if n <= 24:
            idx = np.arange(n)
            return idx, idx
        rng = np.random.RandomState(self.random_state)
        order = rng.permutation(n)
        n_val = int(round(float(self.val_fraction) * n))
        n_val = min(max(n_val, 16), max(n - 16, 16))
        val_idx = order[:n_val]
        tr_idx = order[n_val:]
        if tr_idx.size == 0:
            tr_idx = val_idx
        return tr_idx, val_idx

    def _build_design(self, X, linear_features, nonlinear_terms):
        cols = []
        term_defs = []
        for j in linear_features:
            jj = int(j)
            cols.append(X[:, jj].astype(float))
            term_defs.append(("linear", jj))
        for t in nonlinear_terms:
            jj = int(t["feature"])
            kk = str(t["kind"])
            knot = float(t["knot"])
            cols.append(self._nonlinear_column(X[:, jj], kk, knot))
            term_defs.append(("nonlinear", jj, kk, knot))
        D = np.column_stack(cols) if cols else np.zeros((X.shape[0], 0), dtype=float)
        return D, term_defs

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        x_mean = X.mean(axis=0)
        x_scale = X.std(axis=0)
        x_scale[x_scale < 1e-12] = 1.0
        Xs = (X - x_mean) / x_scale

        idx_tr, idx_va = self._inner_split(n_samples)
        Xtr_std, Xva_std = Xs[idx_tr], Xs[idx_va]
        ytr, yva = y[idx_tr], y[idx_va]

        # Step 1: choose ridge alpha and pruning level using holdout RMSE.
        best = None
        best_val = np.inf
        for alpha in self.alphas:
            inter_std_dense, coef_std_dense = self._ridge_with_intercept(Xtr_std, ytr, alpha)
            max_abs = float(np.max(np.abs(coef_std_dense))) if coef_std_dense.size else 0.0
            for rel_thr in self.prune_rel_grid:
                if max_abs <= 0:
                    feat_idx = np.arange(n_features, dtype=int)
                else:
                    keep_mask = np.abs(coef_std_dense) >= float(rel_thr) * max_abs
                    feat_idx = np.where(keep_mask)[0]
                    if feat_idx.size == 0:
                        feat_idx = np.array([int(np.argmax(np.abs(coef_std_dense)))], dtype=int)
                inter_k, coef_k = self._ridge_with_intercept(Xtr_std[:, feat_idx], ytr, alpha)
                pred_va = inter_k + Xva_std[:, feat_idx] @ coef_k
                val_rmse = self._rmse(yva, pred_va)
                if val_rmse < best_val:
                    full_coef = np.zeros(n_features, dtype=float)
                    full_coef[feat_idx] = coef_k
                    best_val = val_rmse
                    best = {
                        "alpha": float(alpha),
                        "rel_thr": float(rel_thr),
                        "inter_std": float(inter_k),
                        "coef_std": full_coef,
                        "features": feat_idx.copy(),
                    }

        if best is None:
            self.intercept_ = float(np.mean(y))
            self.linear_coef_ = np.zeros(n_features, dtype=float)
            self.nonlinear_terms_ = []
            self.feature_importance_ = np.zeros(n_features, dtype=float)
            self.selected_features_ = []
            self.coef_ = self.linear_coef_.copy()
            self.alpha_selected_ = 0.0
            return self

        # Convert selected standardized linear model to original feature units.
        coef_std = np.asarray(best["coef_std"], dtype=float)
        inter_std = float(best["inter_std"])
        linear_coef = coef_std / x_scale
        linear_intercept = float(inter_std - np.sum(coef_std * x_mean / x_scale))

        # Step 2: forward-select a few nonlinear residual terms.
        pred_tr = linear_intercept + X[idx_tr] @ linear_coef
        pred_va = linear_intercept + X[idx_va] @ linear_coef
        current_val = self._rmse(yva, pred_va)
        nonlinear_terms = []

        for _ in range(int(self.max_nonlinear_terms)):
            resid_tr = ytr - pred_tr
            yc = resid_tr - float(np.mean(resid_tr))
            xc = X[idx_tr] - X[idx_tr].mean(axis=0, keepdims=True)
            xnorm = np.linalg.norm(xc, axis=0) + 1e-12
            ynorm = float(np.linalg.norm(yc)) + 1e-12
            resid_corr = np.abs((xc.T @ yc) / (xnorm * ynorm))
            resid_corr[~np.isfinite(resid_corr)] = 0.0

            lin_imp = np.abs(linear_coef)
            lin_imp = lin_imp / (float(np.max(lin_imp)) + 1e-12)
            screen_score = resid_corr + 0.25 * lin_imp
            screened = np.argsort(screen_score)[::-1][: min(int(self.screen_features), n_features)]

            best_term = None
            best_term_val = current_val
            for j in screened:
                j = int(j)
                if any(int(t["feature"]) == j and str(t["kind"]).startswith("hinge") for t in nonlinear_terms):
                    pass
                xtr_j = X[idx_tr, j]
                xva_j = X[idx_va, j]
                knots = [0.0, float(np.mean(xtr_j)), float(np.median(xtr_j))]
                knots += [float(v) for v in np.quantile(xtr_j, self.nonlinear_quantiles)]
                for knot in np.unique(np.asarray(knots, dtype=float)):
                    if not np.isfinite(knot):
                        continue
                    for kind in ("hinge_pos", "hinge_neg", "abs", "quad"):
                        col_tr = self._nonlinear_column(xtr_j, kind, knot)
                        var = float(np.var(col_tr))
                        if not np.isfinite(var) or var < 1e-12:
                            continue
                        denom = float(col_tr @ col_tr) + 1e-12
                        gamma = float((col_tr @ resid_tr) / denom)
                        col_va = self._nonlinear_column(xva_j, kind, knot)
                        trial_val = self._rmse(yva, pred_va + gamma * col_va)
                        if trial_val < best_term_val:
                            best_term_val = trial_val
                            best_term = {
                                "feature": j,
                                "kind": kind,
                                "knot": float(knot),
                            }

            rel_gain = (current_val - best_term_val) / (abs(current_val) + 1e-12)
            if best_term is None or rel_gain < float(self.min_nonlinear_gain):
                break

            nonlinear_terms.append(best_term)

            # Refit full current additive formula on train split.
            linear_features = [int(i) for i in np.where(np.abs(linear_coef) > self.coef_tol)[0]]
            if not linear_features:
                linear_features = [int(np.argmax(np.abs(linear_coef)))]
            Dtr, term_defs = self._build_design(X[idx_tr], linear_features, nonlinear_terms)
            inter_fit, coef_fit = self._ridge_with_intercept(Dtr, ytr, self.refit_ridge)
            Dva, _ = self._build_design(X[idx_va], linear_features, nonlinear_terms)
            pred_tr = inter_fit + Dtr @ coef_fit
            pred_va = inter_fit + Dva @ coef_fit
            current_val = self._rmse(yva, pred_va)

            # Update linear coefficients from the refit.
            linear_coef_new = np.zeros(n_features, dtype=float)
            cursor = 0
            for term in term_defs:
                if term[0] == "linear":
                    linear_coef_new[int(term[1])] = float(coef_fit[cursor])
                cursor += 1
            linear_coef = linear_coef_new
            linear_intercept = float(inter_fit)

        # Step 3: final full-data refit.
        linear_features = [int(i) for i in np.where(np.abs(linear_coef) > self.coef_tol)[0]]
        if not linear_features:
            linear_features = [int(np.argmax(np.abs(linear_coef)))]

        Dall, term_defs = self._build_design(X, linear_features, nonlinear_terms)
        inter_final, coef_final = self._ridge_with_intercept(Dall, y, self.refit_ridge)

        linear_coef_final = np.zeros(n_features, dtype=float)
        nonlinear_final = []
        cursor = 0
        for term in term_defs:
            if term[0] == "linear":
                linear_coef_final[int(term[1])] = float(coef_final[cursor])
            else:
                nonlinear_final.append({
                    "feature": int(term[1]),
                    "kind": str(term[2]),
                    "knot": float(term[3]),
                    "coef": float(coef_final[cursor]),
                })
            cursor += 1

        # Mild final pruning of tiny linear coefficients.
        max_abs_lin = float(np.max(np.abs(linear_coef_final))) if linear_coef_final.size else 0.0
        if max_abs_lin > 0:
            tiny = np.abs(linear_coef_final) < float(self.min_abs_coef_rel) * max_abs_lin
            linear_coef_final[tiny] = 0.0

        self.intercept_ = float(inter_final)
        self.linear_coef_ = np.asarray(linear_coef_final, dtype=float)
        self.nonlinear_terms_ = [t for t in nonlinear_final if abs(float(t["coef"])) > self.coef_tol]
        self.alpha_selected_ = float(best["alpha"])
        self.prune_rel_selected_ = float(best["rel_thr"])

        self.feature_importance_ = np.abs(self.linear_coef_)
        for t in self.nonlinear_terms_:
            self.feature_importance_[int(t["feature"])] += abs(float(t["coef"]))
        self.selected_features_ = sorted(int(i) for i in np.where(self.feature_importance_ > self.coef_tol)[0])
        self.coef_ = self.linear_coef_.copy()
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "linear_coef_", "feature_importance_"])
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        pred = self.intercept_ + X @ self.linear_coef_
        for t in self.nonlinear_terms_:
            j = int(t["feature"])
            pred = pred + float(t["coef"]) * self._nonlinear_column(
                X[:, j], str(t["kind"]), float(t["knot"])
            )
        return pred

    def __str__(self):
        check_is_fitted(self, ["intercept_", "linear_coef_", "feature_importance_"])
        linear_terms = [(j, float(c)) for j, c in enumerate(self.linear_coef_) if abs(float(c)) > self.coef_tol]
        linear_terms.sort(key=lambda t: abs(t[1]), reverse=True)

        formula_parts = [f"{self.intercept_:+.6f}"]
        for j, c in linear_terms:
            formula_parts.append(f"({c:+.6f})*x{j}")

        for t in self.nonlinear_terms_:
            j = int(t["feature"])
            c = float(t["coef"])
            k = float(t["knot"])
            kind = str(t["kind"])
            if kind == "hinge_pos":
                expr = f"max(0, x{j} - {k:.6f})"
            elif kind == "hinge_neg":
                expr = f"max(0, {k:.6f} - x{j})"
            elif kind == "abs":
                expr = f"abs(x{j} - {k:.6f})"
            else:
                expr = f"(x{j} - {k:.6f})^2"
            formula_parts.append(f"({c:+.6f})*{expr}")

        lines = [
            "Dense Ridge + Adaptive Spline Regressor",
            "Prediction formula (use exactly this arithmetic):",
            "  y = " + " + ".join(formula_parts),
            "",
            "Linear terms sorted by |coefficient|:",
        ]
        if linear_terms:
            for i, (j, c) in enumerate(linear_terms, 1):
                lines.append(f"  {i:2d}. x{j}: {c:+.6f}")
        else:
            lines.append("  (none)")

        lines.append("")
        lines.append("Nonlinear terms:")
        if not self.nonlinear_terms_:
            lines.append("  (none)")
        else:
            for i, t in enumerate(self.nonlinear_terms_, 1):
                j = int(t["feature"])
                c = float(t["coef"])
                k = float(t["knot"])
                kind = str(t["kind"])
                if kind == "hinge_pos":
                    lines.append(f"  {i:2d}. ({c:+.6f}) * max(0, x{j} - {k:.6f})")
                elif kind == "hinge_neg":
                    lines.append(f"  {i:2d}. ({c:+.6f}) * max(0, {k:.6f} - x{j})")
                elif kind == "abs":
                    lines.append(f"  {i:2d}. ({c:+.6f}) * abs(x{j} - {k:.6f})")
                else:
                    lines.append(f"  {i:2d}. ({c:+.6f}) * (x{j} - {k:.6f})^2")

        active = [int(i) for i in self.selected_features_]
        lines.append("")
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
        lines.append(f"Selected linear prune ratio: {self.prune_rel_selected_:.6g}")
        op_count = len(linear_terms) * 2 + max(len(linear_terms) - 1, 0)
        op_count += len(self.nonlinear_terms_) * 3
        lines.append(f"Approximate arithmetic operations: {op_count}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
DenseRidgeAdaptiveSplineRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "DenseRidgeAdaptiveSpline_v1"
model_description = "Dense ridge backbone with holdout-selected pruning and up to three residual one-dimensional spline-like correction terms"
model_defs = [(model_shorthand_name, DenseRidgeAdaptiveSplineRegressor())]


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
