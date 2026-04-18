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


class ValidatedSparseSplineRegressor(BaseEstimator, RegressorMixin):
    """
    Validation-selected sparse ridge backbone with one optional nonlinear term.

    Pipeline:
      1) standardize features
      2) pick (ridge alpha, sparsity level) by inner holdout validation
      3) optionally add one residual basis term (hinge+/hinge-/abs) if it helps
      4) refit a compact exact formula on the full data
    """

    def __init__(
        self,
        alphas=(1e-4, 1e-3, 1e-2, 5e-2, 0.2, 1.0, 5.0),
        max_linear_terms=10,
        min_linear_rel=0.03,
        linear_k_grid=(None, 12, 8, 5, 3),
        screen_features=8,
        candidate_quantiles=(0.2, 0.4, 0.6, 0.8),
        min_val_gain=0.003,
        val_fraction=0.2,
        refit_ridge=1e-6,
        coef_tol=1e-10,
        meaningful_rel=0.12,
        random_state=0,
    ):
        self.alphas = alphas
        self.max_linear_terms = max_linear_terms
        self.min_linear_rel = min_linear_rel
        self.linear_k_grid = linear_k_grid
        self.screen_features = screen_features
        self.candidate_quantiles = candidate_quantiles
        self.min_val_gain = min_val_gain
        self.val_fraction = val_fraction
        self.refit_ridge = refit_ridge
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
        if kind == "hinge_pos":
            return np.maximum(0.0, xj - float(knot))
        if kind == "hinge_neg":
            return np.maximum(0.0, float(knot) - xj)
        if kind == "abs":
            return np.abs(xj - float(knot))
        raise ValueError(f"Unknown nonlinear kind: {kind}")

    def _inner_split(self, n):
        if n <= 8:
            return np.arange(n), np.arange(n)
        rng = np.random.RandomState(self.random_state)
        order = rng.permutation(n)
        n_val = int(round(float(self.val_fraction) * n))
        n_val = min(max(n_val, 8), max(n - 8, 8))
        val_idx = order[:n_val]
        train_idx = order[n_val:]
        if train_idx.size == 0:
            train_idx = val_idx
        return train_idx, val_idx

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
        Xtr, Xva = Xs[idx_tr], Xs[idx_va]
        ytr, yva = y[idx_tr], y[idx_va]

        base_order = np.arange(n_features)
        best = None
        best_val = np.inf

        for alpha in self.alphas:
            inter_dense, coef_dense = self._ridge_with_intercept(Xtr, ytr, alpha)
            pred_dense_va = inter_dense + Xva @ coef_dense
            dense_val = self._rmse(yva, pred_dense_va)
            if dense_val < best_val:
                best_val = dense_val
                best = {
                    "alpha": float(alpha),
                    "features": base_order.copy(),
                    "inter_std": float(inter_dense),
                    "coef_std": np.asarray(coef_dense, dtype=float),
                }

            order = np.argsort(np.abs(coef_dense))[::-1]
            max_abs = float(np.max(np.abs(coef_dense))) if coef_dense.size else 0.0
            for k in self.linear_k_grid:
                if k is None:
                    continue
                k_use = min(int(k), int(self.max_linear_terms), n_features)
                if k_use <= 0:
                    continue
                feat_idx = order[:k_use]
                if max_abs > 0:
                    keep = np.abs(coef_dense[feat_idx]) >= float(self.min_linear_rel) * max_abs
                    feat_idx = feat_idx[keep]
                if feat_idx.size == 0:
                    feat_idx = order[:1]
                feat_idx = np.asarray(sorted(set(int(j) for j in feat_idx)), dtype=int)

                inter_k, coef_k = self._ridge_with_intercept(Xtr[:, feat_idx], ytr, alpha)
                pred_k_va = inter_k + Xva[:, feat_idx] @ coef_k
                k_val = self._rmse(yva, pred_k_va)
                if k_val < best_val:
                    full_coef = np.zeros(n_features, dtype=float)
                    full_coef[feat_idx] = coef_k
                    best_val = k_val
                    best = {
                        "alpha": float(alpha),
                        "features": feat_idx.copy(),
                        "inter_std": float(inter_k),
                        "coef_std": full_coef,
                    }

        if best is None:
            self.intercept_ = float(np.mean(y))
            self.linear_coef_ = np.zeros(n_features, dtype=float)
            self.nonlinear_term_ = None
            self.feature_importance_ = np.zeros(n_features, dtype=float)
            self.selected_features_ = []
            return self

        coef_std = np.asarray(best["coef_std"], dtype=float)
        inter_std = float(best["inter_std"])
        linear_coef_raw = coef_std / self.x_scale_
        linear_intercept_raw = float(inter_std - np.sum(coef_std * self.x_mean_ / self.x_scale_))

        pred_tr = linear_intercept_raw + X[idx_tr] @ linear_coef_raw
        pred_va = linear_intercept_raw + X[idx_va] @ linear_coef_raw
        resid_tr = y[idx_tr] - pred_tr
        base_val = self._rmse(y[idx_va], pred_va)

        xc = X[idx_tr] - X[idx_tr].mean(axis=0, keepdims=True)
        yc = resid_tr - float(np.mean(resid_tr))
        xnorm = np.linalg.norm(xc, axis=0) + 1e-12
        ynorm = float(np.linalg.norm(yc)) + 1e-12
        corr = np.abs((xc.T @ yc) / (xnorm * ynorm))
        corr[~np.isfinite(corr)] = 0.0
        screened = np.argsort(corr)[::-1][: min(int(self.screen_features), n_features)]

        best_nonlinear = None
        best_nonlinear_val = base_val
        for j in screened:
            j = int(j)
            xtr_j = X[idx_tr, j]
            xva_j = X[idx_va, j]
            knots = np.unique(np.quantile(xtr_j, self.candidate_quantiles))
            for knot in knots:
                if not np.isfinite(knot):
                    continue
                for kind in ("hinge_pos", "hinge_neg", "abs"):
                    col_tr = self._nonlinear_column(xtr_j, kind, knot)
                    denom = float(col_tr @ col_tr) + 1e-12
                    gamma = float((col_tr @ resid_tr) / denom)
                    col_va = self._nonlinear_column(xva_j, kind, knot)
                    trial_val = self._rmse(y[idx_va], pred_va + gamma * col_va)
                    if trial_val < best_nonlinear_val:
                        best_nonlinear_val = trial_val
                        best_nonlinear = {
                            "feature": j,
                            "kind": kind,
                            "knot": float(knot),
                            "coef": float(gamma),
                        }

        if best_nonlinear is not None:
            rel_gain = (base_val - best_nonlinear_val) / (abs(base_val) + 1e-12)
            if rel_gain < float(self.min_val_gain):
                best_nonlinear = None

        cols = []
        term_defs = []
        linear_features = [int(j) for j in np.where(np.abs(linear_coef_raw) > self.coef_tol)[0]]
        if not linear_features:
            linear_features = [int(np.argmax(np.abs(linear_coef_raw)))] if n_features > 0 else []
        for j in linear_features:
            cols.append(X[:, j].astype(float))
            term_defs.append(("linear", int(j)))

        if best_nonlinear is not None:
            j = int(best_nonlinear["feature"])
            cols.append(self._nonlinear_column(X[:, j], best_nonlinear["kind"], best_nonlinear["knot"]))
            term_defs.append(("nonlinear", int(j), best_nonlinear["kind"], float(best_nonlinear["knot"])))

        D = np.column_stack(cols) if cols else np.zeros((n_samples, 0), dtype=float)
        inter_final, coef_final = self._ridge_with_intercept(D, y, self.refit_ridge)

        self.intercept_ = float(inter_final)
        self.linear_coef_ = np.zeros(n_features, dtype=float)
        cursor = 0
        for term in term_defs:
            if term[0] == "linear":
                self.linear_coef_[int(term[1])] = float(coef_final[cursor])
                cursor += 1

        if best_nonlinear is not None:
            nl_coef = float(coef_final[cursor])
            self.nonlinear_term_ = {
                "feature": int(best_nonlinear["feature"]),
                "kind": str(best_nonlinear["kind"]),
                "knot": float(best_nonlinear["knot"]),
                "coef": nl_coef,
            }
        else:
            self.nonlinear_term_ = None

        self.alpha_selected_ = float(best["alpha"])
        self.feature_importance_ = np.abs(self.linear_coef_)
        if self.nonlinear_term_ is not None:
            self.feature_importance_[int(self.nonlinear_term_["feature"])] += abs(float(self.nonlinear_term_["coef"]))
        self.selected_features_ = sorted(int(i) for i in np.where(self.feature_importance_ > self.coef_tol)[0])
        self.coef_ = self.linear_coef_.copy()
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "linear_coef_", "feature_importance_"])
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        pred = self.intercept_ + X @ self.linear_coef_
        if self.nonlinear_term_ is not None:
            j = int(self.nonlinear_term_["feature"])
            pred = pred + float(self.nonlinear_term_["coef"]) * self._nonlinear_column(
                X[:, j],
                self.nonlinear_term_["kind"],
                self.nonlinear_term_["knot"],
            )
        return pred

    def __str__(self):
        check_is_fitted(self, ["intercept_", "linear_coef_", "feature_importance_"])
        linear_terms = [(j, float(c)) for j, c in enumerate(self.linear_coef_) if abs(float(c)) > self.coef_tol]
        linear_terms.sort(key=lambda t: abs(t[1]), reverse=True)

        formula_parts = [f"{self.intercept_:+.6f}"]
        for j, c in linear_terms:
            formula_parts.append(f"({c:+.6f})*x{j}")

        if self.nonlinear_term_ is not None and abs(float(self.nonlinear_term_["coef"])) > self.coef_tol:
            j = int(self.nonlinear_term_["feature"])
            c = float(self.nonlinear_term_["coef"])
            k = float(self.nonlinear_term_["knot"])
            kind = self.nonlinear_term_["kind"]
            if kind == "hinge_pos":
                expr = f"max(0, x{j} - {k:.6f})"
            elif kind == "hinge_neg":
                expr = f"max(0, {k:.6f} - x{j})"
            else:
                expr = f"abs(x{j} - {k:.6f})"
            formula_parts.append(f"({c:+.6f})*{expr}")

        lines = [
            "Validated Sparse Spline Regressor",
            "Exact formula for prediction:",
            "  y = " + " + ".join(formula_parts),
            "",
            "Linear terms (sorted by |coefficient|):",
        ]
        if linear_terms:
            for i, (j, c) in enumerate(linear_terms, 1):
                lines.append(f"  {i:2d}. ({c:+.6f}) * x{j}")
        else:
            lines.append("  (none)")

        lines.append("")
        lines.append("Nonlinear term:")
        if self.nonlinear_term_ is None or abs(float(self.nonlinear_term_["coef"])) <= self.coef_tol:
            lines.append("  (none)")
        else:
            j = int(self.nonlinear_term_["feature"])
            c = float(self.nonlinear_term_["coef"])
            k = float(self.nonlinear_term_["knot"])
            kind = self.nonlinear_term_["kind"]
            if kind == "hinge_pos":
                lines.append(f"  add ({c:+.6f}) * max(0, x{j} - {k:.6f})")
            elif kind == "hinge_neg":
                lines.append(f"  add ({c:+.6f}) * max(0, {k:.6f} - x{j})")
            else:
                lines.append(f"  add ({c:+.6f}) * abs(x{j} - {k:.6f})")

        active = [int(i) for i in self.selected_features_]
        lines.append("")
        lines.append("Features used: " + (", ".join(f"x{i}" for i in active) if active else "none"))
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

        op_count = len(linear_terms) * 2 + max(len(linear_terms) - 1, 0)
        if self.nonlinear_term_ is not None and abs(float(self.nonlinear_term_["coef"])) > self.coef_tol:
            op_count += 3
        lines.append(f"Approximate arithmetic operations: {op_count}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
ValidatedSparseSplineRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "ValidatedSparseSpline_v1"
model_description = "Validation-selected sparse ridge backbone with one optional residual hinge/abs nonlinear basis term"
model_defs = [(model_shorthand_name, ValidatedSparseSplineRegressor())]


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
