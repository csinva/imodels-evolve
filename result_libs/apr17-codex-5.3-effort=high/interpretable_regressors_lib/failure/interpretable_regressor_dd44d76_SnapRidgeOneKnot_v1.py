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


class SnapRidgeOneKnotRegressor(BaseEstimator, RegressorMixin):
    """
    Dense-to-sparse ridge backbone with one optional nonlinear correction.

    1) Select ridge alpha and linear sparsity level via holdout validation.
    2) Optionally add one residual nonlinear term if it gives clear gain.
    3) Refit final arithmetic formula and snap coefficients to simple fractions.
    """

    def __init__(
        self,
        alphas=(1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 2e-1, 1.0, 5.0),
        max_linear_terms=14,
        min_linear_rel=0.015,
        linear_k_grid=(None, 14, 10, 7, 5),
        val_fraction=0.2,
        screen_features=6,
        nonlinear_quantiles=(0.25, 0.5, 0.75),
        min_nonlinear_gain=0.02,
        snap_denominators=(1, 2, 4, 8, 16, 32, 64),
        snap_tol=0.015,
        refit_ridge=1e-8,
        coef_tol=1e-10,
        meaningful_rel=0.10,
        random_state=0,
    ):
        self.alphas = alphas
        self.max_linear_terms = max_linear_terms
        self.min_linear_rel = min_linear_rel
        self.linear_k_grid = linear_k_grid
        self.val_fraction = val_fraction
        self.screen_features = screen_features
        self.nonlinear_quantiles = nonlinear_quantiles
        self.min_nonlinear_gain = min_nonlinear_gain
        self.snap_denominators = snap_denominators
        self.snap_tol = snap_tol
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
        knot = float(knot)
        if kind == "hinge_pos":
            return np.maximum(0.0, xj - knot)
        if kind == "hinge_neg":
            return np.maximum(0.0, knot - xj)
        if kind == "abs":
            return np.abs(xj - knot)
        if kind == "quad":
            return (xj - knot) ** 2
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
        Xtr, Xva = Xs[idx_tr], Xs[idx_va]
        ytr, yva = y[idx_tr], y[idx_va]

        best = None
        best_val = np.inf
        all_feats = np.arange(n_features, dtype=int)

        for alpha in self.alphas:
            inter_dense, coef_dense = self._ridge_with_intercept(Xtr, ytr, alpha)
            order = np.argsort(np.abs(coef_dense))[::-1]
            max_abs = float(np.max(np.abs(coef_dense))) if coef_dense.size else 0.0

            for k in self.linear_k_grid:
                if k is None:
                    feat_idx = all_feats.copy()
                else:
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
                pred_va = inter_k + Xva[:, feat_idx] @ coef_k
                val_rmse = self._rmse(yva, pred_va)
                if val_rmse < best_val:
                    full_coef = np.zeros(n_features, dtype=float)
                    full_coef[feat_idx] = coef_k
                    best_val = val_rmse
                    best = {
                        "alpha": float(alpha),
                        "inter_std": float(inter_k),
                        "coef_std": full_coef,
                        "features": feat_idx.copy(),
                    }

        if best is None:
            self.intercept_ = float(np.mean(y))
            self.linear_coef_ = np.zeros(n_features, dtype=float)
            self.nonlinear_term_ = None
            self.feature_importance_ = np.zeros(n_features, dtype=float)
            self.selected_features_ = []
            self.coef_ = self.linear_coef_.copy()
            return self

        coef_std = np.asarray(best["coef_std"], dtype=float)
        inter_std = float(best["inter_std"])
        linear_coef_raw = coef_std / x_scale
        linear_intercept_raw = float(inter_std - np.sum(coef_std * x_mean / x_scale))

        pred_tr_linear = linear_intercept_raw + X[idx_tr] @ linear_coef_raw
        pred_va_linear = linear_intercept_raw + X[idx_va] @ linear_coef_raw
        resid_tr = y[idx_tr] - pred_tr_linear
        base_val = self._rmse(y[idx_va], pred_va_linear)

        # Screen nonlinear candidates by residual correlation and linear importance.
        xc = X[idx_tr] - X[idx_tr].mean(axis=0, keepdims=True)
        yc = resid_tr - float(np.mean(resid_tr))
        xnorm = np.linalg.norm(xc, axis=0) + 1e-12
        ynorm = float(np.linalg.norm(yc)) + 1e-12
        resid_corr = np.abs((xc.T @ yc) / (xnorm * ynorm))
        resid_corr[~np.isfinite(resid_corr)] = 0.0
        lin_imp = np.abs(linear_coef_raw)
        lin_imp = lin_imp / (float(np.max(lin_imp)) + 1e-12)
        screen_score = resid_corr + 0.35 * lin_imp
        screened = np.argsort(screen_score)[::-1][: min(int(self.screen_features), n_features)]

        best_nl = None
        best_nl_val = base_val
        for j in screened:
            j = int(j)
            xtr_j = X[idx_tr, j]
            xva_j = X[idx_va, j]
            knots = [0.0, float(np.median(xtr_j)), float(np.mean(xtr_j))]
            knots += [float(v) for v in np.quantile(xtr_j, self.nonlinear_quantiles)]
            for knot in np.unique(np.asarray(knots, dtype=float)):
                if not np.isfinite(knot):
                    continue
                for kind in ("hinge_pos", "hinge_neg", "abs", "quad"):
                    col_tr = self._nonlinear_column(xtr_j, kind, knot)
                    denom = float(col_tr @ col_tr) + 1e-12
                    gamma = float((col_tr @ resid_tr) / denom)
                    col_va = self._nonlinear_column(xva_j, kind, knot)
                    trial_val = self._rmse(y[idx_va], pred_va_linear + gamma * col_va)
                    if trial_val < best_nl_val:
                        best_nl_val = trial_val
                        best_nl = {"feature": j, "kind": kind, "knot": float(knot), "coef": float(gamma)}

        if best_nl is not None:
            rel_gain = (base_val - best_nl_val) / (abs(base_val) + 1e-12)
            if rel_gain < float(self.min_nonlinear_gain):
                best_nl = None

        linear_features = [int(j) for j in np.where(np.abs(linear_coef_raw) > self.coef_tol)[0]]
        if not linear_features and n_features > 0:
            linear_features = [int(np.argmax(np.abs(linear_coef_raw)))]

        term_defs = []
        cols = []
        for j in linear_features:
            term_defs.append(("linear", j))
            cols.append(X[:, j].astype(float))

        if best_nl is not None:
            j = int(best_nl["feature"])
            term_defs.append(("nonlinear", j, str(best_nl["kind"]), float(best_nl["knot"])))
            cols.append(self._nonlinear_column(X[:, j], best_nl["kind"], best_nl["knot"]))

        D = np.column_stack(cols) if cols else np.zeros((n_samples, 0), dtype=float)
        inter_final, coef_final = self._ridge_with_intercept(D, y, self.refit_ridge)

        linear_coef_final = np.zeros(n_features, dtype=float)
        nl_term = None
        cursor = 0
        for term in term_defs:
            if term[0] == "linear":
                linear_coef_final[int(term[1])] = float(coef_final[cursor])
                cursor += 1
            else:
                nl_term = {
                    "feature": int(term[1]),
                    "kind": str(term[2]),
                    "knot": float(term[3]),
                    "coef": float(coef_final[cursor]),
                }
                cursor += 1

        # Snap coefficients to simple fractions if prediction error stays close.
        base_pred = inter_final + X @ linear_coef_final
        if nl_term is not None:
            base_pred = base_pred + float(nl_term["coef"]) * self._nonlinear_column(
                X[:, int(nl_term["feature"])], nl_term["kind"], float(nl_term["knot"])
            )
        base_rmse = self._rmse(y, base_pred)

        best_snap = {
            "den": None,
            "intercept": float(inter_final),
            "linear_coef": linear_coef_final.copy(),
            "nonlinear_term": None if nl_term is None else dict(nl_term),
        }
        for den in self.snap_denominators:
            den = float(den)
            if den <= 0:
                continue
            inter_q = float(np.round(inter_final * den) / den)
            linear_q = np.round(linear_coef_final * den) / den
            nl_q = None
            if nl_term is not None:
                nl_q = dict(nl_term)
                nl_q["coef"] = float(np.round(float(nl_q["coef"]) * den) / den)
                nl_q["knot"] = float(np.round(float(nl_q["knot"]) * den) / den)

            pred_q = inter_q + X @ linear_q
            if nl_q is not None:
                pred_q = pred_q + float(nl_q["coef"]) * self._nonlinear_column(
                    X[:, int(nl_q["feature"])], nl_q["kind"], float(nl_q["knot"])
                )
            rmse_q = self._rmse(y, pred_q)
            if rmse_q <= base_rmse * (1.0 + float(self.snap_tol)):
                best_snap = {
                    "den": int(den),
                    "intercept": inter_q,
                    "linear_coef": linear_q,
                    "nonlinear_term": nl_q,
                }
                break

        self.intercept_ = float(best_snap["intercept"])
        self.linear_coef_ = np.asarray(best_snap["linear_coef"], dtype=float)
        self.nonlinear_term_ = best_snap["nonlinear_term"]
        self.alpha_selected_ = float(best["alpha"])
        self.snap_denominator_ = best_snap["den"]

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
            elif kind == "abs":
                expr = f"abs(x{j} - {k:.6f})"
            else:
                expr = f"(x{j} - {k:.6f})^2"
            formula_parts.append(f"({c:+.6f})*{expr}")

        lines = [
            "Snap Ridge + One-Knot Residual Regressor",
            "Exact arithmetic formula (all shown coefficients are used directly):",
            "  y = " + " + ".join(formula_parts),
            "",
            "Non-zero linear terms (sorted by |coefficient|):",
        ]
        if linear_terms:
            for i, (j, c) in enumerate(linear_terms, 1):
                lines.append(f"  {i:2d}. ({c:+.6f}) * x{j}")
        else:
            lines.append("  (none)")

        lines.append("")
        lines.append("Optional nonlinear correction:")
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
            elif kind == "abs":
                lines.append(f"  add ({c:+.6f}) * abs(x{j} - {k:.6f})")
            else:
                lines.append(f"  add ({c:+.6f}) * (x{j} - {k:.6f})^2")

        active = [int(i) for i in self.selected_features_]
        lines.append("")
        lines.append("Features used: " + (", ".join(f"x{i}" for i in active) if active else "none"))
        lines.append("Any feature not listed above has coefficient exactly 0.")
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

        if self.snap_denominator_ is not None:
            lines.append(f"Coefficient snap denominator: {self.snap_denominator_}")

        op_count = len(linear_terms) * 2 + max(len(linear_terms) - 1, 0)
        if self.nonlinear_term_ is not None and abs(float(self.nonlinear_term_["coef"])) > self.coef_tol:
            op_count += 3
        lines.append(f"Approximate arithmetic operations: {op_count}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SnapRidgeOneKnotRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SnapRidgeOneKnot_v1"
model_description = "Validation-selected ridge with adaptive linear sparsity, one optional nonlinear residual term, and snapped coefficients"
model_defs = [(model_shorthand_name, SnapRidgeOneKnotRegressor())]


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
