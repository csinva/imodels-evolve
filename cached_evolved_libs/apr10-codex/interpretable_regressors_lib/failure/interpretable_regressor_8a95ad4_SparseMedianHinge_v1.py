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


class SparseMedianHingeRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse additive equation over raw features and median-threshold hinge terms.

    Basis dictionary per feature j:
      - linear: x_j
      - right hinge: max(0, x_j - t_j)
      - left hinge:  max(0, t_j - x_j)
    where t_j is the feature median on the training set.

    Terms are selected greedily on residual correlation, then jointly refit with
    a small ridge penalty. The final string is an explicit raw-feature equation,
    optimized for manual simulation.
    """

    def __init__(
        self,
        max_terms=10,
        min_terms=2,
        alpha_grid=(1e-4, 1e-3, 1e-2, 1e-1, 1.0),
        min_rel_gain=0.03,
        linear_bonus=1.08,
        prune_rel=0.04,
        coef_display_tol=0.03,
    ):
        self.max_terms = max_terms
        self.min_terms = min_terms
        self.alpha_grid = alpha_grid
        self.min_rel_gain = min_rel_gain
        self.linear_bonus = linear_bonus
        self.prune_rel = prune_rel
        self.coef_display_tol = coef_display_tol

    @staticmethod
    def _center_col(v):
        return v - float(np.mean(v))

    @staticmethod
    def _ridge_fit(X, y, alpha):
        if X.shape[1] == 0:
            return np.zeros(0, dtype=float)
        XtX = X.T @ X
        A = XtX + float(alpha) * np.eye(X.shape[1], dtype=float)
        Xty = X.T @ y
        return np.linalg.solve(A, Xty)

    @staticmethod
    def _gcv_alpha(X, y, alpha_grid):
        n = X.shape[0]
        if X.shape[1] == 0:
            return 1.0
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        best_alpha = float(alpha_grid[0])
        best_gcv = None
        for a in alpha_grid:
            a = float(a)
            filt = s / (s * s + a)
            w = Vt.T @ (filt * (U.T @ y))
            pred = X @ w
            mse = float(np.mean((y - pred) ** 2))
            dof = float(np.sum((s * s) / (s * s + a)))
            denom = max(1e-7, 1.0 - dof / max(1.0, float(n)))
            gcv = mse / (denom * denom)
            if (best_gcv is None) or (gcv < best_gcv):
                best_gcv = gcv
                best_alpha = a
        return best_alpha

    def _build_dictionary(self, X):
        n, p = X.shape
        thresholds = np.median(X, axis=0)

        cols = []
        meta = []
        for j in range(p):
            xj = X[:, j]
            t = float(thresholds[j])

            # linear term
            cols.append(self._center_col(xj))
            meta.append((int(j), "lin", t))

            # right hinge: max(0, x - t)
            h_pos = np.maximum(0.0, xj - t)
            cols.append(self._center_col(h_pos))
            meta.append((int(j), "hinge_pos", t))

            # left hinge: max(0, t - x)
            h_neg = np.maximum(0.0, t - xj)
            cols.append(self._center_col(h_neg))
            meta.append((int(j), "hinge_neg", t))

        Phi = np.column_stack(cols) if cols else np.zeros((n, 0), dtype=float)
        return Phi, meta, thresholds

    def _active_term_limit(self, p):
        base = int(np.sqrt(max(1, p)) + 3)
        return int(max(self.min_terms, min(self.max_terms, max(3, base))))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape

        self.n_features_in_ = int(p)
        y_mean = float(np.mean(y))
        yc = y - y_mean

        if p == 0:
            self.thresholds_ = np.zeros(0, dtype=float)
            self.selected_idx_ = np.zeros(0, dtype=int)
            self.selected_meta_ = []
            self.coef_ = np.zeros(0, dtype=float)
            self.intercept_ = y_mean
            self.alpha_ = 1.0
            self.training_mse_ = float(np.mean((y - y_mean) ** 2))
            self.is_fitted_ = True
            return self

        Phi, meta, thresholds = self._build_dictionary(X)
        norms = np.sqrt(np.sum(Phi * Phi, axis=0)) + 1e-12

        max_terms = self._active_term_limit(p)
        min_terms = int(max(1, min(self.min_terms, max_terms)))

        selected = []
        selected_set = set()
        resid = yc.copy()
        y_scale = float(np.linalg.norm(yc)) + 1e-12
        gain_floor = float(self.min_rel_gain) * y_scale / np.sqrt(max(1, n))
        alpha_grid = np.asarray(self.alpha_grid, dtype=float)
        alpha_grid = alpha_grid[np.isfinite(alpha_grid) & (alpha_grid > 0)]
        if alpha_grid.size == 0:
            alpha_grid = np.asarray([1.0], dtype=float)

        best_alpha = 1.0
        w_sel = np.zeros(0, dtype=float)

        for step in range(max_terms):
            best_idx = None
            best_score = -np.inf
            for idx in range(Phi.shape[1]):
                if idx in selected_set:
                    continue
                col = Phi[:, idx]
                corr = abs(float(np.dot(resid, col))) / norms[idx]
                if meta[idx][1] == "lin":
                    corr *= float(self.linear_bonus)
                if corr > best_score:
                    best_score = corr
                    best_idx = idx

            if best_idx is None:
                break
            if step >= min_terms and best_score < gain_floor:
                break

            selected.append(int(best_idx))
            selected_set.add(int(best_idx))

            X_sel = Phi[:, selected]
            best_alpha = self._gcv_alpha(X_sel, yc, alpha_grid)
            w_sel = self._ridge_fit(X_sel, yc, best_alpha)
            resid = yc - X_sel @ w_sel

        if not selected:
            selected = [int(np.argmax(norms))]
            X_sel = Phi[:, selected]
            best_alpha = self._gcv_alpha(X_sel, yc, alpha_grid)
            w_sel = self._ridge_fit(X_sel, yc, best_alpha)

        # Prune tiny terms, then refit.
        abs_w = np.abs(w_sel)
        w_max = float(np.max(abs_w)) if abs_w.size else 0.0
        keep_local = [i for i, w in enumerate(abs_w) if w >= float(self.prune_rel) * max(1e-12, w_max)]
        if len(keep_local) < min_terms:
            order = np.argsort(-abs_w)
            keep_local = [int(i) for i in order[:min_terms]]

        keep_local = sorted(set(keep_local))
        selected = [selected[i] for i in keep_local]
        X_sel = Phi[:, selected]
        best_alpha = self._gcv_alpha(X_sel, yc, alpha_grid)
        w_sel = self._ridge_fit(X_sel, yc, best_alpha)

        intercept = y_mean
        if X_sel.shape[1] > 0:
            # columns are centered, so intercept remains mean(y)
            intercept = y_mean

        pred = intercept + X_sel @ w_sel

        self.thresholds_ = np.asarray(thresholds, dtype=float)
        self.selected_idx_ = np.asarray(selected, dtype=int)
        self.selected_meta_ = [meta[i] for i in selected]
        self.coef_ = np.asarray(w_sel, dtype=float)
        self.intercept_ = float(intercept)
        self.alpha_ = float(best_alpha)
        self.training_mse_ = float(np.mean((y - pred) ** 2))
        self.is_fitted_ = True
        return self

    def _transform_selected(self, X):
        X = np.asarray(X, dtype=float)
        if self.selected_idx_.size == 0:
            return np.zeros((X.shape[0], 0), dtype=float)

        cols = []
        for feat, term_type, threshold in self.selected_meta_:
            xj = X[:, feat]
            if term_type == "lin":
                v = xj
            elif term_type == "hinge_pos":
                v = np.maximum(0.0, xj - threshold)
            else:
                v = np.maximum(0.0, threshold - xj)
            cols.append(v - float(np.mean(v)))
        return np.column_stack(cols)

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        X = np.asarray(X, dtype=float)
        Phi_sel = self._transform_selected(X)
        return self.intercept_ + Phi_sel @ self.coef_

    def __str__(self):
        check_is_fitted(self, "is_fitted_")
        lines = ["Sparse Median-Hinge Equation Regressor:"]
        lines.append(f"  intercept: {self.intercept_:+.4f}")
        lines.append(f"  selected terms: {len(self.selected_meta_)}")
        lines.append(f"  ridge alpha (GCV): {self.alpha_:.6f}")
        lines.append(f"  training MSE: {self.training_mse_:.6f}")

        if len(self.selected_meta_) == 0:
            lines.append("  equation: y = intercept")
            return "\n".join(lines)

        pieces = [f"{self.intercept_:+.4f}"]
        terms = []
        for coef, (feat, term_type, threshold) in zip(self.coef_, self.selected_meta_):
            c = float(coef)
            if term_type == "lin":
                term = f"x{feat}"
            elif term_type == "hinge_pos":
                term = f"max(0, x{feat} - {threshold:.4f})"
            else:
                term = f"max(0, {threshold:.4f} - x{feat})"
            terms.append((feat, term_type, c, threshold, term))

        for _, _, c, _, term in sorted(terms, key=lambda z: -abs(z[2])):
            pieces.append(f"{c:+.4f}*{term}")

        lines.append("  equation:")
        lines.append("    y = " + " ".join(pieces))

        tol = float(max(0.0, self.coef_display_tol))
        visible_terms = [t for t in terms if abs(t[2]) >= tol]
        if visible_terms:
            lines.append("  active terms (sorted by |coefficient|):")
            for feat, term_type, c, threshold, _ in sorted(visible_terms, key=lambda z: -abs(z[2])):
                if term_type == "lin":
                    desc = f"x{feat}"
                elif term_type == "hinge_pos":
                    desc = f"max(0, x{feat} - {threshold:.4f})"
                else:
                    desc = f"max(0, {threshold:.4f} - x{feat})"
                lines.append(f"    {c:+.4f} * {desc}")

        used_features = sorted({int(f) for f, _, _ in self.selected_meta_})
        if used_features:
            lines.append("  used features: " + ", ".join(f"x{j}" for j in used_features))
        approx_ops = 1 + 3 * len(self.selected_meta_)
        lines.append(f"  approximate arithmetic operations: ~{approx_ops}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparseMedianHingeRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SparseMedianHinge_v1"
model_description = "Greedy sparse additive equation over raw linear and median-threshold hinge terms with GCV-ridge refit"
model_defs = [(model_shorthand_name, SparseMedianHingeRegressor())]


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
