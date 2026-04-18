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


class StagewiseHingeAdditiveRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse additive regressor trained with stagewise residual fitting over a
    compact dictionary of univariate terms:
      - linear terms,
      - one-sided hinges at a few quantiles,
      - optional squared terms on top features.
    """

    def __init__(
        self,
        max_terms=14,
        max_screen_features=30,
        n_hinge_quantiles=4,
        square_screen_features=8,
        learning_rate=0.65,
        min_improvement=1e-4,
        ridge=1e-3,
        coef_quant_step=0.05,
        max_quantization_mse_ratio=1.02,
        coef_tol=1e-10,
        meaningful_rel=0.12,
    ):
        self.max_terms = max_terms
        self.max_screen_features = max_screen_features
        self.n_hinge_quantiles = n_hinge_quantiles
        self.square_screen_features = square_screen_features
        self.learning_rate = learning_rate
        self.min_improvement = min_improvement
        self.ridge = ridge
        self.coef_quant_step = coef_quant_step
        self.max_quantization_mse_ratio = max_quantization_mse_ratio
        self.coef_tol = coef_tol
        self.meaningful_rel = meaningful_rel

    @staticmethod
    def _abs_corr_scores(X, y):
        y_centered = y - float(np.mean(y))
        y_norm = float(np.linalg.norm(y_centered)) + 1e-12
        X_centered = X - X.mean(axis=0, keepdims=True)
        x_norm = np.linalg.norm(X_centered, axis=0) + 1e-12
        scores = np.abs((X_centered.T @ y_centered) / (x_norm * y_norm))
        scores[np.isnan(scores)] = 0.0
        return scores

    def _build_term_defs(self, X, y):
        n_features = X.shape[1]
        corr = self._abs_corr_scores(X, y)
        order = np.argsort(corr)[::-1]
        screen = order[: min(int(self.max_screen_features), n_features)]
        square_screen = set(order[: min(int(self.square_screen_features), n_features)].tolist())

        term_defs = []
        for j in screen:
            j = int(j)
            xj = X[:, j]
            term_defs.append(("linear", j))
            if j in square_screen:
                term_defs.append(("square", j))
            if int(self.n_hinge_quantiles) > 0:
                q_vals = np.linspace(0.2, 0.8, int(self.n_hinge_quantiles))
                knots = np.unique(np.quantile(xj, q_vals))
                for knot in knots:
                    k = float(np.round(knot, 6))
                    term_defs.append(("hinge_pos", j, k))
                    term_defs.append(("hinge_neg", j, k))

        return term_defs

    @staticmethod
    def _eval_terms(X, term_defs):
        cols = []
        for term in term_defs:
            kind = term[0]
            if kind == "linear":
                col = X[:, term[1]]
            elif kind == "square":
                xj = X[:, term[1]]
                col = xj * xj
            elif kind == "hinge_pos":
                col = np.maximum(0.0, X[:, term[1]] - term[2])
            elif kind == "hinge_neg":
                col = np.maximum(0.0, term[2] - X[:, term[1]])
            else:
                raise ValueError(f"Unknown term kind: {kind}")
            cols.append(np.asarray(col, dtype=float))
        return np.column_stack(cols) if cols else np.zeros((X.shape[0], 0), dtype=float)

    @staticmethod
    def _solve_with_intercept(W, y, ridge):
        n = W.shape[0]
        if W.shape[1] == 0:
            intercept = float(np.mean(y))
            pred = np.full(n, intercept, dtype=float)
            mse = float(np.mean((y - pred) ** 2))
            return intercept, np.zeros(0, dtype=float), pred, mse

        A = np.column_stack([np.ones(n, dtype=float), W])
        reg = np.eye(A.shape[1], dtype=float)
        reg[0, 0] = 0.0
        lhs = A.T @ A + max(float(ridge), 0.0) * reg
        rhs = A.T @ y
        try:
            sol = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        intercept = float(sol[0])
        coef = np.asarray(sol[1:], dtype=float)
        pred = intercept + W @ coef
        mse = float(np.mean((y - pred) ** 2))
        return intercept, coef, pred, mse

    @staticmethod
    def _quantize(beta, step):
        if step <= 0:
            return beta
        return np.round(beta / step) * step

    def _term_to_str(self, term):
        kind = term[0]
        if kind == "linear":
            return f"x{term[1]}"
        if kind == "square":
            return f"(x{term[1]}^2)"
        if kind == "hinge_pos":
            return f"max(0, x{term[1]} - {term[2]:.4f})"
        if kind == "hinge_neg":
            return f"max(0, {term[2]:.4f} - x{term[1]})"
        return str(term)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        self.all_term_defs_ = self._build_term_defs(X, y)
        Z = self._eval_terms(X, self.all_term_defs_)

        if Z.shape[1] == 0:
            self.intercept_ = float(np.mean(y))
            self.term_defs_ = []
            self.coef_ = np.zeros(0, dtype=float)
            self.feature_importance_ = np.zeros(n_features, dtype=float)
            self.selected_features_ = []
            self.quantization_step_ = 0.0
            return self

        col_norm2 = np.sum(Z * Z, axis=0) + float(self.ridge)
        pred = np.full(n_samples, float(np.mean(y)), dtype=float)
        prev_mse = float(np.mean((y - pred) ** 2))
        min_abs_gain = float(self.min_improvement) * (float(np.var(y)) + 1e-12)

        selected_idx = []
        selected_coef = []
        max_rounds = min(int(self.max_terms), max(1, Z.shape[1]))

        for _ in range(max_rounds):
            residual = y - pred
            dots = Z.T @ residual
            raw_coef = dots / col_norm2
            gains = (dots * dots) / col_norm2 / max(float(n_samples), 1.0)

            best_idx = int(np.argmax(gains))
            if not np.isfinite(gains[best_idx]) or gains[best_idx] <= 0:
                break

            step_coef = float(self.learning_rate) * float(raw_coef[best_idx])
            trial_pred = pred + step_coef * Z[:, best_idx]
            trial_mse = float(np.mean((y - trial_pred) ** 2))
            if prev_mse - trial_mse < min_abs_gain:
                break

            pred = trial_pred
            prev_mse = trial_mse
            selected_idx.append(best_idx)
            selected_coef.append(step_coef)

        if not selected_idx:
            self.intercept_ = float(np.mean(y))
            self.term_defs_ = []
            self.coef_ = np.zeros(0, dtype=float)
            self.feature_importance_ = np.zeros(n_features, dtype=float)
            self.selected_features_ = []
            self.quantization_step_ = 0.0
            return self

        # Merge repeated selections, preserving selection order.
        merged = {}
        order = []
        for idx, c in zip(selected_idx, selected_coef):
            if idx not in merged:
                merged[idx] = 0.0
                order.append(idx)
            merged[idx] += float(c)
        uniq_idx = order
        W = Z[:, uniq_idx]

        _, coef_exact, _, mse_exact = self._solve_with_intercept(W, y, self.ridge)

        step = float(self.coef_quant_step)
        coef_use = np.asarray(coef_exact, dtype=float)
        quant_step = 0.0
        if step > 0:
            coef_q = self._quantize(coef_exact, step)
            inter_q = float(np.mean(y - W @ coef_q))
            mse_q = float(np.mean((y - (inter_q + W @ coef_q)) ** 2))
            if mse_q <= float(mse_exact) * float(self.max_quantization_mse_ratio):
                coef_use = coef_q
                quant_step = step

        keep = np.abs(coef_use) > float(self.coef_tol)
        kept_idx = [uniq_idx[i] for i in range(len(uniq_idx)) if keep[i]]
        kept_coef = coef_use[keep]
        kept_terms = [self.all_term_defs_[i] for i in kept_idx]

        if len(kept_idx) == 0:
            self.intercept_ = float(np.mean(y))
            self.term_defs_ = []
            self.coef_ = np.zeros(0, dtype=float)
        else:
            W_keep = Z[:, kept_idx]
            self.intercept_ = float(np.mean(y - W_keep @ kept_coef))
            self.term_defs_ = kept_terms
            self.coef_ = np.asarray(kept_coef, dtype=float)

        self.quantization_step_ = quant_step

        feat_importance = np.zeros(n_features, dtype=float)
        for c, t in zip(self.coef_, self.term_defs_):
            if t[0] in {"linear", "square", "hinge_pos", "hinge_neg"}:
                feat_importance[int(t[1])] += abs(float(c))
        self.feature_importance_ = feat_importance
        self.selected_features_ = sorted(int(i) for i in np.where(feat_importance > self.coef_tol)[0])
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "term_defs_", "coef_"])
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if len(self.term_defs_) == 0:
            return np.full(X.shape[0], self.intercept_, dtype=float)
        Z = self._eval_terms(X, self.term_defs_)
        return self.intercept_ + Z @ self.coef_

    def __str__(self):
        check_is_fitted(self, ["intercept_", "term_defs_", "coef_", "feature_importance_"])
        lines = [
            "Stagewise Hinge Additive Regressor",
            "Exact prediction recipe:",
            "  1) Start with intercept.",
            "  2) Add each listed term contribution.",
            "  3) Return the final single number.",
            "  Use hinge rule: max(0, a).",
            f"intercept = {self.intercept_:+.5f}",
        ]
        if self.quantization_step_ > 0:
            lines.append(f"coefficient quantization step: {self.quantization_step_:.3f}")

        lines.append("")
        lines.append("Active terms:")
        if len(self.term_defs_) == 0:
            lines.append("  (constant model)")
        else:
            for i, (coef, term) in enumerate(zip(self.coef_, self.term_defs_), 1):
                lines.append(f"  {i:2d}. add ({float(coef):+.5f}) * {self._term_to_str(term)}")

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
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
StagewiseHingeAdditiveRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "StagewiseHingeAdditive_v1"
model_description = "Stagewise residual fitting over sparse linear/hinge/square univariate terms with compact quantized arithmetic form"
model_defs = [(model_shorthand_name, StagewiseHingeAdditiveRegressor())]


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
