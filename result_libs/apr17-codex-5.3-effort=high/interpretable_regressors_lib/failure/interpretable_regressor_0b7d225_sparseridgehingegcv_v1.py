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


class SparseRidgeHingeRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse additive model with:
      1) a ridge-regularized linear backbone (alpha chosen by GCV),
      2) a tiny number of residual hinge terms for threshold-like behavior.

    The final expression is intentionally compact and printed as explicit
    arithmetic for manual simulation.
    """

    def __init__(
        self,
        alphas=(0.0, 0.01, 0.05, 0.2, 1.0, 5.0),
        rel_linear_cutoff=0.08,
        min_linear_terms=3,
        max_linear_terms=16,
        max_hinges=2,
        hinge_screen_features=8,
        hinge_quantiles=(0.2, 0.35, 0.5, 0.65, 0.8),
        min_relative_gain=0.01,
        ridge_refit_multiplier=0.5,
        coef_tol=1e-6,
        quantization=0.01,
    ):
        self.alphas = alphas
        self.rel_linear_cutoff = rel_linear_cutoff
        self.min_linear_terms = min_linear_terms
        self.max_linear_terms = max_linear_terms
        self.max_hinges = max_hinges
        self.hinge_screen_features = hinge_screen_features
        self.hinge_quantiles = hinge_quantiles
        self.min_relative_gain = min_relative_gain
        self.ridge_refit_multiplier = ridge_refit_multiplier
        self.coef_tol = coef_tol
        self.quantization = quantization

    @staticmethod
    def _round_to_step(values, step):
        if step <= 0:
            return values
        return step * np.round(values / step)

    def _fit_ridge_gcv(self, Xs, target):
        n_samples = Xs.shape[0]
        if Xs.size == 0:
            return 0.0, np.zeros(Xs.shape[1], dtype=float)

        U, svals, Vt = np.linalg.svd(Xs, full_matrices=False)
        uy = U.T @ target
        s2 = svals * svals

        best_alpha = float(self.alphas[0]) if self.alphas else 0.0
        best_gcv = np.inf
        best_coef = np.zeros(Xs.shape[1], dtype=float)

        for alpha in self.alphas:
            a = max(float(alpha), 0.0)
            denom = s2 + a
            shrink = s2 / denom
            y_hat = U @ (shrink * uy)
            resid = target - y_hat
            rss = float(resid @ resid)
            df = float(np.sum(shrink))
            gap = max(n_samples - df, 1e-6)
            gcv = rss / (gap * gap)
            if gcv < best_gcv:
                best_gcv = gcv
                best_alpha = a
                best_coef = Vt.T @ ((svals / denom) * uy)

        return best_alpha, np.asarray(best_coef, dtype=float)

    def _select_linear_features(self, coef_dense):
        n_features = coef_dense.shape[0]
        if n_features == 0:
            return []
        abs_coef = np.abs(coef_dense)
        max_abs = float(abs_coef.max()) if abs_coef.size else 0.0
        if max_abs <= 1e-12:
            top_k = min(max(self.min_linear_terms, 1), n_features)
            return [int(i) for i in np.argsort(abs_coef)[::-1][:top_k]]

        active = np.where(abs_coef >= float(self.rel_linear_cutoff) * max_abs)[0].astype(int).tolist()
        if len(active) < self.min_linear_terms:
            top = np.argsort(abs_coef)[::-1][: min(self.min_linear_terms, n_features)]
            active = [int(i) for i in top]
        if len(active) > self.max_linear_terms:
            order = sorted(active, key=lambda i: abs_coef[i], reverse=True)
            active = [int(i) for i in order[: self.max_linear_terms]]
        return sorted(set(active))

    @staticmethod
    def _hinge_column(xcol, threshold, direction):
        if direction > 0:
            return np.maximum(0.0, xcol - threshold)
        return np.maximum(0.0, threshold - xcol)

    def _design_matrix(self, X, linear_feats, hinge_defs):
        cols = []
        term_defs = []

        for feat_idx in linear_feats:
            cols.append(X[:, int(feat_idx)].astype(float))
            term_defs.append(("lin", int(feat_idx)))

        for feat_idx, threshold, direction in hinge_defs:
            col = self._hinge_column(X[:, int(feat_idx)], float(threshold), int(direction))
            cols.append(col.astype(float))
            term_defs.append(("hinge", int(feat_idx), float(threshold), int(direction)))

        if not cols:
            return np.zeros((X.shape[0], 0), dtype=float), term_defs
        return np.column_stack(cols), term_defs

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        n_samples, n_features = X.shape

        self.n_features_in_ = n_features
        self.x_mean_ = X.mean(axis=0)
        self.x_scale_ = X.std(axis=0)
        self.x_scale_[self.x_scale_ < 1e-12] = 1.0

        Xs = (X - self.x_mean_) / self.x_scale_
        y_centered = y - float(y.mean())

        alpha, coef_std = self._fit_ridge_gcv(Xs, y_centered)
        self.alpha_ = float(alpha)

        linear_feats = self._select_linear_features(coef_std)
        if not linear_feats and n_features > 0:
            linear_feats = [int(np.argmax(np.abs(coef_std)))]

        # Initial residual using sparse linear backbone.
        linear_coef_raw = np.zeros(n_features, dtype=float)
        for feat_idx in linear_feats:
            linear_coef_raw[feat_idx] = float(coef_std[feat_idx] / self.x_scale_[feat_idx])
        intercept0 = float(y.mean() - np.dot(self.x_mean_, linear_coef_raw))
        pred = intercept0 + X @ linear_coef_raw
        resid = y - pred
        prev_mse = float(np.mean(resid ** 2))

        # Greedy residual hinge additions.
        hinge_defs = []
        Xc = X - X.mean(axis=0, keepdims=True)
        x_norms = np.linalg.norm(Xc, axis=0) + 1e-12

        for _ in range(int(self.max_hinges)):
            resid_norm = float(np.linalg.norm(resid)) + 1e-12
            corrs = np.abs((Xc.T @ resid) / (x_norms * resid_norm))
            top_k = min(int(self.hinge_screen_features), n_features)
            candidate_features = [int(i) for i in np.argsort(corrs)[::-1][:top_k]]

            best = None
            best_mse = prev_mse
            for feat_idx in candidate_features:
                xj = X[:, feat_idx]
                thresholds = np.unique(np.quantile(xj, self.hinge_quantiles))
                for threshold in thresholds:
                    for direction in (1, -1):
                        candidate_key = (int(feat_idx), float(threshold), int(direction))
                        if candidate_key in hinge_defs:
                            continue
                        h = self._hinge_column(xj, float(threshold), int(direction))
                        h_norm2 = float(h @ h)
                        if h_norm2 <= 1e-12:
                            continue
                        gamma = float((h @ resid) / h_norm2)
                        trial_resid = resid - gamma * h
                        mse = float(np.mean(trial_resid ** 2))
                        if mse < best_mse - 1e-12:
                            best_mse = mse
                            best = (candidate_key, trial_resid)

            if best is None:
                break

            rel_gain = (prev_mse - best_mse) / (abs(prev_mse) + 1e-12)
            if rel_gain < float(self.min_relative_gain):
                break

            hinge_defs.append(best[0])
            resid = best[1]
            prev_mse = best_mse

        # Joint ridge refit on selected raw terms, intercept unpenalized.
        D, term_defs = self._design_matrix(X, linear_feats, hinge_defs)
        if D.shape[1] == 0:
            self.intercept_ = float(np.mean(y))
            self.coef_ = np.zeros(n_features, dtype=float)
            self.hinge_terms_ = []
            self.feature_importance_ = np.zeros(n_features, dtype=float)
            self.selected_features_ = []
            return self

        A = np.column_stack([np.ones(n_samples, dtype=float), D])
        refit_alpha = max(float(self.alpha_) * float(self.ridge_refit_multiplier), 1e-8)
        penalty = np.eye(A.shape[1], dtype=float)
        penalty[0, 0] = 0.0
        lhs = A.T @ A + refit_alpha * penalty
        rhs = A.T @ y
        try:
            aug_coef = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            aug_coef, *_ = np.linalg.lstsq(A, y, rcond=None)

        intercept = float(aug_coef[0])
        term_coef = np.asarray(aug_coef[1:], dtype=float)

        if float(self.quantization) > 0:
            intercept = float(self._round_to_step(np.array([intercept]), float(self.quantization))[0])
            term_coef = self._round_to_step(term_coef, float(self.quantization))

        keep = np.abs(term_coef) > float(self.coef_tol)
        term_defs_kept = [t for t, k in zip(term_defs, keep) if k]
        coef_kept = term_coef[keep]

        self.intercept_ = intercept
        self.coef_ = np.zeros(n_features, dtype=float)
        self.hinge_terms_ = []

        for term, coef_val in zip(term_defs_kept, coef_kept):
            if term[0] == "lin":
                self.coef_[term[1]] = float(coef_val)
            else:
                self.hinge_terms_.append(
                    {
                        "feature": int(term[1]),
                        "threshold": float(term[2]),
                        "direction": int(term[3]),
                        "coef": float(coef_val),
                    }
                )

        importance = np.abs(self.coef_)
        for rule in self.hinge_terms_:
            importance[int(rule["feature"])] += abs(float(rule["coef"]))
        self.feature_importance_ = importance
        self.selected_features_ = sorted(int(i) for i in np.where(importance > self.coef_tol)[0])
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "coef_", "hinge_terms_"])
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        pred = self.intercept_ + X @ self.coef_
        for term in self.hinge_terms_:
            feat_idx = int(term["feature"])
            threshold = float(term["threshold"])
            direction = int(term["direction"])
            coef_val = float(term["coef"])
            pred += coef_val * self._hinge_column(X[:, feat_idx], threshold, direction)
        return pred

    def __str__(self):
        check_is_fitted(self, ["intercept_", "coef_", "hinge_terms_", "feature_importance_"])
        lines = [
            "Sparse Ridge-Hinge Regressor",
            "Compute prediction by explicit arithmetic:",
            f"  1) Start with y = {self.intercept_:+.4f}",
        ]

        nz_linear = [int(i) for i in np.where(np.abs(self.coef_) > self.coef_tol)[0]]
        if nz_linear:
            lines.append("  2) Add linear terms:")
            for feat_idx in sorted(nz_linear, key=lambda i: abs(float(self.coef_[i])), reverse=True):
                lines.append(f"     y += {float(self.coef_[feat_idx]):+.4f} * x{feat_idx}")
        else:
            lines.append("  2) No linear terms.")

        if self.hinge_terms_:
            lines.append("  3) Add hinge terms:")
            for term in sorted(self.hinge_terms_, key=lambda t: abs(float(t["coef"])), reverse=True):
                feat_idx = int(term["feature"])
                threshold = float(term["threshold"])
                direction = int(term["direction"])
                coef_val = float(term["coef"])
                if direction > 0:
                    expr = f"max(0, x{feat_idx} - {threshold:.4f})"
                else:
                    expr = f"max(0, {threshold:.4f} - x{feat_idx})"
                lines.append(f"     y += {coef_val:+.4f} * {expr}")
        else:
            lines.append("  3) No hinge terms.")

        active = sorted(int(i) for i in np.where(self.feature_importance_ > self.coef_tol)[0])
        if active:
            lines.append("")
            lines.append("Active features: " + ", ".join(f"x{i}" for i in active))
        if self.n_features_in_ <= 30 and len(active) < self.n_features_in_:
            inactive = [f"x{i}" for i in range(self.n_features_in_) if i not in set(active)]
            lines.append("Inactive features (zero contribution): " + ", ".join(inactive))
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparseRidgeHingeRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SparseRidgeHingeGCV_v1"
model_description = "GCV-selected ridge backbone pruned to dominant linear terms, plus a few residual hinge threshold terms"
model_defs = [(model_shorthand_name, SparseRidgeHingeRegressor())]


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
