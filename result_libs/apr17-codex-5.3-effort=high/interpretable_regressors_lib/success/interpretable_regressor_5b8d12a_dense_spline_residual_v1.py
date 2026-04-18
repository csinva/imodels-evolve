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


class DenseSplineResidualRegressor(BaseEstimator, RegressorMixin):
    """
    Dense linear ridge backbone with a tiny residual basis:
      1) fit all linear features via closed-form ridge with GCV alpha selection,
      2) greedily add a few residual terms (hinges / interactions),
      3) refit all selected terms jointly with light ridge shrinkage.

    Compared to sparse-only models, this keeps stronger predictive coverage while
    remaining fully arithmetic and explicit for simulation from the model string.
    """

    def __init__(
        self,
        alphas=(0.0, 1e-4, 1e-3, 1e-2, 5e-2, 0.2, 1.0, 5.0),
        nonlinear_screen_features=7,
        hinge_quantiles=(0.2, 0.4, 0.6, 0.8),
        max_hinges=2,
        max_interactions=1,
        min_relative_gain=0.0025,
        refit_ridge=1e-4,
        coef_tol=1e-10,
        meaningful_rel=0.1,
    ):
        self.alphas = alphas
        self.nonlinear_screen_features = nonlinear_screen_features
        self.hinge_quantiles = hinge_quantiles
        self.max_hinges = max_hinges
        self.max_interactions = max_interactions
        self.min_relative_gain = min_relative_gain
        self.refit_ridge = refit_ridge
        self.coef_tol = coef_tol
        self.meaningful_rel = meaningful_rel

    @staticmethod
    def _hinge_column(xcol, threshold, direction):
        if int(direction) > 0:
            return np.maximum(0.0, xcol - float(threshold))
        return np.maximum(0.0, float(threshold) - xcol)

    @staticmethod
    def _solve_linear_system_with_intercept(D, y, ridge):
        n_samples = D.shape[0]
        if D.shape[1] == 0:
            return float(np.mean(y)), np.zeros(0, dtype=float)

        A = np.column_stack([np.ones(n_samples, dtype=float), D])
        reg = np.eye(A.shape[1], dtype=float)
        reg[0, 0] = 0.0  # do not regularize intercept
        lhs = A.T @ A + max(float(ridge), 0.0) * reg
        rhs = A.T @ y
        try:
            sol = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        return float(sol[0]), np.asarray(sol[1:], dtype=float)

    def _fit_ridge_gcv(self, Xs, y_centered):
        n_samples, n_features = Xs.shape
        if n_features == 0:
            return 0.0, np.zeros(0, dtype=float)

        U, svals, Vt = np.linalg.svd(Xs, full_matrices=False)
        uy = U.T @ y_centered
        s2 = svals * svals

        best_alpha = 0.0
        best_score = np.inf
        best_coef = np.zeros(n_features, dtype=float)

        for alpha in self.alphas:
            a = max(float(alpha), 0.0)
            denom = s2 + a
            shrink = s2 / denom
            y_hat = U @ (shrink * uy)
            resid = y_centered - y_hat
            rss = float(resid @ resid)
            df = float(np.sum(shrink))
            denom_gcv = max(n_samples - df, 1e-8)
            gcv = rss / (denom_gcv * denom_gcv)
            if gcv < best_score:
                best_score = gcv
                best_alpha = a
                best_coef = Vt.T @ ((svals / denom) * uy)

        return float(best_alpha), np.asarray(best_coef, dtype=float)

    def _build_design(self, X, extra_terms):
        cols = [X[:, j].astype(float) for j in range(X.shape[1])]
        term_defs = [("lin", int(j)) for j in range(X.shape[1])]

        for term in extra_terms:
            if term[0] == "hinge":
                _, feat, thr, direction = term
                cols.append(self._hinge_column(X[:, int(feat)], float(thr), int(direction)).astype(float))
                term_defs.append(("hinge", int(feat), float(thr), int(direction)))
            else:
                _, fi, fj = term
                cols.append((X[:, int(fi)] * X[:, int(fj)]).astype(float))
                term_defs.append(("inter", int(fi), int(fj)))

        return np.column_stack(cols), term_defs

    def _feature_screen(self, coef_linear):
        n_features = coef_linear.shape[0]
        if n_features == 0:
            return []
        order = np.argsort(np.abs(coef_linear))[::-1]
        top_k = min(int(self.nonlinear_screen_features), n_features)
        return [int(i) for i in order[:top_k]]

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        self.x_mean_ = X.mean(axis=0)
        self.x_scale_ = X.std(axis=0)
        self.x_scale_[self.x_scale_ < 1e-12] = 1.0

        Xs = (X - self.x_mean_) / self.x_scale_
        y_mean = float(y.mean())
        y_centered = y - y_mean

        alpha, coef_std = self._fit_ridge_gcv(Xs, y_centered)
        self.alpha_ = float(alpha)

        coef_linear = coef_std / self.x_scale_
        intercept_linear = y_mean - float(self.x_mean_ @ coef_linear)

        screen_features = self._feature_screen(coef_linear)
        used_hinges = set()
        used_interactions = set()
        extra_terms = []

        D, term_defs = self._build_design(X, extra_terms)
        intercept, coef_all = self._solve_linear_system_with_intercept(D, y, self.refit_ridge)
        pred = intercept + D @ coef_all
        prev_mse = float(np.mean((y - pred) ** 2))

        max_total_extra = int(self.max_hinges) + int(self.max_interactions)
        for _ in range(max_total_extra):
            resid = y - pred
            best_term = None
            best_mse = prev_mse

            if len(used_interactions) < int(self.max_interactions) and len(screen_features) >= 2:
                for a in range(len(screen_features)):
                    for b in range(a + 1, len(screen_features)):
                        fi = int(screen_features[a])
                        fj = int(screen_features[b])
                        key = (min(fi, fj), max(fi, fj))
                        if key in used_interactions:
                            continue
                        col = X[:, fi] * X[:, fj]
                        denom = float(col @ col)
                        if denom <= 1e-12:
                            continue
                        gamma = float((col @ resid) / denom)
                        mse = float(np.mean((resid - gamma * col) ** 2))
                        if mse < best_mse - 1e-12:
                            best_mse = mse
                            best_term = ("inter", key[0], key[1])

            if len(used_hinges) < int(self.max_hinges) and len(screen_features) > 0:
                for feat in screen_features:
                    xj = X[:, int(feat)]
                    thresholds = np.unique(np.quantile(xj, self.hinge_quantiles))
                    for thr in thresholds:
                        for direction in (1, -1):
                            key = (int(feat), float(thr), int(direction))
                            if key in used_hinges:
                                continue
                            col = self._hinge_column(xj, key[1], key[2])
                            denom = float(col @ col)
                            if denom <= 1e-12:
                                continue
                            gamma = float((col @ resid) / denom)
                            mse = float(np.mean((resid - gamma * col) ** 2))
                            if mse < best_mse - 1e-12:
                                best_mse = mse
                                best_term = ("hinge", key[0], key[1], key[2])

            if best_term is None:
                break

            rel_gain = (prev_mse - best_mse) / (abs(prev_mse) + 1e-12)
            if rel_gain < float(self.min_relative_gain):
                break

            extra_terms.append(best_term)
            if best_term[0] == "inter":
                used_interactions.add((int(best_term[1]), int(best_term[2])))
            else:
                used_hinges.add((int(best_term[1]), float(best_term[2]), int(best_term[3])))

            D, term_defs = self._build_design(X, extra_terms)
            intercept, coef_all = self._solve_linear_system_with_intercept(D, y, self.refit_ridge)
            pred = intercept + D @ coef_all
            prev_mse = float(np.mean((y - pred) ** 2))

        self.intercept_ = float(intercept)
        self.coef_ = np.asarray(coef_all[:n_features], dtype=float)

        extra_coef = np.asarray(coef_all[n_features:], dtype=float)
        kept_terms = []
        kept_coef = []
        for term, c in zip(extra_terms, extra_coef):
            if abs(float(c)) > float(self.coef_tol):
                kept_terms.append(term)
                kept_coef.append(float(c))

        self.hinge_terms_ = []
        self.interaction_terms_ = []
        for term, c in zip(kept_terms, kept_coef):
            if term[0] == "hinge":
                self.hinge_terms_.append(
                    {
                        "feature": int(term[1]),
                        "threshold": float(term[2]),
                        "direction": int(term[3]),
                        "coef": float(c),
                    }
                )
            else:
                self.interaction_terms_.append(
                    {"feature_i": int(term[1]), "feature_j": int(term[2]), "coef": float(c)}
                )

        importance = np.abs(self.coef_)
        for term in self.hinge_terms_:
            importance[int(term["feature"])] += abs(float(term["coef"]))
        for term in self.interaction_terms_:
            w = abs(float(term["coef"]))
            importance[int(term["feature_i"])] += w
            importance[int(term["feature_j"])] += w

        self.feature_importance_ = importance
        self.selected_features_ = sorted(int(i) for i in np.where(importance > self.coef_tol)[0])
        self.base_linear_intercept_ = float(intercept_linear)
        self.base_linear_coef_ = np.asarray(coef_linear, dtype=float)
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "coef_", "interaction_terms_", "hinge_terms_"])
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        pred = self.intercept_ + X @ self.coef_
        for term in self.interaction_terms_:
            i = int(term["feature_i"])
            j = int(term["feature_j"])
            pred += float(term["coef"]) * X[:, i] * X[:, j]
        for term in self.hinge_terms_:
            j = int(term["feature"])
            pred += float(term["coef"]) * self._hinge_column(
                X[:, j], float(term["threshold"]), int(term["direction"])
            )
        return pred

    def __str__(self):
        check_is_fitted(
            self,
            ["intercept_", "coef_", "interaction_terms_", "hinge_terms_", "feature_importance_"],
        )
        lines = [
            "Dense Spline Residual Regressor",
            "Compute prediction with this exact arithmetic:",
            f"  1) Start with y = {self.intercept_:+.6f}",
        ]

        nz_linear = [int(i) for i in np.where(np.abs(self.coef_) > self.coef_tol)[0]]
        if nz_linear:
            lines.append("  2) Add linear terms:")
            for feat_idx in sorted(nz_linear):
                lines.append(f"     y += {float(self.coef_[feat_idx]):+.6f} * x{feat_idx}")
        else:
            lines.append("  2) No linear terms.")

        if self.interaction_terms_:
            lines.append("  3) Add interaction terms:")
            for term in sorted(self.interaction_terms_, key=lambda t: abs(float(t["coef"])), reverse=True):
                i = int(term["feature_i"])
                j = int(term["feature_j"])
                lines.append(f"     y += {float(term['coef']):+.6f} * x{i} * x{j}")
        else:
            lines.append("  3) No interaction terms.")

        if self.hinge_terms_:
            lines.append("  4) Add hinge terms:")
            for term in sorted(self.hinge_terms_, key=lambda t: abs(float(t["coef"])), reverse=True):
                feat = int(term["feature"])
                threshold = float(term["threshold"])
                direction = int(term["direction"])
                coef_val = float(term["coef"])
                if direction > 0:
                    expr = f"max(0, x{feat} - {threshold:.6f})"
                else:
                    expr = f"max(0, {threshold:.6f} - x{feat})"
                lines.append(f"     y += {coef_val:+.6f} * {expr}")
        else:
            lines.append("  4) No hinge terms.")

        active = sorted(int(i) for i in np.where(self.feature_importance_ > self.coef_tol)[0])
        lines.append("  5) Features used: " + (", ".join(f"x{i}" for i in active) if active else "none"))

        if self.feature_importance_.size > 0:
            max_imp = float(np.max(self.feature_importance_))
            if max_imp > 0:
                threshold = float(self.meaningful_rel) * max_imp
                meaningful = [f"x{i}" for i in range(self.n_features_in_) if self.feature_importance_[i] >= threshold]
                lines.append(
                    "     Meaningful features (>= "
                    f"{float(self.meaningful_rel):.2f} * max importance): "
                    + (", ".join(meaningful) if meaningful else "none")
                )

        if self.n_features_in_ <= 30 and len(active) < self.n_features_in_:
            active_set = set(active)
            inactive = [f"x{i}" for i in range(self.n_features_in_) if i not in active_set]
            lines.append("     Zero-contribution features: " + ", ".join(inactive))
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
DenseSplineResidualRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "DenseSplineResidual_v1"
model_description = "Dense ridge linear backbone plus greedy residual hinge/interaction basis with exact arithmetic string form"
model_defs = [(model_shorthand_name, DenseSplineResidualRegressor())]


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
