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


class RidgeResidualSymbolicRegressor(BaseEstimator, RegressorMixin):
    """
    Ridge-backed symbolic regressor:
      1) fit a strong ridge linear backbone (alpha chosen by GCV),
      2) add one residual interaction term and one residual hinge term if useful,
      3) refit all retained terms jointly and prune tiny coefficients.

    This keeps performance close to dense linear models while preserving a compact,
    arithmetic representation for human simulation.
    """

    def __init__(
        self,
        alphas=(0.0, 0.001, 0.01, 0.05, 0.2, 1.0, 5.0),
        rel_linear_cutoff=0.015,
        min_linear_terms=4,
        max_linear_terms=40,
        interaction_screen_features=8,
        max_interactions=1,
        max_hinges=1,
        hinge_quantiles=(0.2, 0.35, 0.5, 0.65, 0.8),
        min_relative_gain=0.005,
        linear_prune_rel=0.025,
        refit_ridge=1e-6,
        coef_tol=1e-8,
    ):
        self.alphas = alphas
        self.rel_linear_cutoff = rel_linear_cutoff
        self.min_linear_terms = min_linear_terms
        self.max_linear_terms = max_linear_terms
        self.interaction_screen_features = interaction_screen_features
        self.max_interactions = max_interactions
        self.max_hinges = max_hinges
        self.hinge_quantiles = hinge_quantiles
        self.min_relative_gain = min_relative_gain
        self.linear_prune_rel = linear_prune_rel
        self.refit_ridge = refit_ridge
        self.coef_tol = coef_tol

    def _fit_ridge_gcv(self, Xs, target):
        if Xs.size == 0:
            return 0.0, np.zeros(Xs.shape[1], dtype=float)

        n_samples = Xs.shape[0]
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

    def _select_linear_features(self, coef_raw):
        n_features = coef_raw.shape[0]
        if n_features == 0:
            return []

        abs_coef = np.abs(coef_raw)
        max_abs = float(abs_coef.max()) if abs_coef.size else 0.0
        if max_abs <= 1e-12:
            top_k = min(max(int(self.min_linear_terms), 1), n_features)
            return [int(i) for i in np.argsort(abs_coef)[::-1][:top_k]]

        active = np.where(abs_coef >= float(self.rel_linear_cutoff) * max_abs)[0].astype(int).tolist()
        if len(active) < int(self.min_linear_terms):
            top = np.argsort(abs_coef)[::-1][: min(int(self.min_linear_terms), n_features)]
            active = [int(i) for i in top]
        if len(active) > int(self.max_linear_terms):
            active = sorted(active, key=lambda i: abs_coef[i], reverse=True)[: int(self.max_linear_terms)]
        return sorted(set(active))

    @staticmethod
    def _hinge_column(xcol, threshold, direction):
        if direction > 0:
            return np.maximum(0.0, xcol - threshold)
        return np.maximum(0.0, threshold - xcol)

    def _design_matrix(self, X, linear_feats, interaction_defs, hinge_defs):
        cols = []
        term_defs = []

        for feat_idx in linear_feats:
            cols.append(X[:, int(feat_idx)].astype(float))
            term_defs.append(("lin", int(feat_idx)))

        for feat_i, feat_j in interaction_defs:
            cols.append((X[:, int(feat_i)] * X[:, int(feat_j)]).astype(float))
            term_defs.append(("inter", int(feat_i), int(feat_j)))

        for feat_idx, threshold, direction in hinge_defs:
            col = self._hinge_column(X[:, int(feat_idx)], float(threshold), int(direction))
            cols.append(col.astype(float))
            term_defs.append(("hinge", int(feat_idx), float(threshold), int(direction)))

        if not cols:
            return np.zeros((X.shape[0], 0), dtype=float), term_defs
        return np.column_stack(cols), term_defs

    def _solve_with_intercept(self, D, y):
        n_samples = D.shape[0]
        if D.shape[1] == 0:
            return float(np.mean(y)), np.zeros(0, dtype=float)

        A = np.column_stack([np.ones(n_samples, dtype=float), D])
        penalty = np.eye(A.shape[1], dtype=float)
        penalty[0, 0] = 0.0
        lhs = A.T @ A + max(float(self.refit_ridge), 0.0) * penalty
        rhs = A.T @ y
        try:
            coef = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        return float(coef[0]), np.asarray(coef[1:], dtype=float)

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

        coef_raw = coef_std / self.x_scale_
        linear_feats = self._select_linear_features(coef_raw)
        if not linear_feats and n_features > 0:
            linear_feats = [int(np.argmax(np.abs(coef_raw)))]

        interaction_defs = []
        hinge_defs = []

        D_lin, _ = self._design_matrix(X, linear_feats, interaction_defs, hinge_defs)
        intercept, beta = self._solve_with_intercept(D_lin, y)
        resid = y - (intercept + D_lin @ beta if D_lin.shape[1] else intercept)
        prev_mse = float(np.mean(resid ** 2))

        if int(self.max_interactions) > 0 and len(linear_feats) >= 2:
            ranked = sorted(linear_feats, key=lambda i: abs(float(coef_raw[i])), reverse=True)
            screen_feats = ranked[: min(int(self.interaction_screen_features), len(ranked))]
            best = None
            best_mse = prev_mse
            for a in range(len(screen_feats)):
                for b in range(a + 1, len(screen_feats)):
                    fi, fj = int(screen_feats[a]), int(screen_feats[b])
                    col = X[:, fi] * X[:, fj]
                    norm2 = float(col @ col)
                    if norm2 <= 1e-12:
                        continue
                    gamma = float((col @ resid) / norm2)
                    trial_resid = resid - gamma * col
                    mse = float(np.mean(trial_resid ** 2))
                    if mse < best_mse - 1e-12:
                        best_mse = mse
                        best = (fi, fj)
            if best is not None:
                rel_gain = (prev_mse - best_mse) / (abs(prev_mse) + 1e-12)
                if rel_gain >= float(self.min_relative_gain):
                    interaction_defs.append(best)
                    D_cur, _ = self._design_matrix(X, linear_feats, interaction_defs, hinge_defs)
                    intercept, beta = self._solve_with_intercept(D_cur, y)
                    resid = y - (intercept + D_cur @ beta)
                    prev_mse = float(np.mean(resid ** 2))

        if int(self.max_hinges) > 0:
            Xc = X - X.mean(axis=0, keepdims=True)
            x_norms = np.linalg.norm(Xc, axis=0) + 1e-12
            resid_norm = float(np.linalg.norm(resid)) + 1e-12
            corrs = np.abs((Xc.T @ resid) / (x_norms * resid_norm))
            top_k = min(int(self.interaction_screen_features), n_features)
            candidate_features = [int(i) for i in np.argsort(corrs)[::-1][:top_k]]

            best = None
            best_mse = prev_mse
            for feat_idx in candidate_features:
                xj = X[:, feat_idx]
                thresholds = np.unique(np.quantile(xj, self.hinge_quantiles))
                for threshold in thresholds:
                    for direction in (1, -1):
                        key = (int(feat_idx), float(threshold), int(direction))
                        h = self._hinge_column(xj, key[1], key[2])
                        norm2 = float(h @ h)
                        if norm2 <= 1e-12:
                            continue
                        gamma = float((h @ resid) / norm2)
                        trial_resid = resid - gamma * h
                        mse = float(np.mean(trial_resid ** 2))
                        if mse < best_mse - 1e-12:
                            best_mse = mse
                            best = key
            if best is not None:
                rel_gain = (prev_mse - best_mse) / (abs(prev_mse) + 1e-12)
                if rel_gain >= float(self.min_relative_gain):
                    hinge_defs.append(best)

        D, term_defs = self._design_matrix(X, linear_feats, interaction_defs, hinge_defs)
        intercept, term_coef = self._solve_with_intercept(D, y)

        keep = np.abs(term_coef) > float(self.coef_tol)
        if np.any(keep):
            scale = max(float(np.max(np.abs(term_coef[keep]))), float(self.coef_tol))
            prune_cutoff = max(float(self.linear_prune_rel) * scale, float(self.coef_tol))
            keep = np.abs(term_coef) >= prune_cutoff

            linear_positions = [k for k, t in enumerate(term_defs) if t[0] == "lin"]
            min_keep = min(int(self.min_linear_terms), len(linear_positions))
            kept_linear = [k for k in linear_positions if keep[k]]
            if len(kept_linear) < min_keep:
                top_linear = sorted(linear_positions, key=lambda k: abs(float(term_coef[k])), reverse=True)[:min_keep]
                for k in top_linear:
                    keep[k] = True

        term_defs_kept = [t for t, k in zip(term_defs, keep) if k]
        coef_kept = term_coef[keep]

        self.intercept_ = float(intercept)
        self.coef_ = np.zeros(n_features, dtype=float)
        self.interaction_terms_ = []
        self.hinge_terms_ = []

        for term, coef_val in zip(term_defs_kept, coef_kept):
            if term[0] == "lin":
                self.coef_[int(term[1])] = float(coef_val)
            elif term[0] == "inter":
                self.interaction_terms_.append(
                    {"feature_i": int(term[1]), "feature_j": int(term[2]), "coef": float(coef_val)}
                )
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
        for term in self.interaction_terms_:
            w = abs(float(term["coef"]))
            importance[int(term["feature_i"])] += w
            importance[int(term["feature_j"])] += w
        for term in self.hinge_terms_:
            importance[int(term["feature"])] += abs(float(term["coef"]))
        self.feature_importance_ = importance
        self.selected_features_ = sorted(int(i) for i in np.where(importance > self.coef_tol)[0])
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
            "Ridge Residual Symbolic Regressor",
            "Compute prediction with this exact arithmetic:",
            f"  1) Start with y = {self.intercept_:+.6f}",
        ]

        nz_linear = [int(i) for i in np.where(np.abs(self.coef_) > self.coef_tol)[0]]
        if nz_linear:
            lines.append("  2) Add linear terms:")
            for feat_idx in sorted(nz_linear, key=lambda i: abs(float(self.coef_[i])), reverse=True):
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
        if self.n_features_in_ <= 30 and len(active) < self.n_features_in_:
            inactive = [f"x{i}" for i in range(self.n_features_in_) if i not in set(active)]
            lines.append("     Zero-contribution features: " + ", ".join(inactive))
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
RidgeResidualSymbolicRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "RidgeResidualSymbolic_v1"
model_description = "Ridge backbone with adaptive linear pruning plus one residual interaction and one residual hinge correction"
model_defs = [(model_shorthand_name, RidgeResidualSymbolicRegressor())]


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
