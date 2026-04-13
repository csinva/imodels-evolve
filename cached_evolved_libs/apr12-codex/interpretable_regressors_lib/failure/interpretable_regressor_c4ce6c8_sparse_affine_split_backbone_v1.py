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


class SparseBackboneAffineSplitRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse backbone + one compact affine split correction.

    Model:
      base(x) = b0 + sum_j b_j x_j               (sparse linear)
      y_hat   = base(x) + c_side + sum_{j in S_side} d_j x_j
      where side is left/right from one threshold rule x_k <= t.
    """

    def __init__(
        self,
        max_terms_backbone=8,
        max_leaf_adjust_terms=2,
        top_split_features=8,
        split_quantiles=(0.2, 0.35, 0.5, 0.65, 0.8),
        min_leaf_frac=0.12,
        min_gain=0.008,
        complexity_penalty=0.0015,
        coef_prune_abs=1e-5,
        coef_decimals=4,
        random_state=42,
    ):
        self.max_terms_backbone = max_terms_backbone
        self.max_leaf_adjust_terms = max_leaf_adjust_terms
        self.top_split_features = top_split_features
        self.split_quantiles = split_quantiles
        self.min_leaf_frac = min_leaf_frac
        self.min_gain = min_gain
        self.complexity_penalty = complexity_penalty
        self.coef_prune_abs = coef_prune_abs
        self.coef_decimals = coef_decimals
        self.random_state = random_state

    @staticmethod
    def _ridge_closed_form(Z, y, alpha):
        p = Z.shape[1]
        reg = float(alpha) * np.eye(p)
        reg[0, 0] = 0.0
        A = Z.T @ Z + reg
        b = Z.T @ y
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(A) @ b

    def _impute(self, X):
        X = np.asarray(X, dtype=float).copy()
        bad = ~np.isfinite(X)
        if bad.any():
            X[bad] = np.take(self.feature_medians_, np.where(bad)[1])
        return X

    @staticmethod
    def _corr_abs(a, b):
        ac = a - np.mean(a)
        bc = b - np.mean(b)
        denom = (np.std(ac) + 1e-12) * (np.std(bc) + 1e-12)
        return abs(float(np.mean(ac * bc) / denom))

    def _nonlinear_score(self, xj, y):
        # Reward both linear and one-kink structure.
        return self._corr_abs(xj, y) + 0.6 * self._corr_abs(np.maximum(0.0, xj), y)

    @staticmethod
    def _to_raw_equation(beta_std, x_mean, x_std):
        coef_std = beta_std[1:]
        coef_raw = coef_std / x_std
        intercept_raw = float(beta_std[0] - np.sum((coef_std * x_mean) / x_std))
        return intercept_raw, coef_raw

    def _fit_sparse_backbone(self, X, y):
        n, p = X.shape
        tr_n = max(int(0.8 * n), min(240, n))
        rng = np.random.RandomState(self.random_state)
        idx = rng.permutation(n)
        tr_idx = idx[:tr_n]
        va_idx = idx[tr_n:] if tr_n < n else idx[:tr_n]
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xva, yva = X[va_idx], y[va_idx]

        x_mean = np.mean(Xtr, axis=0)
        x_std = np.std(Xtr, axis=0)
        x_std[x_std < 1e-12] = 1.0
        Xtr_s = (Xtr - x_mean) / x_std
        Xva_s = (Xva - x_mean) / x_std
        Ztr = np.column_stack([np.ones(Xtr_s.shape[0]), Xtr_s])
        Zva = np.column_stack([np.ones(Xva_s.shape[0]), Xva_s])

        alpha_grid = (3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0)
        best = None
        for alpha in alpha_grid:
            beta = self._ridge_closed_form(Ztr, ytr, alpha)
            itc, coef = self._to_raw_equation(beta, x_mean, x_std)
            order = np.argsort(np.abs(coef))[::-1]
            active = order[: min(int(self.max_terms_backbone), p)]
            sparse_coef = np.zeros_like(coef)
            sparse_coef[active] = coef[active]
            pred = itc + Xva @ sparse_coef
            mse = float(np.mean((yva - pred) ** 2))
            score = mse + float(self.complexity_penalty) * len(active)
            if best is None or score < best["score"]:
                best = {
                    "alpha": float(alpha),
                    "intercept": float(itc),
                    "coef": sparse_coef,
                    "active": active.astype(int),
                    "score": float(score),
                    "mse": float(mse),
                }

        # Final refit with selected alpha on all data, then hard-cap to same number of terms.
        x_mean_all = np.mean(X, axis=0)
        x_std_all = np.std(X, axis=0)
        x_std_all[x_std_all < 1e-12] = 1.0
        X_all_s = (X - x_mean_all) / x_std_all
        Zall = np.column_stack([np.ones(n), X_all_s])
        beta_all = self._ridge_closed_form(Zall, y, best["alpha"])
        itc_all, coef_all = self._to_raw_equation(beta_all, x_mean_all, x_std_all)
        order_all = np.argsort(np.abs(coef_all))[::-1]
        active_all = order_all[: len(best["active"])]
        sparse_coef_all = np.zeros_like(coef_all)
        sparse_coef_all[active_all] = coef_all[active_all]
        return {
            "alpha": best["alpha"],
            "intercept": float(itc_all),
            "coef": sparse_coef_all,
            "active": active_all.astype(int),
        }

    @staticmethod
    def _fit_leaf_adjustment(X_leaf, residual_leaf, candidate_features, max_terms):
        if X_leaf.shape[0] == 0:
            return {"offset": 0.0, "coef": np.zeros(0, dtype=float), "features": np.array([], dtype=int)}
        features = np.array(candidate_features[: max(0, int(max_terms))], dtype=int)
        if features.size == 0:
            return {"offset": float(np.mean(residual_leaf)), "coef": np.zeros(0, dtype=float), "features": features}

        Z = np.column_stack([np.ones(X_leaf.shape[0]), X_leaf[:, features]])
        beta, *_ = np.linalg.lstsq(Z, residual_leaf, rcond=None)
        return {
            "offset": float(beta[0]),
            "coef": beta[1:].astype(float),
            "features": features.astype(int),
        }

    @staticmethod
    def _apply_leaf_adjustment(X, adj):
        out = np.full(X.shape[0], float(adj["offset"]), dtype=float)
        if adj["features"].size > 0:
            out += X[:, adj["features"]] @ adj["coef"]
        return out

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_features_in_ = X.shape[1]

        self.feature_medians_ = np.nanmedian(X, axis=0)
        X = self._impute(X)

        base = self._fit_sparse_backbone(X, y)
        base_pred = base["intercept"] + X @ base["coef"]
        base_mse = float(np.mean((y - base_pred) ** 2))

        best = {"use_split": False, "score": base_mse + self.complexity_penalty * len(base["active"])}

        n = X.shape[0]
        min_leaf = max(12, int(float(self.min_leaf_frac) * n))
        nonlinear_scores = np.array([self._nonlinear_score(X[:, j], y) for j in range(self.n_features_in_)])
        split_features = np.argsort(nonlinear_scores)[::-1][: min(int(self.top_split_features), self.n_features_in_)]
        candidate_adjust_features = list(base["active"])
        if len(candidate_adjust_features) == 0:
            candidate_adjust_features = [int(np.argmax(nonlinear_scores))]

        for feat in split_features:
            thresholds = sorted(set(float(np.quantile(X[:, feat], q)) for q in self.split_quantiles))
            for thr in thresholds:
                left = X[:, feat] <= thr
                right = ~left
                n_left, n_right = int(np.sum(left)), int(np.sum(right))
                if n_left < min_leaf or n_right < min_leaf:
                    continue

                residual = y - base_pred
                left_adj = self._fit_leaf_adjustment(
                    X[left], residual[left], candidate_adjust_features, self.max_leaf_adjust_terms
                )
                right_adj = self._fit_leaf_adjustment(
                    X[right], residual[right], candidate_adjust_features, self.max_leaf_adjust_terms
                )

                pred = base_pred.copy()
                pred[left] += self._apply_leaf_adjustment(X[left], left_adj)
                pred[right] += self._apply_leaf_adjustment(X[right], right_adj)
                mse = float(np.mean((y - pred) ** 2))

                complexity = len(base["active"]) + 1
                complexity += len(left_adj["features"]) + len(right_adj["features"]) + 2  # two offsets
                score = mse + float(self.complexity_penalty) * complexity

                if score < best["score"] * (1.0 - float(self.min_gain)):
                    best = {
                        "use_split": True,
                        "score": float(score),
                        "mse": float(mse),
                        "split_feature": int(feat),
                        "split_threshold": float(thr),
                        "left_adj": left_adj,
                        "right_adj": right_adj,
                        "n_left": n_left,
                        "n_right": n_right,
                    }

        q = int(self.coef_decimals)
        self.backbone_alpha_ = float(base["alpha"])
        self.backbone_intercept_ = float(np.round(base["intercept"], q))
        backbone_coef = base["coef"].copy()
        backbone_coef[np.abs(backbone_coef) < float(self.coef_prune_abs)] = 0.0
        self.backbone_coef_ = np.round(backbone_coef, q)
        self.backbone_active_ = np.where(np.abs(self.backbone_coef_) > 0.0)[0].astype(int)

        self.use_split_ = bool(best["use_split"])
        if self.use_split_:
            self.split_feature_ = int(best["split_feature"])
            self.split_threshold_ = float(np.round(best["split_threshold"], q))
            self.left_adj_ = {
                "offset": float(np.round(best["left_adj"]["offset"], q)),
                "coef": np.round(best["left_adj"]["coef"], q),
                "features": best["left_adj"]["features"].astype(int),
            }
            self.right_adj_ = {
                "offset": float(np.round(best["right_adj"]["offset"], q)),
                "coef": np.round(best["right_adj"]["coef"], q),
                "features": best["right_adj"]["features"].astype(int),
            }
            self.leaf_fraction_ = (best["n_left"] / n, best["n_right"] / n)
        else:
            self.split_feature_ = -1
            self.split_threshold_ = 0.0
            self.left_adj_ = {"offset": 0.0, "coef": np.zeros(0), "features": np.array([], dtype=int)}
            self.right_adj_ = {"offset": 0.0, "coef": np.zeros(0), "features": np.array([], dtype=int)}
            self.leaf_fraction_ = (1.0, 0.0)

        importance = np.abs(self.backbone_coef_).copy()
        if self.use_split_:
            importance[self.left_adj_["features"]] += self.leaf_fraction_[0] * np.abs(self.left_adj_["coef"])
            importance[self.right_adj_["features"]] += self.leaf_fraction_[1] * np.abs(self.right_adj_["coef"])
            importance[self.split_feature_] += 0.5 * (abs(self.left_adj_["offset"]) + abs(self.right_adj_["offset"]))
        self.feature_importance_ = importance
        self.feature_rank_ = np.argsort(self.feature_importance_)[::-1]
        self.fitted_mse_ = float(np.mean((y - self.predict(X)) ** 2))
        return self

    def predict(self, X):
        check_is_fitted(self, ["backbone_intercept_", "backbone_coef_", "use_split_"])
        X = self._impute(X)
        pred = self.backbone_intercept_ + X @ self.backbone_coef_
        if self.use_split_:
            left = X[:, self.split_feature_] <= self.split_threshold_
            right = ~left
            pred[left] += self._apply_leaf_adjustment(X[left], self.left_adj_)
            pred[right] += self._apply_leaf_adjustment(X[right], self.right_adj_)
        return pred

    @staticmethod
    def _eq_terms(coef, start_prefix=""):
        active = np.where(np.abs(coef) > 0.0)[0]
        if active.size == 0:
            return []
        order = active[np.argsort(np.abs(coef[active]))[::-1]]
        return [f"{start_prefix}{float(coef[j]):+0.4f}*x{int(j)}" for j in order]

    @staticmethod
    def _leaf_adj_str(adj):
        terms = [f"{float(adj['offset']):+0.4f}"]
        for j, c in zip(adj["features"], adj["coef"]):
            if abs(float(c)) > 0.0:
                terms.append(f"{float(c):+0.4f}*x{int(j)}")
        return " ".join(terms)

    def __str__(self):
        check_is_fitted(self, ["backbone_intercept_", "backbone_coef_", "use_split_"])
        lines = ["SparseBackboneAffineSplitRegressor", "", "Backbone equation:"]
        base_line = f"  base(x) = {self.backbone_intercept_:+0.4f}"
        for t in self._eq_terms(self.backbone_coef_, start_prefix=" "):
            base_line += f" {t}"
        lines.append(base_line)

        if self.use_split_:
            lines.extend(
                [
                    "",
                    f"Split rule: if x{self.split_feature_} <= {self.split_threshold_:+0.4f} then LEFT else RIGHT",
                    f"LEFT adjustment:  add ({self._leaf_adj_str(self.left_adj_)})",
                    f"RIGHT adjustment: add ({self._leaf_adj_str(self.right_adj_)})",
                    "Final prediction:",
                    "  y = base(x) + side_adjustment(x)",
                    f"Leaf coverage: left={self.leaf_fraction_[0]:.1%}, right={self.leaf_fraction_[1]:.1%}",
                ]
            )
        else:
            lines.extend(["", "No split selected. Final prediction: y = base(x)"])

        lines.extend(["", "Feature importance (descending):"])
        for j in self.feature_rank_[: min(10, self.n_features_in_)]:
            lines.append(f"  x{int(j)}: {self.feature_importance_[int(j)]:.4f}")
        near_zero = [f"x{j}" for j, v in enumerate(self.feature_importance_) if v < 1e-6]
        if near_zero:
            lines.append("Near-zero effect features: " + ", ".join(near_zero))
        lines.append(f"Training MSE: {self.fitted_mse_:.6f}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparseBackboneAffineSplitRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SparseAffineSplitBackboneV1"
model_description = "Sparse ridge backbone with one threshold split and compact per-leaf affine residual adjustments over backbone features"
model_defs = [(model_shorthand_name, SparseBackboneAffineSplitRegressor())]

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------

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
