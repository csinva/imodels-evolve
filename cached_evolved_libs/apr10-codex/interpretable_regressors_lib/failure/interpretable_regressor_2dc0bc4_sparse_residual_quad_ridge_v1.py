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


class SparseResidualQuadRidgeRegressor(BaseEstimator, RegressorMixin):
    """
    Transparent equation model:
      1) base ridge on robust-standardized raw features
      2) greedily add up to a tiny number of residual terms from:
         - centered quadratic feature shapes z_j^2
         - centered pairwise interactions z_i * z_k
      3) joint ridge refit on [linear + selected residual terms]
    """

    def __init__(
        self,
        alpha_grid_size=17,
        alpha_min_log10=-6.0,
        alpha_max_log10=3.0,
        max_extra_terms=2,
        interaction_top_features=6,
        min_relative_mse_improvement=0.02,
        display_top_k=8,
    ):
        self.alpha_grid_size = alpha_grid_size
        self.alpha_min_log10 = alpha_min_log10
        self.alpha_max_log10 = alpha_max_log10
        self.max_extra_terms = max_extra_terms
        self.interaction_top_features = interaction_top_features
        self.min_relative_mse_improvement = min_relative_mse_improvement
        self.display_top_k = display_top_k

    @staticmethod
    def _robust_standardize(X):
        center = np.median(X, axis=0)
        q25 = np.percentile(X, 25, axis=0)
        q75 = np.percentile(X, 75, axis=0)
        scale = q75 - q25
        scale = np.where(scale < 1e-8, 1.0, scale)
        Z = (X - center) / scale
        return Z, center, scale

    @staticmethod
    def _ridge_gcv(Xd, y, alpha_min_log10, alpha_max_log10, alpha_grid_size):
        n, p = Xd.shape
        if n == 0:
            return {
                "coef": np.zeros(p, dtype=float),
                "intercept": 0.0,
                "pred": np.zeros(0, dtype=float),
                "alpha": 1.0,
                "gcv": 0.0,
            }
        if p == 0:
            y_mean = float(np.mean(y))
            pred = np.full(n, y_mean, dtype=float)
            mse = float(np.mean((y - pred) ** 2))
            return {
                "coef": np.zeros(0, dtype=float),
                "intercept": y_mean,
                "pred": pred,
                "alpha": 1.0,
                "gcv": mse,
            }

        x_mean = np.mean(Xd, axis=0)
        y_mean = float(np.mean(y))
        Xc = Xd - x_mean
        yc = y - y_mean

        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        s2 = s * s
        uty = U.T @ yc
        alphas = np.logspace(
            float(alpha_min_log10),
            float(alpha_max_log10),
            num=int(max(5, alpha_grid_size)),
        )

        best = None
        for alpha in alphas:
            shrink = s / (s2 + alpha)
            coef = Vt.T @ (shrink * uty)
            intercept = y_mean - float(x_mean @ coef)
            pred = Xd @ coef + intercept
            resid = y - pred
            mse = float(np.mean(resid * resid))
            df = float(np.sum(s2 / (s2 + alpha)))
            denom = max(1e-8, 1.0 - (df / max(n, 1)))
            gcv = mse / (denom * denom)
            cur = {
                "coef": coef,
                "intercept": intercept,
                "pred": pred,
                "alpha": float(alpha),
                "gcv": gcv,
            }
            if best is None or cur["gcv"] < best["gcv"]:
                best = cur
        return best

    @staticmethod
    def _centered(arr):
        return arr - float(np.mean(arr))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape
        self.n_features_in_ = int(p)
        self.feature_names_ = [f"x{j}" for j in range(p)]

        if n == 0:
            self.center_ = np.zeros(p, dtype=float)
            self.scale_ = np.ones(p, dtype=float)
            self.coef_linear_ = np.zeros(p, dtype=float)
            self.extra_terms_ = []
            self.extra_term_means_ = []
            self.coef_extra_ = np.zeros(0, dtype=float)
            self.intercept_ = float(np.mean(y)) if y.size else 0.0
            self.alpha_ = 1.0
            self.training_mse_ = 0.0
            self.is_fitted_ = True
            return self

        Z, center, scale = self._robust_standardize(X)
        self.center_ = center
        self.scale_ = scale

        base = self._ridge_gcv(
            Z,
            y,
            self.alpha_min_log10,
            self.alpha_max_log10,
            self.alpha_grid_size,
        )
        current_pred = base["pred"]
        current_mse = float(np.mean((y - current_pred) ** 2))

        candidates = []
        candidate_values = []
        for j in range(p):
            q = self._centered(Z[:, j] * Z[:, j])
            if float(np.std(q)) > 1e-10:
                candidates.append(("quad", j))
                candidate_values.append(q)

        top_m = int(max(2, min(p, self.interaction_top_features)))
        top_idx = np.argsort(np.abs(base["coef"]))[::-1][:top_m]
        for a in range(len(top_idx)):
            i = int(top_idx[a])
            for b in range(a + 1, len(top_idx)):
                k = int(top_idx[b])
                inter = self._centered(Z[:, i] * Z[:, k])
                if float(np.std(inter)) > 1e-10:
                    candidates.append(("inter", i, k))
                    candidate_values.append(inter)

        selected_ids = []
        selected_terms = []
        selected_values = []
        max_terms = int(max(0, self.max_extra_terms))
        for _ in range(max_terms):
            residual = y - current_pred
            best_id = None
            best_score = -np.inf
            for cid, q in enumerate(candidate_values):
                if cid in selected_ids:
                    continue
                score = abs(float(np.dot(residual, q))) / (float(np.linalg.norm(q)) + 1e-12)
                if score > best_score:
                    best_score = score
                    best_id = cid

            if best_id is None:
                break

            trial_values = selected_values + [candidate_values[best_id]]
            X_extra = np.column_stack(trial_values)
            X_aug = np.hstack([Z, X_extra])
            trial = self._ridge_gcv(
                X_aug,
                y,
                self.alpha_min_log10,
                self.alpha_max_log10,
                self.alpha_grid_size,
            )
            trial_mse = float(np.mean((y - trial["pred"]) ** 2))
            rel_gain = (current_mse - trial_mse) / max(current_mse, 1e-12)
            if rel_gain < float(self.min_relative_mse_improvement):
                break

            selected_ids.append(best_id)
            selected_terms.append(candidates[best_id])
            selected_values = trial_values
            current_pred = trial["pred"]
            current_mse = trial_mse
            base = trial

        n_extra = len(selected_terms)
        self.coef_linear_ = base["coef"][:p]
        self.coef_extra_ = base["coef"][p:p + n_extra] if n_extra else np.zeros(0, dtype=float)
        self.extra_terms_ = selected_terms
        self.extra_term_means_ = []
        for t in selected_terms:
            if t[0] == "quad":
                _, j = t
                self.extra_term_means_.append(float(np.mean(Z[:, j] * Z[:, j])))
            else:
                _, i, k = t
                self.extra_term_means_.append(float(np.mean(Z[:, i] * Z[:, k])))
        self.intercept_ = float(base["intercept"])
        self.alpha_ = float(base["alpha"])
        self.training_mse_ = float(np.mean((y - current_pred) ** 2))
        self.is_fitted_ = True
        return self

    def _build_extra_features(self, Z):
        if not self.extra_terms_:
            return np.zeros((Z.shape[0], 0), dtype=float)
        cols = []
        for t, mu in zip(self.extra_terms_, self.extra_term_means_):
            if t[0] == "quad":
                _, j = t
                cols.append((Z[:, j] * Z[:, j]) - mu)
            else:
                _, i, k = t
                cols.append((Z[:, i] * Z[:, k]) - mu)
        return np.column_stack(cols)

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        X = np.asarray(X, dtype=float)
        Z = (X - self.center_) / self.scale_
        pred = Z @ self.coef_linear_
        if self.coef_extra_.size:
            X_extra = self._build_extra_features(Z)
            pred = pred + X_extra @ self.coef_extra_
        return pred + self.intercept_

    def __str__(self):
        check_is_fitted(self, "is_fitted_")
        lines = ["Sparse Residual Quad-Ridge Equation Regressor:"]
        lines.append(
            "  prediction = ridge(linear z terms + <=2 selected residual quadratic/interaction terms)"
        )
        lines.append(f"  selected alpha (GCV): {self.alpha_:.4g}")
        lines.append(f"  selected extra terms: {len(self.extra_terms_)}")
        lines.append(f"  training MSE: {self.training_mse_:.6f}")

        top_k = int(max(1, self.display_top_k))
        if self.coef_linear_.size:
            lines.append(f"  top linear terms (|coef|): {top_k}")
            order_lin = np.argsort(np.abs(self.coef_linear_))[::-1][:top_k]
            for idx in order_lin:
                lines.append(f"    {self.feature_names_[idx]}: {self.coef_linear_[idx]:+.4f}")
        if self.coef_extra_.size:
            lines.append("  extra residual terms:")
            for t, c in sorted(
                zip(self.extra_terms_, self.coef_extra_),
                key=lambda x: abs(x[1]),
                reverse=True,
            ):
                if t[0] == "quad":
                    _, j = t
                    lines.append(f"    ({self.feature_names_[j]}^2 - mean): {c:+.4f}")
                else:
                    _, i, k = t
                    lines.append(
                        f"    ({self.feature_names_[i]}*{self.feature_names_[k]} - mean): {c:+.4f}"
                    )
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparseResidualQuadRidgeRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SparseResidualQuadRidge_v1"
model_description = "Robust-standardized ridge equation with greedy addition of at most two centered residual quadratic/interaction terms and joint GCV ridge refit"
model_defs = [(model_shorthand_name, SparseResidualQuadRidgeRegressor())]


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
