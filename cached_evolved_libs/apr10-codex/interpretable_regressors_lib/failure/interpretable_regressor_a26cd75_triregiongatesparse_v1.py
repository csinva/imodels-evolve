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


class TriRegionGateSparseRegressor(BaseEstimator, RegressorMixin):
    """
    Compact equation model:
      1) Robust-standardize features.
      2) Keep only top-scoring linear features.
      3) Choose one gating feature and split it into 3 transparent regions.
      4) Fit one ridge equation on sparse linear terms + gate region corrections.
    """

    def __init__(
        self,
        max_linear_features=12,
        alpha_grid_size=31,
        alpha_min_log10=-6.0,
        alpha_max_log10=3.0,
        gate_q1=0.33,
        gate_q2=0.67,
        min_region_frac=0.12,
        max_display_terms=12,
    ):
        self.max_linear_features = max_linear_features
        self.alpha_grid_size = alpha_grid_size
        self.alpha_min_log10 = alpha_min_log10
        self.alpha_max_log10 = alpha_max_log10
        self.gate_q1 = gate_q1
        self.gate_q2 = gate_q2
        self.min_region_frac = min_region_frac
        self.max_display_terms = max_display_terms

    @staticmethod
    def _robust_standardize(X):
        med = np.median(X, axis=0)
        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)
        iqr = q3 - q1
        scale = np.where(iqr < 1e-8, 1.0, iqr / 1.349)
        Z = (X - med) / scale
        return Z, med, scale

    def _ridge_gcv(self, X, y):
        n, p = X.shape
        if n == 0 or p == 0:
            mean_y = float(np.mean(y)) if y.size else 0.0
            return {
                "coef": np.zeros(p, dtype=float),
                "intercept": mean_y,
                "pred": np.full(n, mean_y),
                "alpha": 1.0,
                "gcv": 0.0,
            }

        x_mean = np.mean(X, axis=0)
        y_mean = float(np.mean(y))
        Xc = X - x_mean
        yc = y - y_mean

        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        s2 = s * s
        uty = U.T @ yc

        alphas = np.logspace(
            float(self.alpha_min_log10),
            float(self.alpha_max_log10),
            num=int(max(9, self.alpha_grid_size)),
        )
        best = None
        for alpha in alphas:
            coef = Vt.T @ ((s / (s2 + alpha)) * uty)
            intercept = y_mean - float(x_mean @ coef)
            pred = X @ coef + intercept
            mse = float(np.mean((y - pred) ** 2))
            df = float(np.sum(s2 / (s2 + alpha)))
            denom = max(1e-8, 1.0 - df / max(1, n))
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

    def _select_linear_features(self, Z, y):
        y0 = y - float(np.mean(y))
        scores = np.abs(Z.T @ y0)
        k = int(min(max(1, self.max_linear_features), Z.shape[1]))
        idx = np.argsort(scores)[::-1][:k]
        idx = np.sort(idx)
        return idx, scores

    def _choose_gate_feature(self, Z, resid):
        n, p = Z.shape
        q1_frac = float(self.gate_q1)
        q2_frac = float(self.gate_q2)
        min_count = int(max(5, np.ceil(float(self.min_region_frac) * n)))

        best = None
        resid_mean = float(np.mean(resid))
        for j in range(p):
            z = Z[:, j]
            q1 = float(np.quantile(z, q1_frac))
            q2 = float(np.quantile(z, q2_frac))
            if not np.isfinite(q1) or not np.isfinite(q2) or q2 <= q1:
                continue

            low = z <= q1
            mid = (z > q1) & (z <= q2)
            high = z > q2
            n_low = int(np.sum(low))
            n_mid = int(np.sum(mid))
            n_high = int(np.sum(high))
            if n_low < min_count or n_mid < min_count or n_high < min_count:
                continue

            m_low = float(np.mean(resid[low]))
            m_mid = float(np.mean(resid[mid]))
            m_high = float(np.mean(resid[high]))
            score = (
                n_low * (m_low - resid_mean) ** 2
                + n_mid * (m_mid - resid_mean) ** 2
                + n_high * (m_high - resid_mean) ** 2
            ) / max(1, n)

            cur = {
                "feature": int(j),
                "q1": q1,
                "q2": q2,
                "score": float(score),
            }
            if best is None or cur["score"] > best["score"]:
                best = cur

        if best is None:
            j = int(np.argmax(np.var(Z, axis=0)))
            z = Z[:, j]
            q1 = float(np.quantile(z, q1_frac))
            q2 = float(np.quantile(z, q2_frac))
            if q2 <= q1:
                q2 = q1 + 1e-3
            best = {"feature": j, "q1": q1, "q2": q2, "score": 0.0}
        return best

    def _gate_basis(self, z, q1, q2):
        ind_low = (z <= q1).astype(float)
        ind_high = (z > q2).astype(float)
        hinge_low = np.maximum(0.0, q1 - z)
        hinge_high = np.maximum(0.0, z - q2)

        means = {
            "ind_low": float(np.mean(ind_low)),
            "ind_high": float(np.mean(ind_high)),
            "hinge_low": float(np.mean(hinge_low)),
            "hinge_high": float(np.mean(hinge_high)),
        }

        G = np.column_stack([
            ind_low - means["ind_low"],
            ind_high - means["ind_high"],
            hinge_low - means["hinge_low"],
            hinge_high - means["hinge_high"],
        ])
        return G, means

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape

        self.n_features_in_ = int(p)
        self.feature_names_ = [f"x{j}" for j in range(p)]

        if n == 0:
            self.center_ = np.zeros(p, dtype=float)
            self.scale_ = np.ones(p, dtype=float)
            self.linear_idx_ = np.array([], dtype=int)
            self.gate_feature_ = 0
            self.gate_q1_ = 0.0
            self.gate_q2_ = 1.0
            self.gate_means_ = {"ind_low": 0.0, "ind_high": 0.0, "hinge_low": 0.0, "hinge_high": 0.0}
            self.linear_coef_z_ = np.zeros(0, dtype=float)
            self.gate_coef_ = np.zeros(4, dtype=float)
            self.design_coef_ = np.zeros(4, dtype=float)
            self.intercept_ = float(np.mean(y)) if y.size else 0.0
            self.alpha_ = 1.0
            self.training_mse_ = 0.0
            self.is_fitted_ = True
            return self

        Z, center, scale = self._robust_standardize(X)
        self.center_ = center
        self.scale_ = scale

        linear_idx, linear_scores = self._select_linear_features(Z, y)
        Z_lin = Z[:, linear_idx]
        linear_fit = self._ridge_gcv(Z_lin, y)
        resid = y - linear_fit["pred"]

        gate = self._choose_gate_feature(Z, resid)
        gate_feature = gate["feature"]
        gate_q1 = gate["q1"]
        gate_q2 = gate["q2"]

        G, gate_means = self._gate_basis(Z[:, gate_feature], gate_q1, gate_q2)
        design = np.hstack([Z_lin, G])
        final = self._ridge_gcv(design, y)

        n_lin = Z_lin.shape[1]
        self.linear_idx_ = linear_idx
        self.linear_scores_ = linear_scores
        self.linear_coef_z_ = final["coef"][:n_lin].copy()
        self.gate_coef_ = final["coef"][n_lin:].copy()
        self.design_coef_ = final["coef"].copy()

        self.gate_feature_ = int(gate_feature)
        self.gate_q1_ = float(gate_q1)
        self.gate_q2_ = float(gate_q2)
        self.gate_score_ = float(gate["score"])
        self.gate_means_ = gate_means

        self.alpha_ = float(final["alpha"])
        self.intercept_ = float(final["intercept"])
        self.training_mse_ = float(np.mean((y - final["pred"]) ** 2))

        self.coef_ = np.zeros(p, dtype=float)
        self.coef_[linear_idx] = self.linear_coef_z_ / self.scale_[linear_idx]

        self.is_fitted_ = True
        return self

    def _build_design(self, X):
        Z = (X - self.center_) / self.scale_
        Z_lin = Z[:, self.linear_idx_]

        z_gate = Z[:, self.gate_feature_]
        ind_low = (z_gate <= self.gate_q1_).astype(float) - self.gate_means_["ind_low"]
        ind_high = (z_gate > self.gate_q2_).astype(float) - self.gate_means_["ind_high"]
        hinge_low = np.maximum(0.0, self.gate_q1_ - z_gate) - self.gate_means_["hinge_low"]
        hinge_high = np.maximum(0.0, z_gate - self.gate_q2_) - self.gate_means_["hinge_high"]
        G = np.column_stack([ind_low, ind_high, hinge_low, hinge_high])
        return np.hstack([Z_lin, G])

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        X = np.asarray(X, dtype=float)
        design = self._build_design(X)
        return design @ self.design_coef_ + self.intercept_

    def __str__(self):
        check_is_fitted(self, "is_fitted_")
        lines = ["Tri-Region Gated Sparse Equation Regressor:"]
        lines.append("  prediction = sparse_linear_terms + one gated feature correction")
        lines.append(f"  chosen alpha: {self.alpha_:.4g}")
        lines.append(f"  training MSE: {self.training_mse_:.6f}")
        lines.append(f"  intercept: {self.intercept_:+.6f}")

        lines.append(
            f"  gate feature: x{self.gate_feature_} with z-thresholds q1={self.gate_q1_:+.4f}, q2={self.gate_q2_:+.4f}"
        )
        lines.append(
            "  gate correction terms: "
            f"w_low*1[z<=q1] + w_high*1[z>q2] + w_hl*max(0,q1-z) + w_hh*max(0,z-q2)"
        )
        lines.append(
            f"    w_low={self.gate_coef_[0]:+.6f}, w_high={self.gate_coef_[1]:+.6f}, "
            f"w_hl={self.gate_coef_[2]:+.6f}, w_hh={self.gate_coef_[3]:+.6f}"
        )

        max_terms = int(max(1, self.max_display_terms))
        ranked = np.argsort(np.abs(self.linear_coef_z_))[::-1][:max_terms]
        lines.append("  sparse linear terms (z-scored features):")
        for ridx in ranked:
            j = int(self.linear_idx_[ridx])
            lines.append(f"    x{j}: coef_z={self.linear_coef_z_[ridx]:+.6f}")

        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
TriRegionGateSparseRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "TriRegionGateSparse_v1"
model_description = "Sparse linear equation on top features plus one transparent tri-region gate correction on a selected feature, fit by custom GCV ridge"
model_defs = [(model_shorthand_name, TriRegionGateSparseRegressor())]


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
