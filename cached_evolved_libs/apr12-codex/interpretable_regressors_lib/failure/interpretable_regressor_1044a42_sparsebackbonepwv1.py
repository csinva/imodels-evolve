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
from sklearn.linear_model import RidgeCV
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class SparseBackbonePiecewiseRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse linear backbone + compact piecewise correction on one feature:
    y = b + sum_j w_j x_j + a1*max(0, xk - t1) + a2*max(0, xk - t2)
    """

    def __init__(
        self,
        max_linear_terms=8,
        candidate_features=8,
        hinge_quantiles=(0.15, 0.30, 0.50, 0.70, 0.85),
        complexity_penalty=5e-3,
        min_gain=1e-3,
    ):
        self.max_linear_terms = max_linear_terms
        self.candidate_features = candidate_features
        self.hinge_quantiles = hinge_quantiles
        self.complexity_penalty = complexity_penalty
        self.min_gain = min_gain

    def _impute(self, X):
        X = np.asarray(X, dtype=float).copy()
        nan_mask = ~np.isfinite(X)
        if nan_mask.any():
            X[nan_mask] = np.take(self.feature_medians_, np.where(nan_mask)[1])
        return X

    def _standardize(self, X):
        Xs = (X - self.x_mean_) / self.x_scale_
        return Xs

    def _score_feature_for_nonlinearity(self, xj, residual):
        x0 = xj - np.mean(xj)
        r0 = residual - np.mean(residual)
        denom = (np.std(x0) + 1e-12) * (np.std(r0) + 1e-12)
        c1 = abs(float(np.mean(x0 * r0) / denom))
        x2 = x0 * x0
        x2 = x2 - np.mean(x2)
        denom2 = (np.std(x2) + 1e-12) * (np.std(r0) + 1e-12)
        c2 = abs(float(np.mean(x2 * r0) / denom2))
        return c1 + 0.9 * c2

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.feature_medians_ = np.nanmedian(X, axis=0)
        X = self._impute(X)

        self.x_mean_ = np.mean(X, axis=0)
        self.x_scale_ = np.std(X, axis=0)
        self.x_scale_[self.x_scale_ < 1e-12] = 1.0
        Xs = self._standardize(X)

        ridge = RidgeCV(alphas=(0.03, 0.1, 0.3, 1.0, 3.0, 10.0), cv=3)
        ridge.fit(Xs, y)
        coef_full = ridge.coef_ / self.x_scale_
        intercept_full = float(ridge.intercept_ - np.dot(coef_full, self.x_mean_))

        active = np.where(np.abs(coef_full) > 1e-8)[0]
        if active.size == 0:
            active = np.array([int(np.argmax(np.abs(coef_full)))], dtype=int)
        if active.size > self.max_linear_terms:
            keep = np.argsort(np.abs(coef_full[active]))[::-1][: self.max_linear_terms]
            active = active[keep]
        self.active_linear_features_ = np.array(sorted(active.tolist()), dtype=int)

        Xa = X[:, self.active_linear_features_]
        Xa_design = np.column_stack([np.ones(X.shape[0]), Xa])
        coef_a = np.linalg.lstsq(Xa_design, y, rcond=None)[0]
        self.intercept_ = float(coef_a[0])
        self.linear_coef_ = np.zeros(self.n_features_in_, dtype=float)
        self.linear_coef_[self.active_linear_features_] = coef_a[1:]

        y_base = self.intercept_ + X @ self.linear_coef_
        residual = y - y_base
        base_mse = float(np.mean(residual ** 2))

        scores = np.array(
            [self._score_feature_for_nonlinearity(X[:, j], residual) for j in range(self.n_features_in_)],
            dtype=float,
        )
        ranked = np.argsort(scores)[::-1]
        pool = ranked[: min(self.candidate_features, self.n_features_in_)].tolist()
        for j in self.active_linear_features_:
            j = int(j)
            if j not in pool:
                pool.append(j)

        best = {
            "objective": base_mse + self.complexity_penalty * len(self.active_linear_features_),
            "mse": base_mse,
            "hinges": [],
            "coef_aug": None,
        }

        for j in pool:
            xj = X[:, int(j)]
            thresholds = np.unique(np.quantile(xj, self.hinge_quantiles))
            bases = []
            for thr in thresholds:
                bases.append((f"h_pos_{thr:.6f}", np.maximum(0.0, xj - float(thr)), float(thr), "pos"))
                bases.append((f"h_neg_{thr:.6f}", np.maximum(0.0, float(thr) - xj), float(thr), "neg"))

            # single hinge
            for _, z1, t1, kind1 in bases:
                X_aug = np.column_stack([Xa_design, z1])
                coef_aug = np.linalg.lstsq(X_aug, y, rcond=None)[0]
                pred = X_aug @ coef_aug
                mse = float(np.mean((y - pred) ** 2))
                hinges = [{"j": int(j), "threshold": float(t1), "kind": kind1, "coef_idx": X_aug.shape[1] - 1}]
                objective = mse + self.complexity_penalty * (len(self.active_linear_features_) + len(hinges))
                if objective < best["objective"] and (base_mse - mse) >= self.min_gain:
                    best = {"objective": objective, "mse": mse, "hinges": hinges, "coef_aug": coef_aug}

            # two hinges on the same feature
            for i in range(len(bases)):
                for k in range(i + 1, len(bases)):
                    _, z1, t1, kind1 = bases[i]
                    _, z2, t2, kind2 = bases[k]
                    if abs(t1 - t2) < 1e-8 and kind1 == kind2:
                        continue
                    X_aug = np.column_stack([Xa_design, z1, z2])
                    coef_aug = np.linalg.lstsq(X_aug, y, rcond=None)[0]
                    pred = X_aug @ coef_aug
                    mse = float(np.mean((y - pred) ** 2))
                    hinges = [
                        {"j": int(j), "threshold": float(t1), "kind": kind1, "coef_idx": X_aug.shape[1] - 2},
                        {"j": int(j), "threshold": float(t2), "kind": kind2, "coef_idx": X_aug.shape[1] - 1},
                    ]
                    objective = mse + self.complexity_penalty * (len(self.active_linear_features_) + len(hinges))
                    if objective < best["objective"] and (base_mse - mse) >= self.min_gain:
                        best = {"objective": objective, "mse": mse, "hinges": hinges, "coef_aug": coef_aug}

        self.hinge_terms_ = []
        if best["coef_aug"] is not None:
            coef_aug = best["coef_aug"]
            self.intercept_ = float(coef_aug[0])
            self.linear_coef_ = np.zeros(self.n_features_in_, dtype=float)
            self.linear_coef_[self.active_linear_features_] = coef_aug[1 : 1 + len(self.active_linear_features_)]
            for h in best["hinges"]:
                c = float(coef_aug[h["coef_idx"]])
                if abs(c) > 1e-8:
                    self.hinge_terms_.append(
                        {
                            "j": int(h["j"]),
                            "threshold": float(h["threshold"]),
                            "kind": h["kind"],
                            "coef": c,
                        }
                    )

        fi = np.abs(self.linear_coef_).copy()
        for h in self.hinge_terms_:
            fi[h["j"]] += abs(float(h["coef"]))
        self.feature_importance_ = fi
        return self

    def _eval_hinge(self, X, term):
        xj = X[:, term["j"]]
        if term["kind"] == "pos":
            return np.maximum(0.0, xj - term["threshold"])
        return np.maximum(0.0, term["threshold"] - xj)

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "linear_coef_", "feature_importance_"])
        X = self._impute(X)
        y_hat = self.intercept_ + X @ self.linear_coef_
        for h in self.hinge_terms_:
            y_hat += h["coef"] * self._eval_hinge(X, h)
        return y_hat

    def __str__(self):
        check_is_fitted(self, ["intercept_", "linear_coef_", "feature_importance_"])
        lines = [
            "SparseBackbonePiecewiseRegressor",
            "Single explicit additive equation for prediction:",
            f"y = {self.intercept_:+.4f}",
        ]
        linear_terms = sorted(
            [(int(j), float(self.linear_coef_[j])) for j in self.active_linear_features_],
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        for j, c in linear_terms:
            lines.append(f"  {c:+.4f} * x{j}")
        for h in sorted(self.hinge_terms_, key=lambda t: abs(float(t["coef"])), reverse=True):
            if h["kind"] == "pos":
                basis = f"max(0, x{h['j']} - {h['threshold']:.3f})"
            else:
                basis = f"max(0, {h['threshold']:.3f} - x{h['j']})"
            lines.append(f"  {float(h['coef']):+.4f} * {basis}")

        lines.append("")
        lines.append("All non-listed linear feature coefficients are 0.")
        lines.append(f"Total active terms: {len(linear_terms) + len(self.hinge_terms_)}")
        lines.append("Feature importance (sum of absolute term coefficients):")
        rank = np.argsort(self.feature_importance_)[::-1]
        for j in rank:
            lines.append(f"  x{j}: {self.feature_importance_[j]:.4f}")
        near_zero = [f"x{j}" for j, v in enumerate(self.feature_importance_) if v < 1e-4]
        if near_zero:
            lines.append("Features with near-zero effect: " + ", ".join(near_zero))
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparseBackbonePiecewiseRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SparseBackbonePWV1"
model_description = "Sparse ridge-backed linear equation with an optional jointly-refit one-feature two-hinge piecewise correction"
model_defs = [(model_shorthand_name, SparseBackbonePiecewiseRegressor())]


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
