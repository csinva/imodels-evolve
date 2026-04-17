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


class SparseLinearRuleBoostRegressor(BaseEstimator, RegressorMixin):
    """Sparse linear backbone plus a tiny learned residual stump ensemble."""

    def __init__(
        self,
        max_linear_features=8,
        n_stumps=6,
        learning_rate=0.7,
        threshold_quantiles=(0.1, 0.25, 0.5, 0.75, 0.9),
        min_variance=1e-12,
        random_state=0,
    ):
        self.max_linear_features = max_linear_features
        self.n_stumps = n_stumps
        self.learning_rate = learning_rate
        self.threshold_quantiles = threshold_quantiles
        self.min_variance = min_variance
        self.random_state = random_state

    @staticmethod
    def _safe_standardize(X):
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
        sigma = np.where(sigma > 1e-12, sigma, 1.0)
        return (X - mu) / sigma, mu, sigma

    @staticmethod
    def _ols_fit(X, y):
        if X.shape[1] == 0:
            return float(np.mean(y)), np.zeros(0, dtype=float)
        x_mu = np.mean(X, axis=0)
        Xc = X - x_mu
        y_mu = float(np.mean(y))
        yc = y - y_mu
        try:
            coef, *_ = np.linalg.lstsq(Xc, yc, rcond=None)
        except np.linalg.LinAlgError:
            coef = np.linalg.pinv(Xc) @ yc
        intercept = float(y_mu - np.dot(x_mu, coef))
        return intercept, coef.astype(float)

    def _select_linear_features(self, Xs, y):
        p = Xs.shape[1]
        max_k = max(1, min(int(self.max_linear_features), p))
        y_center = y - float(np.mean(y))
        scores = np.abs(Xs.T @ y_center)
        order = np.argsort(scores)[::-1]
        return np.array(order[:max_k], dtype=int)

    def _fit_best_stump(self, Xs, residual):
        n, p = Xs.shape
        best = None
        best_sse = np.inf

        for j in range(p):
            xj = Xs[:, j]
            if np.var(xj) <= float(self.min_variance):
                continue

            thresholds = []
            for q in self.threshold_quantiles:
                t = float(np.quantile(xj, q))
                thresholds.append(t)
            thresholds = sorted(set(thresholds))

            for t in thresholds:
                left = xj <= t
                right = ~left
                n_left = int(np.sum(left))
                n_right = n - n_left
                if n_left < 5 or n_right < 5:
                    continue

                left_val = float(np.mean(residual[left]))
                right_val = float(np.mean(residual[right]))

                pred = np.where(left, left_val, right_val)
                sse = float(np.sum((residual - pred) ** 2))
                if sse < best_sse:
                    best_sse = sse
                    best = {
                        "feature": int(j),
                        "threshold": float(t),
                        "left_val": left_val,
                        "right_val": right_val,
                    }

        return best, best_sse

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y have incompatible shapes")
        if X.shape[1] == 0:
            raise ValueError("No features provided")

        self.n_features_in_ = X.shape[1]
        Xs, self.x_mu_, self.x_sigma_ = self._safe_standardize(X)

        self.active_linear_features_ = self._select_linear_features(Xs, y)
        X_lin = Xs[:, self.active_linear_features_]
        self.intercept_, self.linear_coef_ = self._ols_fit(X_lin, y)

        pred = self.intercept_ + X_lin @ self.linear_coef_
        residual = y - pred

        self.stumps_ = []
        var_y = float(np.var(y)) + 1e-12
        for _ in range(max(0, int(self.n_stumps))):
            stump, sse_before = self._fit_best_stump(Xs, residual)
            if stump is None:
                break

            xj = Xs[:, stump["feature"]]
            contrib = np.where(xj <= stump["threshold"], stump["left_val"], stump["right_val"])
            updated_residual = residual - float(self.learning_rate) * contrib

            gain = float(np.mean(residual ** 2) - np.mean(updated_residual ** 2))
            if gain <= 1e-5 * var_y:
                break

            self.stumps_.append(stump)
            residual = updated_residual

        feat_imp = np.zeros(self.n_features_in_, dtype=float)
        for j, c in zip(self.active_linear_features_, self.linear_coef_):
            feat_imp[int(j)] += abs(float(c))
        for stump in self.stumps_:
            mag = abs(float(stump["right_val"]) - float(stump["left_val"]))
            feat_imp[int(stump["feature"])] += mag

        max_imp = float(np.max(feat_imp)) if feat_imp.size else 0.0
        self.feature_importances_ = feat_imp / max_imp if max_imp > 0 else feat_imp
        self.active_features_ = np.flatnonzero(feat_imp > 0).astype(int)
        return self

    def predict(self, X):
        check_is_fitted(
            self,
            [
                "intercept_",
                "linear_coef_",
                "active_linear_features_",
                "stumps_",
                "x_mu_",
                "x_sigma_",
            ],
        )

        X = np.asarray(X, dtype=float)
        Xs = (X - self.x_mu_) / self.x_sigma_

        pred = np.full(X.shape[0], float(self.intercept_), dtype=float)
        if self.active_linear_features_.size:
            pred += Xs[:, self.active_linear_features_] @ self.linear_coef_

        for stump in self.stumps_:
            xj = Xs[:, int(stump["feature"])]
            contrib = np.where(xj <= float(stump["threshold"]), float(stump["left_val"]), float(stump["right_val"]))
            pred += float(self.learning_rate) * contrib

        return pred

    def __str__(self):
        check_is_fitted(
            self,
            [
                "intercept_",
                "linear_coef_",
                "active_linear_features_",
                "stumps_",
                "feature_importances_",
                "active_features_",
                "x_mu_",
                "x_sigma_",
            ],
        )

        lines = ["Sparse Linear + Rule Boost Regressor"]
        lines.append("Exact prediction recipe (use raw input features xj):")
        lines.append(f"1) Start with y = {float(self.intercept_):+.6f}")

        lines.append("2) Add sparse linear standardized terms:")
        if self.active_linear_features_.size == 0:
            lines.append("   (no linear terms)")
        else:
            for j, c in zip(self.active_linear_features_, self.linear_coef_):
                mu = float(self.x_mu_[j])
                sigma = float(self.x_sigma_[j])
                lines.append(
                    f"   y += {float(c):+.6f} * ((x{int(j)} - {mu:+.6f}) / {sigma:.6f})"
                )

        lines.append(f"3) Add residual stump contributions (learning_rate={float(self.learning_rate):.3f}):")
        if not self.stumps_:
            lines.append("   (no stump terms)")
        else:
            for k, stump in enumerate(self.stumps_, 1):
                j = int(stump["feature"])
                t = float(stump["threshold"])
                lv = float(stump["left_val"])
                rv = float(stump["right_val"])
                mu = float(self.x_mu_[j])
                sigma = float(self.x_sigma_[j])
                lines.append(
                    f"   Rule {k}: z = (x{j} - {mu:+.6f}) / {sigma:.6f}; "
                    f"if z <= {t:+.6f} add {float(self.learning_rate) * lv:+.6f}, else add {float(self.learning_rate) * rv:+.6f}"
                )

        lines.append("")
        lines.append(
            "Active features used by model: "
            + (", ".join(f"x{int(j)}" for j in self.active_features_) if self.active_features_.size else "none")
        )
        lines.append(
            "Most important features (normalized): "
            + ", ".join(
                f"x{int(i)}={float(self.feature_importances_[i]):.3f}"
                for i in np.argsort(self.feature_importances_)[::-1][: min(8, len(self.feature_importances_))]
                if self.feature_importances_[i] > 0
            )
        )
        lines.append(
            f"Model size: {len(self.active_linear_features_)} linear terms + {len(self.stumps_)} rules"
        )
        lines.append("To simulate a sample: follow steps 1->2->3 in order and sum all additions.")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys

_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])


# Update the model shorthand name and description below to reflect the class above and any changes you make.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
SparseLinearRuleBoostRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "SparseLinearRuleBoost_v1"
model_description = "From-scratch sparse standardized linear backbone plus tiny residual stump boosting with explicit step-by-step simulation rules"
model_defs = [(model_shorthand_name, SparseLinearRuleBoostRegressor())]
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

    # --- Recompute global rank summary from updated performance_results.csv ---
    # Build dataset -> {model: rmse}
    perf_table = defaultdict(dict)
    with open(perf_csv, newline="") as f:
        for row in csv.DictReader(f):
            ds = row["dataset"]
            m = row["model"]
            rmse_s = row.get("rmse", "")
            if rmse_s in ("", None):
                perf_table[ds][m] = float("nan")
            else:
                try:
                    perf_table[ds][m] = float(rmse_s)
                except ValueError:
                    perf_table[ds][m] = float("nan")

    avg_rank, _ = compute_rank_scores(perf_table)
    mean_rank = avg_rank.get(model_name, float("nan"))

    # --- Upsert overall_results.csv ---
    overall_rows = [{
        "commit": git_hash,
        "mean_rank": f"{mean_rank:.2f}" if np.isfinite(mean_rank) else "",
        "frac_interpretability_tests_passed": f"{(n_passed / total):.4f}" if total else "",
        "status": "",  # fill manually after reviewing
        "model_name": model_name,
        "description": model_description,
    }]
    upsert_overall_results(overall_rows, RESULTS_DIR)

    # --- Plot update ---
    overall_csv = os.path.join(RESULTS_DIR, "overall_results.csv")
    plot_interp_vs_performance(
        overall_csv,
        os.path.join(RESULTS_DIR, "interpretability_vs_performance.png"),
    )

    # Print compact summary
    std_names = {t.__name__ for t in ALL_TESTS}
    hard_names = {t.__name__ for t in HARD_TESTS}
    ins_names = {t.__name__ for t in INSIGHT_TESTS}
    n_std = sum(r["passed"] for r in interp_results if r["test"] in std_names)
    n_hard = sum(r["passed"] for r in interp_results if r["test"] in hard_names)
    n_ins = sum(r["passed"] for r in interp_results if r["test"] in ins_names)

    print("\n---")
    print(f"tests_passed:  {n_passed}/{total} ({(n_passed/total):.2%})  "
          f"[std {n_std}/{len(std_names)}  hard {n_hard}/{len(hard_names)}  insight {n_ins}/{len(ins_names)}]")
    print(f"total_seconds: {time.time() - t0:.1f}s")
