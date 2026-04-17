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


class AdditiveStepRidgeRegressor(BaseEstimator, RegressorMixin):
    """Ridge linear backbone with a compact additive set of learned threshold steps."""

    def __init__(
        self,
        alpha_grid=(1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0, 100.0),
        max_rounds=10,
        max_features_per_round=10,
        threshold_quantiles=(0.15, 0.3, 0.5, 0.7, 0.85),
        val_frac=0.2,
        min_rel_gain=0.002,
        random_state=0,
    ):
        self.alpha_grid = alpha_grid
        self.max_rounds = max_rounds
        self.max_features_per_round = max_features_per_round
        self.threshold_quantiles = threshold_quantiles
        self.val_frac = val_frac
        self.min_rel_gain = min_rel_gain
        self.random_state = random_state

    @staticmethod
    def _safe_scale(X):
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
        sigma = np.where(sigma < 1e-12, 1.0, sigma)
        return mu, sigma

    def _val_split(self, n_samples):
        if n_samples <= 2:
            return np.arange(n_samples), np.arange(n_samples)
        n_val = max(1, int(round(float(self.val_frac) * n_samples)))
        n_val = min(n_val, n_samples - 1)
        rng = np.random.RandomState(self.random_state)
        perm = rng.permutation(n_samples)
        return perm[:-n_val], perm[-n_val:]

    @staticmethod
    def _rule_values(x_col, threshold, left_value, right_value):
        return np.where(x_col <= threshold, left_value, right_value)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape
        if n_features == 0:
            raise ValueError("No features provided")
        self.n_features_in_ = n_features

        self.x_mean_, self.x_scale_ = self._safe_scale(X)
        Xs = (X - self.x_mean_) / self.x_scale_

        train_idx, val_idx = self._val_split(n_samples)
        X_train, y_train = Xs[train_idx], y[train_idx]
        X_val, y_val = Xs[val_idx], y[val_idx]

        base = RidgeCV(alphas=np.asarray(self.alpha_grid, dtype=float), cv=5, fit_intercept=True)
        base.fit(X_train, y_train)

        pred_train = base.predict(X_train)
        pred_val = base.predict(X_val)
        current_val_mse = float(np.mean((y_val - pred_val) ** 2))
        initial_val_mse = current_val_mse

        rules = []
        for _ in range(max(0, int(self.max_rounds))):
            residual_train = y_train - pred_train

            centered = X_train - np.mean(X_train, axis=0, keepdims=True)
            denom = np.sqrt(np.sum(centered * centered, axis=0)) + 1e-12
            corr = np.abs((centered.T @ residual_train) / denom)
            feat_order = np.argsort(corr)[::-1]
            candidate_features = feat_order[: max(1, min(int(self.max_features_per_round), n_features))]

            best = None
            best_val_mse = current_val_mse

            for feat_idx in candidate_features:
                feat_idx = int(feat_idx)
                xtr = X_train[:, feat_idx]
                xva = X_val[:, feat_idx]

                thresholds = []
                for q in self.threshold_quantiles:
                    thresholds.append(float(np.quantile(xtr, q)))

                seen = set()
                unique_thresholds = []
                for t in thresholds:
                    key = round(t, 8)
                    if key not in seen:
                        seen.add(key)
                        unique_thresholds.append(t)

                for threshold in unique_thresholds:
                    left_mask = xtr <= threshold
                    right_mask = ~left_mask
                    if np.sum(left_mask) < 5 or np.sum(right_mask) < 5:
                        continue

                    left_value = float(np.mean(residual_train[left_mask]))
                    right_value = float(np.mean(residual_train[right_mask]))

                    delta_train = self._rule_values(xtr, threshold, left_value, right_value)
                    delta_val = self._rule_values(xva, threshold, left_value, right_value)

                    trial_pred_train = pred_train + delta_train
                    trial_pred_val = pred_val + delta_val
                    val_mse = float(np.mean((y_val - trial_pred_val) ** 2))

                    if val_mse < best_val_mse:
                        best_val_mse = val_mse
                        best = {
                            "feat_idx": feat_idx,
                            "threshold": threshold,
                            "left_value": left_value,
                            "right_value": right_value,
                            "delta_train": trial_pred_train - pred_train,
                            "delta_val": trial_pred_val - pred_val,
                        }

            if best is None:
                break

            rel_gain = (current_val_mse - best_val_mse) / max(abs(current_val_mse), 1e-12)
            if rel_gain < float(self.min_rel_gain):
                break

            pred_train = pred_train + best["delta_train"]
            pred_val = pred_val + best["delta_val"]
            current_val_mse = best_val_mse
            rules.append(
                (
                    int(best["feat_idx"]),
                    float(best["threshold"]),
                    float(best["left_value"]),
                    float(best["right_value"]),
                )
            )

        base_full = RidgeCV(alphas=np.asarray(self.alpha_grid, dtype=float), cv=5, fit_intercept=True)
        base_full.fit(Xs, y)

        self.base_alpha_ = float(base_full.alpha_)
        self.base_intercept_ = float(base_full.intercept_)
        self.base_coef_ = np.asarray(base_full.coef_, dtype=float)
        self.rules_ = rules
        self.initial_val_mse_ = float(initial_val_mse)
        self.final_val_mse_ = float(current_val_mse)

        importances = np.abs(self.base_coef_).copy()
        for feat_idx, _, lv, rv in rules:
            importances[int(feat_idx)] += abs(rv - lv)
        max_imp = float(np.max(importances)) if np.max(importances) > 0 else 0.0
        self.feature_importances_ = importances / max_imp if max_imp > 0 else importances
        return self

    def predict(self, X):
        check_is_fitted(self, ["base_intercept_", "base_coef_", "rules_", "x_mean_", "x_scale_"])
        X = np.asarray(X, dtype=float)
        Xs = (X - self.x_mean_) / self.x_scale_

        pred = self.base_intercept_ + Xs @ self.base_coef_
        for feat_idx, threshold, left_value, right_value in self.rules_:
            pred += self._rule_values(Xs[:, int(feat_idx)], threshold, left_value, right_value)
        return pred

    def __str__(self):
        check_is_fitted(
            self,
            [
                "base_intercept_",
                "base_coef_",
                "rules_",
                "x_mean_",
                "x_scale_",
                "feature_importances_",
                "base_alpha_",
                "initial_val_mse_",
                "final_val_mse_",
            ],
        )

        lines = [
            "Additive Step-Ridge Regressor",
            f"Prediction rule: y = intercept + sum_j coef_j * z_j + sum_k step_k",
            "where z_j = (x_j - mean_j) / std_j",
            f"intercept = {self.base_intercept_:+.6f}",
            f"ridge_alpha = {self.base_alpha_:.6g}",
            "",
            "Linear standardized coefficients:",
        ]

        for j, c in enumerate(self.base_coef_):
            lines.append(
                f"x{j}: coef={float(c):+.6f}, mean={float(self.x_mean_[j]):+.6f}, std={float(self.x_scale_[j]):.6f}, importance={float(self.feature_importances_[j]):.3f}"
            )

        lines.append("")
        lines.append("Additive threshold steps (applied in order):")
        if self.rules_:
            for i, (feat_idx, threshold, lv, rv) in enumerate(self.rules_, 1):
                lines.append(
                    f"step{i}: if z{feat_idx} <= {threshold:+.6f} add {lv:+.6f}, else add {rv:+.6f}"
                )
        else:
            lines.append("none")

        lines.append("")
        lines.append(
            "Simulation recipe: compute each z_j, sum intercept + linear terms, then add each step contribution."
        )
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys

_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])

# Update the model shorthand name and description below to reflect the class above and any changes you make.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
AdditiveStepRidgeRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "AdditiveStepRidge_v1"
model_description = "RidgeCV backbone on standardized features plus validation-selected additive one-feature threshold step corrections with explicit simulation recipe"
model_defs = [(model_shorthand_name, AdditiveStepRidgeRegressor())]


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
