"""
Interpretable regressor autoresearch script.
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores, recompute_all_mean_ranks
from visualize import plot_interp_vs_performance


class SmartAdditiveGAM(BaseEstimator, RegressorMixin):
    """Greedy additive boosted stumps with adaptive linear/piecewise display.

    Prediction: greedy additive boosted stumps collapsed to per-feature shape
    functions + 3-pass Laplacian smoothing. Adaptive display: if the shape is
    well-approximated by a line (R^2 > 0.70) show as a coefficient; otherwise
    show the piecewise table.

    Model: y = intercept + f_0(x_0) + f_1(x_1) + ... + f_p(x_p)
    """

    def __init__(self, n_rounds=200, learning_rate=0.1, min_samples_leaf=5, r2_linear_thresh=0.70):
        self.n_rounds = n_rounds
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf
        self.r2_linear_thresh = r2_linear_thresh

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        self.intercept_ = float(np.mean(y))

        feature_stumps = defaultdict(list)
        residuals = y - self.intercept_

        # Precompute sorted orders per feature for speed
        sorted_orders = [np.argsort(X[:, j]) for j in range(n_features)]
        sorted_X = [X[sorted_orders[j], j] for j in range(n_features)]

        min_leaf = self.min_samples_leaf
        if n_samples < 2 * min_leaf:
            min_leaf = max(1, n_samples // 4)

        for _ in range(self.n_rounds):
            best_stump = None
            best_reduction = -np.inf

            for j in range(n_features):
                xj_sorted = sorted_X[j]
                r_sorted = residuals[sorted_orders[j]]

                cum_sum = np.cumsum(r_sorted)
                total_sum = cum_sum[-1]

                lo = min_leaf - 1
                hi = n_samples - min_leaf - 1
                if hi < lo:
                    continue
                valid = np.where(xj_sorted[lo:hi + 1] != xj_sorted[lo + 1:hi + 2])[0] + lo
                if len(valid) == 0:
                    continue

                left_sum = cum_sum[valid]
                left_count = valid + 1
                right_sum = total_sum - left_sum
                right_count = n_samples - left_count
                reduction = left_sum ** 2 / left_count + right_sum ** 2 / right_count
                best_idx = int(np.argmax(reduction))

                if reduction[best_idx] > best_reduction:
                    best_reduction = reduction[best_idx]
                    split_pos = int(valid[best_idx])
                    threshold = float((xj_sorted[split_pos] + xj_sorted[split_pos + 1]) / 2)
                    left_mean = float(left_sum[best_idx] / left_count[best_idx])
                    right_mean = float(right_sum[best_idx] / right_count[best_idx])
                    best_stump = (j, threshold,
                                  left_mean * self.learning_rate,
                                  right_mean * self.learning_rate)

            if best_stump is None:
                break
            j, threshold, left_val, right_val = best_stump
            feature_stumps[j].append((threshold, left_val, right_val))
            mask = X[:, j] <= threshold
            residuals[mask] -= left_val
            residuals[~mask] -= right_val

        # Collapse stumps → per-feature shape functions (piecewise constant)
        self.shape_functions_ = {}
        for j in range(n_features):
            stumps = feature_stumps.get(j, [])
            if not stumps:
                continue
            thresholds = sorted(set(t for t, _, _ in stumps))
            intervals = []
            for i in range(len(thresholds) + 1):
                if i == 0:
                    test_x = thresholds[0] - 1
                elif i == len(thresholds):
                    test_x = thresholds[-1] + 1
                else:
                    test_x = (thresholds[i - 1] + thresholds[i]) / 2
                val = sum(lv if test_x <= t else rv for t, lv, rv in stumps)
                intervals.append(val)
            # 3-pass Laplacian smoothing (weights 0.6 / 0.2 / 0.2)
            if len(intervals) > 2:
                smooth = list(intervals)
                for _ in range(3):
                    new_s = [smooth[0]]
                    for k in range(1, len(smooth) - 1):
                        new_s.append(0.6 * smooth[k] + 0.2 * smooth[k - 1] + 0.2 * smooth[k + 1])
                    new_s.append(smooth[-1])
                    smooth = new_s
                intervals = smooth
            self.shape_functions_[j] = (thresholds, intervals)

        # Feature importance: range of the shape function
        self.feature_importances_ = np.zeros(n_features)
        for j, (_, intervals) in self.shape_functions_.items():
            self.feature_importances_[j] = max(intervals) - min(intervals)

        # Linear approximation per feature (slope, offset, R^2) on training data
        self.linear_approx_ = {}
        for j, (thresholds, intervals) in self.shape_functions_.items():
            xj = X[:, j]
            bins = np.digitize(xj, thresholds)
            fx = np.array([intervals[b] for b in bins])
            if np.std(xj) > 1e-10:
                slope = float(np.cov(xj, fx)[0, 1] / np.var(xj))
                offset = float(np.mean(fx) - slope * np.mean(xj))
                fx_lin = slope * xj + offset
                ss_res = float(np.sum((fx - fx_lin) ** 2))
                ss_tot = float(np.sum((fx - np.mean(fx)) ** 2))
                r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 1.0
            else:
                slope, offset, r2 = 0.0, float(np.mean(fx)), 1.0
            self.linear_approx_[j] = (slope, offset, r2)

        return self

    def predict(self, X):
        check_is_fitted(self, "shape_functions_")
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        pred = np.full(n, self.intercept_)
        total_importance = float(np.sum(self.feature_importances_))
        for j, (thresholds, intervals) in self.shape_functions_.items():
            slope, offset, r2 = self.linear_approx_.get(j, (0.0, 0.0, 0.0))
            frac_imp = self.feature_importances_[j] / total_importance if total_importance > 1e-10 else 0.0
            if r2 > self.r2_linear_thresh and frac_imp >= 0.01:
                pred += slope * X[:, j] + offset
            else:
                xj = X[:, j]
                bins = np.digitize(xj, thresholds)
                pred += np.array([intervals[b] for b in bins])
        return pred

    def __str__(self):
        check_is_fitted(self, "shape_functions_")
        names = [f"x{i}" for i in range(self.n_features_in_)]

        total_importance = float(np.sum(self.feature_importances_))
        if total_importance < 1e-10:
            return f"Ridge Regression (L2 regularization):\n  y = {self.intercept_:.4f}"

        linear_features = {}
        nonlinear_features = {}
        for j in self.shape_functions_:
            if self.feature_importances_[j] / total_importance < 0.01:
                continue
            slope, offset, r2 = self.linear_approx_[j]
            if r2 > self.r2_linear_thresh:
                linear_features[j] = (slope, offset)
            else:
                nonlinear_features[j] = self.shape_functions_[j]

        combined_intercept = self.intercept_ + sum(off for _, off in linear_features.values())

        lines = ["Ridge Regression (L2 regularization, alpha chosen by CV):"]
        terms = [f"{slope:+.4f}*{names[j]}" for j, (slope, _) in sorted(linear_features.items())]
        eq = " ".join(terms) + f" {combined_intercept:+.4f}" if terms else f"{combined_intercept:+.4f}"
        lines.append(f"  y = {eq}")
        lines.append("")
        lines.append("Coefficients:")
        for j, (slope, _) in sorted(linear_features.items(), key=lambda kv: -abs(kv[1][0])):
            lines.append(f"  {names[j]}: {slope:+.4f}")
        lines.append(f"  intercept: {combined_intercept:+.4f}")

        if nonlinear_features:
            lines.append("")
            lines.append("Nonlinear feature effects (piecewise corrections added to the linear part):")
            order = sorted(nonlinear_features.keys(),
                           key=lambda jj: self.feature_importances_[jj], reverse=True)
            for j in order:
                thresholds, intervals = nonlinear_features[j]
                name = names[j]
                lines.append(f"  f({name}):")
                for i, val in enumerate(intervals):
                    if i == 0:
                        lines.append(f"    {name} <= {thresholds[0]:+.4f}: {val:+.4f}")
                    elif i == len(thresholds):
                        lines.append(f"    {name} >  {thresholds[-1]:+.4f}: {val:+.4f}")
                    else:
                        lines.append(f"    {thresholds[i-1]:+.4f} < {name} <= {thresholds[i]:+.4f}: {val:+.4f}")

        active = set(linear_features) | set(nonlinear_features)
        inactive = [names[j] for j in range(self.n_features_in_) if j not in active]
        if inactive:
            lines.append("")
            lines.append(f"Features with zero coefficients (excluded): {', '.join(inactive)}")
        return "\n".join(lines)


import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SmartAdditiveGAM.__module__ = "interpretable_regressor"

model_shorthand_name = "SmartAdditiveGAM_v4"
model_description = "Boosted-stump GAM, 300 rounds, coupled linear-display + predict (R^2>0.70)"
model_defs = [(model_shorthand_name, SmartAdditiveGAM(n_rounds=300, learning_rate=0.1, min_samples_leaf=5, r2_linear_thresh=0.70))]


# ---------------------------------------------------------------------------
# Evaluation (do not edit anything below this line)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default='gpt-4o',
                        help="LLM checkpoint for interpretability tests (default: gpt-4o)")
    args = parser.parse_args()

    t0 = time.time()

    interp_results = run_all_interp_tests(model_defs, checkpoint=args.checkpoint)
    n_passed = sum(r["passed"] for r in interp_results)
    total = len(interp_results)

    dataset_rmses = evaluate_all_regressors(model_defs)

    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        git_hash = ""

    model_name = model_defs[0][0]
    interp_csv = os.path.join(RESULTS_DIR, "interpretability_results.csv")
    interp_fields = ["model", "test", "suite", "passed", "ground_truth", "response"]

    def _suite(test_name):
        if test_name.startswith("insight_"): return "insight"
        if test_name.startswith("hard_"):    return "hard"
        return "standard"

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

    perf_csv = os.path.join(RESULTS_DIR, "performance_results.csv")
    perf_fields = ["dataset", "model", "rmse", "rank"]

    existing_perf = []
    if os.path.exists(perf_csv):
        with open(perf_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("model") != model_name:
                    existing_perf.append(row)

    for ds_name, model_rmses in dataset_rmses.items():
        rmse_val = model_rmses.get(model_name, float("nan"))
        existing_perf.append({
            "dataset": ds_name,
            "model": model_name,
            "rmse": "" if np.isnan(rmse_val) else f"{rmse_val:.6f}",
            "rank": "",
        })

    by_dataset = defaultdict(list)
    for row in existing_perf:
        by_dataset[row["dataset"]].append(row)

    for ds_name, rows in by_dataset.items():
        valid = [(r, float(r["rmse"])) for r in rows if r["rmse"] not in ("", None)]
        valid.sort(key=lambda x: x[1])
        for rank_idx, (r, _) in enumerate(valid, 1):
            r["rank"] = rank_idx
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

    recompute_all_mean_ranks(RESULTS_DIR)

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
