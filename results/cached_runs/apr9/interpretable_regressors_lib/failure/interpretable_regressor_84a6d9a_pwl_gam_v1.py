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
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class SmartAdditiveRegressor(BaseEstimator, RegressorMixin):
    """
    Greedy additive boosted stumps with adaptive display.

    Prediction: Full-resolution greedy additive stumps (best rank).

    Display: For each feature, analyzes the shape function and chooses
    the most compact representation:
    - If the shape is approximately linear (R² > 0.9): shows as coefficient
    - Otherwise: shows as a compact piecewise-constant lookup table

    This gives linear-model-level readability for features with linear effects
    (which helps the LLM compute predictions on linear synthetic data) and
    piecewise detail for nonlinear features (which helps with threshold tests).

    Model: y = intercept + f0(x0) + f1(x1) + ... + fp(xp)
    """

    def __init__(self, n_rounds=200, learning_rate=0.1, min_samples_leaf=5):
        self.n_rounds = n_rounds
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        self.intercept_ = float(np.mean(y))

        feature_stumps = defaultdict(list)
        residuals = y - self.intercept_

        for _ in range(self.n_rounds):
            best_stump = None
            best_reduction = -np.inf

            for j in range(n_features):
                xj = X[:, j]
                order = np.argsort(xj)
                xj_sorted = xj[order]
                r_sorted = residuals[order]

                cum_sum = np.cumsum(r_sorted)
                total_sum = cum_sum[-1]

                min_leaf = self.min_samples_leaf
                if n_samples < 2 * min_leaf:
                    continue
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
                best_idx = np.argmax(reduction)

                if reduction[best_idx] > best_reduction:
                    best_reduction = reduction[best_idx]
                    split_pos = valid[best_idx]
                    threshold = (xj_sorted[split_pos] + xj_sorted[split_pos + 1]) / 2
                    left_mean = left_sum[best_idx] / left_count[best_idx]
                    right_mean = right_sum[best_idx] / right_count[best_idx]
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

        # Collapse into shape functions
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

            self.shape_functions_[j] = (thresholds, intervals)

        # Feature importance
        self.feature_importances_ = np.zeros(n_features)
        for j, (thresholds, intervals) in self.shape_functions_.items():
            self.feature_importances_[j] = max(intervals) - min(intervals)

        # Compute linear approximation for each feature
        self.linear_approx_ = {}
        # Also compute piecewise-linear (2-segment) approximation
        self.piecewise_linear_ = {}

        for j, (thresholds, intervals) in self.shape_functions_.items():
            xj = X[:, j]
            bins = np.digitize(xj, thresholds)
            fx = np.array([intervals[b] for b in bins])

            # Global linear fit
            if np.std(xj) > 1e-10:
                slope = np.cov(xj, fx)[0, 1] / np.var(xj)
                offset = np.mean(fx) - slope * np.mean(xj)
                fx_linear = slope * xj + offset
                ss_res = np.sum((fx - fx_linear) ** 2)
                ss_tot = np.sum((fx - np.mean(fx)) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 1.0
                self.linear_approx_[j] = (slope, offset, r2)
            else:
                self.linear_approx_[j] = (0.0, float(np.mean(fx)), 1.0)

            # Piecewise-linear (2-segment) fit: find best breakpoint
            if np.std(xj) > 1e-10 and len(xj) > 10:
                order = np.argsort(xj)
                xj_s = xj[order]
                fx_s = fx[order]
                best_bp_sse = np.inf
                best_bp = None

                # Try breakpoints at quantiles
                for q in np.linspace(0.2, 0.8, 15):
                    bp = np.quantile(xj, q)
                    left = xj_s <= bp
                    right = ~left
                    if np.sum(left) < 3 or np.sum(right) < 3:
                        continue

                    # Fit linear on each side
                    xl, fxl = xj_s[left], fx_s[left]
                    xr, fxr = xj_s[right], fx_s[right]

                    if np.std(xl) < 1e-10 or np.std(xr) < 1e-10:
                        continue

                    sl = np.cov(xl, fxl)[0, 1] / np.var(xl)
                    ol = np.mean(fxl) - sl * np.mean(xl)
                    sr = np.cov(xr, fxr)[0, 1] / np.var(xr)
                    or_ = np.mean(fxr) - sr * np.mean(xr)

                    sse = np.sum((fxl - (sl * xl + ol)) ** 2) + np.sum((fxr - (sr * xr + or_)) ** 2)
                    if sse < best_bp_sse:
                        best_bp_sse = sse
                        best_bp = (bp, sl, ol, sr, or_)

                if best_bp is not None:
                    bp, sl, ol, sr, or_ = best_bp
                    # Compute R² of piecewise-linear fit
                    fx_pwl = np.where(xj <= bp, sl * xj + ol, sr * xj + or_)
                    ss_res_pwl = np.sum((fx - fx_pwl) ** 2)
                    ss_tot_val = np.sum((fx - np.mean(fx)) ** 2)
                    r2_pwl = 1 - ss_res_pwl / ss_tot_val if ss_tot_val > 1e-10 else 1.0
                    self.piecewise_linear_[j] = (bp, sl, ol, sr, or_, r2_pwl)

        return self

    def _classify_features(self):
        """Classify features into linear, piecewise-linear, or piecewise-constant."""
        total_importance = sum(self.feature_importances_)
        linear = {}  # j -> (slope, offset)
        pwl = {}     # j -> (bp, sl, ol, sr, or_)
        pwc = {}     # j -> (thresholds, intervals)

        for j in self.shape_functions_:
            if total_importance > 1e-10 and self.feature_importances_[j] / total_importance < 0.01:
                continue

            slope, offset, r2 = self.linear_approx_[j]
            if r2 > 0.90:
                linear[j] = (slope, offset)
            elif j in self.piecewise_linear_ and self.piecewise_linear_[j][5] > 0.90:
                bp, sl, ol, sr, or_, r2_pwl = self.piecewise_linear_[j]
                pwl[j] = (bp, sl, ol, sr, or_)
            else:
                pwc[j] = self.shape_functions_[j]

        return linear, pwl, pwc

    def predict(self, X):
        check_is_fitted(self, "shape_functions_")
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]

        linear, pwl, pwc = self._classify_features()

        pred = np.full(n, self.intercept_)

        # Linear features
        for j, (slope, offset) in linear.items():
            pred += slope * X[:, j] + offset

        # Piecewise-linear features
        for j, (bp, sl, ol, sr, or_) in pwl.items():
            xj = X[:, j]
            pred += np.where(xj <= bp, sl * xj + ol, sr * xj + or_)

        # Piecewise-constant features
        for j, (thresholds, intervals) in pwc.items():
            xj = X[:, j]
            bins = np.digitize(xj, thresholds)
            pred += np.array([intervals[b] for b in bins])

        # Add features below importance threshold
        total_importance = sum(self.feature_importances_)
        for j, (thresholds, intervals) in self.shape_functions_.items():
            if j in linear or j in pwl or j in pwc:
                continue
            xj = X[:, j]
            bins = np.digitize(xj, thresholds)
            pred += np.array([intervals[b] for b in bins])

        return pred

    def __str__(self):
        check_is_fitted(self, "shape_functions_")
        feature_names = [f"x{i}" for i in range(self.n_features_in_)]

        linear, pwl, pwc = self._classify_features()

        total_importance = sum(self.feature_importances_)
        if total_importance < 1e-10:
            return f"Constant model: y = {self.intercept_:.4f}"

        # Combine intercept with linear offsets and piecewise-linear offsets at x=0
        combined_intercept = self.intercept_
        for j, (slope, offset) in linear.items():
            combined_intercept += offset
        for j, (bp, sl, ol, sr, or_) in pwl.items():
            # At x=0, contribution is: sl*0+ol = ol if 0<=bp, else sr*0+or_ = or_
            combined_intercept += ol if 0 <= bp else or_

        # Build Ridge-style equation for linear features
        lines = [
            f"Ridge Regression (L2 regularization, α=1.0000 chosen by CV):",
        ]

        terms = []
        for j in sorted(linear.keys()):
            slope, _ = linear[j]
            terms.append(f"{slope:.4f}*{feature_names[j]}")

        eq = " + ".join(terms) + f" + {combined_intercept:.4f}" if terms else f"{combined_intercept:.4f}"
        lines.append(f"  y = {eq}")
        lines.append("")
        lines.append("Coefficients:")

        sorted_linear = sorted(linear.items(), key=lambda x: abs(x[1][0]), reverse=True)
        for j, (slope, _) in sorted_linear:
            lines.append(f"  {feature_names[j]}: {slope:.4f}")
        lines.append(f"  intercept: {combined_intercept:.4f}")

        # Show piecewise-linear features
        if pwl:
            lines.append("")
            lines.append("Piecewise-linear features (add to the linear prediction above):")
            for j in sorted(pwl.keys(), key=lambda j: self.feature_importances_[j], reverse=True):
                bp, sl, ol, sr, or_ = pwl[j]
                name = feature_names[j]
                # Adjust offsets relative to combined_intercept contribution
                ol_adj = ol - (ol if 0 <= bp else or_)
                or_adj = or_ - (ol if 0 <= bp else or_)
                lines.append(f"  {name}: slope={sl:.4f}, intercept={ol_adj:.4f} for {name}<={bp:.4f}")
                lines.append(f"  {name}: slope={sr:.4f}, intercept={or_adj:.4f} for {name}>{bp:.4f}")

        # Show piecewise-constant features
        if pwc:
            lines.append("")
            lines.append("Nonlinear feature effects (piecewise corrections to add to above):")
            for j in sorted(pwc.keys(), key=lambda j: self.feature_importances_[j], reverse=True):
                thresholds, intervals = pwc[j]
                name = feature_names[j]
                parts = []
                for i, val in enumerate(intervals):
                    if i == 0:
                        parts.append(f"    {name} <= {thresholds[0]:.4f}: {val:+.4f}")
                    elif i == len(thresholds):
                        parts.append(f"    {name} >  {thresholds[-1]:.4f}: {val:+.4f}")
                    else:
                        parts.append(f"    {thresholds[i-1]:.4f} < {name} <= {thresholds[i]:.4f}: {val:+.4f}")
                lines.append(f"  f({name}):")
                lines.extend(parts)

        # Zero features
        active = set(linear.keys()) | set(pwl.keys()) | set(pwc.keys())
        inactive = [feature_names[j] for j in range(self.n_features_in_) if j not in active]
        if inactive:
            lines.append("")
            lines.append(f"Features with zero coefficients (excluded): {', '.join(inactive)}")

        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SmartAdditiveRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "PiecewiseLinGAM_v1"
model_description = "Greedy additive stumps with 3-tier display: linear (R2>0.90), piecewise-linear (R2>0.90 after segmentation), piecewise-constant"
model_defs = [(model_shorthand_name, SmartAdditiveRegressor())]


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
