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


class StagewisePiecewiseRegressor(BaseEstimator, RegressorMixin):
    """
    Interpretable two-stage model:
      1) Sparse stagewise linear backbone built from centered/scaled features.
      2) Small number of one-threshold step corrections fit on residuals.
    """

    def __init__(
        self,
        max_linear_terms=6,
        max_step_rules=2,
        top_features=10,
        quantiles=(0.2, 0.4, 0.6, 0.8),
        min_rel_gain=2e-3,
    ):
        self.max_linear_terms = max_linear_terms
        self.max_step_rules = max_step_rules
        self.top_features = top_features
        self.quantiles = quantiles
        self.min_rel_gain = min_rel_gain

    @staticmethod
    def _safe_scale(x):
        s = np.std(x)
        if not np.isfinite(s) or s < 1e-12:
            return 1.0
        return float(s)

    @staticmethod
    def _corr(vec, target):
        vc = vec - np.mean(vec)
        tc = target - np.mean(target)
        denom = (np.linalg.norm(vc) + 1e-12) * (np.linalg.norm(tc) + 1e-12)
        return float(abs(np.dot(vc, tc)) / denom)

    def _ols(self, design, y):
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        pred = design @ beta
        rss = float(np.sum((y - pred) ** 2))
        return beta, pred, rss

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape

        self.x_mean_ = np.mean(X, axis=0)
        self.x_scale_ = np.array([self._safe_scale(X[:, j]) for j in range(p)], dtype=float)
        Xs = (X - self.x_mean_) / self.x_scale_

        corr = np.array([self._corr(Xs[:, j], y) for j in range(p)])
        top_k = min(int(self.top_features), p)
        linear_pool = [int(j) for j in np.argsort(-corr)[:top_k]]

        selected = []
        beta, pred, best_rss = self._ols(np.ones((n, 1), dtype=float), y)

        for _ in range(int(self.max_linear_terms)):
            remain = [j for j in linear_pool if j not in selected]
            if not remain:
                break
            resid = y - pred
            scored = sorted(
                ((self._corr(Xs[:, j], resid), j) for j in remain),
                reverse=True,
                key=lambda z: z[0],
            )
            shortlist = [j for _, j in scored[: min(8, len(scored))]]

            best_trial = None
            for j in shortlist:
                cols = [np.ones(n, dtype=float)] + [Xs[:, k] for k in (selected + [j])]
                design = np.column_stack(cols)
                trial_beta, trial_pred, trial_rss = self._ols(design, y)
                gain = (best_rss - trial_rss) / (best_rss + 1e-12)
                if best_trial is None or gain > best_trial[0]:
                    best_trial = (gain, j, trial_beta, trial_pred, trial_rss)

            if best_trial is None or best_trial[0] < self.min_rel_gain:
                break
            _, chosen, beta, pred, best_rss = best_trial
            selected.append(chosen)

        self.linear_features_ = selected
        self.intercept_ = float(beta[0])
        self.linear_coefs_ = np.asarray(beta[1:], dtype=float)

        # Residual step-rules: each rule adds one constant shift on one side of threshold.
        step_rules = []
        for _ in range(int(self.max_step_rules)):
            resid = y - pred
            base_rss = float(np.sum(resid * resid))
            best_rule = None

            for j in linear_pool:
                xj = X[:, j]
                for q in self.quantiles:
                    thr = float(np.quantile(xj, q))
                    mask = xj >= thr
                    n_hi = int(np.sum(mask))
                    n_lo = n - n_hi
                    if n_hi < 5 or n_lo < 5:
                        continue
                    effect_hi = float(np.mean(resid[mask]))
                    effect_lo = float(np.mean(resid[~mask]))
                    # Keep a one-parameter rule by centering effects to sum to 0 globally.
                    effect = effect_hi - effect_lo
                    centered_shift = np.where(mask, effect * (n_lo / n), -effect * (n_hi / n))
                    trial_pred = pred + centered_shift
                    trial_rss = float(np.sum((y - trial_pred) ** 2))
                    gain = (base_rss - trial_rss) / (base_rss + 1e-12)
                    if best_rule is None or gain > best_rule["gain"]:
                        best_rule = {
                            "gain": gain,
                            "j": int(j),
                            "thr": thr,
                            "effect": effect,
                            "n_hi": n_hi,
                            "n_lo": n_lo,
                            "centered_shift": centered_shift,
                        }

            if best_rule is None or best_rule["gain"] < self.min_rel_gain:
                break

            pred = pred + best_rule["centered_shift"]
            best_rule.pop("centered_shift", None)
            step_rules.append(best_rule)

        self.step_rules_ = step_rules
        self.n_features_in_ = p
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        X = np.asarray(X, dtype=float)
        Xs = (X - self.x_mean_) / self.x_scale_
        yhat = np.full(X.shape[0], self.intercept_, dtype=float)

        for coef, j in zip(self.linear_coefs_, self.linear_features_):
            yhat += coef * Xs[:, j]

        for rule in self.step_rules_:
            j = rule["j"]
            thr = rule["thr"]
            effect = rule["effect"]
            mask = X[:, j] >= thr
            n_hi = max(rule["n_hi"], 1)
            n_lo = max(rule["n_lo"], 1)
            total = n_hi + n_lo
            yhat += np.where(mask, effect * (n_lo / total), -effect * (n_hi / total))

        return yhat

    def __str__(self):
        check_is_fitted(self, "is_fitted_")
        lines = [
            "Stagewise Piecewise Regressor:",
            "  y = intercept + sparse_linear_terms + few_step_rules",
            f"  intercept: {self.intercept_:+.6f}",
            f"  linear_terms: {len(self.linear_features_)}",
            f"  step_rules: {len(self.step_rules_)}",
            "  Linear part (using standardized features z_j = (x_j - mean_j)/std_j):",
        ]

        if self.linear_features_:
            for i, (coef, j) in enumerate(zip(self.linear_coefs_, self.linear_features_), 1):
                lines.append(
                    f"    {i}. {coef:+.6f} * z{j}   [mean={self.x_mean_[j]:.4f}, std={self.x_scale_[j]:.4f}]"
                )
        else:
            lines.append("    none")

        lines.append("  Residual step rules:")
        if self.step_rules_:
            for i, r in enumerate(self.step_rules_, 1):
                lines.append(
                    f"    {i}. if x{r['j']} >= {r['thr']:.4f}, add shift; rule_strength={r['effect']:+.6f}"
                )
        else:
            lines.append("    none")

        active = sorted(set(self.linear_features_) | {r["j"] for r in self.step_rules_})
        lines.append("  Active raw features: " + (", ".join(f"x{j}" for j in active) if active else "none"))
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
StagewisePiecewiseRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "StagewisePiece_v1"
model_description = "Sparse stagewise standardized linear model with a tiny residual one-threshold step-rule layer"
model_defs = [(model_shorthand_name, StagewisePiecewiseRegressor())]


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
