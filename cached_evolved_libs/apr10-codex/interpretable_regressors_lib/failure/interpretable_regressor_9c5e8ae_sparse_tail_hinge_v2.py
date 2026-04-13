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


class SparseTailHingeRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse additive regressor with feature-level group selection.
    Each selected feature contributes one of:
      - linear group: z_j
      - symmetric tail-hinge group: [max(0, z_j-k), max(0, -z_j-k)]
    where z_j is a standardized feature and k is a data-derived knot.
    """

    def __init__(
        self,
        top_features=14,
        max_groups=5,
        shortlist=10,
        min_rel_gain=2e-3,
        knot_quantiles=(0.6, 0.8),
    ):
        self.top_features = top_features
        self.max_groups = max_groups
        self.shortlist = shortlist
        self.min_rel_gain = min_rel_gain
        self.knot_quantiles = knot_quantiles

    @staticmethod
    def _safe_scale(x):
        s = np.std(x)
        if (not np.isfinite(s)) or s < 1e-12:
            return 1.0
        return float(s)

    @staticmethod
    def _centered_corr(a, b):
        ac = a - np.mean(a)
        bc = b - np.mean(b)
        denom = (np.linalg.norm(ac) + 1e-12) * (np.linalg.norm(bc) + 1e-12)
        return float(np.dot(ac, bc) / denom)

    def _group_columns(self, Xs, group):
        z = Xs[:, group["j"]]
        if group["type"] == "linear":
            return [z]
        if group["type"] == "tail_hinge":
            k = group["k"]
            return [np.maximum(0.0, z - k), np.maximum(0.0, -z - k)]
        raise ValueError(f"Unknown group type: {group['type']}")

    def _fit_ols_from_groups(self, Xs, y, groups):
        n = Xs.shape[0]
        cols = [np.ones(n, dtype=float)]
        group_slices = []
        col_idx = 1
        for group in groups:
            gcols = self._group_columns(Xs, group)
            cols.extend(gcols)
            group_slices.append((col_idx, col_idx + len(gcols)))
            col_idx += len(gcols)
        design = np.column_stack(cols)
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        pred = design @ beta
        rss = float(np.sum((y - pred) ** 2))
        return beta, pred, rss, group_slices

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape

        self.x_mean_ = np.mean(X, axis=0)
        self.x_scale_ = np.array([self._safe_scale(X[:, j]) for j in range(p)], dtype=float)
        Xs = (X - self.x_mean_) / self.x_scale_

        feature_scores = np.array([abs(self._centered_corr(Xs[:, j], y)) for j in range(p)])
        top_k = min(int(self.top_features), p)
        feature_pool = [int(j) for j in np.argsort(-feature_scores)[:top_k]]

        candidate_groups = []
        for j in feature_pool:
            candidate_groups.append({"type": "linear", "j": j})
            az = np.abs(Xs[:, j])
            for q in self.knot_quantiles:
                k = float(np.quantile(az, float(q)))
                k = max(k, 0.25)
                candidate_groups.append({"type": "tail_hinge", "j": j, "k": k})

        selected_groups = []
        beta, pred, best_rss, group_slices = self._fit_ols_from_groups(Xs, y, selected_groups)

        for _ in range(int(self.max_groups)):
            used_features = {g["j"] for g in selected_groups}
            remaining = [g for g in candidate_groups if g["j"] not in used_features]
            if not remaining:
                break

            resid = y - pred
            scored = sorted(
                (
                    (
                        max(abs(self._centered_corr(col, resid)) for col in self._group_columns(Xs, g)),
                        g,
                    )
                    for g in remaining
                ),
                key=lambda z: z[0],
                reverse=True,
            )
            shortlist = [g for _, g in scored[: min(int(self.shortlist), len(scored))]]

            best_trial = None
            for group in shortlist:
                trial_groups = selected_groups + [group]
                trial_beta, trial_pred, trial_rss, trial_slices = self._fit_ols_from_groups(Xs, y, trial_groups)
                gain = (best_rss - trial_rss) / (best_rss + 1e-12)
                if (best_trial is None) or (gain > best_trial[0]):
                    best_trial = (gain, group, trial_beta, trial_pred, trial_rss, trial_slices)

            if (best_trial is None) or (best_trial[0] < self.min_rel_gain):
                break

            _, chosen_group, beta, pred, best_rss, group_slices = best_trial
            selected_groups.append(chosen_group)

        self.groups_ = selected_groups
        self.group_slices_ = group_slices
        self.intercept_ = float(beta[0])
        self.group_coefs_ = [np.asarray(beta[s:e], dtype=float) for s, e in self.group_slices_]
        self.n_features_in_ = p
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        X = np.asarray(X, dtype=float)
        Xs = (X - self.x_mean_) / self.x_scale_

        yhat = np.full(X.shape[0], self.intercept_, dtype=float)
        for group, coefs in zip(self.groups_, self.group_coefs_):
            cols = self._group_columns(Xs, group)
            for coef, col in zip(coefs, cols):
                yhat += float(coef) * col
        return yhat

    def __str__(self):
        check_is_fitted(self, "is_fitted_")
        lines = [
            "Sparse Tail-Hinge Regressor:",
            "  y = intercept + sum(feature groups)",
            f"  intercept: {self.intercept_:+.6f}",
            f"  num_groups: {len(self.groups_)}",
            "  Groups:",
        ]

        if len(self.groups_) == 0:
            lines.append("    none")
        else:
            for i, (group, coefs) in enumerate(zip(self.groups_, self.group_coefs_), 1):
                if group["type"] == "linear":
                    j = group["j"]
                    txt = f"{float(coefs[0]):+.6f} * z{j}"
                elif group["type"] == "tail_hinge":
                    j = group["j"]
                    txt = f"{float(coefs[0]):+.6f} * max(0, z{j}-{group['k']:.3f}) + {float(coefs[1]):+.6f} * max(0, -z{j}-{group['k']:.3f})"
                else:
                    txt = f"unknown_group_type={group['type']}"
                lines.append(f"    {i}. {txt}")

        used = sorted({g["j"] for g in self.groups_})
        lines.append("  Active features: " + (", ".join(f"x{j}" for j in used) if used else "none"))
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparseTailHingeRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SparseTailHinge_v2"
model_description = "Greedy sparse additive model selecting per-feature linear or symmetric tail-hinge groups with quantile knots (cache-reset run)"
model_defs = [(model_shorthand_name, SparseTailHingeRegressor())]


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
