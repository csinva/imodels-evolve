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
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class SparseSymbolicRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse symbolic regressor:
    - Builds a modest bank of interpretable basis terms (linear, quadratic, hinge, interaction).
    - Uses L1 selection to keep only a short formula.
    - Refits selected terms with OLS for stable coefficients.
    """

    def __init__(
        self,
        top_features=8,
        quantiles=(0.2, 0.5, 0.8),
        max_terms=8,
        random_state=42,
    ):
        self.top_features = top_features
        self.quantiles = quantiles
        self.max_terms = max_terms
        self.random_state = random_state

    def _impute(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        X = X.copy()
        nan_mask = ~np.isfinite(X)
        if nan_mask.any():
            X[nan_mask] = np.take(self.feature_medians_, np.where(nan_mask)[1])
        return X

    def _feature_corr(self, X, y):
        y_centered = y - y.mean()
        y_std = y_centered.std() + 1e-12
        corrs = np.zeros(X.shape[1], dtype=float)
        for j in range(X.shape[1]):
            xj = X[:, j]
            x_centered = xj - xj.mean()
            denom = (x_centered.std() + 1e-12) * y_std
            corrs[j] = abs(np.mean(x_centered * y_centered) / denom)
        return corrs

    def _build_term_bank(self, X, y):
        n_features = X.shape[1]
        corrs = self._feature_corr(X, y)
        ranked = np.argsort(corrs)[::-1]
        k = min(max(1, self.top_features), n_features)
        top_idx = ranked[:k]

        term_specs = []
        term_names = []

        # Global linear terms for strong baseline predictive performance.
        for j in range(n_features):
            term_specs.append(("lin", j))
            term_names.append(f"x{j}")

        # Nonlinear terms for top features only (keeps formula concise).
        for j in top_idx:
            term_specs.append(("sq", j))
            term_names.append(f"(x{j})^2")
            qvals = np.quantile(X[:, j], self.quantiles)
            for t in qvals:
                term_specs.append(("hinge_pos", j, float(t)))
                term_names.append(f"max(0, x{j} - {t:.3f})")
                term_specs.append(("hinge_neg", j, float(t)))
                term_names.append(f"max(0, {t:.3f} - x{j})")

        # Pairwise interactions among strongest features.
        for a in range(len(top_idx)):
            for b in range(a + 1, len(top_idx)):
                j = int(top_idx[a])
                k = int(top_idx[b])
                term_specs.append(("inter", j, k))
                term_names.append(f"x{j}*x{k}")

        self.term_specs_ = term_specs
        self.term_names_ = term_names

    def _eval_terms(self, X, term_specs):
        Z = np.empty((X.shape[0], len(term_specs)), dtype=float)
        for i, spec in enumerate(term_specs):
            op = spec[0]
            if op == "lin":
                Z[:, i] = X[:, spec[1]]
            elif op == "sq":
                x = X[:, spec[1]]
                Z[:, i] = x * x
            elif op == "hinge_pos":
                Z[:, i] = np.maximum(0.0, X[:, spec[1]] - spec[2])
            elif op == "hinge_neg":
                Z[:, i] = np.maximum(0.0, spec[2] - X[:, spec[1]])
            elif op == "inter":
                Z[:, i] = X[:, spec[1]] * X[:, spec[2]]
            else:
                raise ValueError(f"Unknown term op: {op}")
        return Z

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.feature_medians_ = np.nanmedian(X, axis=0)
        X = self._impute(X)

        self._build_term_bank(X, y)
        Z = self._eval_terms(X, self.term_specs_)

        z_mean = Z.mean(axis=0)
        z_std = Z.std(axis=0)
        z_std[z_std < 1e-12] = 1.0
        Zs = (Z - z_mean) / z_std

        selector = LassoCV(
            cv=3,
            random_state=self.random_state,
            n_alphas=40,
            max_iter=6000,
        )
        selector.fit(Zs, y)
        raw_coef = selector.coef_ / z_std
        raw_intercept = selector.intercept_ - np.dot(raw_coef, z_mean)

        selected = np.where(np.abs(raw_coef) > 1e-7)[0]
        if len(selected) == 0:
            selected = np.array([int(np.argmax(np.abs(raw_coef)))], dtype=int)

        if len(selected) > self.max_terms:
            order = np.argsort(np.abs(raw_coef[selected]))[::-1]
            selected = selected[order[: self.max_terms]]

        self.selected_specs_ = [self.term_specs_[i] for i in selected]
        self.selected_names_ = [self.term_names_[i] for i in selected]

        Z_sel = self._eval_terms(X, self.selected_specs_)
        refit = LinearRegression()
        refit.fit(Z_sel, y)
        self.intercept_ = float(refit.intercept_)
        self.coef_ = refit.coef_.astype(float)

        # Aggregate feature importance across all selected terms for readability.
        feat_imp = np.zeros(self.n_features_in_, dtype=float)
        for c, spec in zip(self.coef_, self.selected_specs_):
            if spec[0] in ("lin", "sq", "hinge_pos", "hinge_neg"):
                feat_imp[spec[1]] += abs(c)
            elif spec[0] == "inter":
                feat_imp[spec[1]] += 0.5 * abs(c)
                feat_imp[spec[2]] += 0.5 * abs(c)
        self.feature_importance_ = feat_imp
        self.selector_intercept_ = float(raw_intercept)
        return self

    def predict(self, X):
        check_is_fitted(self, ["coef_", "intercept_", "selected_specs_"])
        X = self._impute(X)
        Z_sel = self._eval_terms(X, self.selected_specs_)
        return self.intercept_ + Z_sel @ self.coef_

    def __str__(self):
        check_is_fitted(self, ["coef_", "intercept_", "selected_names_", "feature_importance_"])
        lines = [
            "SparseSymbolicRegressor (short additive equation):",
            f"prediction = {self.intercept_:.4f}",
        ]
        order = np.argsort(np.abs(self.coef_))[::-1]
        for i in order:
            lines.append(f"  + ({self.coef_[i]:+.4f}) * {self.selected_names_[i]}")

        lines.append("")
        lines.append("Feature importance (sum of absolute selected-term contributions):")
        rank = np.argsort(self.feature_importance_)[::-1]
        for j in rank:
            lines.append(f"  x{j}: {self.feature_importance_[j]:.4f}")

        low = [f"x{j}" for j, v in enumerate(self.feature_importance_) if v < 1e-4]
        if low:
            lines.append(f"Features with near-zero effect: {', '.join(low)}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparseSymbolicRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SparseSymbolicV1"
model_description = "Sparse symbolic equation via L1-selected linear/quadratic/hinge/interaction terms with OLS refit"
model_defs = [(model_shorthand_name, SparseSymbolicRegressor())]


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
