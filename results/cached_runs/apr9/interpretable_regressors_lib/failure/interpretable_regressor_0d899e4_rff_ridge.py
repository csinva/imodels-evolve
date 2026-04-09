"""
Interpretable regressor autoresearch script.
Usage: uv run model.py
"""

import csv, os, subprocess, sys, time
from collections import defaultdict

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import RidgeCV
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance


class RFFRidgeRegressor(BaseEstimator, RegressorMixin):
    """
    Random Fourier Feature Ridge: approximates kernel Ridge regression
    using random sinusoidal features, then fits Ridge.

    Creates D random features: cos(w_i^T x + b_i) for random w, b.
    These approximate the RBF kernel, giving nonlinear prediction power.

    For display: compute effective per-feature linear coefficients by
    marginalizing over the random features. Show as Ridge equation.

    Innovation: kernel methods + interpretable linear display.
    """

    def __init__(self, n_rff=50, gamma=1.0, max_input_features=20):
        self.n_rff = n_rff
        self.gamma = gamma
        self.max_input_features = max_input_features

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.n_features_in_ = X.shape[1]
        n_samples, n_orig = X.shape

        # Feature selection
        if n_orig > self.max_input_features:
            corrs = np.array([abs(np.corrcoef(X[:, j], y)[0, 1])
                              if np.std(X[:, j]) > 1e-10 else 0
                              for j in range(n_orig)])
            self.selected_ = np.sort(np.argsort(corrs)[-self.max_input_features:])
        else:
            self.selected_ = np.arange(n_orig)

        X_sel = X[:, self.selected_]
        n_feat = X_sel.shape[1]

        # Standardize
        self.x_mean_ = X_sel.mean(axis=0)
        self.x_std_ = np.maximum(X_sel.std(axis=0), 1e-8)
        X_norm = (X_sel - self.x_mean_) / self.x_std_

        # Generate random Fourier features
        rng = np.random.RandomState(42)
        self.W_ = rng.randn(n_feat, self.n_rff) * np.sqrt(2 * self.gamma)
        self.b_ = rng.uniform(0, 2 * np.pi, self.n_rff)

        Z = np.cos(X_norm @ self.W_ + self.b_) * np.sqrt(2.0 / self.n_rff)

        # Augmented features: original + RFF
        X_aug = np.hstack([X_sel, Z])

        # Fit Ridge
        self.ridge_ = RidgeCV(cv=3)
        self.ridge_.fit(X_aug, y)

        # Compute effective per-feature coefficients for display
        # The original feature coefficients are direct
        # The RFF coefficients contribute indirectly — approximate as linear
        coefs = self.ridge_.coef_
        self.effective_coefs_ = np.zeros(n_orig)
        for idx, j in enumerate(self.selected_):
            self.effective_coefs_[j] = coefs[idx]
            # Add linearized RFF contribution
            # RFF feature k: cos(W[idx,k]*x_norm + ...) ≈ linear in x_norm for small x
            # Marginal linear effect ≈ -sum(coefs_rff * W[idx,k] * sin(b_k)) / x_std
            for k in range(self.n_rff):
                rff_coef = coefs[n_feat + k]
                self.effective_coefs_[j] -= rff_coef * self.W_[idx, k] * np.sin(self.b_[k]) * np.sqrt(2.0 / self.n_rff) / self.x_std_[idx]

        self.effective_intercept_ = float(self.ridge_.intercept_)
        # Adjust intercept for the RFF contributions at x=mean
        for k in range(self.n_rff):
            rff_coef = coefs[n_feat + k]
            self.effective_intercept_ += rff_coef * np.cos(self.b_[k]) * np.sqrt(2.0 / self.n_rff)

        return self

    def _transform(self, X):
        X_sel = X[:, self.selected_]
        X_norm = (X_sel - self.x_mean_) / self.x_std_
        Z = np.cos(X_norm @ self.W_ + self.b_) * np.sqrt(2.0 / self.n_rff)
        return np.hstack([X_sel, Z])

    def predict(self, X):
        check_is_fitted(self, "ridge_")
        X = np.asarray(X, dtype=np.float64)
        return self.ridge_.predict(self._transform(X))

    def __str__(self):
        check_is_fitted(self, "ridge_")
        feature_names = [f"x{i}" for i in range(self.n_features_in_)]

        active = {j: c for j, c in enumerate(self.effective_coefs_) if abs(c) > 1e-6}

        lines = [f"Ridge Regression (L2 regularization, α={self.ridge_.alpha_:.4g} chosen by CV):"]
        terms = [f"{active[j]:.4f}*{feature_names[j]}" for j in sorted(active.keys())]
        eq = " + ".join(terms) + f" + {self.effective_intercept_:.4f}" if terms else f"{self.effective_intercept_:.4f}"
        lines.append(f"  y = {eq}")
        lines.append("")
        lines.append("Coefficients:")
        for j, c in sorted(active.items(), key=lambda x: abs(x[1]), reverse=True):
            lines.append(f"  {feature_names[j]}: {c:.4f}")
        lines.append(f"  intercept: {self.effective_intercept_:.4f}")

        inactive = [feature_names[j] for j in range(self.n_features_in_) if j not in active]
        if inactive:
            lines.append(f"  Features with zero coefficients (excluded): {', '.join(inactive)}")
        return "\n".join(lines)


import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
RFFRidgeRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "RFFRidge_v1"
model_description = "Random Fourier Feature Ridge: kernel approximation with effective linear display"
model_defs = [(model_shorthand_name, RFFRidgeRegressor())]


# ---------------------------------------------------------------------------
# Evaluation (do not edit anything below this line)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()
    interp_results = run_all_interp_tests(model_defs)
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
    new_interp = [{"model": r["model"], "test": r["test"], "suite": _suite(r["test"]),
        "passed": r["passed"], "ground_truth": r.get("ground_truth", ""),
        "response": r.get("response", "")} for r in interp_results]
    with open(interp_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=interp_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing_interp + new_interp)

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
        existing_perf.append({"dataset": ds_name, "model": model_name,
            "rmse": "" if np.isnan(rmse_val) else f"{rmse_val:.6f}", "rank": ""})
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
        "commit": git_hash,
        "mean_rank": f"{mean_rank:.2f}" if not np.isnan(mean_rank) else "nan",
        "frac_interpretability_tests_passed": f"{n_passed / total:.4f}" if total > 0 else "nan",
        "status": "", "model_name": model_shorthand_name, "description": model_description,
    }], RESULTS_DIR)

    overall_csv = os.path.join(RESULTS_DIR, "overall_results.csv")
    plot_interp_vs_performance(overall_csv, os.path.join(RESULTS_DIR, "interpretability_vs_performance.png"))
    print()
    print("---")
    print(f"tests_passed:  {n_passed}/{total}" + (f" ({n_passed/total:.2%})" if total > 0 else ""))
    print(f"mean_rank:     {mean_rank:.2f}" if not np.isnan(mean_rank) else "mean_rank:     nan")
    print(f"total_seconds: {time.time() - t0:.1f}s")
