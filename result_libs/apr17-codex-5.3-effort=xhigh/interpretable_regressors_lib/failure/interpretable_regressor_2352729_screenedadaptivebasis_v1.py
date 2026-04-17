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


class ScreenedAdaptiveBasisRegressor(BaseEstimator, RegressorMixin):
    """Sparse screened basis regressor with linear, square, and one-knot hinge atoms."""

    def __init__(
        self,
        ridge_alpha=1.0,
        top_k_features=14,
        min_kept_features=3,
        corr_fraction=0.08,
        include_squares=True,
        include_hinges=True,
        negligible_coef_eps=0.02,
    ):
        self.ridge_alpha = ridge_alpha
        self.top_k_features = top_k_features
        self.min_kept_features = min_kept_features
        self.corr_fraction = corr_fraction
        self.include_squares = include_squares
        self.include_hinges = include_hinges
        self.negligible_coef_eps = negligible_coef_eps

    def _basis_value(self, x_col, term_type):
        if term_type == "linear":
            return x_col
        if term_type == "square":
            return x_col ** 2
        if term_type == "relu_pos":
            return np.maximum(0.0, x_col)
        if term_type == "relu_neg":
            return np.maximum(0.0, -x_col)
        raise ValueError(f"Unknown basis type: {term_type}")

    def _build_basis(self, Xs, feature_idx):
        cols = []
        terms = []
        for j in feature_idx:
            xj = Xs[:, j]
            cols.append(xj)
            terms.append(("linear", int(j)))
            if self.include_squares:
                cols.append(xj ** 2)
                terms.append(("square", int(j)))
            if self.include_hinges:
                cols.append(np.maximum(0.0, xj))
                terms.append(("relu_pos", int(j)))
                cols.append(np.maximum(0.0, -xj))
                terms.append(("relu_neg", int(j)))

        if not cols:
            cols = [Xs[:, 0]]
            terms = [("linear", 0)]
        Z = np.column_stack(cols)
        return Z, terms

    def _fit_ridge(self, Z, y_centered):
        z_mean = np.mean(Z, axis=0)
        z_std = np.std(Z, axis=0)
        z_std[z_std < 1e-12] = 1.0
        Zs = (Z - z_mean) / z_std
        p = Zs.shape[1]
        gram = Zs.T @ Zs
        rhs = Zs.T @ y_centered
        coef_std = np.linalg.solve(gram + float(self.ridge_alpha) * np.eye(p), rhs)
        coef = coef_std / z_std
        return coef, z_mean

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        self.x_mean_ = np.mean(X, axis=0)
        self.x_std_ = np.std(X, axis=0)
        self.x_std_[self.x_std_ < 1e-12] = 1.0
        Xs = (X - self.x_mean_) / self.x_std_

        self.y_mean_ = float(np.mean(y))
        y_centered = y - self.y_mean_
        y_scale = float(np.std(y_centered)) + 1e-12

        cov = np.mean(Xs * y_centered[:, None], axis=0)
        corr_score = np.abs(cov) / y_scale
        order = np.argsort(corr_score)[::-1]

        k_cap = min(max(int(self.top_k_features), 1), n_features)
        cutoff = self.corr_fraction * float(corr_score[order[0]]) if n_features > 0 else 0.0
        selected = [int(j) for j in order if corr_score[j] >= cutoff][:k_cap]
        if len(selected) < min(self.min_kept_features, n_features):
            selected = [int(j) for j in order[: min(max(self.min_kept_features, 1), n_features)]]

        self.selected_features_ = selected

        Z, self.basis_terms_ = self._build_basis(Xs, self.selected_features_)
        self.basis_coef_, self.basis_mean_ = self._fit_ridge(Z, y_centered)
        self.intercept_ = float(self.y_mean_ - np.dot(self.basis_coef_, self.basis_mean_))

        self.feature_importances_ = np.zeros(n_features, dtype=float)
        for coef, (_, j) in zip(self.basis_coef_, self.basis_terms_):
            self.feature_importances_[j] += abs(float(coef))

        return self

    def predict(self, X):
        check_is_fitted(
            self,
            [
                "intercept_",
                "x_mean_",
                "x_std_",
                "basis_terms_",
                "basis_coef_",
                "feature_importances_",
            ],
        )
        X = np.asarray(X, dtype=float)
        Xs = (X - self.x_mean_) / self.x_std_
        pred = np.full(X.shape[0], self.intercept_, dtype=float)
        for coef, (term_type, feat) in zip(self.basis_coef_, self.basis_terms_):
            pred += float(coef) * self._basis_value(Xs[:, feat], term_type)
        return pred

    def _equation_terms(self):
        terms = [f"{self.intercept_:+.6f}"]
        for coef, (term_type, feat) in zip(self.basis_coef_, self.basis_terms_):
            if abs(float(coef)) <= self.negligible_coef_eps:
                continue
            mu = float(self.x_mean_[feat])
            sigma = float(self.x_std_[feat])
            z = f"((x{feat}-{mu:.6f})/{sigma:.6f})"
            if term_type == "linear":
                terms.append(f"{float(coef):+.6f}*{z}")
            elif term_type == "square":
                terms.append(f"{float(coef):+.6f}*({z})^2")
            elif term_type == "relu_pos":
                terms.append(f"{float(coef):+.6f}*max(0, {z})")
            elif term_type == "relu_neg":
                terms.append(f"{float(coef):+.6f}*max(0, -{z})")
        return terms

    def __str__(self):
        check_is_fitted(
            self,
            [
                "intercept_",
                "selected_features_",
                "basis_terms_",
                "basis_coef_",
                "feature_importances_",
            ],
        )
        active = np.where(self.feature_importances_ > self.negligible_coef_eps)[0]
        inactive = [f"x{j}" for j in range(self.n_features_in_) if j not in set(active.tolist())]
        top = np.argsort(self.feature_importances_)[::-1]
        top_txt = ", ".join(
            f"x{int(j)}:{self.feature_importances_[j]:.4f}" for j in top[: min(6, self.n_features_in_)]
        )

        lines = [
            "Screened Adaptive Basis Regressor",
            "Exact prediction equation (use directly for manual calculation):",
            "  y = " + " ".join(self._equation_terms()),
            "",
            f"Selected features after screening ({len(self.selected_features_)}): "
            + ", ".join(f"x{j}" for j in self.selected_features_),
            f"Features with non-negligible total effect ({len(active)}): "
            + (", ".join(f"x{int(j)}" for j in active) if len(active) else "none"),
        ]

        lines.append("Basis coefficient table:")
        for coef, (term_type, feat) in zip(self.basis_coef_, self.basis_terms_):
            if term_type == "linear":
                name = f"x{feat}_z"
            elif term_type == "square":
                name = f"(x{feat}_z)^2"
            elif term_type == "relu_pos":
                name = f"max(0, x{feat}_z)"
            else:
                name = f"max(0, -x{feat}_z)"
            lines.append(f"  {name}: {float(coef):+.6f}")

        lines.extend([
            "Features with near-zero total effect (candidate irrelevant features): "
            + (", ".join(inactive) if inactive else "none"),
            "Top feature influence magnitudes: " + top_txt,
        ])
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
ScreenedAdaptiveBasisRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "ScreenedAdaptiveBasis_v1"
model_description = "Correlation-screened sparse basis regressor with standardized linear, square, and zero-knot hinge atoms fit by ridge for compact simulation-ready equations"
model_defs = [(model_shorthand_name, ScreenedAdaptiveBasisRegressor())]


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
