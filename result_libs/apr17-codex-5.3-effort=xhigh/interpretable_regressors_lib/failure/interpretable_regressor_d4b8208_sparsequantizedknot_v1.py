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


class SparseQuantizedKnotRegressor(BaseEstimator, RegressorMixin):
    """Compact sparse regressor with linear terms plus one-knot hinge atoms."""

    def __init__(
        self,
        ridge_alpha=0.8,
        top_k_linear=10,
        top_k_hinge=4,
        max_active_terms=10,
        corr_fraction=0.06,
        knot_grid=(0.2, 0.35, 0.5, 0.65, 0.8),
        negligible_coef_eps=0.015,
    ):
        self.ridge_alpha = ridge_alpha
        self.top_k_linear = top_k_linear
        self.top_k_hinge = top_k_hinge
        self.max_active_terms = max_active_terms
        self.corr_fraction = corr_fraction
        self.knot_grid = knot_grid
        self.negligible_coef_eps = negligible_coef_eps

    def _basis_value(self, x_col, term_type, knot):
        if term_type == "linear":
            return x_col
        if term_type == "hinge_pos":
            return np.maximum(0.0, x_col - knot)
        if term_type == "hinge_neg":
            return np.maximum(0.0, knot - x_col)
        raise ValueError(f"Unknown basis type: {term_type}")

    def _column_corr(self, a, b):
        a0 = a - np.mean(a)
        b0 = b - np.mean(b)
        denom = (np.std(a0) + 1e-12) * (np.std(b0) + 1e-12)
        return float(np.mean(a0 * b0) / denom)

    def _fit_linear_system(self, Z, y):
        p = Z.shape[1]
        ZTZ = Z.T @ Z
        rhs = Z.T @ y
        coef = np.linalg.solve(ZTZ + float(self.ridge_alpha) * np.eye(p), rhs)
        return coef

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        self.y_mean_ = float(np.mean(y))
        y_centered = y - self.y_mean_
        y_scale = float(np.std(y_centered)) + 1e-12
        corr_score = np.array([abs(self._column_corr(X[:, j], y_centered)) for j in range(n_features)])
        order = np.argsort(corr_score)[::-1]

        k_cap = min(max(int(self.top_k_linear), 1), n_features)
        cutoff = self.corr_fraction * float(corr_score[order[0]]) if n_features > 0 else 0.0
        selected = [int(j) for j in order if corr_score[j] >= cutoff][:k_cap]
        if not selected:
            selected = [int(order[0])] if n_features > 0 else [0]
        self.selected_features_ = selected

        basis_cols = []
        basis_terms = []

        for j in self.selected_features_:
            basis_cols.append(X[:, j])
            basis_terms.append(("linear", int(j), 0.0))

        hinge_candidates = self.selected_features_[: min(len(self.selected_features_), int(self.top_k_hinge))]
        for j in hinge_candidates:
            xj = X[:, j]
            best = None
            for q in self.knot_grid:
                knot = float(np.quantile(xj, q))
                h_pos = np.maximum(0.0, xj - knot)
                h_neg = np.maximum(0.0, knot - xj)
                s_pos = abs(self._column_corr(h_pos, y_centered))
                s_neg = abs(self._column_corr(h_neg, y_centered))
                if (best is None) or (s_pos > best[0]):
                    best = (s_pos, "hinge_pos", knot, h_pos)
                if (best is None) or (s_neg > best[0]):
                    best = (s_neg, "hinge_neg", knot, h_neg)
            if best is not None and best[0] >= 0.12 * (corr_score[j] + 1e-12):
                _, tname, knot, col = best
                basis_cols.append(col)
                basis_terms.append((tname, int(j), float(knot)))

        if not basis_cols:
            basis_cols = [X[:, 0]]
            basis_terms = [("linear", 0, 0.0)]

        Z = np.column_stack(basis_cols)
        coef = self._fit_linear_system(Z, y_centered)

        max_active = min(max(int(self.max_active_terms), 1), len(coef))
        active_idx = np.argsort(np.abs(coef))[::-1][:max_active]
        active_idx = np.sort(active_idx)
        Z_active = Z[:, active_idx]
        coef_active = np.linalg.lstsq(Z_active, y_centered, rcond=None)[0]

        self.basis_terms_ = [basis_terms[i] for i in active_idx]
        self.basis_coef_ = coef_active
        self.intercept_ = float(self.y_mean_)

        self.feature_importances_ = np.zeros(n_features, dtype=float)
        for c, (_, j, _) in zip(self.basis_coef_, self.basis_terms_):
            self.feature_importances_[j] += abs(float(c))
        if np.max(self.feature_importances_) > 0:
            self.feature_importances_ /= np.max(self.feature_importances_)
        self.y_scale_ = y_scale

        return self

    def _term_expr(self, term_type, feat, knot):
        if term_type == "linear":
            return f"x{feat}"
        if term_type == "hinge_pos":
            return f"max(0, x{feat}-{knot:.4f})"
        if term_type == "hinge_neg":
            return f"max(0, {knot:.4f}-x{feat})"
        return f"x{feat}"

    def _build_equation(self):
        terms = [f"{self.intercept_:+.4f}"]
        for c, (term_type, feat, knot) in zip(self.basis_coef_, self.basis_terms_):
            if abs(float(c)) <= self.negligible_coef_eps:
                continue
            expr = self._term_expr(term_type, feat, knot)
            terms.append(f"{float(c):+.4f}*{expr}")
        return " ".join(terms)

    def predict(self, X):
        check_is_fitted(
            self,
            [
                "intercept_",
                "basis_terms_",
                "basis_coef_",
                "feature_importances_",
            ],
        )
        X = np.asarray(X, dtype=float)
        pred = np.full(X.shape[0], self.intercept_, dtype=float)
        for c, (term_type, feat, knot) in zip(self.basis_coef_, self.basis_terms_):
            pred += float(c) * self._basis_value(X[:, feat], term_type, knot)
        return pred

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
            f"x{int(j)}:{self.feature_importances_[j]:.3f}" for j in top[: min(8, self.n_features_in_)]
        )

        lines = [
            "Sparse Quantized Knot Regressor",
            "Prediction equation (directly computable from raw features):",
            "  y = " + self._build_equation(),
            "",
            f"Selected features by screening ({len(self.selected_features_)}): "
            + ", ".join(f"x{j}" for j in self.selected_features_),
            "Active basis terms:",
        ]
        for c, (term_type, feat, knot) in zip(self.basis_coef_, self.basis_terms_):
            lines.append(f"  {float(c):+.4f} * {self._term_expr(term_type, feat, knot)}")

        lines.extend([
            "Features with near-zero effect (candidate irrelevant features): "
            + (", ".join(inactive) if inactive else "none"),
            "Normalized feature influence ranking: " + top_txt,
        ])
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparseQuantizedKnotRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SparseQuantizedKnot_v1"
model_description = "Compact sparse piecewise-linear regressor: correlation-screened linear backbone plus one-knot hinge atoms, hard-capped active terms, and direct raw-feature equation output"
model_defs = [(model_shorthand_name, SparseQuantizedKnotRegressor())]


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
