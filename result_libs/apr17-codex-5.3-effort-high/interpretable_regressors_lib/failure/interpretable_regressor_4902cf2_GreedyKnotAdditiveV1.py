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


class GreedyKnotAdditiveRegressor(BaseEstimator, RegressorMixin):
    """
    Greedy sparse additive regressor with optional one-knot piecewise terms.

    Each selected feature is represented as either:
      linear:      a * x_j
      piecewise:   a * x_j + b * max(0, x_j - t_j)
    where t_j is a learned knot chosen from feature quantiles.
    """

    def __init__(
        self,
        max_features=4,
        ridge_lambda=1e-3,
        min_rel_gain=1e-3,
        knot_quantiles=(0.2, 0.35, 0.5, 0.65, 0.8),
    ):
        self.max_features = max_features
        self.ridge_lambda = ridge_lambda
        self.min_rel_gain = min_rel_gain
        self.knot_quantiles = knot_quantiles

    @staticmethod
    def _fit_ridge(B, y, lam):
        gram = B.T @ B
        rhs = B.T @ y
        coef = np.linalg.solve(gram + lam * np.eye(gram.shape[0]), rhs)
        pred = B @ coef
        mse = float(np.mean((y - pred) ** 2))
        return coef, pred, mse

    def _build_group_linear(self, X, j):
        return np.column_stack([X[:, j]]), {"feature": int(j), "kind": "linear", "knot": None}

    def _build_group_piecewise(self, X, j, knot):
        xj = X[:, j]
        return np.column_stack([xj, np.maximum(0.0, xj - knot)]), {
            "feature": int(j),
            "kind": "piecewise",
            "knot": float(knot),
        }

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        self.intercept_ = float(np.mean(y))
        yc = y - self.intercept_

        selected_groups = []
        selected_cols = []
        used_features = set()
        current_pred = np.zeros(n_samples, dtype=float)
        current_mse = float(np.mean(yc ** 2))

        max_steps = min(self.max_features, n_features)
        for _ in range(max_steps):
            best = None

            for j in range(n_features):
                if j in used_features:
                    continue

                linear_cols, linear_meta = self._build_group_linear(X, j)
                trial_linear = selected_cols + [linear_cols]
                B_linear = np.column_stack(trial_linear)
                coef_linear, pred_linear, mse_linear = self._fit_ridge(B_linear, yc, self.ridge_lambda)
                best_kind_cols = linear_cols
                best_kind_meta = linear_meta
                best_kind_coef = coef_linear
                best_kind_pred = pred_linear
                best_kind_mse = mse_linear

                unique_vals = np.unique(X[:, j])
                if unique_vals.size > 4:
                    knots = []
                    for q in self.knot_quantiles:
                        knots.append(float(np.quantile(X[:, j], q)))
                    knots.append(0.0)
                    knots = sorted(set(knots))
                    for t in knots:
                        piece_cols, piece_meta = self._build_group_piecewise(X, j, t)
                        trial_piece = selected_cols + [piece_cols]
                        B_piece = np.column_stack(trial_piece)
                        coef_piece, pred_piece, mse_piece = self._fit_ridge(B_piece, yc, self.ridge_lambda)
                        if mse_piece < best_kind_mse:
                            best_kind_cols = piece_cols
                            best_kind_meta = piece_meta
                            best_kind_coef = coef_piece
                            best_kind_pred = pred_piece
                            best_kind_mse = mse_piece

                if best is None or best_kind_mse < best["mse"]:
                    best = {
                        "feature": int(j),
                        "cols": best_kind_cols,
                        "meta": best_kind_meta,
                        "coef": best_kind_coef,
                        "pred": best_kind_pred,
                        "mse": best_kind_mse,
                    }

            if best is None:
                break

            rel_gain = (current_mse - best["mse"]) / max(1e-12, current_mse)
            if rel_gain < self.min_rel_gain:
                break

            used_features.add(best["feature"])
            selected_groups.append(best["meta"])
            selected_cols.append(best["cols"])
            current_pred = best["pred"]
            current_mse = best["mse"]
            final_coef = best["coef"]

        self.groups_ = selected_groups
        self.coef_ = np.asarray(final_coef if selected_groups else np.zeros(0), dtype=float)
        self.train_mse_ = current_mse
        return self

    def _design_matrix(self, X):
        cols = []
        for g in self.groups_:
            j = g["feature"]
            if g["kind"] == "linear":
                cols.append(X[:, j:j + 1])
            else:
                xj = X[:, j]
                t = g["knot"]
                cols.append(np.column_stack([xj, np.maximum(0.0, xj - t)]))
        if not cols:
            return np.zeros((X.shape[0], 0))
        return np.column_stack(cols)

    def predict(self, X):
        check_is_fitted(self, ["groups_", "coef_", "intercept_"])
        X = np.asarray(X, dtype=float)
        B = self._design_matrix(X)
        if B.shape[1] == 0:
            return np.full(X.shape[0], self.intercept_, dtype=float)
        return self.intercept_ + B @ self.coef_

    def __str__(self):
        check_is_fitted(self, ["groups_", "coef_", "intercept_"])
        lines = ["Greedy Knot Additive Regressor", "Prediction rule:"]
        if len(self.groups_) == 0:
            lines.append(f"y = {self.intercept_:.6f}")
            return "\n".join(lines)

        terms = [f"{self.intercept_:.6f}"]
        idx = 0
        detail = []
        for g in self.groups_:
            j = g["feature"]
            if g["kind"] == "linear":
                c = self.coef_[idx]
                idx += 1
                terms.append(f"{c:+.6f}*x{j}")
                detail.append(f"x{j}: linear coef={c:+.6f}")
            else:
                c1 = self.coef_[idx]
                c2 = self.coef_[idx + 1]
                idx += 2
                t = g["knot"]
                terms.append(f"{c1:+.6f}*x{j}")
                terms.append(f"{c2:+.6f}*max(0, x{j}-{t:.6f})")
                detail.append(
                    f"x{j}: piecewise knot={t:.6f}, linear={c1:+.6f}, hinge={c2:+.6f}"
                )

        lines.append("y = " + " ".join(terms))
        lines.append("")
        lines.append("Active features:")
        lines.extend(f"  - {d}" for d in detail)
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
GreedyKnotAdditiveRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "GreedyKnotAdditiveV1"
model_description = "Greedy sparse additive regressor selecting up to 4 features with linear or one-knot hinge terms from quantile-based thresholds"
model_defs = [(model_shorthand_name, GreedyKnotAdditiveRegressor())]


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
