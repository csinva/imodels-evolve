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


class DistilledSparseRidgeRegressor(BaseEstimator, RegressorMixin):
    """
    Distilled sparse linear regressor.

    Steps:
      1) Robust-scale features with median/IQR.
      2) Train a ridge "teacher" using a tiny holdout to pick alpha.
      3) Build a compact student equation by greedy feature pursuit on a
         blended target: 0.7 * y + 0.3 * teacher_prediction.
      4) Refit selected terms with mild ridge shrinkage and quantize weights.
    """

    def __init__(
        self,
        teacher_alpha_grid=(0.01, 0.1, 1.0, 10.0, 50.0),
        student_alpha=0.05,
        max_terms=8,
        blend=0.3,
        coef_quant_step=0.05,
        min_abs_coef=0.05,
    ):
        self.teacher_alpha_grid = teacher_alpha_grid
        self.student_alpha = student_alpha
        self.max_terms = max_terms
        self.blend = blend
        self.coef_quant_step = coef_quant_step
        self.min_abs_coef = min_abs_coef

    @staticmethod
    def _robust_scale(X):
        med = np.median(X, axis=0)
        q25 = np.quantile(X, 0.25, axis=0)
        q75 = np.quantile(X, 0.75, axis=0)
        iqr = np.where((q75 - q25) > 1e-9, q75 - q25, 1.0)
        Z = (X - med) / iqr
        return Z, med, iqr

    @staticmethod
    def _ridge_fit_with_intercept(Z, y, alpha):
        n, p = Z.shape
        D = np.column_stack([np.ones(n, dtype=float), Z])
        reg = float(alpha) * np.eye(p + 1, dtype=float)
        reg[0, 0] = 0.0
        return np.linalg.solve(D.T @ D + reg, D.T @ y)

    @staticmethod
    def _quantize(v, step):
        if step <= 0:
            return float(v)
        return float(np.round(v / step) * step)

    @staticmethod
    def _safe_corr(x, y):
        xc = x - np.mean(x)
        yc = y - np.mean(y)
        denom = (np.linalg.norm(xc) + 1e-12) * (np.linalg.norm(yc) + 1e-12)
        return float(np.dot(xc, yc) / denom)

    def _pick_teacher_alpha(self, Z, y):
        n = Z.shape[0]
        if n < 8:
            return float(self.teacher_alpha_grid[0])
        split = int(max(4, min(n - 2, round(0.8 * n))))
        Ztr, Zva = Z[:split], Z[split:]
        ytr, yva = y[:split], y[split:]
        best = None
        for alpha in self.teacher_alpha_grid:
            beta = self._ridge_fit_with_intercept(Ztr, ytr, alpha=float(alpha))
            pred = beta[0] + Zva @ beta[1:]
            mse = float(np.mean((yva - pred) ** 2))
            if best is None or mse < best[0]:
                best = (mse, float(alpha))
        return best[1]

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape

        Z, med, scale = self._robust_scale(X)
        self.feature_medians_ = med
        self.feature_scales_ = scale

        # Teacher: dense ridge for stronger predictive bias.
        teacher_alpha = self._pick_teacher_alpha(Z, y)
        teacher_beta = self._ridge_fit_with_intercept(Z, y, alpha=teacher_alpha)
        y_teacher = teacher_beta[0] + Z @ teacher_beta[1:]

        # Student target balances fidelity-to-data and fidelity-to-teacher.
        blend = float(np.clip(self.blend, 0.0, 1.0))
        y_tilde = (1.0 - blend) * y + blend * y_teacher

        # Greedy forward selection using correlation with current residual.
        selected = []
        residual = y_tilde - np.mean(y_tilde)
        max_terms = int(max(1, min(int(self.max_terms), p)))
        for _ in range(max_terms):
            best_j = -1
            best_score = 0.0
            for j in range(p):
                if j in selected:
                    continue
                score = abs(self._safe_corr(Z[:, j], residual))
                if score > best_score:
                    best_score = score
                    best_j = int(j)
            if best_j < 0 or best_score < 1e-5:
                break
            selected.append(best_j)
            D = np.column_stack([np.ones(n, dtype=float), Z[:, selected]])
            reg = float(self.student_alpha) * np.eye(D.shape[1], dtype=float)
            reg[0, 0] = 0.0
            beta = np.linalg.solve(D.T @ D + reg, D.T @ y_tilde)
            residual = y_tilde - D @ beta

        if not selected:
            selected = [int(np.argmax(np.abs(teacher_beta[1:])))]

        # Final student refit on original y.
        D = np.column_stack([np.ones(n, dtype=float), Z[:, selected]])
        reg = float(self.student_alpha) * np.eye(D.shape[1], dtype=float)
        reg[0, 0] = 0.0
        beta = np.linalg.solve(D.T @ D + reg, D.T @ y)
        intercept = float(beta[0])
        weights = np.asarray(beta[1:], dtype=float)

        qweights = np.array([self._quantize(w, self.coef_quant_step) for w in weights], dtype=float)
        keep = [i for i, w in enumerate(qweights) if abs(float(w)) >= float(self.min_abs_coef)]
        if not keep:
            j = int(np.argmax(np.abs(weights)))
            keep = [j]
            qweights = np.zeros_like(qweights, dtype=float)
            qweights[j] = self._quantize(float(weights[j]), self.coef_quant_step)
            if abs(qweights[j]) < float(self.min_abs_coef):
                qweights[j] = float(np.sign(weights[j]) * self.min_abs_coef)

        self.intercept_ = intercept
        self.selected_features_ = [int(selected[i]) for i in keep]
        self.weights_ = np.array([qweights[i] for i in keep], dtype=float)
        self.teacher_alpha_ = float(teacher_alpha)
        self.teacher_nonzero_hint_ = int(np.sum(np.abs(teacher_beta[1:]) > 1e-8))
        self.n_features_in_ = p
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        X = np.asarray(X, dtype=float)
        Z = (X - self.feature_medians_) / self.feature_scales_
        pred = np.full(X.shape[0], self.intercept_, dtype=float)
        for w, j in zip(self.weights_, self.selected_features_):
            pred += float(w) * Z[:, int(j)]
        return pred

    def __str__(self):
        check_is_fitted(self, "is_fitted_")
        lines = ["Distilled Sparse Ridge Regressor:"]
        lines.append("  z_j = (x_j - median_j) / IQR_j")
        lines.append(f"  teacher alpha: {self.teacher_alpha_:.3g}")
        lines.append(f"  teacher active features (|coef|>1e-8): {self.teacher_nonzero_hint_}")
        lines.append(f"  intercept: {self.intercept_:+.4f}")
        lines.append("  prediction = intercept + sum_k w_k * z_feature_k")
        order = np.argsort(-np.abs(self.weights_))
        for rank, idx in enumerate(order, 1):
            w = float(self.weights_[idx])
            feat = int(self.selected_features_[idx])
            lines.append(f"  {rank}. {w:+.4f} * z{feat}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
DistilledSparseRidgeRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "DistillSparseRidge_v1"
model_description = "Teacher-student robust linear model: dense ridge teacher distilled into a quantized sparse linear student via residual pursuit"
model_defs = [(model_shorthand_name, DistilledSparseRidgeRegressor())]


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
