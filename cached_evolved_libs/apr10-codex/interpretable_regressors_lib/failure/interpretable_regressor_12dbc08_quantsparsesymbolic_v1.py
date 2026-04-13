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


class QuantizedSparseSymbolicRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse symbolic regressor with quantized coefficients.

    Design goals:
    - Keep the final equation short (few terms).
    - Use only simple transforms that are easy to verbalize.
    - Snap coefficients to a coarse grid for cleaner explanations.

    Candidate term families per feature j:
    1) linear:            z_j
    2) clipped linear:    clip(z_j, -c, c)
    3) tail hinge:        sign(z_j) * max(|z_j| - t, 0)
    where z_j = (x_j - median_j) / iqr_j.
    """

    def __init__(
        self,
        max_terms=6,
        screening_features=18,
        l2=0.2,
        complexity_penalty=0.004,
        min_rel_gain=0.003,
        clip_value=2.0,
        tail_threshold=1.0,
        coef_step=0.25,
    ):
        self.max_terms = max_terms
        self.screening_features = screening_features
        self.l2 = l2
        self.complexity_penalty = complexity_penalty
        self.min_rel_gain = min_rel_gain
        self.clip_value = clip_value
        self.tail_threshold = tail_threshold
        self.coef_step = coef_step

    @staticmethod
    def _safe_scale(x):
        med = float(np.median(x))
        q1 = float(np.quantile(x, 0.25))
        q3 = float(np.quantile(x, 0.75))
        scale = q3 - q1
        if scale <= 1e-12:
            scale = float(np.std(x))
        if scale <= 1e-12:
            scale = 1.0
        return med, scale

    @staticmethod
    def _corr_abs(x, y):
        xc = x - np.mean(x)
        yc = y - np.mean(y)
        denom = (np.linalg.norm(xc) + 1e-12) * (np.linalg.norm(yc) + 1e-12)
        return float(abs(np.dot(xc, yc) / denom))

    def _ridge_fit(self, M, y):
        n = M.shape[0]
        D = np.column_stack([np.ones(n, dtype=float), M])
        reg = float(self.l2) * np.eye(D.shape[1])
        reg[0, 0] = 0.0
        A = D.T @ D + reg
        b = D.T @ y
        beta = np.linalg.solve(A, b)
        return float(beta[0]), np.asarray(beta[1:], dtype=float)

    def _normalize(self, X):
        n, p = X.shape
        medians = np.zeros(p, dtype=float)
        scales = np.ones(p, dtype=float)
        Z = np.zeros_like(X, dtype=float)
        for j in range(p):
            med, scale = self._safe_scale(X[:, j])
            medians[j] = med
            scales[j] = scale
            Z[:, j] = (X[:, j] - med) / scale
        return Z, medians, scales

    def _term_values(self, z, kind):
        if kind == "lin":
            return z
        if kind == "clip":
            c = float(self.clip_value)
            return np.clip(z, -c, c)
        t = float(self.tail_threshold)
        return np.sign(z) * np.maximum(np.abs(z) - t, 0.0)

    def _term_label(self, j, kind):
        if kind == "lin":
            return f"z{xj(j)}"
        if kind == "clip":
            return f"clip(z{xj(j)},-{self.clip_value:.1f},{self.clip_value:.1f})"
        return f"tail(z{xj(j)},t={self.tail_threshold:.1f})"

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape

        Z, medians, scales = self._normalize(X)

        corrs = np.array([self._corr_abs(Z[:, j], y) for j in range(p)], dtype=float)
        k = min(max(1, int(self.screening_features)), p)
        screened = list(np.argsort(-corrs)[:k])

        candidates = []
        for j in screened:
            candidates.append((int(j), "lin"))
            candidates.append((int(j), "clip"))
            candidates.append((int(j), "tail"))

        selected = []
        M = np.zeros((n, 0), dtype=float)
        intercept = float(np.mean(y))
        coef = np.zeros(0, dtype=float)
        base_pred = np.full(n, intercept, dtype=float)
        best_obj = float(np.sum((y - base_pred) ** 2))

        max_terms = max(1, int(self.max_terms))
        for _ in range(max_terms):
            best = None
            for cand in candidates:
                if cand in selected:
                    continue
                j, kind = cand
                v = self._term_values(Z[:, j], kind)
                trial_M = np.column_stack([M, v])
                trial_i, trial_w = self._ridge_fit(trial_M, y)
                pred = trial_i + trial_M @ trial_w
                rss = float(np.sum((y - pred) ** 2))
                # Slightly penalize non-linear terms for readability.
                comp = 1.0 if kind == "lin" else 1.25
                obj = rss * (1.0 + float(self.complexity_penalty) * (len(selected) + comp))
                if best is None or obj < best["obj"]:
                    best = {
                        "cand": cand,
                        "obj": obj,
                        "intercept": trial_i,
                        "coef": trial_w,
                        "M": trial_M,
                    }

            if best is None:
                break

            rel_gain = (best_obj - best["obj"]) / (best_obj + 1e-12)
            if rel_gain < float(self.min_rel_gain):
                break

            selected.append(best["cand"])
            M = best["M"]
            intercept = best["intercept"]
            coef = best["coef"]
            best_obj = best["obj"]

        if len(selected) == 0:
            self.medians_ = medians
            self.scales_ = scales
            self.selected_terms_ = []
            self.coef_ = np.zeros(0, dtype=float)
            self.intercept_ = float(np.mean(y))
            self.n_features_in_ = p
            self.is_fitted_ = True
            return self

        step = max(1e-8, float(self.coef_step))
        qcoef = np.round(coef / step) * step

        keep_idx = [i for i, w in enumerate(qcoef) if abs(float(w)) >= 0.5 * step]
        if len(keep_idx) == 0:
            self.medians_ = medians
            self.scales_ = scales
            self.selected_terms_ = []
            self.coef_ = np.zeros(0, dtype=float)
            self.intercept_ = float(np.mean(y))
            self.n_features_in_ = p
            self.is_fitted_ = True
            return self

        kept_terms = [selected[i] for i in keep_idx]
        kept_coef = np.asarray([qcoef[i] for i in keep_idx], dtype=float)

        cols = []
        for j, kind in kept_terms:
            cols.append(self._term_values(Z[:, j], kind))
        M_kept = np.column_stack(cols)
        intercept = float(np.mean(y - M_kept @ kept_coef))

        self.medians_ = medians
        self.scales_ = scales
        self.selected_terms_ = [(int(j), str(kind)) for j, kind in kept_terms]
        self.coef_ = kept_coef
        self.intercept_ = intercept
        self.n_features_in_ = p
        self.is_fitted_ = True
        return self

    def _transform_feature(self, x, j):
        return (x - self.medians_[j]) / self.scales_[j]

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        yhat = np.full(n, self.intercept_, dtype=float)
        for w, (j, kind) in zip(self.coef_, self.selected_terms_):
            z = self._transform_feature(X[:, j], j)
            yhat += float(w) * self._term_values(z, kind)
        return yhat

    def __str__(self):
        check_is_fitted(self, "is_fitted_")
        lines = ["Quantized Sparse Symbolic Regressor:"]
        lines.append(f"  Base value: {self.intercept_:+.4f}")
        if len(self.selected_terms_) == 0:
            lines.append("  Terms: none")
            return "\n".join(lines)

        lines.append("  z_j = (x_j - median_j) / iqr_j")
        lines.append("  Prediction = base + sum_k w_k * term_k(z_j)")
        for w, (j, kind) in zip(self.coef_, self.selected_terms_):
            if kind == "lin":
                term = f"z{xj(j)}"
            elif kind == "clip":
                term = f"clip(z{xj(j)},-{self.clip_value:.1f},{self.clip_value:.1f})"
            else:
                term = f"sign(z{xj(j)})*max(|z{xj(j)}|-{self.tail_threshold:.1f},0)"
            lines.append(
                f"  Term: {w:+.3f} * {term}"
                f"   [median={self.medians_[j]:.4f}, iqr={self.scales_[j]:.4f}]"
            )
        return "\n".join(lines)


def xj(j):
    return str(int(j))


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
QuantizedSparseSymbolicRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "QuantSparseSymbolic_v1"
model_description = "Sparse robust symbolic regressor with linear/clip/tail terms selected greedily and coefficients quantized to 0.25 steps"
model_defs = [(model_shorthand_name, QuantizedSparseSymbolicRegressor())]


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
