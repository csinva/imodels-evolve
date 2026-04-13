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


class PrunedSymbolicRidgeRegressor(BaseEstimator, RegressorMixin):
    """
    Compact symbolic ridge regressor.

    Fits a richer polynomial/interaction basis for predictive strength, then
    prunes and quantizes down to a short equation.
    """

    def __init__(
        self,
        alpha_grid=(0.01, 0.1, 1.0, 10.0),
        screen_features=12,
        max_inter_pairs=8,
        max_terms=10,
        coef_quant_step=0.05,
        min_abs_coef=0.05,
    ):
        self.alpha_grid = alpha_grid
        self.screen_features = screen_features
        self.max_inter_pairs = max_inter_pairs
        self.max_terms = max_terms
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
    def _safe_corr(x, y):
        xc = x - np.mean(x)
        yc = y - np.mean(y)
        denom = (np.linalg.norm(xc) + 1e-12) * (np.linalg.norm(yc) + 1e-12)
        return float(np.dot(xc, yc) / denom)

    @staticmethod
    def _quantize(v, step):
        if step <= 0:
            return float(v)
        return float(np.round(v / step) * step)

    @staticmethod
    def _fit_ridge(D, y, alpha):
        p = D.shape[1]
        reg = float(alpha) * np.eye(p, dtype=float)
        reg[0, 0] = 0.0
        return np.linalg.solve(D.T @ D + reg, D.T @ y)

    def _select_alpha(self, D, y):
        n = D.shape[0]
        if n < 8:
            return float(self.alpha_grid[0])
        split = int(max(4, min(n - 2, round(0.8 * n))))
        Dtr, Dva = D[:split], D[split:]
        ytr, yva = y[:split], y[split:]
        best = None
        for alpha in self.alpha_grid:
            beta = self._fit_ridge(Dtr, ytr, alpha=float(alpha))
            pred = Dva @ beta
            mse = float(np.mean((yva - pred) ** 2))
            if best is None or mse < best[0]:
                best = (mse, float(alpha))
        return best[1]

    def _build_basis(self, Z, y):
        n, p = Z.shape
        # Feature screening by marginal correlation to control complexity/speed.
        corr = np.array([abs(self._safe_corr(Z[:, j], y)) for j in range(p)], dtype=float)
        k = int(max(1, min(int(self.screen_features), p)))
        feat_order = np.argsort(-corr)[:k].astype(int).tolist()

        cols = []
        terms = []

        # Linear + centered quadratic terms.
        for j in feat_order:
            z = Z[:, j]
            cols.append(z)
            terms.append(("lin", int(j)))
            z2 = z * z
            cols.append(z2 - np.mean(z2))
            terms.append(("sq", int(j)))

        # Interaction candidates among top features.
        inter_pool = feat_order[: min(len(feat_order), 6)]
        pair_scores = []
        for a in range(len(inter_pool)):
            for b in range(a + 1, len(inter_pool)):
                j = inter_pool[a]
                kf = inter_pool[b]
                c = Z[:, j] * Z[:, kf]
                score = abs(self._safe_corr(c, y))
                pair_scores.append((score, int(j), int(kf), c))
        pair_scores.sort(key=lambda t: -t[0])
        for _, j, kf, c in pair_scores[: int(max(0, self.max_inter_pairs))]:
            cols.append(c - np.mean(c))
            terms.append(("int", int(j), int(kf)))

        if not cols:
            cols = [np.zeros(n, dtype=float)]
            terms = [("lin", 0)]

        B = np.column_stack(cols)
        D = np.column_stack([np.ones(n, dtype=float), B])
        return D, terms

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        _, p = X.shape

        Z, med, scale = self._robust_scale(X)
        self.feature_medians_ = med
        self.feature_scales_ = scale

        D_full, terms_full = self._build_basis(Z, y)
        alpha = self._select_alpha(D_full, y)
        beta_full = self._fit_ridge(D_full, y, alpha=alpha)
        coef_full = np.asarray(beta_full[1:], dtype=float)

        # Keep only strongest terms for a concise symbolic equation.
        max_terms = int(max(1, min(int(self.max_terms), len(terms_full))))
        keep_order = np.argsort(-np.abs(coef_full))[:max_terms]
        keep_order = [int(i) for i in keep_order if abs(float(coef_full[i])) > 1e-8]
        if not keep_order:
            keep_order = [int(np.argmax(np.abs(coef_full)))]

        keep_terms = [terms_full[i] for i in keep_order]
        B_keep = D_full[:, 1:][:, keep_order]
        D_keep = np.column_stack([np.ones(X.shape[0], dtype=float), B_keep])
        beta_keep = self._fit_ridge(D_keep, y, alpha=alpha)

        intercept = float(beta_keep[0])
        raw_w = np.asarray(beta_keep[1:], dtype=float)
        q_w = np.array([self._quantize(w, self.coef_quant_step) for w in raw_w], dtype=float)

        active = [i for i, w in enumerate(q_w) if abs(float(w)) >= float(self.min_abs_coef)]
        if not active:
            j = int(np.argmax(np.abs(raw_w)))
            active = [j]
            q_w[j] = float(np.sign(raw_w[j]) * max(abs(q_w[j]), self.min_abs_coef))

        self.intercept_ = intercept
        self.term_specs_ = [keep_terms[i] for i in active]
        self.weights_ = np.array([q_w[i] for i in active], dtype=float)
        self.alpha_ = float(alpha)
        self.n_features_in_ = int(p)
        self.is_fitted_ = True
        return self

    def _eval_term(self, Z, spec):
        if spec[0] == "lin":
            return Z[:, int(spec[1])]
        if spec[0] == "sq":
            z = Z[:, int(spec[1])]
            return z * z
        if spec[0] == "int":
            return Z[:, int(spec[1])] * Z[:, int(spec[2])]
        raise ValueError(f"Unknown term spec: {spec}")

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        X = np.asarray(X, dtype=float)
        Z = (X - self.feature_medians_) / self.feature_scales_
        pred = np.full(X.shape[0], self.intercept_, dtype=float)
        for w, spec in zip(self.weights_, self.term_specs_):
            pred += float(w) * self._eval_term(Z, spec)
        return pred

    def __str__(self):
        check_is_fitted(self, "is_fitted_")
        lines = ["Pruned Symbolic Ridge Regressor:"]
        lines.append("  z_j = (x_j - median_j) / IQR_j")
        lines.append(f"  alpha: {self.alpha_:.3g}")
        lines.append(f"  intercept: {self.intercept_:+.4f}")
        lines.append("  prediction = intercept + sum_k w_k * phi_k(z)")
        order = np.argsort(-np.abs(self.weights_))
        for rank, idx in enumerate(order, 1):
            w = float(self.weights_[idx])
            spec = self.term_specs_[idx]
            if spec[0] == "lin":
                rhs = f"z{int(spec[1])}"
            elif spec[0] == "sq":
                rhs = f"z{int(spec[1])}^2"
            else:
                rhs = f"z{int(spec[1])}*z{int(spec[2])}"
            lines.append(f"  {rank}. {w:+.4f} * {rhs}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
PrunedSymbolicRidgeRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "PrunedSymRidge_v1"
model_description = "Robust-scaled symbolic ridge over linear/squared/interaction terms with coefficient-magnitude pruning to a compact quantized equation"
model_defs = [(model_shorthand_name, PrunedSymbolicRidgeRegressor())]


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
