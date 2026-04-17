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


class ResidualBasisRidgeRegressor(BaseEstimator, RegressorMixin):
    """Ridge linear backbone plus a few greedy residual basis terms."""

    def __init__(
        self,
        max_linear_features=24,
        max_residual_terms=4,
        interaction_top_k=8,
        candidate_quantiles=(0.2, 0.35, 0.5, 0.65, 0.8),
        ridge_lambda=0.2,
        min_rel_gain=0.004,
        negligible_coef_eps=0.03,
    ):
        self.max_linear_features = max_linear_features
        self.max_residual_terms = max_residual_terms
        self.interaction_top_k = interaction_top_k
        self.candidate_quantiles = candidate_quantiles
        self.ridge_lambda = ridge_lambda
        self.min_rel_gain = min_rel_gain
        self.negligible_coef_eps = negligible_coef_eps

    def _safe_abs_corr(self, a, b):
        sa = float(np.std(a))
        sb = float(np.std(b))
        if sa < 1e-12 or sb < 1e-12:
            return 0.0
        c = np.corrcoef(a, b)[0, 1]
        if not np.isfinite(c):
            return 0.0
        return float(abs(c))

    def _solve_ridge(self, Z, y):
        z_mean = np.mean(Z, axis=0)
        z_std = np.std(Z, axis=0)
        z_std[z_std < 1e-12] = 1.0
        Zs = (Z - z_mean) / z_std

        y_mean = float(np.mean(y))
        y_c = y - y_mean

        gram = Zs.T @ Zs + self.ridge_lambda * np.eye(Zs.shape[1], dtype=float)
        coef_std = np.linalg.solve(gram, Zs.T @ y_c)
        coef = coef_std / z_std
        intercept = float(y_mean - np.dot(coef, z_mean))
        return intercept, coef

    def _term_values(self, X, term):
        kind = term["kind"]
        if kind == "hinge_pos":
            j = term["feature"]
            t = term["threshold"]
            return np.maximum(0.0, X[:, j] - t)
        if kind == "hinge_neg":
            j = term["feature"]
            t = term["threshold"]
            return np.maximum(0.0, t - X[:, j])
        if kind == "interaction":
            i = term["feature_i"]
            j = term["feature_j"]
            return X[:, i] * X[:, j]
        raise ValueError(f"Unknown term kind: {kind}")

    def _build_residual_candidates(self, X, residual, ranked_feats):
        candidates = []

        nonlinear_feats = ranked_feats[: min(self.interaction_top_k, len(ranked_feats))]
        for feat in nonlinear_feats:
            xj = X[:, feat]
            quantiles = np.unique(np.round(np.quantile(xj, self.candidate_quantiles), 8))
            for thr in quantiles:
                candidates.append({"kind": "hinge_pos", "feature": int(feat), "threshold": float(thr)})
                candidates.append({"kind": "hinge_neg", "feature": int(feat), "threshold": float(thr)})

        inter_feats = ranked_feats[: min(self.interaction_top_k, len(ranked_feats))]
        for a in range(len(inter_feats)):
            for b in range(a + 1, len(inter_feats)):
                i = int(inter_feats[a])
                j = int(inter_feats[b])
                prod = X[:, i] * X[:, j]
                if self._safe_abs_corr(prod, residual) > 0.03:
                    candidates.append({"kind": "interaction", "feature_i": i, "feature_j": j})

        return candidates

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        corr = np.array([self._safe_abs_corr(X[:, j], y) for j in range(n_features)], dtype=float)
        ranked = list(np.argsort(corr)[::-1].astype(int))
        linear_feats = ranked[: min(self.max_linear_features, n_features)]

        Z_linear = X[:, linear_feats]
        self.linear_intercept_, coef_sel = self._solve_ridge(Z_linear, y)
        self.linear_coef_ = np.zeros(n_features, dtype=float)
        self.linear_coef_[np.asarray(linear_feats, dtype=int)] = coef_sel

        pred = self.linear_intercept_ + X @ self.linear_coef_
        residual = y - pred

        candidates = self._build_residual_candidates(X, residual, ranked)
        chosen_terms = []
        used = set()

        for _ in range(self.max_residual_terms):
            base_sse = float(np.dot(residual, residual))
            min_gain = self.min_rel_gain * (base_sse + 1e-12)
            best_idx = None
            best_sse = base_sse

            for idx, cand in enumerate(candidates):
                if idx in used:
                    continue
                z = self._term_values(X, cand)
                denom = float(np.dot(z, z)) + self.ridge_lambda
                if denom < 1e-12:
                    continue
                alpha = float(np.dot(residual, z) / denom)
                new_residual = residual - alpha * z
                sse = float(np.dot(new_residual, new_residual))
                if sse + min_gain < best_sse:
                    best_sse = sse
                    best_idx = idx

            if best_idx is None:
                break

            used.add(best_idx)
            chosen_terms.append(candidates[best_idx])
            z_best = self._term_values(X, candidates[best_idx])
            alpha_best = float(np.dot(residual, z_best) / (float(np.dot(z_best, z_best)) + self.ridge_lambda))
            residual = residual - alpha_best * z_best

        if chosen_terms:
            Z_extra = np.column_stack([self._term_values(X, t) for t in chosen_terms])
            Z_full = np.column_stack([X, Z_extra])
            self.intercept_, coef_full = self._solve_ridge(Z_full, y)
            self.linear_coef_ = coef_full[:n_features]
            self.extra_coefs_ = coef_full[n_features:]
        else:
            self.intercept_ = self.linear_intercept_
            self.extra_coefs_ = np.zeros(0, dtype=float)

        self.extra_terms_ = chosen_terms

        abs_linear = np.abs(self.linear_coef_)
        extra_importance = np.zeros(n_features, dtype=float)
        for coef, term in zip(self.extra_coefs_, self.extra_terms_):
            if term["kind"] in ("hinge_pos", "hinge_neg"):
                extra_importance[term["feature"]] += abs(float(coef))
            elif term["kind"] == "interaction":
                extra_importance[term["feature_i"]] += 0.5 * abs(float(coef))
                extra_importance[term["feature_j"]] += 0.5 * abs(float(coef))

        self.feature_importances_ = abs_linear + extra_importance
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "linear_coef_", "extra_terms_", "extra_coefs_"])
        X = np.asarray(X, dtype=float)
        pred = self.intercept_ + X @ self.linear_coef_
        for coef, term in zip(self.extra_coefs_, self.extra_terms_):
            pred += float(coef) * self._term_values(X, term)
        return pred

    def _term_str(self, term, coef):
        if term["kind"] == "hinge_pos":
            j = term["feature"]
            t = term["threshold"]
            return f"{coef:+.5f}*max(0, x{j}-{t:.5f})"
        if term["kind"] == "hinge_neg":
            j = term["feature"]
            t = term["threshold"]
            return f"{coef:+.5f}*max(0, {t:.5f}-x{j})"
        i = term["feature_i"]
        j = term["feature_j"]
        return f"{coef:+.5f}*(x{i}*x{j})"

    def __str__(self):
        check_is_fitted(self, ["intercept_", "linear_coef_", "extra_terms_", "extra_coefs_", "feature_importances_"])
        active_linear = [j for j in range(self.n_features_in_) if abs(self.linear_coef_[j]) > self.negligible_coef_eps]
        maybe_small = [j for j in range(self.n_features_in_) if abs(self.linear_coef_[j]) <= self.negligible_coef_eps]

        lines = [
            "Residual Basis Ridge Regressor",
            "Model form: y = intercept + linear terms + few residual basis corrections",
            f"Intercept: {self.intercept_:+.5f}",
            f"Residual basis terms: {len(self.extra_terms_)}",
            "",
            "Linear coefficients:",
        ]

        if active_linear:
            order = sorted(active_linear, key=lambda j: abs(self.linear_coef_[j]), reverse=True)
            for j in order:
                lines.append(f"  x{j}: {self.linear_coef_[j]:+.5f}")
        else:
            lines.append("  (all near zero)")

        lines.append("")
        lines.append("Residual basis corrections:")
        if self.extra_terms_:
            for idx, (coef, term) in enumerate(zip(self.extra_coefs_, self.extra_terms_), 1):
                lines.append(f"  term{idx}: {self._term_str(term, float(coef))}")
        else:
            lines.append("  (none)")

        eq_parts = [f"{self.intercept_:+.5f}"]
        for j in sorted(active_linear, key=lambda k: abs(self.linear_coef_[k]), reverse=True):
            eq_parts.append(f"{self.linear_coef_[j]:+.5f}*x{j}")
        for coef, term in zip(self.extra_coefs_, self.extra_terms_):
            eq_parts.append(self._term_str(term, float(coef)))

        order = np.argsort(self.feature_importances_)[::-1]
        top = [f"x{j}:{self.feature_importances_[j]:.4f}" for j in order[: min(10, self.n_features_in_)]]

        lines.extend(
            [
                "",
                "Compact equation:",
                "  y = " + " ".join(eq_parts),
                "",
                "Most influential features (coefficient/basis magnitude):",
                "  " + ", ".join(top),
                "Features with negligible linear effect:",
                "  " + (", ".join(f"x{j}" for j in maybe_small) if maybe_small else "none"),
                "",
                "Simulation recipe: compute intercept + listed linear terms + listed residual basis terms.",
            ]
        )
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
ResidualBasisRidgeRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "ResidualBasisRidge_v1"
model_description = "Ridge linear backbone over top-correlated features plus greedy residual hinge/interaction basis corrections with explicit symbolic equation"
model_defs = [(model_shorthand_name, ResidualBasisRidgeRegressor())]


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
