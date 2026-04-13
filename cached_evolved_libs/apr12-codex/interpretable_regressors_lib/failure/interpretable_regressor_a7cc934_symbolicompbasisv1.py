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


class SymbolicOMPBasisRegressor(BaseEstimator, RegressorMixin):
    """
    Compact symbolic regressor using greedy basis selection + ridge refit.

    Uses a small library of human-readable basis functions:
      linear, square, absolute value, hinge, sinusoid, exp(-abs(.)),
      and a few pairwise interactions.
    """

    def __init__(
        self,
        top_features=8,
        max_terms=10,
        interaction_features=5,
        hinge_quantiles=(0.25, 0.5, 0.75),
        ridge_alpha=5e-3,
        min_mse_gain=1e-4,
        coef_tol=1e-4,
        random_state=42,
    ):
        self.top_features = top_features
        self.max_terms = max_terms
        self.interaction_features = interaction_features
        self.hinge_quantiles = hinge_quantiles
        self.ridge_alpha = ridge_alpha
        self.min_mse_gain = min_mse_gain
        self.coef_tol = coef_tol
        self.random_state = random_state

    def _impute(self, X):
        X = np.asarray(X, dtype=float).copy()
        bad = ~np.isfinite(X)
        if bad.any():
            X[bad] = np.take(self.feature_medians_, np.where(bad)[1])
        return X

    @staticmethod
    def _safe_corr_score(x, y):
        xc = x - np.mean(x)
        yc = y - np.mean(y)
        den = (np.linalg.norm(xc) * np.linalg.norm(yc)) + 1e-12
        return abs(float(np.dot(xc, yc) / den))

    @staticmethod
    def _term_values(X, term):
        kind = term["kind"]
        if kind == "linear":
            return X[:, term["j"]]
        if kind == "square":
            xj = X[:, term["j"]]
            return xj * xj
        if kind == "abs":
            return np.abs(X[:, term["j"]])
        if kind == "sin":
            return np.sin(X[:, term["j"]])
        if kind == "expabs":
            return np.exp(-np.abs(X[:, term["j"]]))
        if kind == "hinge_pos":
            xj = X[:, term["j"]]
            return np.maximum(0.0, xj - term["t"])
        if kind == "hinge_neg":
            xj = X[:, term["j"]]
            return np.maximum(0.0, term["t"] - xj)
        if kind == "interaction":
            return X[:, term["j"]] * X[:, term["k"]]
        raise ValueError(f"Unknown term kind: {kind}")

    @staticmethod
    def _term_name(term):
        kind = term["kind"]
        if kind == "linear":
            return f"x{term['j']}"
        if kind == "square":
            return f"(x{term['j']}^2)"
        if kind == "abs":
            return f"abs(x{term['j']})"
        if kind == "sin":
            return f"sin(x{term['j']})"
        if kind == "expabs":
            return f"exp(-abs(x{term['j']}))"
        if kind == "hinge_pos":
            return f"max(0, x{term['j']} - {term['t']:.4f})"
        if kind == "hinge_neg":
            return f"max(0, {term['t']:.4f} - x{term['j']})"
        if kind == "interaction":
            return f"(x{term['j']} * x{term['k']})"
        return "unknown_term"

    def _ridge_refit(self, y, cols):
        n = y.shape[0]
        if not cols:
            b0 = float(np.mean(y))
            pred = np.full(n, b0, dtype=float)
            return b0, np.array([], dtype=float), pred
        Z = np.column_stack(cols)
        A = np.column_stack([np.ones(n), Z])
        gram = A.T @ A
        reg = float(self.ridge_alpha) * np.eye(gram.shape[0])
        reg[0, 0] = 0.0
        beta = np.linalg.solve(gram + reg, A.T @ y)
        pred = A @ beta
        return float(beta[0]), beta[1:].astype(float), pred

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.feature_medians_ = np.nanmedian(X, axis=0)
        X = self._impute(X)
        n, p = X.shape

        # Feature prescreening by absolute correlation with y
        scores = np.array([self._safe_corr_score(X[:, j], y) for j in range(p)])
        keep = np.argsort(scores)[::-1][: min(self.top_features, p)]
        keep = [int(j) for j in keep]

        # Build symbolic basis library
        candidates = []
        for j in keep:
            candidates.append({"kind": "linear", "j": j})
            candidates.append({"kind": "square", "j": j})
            candidates.append({"kind": "abs", "j": j})
            candidates.append({"kind": "sin", "j": j})
            candidates.append({"kind": "expabs", "j": j})
            xj = X[:, j]
            for q in self.hinge_quantiles:
                thr = float(np.quantile(xj, q))
                candidates.append({"kind": "hinge_pos", "j": j, "t": thr})
                candidates.append({"kind": "hinge_neg", "j": j, "t": thr})

        inter_keep = keep[: min(self.interaction_features, len(keep))]
        for a in range(len(inter_keep)):
            for b in range(a + 1, len(inter_keep)):
                j, k = inter_keep[a], inter_keep[b]
                candidates.append({"kind": "interaction", "j": int(j), "k": int(k)})

        cand_cols = [self._term_values(X, term) for term in candidates]

        # Greedy forward selection with repeated ridge refit
        selected_idx = []
        selected_cols = []
        best_intercept, best_coefs, best_pred = self._ridge_refit(y, selected_cols)
        best_mse = float(np.mean((y - best_pred) ** 2))

        for _ in range(min(self.max_terms, len(candidates))):
            residual = y - best_pred
            best_j = None
            best_score = 0.0
            for j, col in enumerate(cand_cols):
                if j in selected_idx:
                    continue
                cc = col - np.mean(col)
                den = np.linalg.norm(cc) + 1e-12
                score = abs(float(np.dot(residual, cc) / den))
                if score > best_score:
                    best_score = score
                    best_j = j
            if best_j is None:
                break

            trial_cols = selected_cols + [cand_cols[best_j]]
            intercept_t, coefs_t, pred_t = self._ridge_refit(y, trial_cols)
            mse_t = float(np.mean((y - pred_t) ** 2))
            if (best_mse - mse_t) < self.min_mse_gain:
                break

            selected_idx.append(best_j)
            selected_cols = trial_cols
            best_intercept, best_coefs, best_pred, best_mse = intercept_t, coefs_t, pred_t, mse_t

        self.intercept_ = float(best_intercept)
        self.selected_terms_ = []
        self.coefs_ = np.asarray(best_coefs, dtype=float)
        for idx, coef in zip(selected_idx, self.coefs_):
            term = dict(candidates[idx])
            term["name"] = self._term_name(term)
            term["coef"] = float(coef)
            self.selected_terms_.append(term)

        fi = np.zeros(self.n_features_in_, dtype=float)
        for term in self.selected_terms_:
            w = abs(term["coef"])
            if term["kind"] == "interaction":
                fi[term["j"]] += 0.5 * w
                fi[term["k"]] += 0.5 * w
            else:
                fi[term["j"]] += w
        self.feature_importance_ = fi
        self.selected_feature_order_ = np.argsort(fi)[::-1]
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "selected_terms_", "feature_importance_"])
        X = self._impute(X)
        yhat = np.full(X.shape[0], self.intercept_, dtype=float)
        for term in self.selected_terms_:
            yhat += term["coef"] * self._term_values(X, term)
        return yhat

    def __str__(self):
        check_is_fitted(self, ["intercept_", "selected_terms_", "feature_importance_"])
        active = [t for t in self.selected_terms_ if abs(t["coef"]) >= self.coef_tol]
        lines = [
            "SymbolicOMPBasisRegressor",
            "Prediction rule:",
            "  y = intercept + sum_k coef_k * basis_k(x)",
            "",
            f"intercept = {self.intercept_:+.6f}",
            f"active_terms = {len(active)}",
            "",
            "basis terms (in equation order):",
        ]
        for t in active:
            lines.append(f"  {t['coef']:+.6f} * {t['name']}")
        if not active:
            lines.append("  (none)")

        eq = [f"{self.intercept_:+.6f}"] + [f"({t['coef']:+.6f}*{t['name']})" for t in active]
        lines.extend([
            "",
            "compact equation:",
            "  y = " + " + ".join(eq),
        ])
        return "\n".join(lines)



# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SymbolicOMPBasisRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SymbolicOMPBasisV1"
model_description = "Greedy symbolic basis regressor with compact ridge-refit equation over linear, hinge, sinusoid, exp-abs, square, and interaction terms"
model_defs = [(model_shorthand_name, SymbolicOMPBasisRegressor())]


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
