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


class HardThresholdBackboneRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse backbone + tiny nonlinear dictionary:
      y = intercept + sum_j w_j * x_j + sum_k v_k * phi_k(x)

    where phi_k are either one-knot hinge terms or one pairwise interaction.
    The model is intentionally compact and fully explicit for simulatable string output.
    """

    def __init__(
        self,
        max_linear_terms=8,
        max_extra_terms=2,
        top_nonlinear_features=6,
        ridge_alpha=1e-2,
        min_gain=1e-4,
        coef_tol=1e-7,
    ):
        self.max_linear_terms = max_linear_terms
        self.max_extra_terms = max_extra_terms
        self.top_nonlinear_features = top_nonlinear_features
        self.ridge_alpha = ridge_alpha
        self.min_gain = min_gain
        self.coef_tol = coef_tol

    def _impute(self, X):
        X = np.asarray(X, dtype=float).copy()
        mask = ~np.isfinite(X)
        if mask.any():
            X[mask] = np.take(self.feature_medians_, np.where(mask)[1])
        return X

    def _ridge_with_intercept(self, A, y):
        if A.shape[1] == 0:
            return float(np.mean(y)), np.zeros(0, dtype=float)
        y_mean = float(np.mean(y))
        yc = y - y_mean
        a_mean = np.mean(A, axis=0)
        Ac = A - a_mean
        gram = Ac.T @ Ac
        rhs = Ac.T @ yc
        coef = np.linalg.solve(gram + self.ridge_alpha * np.eye(A.shape[1]), rhs)
        intercept = y_mean - float(np.dot(a_mean, coef))
        return intercept, coef

    def _fit_sparse_backbone(self, X, y):
        intercept_full, coef_full = self._ridge_with_intercept(X, y)
        p = X.shape[1]
        k = min(max(1, self.max_linear_terms), p)
        order = np.argsort(np.abs(coef_full))[::-1]
        active = np.sort(order[:k])

        A = X[:, active]
        intercept, coef_active = self._ridge_with_intercept(A, y)
        coef = np.zeros(p, dtype=float)
        coef[active] = coef_active

        pred = intercept + X @ coef
        mse = float(np.mean((y - pred) ** 2))
        if not np.isfinite(mse):
            intercept, coef = intercept_full, coef_full
            active = np.where(np.abs(coef) > self.coef_tol)[0]
            mse = float(np.mean((y - (intercept + X @ coef)) ** 2))
        return intercept, coef, active, mse

    @staticmethod
    def _term_eval(X, term):
        if term["kind"] == "hinge_pos":
            return np.maximum(0.0, X[:, term["j"]] - term["t"])
        if term["kind"] == "hinge_neg":
            return np.maximum(0.0, term["t"] - X[:, term["j"]])
        if term["kind"] == "interaction":
            return X[:, term["j"]] * X[:, term["k"]]
        raise ValueError(f"unknown term kind: {term['kind']}")

    def _build_candidates(self, X, residual, active_linear):
        p = X.shape[1]
        var = np.std(X, axis=0) + 1e-12
        corr = np.abs(np.mean((X / var) * residual[:, None], axis=0))
        feat_order = np.argsort(corr)[::-1]
        top_feats = feat_order[: min(self.top_nonlinear_features, p)]

        candidates = []
        seen = set()

        def add_term(term):
            key = tuple(sorted(term.items()))
            if key not in seen:
                seen.add(key)
                z = self._term_eval(X, term)
                if float(np.std(z)) > 1e-10:
                    candidates.append(term)

        for j in top_feats:
            xj = X[:, j]
            for q in (0.25, 0.5, 0.75):
                t = float(np.quantile(xj, q))
                add_term({"kind": "hinge_pos", "j": int(j), "t": t})
                add_term({"kind": "hinge_neg", "j": int(j), "t": t})

        # allow only a few interaction candidates to preserve compactness
        interaction_feats = list(np.argsort(np.abs(active_linear))[::-1][: min(4, len(active_linear))])
        for i in range(len(interaction_feats)):
            for j in range(i + 1, len(interaction_feats)):
                add_term({"kind": "interaction", "j": int(interaction_feats[i]), "k": int(interaction_feats[j])})

        return candidates

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_features_in_ = X.shape[1]

        self.feature_medians_ = np.nanmedian(X, axis=0)
        X = self._impute(X)

        base_intercept, base_coef, base_active, base_mse = self._fit_sparse_backbone(X, y)
        base_pred = base_intercept + X @ base_coef
        residual = y - base_pred

        candidates = self._build_candidates(X, residual, base_active)

        linear_idx = np.array(base_active, dtype=int)
        selected_terms = []
        selected_term_idx = []
        best_intercept = base_intercept
        best_linear_coef = base_coef[linear_idx].copy()
        best_extra_coef = np.zeros(0, dtype=float)
        best_mse = base_mse

        term_columns = [self._term_eval(X, t) for t in candidates]
        remaining = list(range(len(candidates)))

        for _ in range(self.max_extra_terms):
            if not remaining:
                break

            best_trial = None
            for idx in remaining:
                trial_idx = selected_term_idx + [idx]
                A_parts = [X[:, linear_idx]]
                A_parts.extend(term_columns[i][:, None] for i in trial_idx)
                A = np.column_stack(A_parts)

                intercept, coef_all = self._ridge_with_intercept(A, y)
                pred = intercept + A @ coef_all
                mse = float(np.mean((y - pred) ** 2))

                if (best_trial is None) or (mse < best_trial["mse"]):
                    best_trial = {
                        "idx": idx,
                        "mse": mse,
                        "intercept": intercept,
                        "coef_all": coef_all,
                    }

            if best_trial is None or (best_mse - best_trial["mse"]) < self.min_gain:
                break

            chosen = best_trial["idx"]
            selected_term_idx.append(chosen)
            selected_terms.append(candidates[chosen])
            remaining.remove(chosen)

            n_lin = len(linear_idx)
            best_intercept = float(best_trial["intercept"])
            best_linear_coef = best_trial["coef_all"][:n_lin].copy()
            best_extra_coef = best_trial["coef_all"][n_lin:].copy()
            best_mse = float(best_trial["mse"])

        self.intercept_ = float(best_intercept)
        self.linear_features_ = linear_idx
        self.linear_coef_ = np.asarray(best_linear_coef, dtype=float)

        self.extra_terms_ = []
        self.extra_coef_ = []
        for term, coef in zip(selected_terms, best_extra_coef):
            if abs(float(coef)) > self.coef_tol:
                self.extra_terms_.append(term)
                self.extra_coef_.append(float(coef))
        self.extra_coef_ = np.asarray(self.extra_coef_, dtype=float)

        # Final compact refit after coefficient-pruning
        A_parts = [X[:, self.linear_features_]]
        if self.extra_terms_:
            A_parts.extend(self._term_eval(X, t)[:, None] for t in self.extra_terms_)
        A = np.column_stack(A_parts) if A_parts else np.zeros((X.shape[0], 0), dtype=float)
        self.intercept_, coef_all = self._ridge_with_intercept(A, y)

        n_lin = len(self.linear_features_)
        self.linear_coef_ = coef_all[:n_lin] if n_lin > 0 else np.zeros(0, dtype=float)
        self.extra_coef_ = coef_all[n_lin:] if len(self.extra_terms_) > 0 else np.zeros(0, dtype=float)

        fi = np.zeros(self.n_features_in_, dtype=float)
        for j, coef in zip(self.linear_features_, self.linear_coef_):
            fi[int(j)] += abs(float(coef))
        for term, coef in zip(self.extra_terms_, self.extra_coef_):
            mag = abs(float(coef))
            fi[int(term["j"])] += mag
            if term["kind"] == "interaction":
                fi[int(term["k"])] += mag
        self.feature_importance_ = fi
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "linear_features_", "linear_coef_", "extra_terms_", "extra_coef_"])
        X = self._impute(X)
        y_hat = np.full(X.shape[0], self.intercept_, dtype=float)
        if len(self.linear_features_) > 0:
            y_hat += X[:, self.linear_features_] @ self.linear_coef_
        for term, coef in zip(self.extra_terms_, self.extra_coef_):
            y_hat += coef * self._term_eval(X, term)
        return y_hat

    @staticmethod
    def _term_to_text(term):
        if term["kind"] == "hinge_pos":
            return f"max(0, x{term['j']} - {term['t']:.4f})"
        if term["kind"] == "hinge_neg":
            return f"max(0, {term['t']:.4f} - x{term['j']})"
        if term["kind"] == "interaction":
            return f"(x{term['j']} * x{term['k']})"
        return str(term)

    def __str__(self):
        check_is_fitted(self, ["intercept_", "linear_features_", "linear_coef_", "extra_terms_", "extra_coef_", "feature_importance_"])
        lines = [
            "HardThresholdBackboneRegressor",
            "Prediction equation:",
            f"y = {self.intercept_:+.5f}",
        ]

        terms = []
        for j, c in zip(self.linear_features_, self.linear_coef_):
            if abs(float(c)) > self.coef_tol:
                terms.append((abs(float(c)), f"{c:+.5f} * x{int(j)}"))
        for term, c in zip(self.extra_terms_, self.extra_coef_):
            if abs(float(c)) > self.coef_tol:
                terms.append((abs(float(c)), f"{c:+.5f} * {self._term_to_text(term)}"))
        terms.sort(key=lambda x: -x[0])
        for _, text in terms:
            lines.append("  " + text)

        lines.append("")
        lines.append(f"Model size: {len(terms)} active terms")
        active = [j for j in np.argsort(self.feature_importance_)[::-1] if self.feature_importance_[j] > self.coef_tol]
        lines.append("Feature importance (sum abs contributions):")
        for j in active:
            lines.append(f"  x{j}: {self.feature_importance_[j]:.5f}")
        inactive = [f"x{j}" for j in range(self.n_features_in_) if self.feature_importance_[j] <= self.coef_tol]
        if inactive:
            lines.append("Features with near-zero effect: " + ", ".join(inactive))
        return "\n".join(lines)



# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
HardThresholdBackboneRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "HardThreshBackboneV1"
model_description = "Hard-thresholded sparse linear backbone with up to two greedy nonlinear terms (hinges/interactions) and a joint ridge refit"
model_defs = [(model_shorthand_name, HardThresholdBackboneRegressor())]


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
