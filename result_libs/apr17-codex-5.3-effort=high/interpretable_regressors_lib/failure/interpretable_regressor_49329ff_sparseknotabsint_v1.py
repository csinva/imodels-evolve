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


class SparseAdaptiveKnotRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse additive regressor with per-feature learned knots.

    For each selected feature x_j, the model uses two basis terms:
      - linear term: x_j
      - adaptive kink term: |x_j - t_j|

    This keeps the representation compact while allowing piecewise-linear effects
    with an explicit, human-readable threshold t_j. Optionally, a tiny number of
    pairwise interactions can be added for non-additive structure.
    """

    def __init__(
        self,
        max_features=4,
        max_interactions=1,
        feature_screen=24,
        knot_quantiles=(0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90),
        min_relative_gain=1e-3,
        coef_tol=1e-8,
    ):
        self.max_features = max_features
        self.max_interactions = max_interactions
        self.feature_screen = feature_screen
        self.knot_quantiles = knot_quantiles
        self.min_relative_gain = min_relative_gain
        self.coef_tol = coef_tol

    @staticmethod
    def _knot_candidates(x, quantiles):
        if x.size == 0:
            return [0.0]
        qs = np.quantile(x, quantiles)
        uniq = np.unique(np.asarray(qs, dtype=float))
        if uniq.size == 0:
            return [0.0]
        return [float(v) for v in uniq]

    @staticmethod
    def _eval_term(X, term):
        kind = term[0]
        if kind == "lin":
            return X[:, term[1]]
        if kind == "abs":
            return np.abs(X[:, term[1]] - term[2])
        if kind == "int":
            return X[:, term[1]] * X[:, term[2]]
        raise ValueError(f"Unknown term kind: {kind}")

    def _design_from_terms(self, X, term_defs):
        if not term_defs:
            return np.zeros((X.shape[0], 0), dtype=float)
        cols = [self._eval_term(X, term).astype(float) for term in term_defs]
        return np.column_stack(cols)

    def _fit_with_terms(self, X, target, term_defs):
        if not term_defs:
            pred = np.zeros_like(target)
            mse = float(np.mean((target - pred) ** 2))
            return np.zeros(0, dtype=float), pred, mse
        D = self._design_from_terms(X, term_defs)
        coef, *_ = np.linalg.lstsq(D, target, rcond=None)
        pred = D @ coef
        mse = float(np.mean((target - pred) ** 2))
        return np.asarray(coef, dtype=float), pred, mse

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        self.n_features_in_ = X.shape[1]

        self.intercept_ = float(y.mean())
        target = y - self.intercept_

        n_samples, n_features = X.shape
        centered_X = X - X.mean(axis=0, keepdims=True)
        x_norms = np.linalg.norm(centered_X, axis=0) + 1e-12

        selected_features = []
        selected_terms = []
        _, current_pred, prev_mse = self._fit_with_terms(X, target, selected_terms)

        max_rounds = min(self.max_features, n_features)
        for _ in range(max_rounds):
            resid = target - current_pred
            y_norm = float(np.linalg.norm(resid)) + 1e-12
            corrs = np.abs((centered_X.T @ resid) / (x_norms * y_norm))
            top_k = min(self.feature_screen, n_features)
            candidate_feats = np.argsort(corrs)[::-1][:top_k]

            best_feature = None
            best_terms = None
            best_pred = None
            best_mse = prev_mse

            for feat_idx in candidate_feats:
                feat_idx = int(feat_idx)
                if feat_idx in selected_features:
                    continue
                xj = X[:, feat_idx]
                knots = self._knot_candidates(xj, self.knot_quantiles)
                for knot in knots:
                    trial_terms = selected_terms + [("lin", feat_idx), ("abs", feat_idx, float(knot))]
                    _, trial_pred, mse = self._fit_with_terms(X, target, trial_terms)
                    if mse < best_mse - 1e-12:
                        best_mse = mse
                        best_feature = feat_idx
                        best_terms = trial_terms
                        best_pred = trial_pred

            if best_feature is None:
                break

            rel_gain = (prev_mse - best_mse) / (abs(prev_mse) + 1e-12)
            if selected_features and rel_gain < self.min_relative_gain:
                break

            selected_features.append(best_feature)
            selected_terms = best_terms
            current_pred = best_pred
            prev_mse = best_mse

        # Optional tiny interaction budget to recover non-additive structure.
        interaction_terms = []
        for _ in range(self.max_interactions):
            if len(selected_features) < 2:
                break

            best_pair_term = None
            best_terms = None
            best_pred = None
            best_mse = prev_mse

            for i in range(len(selected_features)):
                for j in range(i + 1, len(selected_features)):
                    fi = int(selected_features[i])
                    fj = int(selected_features[j])
                    term = ("int", fi, fj)
                    if term in interaction_terms:
                        continue
                    trial_terms = selected_terms + [term]
                    _, trial_pred, mse = self._fit_with_terms(X, target, trial_terms)
                    if mse < best_mse - 1e-12:
                        best_mse = mse
                        best_pair_term = term
                        best_terms = trial_terms
                        best_pred = trial_pred

            if best_pair_term is None:
                break

            rel_gain = (prev_mse - best_mse) / (abs(prev_mse) + 1e-12)
            if rel_gain < 0.5 * self.min_relative_gain:
                break

            interaction_terms.append(best_pair_term)
            selected_terms = best_terms
            current_pred = best_pred
            prev_mse = best_mse

        coef, _, _ = self._fit_with_terms(X, target, selected_terms)
        if coef.size:
            keep = np.abs(coef) > self.coef_tol
            if not np.all(keep):
                selected_terms = [term for term, k in zip(selected_terms, keep) if k]
                coef, _, _ = self._fit_with_terms(X, target, selected_terms)

        self.term_defs_ = selected_terms
        self.coef_ = np.asarray(coef, dtype=float)
        self.selected_features_ = sorted({
            term[1] for term in self.term_defs_ if term[0] in {"lin", "abs"}
        } | {
            idx for term in self.term_defs_ if term[0] == "int" for idx in (term[1], term[2])
        })

        feat_importance = np.zeros(self.n_features_in_, dtype=float)
        for c, term in zip(self.coef_, self.term_defs_):
            cabs = abs(float(c))
            if term[0] in {"lin", "abs"}:
                feat_importance[term[1]] += cabs
            elif term[0] == "int":
                feat_importance[term[1]] += 0.5 * cabs
                feat_importance[term[2]] += 0.5 * cabs
        self.feature_importance_ = feat_importance
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "term_defs_", "coef_"])
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if len(self.term_defs_) == 0:
            return np.full(X.shape[0], self.intercept_, dtype=float)
        D = self._design_from_terms(X, self.term_defs_)
        return self.intercept_ + D @ self.coef_

    @staticmethod
    def _term_to_string(term):
        kind = term[0]
        if kind == "lin":
            return f"x{term[1]}"
        if kind == "abs":
            return f"|x{term[1]} - {term[2]:.4f}|"
        if kind == "int":
            return f"(x{term[1]} * x{term[2]})"
        return str(term)

    def __str__(self):
        check_is_fitted(self, ["intercept_", "term_defs_", "coef_", "feature_importance_"])

        lines = [
            "Sparse Adaptive-Knot Regressor",
            "Prediction rule: y = intercept + sum_k coef_k * term_k",
            f"intercept: {self.intercept_:+.6f}",
            "",
            "Active terms:",
        ]

        if len(self.term_defs_) == 0:
            lines.append("  (no active terms; constant predictor)")
        else:
            order = np.argsort(np.abs(self.coef_))[::-1]
            for rank, idx in enumerate(order, 1):
                coef = float(self.coef_[idx])
                term = self._term_to_string(self.term_defs_[idx])
                lines.append(f"  {rank:2d}. {coef:+.6f} * {term}")

        lines.append("")
        lines.append("Per-feature kink thresholds (from |xj - t| terms):")
        knot_lines = []
        for term in self.term_defs_:
            if term[0] == "abs":
                knot_lines.append(f"  x{term[1]} kink at t={term[2]:.4f}")
        if knot_lines:
            lines.extend(sorted(set(knot_lines)))
        else:
            lines.append("  (none)")

        lines.append("")
        lines.append(f"Model size: {len(self.term_defs_)} terms")
        if self.selected_features_:
            lines.append("Active features: " + ", ".join(f"x{i}" for i in self.selected_features_))

        unused = [f"x{i}" for i in range(self.n_features_in_) if i not in set(self.selected_features_)]
        if unused and len(unused) <= 30:
            lines.append("Unused features (zero direct effect): " + ", ".join(unused))

        lines.append("Computation note: evaluate each listed term, multiply by its coefficient, add intercept.")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparseAdaptiveKnotRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SparseKnotAbsInt_v1"
model_description = "Forward-selected sparse additive model with per-feature adaptive |x-t| kink terms plus at most one interaction"
model_defs = [(model_shorthand_name, SparseAdaptiveKnotRegressor())]


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
