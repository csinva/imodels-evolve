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


class SparseLinearHingeRefineRegressor(BaseEstimator, RegressorMixin):
    """
    Compact sparse regressor with a linear backbone and a few hinge refinements.

    Structure:
      - sparse linear backbone: sum_j w_j * x_j over selected features
      - optional hinge corrections: a_k * max(0, x_j - t) or a_k * max(0, t - x_j)
      - optional single interaction term: b * (x_i * x_j)

    The fit loop is greedy and keeps the model intentionally small so the final
    string form remains directly simulatable.
    """

    def __init__(
        self,
        max_linear_features=8,
        max_hinges=2,
        max_interactions=1,
        hinge_feature_screen=8,
        knot_quantiles=(0.2, 0.4, 0.6, 0.8),
        knot_round_decimals=2,
        min_relative_gain=1e-3,
        coef_tol=1e-8,
        quantization=0.01,
    ):
        self.max_linear_features = max_linear_features
        self.max_hinges = max_hinges
        self.max_interactions = max_interactions
        self.hinge_feature_screen = hinge_feature_screen
        self.knot_quantiles = knot_quantiles
        self.knot_round_decimals = knot_round_decimals
        self.min_relative_gain = min_relative_gain
        self.coef_tol = coef_tol
        self.quantization = quantization

    @staticmethod
    def _eval_term(X, term):
        kind = term[0]
        if kind == "lin":
            return X[:, term[1]]
        if kind == "hinge_pos":
            return np.maximum(0.0, X[:, term[1]] - term[2])
        if kind == "hinge_neg":
            return np.maximum(0.0, term[2] - X[:, term[1]])
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
        coef = np.asarray(coef, dtype=float)
        pred = D @ coef
        mse = float(np.mean((target - pred) ** 2))
        return coef, pred, mse

    def _knot_candidates(self, x):
        if x.size == 0:
            return [0.0]
        qs = np.quantile(x, self.knot_quantiles)
        qs = np.round(np.asarray(qs, dtype=float), int(self.knot_round_decimals))
        uniq = np.unique(qs)
        if uniq.size == 0:
            return [0.0]
        return [float(v) for v in uniq]

    @staticmethod
    def _round_to_step(values, step):
        if step <= 0:
            return values
        return step * np.round(values / step)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        self.n_features_in_ = X.shape[1]

        self.intercept_ = float(y.mean())
        target = y - self.intercept_

        n_samples, n_features = X.shape
        centered_X = X - X.mean(axis=0, keepdims=True)
        x_norms = np.linalg.norm(centered_X, axis=0) + 1e-12
        target_norm = float(np.linalg.norm(target)) + 1e-12

        corrs = np.abs((centered_X.T @ target) / (x_norms * target_norm))
        n_linear = min(int(self.max_linear_features), n_features)
        linear_features = [int(i) for i in np.argsort(corrs)[::-1][:n_linear]]

        term_defs = [("lin", j) for j in linear_features]
        coef, current_pred, prev_mse = self._fit_with_terms(X, target, term_defs)

        for _ in range(int(self.max_hinges)):
            resid = target - current_pred
            resid_norm = float(np.linalg.norm(resid)) + 1e-12
            resid_corrs = np.abs((centered_X.T @ resid) / (x_norms * resid_norm))
            n_screen = min(int(self.hinge_feature_screen), n_features)
            feat_candidates = [int(i) for i in np.argsort(resid_corrs)[::-1][:n_screen]]

            best_term = None
            best_terms = None
            best_pred = None
            best_mse = prev_mse

            for feat_idx in feat_candidates:
                knots = self._knot_candidates(X[:, feat_idx])
                for t in knots:
                    for kind in ("hinge_pos", "hinge_neg"):
                        candidate = (kind, int(feat_idx), float(t))
                        if candidate in term_defs:
                            continue
                        trial_terms = term_defs + [candidate]
                        _, trial_pred, trial_mse = self._fit_with_terms(X, target, trial_terms)
                        if trial_mse < best_mse - 1e-12:
                            best_mse = trial_mse
                            best_term = candidate
                            best_terms = trial_terms
                            best_pred = trial_pred

            if best_term is None:
                break

            rel_gain = (prev_mse - best_mse) / (abs(prev_mse) + 1e-12)
            if rel_gain < self.min_relative_gain:
                break

            term_defs = best_terms
            current_pred = best_pred
            prev_mse = best_mse

        used_for_interactions = sorted({
            term[1] for term in term_defs if term[0] in {"lin", "hinge_pos", "hinge_neg"}
        })
        for _ in range(int(self.max_interactions)):
            if len(used_for_interactions) < 2:
                break

            best_term = None
            best_terms = None
            best_pred = None
            best_mse = prev_mse

            for i in range(len(used_for_interactions)):
                for j in range(i + 1, len(used_for_interactions)):
                    fi = int(used_for_interactions[i])
                    fj = int(used_for_interactions[j])
                    cand = ("int", fi, fj)
                    if cand in term_defs:
                        continue
                    trial_terms = term_defs + [cand]
                    _, trial_pred, trial_mse = self._fit_with_terms(X, target, trial_terms)
                    if trial_mse < best_mse - 1e-12:
                        best_mse = trial_mse
                        best_term = cand
                        best_terms = trial_terms
                        best_pred = trial_pred

            if best_term is None:
                break

            rel_gain = (prev_mse - best_mse) / (abs(prev_mse) + 1e-12)
            if rel_gain < 0.5 * self.min_relative_gain:
                break

            term_defs = best_terms
            current_pred = best_pred
            prev_mse = best_mse

        coef, _, _ = self._fit_with_terms(X, target, term_defs)
        if coef.size:
            keep = np.abs(coef) > self.coef_tol
            if not np.all(keep):
                term_defs = [term for term, k in zip(term_defs, keep) if k]
                coef, _, _ = self._fit_with_terms(X, target, term_defs)

        if coef.size and self.quantization > 0:
            coef = self._round_to_step(coef, float(self.quantization))
            keep = np.abs(coef) > self.coef_tol
            term_defs = [term for term, k in zip(term_defs, keep) if k]
            coef = coef[keep]

            if len(term_defs) > 0:
                coef, _, _ = self._fit_with_terms(X, target, term_defs)
                coef = self._round_to_step(coef, float(self.quantization))
                keep = np.abs(coef) > self.coef_tol
                term_defs = [term for term, k in zip(term_defs, keep) if k]
                coef = coef[keep]

        if self.quantization > 0:
            self.intercept_ = float(self._round_to_step(np.array([self.intercept_]), float(self.quantization))[0])

        self.term_defs_ = list(term_defs)
        self.coef_ = np.asarray(coef, dtype=float)

        used_feats = set()
        for term in self.term_defs_:
            if term[0] in {"lin", "hinge_pos", "hinge_neg"}:
                used_feats.add(int(term[1]))
            elif term[0] == "int":
                used_feats.add(int(term[1]))
                used_feats.add(int(term[2]))
        self.selected_features_ = sorted(used_feats)

        feat_importance = np.zeros(self.n_features_in_, dtype=float)
        for c, term in zip(self.coef_, self.term_defs_):
            cabs = abs(float(c))
            if term[0] in {"lin", "hinge_pos", "hinge_neg"}:
                feat_importance[int(term[1])] += cabs
            elif term[0] == "int":
                feat_importance[int(term[1])] += 0.5 * cabs
                feat_importance[int(term[2])] += 0.5 * cabs
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
        if kind == "hinge_pos":
            return f"max(0, x{term[1]} - {term[2]:.2f})"
        if kind == "hinge_neg":
            return f"max(0, {term[2]:.2f} - x{term[1]})"
        if kind == "int":
            return f"(x{term[1]} * x{term[2]})"
        return str(term)

    def __str__(self):
        check_is_fitted(self, ["intercept_", "term_defs_", "coef_", "feature_importance_"])

        lines = [
            "Sparse Linear + Hinge Refinement Regressor",
            "Prediction rule: y = intercept + sum_k coef_k * term_k",
            f"intercept: {self.intercept_:+.4f}",
            "",
            "Active terms (largest |coef| first):",
        ]

        if len(self.term_defs_) == 0:
            lines.append("  (no active terms; constant predictor)")
        else:
            order = np.argsort(np.abs(self.coef_))[::-1]
            for rank, idx in enumerate(order, 1):
                coef = float(self.coef_[idx])
                term_str = self._term_to_string(self.term_defs_[idx])
                lines.append(f"  {rank:2d}. {coef:+.4f} * {term_str}")

        lines.append("")
        lines.append(f"Model size: {len(self.term_defs_)} terms")
        if self.selected_features_:
            lines.append("Active features: " + ", ".join(f"x{i}" for i in self.selected_features_))

        unused = [f"x{i}" for i in range(self.n_features_in_) if i not in set(self.selected_features_)]
        if unused and len(unused) <= 30:
            lines.append("Unused features (zero direct effect): " + ", ".join(unused))

        lines.append("Computation note: evaluate each listed term, multiply by its coefficient, then add intercept.")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparseLinearHingeRefineRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SparseLinHingeRefine_v1"
model_description = "Sparse linear backbone with greedy one-sided hinge refinements, optional interaction, and coefficient quantization for simulatability"
model_defs = [(model_shorthand_name, SparseLinearHingeRefineRegressor())]


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
