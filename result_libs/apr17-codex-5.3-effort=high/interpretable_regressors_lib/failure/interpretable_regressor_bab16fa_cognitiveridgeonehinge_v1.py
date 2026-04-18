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


class CognitiveRidgeOneHingeRegressor(BaseEstimator, RegressorMixin):
    """
    Dense ridge backbone with one optional hinge residual.

    Design goals:
      - Keep the predictive strength of ridge on tabular datasets.
      - Add at most one piecewise term to capture clear threshold effects.
      - Select a validation-tuned linear pruning level that balances RMSE with
        cognitive complexity (number of active terms).
    """

    def __init__(
        self,
        val_fraction=0.2,
        alphas=(0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0),
        hinge_top_features=8,
        hinge_quantiles=(0.2, 0.35, 0.5, 0.65, 0.8),
        linear_prune_quantiles=(0.0, 0.1, 0.2, 0.3, 0.4),
        complexity_penalty=0.001,
        min_hinge_relative_gain=1e-3,
        coef_tol=1e-10,
        decimals=6,
        random_state=0,
    ):
        self.val_fraction = val_fraction
        self.alphas = alphas
        self.hinge_top_features = hinge_top_features
        self.hinge_quantiles = hinge_quantiles
        self.linear_prune_quantiles = linear_prune_quantiles
        self.complexity_penalty = complexity_penalty
        self.min_hinge_relative_gain = min_hinge_relative_gain
        self.coef_tol = coef_tol
        self.decimals = decimals
        self.random_state = random_state

    @staticmethod
    def _rmse(y_true, y_pred):
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    def _split_indices(self, n):
        if n <= 80:
            idx = np.arange(n, dtype=int)
            return idx, idx
        rng = np.random.RandomState(self.random_state)
        order = rng.permutation(n)
        n_val = int(round(float(self.val_fraction) * n))
        n_val = min(max(n_val, 40), n - 40)
        va_idx = order[:n_val]
        tr_idx = order[n_val:]
        if tr_idx.size == 0:
            tr_idx = va_idx
        return tr_idx, va_idx

    @staticmethod
    def _solve_ridge_with_intercept(D, y, alpha):
        n, p = D.shape
        if p == 0:
            return float(np.mean(y)), np.zeros(0, dtype=float)
        scale = np.std(D, axis=0).astype(float)
        scale[scale < 1e-12] = 1.0
        Ds = D / scale
        A = np.column_stack([np.ones(n, dtype=float), Ds])
        reg = np.eye(p + 1, dtype=float)
        reg[0, 0] = 0.0
        lhs = A.T @ A + float(max(alpha, 0.0)) * reg
        rhs = A.T @ y
        try:
            sol = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        intercept = float(sol[0])
        coefs = np.asarray(sol[1:], dtype=float) / scale
        return intercept, coefs

    @staticmethod
    def _hinge_column(X, spec):
        kind, feat, thr = spec
        xj = X[:, int(feat)]
        if kind == "hinge_pos":
            return np.maximum(0.0, xj - float(thr))
        return np.maximum(0.0, float(thr) - xj)

    @staticmethod
    def _hinge_str(spec, decimals):
        kind, feat, thr = spec
        if kind == "hinge_pos":
            return f"max(0, x{feat} - {float(thr):.{decimals}f})"
        return f"max(0, {float(thr):.{decimals}f} - x{feat})"

    def _objective(self, rmse, n_active_linear, has_hinge):
        term_count = int(n_active_linear) + (1 if has_hinge else 0)
        penalty = 1.0 + float(self.complexity_penalty) * max(0, term_count - 8)
        return float(rmse) * penalty

    def _fit_design(self, X, y, alpha, active_mask, hinge_spec):
        active_idx = np.where(active_mask)[0]
        cols = []
        for j in active_idx:
            cols.append(X[:, int(j)])
        if hinge_spec is not None:
            cols.append(self._hinge_column(X, hinge_spec))
        D = np.column_stack(cols) if cols else np.zeros((X.shape[0], 0), dtype=float)

        inter, coef = self._solve_ridge_with_intercept(D, y, alpha)
        full_linear = np.zeros(X.shape[1], dtype=float)
        if active_idx.size > 0:
            full_linear[active_idx] = coef[:active_idx.size]
        hinge_coef = float(coef[-1]) if hinge_spec is not None else 0.0
        return inter, full_linear, hinge_coef

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        n_samples, n_features = X.shape
        self.n_features_in_ = int(n_features)

        tr_idx, va_idx = self._split_indices(n_samples)
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xva, yva = X[va_idx], y[va_idx]

        # 1) Validation-select alpha for dense linear backbone.
        base_best = None
        full_mask = np.ones(n_features, dtype=bool)
        for alpha in self.alphas:
            inter, linear_coef, _ = self._fit_design(Xtr, ytr, alpha, full_mask, None)
            pred_va = inter + Xva @ linear_coef
            rmse_va = self._rmse(yva, pred_va)
            obj = self._objective(rmse_va, np.sum(np.abs(linear_coef) > self.coef_tol), False)
            if base_best is None or obj < base_best["obj"]:
                base_best = {
                    "alpha": float(alpha),
                    "inter": float(inter),
                    "linear": np.asarray(linear_coef, dtype=float),
                    "rmse_va": float(rmse_va),
                    "obj": float(obj),
                }

        # 2) Candidate one-hinge residual over top linear features.
        x_scale = np.std(Xtr, axis=0) + 1e-12
        contributions = np.abs(base_best["linear"]) * x_scale
        feat_order = np.argsort(contributions)[::-1]
        top = feat_order[: min(int(self.hinge_top_features), n_features)]

        hinge_specs = [None]
        for j in top:
            q_vals = np.unique(np.quantile(Xtr[:, int(j)], np.asarray(self.hinge_quantiles, dtype=float)))
            for t in q_vals:
                hinge_specs.append(("hinge_pos", int(j), float(t)))
                hinge_specs.append(("hinge_neg", int(j), float(t)))

        structure_best = {
            "alpha": float(base_best["alpha"]),
            "hinge": None,
            "rmse_va": float(base_best["rmse_va"]),
            "inter": float(base_best["inter"]),
            "linear": np.asarray(base_best["linear"], dtype=float),
            "hinge_coef": 0.0,
            "obj": float(base_best["obj"]),
        }

        for hinge_spec in hinge_specs:
            has_hinge = hinge_spec is not None
            for alpha in self.alphas:
                inter, linear_coef, hinge_coef = self._fit_design(Xtr, ytr, alpha, full_mask, hinge_spec)
                pred_va = inter + Xva @ linear_coef
                if has_hinge:
                    pred_va = pred_va + hinge_coef * self._hinge_column(Xva, hinge_spec)
                rmse_va = self._rmse(yva, pred_va)
                n_active = np.sum(np.abs(linear_coef) > self.coef_tol)
                obj = self._objective(rmse_va, n_active, has_hinge)
                if obj < structure_best["obj"]:
                    structure_best = {
                        "alpha": float(alpha),
                        "hinge": hinge_spec,
                        "rmse_va": float(rmse_va),
                        "inter": float(inter),
                        "linear": np.asarray(linear_coef, dtype=float),
                        "hinge_coef": float(hinge_coef),
                        "obj": float(obj),
                    }

        # Keep hinge only if it gives meaningful validation RMSE gain.
        use_hinge = structure_best["hinge"] is not None
        if use_hinge:
            rel_gain = (base_best["rmse_va"] - structure_best["rmse_va"]) / max(base_best["rmse_va"], 1e-12)
            if rel_gain < float(self.min_hinge_relative_gain):
                structure_best["hinge"] = None
                structure_best["hinge_coef"] = 0.0
                structure_best["alpha"] = float(base_best["alpha"])

        # 3) Validation-select linear pruning level with chosen structure.
        prune_best = None
        positive = contributions[contributions > 0]
        for q in np.asarray(self.linear_prune_quantiles, dtype=float):
            if positive.size > 0:
                thr = float(np.quantile(positive, min(max(float(q), 0.0), 1.0)))
            else:
                thr = 0.0
            active_mask = contributions >= thr
            if not np.any(active_mask):
                active_mask[np.argmax(contributions)] = True

            inter, linear_coef, hinge_coef = self._fit_design(
                Xtr,
                ytr,
                structure_best["alpha"],
                active_mask,
                structure_best["hinge"],
            )
            pred_va = inter + Xva @ linear_coef
            if structure_best["hinge"] is not None:
                pred_va = pred_va + hinge_coef * self._hinge_column(Xva, structure_best["hinge"])
            rmse_va = self._rmse(yva, pred_va)
            n_active = int(np.sum(np.abs(linear_coef) > self.coef_tol))
            has_hinge = structure_best["hinge"] is not None and abs(float(hinge_coef)) > self.coef_tol
            obj = self._objective(rmse_va, n_active, has_hinge)
            if prune_best is None or obj < prune_best["obj"]:
                prune_best = {
                    "mask": np.asarray(active_mask, dtype=bool),
                    "obj": float(obj),
                }

        # 4) Refit chosen structure on full data.
        inter_full, linear_full, hinge_coef_full = self._fit_design(
            X,
            y,
            structure_best["alpha"],
            prune_best["mask"],
            structure_best["hinge"],
        )
        if structure_best["hinge"] is not None and abs(float(hinge_coef_full)) <= self.coef_tol:
            structure_best["hinge"] = None
            hinge_coef_full = 0.0

        self.alpha_selected_ = float(structure_best["alpha"])
        self.intercept_ = float(inter_full)
        self.linear_coefs_ = np.asarray(linear_full, dtype=float)
        self.hinge_spec_ = structure_best["hinge"]
        self.hinge_coef_ = float(hinge_coef_full)

        self.linear_coefs_[np.abs(self.linear_coefs_) < self.coef_tol] = 0.0
        if abs(self.hinge_coef_) < self.coef_tol:
            self.hinge_coef_ = 0.0
            self.hinge_spec_ = None

        feature_imp = np.abs(self.linear_coefs_)
        if self.hinge_spec_ is not None:
            feature_imp[int(self.hinge_spec_[1])] += abs(float(self.hinge_coef_))
        self.feature_importance_ = feature_imp
        self.selected_features_ = sorted(int(i) for i in np.where(feature_imp > self.coef_tol)[0])
        self.operations_ = int(1 + 2 * len(self.selected_features_) + (4 if self.hinge_spec_ is not None else 0))
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "linear_coefs_", "hinge_spec_", "hinge_coef_"])
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        pred = self.intercept_ + X @ self.linear_coefs_
        if self.hinge_spec_ is not None and abs(float(self.hinge_coef_)) > self.coef_tol:
            pred = pred + float(self.hinge_coef_) * self._hinge_column(X, self.hinge_spec_)
        return pred

    def __str__(self):
        check_is_fitted(self, ["intercept_", "linear_coefs_", "hinge_spec_", "hinge_coef_"])
        dec = int(self.decimals)

        terms = []
        for j, c in enumerate(self.linear_coefs_):
            if abs(float(c)) > self.coef_tol:
                terms.append((f"x{j}", float(c)))
        if self.hinge_spec_ is not None and abs(float(self.hinge_coef_)) > self.coef_tol:
            terms.append((self._hinge_str(self.hinge_spec_, dec), float(self.hinge_coef_)))
        terms.sort(key=lambda t: abs(t[1]), reverse=True)

        rhs = [f"{self.intercept_:+.{dec}f}"] + [f"({coef:+.{dec}f})*{name}" for name, coef in terms]
        lines = [
            "Cognitive Ridge + One-Hinge Regressor",
            "Answer protocol: return one numeric value only.",
            "Exact prediction equation:",
            "  y = " + " + ".join(rhs),
            "",
            "Active terms (sorted by absolute coefficient):",
        ]
        if terms:
            for rank, (name, coef) in enumerate(terms, 1):
                lines.append(f"  {rank:2d}. coef={coef:+.{dec}f}  term={name}")
        else:
            lines.append("  (intercept only)")

        active = [int(i) for i in self.selected_features_]
        lines.append("")
        lines.append("Active features: " + (", ".join(f"x{i}" for i in active) if active else "none"))
        if self.n_features_in_ <= 30 and len(active) < self.n_features_in_:
            active_set = set(active)
            zero = [f"x{i}" for i in range(self.n_features_in_) if i not in active_set]
            lines.append("Zero-contribution features: " + ", ".join(zero))

        lines.append(f"Selected ridge alpha: {self.alpha_selected_:.6g}")
        lines.append(f"Approximate arithmetic operations: {self.operations_}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
CognitiveRidgeOneHingeRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "CognitiveRidgeOneHinge_v1"
model_description = "Validation-tuned dense ridge with optional single hinge residual and cognitive-complexity-aware linear pruning"
model_defs = [(model_shorthand_name, CognitiveRidgeOneHingeRegressor())]


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
