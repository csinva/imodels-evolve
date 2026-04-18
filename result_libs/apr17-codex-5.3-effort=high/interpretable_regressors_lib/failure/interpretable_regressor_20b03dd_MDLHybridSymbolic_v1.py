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


class MDLHybridSymbolicRegressor(BaseEstimator, RegressorMixin):
    """
    Validation-selected symbolic hybrid regressor.

    Candidate structures:
      1) Dense ridge over all features.
      2) Sparse quantized linear model (forward-selected features).
      3) Sparse quantized linear + one hinge term max(0, xj - t).
      4) Sparse quantized linear + one step term I(xj > t).

    Selection criterion:
      choose the smallest-operation candidate whose validation RMSE is within
      rmse_tolerance of the best-RMSE candidate.
    """

    def __init__(
        self,
        val_fraction=0.2,
        alphas=(0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0),
        max_linear_terms=8,
        top_nonlinear_features=4,
        nonlinear_quantiles=(0.2, 0.5, 0.8),
        rmse_tolerance=0.03,
        quant_step=0.25,
        coef_tol=1e-10,
        decimals=6,
        random_state=0,
    ):
        self.val_fraction = val_fraction
        self.alphas = alphas
        self.max_linear_terms = max_linear_terms
        self.top_nonlinear_features = top_nonlinear_features
        self.nonlinear_quantiles = nonlinear_quantiles
        self.rmse_tolerance = rmse_tolerance
        self.quant_step = quant_step
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

    def _quantize(self, arr):
        step = float(self.quant_step)
        if step <= 0:
            return np.asarray(arr, dtype=float)
        return np.round(np.asarray(arr, dtype=float) / step) * step

    @staticmethod
    def _extra_column(X, term):
        feat = int(term["feature"])
        knot = float(term["knot"])
        xj = X[:, feat]
        if term["kind"] == "hinge":
            return np.maximum(0.0, xj - knot)
        return (xj > knot).astype(float)

    def _build_design(self, X, linear_features, extra_terms):
        cols = []
        for f in linear_features:
            cols.append(X[:, int(f)])
        for t in extra_terms:
            cols.append(self._extra_column(X, t))
        if cols:
            return np.column_stack(cols)
        return np.zeros((X.shape[0], 0), dtype=float)

    def _fit_structure(self, Xtr, ytr, Xva, yva, linear_features, extra_terms, quantize):
        Dtr = self._build_design(Xtr, linear_features, extra_terms)
        Dva = self._build_design(Xva, linear_features, extra_terms)

        best = None
        for alpha in self.alphas:
            inter, coef = self._solve_ridge_with_intercept(Dtr, ytr, alpha)
            if quantize:
                inter = float(self._quantize([inter])[0])
                coef = self._quantize(coef)
            pred_va = inter + Dva @ coef
            rmse_va = self._rmse(yva, pred_va)
            if best is None or rmse_va < best["rmse_va"]:
                best = {
                    "alpha": float(alpha),
                    "intercept": float(inter),
                    "coef": np.asarray(coef, dtype=float),
                    "rmse_va": float(rmse_va),
                }

        p = Xtr.shape[1]
        linear = np.zeros(p, dtype=float)
        for i, feat in enumerate(linear_features):
            if i < best["coef"].size:
                linear[int(feat)] = float(best["coef"][i])

        extras = []
        offset = len(linear_features)
        for k, term in enumerate(extra_terms):
            coef_idx = offset + k
            coef_val = float(best["coef"][coef_idx]) if coef_idx < best["coef"].size else 0.0
            extras.append(
                {
                    "kind": term["kind"],
                    "feature": int(term["feature"]),
                    "knot": float(term["knot"]),
                    "coef": coef_val,
                }
            )

        linear[np.abs(linear) < self.coef_tol] = 0.0
        for t in extras:
            if abs(float(t["coef"])) < self.coef_tol:
                t["coef"] = 0.0
        extras = [t for t in extras if abs(float(t["coef"])) > self.coef_tol]

        op_count = 1 + 2 * int(np.sum(np.abs(linear) > self.coef_tol))
        for t in extras:
            op_count += 3 if t["kind"] == "hinge" else 2

        return {
            "alpha": float(best["alpha"]),
            "intercept": float(best["intercept"]),
            "linear_coefs": linear,
            "extra_terms": extras,
            "rmse_va": float(best["rmse_va"]),
            "ops": int(op_count),
            "linear_features": [int(f) for f in linear_features],
            "quantized": bool(quantize),
        }

    def _forward_select_features(self, X, y):
        n_features = X.shape[1]
        max_terms = min(int(self.max_linear_terms), n_features)
        if max_terms <= 0:
            return []

        selected = []
        residual = y - float(np.mean(y))
        for _ in range(max_terms):
            corr = np.abs(X.T @ residual)
            if selected:
                corr[np.asarray(selected, dtype=int)] = -np.inf
            j = int(np.argmax(corr))
            if not np.isfinite(corr[j]) or corr[j] <= 1e-12:
                break
            selected.append(j)
            D = X[:, selected]
            inter, beta = self._solve_ridge_with_intercept(D, y, alpha=1e-4)
            residual = y - (inter + D @ beta)
        return [int(f) for f in selected]

    def _top_features_for_nonlinear(self, X, y, sparse_features):
        centered_y = y - float(np.mean(y))
        scores = np.abs(X.T @ centered_y)
        order = [int(i) for i in np.argsort(scores)[::-1]]
        merged = []
        for f in list(sparse_features) + order:
            if f not in merged:
                merged.append(int(f))
            if len(merged) >= int(self.top_nonlinear_features):
                break
        return merged

    def _choose_candidate(self, candidates):
        best_rmse = min(c["rmse_va"] for c in candidates)
        tol = float(self.rmse_tolerance)
        near_best = [c for c in candidates if c["rmse_va"] <= best_rmse * (1.0 + tol)]
        if not near_best:
            near_best = candidates
        return min(near_best, key=lambda c: (c["ops"], c["rmse_va"]))

    def _refit_on_full(self, X, y, structure):
        linear_features = structure["linear_features"]
        template_terms = [
            {"kind": t["kind"], "feature": t["feature"], "knot": t["knot"]}
            for t in structure["extra_terms"]
        ]
        D = self._build_design(X, linear_features, template_terms)
        inter, coef = self._solve_ridge_with_intercept(D, y, structure["alpha"])
        if structure["quantized"]:
            inter = float(self._quantize([inter])[0])
            coef = self._quantize(coef)

        p = X.shape[1]
        linear = np.zeros(p, dtype=float)
        for i, feat in enumerate(linear_features):
            if i < coef.size:
                linear[int(feat)] = float(coef[i])
        linear[np.abs(linear) < self.coef_tol] = 0.0

        extras = []
        offset = len(linear_features)
        for k, t in enumerate(template_terms):
            coef_idx = offset + k
            c = float(coef[coef_idx]) if coef_idx < coef.size else 0.0
            if abs(c) > self.coef_tol:
                extras.append(
                    {
                        "kind": t["kind"],
                        "feature": int(t["feature"]),
                        "knot": float(t["knot"]),
                        "coef": c,
                    }
                )

        op_count = 1 + 2 * int(np.sum(np.abs(linear) > self.coef_tol))
        for t in extras:
            op_count += 3 if t["kind"] == "hinge" else 2

        return float(inter), linear, extras, int(op_count)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        n_samples, n_features = X.shape
        self.n_features_in_ = int(n_features)

        tr_idx, va_idx = self._split_indices(n_samples)
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xva, yva = X[va_idx], y[va_idx]

        sparse_features = self._forward_select_features(Xtr, ytr)
        if not sparse_features:
            sparse_features = [int(np.argmax(np.abs(Xtr.T @ (ytr - np.mean(ytr)))))]

        candidates = []

        dense_features = list(range(n_features))
        dense = self._fit_structure(Xtr, ytr, Xva, yva, dense_features, [], quantize=False)
        dense["name"] = "dense_ridge"
        candidates.append(dense)

        sparse = self._fit_structure(Xtr, ytr, Xva, yva, sparse_features, [], quantize=True)
        sparse["name"] = "sparse_quantized_linear"
        candidates.append(sparse)

        top_feats = self._top_features_for_nonlinear(Xtr, ytr, sparse_features)
        quantiles = np.asarray(self.nonlinear_quantiles, dtype=float)

        best_hinge = None
        best_step = None
        for feat in top_feats:
            knots = np.unique(np.quantile(Xtr[:, int(feat)], quantiles))
            for knot in knots:
                hinge = self._fit_structure(
                    Xtr,
                    ytr,
                    Xva,
                    yva,
                    sparse_features,
                    [{"kind": "hinge", "feature": int(feat), "knot": float(knot)}],
                    quantize=True,
                )
                hinge["name"] = "sparse_hinge"
                if best_hinge is None or hinge["rmse_va"] < best_hinge["rmse_va"]:
                    best_hinge = hinge

                step = self._fit_structure(
                    Xtr,
                    ytr,
                    Xva,
                    yva,
                    sparse_features,
                    [{"kind": "step", "feature": int(feat), "knot": float(knot)}],
                    quantize=True,
                )
                step["name"] = "sparse_step"
                if best_step is None or step["rmse_va"] < best_step["rmse_va"]:
                    best_step = step

        if best_hinge is not None:
            candidates.append(best_hinge)
        if best_step is not None:
            candidates.append(best_step)

        chosen = self._choose_candidate(candidates)
        inter, linear, extras, op_count = self._refit_on_full(X, y, chosen)

        self.candidate_name_ = chosen["name"]
        self.alpha_selected_ = float(chosen["alpha"])
        self.validation_rmse_ = float(chosen["rmse_va"])
        self.intercept_ = float(inter)
        self.linear_coefs_ = np.asarray(linear, dtype=float)
        self.extra_terms_ = extras
        self.operations_ = int(op_count)

        feature_imp = np.abs(self.linear_coefs_).copy()
        for t in self.extra_terms_:
            feature_imp[int(t["feature"])] += abs(float(t["coef"]))
        self.feature_importance_ = feature_imp
        self.selected_features_ = sorted(int(i) for i in np.where(feature_imp > self.coef_tol)[0])
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "linear_coefs_", "extra_terms_"])
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        pred = self.intercept_ + X @ self.linear_coefs_
        for t in self.extra_terms_:
            feat = int(t["feature"])
            knot = float(t["knot"])
            coef = float(t["coef"])
            if t["kind"] == "hinge":
                pred = pred + coef * np.maximum(0.0, X[:, feat] - knot)
            else:
                pred = pred + coef * (X[:, feat] > knot).astype(float)
        return pred

    def __str__(self):
        check_is_fitted(
            self,
            [
                "candidate_name_",
                "alpha_selected_",
                "validation_rmse_",
                "intercept_",
                "linear_coefs_",
                "extra_terms_",
                "operations_",
            ],
        )
        dec = int(self.decimals)

        linear_terms = [
            (f"x{j}", float(c))
            for j, c in enumerate(self.linear_coefs_)
            if abs(float(c)) > self.coef_tol
        ]

        nonlinear_terms = []
        for t in self.extra_terms_:
            coef = float(t["coef"])
            if abs(coef) <= self.coef_tol:
                continue
            if t["kind"] == "hinge":
                name = f"max(0, x{int(t['feature'])} - {float(t['knot']):.{dec}f})"
            else:
                name = f"I(x{int(t['feature'])} > {float(t['knot']):.{dec}f})"
            nonlinear_terms.append((name, coef))

        all_terms = linear_terms + nonlinear_terms
        all_terms.sort(key=lambda z: abs(z[1]), reverse=True)
        rhs = [f"{self.intercept_:+.{dec}f}"] + [f"({coef:+.{dec}f})*{name}" for name, coef in all_terms]

        active_feats = sorted(set(int(i) for i in self.selected_features_))
        inactive_feats = [f"x{i}" for i in range(self.n_features_in_) if i not in active_feats]

        lines = [
            "MDL Hybrid Symbolic Regressor",
            f"Chosen structure: {self.candidate_name_}",
            f"Validation RMSE: {self.validation_rmse_:.6f}",
            f"Selected ridge alpha: {self.alpha_selected_:.6g}",
            "Exact prediction equation:",
            "  y = " + " + ".join(rhs),
            "",
            "Only terms shown above contribute to the prediction.",
            "Active features: " + (", ".join(f"x{i}" for i in active_feats) if active_feats else "(none)"),
            "Inactive (coefficient 0): " + (", ".join(inactive_feats) if inactive_feats else "(none)"),
            f"Approximate arithmetic operations: {self.operations_}",
        ]
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
MDLHybridSymbolicRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "MDLHybridSymbolic_v1"
model_description = "Validation-selected symbolic hybrid: dense ridge vs sparse quantized linear/hinge/step, choosing simplest near-best RMSE structure"
model_defs = [(model_shorthand_name, MDLHybridSymbolicRegressor())]


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
