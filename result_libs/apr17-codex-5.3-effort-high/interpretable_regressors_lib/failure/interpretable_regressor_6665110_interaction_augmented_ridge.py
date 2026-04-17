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


class InteractionAugmentedRidgeRegressor(BaseEstimator, RegressorMixin):
    """
    Dense GCV ridge backbone with a tiny explicit interaction augmentation.

    Stage 1: fit a closed-form ridge model with GCV alpha selection.
    Stage 2: greedily add a few pairwise x_i*x_j terms if they improve
             holdout MSE.
    """

    def __init__(
        self,
        alpha_grid=(1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0),
        interaction_ridge=1.0,
        top_linear_features=12,
        max_interactions=3,
        min_interaction_rel_gain=0.004,
        val_fraction=0.2,
        random_state=42,
    ):
        self.alpha_grid = alpha_grid
        self.interaction_ridge = interaction_ridge
        self.top_linear_features = top_linear_features
        self.max_interactions = max_interactions
        self.min_interaction_rel_gain = min_interaction_rel_gain
        self.val_fraction = val_fraction
        self.random_state = random_state

    @staticmethod
    def _safe_std(v):
        s = float(np.std(v))
        return s if s > 1e-12 else 1.0

    def _fit_ridge_standardized(self, Z, yc):
        n, p = Z.shape
        if p == 0:
            return np.zeros(0, dtype=float), float(self.alpha_grid[0])
        U, s, Vt = np.linalg.svd(Z, full_matrices=False)
        Uy = U.T @ yc

        best_alpha = float(self.alpha_grid[0])
        best_gcv = np.inf
        best_coef = np.zeros(p, dtype=float)
        n_eff = max(n, 1)

        for a in self.alpha_grid:
            alpha = float(max(a, 1e-12))
            shrink = s / (s * s + alpha)
            coef = Vt.T @ (shrink * Uy)
            resid = yc - Z @ coef
            trace_h = float(np.sum((s * s) / (s * s + alpha)))
            denom = max(1.0 - trace_h / n_eff, 1e-6)
            gcv = float(np.mean(resid * resid) / (denom * denom))
            if gcv < best_gcv:
                best_gcv = gcv
                best_alpha = alpha
                best_coef = coef
        return best_coef, best_alpha

    @staticmethod
    def _pair_feature(X, i, j):
        return X[:, i] * X[:, j]

    def _best_interaction_term(self, Xtr, ytr, pred_tr, Xva, yva, pred_va, pairs, used):
        base_mse = float(np.mean((yva - pred_va) ** 2))
        best = None
        lam = float(max(self.interaction_ridge, 1e-9))

        for (i, j) in pairs:
            key = (int(i), int(j))
            if key in used:
                continue
            z_tr = self._pair_feature(Xtr, i, j)
            z_va = self._pair_feature(Xva, i, j)
            z_mean = float(np.mean(z_tr))
            z_tr = z_tr - z_mean
            z_va = z_va - z_mean
            denom = float(np.dot(z_tr, z_tr) + lam)
            if denom <= 1e-12:
                continue

            coef = float(np.dot(ytr - pred_tr, z_tr) / denom)
            cand_pred_va = pred_va + coef * z_va
            mse = float(np.mean((yva - cand_pred_va) ** 2))
            rel_gain = (base_mse - mse) / max(base_mse, 1e-12)
            if best is None or rel_gain > best["gain"]:
                best = {
                    "i": int(i),
                    "j": int(j),
                    "coef": coef,
                    "mean": z_mean,
                    "gain": float(rel_gain),
                }
        return best

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape
        self.n_features_in_ = p

        rng = np.random.RandomState(self.random_state)

        self.x_mean_ = np.mean(X, axis=0)
        self.x_scale_ = np.array([self._safe_std(X[:, j]) for j in range(p)], dtype=float)
        Z = (X - self.x_mean_) / self.x_scale_

        y_mean = float(np.mean(y))
        yc = y - y_mean

        coef_std, alpha = self._fit_ridge_standardized(Z, yc)
        coef_raw = coef_std / self.x_scale_
        intercept = y_mean - float(np.dot(self.x_mean_, coef_raw))

        pred = intercept + X @ coef_raw

        if p >= 2 and n >= 25:
            idx = np.arange(n, dtype=int)
            rng.shuffle(idx)
            n_va = int(max(5, min(n - 5, round(float(self.val_fraction) * n))))
            va_idx = idx[:n_va]
            tr_idx = idx[n_va:]
            if len(tr_idx) < 5:
                tr_idx = idx
                va_idx = idx[: max(5, n // 5)]
        else:
            tr_idx = np.arange(n, dtype=int)
            va_idx = np.arange(min(max(5, n // 5), n), dtype=int)

        Xtr, ytr, pred_tr = X[tr_idx], y[tr_idx], pred[tr_idx]
        Xva, yva, pred_va = X[va_idx], y[va_idx], pred[va_idx]

        k = int(min(max(2, self.top_linear_features), p))
        top = np.argsort(-np.abs(coef_raw))[:k].astype(int)
        pairs = []
        for a in range(len(top)):
            for b in range(a + 1, len(top)):
                pairs.append((int(top[a]), int(top[b])))

        interactions = []
        used = set()
        for _ in range(int(self.max_interactions)):
            best = self._best_interaction_term(Xtr, ytr, pred_tr, Xva, yva, pred_va, pairs, used)
            if best is None or best["gain"] < float(self.min_interaction_rel_gain):
                break
            interactions.append(best)
            used.add((best["i"], best["j"]))
            z_all = self._pair_feature(X, best["i"], best["j"]) - best["mean"]
            pred = pred + best["coef"] * z_all
            pred_tr = pred[tr_idx]
            pred_va = pred[va_idx]

        self.alpha_ = float(alpha)
        self.intercept_ = float(intercept)
        self.coef_ = coef_raw
        self.interactions_ = interactions

        abs_c = np.abs(self.coef_)
        max_abs = float(np.max(abs_c)) if len(abs_c) else 0.0
        thr = 0.08 * max(max_abs, 1e-12)
        self.meaningful_features_ = np.where(abs_c >= thr)[0].astype(int)
        self.negligible_features_ = np.where(abs_c < thr)[0].astype(int)
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "coef_", "interactions_"])
        X = np.asarray(X, dtype=float)
        pred = self.intercept_ + X @ self.coef_
        for term in self.interactions_:
            z = self._pair_feature(X, term["i"], term["j"]) - term["mean"]
            pred = pred + term["coef"] * z
        return pred

    def __str__(self):
        check_is_fitted(self, ["intercept_", "coef_", "interactions_"])
        lines = [
            "Interaction-Augmented Ridge Regressor",
            "Exact prediction recipe:",
            "1) Start from linear ridge backbone.",
            "2) Add each centered pairwise interaction term.",
        ]

        eq = f"y(x) = {self.intercept_:+.6f}"
        active_sorted = np.argsort(-np.abs(self.coef_))
        for j in active_sorted:
            c = float(self.coef_[j])
            if abs(c) < 1e-12:
                continue
            eq += f" {c:+.6f}*x{int(j)}"
        for term in self.interactions_:
            eq += (
                f" {float(term['coef']):+.6f}*(x{int(term['i'])}*x{int(term['j'])}"
                f" {(-float(term['mean'])):+.6f})"
            )
        lines.append(eq)
        lines.append(f"Ridge alpha (GCV): {self.alpha_:.6g}")

        if self.interactions_:
            lines.append("Interaction terms (from residual holdout gain):")
            for i, term in enumerate(self.interactions_, 1):
                lines.append(
                    f"  Term {i}: {float(term['coef']):+.6f}*(x{int(term['i'])}*x{int(term['j'])} {(-float(term['mean'])):+.6f})"
                )
        else:
            lines.append("Interaction terms: none selected.")

        lines.append("Features ranked by absolute linear coefficient:")
        for idx in active_sorted:
            lines.append(
                f"  x{int(idx)}: coef={float(self.coef_[idx]):+.6f}"
            )

        if len(self.meaningful_features_) > 0:
            lines.append("Meaningfully used features: " + ", ".join(f"x{int(j)}" for j in self.meaningful_features_))
        if len(self.negligible_features_) > 0:
            lines.append("Features with negligible effect: " + ", ".join(f"x{int(j)}" for j in self.negligible_features_))
        lines.append("Model is simulatable as one explicit equation with a short list of interaction terms.")
        return "\n".join(lines)


import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
InteractionAugmentedRidgeRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "InteractionAugmentedRidgeV1"
model_description = "Dense GCV ridge backbone with greedy holdout-gated centered pairwise interaction terms on top linear features, expressed as one exact simulatable equation"
model_defs = [(model_shorthand_name, InteractionAugmentedRidgeRegressor())]

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
