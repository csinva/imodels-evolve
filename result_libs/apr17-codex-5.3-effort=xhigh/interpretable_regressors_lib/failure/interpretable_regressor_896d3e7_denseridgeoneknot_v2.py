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


class DenseRidgeOneKnotRegressor(BaseEstimator, RegressorMixin):
    """Dense ridge backbone with one optional data-driven hinge correction."""

    def __init__(
        self,
        alpha_grid=(0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0),
        cv_folds=3,
        max_knot_features=12,
        knot_quantiles=(0.3, 0.5, 0.7),
        min_rel_improvement=0.003,
        tiny_coef=1e-10,
        random_state=0,
    ):
        self.alpha_grid = alpha_grid
        self.cv_folds = cv_folds
        self.max_knot_features = max_knot_features
        self.knot_quantiles = knot_quantiles
        self.min_rel_improvement = min_rel_improvement
        self.tiny_coef = tiny_coef
        self.random_state = random_state

    @staticmethod
    def _fit_ridge_raw(X, y, alpha):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        x_mean = np.mean(X, axis=0)
        x_std = np.std(X, axis=0)
        x_std[x_std < 1e-12] = 1.0

        y_mean = float(np.mean(y))
        y_centered = y - y_mean
        Xs = (X - x_mean) / x_std

        p = Xs.shape[1]
        gram = Xs.T @ Xs + float(alpha) * np.eye(p)
        rhs = Xs.T @ y_centered
        try:
            coef_std = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            coef_std = np.linalg.lstsq(gram, rhs, rcond=None)[0]

        coef_raw = coef_std / x_std
        intercept = y_mean - float(x_mean @ coef_raw)
        return coef_raw, intercept

    def _make_folds(self, n):
        k = max(2, min(int(self.cv_folds), n))
        rng = np.random.RandomState(self.random_state)
        order = rng.permutation(n)
        return np.array_split(order, k)

    def _cv_alpha(self, X, y):
        n = X.shape[0]
        folds = self._make_folds(n)

        best_alpha = None
        best_mse = float("inf")
        for alpha in self.alpha_grid:
            mse_vals = []
            for i in range(len(folds)):
                val_idx = folds[i]
                if val_idx.size == 0:
                    continue
                train_idx = np.concatenate([folds[j] for j in range(len(folds)) if j != i])
                X_train, y_train = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]

                coef, intercept = self._fit_ridge_raw(X_train, y_train, alpha)
                pred = intercept + X_val @ coef
                mse_vals.append(float(np.mean((y_val - pred) ** 2)))

            if not mse_vals:
                continue
            mse = float(np.mean(mse_vals))
            if mse < best_mse:
                best_mse = mse
                best_alpha = float(alpha)

        if best_alpha is None:
            best_alpha = float(self.alpha_grid[0])
            best_mse = float("inf")
        return best_alpha, best_mse

    def _fit_augmented(self, X, y, alpha, feat_idx, knot):
        h = np.maximum(0.0, X[:, feat_idx] - knot).reshape(-1, 1)
        Xa = np.hstack([X, h])
        coef_aug, intercept = self._fit_ridge_raw(Xa, y, alpha)
        return coef_aug[:-1], float(coef_aug[-1]), intercept

    def _cv_augmented_mse(self, X, y, alpha, feat_idx, knot, folds):
        errs = []
        for i in range(len(folds)):
            val_idx = folds[i]
            if val_idx.size == 0:
                continue
            train_idx = np.concatenate([folds[j] for j in range(len(folds)) if j != i])

            Xt, yt = X[train_idx], y[train_idx]
            Xv, yv = X[val_idx], y[val_idx]
            coef_lin, coef_h, intercept = self._fit_augmented(Xt, yt, alpha, feat_idx, knot)
            pred = intercept + Xv @ coef_lin + coef_h * np.maximum(0.0, Xv[:, feat_idx] - knot)
            errs.append(float(np.mean((yv - pred) ** 2)))
        return float(np.mean(errs)) if errs else float("inf")

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        if n_features == 0:
            raise ValueError("No features provided")

        self.selected_alpha_, best_linear_cv_mse = self._cv_alpha(X, y)
        coef_full, intercept = self._fit_ridge_raw(X, y, self.selected_alpha_)

        base_pred = intercept + X @ coef_full
        residual = y - base_pred

        centered = X - np.mean(X, axis=0, keepdims=True)
        denom = np.sqrt(np.sum(centered ** 2, axis=0)) + 1e-12
        corrs = np.abs((centered.T @ residual) / denom)
        feat_order = np.argsort(corrs)[::-1]
        candidate_features = feat_order[: max(1, min(int(self.max_knot_features), n_features))]

        folds = self._make_folds(n_samples)
        best_aug_mse = best_linear_cv_mse
        best_aug = None
        for feat_idx in candidate_features:
            for q in self.knot_quantiles:
                knot = float(np.quantile(X[:, feat_idx], q))
                mse = self._cv_augmented_mse(X, y, self.selected_alpha_, int(feat_idx), knot, folds)
                if mse < best_aug_mse:
                    best_aug_mse = mse
                    best_aug = (int(feat_idx), knot)

        rel_gain = (best_linear_cv_mse - best_aug_mse) / max(abs(best_linear_cv_mse), 1e-12)
        if best_aug is not None and rel_gain >= float(self.min_rel_improvement):
            feat_idx, knot = best_aug
            coef_full, hinge_coef, intercept = self._fit_augmented(X, y, self.selected_alpha_, feat_idx, knot)
            self.hinge_feature_ = int(feat_idx)
            self.hinge_threshold_ = float(knot)
            self.hinge_coef_ = float(hinge_coef)
            self.used_hinge_ = True
        else:
            self.hinge_feature_ = -1
            self.hinge_threshold_ = 0.0
            self.hinge_coef_ = 0.0
            self.used_hinge_ = False

        coef_full[np.abs(coef_full) < self.tiny_coef] = 0.0
        importances = np.abs(coef_full)
        if self.used_hinge_:
            importances[self.hinge_feature_] += abs(self.hinge_coef_)
        mx = float(np.max(importances)) if np.max(importances) > 0 else 0.0
        self.feature_importances_ = importances / mx if mx > 0 else importances

        self.intercept_ = intercept
        self.coef_ = coef_full
        self.rel_gain_ = float(rel_gain)
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "coef_", "used_hinge_"])
        X = np.asarray(X, dtype=float)
        pred = self.intercept_ + X @ self.coef_
        if self.used_hinge_:
            pred = pred + self.hinge_coef_ * np.maximum(0.0, X[:, self.hinge_feature_] - self.hinge_threshold_)
        return pred

    def __str__(self):
        check_is_fitted(
            self,
            [
                "intercept_",
                "coef_",
                "feature_importances_",
                "selected_alpha_",
                "used_hinge_",
                "hinge_feature_",
                "hinge_threshold_",
                "hinge_coef_",
                "rel_gain_",
            ],
        )

        order = np.argsort(np.abs(self.coef_))[::-1]
        active = [int(j) for j in order if abs(float(self.coef_[j])) > self.tiny_coef]
        if not active:
            active = [int(order[0])] if len(order) else []

        eq_terms = [f"{self.intercept_:+.4f}"]
        for j in active:
            eq_terms.append(f"{float(self.coef_[j]):+.4f}*x{j}")
        if self.used_hinge_ and abs(self.hinge_coef_) > self.tiny_coef:
            eq_terms.append(
                f"{float(self.hinge_coef_):+.4f}*max(0, x{self.hinge_feature_}-{float(self.hinge_threshold_):.4f})"
            )

        top_lines = [
            f"x{j}: coef={float(self.coef_[j]):+.4f}, importance={float(self.feature_importances_[j]):.3f}"
            for j in active[: min(len(active), 12)]
        ]
        lines = [
            "Dense Ridge + One-Knot Regressor",
            f"Chosen ridge alpha by CV: {self.selected_alpha_:.4g}",
            "Prediction equation:",
            "  y = " + " ".join(eq_terms),
            "",
            "Top features by absolute effect:",
            *("  " + line for line in top_lines),
        ]
        if self.used_hinge_:
            lines.extend(
                [
                    "",
                    "Nonlinear correction:",
                    f"  hinge feature: x{self.hinge_feature_}",
                    f"  threshold: {self.hinge_threshold_:.4f}",
                    f"  hinge coefficient: {self.hinge_coef_:+.4f}",
                    f"  relative CV gain vs. linear backbone: {100.0 * self.rel_gain_:.2f}%",
                ]
            )
        else:
            lines.extend(["", "Nonlinear correction: none (linear backbone kept)"])
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys

_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])

# Update the model shorthand name and description below to reflect the class above and any changes you make.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
DenseRidgeOneKnotRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "DenseRidgeOneKnot_v2"
model_description = "Dense custom CV ridge backbone with optional single feature hinge correction selected by cross-validated residual gain (cache-safe rerun)"
model_defs = [(model_shorthand_name, DenseRidgeOneKnotRegressor())]


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

    # --- Recompute global rank summary from updated performance_results.csv ---
    # Build dataset -> {model: rmse}
    perf_table = defaultdict(dict)
    with open(perf_csv, newline="") as f:
        for row in csv.DictReader(f):
            ds = row["dataset"]
            m = row["model"]
            rmse_s = row.get("rmse", "")
            if rmse_s in ("", None):
                perf_table[ds][m] = float("nan")
            else:
                try:
                    perf_table[ds][m] = float(rmse_s)
                except ValueError:
                    perf_table[ds][m] = float("nan")

    avg_rank, _ = compute_rank_scores(perf_table)
    mean_rank = avg_rank.get(model_name, float("nan"))

    # --- Upsert overall_results.csv ---
    overall_rows = [{
        "commit": git_hash,
        "mean_rank": f"{mean_rank:.2f}" if np.isfinite(mean_rank) else "",
        "frac_interpretability_tests_passed": f"{(n_passed / total):.4f}" if total else "",
        "status": "",  # fill manually after reviewing
        "model_name": model_name,
        "description": model_description,
    }]
    upsert_overall_results(overall_rows, RESULTS_DIR)

    # --- Plot update ---
    overall_csv = os.path.join(RESULTS_DIR, "overall_results.csv")
    plot_interp_vs_performance(
        overall_csv,
        os.path.join(RESULTS_DIR, "interpretability_vs_performance.png"),
    )

    # Print compact summary
    std_names = {t.__name__ for t in ALL_TESTS}
    hard_names = {t.__name__ for t in HARD_TESTS}
    ins_names = {t.__name__ for t in INSIGHT_TESTS}
    n_std = sum(r["passed"] for r in interp_results if r["test"] in std_names)
    n_hard = sum(r["passed"] for r in interp_results if r["test"] in hard_names)
    n_ins = sum(r["passed"] for r in interp_results if r["test"] in ins_names)

    print("\n---")
    print(f"tests_passed:  {n_passed}/{total} ({(n_passed/total):.2%})  "
          f"[std {n_std}/{len(std_names)}  hard {n_hard}/{len(hard_names)}  insight {n_ins}/{len(ins_names)}]")
    print(f"total_seconds: {time.time() - t0:.1f}s")
