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
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class TwinHingeAdditiveRidgeRegressor(BaseEstimator, RegressorMixin):
    """RidgeCV linear backbone with up to two greedy hinge corrections."""

    def __init__(
        self,
        alpha_grid=(1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0, 100.0),
        max_hinge_terms=2,
        max_hinge_features=6,
        hinge_quantiles=(0.2, 0.4, 0.6, 0.8),
        val_frac=0.2,
        min_rel_improvement=0.003,
        tiny_coef=1e-8,
        random_state=0,
    ):
        self.alpha_grid = alpha_grid
        self.max_hinge_terms = max_hinge_terms
        self.max_hinge_features = max_hinge_features
        self.hinge_quantiles = hinge_quantiles
        self.val_frac = val_frac
        self.min_rel_improvement = min_rel_improvement
        self.tiny_coef = tiny_coef
        self.random_state = random_state

    @staticmethod
    def _hinge_feature(x_col, threshold, direction):
        if direction == "right":
            return np.maximum(0.0, x_col - threshold)
        return np.maximum(0.0, threshold - x_col)

    def _build_hinge_column(self, X, spec):
        feat_idx, threshold, direction = spec
        return self._hinge_feature(X[:, feat_idx], threshold, direction).reshape(-1, 1)

    def _fit_ridge(self, X, y, alpha):
        model = Ridge(alpha=float(alpha), fit_intercept=True)
        model.fit(X, y)
        return model

    def _val_split(self, n_samples):
        n_val = max(1, int(round(float(self.val_frac) * n_samples)))
        n_val = min(n_val, max(1, n_samples - 1))
        rng = np.random.RandomState(self.random_state)
        perm = rng.permutation(n_samples)
        return perm[:-n_val], perm[-n_val:]

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape
        if n_features == 0:
            raise ValueError("No features provided")
        self.n_features_in_ = n_features

        train_idx, val_idx = self._val_split(n_samples)
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        base_cv = RidgeCV(alphas=np.asarray(self.alpha_grid, dtype=float), cv=5, fit_intercept=True)
        base_cv.fit(X_train, y_train)
        alpha = float(base_cv.alpha_)

        base_model = self._fit_ridge(X_train, y_train, alpha)
        base_val_pred = base_model.predict(X_val)
        best_val_mse = float(np.mean((y_val - base_val_pred) ** 2))
        baseline_val_mse = best_val_mse

        residual_train = y_train - base_model.predict(X_train)
        centered = X_train - np.mean(X_train, axis=0, keepdims=True)
        denom = np.sqrt(np.sum(centered**2, axis=0)) + 1e-12
        corr = np.abs((centered.T @ residual_train) / denom)
        feature_order = np.argsort(corr)[::-1]
        candidate_features = feature_order[: max(1, min(int(self.max_hinge_features), n_features))]

        all_specs = []
        for feat_idx in candidate_features:
            x_col = X_train[:, int(feat_idx)]
            for q in self.hinge_quantiles:
                thr = float(np.quantile(x_col, q))
                all_specs.append((int(feat_idx), thr, "right"))
                all_specs.append((int(feat_idx), thr, "left"))

        selected_specs = []
        for _ in range(max(0, int(self.max_hinge_terms))):
            best_spec = None
            best_candidate_mse = best_val_mse
            for spec in all_specs:
                if spec in selected_specs:
                    continue
                cols_train = [X_train]
                cols_val = [X_val]
                for s in selected_specs:
                    cols_train.append(self._build_hinge_column(X_train, s))
                    cols_val.append(self._build_hinge_column(X_val, s))
                cols_train.append(self._build_hinge_column(X_train, spec))
                cols_val.append(self._build_hinge_column(X_val, spec))
                X_aug_train = np.hstack(cols_train)
                X_aug_val = np.hstack(cols_val)

                m = self._fit_ridge(X_aug_train, y_train, alpha)
                mse = float(np.mean((y_val - m.predict(X_aug_val)) ** 2))
                if mse < best_candidate_mse:
                    best_candidate_mse = mse
                    best_spec = spec

            rel_gain = (best_val_mse - best_candidate_mse) / max(abs(best_val_mse), 1e-12)
            if best_spec is None or rel_gain < float(self.min_rel_improvement):
                break
            selected_specs.append(best_spec)
            best_val_mse = best_candidate_mse

        X_full_parts = [X]
        for spec in selected_specs:
            X_full_parts.append(self._build_hinge_column(X, spec))
        X_full_aug = np.hstack(X_full_parts)
        final_model = self._fit_ridge(X_full_aug, y, alpha)

        coef_all = np.asarray(final_model.coef_, dtype=float).copy()
        coef_linear = coef_all[:n_features]
        coef_hinge = coef_all[n_features:] if len(coef_all) > n_features else np.zeros(0, dtype=float)
        coef_linear[np.abs(coef_linear) < self.tiny_coef] = 0.0
        coef_hinge[np.abs(coef_hinge) < self.tiny_coef] = 0.0

        importances = np.abs(coef_linear).copy()
        for k, spec in enumerate(selected_specs):
            importances[spec[0]] += abs(coef_hinge[k])
        max_imp = float(np.max(importances)) if np.max(importances) > 0 else 0.0
        self.feature_importances_ = importances / max_imp if max_imp > 0 else importances

        self.selected_alpha_ = alpha
        self.intercept_ = float(final_model.intercept_)
        self.coef_ = coef_linear
        self.hinge_specs_ = selected_specs
        self.hinge_coefs_ = coef_hinge
        self.baseline_val_mse_ = float(baseline_val_mse)
        self.final_val_mse_ = float(best_val_mse)
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "coef_", "hinge_specs_", "hinge_coefs_"])
        X = np.asarray(X, dtype=float)
        pred = self.intercept_ + X @ self.coef_
        for k, spec in enumerate(self.hinge_specs_):
            pred += float(self.hinge_coefs_[k]) * self._build_hinge_column(X, spec).ravel()
        return pred

    def __str__(self):
        check_is_fitted(
            self,
            [
                "intercept_",
                "coef_",
                "hinge_specs_",
                "hinge_coefs_",
                "feature_importances_",
                "selected_alpha_",
                "baseline_val_mse_",
                "final_val_mse_",
            ],
        )
        terms = [f"{self.intercept_:+.6f}"]
        for j, c in enumerate(self.coef_):
            terms.append(f"{float(c):+.6f}*x{j}")
        for coef, spec in zip(self.hinge_coefs_, self.hinge_specs_):
            feat_idx, threshold, direction = spec
            if direction == "right":
                basis = f"max(0, x{feat_idx}-{threshold:.6f})"
            else:
                basis = f"max(0, {threshold:.6f}-x{feat_idx})"
            terms.append(f"{float(coef):+.6f}*{basis}")

        sorted_idx = np.argsort(np.abs(self.coef_))[::-1]
        active = [j for j in sorted_idx if abs(self.coef_[j]) > self.tiny_coef]
        inactive = [j for j in range(len(self.coef_)) if abs(self.coef_[j]) <= self.tiny_coef]

        lines = [
            "Prediction equation:",
            "y = " + " ".join(terms),
            "",
            f"Ridge alpha: {self.selected_alpha_:.6g}",
            "Simulation rule: multiply each xj by its coefficient, add intercept and hinge terms.",
            "Hinge definition: max(0, a) returns a when a>0 else 0.",
            "",
            "Top linear features:",
        ]
        for j in active[: min(10, len(active))]:
            lines.append(
                f"x{j}: coef={float(self.coef_[j]):+.4f}, importance={float(self.feature_importances_[j]):.3f}"
            )
        if inactive:
            lines.append("")
            lines.append("Near-zero linear features:")
            lines.append(", ".join(f"x{j}" for j in inactive[:20]))
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys

_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])

# Update the model shorthand name and description below to reflect the class above and any changes you make.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
TwinHingeAdditiveRidgeRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "TwinHingeAdditiveRidge_v1"
model_description = "RidgeCV linear backbone with up to two validation-selected one-feature hinge corrections for compact piecewise-additive equations"
model_defs = [(model_shorthand_name, TwinHingeAdditiveRidgeRegressor())]


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
