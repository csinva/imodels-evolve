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


class SparseRidgeOneHingeRegressor(BaseEstimator, RegressorMixin):
    """Sparse linear regressor with one validation-gated hinge correction."""

    def __init__(
        self,
        max_linear_terms=10,
        min_linear_terms=3,
        ridge_alpha=0.8,
        relative_coef_keep=0.08,
        val_fraction=0.2,
        hinge_quantiles=(0.2, 0.4, 0.6, 0.8),
        max_hinge_features=3,
        min_hinge_gain=0.01,
        negligible_coef_eps=1e-3,
        random_state=0,
    ):
        self.max_linear_terms = max_linear_terms
        self.min_linear_terms = min_linear_terms
        self.ridge_alpha = ridge_alpha
        self.relative_coef_keep = relative_coef_keep
        self.val_fraction = val_fraction
        self.hinge_quantiles = hinge_quantiles
        self.max_hinge_features = max_hinge_features
        self.min_hinge_gain = min_hinge_gain
        self.negligible_coef_eps = negligible_coef_eps
        self.random_state = random_state

    def _solve_weighted_linear(self, Z, y, alpha):
        z_mean = np.mean(Z, axis=0)
        z_std = np.std(Z, axis=0)
        z_std[z_std < 1e-12] = 1.0
        Zs = (Z - z_mean) / z_std

        y_mean = float(np.mean(y))
        y_centered = y - y_mean

        p = Zs.shape[1]
        gram = Zs.T @ Zs
        rhs = Zs.T @ y_centered
        coef_std = np.linalg.solve(gram + float(alpha) * np.eye(p), rhs)

        coef = coef_std / z_std
        intercept = float(y_mean - np.dot(coef, z_mean))
        return intercept, coef

    def _val_mse(self, X_val, y_val, intercept, coef, hinge=None, hinge_coef=0.0):
        pred = intercept + X_val @ coef
        if hinge is not None:
            z_val = self._hinge_values(X_val[:, hinge["feature"]], hinge["knot"], hinge["direction"])
            pred = pred + hinge_coef * z_val
        return float(np.mean((y_val - pred) ** 2))

    def _hinge_values(self, x_col, knot, direction):
        if direction == "right":
            return np.maximum(0.0, x_col - knot)
        return np.maximum(0.0, knot - x_col)

    def _choose_linear_features(self, coef):
        n_features = coef.shape[0]
        order = np.argsort(np.abs(coef))[::-1]
        max_terms = int(min(max(1, self.max_linear_terms), n_features))
        min_terms = int(min(max(1, self.min_linear_terms), max_terms))

        abs_coef = np.abs(coef[order])
        total = float(np.sum(abs_coef)) + 1e-12
        csum = np.cumsum(abs_coef) / total
        k_mass = int(np.searchsorted(csum, 0.97) + 1)
        k_mass = int(min(max(k_mass, min_terms), max_terms))

        keep = list(order[:k_mass])
        strongest = float(abs_coef[0]) if len(abs_coef) else 0.0
        for j in order:
            if abs(coef[j]) >= self.relative_coef_keep * strongest and j not in keep:
                keep.append(int(j))
            if len(keep) >= max_terms:
                break

        keep = sorted(set(int(j) for j in keep))
        if len(keep) < min_terms:
            keep = sorted(int(j) for j in order[:min_terms])
        return keep

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        rng = np.random.RandomState(self.random_state)
        idx = np.arange(n_samples)
        rng.shuffle(idx)
        n_val = int(max(1, min(n_samples - 1, round(self.val_fraction * n_samples))))
        val_idx = idx[:n_val]
        tr_idx = idx[n_val:]
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        base_intercept, base_coef = self._solve_weighted_linear(X_tr, y_tr, alpha=self.ridge_alpha)
        keep = self._choose_linear_features(base_coef)
        self.selected_linear_features_ = np.asarray(keep, dtype=int)

        X_tr_sel = X_tr[:, self.selected_linear_features_]
        X_val_sel = X_val[:, self.selected_linear_features_]
        lin_intercept, lin_coef_sel = self._solve_weighted_linear(X_tr_sel, y_tr, alpha=1e-8)

        linear_coef_full = np.zeros(n_features, dtype=float)
        linear_coef_full[self.selected_linear_features_] = lin_coef_sel
        best_mse = self._val_mse(X_val, y_val, lin_intercept, linear_coef_full)

        self.hinge_term_ = None
        self.hinge_coef_ = 0.0

        hinge_candidates = self.selected_linear_features_[
            np.argsort(np.abs(lin_coef_sel))[::-1][: min(self.max_hinge_features, len(self.selected_linear_features_))]
        ]

        for feat in hinge_candidates:
            x_col = X_tr[:, feat]
            knots = np.unique(np.quantile(x_col, self.hinge_quantiles))
            for knot in knots:
                for direction in ("right", "left"):
                    z_tr = self._hinge_values(x_col, float(knot), direction)
                    if float(np.std(z_tr)) < 1e-10:
                        continue
                    Z_tr = np.column_stack([X_tr_sel, z_tr])
                    try_intercept, try_coef = self._solve_weighted_linear(Z_tr, y_tr, alpha=1e-8)
                    lin_try = np.zeros(n_features, dtype=float)
                    lin_try[self.selected_linear_features_] = try_coef[:-1]
                    hinge_coef = float(try_coef[-1])
                    hinge = {"feature": int(feat), "knot": float(knot), "direction": direction}
                    mse_try = self._val_mse(X_val, y_val, try_intercept, lin_try, hinge=hinge, hinge_coef=hinge_coef)
                    if mse_try < best_mse:
                        best_mse = mse_try
                        self.hinge_term_ = hinge
                        self.hinge_coef_ = hinge_coef
                        lin_intercept = try_intercept
                        linear_coef_full = lin_try

        if self.hinge_term_ is not None:
            base_pred = self._val_mse(
                X_val, y_val, lin_intercept, linear_coef_full, hinge=self.hinge_term_, hinge_coef=self.hinge_coef_
            )
            no_hinge_mse = self._val_mse(X_val, y_val, lin_intercept, linear_coef_full)
            rel_gain = (no_hinge_mse - base_pred) / (no_hinge_mse + 1e-12)
            if rel_gain < self.min_hinge_gain:
                self.hinge_term_ = None
                self.hinge_coef_ = 0.0

        X_full_sel = X[:, self.selected_linear_features_]
        if self.hinge_term_ is None:
            self.intercept_, coef_sel = self._solve_weighted_linear(X_full_sel, y, alpha=1e-8)
            self.linear_coef_ = np.zeros(n_features, dtype=float)
            self.linear_coef_[self.selected_linear_features_] = coef_sel
        else:
            z_full = self._hinge_values(
                X[:, self.hinge_term_["feature"]],
                self.hinge_term_["knot"],
                self.hinge_term_["direction"],
            )
            Z_full = np.column_stack([X_full_sel, z_full])
            self.intercept_, coef_all = self._solve_weighted_linear(Z_full, y, alpha=1e-8)
            self.linear_coef_ = np.zeros(n_features, dtype=float)
            self.linear_coef_[self.selected_linear_features_] = coef_all[:-1]
            self.hinge_coef_ = float(coef_all[-1])

        self.feature_importances_ = np.abs(self.linear_coef_)
        if self.hinge_term_ is not None:
            self.feature_importances_[self.hinge_term_["feature"]] += abs(self.hinge_coef_)

        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "linear_coef_", "selected_linear_features_", "feature_importances_"])
        X = np.asarray(X, dtype=float)
        pred = self.intercept_ + X @ self.linear_coef_
        if self.hinge_term_ is not None:
            z = self._hinge_values(
                X[:, self.hinge_term_["feature"]],
                self.hinge_term_["knot"],
                self.hinge_term_["direction"],
            )
            pred = pred + self.hinge_coef_ * z
        return pred

    def _equation_terms(self):
        terms = [f"{self.intercept_:+.6f}"]
        active = np.where(np.abs(self.linear_coef_) > self.negligible_coef_eps)[0]
        for j in active:
            terms.append(f"{self.linear_coef_[j]:+.6f}*x{int(j)}")
        if self.hinge_term_ is not None and abs(self.hinge_coef_) > self.negligible_coef_eps:
            feat = self.hinge_term_["feature"]
            knot = self.hinge_term_["knot"]
            if self.hinge_term_["direction"] == "right":
                terms.append(f"{self.hinge_coef_:+.6f}*max(0, x{feat}-{knot:.6f})")
            else:
                terms.append(f"{self.hinge_coef_:+.6f}*max(0, {knot:.6f}-x{feat})")
        return terms

    def __str__(self):
        check_is_fitted(self, ["intercept_", "linear_coef_", "feature_importances_"])
        active = np.where(np.abs(self.linear_coef_) > self.negligible_coef_eps)[0]
        inactive = [f"x{j}" for j in range(self.n_features_in_) if j not in set(active.tolist())]
        top = np.argsort(self.feature_importances_)[::-1]
        top_txt = ", ".join(
            f"x{int(j)}:{self.feature_importances_[j]:.4f}" for j in top[: min(6, self.n_features_in_)]
        )

        lines = [
            "Sparse Ridge + One-Hinge Regressor",
            "Exact prediction equation (use directly for manual calculation):",
            "  y = " + " ".join(self._equation_terms()),
            "",
            f"Active linear features ({len(active)}): "
            + (", ".join(f"x{int(j)}" for j in active) if len(active) else "none"),
        ]
        if self.hinge_term_ is None:
            lines.append("Hinge correction: none")
        else:
            feat = self.hinge_term_["feature"]
            knot = self.hinge_term_["knot"]
            direction = self.hinge_term_["direction"]
            if direction == "right":
                hinge_txt = f"max(0, x{feat}-{knot:.6f})"
            else:
                hinge_txt = f"max(0, {knot:.6f}-x{feat})"
            lines.append(f"Hinge correction: {self.hinge_coef_:+.6f} * {hinge_txt}")

        lines.extend([
            "Features with near-zero linear effect: "
            + (", ".join(inactive) if inactive else "none"),
            "Top feature influence magnitudes: " + top_txt,
        ])
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparseRidgeOneHingeRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SparseRidgeOneHinge_v1"
model_description = "Sparse ridge linear backbone with validation-selected one-feature hinge correction and calculator-first equation output"
model_defs = [(model_shorthand_name, SparseRidgeOneHingeRegressor())]


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
