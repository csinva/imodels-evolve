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
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class SparseAdditiveHingeRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse symbolic additive model:
    - Build an interpretable basis (linear + hinge + abs + few interactions).
    - Use L1 selection to keep only a small number of basis terms.
    - Refit selected terms with OLS for calibration.
    """

    def __init__(
        self,
        screen_features=14,
        max_terms=8,
        interaction_candidates=3,
        random_state=42,
    ):
        self.screen_features = screen_features
        self.max_terms = max_terms
        self.interaction_candidates = interaction_candidates
        self.random_state = random_state

    def _impute(self, X):
        X = np.asarray(X, dtype=float).copy()
        nan_mask = ~np.isfinite(X)
        if nan_mask.any():
            X[nan_mask] = np.take(self.feature_medians_, np.where(nan_mask)[1])
        return X

    def _corr_abs(self, a, b):
        ac = a - np.mean(a)
        bc = b - np.mean(b)
        return abs(float(np.mean(ac * bc) / ((np.std(ac) + 1e-12) * (np.std(bc) + 1e-12))))

    def _screen_indices(self, X, y):
        scores = []
        for j in range(X.shape[1]):
            xj = X[:, j]
            s = self._corr_abs(xj, y)
            s += 0.5 * self._corr_abs(xj * xj, y)
            s += 0.25 * self._corr_abs(np.abs(xj), y)
            scores.append(s)
        k = min(self.screen_features, X.shape[1])
        return np.argsort(np.asarray(scores))[::-1][:k]

    def _term_values(self, X, spec):
        kind = spec["kind"]
        if kind == "linear":
            return X[:, spec["j"]]
        if kind == "abs":
            return np.abs(X[:, spec["j"]])
        if kind == "hinge_pos":
            return np.maximum(0.0, X[:, spec["j"]] - spec["knot"])
        if kind == "hinge_neg":
            return np.maximum(0.0, spec["knot"] - X[:, spec["j"]])
        if kind == "interaction":
            return X[:, spec["j1"]] * X[:, spec["j2"]]
        raise ValueError(f"Unknown term kind: {kind}")

    def _term_to_text(self, spec):
        kind = spec["kind"]
        if kind == "linear":
            return f"x{spec['j']}"
        if kind == "abs":
            return f"|x{spec['j']}|"
        if kind == "hinge_pos":
            return f"max(0, x{spec['j']} - {spec['knot']:.3f})"
        if kind == "hinge_neg":
            return f"max(0, {spec['knot']:.3f} - x{spec['j']})"
        if kind == "interaction":
            return f"x{spec['j1']}*x{spec['j2']}"
        return str(spec)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.feature_medians_ = np.nanmedian(X, axis=0)
        X = self._impute(X)

        screened = self._screen_indices(X, y)
        self.screened_features_ = screened.astype(int)

        specs = []
        for j in self.screened_features_:
            xj = X[:, j]
            q40, q60 = np.quantile(xj, [0.40, 0.60])
            specs.append({"kind": "linear", "j": int(j)})
            specs.append({"kind": "abs", "j": int(j)})
            specs.append({"kind": "hinge_pos", "j": int(j), "knot": float(q60)})
            specs.append({"kind": "hinge_neg", "j": int(j), "knot": float(q40)})

        top_inter_feats = self.screened_features_[: min(6, len(self.screened_features_))]
        interaction_specs = []
        if len(top_inter_feats) >= 2:
            int_scores = []
            for a in range(len(top_inter_feats)):
                for b in range(a + 1, len(top_inter_feats)):
                    j1 = int(top_inter_feats[a])
                    j2 = int(top_inter_feats[b])
                    v = X[:, j1] * X[:, j2]
                    int_scores.append((self._corr_abs(v, y), j1, j2))
            int_scores.sort(reverse=True)
            for _, j1, j2 in int_scores[: self.interaction_candidates]:
                interaction_specs.append({"kind": "interaction", "j1": j1, "j2": j2})
        specs.extend(interaction_specs)

        Z = np.column_stack([self._term_values(X, s) for s in specs]) if specs else np.zeros((X.shape[0], 0))
        if Z.shape[1] == 0:
            self.intercept_ = float(np.mean(y))
            self.active_specs_ = []
            self.active_coef_ = np.array([], dtype=float)
            self.feature_importance_ = np.zeros(self.n_features_in_, dtype=float)
            return self

        z_mean = np.mean(Z, axis=0)
        z_std = np.std(Z, axis=0)
        z_std[z_std < 1e-12] = 1.0
        Zs = (Z - z_mean) / z_std

        selector = LassoCV(cv=3, random_state=self.random_state, n_alphas=25, max_iter=3000)
        selector.fit(Zs, y)
        coef_scaled = selector.coef_ / z_std

        active = np.where(np.abs(coef_scaled) > 1e-8)[0]
        if active.size == 0:
            active = np.array([int(np.argmax(np.abs(coef_scaled)))], dtype=int)
        if active.size > self.max_terms:
            keep_order = np.argsort(np.abs(coef_scaled[active]))[::-1][: self.max_terms]
            active = active[keep_order]

        reg = LinearRegression()
        reg.fit(Z[:, active], y)

        self.intercept_ = float(reg.intercept_)
        self.active_specs_ = [specs[i] for i in active]
        self.active_coef_ = reg.coef_.astype(float)

        fi = np.zeros(self.n_features_in_, dtype=float)
        for c, spec in zip(self.active_coef_, self.active_specs_):
            w = abs(float(c))
            if spec["kind"] in {"linear", "abs", "hinge_pos", "hinge_neg"}:
                fi[spec["j"]] += w
            elif spec["kind"] == "interaction":
                fi[spec["j1"]] += 0.5 * w
                fi[spec["j2"]] += 0.5 * w
        self.feature_importance_ = fi
        return self

    def predict(self, X):
        check_is_fitted(self, ["feature_importance_", "active_specs_", "active_coef_", "intercept_"])
        X = self._impute(X)
        if len(self.active_specs_) == 0:
            return np.full(X.shape[0], self.intercept_, dtype=float)
        Z = np.column_stack([self._term_values(X, s) for s in self.active_specs_])
        return self.intercept_ + Z @ self.active_coef_

    def __str__(self):
        check_is_fitted(self, ["feature_importance_", "active_specs_", "active_coef_", "intercept_"])
        lines = ["SparseAdditiveHingeRegressor", "Equation:", f"y = {self.intercept_:+.4f}"]
        for c, spec in sorted(
            zip(self.active_coef_, self.active_specs_),
            key=lambda x: abs(x[0]),
            reverse=True,
        ):
            lines.append(f"  {c:+.4f} * {self._term_to_text(spec)}")

        lines.append("")
        lines.append("Feature importance (sum of absolute term coefficients):")
        rank = np.argsort(self.feature_importance_)[::-1]
        for j in rank:
            lines.append(f"  x{j}: {self.feature_importance_[j]:.4f}")
        near_zero = [f"x{j}" for j, v in enumerate(self.feature_importance_) if v < 1e-4]
        if near_zero:
            lines.append("Features with near-zero effect: " + ", ".join(near_zero))
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparseAdditiveHingeRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SparseAddHingeV1"
model_description = "Sparse additive symbolic regressor with linear, hinge, abs, and few interaction terms selected by L1"
model_defs = [(model_shorthand_name, SparseAdditiveHingeRegressor())]


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
