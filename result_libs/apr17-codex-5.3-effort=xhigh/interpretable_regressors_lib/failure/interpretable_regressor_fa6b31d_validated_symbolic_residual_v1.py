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


class ValidatedSymbolicResidualRegressor(BaseEstimator, RegressorMixin):
    """OLS backbone with up to two validation-selected symbolic residual terms."""

    def __init__(
        self,
        max_candidate_features=8,
        candidate_quantiles=(0.25, 0.5, 0.75),
        max_extra_terms=2,
        min_val_gain=0.002,
        val_fraction=0.2,
        random_state=0,
        negligible_coef_eps=0.015,
    ):
        self.max_candidate_features = max_candidate_features
        self.candidate_quantiles = candidate_quantiles
        self.max_extra_terms = max_extra_terms
        self.min_val_gain = min_val_gain
        self.val_fraction = val_fraction
        self.random_state = random_state
        self.negligible_coef_eps = negligible_coef_eps

    def _safe_abs_corr(self, a, b):
        sa = float(np.std(a))
        sb = float(np.std(b))
        if sa < 1e-12 or sb < 1e-12:
            return 0.0
        c = np.corrcoef(a, b)[0, 1]
        if not np.isfinite(c):
            return 0.0
        return float(abs(c))

    def _solve_ols(self, Z, y):
        z_mean = np.mean(Z, axis=0)
        z_std = np.std(Z, axis=0)
        z_std[z_std < 1e-12] = 1.0
        Zs = (Z - z_mean) / z_std

        y_mean = float(np.mean(y))
        y_c = y - y_mean

        coef_std, *_ = np.linalg.lstsq(Zs, y_c, rcond=None)
        coef = coef_std / z_std
        intercept = float(y_mean - np.dot(coef, z_mean))
        return intercept, coef

    def _basis_values(self, X, term):
        feat = term["feature"]
        knot = term["knot"]
        x = X[:, feat]
        if term["kind"] == "hinge_pos":
            return np.maximum(0.0, x - knot)
        if term["kind"] == "hinge_neg":
            return np.maximum(0.0, knot - x)
        if term["kind"] == "abs":
            return np.abs(x - knot)
        if term["kind"] == "square":
            return (x - knot) ** 2
        if term["kind"] == "interaction":
            feat2 = term["feature2"]
            return x * X[:, feat2]
        raise ValueError(f"Unknown basis kind {term['kind']}")

    def _build_candidates(self, X, residual):
        n_features = X.shape[1]
        feat_corr = np.array([self._safe_abs_corr(X[:, j], residual) for j in range(n_features)], dtype=float)
        ranked_feats = np.argsort(feat_corr)[::-1]
        top = [int(f) for f in ranked_feats[: min(self.max_candidate_features, n_features)]]
        terms = []

        for feat in top:
            xj = X[:, feat]
            knots = np.unique(np.round(np.quantile(xj, self.candidate_quantiles), 8))
            mean_knot = float(np.mean(xj))
            terms.append({"kind": "square", "feature": feat, "knot": mean_knot})
            for knot in knots:
                terms.append({"kind": "hinge_pos", "feature": feat, "knot": float(knot)})
                terms.append({"kind": "hinge_neg", "feature": feat, "knot": float(knot)})
                terms.append({"kind": "abs", "feature": feat, "knot": float(knot)})

        for i in range(len(top)):
            for j in range(i + 1, len(top)):
                terms.append({
                    "kind": "interaction",
                    "feature": int(top[i]),
                    "feature2": int(top[j]),
                    "knot": 0.0,
                })

        return terms

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

        self.selected_terms_ = []

        base_intercept, base_coef = self._solve_ols(X_tr, y_tr)
        pred_val = base_intercept + X_val @ base_coef
        best_val_mse = float(np.mean((y_val - pred_val) ** 2))

        base_pred_tr = base_intercept + X_tr @ base_coef
        residual_tr = y_tr - base_pred_tr

        for _ in range(self.max_extra_terms):
            candidates = self._build_candidates(X_tr, residual_tr)
            chosen = None
            chosen_coef_full = None
            chosen_intercept = None
            chosen_mse = best_val_mse

            extra_tr_cols = [self._basis_values(X_tr, t) for t in self.selected_terms_]
            extra_val_cols = [self._basis_values(X_val, t) for t in self.selected_terms_]
            Z_tr_base = X_tr if not extra_tr_cols else np.column_stack([X_tr] + extra_tr_cols)
            Z_val_base = X_val if not extra_val_cols else np.column_stack([X_val] + extra_val_cols)

            for term in candidates:
                z_tr = self._basis_values(X_tr, term)
                z_val = self._basis_values(X_val, term)
                if float(np.std(z_tr)) < 1e-10:
                    continue
                Z_tr_try = np.column_stack([Z_tr_base, z_tr])
                Z_val_try = np.column_stack([Z_val_base, z_val])
                intercept_try, coef_try = self._solve_ols(Z_tr_try, y_tr)
                val_pred_try = intercept_try + Z_val_try @ coef_try
                mse_try = float(np.mean((y_val - val_pred_try) ** 2))
                if mse_try < chosen_mse:
                    chosen = term
                    chosen_mse = mse_try
                    chosen_intercept = intercept_try
                    chosen_coef_full = coef_try

            rel_gain = (best_val_mse - chosen_mse) / (best_val_mse + 1e-12)
            if chosen is None or rel_gain < self.min_val_gain:
                break

            self.selected_terms_.append(chosen)
            best_val_mse = chosen_mse

            n_main = n_features
            n_extra = len(self.selected_terms_)
            self.intercept_ = float(chosen_intercept)
            self.linear_coef_ = np.asarray(chosen_coef_full[:n_main], dtype=float)
            self.extra_coefs_ = np.asarray(chosen_coef_full[n_main:n_main + n_extra], dtype=float)

            residual_tr = y_tr - (
                self.intercept_
                + X_tr @ self.linear_coef_
                + sum(
                    self.extra_coefs_[k] * self._basis_values(X_tr, self.selected_terms_[k])
                    for k in range(n_extra)
                )
            )

        if not self.selected_terms_:
            self.intercept_, self.linear_coef_ = self._solve_ols(X, y)
            self.extra_coefs_ = np.zeros(0, dtype=float)
        else:
            full_cols = [self._basis_values(X, t) for t in self.selected_terms_]
            Z_full = np.column_stack([X] + full_cols)
            self.intercept_, coef_full = self._solve_ols(Z_full, y)
            self.linear_coef_ = np.asarray(coef_full[:n_features], dtype=float)
            self.extra_coefs_ = np.asarray(coef_full[n_features:], dtype=float)

        abs_linear = np.abs(self.linear_coef_)
        self.feature_importances_ = abs_linear.copy()
        for coef, term in zip(self.extra_coefs_, self.selected_terms_):
            self.feature_importances_[term["feature"]] += abs(float(coef))
            if term["kind"] == "interaction":
                self.feature_importances_[term["feature2"]] += abs(float(coef))

        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "linear_coef_", "selected_terms_", "extra_coefs_"])
        X = np.asarray(X, dtype=float)
        pred = self.intercept_ + X @ self.linear_coef_
        for coef, term in zip(self.extra_coefs_, self.selected_terms_):
            pred += coef * self._basis_values(X, term)
        return pred

    def _term_to_str(self, coef, term):
        feat = term["feature"]
        knot = term["knot"]
        kind = term["kind"]
        if kind == "hinge_pos":
            return f"{coef:+.8f}*max(0, x{feat}-{knot:.6f})"
        if kind == "hinge_neg":
            return f"{coef:+.8f}*max(0, {knot:.6f}-x{feat})"
        if kind == "abs":
            return f"{coef:+.8f}*abs(x{feat}-{knot:.6f})"
        if kind == "square":
            return f"{coef:+.8f}*(x{feat}-{knot:.6f})^2"
        if kind == "interaction":
            return f"{coef:+.8f}*x{feat}*x{term['feature2']}"
        return None

    def _extra_term_summary(self):
        if len(self.selected_terms_) == 0:
            return None
        return [self._term_to_str(float(c), t) for c, t in zip(self.extra_coefs_, self.selected_terms_)]

    def __str__(self):
        check_is_fitted(self, ["intercept_", "linear_coef_", "feature_importances_", "selected_terms_", "extra_coefs_"])

        order = np.argsort(np.abs(self.linear_coef_))[::-1]
        active = [int(j) for j in order if abs(self.linear_coef_[j]) > self.negligible_coef_eps]
        negligible = [f"x{j}" for j in range(self.n_features_in_) if abs(self.linear_coef_[j]) <= self.negligible_coef_eps]

        equation_terms = [f"{self.intercept_:+.8f}"]
        for j in range(self.n_features_in_):
            equation_terms.append(f"{self.linear_coef_[j]:+.8f}*x{j}")
        extra_terms = self._extra_term_summary()
        if extra_terms is not None:
            equation_terms.extend(extra_terms)

        top_importance = np.argsort(self.feature_importances_)[::-1]
        top_text = ", ".join(
            f"x{j}:{self.feature_importances_[j]:.4f}" for j in top_importance[: min(10, self.n_features_in_)]
        )

        lines = [
            "Validated Symbolic Residual Regressor",
            "Model form: y = intercept + linear terms + up to two symbolic residual corrections",
            "Compact equation:",
            "  y = " + " ".join(equation_terms),
            "",
            "Linear coefficients (sorted by absolute magnitude):",
        ]

        for j in active:
            lines.append(f"  x{j}: {self.linear_coef_[j]:+.8f}")

        lines.extend([
            "",
            "Residual correction terms:",
        ])
        if extra_terms is None:
            lines.append("  (none)")
        else:
            for t in extra_terms:
                lines.append(f"  {t}")

        lines.extend([
            "",
            "Features with negligible linear effect (near-zero coefficients):",
            "  " + (", ".join(negligible) if negligible else "none"),
            "",
            "Most influential features (linear coefficient magnitude + correction attribution):",
            "  " + top_text,
            "",
            "Simulation recipe: plug feature values into the compact equation directly.",
        ])

        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
ValidatedSymbolicResidualRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "ValidatedSymbolicResidual_v1"
model_description = "OLS backbone with validation-gated symbolic residual terms (hinge, abs, square, interaction) selected greedily for compact nonlinear correction"
model_defs = [(model_shorthand_name, ValidatedSymbolicResidualRegressor())]


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
