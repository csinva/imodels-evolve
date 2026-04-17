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


class SymbolicResidualHybridRegressor(BaseEstimator, RegressorMixin):
    """Ridge linear backbone plus a tiny residual symbolic correction set."""

    def __init__(
        self,
        ridge_alpha=1.0,
        max_screen_features=10,
        max_extra_terms=2,
        min_residual_gain=1e-3,
        top_terms_for_display=12,
        negligible_coef_eps=1e-4,
    ):
        self.ridge_alpha = ridge_alpha
        self.max_screen_features = max_screen_features
        self.max_extra_terms = max_extra_terms
        self.min_residual_gain = min_residual_gain
        self.top_terms_for_display = top_terms_for_display
        self.negligible_coef_eps = negligible_coef_eps

    @staticmethod
    def _corr(a, b):
        a0 = a - np.mean(a)
        b0 = b - np.mean(b)
        den = (np.std(a0) + 1e-12) * (np.std(b0) + 1e-12)
        return float(np.mean(a0 * b0) / den)

    @staticmethod
    def _ridge_fit(X, y, alpha):
        p = X.shape[1]
        return np.linalg.solve(X.T @ X + float(alpha) * np.eye(p), X.T @ y)

    def _best_hinge_for_feature(self, x_col, residual):
        best = None
        for q in (0.2, 0.35, 0.5, 0.65, 0.8):
            knot = float(np.quantile(x_col, q))
            hp = np.maximum(0.0, x_col - knot)
            hn = np.maximum(0.0, knot - x_col)
            sp = abs(self._corr(hp, residual))
            sn = abs(self._corr(hn, residual))
            if (best is None) or (sp > best[0]):
                best = (sp, "hinge_pos", knot, hp)
            if (best is None) or (sn > best[0]):
                best = (sn, "hinge_neg", knot, hn)
        return best

    def _term_values(self, X, term):
        t = term["type"]
        if t == "hinge_pos":
            return np.maximum(0.0, X[:, term["feat"]] - term["knot"])
        if t == "hinge_neg":
            return np.maximum(0.0, term["knot"] - X[:, term["feat"]])
        if t == "interaction":
            return X[:, term["feat_a"]] * X[:, term["feat_b"]]
        raise ValueError(f"Unknown term type: {t}")

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        self.x_mean_ = np.mean(X, axis=0)
        self.x_scale_ = np.std(X, axis=0) + 1e-12
        Xz = (X - self.x_mean_) / self.x_scale_

        self.y_mean_ = float(np.mean(y))
        y_centered = y - self.y_mean_

        # Dense ridge backbone in standardized coordinates.
        beta_z = self._ridge_fit(Xz, y_centered, self.ridge_alpha)
        beta_raw = beta_z / self.x_scale_
        intercept = self.y_mean_ - float(np.dot(self.x_mean_, beta_raw))

        y_lin = intercept + X @ beta_raw
        residual = y - y_lin

        # Screen features by residual correlation and add at most 2 symbolic residual terms.
        corr = np.array([abs(self._corr(X[:, j], residual)) for j in range(n_features)])
        order = np.argsort(corr)[::-1]
        screened = [int(j) for j in order[: min(max(int(self.max_screen_features), 1), n_features)]]

        candidate_terms = []
        for j in screened:
            hbest = self._best_hinge_for_feature(X[:, j], residual)
            if hbest is not None:
                score, tname, knot, _ = hbest
                candidate_terms.append(
                    (score, {"type": tname, "feat": int(j), "knot": float(knot)})
                )

        for i, a in enumerate(screened):
            for b in screened[i + 1 :]:
                col = X[:, a] * X[:, b]
                score = abs(self._corr(col, residual))
                candidate_terms.append(
                    (score, {"type": "interaction", "feat_a": int(a), "feat_b": int(b)})
                )

        candidate_terms.sort(key=lambda x: x[0], reverse=True)
        selected_terms = []
        work_resid = residual.copy()
        for score, term in candidate_terms:
            if len(selected_terms) >= max(int(self.max_extra_terms), 0):
                break
            col = self._term_values(X, term)
            gain = abs(self._corr(col, work_resid))
            if gain < float(self.min_residual_gain):
                continue
            selected_terms.append(term)
            c = float(np.dot(col, work_resid) / (np.dot(col, col) + 1e-12))
            work_resid = work_resid - c * col

        # Refit joint model: all linear terms + selected symbolic terms.
        cols = [X[:, j] for j in range(n_features)]
        cols.extend(self._term_values(X, t) for t in selected_terms)
        Z = np.column_stack(cols)
        coef_joint = self._ridge_fit(Z, y - np.mean(y), self.ridge_alpha)

        self.linear_coef_ = coef_joint[:n_features]
        self.extra_terms_ = selected_terms
        self.extra_coef_ = coef_joint[n_features:]
        self.intercept_ = float(np.mean(y))

        # Importances for eval/string readability.
        imp = np.abs(self.linear_coef_).astype(float)
        for c, t in zip(self.extra_coef_, self.extra_terms_):
            if t["type"].startswith("hinge"):
                imp[t["feat"]] += abs(float(c))
            else:
                imp[t["feat_a"]] += 0.5 * abs(float(c))
                imp[t["feat_b"]] += 0.5 * abs(float(c))
        if np.max(imp) > 0:
            imp = imp / np.max(imp)
        self.feature_importances_ = imp

        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "linear_coef_", "extra_terms_", "extra_coef_"])
        X = np.asarray(X, dtype=float)
        pred = self.intercept_ + X @ self.linear_coef_
        for c, t in zip(self.extra_coef_, self.extra_terms_):
            pred += float(c) * self._term_values(X, t)
        return pred

    def _term_to_str(self, term):
        if term["type"] == "hinge_pos":
            return f"max(0, x{term['feat']}-{term['knot']:.4f})"
        if term["type"] == "hinge_neg":
            return f"max(0, {term['knot']:.4f}-x{term['feat']})"
        return f"x{term['feat_a']}*x{term['feat_b']}"

    def __str__(self):
        check_is_fitted(self, ["intercept_", "linear_coef_", "extra_terms_", "extra_coef_", "feature_importances_"])

        shown = np.argsort(np.abs(self.linear_coef_))[::-1]

        eq_parts = [f"{self.intercept_:+.4f}"]
        for j in shown:
            c = float(self.linear_coef_[j])
            if abs(c) > self.negligible_coef_eps:
                eq_parts.append(f"{c:+.4f}*x{int(j)}")
        for c, t in zip(self.extra_coef_, self.extra_terms_):
            if abs(float(c)) > self.negligible_coef_eps:
                eq_parts.append(f"{float(c):+.4f}*{self._term_to_str(t)}")

        ranked = np.argsort(self.feature_importances_)[::-1]
        ranking_txt = ", ".join(
            f"x{int(j)}:{self.feature_importances_[j]:.3f}" for j in ranked[: min(10, self.n_features_in_)]
        )

        lines = [
            "Symbolic Residual Hybrid Regressor",
            "Exact prediction rule:",
            "  y = " + " ".join(eq_parts),
            "",
            "Top feature influence ranking: " + ranking_txt,
        ]

        if self.extra_terms_:
            lines.append("Residual symbolic corrections:")
            for c, t in zip(self.extra_coef_, self.extra_terms_):
                lines.append(f"  {float(c):+.4f} * {self._term_to_str(t)}")

        near_zero = [f"x{i}" for i, c in enumerate(self.linear_coef_) if abs(float(c)) < self.negligible_coef_eps]
        lines.append("Near-zero linear effects: " + (", ".join(near_zero) if near_zero else "none"))

        return "\n".join(lines)
# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SymbolicResidualHybridRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SymbolicResidualHybridDense_v1"
model_description = "Dense ridge linear backbone with at most two residual symbolic corrections (hinge/interaction), exported as an exact raw-feature equation"
model_defs = [(model_shorthand_name, SymbolicResidualHybridRegressor())]


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
