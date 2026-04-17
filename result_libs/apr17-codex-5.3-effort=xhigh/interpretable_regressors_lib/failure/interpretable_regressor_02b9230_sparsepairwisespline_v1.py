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


class SparsePairwiseSplineRegressor(BaseEstimator, RegressorMixin):
    """Sparse additive spline model with optional single interaction and explicit symbolic output."""

    def __init__(
        self,
        ridge_alpha=1.5,
        max_active_features=12,
        max_retained_features=5,
        knots_per_feature=2,
        include_interaction=True,
        interaction_screen_features=6,
        min_group_norm=1e-4,
        negligible_coef_eps=1e-8,
    ):
        self.ridge_alpha = ridge_alpha
        self.max_active_features = max_active_features
        self.max_retained_features = max_retained_features
        self.knots_per_feature = knots_per_feature
        self.include_interaction = include_interaction
        self.interaction_screen_features = interaction_screen_features
        self.min_group_norm = min_group_norm
        self.negligible_coef_eps = negligible_coef_eps

    @staticmethod
    def _safe_corr(a, b):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        a0 = a - np.mean(a)
        b0 = b - np.mean(b)
        den = np.sqrt(np.mean(a0 * a0) * np.mean(b0 * b0))
        if den <= 1e-12:
            return 0.0
        return float(np.mean(a0 * b0) / den)

    @staticmethod
    def _ridge_fit(X, y, alpha):
        p = X.shape[1]
        return np.linalg.solve(X.T @ X + float(alpha) * np.eye(p), X.T @ y)

    @staticmethod
    def _ols_fit(X, y):
        return np.linalg.lstsq(X, y, rcond=None)[0]

    def _quantile_knots(self, x):
        n_knots = max(0, int(self.knots_per_feature))
        if n_knots == 0:
            return []
        qs = np.linspace(0.2, 0.8, n_knots + 2)[1:-1]
        knots = []
        for q in qs:
            k = float(np.quantile(x, q))
            if not knots or abs(k - knots[-1]) > 1e-10:
                knots.append(k)
        return knots

    def _build_design(self, X, active_features, include_pair=None):
        n = X.shape[0]
        cols = [np.ones(n, dtype=float)]
        terms = [("bias", None)]
        group_of_term = [-1]

        feature_knots = {}
        for j in active_features:
            xj = X[:, j]
            cols.append(xj)
            terms.append(("linear", int(j)))
            group_of_term.append(int(j))

            knots = self._quantile_knots(xj)
            feature_knots[int(j)] = knots
            for k in knots:
                cols.append(np.maximum(0.0, xj - k))
                terms.append(("hinge_plus", (int(j), float(k))))
                group_of_term.append(int(j))
                cols.append(np.maximum(0.0, k - xj))
                terms.append(("hinge_minus", (int(j), float(k))))
                group_of_term.append(int(j))

        if include_pair is not None:
            j, k = include_pair
            xj = X[:, j] - self.feature_means_[j]
            xk = X[:, k] - self.feature_means_[k]
            cols.append(xj * xk)
            terms.append(("interaction", (int(j), int(k))))
            group_of_term.append(-2)

        return np.column_stack(cols), terms, group_of_term, feature_knots

    def _term_values(self, X, term_type, meta):
        if term_type == "bias":
            return np.ones(X.shape[0], dtype=float)
        if term_type == "linear":
            return X[:, int(meta)]
        if term_type == "hinge_plus":
            j, k = meta
            return np.maximum(0.0, X[:, int(j)] - float(k))
        if term_type == "hinge_minus":
            j, k = meta
            return np.maximum(0.0, float(k) - X[:, int(j)])
        if term_type == "interaction":
            j, k = meta
            return (X[:, int(j)] - self.feature_means_[int(j)]) * (X[:, int(k)] - self.feature_means_[int(k)])
        raise ValueError(f"Unknown term type: {term_type}")

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        _, n_features = X.shape
        self.n_features_in_ = n_features
        self.feature_means_ = np.mean(X, axis=0)

        self.intercept_ = float(np.mean(y))
        y_centered = y - self.intercept_

        corr = np.array([abs(self._safe_corr(X[:, j], y_centered)) for j in range(n_features)], dtype=float)
        screened = np.argsort(corr)[::-1]
        n_active = max(1, min(int(self.max_active_features), n_features))
        self.active_features_ = np.array([int(j) for j in screened[:n_active]], dtype=int)

        interaction_pair = None
        if self.include_interaction and len(self.active_features_) >= 2:
            cand = self.active_features_[: max(2, min(int(self.interaction_screen_features), len(self.active_features_)))]
            best_score = -1.0
            for i in range(len(cand)):
                fi = int(cand[i])
                for j in range(i + 1, len(cand)):
                    fj = int(cand[j])
                    prod = (X[:, fi] - self.feature_means_[fi]) * (X[:, fj] - self.feature_means_[fj])
                    score = abs(self._safe_corr(prod, y_centered))
                    if score > best_score:
                        best_score = score
                        interaction_pair = (fi, fj)

        design_full, terms_full, groups_full, full_knots = self._build_design(
            X, self.active_features_, include_pair=interaction_pair
        )
        coef_full = self._ridge_fit(design_full, y_centered, self.ridge_alpha)

        group_norms = {}
        for idx, g in enumerate(groups_full):
            if g < 0:
                continue
            group_norms[g] = group_norms.get(g, 0.0) + float(coef_full[idx] ** 2)
        for g in list(group_norms.keys()):
            group_norms[g] = np.sqrt(group_norms[g])

        ranked_groups = sorted(group_norms.items(), key=lambda kv: kv[1], reverse=True)
        kept_groups = [
            int(g) for g, norm in ranked_groups
            if norm >= float(self.min_group_norm)
        ][: max(1, int(self.max_retained_features))]
        if not kept_groups and len(self.active_features_) > 0:
            kept_groups = [int(self.active_features_[0])]

        keep_term_idx = [0]
        for idx, g in enumerate(groups_full):
            if g in kept_groups:
                keep_term_idx.append(idx)
        if interaction_pair is not None and groups_full[-1] == -2 and abs(float(coef_full[-1])) > 0.02:
            if interaction_pair[0] in kept_groups and interaction_pair[1] in kept_groups:
                keep_term_idx.append(len(groups_full) - 1)

        keep_term_idx = np.array(sorted(set(keep_term_idx)), dtype=int)
        terms_small = [terms_full[i] for i in keep_term_idx]
        design_small = np.column_stack([self._term_values(X, t, m) for t, m in terms_small])
        coef_small = self._ols_fit(design_small, y_centered)
        coef_small[np.abs(coef_small) < self.negligible_coef_eps] = 0.0

        self.selected_terms_ = terms_small
        self.coef_ = coef_small
        self.kept_features_ = np.array(sorted(set(kept_groups)), dtype=int)
        self.feature_knots_ = {j: full_knots.get(j, []) for j in self.kept_features_}
        self.interaction_pair_ = interaction_pair

        importances = np.zeros(n_features, dtype=float)
        linear_raw = np.zeros(n_features, dtype=float)
        for c, (tt, meta) in zip(self.coef_, self.selected_terms_):
            mag = abs(float(c))
            if tt == "linear":
                j = int(meta)
                importances[j] += mag
                linear_raw[j] += float(c)
            elif tt in {"hinge_plus", "hinge_minus"}:
                j = int(meta[0])
                importances[j] += mag
            elif tt == "interaction":
                j, k = meta
                importances[int(j)] += 0.5 * mag
                importances[int(k)] += 0.5 * mag

        mx = np.max(importances) if importances.size else 0.0
        self.feature_importances_ = importances / mx if mx > 0 else importances
        self.linear_coef_full_ = linear_raw
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "selected_terms_", "coef_", "feature_means_"])
        X = np.asarray(X, dtype=float)
        pred = np.full(X.shape[0], self.intercept_, dtype=float)
        for c, (tt, meta) in zip(self.coef_, self.selected_terms_):
            pred += float(c) * self._term_values(X, tt, meta)
        return pred

    def __str__(self):
        check_is_fitted(
            self,
            [
                "intercept_",
                "n_features_in_",
                "active_features_",
                "kept_features_",
                "selected_terms_",
                "coef_",
                "feature_importances_",
                "linear_coef_full_",
            ],
        )

        eq = [f"{self.intercept_:+.4f}"]
        for c, (tt, meta) in zip(self.coef_, self.selected_terms_):
            if abs(float(c)) < self.negligible_coef_eps or tt == "bias":
                continue
            if tt == "linear":
                j = int(meta)
                eq.append(f"{float(c):+.4f}*x{j}")
            elif tt == "hinge_plus":
                j, k = meta
                eq.append(f"{float(c):+.4f}*max(0, x{int(j)}-{float(k):.4f})")
            elif tt == "hinge_minus":
                j, k = meta
                eq.append(f"{float(c):+.4f}*max(0, {float(k):.4f}-x{int(j)})")
            elif tt == "interaction":
                j, k = meta
                mj = float(self.feature_means_[int(j)])
                mk = float(self.feature_means_[int(k)])
                eq.append(f"{float(c):+.4f}*(x{int(j)}-{mj:+.4f})*(x{int(k)}-{mk:+.4f})")

        ranked = np.argsort(self.feature_importances_)[::-1]
        important = [int(j) for j in ranked if self.feature_importances_[j] > 0]
        inactive = [f"x{j}" for j in range(self.n_features_in_) if j not in set(self.kept_features_)]

        summary_lines = []
        for j in important[:10]:
            base = float(self.linear_coef_full_[j])
            knot_terms = []
            for c, (tt, meta) in zip(self.coef_, self.selected_terms_):
                if tt in {"hinge_plus", "hinge_minus"} and int(meta[0]) == j:
                    direction = "right-slope+"
                    if tt == "hinge_minus":
                        direction = "left-slope-"
                    knot_terms.append(f"{direction}{float(c):+.3f}@{float(meta[1]):.3f}")
            knots_txt = ", ".join(knot_terms) if knot_terms else "none"
            summary_lines.append(
                f"x{j}: importance={self.feature_importances_[j]:.3f}, base_linear={base:+.4f}, hinges={knots_txt}"
            )

        lines = [
            "Sparse Pairwise Spline Regressor",
            "Prediction equation (raw features):",
            "  y = " + " ".join(eq),
            "",
            "Most influential features:",
            *("  " + s for s in (summary_lines if summary_lines else ["none"])),
            "Screened active features: " + ", ".join(f"x{j}" for j in self.active_features_),
            "Inactive (pruned) features: " + (", ".join(inactive) if inactive else "none"),
        ]
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys

_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])

# Update the model shorthand name and description below to reflect the class above and any changes you make.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
SparsePairwiseSplineRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "SparsePairwiseSpline_v1"
model_description = "Two-stage sparse additive spline regressor: correlation screening, quantile hinge pairs per feature, optional centered interaction, group pruning, and symbolic raw-feature equation"
model_defs = [(model_shorthand_name, SparsePairwiseSplineRegressor())]


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
