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


class ScreenedOMPFormulaRegressor(BaseEstimator, RegressorMixin):
    """
    Compact symbolic regressor:
    1) generate transparent raw-feature basis terms (linear/abs/square/hinge/interaction),
    2) greedily select a small subset by residual correlation,
    3) ridge-refit the selected equation for stability.
    """

    def __init__(
        self,
        max_terms=10,
        nonlinear_screen_features=6,
        interaction_screen_features=8,
        max_interactions=10,
        ridge_alpha=0.08,
        min_rel_improvement=1e-4,
        coef_tol=1e-4,
        random_state=42,
    ):
        self.max_terms = max_terms
        self.nonlinear_screen_features = nonlinear_screen_features
        self.interaction_screen_features = interaction_screen_features
        self.max_interactions = max_interactions
        self.ridge_alpha = ridge_alpha
        self.min_rel_improvement = min_rel_improvement
        self.coef_tol = coef_tol
        self.random_state = random_state

    def _impute(self, X):
        X = np.asarray(X, dtype=float).copy()
        bad = ~np.isfinite(X)
        if bad.any():
            X[bad] = np.take(self.feature_medians_, np.where(bad)[1])
        return X

    @staticmethod
    def _center(v):
        return v - float(np.mean(v))

    @staticmethod
    def _corr_score(a, b):
        ac = a - float(np.mean(a))
        bc = b - float(np.mean(b))
        denom = (float(np.std(ac)) + 1e-12) * (float(np.std(bc)) + 1e-12)
        return abs(float(np.mean(ac * bc)) / denom)

    @staticmethod
    def _term_value(X, meta):
        t = meta["kind"]
        if t == "lin":
            return X[:, meta["j"]]
        if t == "abs":
            return np.abs(X[:, meta["j"]])
        if t == "sq":
            xj = X[:, meta["j"]]
            return xj * xj
        if t == "hinge":
            return np.maximum(0.0, X[:, meta["j"]] - meta["thr"])
        if t == "int":
            return X[:, meta["j"]] * X[:, meta["k"]]
        raise ValueError(f"Unknown term kind: {t}")

    @staticmethod
    def _term_name(meta):
        t = meta["kind"]
        if t == "lin":
            return f"x{meta['j']}"
        if t == "abs":
            return f"abs(x{meta['j']})"
        if t == "sq":
            return f"x{meta['j']}^2"
        if t == "hinge":
            return f"max(0, x{meta['j']} - {meta['thr']:+.4f})"
        if t == "int":
            return f"x{meta['j']}*x{meta['k']}"
        return "unknown"

    def _build_terms(self, X, y):
        p = X.shape[1]
        terms = [{"kind": "lin", "j": int(j)} for j in range(p)]

        screen = np.array([
            self._corr_score(X[:, j], y)
            + 0.4 * self._corr_score(np.abs(X[:, j]), y)
            + 0.4 * self._corr_score(X[:, j] * X[:, j], y)
            for j in range(p)
        ])
        top_nonlin = np.argsort(screen)[::-1][: min(int(self.nonlinear_screen_features), p)]
        for j in top_nonlin:
            j = int(j)
            terms.append({"kind": "abs", "j": j})
            terms.append({"kind": "sq", "j": j})
            terms.append({"kind": "hinge", "j": j, "thr": float(np.median(X[:, j]))})

        top_int_base = np.argsort(screen)[::-1][: min(int(self.interaction_screen_features), p)]
        pair_scores = []
        for a in range(len(top_int_base)):
            j = int(top_int_base[a])
            for b in range(a + 1, len(top_int_base)):
                k = int(top_int_base[b])
                pair_scores.append((self._corr_score(X[:, j] * X[:, k], y), j, k))
        pair_scores.sort(key=lambda z: z[0], reverse=True)
        for _, j, k in pair_scores[: int(self.max_interactions)]:
            terms.append({"kind": "int", "j": int(j), "k": int(k)})

        term_values = [self._term_value(X, m) for m in terms]
        term_values_centered = [self._center(v) for v in term_values]
        return terms, term_values, term_values_centered

    def _ridge_refit(self, Z, y):
        n = Z.shape[0]
        A = np.column_stack([np.ones(n, dtype=float), Z])
        reg = np.eye(A.shape[1], dtype=float)
        reg[0, 0] = 0.0
        lhs = A.T @ A + float(self.ridge_alpha) * reg
        rhs = A.T @ y
        beta = np.linalg.solve(lhs, rhs)
        return float(beta[0]), beta[1:]

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_features_in_ = X.shape[1]

        self.feature_medians_ = np.nanmedian(X, axis=0)
        X = self._impute(X)

        terms, values, values_centered = self._build_terms(X, y)
        m = len(terms)

        selected = []
        best_sse = float(np.sum((y - float(np.mean(y))) ** 2))
        resid = y - float(np.mean(y))
        max_terms = min(int(self.max_terms), m)

        for _ in range(max_terms):
            best_idx = None
            best_score = -1.0
            for idx in range(m):
                if idx in selected:
                    continue
                v = values_centered[idx]
                score = abs(float(np.dot(v, resid))) / (float(np.linalg.norm(v)) + 1e-12)
                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx is None:
                break

            selected.append(best_idx)
            Z = np.column_stack([values[i] for i in selected])
            intercept, coefs = self._ridge_refit(Z, y)
            pred = intercept + Z @ coefs
            sse = float(np.sum((y - pred) ** 2))

            rel_improve = (best_sse - sse) / (abs(best_sse) + 1e-12)
            resid = y - pred
            best_sse = sse
            if rel_improve < float(self.min_rel_improvement):
                break

        if selected:
            Z = np.column_stack([values[i] for i in selected])
            intercept, coefs = self._ridge_refit(Z, y)
            keep = np.where(np.abs(coefs) >= float(self.coef_tol))[0]
            if keep.size > 0:
                selected = [selected[int(i)] for i in keep]
                coefs = coefs[keep]
                Z = Z[:, keep]
                intercept, coefs = self._ridge_refit(Z, y)
            else:
                selected = []
                coefs = np.zeros(0, dtype=float)
        else:
            intercept = float(np.mean(y))
            coefs = np.zeros(0, dtype=float)

        self.all_terms_ = terms
        self.selected_term_indices_ = selected
        self.selected_terms_ = [terms[i] for i in selected]
        self.coef_ = np.asarray(coefs, dtype=float)
        self.intercept_ = float(intercept)

        if selected:
            self.term_names_ = [self._term_name(m) for m in self.selected_terms_]
            self.term_strength_ = np.abs(self.coef_)
            self.term_order_ = np.argsort(self.term_strength_)[::-1]
            Zs = np.column_stack([self._term_value(X, m) for m in self.selected_terms_])
            self.fitted_mse_ = float(np.mean((y - (self.intercept_ + Zs @ self.coef_)) ** 2))
        else:
            self.term_names_ = []
            self.term_strength_ = np.zeros(0, dtype=float)
            self.term_order_ = np.zeros(0, dtype=int)
            self.fitted_mse_ = float(np.mean((y - self.intercept_) ** 2))
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "selected_terms_", "coef_"])
        X = self._impute(X)
        yhat = np.full(X.shape[0], self.intercept_, dtype=float)
        if len(self.selected_terms_) > 0:
            for c, m in zip(self.coef_, self.selected_terms_):
                yhat += float(c) * self._term_value(X, m)
        return yhat

    def __str__(self):
        check_is_fitted(self, ["intercept_", "selected_terms_", "coef_"])
        lines = [
            "ScreenedOMPFormulaRegressor",
            "Prediction equation:",
        ]

        if len(self.selected_terms_) == 0:
            lines.append(f"  y = {self.intercept_:+.6f}")
            lines.append("")
            lines.append("No non-constant terms selected.")
            lines.append(f"Training MSE: {self.fitted_mse_:.6f}")
            return "\n".join(lines)

        eq_terms = [f"({c:+.6f})*{self._term_name(m)}" for c, m in zip(self.coef_, self.selected_terms_)]
        lines.append(f"  y = {self.intercept_:+.6f} " + " ".join(["+ " + t for t in eq_terms]))
        lines.append("")
        lines.append("Selected terms (ordered by absolute coefficient):")
        for idx in self.term_order_:
            c = float(self.coef_[idx])
            name = self.term_names_[idx]
            lines.append(f"  {name}: coef={c:+.6f}")

        used_features = set()
        for m in self.selected_terms_:
            used_features.add(int(m["j"]))
            if "k" in m:
                used_features.add(int(m["k"]))
        negligible = [f"x{j}" for j in range(self.n_features_in_) if j not in used_features]
        if negligible:
            lines.append("")
            lines.append("Features not used: " + ", ".join(negligible))
        lines.append(f"Training MSE: {self.fitted_mse_:.6f}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
ScreenedOMPFormulaRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "ScreenedOMPFormulaV1"
model_description = "Greedy sparse symbolic equation over linear, abs, square, hinge, and interaction raw-feature terms with ridge refit"
model_defs = [(model_shorthand_name, ScreenedOMPFormulaRegressor())]

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------

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
