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


class OrthogonalSymbolicRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse symbolic regressor with explicit raw-feature equation.
    Uses greedy orthogonal forward selection over a compact nonlinear basis.
    """

    def __init__(
        self,
        max_terms=9,
        ridge_lambda=1e-2,
        screen_features=10,
        max_interaction_features=3,
        min_rel_gain=2e-3,
    ):
        self.max_terms = max_terms
        self.ridge_lambda = ridge_lambda
        self.screen_features = screen_features
        self.max_interaction_features = max_interaction_features
        self.min_rel_gain = min_rel_gain

    @staticmethod
    def _ridge_fit(B, y, lam):
        n = B.shape[0]
        if B.shape[1] == 0:
            mu = float(np.mean(y))
            return mu, np.zeros(0, dtype=float), np.full(n, mu, dtype=float)
        Xd = np.hstack([np.ones((n, 1), dtype=float), B])
        p = Xd.shape[1]
        eye = np.eye(p, dtype=float)
        eye[0, 0] = 0.0
        A = Xd.T @ Xd + max(lam, 1e-12) * eye
        b = Xd.T @ y
        beta = np.linalg.solve(A, b)
        intercept = float(beta[0])
        coef = np.asarray(beta[1:], dtype=float)
        pred = intercept + B @ coef
        return intercept, coef, pred

    @staticmethod
    def _term_expr(term):
        kind = term["kind"]
        if kind == "lin":
            return f"x{term['j']}"
        if kind == "hinge_pos":
            return f"max(0, x{term['j']})"
        if kind == "hinge_neg":
            return f"max(0, -x{term['j']})"
        if kind == "quad":
            return f"(x{term['j']}^2)"
        if kind == "int":
            return f"(x{term['j']}*x{term['k']})"
        return str(term)

    def _term_col(self, X, term):
        kind = term["kind"]
        if kind == "lin":
            return X[:, term["j"]]
        if kind == "hinge_pos":
            return np.maximum(0.0, X[:, term["j"]])
        if kind == "hinge_neg":
            return np.maximum(0.0, -X[:, term["j"]])
        if kind == "quad":
            return X[:, term["j"]] ** 2
        if kind == "int":
            return X[:, term["j"]] * X[:, term["k"]]
        raise ValueError(f"Unknown term kind: {kind}")

    def _build_candidates(self, X, y):
        yc = y - float(np.mean(y))
        corr = np.abs(X.T @ yc)
        keep = min(X.shape[1], max(1, int(self.screen_features)))
        feats = np.argsort(corr)[-keep:]
        feats = np.sort(feats.astype(int))

        candidates = []
        for j in feats:
            candidates.append({"kind": "lin", "j": int(j)})
            candidates.append({"kind": "hinge_pos", "j": int(j)})
            candidates.append({"kind": "hinge_neg", "j": int(j)})
            candidates.append({"kind": "quad", "j": int(j)})

        top = feats[np.argsort(corr[feats])[::-1][: min(len(feats), self.max_interaction_features)]]
        for i in range(len(top)):
            for k in range(i + 1, len(top)):
                candidates.append({"kind": "int", "j": int(top[i]), "k": int(top[k])})

        return candidates, feats

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape
        self.n_features_in_ = p

        candidates, feats = self._build_candidates(X, y)
        self.screened_features_ = feats
        if not candidates:
            self.selected_terms_ = []
            self.coef_ = np.zeros(0, dtype=float)
            self.intercept_ = float(np.mean(y))
            self.train_mse_ = float(np.mean((y - self.intercept_) ** 2))
            return self

        C = np.column_stack([self._term_col(X, t) for t in candidates])
        C_norm = np.linalg.norm(C, axis=0) + 1e-12
        C = C / C_norm
        selected = []
        current_pred = np.full(n, np.mean(y), dtype=float)
        current_mse = float(np.mean((y - current_pred) ** 2))
        best_intercept = float(np.mean(y))
        best_coef = np.zeros(0, dtype=float)

        for _ in range(min(self.max_terms, C.shape[1])):
            residual = y - current_pred
            scores = np.abs(C.T @ residual)
            if selected:
                scores[np.array(selected, dtype=int)] = -np.inf
            order = np.argsort(scores)[::-1]

            improved = False
            for idx in order[: min(12, len(order))]:
                if idx in selected:
                    continue
                trial = selected + [int(idx)]
                B = C[:, trial]
                intercept, coef_scaled, pred = self._ridge_fit(B, y, self.ridge_lambda)
                trial_mse = float(np.mean((y - pred) ** 2))
                rel_gain = (current_mse - trial_mse) / max(current_mse, 1e-12)
                if rel_gain >= self.min_rel_gain:
                    improved = True
                    selected = trial
                    current_mse = trial_mse
                    current_pred = pred
                    best_intercept = intercept
                    best_coef = coef_scaled / C_norm[np.array(selected, dtype=int)]
                    break
            if not improved:
                break

        if len(selected) == 0:
            self.selected_terms_ = []
            self.coef_ = np.zeros(0, dtype=float)
            self.intercept_ = float(np.mean(y))
            self.train_mse_ = float(np.mean((y - self.intercept_) ** 2))
            return self

        # Prune tiny terms once for readability and refit.
        active = [i for i, c in enumerate(best_coef) if abs(c) > 1e-4]
        if len(active) < len(selected):
            selected = [selected[i] for i in active]
            if len(selected) == 0:
                self.selected_terms_ = []
                self.coef_ = np.zeros(0, dtype=float)
                self.intercept_ = float(np.mean(y))
                self.train_mse_ = float(np.mean((y - self.intercept_) ** 2))
                return self
            B = np.column_stack([self._term_col(X, candidates[t]) for t in selected])
            best_intercept, best_coef, current_pred = self._ridge_fit(B, y, self.ridge_lambda)
            current_mse = float(np.mean((y - current_pred) ** 2))

        self.selected_terms_ = [candidates[i] for i in selected]
        self.coef_ = np.asarray(best_coef, dtype=float)
        self.intercept_ = float(best_intercept)
        self.train_mse_ = float(current_mse)
        return self

    def predict(self, X):
        check_is_fitted(self, ["selected_terms_", "coef_", "intercept_"])
        X = np.asarray(X, dtype=float)
        yhat = np.full(X.shape[0], self.intercept_, dtype=float)
        for coef, term in zip(self.coef_, self.selected_terms_):
            yhat += coef * self._term_col(X, term)
        return yhat

    def __str__(self):
        check_is_fitted(self, ["selected_terms_", "coef_", "intercept_"])
        lines = ["Orthogonal Symbolic Regressor", "Explicit prediction equation on raw features:"]
        eq = [f"{self.intercept_:.6f}"]
        for c, t in sorted(zip(self.coef_, self.selected_terms_), key=lambda z: -abs(z[0])):
            eq.append(f"{c:+.6f}*{self._term_expr(t)}")
        lines.append("  y = " + " ".join(eq))

        used = set()
        for t in self.selected_terms_:
            used.add(int(t["j"]))
            if t["kind"] == "int":
                used.add(int(t["k"]))
        inactive = [f"x{i}" for i in range(self.n_features_in_) if i not in used]

        lines.append("")
        lines.append("Terms used (largest |coefficient| first):")
        for rank, (c, t) in enumerate(sorted(zip(self.coef_, self.selected_terms_), key=lambda z: -abs(z[0])), 1):
            lines.append(f"  {rank:2d}. {c:+.6f} * {self._term_expr(t)}")
        lines.append("")
        lines.append("Features with no direct effect:")
        lines.append("  " + (", ".join(inactive) if inactive else "none"))
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
OrthogonalSymbolicRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "OrthoSymbolicV2"
model_description = "Orthogonal forward-selected sparse symbolic regressor on raw features with linear, one-sided hinge, quadratic, and limited interaction terms (cache-busted rerun)"
model_defs = [(model_shorthand_name, OrthogonalSymbolicRegressor())]


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
