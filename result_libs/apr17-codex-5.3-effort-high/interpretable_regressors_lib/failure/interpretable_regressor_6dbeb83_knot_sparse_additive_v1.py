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


class SparseKnotAdditiveRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse additive regressor with feature-wise learned hinge knots.
    Terms are selected by greedy forward search with ridge refits.
    """

    def __init__(
        self,
        max_terms=8,
        screen_features=10,
        knots_per_feature=5,
        max_interaction_features=3,
        ridge_lambda=1e-2,
        min_rel_gain=1e-3,
    ):
        self.max_terms = max_terms
        self.screen_features = screen_features
        self.knots_per_feature = knots_per_feature
        self.max_interaction_features = max_interaction_features
        self.ridge_lambda = ridge_lambda
        self.min_rel_gain = min_rel_gain

    @staticmethod
    def _ridge_fit(B, y, lam):
        n = B.shape[0]
        if B.shape[1] == 0:
            mu = float(np.mean(y))
            return mu, np.zeros(0, dtype=float), np.full(n, mu, dtype=float)
        Xd = np.hstack([np.ones((n, 1), dtype=float), B])
        p = Xd.shape[1]
        I = np.eye(p, dtype=float)
        I[0, 0] = 0.0
        beta = np.linalg.solve(Xd.T @ Xd + max(lam, 1e-12) * I, Xd.T @ y)
        intercept = float(beta[0])
        coef = np.asarray(beta[1:], dtype=float)
        pred = intercept + B @ coef
        return intercept, coef, pred

    @staticmethod
    def _quantile_knots(x, n_knots):
        q = np.linspace(0.15, 0.85, max(2, n_knots))
        knots = np.unique(np.quantile(x, q))
        if knots.size <= 1:
            return np.array([float(np.median(x))], dtype=float)
        return knots.astype(float)

    def _term_col(self, X, term):
        kind = term["kind"]
        if kind == "lin":
            return X[:, term["j"]]
        if kind == "hinge_pos":
            return np.maximum(0.0, X[:, term["j"]] - term["knot"])
        if kind == "hinge_neg":
            return np.maximum(0.0, term["knot"] - X[:, term["j"]])
        if kind == "int":
            return (X[:, term["j"]] - term["mu_j"]) * (X[:, term["k"]] - term["mu_k"])
        raise ValueError(f"Unknown term kind: {kind}")

    @staticmethod
    def _term_expr(term):
        kind = term["kind"]
        if kind == "lin":
            return f"x{term['j']}"
        if kind == "hinge_pos":
            return f"max(0, x{term['j']}-{term['knot']:.4f})"
        if kind == "hinge_neg":
            return f"max(0, {term['knot']:.4f}-x{term['j']})"
        if kind == "int":
            return f"((x{term['j']}-{term['mu_j']:.4f})*(x{term['k']}-{term['mu_k']:.4f}))"
        return str(term)

    def _build_candidates(self, X, y):
        y0 = y - float(np.mean(y))
        corr = np.abs(X.T @ y0)
        keep = min(X.shape[1], max(1, int(self.screen_features)))
        feats = np.argsort(corr)[-keep:]
        feats = np.sort(feats.astype(int))

        candidates = []
        for j in feats:
            xj = X[:, j]
            candidates.append({"kind": "lin", "j": int(j)})
            knots = self._quantile_knots(xj, self.knots_per_feature)
            best_pos = None
            best_pos_score = -np.inf
            best_neg = None
            best_neg_score = -np.inf
            for t in knots:
                hpos = np.maximum(0.0, xj - t)
                hneg = np.maximum(0.0, t - xj)
                s_pos = float(abs(np.dot(hpos, y0)))
                s_neg = float(abs(np.dot(hneg, y0)))
                if s_pos > best_pos_score:
                    best_pos_score = s_pos
                    best_pos = float(t)
                if s_neg > best_neg_score:
                    best_neg_score = s_neg
                    best_neg = float(t)
            candidates.append({"kind": "hinge_pos", "j": int(j), "knot": best_pos})
            candidates.append({"kind": "hinge_neg", "j": int(j), "knot": best_neg})

        top = feats[np.argsort(corr[feats])[::-1][: min(len(feats), self.max_interaction_features)]]
        for a in range(len(top)):
            for b in range(a + 1, len(top)):
                j, k = int(top[a]), int(top[b])
                candidates.append(
                    {"kind": "int", "j": j, "k": k, "mu_j": float(np.mean(X[:, j])), "mu_k": float(np.mean(X[:, k]))}
                )
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

        C_raw = np.column_stack([self._term_col(X, t) for t in candidates])
        C_norm = np.linalg.norm(C_raw, axis=0) + 1e-12
        C = C_raw / C_norm

        selected = []
        current_pred = np.full(n, np.mean(y), dtype=float)
        current_mse = float(np.mean((y - current_pred) ** 2))
        best_intercept = float(np.mean(y))
        best_coef_raw = np.zeros(0, dtype=float)

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
                intercept, coef_scaled, pred = self._ridge_fit(C[:, trial], y, self.ridge_lambda)
                mse = float(np.mean((y - pred) ** 2))
                gain = (current_mse - mse) / max(current_mse, 1e-12)
                if gain >= self.min_rel_gain:
                    selected = trial
                    current_mse = mse
                    current_pred = pred
                    best_intercept = intercept
                    best_coef_raw = coef_scaled / C_norm[np.array(selected, dtype=int)]
                    improved = True
                    break
            if not improved:
                break

        if len(selected) == 0:
            self.selected_terms_ = []
            self.coef_ = np.zeros(0, dtype=float)
            self.intercept_ = float(np.mean(y))
            self.train_mse_ = float(np.mean((y - self.intercept_) ** 2))
            return self

        active = [i for i, c in enumerate(best_coef_raw) if abs(c) > 5e-4]
        if len(active) < len(selected):
            selected = [selected[i] for i in active]
            if len(selected) == 0:
                self.selected_terms_ = []
                self.coef_ = np.zeros(0, dtype=float)
                self.intercept_ = float(np.mean(y))
                self.train_mse_ = float(np.mean((y - self.intercept_) ** 2))
                return self
            B = np.column_stack([self._term_col(X, candidates[i]) for i in selected])
            best_intercept, best_coef_raw, current_pred = self._ridge_fit(B, y, self.ridge_lambda)
            current_mse = float(np.mean((y - current_pred) ** 2))

        self.selected_terms_ = [candidates[i] for i in selected]
        self.coef_ = np.asarray(best_coef_raw, dtype=float)
        self.intercept_ = float(best_intercept)
        self.train_mse_ = float(current_mse)
        return self

    def predict(self, X):
        check_is_fitted(self, ["selected_terms_", "coef_", "intercept_"])
        X = np.asarray(X, dtype=float)
        yhat = np.full(X.shape[0], self.intercept_, dtype=float)
        for c, t in zip(self.coef_, self.selected_terms_):
            yhat += c * self._term_col(X, t)
        return yhat

    def __str__(self):
        check_is_fitted(self, ["selected_terms_", "coef_", "intercept_"])
        lines = ["Sparse Knot Additive Regressor", "Explicit prediction equation on raw features:"]
        eq = [f"{self.intercept_:.6f}"]
        pairs = sorted(zip(self.coef_, self.selected_terms_), key=lambda z: -abs(z[0]))
        for c, t in pairs:
            eq.append(f"{c:+.6f}*{self._term_expr(t)}")
        lines.append("  y = " + " ".join(eq))
        lines.append("")
        lines.append("Selected terms (ordered by |coefficient|):")
        for i, (c, t) in enumerate(pairs, 1):
            lines.append(f"  {i:2d}. {c:+.6f} * {self._term_expr(t)}")

        used = set()
        for t in self.selected_terms_:
            used.add(int(t["j"]))
            if t["kind"] == "int":
                used.add(int(t["k"]))
        inactive = [f"x{i}" for i in range(self.n_features_in_) if i not in used]
        lines.append("")
        lines.append("Features with no direct effect:")
        lines.append("  " + (", ".join(inactive) if inactive else "none"))
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparseKnotAdditiveRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "KnotSparseAdditiveV1"
model_description = "Sparse additive regressor with feature-wise quantile knot hinges and centered pairwise interactions chosen by greedy ridge-refit forward selection"
model_defs = [(model_shorthand_name, SparseKnotAdditiveRegressor())]


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
