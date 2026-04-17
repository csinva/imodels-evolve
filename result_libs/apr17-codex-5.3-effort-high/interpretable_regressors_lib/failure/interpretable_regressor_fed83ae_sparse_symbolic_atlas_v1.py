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


class SparseSymbolicAtlasRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse forward-selected symbolic additive regressor with ridge refits.

    Candidate terms are generated from a compact basis library per feature:
    linear, abs, quadratic, sinusoid, and one-knot hinges at feature quantiles.
    Terms are added greedily only when they improve validation MSE.
    """

    def __init__(
        self,
        alpha_grid=(0.001, 0.01, 0.1, 1.0, 3.0, 10.0),
        holdout_frac=0.2,
        max_screen_features=10,
        max_terms=10,
        min_rel_improve=0.003,
        quantiles=(0.25, 0.5, 0.75),
        random_state=0,
    ):
        self.alpha_grid = alpha_grid
        self.holdout_frac = holdout_frac
        self.max_screen_features = max_screen_features
        self.max_terms = max_terms
        self.min_rel_improve = min_rel_improve
        self.quantiles = quantiles
        self.random_state = random_state

    @staticmethod
    def _mse(y_true, y_pred):
        err = y_true - y_pred
        return float(np.mean(err * err))

    @staticmethod
    def _fit_ridge(X, y, alpha):
        n, p = X.shape
        D = np.hstack([np.ones((n, 1), dtype=float), X])
        reg = np.zeros(p + 1, dtype=float)
        reg[1:] = float(alpha)
        A = D.T @ D + np.diag(reg)
        b = D.T @ y
        try:
            beta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(A) @ b
        return float(beta[0]), np.asarray(beta[1:], dtype=float)

    def _fit_best_ridge(self, Xtr, ytr, Xval, yval):
        best = None
        for alpha in self.alpha_grid:
            intercept, coef = self._fit_ridge(Xtr, ytr, alpha=float(alpha))
            pred_val = intercept + Xval @ coef
            mse_val = self._mse(yval, pred_val)
            if best is None or mse_val < best["mse"]:
                best = {
                    "mse": float(mse_val),
                    "alpha": float(alpha),
                    "intercept": float(intercept),
                    "coef": coef,
                }
        return best

    def _build_candidates(self, X, feature_ids):
        cands = []
        for j in feature_ids:
            xj = X[:, j]
            mean_j = float(np.mean(xj))
            std_j = float(np.std(xj))
            std_j = max(std_j, 1e-8)

            cands.append({
                "name": f"x{j}",
                "feature": int(j),
                "kind": "linear",
                "params": {},
                "values": xj,
            })
            cands.append({
                "name": f"|x{j}|",
                "feature": int(j),
                "kind": "abs",
                "params": {},
                "values": np.abs(xj),
            })
            cands.append({
                "name": f"(x{j}-{mean_j:.3f})^2",
                "feature": int(j),
                "kind": "quad_center",
                "params": {"mu": mean_j},
                "values": (xj - mean_j) ** 2,
            })
            cands.append({
                "name": f"sin(x{j})",
                "feature": int(j),
                "kind": "sin",
                "params": {},
                "values": np.sin(xj),
            })

            for q in self.quantiles:
                knot = float(np.quantile(xj, q))
                cands.append({
                    "name": f"max(0, x{j}-{knot:.3f})",
                    "feature": int(j),
                    "kind": "hinge_pos",
                    "params": {"knot": knot},
                    "values": np.maximum(0.0, xj - knot),
                })
                cands.append({
                    "name": f"max(0, {knot:.3f}-x{j})",
                    "feature": int(j),
                    "kind": "hinge_neg",
                    "params": {"knot": knot},
                    "values": np.maximum(0.0, knot - xj),
                })
        return cands

    def _eval_term(self, X, term):
        j = term["feature"]
        xj = X[:, j]
        kind = term["kind"]
        if kind == "linear":
            return xj
        if kind == "abs":
            return np.abs(xj)
        if kind == "quad_center":
            return (xj - float(term["params"]["mu"])) ** 2
        if kind == "sin":
            return np.sin(xj)
        if kind == "hinge_pos":
            return np.maximum(0.0, xj - float(term["params"]["knot"]))
        if kind == "hinge_neg":
            return np.maximum(0.0, float(term["params"]["knot"]) - xj)
        raise ValueError(f"Unknown term kind: {kind}")

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape
        self.n_features_in_ = p

        rng = np.random.RandomState(int(self.random_state))
        n_val = max(20, int(float(self.holdout_frac) * n))
        if n - n_val < 20:
            n_val = max(1, n // 5)
        perm = rng.permutation(n)
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]
        if len(tr_idx) < 10:
            tr_idx = perm
            val_idx = perm[: max(1, min(10, n // 4))]

        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xval, yval = X[val_idx], y[val_idx]

        yc = ytr - float(np.mean(ytr))
        scores = []
        for j in range(p):
            xj = Xtr[:, j]
            xj_c = xj - float(np.mean(xj))
            denom = float(np.sqrt(np.dot(xj_c, xj_c) * np.dot(yc, yc))) + 1e-12
            corr = float(np.dot(xj_c, yc) / denom)
            scores.append(abs(corr))
        order = np.argsort(-np.asarray(scores))
        k = min(int(self.max_screen_features), p)
        screen = np.sort(order[:k])

        candidates = self._build_candidates(Xtr, screen)

        selected_terms = []
        selected_cols_tr = []
        selected_cols_val = []

        base_pred = np.full(len(yval), float(np.mean(ytr)))
        best_val_mse = self._mse(yval, base_pred)
        chosen_alpha = float(self.alpha_grid[0])
        chosen_intercept = float(np.mean(ytr))
        chosen_coef = np.zeros(0, dtype=float)

        remaining = list(range(len(candidates)))
        for _ in range(int(self.max_terms)):
            if not remaining:
                break

            resid = ytr - (chosen_intercept + (np.column_stack(selected_cols_tr) @ chosen_coef if selected_cols_tr else 0.0))
            corr_scores = []
            for idx in remaining:
                col = candidates[idx]["values"]
                col_c = col - float(np.mean(col))
                denom = float(np.sqrt(np.dot(col_c, col_c) * np.dot(resid, resid))) + 1e-12
                corr_scores.append(abs(float(np.dot(col_c, resid) / denom)))
            best_idx = remaining[int(np.argmax(corr_scores))]

            trial_cols_tr = selected_cols_tr + [candidates[best_idx]["values"]]
            trial_cols_val = selected_cols_val + [self._eval_term(Xval, candidates[best_idx])]
            Xtr_terms = np.column_stack(trial_cols_tr)
            Xval_terms = np.column_stack(trial_cols_val)
            ridge_fit = self._fit_best_ridge(Xtr_terms, ytr, Xval_terms, yval)

            rel_gain = (best_val_mse - ridge_fit["mse"]) / max(best_val_mse, 1e-12)
            if rel_gain >= float(self.min_rel_improve):
                selected_terms.append(candidates[best_idx])
                selected_cols_tr = trial_cols_tr
                selected_cols_val = trial_cols_val
                best_val_mse = float(ridge_fit["mse"])
                chosen_alpha = float(ridge_fit["alpha"])
                chosen_intercept = float(ridge_fit["intercept"])
                chosen_coef = np.asarray(ridge_fit["coef"], dtype=float)
            remaining.remove(best_idx)

        if not selected_terms:
            j0 = int(order[0]) if len(order) else 0
            selected_terms = [{
                "name": f"x{j0}",
                "feature": int(j0),
                "kind": "linear",
                "params": {},
            }]
            Xtr_terms = Xtr[:, [j0]]
            Xval_terms = Xval[:, [j0]]
            ridge_fit = self._fit_best_ridge(Xtr_terms, ytr, Xval_terms, yval)
            chosen_alpha = float(ridge_fit["alpha"])
            chosen_intercept = float(ridge_fit["intercept"])
            chosen_coef = np.asarray(ridge_fit["coef"], dtype=float)

        self.terms_ = selected_terms
        self.term_coefs_ = chosen_coef
        self.intercept_ = float(chosen_intercept)
        self.alpha_ = float(chosen_alpha)
        self.validation_mse_ = float(best_val_mse)
        self.screened_features_ = np.asarray(screen, dtype=int)
        self.active_features_ = np.asarray(sorted({t["feature"] for t in selected_terms}), dtype=int)
        self.inactive_features_ = np.asarray(
            [j for j in range(p) if j not in set(self.active_features_)], dtype=int
        )
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "terms_", "term_coefs_"])
        X = np.asarray(X, dtype=float)
        out = np.full(X.shape[0], float(self.intercept_), dtype=float)
        for c, term in zip(self.term_coefs_, self.terms_):
            out += float(c) * self._eval_term(X, term)
        return out

    def __str__(self):
        check_is_fitted(self, ["intercept_", "terms_", "term_coefs_"])
        lines = [
            "Sparse Symbolic Atlas Regressor",
            f"Ridge alpha: {self.alpha_:.6g}",
            "Prediction equation:",
        ]

        eq = f"y = {self.intercept_:.6f}"
        for c, term in zip(self.term_coefs_, self.terms_):
            eq += f" {float(c):+.6f}*{term['name']}"
        lines.append(eq)

        lines.append("Active symbolic terms:")
        for c, term in sorted(
            zip(self.term_coefs_, self.terms_),
            key=lambda z: -abs(float(z[0])),
        ):
            lines.append(f"  {term['name']}: coef={float(c):+.6f}")

        if len(self.inactive_features_) > 0:
            inactive_txt = ", ".join(f"x{int(j)}" for j in self.inactive_features_)
            lines.append(f"Inactive features: {inactive_txt}")
        lines.append(f"Validation MSE: {self.validation_mse_:.6f}")
        lines.append(f"Approx arithmetic ops to evaluate: {3 * len(self.terms_)}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
SparseSymbolicAtlasRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "SparseSymbolicAtlasV1"
model_description = "Forward-selected sparse symbolic additive regressor over linear/hinge/quadratic/abs/sine basis terms with validation-gated term inclusion and ridge refit"
model_defs = [(model_shorthand_name, SparseSymbolicAtlasRegressor())]

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
