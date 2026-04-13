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


class SparseHierarchicalInteractionRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse symbolic regressor:
    y = intercept + sum(main linear terms) + sum(limited pairwise interactions).

    Terms are chosen greedily by residual fit with a complexity penalty favoring
    main effects over interactions, then jointly refit with ridge stabilization.
    """

    def __init__(
        self,
        top_features=12,
        interaction_pool=6,
        max_terms=8,
        l2=2e-3,
        complexity_penalty=1e-3,
        min_gain=1e-4,
    ):
        self.top_features = top_features
        self.interaction_pool = interaction_pool
        self.max_terms = max_terms
        self.l2 = l2
        self.complexity_penalty = complexity_penalty
        self.min_gain = min_gain

    @staticmethod
    def _safe_scale(col):
        s = np.std(col)
        if (not np.isfinite(s)) or s < 1e-12:
            return 1.0
        return float(s)

    @staticmethod
    def _center(v):
        return np.asarray(v, dtype=float) - float(np.mean(v))

    @staticmethod
    def _corr(x, y):
        xc = x - np.mean(x)
        yc = y - np.mean(y)
        denom = (np.linalg.norm(xc) + 1e-12) * (np.linalg.norm(yc) + 1e-12)
        return float(np.dot(xc, yc) / denom)

    @staticmethod
    def _term_complexity(term):
        return 2 if term["kind"] == "inter" else 1

    def _build_candidates(self, Xs, y):
        n, p = Xs.shape
        yc = y - np.mean(y)
        corr = np.array([self._corr(Xs[:, j], yc) for j in range(p)], dtype=float)
        pool_size = min(max(1, int(self.top_features)), p)
        main_pool = [int(j) for j in np.argsort(-np.abs(corr))[:pool_size]]

        candidates = []
        for j in main_pool:
            vec = self._center(Xs[:, j])
            if np.dot(vec, vec) > 1e-12:
                candidates.append({"kind": "main", "j": j, "k": None, "vec": vec})

        inter_size = min(max(2, int(self.interaction_pool)), len(main_pool))
        inter_pool = main_pool[:inter_size]
        for a in range(len(inter_pool)):
            j = inter_pool[a]
            for b in range(a + 1, len(inter_pool)):
                k = inter_pool[b]
                vec = self._center(Xs[:, j] * Xs[:, k])
                if np.dot(vec, vec) > 1e-12:
                    candidates.append({"kind": "inter", "j": j, "k": k, "vec": vec})

        return candidates

    @staticmethod
    def _term_key(term):
        if term["kind"] == "main":
            return ("main", int(term["j"]))
        j, k = int(term["j"]), int(term["k"])
        if j > k:
            j, k = k, j
        return ("inter", j, k)

    def _term_vector(self, Xs, spec):
        if spec["kind"] == "main":
            return self._center(Xs[:, spec["j"]])
        return self._center(Xs[:, spec["j"]] * Xs[:, spec["k"]])

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape

        self.x_mean_ = np.mean(X, axis=0)
        self.x_scale_ = np.array([self._safe_scale(X[:, j]) for j in range(p)], dtype=float)
        Xs = (X - self.x_mean_) / self.x_scale_

        self.intercept_ = float(np.mean(y))
        yc = y - self.intercept_

        candidates = self._build_candidates(Xs, y)

        selected_specs = []
        selected_keys = set()
        pred = np.zeros(n, dtype=float)
        resid = yc.copy()

        y_var = float(np.var(yc) + 1e-12)

        for _ in range(int(self.max_terms)):
            best = None
            for cand in candidates:
                key = self._term_key(cand)
                if key in selected_keys:
                    continue

                vec = cand["vec"]
                denom = float(np.dot(vec, vec) + self.l2)
                if denom < 1e-12:
                    continue

                coef = float(np.dot(resid, vec) / denom)
                gain = float((coef * coef) * np.dot(vec, vec))
                penalized_gain = gain - float(self.complexity_penalty) * self._term_complexity(cand) * y_var

                # weak heredity for interactions: require at least one parent main term selected
                if cand["kind"] == "inter":
                    parent_a = ("main", int(cand["j"]))
                    parent_b = ("main", int(cand["k"]))
                    if parent_a not in selected_keys and parent_b not in selected_keys:
                        penalized_gain *= 0.7

                if (best is None) or (penalized_gain > best["penalized_gain"]):
                    best = {
                        "cand": cand,
                        "coef": coef,
                        "gain": gain,
                        "penalized_gain": penalized_gain,
                    }

            if best is None:
                break
            if best["gain"] < float(self.min_gain) or best["penalized_gain"] <= 0.0:
                break

            cand = best["cand"]
            key = self._term_key(cand)
            selected_keys.add(key)
            selected_specs.append({"kind": cand["kind"], "j": int(cand["j"]), "k": cand["k"]})
            pred += best["coef"] * cand["vec"]
            resid = yc - pred

        if selected_specs:
            design = np.column_stack([self._term_vector(Xs, s) for s in selected_specs])
            A = design.T @ design + float(self.l2) * np.eye(design.shape[1])
            b = design.T @ yc
            coef = np.linalg.solve(A, b)

            max_abs = float(np.max(np.abs(coef))) if coef.size else 0.0
            prune_tol = max(1e-8, 0.01 * max_abs)

            self.terms_ = []
            for s, c in zip(selected_specs, coef):
                c = float(c)
                if np.isfinite(c) and abs(c) >= prune_tol:
                    self.terms_.append({
                        "kind": s["kind"],
                        "j": int(s["j"]),
                        "k": None if s["k"] is None else int(s["k"]),
                        "coef": c,
                    })
        else:
            self.terms_ = []

        self.n_features_in_ = p
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        X = np.asarray(X, dtype=float)
        Xs = (X - self.x_mean_) / self.x_scale_

        yhat = np.full(X.shape[0], self.intercept_, dtype=float)
        for t in self.terms_:
            if t["kind"] == "main":
                yhat += float(t["coef"]) * self._center(Xs[:, t["j"]])
            else:
                yhat += float(t["coef"]) * self._center(Xs[:, t["j"]] * Xs[:, t["k"]])
        return yhat

    def __str__(self):
        check_is_fitted(self, "is_fitted_")
        lines = [
            "Sparse Hierarchical Interaction Regressor:",
            "  y = intercept + sum(main terms) + sum(pairwise interactions)",
            f"  intercept: {self.intercept_:+.6f}",
            f"  num_terms: {len(self.terms_)}",
            "  Terms:",
        ]

        if not self.terms_:
            lines.append("    none")
        else:
            sorted_terms = sorted(self.terms_, key=lambda t: -abs(float(t["coef"])))
            for i, t in enumerate(sorted_terms, start=1):
                c = float(t["coef"])
                if t["kind"] == "main":
                    expr = f"z{t['j']}"
                else:
                    expr = f"z{t['j']} * z{t['k']}"
                direction = "increases" if c > 0 else "decreases"
                lines.append(f"    {i}. {c:+.6f} * {expr}  ({direction} prediction)")

        active = set()
        for t in self.terms_:
            active.add(int(t["j"]))
            if t["k"] is not None:
                active.add(int(t["k"]))
        lines.append("  Active features: " + (", ".join(f"x{j}" for j in sorted(active)) if active else "none"))
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparseHierarchicalInteractionRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SparseHierInter_v1"
model_description = "Sparse symbolic linear+pairwise regressor with complexity-penalized greedy selection and hierarchical interaction preference"
model_defs = [(model_shorthand_name, SparseHierarchicalInteractionRegressor())]


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

    std_tests = {t.__name__ for t in ALL_TESTS}
    hard_tests = {t.__name__ for t in HARD_TESTS}
    insight_tests = {t.__name__ for t in INSIGHT_TESTS}
    std_passed = sum(r["passed"] for r in interp_results if r["test"] in std_tests)
    hard_passed = sum(r["passed"] for r in interp_results if r["test"] in hard_tests)
    insight_passed = sum(r["passed"] for r in interp_results if r["test"] in insight_tests)
    print(f"[std {std_passed}/{len(std_tests)}  hard {hard_passed}/{len(hard_tests)}  insight {insight_passed}/{len(insight_tests)}]")
    print(f"total_seconds: {time.time() - t0:.1f}s")
