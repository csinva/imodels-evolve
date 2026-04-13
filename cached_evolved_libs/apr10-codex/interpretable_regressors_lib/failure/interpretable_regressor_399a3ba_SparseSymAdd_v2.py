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


class SparseSymbolicAdditiveRegressor(BaseEstimator, RegressorMixin):
    """
    Compact symbolic model learned by greedy forward selection from a small basis:
      - linear terms: x_j
      - hinge terms: max(0, x_j - t) for t at selected quantiles
      - quadratic terms: x_j^2
      - pairwise interactions: x_j * x_k (few top candidates only)

    The final model is a tiny additive expression solved by least squares.
    """

    def __init__(
        self,
        max_terms=5,
        top_features=8,
        quantiles=(0.25, 0.5, 0.75),
        min_rel_gain=2e-3,
        max_interactions=6,
    ):
        self.max_terms = max_terms
        self.top_features = top_features
        self.quantiles = quantiles
        self.min_rel_gain = min_rel_gain
        self.max_interactions = max_interactions

    @staticmethod
    def _corr_score(vec, target):
        vc = vec - np.mean(vec)
        tc = target - np.mean(target)
        denom = (np.linalg.norm(vc) + 1e-12) * (np.linalg.norm(tc) + 1e-12)
        return float(abs(np.dot(vc, tc)) / denom)

    @staticmethod
    def _term_values(X, term):
        ttype = term["type"]
        if ttype == "lin":
            return X[:, term["j"]]
        if ttype == "quad":
            xj = X[:, term["j"]]
            return xj * xj
        if ttype == "hinge":
            return np.maximum(0.0, X[:, term["j"]] - term["t"])
        if ttype == "inter":
            return X[:, term["j"]] * X[:, term["k"]]
        raise ValueError(f"unknown term type: {ttype}")

    def _fit_with_terms(self, X, y, terms):
        cols = [np.ones(X.shape[0], dtype=float)]
        for term in terms:
            cols.append(self._term_values(X, term))
        design = np.column_stack(cols)
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        pred = design @ beta
        rss = float(np.sum((y - pred) ** 2))
        return beta, pred, rss

    def _build_candidates(self, X, y):
        n_features = X.shape[1]
        # Rank features by marginal correlation, then create a compact candidate library.
        corr = np.array([self._corr_score(X[:, j], y) for j in range(n_features)])
        top_k = min(int(self.top_features), n_features)
        top_idx = [int(j) for j in np.argsort(-corr)[:top_k]]

        cands = []
        seen = set()

        def _push(term):
            key = tuple(sorted(term.items()))
            if key not in seen:
                seen.add(key)
                cands.append(term)

        for j in top_idx:
            _push({"type": "lin", "j": j})
            _push({"type": "quad", "j": j})
            xj = X[:, j]
            for q in self.quantiles:
                t = float(np.quantile(xj, q))
                _push({"type": "hinge", "j": j, "t": t})

        # Add a small number of interaction terms among top features.
        pairs = []
        for a in range(len(top_idx)):
            for b in range(a + 1, len(top_idx)):
                j, k = top_idx[a], top_idx[b]
                score = corr[j] * corr[k]
                pairs.append((score, j, k))
        pairs.sort(reverse=True)
        for _, j, k in pairs[: int(self.max_interactions)]:
            _push({"type": "inter", "j": int(j), "k": int(k)})

        return cands

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        candidates = self._build_candidates(X, y)
        selected = []
        beta, pred, best_rss = self._fit_with_terms(X, y, selected)

        for _ in range(int(self.max_terms)):
            remaining = [t for t in candidates if t not in selected]
            if not remaining:
                break

            residual = y - pred
            # First prune by correlation to residual, then evaluate top few exactly.
            scored = [
                (self._corr_score(self._term_values(X, t), residual), t)
                for t in remaining
            ]
            scored.sort(reverse=True, key=lambda z: z[0])
            shortlist = [t for _, t in scored[: min(12, len(scored))]]

            best_trial = None
            for t in shortlist:
                trial_terms = selected + [t]
                trial_beta, trial_pred, trial_rss = self._fit_with_terms(X, y, trial_terms)
                gain = (best_rss - trial_rss) / (best_rss + 1e-12)
                if best_trial is None or gain > best_trial[0]:
                    best_trial = (gain, t, trial_beta, trial_pred, trial_rss)

            if best_trial is None or best_trial[0] < self.min_rel_gain:
                break

            _, chosen, beta, pred, best_rss = best_trial
            selected.append(chosen)

        self.terms_ = selected
        self.intercept_ = float(beta[0])
        self.coefs_ = np.asarray(beta[1:], dtype=float)
        self.n_features_in_ = X.shape[1]
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        X = np.asarray(X, dtype=float)
        yhat = np.full(X.shape[0], self.intercept_, dtype=float)
        for c, term in zip(self.coefs_, self.terms_):
            yhat += c * self._term_values(X, term)
        return yhat

    @staticmethod
    def _term_str(term):
        ttype = term["type"]
        if ttype == "lin":
            return f"x{term['j']}"
        if ttype == "quad":
            return f"(x{term['j']}^2)"
        if ttype == "hinge":
            return f"max(0, x{term['j']} - {term['t']:.4f})"
        if ttype == "inter":
            return f"(x{term['j']}*x{term['k']})"
        return "?"

    def __str__(self):
        check_is_fitted(self, "is_fitted_")

        lines = [
            "Sparse Symbolic Additive Regressor:",
            "  y = intercept + sum(coef_i * basis_i)",
            f"  intercept: {self.intercept_:+.6f}",
            f"  active_terms: {len(self.terms_)}",
        ]

        if not self.terms_:
            lines.append("  No active terms (constant model).")
        else:
            lines.append("  Basis terms:")
            for i, (coef, term) in enumerate(zip(self.coefs_, self.terms_), 1):
                lines.append(f"    {i}. {coef:+.6f} * {self._term_str(term)}")

        eq = f"{self.intercept_:+.6f}"
        for coef, term in zip(self.coefs_, self.terms_):
            eq += f" {coef:+.6f}*{self._term_str(term)}"
        lines.append(f"  Equation: y = {eq}")

        # Explicitly list active raw features to help sparse-feature reasoning tests.
        active_feats = set()
        for term in self.terms_:
            if "j" in term:
                active_feats.add(int(term["j"]))
            if "k" in term:
                active_feats.add(int(term["k"]))
        if active_feats:
            lines.append("  Active raw features: " + ", ".join(f"x{i}" for i in sorted(active_feats)))
        else:
            lines.append("  Active raw features: none")

        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparseSymbolicAdditiveRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SparseSymAdd_v2"
model_description = "Greedy sparse symbolic additive model over linear, hinge, quadratic, and limited interaction basis terms"
model_defs = [(model_shorthand_name, SparseSymbolicAdditiveRegressor())]


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
