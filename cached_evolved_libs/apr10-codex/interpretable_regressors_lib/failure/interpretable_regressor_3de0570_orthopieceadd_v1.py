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


class SparseOrthoPiecewiseAdditiveRegressor(BaseEstimator, RegressorMixin):
    """
    Greedy sparse additive model over centered linear and one-knot hinge terms.
    Candidate terms are orthogonalized against selected terms before scoring,
    then all active terms are jointly refit with mild L2 stabilization.
    """

    def __init__(
        self,
        top_features=14,
        max_terms=8,
        quantiles=(0.2, 0.5, 0.8),
        min_gain=1e-4,
        l2=1e-3,
    ):
        self.top_features = top_features
        self.max_terms = max_terms
        self.quantiles = quantiles
        self.min_gain = min_gain
        self.l2 = l2

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

    def _candidate_terms_for_feature(self, z, feat_idx):
        terms = []

        # linear component
        lin = self._center(z)
        if np.dot(lin, lin) > 1e-12:
            terms.append({
                "feature": feat_idx,
                "kind": "lin",
                "knot": 0.0,
                "vec": lin,
            })

        # one-knot hinge components at robust quantiles
        qs = np.unique(np.asarray(self.quantiles, dtype=float))
        knots = np.quantile(z, qs)
        for knot in np.unique(np.asarray(knots, dtype=float)):
            h_pos = self._center(np.maximum(0.0, z - knot))
            if np.dot(h_pos, h_pos) > 1e-12:
                terms.append({
                    "feature": feat_idx,
                    "kind": "hinge_pos",
                    "knot": float(knot),
                    "vec": h_pos,
                })

            h_neg = self._center(np.maximum(0.0, knot - z))
            if np.dot(h_neg, h_neg) > 1e-12:
                terms.append({
                    "feature": feat_idx,
                    "kind": "hinge_neg",
                    "knot": float(knot),
                    "vec": h_neg,
                })

        return terms

    @staticmethod
    def _orthogonalize(v, basis):
        out = np.asarray(v, dtype=float).copy()
        if basis is None or basis.size == 0:
            return out
        for j in range(basis.shape[1]):
            b = basis[:, j]
            denom = float(np.dot(b, b))
            if denom > 1e-12:
                out -= (float(np.dot(out, b)) / denom) * b
        return out

    def _eval_term(self, resid, vec, basis):
        ortho = self._orthogonalize(vec, basis)
        denom = float(np.dot(ortho, ortho))
        if denom < 1e-12:
            return None

        coef = float(np.dot(resid, ortho) / (denom + 1e-12))
        step = coef * ortho
        gain = float(2.0 * np.dot(resid, step) - np.dot(step, step))
        return {
            "ortho": ortho,
            "coef": coef,
            "gain": gain,
        }

    def _term_values(self, Xs, spec):
        z = Xs[:, spec["feature"]]
        if spec["kind"] == "lin":
            return self._center(z)
        if spec["kind"] == "hinge_pos":
            return self._center(np.maximum(0.0, z - spec["knot"]))
        return self._center(np.maximum(0.0, spec["knot"] - z))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape

        self.x_mean_ = np.mean(X, axis=0)
        self.x_scale_ = np.array([self._safe_scale(X[:, j]) for j in range(p)], dtype=float)
        Xs = (X - self.x_mean_) / self.x_scale_

        yc = y - np.mean(y)
        corr = np.array([self._corr(Xs[:, j], yc) for j in range(p)], dtype=float)
        pool_size = min(max(1, int(self.top_features)), p)
        pool = [int(j) for j in np.argsort(-np.abs(corr))[:pool_size]]

        candidates = []
        for j in pool:
            candidates.extend(self._candidate_terms_for_feature(Xs[:, j], j))

        self.intercept_ = float(np.mean(y))
        pred = np.full(n, self.intercept_, dtype=float)
        resid = y - pred

        selected_specs = []
        selected_cols = []

        for _ in range(int(self.max_terms)):
            basis = np.column_stack(selected_cols) if selected_cols else None
            best = None
            for cand in candidates:
                score = self._eval_term(resid, cand["vec"], basis)
                if score is None:
                    continue
                if (best is None) or (score["gain"] > best["gain"]):
                    best = {"cand": cand, **score}

            if best is None or best["gain"] < float(self.min_gain):
                break

            selected_specs.append({
                "feature": best["cand"]["feature"],
                "kind": best["cand"]["kind"],
                "knot": best["cand"]["knot"],
            })
            selected_cols.append(best["cand"]["vec"])
            pred += best["coef"] * best["ortho"]
            resid = y - pred

            # Avoid selecting exact duplicate specification again.
            candidates = [
                c for c in candidates
                if not (
                    c["feature"] == best["cand"]["feature"]
                    and c["kind"] == best["cand"]["kind"]
                    and abs(c["knot"] - best["cand"]["knot"]) < 1e-12
                )
            ]

        if selected_specs:
            design = np.column_stack([self._term_values(Xs, s) for s in selected_specs])
            A = design.T @ design + float(self.l2) * np.eye(design.shape[1])
            b = design.T @ (y - np.mean(y))
            coef = np.linalg.solve(A, b)
            self.intercept_ = float(np.mean(y))
            self.terms_ = [
                {
                    "feature": s["feature"],
                    "kind": s["kind"],
                    "knot": s["knot"],
                    "coef": float(c),
                }
                for s, c in zip(selected_specs, coef)
                if np.isfinite(c) and abs(float(c)) > 1e-8
            ]
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
            spec = {
                "feature": t["feature"],
                "kind": t["kind"],
                "knot": t["knot"],
            }
            yhat += float(t["coef"]) * self._term_values(Xs, spec)
        return yhat

    def __str__(self):
        check_is_fitted(self, "is_fitted_")
        lines = [
            "Sparse Ortho Piecewise Additive Regressor:",
            "  y = intercept + sum_k coef_k * basis_k(x_j)",
            f"  intercept: {self.intercept_:+.6f}",
            f"  num_terms: {len(self.terms_)}",
            "  Terms:",
        ]
        if not self.terms_:
            lines.append("    none")
        else:
            for i, t in enumerate(self.terms_, start=1):
                if t["kind"] == "lin":
                    basis = f"z{t['feature']}"
                elif t["kind"] == "hinge_pos":
                    basis = f"max(0, z{t['feature']} - {t['knot']:+.3f})"
                else:
                    basis = f"max(0, {t['knot']:+.3f} - z{t['feature']})"
                lines.append(f"    {i}. {float(t['coef']):+.6f} * {basis}")

        active = sorted({t["feature"] for t in self.terms_})
        lines.append("  Active features: " + (", ".join(f"x{j}" for j in active) if active else "none"))
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparseOrthoPiecewiseAdditiveRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "OrthoPieceAdd_v1"
model_description = "Greedy sparse additive model over centered linear and one-knot hinge terms, with orthogonalized selection and joint L2 refit"
model_defs = [(model_shorthand_name, SparseOrthoPiecewiseAdditiveRegressor())]


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
