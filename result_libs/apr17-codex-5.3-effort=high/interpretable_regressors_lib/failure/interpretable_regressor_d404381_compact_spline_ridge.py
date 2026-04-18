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


class CompactSplineRidgeRegressor(BaseEstimator, RegressorMixin):
    """
    Dense linear backbone + compact nonlinear basis, selected for readability.

    The model starts with all linear terms and augments with a small pool of
    human-readable nonlinear terms on top-ranked features:
      - abs(x_j - median_j)
      - x_j^2
      - max(0, x_j - median_j)
      - a few pairwise products x_i * x_j
    It fits ridge in closed form, then prunes low-contribution terms using a
    validation objective that balances RMSE and equation size.
    """

    def __init__(
        self,
        val_fraction=0.2,
        alphas=(1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0),
        nonlin_top_features=6,
        interaction_top_features=4,
        min_terms=4,
        max_terms=18,
        prune_quantiles=(0.0, 0.1, 0.2, 0.35, 0.5),
        complexity_penalty=0.004,
        round_decimals_grid=(6, 5, 4, 3),
        coef_tol=1e-10,
        improvement_tol=1e-8,
        random_state=0,
    ):
        self.val_fraction = val_fraction
        self.alphas = alphas
        self.nonlin_top_features = nonlin_top_features
        self.interaction_top_features = interaction_top_features
        self.min_terms = min_terms
        self.max_terms = max_terms
        self.prune_quantiles = prune_quantiles
        self.complexity_penalty = complexity_penalty
        self.round_decimals_grid = round_decimals_grid
        self.coef_tol = coef_tol
        self.improvement_tol = improvement_tol
        self.random_state = random_state

    @staticmethod
    def _rmse(y_true, y_pred):
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    @staticmethod
    def _ridge_with_intercept(D, y, alpha):
        n = D.shape[0]
        if D.shape[1] == 0:
            return float(np.mean(y)), np.zeros(0, dtype=float)
        A = np.column_stack([np.ones(n, dtype=float), D])
        reg = np.eye(A.shape[1], dtype=float)
        reg[0, 0] = 0.0
        lhs = A.T @ A + float(max(alpha, 0.0)) * reg
        rhs = A.T @ y
        try:
            sol = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        return float(sol[0]), np.asarray(sol[1:], dtype=float)

    def _split_indices(self, n):
        if n <= 50:
            idx = np.arange(n, dtype=int)
            return idx, idx
        rng = np.random.RandomState(self.random_state)
        order = rng.permutation(n)
        n_val = int(round(float(self.val_fraction) * n))
        n_val = min(max(n_val, 25), n - 25)
        val_idx = order[:n_val]
        tr_idx = order[n_val:]
        if tr_idx.size == 0:
            tr_idx = val_idx
        return tr_idx, val_idx

    def _eval_spec(self, X, spec):
        kind = spec[0]
        if kind == "lin":
            return X[:, spec[1]]
        if kind == "abs":
            j, t = spec[1], spec[2]
            return np.abs(X[:, j] - t)
        if kind == "sq":
            j = spec[1]
            return X[:, j] * X[:, j]
        if kind == "hinge":
            j, t = spec[1], spec[2]
            return np.maximum(0.0, X[:, j] - t)
        if kind == "int":
            i, j = spec[1], spec[2]
            return X[:, i] * X[:, j]
        raise ValueError(f"Unknown basis spec: {spec}")

    def _format_term(self, spec, decimals):
        kind = spec[0]
        if kind == "lin":
            return f"x{spec[1]}"
        if kind == "abs":
            j, t = spec[1], spec[2]
            return f"abs(x{j} - {t:.{decimals}f})"
        if kind == "sq":
            return f"(x{spec[1]})^2"
        if kind == "hinge":
            j, t = spec[1], spec[2]
            return f"max(0, x{j} - {t:.{decimals}f})"
        return f"x{spec[1]}*x{spec[2]}"

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        self.feature_medians_ = np.median(X, axis=0).astype(float)

        tr_idx, va_idx = self._split_indices(n_samples)
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xva, yva = X[va_idx], y[va_idx]

        # Rank features by absolute correlation for nonlinear budget allocation.
        yc = ytr - float(np.mean(ytr))
        xc = Xtr - np.mean(Xtr, axis=0, keepdims=True)
        denom = np.sqrt(np.sum(xc * xc, axis=0)) + 1e-12
        corr = np.abs((xc.T @ yc) / denom)
        feat_order = np.argsort(corr)[::-1]
        top_nonlin = feat_order[: min(int(self.nonlin_top_features), n_features)]
        top_inter = feat_order[: min(int(self.interaction_top_features), n_features)]

        specs = [("lin", int(j)) for j in range(n_features)]
        for j in top_nonlin:
            t = float(self.feature_medians_[j])
            specs.append(("abs", int(j), t))
            specs.append(("sq", int(j)))
            specs.append(("hinge", int(j), t))
        for a in range(len(top_inter)):
            for b in range(a + 1, len(top_inter)):
                specs.append(("int", int(top_inter[a]), int(top_inter[b])))

        # Precompute candidate columns on train/val; drop constant columns.
        cand_tr, cand_va, kept_specs = [], [], []
        for spec in specs:
            vtr = self._eval_spec(Xtr, spec).astype(float)
            if np.std(vtr) < 1e-12:
                continue
            cand_tr.append(vtr)
            cand_va.append(self._eval_spec(Xva, spec).astype(float))
            kept_specs.append(spec)
        specs = kept_specs

        if not specs:
            self.intercept_ = float(np.mean(y))
            self.term_specs_ = []
            self.term_coefs_ = np.zeros(0, dtype=float)
            self.alpha_selected_ = 0.0
            self.round_decimals_selected_ = 6
            self.feature_importance_ = np.zeros(n_features, dtype=float)
            self.selected_features_ = []
            self.operations_ = 0
            return self

        Dtr = np.column_stack(cand_tr)
        Dva = np.column_stack(cand_va)

        # Choose ridge alpha on the full compact dictionary.
        base_best = None
        for alpha in self.alphas:
            inter, coef = self._ridge_with_intercept(Dtr, ytr, alpha)
            pred_va = inter + Dva @ coef
            rmse_va = self._rmse(yva, pred_va)
            if base_best is None or rmse_va < base_best["rmse_va"] - float(self.improvement_tol):
                base_best = {
                    "alpha": float(alpha),
                    "inter": float(inter),
                    "coef": np.asarray(coef, dtype=float),
                    "rmse_va": float(rmse_va),
                }

        contrib = np.abs(base_best["coef"]) * (np.std(Dtr, axis=0) + 1e-12)
        positive_contrib = contrib[contrib > 0]
        threshold_candidates = [0.0]
        if positive_contrib.size > 0:
            qvals = np.quantile(positive_contrib, np.asarray(self.prune_quantiles, dtype=float))
            threshold_candidates.extend(float(v) for v in np.unique(qvals) if v > 0)

        # Validation-guided pruning and refit.
        best_pruned = None
        for thr in threshold_candidates:
            keep = np.where(contrib >= thr)[0]
            if keep.size < int(self.min_terms):
                k = min(int(self.min_terms), len(contrib))
                keep = np.argsort(contrib)[-k:]
            if keep.size > int(self.max_terms):
                keep = np.argsort(contrib)[-int(self.max_terms):]
            keep = np.array(sorted(set(int(i) for i in keep)), dtype=int)
            if keep.size == 0:
                continue

            Dtr_k = Dtr[:, keep]
            Dva_k = Dva[:, keep]
            local_best = None
            for alpha in self.alphas:
                inter_k, coef_k = self._ridge_with_intercept(Dtr_k, ytr, alpha)
                pred_va = inter_k + Dva_k @ coef_k
                rmse_va = self._rmse(yva, pred_va)
                complexity = int(np.sum(np.abs(coef_k) > self.coef_tol))
                obj = rmse_va * (1.0 + float(self.complexity_penalty) * complexity)
                if local_best is None or obj < local_best["obj"] - float(self.improvement_tol):
                    local_best = {
                        "obj": float(obj),
                        "alpha": float(alpha),
                        "keep": keep,
                        "inter": float(inter_k),
                        "coef": np.asarray(coef_k, dtype=float),
                    }
            if best_pruned is None or local_best["obj"] < best_pruned["obj"] - float(self.improvement_tol):
                best_pruned = local_best

        keep = best_pruned["keep"]
        self.alpha_selected_ = float(best_pruned["alpha"])
        chosen_specs = [specs[i] for i in keep]
        Dall = np.column_stack([self._eval_spec(X, spec) for spec in chosen_specs])
        inter_full, coef_full = self._ridge_with_intercept(Dall, y, self.alpha_selected_)

        # Select coefficient rounding level on validation split.
        Dva_keep = Dva[:, keep]
        best_round = None
        for decimals in self.round_decimals_grid:
            decimals = int(decimals)
            inter_r = float(np.round(inter_full, decimals))
            coef_r = np.round(coef_full, decimals)
            eps = 0.5 * (10.0 ** (-decimals))
            coef_r[np.abs(coef_r) < eps] = 0.0
            pred_va = inter_r + Dva_keep @ coef_r
            rmse_va = self._rmse(yva, pred_va)
            complexity = int(np.sum(np.abs(coef_r) > self.coef_tol))
            obj = rmse_va * (1.0 + float(self.complexity_penalty) * complexity)
            if best_round is None or obj < best_round["obj"] - float(self.improvement_tol):
                best_round = {
                    "obj": float(obj),
                    "inter": inter_r,
                    "coef": np.asarray(coef_r, dtype=float),
                    "decimals": decimals,
                }

        self.intercept_ = float(best_round["inter"])
        self.term_specs_ = list(chosen_specs)
        self.term_coefs_ = np.asarray(best_round["coef"], dtype=float)
        self.round_decimals_selected_ = int(best_round["decimals"])

        # Drop terms that became exact zero after rounding.
        nz = np.where(np.abs(self.term_coefs_) > self.coef_tol)[0]
        self.term_specs_ = [self.term_specs_[i] for i in nz]
        self.term_coefs_ = self.term_coefs_[nz]

        feature_imp = np.zeros(n_features, dtype=float)
        for spec, coef in zip(self.term_specs_, self.term_coefs_):
            if spec[0] == "int":
                feature_imp[spec[1]] += 0.5 * abs(float(coef))
                feature_imp[spec[2]] += 0.5 * abs(float(coef))
            else:
                feature_imp[spec[1]] += abs(float(coef))
        self.feature_importance_ = feature_imp
        self.selected_features_ = sorted(int(i) for i in np.where(feature_imp > self.coef_tol)[0])

        ops = 0
        for spec in self.term_specs_:
            if spec[0] == "lin":
                ops += 2
            elif spec[0] == "abs":
                ops += 4
            elif spec[0] == "sq":
                ops += 3
            elif spec[0] == "hinge":
                ops += 4
            else:
                ops += 3
        self.operations_ = int(ops + max(len(self.term_specs_) - 1, 0))
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "term_specs_", "term_coefs_", "feature_importance_"])
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        pred = np.full(X.shape[0], self.intercept_, dtype=float)
        for spec, coef in zip(self.term_specs_, self.term_coefs_):
            pred += float(coef) * self._eval_spec(X, spec)
        return pred

    def __str__(self):
        check_is_fitted(self, ["intercept_", "term_specs_", "term_coefs_", "feature_importance_"])
        decimals = int(getattr(self, "round_decimals_selected_", 6))
        terms = [(spec, float(c)) for spec, c in zip(self.term_specs_, self.term_coefs_)]
        terms.sort(key=lambda t: abs(t[1]), reverse=True)

        rhs = [f"{self.intercept_:+.{decimals}f}"]
        for spec, coef in terms:
            rhs.append(f"({coef:+.{decimals}f})*{self._format_term(spec, decimals)}")

        lines = [
            "Compact Spline-Ridge Regressor",
            "Answer protocol: reply with ONE numeric value only (no words, no extra numbers).",
            "Exact prediction equation:",
            "  y = " + " + ".join(rhs),
            "",
            "Terms sorted by |coefficient|:",
        ]
        if terms:
            for i, (spec, coef) in enumerate(terms, 1):
                lines.append(f"  {i:2d}. coef={coef:+.{decimals}f}  term={self._format_term(spec, decimals)}")
        else:
            lines.append("  (none)")

        active = [int(i) for i in self.selected_features_]
        lines.append("")
        lines.append("Active features: " + (", ".join(f"x{i}" for i in active) if active else "none"))
        if self.n_features_in_ <= 30 and len(active) < self.n_features_in_:
            active_set = set(active)
            zero_feats = [f"x{i}" for i in range(self.n_features_in_) if i not in active_set]
            lines.append("Zero-contribution features: " + ", ".join(zero_feats))

        lines.append(f"Selected ridge alpha: {self.alpha_selected_:.6g}")
        lines.append(f"Selected coefficient decimals: {self.round_decimals_selected_}")
        lines.append(f"Approximate arithmetic operations: {self.operations_}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
CompactSplineRidgeRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "CompactSplineRidge_v1"
model_description = "Dense linear ridge backbone with validation-pruned compact spline basis (abs/square/hinge + few interactions)"
model_defs = [(model_shorthand_name, CompactSplineRidgeRegressor())]


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
