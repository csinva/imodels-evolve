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


class StagewiseSymbolicRegressor(BaseEstimator, RegressorMixin):
    """
    Compact symbolic regressor:
    1) Dense ridge screening on all raw features.
    2) Keep a compact linear backbone (top coefficient-mass features).
    3) Add a few validation-gated nonlinear residual terms (hinge/quadratic/interaction).
    """

    def __init__(
        self,
        alpha_grid=(0.001, 0.01, 0.1, 1.0, 3.0, 10.0, 30.0),
        holdout_frac=0.2,
        max_linear_terms=10,
        linear_mass_keep=0.97,
        max_seed_features=6,
        max_nonlinear_terms=3,
        min_rel_gain=0.003,
        quantiles=(0.33, 0.5, 0.67),
        random_state=0,
    ):
        self.alpha_grid = alpha_grid
        self.holdout_frac = holdout_frac
        self.max_linear_terms = max_linear_terms
        self.linear_mass_keep = linear_mass_keep
        self.max_seed_features = max_seed_features
        self.max_nonlinear_terms = max_nonlinear_terms
        self.min_rel_gain = min_rel_gain
        self.quantiles = quantiles
        self.random_state = random_state

    @staticmethod
    def _mse(y_true, y_pred):
        d = y_true - y_pred
        return float(np.mean(d * d))

    @staticmethod
    def _ridge_fit(D, y, alpha):
        n, p = D.shape
        A = D.T @ D + float(alpha) * np.eye(p, dtype=float)
        A[0, 0] -= float(alpha)  # no regularization on intercept
        b = D.T @ y
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(A) @ b

    def _fit_best_alpha(self, Dtr, ytr, Dval, yval):
        best = None
        for alpha in self.alpha_grid:
            w = self._ridge_fit(Dtr, ytr, alpha=float(alpha))
            mse = self._mse(yval, Dval @ w)
            if best is None or mse < best["mse"]:
                best = {"mse": float(mse), "alpha": float(alpha), "w": np.asarray(w, dtype=float)}
        return best

    @staticmethod
    def _feature_corr_scores(X, y):
        yc = y - float(np.mean(y))
        yn = float(np.sqrt(np.dot(yc, yc))) + 1e-12
        out = []
        for j in range(X.shape[1]):
            xj = X[:, j]
            xjc = xj - float(np.mean(xj))
            xn = float(np.sqrt(np.dot(xjc, xjc))) + 1e-12
            out.append(abs(float(np.dot(xjc, yc) / (xn * yn))))
        return np.asarray(out, dtype=float)

    @staticmethod
    def _eval_term(X, term):
        kind = term["kind"]
        if kind == "hinge_pos":
            xj = X[:, int(term["feature"])]
            return np.maximum(0.0, xj - float(term["knot"]))
        if kind == "hinge_neg":
            xj = X[:, int(term["feature"])]
            return np.maximum(0.0, float(term["knot"]) - xj)
        if kind == "quadratic":
            xj = X[:, int(term["feature"])]
            c = float(term["center"])
            return (xj - c) * (xj - c)
        if kind == "interaction":
            xa = X[:, int(term["feature_a"])]
            xb = X[:, int(term["feature_b"])]
            return xa * xb
        raise ValueError(f"Unknown term kind: {kind}")

    def _design(self, X, linear_features, nonlinear_terms):
        cols = [np.ones(len(X), dtype=float)]
        if len(linear_features) > 0:
            cols.extend([X[:, int(j)] for j in linear_features])
        cols.extend([self._eval_term(X, t) for t in nonlinear_terms])
        return np.column_stack(cols)

    def _build_candidates(self, Xtr, ytr, seed_features):
        cands = []
        seen = set()
        for j in seed_features:
            j = int(j)
            xj = Xtr[:, j]
            for q in self.quantiles:
                knot = float(np.quantile(xj, q))
                kq = round(knot, 6)
                key_pos = ("hinge_pos", j, kq)
                key_neg = ("hinge_neg", j, kq)
                if key_pos not in seen:
                    cands.append({"kind": "hinge_pos", "feature": j, "knot": knot, "name": f"max(0, x{j}-{knot:.4f})"})
                    seen.add(key_pos)
                if key_neg not in seen:
                    cands.append({"kind": "hinge_neg", "feature": j, "knot": knot, "name": f"max(0, {knot:.4f}-x{j})"})
                    seen.add(key_neg)
            c = float(np.mean(xj))
            key_q = ("quadratic", j, round(c, 6))
            if key_q not in seen:
                cands.append({"kind": "quadratic", "feature": j, "center": c, "name": f"(x{j}-{c:.4f})^2"})
                seen.add(key_q)

        for a_i, a in enumerate(seed_features):
            for b in seed_features[a_i + 1:]:
                a = int(a)
                b = int(b)
                key_i = ("interaction", min(a, b), max(a, b))
                if key_i not in seen:
                    cands.append({"kind": "interaction", "feature_a": a, "feature_b": b, "name": f"x{a}*x{b}"})
                    seen.add(key_i)

        # Order by residual alignment to encourage useful first picks.
        base_scores = self._feature_corr_scores(Xtr, ytr)
        def _score(t):
            if t["kind"] in ("hinge_pos", "hinge_neg", "quadratic"):
                return base_scores[int(t["feature"])]
            return 0.5 * (base_scores[int(t["feature_a"])] + base_scores[int(t["feature_b"])])
        cands.sort(key=_score, reverse=True)
        return cands

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

        # Stage 1: dense ridge screening on all features.
        Dtr_dense = np.column_stack([np.ones(len(Xtr), dtype=float), Xtr])
        Dval_dense = np.column_stack([np.ones(len(Xval), dtype=float), Xval])
        dense_best = self._fit_best_alpha(Dtr_dense, ytr, Dval_dense, yval)
        dense_coef = np.asarray(dense_best["w"][1:], dtype=float)

        # Keep compact linear backbone via coefficient mass.
        order = np.argsort(-np.abs(dense_coef))
        abs_coef = np.abs(dense_coef[order])
        mass = np.cumsum(abs_coef) / max(float(np.sum(abs_coef)), 1e-12)
        keep_n = min(int(self.max_linear_terms), p)
        for i in range(len(order)):
            if i + 1 >= keep_n or mass[i] >= float(self.linear_mass_keep):
                keep_n = max(2, i + 1)
                break
        linear_features = np.sort(order[:keep_n].astype(int))

        # Refit compact linear backbone.
        Dtr_lin = self._design(Xtr, linear_features=linear_features, nonlinear_terms=[])
        Dval_lin = self._design(Xval, linear_features=linear_features, nonlinear_terms=[])
        linear_best = self._fit_best_alpha(Dtr_lin, ytr, Dval_lin, yval)
        best_mse = float(linear_best["mse"])
        chosen_alpha = float(linear_best["alpha"])
        selected_terms = []

        # Stage 2: forward-add nonlinear terms on top of compact linear backbone.
        corr_scores = self._feature_corr_scores(Xtr, ytr)
        top_corr = np.argsort(-corr_scores)[: min(int(self.max_seed_features), p)]
        seed_features = np.unique(np.concatenate([linear_features, top_corr])).astype(int).tolist()
        candidates = self._build_candidates(Xtr, ytr, seed_features)
        remaining = list(range(len(candidates)))

        for _ in range(int(self.max_nonlinear_terms)):
            if not remaining:
                break
            local_best = None
            for idx in remaining:
                trial_terms = selected_terms + [candidates[idx]]
                Dtr = self._design(Xtr, linear_features=linear_features, nonlinear_terms=trial_terms)
                Dval = self._design(Xval, linear_features=linear_features, nonlinear_terms=trial_terms)
                fit = self._fit_best_alpha(Dtr, ytr, Dval, yval)
                if local_best is None or fit["mse"] < local_best["fit"]["mse"]:
                    local_best = {"idx": idx, "fit": fit}

            rel_gain = (best_mse - float(local_best["fit"]["mse"])) / max(best_mse, 1e-12)
            if rel_gain >= float(self.min_rel_gain):
                best_mse = float(local_best["fit"]["mse"])
                chosen_alpha = float(local_best["fit"]["alpha"])
                selected_terms.append(candidates[local_best["idx"]])
                remaining.remove(local_best["idx"])
            else:
                break

        # Refit final equation on full data with chosen structure.
        Dfull = self._design(X, linear_features=linear_features, nonlinear_terms=selected_terms)
        w = self._ridge_fit(Dfull, y, alpha=chosen_alpha)
        n_lin = len(linear_features)

        self.intercept_ = float(w[0])
        self.linear_features_ = np.asarray(linear_features, dtype=int)
        self.linear_coef_ = np.asarray(w[1 : 1 + n_lin], dtype=float)
        self.nonlinear_terms_ = selected_terms
        self.nonlinear_coef_ = np.asarray(w[1 + n_lin :], dtype=float)
        self.alpha_ = float(chosen_alpha)
        self.validation_mse_ = float(best_mse)

        active = set(int(j) for j in self.linear_features_)
        for t in self.nonlinear_terms_:
            if t["kind"] in ("hinge_pos", "hinge_neg", "quadratic"):
                active.add(int(t["feature"]))
            else:
                active.add(int(t["feature_a"]))
                active.add(int(t["feature_b"]))
        self.active_features_ = np.asarray(sorted(active), dtype=int)
        self.inactive_features_ = np.asarray([j for j in range(p) if j not in active], dtype=int)
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "linear_features_", "linear_coef_", "nonlinear_terms_", "nonlinear_coef_"])
        X = np.asarray(X, dtype=float)
        out = np.full(X.shape[0], self.intercept_, dtype=float)
        if len(self.linear_features_) > 0:
            out += X[:, self.linear_features_] @ self.linear_coef_
        for c, term in zip(self.nonlinear_coef_, self.nonlinear_terms_):
            out += float(c) * self._eval_term(X, term)
        return out

    def __str__(self):
        check_is_fitted(self, ["intercept_", "linear_features_", "linear_coef_", "nonlinear_terms_", "nonlinear_coef_"])
        lines = [
            "Stagewise Symbolic Regressor",
            f"ridge_alpha={self.alpha_:.6g}",
            "Exact prediction equation in raw features:",
        ]

        eq = f"y = {self.intercept_:+.6f}"
        for j, c in zip(self.linear_features_, self.linear_coef_):
            eq += f" {float(c):+.6f}*x{int(j)}"
        for c, t in zip(self.nonlinear_coef_, self.nonlinear_terms_):
            eq += f" {float(c):+.6f}*{t['name']}"
        lines.append(eq)

        lines.append("Linear backbone terms:")
        ranked_linear = sorted(zip(self.linear_features_, self.linear_coef_), key=lambda z: -abs(float(z[1])))
        for j, c in ranked_linear:
            lines.append(f"  x{int(j)}: {float(c):+.6f}")

        if self.nonlinear_terms_:
            lines.append("Nonlinear residual terms:")
            ranked_nonlin = sorted(zip(self.nonlinear_coef_, self.nonlinear_terms_), key=lambda z: -abs(float(z[0])))
            for c, t in ranked_nonlin:
                lines.append(f"  {t['name']}: {float(c):+.6f}")
        else:
            lines.append("Nonlinear residual terms: none")

        if len(self.inactive_features_) > 0:
            lines.append("Features with negligible total effect include: " + ", ".join(f"x{int(j)}" for j in self.inactive_features_))
        lines.append(f"Validation MSE: {self.validation_mse_:.6f}")
        ops = int(1 + len(self.linear_features_) + 4 * len(self.nonlinear_terms_))
        lines.append(f"Approx arithmetic operations: {ops}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
StagewiseSymbolicRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "StagewiseSymbolicV1"
model_description = "Dense ridge screening compressed into a compact linear backbone, then validation-gated forward selection of a few symbolic nonlinear residual terms (hinge/quadratic/interaction)"
model_defs = [(model_shorthand_name, StagewiseSymbolicRegressor())]

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
