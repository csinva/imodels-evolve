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


class ResidualizedHingeRidgeRegressor(BaseEstimator, RegressorMixin):
    """
    Dense ridge backbone plus a tiny set of validation-gated residual hinge terms.

    This keeps a closed-form explicit equation while allowing lightweight
    piecewise-linear corrections for threshold-like structure.
    """

    def __init__(
        self,
        alpha_grid=(0.001, 0.01, 0.1, 1.0, 3.0, 10.0, 30.0),
        rule_alpha_multipliers=(0.25, 1.0, 4.0),
        holdout_frac=0.2,
        max_hinge_features=8,
        max_hinges=2,
        min_rel_gain=0.002,
        quantiles=(0.2, 0.4, 0.6, 0.8),
        random_state=0,
    ):
        self.alpha_grid = alpha_grid
        self.rule_alpha_multipliers = rule_alpha_multipliers
        self.holdout_frac = holdout_frac
        self.max_hinge_features = max_hinge_features
        self.max_hinges = max_hinges
        self.min_rel_gain = min_rel_gain
        self.quantiles = quantiles
        self.random_state = random_state

    @staticmethod
    def _mse(y_true, y_pred):
        d = y_true - y_pred
        return float(np.mean(d * d))

    @staticmethod
    def _fit_weighted_ridge(X_lin, X_rule, y, alpha_lin, alpha_rule):
        n = len(y)
        p_lin = X_lin.shape[1]
        p_rule = X_rule.shape[1] if X_rule is not None else 0
        D = np.hstack(
            [np.ones((n, 1), dtype=float), X_lin, X_rule if p_rule > 0 else np.zeros((n, 0), dtype=float)]
        )
        reg = np.zeros(1 + p_lin + p_rule, dtype=float)
        reg[1 : 1 + p_lin] = float(alpha_lin)
        if p_rule > 0:
            reg[1 + p_lin :] = float(alpha_rule)
        A = D.T @ D + np.diag(reg)
        b = D.T @ y
        try:
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            w = np.linalg.pinv(A) @ b
        return float(w[0]), np.asarray(w[1 : 1 + p_lin], dtype=float), np.asarray(w[1 + p_lin :], dtype=float)

    def _best_ridge_combo(self, Xtr, Rtr, ytr, Xval, Rval, yval):
        best = None
        for alpha in self.alpha_grid:
            for mult in self.rule_alpha_multipliers:
                alpha_lin = float(alpha)
                alpha_rule = float(alpha) * float(mult)
                intercept, b_lin, b_rule = self._fit_weighted_ridge(Xtr, Rtr, ytr, alpha_lin, alpha_rule)
                pred_val = intercept + Xval @ b_lin
                if Rval is not None and Rval.shape[1] > 0:
                    pred_val = pred_val + Rval @ b_rule
                mse = self._mse(yval, pred_val)
                if best is None or mse < best["mse"]:
                    best = {
                        "mse": float(mse),
                        "alpha_lin": alpha_lin,
                        "alpha_rule": alpha_rule,
                        "intercept": float(intercept),
                        "b_lin": b_lin,
                        "b_rule": b_rule,
                    }
        return best

    @staticmethod
    def _eval_rule(X, rule):
        xj = X[:, int(rule["feature"])]
        knot = float(rule["knot"])
        if rule["kind"] == "hinge_pos":
            return np.maximum(0.0, xj - knot)
        return np.maximum(0.0, knot - xj)

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

        # Base dense ridge model.
        base = self._best_ridge_combo(
            Xtr=Xtr,
            Rtr=np.zeros((len(Xtr), 0), dtype=float),
            ytr=ytr,
            Xval=Xval,
            Rval=np.zeros((len(Xval), 0), dtype=float),
            yval=yval,
        )
        best_mse = float(base["mse"])
        chosen_alpha_lin = float(base["alpha_lin"])
        chosen_alpha_rule = float(base["alpha_rule"])

        # Candidate rule features: high linear weight + high residual correlation.
        resid = ytr - (base["intercept"] + Xtr @ base["b_lin"])
        corr = []
        for j in range(p):
            xj = Xtr[:, j]
            xjc = xj - float(np.mean(xj))
            rc = resid - float(np.mean(resid))
            denom = float(np.sqrt(np.dot(xjc, xjc) * np.dot(rc, rc))) + 1e-12
            corr.append(abs(float(np.dot(xjc, rc) / denom)))
        top_corr = np.argsort(-np.asarray(corr))[: min(int(self.max_hinge_features), p)]
        top_coef = np.argsort(-np.abs(base["b_lin"]))[: min(int(self.max_hinge_features), p)]
        feat_pool = sorted(set(int(j) for j in np.concatenate([top_corr, top_coef])))

        candidates = []
        for j in feat_pool:
            xj = Xtr[:, j]
            for q in self.quantiles:
                knot = float(np.quantile(xj, q))
                candidates.append(
                    {"feature": int(j), "kind": "hinge_pos", "knot": knot, "name": f"max(0, x{j}-{knot:.4f})"}
                )
                candidates.append(
                    {"feature": int(j), "kind": "hinge_neg", "knot": knot, "name": f"max(0, {knot:.4f}-x{j})"}
                )

        selected_rules = []
        selected_tr_cols = []
        selected_val_cols = []
        remaining = list(range(len(candidates)))
        for _ in range(int(self.max_hinges)):
            if not remaining:
                break
            local_best = None
            for idx in remaining:
                trial_tr_cols = selected_tr_cols + [self._eval_rule(Xtr, candidates[idx])]
                trial_val_cols = selected_val_cols + [self._eval_rule(Xval, candidates[idx])]
                Rtr = np.column_stack(trial_tr_cols)
                Rval = np.column_stack(trial_val_cols)
                fit = self._best_ridge_combo(Xtr, Rtr, ytr, Xval, Rval, yval)
                if local_best is None or fit["mse"] < local_best["fit"]["mse"]:
                    local_best = {"idx": idx, "fit": fit, "Rtr": Rtr, "Rval": Rval}

            rel_gain = (best_mse - local_best["fit"]["mse"]) / max(best_mse, 1e-12)
            if rel_gain >= float(self.min_rel_gain):
                best_mse = float(local_best["fit"]["mse"])
                chosen_alpha_lin = float(local_best["fit"]["alpha_lin"])
                chosen_alpha_rule = float(local_best["fit"]["alpha_rule"])
                selected_rules.append(candidates[local_best["idx"]])
                selected_tr_cols = [local_best["Rtr"][:, i] for i in range(local_best["Rtr"].shape[1])]
                selected_val_cols = [local_best["Rval"][:, i] for i in range(local_best["Rval"].shape[1])]
                remaining.remove(local_best["idx"])
            else:
                break

        # Refit on full data with selected structure and chosen penalties.
        Rfull = None
        if selected_rules:
            Rfull = np.column_stack([self._eval_rule(X, rule) for rule in selected_rules])
        intercept, b_lin, b_rule = self._fit_weighted_ridge(
            X,
            Rfull if Rfull is not None else np.zeros((n, 0), dtype=float),
            y,
            alpha_lin=chosen_alpha_lin,
            alpha_rule=chosen_alpha_rule,
        )

        self.intercept_ = float(intercept)
        self.linear_coef_ = np.asarray(b_lin, dtype=float)
        self.rules_ = selected_rules
        self.rule_coef_ = np.asarray(b_rule, dtype=float)
        self.alpha_lin_ = float(chosen_alpha_lin)
        self.alpha_rule_ = float(chosen_alpha_rule)
        self.validation_mse_ = float(best_mse)
        self.active_linear_features_ = np.where(np.abs(self.linear_coef_) > 1e-6)[0].astype(int)
        active_rule_feats = {int(r["feature"]) for r in self.rules_}
        self.active_features_ = np.asarray(sorted(set(self.active_linear_features_.tolist()) | active_rule_feats), dtype=int)
        self.inactive_features_ = np.asarray([j for j in range(p) if j not in set(self.active_features_)], dtype=int)
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "linear_coef_", "rules_", "rule_coef_"])
        X = np.asarray(X, dtype=float)
        out = self.intercept_ + X @ self.linear_coef_
        for c, rule in zip(self.rule_coef_, self.rules_):
            out += float(c) * self._eval_rule(X, rule)
        return out

    def __str__(self):
        check_is_fitted(self, ["intercept_", "linear_coef_", "rules_", "rule_coef_"])
        lines = [
            "Residualized Hinge Ridge Regressor",
            f"alpha_linear={self.alpha_lin_:.6g}, alpha_rule={self.alpha_rule_:.6g}",
            "Exact prediction equation in raw features:",
        ]

        eq = f"y = {self.intercept_:+.6f}"
        for j, c in enumerate(self.linear_coef_):
            if abs(float(c)) > 1e-8:
                eq += f" {float(c):+.6f}*x{j}"
        for c, rule in zip(self.rule_coef_, self.rules_):
            eq += f" {float(c):+.6f}*{rule['name']}"
        lines.append(eq)

        lines.append("Linear coefficients:")
        for j, c in sorted(enumerate(self.linear_coef_), key=lambda z: -abs(float(z[1]))):
            lines.append(f"  x{j}: {float(c):+.6f}")

        if self.rules_:
            lines.append("Residual hinge corrections:")
            for c, rule in sorted(zip(self.rule_coef_, self.rules_), key=lambda z: -abs(float(z[0]))):
                lines.append(f"  {rule['name']}: {float(c):+.6f}")
        else:
            lines.append("Residual hinge corrections: none")

        if len(self.inactive_features_) > 0:
            lines.append("Features with near-zero total effect include: " + ", ".join(f"x{int(j)}" for j in self.inactive_features_))
        lines.append(f"Validation MSE: {self.validation_mse_:.6f}")
        ops = int(np.count_nonzero(np.abs(self.linear_coef_) > 1e-8) + 3 * len(self.rules_) + 1)
        lines.append(f"Approx arithmetic operations: {ops}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
ResidualizedHingeRidgeRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "ResidualizedHingeRidgeV2"
model_description = "Dense ridge backbone with validation-gated residual hinge corrections on high-impact features, yielding an exact raw-feature equation with compact piecewise-linear nonlinearity"
model_defs = [(model_shorthand_name, ResidualizedHingeRidgeRegressor())]

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
