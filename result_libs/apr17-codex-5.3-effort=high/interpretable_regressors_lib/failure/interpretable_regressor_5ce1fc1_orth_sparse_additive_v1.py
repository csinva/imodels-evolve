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


class OrthSparseAdditiveRegressor(BaseEstimator, RegressorMixin):
    """
    Ridge linear backbone + tiny orthogonal additive correction set.

    The model keeps a dense linear core for predictive strength, then greedily
    adds a few residual terms chosen from simple human-readable atoms:
      - max(0, x_j - t), max(0, t - x_j)
      - abs(x_j - t), (x_j - t)^2
      - x_i * x_j (few top features only)
    A small validation split chooses regularization, correction terms, linear
    pruning, and rounding for a compact equation that remains easy to simulate.
    """

    def __init__(
        self,
        val_fraction=0.2,
        alphas=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0),
        top_features_for_nonlinear=6,
        top_features_for_interactions=3,
        max_extra_terms=3,
        min_extra_terms=0,
        linear_prune_quantiles=(0.0, 0.2, 0.4, 0.6),
        complexity_penalty=0.002,
        round_decimals_grid=(6, 5, 4),
        coef_tol=1e-10,
        min_relative_improvement=1e-3,
        random_state=0,
    ):
        self.val_fraction = val_fraction
        self.alphas = alphas
        self.top_features_for_nonlinear = top_features_for_nonlinear
        self.top_features_for_interactions = top_features_for_interactions
        self.max_extra_terms = max_extra_terms
        self.min_extra_terms = min_extra_terms
        self.linear_prune_quantiles = linear_prune_quantiles
        self.complexity_penalty = complexity_penalty
        self.round_decimals_grid = round_decimals_grid
        self.coef_tol = coef_tol
        self.min_relative_improvement = min_relative_improvement
        self.random_state = random_state

    @staticmethod
    def _rmse(y_true, y_pred):
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    @staticmethod
    def _ridge_with_intercept(D, y, alpha):
        n, p = D.shape
        if p == 0:
            return float(np.mean(y)), np.zeros(0, dtype=float)
        scale = np.std(D, axis=0).astype(float)
        scale[scale < 1e-12] = 1.0
        Ds = D / scale
        A = np.column_stack([np.ones(n, dtype=float), Ds])
        reg = np.eye(p + 1, dtype=float)
        reg[0, 0] = 0.0
        lhs = A.T @ A + float(max(alpha, 0.0)) * reg
        rhs = A.T @ y
        try:
            sol = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        intercept = float(sol[0])
        coefs = np.asarray(sol[1:], dtype=float) / scale
        return intercept, coefs

    def _split_indices(self, n):
        if n <= 60:
            idx = np.arange(n, dtype=int)
            return idx, idx
        rng = np.random.RandomState(self.random_state)
        order = rng.permutation(n)
        n_val = int(round(float(self.val_fraction) * n))
        n_val = min(max(n_val, 30), n - 30)
        val_idx = order[:n_val]
        tr_idx = order[n_val:]
        if tr_idx.size == 0:
            tr_idx = val_idx
        return tr_idx, val_idx

    def _eval_spec(self, X, spec):
        kind = spec[0]
        if kind == "hinge_pos":
            j, t = int(spec[1]), float(spec[2])
            return np.maximum(0.0, X[:, j] - t)
        if kind == "hinge_neg":
            j, t = int(spec[1]), float(spec[2])
            return np.maximum(0.0, t - X[:, j])
        if kind == "abs":
            j, t = int(spec[1]), float(spec[2])
            return np.abs(X[:, j] - t)
        if kind == "sq":
            j, t = int(spec[1]), float(spec[2])
            d = X[:, j] - t
            return d * d
        if kind == "int":
            i, j = int(spec[1]), int(spec[2])
            return X[:, i] * X[:, j]
        raise ValueError(f"Unknown basis spec: {spec}")

    def _format_spec(self, spec, decimals):
        kind = spec[0]
        if kind == "hinge_pos":
            return f"max(0, x{spec[1]} - {float(spec[2]):.{decimals}f})"
        if kind == "hinge_neg":
            return f"max(0, {float(spec[2]):.{decimals}f} - x{spec[1]})"
        if kind == "abs":
            return f"abs(x{spec[1]} - {float(spec[2]):.{decimals}f})"
        if kind == "sq":
            return f"(x{spec[1]} - {float(spec[2]):.{decimals}f})^2"
        return f"x{spec[1]}*x{spec[2]}"

    @staticmethod
    def _stack_columns(base_matrix, extra_cols):
        if len(extra_cols) == 0:
            return base_matrix
        return np.column_stack([base_matrix] + extra_cols)

    def _build_candidate_specs(self, Xtr, ytr):
        n_features = Xtr.shape[1]
        yc = ytr - float(np.mean(ytr))
        xc = Xtr - np.mean(Xtr, axis=0, keepdims=True)
        denom = np.sqrt(np.sum(xc * xc, axis=0) * (np.sum(yc * yc) + 1e-12)) + 1e-12
        corr = np.abs((xc.T @ yc) / denom)
        feat_order = np.argsort(corr)[::-1]

        top_nonlin = feat_order[: min(int(self.top_features_for_nonlinear), n_features)]
        top_inter = feat_order[: min(int(self.top_features_for_interactions), n_features)]

        specs = []
        for j in top_nonlin:
            j = int(j)
            q25, q50, q75 = np.quantile(Xtr[:, j], [0.25, 0.5, 0.75])
            for t in (float(q25), float(q50), float(q75)):
                specs.append(("hinge_pos", j, t))
                specs.append(("hinge_neg", j, t))
            specs.append(("abs", j, float(q50)))
            specs.append(("sq", j, float(q50)))

        for a in range(len(top_inter)):
            for b in range(a + 1, len(top_inter)):
                specs.append(("int", int(top_inter[a]), int(top_inter[b])))

        # De-duplicate while preserving order.
        seen = set()
        deduped = []
        for s in specs:
            if s not in seen:
                deduped.append(s)
                seen.add(s)
        return deduped

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        n_samples, n_features = X.shape
        self.n_features_in_ = int(n_features)

        tr_idx, va_idx = self._split_indices(n_samples)
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xva, yva = X[va_idx], y[va_idx]

        # 1) Select alpha for linear backbone.
        linear_best = None
        for alpha in self.alphas:
            inter, coef = self._ridge_with_intercept(Xtr, ytr, alpha)
            rmse_va = self._rmse(yva, inter + Xva @ coef)
            if linear_best is None or rmse_va < linear_best["rmse_va"]:
                linear_best = {
                    "alpha": float(alpha),
                    "inter": float(inter),
                    "coef": np.asarray(coef, dtype=float),
                    "rmse_va": float(rmse_va),
                }

        candidate_specs = self._build_candidate_specs(Xtr, ytr)
        cand_specs, cand_tr, cand_va = [], [], []
        for spec in candidate_specs:
            col_tr = self._eval_spec(Xtr, spec).astype(float)
            if np.std(col_tr) < 1e-12:
                continue
            cand_specs.append(spec)
            cand_tr.append(col_tr)
            cand_va.append(self._eval_spec(Xva, spec).astype(float))

        # 2) Greedy forward selection of a tiny nonlinear correction set.
        selected = []
        remaining = list(range(len(cand_specs)))
        best_state = {
            "selected": [],
            "alpha": float(linear_best["alpha"]),
            "inter": float(linear_best["inter"]),
            "coef": np.asarray(linear_best["coef"], dtype=float),
            "rmse_va": float(linear_best["rmse_va"]),
            "obj": float(linear_best["rmse_va"]),
        }
        current_rmse = float(linear_best["rmse_va"])

        for _ in range(int(self.max_extra_terms)):
            if not remaining:
                break
            step_best = None
            for idx in remaining:
                trial_sel = selected + [idx]
                Dtr = self._stack_columns(Xtr, [cand_tr[i] for i in trial_sel])
                Dva = self._stack_columns(Xva, [cand_va[i] for i in trial_sel])
                for alpha in self.alphas:
                    inter, coef = self._ridge_with_intercept(Dtr, ytr, alpha)
                    pred_va = inter + Dva @ coef
                    rmse_va = self._rmse(yva, pred_va)
                    nz_lin = int(np.sum(np.abs(coef[:n_features]) > self.coef_tol))
                    complexity = nz_lin + len(trial_sel)
                    obj = rmse_va * (1.0 + float(self.complexity_penalty) * max(0, complexity - 8))
                    if step_best is None or obj < step_best["obj"]:
                        step_best = {
                            "idx": int(idx),
                            "selected": list(trial_sel),
                            "alpha": float(alpha),
                            "inter": float(inter),
                            "coef": np.asarray(coef, dtype=float),
                            "rmse_va": float(rmse_va),
                            "obj": float(obj),
                        }

            if step_best is None:
                break

            rel_impr = (current_rmse - step_best["rmse_va"]) / max(current_rmse, 1e-12)
            force_keep = len(selected) < int(self.min_extra_terms)
            if not force_keep and rel_impr < float(self.min_relative_improvement):
                break

            selected = step_best["selected"]
            remaining.remove(step_best["idx"])
            current_rmse = float(step_best["rmse_va"])
            if step_best["obj"] < best_state["obj"]:
                best_state = step_best

        self.alpha_selected_ = float(best_state["alpha"])
        self.extra_specs_ = [cand_specs[i] for i in best_state["selected"]]

        # 3) Refit on full data with selected terms.
        extra_all = [self._eval_spec(X, spec) for spec in self.extra_specs_]
        Dall = self._stack_columns(X, extra_all)
        inter_full, coef_full = self._ridge_with_intercept(Dall, y, self.alpha_selected_)
        lin_coef = np.asarray(coef_full[:n_features], dtype=float)
        extra_coef = np.asarray(coef_full[n_features:], dtype=float)

        # 4) Validation-guided linear pruning.
        extra_va = [self._eval_spec(Xva, spec) for spec in self.extra_specs_]
        Dva_extra = np.column_stack(extra_va) if extra_va else np.zeros((len(Xva), 0), dtype=float)
        lin_scale = np.std(Xtr, axis=0) + 1e-12
        contrib = np.abs(lin_coef) * lin_scale
        positive = contrib[contrib > 0]

        prune_best = None
        q_grid = np.asarray(self.linear_prune_quantiles, dtype=float)
        for q in q_grid:
            if positive.size > 0:
                thr = float(np.quantile(positive, min(max(q, 0.0), 1.0)))
            else:
                thr = 0.0
            lin_trial = lin_coef.copy()
            lin_trial[contrib < thr] = 0.0
            pred_va = inter_full + Xva @ lin_trial
            if extra_coef.size > 0:
                pred_va = pred_va + Dva_extra @ extra_coef
            rmse_va = self._rmse(yva, pred_va)
            complexity = int(np.sum(np.abs(lin_trial) > self.coef_tol) + np.sum(np.abs(extra_coef) > self.coef_tol))
            obj = rmse_va * (1.0 + float(self.complexity_penalty) * max(0, complexity - 8))
            if prune_best is None or obj < prune_best["obj"]:
                prune_best = {"obj": obj, "lin": lin_trial}

        lin_coef = np.asarray(prune_best["lin"], dtype=float)

        # 5) Validation-guided rounding to align equation text and predictions.
        round_best = None
        for dec in self.round_decimals_grid:
            dec = int(dec)
            inter_r = float(np.round(inter_full, dec))
            lin_r = np.round(lin_coef, dec)
            extra_r = np.round(extra_coef, dec)
            eps = 0.5 * (10.0 ** (-dec))
            lin_r[np.abs(lin_r) < eps] = 0.0
            extra_r[np.abs(extra_r) < eps] = 0.0

            pred_va = inter_r + Xva @ lin_r
            if extra_r.size > 0:
                pred_va = pred_va + Dva_extra @ extra_r
            rmse_va = self._rmse(yva, pred_va)
            complexity = int(np.sum(np.abs(lin_r) > self.coef_tol) + np.sum(np.abs(extra_r) > self.coef_tol))
            obj = rmse_va * (1.0 + float(self.complexity_penalty) * max(0, complexity - 8))
            if round_best is None or obj < round_best["obj"]:
                round_best = {
                    "inter": inter_r,
                    "lin": lin_r,
                    "extra": extra_r,
                    "decimals": dec,
                    "obj": obj,
                }

        self.intercept_ = float(round_best["inter"])
        self.linear_coefs_ = np.asarray(round_best["lin"], dtype=float)
        self.extra_coefs_ = np.asarray(round_best["extra"], dtype=float)
        self.round_decimals_selected_ = int(round_best["decimals"])

        # Drop rounded-out extra terms.
        nz_extra = np.where(np.abs(self.extra_coefs_) > self.coef_tol)[0]
        self.extra_specs_ = [self.extra_specs_[i] for i in nz_extra]
        self.extra_coefs_ = self.extra_coefs_[nz_extra]

        feature_imp = np.abs(self.linear_coefs_)
        for spec, coef in zip(self.extra_specs_, self.extra_coefs_):
            if spec[0] == "int":
                feature_imp[int(spec[1])] += 0.5 * abs(float(coef))
                feature_imp[int(spec[2])] += 0.5 * abs(float(coef))
            else:
                feature_imp[int(spec[1])] += abs(float(coef))
        self.feature_importance_ = feature_imp
        self.selected_features_ = sorted(int(i) for i in np.where(feature_imp > self.coef_tol)[0])

        ops = 1 + int(np.sum(np.abs(self.linear_coefs_) > self.coef_tol)) * 2
        for spec in self.extra_specs_:
            if spec[0] in ("hinge_pos", "hinge_neg", "abs"):
                ops += 4
            elif spec[0] == "sq":
                ops += 4
            else:
                ops += 3
        self.operations_ = int(ops + max(int(np.sum(np.abs(self.extra_coefs_) > self.coef_tol)) - 1, 0))
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "linear_coefs_", "extra_specs_", "extra_coefs_"])
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        pred = self.intercept_ + X @ self.linear_coefs_
        for spec, coef in zip(self.extra_specs_, self.extra_coefs_):
            pred += float(coef) * self._eval_spec(X, spec)
        return pred

    def __str__(self):
        check_is_fitted(self, ["intercept_", "linear_coefs_", "extra_specs_", "extra_coefs_"])
        dec = int(getattr(self, "round_decimals_selected_", 6))

        terms = []
        for j, coef in enumerate(self.linear_coefs_):
            if abs(float(coef)) > self.coef_tol:
                terms.append((f"x{j}", float(coef)))
        for spec, coef in zip(self.extra_specs_, self.extra_coefs_):
            if abs(float(coef)) > self.coef_tol:
                terms.append((self._format_spec(spec, dec), float(coef)))
        terms.sort(key=lambda t: abs(t[1]), reverse=True)

        rhs = [f"{self.intercept_:+.{dec}f}"] + [f"({c:+.{dec}f})*{name}" for name, c in terms]
        lines = [
            "Orthogonal Sparse Additive Regressor",
            "Answer protocol: return one numeric value only.",
            "Exact prediction equation:",
            "  y = " + " + ".join(rhs),
            "",
            "Terms sorted by |coefficient|:",
        ]
        if terms:
            for i, (name, coef) in enumerate(terms, 1):
                lines.append(f"  {i:2d}. coef={coef:+.{dec}f}  term={name}")
        else:
            lines.append("  (intercept-only)")

        active = [int(i) for i in self.selected_features_]
        lines.append("")
        lines.append("Active features: " + (", ".join(f"x{i}" for i in active) if active else "none"))
        if self.n_features_in_ <= 30 and len(active) < self.n_features_in_:
            active_set = set(active)
            zero = [f"x{i}" for i in range(self.n_features_in_) if i not in active_set]
            lines.append("Zero-contribution features: " + ", ".join(zero))

        lines.append(f"Selected ridge alpha: {self.alpha_selected_:.6g}")
        lines.append(f"Selected coefficient decimals: {self.round_decimals_selected_}")
        lines.append(f"Approximate arithmetic operations: {self.operations_}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
OrthSparseAdditiveRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "OrthSparseAdditive_v1"
model_description = "Ridge linear backbone plus orthogonal forward-selected tiny additive correction set (hinge/abs/square + few interactions)"
model_defs = [(model_shorthand_name, OrthSparseAdditiveRegressor())]


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
