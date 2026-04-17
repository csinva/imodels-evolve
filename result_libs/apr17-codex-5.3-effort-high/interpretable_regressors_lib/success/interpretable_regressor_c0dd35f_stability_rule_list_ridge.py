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


class StabilityRuleListRidgeRegressor(BaseEstimator, RegressorMixin):
    """
    Bootstrap-stable sparse ridge backbone plus a short residual rule list.

    Stage 1: identify stable linear features using bootstrap ridge frequency.
    Stage 2: fit sparse ridge on active stable features.
    Stage 3: greedily add a few one-feature threshold rules on residuals.
    """

    def __init__(
        self,
        alpha_grid=(1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0),
        n_bootstraps=5,
        bootstrap_frac=0.75,
        stability_top_k=12,
        stability_freq_min=0.5,
        max_rules=4,
        rule_quantiles=(0.15, 0.3, 0.5, 0.7, 0.85),
        rule_learning_rate=0.9,
        min_rule_rel_gain=0.008,
        random_state=42,
    ):
        self.alpha_grid = alpha_grid
        self.n_bootstraps = n_bootstraps
        self.bootstrap_frac = bootstrap_frac
        self.stability_top_k = stability_top_k
        self.stability_freq_min = stability_freq_min
        self.max_rules = max_rules
        self.rule_quantiles = rule_quantiles
        self.rule_learning_rate = rule_learning_rate
        self.min_rule_rel_gain = min_rule_rel_gain
        self.random_state = random_state

    @staticmethod
    def _safe_std(v):
        s = float(np.std(v))
        return s if s > 1e-12 else 1.0

    def _fit_ridge_standardized(self, Z, yc):
        n, p = Z.shape
        if p == 0:
            return np.zeros(0, dtype=float), float(self.alpha_grid[0])
        U, s, Vt = np.linalg.svd(Z, full_matrices=False)
        Uy = U.T @ yc

        best_alpha = float(self.alpha_grid[0])
        best_gcv = np.inf
        best_coef = np.zeros(p, dtype=float)
        n_eff = max(n, 1)

        for a in self.alpha_grid:
            alpha = float(max(a, 1e-12))
            shrink = s / (s * s + alpha)
            coef = Vt.T @ (shrink * Uy)
            resid = yc - Z @ coef
            trace_h = float(np.sum((s * s) / (s * s + alpha)))
            denom = max(1.0 - trace_h / n_eff, 1e-6)
            gcv = float(np.mean(resid * resid) / (denom * denom))
            if gcv < best_gcv:
                best_gcv = gcv
                best_alpha = alpha
                best_coef = coef
        return best_coef, best_alpha

    def _stability_select(self, Z, yc, rng):
        n, p = Z.shape
        if p == 0:
            return np.zeros(0, dtype=int), np.zeros(0, dtype=float), 0.0

        freq = np.zeros(p, dtype=float)
        coef_accum = np.zeros(p, dtype=float)
        bsize = max(8, int(float(self.bootstrap_frac) * n))

        for _ in range(int(self.n_bootstraps)):
            idx = rng.choice(n, size=bsize, replace=True)
            Zb = Z[idx]
            yb = yc[idx]
            coef_b, _ = self._fit_ridge_standardized(Zb, yb)
            k = int(min(max(1, self.stability_top_k), p))
            top = np.argsort(-np.abs(coef_b))[:k]
            freq[top] += 1.0
            coef_accum += coef_b

        freq /= max(float(self.n_bootstraps), 1.0)
        coef_mean = coef_accum / max(float(self.n_bootstraps), 1.0)

        active = np.where(freq >= float(self.stability_freq_min))[0].astype(int)
        if len(active) == 0:
            active = np.argsort(-freq)[: min(max(1, self.stability_top_k // 2), p)].astype(int)
        if len(active) == 0:
            active = np.array([int(np.argmax(np.abs(coef_mean)))], dtype=int)

        return active, freq, float(np.mean(freq[active]))

    @staticmethod
    def _rule_values(x, threshold, left_value, right_value):
        return np.where(x <= threshold, left_value, right_value)

    def _fit_best_residual_rule(self, X, residual, candidate_features, current_pred):
        y_target = current_pred + residual
        base_mse = float(np.mean((y_target - current_pred) ** 2))
        best = None

        for j in candidate_features:
            xj = X[:, j]
            thr_values = []
            for q in self.rule_quantiles:
                t = float(np.quantile(xj, float(q)))
                if np.isfinite(t):
                    thr_values.append(t)
            if not thr_values:
                continue
            thr_values = np.unique(np.asarray(thr_values, dtype=float))

            for thr in thr_values:
                left = xj <= thr
                right = ~left
                nl = int(np.sum(left))
                nr = int(np.sum(right))
                if nl < 5 or nr < 5:
                    continue

                left_val = float(np.mean(residual[left]))
                right_val = float(np.mean(residual[right]))
                step = self._rule_values(xj, thr, left_val, right_val)
                cand_pred = current_pred + float(self.rule_learning_rate) * step
                mse = float(np.mean((y_target - cand_pred) ** 2))
                rel_gain = (base_mse - mse) / max(base_mse, 1e-12)

                if best is None or rel_gain > best["gain"]:
                    best = {
                        "feature": int(j),
                        "threshold": float(thr),
                        "left_value": left_val,
                        "right_value": right_val,
                        "gain": float(rel_gain),
                    }
        return best

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape
        self.n_features_in_ = p

        rng = np.random.RandomState(self.random_state)

        self.x_mean_ = np.mean(X, axis=0)
        self.x_scale_ = np.array([self._safe_std(X[:, j]) for j in range(p)], dtype=float)
        Z = (X - self.x_mean_) / self.x_scale_

        y_mean = float(np.mean(y))
        yc = y - y_mean

        active, stability_freq, mean_active_freq = self._stability_select(Z, yc, rng)
        Za = Z[:, active]
        coef_a, alpha = self._fit_ridge_standardized(Za, yc)

        coef_std = np.zeros(p, dtype=float)
        coef_std[active] = coef_a
        coef_raw = coef_std / self.x_scale_
        intercept = y_mean - float(np.dot(self.x_mean_, coef_raw))

        pred = intercept + X @ coef_raw
        y_target = y.copy()

        candidate_features = np.argsort(-np.abs(coef_raw))[: min(max(2, len(active)), p)].astype(int)
        if len(candidate_features) == 0 and p > 0:
            candidate_features = np.arange(min(3, p), dtype=int)

        rules = []
        for _ in range(int(self.max_rules)):
            residual = y_target - pred
            best = self._fit_best_residual_rule(X, residual, candidate_features, pred)
            if best is None or best["gain"] < float(self.min_rule_rel_gain):
                break
            rules.append(best)
            pred = pred + float(self.rule_learning_rate) * self._rule_values(
                X[:, best["feature"]],
                best["threshold"],
                best["left_value"],
                best["right_value"],
            )

        self.alpha_ = float(alpha)
        self.intercept_ = float(intercept)
        self.coef_ = coef_raw
        self.active_features_ = np.asarray(active, dtype=int)
        self.stability_freq_ = stability_freq
        self.mean_active_stability_ = float(mean_active_freq)
        self.rules_ = rules

        abs_c = np.abs(self.coef_)
        max_abs = float(np.max(abs_c)) if len(abs_c) else 0.0
        thr = 0.08 * max(max_abs, 1e-12)
        self.meaningful_features_ = np.where(abs_c >= thr)[0].astype(int)
        self.negligible_features_ = np.where(abs_c < thr)[0].astype(int)
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "coef_", "rules_"])
        X = np.asarray(X, dtype=float)
        pred = self.intercept_ + X @ self.coef_
        lr = float(self.rule_learning_rate)
        for rule in self.rules_:
            pred = pred + lr * self._rule_values(
                X[:, rule["feature"]],
                rule["threshold"],
                rule["left_value"],
                rule["right_value"],
            )
        return pred

    def __str__(self):
        check_is_fitted(self, ["intercept_", "coef_", "rules_"])
        lines = [
            "Stability Rule-List Ridge Regressor",
            "Exact prediction recipe:",
            "1) Start from sparse linear backbone.",
            "2) Add each residual threshold rule in order.",
        ]

        eq = f"base(x) = {self.intercept_:+.6f}"
        active_sorted = np.argsort(-np.abs(self.coef_))
        for j in active_sorted:
            c = float(self.coef_[j])
            if abs(c) < 1e-12:
                continue
            eq += f" {c:+.6f}*x{int(j)}"
        lines.append(eq)
        lines.append(f"Ridge alpha (GCV): {self.alpha_:.6g}")
        lines.append(
            f"Bootstrap stability: {len(self.active_features_)} active features, "
            f"mean inclusion frequency {self.mean_active_stability_:.2f}"
        )

        if self.rules_:
            lines.append("Residual rule list (apply sequentially):")
            for i, rule in enumerate(self.rules_, 1):
                lines.append(
                    f"  Rule {i}: add {self.rule_learning_rate:.3f}*({rule['left_value']:+.6f} if x{rule['feature']} <= {rule['threshold']:.6g} else {rule['right_value']:+.6f})"
                )
        else:
            lines.append("Residual rule list: none selected.")

        lines.append("Features ranked by absolute linear effect:")
        for idx in active_sorted:
            lines.append(
                f"  x{int(idx)}: coef={float(self.coef_[idx]):+.6f}, stability={float(self.stability_freq_[idx]):.2f}"
            )

        if len(self.meaningful_features_) > 0:
            lines.append("Meaningfully used features: " + ", ".join(f"x{int(j)}" for j in self.meaningful_features_))
        if len(self.negligible_features_) > 0:
            lines.append("Features with negligible effect: " + ", ".join(f"x{int(j)}" for j in self.negligible_features_))
        lines.append("Model is simulatable as one sparse equation plus a short ordered rule list.")
        return "\n".join(lines)


import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
StabilityRuleListRidgeRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "StabilityRuleListRidgeV1"
model_description = "Bootstrap-stable sparse ridge backbone with a short residual threshold rule list trained by greedy residual gain, yielding an explicit linear-plus-rules simulator"
model_defs = [(model_shorthand_name, StabilityRuleListRidgeRegressor())]

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
