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
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class LinearBoostedRulesRegressor(BaseEstimator, RegressorMixin):
    """CV-ridge linear backbone plus a tiny centered stump-rule residual booster."""

    def __init__(
        self,
        alpha_grid=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0),
        max_linear_features=12,
        max_rules=4,
        rule_learning_rate=0.8,
        max_rule_features=10,
        min_rule_gain=1e-4,
        quantiles=(0.15, 0.3, 0.5, 0.7, 0.85),
        n_splits=4,
        linear_coef_prune_ratio=0.015,
        random_state=0,
    ):
        self.alpha_grid = alpha_grid
        self.max_linear_features = max_linear_features
        self.max_rules = max_rules
        self.rule_learning_rate = rule_learning_rate
        self.max_rule_features = max_rule_features
        self.min_rule_gain = min_rule_gain
        self.quantiles = quantiles
        self.n_splits = n_splits
        self.linear_coef_prune_ratio = linear_coef_prune_ratio
        self.random_state = random_state

    @staticmethod
    def _safe_standardize(X):
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
        sigma = np.where(sigma > 1e-12, sigma, 1.0)
        return (X - mu) / sigma, mu, sigma

    @staticmethod
    def _ridge_solve(X, y, alpha):
        lhs = X.T @ X + float(alpha) * np.eye(X.shape[1])
        rhs = X.T @ y
        try:
            return np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(lhs) @ rhs

    def _select_features(self, Xz, yc, max_features):
        denom = np.sqrt(np.sum(Xz**2, axis=0) * np.sum(yc**2)) + 1e-12
        corr = np.abs((Xz.T @ yc) / denom)
        order = np.argsort(corr)[::-1]
        top_k = max(1, min(int(max_features), Xz.shape[1]))
        keep = order[:top_k]
        if keep.size == 0:
            keep = order[:1]
        return np.sort(keep).astype(int), corr

    def _cv_alpha(self, Xz, yc):
        n = Xz.shape[0]
        n_splits = min(max(2, int(self.n_splits)), n)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        alpha_choices = [float(a) for a in self.alpha_grid] or [1.0]

        best_alpha = alpha_choices[0]
        best_mse = np.inf
        for alpha in alpha_choices:
            mses = []
            for tr, va in kf.split(Xz):
                Xtr, ytr = Xz[tr], yc[tr]
                Xva, yva = Xz[va], yc[va]

                beta = self._ridge_solve(Xtr, ytr, alpha)
                pred = Xva @ beta
                mses.append(float(np.mean((yva - pred) ** 2)))
            mse = float(np.mean(mses))
            if mse < best_mse:
                best_mse = mse
                best_alpha = alpha
        return float(best_alpha)

    @staticmethod
    def _centered_rule_column(x_col, threshold):
        indicator = (x_col > threshold).astype(float)
        p = float(np.mean(indicator))
        h = indicator - p
        denom = float(np.dot(h, h))
        return h, p, denom

    def _find_best_rule(self, X, residual, candidate_features):
        best = None
        sse0 = float(np.dot(residual, residual))
        for j in candidate_features:
            xj = X[:, int(j)]
            thresholds = np.unique(np.quantile(xj, self.quantiles))
            for t in thresholds:
                h, p, denom = self._centered_rule_column(xj, float(t))
                if denom < 1e-10:
                    continue
                gamma = float(np.dot(residual, h) / denom)
                update = gamma * h
                sse = float(np.dot(residual - update, residual - update))
                gain = sse0 - sse
                if best is None or gain > best["gain"]:
                    best = {
                        "feature": int(j),
                        "threshold": float(t),
                        "gamma": gamma,
                        "p": p,
                        "gain": float(gain),
                    }
        return best

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y have incompatible shapes")
        if X.shape[1] == 0:
            raise ValueError("No features provided")

        self.n_features_in_ = X.shape[1]
        Xz, self.x_mean_, self.x_scale_ = self._safe_standardize(X)
        self.y_mean_ = float(np.mean(y))
        yc = y - self.y_mean_

        self.selected_features_, self.correlation_signal_ = self._select_features(
            Xz, yc, self.max_linear_features
        )
        Xz_lin = Xz[:, self.selected_features_]
        self.alpha_ = self._cv_alpha(Xz_lin, yc)
        coef_std = self._ridge_solve(Xz_lin, yc, self.alpha_)
        coef_raw = coef_std / self.x_scale_[self.selected_features_]
        self.linear_intercept_ = float(
            self.y_mean_ - np.dot(coef_raw, self.x_mean_[self.selected_features_])
        )
        self.linear_coef_raw_ = coef_raw.copy()

        max_abs = float(np.max(np.abs(self.linear_coef_raw_))) if self.linear_coef_raw_.size else 0.0
        if max_abs > 0:
            keep_mask = np.abs(self.linear_coef_raw_) >= float(self.linear_coef_prune_ratio) * max_abs
            self.linear_coef_raw_ = self.linear_coef_raw_ * keep_mask.astype(float)

        linear_pred = np.full(X.shape[0], self.linear_intercept_, dtype=float)
        self.raw_linear_coef_ = np.zeros(self.n_features_in_, dtype=float)
        for local_idx, j in enumerate(self.selected_features_):
            c = float(self.linear_coef_raw_[local_idx])
            self.raw_linear_coef_[j] = c
            linear_pred += c * X[:, j]
        self.raw_intercept_ = float(self.linear_intercept_)

        rule_candidates, _ = self._select_features(Xz, yc, self.max_rule_features)
        residual = y - linear_pred
        self.rule_terms_ = []
        lr = float(self.rule_learning_rate)
        for _ in range(int(self.max_rules)):
            best = self._find_best_rule(X, residual, rule_candidates)
            if best is None or best["gain"] < float(self.min_rule_gain):
                break
            feature = best["feature"]
            threshold = best["threshold"]
            gamma = lr * best["gamma"]
            p = best["p"]
            h, _, _ = self._centered_rule_column(X[:, feature], threshold)
            residual = residual - gamma * h
            self.rule_terms_.append(
                {
                    "feature": int(feature),
                    "threshold": float(threshold),
                    "gamma": float(gamma),
                    "p": float(p),
                    "gain": float(best["gain"]),
                }
            )

        feat_scores = np.abs(self.raw_linear_coef_).copy()
        for rule in self.rule_terms_:
            feat_scores[int(rule["feature"])] += abs(float(rule["gamma"]))
        max_score = float(np.max(feat_scores)) if feat_scores.size else 0.0
        self.feature_importances_ = feat_scores / max_score if max_score > 0 else feat_scores
        self.active_features_ = np.flatnonzero(feat_scores > 1e-12).astype(int)
        return self

    def predict(self, X):
        check_is_fitted(self, ["raw_intercept_", "raw_linear_coef_", "rule_terms_"])
        X = np.asarray(X, dtype=float)
        pred = np.full(X.shape[0], float(self.raw_intercept_), dtype=float)
        pred += X @ self.raw_linear_coef_
        for rule in self.rule_terms_:
            j = int(rule["feature"])
            t = float(rule["threshold"])
            gamma = float(rule["gamma"])
            p = float(rule["p"])
            pred += gamma * (((X[:, j] > t).astype(float)) - p)
        return pred

    def __str__(self):
        check_is_fitted(
            self,
            [
                "raw_intercept_",
                "raw_linear_coef_",
                "selected_features_",
                "active_features_",
                "feature_importances_",
                "alpha_",
                "rule_terms_",
            ],
        )

        lines = [
            "Linear + Boosted Centered Rules Regressor",
            "Exact prediction equation on raw features:",
            f"y = {self.raw_intercept_:+.6f}",
        ]
        for j, c in enumerate(self.raw_linear_coef_):
            if abs(c) > 1e-12:
                lines.append(f"    {c:+.6f} * x{j}")
        for idx, rule in enumerate(self.rule_terms_, 1):
            lines.append(
                "    "
                f"{rule['gamma']:+.6f} * (I(x{int(rule['feature'])} > {rule['threshold']:.6f}) - {rule['p']:.6f})"
                f"    [rule {idx}]"
            )

        lines.append("")
        lines.append(f"ridge_alpha = {self.alpha_:.6g}")
        lines.append(f"num_rules = {len(self.rule_terms_)}")
        lines.append(
            "selected_features = "
            + ", ".join(f"x{int(j)}" for j in self.selected_features_)
        )
        lines.append(
            "active_features = "
            + (", ".join(f"x{int(j)}" for j in self.active_features_) if self.active_features_.size else "none")
        )
        lines.append("feature_importance_by_abs_contribution:")
        for j, s in enumerate(self.feature_importances_):
            if s > 1e-8:
                lines.append(f"  x{j}: {float(s):.4f}")
        lines.append("")
        lines.append("Simulation recipe: compute linear part, then add each centered rule term exactly as written.")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys

_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])

# Update the model shorthand name and description below to reflect the class above and any changes you make.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
LinearBoostedRulesRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "LinearBoostedRules_v1"
model_description = "CV-ridge linear backbone with a tiny greedy ensemble of centered threshold-rule residual corrections and exact symbolic equation"
model_defs = [(model_shorthand_name, LinearBoostedRulesRegressor())]


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

    # --- Recompute global rank summary from updated performance_results.csv ---
    # Build dataset -> {model: rmse}
    perf_table = defaultdict(dict)
    with open(perf_csv, newline="") as f:
        for row in csv.DictReader(f):
            ds = row["dataset"]
            m = row["model"]
            rmse_s = row.get("rmse", "")
            if rmse_s in ("", None):
                perf_table[ds][m] = float("nan")
            else:
                try:
                    perf_table[ds][m] = float(rmse_s)
                except ValueError:
                    perf_table[ds][m] = float("nan")

    avg_rank, _ = compute_rank_scores(perf_table)
    mean_rank = avg_rank.get(model_name, float("nan"))

    # --- Upsert overall_results.csv ---
    overall_rows = [{
        "commit": git_hash,
        "mean_rank": f"{mean_rank:.2f}" if np.isfinite(mean_rank) else "",
        "frac_interpretability_tests_passed": f"{(n_passed / total):.4f}" if total else "",
        "status": "",  # fill manually after reviewing
        "model_name": model_name,
        "description": model_description,
    }]
    upsert_overall_results(overall_rows, RESULTS_DIR)

    # --- Plot update ---
    overall_csv = os.path.join(RESULTS_DIR, "overall_results.csv")
    plot_interp_vs_performance(
        overall_csv,
        os.path.join(RESULTS_DIR, "interpretability_vs_performance.png"),
    )

    # Print compact summary
    std_names = {t.__name__ for t in ALL_TESTS}
    hard_names = {t.__name__ for t in HARD_TESTS}
    ins_names = {t.__name__ for t in INSIGHT_TESTS}
    n_std = sum(r["passed"] for r in interp_results if r["test"] in std_names)
    n_hard = sum(r["passed"] for r in interp_results if r["test"] in hard_names)
    n_ins = sum(r["passed"] for r in interp_results if r["test"] in ins_names)

    print("\n---")
    print(f"tests_passed:  {n_passed}/{total} ({(n_passed/total):.2%})  "
          f"[std {n_std}/{len(std_names)}  hard {n_hard}/{len(hard_names)}  insight {n_ins}/{len(ins_names)}]")
    print(f"total_seconds: {time.time() - t0:.1f}s")
