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


class HierSparseInteractionRidgeRegressor(BaseEstimator, RegressorMixin):
    """
    Ridge backbone with validation-gated hierarchical pairwise interactions.

    Form:
      y = intercept + sum_j w_j * x_j + sum_(i,j in S) v_ij * x_i * x_j

    Interaction candidates are restricted to pairs among the strongest main
    effects from a RidgeCV backbone, and only retained when holdout MSE
    improves materially.
    """

    def __init__(
        self,
        ridge_alphas=(1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0, 100.0),
        max_main_features=10,
        max_interactions=2,
        interaction_corr_min=0.03,
        min_val_gain=0.0025,
        max_display_linear_terms=25,
        linear_coef_threshold=1e-6,
        random_state=42,
    ):
        self.ridge_alphas = ridge_alphas
        self.max_main_features = max_main_features
        self.max_interactions = max_interactions
        self.interaction_corr_min = interaction_corr_min
        self.min_val_gain = min_val_gain
        self.max_display_linear_terms = max_display_linear_terms
        self.linear_coef_threshold = linear_coef_threshold
        self.random_state = random_state

    @staticmethod
    def _safe_std(x):
        s = np.std(x, axis=0)
        s[s < 1e-12] = 1.0
        return s

    def _impute(self, X):
        X = np.asarray(X, dtype=float).copy()
        bad = ~np.isfinite(X)
        if bad.any():
            X[bad] = np.take(self.feature_medians_, np.where(bad)[1])
        return X

    def _fit_ridge_design(self, A, y):
        from sklearn.linear_model import RidgeCV

        m = np.mean(A, axis=0)
        s = self._safe_std(A)
        As = (A - m) / s
        reg = RidgeCV(alphas=np.asarray(self.ridge_alphas, dtype=float), cv=3)
        reg.fit(As, y)
        return reg, m, s

    @staticmethod
    def _predict_ridge_design(reg, A, m, s):
        return reg.predict((A - m) / s)

    def _build_design(self, X, pairs):
        if not pairs:
            return X
        cols = [X]
        for i, j in pairs:
            cols.append((X[:, i] * X[:, j])[:, None])
        return np.concatenate(cols, axis=1)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape
        self.n_features_in_ = p

        self.feature_medians_ = np.nanmedian(X, axis=0)
        self.feature_medians_ = np.where(np.isfinite(self.feature_medians_), self.feature_medians_, 0.0)
        X = self._impute(X)

        rng = np.random.RandomState(self.random_state)
        idx = np.arange(n)
        rng.shuffle(idx)
        n_val = max(40, int(0.2 * n))
        val_idx = idx[:n_val]
        tr_idx = idx[n_val:]
        if len(tr_idx) < max(20, int(0.5 * n)):
            tr_idx = idx
            val_idx = idx[: min(80, n)]

        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xva, yva = X[val_idx], y[val_idx]

        reg_base, A_m_base, A_s_base = self._fit_ridge_design(Xtr, ytr)
        pred_tr = self._predict_ridge_design(reg_base, Xtr, A_m_base, A_s_base)
        pred_va = self._predict_ridge_design(reg_base, Xva, A_m_base, A_s_base)
        base_val_mse = float(np.mean((yva - pred_va) ** 2))
        resid_tr = ytr - pred_tr

        # Candidate features: strongest absolute main effects.
        coef_main_raw = reg_base.coef_ / A_s_base
        main_order = np.argsort(np.abs(coef_main_raw))[::-1]
        k_main = min(max(2, int(self.max_main_features)), p)
        top_main = main_order[:k_main]

        # Score pairwise products by residual correlation.
        r = resid_tr - np.mean(resid_tr)
        r_norm = np.sqrt(np.dot(r, r)) + 1e-12
        pair_scores = []
        for a in range(len(top_main)):
            for b in range(a + 1, len(top_main)):
                i, j = int(top_main[a]), int(top_main[b])
                z = Xtr[:, i] * Xtr[:, j]
                zc = z - np.mean(z)
                denom = np.sqrt(np.dot(zc, zc)) + 1e-12
                corr = abs(float(np.dot(zc, r) / (denom * r_norm)))
                if corr >= float(self.interaction_corr_min):
                    pair_scores.append(((i, j), corr))
        pair_scores.sort(key=lambda x: x[1], reverse=True)

        # Forward-select interactions with holdout gating.
        selected_pairs = []
        current_val_mse = base_val_mse
        max_pool = min(len(pair_scores), max(0, int(self.max_interactions)) * 6)
        for cand_pair, _ in pair_scores[:max_pool]:
            if len(selected_pairs) >= int(self.max_interactions):
                break
            trial_pairs = selected_pairs + [cand_pair]
            Atr = self._build_design(Xtr, trial_pairs)
            Ava = self._build_design(Xva, trial_pairs)
            reg_trial, A_m_trial, A_s_trial = self._fit_ridge_design(Atr, ytr)
            pred_trial = self._predict_ridge_design(reg_trial, Ava, A_m_trial, A_s_trial)
            trial_val_mse = float(np.mean((yva - pred_trial) ** 2))
            if trial_val_mse + 1e-12 < current_val_mse:
                selected_pairs = trial_pairs
                current_val_mse = trial_val_mse

        self.use_interactions_ = (base_val_mse - current_val_mse) >= float(self.min_val_gain) and len(selected_pairs) > 0
        self.selected_pairs_ = selected_pairs if self.use_interactions_ else []

        # Refit final model on full data.
        Afinal = self._build_design(X, self.selected_pairs_)
        reg_final, A_m_final, A_s_final = self._fit_ridge_design(Afinal, y)

        self._reg_ = reg_final
        self._design_mean_ = A_m_final
        self._design_std_ = A_s_final
        self.alpha_ = float(reg_final.alpha_)

        coef_std = reg_final.coef_.astype(float)
        coef_design_raw = coef_std / A_s_final
        intercept_raw = float(reg_final.intercept_ - np.dot(coef_design_raw, A_m_final))

        self.coef_ = coef_design_raw[:p].astype(float)
        self.intercept_ = intercept_raw

        self.interaction_terms_ = []
        for k, (i, j) in enumerate(self.selected_pairs_):
            c = float(coef_design_raw[p + k])
            self.interaction_terms_.append((int(i), int(j), c))

        imp = np.abs(self.coef_).copy()
        for i, j, c in self.interaction_terms_:
            share = 0.5 * abs(c)
            imp[i] += share
            imp[j] += share
        self.feature_importance_ = imp
        self.feature_rank_ = np.argsort(imp)[::-1]
        return self

    def predict(self, X):
        check_is_fitted(self, ["_reg_", "_design_mean_", "_design_std_"])
        X = self._impute(X)
        A = self._build_design(X, self.selected_pairs_)
        return self._predict_ridge_design(self._reg_, A, self._design_mean_, self._design_std_)

    def __str__(self):
        check_is_fitted(self, ["coef_", "intercept_", "feature_rank_"])
        lines = [
            "HierSparseInteractionRidgeRegressor",
            f"Backbone: RidgeCV with standardized design (alpha={self.alpha_:.4g})",
            f"Validation-gated interactions used: {len(self.interaction_terms_)}",
            "",
            "Prediction equation:",
        ]

        eq_terms = [f"{self.intercept_:+.6f}"]
        if self.n_features_in_ <= int(self.max_display_linear_terms):
            linear_idx = np.arange(self.n_features_in_)
        else:
            abs_c = np.abs(self.coef_)
            keep = np.where(abs_c >= float(self.linear_coef_threshold))[0]
            if keep.size == 0:
                keep = np.array([int(np.argmax(abs_c))], dtype=int)
            order = np.argsort(abs_c[keep])[::-1]
            keep = keep[order[: int(self.max_display_linear_terms)]]
            linear_idx = np.array(sorted(set(int(k) for k in keep.tolist())), dtype=int)

        for j in linear_idx:
            c = float(self.coef_[int(j)])
            eq_terms.append(f"{c:+.6f}*x{int(j)}")
        for i, j, c in self.interaction_terms_:
            eq_terms.append(f"{c:+.6f}*x{i}*x{j}")
        lines.append("  y = " + " ".join(eq_terms))

        hidden = self.n_features_in_ - len(linear_idx)
        if hidden > 0:
            lines.append(f"  (+ {hidden} small linear terms omitted for readability)")

        lines.extend(["", "Feature importance (top 12):"])
        for j in self.feature_rank_[: min(12, self.n_features_in_)]:
            lines.append(f"  x{int(j)}: {self.feature_importance_[int(j)]:.6f}")

        near_zero = [f"x{j}" for j, v in enumerate(np.abs(self.coef_)) if v < 1e-5]
        if near_zero:
            lines.append("Near-zero linear coefficients: " + ", ".join(near_zero[:20]))
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys

_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
HierSparseInteractionRidgeRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "HierSparseInterRidgeV1"
model_description = "RidgeCV backbone with validation-gated sparse pairwise interactions selected among top main-effect features"
model_defs = [(model_shorthand_name, HierSparseInteractionRidgeRegressor())]


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
