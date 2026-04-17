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


class StabilitySparseHingeRegressor(BaseEstimator, RegressorMixin):
    """
    Stability-screened sparse ridge with optional single hinge correction.

    1) Fit bootstrapped dense ridge models on standardized features.
    2) Keep features with large and sign-stable coefficients (sparse backbone).
    3) Fit sparse ridge on the selected features.
    4) Optionally add one hinge basis term chosen by validation gain.
    """

    def __init__(
        self,
        ridge_lambda=0.4,
        screening_lambda=1.0,
        hinge_lambda=0.05,
        n_bootstraps=5,
        max_active_features=10,
        min_stability=0.45,
        min_hinge_gain=0.01,
        random_state=42,
    ):
        self.ridge_lambda = ridge_lambda
        self.screening_lambda = screening_lambda
        self.hinge_lambda = hinge_lambda
        self.n_bootstraps = n_bootstraps
        self.max_active_features = max_active_features
        self.min_stability = min_stability
        self.min_hinge_gain = min_hinge_gain
        self.random_state = random_state

    @staticmethod
    def _safe_scale(X):
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
        sigma = np.where(sigma < 1e-8, 1.0, sigma)
        return mu, sigma

    @staticmethod
    def _hinge_col(x, thr, sign):
        return np.maximum(0.0, sign * (x - thr))

    @staticmethod
    def _solve_linear_system(D, y, reg):
        A = D.T @ D + np.diag(reg)
        b = D.T @ y
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(A) @ b

    def _fit_subset(self, Xs_sub, y, lam):
        n, k = Xs_sub.shape
        D = np.hstack([np.ones((n, 1), dtype=float), Xs_sub])
        reg = np.zeros(k + 1, dtype=float)
        reg[1:] = max(float(lam), 0.0)
        beta = self._solve_linear_system(D, y, reg)
        pred = D @ beta
        return float(beta[0]), np.asarray(beta[1:], dtype=float), pred

    def _fit_subset_with_hinge(self, Xs_sub, y, xj_all, thr, sign):
        n, k = Xs_sub.shape
        z = self._hinge_col(xj_all, thr, sign)
        D = np.hstack([np.ones((n, 1), dtype=float), Xs_sub, z[:, None]])
        reg = np.zeros(k + 2, dtype=float)
        reg[1 : 1 + k] = max(float(self.ridge_lambda), 0.0)
        reg[-1] = max(float(self.hinge_lambda), 0.0)
        beta = self._solve_linear_system(D, y, reg)
        pred = D @ beta
        return float(beta[0]), np.asarray(beta[1 : 1 + k], dtype=float), float(beta[-1]), pred

    def _screen_features(self, Xs, y):
        n, p = Xs.shape
        rng = np.random.RandomState(self.random_state)
        coefs = []
        sample_size = max(16, int(0.8 * n))
        for _ in range(max(1, int(self.n_bootstraps))):
            idx = rng.randint(0, n, size=sample_size)
            _, w, _ = self._fit_subset(Xs[idx], y[idx], self.screening_lambda)
            coefs.append(w)
        W = np.vstack(coefs)
        mag = np.median(np.abs(W), axis=0)
        sign_stability = np.abs(np.mean(np.sign(W), axis=0))
        score = mag * sign_stability

        order = np.argsort(score)[::-1]
        selected = []
        for j in order:
            if len(selected) >= int(self.max_active_features):
                break
            if sign_stability[j] >= float(self.min_stability) and score[j] > 1e-8:
                selected.append(int(j))

        if not selected:
            selected = [int(np.argmax(score))]
        selected.sort()
        return np.asarray(selected, dtype=int), score, sign_stability

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape
        self.n_features_in_ = p

        mu, sigma = self._safe_scale(X)
        Xs = (X - mu) / sigma

        active_idx, screen_score, screen_stability = self._screen_features(Xs, y)
        Xs_active = Xs[:, active_idx]
        b_lin, w_lin, pred_lin = self._fit_subset(Xs_active, y, self.ridge_lambda)
        mse_lin = float(np.mean((y - pred_lin) ** 2))

        # Validation-gated hinge search to reduce overfitting.
        rng = np.random.RandomState(self.random_state + 17)
        perm = rng.permutation(n)
        n_val = max(20, n // 5)
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]
        if len(tr_idx) < 10:
            tr_idx = perm
            val_idx = perm

        b_tr, w_tr, pred_tr = self._fit_subset(Xs_active[tr_idx], y[tr_idx], self.ridge_lambda)
        pred_val = b_tr + Xs_active[val_idx] @ w_tr
        mse_val_base = float(np.mean((y[val_idx] - pred_val) ** 2))

        best_hinge = None
        best_val_mse = mse_val_base
        # Try hinge only on top-importance active features (keeps model compact and search cheap).
        top_local = np.argsort(np.abs(w_tr))[::-1][: min(3, len(active_idx))]
        for local_j in top_local:
            global_j = int(active_idx[local_j])
            xj_train = Xs[tr_idx, global_j]
            thr_candidates = np.unique(np.quantile(xj_train, [0.2, 0.4, 0.6, 0.8]))
            for thr in thr_candidates:
                for sign in (-1.0, 1.0):
                    b_h, w_h, a_h, _ = self._fit_subset_with_hinge(
                        Xs_active[tr_idx], y[tr_idx], Xs[tr_idx, global_j], float(thr), float(sign)
                    )
                    z_val = self._hinge_col(Xs[val_idx, global_j], float(thr), float(sign))
                    pred_val_h = b_h + Xs_active[val_idx] @ w_h + a_h * z_val
                    mse_val = float(np.mean((y[val_idx] - pred_val_h) ** 2))
                    if mse_val < best_val_mse:
                        best_val_mse = mse_val
                        best_hinge = (global_j, float(thr), float(sign))

        val_gain = (mse_val_base - best_val_mse) / max(mse_val_base, 1e-12)
        if best_hinge is not None and val_gain >= float(self.min_hinge_gain):
            jh, thr_h, sign_h = best_hinge
            b_std, w_std_active, a_std, pred = self._fit_subset_with_hinge(Xs_active, y, Xs[:, jh], thr_h, sign_h)
            self.hinge_feature_ = int(jh)
            self.hinge_threshold_std_ = float(thr_h)
            self.hinge_sign_ = float(sign_h)
            self.hinge_coef_std_ = float(a_std)
            train_mse = float(np.mean((y - pred) ** 2))
        else:
            b_std = b_lin
            w_std_active = w_lin
            self.hinge_feature_ = None
            self.hinge_threshold_std_ = 0.0
            self.hinge_sign_ = 1.0
            self.hinge_coef_std_ = 0.0
            train_mse = mse_lin

        coef_std_full = np.zeros(p, dtype=float)
        coef_std_full[active_idx] = w_std_active
        coef_raw_full = coef_std_full / sigma
        intercept_raw = float(b_std - np.dot(coef_raw_full, mu))

        self.feature_mu_ = mu
        self.feature_sigma_ = sigma
        self.active_features_ = active_idx
        self.screen_score_ = screen_score
        self.screen_stability_ = screen_stability
        self.linear_coef_ = coef_raw_full
        self.intercept_ = intercept_raw
        self.train_mse_ = train_mse

        if self.hinge_feature_ is None:
            self.hinge_threshold_raw_ = 0.0
            self.hinge_coef_raw_ = 0.0
        else:
            j = self.hinge_feature_
            self.hinge_threshold_raw_ = float(mu[j] + sigma[j] * self.hinge_threshold_std_)
            self.hinge_coef_raw_ = float(self.hinge_coef_std_ / sigma[j])

        return self

    def predict(self, X):
        check_is_fitted(
            self,
            [
                "intercept_",
                "linear_coef_",
                "active_features_",
                "hinge_feature_",
                "hinge_threshold_raw_",
                "hinge_sign_",
                "hinge_coef_raw_",
            ],
        )
        X = np.asarray(X, dtype=float)
        yhat = self.intercept_ + X @ self.linear_coef_
        if self.hinge_feature_ is not None and abs(self.hinge_coef_raw_) > 1e-12:
            xj = X[:, self.hinge_feature_]
            yhat += self.hinge_coef_raw_ * self._hinge_col(xj, self.hinge_threshold_raw_, self.hinge_sign_)
        return yhat

    def __str__(self):
        check_is_fitted(self, ["intercept_", "linear_coef_", "active_features_", "hinge_feature_", "hinge_threshold_raw_", "hinge_sign_", "hinge_coef_raw_"])
        lines = [
            "Stability-Screened Sparse Ridge + Optional Hinge Regressor",
            f"Active linear features: {len(self.active_features_)} / {self.n_features_in_}",
            "Prediction equation on raw features:",
        ]
        eq_terms = [f"{self.intercept_:.6f}"]
        for j in self.active_features_:
            c = self.linear_coef_[j]
            eq_terms.append(f"{c:+.6f}*x{j}")
        if self.hinge_feature_ is not None and abs(self.hinge_coef_raw_) > 1e-8:
            if self.hinge_sign_ > 0:
                hinge_txt = f"max(0, x{self.hinge_feature_} - {self.hinge_threshold_raw_:.6f})"
            else:
                hinge_txt = f"max(0, {self.hinge_threshold_raw_:.6f} - x{self.hinge_feature_})"
            eq_terms.append(f"{self.hinge_coef_raw_:+.6f}*{hinge_txt}")
        lines.append("  y = " + " ".join(eq_terms))
        lines.append("")
        lines.append("Linear coefficients (active only):")
        for j in sorted(self.active_features_, key=lambda k: -abs(self.linear_coef_[k])):
            lines.append(f"  x{j}: {self.linear_coef_[j]:+.6f}")
        if self.hinge_feature_ is None or abs(self.hinge_coef_raw_) <= 1e-8:
            lines.append("Hinge term: none")
        else:
            lines.append(
                f"Hinge term: {self.hinge_coef_raw_:+.6f} on x{self.hinge_feature_} with threshold {self.hinge_threshold_raw_:.6f}"
            )
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
StabilitySparseHingeRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "StabilitySparseHingeV1"
model_description = "Bootstrapped sign-stability screened sparse ridge backbone with validation-gated single hinge correction on a top active feature"
model_defs = [(model_shorthand_name, StabilitySparseHingeRegressor())]


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
