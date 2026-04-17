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


class SparseAdditiveInteractionRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse linear backbone plus a tiny set of nonlinear basis terms.

    - Bootstrap sign-stability screening picks a compact linear support.
    - Validation-gated forward selection adds <= 2 nonlinear terms:
      one-dimensional hinges and/or pairwise interactions.
    - Final predictor is an explicit raw-feature equation.
    """

    def __init__(
        self,
        screening_lambda=0.8,
        linear_lambda=0.2,
        nonlinear_lambda=0.08,
        n_bootstraps=7,
        bootstrap_frac=0.75,
        max_linear_features=8,
        candidate_features=4,
        n_quantiles=5,
        max_nonlinear_terms=2,
        min_stability=0.35,
        min_relative_gain=0.012,
        random_state=42,
    ):
        self.screening_lambda = screening_lambda
        self.linear_lambda = linear_lambda
        self.nonlinear_lambda = nonlinear_lambda
        self.n_bootstraps = n_bootstraps
        self.bootstrap_frac = bootstrap_frac
        self.max_linear_features = max_linear_features
        self.candidate_features = candidate_features
        self.n_quantiles = n_quantiles
        self.max_nonlinear_terms = max_nonlinear_terms
        self.min_stability = min_stability
        self.min_relative_gain = min_relative_gain
        self.random_state = random_state

    @staticmethod
    def _safe_scale(X):
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
        sigma = np.where(sigma < 1e-8, 1.0, sigma)
        return mu, sigma

    @staticmethod
    def _solve_ridge(X, y, lam):
        n, p = X.shape
        D = np.hstack([np.ones((n, 1), dtype=float), X])
        reg = np.zeros(p + 1, dtype=float)
        if p > 0:
            reg[1:] = max(float(lam), 0.0)
        A = D.T @ D + np.diag(reg)
        b = D.T @ y
        try:
            beta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(A) @ b
        return float(beta[0]), np.asarray(beta[1:], dtype=float), D @ beta

    @staticmethod
    def _hinge_col(x, thr, sign):
        return np.maximum(0.0, sign * (x - thr))

    def _screen_features(self, Xs, y):
        n, p = Xs.shape
        rng = np.random.RandomState(self.random_state)
        boot_size = max(16, int(float(self.bootstrap_frac) * n))
        W = []
        for _ in range(max(1, int(self.n_bootstraps))):
            idx = rng.randint(0, n, size=boot_size)
            _, w, _ = self._solve_ridge(Xs[idx], y[idx], self.screening_lambda)
            W.append(w)
        W = np.vstack(W)

        mag = np.median(np.abs(W), axis=0)
        sign_stab = np.abs(np.mean(np.sign(W), axis=0))
        score = mag * sign_stab

        order = np.argsort(score)[::-1]
        selected = []
        for j in order:
            if len(selected) >= int(self.max_linear_features):
                break
            if score[j] > 1e-9 and sign_stab[j] >= float(self.min_stability):
                selected.append(int(j))
        if not selected:
            selected = [int(order[0])]
        selected = np.array(sorted(set(selected)), dtype=int)
        return selected, score, sign_stab

    def _build_candidates(self, Xs, active_idx, screen_score):
        candidates = []
        if len(active_idx) == 0:
            return candidates

        local_rank = np.argsort(screen_score[active_idx])[::-1]
        top_local = local_rank[: min(int(self.candidate_features), len(active_idx))]
        top_global = [int(active_idx[k]) for k in top_local]

        q_grid = np.linspace(0.15, 0.85, max(2, int(self.n_quantiles)))
        for j in top_global:
            xj = Xs[:, j]
            thrs = np.unique(np.quantile(xj, q_grid))
            for thr in thrs:
                thr = float(thr)
                for sign in (-1.0, 1.0):
                    candidates.append(
                        {
                            "kind": "hinge",
                            "j": int(j),
                            "thr_std": thr,
                            "sign": float(sign),
                            "z": self._hinge_col(xj, thr, sign),
                        }
                    )

        for a in range(len(top_global)):
            for b in range(a + 1, len(top_global)):
                i = top_global[a]
                j = top_global[b]
                candidates.append(
                    {
                        "kind": "interaction",
                        "i": int(i),
                        "j": int(j),
                        "z": Xs[:, i] * Xs[:, j],
                    }
                )
        return candidates

    def _fit_with_terms(self, X_lin, y, z_list):
        n = X_lin.shape[0]
        if z_list:
            Z = np.column_stack(z_list)
            D = np.hstack([np.ones((n, 1), dtype=float), X_lin, Z])
        else:
            D = np.hstack([np.ones((n, 1), dtype=float), X_lin])

        n_lin = X_lin.shape[1]
        n_nonlin = len(z_list)
        reg = np.zeros(1 + n_lin + n_nonlin, dtype=float)
        if n_lin > 0:
            reg[1 : 1 + n_lin] = max(float(self.linear_lambda), 0.0)
        if n_nonlin > 0:
            reg[1 + n_lin :] = max(float(self.nonlinear_lambda), 0.0)

        A = D.T @ D + np.diag(reg)
        b = D.T @ y
        try:
            beta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(A) @ b
        pred = D @ beta
        return beta, pred

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape
        self.n_features_in_ = p

        mu, sigma = self._safe_scale(X)
        Xs = (X - mu) / sigma

        active_idx, screen_score, sign_stability = self._screen_features(Xs, y)
        X_lin = Xs[:, active_idx]
        candidates = self._build_candidates(Xs, active_idx, screen_score)

        rng = np.random.RandomState(self.random_state + 19)
        perm = rng.permutation(n)
        n_val = max(20, n // 5)
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]
        if len(tr_idx) < 16:
            tr_idx = perm
            val_idx = perm

        Xlin_tr = X_lin[tr_idx]
        Xlin_val = X_lin[val_idx]
        y_tr = y[tr_idx]
        y_val = y[val_idx]

        beta_base, _ = self._fit_with_terms(Xlin_tr, y_tr, [])
        pred_val_base = beta_base[0] + Xlin_val @ beta_base[1:]
        mse_base = float(np.mean((y_val - pred_val_base) ** 2))
        best_mse = mse_base

        selected_terms = []
        for _ in range(max(0, int(self.max_nonlinear_terms))):
            best = None
            for c_idx, c in enumerate(candidates):
                if c_idx in selected_terms:
                    continue
                trial_terms = selected_terms + [c_idx]
                z_tr = [candidates[k]["z"][tr_idx] for k in trial_terms]
                beta_trial, _ = self._fit_with_terms(Xlin_tr, y_tr, z_tr)
                n_lin = Xlin_tr.shape[1]
                b0 = float(beta_trial[0])
                w = beta_trial[1 : 1 + n_lin]
                a = beta_trial[1 + n_lin :]
                z_val = np.column_stack([candidates[k]["z"][val_idx] for k in trial_terms])
                pred_val = b0 + Xlin_val @ w + z_val @ a
                mse_val = float(np.mean((y_val - pred_val) ** 2))
                if (best is None) or (mse_val < best["mse"]):
                    best = {"idx": c_idx, "mse": mse_val}

            if best is None:
                break
            rel_gain = (best_mse - best["mse"]) / max(best_mse, 1e-12)
            if rel_gain < float(self.min_relative_gain):
                break
            selected_terms.append(best["idx"])
            best_mse = best["mse"]

        z_all = [candidates[k]["z"] for k in selected_terms]
        beta_final, pred_train = self._fit_with_terms(X_lin, y, z_all)
        n_lin = X_lin.shape[1]
        b_std = float(beta_final[0])
        w_std = np.asarray(beta_final[1 : 1 + n_lin], dtype=float)
        a_std = np.asarray(beta_final[1 + n_lin :], dtype=float) if selected_terms else np.zeros(0, dtype=float)

        coef_raw = np.zeros(p, dtype=float)
        coef_raw[active_idx] = w_std / sigma[active_idx]
        intercept_raw = float(b_std - np.dot(coef_raw, mu))

        pair_terms = []
        hinge_terms = []
        for k, c_idx in enumerate(selected_terms):
            c = candidates[c_idx]
            if c["kind"] == "interaction":
                i = int(c["i"])
                j = int(c["j"])
                a_raw = float(a_std[k] / (sigma[i] * sigma[j]))
                coef_raw[i] += -a_raw * mu[j]
                coef_raw[j] += -a_raw * mu[i]
                intercept_raw += a_raw * mu[i] * mu[j]
                pair_terms.append({"i": i, "j": j, "coef": a_raw})
            elif c["kind"] == "hinge":
                j = int(c["j"])
                a_raw = float(a_std[k] / sigma[j])
                thr_raw = float(mu[j] + sigma[j] * c["thr_std"])
                hinge_terms.append(
                    {
                        "j": j,
                        "thr": thr_raw,
                        "sign": float(c["sign"]),
                        "coef": a_raw,
                    }
                )

        self.feature_mu_ = mu
        self.feature_sigma_ = sigma
        self.active_features_ = active_idx
        self.screen_score_ = screen_score
        self.sign_stability_ = sign_stability
        self.intercept_ = intercept_raw
        self.linear_coef_ = coef_raw
        self.pair_terms_ = pair_terms
        self.hinge_terms_ = hinge_terms
        self.selected_term_indices_ = np.asarray(selected_terms, dtype=int)
        self.train_mse_ = float(np.mean((y - pred_train) ** 2))
        self.val_mse_base_ = mse_base
        self.val_mse_final_ = best_mse
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "linear_coef_", "pair_terms_", "hinge_terms_"])
        X = np.asarray(X, dtype=float)
        yhat = self.intercept_ + X @ self.linear_coef_
        for t in self.pair_terms_:
            yhat += float(t["coef"]) * X[:, int(t["i"])] * X[:, int(t["j"])]
        for t in self.hinge_terms_:
            j = int(t["j"])
            if float(t["sign"]) > 0:
                yhat += float(t["coef"]) * np.maximum(0.0, X[:, j] - float(t["thr"]))
            else:
                yhat += float(t["coef"]) * np.maximum(0.0, float(t["thr"]) - X[:, j])
        return yhat

    def __str__(self):
        check_is_fitted(self, ["intercept_", "linear_coef_", "pair_terms_", "hinge_terms_"])
        lines = ["Sparse Additive Interaction Regressor", "Equation (raw features):"]
        eq_terms = [f"{self.intercept_:.6f}"]

        lin_active = [j for j in range(self.n_features_in_) if abs(self.linear_coef_[j]) > 1e-10]
        for j in sorted(lin_active, key=lambda k: -abs(self.linear_coef_[k])):
            eq_terms.append(f"{self.linear_coef_[j]:+,.6f}*x{j}".replace(",", ""))

        for t in sorted(self.pair_terms_, key=lambda d: -abs(float(d["coef"]))):
            i = int(t["i"])
            j = int(t["j"])
            c = float(t["coef"])
            eq_terms.append(f"{c:+,.6f}*x{i}*x{j}".replace(",", ""))

        for t in sorted(self.hinge_terms_, key=lambda d: -abs(float(d["coef"]))):
            j = int(t["j"])
            c = float(t["coef"])
            thr = float(t["thr"])
            if float(t["sign"]) > 0:
                eq_terms.append(f"{c:+,.6f}*max(0, x{j} - {thr:.6f})".replace(",", ""))
            else:
                eq_terms.append(f"{c:+,.6f}*max(0, {thr:.6f} - x{j})".replace(",", ""))
        lines.append("y = " + " ".join(eq_terms))
        lines.append("")
        lines.append("Omitted terms have coefficient 0.")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparseAdditiveInteractionRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SparseAdditiveInteractionV1"
model_description = "Bootstrap-stability screened sparse linear backbone with validation-selected hinge and pairwise interaction basis terms in an explicit raw-feature equation"
model_defs = [(model_shorthand_name, SparseAdditiveInteractionRegressor())]


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
