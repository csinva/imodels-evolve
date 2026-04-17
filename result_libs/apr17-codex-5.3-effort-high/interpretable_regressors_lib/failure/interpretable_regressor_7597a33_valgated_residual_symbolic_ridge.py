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


class ValGatedResidualSymbolicRidgeV1(BaseEstimator, RegressorMixin):
    """
    Dense-from-scratch ridge with a tiny symbolic residual dictionary.

    1) Fit a dense linear backbone with SVD-GCV ridge.
    2) Build a compact residual dictionary (hinge, quadratic, one interaction).
    3) Greedily add at most two residual terms only when validation gain is real.
    4) Jointly refit backbone + accepted terms with ridge for one explicit equation.
    """

    def __init__(
        self,
        alpha_grid=None,
        val_frac=0.2,
        top_features_for_residuals=4,
        max_extra_terms=2,
        min_rel_gain=0.01,
        seed=42,
        eps=1e-12,
    ):
        self.alpha_grid = alpha_grid
        self.val_frac = val_frac
        self.top_features_for_residuals = top_features_for_residuals
        self.max_extra_terms = max_extra_terms
        self.min_rel_gain = min_rel_gain
        self.seed = seed
        self.eps = eps

    @staticmethod
    def _safe_scale(X):
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
        sigma = np.where(sigma > 1e-12, sigma, 1.0)
        return mu.astype(float), sigma.astype(float)

    def _ridge_fit(self, D, y, alpha):
        p = D.shape[1]
        reg = np.eye(p, dtype=float) * float(alpha)
        reg[0, 0] = 0.0
        A = D.T @ D + reg
        b = D.T @ y
        try:
            theta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            theta = np.linalg.pinv(A) @ b
        return np.asarray(theta, dtype=float)

    def _gcv_alpha(self, X, y):
        n = X.shape[0]
        D = np.column_stack([np.ones(n, dtype=float), X])
        alphas = (
            np.asarray(self.alpha_grid, dtype=float)
            if self.alpha_grid is not None
            else np.logspace(-6, 3, 20)
        )
        U, s, _ = np.linalg.svd(D, full_matrices=False)
        Uy = U.T @ y
        reg_mask = np.ones_like(s)
        if reg_mask.size > 0:
            reg_mask[0] = 0.0

        best_alpha = float(alphas[0])
        best_gcv = None
        for a in alphas:
            denom = s**2 + float(a) * reg_mask
            f = np.where(denom > 0, (s**2) / denom, 0.0)
            yhat = U @ (f * Uy)
            rss = float(np.sum((y - yhat) ** 2))
            df = float(np.sum(f))
            gcv = rss / max((n - df) ** 2, self.eps)
            if (best_gcv is None) or (gcv < best_gcv):
                best_gcv = gcv
                best_alpha = float(a)
        return best_alpha

    @staticmethod
    def _hinge(x, thr, side):
        if side > 0:
            return np.maximum(0.0, x - thr)
        return np.maximum(0.0, thr - x)

    def _term_column(self, X, term):
        t = term["type"]
        if t == "hinge":
            return self._hinge(X[:, term["j"]], term["thr"], term["side"])
        if t == "quad":
            z = X[:, term["j"]] - term["center"]
            return z * z
        if t == "inter":
            return (X[:, term["j1"]] - term["c1"]) * (X[:, term["j2"]] - term["c2"])
        raise ValueError(f"Unknown term type: {t}")

    def _term_str(self, term, coef):
        if term["type"] == "hinge":
            if term["side"] > 0:
                return f"{coef:+.6f}*max(0, x{term['j']} - ({term['thr']:+.6f}))"
            return f"{coef:+.6f}*max(0, ({term['thr']:+.6f}) - x{term['j']})"
        if term["type"] == "quad":
            return f"{coef:+.6f}*(x{term['j']} - ({term['center']:+.6f}))^2"
        return (
            f"{coef:+.6f}*(x{term['j1']} - ({term['c1']:+.6f}))"
            f"*(x{term['j2']} - ({term['c2']:+.6f}))"
        )

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape
        self.n_features_in_ = p

        mu, sigma = self._safe_scale(X)
        Xs = (X - mu) / sigma

        rng = np.random.RandomState(self.seed)
        idx = np.arange(n)
        rng.shuffle(idx)
        n_val = max(1, int(round(float(self.val_frac) * n)))
        n_val = min(n_val, max(1, n - 5))
        idx_val = idx[:n_val]
        idx_tr = idx[n_val:]
        if idx_tr.size == 0:
            idx_tr = idx
            idx_val = idx[: min(1, n)]

        X_tr, y_tr = Xs[idx_tr], y[idx_tr]
        X_val, y_val = Xs[idx_val], y[idx_val]

        alpha_base = self._gcv_alpha(X_tr, y_tr)
        D_tr = np.column_stack([np.ones(X_tr.shape[0], dtype=float), X_tr])
        theta_base = self._ridge_fit(D_tr, y_tr, alpha_base)
        pred_val_base = np.column_stack([np.ones(X_val.shape[0], dtype=float), X_val]) @ theta_base
        val_mse_base = float(np.mean((y_val - pred_val_base) ** 2))

        w_base = np.asarray(theta_base[1:], dtype=float)
        top_k = int(min(max(1, self.top_features_for_residuals), p))
        top_feats = np.argsort(-np.abs(w_base))[:top_k].astype(int).tolist()

        candidates = []
        for j in top_feats:
            col = Xs[:, j]
            q20, q50, q80 = np.quantile(col, [0.2, 0.5, 0.8])
            for thr in (float(q20), float(q50), float(q80)):
                candidates.append({"type": "hinge", "j": int(j), "thr": thr, "side": 1})
                candidates.append({"type": "hinge", "j": int(j), "thr": thr, "side": -1})
            candidates.append({"type": "quad", "j": int(j), "center": float(np.median(col))})
        if len(top_feats) >= 2:
            j1, j2 = int(top_feats[0]), int(top_feats[1])
            candidates.append({
                "type": "inter",
                "j1": j1,
                "j2": j2,
                "c1": float(np.mean(Xs[:, j1])),
                "c2": float(np.mean(Xs[:, j2])),
            })

        selected = []
        selected_cols = []
        current_val_mse = val_mse_base

        for _ in range(int(self.max_extra_terms)):
            best = None
            best_mse = current_val_mse
            for term in candidates:
                if term in selected:
                    continue
                z_full = self._term_column(Xs, term)
                if np.std(z_full[idx_tr]) < 1e-10:
                    continue
                trial_cols = selected_cols + [z_full]
                Z_tr = np.column_stack(trial_cols)
                D_trial_tr = np.column_stack([np.ones(X_tr.shape[0], dtype=float), X_tr, Z_tr[idx_tr]])
                a_trial = self._gcv_alpha(D_trial_tr[:, 1:], y_tr)
                theta_trial = self._ridge_fit(D_trial_tr, y_tr, a_trial)

                D_trial_val = np.column_stack([np.ones(X_val.shape[0], dtype=float), X_val, Z_tr[idx_val]])
                pred_val = D_trial_val @ theta_trial
                mse_val = float(np.mean((y_val - pred_val) ** 2))
                if mse_val < best_mse:
                    best_mse = mse_val
                    best = (term, z_full, a_trial)

            rel_gain = (current_val_mse - best_mse) / max(current_val_mse, self.eps)
            if best is None or rel_gain < float(self.min_rel_gain):
                break
            term, z_full, _ = best
            selected.append(term)
            selected_cols.append(z_full)
            current_val_mse = best_mse

        if selected_cols:
            Z = np.column_stack(selected_cols)
            D_final = np.column_stack([np.ones(n, dtype=float), Xs, Z])
        else:
            Z = np.zeros((n, 0), dtype=float)
            D_final = np.column_stack([np.ones(n, dtype=float), Xs])

        alpha_final = self._gcv_alpha(D_final[:, 1:], y)
        theta_final = self._ridge_fit(D_final, y, alpha_final)

        b_std = float(theta_final[0])
        w_std = np.asarray(theta_final[1 : 1 + p], dtype=float)
        extra_coef = np.asarray(theta_final[1 + p :], dtype=float)

        self.coef_ = w_std / sigma
        self.intercept_ = float(b_std - np.dot(self.coef_, mu))
        self.extra_terms_ = []
        for k, term in enumerate(selected):
            c = float(extra_coef[k])
            if abs(c) <= 1e-10:
                continue
            if term["type"] == "hinge":
                j = int(term["j"])
                thr_raw = float(mu[j] + sigma[j] * term["thr"])
                c_raw = float(c / sigma[j])
                term_raw = {"type": "hinge", "j": j, "thr": thr_raw, "side": int(term["side"])}
            elif term["type"] == "quad":
                j = int(term["j"])
                center_raw = float(mu[j] + sigma[j] * term["center"])
                c_raw = float(c / (sigma[j] ** 2))
                term_raw = {"type": "quad", "j": j, "center": center_raw}
            else:
                j1, j2 = int(term["j1"]), int(term["j2"])
                c1_raw = float(mu[j1] + sigma[j1] * term["c1"])
                c2_raw = float(mu[j2] + sigma[j2] * term["c2"])
                c_raw = float(c / (sigma[j1] * sigma[j2]))
                term_raw = {
                    "type": "inter",
                    "j1": j1,
                    "j2": j2,
                    "c1": c1_raw,
                    "c2": c2_raw,
                }
            self.extra_terms_.append((term_raw, c_raw))

        self.alpha_base_ = float(alpha_base)
        self.alpha_final_ = float(alpha_final)
        self.val_mse_base_ = float(val_mse_base)
        self.val_mse_final_ = float(current_val_mse)

        mass = np.abs(self.coef_)
        norm = mass / max(float(np.sum(mass)), self.eps)
        self.meaningful_features_ = np.where(norm >= 0.06)[0].astype(int)
        if self.meaningful_features_.size == 0:
            self.meaningful_features_ = np.array([int(np.argmax(mass))], dtype=int)
        self.negligible_features_ = np.where(norm < 0.01)[0].astype(int)
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "coef_", "extra_terms_"])
        X = np.asarray(X, dtype=float)
        yhat = self.intercept_ + X @ self.coef_
        for term, c in self.extra_terms_:
            z = self._term_column(X, term)
            yhat = yhat + float(c) * z
        return yhat

    def __str__(self):
        check_is_fitted(self, ["intercept_", "coef_", "extra_terms_"])
        lines = [
            "Validation-Gated Residual Symbolic Ridge Regressor",
            "Exact prediction equation in raw features:",
        ]
        terms = [f"{self.intercept_:+.6f}"]
        for j in np.where(np.abs(self.coef_) > 1e-12)[0]:
            terms.append(f"{float(self.coef_[j]):+.6f}*x{int(j)}")
        for term, c in self.extra_terms_:
            terms.append(self._term_str(term, float(c)))
        lines.append("  y = " + " ".join(terms))

        lines.append("")
        lines.append("Linear coefficients (sorted by absolute magnitude):")
        for j in np.argsort(-np.abs(self.coef_)):
            lines.append(f"  x{int(j)}: {float(self.coef_[j]):+.6f}")

        lines.append("")
        if self.extra_terms_:
            lines.append("Residual symbolic terms (validation-gated):")
            for i, (term, c) in enumerate(self.extra_terms_, 1):
                lines.append(f"  t{i}: {self._term_str(term, float(c))}")
        else:
            lines.append("Residual symbolic terms: none")

        lines.append("")
        lines.append(f"GCV alpha (base): {self.alpha_base_:.6g}")
        lines.append(f"GCV alpha (final): {self.alpha_final_:.6g}")
        lines.append(f"Validation MSE (base): {self.val_mse_base_:.6f}")
        lines.append(f"Validation MSE (selected): {self.val_mse_final_:.6f}")
        lines.append("Meaningful features: " + ", ".join(f"x{int(i)}" for i in self.meaningful_features_))
        if len(self.negligible_features_) > 0:
            lines.append("Negligible features: " + ", ".join(f"x{int(i)}" for i in self.negligible_features_))
        return "\n".join(lines)


import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
ValGatedResidualSymbolicRidgeV1.__module__ = "interpretable_regressor"

model_shorthand_name = "ValGatedResidualSymbolicRidgeV1"
model_description = "Dense from-scratch SVD-GCV ridge backbone with a tiny validation-gated residual symbolic dictionary (hinge/quadratic/one interaction) jointly ridge-refit into one explicit equation"
model_defs = [(model_shorthand_name, ValGatedResidualSymbolicRidgeV1())]

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
