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


class SymbolicResidualRidgeV1(BaseEstimator, RegressorMixin):
    """
    Validation-gated symbolic residual ridge:
    1) fit a strong dense ridge linear backbone (alpha by holdout search)
    2) build symbolic nonlinear candidates (one-knot hinges and pair interactions)
    3) greedily add at most a few terms only when holdout MSE improves.
    """

    def __init__(
        self,
        alpha_grid=None,
        term_ridge=0.3,
        val_fraction=0.2,
        top_features_for_terms=6,
        max_terms=2,
        min_relative_gain=0.005,
        random_state=42,
        eps=1e-12,
    ):
        self.alpha_grid = alpha_grid
        self.term_ridge = term_ridge
        self.val_fraction = val_fraction
        self.top_features_for_terms = top_features_for_terms
        self.max_terms = max_terms
        self.min_relative_gain = min_relative_gain
        self.random_state = random_state
        self.eps = eps

    @staticmethod
    def _safe_scale(x):
        s = np.std(x, axis=0)
        return np.where(s > 1e-12, s, 1.0)

    @staticmethod
    def _build_split(n, val_fraction, random_state):
        n_val = int(max(20, round(n * val_fraction)))
        n_val = min(max(n_val, 1), max(n - 1, 1))
        rng = np.random.RandomState(random_state)
        idx = np.arange(n)
        rng.shuffle(idx)
        val_idx = idx[:n_val]
        tr_idx = idx[n_val:]
        if tr_idx.size == 0:
            tr_idx = idx[:-1]
            val_idx = idx[-1:]
        return tr_idx, val_idx

    def _fit_joint_ridge(self, Xtr_std, ytr, Ztr=None, alpha=1.0):
        n, p = Xtr_std.shape
        if Ztr is None:
            Ztr = np.zeros((n, 0), dtype=float)
        q = Ztr.shape[1]
        D = np.column_stack([np.ones(n), Xtr_std, Ztr])
        reg = np.zeros(1 + p + q, dtype=float)
        reg[1 : 1 + p] = float(alpha)
        if q > 0:
            reg[1 + p :] = float(self.term_ridge)
        A = D.T @ D + np.diag(reg)
        b = D.T @ ytr
        try:
            theta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            theta = np.linalg.pinv(A) @ b
        b0 = float(theta[0])
        w = np.asarray(theta[1 : 1 + p], dtype=float)
        g = np.asarray(theta[1 + p :], dtype=float)
        return b0, w, g

    def _term_matrix(self, X, terms):
        if not terms:
            return np.zeros((X.shape[0], 0), dtype=float)
        cols = [self._eval_term(X, t) for t in terms]
        return np.column_stack(cols)

    @staticmethod
    def _eval_term(X, term):
        kind = term["kind"]
        if kind == "hinge_pos":
            return np.maximum(0.0, X[:, term["j"]] - term["t"])
        if kind == "hinge_neg":
            return np.maximum(0.0, term["t"] - X[:, term["j"]])
        if kind == "interaction":
            return (X[:, term["j"]] - term["mj"]) * (X[:, term["k"]] - term["mk"])
        raise ValueError(f"unknown term kind: {kind}")

    def _format_term(self, coef, term):
        if term["kind"] == "hinge_pos":
            return f"{coef:+.6f}*max(0, x{term['j']} - ({term['t']:+.6f}))"
        if term["kind"] == "hinge_neg":
            return f"{coef:+.6f}*max(0, ({term['t']:+.6f}) - x{term['j']})"
        return (
            f"{coef:+.6f}*(x{term['j']} - ({term['mj']:+.6f}))"
            f"*(x{term['k']} - ({term['mk']:+.6f}))"
        )

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape
        self.n_features_in_ = p

        tr_idx, val_idx = self._build_split(n, float(self.val_fraction), int(self.random_state))
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xval, yval = X[val_idx], y[val_idx]

        mu = np.mean(Xtr, axis=0)
        scale = self._safe_scale(Xtr)
        Xtr_std = (Xtr - mu) / scale
        Xval_std = (Xval - mu) / scale

        alphas = np.logspace(-4, 3, 16) if self.alpha_grid is None else np.asarray(self.alpha_grid, dtype=float)
        best = None
        for alpha in alphas:
            b0, w_std, _ = self._fit_joint_ridge(Xtr_std, ytr, Ztr=None, alpha=float(alpha))
            val_pred = b0 + Xval_std @ w_std
            mse = float(np.mean((yval - val_pred) ** 2))
            if (best is None) or (mse < best[0]):
                best = (mse, float(alpha), b0, w_std)

        base_val_mse, alpha_best, b0, w_std = best
        base_tr_pred = b0 + Xtr_std @ w_std
        residual_tr = ytr - base_tr_pred

        coef_raw_linear = w_std / scale
        abs_coef = np.abs(coef_raw_linear)
        k = int(min(max(3, self.top_features_for_terms), p))
        top_idx = np.argsort(-abs_coef)[:k]

        candidates = []
        for j in top_idx:
            qvals = np.unique(np.quantile(Xtr[:, j], [0.25, 0.5, 0.75]))
            for t in qvals:
                candidates.append({"kind": "hinge_pos", "j": int(j), "t": float(t)})
                candidates.append({"kind": "hinge_neg", "j": int(j), "t": float(t)})

        for a in range(len(top_idx)):
            for b in range(a + 1, len(top_idx)):
                j = int(top_idx[a])
                k2 = int(top_idx[b])
                candidates.append({
                    "kind": "interaction",
                    "j": j,
                    "k": k2,
                    "mj": float(mu[j]),
                    "mk": float(mu[k2]),
                })

        if candidates:
            scores = []
            res_norm = float(np.linalg.norm(residual_tr)) + self.eps
            for term in candidates:
                z = self._eval_term(Xtr, term)
                corr = float(abs(z @ residual_tr) / (res_norm * (np.linalg.norm(z) + self.eps)))
                scores.append(corr)
            order = np.argsort(-np.asarray(scores))
            candidates = [candidates[i] for i in order[: min(14, len(order))]]

        selected_terms = []
        selected_gamma = np.zeros(0, dtype=float)
        current_val_mse = base_val_mse
        current_b0, current_w_std = b0, w_std

        for _ in range(int(max(0, self.max_terms))):
            best_step = None
            for term in candidates:
                if term in selected_terms:
                    continue
                trial_terms = selected_terms + [term]
                Ztr = self._term_matrix(Xtr, trial_terms)
                Zval = self._term_matrix(Xval, trial_terms)
                tb0, tw_std, tg = self._fit_joint_ridge(Xtr_std, ytr, Ztr=Ztr, alpha=alpha_best)
                val_pred = tb0 + Xval_std @ tw_std + Zval @ tg
                mse = float(np.mean((yval - val_pred) ** 2))
                if (best_step is None) or (mse < best_step[0]):
                    best_step = (mse, term, tb0, tw_std, tg)

            if best_step is None:
                break

            gain = (current_val_mse - best_step[0]) / max(current_val_mse, self.eps)
            if gain < float(self.min_relative_gain):
                break

            current_val_mse = best_step[0]
            selected_terms.append(best_step[1])
            current_b0 = best_step[2]
            current_w_std = best_step[3]
            selected_gamma = best_step[4]

        coef_raw = current_w_std / scale
        intercept_raw = float(current_b0 - np.dot(coef_raw, mu))

        lin_train_pred = intercept_raw + Xtr @ coef_raw
        if selected_terms:
            Ztr_sel = self._term_matrix(Xtr, selected_terms)
            lin_train_pred = lin_train_pred + Ztr_sel @ selected_gamma
        final_train_mse = float(np.mean((ytr - lin_train_pred) ** 2))

        self.alpha_ = float(alpha_best)
        self.intercept_ = intercept_raw
        self.coef_ = coef_raw
        self.selected_terms_ = selected_terms
        self.term_coefs_ = np.asarray(selected_gamma, dtype=float)
        self.val_mse_linear_ = float(base_val_mse)
        self.val_mse_final_ = float(current_val_mse)
        self.train_mse_final_ = final_train_mse
        self.linear_coef_mass_ = float(np.sum(np.abs(self.coef_)))

        norm_mass = np.abs(self.coef_) / max(self.linear_coef_mass_, self.eps)
        self.meaningful_features_ = np.where(norm_mass >= 0.06)[0].astype(int)
        if self.meaningful_features_.size == 0:
            self.meaningful_features_ = np.array([int(np.argmax(np.abs(self.coef_)))], dtype=int)
        self.negligible_features_ = np.where(norm_mass < 0.01)[0].astype(int)
        return self

    def predict(self, X):
        check_is_fitted(self, ["coef_", "intercept_", "selected_terms_", "term_coefs_"])
        X = np.asarray(X, dtype=float)
        yhat = self.intercept_ + X @ self.coef_
        if len(self.selected_terms_) > 0:
            Z = self._term_matrix(X, self.selected_terms_)
            yhat = yhat + Z @ self.term_coefs_
        return yhat

    def __str__(self):
        check_is_fitted(self, ["coef_", "intercept_", "selected_terms_", "term_coefs_"])
        lines = [
            "Symbolic Residual Ridge Regressor",
            "Exact prediction equation on raw features:",
        ]

        eq_parts = [f"{self.intercept_:+.6f}"]
        for j in range(len(self.coef_)):
            eq_parts.append(f"{self.coef_[j]:+.6f}*x{j}")
        for coef, term in zip(self.term_coefs_, self.selected_terms_):
            eq_parts.append(self._format_term(float(coef), term))
        lines.append("  y = " + " ".join(eq_parts))
        lines.append("")
        lines.append(f"Linear ridge alpha selected by validation: {self.alpha_:.6g}")
        lines.append(f"Validation MSE linear-only: {self.val_mse_linear_:.6f}")
        lines.append(f"Validation MSE final: {self.val_mse_final_:.6f}")
        lines.append(f"Train MSE final: {self.train_mse_final_:.6f}")

        lines.append("")
        lines.append("Linear coefficients (largest magnitude first):")
        for j in np.argsort(-np.abs(self.coef_)):
            lines.append(f"  x{int(j)}: {float(self.coef_[j]):+.6f}")

        lines.append("")
        if len(self.selected_terms_) == 0:
            lines.append("Selected nonlinear terms: none")
        else:
            lines.append("Selected nonlinear terms:")
            for idx, (coef, term) in enumerate(zip(self.term_coefs_, self.selected_terms_), 1):
                lines.append(f"  term {idx}: {self._format_term(float(coef), term)}")

        if len(self.meaningful_features_) > 0:
            lines.append("Meaningful features: " + ", ".join(f"x{int(i)}" for i in self.meaningful_features_))
        if len(self.negligible_features_) > 0:
            lines.append("Negligible features: " + ", ".join(f"x{int(i)}" for i in self.negligible_features_))
        return "\n".join(lines)


import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SymbolicResidualRidgeV1.__module__ = "interpretable_regressor"

model_shorthand_name = "SymbolicResidualRidgeV1"
model_description = "Validation-selected dense ridge backbone with greedy holdout-gated symbolic residual terms (hinge and pairwise interaction), capped to keep a concise simulatable equation"
model_defs = [(model_shorthand_name, SymbolicResidualRidgeV1())]

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
