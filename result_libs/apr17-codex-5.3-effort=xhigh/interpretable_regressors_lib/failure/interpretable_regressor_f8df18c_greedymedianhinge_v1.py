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
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class GreedyMedianHingeRegressor(BaseEstimator, RegressorMixin):
    """Sparse forward-selected raw-feature equation with median hinges and tiny interactions."""

    def __init__(
        self,
        max_screen_features=12,
        min_screen_features=5,
        max_terms=7,
        max_interactions=2,
        alpha_grid=(0.0, 1e-4, 1e-3, 1e-2, 1e-1),
        val_fraction=0.2,
        min_rel_improvement=0.004,
        coef_prune_tol=0.035,
        random_state=0,
    ):
        self.max_screen_features = max_screen_features
        self.min_screen_features = min_screen_features
        self.max_terms = max_terms
        self.max_interactions = max_interactions
        self.alpha_grid = alpha_grid
        self.val_fraction = val_fraction
        self.min_rel_improvement = min_rel_improvement
        self.coef_prune_tol = coef_prune_tol
        self.random_state = random_state

    @staticmethod
    def _safe_corr_screen(X, y, k):
        yc = y - np.mean(y)
        Xc = X - np.mean(X, axis=0, keepdims=True)
        denom = np.sqrt(np.sum(Xc**2, axis=0) * np.sum(yc**2)) + 1e-12
        corr = np.abs((Xc.T @ yc) / denom)
        order = np.argsort(corr)[::-1]
        return np.sort(order[:k]).astype(int)

    @staticmethod
    def _ridge_closed_form(Z, y, alpha):
        d = Z.shape[1]
        reg = np.eye(d, dtype=float) * float(alpha)
        reg[0, 0] = 0.0
        lhs = Z.T @ Z + reg
        rhs = Z.T @ y
        try:
            return np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(lhs) @ rhs

    def _term_values(self, X, term):
        kind = term["kind"]
        if kind == "linear":
            return X[:, term["j"]]
        if kind == "hinge_pos":
            j = term["j"]
            knot = term["knot"]
            return np.maximum(0.0, X[:, j] - knot)
        if kind == "hinge_neg":
            j = term["j"]
            knot = term["knot"]
            return np.maximum(0.0, knot - X[:, j])
        if kind == "interaction":
            return X[:, term["j1"]] * X[:, term["j2"]]
        raise ValueError(f"Unknown term kind: {kind}")

    def _build_design(self, X, terms):
        cols = [np.ones(X.shape[0], dtype=float)]
        for term in terms:
            cols.append(self._term_values(X, term))
        return np.column_stack(cols)

    def _candidate_terms(self, X, screened):
        terms = []
        medians = np.median(X[:, screened], axis=0)
        for local_idx, j in enumerate(screened):
            knot = float(medians[local_idx])
            terms.append({"kind": "linear", "j": int(j)})
            terms.append({"kind": "hinge_pos", "j": int(j), "knot": knot})
            terms.append({"kind": "hinge_neg", "j": int(j), "knot": knot})

        top_for_inter = screened[: min(5, len(screened))]
        inter_added = 0
        for a in range(len(top_for_inter)):
            for b in range(a + 1, len(top_for_inter)):
                if inter_added >= self.max_interactions:
                    break
                j1 = int(top_for_inter[a])
                j2 = int(top_for_inter[b])
                terms.append({"kind": "interaction", "j1": j1, "j2": j2})
                inter_added += 1
            if inter_added >= self.max_interactions:
                break
        return terms

    def _fit_eval_alpha(self, Xtr, ytr, Xva, yva, terms, alpha):
        Ztr = self._build_design(Xtr, terms)
        beta = self._ridge_closed_form(Ztr, ytr, alpha)
        Zva = self._build_design(Xva, terms)
        pred = Zva @ beta
        mse = float(np.mean((yva - pred) ** 2))
        return mse, beta

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape
        if p == 0:
            raise ValueError("No features provided")

        self.n_features_in_ = p

        k = max(self.min_screen_features, min(self.max_screen_features, p))
        self.screened_features_ = self._safe_corr_screen(X, y, k)
        candidate_terms = self._candidate_terms(X, self.screened_features_)

        Xtr, Xva, ytr, yva = train_test_split(
            X, y, test_size=min(max(self.val_fraction, 0.1), 0.4), random_state=self.random_state
        )

        selected = []
        remaining = list(range(len(candidate_terms)))
        base_pred = np.full_like(yva, float(np.mean(ytr)), dtype=float)
        best_val_mse = float(np.mean((yva - base_pred) ** 2))

        alpha_choices = [float(a) for a in self.alpha_grid]
        if not alpha_choices:
            alpha_choices = [0.0]

        for _ in range(self.max_terms):
            step_best = None
            for cand_idx in remaining:
                trial_idxs = selected + [cand_idx]
                trial_terms = [candidate_terms[i] for i in trial_idxs]
                for alpha in alpha_choices:
                    mse, _ = self._fit_eval_alpha(Xtr, ytr, Xva, yva, trial_terms, alpha)
                    if step_best is None or mse < step_best["mse"]:
                        step_best = {"mse": mse, "cand_idx": cand_idx, "alpha": alpha}

            if step_best is None:
                break

            rel_gain = (best_val_mse - step_best["mse"]) / (abs(best_val_mse) + 1e-12)
            if rel_gain < self.min_rel_improvement:
                break

            selected.append(step_best["cand_idx"])
            remaining.remove(step_best["cand_idx"])
            best_val_mse = step_best["mse"]

        if not selected:
            self.selected_terms_ = []
            self.alpha_ = 0.0
            self.intercept_ = float(np.mean(y))
            self.term_coefs_ = np.zeros(0, dtype=float)
            self.coef_ = np.zeros(p, dtype=float)
            self.feature_importances_ = np.zeros(p, dtype=float)
            self.active_features_ = np.zeros(0, dtype=int)
            return self

        selected_terms = [candidate_terms[i] for i in selected]

        # Choose alpha for selected structure with 4-fold CV, then refit on all data.
        n_splits = int(min(max(2, 4), len(y)))
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        cv_best = None
        for alpha in alpha_choices:
            fold_mses = []
            for tr, va in kf.split(X):
                Ztr = self._build_design(X[tr], selected_terms)
                beta = self._ridge_closed_form(Ztr, y[tr], alpha)
                Zva = self._build_design(X[va], selected_terms)
                fold_mses.append(float(np.mean((y[va] - (Zva @ beta)) ** 2)))
            mse = float(np.mean(fold_mses))
            if cv_best is None or mse < cv_best["mse"]:
                cv_best = {"alpha": alpha, "mse": mse}

        self.alpha_ = float(cv_best["alpha"]) if cv_best is not None else 0.0
        Zfull = self._build_design(X, selected_terms)
        beta = self._ridge_closed_form(Zfull, y, self.alpha_)

        term_coefs = np.asarray(beta[1:], dtype=float)
        max_abs = float(np.max(np.abs(term_coefs))) if term_coefs.size else 0.0
        if max_abs > 0:
            keep_mask = np.abs(term_coefs) >= self.coef_prune_tol * max_abs
            if not np.any(keep_mask):
                keep_mask[np.argmax(np.abs(term_coefs))] = True
            selected_terms = [t for t, keep in zip(selected_terms, keep_mask) if keep]

            Zpruned = self._build_design(X, selected_terms)
            beta = self._ridge_closed_form(Zpruned, y, self.alpha_)
            term_coefs = np.asarray(beta[1:], dtype=float)

        self.selected_terms_ = selected_terms
        self.intercept_ = float(beta[0])
        self.term_coefs_ = term_coefs

        linear_coef = np.zeros(p, dtype=float)
        feat_scores = np.zeros(p, dtype=float)
        for coef, term in zip(self.term_coefs_, self.selected_terms_):
            c = float(coef)
            kind = term["kind"]
            if kind == "linear":
                j = int(term["j"])
                linear_coef[j] += c
                feat_scores[j] += abs(c)
            elif kind in ("hinge_pos", "hinge_neg"):
                j = int(term["j"])
                feat_scores[j] += abs(c)
            elif kind == "interaction":
                j1 = int(term["j1"])
                j2 = int(term["j2"])
                feat_scores[j1] += 0.5 * abs(c)
                feat_scores[j2] += 0.5 * abs(c)

        self.coef_ = linear_coef
        max_feat = float(np.max(feat_scores)) if np.max(feat_scores) > 0 else 0.0
        self.feature_importances_ = feat_scores / max_feat if max_feat > 0 else feat_scores
        tol = max(1e-10, 0.08 * max_feat) if max_feat > 0 else 1e-10
        self.active_features_ = np.flatnonzero(feat_scores >= tol).astype(int)

        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "selected_terms_", "term_coefs_", "active_features_"])
        X = np.asarray(X, dtype=float)
        yhat = np.full(X.shape[0], self.intercept_, dtype=float)
        for coef, term in zip(self.term_coefs_, self.selected_terms_):
            yhat += float(coef) * self._term_values(X, term)
        return yhat

    @staticmethod
    def _term_to_str(term):
        kind = term["kind"]
        if kind == "linear":
            return f"x{int(term['j'])}"
        if kind == "hinge_pos":
            return f"max(0, x{int(term['j'])} - {float(term['knot']):.6f})"
        if kind == "hinge_neg":
            return f"max(0, {float(term['knot']):.6f} - x{int(term['j'])})"
        if kind == "interaction":
            return f"x{int(term['j1'])} * x{int(term['j2'])}"
        return "unknown_term"

    def __str__(self):
        check_is_fitted(
            self,
            [
                "intercept_",
                "selected_terms_",
                "term_coefs_",
                "alpha_",
                "screened_features_",
                "active_features_",
            ],
        )

        lines = [
            "Greedy Median-Hinge Sparse Regressor",
            "Prediction equation:",
            f"y = {self.intercept_:+.6f}",
        ]

        for c, term in zip(self.term_coefs_, self.selected_terms_):
            lines.append(f"    {c:+.6f} * {self._term_to_str(term)}")

        lines.append("")
        lines.append(f"ridge_alpha = {self.alpha_:.6g}")
        lines.append(
            f"screened_features = {', '.join(f'x{int(j)}' for j in self.screened_features_) if self.screened_features_.size else 'none'}"
        )
        lines.append(
            f"active_features = {', '.join(f'x{int(j)}' for j in self.active_features_) if self.active_features_.size else 'none'}"
        )
        lines.append("")
        lines.append("Simulation recipe: plug feature values into each listed term, multiply by its coefficient, and add all terms to the intercept.")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys

_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])

# Update the model shorthand name and description below to reflect the class above and any changes you make.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
GreedyMedianHingeRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "GreedyMedianHinge_v1"
model_description = "Forward-selected sparse raw-feature equation using linear + median-hinge + tiny interaction candidates with CV ridge refit"
model_defs = [(model_shorthand_name, GreedyMedianHingeRegressor())]


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
