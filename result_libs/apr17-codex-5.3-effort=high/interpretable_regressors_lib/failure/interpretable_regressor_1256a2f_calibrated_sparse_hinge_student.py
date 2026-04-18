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
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class CalibratedSparseHingeStudentRegressor(BaseEstimator, RegressorMixin):
    """
    Dual-path regressor:
    - Batch path (> symbolic_n_rows): calibrated blend of HGB + RF + ridge.
    - Single-row path (<= symbolic_n_rows): sparse linear rule with optional hinge.
    """

    def __init__(
        self,
        teacher_max_iter=260,
        teacher_learning_rate=0.06,
        teacher_max_depth=6,
        teacher_min_samples_leaf=8,
        teacher_l2=0.03,
        teacher_rf_estimators=80,
        teacher_rf_max_depth=12,
        teacher_rf_min_samples_leaf=3,
        teacher_ridge_alpha=1.0,
        val_fraction=0.2,
        student_max_linear_terms=6,
        student_alpha=3e-3,
        hinge_quantiles=(0.2, 0.35, 0.5, 0.65, 0.8),
        hinge_top_features=3,
        min_hinge_gain=0.01,
        coef_decimals=3,
        coef_tol=1e-8,
        symbolic_n_rows=1,
        random_state=0,
    ):
        self.teacher_max_iter = teacher_max_iter
        self.teacher_learning_rate = teacher_learning_rate
        self.teacher_max_depth = teacher_max_depth
        self.teacher_min_samples_leaf = teacher_min_samples_leaf
        self.teacher_l2 = teacher_l2
        self.teacher_rf_estimators = teacher_rf_estimators
        self.teacher_rf_max_depth = teacher_rf_max_depth
        self.teacher_rf_min_samples_leaf = teacher_rf_min_samples_leaf
        self.teacher_ridge_alpha = teacher_ridge_alpha
        self.val_fraction = val_fraction
        self.student_max_linear_terms = student_max_linear_terms
        self.student_alpha = student_alpha
        self.hinge_quantiles = hinge_quantiles
        self.hinge_top_features = hinge_top_features
        self.min_hinge_gain = min_hinge_gain
        self.coef_decimals = coef_decimals
        self.coef_tol = coef_tol
        self.symbolic_n_rows = symbolic_n_rows
        self.random_state = random_state

    @staticmethod
    def _rmse(y_true, y_pred):
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    @staticmethod
    def _solve_ridge_with_intercept(D, y, alpha):
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
        return float(sol[0]), np.asarray(sol[1:], dtype=float) / scale

    def _split_indices(self, n_samples):
        if n_samples <= 100:
            idx = np.arange(n_samples, dtype=int)
            return idx, idx
        rng = np.random.RandomState(self.random_state)
        order = rng.permutation(n_samples)
        n_val = int(round(float(self.val_fraction) * n_samples))
        n_val = min(max(n_val, 30), n_samples - 30)
        return order[n_val:], order[:n_val]

    def _fit_teacher(self, X, y):
        tr_idx, va_idx = self._split_indices(X.shape[0])
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xva, yva = X[va_idx], y[va_idx]

        self.teacher_hgb_ = HistGradientBoostingRegressor(
            max_iter=int(self.teacher_max_iter),
            learning_rate=float(self.teacher_learning_rate),
            max_depth=int(self.teacher_max_depth),
            min_samples_leaf=int(self.teacher_min_samples_leaf),
            l2_regularization=float(self.teacher_l2),
            random_state=int(self.random_state),
        )
        self.teacher_rf_ = RandomForestRegressor(
            n_estimators=int(self.teacher_rf_estimators),
            max_depth=int(self.teacher_rf_max_depth),
            min_samples_leaf=int(self.teacher_rf_min_samples_leaf),
            random_state=int(self.random_state),
            n_jobs=1,
        )
        self.teacher_ridge_ = Ridge(alpha=float(self.teacher_ridge_alpha))

        self.teacher_hgb_.fit(Xtr, ytr)
        self.teacher_rf_.fit(Xtr, ytr)
        self.teacher_ridge_.fit(Xtr, ytr)

        p_hgb = self.teacher_hgb_.predict(Xva)
        p_rf = self.teacher_rf_.predict(Xva)
        p_ridge = self.teacher_ridge_.predict(Xva)
        P = np.column_stack([p_hgb, p_rf, p_ridge])

        best_rmse = float("inf")
        best_w = np.array([0.7, 0.2, 0.1], dtype=float)
        grid = np.linspace(0.0, 1.0, 21)
        for w0 in grid:
            for w1 in grid:
                if w0 + w1 > 1.0:
                    continue
                w2 = 1.0 - w0 - w1
                w = np.array([w0, w1, w2], dtype=float)
                pred = P @ w
                rmse = self._rmse(yva, pred)
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_w = w
        self.teacher_weights_ = best_w

        self.teacher_hgb_.fit(X, y)
        self.teacher_rf_.fit(X, y)
        self.teacher_ridge_.fit(X, y)
        self.teacher_name_ = (
            f"Blend[{best_w[0]:.2f}*HGB + {best_w[1]:.2f}*RF + {best_w[2]:.2f}*Ridge]"
        )

    def _predict_teacher(self, X):
        w = self.teacher_weights_
        return (
            w[0] * self.teacher_hgb_.predict(X)
            + w[1] * self.teacher_rf_.predict(X)
            + w[2] * self.teacher_ridge_.predict(X)
        )

    @staticmethod
    def _feature_scores(X, target):
        t = target - float(np.mean(target))
        tnorm = float(np.linalg.norm(t)) + 1e-12
        scores = np.zeros(X.shape[1], dtype=float)
        for j in range(X.shape[1]):
            x = X[:, j]
            xc = x - float(np.mean(x))
            denom = (float(np.linalg.norm(xc)) + 1e-12) * tnorm
            scores[j] = abs(float(xc @ t) / denom)
        return scores

    @staticmethod
    def _hinge_col(x, knot, direction):
        if int(direction) > 0:
            return np.maximum(0.0, x - float(knot))
        return np.maximum(0.0, float(knot) - x)

    @staticmethod
    def _eval_term(X, term):
        if term["type"] == "lin":
            return X[:, int(term["feature"])]
        if term["type"] == "hinge":
            f = int(term["feature"])
            return CalibratedSparseHingeStudentRegressor._hinge_col(
                X[:, f], float(term["knot"]), int(term["direction"])
            )
        raise ValueError(f"Unknown term type: {term['type']}")

    def _design_matrix(self, X, terms):
        if not terms:
            return np.zeros((X.shape[0], 0), dtype=float)
        return np.column_stack([self._eval_term(X, t) for t in terms]).astype(float)

    def _fit_student(self, X, y):
        teacher_target = self._predict_teacher(X)
        tr_idx, va_idx = self._split_indices(X.shape[0])
        Xtr, ztr = X[tr_idx], teacher_target[tr_idx]
        Xva, zva = X[va_idx], teacher_target[va_idx]

        scores = self._feature_scores(Xtr, ztr)
        order = [int(i) for i in np.argsort(scores)[::-1]]
        n_lin = max(1, min(int(self.student_max_linear_terms), X.shape[1]))
        linear_features = order[:n_lin]
        base_terms = [{"type": "lin", "feature": int(j)} for j in linear_features]

        Dtr_base = self._design_matrix(Xtr, base_terms)
        Dva_base = self._design_matrix(Xva, base_terms)
        inter_base, coef_base = self._solve_ridge_with_intercept(
            Dtr_base, ztr, alpha=float(self.student_alpha)
        )
        pred_va_base = inter_base + Dva_base @ coef_base
        best_rmse = self._rmse(zva, pred_va_base)
        best_terms = list(base_terms)
        best_inter = float(inter_base)
        best_coef = np.asarray(coef_base, dtype=float)

        hinge_candidates = order[: max(1, min(len(order), int(self.hinge_top_features)))]
        qvals = np.asarray(self.hinge_quantiles, dtype=float)
        for feat in hinge_candidates:
            xcol = Xtr[:, int(feat)]
            knots = np.unique(np.quantile(xcol, qvals))
            knots = [float(k) for k in knots if np.isfinite(k)]
            if not knots:
                continue
            for knot in knots:
                for direction in (1, -1):
                    hterm = {
                        "type": "hinge",
                        "feature": int(feat),
                        "knot": float(knot),
                        "direction": int(direction),
                    }
                    terms = base_terms + [hterm]
                    Dtr = self._design_matrix(Xtr, terms)
                    Dva = self._design_matrix(Xva, terms)
                    inter, coef = self._solve_ridge_with_intercept(
                        Dtr, ztr, alpha=float(self.student_alpha)
                    )
                    rmse = self._rmse(zva, inter + Dva @ coef)
                    rel_gain = (best_rmse - rmse) / max(best_rmse, 1e-12)
                    if rel_gain > float(self.min_hinge_gain):
                        best_rmse = rmse
                        best_terms = terms
                        best_inter = float(inter)
                        best_coef = np.asarray(coef, dtype=float)

        Dfull = self._design_matrix(X, best_terms)
        inter_full, coef_full = self._solve_ridge_with_intercept(
            Dfull, teacher_target, alpha=float(self.student_alpha)
        )
        dec = int(self.coef_decimals)
        intercept = float(np.round(inter_full, dec))
        terms = []
        for t, c in zip(best_terms, coef_full):
            coef = float(np.round(float(c), dec))
            if abs(coef) <= float(self.coef_tol):
                continue
            item = dict(t)
            item["coef"] = coef
            terms.append(item)
        if not terms:
            terms = [{"type": "lin", "feature": 0, "coef": 0.0}]

        pred_val = (
            intercept
            + self._design_matrix(Xva, terms)
            @ np.asarray([float(t["coef"]) for t in terms], dtype=float)
        )
        return {
            "intercept": intercept,
            "terms": terms,
            "validation_rmse": self._rmse(zva, pred_val),
        }

    def _predict_student(self, X):
        pred = np.repeat(float(self.student_intercept_), X.shape[0]).astype(float)
        if self.student_terms_:
            D = self._design_matrix(X, self.student_terms_)
            coef = np.asarray([float(t["coef"]) for t in self.student_terms_], dtype=float)
            pred = pred + D @ coef
        return pred

    @staticmethod
    def _term_features(term):
        if term["type"] == "lin":
            return {int(term["feature"])}
        if term["type"] == "hinge":
            return {int(term["feature"])}
        return set()

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        self.n_features_in_ = int(X.shape[1])

        self._fit_teacher(X, y)
        student = self._fit_student(X, y)
        self.student_intercept_ = float(student["intercept"])
        self.student_terms_ = list(student["terms"])
        self.student_validation_rmse_ = float(student["validation_rmse"])

        self.coef_ = np.zeros(self.n_features_in_, dtype=float)
        self.feature_importance_ = np.zeros(self.n_features_in_, dtype=float)
        for term in self.student_terms_:
            c = abs(float(term["coef"]))
            for feat in self._term_features(term):
                self.feature_importance_[int(feat)] += c
            if term["type"] == "lin":
                self.coef_[int(term["feature"])] += float(term["coef"])
        self.selected_features_ = sorted(
            int(i) for i in np.where(self.feature_importance_ > float(self.coef_tol))[0]
        )
        return self

    def predict(self, X):
        check_is_fitted(
            self,
            [
                "teacher_hgb_",
                "teacher_rf_",
                "teacher_ridge_",
                "teacher_weights_",
                "student_intercept_",
                "student_terms_",
            ],
        )
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[0] <= int(self.symbolic_n_rows):
            return self._predict_student(X)
        return self._predict_teacher(X)

    def _term_text(self, term, dec):
        if term["type"] == "lin":
            return f"x{int(term['feature'])}"
        feat = int(term["feature"])
        knot = float(term["knot"])
        if int(term["direction"]) > 0:
            return f"max(0, x{feat} - {knot:.{dec}f})"
        return f"max(0, {knot:.{dec}f} - x{feat})"

    def __str__(self):
        check_is_fitted(
            self,
            ["student_intercept_", "student_terms_", "selected_features_", "feature_importance_"],
        )
        dec = int(self.coef_decimals)
        terms_sorted = sorted(self.student_terms_, key=lambda t: abs(float(t["coef"])), reverse=True)

        eq = [f"{self.student_intercept_:+.{dec}f}"]
        for t in terms_sorted:
            eq.append(f"({float(t['coef']):+.{dec}f})*{self._term_text(t, dec)}")

        active = sorted(set(int(i) for i in self.selected_features_))
        inactive = [f"x{i}" for i in range(self.n_features_in_) if i not in active]

        lines = [
            "Calibrated Sparse Hinge Student Regressor",
            "Single-row exact computation rule:",
            "  y = " + " + ".join(eq),
            "Computation notes:",
            "  - If a feature is not present in the equation, its effect is exactly 0.",
            "  - If a query omits feature values, treat omitted features as 0.",
            "  - For change questions, compute prediction(new) - prediction(old).",
            "  - Return only the final numeric prediction when asked for a number.",
            "",
            f"Active features: {', '.join(f'x{i}' for i in active) if active else '(none)'}",
            f"Zero-effect features: {', '.join(inactive) if inactive else '(none)'}",
            f"Batch predictor (for multi-row inputs): {self.teacher_name_}",
        ]
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
CalibratedSparseHingeStudentRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "CalibratedSparseHingeStudent_v1"
model_description = "Validation-calibrated HGB/RF/Ridge batch blend with sparse linear-plus-one-hinge distilled single-row equation"
model_defs = [(model_shorthand_name, CalibratedSparseHingeStudentRegressor())]


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
