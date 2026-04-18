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
from itertools import combinations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class TeacherStudentCompactDistillRegressor(BaseEstimator, RegressorMixin):
    """
    Two-path regressor:
    - Batch path (> symbolic_n_rows): blended teacher (GBM + ridge).
    - Single-row path (<= symbolic_n_rows): compact sparse symbolic equation.
    """

    def __init__(
        self,
        teacher_n_estimators=180,
        teacher_learning_rate=0.05,
        teacher_max_depth=3,
        teacher_subsample=0.85,
        val_fraction=0.2,
        max_student_features=8,
        max_student_terms=4,
        alpha_student=2e-3,
        corr_screen_rel=0.04,
        hinge_quantiles=(0.25, 0.5, 0.75),
        interaction_top_features=3,
        candidate_eval_topk=12,
        complexity_penalty=0.015,
        min_rel_gain=0.004,
        coef_decimals=3,
        coef_tol=1e-8,
        symbolic_n_rows=1,
        random_state=0,
    ):
        self.teacher_n_estimators = teacher_n_estimators
        self.teacher_learning_rate = teacher_learning_rate
        self.teacher_max_depth = teacher_max_depth
        self.teacher_subsample = teacher_subsample
        self.val_fraction = val_fraction
        self.max_student_features = max_student_features
        self.max_student_terms = max_student_terms
        self.alpha_student = alpha_student
        self.corr_screen_rel = corr_screen_rel
        self.hinge_quantiles = hinge_quantiles
        self.interaction_top_features = interaction_top_features
        self.candidate_eval_topk = candidate_eval_topk
        self.complexity_penalty = complexity_penalty
        self.min_rel_gain = min_rel_gain
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
        if n_samples <= 80:
            idx = np.arange(n_samples, dtype=int)
            return idx, idx
        rng = np.random.RandomState(self.random_state)
        order = rng.permutation(n_samples)
        n_val = int(round(float(self.val_fraction) * n_samples))
        n_val = min(max(n_val, 24), n_samples - 24)
        return order[n_val:], order[:n_val]

    def _fit_teacher(self, X, y):
        tr_idx, va_idx = self._split_indices(X.shape[0])
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xva, yva = X[va_idx], y[va_idx]

        self.teacher_gbm_ = GradientBoostingRegressor(
            n_estimators=int(self.teacher_n_estimators),
            learning_rate=float(self.teacher_learning_rate),
            max_depth=int(self.teacher_max_depth),
            subsample=float(self.teacher_subsample),
            random_state=int(self.random_state),
        )
        self.teacher_ridge_ = Ridge(alpha=1.0)
        self.teacher_gbm_.fit(Xtr, ytr)
        self.teacher_ridge_.fit(Xtr, ytr)

        if len(Xva) == 0:
            w = 1.0
        else:
            p_gbm = self.teacher_gbm_.predict(Xva)
            p_ridge = self.teacher_ridge_.predict(Xva)
            diff = p_gbm - p_ridge
            denom = float(diff @ diff) + 1e-12
            w = float((diff @ (yva - p_ridge)) / denom)
            if not np.isfinite(w):
                w = 1.0
            w = float(np.clip(w, 0.0, 1.0))

        self.teacher_weight_ = w
        self.teacher_name_ = (
            f"Blend[{w:.2f}*GBM + {1.0 - w:.2f}*Ridge], "
            f"GBM(n_estimators={int(self.teacher_n_estimators)}, "
            f"lr={float(self.teacher_learning_rate):.3f}, "
            f"depth={int(self.teacher_max_depth)})"
        )

    @staticmethod
    def _term_features(term):
        t = term["type"]
        if t in {"lin", "sq", "abs", "step", "hinge"}:
            return {int(term["feature"])}
        if t == "int":
            return {int(term["a"]), int(term["b"])}
        return set()

    @staticmethod
    def _term_complexity(term):
        t = term["type"]
        if t in {"lin", "step"}:
            return 1
        if t in {"sq", "abs", "int"}:
            return 2
        if t == "hinge":
            return 3
        return 2

    @staticmethod
    def _eval_term(X, term):
        t = term["type"]
        if t == "lin":
            return X[:, int(term["feature"])]
        if t == "sq":
            col = X[:, int(term["feature"])]
            return col * col
        if t == "abs":
            return np.abs(X[:, int(term["feature"])])
        if t == "step":
            feat = int(term["feature"])
            knot = float(term["knot"])
            return (X[:, feat] > knot).astype(float)
        if t == "hinge":
            feat = int(term["feature"])
            knot = float(term["knot"])
            if int(term["direction"]) > 0:
                return np.maximum(0.0, X[:, feat] - knot)
            return np.maximum(0.0, knot - X[:, feat])
        if t == "int":
            return X[:, int(term["a"])] * X[:, int(term["b"])]
        raise ValueError(f"Unknown term type: {t}")

    @staticmethod
    def _dedupe_terms(terms):
        out = []
        seen = set()
        for t in terms:
            if t["type"] in {"lin", "sq", "abs"}:
                key = (t["type"], int(t["feature"]))
            elif t["type"] in {"step", "hinge"}:
                key = (
                    t["type"],
                    int(t["feature"]),
                    round(float(t["knot"]), 6),
                    int(t.get("direction", 0)),
                )
            elif t["type"] == "int":
                a, b = sorted((int(t["a"]), int(t["b"])))
                key = ("int", a, b)
            else:
                key = tuple(sorted(t.items()))
            if key not in seen:
                seen.add(key)
                out.append(t)
        return out

    def _screen_features(self, X, y):
        n_features = X.shape[1]
        if n_features == 0:
            return []

        yc = y - float(np.mean(y))
        y_norm = float(np.linalg.norm(yc)) + 1e-12
        scores = np.zeros(n_features, dtype=float)

        for j in range(n_features):
            x = X[:, j].astype(float)
            transforms = [
                x,
                x * x,
                np.abs(x),
                np.maximum(0.0, x),
                (x > 0.0).astype(float),
            ]
            best = 0.0
            for col in transforms:
                cc = col - float(np.mean(col))
                denom = (float(np.linalg.norm(cc)) + 1e-12) * y_norm
                corr = abs(float(cc @ yc) / denom)
                if corr > best:
                    best = corr
            scores[j] = best

        order = [int(i) for i in np.argsort(scores)[::-1]]
        if not order:
            return [0]

        max_score = float(scores[order[0]])
        if max_score <= 1e-12:
            return list(range(min(int(self.max_student_features), n_features)))

        selected = []
        for j in order:
            if scores[j] >= float(self.corr_screen_rel) * max_score:
                selected.append(int(j))
            if len(selected) >= int(self.max_student_features):
                break
        if not selected:
            selected = [int(order[0])]
        return selected

    def _build_candidates(self, X, screened):
        if not screened:
            return []
        terms = []
        qvals = np.asarray(self.hinge_quantiles, dtype=float)
        for feat in screened:
            feat = int(feat)
            xcol = X[:, feat]
            terms.append({"type": "lin", "feature": feat})
            terms.append({"type": "sq", "feature": feat})
            terms.append({"type": "abs", "feature": feat})
            knots = [0.0, float(np.median(xcol))]
            if qvals.size > 0:
                knots.extend(np.quantile(xcol, qvals).tolist())
            for knot in np.unique(np.asarray(knots, dtype=float)):
                if not np.isfinite(knot):
                    continue
                terms.append({"type": "step", "feature": feat, "knot": float(knot)})
                terms.append({"type": "hinge", "feature": feat, "knot": float(knot), "direction": 1})
                terms.append({"type": "hinge", "feature": feat, "knot": float(knot), "direction": -1})

        inter_feats = screened[: max(2, min(len(screened), int(self.interaction_top_features)))]
        for a, b in combinations(inter_feats, 2):
            terms.append({"type": "int", "a": int(a), "b": int(b)})
        return self._dedupe_terms(terms)

    def _design_matrix(self, X, terms):
        if not terms:
            return np.zeros((X.shape[0], 0), dtype=float)
        cols = [self._eval_term(X, t) for t in terms]
        return np.column_stack(cols).astype(float)

    def _fit_student(self, X, y):
        tr_idx, va_idx = self._split_indices(X.shape[0])
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xva, yva = X[va_idx], y[va_idx]

        screened = self._screen_features(Xtr, ytr)
        candidates = self._build_candidates(Xtr, screened)
        if not candidates:
            return {
                "intercept": float(np.mean(y)),
                "terms": [],
                "validation_rmse": self._rmse(yva, np.repeat(float(np.mean(ytr)), len(yva))),
            }

        Dtr = self._design_matrix(Xtr, candidates)
        Dva = self._design_matrix(Xva, candidates)

        pred_tr = np.repeat(float(np.mean(ytr)), len(ytr))
        pred_va = np.repeat(float(np.mean(ytr)), len(yva))
        selected = []
        best_score = self._rmse(yva, pred_va)

        for _ in range(int(self.max_student_terms)):
            residual = ytr - pred_tr
            corr_scores = []
            for j in range(Dtr.shape[1]):
                if j in selected:
                    continue
                col = Dtr[:, j]
                denom = float(np.linalg.norm(col)) + 1e-12
                corr_scores.append((abs(float(col @ residual)) / denom, j))
            if not corr_scores:
                break
            corr_scores.sort(reverse=True)
            top = [j for _, j in corr_scores[: max(1, int(self.candidate_eval_topk))]]

            candidate_best = None
            for j in top:
                idx = selected + [int(j)]
                inter, coef = self._solve_ridge_with_intercept(
                    Dtr[:, idx], ytr, alpha=float(self.alpha_student)
                )
                pred_va_trial = inter + Dva[:, idx] @ coef
                rmse_va = self._rmse(yva, pred_va_trial)
                ops = sum(self._term_complexity(candidates[k]) for k in idx)
                score = rmse_va + float(self.complexity_penalty) * float(ops)
                if candidate_best is None or score < candidate_best["score"]:
                    candidate_best = {"j": int(j), "score": float(score)}

            if candidate_best is None:
                break

            rel_gain = (best_score - float(candidate_best["score"])) / max(best_score, 1e-12)
            if rel_gain < float(self.min_rel_gain):
                break

            selected.append(int(candidate_best["j"]))
            inter, coef = self._solve_ridge_with_intercept(
                Dtr[:, selected], ytr, alpha=float(self.alpha_student)
            )
            pred_tr = inter + Dtr[:, selected] @ coef
            pred_va = inter + Dva[:, selected] @ coef
            ops = sum(self._term_complexity(candidates[k]) for k in selected)
            best_score = self._rmse(yva, pred_va) + float(self.complexity_penalty) * float(ops)

        if not selected:
            return {
                "intercept": float(np.mean(y)),
                "terms": [],
                "validation_rmse": self._rmse(yva, np.repeat(float(np.mean(ytr)), len(yva))),
            }

        selected_terms = [candidates[j] for j in selected]
        Dfull = self._design_matrix(X, selected_terms)
        inter_full, coef_full = self._solve_ridge_with_intercept(
            Dfull, y, alpha=float(self.alpha_student)
        )

        dec = int(self.coef_decimals)
        intercept = float(np.round(inter_full, dec))
        terms = []
        for t, c in zip(selected_terms, coef_full):
            coef = float(np.round(float(c), dec))
            if abs(coef) <= float(self.coef_tol):
                continue
            item = dict(t)
            item["coef"] = coef
            terms.append(item)

        terms.sort(key=lambda t: abs(float(t["coef"])), reverse=True)
        va_rmse = self._rmse(yva, intercept + self._design_matrix(Xva, terms) @ np.asarray([t["coef"] for t in terms], dtype=float)) if terms else self._rmse(yva, np.repeat(intercept, len(yva)))
        return {"intercept": intercept, "terms": terms, "validation_rmse": float(va_rmse)}

    def _predict_student(self, X):
        pred = np.repeat(float(self.student_intercept_), X.shape[0]).astype(float)
        if self.student_terms_:
            D = self._design_matrix(X, self.student_terms_)
            coef = np.asarray([float(t["coef"]) for t in self.student_terms_], dtype=float)
            pred = pred + D @ coef
        return pred

    def _predict_teacher(self, X):
        p_gbm = self.teacher_gbm_.predict(X)
        p_ridge = self.teacher_ridge_.predict(X)
        w = float(self.teacher_weight_)
        return w * p_gbm + (1.0 - w) * p_ridge

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
        self.selected_features_ = sorted(int(i) for i in np.where(self.feature_importance_ > float(self.coef_tol))[0])
        self.student_operation_count_ = sum(self._term_complexity(t) for t in self.student_terms_)
        return self

    def predict(self, X):
        check_is_fitted(
            self,
            [
                "teacher_gbm_",
                "teacher_ridge_",
                "teacher_weight_",
                "student_intercept_",
                "student_terms_",
                "feature_importance_",
            ],
        )
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[0] <= int(self.symbolic_n_rows):
            return self._predict_student(X)
        return self._predict_teacher(X)

    def _term_text(self, term, dec):
        t = term["type"]
        if t == "lin":
            return f"x{int(term['feature'])}"
        if t == "sq":
            return f"(x{int(term['feature'])}^2)"
        if t == "abs":
            return f"abs(x{int(term['feature'])})"
        if t == "step":
            return f"1[x{int(term['feature'])} > {float(term['knot']):.{dec}f}]"
        if t == "hinge":
            feat = int(term["feature"])
            knot = float(term["knot"])
            if int(term["direction"]) > 0:
                return f"max(0, x{feat} - {knot:.{dec}f})"
            return f"max(0, {knot:.{dec}f} - x{feat})"
        if t == "int":
            return f"(x{int(term['a'])} * x{int(term['b'])})"
        return "term"

    def __str__(self):
        check_is_fitted(
            self,
            [
                "student_intercept_",
                "student_terms_",
                "selected_features_",
                "feature_importance_",
                "student_operation_count_",
            ],
        )
        dec = int(self.coef_decimals)
        terms_sorted = sorted(self.student_terms_, key=lambda t: abs(float(t["coef"])), reverse=True)
        eq_parts = [f"{self.student_intercept_:+.{dec}f}"]
        for t in terms_sorted:
            eq_parts.append(f"({float(t['coef']):+.{dec}f})*{self._term_text(t, dec)}")

        lines = [
            "Teacher-Student Compact Distillation Regressor",
            "Single-row symbolic rule for manual simulation (exact):",
            "  y = " + " + ".join(eq_parts),
            "",
            "Usage notes:",
            "  - If a feature is not shown in the equation, its effect is 0.",
            "  - If a question omits a feature value, treat that feature value as 0.",
            "  - For change questions, compute prediction(new) - prediction(old).",
            "",
            f"Active symbolic terms ({len(terms_sorted)} total):",
        ]
        if terms_sorted:
            for i, t in enumerate(terms_sorted, 1):
                lines.append(f"  t{i}: {float(t['coef']):+.{dec}f} * {self._term_text(t, dec)}")
        else:
            lines.append("  (none)")

        feat_rank = sorted(
            [(i, v) for i, v in enumerate(self.feature_importance_) if v > float(self.coef_tol)],
            key=lambda z: z[1],
            reverse=True,
        )
        lines.append("")
        lines.append("Feature influence ranking (sum of |coefficients| over touching terms):")
        if feat_rank:
            for i, v in feat_rank:
                lines.append(f"  x{i}: {float(v):.{dec}f}")
        else:
            lines.append("  (none)")

        active = sorted(set(int(i) for i in self.selected_features_))
        inactive = [f"x{i}" for i in range(self.n_features_in_) if i not in active]
        lines.append("")
        lines.append("Active features: " + (", ".join(f"x{i}" for i in active) if active else "(none)"))
        lines.append("Zero-effect features: " + (", ".join(inactive) if inactive else "(none)"))
        lines.append(
            f"Estimated arithmetic operations for one-row simulation: {int(self.student_operation_count_)}"
        )
        lines.append(f"Large-batch predictor: {self.teacher_name_}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
TeacherStudentCompactDistillRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "TeacherStudentCompactDistill_v1"
model_description = "Blended GBM+ridge batch teacher with complexity-penalized compact symbolic student (linear/square/abs/hinge/step/interaction) for single-row simulatability"
model_defs = [(model_shorthand_name, TeacherStudentCompactDistillRegressor())]


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
