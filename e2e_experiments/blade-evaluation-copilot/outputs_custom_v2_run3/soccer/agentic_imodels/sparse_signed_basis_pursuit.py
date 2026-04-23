"""sparse_signed_basis_pursuit — SparseSignedBasisPursuitRegressor from the agentic-imodels library.

Generated from: result_libs/apr17-codex-5.3-effort=high/interpretable_regressors_lib/success/interpretable_regressor_029630d_sparse_signed_basis_pursuit.py

Shorthand: SparseSignedBasisPursuit_v1
Mean global rank (lower is better): 272.70   (pooled 65 dev datasets)
Interpretability (fraction passed, higher is better):
    dev  (43 tests):  0.674
    test (157 tests): 0.758
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


# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class SparseSignedBasisPursuitRegressor(BaseEstimator, RegressorMixin):
    """
    Interpretable sparse dictionary regressor.

    The model uses forward selection over a compact basis dictionary:
      - centered linear terms
      - positive/negative hinges at feature medians
      - centered quadratic terms
      - a few centered pairwise interactions
    Then it refits selected terms with ridge and light coefficient rounding.
    """

    def __init__(
        self,
        max_terms=10,
        min_terms=2,
        val_fraction=0.2,
        alphas=(0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0),
        nonlin_top_features=10,
        interaction_top_features=6,
        round_decimals_grid=(6, 5, 4, 3, 2),
        complexity_penalty=0.01,
        coef_tol=1e-10,
        improvement_tol=1e-6,
        random_state=0,
    ):
        self.max_terms = max_terms
        self.min_terms = min_terms
        self.val_fraction = val_fraction
        self.alphas = alphas
        self.nonlin_top_features = nonlin_top_features
        self.interaction_top_features = interaction_top_features
        self.round_decimals_grid = round_decimals_grid
        self.complexity_penalty = complexity_penalty
        self.coef_tol = coef_tol
        self.improvement_tol = improvement_tol
        self.random_state = random_state

    @staticmethod
    def _rmse(y_true, y_pred):
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    @staticmethod
    def _ridge_with_intercept(D, y, alpha):
        n = D.shape[0]
        if D.shape[1] == 0:
            return float(np.mean(y)), np.zeros(0, dtype=float)
        A = np.column_stack([np.ones(n, dtype=float), D])
        reg = np.eye(A.shape[1], dtype=float)
        reg[0, 0] = 0.0
        lhs = A.T @ A + float(max(alpha, 0.0)) * reg
        rhs = A.T @ y
        try:
            sol = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        return float(sol[0]), np.asarray(sol[1:], dtype=float)

    def _split_indices(self, n):
        if n <= 40:
            idx = np.arange(n, dtype=int)
            return idx, idx
        rng = np.random.RandomState(self.random_state)
        order = rng.permutation(n)
        n_val = int(round(float(self.val_fraction) * n))
        n_val = min(max(n_val, 20), n - 20)
        val_idx = order[:n_val]
        tr_idx = order[n_val:]
        if tr_idx.size == 0:
            tr_idx = val_idx
        return tr_idx, val_idx

    def _eval_spec(self, X, spec):
        kind = spec[0]
        if kind == "lin":
            j = spec[1]
            c = self.feature_centers_[j]
            return X[:, j] - c
        if kind == "plus":
            j = spec[1]
            c = self.feature_centers_[j]
            return np.maximum(0.0, X[:, j] - c)
        if kind == "minus":
            j = spec[1]
            c = self.feature_centers_[j]
            return np.maximum(0.0, c - X[:, j])
        if kind == "sq":
            j = spec[1]
            c = self.feature_centers_[j]
            v = X[:, j] - c
            return v * v
        if kind == "int":
            i, j = spec[1], spec[2]
            ci = self.feature_centers_[i]
            cj = self.feature_centers_[j]
            return (X[:, i] - ci) * (X[:, j] - cj)
        raise ValueError(f"Unknown basis spec: {spec}")

    def _format_term(self, spec, decimals):
        kind = spec[0]
        if kind == "lin":
            j = spec[1]
            c = self.feature_centers_[j]
            return f"(x{j} - {c:.{decimals}f})"
        if kind == "plus":
            j = spec[1]
            c = self.feature_centers_[j]
            return f"max(0, x{j} - {c:.{decimals}f})"
        if kind == "minus":
            j = spec[1]
            c = self.feature_centers_[j]
            return f"max(0, {c:.{decimals}f} - x{j})"
        if kind == "sq":
            j = spec[1]
            c = self.feature_centers_[j]
            return f"(x{j} - {c:.{decimals}f})^2"
        i, j = spec[1], spec[2]
        ci = self.feature_centers_[i]
        cj = self.feature_centers_[j]
        return f"(x{i} - {ci:.{decimals}f})*(x{j} - {cj:.{decimals}f})"

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        self.feature_centers_ = np.median(X, axis=0).astype(float)

        tr_idx, va_idx = self._split_indices(n_samples)
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xva, yva = X[va_idx], y[va_idx]

        # Candidate set: all linear terms, plus nonlinear terms on most correlated features.
        xc = Xtr - self.feature_centers_
        y_center = ytr - float(np.mean(ytr))
        denom = np.sqrt(np.sum(xc * xc, axis=0)) + 1e-12
        corr = np.abs((xc.T @ y_center) / denom)
        feature_order = np.argsort(corr)[::-1]
        top_nonlin = feature_order[: min(int(self.nonlin_top_features), n_features)]
        top_inter = feature_order[: min(int(self.interaction_top_features), n_features)]

        specs = [("lin", j) for j in range(n_features)]
        for j in top_nonlin:
            specs.append(("plus", int(j)))
            specs.append(("minus", int(j)))
            specs.append(("sq", int(j)))
        for a in range(len(top_inter)):
            for b in range(a + 1, len(top_inter)):
                specs.append(("int", int(top_inter[a]), int(top_inter[b])))

        # Precompute candidate columns on train/val; drop near-constant terms.
        cand_tr, cand_va, kept_specs = [], [], []
        for spec in specs:
            vtr = self._eval_spec(Xtr, spec).astype(float)
            if np.std(vtr) < 1e-12:
                continue
            cand_tr.append(vtr)
            cand_va.append(self._eval_spec(Xva, spec).astype(float))
            kept_specs.append(spec)
        specs = kept_specs

        if not specs:
            self.intercept_ = float(np.mean(y))
            self.term_specs_ = []
            self.term_coefs_ = np.zeros(0, dtype=float)
            self.round_decimals_selected_ = 6
            self.alpha_selected_ = 0.0
            self.feature_importance_ = np.zeros(n_features, dtype=float)
            self.selected_features_ = []
            self.operations_ = 0
            return self

        selected = []
        selected_set = set()
        best_global = None

        # Residual for greedy search starts at mean-only model.
        current_pred_tr = np.full_like(ytr, np.mean(ytr), dtype=float)
        residual = ytr - current_pred_tr

        max_steps = min(int(self.max_terms), len(specs))
        for step in range(max_steps):
            best_idx = None
            best_score = -np.inf
            for i, col in enumerate(cand_tr):
                if i in selected_set:
                    continue
                centered = col - np.mean(col)
                score = abs(float(centered @ residual)) / (np.linalg.norm(centered) + 1e-12)
                if score > best_score:
                    best_score = score
                    best_idx = i
            if best_idx is None:
                break

            selected.append(best_idx)
            selected_set.add(best_idx)

            Dtr = np.column_stack([cand_tr[i] for i in selected])
            Dva = np.column_stack([cand_va[i] for i in selected])

            local_best = None
            for alpha in self.alphas:
                inter, coef = self._ridge_with_intercept(Dtr, ytr, alpha)
                pred_tr = inter + Dtr @ coef
                pred_va = inter + Dva @ coef
                rmse_va = self._rmse(yva, pred_va)
                complexity = len(selected)
                obj = rmse_va * (1.0 + float(self.complexity_penalty) * complexity)
                if local_best is None or obj < local_best["obj"]:
                    local_best = {
                        "obj": float(obj),
                        "rmse_va": float(rmse_va),
                        "alpha": float(alpha),
                        "inter": float(inter),
                        "coef": np.asarray(coef, dtype=float),
                        "pred_tr": np.asarray(pred_tr, dtype=float),
                    }

            current_pred_tr = local_best["pred_tr"]
            residual = ytr - current_pred_tr

            if best_global is None or local_best["obj"] < best_global["obj"] - float(self.improvement_tol):
                best_global = {
                    "obj": float(local_best["obj"]),
                    "alpha": float(local_best["alpha"]),
                    "selected": list(selected),
                }

            if step + 1 >= int(self.min_terms):
                no_improve_steps = (step + 1) - len(best_global["selected"])
                if no_improve_steps >= 2:
                    break

        if best_global is None or len(best_global["selected"]) == 0:
            best_global = {"alpha": 0.0, "selected": [int(np.argmax(corr)) if n_features else 0]}

        self.alpha_selected_ = float(best_global["alpha"])
        chosen = best_global["selected"]
        D_all = np.column_stack([self._eval_spec(X, specs[i]) for i in chosen])
        inter_full, coef_full = self._ridge_with_intercept(D_all, y, self.alpha_selected_)

        # Light rounding selected on validation split.
        Dva_chosen = np.column_stack([cand_va[i] for i in chosen])
        best_round = None
        for decimals in self.round_decimals_grid:
            inter_r = float(np.round(inter_full, int(decimals)))
            coef_r = np.round(coef_full, int(decimals))
            eps = 0.5 * (10.0 ** (-int(decimals)))
            coef_r[np.abs(coef_r) < eps] = 0.0
            pred_va = inter_r + Dva_chosen @ coef_r
            rmse_va = self._rmse(yva, pred_va)
            complexity = int(np.sum(np.abs(coef_r) > self.coef_tol))
            obj = rmse_va * (1.0 + float(self.complexity_penalty) * complexity)
            if best_round is None or obj < best_round["obj"]:
                best_round = {
                    "obj": float(obj),
                    "inter": inter_r,
                    "coef": np.asarray(coef_r, dtype=float),
                    "decimals": int(decimals),
                }

        self.intercept_ = float(best_round["inter"])
        self.term_specs_ = [specs[i] for i in chosen]
        self.term_coefs_ = np.asarray(best_round["coef"], dtype=float)
        self.round_decimals_selected_ = int(best_round["decimals"])

        # Remove exact zero terms after rounding.
        nonzero = np.where(np.abs(self.term_coefs_) > self.coef_tol)[0]
        self.term_specs_ = [self.term_specs_[i] for i in nonzero]
        self.term_coefs_ = self.term_coefs_[nonzero]

        feature_imp = np.zeros(n_features, dtype=float)
        for spec, coef in zip(self.term_specs_, self.term_coefs_):
            if spec[0] == "int":
                feature_imp[spec[1]] += 0.5 * abs(float(coef))
                feature_imp[spec[2]] += 0.5 * abs(float(coef))
            else:
                feature_imp[spec[1]] += abs(float(coef))
        self.feature_importance_ = feature_imp
        self.selected_features_ = sorted(int(i) for i in np.where(feature_imp > self.coef_tol)[0])

        ops = 0
        for spec in self.term_specs_:
            if spec[0] == "lin":
                ops += 2
            elif spec[0] in ("plus", "minus"):
                ops += 3
            elif spec[0] == "sq":
                ops += 4
            else:
                ops += 5
        self.operations_ = int(ops + max(len(self.term_specs_) - 1, 0))
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "term_specs_", "term_coefs_", "feature_importance_"])
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        pred = np.full(X.shape[0], self.intercept_, dtype=float)
        for spec, coef in zip(self.term_specs_, self.term_coefs_):
            pred += float(coef) * self._eval_spec(X, spec)
        return pred

    def __str__(self):
        check_is_fitted(self, ["intercept_", "term_specs_", "term_coefs_", "feature_importance_"])
        decimals = int(getattr(self, "round_decimals_selected_", 6))

        terms = [(spec, float(c)) for spec, c in zip(self.term_specs_, self.term_coefs_)]
        terms.sort(key=lambda t: abs(t[1]), reverse=True)

        rhs_parts = [f"{self.intercept_:+.{decimals}f}"]
        for spec, coef in terms:
            rhs_parts.append(f"({coef:+.{decimals}f})*{self._format_term(spec, decimals)}")
        equation = " + ".join(rhs_parts)

        lines = [
            "Sparse Signed-Basis Pursuit Regressor",
            "Answer protocol: reply with ONE numeric value only (no words, no extra numbers).",
            "Exact prediction equation:",
            "  y = " + equation,
            "",
            "Basis terms used (sorted by |coefficient|):",
        ]
        if terms:
            for i, (spec, coef) in enumerate(terms, 1):
                lines.append(f"  {i:2d}. coef={coef:+.{decimals}f}  term={self._format_term(spec, decimals)}")
        else:
            lines.append("  (none)")

        active = [int(i) for i in self.selected_features_]
        lines.append("")
        lines.append("Active features: " + (", ".join(f"x{i}" for i in active) if active else "none"))
        if self.n_features_in_ <= 30 and len(active) < self.n_features_in_:
            active_set = set(active)
            zero_feats = [f"x{i}" for i in range(self.n_features_in_) if i not in active_set]
            lines.append("Zero-contribution features: " + ", ".join(zero_feats))

        lines.append(f"Selected ridge alpha: {self.alpha_selected_:.6g}")
        lines.append(f"Selected coefficient decimals: {self.round_decimals_selected_}")
        lines.append(f"Approximate arithmetic operations: {self.operations_}")
        return "\n".join(lines)
