"""tiny_dt — TinyDTDepth2Regressor from the agentic-imodels library.

Generated from: result_libs/apr19-claude-4.7-effort=medium-rerun3/interpretable_regressors_lib/failure/interpretable_regressor_fa6e001_TinyDTDepth2_v1.py

Shorthand: TinyDTDepth2_v1
Mean global rank (lower is better): 334.01   (pooled 65 dev datasets)
Interpretability (fraction passed, higher is better):
    dev  (43 tests):  0.674
    test (157 tests): 0.713
"""

import argparse, csv, os, subprocess, sys, time
from collections import defaultdict
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV, Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.utils.validation import check_is_fitted


# =========== CLASS ============


class TinyDTDepth2Regressor(BaseEstimator, RegressorMixin):
    """Tiny depth-2 decision tree (4-leaf max)."""

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        self.d_ = X.shape[1]
        self.tree_ = DecisionTreeRegressor(max_depth=2, min_samples_leaf=4, random_state=42).fit(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self, "tree_")
        return self.tree_.predict(np.asarray(X, dtype=float))

    def __str__(self):
        check_is_fitted(self, "tree_")
        names = [f"x{i}" for i in range(self.d_)]
        return f"Decision Tree Regressor (max_depth=2):\n{export_text(self.tree_, feature_names=names)}"
    """ElasticNetCV with low l1_ratio=0.1 (mostly ridge) for less aggressive sparsity."""

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        self.d_ = X.shape[1]
        scaler = StandardScaler().fit(X); Xs = scaler.transform(X)
        en = ElasticNetCV(l1_ratio=[0.05, 0.1, 0.2, 0.5], cv=3, n_alphas=30, max_iter=5000).fit(Xs, y)
        self.coef_ = en.coef_ / scaler.scale_
        self.intercept_ = float(en.intercept_ - np.sum(self.coef_ * scaler.mean_))
        self.alpha_ = float(en.alpha_); self.l1_ratio_ = float(en.l1_ratio_)
        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_

    def __str__(self):
        check_is_fitted(self, "coef_")
        active = [(i, c) for i, c in enumerate(self.coef_) if abs(c) > 1e-10]
        zeroed = [f"x{i}" for i, c in enumerate(self.coef_) if abs(c) <= 1e-10]
        eq = " + ".join(f"{c:+.4g}*x{i}" for i, c in active) or "0"
        lines = [f"ElasticNet (α={self.alpha_:.4g}, l1_ratio={self.l1_ratio_:.2f}):",
                 f"  y = {self.intercept_:+.4f} + {eq}", "", "Coefficients:",
                 f"  intercept: {self.intercept_:.4f}"]
        for i, c in active: lines.append(f"  x{i}: {c:.4f}")
        if zeroed: lines.append(f"Features with zero coefficients (excluded): {', '.join(zeroed)}")
        return "\n".join(lines)
    """Orthogonal Matching Pursuit with CV; clean linear formula."""

    def fit(self, X, y):
        from sklearn.linear_model import OrthogonalMatchingPursuitCV
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        self.d_ = X.shape[1]
        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)
        omp = OrthogonalMatchingPursuitCV(cv=3).fit(Xs, y)
        self.coef_ = omp.coef_ / scaler.scale_
        self.intercept_ = float(omp.intercept_ - np.sum(self.coef_ * scaler.mean_))
        self.n_nonzero_ = int(getattr(omp, "n_nonzero_coefs_", np.count_nonzero(omp.coef_)))
        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_

    def __str__(self):
        check_is_fitted(self, "coef_")
        active = [(i, c) for i, c in enumerate(self.coef_) if abs(c) > 1e-10]
        zeroed = [f"x{i}" for i, c in enumerate(self.coef_) if abs(c) <= 1e-10]
        eq = " + ".join(f"{c:+.4g}*x{i}" for i, c in active) or "0"
        lines = [f"OrthogonalMatchingPursuit (CV-selected {self.n_nonzero_} nonzero coefficients):",
                 f"  y = {self.intercept_:+.4f} + {eq}", "", "Coefficients:",
                 f"  intercept: {self.intercept_:.4f}"]
        for i, c in active: lines.append(f"  x{i}: {c:.4f}")
        if zeroed: lines.append(f"Features with zero coefficients (excluded): {', '.join(zeroed)}")
        return "\n".join(lines)
    """BayesianRidge regression on standardized features; formula in original scale."""

    def fit(self, X, y):
        from sklearn.linear_model import BayesianRidge
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        self.d_ = X.shape[1]
        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)
        br = BayesianRidge().fit(Xs, y)
        self.coef_ = br.coef_ / scaler.scale_
        self.intercept_ = float(br.intercept_ - np.sum(self.coef_ * scaler.mean_))
        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_

    def __str__(self):
        check_is_fitted(self, "coef_")
        eq = " + ".join(f"{c:+.4g}*x{i}" for i, c in enumerate(self.coef_))
        lines = ["BayesianRidge Linear Regression (automatic hyperparameter tuning via type-II ML):",
                 f"  y = {self.intercept_:+.4f} + {eq}", "", "Coefficients:",
                 f"  intercept: {self.intercept_:.4f}"]
        for i, c in enumerate(self.coef_): lines.append(f"  x{i}: {c:.4f}")
        return "\n".join(lines)


class _Unused15(BaseEstimator, RegressorMixin):
    """Depth-3 decision tree."""

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        self.d_ = X.shape[1]
        self.tree_ = DecisionTreeRegressor(max_depth=3, min_samples_leaf=8, random_state=42).fit(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self, "tree_")
        return self.tree_.predict(np.asarray(X, dtype=float))

    def __str__(self):
        check_is_fitted(self, "tree_")
        names = [f"x{i}" for i in range(self.d_)]
        return f"Decision Tree Regressor (max_depth=3):\n{export_text(self.tree_, feature_names=names)}"


class _Unused14(BaseEstimator, RegressorMixin):
    """Plain OLS with output sorted by |std-coef|, emphasizing dominant features first."""

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        self.d_ = X.shape[1]
        self.x_std_ = X.std(axis=0) + 1e-12
        self.ols_ = LinearRegression().fit(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self, "ols_")
        return self.ols_.predict(np.asarray(X, dtype=float))

    def __str__(self):
        check_is_fitted(self, "ols_")
        coef = self.ols_.coef_; b = float(self.ols_.intercept_)
        importance = np.abs(coef * self.x_std_)  # approximate "standardized effect"
        order = np.argsort(-importance)
        eq = " + ".join(f"{coef[i]:+.4g}*x{i}" for i in range(len(coef)))
        lines = [
            "OLS Linear Regression:",
            f"  y = {b:+.4f} + {eq}",
            "",
            "Coefficients (sorted by effect size |coef * std(x)|):",
            f"  intercept: {b:.4f}",
        ]
        for i in order:
            lines.append(f"  x{i}: {coef[i]:+.4f}    (|coef * std(x{i})| = {importance[i]:.3f})")
        return "\n".join(lines)


class _Unused13(BaseEstimator, RegressorMixin):
    """Plain LassoCV; clean string with alpha and active/excluded features."""

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        self.d_ = X.shape[1]
        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)
        lasso = LassoCV(cv=3, n_alphas=20, max_iter=5000).fit(Xs, y)
        self.coef_ = lasso.coef_ / scaler.scale_
        self.intercept_ = float(lasso.intercept_ - np.sum(self.coef_ * scaler.mean_))
        self.alpha_ = float(lasso.alpha_)
        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_

    def __str__(self):
        check_is_fitted(self, "coef_")
        active = [(i, c) for i, c in enumerate(self.coef_) if abs(c) > 1e-10]
        zeroed = [f"x{i}" for i, c in enumerate(self.coef_) if abs(c) <= 1e-10]
        eq = " + ".join(f"{c:+.4g}*x{i}" for i, c in active) or "0"
        lines = [f"LASSO Regression (L1 regularization, α={self.alpha_:.4g} chosen by CV — promotes sparsity):",
                 f"  y = {self.intercept_:+.4f} + {eq}", "", "Coefficients:",
                 f"  intercept: {self.intercept_:.4f}"]
        for i, c in active: lines.append(f"  x{i}: {c:.4f}")
        if zeroed: lines.append(f"Features with zero coefficients (excluded): {', '.join(zeroed)}")
        return "\n".join(lines)


class _Unused12(BaseEstimator, RegressorMixin):
    """OLS over {x_i, x_i^2} for each feature; Lasso-select then OLS refit; per-feature polynomial reporting."""

    def __init__(self, max_basis=12):
        self.max_basis = max_basis

    def _basis(self, X):
        return np.hstack([X, X * X])

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        n, d = X.shape; self.d_ = d
        B = self._basis(X)
        scaler = StandardScaler().fit(B)
        Bs = scaler.transform(B)
        lasso = LassoCV(cv=min(3, max(2, n // 20)), n_alphas=25, max_iter=5000).fit(Bs, y)
        order = np.argsort(-np.abs(lasso.coef_))
        kept = [int(i) for i in order if lasso.coef_[i] != 0][: self.max_basis]
        if not kept:
            kept = [int(np.argmax(np.abs(lasso.coef_)))] if np.any(lasso.coef_) else [0]
        self.support_ = sorted(kept)
        ols = LinearRegression().fit(B[:, self.support_], y)
        self.coef_ = ols.coef_; self.intercept_ = float(ols.intercept_)
        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        return self._basis(np.asarray(X, dtype=float))[:, self.support_] @ self.coef_ + self.intercept_

    def __str__(self):
        check_is_fitted(self, "coef_")
        d = self.d_
        per_feat = defaultdict(list)
        for c, idx in zip(self.coef_, self.support_):
            if idx < d: per_feat[idx].append(("x", c))
            else: per_feat[idx - d].append(("x^2", c))
        lines = ["Quadratic Regression: y = intercept + sum_j (a_j*x_j + b_j*x_j^2)",
                 f"  intercept: {self.intercept_:.4f}", "", "Per-feature polynomial shape:"]
        for j in range(d):
            terms = per_feat.get(j, [])
            if not terms: lines.append(f"\n  x{j}: (excluded, no effect)"); continue
            parts = [f"{c:+.4g} * {'x' + str(j) if t == 'x' else f'x{j}^2'}" for t, c in terms]
            lines.append(f"\n  x{j}: f(x{j}) = " + " + ".join(parts))
        return "\n".join(lines)


class _Unused11(BaseEstimator, RegressorMixin):
    """Robust linear regression (Huber loss); prints as OLS-style linear formula."""

    def fit(self, X, y):
        from sklearn.linear_model import HuberRegressor
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        self.d_ = X.shape[1]
        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)
        hub = HuberRegressor(max_iter=200, alpha=1e-4).fit(Xs, y)
        self.coef_ = hub.coef_ / scaler.scale_
        self.intercept_ = float(hub.intercept_ - np.sum(self.coef_ * scaler.mean_))
        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_

    def __str__(self):
        check_is_fitted(self, "coef_")
        eq = " + ".join(f"{c:+.4g}*x{i}" for i, c in enumerate(self.coef_))
        lines = ["Robust Linear Regression (Huber loss, less sensitive to outliers):",
                 f"  y = {self.intercept_:+.4f} + {eq}", "", "Coefficients:",
                 f"  intercept: {self.intercept_:.4f}"]
        for i, c in enumerate(self.coef_): lines.append(f"  x{i}: {c:.4f}")
        return "\n".join(lines)


class _Unused10(BaseEstimator, RegressorMixin):
    """Plain OLS on centered-only features (no scaling); rescale coefficients as-is."""

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        self.d_ = X.shape[1]
        # sklearn's LinearRegression already includes a bias; center explicitly for numerical stability
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        lr = LinearRegression(fit_intercept=True).fit(Xc, y)
        self.coef_ = lr.coef_
        self.intercept_ = float(lr.intercept_ - np.dot(self.coef_, self.mean_))
        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_

    def __str__(self):
        check_is_fitted(self, "coef_")
        eq = " + ".join(f"{c:+.4g}*x{i}" for i, c in enumerate(self.coef_))
        lines = [f"OLS Linear Regression:", f"  y = {self.intercept_:+.4f} + {eq}", "",
                 "Coefficients:", f"  intercept: {self.intercept_:.4f}"]
        for i, c in enumerate(self.coef_): lines.append(f"  x{i}: {c:.4f}")
        return "\n".join(lines)


class _Unused9(BaseEstimator, RegressorMixin):
    """LassoCV for selection, OLS refit on ALL features that got non-zero (no cap)."""

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        n, d = X.shape; self.d_ = d
        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)
        lasso = LassoCV(cv=min(3, max(2, n // 20)), n_alphas=30, max_iter=8000).fit(Xs, y)
        keep = np.where(np.abs(lasso.coef_) > 1e-10)[0]
        if keep.size == 0:
            keep = np.array([int(np.argmax(np.abs(lasso.coef_)))]) if np.any(lasso.coef_) else np.arange(min(d, 1))
        self.support_ = sorted(keep.tolist())
        ols = LinearRegression().fit(X[:, self.support_], y)
        self.coef_ = ols.coef_; self.intercept_ = float(ols.intercept_)
        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        return np.asarray(X, dtype=float)[:, self.support_] @ self.coef_ + self.intercept_

    def __str__(self):
        check_is_fitted(self, "coef_")
        terms = " + ".join(f"{c:+.4g}*x{j}" for c, j in zip(self.coef_, self.support_))
        excluded = [f"x{j}" for j in range(self.d_) if j not in self.support_]
        lines = [f"LassoOLS Linear Regression ({len(self.support_)} features selected by LassoCV, refit with OLS):",
                 f"  y = {self.intercept_:+.4f} + {terms}", "", "Coefficients:",
                 f"  intercept: {self.intercept_:.4f}"]
        for c, j in zip(self.coef_, self.support_): lines.append(f"  x{j}: {c:.4f}")
        if excluded: lines.append(f"Features with zero coefficients (excluded): {', '.join(excluded)}")
        return "\n".join(lines)


class _Unused8(BaseEstimator, RegressorMixin):
    """Apply sign-preserving sqrt transform to X to tame outliers, fit OLS on transformed X.
    Reports as y ≈ intercept + sum c_i * sqrt_sign(x_i) with explanation."""

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        self.d_ = X.shape[1]
        Xt = np.sign(X) * np.sqrt(np.abs(X))
        self.ols_ = LinearRegression().fit(Xt, y)
        return self

    def predict(self, X):
        check_is_fitted(self, "ols_")
        X = np.asarray(X, dtype=float)
        Xt = np.sign(X) * np.sqrt(np.abs(X))
        return self.ols_.predict(Xt)

    def __str__(self):
        check_is_fitted(self, "ols_")
        coef = self.ols_.coef_; b = float(self.ols_.intercept_)
        eq = " + ".join(f"{c:+.4g}*s(x{i})" for i, c in enumerate(coef))
        lines = [
            "OLS Linear Regression in sqrt-sign transformed features:",
            "  where s(x) = sign(x) * sqrt(|x|), a monotone outlier-tamed transform",
            f"  y = {b:+.4f} + {eq}",
            "",
            "Coefficients on s(x_i):",
            f"  intercept: {b:.4f}",
        ]
        for i, c in enumerate(coef): lines.append(f"  s(x{i}): {c:.4f}")
        return "\n".join(lines)


class _Unused7(BaseEstimator, RegressorMixin):
    """Choose between OLS (n big) and RidgeCV (n small); always report as clean linear formula."""

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        n, d = X.shape; self.d_ = d
        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)
        if n > 5 * d:
            mdl = LinearRegression().fit(Xs, y); self.method_ = "OLS"
            alpha_str = ""
        else:
            mdl = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(Xs, y)
            self.method_ = f"RidgeCV(α={mdl.alpha_:.4g})"
            alpha_str = f" (ridge α={mdl.alpha_:.4g})"
        self.alpha_str_ = alpha_str
        coef_std = mdl.coef_
        self.coef_ = coef_std / scaler.scale_
        self.intercept_ = float(mdl.intercept_ - np.sum(self.coef_ * scaler.mean_))
        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_

    def __str__(self):
        check_is_fitted(self, "coef_")
        eq = " + ".join(f"{c:+.4g}*x{i}" for i, c in enumerate(self.coef_))
        lines = [
            f"Linear Regression ({self.method_}):",
            f"  y = {self.intercept_:+.4f} + {eq}",
            "",
            "Coefficients:",
            f"  intercept: {self.intercept_:.4f}",
        ]
        for i, c in enumerate(self.coef_): lines.append(f"  x{i}: {c:.4f}")
        return "\n".join(lines)


class _Unused6(BaseEstimator, RegressorMixin):
    """ElasticNetCV on standardized features; report as OLS-style linear formula in original scale."""

    def __init__(self, l1_ratio=(0.1, 0.5, 0.9)):
        self.l1_ratio = l1_ratio

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        n, d = X.shape; self.d_ = d
        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)
        en = ElasticNetCV(l1_ratio=list(self.l1_ratio), cv=min(3, max(2, n // 20)),
                          n_alphas=20, max_iter=5000).fit(Xs, y)
        coef_std = en.coef_
        self.coef_ = coef_std / scaler.scale_
        self.intercept_ = float(en.intercept_ - np.sum(self.coef_ * scaler.mean_))
        self.alpha_ = float(en.alpha_); self.l1_ratio_ = float(en.l1_ratio_)
        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_

    def __str__(self):
        check_is_fitted(self, "coef_")
        active = [(i, c) for i, c in enumerate(self.coef_) if abs(c) > 1e-10]
        zeroed = [f"x{i}" for i, c in enumerate(self.coef_) if abs(c) <= 1e-10]
        eq = " + ".join(f"{c:+.4g}*x{i}" for i, c in active) if active else "0"
        lines = [
            f"ElasticNet Regression (α={self.alpha_:.4g}, l1_ratio={self.l1_ratio_:.2f} chosen by CV):",
            f"  y = {self.intercept_:+.4f} + {eq}",
            "",
            "Coefficients:",
            f"  intercept: {self.intercept_:.4f}",
        ]
        for i, c in active: lines.append(f"  x{i}: {c:.4f}")
        if zeroed: lines.append(f"Features with zero coefficients (excluded): {', '.join(zeroed)}")
        return "\n".join(lines)


class _Unused5(BaseEstimator, RegressorMixin):
    """Per-feature basis: {x, max(0,x-med), max(0,-x+med)} → Lasso-select, OLS refit.
    Output grouped as per-feature piecewise-linear shape f_j(x_j)."""

    def __init__(self, max_basis=10):
        self.max_basis = max_basis

    def _basis(self, X):
        meds = self.medians_[None, :]
        pos = np.maximum(0.0, X - meds)
        neg = np.maximum(0.0, meds - X)
        return np.hstack([X, pos, neg])

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        n, d = X.shape; self.d_ = d
        self.medians_ = np.median(X, axis=0)
        B = self._basis(X)
        scaler = StandardScaler().fit(B)
        Bs = scaler.transform(B)
        lasso = LassoCV(cv=min(3, max(2, n // 20)), n_alphas=25, max_iter=5000).fit(Bs, y)
        order = np.argsort(-np.abs(lasso.coef_))
        kept = [int(i) for i in order if lasso.coef_[i] != 0][: self.max_basis]
        if not kept:
            kept = [int(np.argmax(np.abs(lasso.coef_)))] if np.any(lasso.coef_) else [0]
        self.support_ = sorted(kept)
        ols = LinearRegression().fit(B[:, self.support_], y)
        self.coef_ = ols.coef_; self.intercept_ = float(ols.intercept_)
        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        return self._basis(np.asarray(X, dtype=float))[:, self.support_] @ self.coef_ + self.intercept_

    def __str__(self):
        check_is_fitted(self, "coef_")
        d = self.d_
        per_feat = defaultdict(list)
        for c, idx in zip(self.coef_, self.support_):
            if idx < d: per_feat[idx].append(("lin", c, None))
            elif idx < 2*d:
                j = idx - d; per_feat[j].append(("pos", c, float(self.medians_[j])))
            else:
                j = idx - 2*d; per_feat[j].append(("neg", c, float(self.medians_[j])))
        lines = [
            "HingeGAM — additive piecewise-linear per feature:  y = intercept + sum_j f_j(x_j)",
            "  where f_j(x) = a_j * x + b_j * max(0, x - m_j) + c_j * max(0, m_j - x)",
            f"  intercept: {self.intercept_:.4f}",
            "",
            "Per-feature shape functions:",
        ]
        for j in range(d):
            terms = per_feat.get(j, [])
            if not terms:
                lines.append(f"\n  x{j}: f(x{j}) = 0  (feature excluded)")
                continue
            parts = []
            for kind, c, m in terms:
                if kind == "lin": parts.append(f"{c:+.4g} * x{j}")
                elif kind == "pos": parts.append(f"{c:+.4g} * max(0, x{j} - {m:+.3f})")
                else: parts.append(f"{c:+.4g} * max(0, {m:+.3f} - x{j})")
            lines.append(f"\n  x{j}: f(x{j}) = " + " + ".join(parts))
        return "\n".join(lines)


class _Unused4(BaseEstimator, RegressorMixin):
    """Sum of K decision stumps (depth-1 trees), each on one feature. Stagewise boosting with shrinkage."""

    def __init__(self, n_stumps=30, learning_rate=0.1, max_leaf_nodes=2):
        self.n_stumps = n_stumps
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        self.d_ = X.shape[1]
        self.init_ = float(np.mean(y))
        self.stumps_ = []
        resid = y - self.init_
        for _ in range(self.n_stumps):
            stump = DecisionTreeRegressor(max_depth=1, max_leaf_nodes=self.max_leaf_nodes,
                                          random_state=42).fit(X, resid)
            pred = stump.predict(X)
            if np.allclose(pred.std(), 0): break
            resid = resid - self.learning_rate * pred
            self.stumps_.append(stump)
        return self

    def predict(self, X):
        check_is_fitted(self, "stumps_")
        X = np.asarray(X, dtype=float)
        out = np.full(X.shape[0], self.init_, dtype=float)
        for s in self.stumps_:
            out += self.learning_rate * s.predict(X)
        return out

    def __str__(self):
        check_is_fitted(self, "stumps_")
        # Aggregate per-feature contributions
        agg = defaultdict(list)  # j -> list of (threshold, left_val, right_val) scaled by lr
        for s in self.stumps_:
            tree = s.tree_
            j = int(tree.feature[0])
            thr = float(tree.threshold[0])
            left_idx = tree.children_left[0]
            right_idx = tree.children_right[0]
            lv = float(tree.value[left_idx][0, 0]) * self.learning_rate
            rv = float(tree.value[right_idx][0, 0]) * self.learning_rate
            agg[j].append((thr, lv, rv))
        lines = [
            f"BoostedStumps: y = {self.init_:+.4f} + sum_k f_k(x)   ({len(self.stumps_)} stumps, lr={self.learning_rate})",
            "  each f_k is: if x_j <= thr then left_val else right_val",
            "",
            "Per-feature stump contributions (additive, summed over all stumps on that feature):",
        ]
        used = sorted(agg.keys())
        for j in used:
            lines.append(f"\n  x{j}:")
            for thr, lv, rv in agg[j][:8]:
                lines.append(f"    if x{j} <= {thr:+.3f}: {lv:+.4f}  else: {rv:+.4f}")
            if len(agg[j]) > 8:
                lines.append(f"    ... ({len(agg[j]) - 8} more stumps on x{j})")
        inactive = [f"x{i}" for i in range(self.d_) if i not in agg]
        if inactive:
            lines.append(f"\nFeatures with no stumps (zero effect on y): {', '.join(inactive)}")
        return "\n".join(lines)


class _Unused3(BaseEstimator, RegressorMixin):
    """Fit full OLS, drop coefs with |coef_std| < threshold * max (threshold=0.05), refit OLS."""

    def __init__(self, threshold=0.05):
        self.threshold = threshold

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        n, d = X.shape
        self.d_ = d
        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)
        full = LinearRegression().fit(Xs, y)
        std_coef = np.abs(full.coef_)
        if std_coef.max() > 0:
            keep_mask = std_coef >= self.threshold * std_coef.max()
        else:
            keep_mask = np.zeros(d, bool); keep_mask[0] = True
        self.support_ = sorted(np.where(keep_mask)[0].tolist())
        ols = LinearRegression().fit(X[:, self.support_], y)
        self.coef_ = ols.coef_
        self.intercept_ = float(ols.intercept_)
        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        return np.asarray(X, dtype=float)[:, self.support_] @ self.coef_ + self.intercept_

    def __str__(self):
        check_is_fitted(self, "coef_")
        terms = " + ".join(f"{c:+.4g}*x{j}" for c, j in zip(self.coef_, self.support_))
        excluded = [f"x{j}" for j in range(self.d_) if j not in self.support_]
        lines = [
            f"ThresholdOLS (full OLS then drop features with |std-coef| < {self.threshold}*max):",
            f"  y = {self.intercept_:+.4f} + {terms}",
            "",
            "Coefficients:",
            f"  intercept: {self.intercept_:.4f}",
        ]
        for c, j in zip(self.coef_, self.support_):
            lines.append(f"  x{j}: {c:.4f}")
        if excluded:
            lines.append(f"Features excluded (negligible effect): {', '.join(excluded)}")
        return "\n".join(lines)
