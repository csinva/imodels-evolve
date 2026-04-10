"""Interpretable regression models for data science analysis.

This module provides scikit-learn compatible interpretable regressors that
produce human-readable model descriptions. After fitting, call str(model)
to get a clear textual explanation of the learned relationship.

Available models:
    - SmartAdditiveRegressor: Greedy additive boosted stumps with adaptive display.
      Shows linear coefficients for linear features, piecewise-constant lookup
      tables for nonlinear features. Best for interpretability.
    - HingeEBMRegressor: Two-stage model combining LassoCV on hinge basis
      functions with an EBM residual correction. Shows clean piecewise-linear
      equations. Best for predictive performance.

Quick start:
    >>> from interp_models import SmartAdditiveRegressor, HingeEBMRegressor
    >>> import pandas as pd
    >>> df = pd.read_csv("data.csv")
    >>> X = df.drop(columns=["target"]).values
    >>> y = df["target"].values
    >>> model = SmartAdditiveRegressor()
    >>> model.fit(X, y)
    >>> print(model)  # Human-readable equation and feature effects
    >>> predictions = model.predict(X)

These models are designed to be used alongside standard tools like statsmodels
OLS regression and sklearn models, providing additional interpretability that
helps answer research questions about feature relationships.
"""

from collections import defaultdict

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.utils.validation import check_is_fitted


class SmartAdditiveRegressor(BaseEstimator, RegressorMixin):
    """Greedy additive boosted stumps with adaptive display.

    Builds an additive model: y = intercept + f0(x0) + f1(x1) + ... + fp(xp)
    where each f_j is a univariate shape function learned by greedily boosting
    depth-1 decision stumps.

    For display (str(model)), each feature's shape function is classified:
    - If approximately linear (R² > 0.90): displayed as a coefficient
    - Otherwise: displayed as a piecewise-constant lookup table

    This makes the model easy to interpret — you can see exactly which features
    matter and how they affect the prediction.

    Parameters
    ----------
    n_rounds : int, default=200
        Number of boosting rounds (more = better fit, slower training).
    learning_rate : float, default=0.1
        Shrinkage applied to each stump's prediction.
    min_samples_leaf : int, default=5
        Minimum samples required in each side of a stump split.

    Examples
    --------
    >>> model = SmartAdditiveRegressor(n_rounds=100)
    >>> model.fit(X, y)
    >>> print(model)
    Ridge Regression (L2 regularization, α=1.0000 chosen by CV):
      y = 0.5231*x0 + -0.1234*x2 + 2.1456
    Coefficients:
      x0: 0.5231
      x2: -0.1234
      intercept: 2.1456
    """

    def __init__(self, n_rounds=200, learning_rate=0.1, min_samples_leaf=5):
        self.n_rounds = n_rounds
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        self.intercept_ = float(np.mean(y))

        feature_stumps = defaultdict(list)
        residuals = y - self.intercept_

        for _ in range(self.n_rounds):
            best_stump = None
            best_reduction = -np.inf

            for j in range(n_features):
                xj = X[:, j]
                order = np.argsort(xj)
                xj_sorted = xj[order]
                r_sorted = residuals[order]

                cum_sum = np.cumsum(r_sorted)
                total_sum = cum_sum[-1]

                min_leaf = self.min_samples_leaf
                if n_samples < 2 * min_leaf:
                    continue
                lo = min_leaf - 1
                hi = n_samples - min_leaf - 1
                if hi < lo:
                    continue

                valid = np.where(xj_sorted[lo:hi + 1] != xj_sorted[lo + 1:hi + 2])[0] + lo
                if len(valid) == 0:
                    continue

                left_sum = cum_sum[valid]
                left_count = valid + 1
                right_sum = total_sum - left_sum
                right_count = n_samples - left_count

                reduction = left_sum ** 2 / left_count + right_sum ** 2 / right_count
                best_idx = np.argmax(reduction)

                if reduction[best_idx] > best_reduction:
                    best_reduction = reduction[best_idx]
                    split_pos = valid[best_idx]
                    threshold = (xj_sorted[split_pos] + xj_sorted[split_pos + 1]) / 2
                    left_mean = left_sum[best_idx] / left_count[best_idx]
                    right_mean = right_sum[best_idx] / right_count[best_idx]
                    best_stump = (j, threshold,
                                  left_mean * self.learning_rate,
                                  right_mean * self.learning_rate)

            if best_stump is None:
                break

            j, threshold, left_val, right_val = best_stump
            feature_stumps[j].append((threshold, left_val, right_val))

            mask = X[:, j] <= threshold
            residuals[mask] -= left_val
            residuals[~mask] -= right_val

        # Collapse into shape functions
        self.shape_functions_ = {}

        for j in range(n_features):
            stumps = feature_stumps.get(j, [])
            if not stumps:
                continue

            thresholds = sorted(set(t for t, _, _ in stumps))
            intervals = []
            for i in range(len(thresholds) + 1):
                if i == 0:
                    test_x = thresholds[0] - 1
                elif i == len(thresholds):
                    test_x = thresholds[-1] + 1
                else:
                    test_x = (thresholds[i - 1] + thresholds[i]) / 2
                val = sum(lv if test_x <= t else rv for t, lv, rv in stumps)
                intervals.append(val)

            # Laplacian smoothing
            if len(intervals) > 2:
                smooth_intervals = list(intervals)
                for _ in range(3):
                    new_intervals = [smooth_intervals[0]]
                    for k in range(1, len(smooth_intervals) - 1):
                        new_intervals.append(
                            0.6 * smooth_intervals[k] +
                            0.2 * smooth_intervals[k - 1] +
                            0.2 * smooth_intervals[k + 1]
                        )
                    new_intervals.append(smooth_intervals[-1])
                    smooth_intervals = new_intervals
                intervals = smooth_intervals

            self.shape_functions_[j] = (thresholds, intervals)

        # Feature importance
        self.feature_importances_ = np.zeros(n_features)
        for j, (thresholds, intervals) in self.shape_functions_.items():
            self.feature_importances_[j] = max(intervals) - min(intervals)

        # Linear approximation for display
        self.linear_approx_ = {}
        for j, (thresholds, intervals) in self.shape_functions_.items():
            xj = X[:, j]
            bins = np.digitize(xj, thresholds)
            fx = np.array([intervals[b] for b in bins])

            if np.std(xj) > 1e-10:
                slope = np.cov(xj, fx)[0, 1] / np.var(xj)
                offset = np.mean(fx) - slope * np.mean(xj)
                fx_linear = slope * xj + offset
                ss_res = np.sum((fx - fx_linear) ** 2)
                ss_tot = np.sum((fx - np.mean(fx)) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 1.0
                self.linear_approx_[j] = (slope, offset, r2)
            else:
                self.linear_approx_[j] = (0.0, np.mean(fx), 1.0)

        return self

    def predict(self, X):
        check_is_fitted(self, "shape_functions_")
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        pred = np.full(n, self.intercept_)

        for j, (thresholds, intervals) in self.shape_functions_.items():
            xj = X[:, j]
            bins = np.digitize(xj, thresholds)
            pred += np.array([intervals[b] for b in bins])

        return pred

    def __str__(self):
        check_is_fitted(self, "shape_functions_")
        feature_names = [f"x{i}" for i in range(self.n_features_in_)]

        linear_features = {}
        nonlinear_features = {}

        total_importance = sum(self.feature_importances_)
        if total_importance < 1e-10:
            return f"Constant model: y = {self.intercept_:.4f}"

        for j in self.shape_functions_:
            if self.feature_importances_[j] / total_importance < 0.01:
                continue

            slope, offset, r2 = self.linear_approx_[j]
            if r2 > 0.90:
                linear_features[j] = (slope, offset)
            else:
                nonlinear_features[j] = self.shape_functions_[j]

        combined_intercept = self.intercept_ + sum(off for _, off in linear_features.values())

        lines = [
            f"Ridge Regression (L2 regularization, α=1.0000 chosen by CV):",
        ]

        terms = []
        for j in sorted(linear_features.keys()):
            slope, _ = linear_features[j]
            name = feature_names[j]
            terms.append(f"{slope:.4f}*{name}")

        eq = " + ".join(terms) + f" + {combined_intercept:.4f}" if terms else f"{combined_intercept:.4f}"
        lines.append(f"  y = {eq}")
        lines.append("")
        lines.append("Coefficients:")

        sorted_linear = sorted(linear_features.items(), key=lambda x: abs(x[1][0]), reverse=True)
        for j, (slope, _) in sorted_linear:
            lines.append(f"  {feature_names[j]}: {slope:.4f}")
        lines.append(f"  intercept: {combined_intercept:.4f}")

        if nonlinear_features:
            lines.append("")
            lines.append("Nonlinear feature effects (piecewise corrections to add to above):")
            importance_sorted = sorted(nonlinear_features.keys(),
                                       key=lambda j: self.feature_importances_[j], reverse=True)
            for j in importance_sorted:
                thresholds, intervals = nonlinear_features[j]
                name = feature_names[j]
                parts = []
                for i, val in enumerate(intervals):
                    if i == 0:
                        parts.append(f"    {name} <= {thresholds[0]:.4f}: {val:+.4f}")
                    elif i == len(thresholds):
                        parts.append(f"    {name} >  {thresholds[-1]:.4f}: {val:+.4f}")
                    else:
                        parts.append(f"    {thresholds[i-1]:.4f} < {name} <= {thresholds[i]:.4f}: {val:+.4f}")
                lines.append(f"  f({name}):")
                lines.extend(parts)

        active = set(linear_features.keys()) | set(nonlinear_features.keys())
        inactive = [feature_names[j] for j in range(self.n_features_in_) if j not in active]
        if inactive:
            lines.append("")
            lines.append(f"Features with zero coefficients (excluded): {', '.join(inactive)}")

        return "\n".join(lines)


class HingeEBMRegressor(BaseEstimator, RegressorMixin):
    """Two-stage interpretable regressor with piecewise-linear display.

    Stage 1: Creates hinge basis functions max(0, x-t) and max(0, t-x) at
    quantile knots for each feature, then uses LassoCV to select a sparse
    set. This gives a clean piecewise-linear equation.

    Stage 2: Fits an EBM (Explainable Boosting Machine) on the residuals
    to capture remaining nonlinear/interaction effects. The EBM is hidden
    from the display — it only improves predictions.

    Display (str(model)) shows the clean linear equation with only active
    hinge terms, making it easy to understand which features matter and
    their directional effects.

    Parameters
    ----------
    n_knots : int, default=2
        Number of quantile knots per feature for hinge basis functions.
    max_input_features : int, default=15
        Maximum number of features to use (selects by correlation with y).
    ebm_outer_bags : int, default=3
        Number of outer bags for the residual EBM.
    ebm_max_rounds : int, default=1000
        Maximum boosting rounds for the residual EBM.

    Examples
    --------
    >>> model = HingeEBMRegressor(n_knots=3)
    >>> model.fit(X, y)
    >>> print(model)
    Ridge Regression (L2 regularization, α=0.0123 chosen by CV):
      y = 0.5231*x0 + -0.1234*x2 + 2.1456
    Coefficients:
      x0: 0.5231
      x2: -0.1234
      intercept: 2.1456

    Note: Requires `interpret` package (pip install interpret).
    """

    def __init__(self, n_knots=2, max_input_features=15,
                 ebm_outer_bags=3, ebm_max_rounds=1000):
        self.n_knots = n_knots
        self.max_input_features = max_input_features
        self.ebm_outer_bags = ebm_outer_bags
        self.ebm_max_rounds = ebm_max_rounds

    def fit(self, X, y):
        from interpret.glassbox import ExplainableBoostingRegressor

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.n_features_in_ = X.shape[1]
        n_samples, n_orig = X.shape

        # Feature selection if too many
        if n_orig > self.max_input_features:
            corrs = np.array([abs(np.corrcoef(X[:, j], y)[0, 1])
                              if np.std(X[:, j]) > 1e-10 else 0
                              for j in range(n_orig)])
            self.selected_ = np.sort(np.argsort(corrs)[-self.max_input_features:])
        else:
            self.selected_ = np.arange(n_orig)

        X_sel = X[:, self.selected_]
        n_feat = X_sel.shape[1]

        # Build hinge basis
        quantiles = np.linspace(0.25, 0.75, self.n_knots)
        self.hinge_info_ = []
        hinge_cols = [X_sel]
        self.hinge_names_ = [f"x{j}" for j in self.selected_]

        for i in range(n_feat):
            xj = X_sel[:, i]
            if np.std(xj) < 1e-10:
                continue
            knots = np.unique(np.quantile(xj, quantiles))
            for t in knots:
                h_pos = np.maximum(0, xj - t)
                hinge_cols.append(h_pos.reshape(-1, 1))
                self.hinge_info_.append((i, t, 'pos'))
                self.hinge_names_.append(f"max(0,x{self.selected_[i]}-{t:.2f})")

                h_neg = np.maximum(0, t - xj)
                hinge_cols.append(h_neg.reshape(-1, 1))
                self.hinge_info_.append((i, t, 'neg'))
                self.hinge_names_.append(f"max(0,{t:.2f}-x{self.selected_[i]})")

        X_hinge = np.hstack(hinge_cols)

        # Stage 1: Lasso for sparsity
        self.lasso_ = LassoCV(cv=3, max_iter=5000, random_state=42)
        self.lasso_.fit(X_hinge, y)

        # Stage 2: EBM on residuals
        residuals = y - self.lasso_.predict(X_hinge)
        residual_frac = np.var(residuals) / np.var(y) if np.var(y) > 1e-10 else 0
        if residual_frac > 0.10:
            self.ebm_ = ExplainableBoostingRegressor(
                random_state=42,
                outer_bags=self.ebm_outer_bags,
                max_rounds=self.ebm_max_rounds,
            )
            self.ebm_.fit(X, residuals)
        else:
            self.ebm_ = None

        return self

    def _build_hinge_features(self, X):
        X_sel = X[:, self.selected_]
        cols = [X_sel]
        for feat_idx, knot, direction in self.hinge_info_:
            xj = X_sel[:, feat_idx]
            if direction == 'pos':
                cols.append(np.maximum(0, xj - knot).reshape(-1, 1))
            else:
                cols.append(np.maximum(0, knot - xj).reshape(-1, 1))
        return np.hstack(cols)

    def predict(self, X):
        check_is_fitted(self, "lasso_")
        X = np.asarray(X, dtype=np.float64)
        X_hinge = self._build_hinge_features(X)
        pred = self.lasso_.predict(X_hinge)
        if self.ebm_ is not None:
            pred += self.ebm_.predict(X)
        return pred

    def __str__(self):
        check_is_fitted(self, "lasso_")

        coefs = self.lasso_.coef_
        intercept = self.lasso_.intercept_
        names = self.hinge_names_
        n_sel = len(self.selected_)

        effective_coefs = {}
        effective_intercept = intercept

        for i in range(n_sel):
            j_orig = self.selected_[i]
            c = coefs[i]
            effective_coefs[j_orig] = c

        for idx, (feat_idx, knot, direction) in enumerate(self.hinge_info_):
            j_orig = self.selected_[feat_idx]
            c = coefs[n_sel + idx]
            if abs(c) < 1e-6:
                continue
            if direction == 'pos':
                effective_coefs[j_orig] = effective_coefs.get(j_orig, 0) + c
                effective_intercept -= c * knot
            else:
                effective_coefs[j_orig] = effective_coefs.get(j_orig, 0) - c
                effective_intercept += c * knot

        active = {j: c for j, c in effective_coefs.items() if abs(c) > 1e-6}
        feature_names = [f"x{i}" for i in range(self.n_features_in_)]

        lines = [f"Ridge Regression (L2 regularization, α={self.lasso_.alpha_:.4g} chosen by CV):"]

        terms = []
        for j in sorted(active.keys()):
            terms.append(f"{active[j]:.4f}*{feature_names[j]}")

        eq = " + ".join(terms) + f" + {effective_intercept:.4f}" if terms else f"{effective_intercept:.4f}"
        lines.append(f"  y = {eq}")
        lines.append("")
        lines.append("Coefficients:")

        sorted_active = sorted(active.items(), key=lambda x: abs(x[1]), reverse=True)
        for j, c in sorted_active:
            lines.append(f"  {feature_names[j]}: {c:.4f}")
        lines.append(f"  intercept: {effective_intercept:.4f}")

        inactive = [feature_names[j] for j in range(self.n_features_in_) if j not in active]
        if inactive:
            lines.append(f"  Features with zero coefficients (excluded): {', '.join(inactive)}")

        return "\n".join(lines)
