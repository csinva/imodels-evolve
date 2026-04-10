import json

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, t as t_dist


def ols_with_inference(x: np.ndarray, y: np.ndarray):
    """
    Fit OLS via normal equations and return coefficients, p-values, and CI.
    x must include an intercept column.
    """
    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    residuals = y - x @ beta
    n, k = x.shape
    dof = n - k
    sigma2 = float((residuals.T @ residuals) / dof)
    var_beta = sigma2 * xtx_inv
    se = np.sqrt(np.diag(var_beta))
    t_stats = beta / se
    pvals = 2 * (1 - t_dist.cdf(np.abs(t_stats), df=dof))
    t_crit = t_dist.ppf(0.975, df=dof)
    ci_low = beta - t_crit * se
    ci_high = beta + t_crit * se
    return beta, pvals, ci_low, ci_high


def main() -> None:
    # Load data and derive key variables for the question.
    df = pd.read_csv("caschools.csv")
    df["stratio"] = df["students"] / df["teachers"]
    df["avg_score"] = (df["read"] + df["math"]) / 2.0

    # Basic exploration summary.
    key_cols = [
        "stratio",
        "avg_score",
        "lunch",
        "calworks",
        "english",
        "income",
        "expenditure",
    ]
    analysis_df = df[key_cols].dropna().copy()
    n_obs = len(analysis_df)

    # Bivariate association.
    corr, corr_p = pearsonr(analysis_df["stratio"], analysis_df["avg_score"])

    # Multivariate model to test whether the relationship persists with controls.
    predictors = ["stratio", "lunch", "calworks", "english", "income", "expenditure"]
    x_no_const = analysis_df[predictors].to_numpy(dtype=float)
    x = np.column_stack([np.ones(n_obs), x_no_const])
    y = analysis_df["avg_score"].to_numpy(dtype=float)
    beta, pvals, ci_lows, ci_highs = ols_with_inference(x, y)

    str_idx = 1  # After intercept, first predictor is stratio.
    coef = float(beta[str_idx])
    pval = float(pvals[str_idx])
    ci_low = float(ci_lows[str_idx])
    ci_high = float(ci_highs[str_idx])

    # Convert significance/direction into the required Likert-style response.
    if coef < 0 and pval < 0.01 and corr < 0 and corr_p < 0.01:
        response = 93
    elif coef < 0 and pval < 0.05:
        response = 85
    elif coef < 0 and pval < 0.10:
        response = 65
    elif pval >= 0.05:
        response = 18
    else:
        # Significant relationship in the opposite direction from the hypothesis.
        response = 8

    explanation = (
        f"Using {n_obs} districts, the student-teacher ratio (students/teachers) is "
        f"negatively correlated with average test score (r={corr:.3f}, p={corr_p:.3g}). "
        f"In an OLS model controlling for lunch, calworks, english learners, income, and "
        f"expenditure, the student-teacher-ratio coefficient is {coef:.3f} "
        f"(p={pval:.3g}, 95% CI [{ci_low:.3f}, {ci_high:.3f}]). "
        "This indicates that lower student-teacher ratios are associated with higher "
        "academic performance, though this is an observational association rather than "
        "proof of causality."
    )

    result = {"response": int(response), "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
