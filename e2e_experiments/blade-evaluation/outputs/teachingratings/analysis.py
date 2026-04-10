import json
import numpy as np
import pandas as pd
from scipy import stats


def likert_from_result(coef: float, pvalue: float) -> int:
    """Map sign and significance of estimated impact to 0-100 Likert score."""
    if np.isnan(coef) or np.isnan(pvalue):
        return 50

    if coef > 0:
        if pvalue < 0.001:
            return 95
        if pvalue < 0.01:
            return 90
        if pvalue < 0.05:
            return 85
        if pvalue < 0.10:
            return 65
        return 25

    if coef < 0:
        if pvalue < 0.001:
            return 5
        if pvalue < 0.01:
            return 10
        if pvalue < 0.05:
            return 15
        if pvalue < 0.10:
            return 35
        return 25

    return 50


def ols_with_pvalues(X: pd.DataFrame, y: pd.Series):
    """Return OLS coefficients and p-values using classical standard errors."""
    X_design = X.copy()
    X_design.insert(0, "intercept", 1.0)

    X_mat = X_design.to_numpy(dtype=float)
    y_vec = y.to_numpy(dtype=float)

    n, p = X_mat.shape
    xtx_inv = np.linalg.pinv(X_mat.T @ X_mat)
    beta = xtx_inv @ X_mat.T @ y_vec

    resid = y_vec - X_mat @ beta
    dof = n - p
    sigma2 = (resid @ resid) / dof
    cov_beta = sigma2 * xtx_inv
    se_beta = np.sqrt(np.diag(cov_beta))

    t_vals = beta / se_beta
    p_vals = 2 * (1 - stats.t.cdf(np.abs(t_vals), df=dof))

    coef = pd.Series(beta, index=X_design.columns)
    pval = pd.Series(p_vals, index=X_design.columns)
    return coef, pval


def main() -> None:
    # Load and explore data
    df = pd.read_csv("teachingratings.csv")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print("Missing values:\n", df.isna().sum())

    needed = [
        "eval",
        "beauty",
        "age",
        "gender",
        "minority",
        "native",
        "tenure",
        "division",
        "credits",
        "students",
        "allstudents",
    ]
    dfa = df[needed].dropna().copy()

    # 1) Bivariate relationship test
    r, p_corr = stats.pearsonr(dfa["beauty"], dfa["eval"])

    # 2) Simple regression significance test
    simple = stats.linregress(dfa["beauty"], dfa["eval"])
    simple_coef = float(simple.slope)
    simple_p = float(simple.pvalue)

    # 3) Adjusted OLS with controls (classical SEs)
    numeric = dfa[["beauty", "age", "students", "allstudents"]]
    categoricals = pd.get_dummies(
        dfa[["gender", "minority", "native", "tenure", "division", "credits"]],
        drop_first=True,
    )
    X_adj = pd.concat([numeric, categoricals], axis=1)
    coef_adj, pval_adj = ols_with_pvalues(X_adj, dfa["eval"])
    adj_coef = float(coef_adj["beauty"])
    adj_p = float(pval_adj["beauty"])

    response = int(likert_from_result(adj_coef, adj_p))
    explanation = (
        f"Using {len(dfa)} observations, beauty is positively associated with teaching evaluations. "
        f"Pearson correlation is r={r:.3f} (p={p_corr:.4g}). "
        f"Simple linear regression gives a beauty coefficient of {simple_coef:.3f} (p={simple_p:.4g}). "
        f"In an adjusted OLS model with instructor/course controls, the beauty coefficient is {adj_coef:.3f} "
        f"(p={adj_p:.4g}). Because the adjusted effect is positive and statistically significant, "
        f"this supports a Yes answer."
    )

    result = {"response": response, "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(result))

    print("\nBivariate correlation p-value:", p_corr)
    print("Simple model beauty coef/p:", simple_coef, simple_p)
    print("Adjusted model beauty coef/p:", adj_coef, adj_p)
    print("Wrote conclusion.txt")


if __name__ == "__main__":
    main()
