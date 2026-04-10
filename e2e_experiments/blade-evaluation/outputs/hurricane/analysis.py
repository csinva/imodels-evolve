import json
import numpy as np
import pandas as pd
from scipy import stats


def ols_with_inference(X: np.ndarray, y: np.ndarray):
    """Return OLS coefficients, standard errors, t-stats, p-values, and dof."""
    n, p = X.shape
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ (X.T @ y)
    resid = y - X @ beta
    dof = n - p
    sigma2 = (resid @ resid) / dof
    se = np.sqrt(np.diag(sigma2 * xtx_inv))
    t_stats = beta / se
    p_vals = 2 * stats.t.sf(np.abs(t_stats), dof)
    return beta, se, t_stats, p_vals, dof


def main():
    # 1) Load data
    df = pd.read_csv("hurricane.csv")

    # 2) Basic exploration
    print("Rows:", len(df))
    print("Columns:", list(df.columns))
    print("Missing values per column:\n", df.isna().sum())

    required = ["masfem", "alldeaths", "wind", "ndam15", "year"]
    dfa = df.dropna(subset=required).copy()
    print("Rows after dropping NA in required vars:", len(dfa))

    # Distribution transforms for skewed outcomes/exposures
    dfa["log_deaths"] = np.log1p(dfa["alldeaths"])
    dfa["log_ndam15"] = np.log1p(dfa["ndam15"])

    print("\nSummary stats (analysis vars):")
    print(dfa[["masfem", "alldeaths", "log_deaths", "wind", "ndam15", "log_ndam15", "year"]].describe())

    # 3) Significance tests for relationship
    # 3a) Unadjusted nonparametric test
    spearman = stats.spearmanr(dfa["masfem"], dfa["alldeaths"])
    print("\nSpearman correlation masfem vs alldeaths:")
    print("rho=%.4f, p=%.4g" % (spearman.statistic, spearman.pvalue))

    # 3b) Unadjusted linear relationship on log deaths
    pearson_log = stats.pearsonr(dfa["masfem"], dfa["log_deaths"])
    print("Pearson correlation masfem vs log_deaths:")
    print("r=%.4f, p=%.4g" % (pearson_log.statistic, pearson_log.pvalue))

    # 3c) Adjusted OLS: log_deaths ~ masfem + wind + log_ndam15 + year
    y = dfa["log_deaths"].to_numpy()
    X = np.column_stack(
        [
            np.ones(len(dfa)),
            dfa["masfem"].to_numpy(),
            dfa["wind"].to_numpy(),
            dfa["log_ndam15"].to_numpy(),
            dfa["year"].to_numpy(),
        ]
    )
    beta, se, t_stats, p_vals, dof = ols_with_inference(X, y)
    names = ["intercept", "masfem", "wind", "log_ndam15", "year"]

    print("\nAdjusted OLS coefficients (dependent var: log_deaths):")
    for i, name in enumerate(names):
        print(
            f"{name:>10s}: coef={beta[i]: .6f}, se={se[i]: .6f}, "
            f"t={t_stats[i]: .3f}, p={p_vals[i]:.4g}"
        )
    print("Degrees of freedom:", dof)

    # 3d) Partial correlation check (masfem and log_deaths controlling covariates)
    C = np.column_stack(
        [
            np.ones(len(dfa)),
            dfa["wind"].to_numpy(),
            dfa["log_ndam15"].to_numpy(),
            dfa["year"].to_numpy(),
        ]
    )
    by = np.linalg.lstsq(C, y, rcond=None)[0]
    bm = np.linalg.lstsq(C, dfa["masfem"].to_numpy(), rcond=None)[0]
    res_y = y - C @ by
    res_m = dfa["masfem"].to_numpy() - C @ bm
    partial = stats.pearsonr(res_m, res_y)

    print("\nPartial correlation (masfem vs log_deaths | wind, log_ndam15, year):")
    print("r=%.4f, p=%.4g" % (partial.statistic, partial.pvalue))

    # 4) Interpret in context of the research question
    # Hypothesis expects a positive and statistically significant relationship.
    masfem_coef = float(beta[1])
    masfem_p = float(p_vals[1])

    significant_positive = (
        (spearman.pvalue < 0.05 and spearman.statistic > 0)
        and (masfem_p < 0.05 and masfem_coef > 0)
    )

    if significant_positive:
        score = 85
        explanation = (
            "Evidence supports the claim: hurricanes with more feminine names show "
            "a statistically significant positive association with fatalities in both "
            "unadjusted and adjusted tests."
        )
    else:
        score = 10
        explanation = (
            "The data do not support the claim. The relationship between femininity "
            "of hurricane names (masfem) and deaths is not statistically significant "
            "in unadjusted testing (Spearman p={:.3f}) or in an adjusted OLS model "
            "controlling for wind, damage, and year (masfem p={:.3f}). "
            "Without significant evidence, the hypothesis receives a low score."
        ).format(spearman.pvalue, masfem_p)

    result = {"response": int(score), "explanation": explanation}

    # 5) Required output file containing ONLY JSON
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(result, f)

    print("\nWrote conclusion.txt")
    print(result)


if __name__ == "__main__":
    main()
