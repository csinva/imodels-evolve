import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge
import json

df = pd.read_csv("caschools.csv")

# Compute student-teacher ratio
df["str"] = df["students"] / df["teachers"]

# Academic performance: average of read and math
df["score"] = (df["read"] + df["math"]) / 2

print("Summary stats for student-teacher ratio:")
print(df["str"].describe())
print("\nSummary stats for average score:")
print(df["score"].describe())

# Pearson correlation
corr, pval = stats.pearsonr(df["str"], df["score"])
print(f"\nPearson r(str, score) = {corr:.4f}, p = {pval:.4e}")

# Spearman correlation
scorr, spval = stats.spearmanr(df["str"], df["score"])
print(f"Spearman r(str, score) = {scorr:.4f}, p = {spval:.4e}")

# OLS regression: score ~ str (simple)
X_simple = sm.add_constant(df["str"])
ols_simple = sm.OLS(df["score"], X_simple).fit()
print("\nSimple OLS (score ~ str):")
print(ols_simple.summary().tables[1])

# OLS regression controlling for confounders
controls = ["str", "lunch", "english", "income", "expenditure"]
X_full = sm.add_constant(df[controls])
ols_full = sm.OLS(df["score"], X_full).fit()
print("\nMultiple OLS (score ~ str + lunch + english + income + expenditure):")
print(ols_full.summary().tables[1])

# Quartile split analysis
df["str_quartile"] = pd.qcut(df["str"], 4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"])
quartile_means = df.groupby("str_quartile")["score"].mean()
print("\nMean score by STR quartile:")
print(quartile_means)

# ANOVA across quartiles
groups = [df[df["str_quartile"] == q]["score"].values for q in df["str_quartile"].cat.categories]
f_stat, anova_p = stats.f_oneway(*groups)
print(f"\nANOVA F={f_stat:.4f}, p={anova_p:.4e}")

# Interpret results
simple_coef = ols_simple.params["str"]
simple_p = ols_simple.pvalues["str"]
full_coef = ols_full.params["str"]
full_p = ols_full.pvalues["str"]

print(f"\nSimple regression: coef(str)={simple_coef:.4f}, p={simple_p:.4e}")
print(f"Multiple regression: coef(str)={full_coef:.4f}, p={full_p:.4e}")

# Decision: significant negative relationship in simple regression?
# Lower STR -> higher score means negative correlation
significant_simple = (pval < 0.05) and (corr < 0)
significant_full = (full_p < 0.05) and (full_coef < 0)

print(f"\nSimple model: significant negative relationship = {significant_simple}")
print(f"Full model: significant negative relationship = {significant_full}")

# Score: strong evidence if both simple and multiple regressions show significant negative relationship
if significant_simple and significant_full:
    response = 80
    explanation = (
        f"Yes, lower student-teacher ratio is significantly associated with higher academic performance. "
        f"Simple regression: r={corr:.3f}, p={pval:.2e}, coef={simple_coef:.3f}. "
        f"Multiple regression controlling for lunch, english, income, expenditure: coef={full_coef:.3f}, p={full_p:.2e}. "
        f"Both analyses show a significant negative relationship (lower STR -> higher scores). "
        f"Mean scores by STR quartile confirm the trend: Q1={quartile_means.iloc[0]:.1f}, Q4={quartile_means.iloc[3]:.1f}."
    )
elif significant_simple and not significant_full:
    response = 40
    explanation = (
        f"Partially. Simple regression shows a significant negative relationship (r={corr:.3f}, p={pval:.2e}), "
        f"but after controlling for confounders (lunch, english, income, expenditure), the relationship is no longer significant "
        f"(coef={full_coef:.3f}, p={full_p:.2e}). The raw correlation may be confounded by socioeconomic factors."
    )
elif not significant_simple:
    response = 15
    explanation = (
        f"No significant association found. Pearson r={corr:.3f}, p={pval:.2e}. "
        f"The student-teacher ratio does not significantly predict academic performance in this dataset."
    )
else:
    response = 60
    explanation = (
        f"Mixed evidence. Multiple regression significant (coef={full_coef:.3f}, p={full_p:.2e}) "
        f"but simple correlation weaker (r={corr:.3f}, p={pval:.2e})."
    )

conclusion = {"response": response, "explanation": explanation}
with open("conclusion.txt", "w") as f:
    json.dump(conclusion, f)

print(f"\nConclusion written: response={response}")
print(f"Explanation: {explanation}")
