import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor

# Load data
df = pd.read_csv("caschools.csv")

# Compute student-teacher ratio
df["str"] = df["students"] / df["teachers"]

# Academic performance: average of read and math
df["score"] = (df["read"] + df["math"]) / 2

print("Dataset shape:", df.shape)
print("\nStudent-teacher ratio stats:")
print(df["str"].describe())
print("\nScore stats:")
print(df["score"].describe())

# Correlation between STR and scores
r_read, p_read = stats.pearsonr(df["str"], df["read"])
r_math, p_math = stats.pearsonr(df["str"], df["math"])
r_score, p_score = stats.pearsonr(df["str"], df["score"])

print(f"\nPearson r (STR vs read): {r_read:.4f}, p={p_read:.4e}")
print(f"Pearson r (STR vs math): {r_math:.4f}, p={p_math:.4e}")
print(f"Pearson r (STR vs score): {r_score:.4f}, p={p_score:.4e}")

# OLS regression: simple
X_simple = sm.add_constant(df["str"])
model_simple = sm.OLS(df["score"], X_simple).fit()
print("\n--- Simple OLS: score ~ STR ---")
print(model_simple.summary())

# Multiple regression controlling for confounders
features = ["str", "calworks", "lunch", "income", "english", "expenditure"]
X_multi = sm.add_constant(df[features].dropna())
y_multi = df.loc[df[features].dropna().index, "score"]
model_multi = sm.OLS(y_multi, X_multi).fit()
print("\n--- Multiple OLS: score ~ STR + controls ---")
print(model_multi.summary())

# Split into high/low STR groups and t-test
median_str = df["str"].median()
low_str = df[df["str"] <= median_str]["score"]
high_str = df[df["str"] > median_str]["score"]
t_stat, t_pval = stats.ttest_ind(low_str, high_str)
print(f"\nMedian STR: {median_str:.2f}")
print(f"Mean score (low STR): {low_str.mean():.2f}")
print(f"Mean score (high STR): {high_str.mean():.2f}")
print(f"t-test: t={t_stat:.4f}, p={t_pval:.4e}")

# Interpretable model: Decision Tree
X_dt = df[["str"]].values
y_dt = df["score"].values
dt = DecisionTreeRegressor(max_depth=3)
dt.fit(X_dt, y_dt)

# EBM model
try:
    from interpret.glassbox import ExplainableBoostingRegressor
    ebm = ExplainableBoostingRegressor(random_state=42)
    ebm.fit(df[features].values, df["score"].values)
    ebm_importances = dict(zip(features, ebm.term_importances()))
    print("\nEBM feature importances:")
    for feat, imp in sorted(ebm_importances.items(), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.4f}")
    ebm_available = True
except Exception as e:
    print(f"EBM not available: {e}")
    ebm_available = False

# Determine conclusion
# Simple OLS: is STR coefficient negative and significant?
str_coef_simple = model_simple.params["str"]
str_pval_simple = model_simple.pvalues["str"]
str_coef_multi = model_multi.params["str"]
str_pval_multi = model_multi.pvalues["str"]

print(f"\nSimple OLS STR coef: {str_coef_simple:.4f}, p={str_pval_simple:.4e}")
print(f"Multiple OLS STR coef: {str_coef_multi:.4f}, p={str_pval_multi:.4e}")

# Build response score
# Simple relationship: negative and significant -> supports "yes"
# With controls: if still significant, stronger evidence
simple_significant = str_pval_simple < 0.05 and str_coef_simple < 0
multi_significant = str_pval_multi < 0.05 and str_coef_multi < 0
ttest_significant = t_pval < 0.05 and low_str.mean() > high_str.mean()

print(f"\nSimple OLS significant & negative: {simple_significant}")
print(f"Multiple OLS significant & negative: {multi_significant}")
print(f"T-test significant & low>high: {ttest_significant}")

# Score determination
if simple_significant and multi_significant:
    response = 80
    explanation = (
        f"Both simple and multiple regression analyses show a statistically significant "
        f"negative relationship between student-teacher ratio (STR) and academic performance. "
        f"Simple OLS: coef={str_coef_simple:.3f}, p={str_pval_simple:.4f}. "
        f"Multiple OLS (controlling for calworks, lunch, income, english, expenditure): "
        f"coef={str_coef_multi:.3f}, p={str_pval_multi:.4f}. "
        f"Pearson r={r_score:.3f}, p={p_score:.4f}. "
        f"Districts with lower STR score higher on average ({low_str.mean():.1f} vs {high_str.mean():.1f}, "
        f"t-test p={t_pval:.4f}). "
        f"Evidence consistently supports a negative association between STR and test scores."
    )
elif simple_significant and not multi_significant:
    response = 45
    explanation = (
        f"Simple regression shows a significant negative relationship (coef={str_coef_simple:.3f}, p={str_pval_simple:.4f}), "
        f"but this becomes non-significant when controlling for confounders like income and poverty "
        f"(multiple OLS: coef={str_coef_multi:.3f}, p={str_pval_multi:.4f}). "
        f"The raw association may be confounded by socioeconomic factors."
    )
else:
    response = 20
    explanation = (
        f"The analysis does not find a statistically significant relationship between STR and academic performance. "
        f"Simple OLS: coef={str_coef_simple:.3f}, p={str_pval_simple:.4f}. "
        f"Multiple OLS: coef={str_coef_multi:.3f}, p={str_pval_multi:.4f}."
    )

conclusion = {"response": response, "explanation": explanation}
with open("conclusion.txt", "w") as f:
    json.dump(conclusion, f)

print(f"\nConclusion written: response={response}")
print(f"Explanation: {explanation}")
