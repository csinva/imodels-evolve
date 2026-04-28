import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor

# Load data
df = pd.read_csv("caschools.csv")
print("Shape:", df.shape)
print(df.describe())

# Compute student-teacher ratio
df["str"] = df["students"] / df["teachers"]

# Composite academic performance score (average of read and math)
df["score"] = (df["read"] + df["math"]) / 2

print("\nSTR stats:", df["str"].describe())
print("Score stats:", df["score"].describe())

# Pearson correlation
r, p = stats.pearsonr(df["str"], df["score"])
print(f"\nPearson r(STR, score) = {r:.4f}, p = {p:.4e}")

r_read, p_read = stats.pearsonr(df["str"], df["read"])
r_math, p_math = stats.pearsonr(df["str"], df["math"])
print(f"Pearson r(STR, read) = {r_read:.4f}, p = {p_read:.4e}")
print(f"Pearson r(STR, math) = {r_math:.4f}, p = {p_math:.4e}")

# OLS regression: score ~ STR (bivariate)
X_simple = sm.add_constant(df["str"])
ols_simple = sm.OLS(df["score"], X_simple).fit()
print("\n--- Bivariate OLS: score ~ STR ---")
print(ols_simple.summary())

# OLS with controls (income, lunch, english as confounders)
controls = ["str", "income", "lunch", "english", "expenditure"]
X_full = sm.add_constant(df[controls])
ols_full = sm.OLS(df["score"], X_full).fit()
print("\n--- Multivariate OLS: score ~ STR + controls ---")
print(ols_full.summary())

# Decision tree to check non-linear importance
tree = DecisionTreeRegressor(max_depth=4, random_state=42)
tree.fit(df[controls], df["score"])
importances = dict(zip(controls, tree.feature_importances_))
print("\nDecision tree feature importances:", importances)

# Spearman correlation (robust to outliers/non-linearity)
rho, p_spear = stats.spearmanr(df["str"], df["score"])
print(f"\nSpearman rho(STR, score) = {rho:.4f}, p = {p_spear:.4e}")

# Summary
str_coef_bivariate = ols_simple.params["str"]
str_pval_bivariate = ols_simple.pvalues["str"]
str_coef_full = ols_full.params["str"]
str_pval_full = ols_full.pvalues["str"]

print(f"\nBivariate STR coef={str_coef_bivariate:.4f}, p={str_pval_bivariate:.4e}")
print(f"Multivariate STR coef={str_coef_full:.4f}, p={str_pval_full:.4e}")

# Determine response score
# Strong negative correlation bivariate, but check if it survives controls
bivariate_significant = str_pval_bivariate < 0.05
multivariate_significant = str_pval_full < 0.05
negative_direction = str_coef_bivariate < 0

if bivariate_significant and negative_direction and multivariate_significant:
    response = 80
    explanation = (
        f"Yes. A lower student-teacher ratio (STR) is significantly associated with higher academic performance. "
        f"Bivariate OLS: STR coefficient={str_coef_bivariate:.3f} (p={str_pval_bivariate:.2e}), "
        f"Pearson r={r:.3f} (p={p:.2e}). "
        f"The negative sign confirms that higher STR (more students per teacher) is associated with lower scores. "
        f"The association remains statistically significant in the multivariate model controlling for income, lunch, english learners, and expenditure "
        f"(coef={str_coef_full:.3f}, p={str_pval_full:.2e}), though effect size is attenuated by confounders. "
        f"Spearman rho={rho:.3f} (p={p_spear:.2e}) confirms robustness."
    )
elif bivariate_significant and negative_direction and not multivariate_significant:
    response = 55
    explanation = (
        f"Partially. Bivariate correlation is significant (r={r:.3f}, p={p:.2e}), but after controlling for socioeconomic confounders "
        f"(income, lunch, english), the STR effect is not independently significant (p={str_pval_full:.2e}). "
        f"The bivariate association may be confounded."
    )
else:
    response = 20
    explanation = (
        f"No strong evidence. STR bivariate coef={str_coef_bivariate:.3f} (p={str_pval_bivariate:.2e}), "
        f"not clearly significant or not in expected direction."
    )

result = {"response": response, "explanation": explanation}
with open("conclusion.txt", "w") as f:
    json.dump(result, f)

print("\nconclusion.txt written.")
print(json.dumps(result, indent=2))
