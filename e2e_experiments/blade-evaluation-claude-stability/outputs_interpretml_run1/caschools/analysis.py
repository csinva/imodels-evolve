import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("caschools.csv")

# Compute student-teacher ratio
df["str"] = df["students"] / df["teachers"]

# Academic performance: average of read and math scores
df["score"] = (df["read"] + df["math"]) / 2

print("=== Summary Statistics ===")
print(df[["str", "score", "read", "math"]].describe())

# Correlation between STR and test scores
r_score, p_score = stats.pearsonr(df["str"], df["score"])
r_read, p_read = stats.pearsonr(df["str"], df["read"])
r_math, p_math = stats.pearsonr(df["str"], df["math"])

print(f"\nCorrelation STR vs avg score: r={r_score:.4f}, p={p_score:.4e}")
print(f"Correlation STR vs read:      r={r_read:.4f}, p={p_read:.4e}")
print(f"Correlation STR vs math:      r={r_math:.4f}, p={p_math:.4e}")

# OLS regression: score ~ str (bivariate)
X_simple = sm.add_constant(df["str"])
ols_simple = sm.OLS(df["score"], X_simple).fit()
print("\n=== OLS: score ~ str ===")
print(ols_simple.summary())

# Controlled OLS: score ~ str + lunch + english + income (common confounders)
controls = ["str", "lunch", "english", "income"]
X_ctrl = sm.add_constant(df[controls].dropna())
y_ctrl = df["score"].loc[X_ctrl.index]
ols_ctrl = sm.OLS(y_ctrl, X_ctrl).fit()
print("\n=== OLS: score ~ str + lunch + english + income ===")
print(ols_ctrl.summary())

# Split by high/low STR (median split) and t-test
median_str = df["str"].median()
high_str = df.loc[df["str"] > median_str, "score"]
low_str = df.loc[df["str"] <= median_str, "score"]
t_stat, t_p = stats.ttest_ind(low_str, high_str)
print(f"\nMedian STR={median_str:.2f}")
print(f"Low-STR mean score:  {low_str.mean():.2f}")
print(f"High-STR mean score: {high_str.mean():.2f}")
print(f"T-test: t={t_stat:.4f}, p={t_p:.4e}")

# Ridge regression with standardized features to get relative importance
feat_cols = ["str", "lunch", "english", "income", "expenditure", "calworks"]
df_clean = df[feat_cols + ["score"]].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean[feat_cols])
ridge = Ridge(alpha=1.0)
ridge.fit(X_scaled, df_clean["score"])
print("\n=== Ridge coefficients (standardized) ===")
for col, coef in zip(feat_cols, ridge.coef_):
    print(f"  {col}: {coef:.4f}")

# Conclusion
str_coef_bivariate = ols_simple.params["str"]
str_pval_bivariate = ols_simple.pvalues["str"]
str_coef_ctrl = ols_ctrl.params["str"]
str_pval_ctrl = ols_ctrl.pvalues["str"]

# Bivariate: negative coefficient + significant => lower STR → higher score
bivariate_sig = str_pval_bivariate < 0.05
controlled_sig = str_pval_ctrl < 0.05
negative_bivariate = str_coef_bivariate < 0
negative_ctrl = str_coef_ctrl < 0

print(f"\nBivariate: coef={str_coef_bivariate:.4f}, p={str_pval_bivariate:.4e}, sig={bivariate_sig}, negative={negative_bivariate}")
print(f"Controlled: coef={str_coef_ctrl:.4f}, p={str_pval_ctrl:.4e}, sig={controlled_sig}, negative={negative_ctrl}")

# Score assignment
if bivariate_sig and negative_bivariate and controlled_sig and negative_ctrl:
    response = 80
    explanation = (
        f"Both bivariate and controlled OLS regressions show a statistically significant "
        f"negative relationship between student-teacher ratio (STR) and academic performance. "
        f"Bivariate: coef={str_coef_bivariate:.3f}, p={str_pval_bivariate:.2e}; "
        f"Controlled (+ lunch, english, income): coef={str_coef_ctrl:.3f}, p={str_pval_ctrl:.2e}. "
        f"Pearson r={r_score:.3f} (p={p_score:.2e}). "
        f"Districts with lower STR score higher on average (low-STR mean={low_str.mean():.1f} vs high-STR mean={high_str.mean():.1f}). "
        f"The evidence supports that lower student-teacher ratio is associated with higher academic performance, "
        f"though the effect size diminishes when controlling for socioeconomic confounders."
    )
elif bivariate_sig and negative_bivariate:
    response = 65
    explanation = (
        f"Bivariate OLS shows a significant negative relationship (coef={str_coef_bivariate:.3f}, p={str_pval_bivariate:.2e}), "
        f"but this becomes non-significant after controlling for confounders (coef={str_coef_ctrl:.3f}, p={str_pval_ctrl:.2e}), "
        f"suggesting the raw association may be confounded by socioeconomic factors."
    )
else:
    response = 30
    explanation = (
        f"No statistically significant negative association found. "
        f"Bivariate: coef={str_coef_bivariate:.3f}, p={str_pval_bivariate:.2e}. "
        f"Controlled: coef={str_coef_ctrl:.3f}, p={str_pval_ctrl:.2e}."
    )

result = {"response": response, "explanation": explanation}
with open("conclusion.txt", "w") as f:
    json.dump(result, f)

print(f"\nconclusion.txt written: response={response}")
