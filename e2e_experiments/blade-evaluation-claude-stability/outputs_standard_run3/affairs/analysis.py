import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("affairs.csv")
print(df.head())
print(df.describe())
print(df["children"].value_counts())

# Encode children as binary
df["children_bin"] = (df["children"] == "yes").astype(int)

# Group comparison
with_children = df[df["children_bin"] == 1]["affairs"]
without_children = df[df["children_bin"] == 0]["affairs"]

print(f"\nMean affairs (with children): {with_children.mean():.4f}")
print(f"Mean affairs (without children): {without_children.mean():.4f}")

# t-test
t_stat, p_value = stats.ttest_ind(with_children, without_children)
print(f"\nT-test: t={t_stat:.4f}, p={p_value:.4f}")

# Mann-Whitney U (non-parametric, since affairs is skewed)
u_stat, p_mann = stats.mannwhitneyu(with_children, without_children, alternative="two-sided")
print(f"Mann-Whitney U: U={u_stat:.4f}, p={p_mann:.4f}")

# OLS regression controlling for confounders
df["gender_bin"] = (df["gender"] == "male").astype(int)
X = df[["children_bin", "age", "yearsmarried", "religiousness", "education", "occupation", "rating", "gender_bin"]]
X_const = sm.add_constant(X)
y = df["affairs"]
ols = sm.OLS(y, X_const).fit()
print("\nOLS Summary:")
print(ols.summary())

children_coef = ols.params["children_bin"]
children_pval = ols.pvalues["children_bin"]
print(f"\nChildren coefficient: {children_coef:.4f}, p-value: {children_pval:.4f}")

# Ridge for feature importances
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
ridge = Ridge(alpha=1.0)
ridge.fit(X_scaled, y)
feat_imp = pd.Series(ridge.coef_, index=X.columns).sort_values()
print("\nRidge coefficients (standardized):")
print(feat_imp)

# Interpretation
# The question is: does having children DECREASE affairs?
# If children_coef > 0 and significant: children INCREASES affairs (No to decrease)
# If children_coef < 0 and significant: children DECREASES affairs (Yes to decrease)
# If not significant: no clear effect

# Raw means: with_children mean vs without_children mean
raw_diff = with_children.mean() - without_children.mean()
print(f"\nRaw difference (with - without): {raw_diff:.4f}")

# Summary: children_bin coefficient in OLS (controlled)
print(f"\nControlled effect of children: coef={children_coef:.4f}, p={children_pval:.4f}")

# Score determination:
# - Raw means show with-children have MORE affairs
# - OLS controlled shows direction after confounders
# - If children increases affairs (positive coef, significant), response should be low (No)
# - If children decreases affairs (negative coef, significant), response should be high (Yes)

if children_pval < 0.05:
    if children_coef < 0:
        response = 75
        explanation = (
            f"Children significantly DECREASE extramarital affairs. "
            f"OLS coef={children_coef:.3f} (p={children_pval:.4f}). "
            f"Raw means: with_children={with_children.mean():.3f}, without={without_children.mean():.3f}. "
            f"Mann-Whitney p={p_mann:.4f}."
        )
    else:
        response = 20
        explanation = (
            f"Children are associated with MORE extramarital affairs (opposite of question hypothesis). "
            f"OLS coef={children_coef:.3f} (p={children_pval:.4f}). "
            f"Raw means: with_children={with_children.mean():.3f}, without={without_children.mean():.3f}. "
            f"Mann-Whitney p={p_mann:.4f}."
        )
else:
    response = 35
    explanation = (
        f"No statistically significant effect of children on extramarital affairs after controlling for confounders. "
        f"OLS coef={children_coef:.3f} (p={children_pval:.4f}). "
        f"Raw means: with_children={with_children.mean():.3f}, without={without_children.mean():.3f}. "
        f"Mann-Whitney p={p_mann:.4f}."
    )

result = {"response": response, "explanation": explanation}
with open("conclusion.txt", "w") as f:
    json.dump(result, f)

print(f"\nConclusion: {result}")
