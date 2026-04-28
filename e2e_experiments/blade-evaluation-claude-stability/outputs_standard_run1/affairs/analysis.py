import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv("affairs.csv")
print("Shape:", df.shape)
print(df.head())
print(df.describe())
print(df["children"].value_counts())

# Encode children as binary
df["children_bin"] = (df["children"] == "yes").astype(int)

# Summary stats by children group
print("\nMean affairs by children:")
print(df.groupby("children")["affairs"].describe())

# t-test: do those with children have significantly different affairs count?
group_yes = df[df["children"] == "yes"]["affairs"]
group_no = df[df["children"] == "no"]["affairs"]
t_stat, p_val = stats.ttest_ind(group_yes, group_no)
print(f"\nt-test: t={t_stat:.4f}, p={p_val:.4f}")
print(f"Mean affairs (children=yes): {group_yes.mean():.4f}")
print(f"Mean affairs (children=no): {group_no.mean():.4f}")

# Mann-Whitney U test (non-parametric, since affairs is not normal)
u_stat, p_mw = stats.mannwhitneyu(group_yes, group_no, alternative="two-sided")
print(f"\nMann-Whitney U test: U={u_stat:.4f}, p={p_mw:.4f}")

# OLS regression controlling for confounders
df["gender_bin"] = (df["gender"] == "male").astype(int)
features = ["children_bin", "age", "yearsmarried", "religiousness", "education", "occupation", "rating", "gender_bin"]
X = df[features]
X = sm.add_constant(X)
y = df["affairs"]
model = sm.OLS(y, X).fit()
print("\nOLS Regression Summary:")
print(model.summary())

children_coef = model.params["children_bin"]
children_pval = model.pvalues["children_bin"]
print(f"\nChildren coefficient: {children_coef:.4f}, p-value: {children_pval:.4f}")

# Logistic regression: any affair (binary)
df["had_affair"] = (df["affairs"] > 0).astype(int)
log_model = LogisticRegression(max_iter=1000)
log_model.fit(df[features], df["had_affair"])
coef_dict = dict(zip(features, log_model.coef_[0]))
print("\nLogistic regression coefficients:")
for k, v in coef_dict.items():
    print(f"  {k}: {v:.4f}")

# Correlation
corr_point_biserial, p_corr = stats.pointbiserialr(df["children_bin"], df["affairs"])
print(f"\nPoint-biserial correlation (children vs affairs): r={corr_point_biserial:.4f}, p={p_corr:.4f}")

# Determine conclusion
# Key findings:
# 1. Raw means comparison
# 2. t-test and Mann-Whitney p-values
# 3. OLS coefficient and p-value (controlling for confounders)

direction = "decrease" if children_coef < 0 else "increase"
significant = children_pval < 0.05
raw_higher_with_children = group_yes.mean() > group_no.mean()

print(f"\nRaw: children=yes mean={group_yes.mean():.3f}, children=no mean={group_no.mean():.3f}")
print(f"OLS: children coefficient={children_coef:.4f}, p={children_pval:.4f}")
print(f"Children associated with {direction} in affairs (controlling for confounders)")
print(f"Statistically significant in OLS: {significant}")

# The research question asks: Does having children DECREASE extramarital affairs?
# If children_coef < 0 and significant -> Yes (children decrease affairs)
# If children_coef > 0 or not significant -> No

if children_coef < 0 and significant:
    # Children significantly decrease affairs
    # Score based on effect size and significance
    response = 65
    explanation = (
        f"Having children is associated with a statistically significant DECREASE in extramarital affairs "
        f"when controlling for age, years married, religiousness, education, occupation, marriage rating, and gender. "
        f"OLS regression: children coefficient = {children_coef:.4f} (p = {children_pval:.4f}). "
        f"Raw means: with children = {group_yes.mean():.3f}, without children = {group_no.mean():.3f}. "
        f"Mann-Whitney U test p = {p_mw:.4f}. "
        f"The negative coefficient indicates that having children is associated with fewer extramarital affairs."
    )
elif children_coef > 0 and significant:
    # Children significantly increase affairs
    response = 15
    explanation = (
        f"Controlling for confounders, having children is associated with a statistically significant INCREASE "
        f"(not decrease) in extramarital affairs. OLS regression: children coefficient = {children_coef:.4f} "
        f"(p = {children_pval:.4f}). Raw means: with children = {group_yes.mean():.3f}, "
        f"without children = {group_no.mean():.3f}. Mann-Whitney U test p = {p_mw:.4f}. "
        f"The evidence does not support a decrease; if anything, children are associated with more affairs."
    )
else:
    # Not significant
    response = 30
    explanation = (
        f"The relationship between having children and extramarital affairs is not statistically significant "
        f"when controlling for confounders. OLS regression: children coefficient = {children_coef:.4f} "
        f"(p = {children_pval:.4f}). Raw means: with children = {group_yes.mean():.3f}, "
        f"without children = {group_no.mean():.3f}. Mann-Whitney U test p = {p_mw:.4f}. "
        f"There is insufficient evidence to conclude that having children decreases extramarital affairs."
    )

result = {"response": response, "explanation": explanation}
with open("conclusion.txt", "w") as f:
    json.dump(result, f)

print(f"\nConclusion written: response={response}")
print(f"Explanation: {explanation}")
