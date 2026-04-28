import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv("affairs.csv")
print("Shape:", df.shape)
print(df.head())
print(df.describe())

# Encode children
df["children_bin"] = (df["children"] == "yes").astype(int)
df["had_affair"] = (df["affairs"] > 0).astype(int)

# Summary by children group
print("\nAffairs by children group:")
print(df.groupby("children")["affairs"].describe())
print("\nMean affairs:")
print(df.groupby("children")["affairs"].mean())

with_children = df[df["children"] == "yes"]["affairs"]
without_children = df[df["children"] == "no"]["affairs"]

print(f"\nWith children: n={len(with_children)}, mean={with_children.mean():.4f}, median={with_children.median():.1f}")
print(f"Without children: n={len(without_children)}, mean={without_children.mean():.4f}, median={without_children.median():.1f}")

# Mann-Whitney U test (affairs is ordinal/skewed)
u_stat, p_mw = stats.mannwhitneyu(with_children, without_children, alternative="two-sided")
print(f"\nMann-Whitney U: U={u_stat:.1f}, p={p_mw:.4f}")

# t-test
t_stat, p_t = stats.ttest_ind(with_children, without_children)
print(f"t-test: t={t_stat:.4f}, p={p_t:.4f}")

# Chi-square: had affair vs children
ct = pd.crosstab(df["children"], df["had_affair"])
print("\nCrosstab (children vs had_affair):")
print(ct)
chi2, p_chi2, dof, expected = stats.chi2_contingency(ct)
print(f"Chi-square: chi2={chi2:.4f}, p={p_chi2:.4f}")

# OLS regression controlling for confounders
df["gender_bin"] = (df["gender"] == "male").astype(int)
X = df[["children_bin", "age", "yearsmarried", "religiousness", "education", "occupation", "rating", "gender_bin"]]
X = sm.add_constant(X)
y = df["affairs"]
ols = sm.OLS(y, X).fit()
print("\nOLS Regression:")
print(ols.summary())

children_coef = ols.params["children_bin"]
children_pval = ols.pvalues["children_bin"]
print(f"\nChildren coefficient: {children_coef:.4f}, p-value: {children_pval:.4f}")

# Logistic regression for probability of any affair
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(df[["children_bin", "age", "yearsmarried", "religiousness", "education", "occupation", "rating", "gender_bin"]], df["had_affair"])
children_lr_coef = log_reg.coef_[0][0]
print(f"\nLogistic regression children coefficient (log-odds): {children_lr_coef:.4f}")

# Interpret: EBM
try:
    from interpret.glassbox import ExplainableBoostingRegressor
    ebm = ExplainableBoostingRegressor(random_state=42)
    ebm.fit(df[["children_bin", "age", "yearsmarried", "religiousness", "education", "occupation", "rating", "gender_bin"]], df["affairs"])
    importances = dict(zip(["children_bin", "age", "yearsmarried", "religiousness", "education", "occupation", "rating", "gender_bin"], ebm.term_importances()))
    print("\nEBM feature importances:", importances)
    ebm_children_importance = importances["children_bin"]
except Exception as e:
    print(f"EBM failed: {e}")
    ebm_children_importance = None

# Summary
print("\n=== SUMMARY ===")
print(f"Mean affairs with children: {with_children.mean():.4f}")
print(f"Mean affairs without children: {without_children.mean():.4f}")
print(f"Difference: {with_children.mean() - without_children.mean():.4f}")
print(f"OLS children coefficient: {children_coef:.4f}, p={children_pval:.4f}")
print(f"Mann-Whitney p: {p_mw:.4f}")
print(f"Chi-square p: {p_chi2:.4f}")

# Determine response
# If children_coef is negative and significant -> children decrease affairs -> "Yes" (high score)
# Higher affairs in those with children (positive coef) -> "No" (low score)
# Unadjusted: with children tend to have higher affairs (confounded by duration of marriage)
# Adjusted: OLS controls for confounders

significant = children_pval < 0.05
decreases = children_coef < 0  # negative means children associated with fewer affairs

if significant and decreases:
    response = 75
    explanation = (
        f"The OLS regression controlling for age, years married, religiousness, education, "
        f"occupation, marriage rating, and gender shows that having children is associated with "
        f"a {abs(children_coef):.4f} decrease in affairs (p={children_pval:.4f}), which is statistically significant. "
        f"This suggests children do decrease extramarital affair engagement."
    )
elif significant and not decreases:
    response = 20
    explanation = (
        f"The OLS regression shows having children is associated with MORE affairs (coef={children_coef:.4f}, p={children_pval:.4f}), "
        f"not fewer. Unadjusted mean: with children={with_children.mean():.4f}, without={without_children.mean():.4f}. "
        f"Evidence does NOT support that children decrease affairs."
    )
else:
    # Not significant - check direction
    if decreases:
        response = 35
        explanation = (
            f"After controlling for confounders, having children shows a slight negative association with affairs "
            f"(coef={children_coef:.4f}) but this is NOT statistically significant (p={children_pval:.4f}). "
            f"Unadjusted: with children mean={with_children.mean():.4f}, without={without_children.mean():.4f}. "
            f"No strong evidence that children decrease extramarital affairs."
        )
    else:
        response = 25
        explanation = (
            f"After controlling for confounders, having children shows no significant association with affairs "
            f"(coef={children_coef:.4f}, p={children_pval:.4f}). "
            f"Unadjusted: with children mean={with_children.mean():.4f}, without={without_children.mean():.4f}. "
            f"No evidence that children decrease extramarital affairs."
        )

result = {"response": response, "explanation": explanation}
with open("conclusion.txt", "w") as f:
    json.dump(result, f)

print(f"\nconclusion.txt written: {result}")
