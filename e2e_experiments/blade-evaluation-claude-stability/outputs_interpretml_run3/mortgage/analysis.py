import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv("mortgage.csv")
print("Shape:", df.shape)
print(df.head())
print(df.describe())

# Drop rows with missing female or deny values for analysis
df_valid = df.dropna(subset=["female", "deny"])
print(f"\nRows after dropping NaN in female/deny: {len(df_valid)}")

# Key columns
print("\nDeny rate by gender:")
deny_by_gender = df_valid.groupby("female")["deny"].agg(["mean", "count"])
print(deny_by_gender)

female_deny = df_valid[df_valid["female"] == 1]["deny"]
male_deny = df_valid[df_valid["female"] == 0]["deny"]

print(f"\nFemale denial rate: {female_deny.mean():.4f} (n={len(female_deny)})")
print(f"Male denial rate:   {male_deny.mean():.4f} (n={len(male_deny)})")

# Chi-square test for independence
contingency = pd.crosstab(df_valid["female"], df_valid["deny"])
print("\nContingency table:")
print(contingency)
chi2, p_chi2, dof, expected = stats.chi2_contingency(contingency)
print(f"Chi-square: {chi2:.4f}, p-value: {p_chi2:.4f}, dof: {dof}")

# t-test on deny rates
t_stat, p_ttest = stats.ttest_ind(female_deny, male_deny)
print(f"\nt-test: t={t_stat:.4f}, p={p_ttest:.4f}")

# Logistic regression: univariate (female -> deny)
X_uni = sm.add_constant(df_valid["female"])
logit_uni = sm.Logit(df_valid["deny"], X_uni).fit(disp=False)
print("\nUnivariate logistic regression (female -> deny):")
print(logit_uni.summary2())

# Multivariate logistic regression controlling for confounders
features = ["female", "black", "housing_expense_ratio", "self_employed",
            "married", "mortgage_credit", "consumer_credit", "bad_history",
            "PI_ratio", "loan_to_value", "denied_PMI"]
df_clean = df_valid[features + ["deny"]].dropna()
X_multi = sm.add_constant(df_clean[features])
logit_multi = sm.Logit(df_clean["deny"], X_multi).fit(disp=False)
print("\nMultivariate logistic regression:")
print(logit_multi.summary2())

female_coef = logit_multi.params["female"]
female_pval = logit_multi.pvalues["female"]
female_or = np.exp(female_coef)
print(f"\nFemale coefficient: {female_coef:.4f}, OR: {female_or:.4f}, p-value: {female_pval:.4f}")

# Use interpret EBM for interpretable nonlinear model
try:
    from interpret.glassbox import ExplainableBoostingClassifier
    ebm = ExplainableBoostingClassifier(random_state=42)
    X_ebm = df_clean[features]
    y_ebm = df_clean["deny"]
    ebm.fit(X_ebm, y_ebm)
    importances = dict(zip(features, ebm.term_importances()))
    print("\nEBM Feature importances:")
    for f, imp in sorted(importances.items(), key=lambda x: -x[1]):
        print(f"  {f}: {imp:.4f}")
    female_importance = importances.get("female", None)
    print(f"\nEBM female importance: {female_importance:.4f}")
except Exception as e:
    print(f"EBM failed: {e}")
    female_importance = None

# Conclusion
# Determine score based on statistical significance and effect direction
# Univariate: female denial rate vs male
diff = female_deny.mean() - male_deny.mean()
print(f"\nDifference in denial rates (female - male): {diff:.4f}")

# If p-value from multivariate logistic is significant, gender affects approval
# The question: does gender affect mortgage approval?
# Score: 0=strong No, 100=strong Yes
# We look at multivariate p-value for female coefficient

if female_pval < 0.05:
    if female_coef < 0:  # being female reduces denial = increases approval
        score = 70
        direction = "females are less likely to be denied (more likely approved)"
    else:
        score = 70
        direction = "females are more likely to be denied (less likely approved)"
else:
    # Not significant in multivariate model
    # Check univariate
    uni_female_pval = logit_uni.pvalues["female"]
    if uni_female_pval < 0.05:
        score = 45
        direction = "univariate association exists but disappears when controlling for confounders"
    else:
        score = 20
        direction = "no significant association between gender and mortgage approval"

explanation = (
    f"Research question: Does gender affect mortgage approval? "
    f"Univariate analysis: female denial rate = {female_deny.mean():.3f}, male denial rate = {male_deny.mean():.3f}, "
    f"difference = {diff:.3f}. "
    f"Chi-square test: chi2={chi2:.3f}, p={p_chi2:.4f}. "
    f"Multivariate logistic regression controlling for race, credit scores, debt ratios, etc.: "
    f"female coefficient = {female_coef:.4f} (OR={female_or:.4f}), p-value = {female_pval:.4f}. "
    f"Conclusion: {direction}. "
    f"The multivariate p-value {'is' if female_pval < 0.05 else 'is not'} statistically significant (p={'<0.05' if female_pval < 0.05 else '>0.05'}), "
    f"suggesting gender {'does' if female_pval < 0.05 else 'does not'} independently affect mortgage approval after controlling for creditworthiness factors."
)

result = {"response": score, "explanation": explanation}
print("\nFinal result:", result)

with open("conclusion.txt", "w") as f:
    json.dump(result, f)

print("conclusion.txt written successfully.")
