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

# Research question: How does gender affect mortgage approval?
# female=1 means female, deny=1 means denied (accept=0)

df_gender = df.dropna(subset=["female"])

print("\n--- Denial rates by gender ---")
denial_by_gender = df_gender.groupby("female")["deny"].agg(["mean", "count"])
denial_by_gender.index = ["Male", "Female"]
print(denial_by_gender)

male_denials = df_gender[df_gender["female"] == 0]["deny"]
female_denials = df_gender[df_gender["female"] == 1]["deny"]

print(f"\nMale denial rate: {male_denials.mean():.4f} (n={len(male_denials)})")
print(f"Female denial rate: {female_denials.mean():.4f} (n={len(female_denials)})")

# Chi-square test
from scipy.stats import chi2_contingency
ct = pd.crosstab(df_gender["female"], df_gender["deny"])
chi2, p_chi2, dof, expected = chi2_contingency(ct)
print(f"\nChi-square test: chi2={chi2:.4f}, p={p_chi2:.4f}")

# Two-proportion z-test (equivalent to chi-square)
t_stat, p_ttest = stats.ttest_ind(female_denials, male_denials)
print(f"T-test: t={t_stat:.4f}, p={p_ttest:.4f}")

# Logistic regression - univariate (just gender)
print("\n--- Univariate logistic regression (deny ~ female) ---")
X_uni = sm.add_constant(df_gender[["female"]])
logit_uni = sm.Logit(df_gender["deny"], X_uni).fit(disp=0)
print(logit_uni.summary())

# Logistic regression - multivariate (controlling for confounders)
print("\n--- Multivariate logistic regression (controlling for confounders) ---")
features = ["female", "black", "housing_expense_ratio", "self_employed", "married",
            "mortgage_credit", "consumer_credit", "bad_history", "PI_ratio",
            "loan_to_value", "denied_PMI"]
df_clean = df_gender[features + ["deny"]].dropna()
X_multi = sm.add_constant(df_clean[features])
logit_multi = sm.Logit(df_clean["deny"], X_multi).fit(disp=0)
print(logit_multi.summary())

# Extract female coefficient and p-value
female_coef = logit_multi.params["female"]
female_pval = logit_multi.pvalues["female"]
female_coef_uni = logit_uni.params["female"]
female_pval_uni = logit_uni.pvalues["female"]

print(f"\nUnivariate - female coef: {female_coef_uni:.4f}, p={female_pval_uni:.4f}")
print(f"Multivariate - female coef: {female_coef:.4f}, p={female_pval:.4f}")

# Odds ratios
print(f"\nUnivariate odds ratio for female: {np.exp(female_coef_uni):.4f}")
print(f"Multivariate odds ratio for female: {np.exp(female_coef):.4f}")

# Determine response score
# Significant effect of gender on mortgage approval?
alpha = 0.05
significant_uni = female_pval_uni < alpha
significant_multi = female_pval < alpha

print(f"\nUnivariate significant: {significant_uni}")
print(f"Multivariate significant: {significant_multi}")

male_denial_rate = male_denials.mean()
female_denial_rate = female_denials.mean()

# Determine response
# If female has significantly lower denial rate (positive effect = gender matters)
# Univariate: is there a raw difference?
# Multivariate: does it hold after controlling?

if significant_uni and significant_multi:
    # Strong evidence gender affects approval
    if female_coef < 0:
        # Being female associated with lower denial (more approvals)
        response = 75
        direction = "females are less likely to be denied"
    else:
        response = 75
        direction = "females are more likely to be denied"
elif significant_uni and not significant_multi:
    # Effect disappears after controlling -> likely confounded
    response = 35
    direction = "raw difference but not significant after controlling for confounders"
elif not significant_uni:
    response = 20
    direction = "no significant raw difference"
else:
    response = 50
    direction = "mixed evidence"

explanation = (
    f"The research question asks how gender affects mortgage approval. "
    f"Male denial rate: {male_denial_rate:.3f}, Female denial rate: {female_denial_rate:.3f}. "
    f"Chi-square test: p={p_chi2:.4f}. "
    f"Univariate logistic regression: female coef={female_coef_uni:.4f} (OR={np.exp(female_coef_uni):.3f}), p={female_pval_uni:.4f}. "
    f"Multivariate logistic regression (controlling for credit history, debt ratios, etc.): "
    f"female coef={female_coef:.4f} (OR={np.exp(female_coef):.3f}), p={female_pval:.4f}. "
    f"Conclusion: {direction}. "
    f"{'Gender does have a statistically significant effect on mortgage approval.' if (significant_uni or significant_multi) else 'Gender does not have a statistically significant effect on mortgage approval.'}"
)

print(f"\nResponse score: {response}")
print(f"Explanation: {explanation}")

result = {"response": response, "explanation": explanation}

with open("conclusion.txt", "w") as f:
    json.dump(result, f)

print("\nconclusion.txt written successfully.")
