import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("mortgage.csv")
df = df.dropna(subset=["female", "accept"])
print("Shape after dropping NaN in female/accept:", df.shape)
print(df.head())
print(df.describe())

# Research question: How does gender affect mortgage approval?
# female=1 means female, deny=1 means denied, accept=1 means accepted

print("\n--- Approval rates by gender ---")
gender_groups = df.groupby("female")["accept"].agg(["mean", "count"])
print(gender_groups)

female_accept = df[df["female"] == 1]["accept"]
male_accept = df[df["female"] == 0]["accept"]

print(f"\nFemale acceptance rate: {female_accept.mean():.4f} (n={len(female_accept)})")
print(f"Male acceptance rate:   {male_accept.mean():.4f} (n={len(male_accept)})")

# Chi-square test on accept vs gender
ct = pd.crosstab(df["female"], df["accept"])
chi2, p_chi2, dof, expected = stats.chi2_contingency(ct)
print(f"\nChi-square test: chi2={chi2:.4f}, p={p_chi2:.6f}")

# t-test
t_stat, p_ttest = stats.ttest_ind(female_accept, male_accept)
print(f"T-test: t={t_stat:.4f}, p={p_ttest:.6f}")

# Logistic regression: univariate
X_uni = sm.add_constant(df[["female"]])
logit_uni = sm.Logit(df["accept"], X_uni).fit(disp=False)
print("\n--- Univariate logistic regression (accept ~ female) ---")
print(logit_uni.summary2())

# Multivariate logistic regression controlling for confounders
features = ["female", "black", "housing_expense_ratio", "self_employed",
            "married", "mortgage_credit", "consumer_credit", "bad_history",
            "PI_ratio", "loan_to_value", "denied_PMI"]
df_clean = df[features + ["accept"]].dropna()

X_multi = sm.add_constant(df_clean[features])
logit_multi = sm.Logit(df_clean["accept"], X_multi).fit(disp=False)
print("\n--- Multivariate logistic regression (accept ~ female + controls) ---")
print(logit_multi.summary2())

female_coef = logit_multi.params["female"]
female_pval = logit_multi.pvalues["female"]
female_ci = logit_multi.conf_int().loc["female"]
print(f"\nFemale coefficient: {female_coef:.4f}, p-value: {female_pval:.6f}")
print(f"95% CI: [{female_ci[0]:.4f}, {female_ci[1]:.4f}]")

# Determine response
# Univariate effect
diff = female_accept.mean() - male_accept.mean()
print(f"\nRaw difference (female - male accept rate): {diff:.4f}")

# Decision
# p-value from multivariate logistic (controlling for confounders)
if female_pval < 0.05:
    significant = True
else:
    significant = False

if significant:
    if female_coef > 0:
        direction = "females are more likely to be approved"
        response = 70
    else:
        direction = "females are less likely to be approved"
        response = 70
else:
    direction = "no statistically significant gender effect after controlling for other factors"
    # Check univariate
    if p_chi2 < 0.05:
        response = 45
        direction += " (but univariate difference exists, likely confounded)"
    else:
        response = 20

explanation = (
    f"Research question: Does gender affect mortgage approval? "
    f"Female acceptance rate: {female_accept.mean():.3f}, Male acceptance rate: {male_accept.mean():.3f} "
    f"(raw diff={diff:+.3f}). "
    f"Chi-square test p={p_chi2:.4f}. "
    f"Multivariate logistic regression controlling for race, credit scores, debt ratios, etc.: "
    f"female coefficient={female_coef:.4f}, p={female_pval:.4f}, "
    f"95% CI=[{female_ci[0]:.4f}, {female_ci[1]:.4f}]. "
    f"Conclusion: {direction}."
)

print(f"\nResponse: {response}")
print(f"Explanation: {explanation}")

result = {"response": response, "explanation": explanation}
with open("conclusion.txt", "w") as f:
    json.dump(result, f)

print("\nconclusion.txt written.")
