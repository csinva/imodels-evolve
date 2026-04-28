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
# female=1 means female applicant, deny=1 means denied

print("\n--- Denial rates by gender ---")
df_gender = df.dropna(subset=["female"])
denial_by_gender = df_gender.groupby("female")["deny"].agg(["mean", "count"])
denial_by_gender.index = ["Male", "Female"]
print(denial_by_gender)

male_deny = df_gender[df_gender["female"] == 0]["deny"]
female_deny = df_gender[df_gender["female"] == 1]["deny"]

print(f"\nMale denial rate: {male_deny.mean():.4f} (n={len(male_deny)})")
print(f"Female denial rate: {female_deny.mean():.4f} (n={len(female_deny)})")

# Chi-square test
from scipy.stats import chi2_contingency
contingency = pd.crosstab(df_gender["female"], df_gender["deny"])
print("\nContingency table:")
print(contingency)
chi2, p_chi2, dof, expected = chi2_contingency(contingency)
print(f"Chi-square test: chi2={chi2:.4f}, p={p_chi2:.6f}")

# t-test on denial rates
t_stat, p_ttest = stats.ttest_ind(female_deny, male_deny)
print(f"T-test: t={t_stat:.4f}, p={p_ttest:.6f}")

# Logistic regression: univariate (gender only)
X_uni = sm.add_constant(df_gender[["female"]])
logit_uni = sm.Logit(df_gender["deny"], X_uni).fit(disp=0)
print("\n--- Univariate logistic regression (deny ~ female) ---")
print(logit_uni.summary())

# Multivariate logistic regression controlling for creditworthiness
features = ["female", "black", "housing_expense_ratio", "self_employed", "married",
            "mortgage_credit", "consumer_credit", "bad_history", "PI_ratio",
            "loan_to_value", "denied_PMI"]
df_model = df[features + ["deny"]].dropna()
X_multi = sm.add_constant(df_model[features])
logit_multi = sm.Logit(df_model["deny"], X_multi).fit(disp=0)
print("\n--- Multivariate logistic regression ---")
print(logit_multi.summary())

female_coef = logit_multi.params["female"]
female_pval = logit_multi.pvalues["female"]
print(f"\nFemale coefficient: {female_coef:.4f}, p-value: {female_pval:.6f}")

# EBM for interpretable feature importances
try:
    from interpret.glassbox import ExplainableBoostingClassifier
    ebm = ExplainableBoostingClassifier(random_state=42)
    ebm.fit(df_model[features], df_model["deny"])
    importances = dict(zip(features, ebm.term_importances()))
    print("\n--- EBM Feature Importances ---")
    for k, v in sorted(importances.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v:.4f}")
    female_importance = importances.get("female", 0)
except Exception as e:
    print(f"EBM failed: {e}")
    female_importance = None

# Summarize findings
raw_diff = female_deny.mean() - male_deny.mean()
print(f"\n--- Summary ---")
print(f"Raw denial rate difference (female - male): {raw_diff:.4f}")
print(f"Univariate p-value (chi2): {p_chi2:.6f}")
print(f"Multivariate logistic female p-value: {female_pval:.6f}")
print(f"Multivariate female coef: {female_coef:.4f}")

# Determine score
# If p < 0.05 in multivariate: gender has a significant effect after controls
significant_after_controls = female_pval < 0.05
significant_univariate = p_chi2 < 0.05

# Raw denial rates
male_rate = male_deny.mean()
female_rate = female_deny.mean()

if significant_after_controls:
    # Gender matters even after controlling for creditworthiness
    if female_coef < 0:
        # Negative coef means female -> less denial -> banks favor women
        score = 70
        explanation = (
            f"Gender significantly affects mortgage approval (multivariate logistic p={female_pval:.4f}). "
            f"Female applicants have a lower denial rate ({female_rate:.3f}) vs males ({male_rate:.3f}). "
            f"After controlling for creditworthiness variables, the female coefficient is {female_coef:.4f} (p={female_pval:.4f}), "
            f"indicating females are significantly less likely to be denied. "
            f"Gender does affect approval outcomes, with females faring better."
        )
    else:
        score = 70
        explanation = (
            f"Gender significantly affects mortgage approval (multivariate logistic p={female_pval:.4f}). "
            f"Female denial rate: {female_rate:.3f}, male denial rate: {male_rate:.3f}. "
            f"Female coefficient: {female_coef:.4f}. Gender is a significant predictor even after controlling for creditworthiness."
        )
else:
    # Not significant after controls
    if significant_univariate:
        score = 35
        explanation = (
            f"Univariate analysis shows a difference (chi2 p={p_chi2:.4f}): female denial rate {female_rate:.3f} vs male {male_rate:.3f}. "
            f"However, after controlling for creditworthiness (housing expense ratio, credit scores, debt ratios), "
            f"the gender effect is not statistically significant (multivariate p={female_pval:.4f}). "
            f"The apparent gender gap largely reflects differences in applicant financial profiles, not gender discrimination per se."
        )
    else:
        score = 15
        explanation = (
            f"Gender does not significantly affect mortgage approval. "
            f"Female denial rate: {female_rate:.3f}, male denial rate: {male_rate:.3f}. "
            f"Neither univariate (p={p_chi2:.4f}) nor multivariate (p={female_pval:.4f}) analysis shows significance."
        )

print(f"\nFinal score: {score}")
print(f"Explanation: {explanation}")

result = {"response": score, "explanation": explanation}
with open("conclusion.txt", "w") as f:
    json.dump(result, f)

print("\nWrote conclusion.txt")
