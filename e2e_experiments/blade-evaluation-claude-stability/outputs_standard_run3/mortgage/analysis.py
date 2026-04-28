import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("mortgage.csv")
df = df.dropna(subset=['female'])

print("Shape:", df.shape)
print(df.head())
print(df.describe())

# Gender breakdown
female = df[df['female'] == 1]
male = df[df['female'] == 0]

print("\nFemale applicants:", len(female), "Approval rate:", female['accept'].mean())
print("Male applicants:", len(male), "Approval rate:", male['accept'].mean())

# Chi-square test on gender vs acceptance
ct = pd.crosstab(df['female'], df['accept'])
print("\nCrosstab:\n", ct)
chi2, p_chi2, dof, expected = stats.chi2_contingency(ct)
print(f"Chi-square: {chi2:.4f}, p-value: {p_chi2:.4f}")

# T-test on accept rate by gender
t_stat, p_ttest = stats.ttest_ind(female['accept'], male['accept'])
print(f"T-test: t={t_stat:.4f}, p={p_ttest:.4f}")

# Logistic regression: gender alone
X_simple = df[['female']]
X_simple_const = sm.add_constant(X_simple)
logit_simple = sm.Logit(df['accept'], X_simple_const).fit(disp=0)
print("\nLogistic regression (gender only):")
print(logit_simple.summary2())

# Logistic regression: gender + controls
controls = ['female', 'black', 'housing_expense_ratio', 'self_employed', 'married',
            'mortgage_credit', 'consumer_credit', 'bad_history', 'PI_ratio', 'loan_to_value', 'denied_PMI']
df_clean = df[controls + ['accept']].dropna()
X_full = sm.add_constant(df_clean[controls])
logit_full = sm.Logit(df_clean['accept'], X_full).fit(disp=0)
print("\nLogistic regression (with controls):")
print(logit_full.summary2())

female_coef = logit_full.params['female']
female_pval = logit_full.pvalues['female']
print(f"\nFemale coefficient: {female_coef:.4f}, p-value: {female_pval:.4f}")

# Decision
# Raw approval rates
female_rate = female['accept'].mean()
male_rate = male['accept'].mean()
rate_diff = female_rate - male_rate

# Use controlled logistic regression p-value as primary evidence
if female_pval < 0.05:
    significant = True
else:
    significant = False

print(f"\nFemale approval rate: {female_rate:.4f}")
print(f"Male approval rate: {male_rate:.4f}")
print(f"Difference: {rate_diff:.4f}")
print(f"Controlled p-value for gender: {female_pval:.4f}, significant: {significant}")

# Score: if not significant after controls, lean toward No (low score)
# If significant with a positive coef (female approved more), lean toward 50+
# If significant with a negative coef (female approved less), still indicates effect exists -> higher score
if significant:
    # Effect exists; score based on direction and magnitude
    score = 65
    explanation = (
        f"Gender (female) has a statistically significant effect on mortgage approval even after controlling "
        f"for creditworthiness variables (coef={female_coef:.3f}, p={female_pval:.4f}). "
        f"Female applicants have an approval rate of {female_rate:.3f} vs {male_rate:.3f} for males "
        f"(raw difference {rate_diff:+.3f}). The chi-square test also shows significance (p={p_chi2:.4f}). "
        f"This indicates gender does affect approval decisions."
    )
else:
    score = 25
    explanation = (
        f"After controlling for creditworthiness variables, gender (female) does NOT have a statistically "
        f"significant effect on mortgage approval (coef={female_coef:.3f}, p={female_pval:.4f}). "
        f"Female applicants have an approval rate of {female_rate:.3f} vs {male_rate:.3f} for males "
        f"(raw difference {rate_diff:+.3f}), but this disappears after controlling for other factors. "
        f"The raw chi-square test p={p_chi2:.4f}."
    )

result = {"response": score, "explanation": explanation}
with open("conclusion.txt", "w") as f:
    json.dump(result, f)

print("\nconclusion.txt written.")
print(json.dumps(result, indent=2))
