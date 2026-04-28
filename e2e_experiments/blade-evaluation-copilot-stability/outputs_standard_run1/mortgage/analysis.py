import pandas as pd
import numpy as np
import json
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.formula.api import logit

# Load the data
df = pd.read_csv('mortgage.csv')

# Drop rows with missing values
df = df.dropna()

# Research Question: How does gender affect whether banks approve an individual's mortgage application?
# The outcome variable is 'accept' (1 if approved, 0 if denied)
# The key predictor is 'female' (1 if female, 0 if male)

print("=" * 80)
print("ANALYSIS: Gender Effect on Mortgage Application Approval")
print("=" * 80)
print()

# 1. Exploratory Data Analysis
print("1. DATA OVERVIEW")
print("-" * 80)
print(f"Total applications: {len(df)}")
print(f"Number of females: {df['female'].sum()} ({df['female'].mean()*100:.1f}%)")
print(f"Number of males: {(1-df['female']).sum()} ({(1-df['female']).mean()*100:.1f}%)")
print(f"Overall acceptance rate: {df['accept'].mean()*100:.1f}%")
print()

# 2. Approval rates by gender
print("2. APPROVAL RATES BY GENDER")
print("-" * 80)
approval_by_gender = df.groupby('female')['accept'].agg(['mean', 'count', 'sum'])
approval_by_gender.index = ['Male', 'Female']
approval_by_gender.columns = ['Approval Rate', 'Total Applications', 'Approved Applications']
print(approval_by_gender)
print()

female_approval_rate = df[df['female'] == 1]['accept'].mean()
male_approval_rate = df[df['female'] == 0]['accept'].mean()
print(f"Female approval rate: {female_approval_rate*100:.2f}%")
print(f"Male approval rate: {male_approval_rate*100:.2f}%")
print(f"Difference: {(female_approval_rate - male_approval_rate)*100:.2f} percentage points")
print()

# 3. Chi-square test for independence
print("3. CHI-SQUARE TEST")
print("-" * 80)
contingency_table = pd.crosstab(df['female'], df['accept'])
chi2, p_value_chi, dof, expected = stats.chi2_contingency(contingency_table)
print(f"Chi-square statistic: {chi2:.4f}")
print(f"P-value: {p_value_chi:.6f}")
print(f"Degrees of freedom: {dof}")
if p_value_chi < 0.05:
    print("Result: SIGNIFICANT relationship between gender and approval (p < 0.05)")
else:
    print("Result: NO significant relationship between gender and approval (p >= 0.05)")
print()

# 4. Two-sample t-test
print("4. TWO-SAMPLE T-TEST")
print("-" * 80)
female_accepts = df[df['female'] == 1]['accept']
male_accepts = df[df['female'] == 0]['accept']
t_stat, p_value_ttest = stats.ttest_ind(female_accepts, male_accepts)
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value_ttest:.6f}")
if p_value_ttest < 0.05:
    print("Result: SIGNIFICANT difference in approval rates by gender (p < 0.05)")
else:
    print("Result: NO significant difference in approval rates by gender (p >= 0.05)")
print()

# 5. Logistic Regression - Univariate (gender only)
print("5. LOGISTIC REGRESSION - UNIVARIATE (GENDER ONLY)")
print("-" * 80)
X_univariate = df[['female']]
y = df['accept']
X_univariate_sm = sm.add_constant(X_univariate)
logit_univariate = sm.Logit(y, X_univariate_sm)
result_univariate = logit_univariate.fit(disp=0)
print(result_univariate.summary())
print()

# 6. Logistic Regression - Multivariate (controlling for other factors)
print("6. LOGISTIC REGRESSION - MULTIVARIATE (CONTROLLING FOR OTHER FACTORS)")
print("-" * 80)
# Include all relevant control variables
control_vars = ['female', 'black', 'housing_expense_ratio', 'self_employed', 'married',
                'mortgage_credit', 'consumer_credit', 'bad_history', 'PI_ratio',
                'loan_to_value', 'denied_PMI']
X_multivariate = df[control_vars]
X_multivariate_sm = sm.add_constant(X_multivariate)
logit_multivariate = sm.Logit(y, X_multivariate_sm)
result_multivariate = logit_multivariate.fit(disp=0)
print(result_multivariate.summary())
print()

# 7. Extract key statistics
print("7. KEY FINDINGS")
print("-" * 80)
female_coef_univariate = result_univariate.params['female']
female_pval_univariate = result_univariate.pvalues['female']
female_coef_multivariate = result_multivariate.params['female']
female_pval_multivariate = result_multivariate.pvalues['female']

print(f"Univariate model:")
print(f"  - Female coefficient: {female_coef_univariate:.4f}")
print(f"  - P-value: {female_pval_univariate:.6f}")
print(f"  - Odds ratio: {np.exp(female_coef_univariate):.4f}")
print()
print(f"Multivariate model (controlling for creditworthiness, demographics, loan characteristics):")
print(f"  - Female coefficient: {female_coef_multivariate:.4f}")
print(f"  - P-value: {female_pval_multivariate:.6f}")
print(f"  - Odds ratio: {np.exp(female_coef_multivariate):.4f}")
print()

# 8. Decision and Conclusion
print("8. DECISION")
print("=" * 80)

# Determine the response based on statistical significance
# The key question is whether gender affects approval
# We need to look at both univariate and multivariate results

# In the univariate model, we see if there's a raw association
# In the multivariate model, we see if gender has an effect after controlling for creditworthiness

if female_pval_multivariate < 0.05:
    # Significant effect in multivariate model
    if abs(female_coef_multivariate) > 0.2:
        response_score = 85  # Strong evidence
        explanation = f"Gender has a statistically significant effect on mortgage approval (p={female_pval_multivariate:.4f}). Even after controlling for creditworthiness, employment, and loan characteristics, being female is associated with an odds ratio of {np.exp(female_coef_multivariate):.3f} for approval. This suggests systematic gender-based differences in lending decisions."
    else:
        response_score = 70  # Moderate evidence
        explanation = f"Gender has a statistically significant but modest effect on mortgage approval (p={female_pval_multivariate:.4f}). After controlling for creditworthiness and other factors, being female is associated with an odds ratio of {np.exp(female_coef_multivariate):.3f} for approval."
elif female_pval_univariate < 0.05 and female_pval_multivariate >= 0.05:
    # Significant in univariate but not multivariate
    response_score = 40  # Weak evidence
    explanation = f"There is a raw association between gender and approval rates (univariate p={female_pval_univariate:.4f}), but this effect becomes non-significant (p={female_pval_multivariate:.4f}) when controlling for creditworthiness and other legitimate lending factors. This suggests the gender difference in approval rates is largely explained by differences in applicant qualifications rather than gender discrimination."
else:
    # Not significant in either model
    response_score = 15  # No evidence
    explanation = f"Gender does not have a statistically significant effect on mortgage approval. Neither the univariate (p={female_pval_univariate:.4f}) nor multivariate (p={female_pval_multivariate:.4f}) analyses show a significant relationship between gender and approval decisions. The data does not support gender-based discrimination in lending."

print(f"Response Score (0-100 Likert scale): {response_score}")
print(f"Explanation: {explanation}")
print()

# Write the conclusion to file
conclusion = {
    "response": response_score,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("Conclusion written to conclusion.txt")
print("=" * 80)
