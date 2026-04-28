import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from interpret.glassbox import ExplainableBoostingClassifier

# Load the data
df = pd.read_csv('mortgage.csv')

# Drop rows with missing values in key variables
df_clean = df.dropna(subset=['female', 'accept', 'married', 'PI_ratio'])

# Print basic statistics
print("=" * 80)
print("DATASET OVERVIEW")
print("=" * 80)
print(f"Total samples (original): {len(df)}")
print(f"Total samples (after dropping missing): {len(df_clean)}")
print(f"\nColumns: {df_clean.columns.tolist()}")
print(f"\nMissing values (original):\n{df.isnull().sum()}")

# Explore the target variable and gender
print("\n" + "=" * 80)
print("GENDER AND MORTGAGE APPROVAL DISTRIBUTION")
print("=" * 80)
print(f"\nGender distribution:")
print(df_clean['female'].value_counts())
print(f"\nMortgage approval distribution:")
print(df_clean['accept'].value_counts())

# Cross-tabulation of gender and acceptance
print(f"\nCross-tabulation of gender and mortgage approval:")
crosstab = pd.crosstab(df_clean['female'], df_clean['accept'], margins=True)
print(crosstab)

# Calculate approval rates by gender
female_data = df_clean[df_clean['female'] == 1]
male_data = df_clean[df_clean['female'] == 0]

female_approval_rate = female_data['accept'].mean()
male_approval_rate = male_data['accept'].mean()

print(f"\nApproval rates:")
print(f"  Female approval rate: {female_approval_rate:.4f} ({female_approval_rate*100:.2f}%)")
print(f"  Male approval rate: {male_approval_rate:.4f} ({male_approval_rate*100:.2f}%)")
print(f"  Difference: {(male_approval_rate - female_approval_rate)*100:.2f} percentage points")

# Chi-square test for independence
print("\n" + "=" * 80)
print("CHI-SQUARE TEST FOR INDEPENDENCE")
print("=" * 80)
contingency_table = pd.crosstab(df_clean['female'], df_clean['accept'])
chi2, p_value_chi, dof, expected = stats.chi2_contingency(contingency_table)
print(f"Chi-square statistic: {chi2:.4f}")
print(f"P-value: {p_value_chi:.6f}")
print(f"Degrees of freedom: {dof}")
if p_value_chi < 0.05:
    print("Result: SIGNIFICANT at 0.05 level - gender and approval are associated")
else:
    print("Result: NOT SIGNIFICANT at 0.05 level - no evidence of association")

# Two-proportion z-test
print("\n" + "=" * 80)
print("TWO-PROPORTION Z-TEST")
print("=" * 80)
n_female = len(female_data)
n_male = len(male_data)
n_female_approved = female_data['accept'].sum()
n_male_approved = male_data['accept'].sum()

from statsmodels.stats.proportion import proportions_ztest
z_stat, p_value_z = proportions_ztest([n_female_approved, n_male_approved], [n_female, n_male])
print(f"Z-statistic: {z_stat:.4f}")
print(f"P-value (two-tailed): {p_value_z:.6f}")
if p_value_z < 0.05:
    print("Result: SIGNIFICANT at 0.05 level - approval rates differ by gender")
else:
    print("Result: NOT SIGNIFICANT at 0.05 level - no evidence of difference")

# Simple logistic regression: gender only
print("\n" + "=" * 80)
print("SIMPLE LOGISTIC REGRESSION (Gender Only)")
print("=" * 80)
X_simple = df_clean[['female']]
y = df_clean['accept']
X_simple_const = sm.add_constant(X_simple)
logit_simple = sm.Logit(y, X_simple_const).fit(disp=0)
print(logit_simple.summary())

# Multiple logistic regression: controlling for other factors
print("\n" + "=" * 80)
print("MULTIPLE LOGISTIC REGRESSION (Controlling for Confounders)")
print("=" * 80)
# Include relevant financial and demographic variables
control_vars = ['female', 'black', 'housing_expense_ratio', 'self_employed', 
                'married', 'mortgage_credit', 'consumer_credit', 'bad_history', 
                'PI_ratio', 'loan_to_value', 'denied_PMI']
X_multi = df_clean[control_vars]
X_multi_const = sm.add_constant(X_multi)
logit_multi = sm.Logit(y, X_multi_const).fit(disp=0)
print(logit_multi.summary())

female_coef = logit_multi.params['female']
female_pval = logit_multi.pvalues['female']
female_odds_ratio = np.exp(female_coef)
print(f"\nFemale coefficient: {female_coef:.4f}")
print(f"P-value: {female_pval:.6f}")
print(f"Odds ratio: {female_odds_ratio:.4f}")
print(f"Interpretation: Being female is associated with {(female_odds_ratio-1)*100:.2f}% change in odds of approval")
if female_pval < 0.05:
    print("Result: Gender is SIGNIFICANT at 0.05 level after controlling for other factors")
else:
    print("Result: Gender is NOT SIGNIFICANT at 0.05 level after controlling for other factors")

# Explainable Boosting Classifier for interpretability
print("\n" + "=" * 80)
print("EXPLAINABLE BOOSTING CLASSIFIER")
print("=" * 80)
X_ebc = df_clean[control_vars]
ebc = ExplainableBoostingClassifier(random_state=42)
ebc.fit(X_ebc, y)

# Get feature importances
importances = ebc.term_importances()
print("\nFeature importances from EBC:")
for i, feature in enumerate(control_vars):
    print(f"  {feature}: {importances[i]:.4f}")

# Compare descriptive statistics by gender
print("\n" + "=" * 80)
print("DESCRIPTIVE STATISTICS BY GENDER")
print("=" * 80)
print("\nFemale applicants:")
print(female_data[['housing_expense_ratio', 'mortgage_credit', 'consumer_credit', 
                   'bad_history', 'PI_ratio', 'loan_to_value']].describe())
print("\nMale applicants:")
print(male_data[['housing_expense_ratio', 'mortgage_credit', 'consumer_credit', 
                 'bad_history', 'PI_ratio', 'loan_to_value']].describe())

# T-tests for differences in financial characteristics
print("\n" + "=" * 80)
print("T-TESTS FOR DIFFERENCES IN FINANCIAL CHARACTERISTICS")
print("=" * 80)
for var in ['housing_expense_ratio', 'mortgage_credit', 'consumer_credit', 'PI_ratio', 'loan_to_value']:
    t_stat, p_val = stats.ttest_ind(female_data[var].dropna(), male_data[var].dropna())
    print(f"{var}: t={t_stat:.4f}, p={p_val:.6f}", end="")
    if p_val < 0.05:
        print(" (SIGNIFICANT)")
    else:
        print(" (not significant)")

# Conclusion
print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

# Determine the response score based on statistical evidence
if p_value_chi < 0.05 and female_pval < 0.05:
    # Both bivariate and multivariate tests significant
    response = 85
    explanation = (
        f"Yes, gender significantly affects mortgage approval. "
        f"Chi-square test shows significant association (p={p_value_chi:.6f}). "
        f"In multivariate logistic regression controlling for creditworthiness and other factors, "
        f"gender remains significant (p={female_pval:.6f}). "
        f"Female applicants have {abs((female_odds_ratio-1)*100):.1f}% {'lower' if female_odds_ratio < 1 else 'higher'} odds of approval. "
        f"Raw approval rates: {female_approval_rate*100:.1f}% (female) vs {male_approval_rate*100:.1f}% (male)."
    )
elif p_value_chi < 0.05 and female_pval >= 0.05:
    # Bivariate significant but multivariate not significant
    response = 35
    explanation = (
        f"Gender shows a bivariate association with approval (chi-square p={p_value_chi:.6f}), "
        f"with raw rates of {female_approval_rate*100:.1f}% (female) vs {male_approval_rate*100:.1f}% (male). "
        f"However, when controlling for creditworthiness and financial factors in logistic regression, "
        f"gender is not significant (p={female_pval:.6f}). "
        f"This suggests the apparent gender difference is largely explained by differences in financial qualifications."
    )
elif p_value_chi >= 0.05:
    # Not even bivariate significant
    response = 15
    explanation = (
        f"No significant relationship between gender and mortgage approval. "
        f"Chi-square test shows no association (p={p_value_chi:.6f}). "
        f"Approval rates are similar: {female_approval_rate*100:.1f}% (female) vs {male_approval_rate*100:.1f}% (male). "
        f"The difference of {abs((male_approval_rate - female_approval_rate)*100):.1f} percentage points is not statistically significant."
    )
else:
    # Edge case
    response = 50
    explanation = (
        f"Mixed evidence. Statistical tests yield inconclusive results regarding gender's effect on approval."
    )

print(f"\nResponse score: {response}/100")
print(f"Explanation: {explanation}")

# Write conclusion to file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n" + "=" * 80)
print("Analysis complete. Results written to conclusion.txt")
print("=" * 80)
