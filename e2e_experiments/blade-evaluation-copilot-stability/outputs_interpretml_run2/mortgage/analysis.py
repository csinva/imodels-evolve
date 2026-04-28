import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import json

# Load the data
df = pd.read_csv('mortgage.csv')

# Handle missing values - drop rows with missing values
print(f"Original dataset size: {len(df)}")
print(f"Missing values per column:\n{df.isnull().sum()}")
df = df.dropna()
print(f"Dataset size after removing missing values: {len(df)}")

# Research question: How does gender affect whether banks approve an individual's mortgage application?
# Target variable: accept (1 if accepted, 0 if denied)
# Key predictor: female (1 if female, 0 if male)

print("=" * 80)
print("MORTGAGE APPROVAL ANALYSIS: GENDER EFFECT")
print("=" * 80)

# 1. EXPLORATORY DATA ANALYSIS
print("\n1. DATA OVERVIEW")
print(f"Total applications: {len(df)}")
print(f"Overall acceptance rate: {df['accept'].mean():.2%}")

print("\n2. ACCEPTANCE RATES BY GENDER")
gender_approval = df.groupby('female')['accept'].agg(['mean', 'count'])
gender_approval.index = ['Male', 'Female']
print(gender_approval)

male_acceptance = df[df['female'] == 0]['accept'].mean()
female_acceptance = df[df['female'] == 1]['accept'].mean()
print(f"\nMale acceptance rate: {male_acceptance:.2%}")
print(f"Female acceptance rate: {female_acceptance:.2%}")
print(f"Difference: {female_acceptance - male_acceptance:.2%}")

# 2. STATISTICAL TEST: Chi-square test for independence
print("\n3. CHI-SQUARE TEST (Gender vs Acceptance)")
contingency_table = pd.crosstab(df['female'], df['accept'])
chi2, p_value_chi, dof, expected = stats.chi2_contingency(contingency_table)
print(f"Chi-square statistic: {chi2:.4f}")
print(f"P-value: {p_value_chi:.4f}")
print(f"Degrees of freedom: {dof}")

# 3. T-TEST for difference in means
print("\n4. T-TEST (Difference in acceptance rates)")
male_accept = df[df['female'] == 0]['accept']
female_accept = df[df['female'] == 1]['accept']
t_stat, p_value_t = stats.ttest_ind(female_accept, male_accept)
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value_t:.4f}")

# 4. LOGISTIC REGRESSION: Simple model (gender only)
print("\n5. SIMPLE LOGISTIC REGRESSION (Gender Only)")
X_simple = df[['female']]
y = df['accept']
model_simple = LogisticRegression()
model_simple.fit(X_simple, y)
print(f"Coefficient for female: {model_simple.coef_[0][0]:.4f}")
print(f"Odds ratio: {np.exp(model_simple.coef_[0][0]):.4f}")

# Get p-value using statsmodels
X_simple_sm = sm.add_constant(X_simple)
logit_simple = sm.Logit(y, X_simple_sm)
result_simple = logit_simple.fit(disp=0)
print("\nStatsmodels Logistic Regression Summary:")
print(result_simple.summary2().tables[1])

# 5. MULTIPLE LOGISTIC REGRESSION: Control for confounders
print("\n6. MULTIPLE LOGISTIC REGRESSION (Controlling for Confounders)")
# Include relevant covariates that might affect both gender and approval
control_vars = ['female', 'black', 'housing_expense_ratio', 'self_employed', 
                'married', 'mortgage_credit', 'consumer_credit', 'bad_history',
                'PI_ratio', 'loan_to_value', 'denied_PMI']

X_full = df[control_vars]
logit_full = sm.Logit(y, sm.add_constant(X_full))
result_full = logit_full.fit(disp=0)
print("\nFull Model Summary (Key Coefficients):")
print(result_full.summary2().tables[1])

# Extract female coefficient and p-value from full model
female_coef = result_full.params['female']
female_pvalue = result_full.pvalues['female']
female_odds_ratio = np.exp(female_coef)

print(f"\n7. KEY FINDINGS")
print(f"Female coefficient (controlled): {female_coef:.4f}")
print(f"Female p-value (controlled): {female_pvalue:.4f}")
print(f"Female odds ratio (controlled): {female_odds_ratio:.4f}")

# 6. INTERPRETATION AND CONCLUSION
print("\n8. INTERPRETATION")
print("=" * 80)

# Determine response score based on statistical significance and effect size
alpha = 0.05

# Consider multiple tests and effect sizes
simple_significant = p_value_chi < alpha and p_value_t < alpha
controlled_significant = female_pvalue < alpha

# Effect size consideration
effect_size = abs(female_acceptance - male_acceptance)
controlled_effect = abs(female_coef)

print(f"\nSimple analysis (unadjusted):")
print(f"  - Chi-square p-value: {p_value_chi:.4f} (significant: {simple_significant})")
print(f"  - T-test p-value: {p_value_t:.4f}")
print(f"  - Raw difference in acceptance: {(female_acceptance - male_acceptance)*100:.2f} percentage points")

print(f"\nControlled analysis (adjusted for confounders):")
print(f"  - Female coefficient p-value: {female_pvalue:.4f} (significant: {controlled_significant})")
print(f"  - Coefficient magnitude: {female_coef:.4f}")
print(f"  - Odds ratio: {female_odds_ratio:.4f}")

# Determine score
# The research question asks: "How does gender affect whether banks approve an individual's mortgage application?"
# We need to assess if there IS an effect (not necessarily the direction)

if controlled_significant:
    # Statistically significant effect even after controlling for confounders
    if abs(female_coef) > 0.3:  # Substantial effect
        score = 85
        explanation = f"Yes, gender significantly affects mortgage approval. After controlling for creditworthiness and other factors, being female is associated with a significant effect (p={female_pvalue:.4f}, coefficient={female_coef:.4f}, odds ratio={female_odds_ratio:.2f}). The effect remains statistically significant even when accounting for credit scores, debt ratios, employment status, and other relevant financial variables."
    elif abs(female_coef) > 0.15:  # Moderate effect
        score = 70
        explanation = f"Yes, gender has a moderate effect on mortgage approval. The effect is statistically significant (p={female_pvalue:.4f}) even after controlling for confounders, with a coefficient of {female_coef:.4f} (odds ratio={female_odds_ratio:.2f}). This indicates that gender does influence approval decisions beyond what can be explained by financial qualifications alone."
    else:  # Small but significant effect
        score = 60
        explanation = f"Gender has a small but statistically significant effect on mortgage approval (p={female_pvalue:.4f}, coefficient={female_coef:.4f}). While the effect size is modest, it persists even after controlling for creditworthiness, suggesting that gender does play a role in approval decisions."
elif simple_significant and not controlled_significant:
    # Significant in simple analysis but not when controlled
    score = 30
    explanation = f"Gender shows a relationship with mortgage approval in unadjusted analysis (p={p_value_chi:.4f}), but this effect is not statistically significant when controlling for financial qualifications (p={female_pvalue:.4f}). This suggests that gender differences in approval rates are largely explained by differences in creditworthiness and financial factors, not gender discrimination per se."
else:
    # Not significant even in simple analysis
    score = 10
    explanation = f"No significant effect of gender on mortgage approval was detected. Neither the unadjusted analysis (p={p_value_chi:.4f}) nor the controlled analysis (p={female_pvalue:.4f}) show statistically significant differences in approval rates based on gender. The data does not support a gender effect on mortgage approval decisions."

print(f"\nFINAL SCORE: {score}/100")
print(f"EXPLANATION: {explanation}")

# Write conclusion to file
conclusion = {
    "response": score,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE - conclusion.txt written")
print("=" * 80)
