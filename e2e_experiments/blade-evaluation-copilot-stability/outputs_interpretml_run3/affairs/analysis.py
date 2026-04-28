import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder
from interpret.glassbox import ExplainableBoostingRegressor, ExplainableBoostingClassifier
import json

# Load the dataset
df = pd.read_csv('affairs.csv')

# Display basic information
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nDataset info:")
print(df.info())
print("\nSummary statistics:")
print(df.describe())

# Focus on the research question: Does having children decrease engagement in extramarital affairs?
print("\n" + "="*80)
print("RESEARCH QUESTION: Does having children decrease engagement in extramarital affairs?")
print("="*80)

# Check the distribution of affairs by children status
print("\n1. DESCRIPTIVE STATISTICS BY CHILDREN STATUS")
print("-"*80)
affairs_by_children = df.groupby('children')['affairs'].agg(['count', 'mean', 'std', 'median'])
print(affairs_by_children)

# Calculate percentage with any affairs
df['has_affair'] = (df['affairs'] > 0).astype(int)
affair_rate = df.groupby('children')['has_affair'].agg(['count', 'sum', 'mean'])
affair_rate.columns = ['total', 'num_with_affairs', 'rate_with_affairs']
print("\nAffair rates by children status:")
print(affair_rate)

# 2. T-TEST: Compare mean affairs between groups
print("\n2. INDEPENDENT T-TEST")
print("-"*80)
children_yes = df[df['children'] == 'yes']['affairs']
children_no = df[df['children'] == 'no']['affairs']

t_stat, p_value_ttest = stats.ttest_ind(children_yes, children_no)
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value_ttest:.4f}")
print(f"Mean affairs (children=yes): {children_yes.mean():.4f}")
print(f"Mean affairs (children=no): {children_no.mean():.4f}")
print(f"Difference: {children_yes.mean() - children_no.mean():.4f}")

# 3. MANN-WHITNEY U TEST (non-parametric alternative)
print("\n3. MANN-WHITNEY U TEST (Non-parametric)")
print("-"*80)
u_stat, p_value_mann = stats.mannwhitneyu(children_yes, children_no, alternative='two-sided')
print(f"U-statistic: {u_stat:.4f}")
print(f"P-value: {p_value_mann:.4f}")

# 4. CHI-SQUARE TEST: For binary affair occurrence
print("\n4. CHI-SQUARE TEST (Has affair vs No affair)")
print("-"*80)
contingency_table = pd.crosstab(df['children'], df['has_affair'])
print("Contingency table:")
print(contingency_table)
chi2, p_value_chi2, dof, expected = stats.chi2_contingency(contingency_table)
print(f"\nChi-square statistic: {chi2:.4f}")
print(f"P-value: {p_value_chi2:.4f}")
print(f"Degrees of freedom: {dof}")

# 5. LINEAR REGRESSION with statsmodels (for p-values)
print("\n5. LINEAR REGRESSION (OLS with p-values)")
print("-"*80)
# Encode children as binary
df['children_binary'] = (df['children'] == 'yes').astype(int)

# Simple regression: affairs ~ children
X = df[['children_binary']]
y = df['affairs']
X_with_const = sm.add_constant(X)
model_simple = sm.OLS(y, X_with_const).fit()
print(model_simple.summary())

# 6. MULTIPLE REGRESSION controlling for confounders
print("\n6. MULTIPLE REGRESSION (Controlling for confounders)")
print("-"*80)
# Encode categorical variables
df['gender_binary'] = (df['gender'] == 'male').astype(int)

# Include potential confounders
X_full = df[['children_binary', 'age', 'yearsmarried', 'gender_binary', 
             'religiousness', 'education', 'rating']]
X_full_const = sm.add_constant(X_full)
model_full = sm.OLS(y, X_full_const).fit()
print(model_full.summary())

# 7. INTERPRETABLE MODEL: ExplainableBoostingRegressor
print("\n7. EXPLAINABLE BOOSTING REGRESSOR")
print("-"*80)
X_interpret = df[['children_binary', 'age', 'yearsmarried', 'gender_binary', 
                  'religiousness', 'education', 'rating']]
try:
    ebm = ExplainableBoostingRegressor(random_state=42)
    ebm.fit(X_interpret, y)
    print("EBM model fitted successfully")
    print(f"Model score: {ebm.score(X_interpret, y):.4f}")
except Exception as e:
    print(f"EBM error (skipping): {e}")

# 8. LOGISTIC REGRESSION for binary outcome (has affair or not)
print("\n8. LOGISTIC REGRESSION (Has affair or not)")
print("-"*80)
y_binary = df['has_affair']
X_logit_const = sm.add_constant(X_full)
model_logit = sm.Logit(y_binary, X_logit_const).fit()
print(model_logit.summary())

# Extract coefficient and p-value for children
children_coef_simple = model_simple.params['children_binary']
children_pval_simple = model_simple.pvalues['children_binary']
children_coef_full = model_full.params['children_binary']
children_pval_full = model_full.pvalues['children_binary']
children_coef_logit = model_logit.params['children_binary']
children_pval_logit = model_logit.pvalues['children_binary']

# CONCLUSION
print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

# Determine response based on statistical tests
# The question is: Does having children DECREASE engagement in affairs?
# If children coefficient is negative and significant, the answer is YES
# If not significant, the answer is NO or UNCERTAIN

significant_threshold = 0.05

# Check if children has a significant effect
tests_results = {
    't-test': p_value_ttest,
    'mann-whitney': p_value_mann,
    'chi-square': p_value_chi2,
    'simple_regression': children_pval_simple,
    'multiple_regression': children_pval_full,
    'logistic_regression': children_pval_logit
}

print(f"\nStatistical test p-values:")
for test_name, pval in tests_results.items():
    sig_status = "SIGNIFICANT" if pval < significant_threshold else "NOT SIGNIFICANT"
    print(f"  {test_name}: {pval:.4f} ({sig_status})")

print(f"\nEffect direction (regression coefficients):")
print(f"  Simple regression: {children_coef_simple:.4f}")
print(f"  Multiple regression: {children_coef_full:.4f}")
print(f"  Logistic regression: {children_coef_logit:.4f}")

print(f"\nMean affairs difference: {children_yes.mean() - children_no.mean():.4f}")
print(f"  (Positive = more affairs with children, Negative = fewer affairs with children)")

# Determine the final response score
# Check if the relationship is statistically significant in most tests
sig_count = sum(1 for pval in tests_results.values() if pval < significant_threshold)
total_tests = len(tests_results)

# Check if effect is in the expected direction (negative = decrease)
effect_is_negative = (children_yes.mean() < children_no.mean())

explanation_parts = []

if sig_count >= 4:  # Majority of tests are significant
    if effect_is_negative:
        # Strong evidence that children DECREASE affairs
        response_score = 80
        explanation_parts.append(f"Strong evidence: {sig_count}/{total_tests} statistical tests show significant effects (p<0.05).")
        explanation_parts.append(f"Having children is associated with FEWER affairs (mean: {children_yes.mean():.2f} vs {children_no.mean():.2f}).")
    else:
        # Significant but opposite direction
        response_score = 15
        explanation_parts.append(f"Statistical tests show significance, but in the OPPOSITE direction.")
        explanation_parts.append(f"Having children is associated with MORE affairs (mean: {children_yes.mean():.2f} vs {children_no.mean():.2f}).")
elif sig_count >= 2:  # Some evidence
    if effect_is_negative:
        response_score = 60
        explanation_parts.append(f"Moderate evidence: {sig_count}/{total_tests} tests significant.")
        explanation_parts.append(f"Children associated with fewer affairs, but effect not consistent across all tests.")
    else:
        response_score = 25
        explanation_parts.append(f"Some significant tests, but effect is in opposite direction or inconsistent.")
else:  # Weak or no evidence
    if abs(children_yes.mean() - children_no.mean()) < 0.2:
        response_score = 10
        explanation_parts.append(f"No significant relationship found in most tests (only {sig_count}/{total_tests} significant).")
        explanation_parts.append(f"Mean difference is minimal: {abs(children_yes.mean() - children_no.mean()):.2f}")
    else:
        response_score = 30
        explanation_parts.append(f"Tests not significant, but descriptive statistics suggest a trend.")

explanation_parts.append(f"Key p-values: multiple regression={children_pval_full:.4f}, t-test={p_value_ttest:.4f}.")

explanation = " ".join(explanation_parts)

print(f"\n{'='*80}")
print(f"FINAL RESPONSE SCORE: {response_score}/100")
print(f"EXPLANATION: {explanation}")
print(f"{'='*80}")

# Write conclusion to file
conclusion = {
    "response": response_score,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n✓ Conclusion written to conclusion.txt")
