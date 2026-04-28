import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from interpret.glassbox import ExplainableBoostingRegressor
import json

# Load the data
df = pd.read_csv('affairs.csv')

# Display basic information
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nSummary statistics:")
print(df.describe())

# Focus on the research question: Does having children decrease engagement in extramarital affairs?
print("\n" + "="*70)
print("RESEARCH QUESTION: Does having children decrease engagement in extramarital affairs?")
print("="*70)

# 1. Basic comparison: affairs by children status
print("\n1. DESCRIPTIVE STATISTICS BY CHILDREN STATUS")
print("-"*50)
affairs_by_children = df.groupby('children')['affairs'].agg(['count', 'mean', 'std', 'median'])
print(affairs_by_children)

children_yes = df[df['children'] == 'yes']['affairs']
children_no = df[df['children'] == 'no']['affairs']

print(f"\nMean affairs (with children): {children_yes.mean():.3f}")
print(f"Mean affairs (without children): {children_no.mean():.3f}")
print(f"Difference: {children_yes.mean() - children_no.mean():.3f}")

# 2. Statistical test: t-test
print("\n2. T-TEST: Comparing affairs between those with and without children")
print("-"*50)
t_stat, p_value = stats.ttest_ind(children_yes, children_no)
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Significant at α=0.05? {'YES' if p_value < 0.05 else 'NO'}")

# 3. Mann-Whitney U test (non-parametric alternative)
print("\n3. MANN-WHITNEY U TEST (non-parametric)")
print("-"*50)
u_stat, p_value_mw = stats.mannwhitneyu(children_yes, children_no, alternative='two-sided')
print(f"U-statistic: {u_stat:.4f}")
print(f"p-value: {p_value_mw:.4f}")
print(f"Significant at α=0.05? {'YES' if p_value_mw < 0.05 else 'NO'}")

# 4. Regression analysis with statsmodels for p-values
print("\n4. REGRESSION ANALYSIS (statsmodels OLS)")
print("-"*50)

# Create dummy variable for children
df['children_dummy'] = (df['children'] == 'yes').astype(int)

# Simple regression: affairs ~ children
X_simple = sm.add_constant(df['children_dummy'])
y = df['affairs']
model_simple = sm.OLS(y, X_simple).fit()
print(model_simple.summary())

# Multiple regression: controlling for confounders
print("\n5. MULTIPLE REGRESSION (controlling for confounders)")
print("-"*50)

# Create dummy for gender
df['gender_dummy'] = (df['gender'] == 'male').astype(int)

# Multiple regression including other factors
X_multi = df[['children_dummy', 'age', 'yearsmarried', 'religiousness', 'education', 'rating', 'gender_dummy']]
X_multi = sm.add_constant(X_multi)
model_multi = sm.OLS(y, X_multi).fit()
print(model_multi.summary())

print(f"\nChildren coefficient in multiple regression: {model_multi.params['children_dummy']:.4f}")
print(f"P-value for children: {model_multi.pvalues['children_dummy']:.4f}")

# 6. Interpretable ML model using EBM
print("\n6. EXPLAINABLE BOOSTING MACHINE (EBM)")
print("-"*50)

X_features = df[['children_dummy', 'age', 'yearsmarried', 'religiousness', 'education', 'rating', 'gender_dummy']]
ebm = ExplainableBoostingRegressor(random_state=42)
ebm.fit(X_features, y)

print("\nFeature Importances from EBM:")
feature_names = X_features.columns.tolist()
for name, importance in zip(feature_names, ebm.term_importances()):
    print(f"  {name}: {importance:.4f}")

# 7. Correlation analysis
print("\n7. CORRELATION ANALYSIS")
print("-"*50)
corr = df[['affairs', 'children_dummy']].corr()
print(corr)

# Determine the conclusion
print("\n" + "="*70)
print("CONCLUSION")
print("="*70)

# Analysis of results
has_significant_effect = p_value < 0.05
coefficient_simple = model_simple.params['children_dummy']
coefficient_multi = model_multi.params['children_dummy']
p_value_multi = model_multi.pvalues['children_dummy']

print(f"\nKey findings:")
print(f"1. Mean difference: {children_yes.mean() - children_no.mean():.3f}")
print(f"   (Those WITH children have {'more' if coefficient_simple > 0 else 'fewer'} affairs on average)")
print(f"2. T-test p-value: {p_value:.4f} ({'significant' if has_significant_effect else 'not significant'})")
print(f"3. Simple regression coefficient: {coefficient_simple:.4f} (p={model_simple.pvalues['children_dummy']:.4f})")
print(f"4. Multiple regression coefficient: {coefficient_multi:.4f} (p={p_value_multi:.4f})")

# Determine response score (0-100 scale)
# The question asks: "Does having children DECREASE engagement in affairs?"
# If coefficient is negative and significant -> YES (high score)
# If coefficient is negative but not significant -> MAYBE (medium score)
# If coefficient is positive -> NO (low score)

if coefficient_multi < 0 and p_value_multi < 0.05:
    # Significant negative effect - having children DOES decrease affairs
    response_score = 75  # Strong Yes
    explanation = f"Yes, having children significantly decreases extramarital affairs. Multiple regression controlling for age, years married, religiousness, education, marital rating, and gender shows a coefficient of {coefficient_multi:.3f} (p={p_value_multi:.4f}), indicating that people with children engage in fewer affairs. The effect is statistically significant at the 0.05 level."
elif coefficient_multi < 0 and p_value_multi < 0.10:
    # Marginally significant negative effect
    response_score = 60  # Moderate Yes
    explanation = f"There is moderate evidence that having children decreases extramarital affairs. The coefficient is {coefficient_multi:.3f} (p={p_value_multi:.4f}), suggesting fewer affairs among those with children, though the effect is only marginally significant."
elif coefficient_multi < 0:
    # Negative but not significant
    response_score = 40  # Weak evidence
    explanation = f"There is weak evidence for the hypothesis. While the coefficient is negative ({coefficient_multi:.3f}), suggesting fewer affairs among those with children, the effect is not statistically significant (p={p_value_multi:.4f})."
elif coefficient_multi > 0 and p_value_multi < 0.05:
    # Significant positive effect - opposite direction
    response_score = 15  # Strong No
    explanation = f"No, the evidence suggests the opposite. Having children is associated with MORE affairs (coefficient: {coefficient_multi:.3f}, p={p_value_multi:.4f}). This contradicts the hypothesis."
else:
    # Not significant, direction unclear
    response_score = 35  # Unclear/No strong evidence
    explanation = f"There is no clear evidence that having children decreases affairs. The coefficient is {coefficient_multi:.3f} (p={p_value_multi:.4f}), which is not statistically significant."

print(f"\nFinal Assessment:")
print(f"Response Score: {response_score}/100")
print(f"Explanation: {explanation}")

# Write conclusion to file
conclusion = {
    "response": response_score,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\nConclusion written to conclusion.txt")
