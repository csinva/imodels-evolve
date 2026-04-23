import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
import statsmodels.api as sm
from imodels import RuleFitRegressor, FIGSRegressor
import json

# Load the data
df = pd.read_csv('affairs.csv')

print("=" * 80)
print("RESEARCH QUESTION: Does having children decrease engagement in extramarital affairs?")
print("=" * 80)

# Basic data exploration
print("\n1. DATA OVERVIEW")
print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

print(f"\nData types:")
print(df.dtypes)

print(f"\nBasic statistics:")
print(df.describe())

print(f"\nMissing values:")
print(df.isnull().sum())

# Focus on the key variables: affairs (outcome) and children (predictor)
print("\n" + "=" * 80)
print("2. DESCRIPTIVE STATISTICS BY CHILDREN STATUS")
print("=" * 80)

# Group by children and compute statistics
grouped = df.groupby('children')['affairs'].describe()
print("\nAffairs statistics by children status:")
print(grouped)

# Calculate means
affairs_with_children = df[df['children'] == 'yes']['affairs']
affairs_no_children = df[df['children'] == 'no']['affairs']

mean_with_children = affairs_with_children.mean()
mean_no_children = affairs_no_children.mean()

print(f"\nMean affairs with children: {mean_with_children:.3f}")
print(f"Mean affairs without children: {mean_no_children:.3f}")
print(f"Difference: {mean_with_children - mean_no_children:.3f}")

# Check distribution
print(f"\nProportion with any affairs (children=yes): {(affairs_with_children > 0).mean():.3f}")
print(f"Proportion with any affairs (children=no): {(affairs_no_children > 0).mean():.3f}")

# Statistical tests
print("\n" + "=" * 80)
print("3. STATISTICAL SIGNIFICANCE TESTS")
print("=" * 80)

# T-test comparing affairs between those with and without children
t_stat, p_value = stats.ttest_ind(affairs_with_children, affairs_no_children)
print(f"\nIndependent t-test:")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_value:.4f}")
print(f"  Significant at α=0.05? {p_value < 0.05}")

# Mann-Whitney U test (non-parametric alternative)
u_stat, p_value_mw = stats.mannwhitneyu(affairs_with_children, affairs_no_children, alternative='two-sided')
print(f"\nMann-Whitney U test (non-parametric):")
print(f"  U-statistic: {u_stat:.4f}")
print(f"  p-value: {p_value_mw:.4f}")
print(f"  Significant at α=0.05? {p_value_mw < 0.05}")

# Chi-square test for binary affair status
df['has_affair'] = (df['affairs'] > 0).astype(int)
df['has_children'] = (df['children'] == 'yes').astype(int)

contingency_table = pd.crosstab(df['has_children'], df['has_affair'])
print(f"\nContingency table:")
print(contingency_table)

chi2, p_value_chi2, dof, expected = stats.chi2_contingency(contingency_table)
print(f"\nChi-square test:")
print(f"  χ² statistic: {chi2:.4f}")
print(f"  p-value: {p_value_chi2:.4f}")
print(f"  Significant at α=0.05? {p_value_chi2 < 0.05}")

# Regression analysis controlling for confounders
print("\n" + "=" * 80)
print("4. REGRESSION ANALYSIS (Controlling for Confounders)")
print("=" * 80)

# Prepare data for regression
df_reg = df.copy()
df_reg['children_binary'] = (df_reg['children'] == 'yes').astype(int)
df_reg['gender_binary'] = (df_reg['gender'] == 'male').astype(int)

# Simple linear regression with just children
X_simple = df_reg[['children_binary']]
y = df_reg['affairs']

model_simple = sm.OLS(y, sm.add_constant(X_simple)).fit()
print("\nSimple regression (affairs ~ children):")
print(model_simple.summary().tables[1])

# Multiple regression controlling for confounders
X_full = df_reg[['children_binary', 'gender_binary', 'age', 'yearsmarried', 
                  'religiousness', 'education', 'occupation', 'rating']]
model_full = sm.OLS(y, sm.add_constant(X_full)).fit()
print("\nMultiple regression (controlling for confounders):")
print(model_full.summary().tables[1])

# Extract coefficient and p-value for children
children_coef = model_full.params['children_binary']
children_pval = model_full.pvalues['children_binary']
print(f"\nChildren coefficient: {children_coef:.4f}")
print(f"Children p-value: {children_pval:.4f}")
print(f"Significant at α=0.05? {children_pval < 0.05}")

# Interpretable models
print("\n" + "=" * 80)
print("5. INTERPRETABLE MODELS")
print("=" * 80)

# Decision Tree
X_tree = df_reg[['children_binary', 'gender_binary', 'age', 'yearsmarried', 
                  'religiousness', 'education', 'occupation', 'rating']].values
y_tree = df_reg['affairs'].values

tree = DecisionTreeRegressor(max_depth=4, random_state=42)
tree.fit(X_tree, y_tree)

feature_names = ['children', 'gender', 'age', 'yearsmarried', 
                 'religiousness', 'education', 'occupation', 'rating']
importances = tree.feature_importances_

print("\nDecision Tree Feature Importance:")
for name, imp in zip(feature_names, importances):
    print(f"  {name}: {imp:.4f}")

# FIGS Regressor (interpretable tree-based model from imodels)
try:
    figs = FIGSRegressor(max_rules=10, random_state=42)
    figs.fit(X_tree, y_tree)
    print("\nFIGS Model trained successfully")
    print(f"Number of rules: {len(figs.trees_)}")
    print(f"FIGS Feature importance for 'children': {figs.feature_importances_[0]:.4f}")
except Exception as e:
    print(f"\nFIGS model error: {e}")

# Effect size calculation
print("\n" + "=" * 80)
print("6. EFFECT SIZE")
print("=" * 80)

# Cohen's d
pooled_std = np.sqrt(((len(affairs_with_children) - 1) * affairs_with_children.std()**2 + 
                       (len(affairs_no_children) - 1) * affairs_no_children.std()**2) / 
                      (len(affairs_with_children) + len(affairs_no_children) - 2))
cohens_d = (mean_with_children - mean_no_children) / pooled_std

print(f"\nCohen's d: {cohens_d:.4f}")
print(f"Effect size interpretation: ", end="")
if abs(cohens_d) < 0.2:
    print("negligible")
elif abs(cohens_d) < 0.5:
    print("small")
elif abs(cohens_d) < 0.8:
    print("medium")
else:
    print("large")

# Summary and conclusion
print("\n" + "=" * 80)
print("7. CONCLUSION")
print("=" * 80)

# Determine response based on evidence
response_score = 0
explanation_parts = []

# Factor 1: Direction of effect
if mean_with_children < mean_no_children:
    explanation_parts.append(f"People with children have LOWER mean affairs ({mean_with_children:.2f}) than those without ({mean_no_children:.2f})")
    direction_points = 40
else:
    explanation_parts.append(f"People with children have HIGHER mean affairs ({mean_with_children:.2f}) than those without ({mean_no_children:.2f})")
    direction_points = 0

# Factor 2: Statistical significance
if p_value < 0.05 or p_value_chi2 < 0.05:
    explanation_parts.append(f"Difference is statistically significant (t-test p={p_value:.4f}, chi-square p={p_value_chi2:.4f})")
    sig_points = 40
else:
    explanation_parts.append(f"Difference is NOT statistically significant (t-test p={p_value:.4f}, chi-square p={p_value_chi2:.4f})")
    sig_points = 0

# Factor 3: Regression controlling for confounders
if children_pval < 0.05 and children_coef < 0:
    explanation_parts.append(f"Multiple regression confirms negative effect controlling for confounders (coef={children_coef:.4f}, p={children_pval:.4f})")
    reg_points = 20
elif children_pval < 0.05 and children_coef > 0:
    explanation_parts.append(f"Multiple regression shows positive effect (coef={children_coef:.4f}, p={children_pval:.4f})")
    reg_points = 0
else:
    explanation_parts.append(f"Multiple regression shows no significant effect (p={children_pval:.4f})")
    reg_points = 0

response_score = direction_points + sig_points + reg_points

# Adjust based on overall evidence
if response_score > 0 and abs(cohens_d) < 0.1:
    explanation_parts.append(f"However, effect size is very small (Cohen's d={cohens_d:.4f})")
    response_score = max(response_score - 20, 20)

print("\nBased on the analysis:")
for part in explanation_parts:
    print(f"  • {part}")

print(f"\nFinal assessment score: {response_score}/100")

if response_score >= 70:
    print("CONCLUSION: Strong evidence that having children DECREASES extramarital affairs")
elif response_score >= 50:
    print("CONCLUSION: Moderate evidence that having children DECREASES extramarital affairs")
elif response_score >= 30:
    print("CONCLUSION: Weak evidence that having children DECREASES extramarital affairs")
else:
    print("CONCLUSION: No evidence that having children decreases extramarital affairs")

# Write conclusion to file
explanation = " ".join(explanation_parts)
conclusion = {
    "response": response_score,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print(f"\nConclusion written to conclusion.txt")
print("=" * 80)
