import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
import json
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('panda_nuts.csv')

print("="*60)
print("DATASET OVERVIEW")
print("="*60)
print(f"Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head(10))

# Create efficiency metric: nuts opened per second
df['efficiency'] = df['nuts_opened'] / df['seconds']

print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(df.describe())
print(f"\nEfficiency statistics:")
print(df['efficiency'].describe())

# Encode categorical variables for modeling
df['sex_encoded'] = LabelEncoder().fit_transform(df['sex'])  # f=0, m=1
df['help_encoded'] = LabelEncoder().fit_transform(df['help'])  # N=0, y=1

print("\n" + "="*60)
print("RESEARCH QUESTION ANALYSIS")
print("="*60)
print("How do age, sex, and receiving help influence nut-cracking efficiency?")
print()

# 1. Correlation analysis
print("\n1. CORRELATION ANALYSIS")
print("-"*60)
corr_vars = ['age', 'sex_encoded', 'help_encoded', 'efficiency']
corr_matrix = df[corr_vars].corr()
print("Correlation with efficiency:")
print(corr_matrix['efficiency'].sort_values(ascending=False))

# 2. Statistical tests for each predictor
print("\n2. STATISTICAL SIGNIFICANCE TESTS")
print("-"*60)

# Age vs Efficiency (Pearson correlation)
age_corr, age_pval = stats.pearsonr(df['age'], df['efficiency'])
print(f"\nAge vs Efficiency:")
print(f"  Pearson correlation: r = {age_corr:.4f}, p = {age_pval:.4f}")
print(f"  Significant: {age_pval < 0.05}")

# Sex vs Efficiency (t-test)
male_eff = df[df['sex'] == 'm']['efficiency']
female_eff = df[df['sex'] == 'f']['efficiency']
sex_tstat, sex_pval = stats.ttest_ind(male_eff, female_eff, nan_policy='omit')
print(f"\nSex vs Efficiency (t-test):")
print(f"  Male efficiency mean: {male_eff.mean():.4f}")
print(f"  Female efficiency mean: {female_eff.mean():.4f}")
print(f"  t-statistic: {sex_tstat:.4f}, p = {sex_pval:.4f}")
print(f"  Significant: {sex_pval < 0.05}")

# Help vs Efficiency (t-test)
help_yes = df[df['help'] == 'y']['efficiency']
help_no = df[df['help'] == 'N']['efficiency']
help_tstat, help_pval = stats.ttest_ind(help_yes, help_no, nan_policy='omit')
print(f"\nHelp vs Efficiency (t-test):")
print(f"  With help mean: {help_yes.mean():.4f}")
print(f"  Without help mean: {help_no.mean():.4f}")
print(f"  t-statistic: {help_tstat:.4f}, p = {help_pval:.4f}")
print(f"  Significant: {help_pval < 0.05}")

# 3. Linear regression with statsmodels (for p-values)
print("\n3. MULTIPLE LINEAR REGRESSION")
print("-"*60)
X = df[['age', 'sex_encoded', 'help_encoded']]
y = df['efficiency']

# Remove any NaN values
mask = ~(X.isna().any(axis=1) | y.isna())
X_clean = X[mask]
y_clean = y[mask]

X_with_const = sm.add_constant(X_clean)
model = sm.OLS(y_clean, X_with_const).fit()
print(model.summary())

# Extract coefficients and p-values
print("\nCoefficient interpretation:")
var_names = ['const', 'age', 'sex_encoded', 'help_encoded']
for var in var_names:
    coef = model.params[var]
    pval = model.pvalues[var]
    print(f"  {var}: coef = {coef:.4f}, p = {pval:.4f}, significant = {pval < 0.05}")

# 4. Decision tree for interpretability
print("\n4. DECISION TREE ANALYSIS")
print("-"*60)
dt_model = DecisionTreeRegressor(max_depth=3, random_state=42)
dt_model.fit(X_clean, y_clean)
feature_importance = dt_model.feature_importances_
print("Feature importances:")
for i, var in enumerate(['age', 'sex_encoded', 'help_encoded']):
    print(f"  {var}: {feature_importance[i]:.4f}")

# 5. R-squared to assess overall model fit
print("\n5. MODEL FIT")
print("-"*60)
print(f"R-squared: {model.rsquared:.4f}")
print(f"Adjusted R-squared: {model.rsquared_adj:.4f}")
print(f"F-statistic: {model.fvalue:.4f}, p = {model.f_pvalue:.4f}")

# Determine conclusion
print("\n" + "="*60)
print("CONCLUSION")
print("="*60)

# Count significant predictors
significant_count = 0
explanations = []

if age_pval < 0.05:
    significant_count += 1
    explanations.append(f"age (p={age_pval:.4f}, r={age_corr:.3f})")
    
if sex_pval < 0.05:
    significant_count += 1
    explanations.append(f"sex (p={sex_pval:.4f})")
    
if help_pval < 0.05:
    significant_count += 1
    explanations.append(f"help (p={help_pval:.4f})")

# Also check overall model significance
overall_significant = model.f_pvalue < 0.05

print(f"\nSignificant predictors: {significant_count}/3")
print(f"Overall model p-value: {model.f_pvalue:.4f}")
print(f"Model R-squared: {model.rsquared:.4f}")

# Determine response score (0-100)
if overall_significant and significant_count >= 1:
    # At least one predictor is significant and overall model is significant
    if significant_count == 3:
        response = 85  # Strong yes - all three influence efficiency
    elif significant_count == 2:
        response = 70  # Moderate-strong yes - two influence
    else:
        response = 60  # Moderate yes - one influences
else:
    # No significant relationship or weak model
    if model.f_pvalue < 0.10:
        response = 40  # Weak relationship
    else:
        response = 20  # No significant relationship

# Build explanation
if significant_count > 0:
    explanation = f"Statistical analysis shows that {', '.join(explanations)} significantly influence(s) nut-cracking efficiency (nuts/second). "
    if significant_count == 3:
        explanation += "All three variables have significant effects. "
    elif significant_count == 2:
        explanation += "Two of the three variables have significant effects. "
    else:
        explanation += "One variable has a significant effect. "
    explanation += f"The overall model is {'significant' if overall_significant else 'not significant'} (F-test p={model.f_pvalue:.4f}) with R²={model.rsquared:.3f}."
else:
    explanation = f"Statistical analysis shows no significant individual effects from age (p={age_pval:.3f}), sex (p={sex_pval:.3f}), or help (p={help_pval:.3f}) on nut-cracking efficiency. The overall model is not significant (F-test p={model.f_pvalue:.4f}, R²={model.rsquared:.3f})."

print(f"\nResponse score: {response}/100")
print(f"Explanation: {explanation}")

# Write conclusion to file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n" + "="*60)
print("Analysis complete. Results written to conclusion.txt")
print("="*60)
