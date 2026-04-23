import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
import json

# Load the dataset
df = pd.read_csv('panda_nuts.csv')

# Research question: How do age, sex, and receiving help from another chimpanzee 
# influence the nut-cracking efficiency of western chimpanzees?

# Calculate nut-cracking efficiency (nuts per second)
df['efficiency'] = df['nuts_opened'] / df['seconds']

# Data exploration
print("=" * 80)
print("DATA EXPLORATION")
print("=" * 80)
print("\nDataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nSummary statistics:")
print(df.describe())
print("\nEfficiency statistics:")
print(df['efficiency'].describe())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Encode categorical variables
df['sex_encoded'] = (df['sex'] == 'm').astype(int)  # 1 for male, 0 for female
df['help_encoded'] = (df['help'] == 'y').astype(int)  # 1 for yes, 0 for no

print("\n" + "=" * 80)
print("CORRELATION ANALYSIS")
print("=" * 80)
# Correlation with efficiency
correlation_vars = ['age', 'sex_encoded', 'help_encoded']
print("\nCorrelations with efficiency:")
for var in correlation_vars:
    corr = df[var].corr(df['efficiency'])
    print(f"{var}: {corr:.4f}")

print("\n" + "=" * 80)
print("STATISTICAL TESTS")
print("=" * 80)

# Test 1: Age effect - correlation test
print("\n1. AGE EFFECT:")
age_corr, age_pval = stats.pearsonr(df['age'], df['efficiency'])
print(f"   Pearson correlation: r={age_corr:.4f}, p={age_pval:.4f}")
if age_pval < 0.05:
    print(f"   -> Significant relationship (p < 0.05)")
else:
    print(f"   -> No significant relationship (p >= 0.05)")

# Test 2: Sex effect - t-test
print("\n2. SEX EFFECT:")
male_eff = df[df['sex'] == 'm']['efficiency']
female_eff = df[df['sex'] == 'f']['efficiency']
sex_tstat, sex_pval = stats.ttest_ind(male_eff, female_eff)
print(f"   Male efficiency: mean={male_eff.mean():.4f}, std={male_eff.std():.4f}, n={len(male_eff)}")
print(f"   Female efficiency: mean={female_eff.mean():.4f}, std={female_eff.std():.4f}, n={len(female_eff)}")
print(f"   T-test: t={sex_tstat:.4f}, p={sex_pval:.4f}")
if sex_pval < 0.05:
    print(f"   -> Significant difference (p < 0.05)")
else:
    print(f"   -> No significant difference (p >= 0.05)")

# Test 3: Help effect - t-test
print("\n3. HELP EFFECT:")
help_yes = df[df['help'] == 'y']['efficiency']
help_no = df[df['help'] == 'N']['efficiency']
help_tstat, help_pval = stats.ttest_ind(help_yes, help_no)
print(f"   With help: mean={help_yes.mean():.4f}, std={help_yes.std():.4f}, n={len(help_yes)}")
print(f"   Without help: mean={help_no.mean():.4f}, std={help_no.std():.4f}, n={len(help_no)}")
print(f"   T-test: t={help_tstat:.4f}, p={help_pval:.4f}")
if help_pval < 0.05:
    print(f"   -> Significant difference (p < 0.05)")
else:
    print(f"   -> No significant difference (p >= 0.05)")

print("\n" + "=" * 80)
print("MULTIPLE REGRESSION ANALYSIS")
print("=" * 80)

# Multiple regression with all three predictors
X = df[['age', 'sex_encoded', 'help_encoded']]
y = df['efficiency']
X_with_const = sm.add_constant(X)
model = sm.OLS(y, X_with_const).fit()
print("\nOLS Regression Results:")
print(model.summary())

# Extract coefficients and p-values
print("\nCoefficient interpretation:")
for var in ['age', 'sex_encoded', 'help_encoded']:
    coef = model.params[var]
    pval = model.pvalues[var]
    print(f"{var}: coef={coef:.4f}, p={pval:.4f}")
    if pval < 0.05:
        print(f"   -> Significant predictor (p < 0.05)")
    else:
        print(f"   -> Not significant (p >= 0.05)")

print("\n" + "=" * 80)
print("INTERPRETABLE MODEL: DECISION TREE")
print("=" * 80)

# Decision tree for interpretability
tree = DecisionTreeRegressor(max_depth=3, random_state=42)
tree.fit(X, y)
print("\nFeature importances:")
for i, var in enumerate(['age', 'sex_encoded', 'help_encoded']):
    print(f"{var}: {tree.feature_importances_[i]:.4f}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

# Synthesize results
significant_factors = []
if age_pval < 0.05:
    significant_factors.append("age")
if sex_pval < 0.05:
    significant_factors.append("sex")
if help_pval < 0.05:
    significant_factors.append("help")

# Check model significance
model_f_pval = model.f_pvalue

print(f"\nOverall model F-test p-value: {model_f_pval:.4f}")
print(f"Significant individual factors: {significant_factors}")

# Determine response score
# Score based on:
# - Number of significant factors (0-3)
# - Overall model significance
# - Strength of relationships

if len(significant_factors) == 0:
    # No significant factors at all
    response = 10
    explanation = "Statistical tests show no significant individual relationships between age, sex, or help and nut-cracking efficiency. None of the univariate tests (age correlation, sex t-test, help t-test) reached significance (p<0.05)."
elif len(significant_factors) == 1:
    # One significant factor
    response = 50
    explanation = f"Only {significant_factors[0]} shows a statistically significant relationship with nut-cracking efficiency (p<0.05). The other factors (age, sex, help) do not show significant individual effects."
elif len(significant_factors) == 2:
    # Two significant factors
    response = 75
    explanation = f"Two factors show significant relationships: {', '.join(significant_factors)}. These variables have statistically significant effects (p<0.05) on nut-cracking efficiency, indicating moderate support for the influence of these variables."
else:
    # All three factors significant
    response = 90
    explanation = "All three factors (age, sex, and help) show statistically significant relationships with nut-cracking efficiency (p<0.05). The multiple regression model confirms that these variables collectively influence nut-cracking performance."

# Additional detail from regression model
sig_from_regression = []
for var in ['age', 'sex_encoded', 'help_encoded']:
    if model.pvalues[var] < 0.05:
        sig_from_regression.append(var.replace('_encoded', ''))

if model_f_pval < 0.05:
    explanation += f" The overall regression model is significant (F-test p={model_f_pval:.4f})."
    if len(sig_from_regression) > 0:
        explanation += f" In the multiple regression, {', '.join(sig_from_regression)} remain significant predictors."

print(f"\nResponse score: {response}")
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
