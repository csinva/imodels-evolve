#!/usr/bin/env python3
import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('panda_nuts.csv')

# Create efficiency metric: nuts opened per second
df['efficiency'] = df['nuts_opened'] / df['seconds']

# Display basic statistics
print("=" * 80)
print("DATA EXPLORATION")
print("=" * 80)
print("\nDataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nSummary statistics:")
print(df.describe())
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())

# Explore the efficiency variable
print("\n" + "=" * 80)
print("EFFICIENCY ANALYSIS (nuts_opened/seconds)")
print("=" * 80)
print(f"\nMean efficiency: {df['efficiency'].mean():.4f} nuts/second")
print(f"Median efficiency: {df['efficiency'].median():.4f} nuts/second")
print(f"Std efficiency: {df['efficiency'].std():.4f} nuts/second")

# Analyze by age
print("\n" + "=" * 80)
print("1. AGE INFLUENCE ON EFFICIENCY")
print("=" * 80)
print("\nEfficiency by age:")
age_stats = df.groupby('age')['efficiency'].agg(['mean', 'std', 'count'])
print(age_stats)

# Correlation between age and efficiency
age_corr, age_pval = stats.pearsonr(df['age'], df['efficiency'])
print(f"\nPearson correlation (age vs efficiency): r={age_corr:.4f}, p={age_pval:.4f}")

# Linear regression for age
X_age = df[['age']].values
y = df['efficiency'].values
model_age = LinearRegression()
model_age.fit(X_age, y)
print(f"Linear regression coefficient: {model_age.coef_[0]:.6f}")
print(f"Interpretation: Each additional year of age is associated with {model_age.coef_[0]:.6f} change in efficiency")

# Analyze by sex
print("\n" + "=" * 80)
print("2. SEX INFLUENCE ON EFFICIENCY")
print("=" * 80)
print("\nEfficiency by sex:")
sex_stats = df.groupby('sex')['efficiency'].agg(['mean', 'std', 'count'])
print(sex_stats)

# T-test for sex differences
female_eff = df[df['sex'] == 'f']['efficiency']
male_eff = df[df['sex'] == 'm']['efficiency']
sex_tstat, sex_pval = stats.ttest_ind(female_eff, male_eff)
print(f"\nIndependent t-test (sex): t={sex_tstat:.4f}, p={sex_pval:.4f}")
print(f"Mean difference (f-m): {female_eff.mean() - male_eff.mean():.4f}")

# Analyze by help
print("\n" + "=" * 80)
print("3. HELP INFLUENCE ON EFFICIENCY")
print("=" * 80)
print("\nEfficiency by help status:")
help_stats = df.groupby('help')['efficiency'].agg(['mean', 'std', 'count'])
print(help_stats)

# T-test for help differences
help_y = df[df['help'] == 'y']['efficiency']
help_n = df[df['help'] == 'N']['efficiency']
help_tstat, help_pval = stats.ttest_ind(help_y, help_n)
print(f"\nIndependent t-test (help): t={help_tstat:.4f}, p={help_pval:.4f}")
print(f"Mean difference (yes-no): {help_y.mean() - help_n.mean():.4f}")

# Multiple regression analysis with all three variables
print("\n" + "=" * 80)
print("4. MULTIPLE REGRESSION: AGE + SEX + HELP -> EFFICIENCY")
print("=" * 80)

# Encode categorical variables
df['sex_encoded'] = LabelEncoder().fit_transform(df['sex'])  # f=0, m=1
df['help_encoded'] = LabelEncoder().fit_transform(df['help'])  # N=0, y=1

# Prepare data for statsmodels OLS (to get p-values)
X = df[['age', 'sex_encoded', 'help_encoded']]
X = sm.add_constant(X)  # Add intercept
y = df['efficiency']

# Fit OLS model
model = sm.OLS(y, X).fit()
print(model.summary())

# Extract coefficients and p-values
print("\n" + "=" * 80)
print("COEFFICIENT INTERPRETATION")
print("=" * 80)
coef_age = model.params['age']
pval_age = model.pvalues['age']
coef_sex = model.params['sex_encoded']
pval_sex = model.pvalues['sex_encoded']
coef_help = model.params['help_encoded']
pval_help = model.pvalues['help_encoded']

print(f"\nAge: coefficient={coef_age:.6f}, p-value={pval_age:.4f}")
print(f"  Significant: {pval_age < 0.05}")
print(f"  Effect size: {'Moderate' if abs(coef_age) > 0.01 else 'Small'}")

print(f"\nSex (male vs female): coefficient={coef_sex:.6f}, p-value={pval_sex:.4f}")
print(f"  Significant: {pval_sex < 0.05}")
print(f"  Effect size: {'Moderate' if abs(coef_sex) > 0.01 else 'Small'}")

print(f"\nHelp (yes vs no): coefficient={coef_help:.6f}, p-value={pval_help:.4f}")
print(f"  Significant: {pval_help < 0.05}")
print(f"  Effect size: {'Moderate' if abs(coef_help) > 0.01 else 'Small'}")

print(f"\nModel R-squared: {model.rsquared:.4f}")
print(f"Adjusted R-squared: {model.rsquared_adj:.4f}")

# Decision: Generate conclusion based on statistical significance
print("\n" + "=" * 80)
print("FINAL CONCLUSION")
print("=" * 80)

# Count significant factors
significant_factors = []
if pval_age < 0.05:
    significant_factors.append(f"age (p={pval_age:.4f})")
if pval_sex < 0.05:
    significant_factors.append(f"sex (p={pval_sex:.4f})")
if pval_help < 0.05:
    significant_factors.append(f"help (p={pval_help:.4f})")

num_significant = len(significant_factors)

# Calculate response score (0-100 Likert scale)
# All 3 significant -> high score (80-100)
# 2 significant -> moderate-high score (60-80)
# 1 significant -> moderate score (40-60)
# 0 significant -> low score (0-30)

if num_significant == 3:
    response_score = 85
    explanation = f"Multiple regression analysis shows that all three factors significantly influence nut-cracking efficiency. Age (coef={coef_age:.4f}, p={pval_age:.4f}), sex (coef={coef_sex:.4f}, p={pval_sex:.4f}), and receiving help (coef={coef_help:.4f}, p={pval_help:.4f}) are all statistically significant predictors. The model explains {model.rsquared*100:.1f}% of variance in efficiency."
elif num_significant == 2:
    response_score = 70
    explanation = f"Multiple regression analysis shows that {num_significant} of the three factors significantly influence nut-cracking efficiency: {', '.join(significant_factors)}. The model explains {model.rsquared*100:.1f}% of variance in efficiency."
elif num_significant == 1:
    response_score = 50
    explanation = f"Multiple regression analysis shows that only {significant_factors[0]} significantly influences nut-cracking efficiency. The other factors do not show significant effects. The model explains {model.rsquared*100:.1f}% of variance in efficiency."
else:
    response_score = 20
    explanation = f"Multiple regression analysis indicates that none of the three factors (age p={pval_age:.4f}, sex p={pval_sex:.4f}, help p={pval_help:.4f}) significantly influence nut-cracking efficiency at the conventional alpha=0.05 level. The model explains only {model.rsquared*100:.1f}% of variance."

print(f"\nSignificant factors: {num_significant}/3")
if significant_factors:
    for factor in significant_factors:
        print(f"  - {factor}")
else:
    print("  - None")

print(f"\nLikert response score: {response_score}/100")
print(f"\nExplanation: {explanation}")

# Write conclusion to file
conclusion = {
    "response": response_score,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE - conclusion.txt created")
print("=" * 80)
