import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import LabelEncoder
import json

# Load the data
df = pd.read_csv('panda_nuts.csv')

print("=" * 80)
print("DATASET OVERVIEW")
print("=" * 80)
print(f"Shape: {df.shape}")
print("\nFirst few rows:")
print(df.head(10))
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())

# Calculate nut-cracking efficiency (nuts per second)
df['efficiency'] = df['nuts_opened'] / df['seconds']
df['efficiency'] = df['efficiency'].replace([np.inf, -np.inf], 0)  # Handle division by zero

print("\n" + "=" * 80)
print("DESCRIPTIVE STATISTICS")
print("=" * 80)
print("\nOverall statistics:")
print(df[['age', 'nuts_opened', 'seconds', 'efficiency']].describe())

print("\n\nEfficiency by Sex:")
print(df.groupby('sex')['efficiency'].describe())

print("\n\nEfficiency by Help Status:")
print(df.groupby('help')['efficiency'].describe())

print("\n\nCorrelation with Age:")
print(f"Pearson correlation (age vs efficiency): {stats.pearsonr(df['age'], df['efficiency'])}")

# Encode categorical variables for modeling
df['sex_encoded'] = LabelEncoder().fit_transform(df['sex'])  # f=0, m=1
df['help_encoded'] = LabelEncoder().fit_transform(df['help'])  # N=0, y=1

print("\n" + "=" * 80)
print("STATISTICAL TESTS")
print("=" * 80)

# Test 1: Effect of Sex on Efficiency
print("\n1. SEX EFFECT (t-test)")
male_eff = df[df['sex'] == 'm']['efficiency']
female_eff = df[df['sex'] == 'f']['efficiency']
sex_ttest = stats.ttest_ind(male_eff, female_eff)
print(f"   Male efficiency: {male_eff.mean():.4f} ± {male_eff.std():.4f}")
print(f"   Female efficiency: {female_eff.mean():.4f} ± {female_eff.std():.4f}")
print(f"   t-statistic: {sex_ttest.statistic:.4f}, p-value: {sex_ttest.pvalue:.6f}")
print(f"   Significant at α=0.05: {sex_ttest.pvalue < 0.05}")

# Test 2: Effect of Help on Efficiency
print("\n2. HELP EFFECT (t-test)")
help_yes = df[df['help'] == 'y']['efficiency']
help_no = df[df['help'] == 'N']['efficiency']
help_ttest = stats.ttest_ind(help_yes, help_no)
print(f"   With help efficiency: {help_yes.mean():.4f} ± {help_yes.std():.4f}")
print(f"   Without help efficiency: {help_no.mean():.4f} ± {help_no.std():.4f}")
print(f"   t-statistic: {help_ttest.statistic:.4f}, p-value: {help_ttest.pvalue:.6f}")
print(f"   Significant at α=0.05: {help_ttest.pvalue < 0.05}")

# Test 3: Effect of Age on Efficiency (correlation)
print("\n3. AGE EFFECT (Pearson correlation)")
age_corr = stats.pearsonr(df['age'], df['efficiency'])
print(f"   Correlation: {age_corr.statistic:.4f}, p-value: {age_corr.pvalue:.6f}")
print(f"   Significant at α=0.05: {age_corr.pvalue < 0.05}")

# Test 4: Linear regression with all three predictors
print("\n4. MULTIPLE REGRESSION (Age, Sex, Help)")
X = df[['age', 'sex_encoded', 'help_encoded']]
y = df['efficiency']
X_with_const = sm.add_constant(X)
model = sm.OLS(y, X_with_const).fit()
print(model.summary())

print("\n" + "=" * 80)
print("INTERPRETABLE MODEL RESULTS")
print("=" * 80)

# Simple linear regression for interpretation
lr = LinearRegression()
lr.fit(X, y)
print(f"\nLinear Regression Coefficients:")
print(f"   Age coefficient: {lr.coef_[0]:.6f}")
print(f"   Sex coefficient (male effect): {lr.coef_[1]:.6f}")
print(f"   Help coefficient: {lr.coef_[2]:.6f}")
print(f"   Intercept: {lr.intercept_:.6f}")
print(f"   R² score: {lr.score(X, y):.4f}")

print("\n" + "=" * 80)
print("CONCLUSION REASONING")
print("=" * 80)

# Analyze significance of each factor
age_significant = age_corr.pvalue < 0.05
sex_significant = sex_ttest.pvalue < 0.05
help_significant = help_ttest.pvalue < 0.05

# Check p-values from regression model
age_pval = model.pvalues['age']
sex_pval = model.pvalues['sex_encoded']
help_pval = model.pvalues['help_encoded']

print(f"\nSignificance Summary (α = 0.05):")
print(f"   Age: p = {age_pval:.6f} {'✓ Significant' if age_pval < 0.05 else '✗ Not significant'}")
print(f"   Sex: p = {sex_pval:.6f} {'✓ Significant' if sex_pval < 0.05 else '✗ Not significant'}")
print(f"   Help: p = {help_pval:.6f} {'✓ Significant' if help_pval < 0.05 else '✗ Not significant'}")

# Count significant factors
significant_factors = sum([age_pval < 0.05, sex_pval < 0.05, help_pval < 0.05])
print(f"\nNumber of significant factors: {significant_factors} out of 3")

# Determine response score based on how many factors are significant and their effect sizes
if significant_factors == 3:
    response = 90  # Strong Yes - all three factors significant
    explanation = f"All three factors significantly influence nut-cracking efficiency. Age (p={age_pval:.4f}, coef={lr.coef_[0]:.4f}), sex (p={sex_pval:.4f}, coef={lr.coef_[1]:.4f}), and receiving help (p={help_pval:.4f}, coef={lr.coef_[2]:.4f}) all show statistically significant effects in the multiple regression model."
elif significant_factors == 2:
    sig_factors = []
    if age_pval < 0.05:
        sig_factors.append(f"age (p={age_pval:.4f})")
    if sex_pval < 0.05:
        sig_factors.append(f"sex (p={sex_pval:.4f})")
    if help_pval < 0.05:
        sig_factors.append(f"help (p={help_pval:.4f})")
    response = 70  # Moderate Yes - two factors significant
    explanation = f"Two out of three factors significantly influence nut-cracking efficiency: {' and '.join(sig_factors)}. The third factor does not show a significant effect."
elif significant_factors == 1:
    if age_pval < 0.05:
        response = 50  # Moderate - only age significant
        explanation = f"Only age significantly influences nut-cracking efficiency (p={age_pval:.4f}). Sex (p={sex_pval:.4f}) and help (p={help_pval:.4f}) do not show significant effects."
    elif sex_pval < 0.05:
        response = 50  # Moderate - only sex significant
        explanation = f"Only sex significantly influences nut-cracking efficiency (p={sex_pval:.4f}). Age (p={age_pval:.4f}) and help (p={help_pval:.4f}) do not show significant effects."
    else:
        response = 50  # Moderate - only help significant
        explanation = f"Only receiving help significantly influences nut-cracking efficiency (p={help_pval:.4f}). Age (p={age_pval:.4f}) and sex (p={sex_pval:.4f}) do not show significant effects."
else:
    response = 10  # Strong No - none significant
    explanation = f"None of the three factors show statistically significant influence on nut-cracking efficiency in the multiple regression model. Age (p={age_pval:.4f}), sex (p={sex_pval:.4f}), and help (p={help_pval:.4f}) all have p-values above 0.05."

print(f"\n\nFINAL RESPONSE: {response}/100")
print(f"EXPLANATION: {explanation}")

# Write conclusion to file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n" + "=" * 80)
print("CONCLUSION SAVED TO conclusion.txt")
print("=" * 80)
