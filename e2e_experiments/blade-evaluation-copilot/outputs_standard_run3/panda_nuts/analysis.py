import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
import json

# Load the dataset
df = pd.read_csv('panda_nuts.csv')

# Display basic info
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nSummary statistics:")
print(df.describe())

# Research question: How do age, sex, and receiving help from another chimpanzee 
# influence the nut-cracking efficiency of western chimpanzees?

# Define efficiency as nuts opened per second
df['efficiency'] = df['nuts_opened'] / df['seconds']

print("\n" + "="*80)
print("RESEARCH QUESTION ANALYSIS")
print("="*80)
print("\nEfficiency metric: nuts_opened / seconds")
print("Mean efficiency:", df['efficiency'].mean())
print("Std efficiency:", df['efficiency'].std())

# Encode categorical variables
sex_encoded = LabelEncoder().fit_transform(df['sex'])
help_encoded = LabelEncoder().fit_transform(df['help'])

print("\n" + "="*80)
print("1. CORRELATION ANALYSIS")
print("="*80)

# Correlation between age and efficiency
corr_age, p_age = stats.pearsonr(df['age'], df['efficiency'])
print(f"\nAge vs Efficiency:")
print(f"  Correlation: {corr_age:.4f}")
print(f"  P-value: {p_age:.4f}")
print(f"  Significant: {p_age < 0.05}")

# Sex and efficiency (t-test)
male_eff = df[df['sex'] == 'm']['efficiency']
female_eff = df[df['sex'] == 'f']['efficiency']
t_stat_sex, p_sex = stats.ttest_ind(male_eff, female_eff)
print(f"\nSex vs Efficiency (t-test):")
print(f"  Male mean: {male_eff.mean():.4f}")
print(f"  Female mean: {female_eff.mean():.4f}")
print(f"  T-statistic: {t_stat_sex:.4f}")
print(f"  P-value: {p_sex:.4f}")
print(f"  Significant: {p_sex < 0.05}")

# Help and efficiency (t-test)
help_yes = df[df['help'] == 'y']['efficiency']
help_no = df[df['help'] == 'N']['efficiency']
t_stat_help, p_help = stats.ttest_ind(help_yes, help_no)
print(f"\nHelp vs Efficiency (t-test):")
print(f"  With help mean: {help_yes.mean():.4f}")
print(f"  Without help mean: {help_no.mean():.4f}")
print(f"  T-statistic: {t_stat_help:.4f}")
print(f"  P-value: {p_help:.4f}")
print(f"  Significant: {p_help < 0.05}")

print("\n" + "="*80)
print("2. MULTIPLE REGRESSION ANALYSIS")
print("="*80)

# Prepare data for regression
X = pd.DataFrame({
    'age': df['age'],
    'sex': sex_encoded,
    'help': help_encoded
})
y = df['efficiency']

# Add constant for statsmodels
X_sm = sm.add_constant(X)

# Fit OLS model to get p-values
model_ols = sm.OLS(y, X_sm).fit()
print("\nOLS Regression Summary:")
print(model_ols.summary())

print("\n" + "="*80)
print("3. INTERPRETABLE MODEL - LINEAR REGRESSION")
print("="*80)

# Fit sklearn linear regression
lr = LinearRegression()
lr.fit(X, y)

print("\nLinear Regression Coefficients:")
print(f"  Intercept: {lr.intercept_:.4f}")
for i, col in enumerate(X.columns):
    print(f"  {col}: {lr.coef_[i]:.4f}")

print(f"\nR² score: {lr.score(X, y):.4f}")

print("\n" + "="*80)
print("4. INTERPRETABLE MODEL - DECISION TREE")
print("="*80)

# Fit decision tree
dt = DecisionTreeRegressor(max_depth=3, random_state=42)
dt.fit(X, y)

print("\nDecision Tree Feature Importances:")
for i, col in enumerate(X.columns):
    print(f"  {col}: {dt.feature_importances_[i]:.4f}")

print(f"\nR² score: {dt.score(X, y):.4f}")

print("\n" + "="*80)
print("5. CONCLUSION")
print("="*80)

# Analyze significance
significant_factors = []
if p_age < 0.05:
    significant_factors.append(f"age (p={p_age:.4f})")
if p_sex < 0.05:
    significant_factors.append(f"sex (p={p_sex:.4f})")
if p_help < 0.05:
    significant_factors.append(f"help (p={p_help:.4f})")

# Check regression coefficients p-values from OLS
ols_pvalues = model_ols.pvalues
age_p_ols = ols_pvalues['age']
sex_p_ols = ols_pvalues['sex']
help_p_ols = ols_pvalues['help']

print(f"\nStatistical significance (α=0.05):")
print(f"  Age: p={age_p_ols:.4f} {'✓ Significant' if age_p_ols < 0.05 else '✗ Not significant'}")
print(f"  Sex: p={sex_p_ols:.4f} {'✓ Significant' if sex_p_ols < 0.05 else '✗ Not significant'}")
print(f"  Help: p={help_p_ols:.4f} {'✓ Significant' if help_p_ols < 0.05 else '✗ Not significant'}")

# Determine response score
sig_count = sum([age_p_ols < 0.05, sex_p_ols < 0.05, help_p_ols < 0.05])

# Response logic: 
# - If all three are significant: high score (80-95)
# - If two are significant: moderate-high score (60-75)
# - If one is significant: moderate score (40-55)
# - If none are significant: low score (5-20)

if sig_count == 3:
    response_score = 85
    explanation = f"All three factors (age, sex, and help) show statistically significant influences on nut-cracking efficiency (p < 0.05). Age (p={age_p_ols:.4f}), sex (p={sex_p_ols:.4f}), and receiving help (p={help_p_ols:.4f}) all significantly influence efficiency. The multiple regression model explains a meaningful portion of the variance (R²={lr.score(X, y):.3f})."
elif sig_count == 2:
    sig_vars = []
    if age_p_ols < 0.05:
        sig_vars.append(f"age (p={age_p_ols:.4f})")
    if sex_p_ols < 0.05:
        sig_vars.append(f"sex (p={sex_p_ols:.4f})")
    if help_p_ols < 0.05:
        sig_vars.append(f"help (p={help_p_ols:.4f})")
    response_score = 65
    explanation = f"Two out of three factors show statistically significant influences on nut-cracking efficiency: {' and '.join(sig_vars)}. This indicates a substantial but not complete relationship between the tested factors and efficiency."
elif sig_count == 1:
    sig_var = None
    if age_p_ols < 0.05:
        sig_var = f"age (p={age_p_ols:.4f})"
    elif sex_p_ols < 0.05:
        sig_var = f"sex (p={sex_p_ols:.4f})"
    elif help_p_ols < 0.05:
        sig_var = f"help (p={help_p_ols:.4f})"
    response_score = 45
    explanation = f"Only one factor shows statistically significant influence on nut-cracking efficiency: {sig_var}. The other two factors (among age, sex, and help) do not show significant effects, indicating a partial relationship."
else:
    response_score = 15
    explanation = f"None of the three factors (age p={age_p_ols:.4f}, sex p={sex_p_ols:.4f}, help p={help_p_ols:.4f}) show statistically significant influences on nut-cracking efficiency at the α=0.05 level. The relationships are not strong enough to conclude a significant influence."

print(f"\nFinal Assessment:")
print(f"  Response Score: {response_score}/100")
print(f"  Explanation: {explanation}")

# Write conclusion to file
conclusion = {
    "response": response_score,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n✓ conclusion.txt written successfully")
