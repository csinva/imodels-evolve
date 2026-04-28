import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('panda_nuts.csv')

# Calculate nut-cracking efficiency (nuts per second)
df['efficiency'] = df['nuts_opened'] / df['seconds']

# Display basic statistics
print("=" * 60)
print("DATA EXPLORATION")
print("=" * 60)
print(f"\nDataset shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print("\nBasic statistics:")
print(df.describe())
print("\nMissing values:")
print(df.isnull().sum())

# Explore the target variable: efficiency
print("\n" + "=" * 60)
print("EFFICIENCY ANALYSIS")
print("=" * 60)
print(f"\nEfficiency statistics:")
print(f"Mean: {df['efficiency'].mean():.4f} nuts/sec")
print(f"Median: {df['efficiency'].median():.4f} nuts/sec")
print(f"Std: {df['efficiency'].std():.4f}")
print(f"Min: {df['efficiency'].min():.4f}, Max: {df['efficiency'].max():.4f}")

# Analyze by age
print("\n" + "=" * 60)
print("AGE ANALYSIS")
print("=" * 60)
print("\nEfficiency by age groups:")
age_groups = pd.cut(df['age'], bins=[0, 6, 10, 20], labels=['Young (3-6)', 'Mid (7-10)', 'Old (11+)'])
print(df.groupby(age_groups)['efficiency'].agg(['count', 'mean', 'std']))

# Correlation between age and efficiency
corr_age, pval_age = stats.pearsonr(df['age'], df['efficiency'])
print(f"\nPearson correlation (age vs efficiency): r={corr_age:.4f}, p={pval_age:.4f}")

# Spearman correlation (non-parametric)
spearman_age, spval_age = stats.spearmanr(df['age'], df['efficiency'])
print(f"Spearman correlation (age vs efficiency): rho={spearman_age:.4f}, p={spval_age:.4f}")

# Analyze by sex
print("\n" + "=" * 60)
print("SEX ANALYSIS")
print("=" * 60)
print("\nEfficiency by sex:")
print(df.groupby('sex')['efficiency'].agg(['count', 'mean', 'std']))

# T-test for sex differences
male_eff = df[df['sex'] == 'm']['efficiency']
female_eff = df[df['sex'] == 'f']['efficiency']
t_stat_sex, pval_sex = stats.ttest_ind(male_eff, female_eff)
print(f"\nT-test (sex): t={t_stat_sex:.4f}, p={pval_sex:.4f}")
print(f"Mean difference (m-f): {male_eff.mean() - female_eff.mean():.4f}")

# Analyze by help
print("\n" + "=" * 60)
print("HELP ANALYSIS")
print("=" * 60)
print("\nEfficiency by help received:")
print(df.groupby('help')['efficiency'].agg(['count', 'mean', 'std']))

# T-test for help differences
help_yes = df[df['help'] == 'y']['efficiency']
help_no = df[df['help'] == 'N']['efficiency']
t_stat_help, pval_help = stats.ttest_ind(help_yes, help_no)
print(f"\nT-test (help): t={t_stat_help:.4f}, p={pval_help:.4f}")
print(f"Mean difference (help_yes - help_no): {help_yes.mean() - help_no.mean():.4f}")

# Multiple regression analysis
print("\n" + "=" * 60)
print("MULTIPLE REGRESSION ANALYSIS")
print("=" * 60)

# Encode categorical variables
df_reg = df.copy()
df_reg['sex_encoded'] = (df_reg['sex'] == 'm').astype(int)  # 1 for male, 0 for female
df_reg['help_encoded'] = (df_reg['help'] == 'y').astype(int)  # 1 for help, 0 for no help

# Prepare features
X = df_reg[['age', 'sex_encoded', 'help_encoded']]
y = df_reg['efficiency']

# OLS regression with statsmodels for p-values
X_with_const = sm.add_constant(X)
model = sm.OLS(y, X_with_const).fit()
print("\nOLS Regression Results:")
print(model.summary())

# Extract key statistics
age_coef = model.params['age']
age_pval = model.pvalues['age']
sex_coef = model.params['sex_encoded']
sex_pval = model.pvalues['sex_encoded']
help_coef = model.params['help_encoded']
help_pval = model.pvalues['help_encoded']
r_squared = model.rsquared

print("\n" + "=" * 60)
print("KEY FINDINGS")
print("=" * 60)
print(f"\nAge coefficient: {age_coef:.4f}, p-value: {age_pval:.4f}")
print(f"Sex coefficient (male): {sex_coef:.4f}, p-value: {sex_pval:.4f}")
print(f"Help coefficient: {help_coef:.4f}, p-value: {help_pval:.4f}")
print(f"R-squared: {r_squared:.4f}")

# Determine significance level (typical threshold: 0.05)
alpha = 0.05
sig_findings = []
if age_pval < alpha:
    sig_findings.append(f"age (p={age_pval:.4f}, coef={age_coef:.4f})")
if sex_pval < alpha:
    sig_findings.append(f"sex (p={sex_pval:.4f}, coef={sex_coef:.4f})")
if help_pval < alpha:
    sig_findings.append(f"help (p={help_pval:.4f}, coef={help_coef:.4f})")

print(f"\nSignificant predictors (p < {alpha}): {', '.join(sig_findings) if sig_findings else 'None'}")

# Calculate conclusion score
# The research question asks about influence of age, sex, and help on efficiency
# Score is based on:
# 1. Statistical significance of the predictors
# 2. Effect sizes
# 3. Model fit (R-squared)

num_significant = sum([age_pval < alpha, sex_pval < alpha, help_pval < alpha])
base_score = (num_significant / 3) * 70  # Up to 70 points for significance

# Add points for model fit
r_squared_score = r_squared * 30  # Up to 30 points for R-squared

total_score = int(base_score + r_squared_score)
total_score = min(100, max(0, total_score))  # Clamp to [0, 100]

print(f"\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)
print(f"Total score: {total_score}/100")

# Generate explanation
explanation_parts = []

if age_pval < alpha:
    direction = "positive" if age_coef > 0 else "negative"
    explanation_parts.append(f"Age has a significant {direction} effect (p={age_pval:.4f}, coef={age_coef:.4f})")
else:
    explanation_parts.append(f"Age does not significantly influence efficiency (p={age_pval:.4f})")

if sex_pval < alpha:
    direction = "males" if sex_coef > 0 else "females"
    explanation_parts.append(f"Sex is significant with {direction} showing higher efficiency (p={sex_pval:.4f})")
else:
    explanation_parts.append(f"Sex does not significantly influence efficiency (p={sex_pval:.4f})")

if help_pval < alpha:
    direction = "increases" if help_coef > 0 else "decreases"
    explanation_parts.append(f"Receiving help significantly {direction} efficiency (p={help_pval:.4f}, coef={help_coef:.4f})")
else:
    explanation_parts.append(f"Receiving help does not significantly influence efficiency (p={help_pval:.4f})")

explanation_parts.append(f"The model explains {r_squared*100:.1f}% of variance in nut-cracking efficiency")

explanation = ". ".join(explanation_parts) + "."

# Write conclusion to file
conclusion = {
    "response": total_score,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print(f"\nExplanation: {explanation}")
print(f"\nConclusion written to conclusion.txt")
