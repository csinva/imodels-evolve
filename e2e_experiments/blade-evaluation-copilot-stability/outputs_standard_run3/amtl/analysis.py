import pandas as pd
import numpy as np
import json
from scipy import stats
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load the data
df = pd.read_csv('amtl.csv')

print("=" * 80)
print("DATASET OVERVIEW")
print("=" * 80)
print(f"Shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())
print("\nBasic statistics:")
print(df.describe())

# Explore the data by genus
print("\n" + "=" * 80)
print("EXPLORATORY ANALYSIS BY GENUS")
print("=" * 80)
print("\nGenus distribution:")
print(df['genus'].value_counts())

# Calculate AMTL rate (proportion of missing teeth)
df['amtl_rate'] = df['num_amtl'] / df['sockets']

print("\n\nAMTL Rate by Genus:")
genus_summary = df.groupby('genus').agg({
    'amtl_rate': ['mean', 'std', 'count'],
    'age': 'mean',
    'prob_male': 'mean'
})
print(genus_summary)

# Create binary indicator for Homo sapiens
df['is_human'] = (df['genus'] == 'Homo sapiens').astype(int)

# Encode tooth_class as dummy variables
tooth_class_dummies = pd.get_dummies(df['tooth_class'], prefix='tooth', drop_first=True)
df = pd.concat([df, tooth_class_dummies], axis=1)

print("\n" + "=" * 80)
print("CORRELATION ANALYSIS")
print("=" * 80)
print("\nCorrelation between AMTL rate and key variables:")
corr_vars = ['amtl_rate', 'age', 'prob_male', 'is_human']
print(df[corr_vars].corr())

# Test for difference in AMTL rates between humans and non-human primates
print("\n" + "=" * 80)
print("STATISTICAL TESTS: HUMANS vs NON-HUMAN PRIMATES")
print("=" * 80)

human_amtl = df[df['is_human'] == 1]['amtl_rate']
non_human_amtl = df[df['is_human'] == 0]['amtl_rate']

print(f"\nHuman AMTL rate: Mean={human_amtl.mean():.4f}, Std={human_amtl.std():.4f}, N={len(human_amtl)}")
print(f"Non-human AMTL rate: Mean={non_human_amtl.mean():.4f}, Std={non_human_amtl.std():.4f}, N={len(non_human_amtl)}")

# T-test
t_stat, t_pval = stats.ttest_ind(human_amtl, non_human_amtl)
print(f"\nT-test: t={t_stat:.4f}, p-value={t_pval:.6f}")

# Mann-Whitney U test (non-parametric alternative)
u_stat, u_pval = stats.mannwhitneyu(human_amtl, non_human_amtl, alternative='two-sided')
print(f"Mann-Whitney U test: U={u_stat:.4f}, p-value={u_pval:.6f}")

# Linear regression with controls
print("\n" + "=" * 80)
print("LINEAR REGRESSION: AMTL RATE ~ HUMAN + AGE + SEX + TOOTH_CLASS")
print("=" * 80)

# Prepare data for regression
reg_data = df[['amtl_rate', 'is_human', 'age', 'prob_male'] + 
              [col for col in df.columns if col.startswith('tooth_')]].dropna()

# Fit OLS regression with statsmodels for p-values
formula = 'amtl_rate ~ is_human + age + prob_male'
if 'tooth_Posterior' in reg_data.columns:
    formula += ' + tooth_Posterior'
if 'tooth_Premolar' in reg_data.columns:
    formula += ' + tooth_Premolar'

ols_model = smf.ols(formula, data=reg_data).fit()
print(ols_model.summary())

# Extract key results
human_coef = ols_model.params['is_human']
human_pval = ols_model.pvalues['is_human']
human_ci = ols_model.conf_int().loc['is_human']

print("\n" + "=" * 80)
print("KEY FINDING: HUMAN COEFFICIENT")
print("=" * 80)
print(f"Coefficient for 'is_human': {human_coef:.6f}")
print(f"95% CI: [{human_ci[0]:.6f}, {human_ci[1]:.6f}]")
print(f"P-value: {human_pval:.6f}")
print(f"Significant at 0.05 level: {human_pval < 0.05}")

# Binomial regression (logistic regression for proportions)
# We need to model num_amtl with sockets as weights
print("\n" + "=" * 80)
print("BINOMIAL REGRESSION: NUM_AMTL ~ HUMAN + AGE + SEX + TOOTH_CLASS")
print("=" * 80)

# Use statsmodels for binomial regression
binomial_data = df[['num_amtl', 'sockets', 'is_human', 'age', 'prob_male'] + 
                    [col for col in df.columns if col.startswith('tooth_')]].dropna()

# Create proportion and prepare for GLM
binomial_data['prop_amtl'] = binomial_data['num_amtl'] / binomial_data['sockets']

# GLM with binomial family
formula_binom = 'prop_amtl ~ is_human + age + prob_male'
if 'tooth_Posterior' in binomial_data.columns:
    formula_binom += ' + tooth_Posterior'
if 'tooth_Premolar' in binomial_data.columns:
    formula_binom += ' + tooth_Premolar'

glm_model = smf.glm(formula_binom, data=binomial_data, 
                     family=sm.families.Binomial(), 
                     var_weights=binomial_data['sockets']).fit()
print(glm_model.summary())

glm_human_coef = glm_model.params['is_human']
glm_human_pval = glm_model.pvalues['is_human']
glm_human_ci = glm_model.conf_int().loc['is_human']

print("\n" + "=" * 80)
print("KEY FINDING: HUMAN COEFFICIENT (BINOMIAL MODEL)")
print("=" * 80)
print(f"Coefficient for 'is_human': {glm_human_coef:.6f}")
print(f"95% CI: [{glm_human_ci[0]:.6f}, {glm_human_ci[1]:.6f}]")
print(f"P-value: {glm_human_pval:.6f}")
print(f"Significant at 0.05 level: {glm_human_pval < 0.05}")

# ANOVA by genus
print("\n" + "=" * 80)
print("ANOVA: AMTL RATE BY GENUS")
print("=" * 80)

genus_groups = [df[df['genus'] == g]['amtl_rate'].dropna() for g in df['genus'].unique()]
f_stat, anova_pval = stats.f_oneway(*genus_groups)
print(f"F-statistic: {f_stat:.4f}")
print(f"P-value: {anova_pval:.6f}")

# Determine conclusion
print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

# Decision logic based on statistical evidence
# The research question asks if humans have HIGHER frequencies of AMTL after accounting for age, sex, and tooth class

if glm_human_pval < 0.05 and glm_human_coef > 0:
    # Significant positive effect
    response = 85
    explanation = f"Yes, humans show significantly higher AMTL rates. The binomial regression controlling for age, sex, and tooth class shows humans have a coefficient of {glm_human_coef:.4f} (p={glm_human_pval:.4f}), indicating significantly higher AMTL frequencies compared to non-human primates (Pan, Pongo, Papio)."
elif glm_human_pval < 0.05 and glm_human_coef < 0:
    # Significant negative effect (humans have LOWER AMTL)
    response = 15
    explanation = f"No, humans actually show significantly lower AMTL rates. The binomial regression controlling for age, sex, and tooth class shows humans have a coefficient of {glm_human_coef:.4f} (p={glm_human_pval:.4f}), indicating significantly lower AMTL frequencies compared to non-human primates."
elif glm_human_pval >= 0.05 and human_pval < 0.05 and human_coef > 0:
    # OLS significant but GLM not - moderate confidence
    response = 65
    explanation = f"Moderate evidence for higher AMTL in humans. Linear regression shows a significant positive effect (coef={human_coef:.4f}, p={human_pval:.4f}), but binomial regression is not significant (p={glm_human_pval:.4f}). This suggests some evidence but not conclusive."
elif glm_human_pval >= 0.05:
    # Not significant in the appropriate model
    response = 30
    explanation = f"No significant difference found. After controlling for age, sex, and tooth class in binomial regression, the human effect is not statistically significant (coef={glm_human_coef:.4f}, p={glm_human_pval:.4f}). Cannot conclude humans have higher AMTL rates."
else:
    response = 50
    explanation = "Mixed or unclear results from the analyses."

print(f"\nResponse Score: {response}")
print(f"Explanation: {explanation}")

# Write conclusion to file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE - conclusion.txt created")
print("=" * 80)
