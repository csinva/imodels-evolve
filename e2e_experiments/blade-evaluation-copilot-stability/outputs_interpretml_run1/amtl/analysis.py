import pandas as pd
import numpy as np
import json
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('amtl.csv')

print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nBasic statistics:")
print(df.describe())

# Calculate AMTL rate (proportion of missing teeth)
df['amtl_rate'] = df['num_amtl'] / df['sockets']

print("\n" + "="*80)
print("RESEARCH QUESTION ANALYSIS")
print("="*80)
print("\nResearch Question: Do modern humans (Homo sapiens) have higher frequencies")
print("of AMTL compared to non-human primates (Pan, Pongo, Papio), after accounting")
print("for age, sex, and tooth class?")

# Create binary indicator for Homo sapiens vs non-human primates
df['is_human'] = (df['genus'] == 'Homo sapiens').astype(int)

print("\n" + "-"*80)
print("1. EXPLORATORY ANALYSIS")
print("-"*80)

# Summary by genus
print("\nAMTL Rate by Genus:")
genus_summary = df.groupby('genus').agg({
    'amtl_rate': ['mean', 'std', 'count'],
    'num_amtl': ['sum'],
    'sockets': ['sum']
}).round(4)
print(genus_summary)

# Calculate overall AMTL rate per genus
for genus in df['genus'].unique():
    genus_df = df[df['genus'] == genus]
    total_amtl = genus_df['num_amtl'].sum()
    total_sockets = genus_df['sockets'].sum()
    rate = total_amtl / total_sockets
    print(f"{genus}: {rate:.4f} ({total_amtl}/{total_sockets})")

print("\nAMTL Rate by Genus and Tooth Class:")
tooth_genus_summary = df.groupby(['genus', 'tooth_class'])['amtl_rate'].mean().round(4)
print(tooth_genus_summary)

# Check correlations with age and sex
print("\nCorrelation with Age (for all specimens):")
print(f"Age vs AMTL rate: {df[['age', 'amtl_rate']].corr().iloc[0,1]:.4f}")

print("\nCorrelation with Probability of Male:")
print(f"prob_male vs AMTL rate: {df[['prob_male', 'amtl_rate']].corr().iloc[0,1]:.4f}")

print("\n" + "-"*80)
print("2. STATISTICAL TESTING: T-TEST (Humans vs Non-Human Primates)")
print("-"*80)

human_amtl_rates = df[df['is_human'] == 1]['amtl_rate']
nonhuman_amtl_rates = df[df['is_human'] == 0]['amtl_rate']

print(f"\nHomo sapiens AMTL rate: mean={human_amtl_rates.mean():.4f}, std={human_amtl_rates.std():.4f}, n={len(human_amtl_rates)}")
print(f"Non-human primates AMTL rate: mean={nonhuman_amtl_rates.mean():.4f}, std={nonhuman_amtl_rates.std():.4f}, n={len(nonhuman_amtl_rates)}")

# Perform t-test
t_stat, p_value_ttest = stats.ttest_ind(human_amtl_rates, nonhuman_amtl_rates)
print(f"\nT-test: t={t_stat:.4f}, p-value={p_value_ttest:.6f}")

if p_value_ttest < 0.05:
    print("Result: SIGNIFICANT difference (p < 0.05)")
else:
    print("Result: NOT significant (p >= 0.05)")

print("\n" + "-"*80)
print("3. BINOMIAL LOGISTIC REGRESSION (Accounting for Covariates)")
print("-"*80)

# Prepare data for regression
# Create dummy variables for tooth_class
tooth_dummies = pd.get_dummies(df['tooth_class'], prefix='tooth', drop_first=True)
df_model = pd.concat([df, tooth_dummies], axis=1)

# Features: is_human, age, prob_male, tooth_class dummies
X_cols = ['is_human', 'age', 'prob_male'] + [col for col in tooth_dummies.columns]
X = df_model[X_cols].astype(float)
X = sm.add_constant(X)  # Add intercept

# Dependent variable: proportion of teeth lost (binomial)
# We'll use weighted logistic regression where we model the proportion
# For statsmodels, we can use GLM with binomial family

# Create success/failure counts for binomial regression
df_model['num_present'] = df_model['sockets'] - df_model['num_amtl']

# Use GLM with binomial family
# The endog should be proportion, and we provide counts as frequency weights
glm_binom = sm.GLM(df_model['amtl_rate'], X, 
                   family=sm.families.Binomial(),
                   freq_weights=df_model['sockets'])
result = glm_binom.fit()

print("\nBinomial GLM Results:")
print(result.summary())

# Extract key results
is_human_coef = result.params['is_human']
is_human_pval = result.pvalues['is_human']
is_human_conf = result.conf_int().loc['is_human']

print("\n" + "="*80)
print("KEY FINDING: is_human coefficient")
print("="*80)
print(f"Coefficient: {is_human_coef:.4f}")
print(f"P-value: {is_human_pval:.6f}")
print(f"95% CI: [{is_human_conf[0]:.4f}, {is_human_conf[1]:.4f}]")

if is_human_pval < 0.05:
    if is_human_coef > 0:
        direction = "HIGHER"
        print(f"\nConclusion: Homo sapiens have SIGNIFICANTLY {direction} AMTL rates")
        print(f"compared to non-human primates (p={is_human_pval:.6f} < 0.05),")
        print("after controlling for age, sex, and tooth class.")
    else:
        direction = "LOWER"
        print(f"\nConclusion: Homo sapiens have SIGNIFICANTLY {direction} AMTL rates")
        print(f"compared to non-human primates (p={is_human_pval:.6f} < 0.05),")
        print("after controlling for age, sex, and tooth class.")
else:
    print(f"\nConclusion: NO significant difference in AMTL rates between")
    print(f"Homo sapiens and non-human primates (p={is_human_pval:.6f} >= 0.05),")
    print("after controlling for age, sex, and tooth class.")

print("\n" + "-"*80)
print("4. ADDITIONAL ANALYSIS: Effect of Other Variables")
print("-"*80)

print("\nCoefficients of all predictors:")
for var in X_cols:
    if var in result.params.index:
        coef = result.params[var]
        pval = result.pvalues[var]
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        print(f"{var:20s}: coef={coef:7.4f}, p={pval:.6f} {sig}")

print("\n" + "="*80)
print("FINAL CONCLUSION")
print("="*80)

# Determine response score (0-100 Likert scale)
# 0 = strong "No", 100 = strong "Yes"
# The question asks if humans have HIGHER frequencies of AMTL

if is_human_pval < 0.001:
    # Very strong evidence
    if is_human_coef > 0:
        response_score = 95  # Strong Yes
        explanation = f"Very strong evidence (p={is_human_pval:.6f}) that Homo sapiens have significantly higher AMTL rates than non-human primates after controlling for age, sex, and tooth class. The binomial GLM coefficient for is_human is {is_human_coef:.4f} with p<0.001."
    else:
        response_score = 5  # Strong No
        explanation = f"Very strong evidence (p={is_human_pval:.6f}) that Homo sapiens have significantly LOWER, not higher, AMTL rates than non-human primates after controlling for age, sex, and tooth class. The binomial GLM coefficient for is_human is {is_human_coef:.4f} with p<0.001."
elif is_human_pval < 0.01:
    # Strong evidence
    if is_human_coef > 0:
        response_score = 85  # Yes
        explanation = f"Strong evidence (p={is_human_pval:.6f}) that Homo sapiens have significantly higher AMTL rates than non-human primates after controlling for age, sex, and tooth class. The binomial GLM coefficient for is_human is {is_human_coef:.4f} with p<0.01."
    else:
        response_score = 15  # No
        explanation = f"Strong evidence (p={is_human_pval:.6f}) that Homo sapiens have significantly LOWER, not higher, AMTL rates than non-human primates after controlling for age, sex, and tooth class. The binomial GLM coefficient for is_human is {is_human_coef:.4f} with p<0.01."
elif is_human_pval < 0.05:
    # Moderate evidence
    if is_human_coef > 0:
        response_score = 75  # Moderate Yes
        explanation = f"Moderate evidence (p={is_human_pval:.6f}) that Homo sapiens have significantly higher AMTL rates than non-human primates after controlling for age, sex, and tooth class. The binomial GLM coefficient for is_human is {is_human_coef:.4f} with p<0.05."
    else:
        response_score = 25  # Moderate No
        explanation = f"Moderate evidence (p={is_human_pval:.6f}) that Homo sapiens have significantly LOWER, not higher, AMTL rates than non-human primates after controlling for age, sex, and tooth class. The binomial GLM coefficient for is_human is {is_human_coef:.4f} with p<0.05."
else:
    # No significant evidence
    response_score = 50  # Neutral
    explanation = f"No significant evidence (p={is_human_pval:.6f}) that Homo sapiens have different AMTL rates compared to non-human primates after controlling for age, sex, and tooth class. The binomial GLM coefficient for is_human is {is_human_coef:.4f} but is not statistically significant (p>=0.05)."

print(f"\nResponse Score: {response_score}/100")
print(f"Explanation: {explanation}")

# Write conclusion to file
conclusion = {
    "response": response_score,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n" + "="*80)
print("Analysis complete. Results written to conclusion.txt")
print("="*80)
