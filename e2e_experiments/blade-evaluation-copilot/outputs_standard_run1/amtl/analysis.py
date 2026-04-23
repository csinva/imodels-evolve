import pandas as pd
import numpy as np
import json
from scipy import stats
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('amtl.csv')

print("="*80)
print("EXPLORATORY DATA ANALYSIS")
print("="*80)

# Basic info
print(f"\nDataset shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nGenus distribution:")
print(df['genus'].value_counts())

# Create a binary indicator for whether any AMTL occurred
df['has_amtl'] = (df['num_amtl'] > 0).astype(int)

# Create AMTL rate (proportion)
df['amtl_rate'] = df['num_amtl'] / df['sockets']

# Create binary indicator for Homo sapiens
df['is_homo'] = (df['genus'] == 'Homo sapiens').astype(int)

print(f"\nAMTL rate by genus:")
amtl_by_genus = df.groupby('genus').agg({
    'num_amtl': 'sum',
    'sockets': 'sum',
    'has_amtl': 'mean',
    'age': 'mean'
})
amtl_by_genus['total_rate'] = amtl_by_genus['num_amtl'] / amtl_by_genus['sockets']
print(amtl_by_genus)

print(f"\nDescriptive statistics by genus:")
print(df.groupby('genus')[['amtl_rate', 'age', 'prob_male']].describe())

print("\n" + "="*80)
print("STATISTICAL ANALYSIS")
print("="*80)

# Approach 1: Logistic regression on binomial data (proper approach for count data)
print("\n1. LOGISTIC REGRESSION ON BINOMIAL OUTCOME")
print("-" * 60)

# Create dummy variables for categorical predictors
df_encoded = pd.get_dummies(df, columns=['tooth_class'], drop_first=True)

# Prepare features for logistic regression
feature_cols = ['is_homo', 'age', 'prob_male', 'tooth_class_Posterior', 'tooth_class_Premolar']
X = df_encoded[feature_cols].copy().astype(float)
X = sm.add_constant(X)

# For logistic regression, we'll model whether AMTL occurred (binary outcome)
y = df_encoded['has_amtl'].astype(float)

# Fit logistic regression
logit_model = sm.Logit(y, X).fit(disp=0)
print(logit_model.summary())

print("\nOdds Ratios (exponentiated coefficients):")
odds_ratios = np.exp(logit_model.params)
conf_int = np.exp(logit_model.conf_int())
for var in ['is_homo', 'age', 'prob_male']:
    if var in logit_model.params.index:
        print(f"{var}: OR = {odds_ratios[var]:.3f}, 95% CI = [{conf_int.loc[var, 0]:.3f}, {conf_int.loc[var, 1]:.3f}], p = {logit_model.pvalues[var]:.4f}")

# Approach 2: Linear regression on AMTL rate
print("\n\n2. LINEAR REGRESSION ON AMTL RATE")
print("-" * 60)

# OLS regression on AMTL rate
X_rate = df_encoded[feature_cols].copy().astype(float)
X_rate = sm.add_constant(X_rate)
y_rate = df_encoded['amtl_rate'].astype(float)

ols_model = sm.OLS(y_rate, X_rate).fit()
print(ols_model.summary())

print("\nKey coefficients for AMTL rate:")
for var in ['is_homo', 'age', 'prob_male']:
    if var in ols_model.params.index:
        print(f"{var}: coef = {ols_model.params[var]:.5f}, 95% CI = [{ols_model.conf_int().loc[var, 0]:.5f}, {ols_model.conf_int().loc[var, 1]:.5f}], p = {ols_model.pvalues[var]:.4f}")

# Approach 3: Compare Homo sapiens vs all non-human primates directly
print("\n\n3. DIRECT COMPARISON: HOMO SAPIENS VS NON-HUMAN PRIMATES")
print("-" * 60)

homo_data = df[df['genus'] == 'Homo sapiens']
non_homo_data = df[df['genus'] != 'Homo sapiens']

print(f"Homo sapiens: n={len(homo_data)}, mean AMTL rate={homo_data['amtl_rate'].mean():.4f}")
print(f"Non-human primates: n={len(non_homo_data)}, mean AMTL rate={non_homo_data['amtl_rate'].mean():.4f}")

# T-test
t_stat, p_value = stats.ttest_ind(homo_data['amtl_rate'], non_homo_data['amtl_rate'])
print(f"\nT-test: t={t_stat:.3f}, p={p_value:.4f}")

# Mann-Whitney U test (non-parametric alternative)
u_stat, p_value_mw = stats.mannwhitneyu(homo_data['amtl_rate'], non_homo_data['amtl_rate'], alternative='two-sided')
print(f"Mann-Whitney U test: U={u_stat:.1f}, p={p_value_mw:.4f}")

# Approach 4: ANCOVA-style analysis controlling for covariates
print("\n\n4. ANCOVA: COMPARING GROUPS WHILE CONTROLLING FOR COVARIATES")
print("-" * 60)

# Create a formula for the model
formula = "amtl_rate ~ is_homo + age + prob_male + C(tooth_class)"
ancova_model = smf.ols(formula, data=df).fit()
print(ancova_model.summary())

# Approach 5: Aggregated analysis at specimen level
print("\n\n5. SPECIMEN-LEVEL ANALYSIS (AGGREGATED)")
print("-" * 60)

# Aggregate to specimen level
specimen_df = df.groupby('specimen').agg({
    'num_amtl': 'sum',
    'sockets': 'sum',
    'age': 'first',
    'prob_male': 'first',
    'genus': 'first'
}).reset_index()

specimen_df['amtl_rate'] = specimen_df['num_amtl'] / specimen_df['sockets']
specimen_df['is_homo'] = (specimen_df['genus'] == 'Homo sapiens').astype(int)

print(f"Specimen-level comparison:")
homo_spec = specimen_df[specimen_df['is_homo'] == 1]
non_homo_spec = specimen_df[specimen_df['is_homo'] == 0]

print(f"Homo sapiens specimens: n={len(homo_spec)}, mean rate={homo_spec['amtl_rate'].mean():.4f}")
print(f"Non-human specimens: n={len(non_homo_spec)}, mean rate={non_homo_spec['amtl_rate'].mean():.4f}")

# Regression at specimen level
X_spec = sm.add_constant(specimen_df[['is_homo', 'age', 'prob_male']].astype(float))
y_spec = specimen_df['amtl_rate'].astype(float)
spec_model = sm.OLS(y_spec, X_spec).fit()
print("\nSpecimen-level regression:")
print(spec_model.summary())

print("\n" + "="*80)
print("INTERPRETABLE MODELS")
print("="*80)

# Try imodels for interpretable analysis
try:
    from imodels import RuleFitRegressor, FIGSRegressor
    
    print("\n6. RULEFIT REGRESSION")
    print("-" * 60)
    
    # Prepare data for RuleFit
    X_rf = df_encoded[['is_homo', 'age', 'prob_male', 'tooth_class_Posterior', 'tooth_class_Premolar']].values
    y_rf = df_encoded['amtl_rate'].values
    
    # Fit RuleFit
    rf_model = RuleFitRegressor(max_rules=10, random_state=42)
    rf_model.fit(X_rf, y_rf)
    
    print(f"RuleFit R^2: {rf_model.score(X_rf, y_rf):.3f}")
    print("\nTop rules and feature importances:")
    
    # Get feature importances
    feature_names = ['is_homo', 'age', 'prob_male', 'tooth_class_Posterior', 'tooth_class_Premolar']
    
    # Display rules if available
    if hasattr(rf_model, 'rules_'):
        for i, rule in enumerate(rf_model.rules_[:5]):  # Top 5 rules
            print(f"Rule {i+1}: {rule}")
    
except Exception as e:
    print(f"Could not fit imodels: {e}")

print("\n" + "="*80)
print("FINAL INTERPRETATION")
print("="*80)

# Extract key statistics
homo_coef = logit_model.params.get('is_homo', 0)
homo_pval = logit_model.pvalues.get('is_homo', 1)
homo_or = np.exp(homo_coef)

ols_coef = ols_model.params.get('is_homo', 0)
ols_pval = ols_model.pvalues.get('is_homo', 1)

print(f"\nKey findings:")
print(f"1. Logistic regression (has_amtl): Homo sapiens OR = {homo_or:.3f}, p = {homo_pval:.4f}")
print(f"2. Linear regression (AMTL rate): Homo sapiens coef = {ols_coef:.5f}, p = {ols_pval:.4f}")
print(f"3. Direct comparison p-value: {p_value:.4f}")
print(f"4. ANCOVA is_homo coefficient: {ancova_model.params.get('is_homo', 0):.5f}, p = {ancova_model.pvalues.get('is_homo', 1):.4f}")
print(f"5. Specimen-level is_homo coefficient: {spec_model.params.get('is_homo', 0):.5f}, p = {spec_model.pvalues.get('is_homo', 1):.4f}")

# Decision logic
print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

# Determine response based on multiple lines of evidence
significant_count = 0
total_tests = 0

# Check each statistical test
tests = [
    ("Logistic regression", homo_pval),
    ("Linear regression (OLS)", ols_pval),
    ("ANCOVA", ancova_model.pvalues.get('is_homo', 1)),
    ("Specimen-level regression", spec_model.pvalues.get('is_homo', 1)),
    ("Direct t-test", p_value)
]

print("\nStatistical significance summary (α = 0.05):")
for test_name, p_val in tests:
    total_tests += 1
    is_sig = p_val < 0.05
    if is_sig:
        significant_count += 1
    print(f"  {test_name}: p = {p_val:.4f} {'✓ Significant' if is_sig else '✗ Not significant'}")

# Check direction of effect
effect_positive = homo_or > 1 and ols_coef > 0 and ancova_model.params.get('is_homo', 0) > 0

print(f"\nSignificant tests: {significant_count}/{total_tests}")
print(f"Effect direction: {'Homo sapiens has HIGHER AMTL' if effect_positive else 'Homo sapiens has LOWER or no difference in AMTL'}")

# Calculate Likert score
# Strong evidence (4-5 significant tests): 80-100
# Moderate evidence (2-3 significant): 50-80
# Weak evidence (1 significant): 20-50
# No evidence (0 significant): 0-20

if significant_count >= 4 and effect_positive:
    response = 90
    explanation = f"Strong evidence: {significant_count} out of {total_tests} statistical tests show significant differences (p<0.05), with Homo sapiens having higher AMTL rates after controlling for age, sex, and tooth class. Logistic regression shows OR={homo_or:.2f} (p={homo_pval:.4f}), indicating humans have substantially higher odds of tooth loss."
elif significant_count >= 3 and effect_positive:
    response = 75
    explanation = f"Moderate-to-strong evidence: {significant_count} out of {total_tests} tests are significant (p<0.05), suggesting Homo sapiens has higher AMTL rates after controlling for covariates. The effect is consistent across multiple modeling approaches."
elif significant_count >= 2 and effect_positive:
    response = 60
    explanation = f"Moderate evidence: {significant_count} out of {total_tests} tests show significance (p<0.05), providing some support that Homo sapiens has higher AMTL rates when controlling for age, sex, and tooth class, though not all analyses agree."
elif significant_count >= 1 and effect_positive:
    response = 40
    explanation = f"Weak evidence: Only {significant_count} out of {total_tests} tests reached significance (p<0.05), suggesting limited statistical support for the claim that Homo sapiens has higher AMTL rates after controlling for covariates."
elif significant_count == 0 and effect_positive:
    response = 20
    explanation = f"No significant evidence: None of the {total_tests} statistical tests reached significance (p<0.05). While the direction of effects suggests higher AMTL in humans, the evidence is not statistically reliable after controlling for age, sex, and tooth class."
elif significant_count >= 3 and not effect_positive:
    response = 10
    explanation = f"Strong evidence against: {significant_count} out of {total_tests} tests are significant, but the effect direction shows Homo sapiens has LOWER or no different AMTL rates compared to non-human primates after controlling for covariates."
else:
    response = 15
    explanation = f"No clear evidence: Statistical tests do not provide convincing support for higher AMTL rates in Homo sapiens. After controlling for age, sex, and tooth class, the relationship is not reliably established."

print(f"\nFinal Likert score: {response}/100")
print(f"Explanation: {explanation}")

# Write conclusion to file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n" + "="*80)
print("ANALYSIS COMPLETE - conclusion.txt written")
print("="*80)
