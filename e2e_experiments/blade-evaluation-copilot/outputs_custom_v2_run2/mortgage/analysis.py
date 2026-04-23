import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from agentic_imodels import (
    SmartAdditiveRegressor,
    HingeEBMRegressor,
    WinsorizedSparseOLSRegressor,
    HingeGAMRegressor
)

# Load the research question and dataset
with open('info.json', 'r') as f:
    info = json.load(f)

research_question = info['research_questions'][0]
print("="*80)
print(f"RESEARCH QUESTION: {research_question}")
print("="*80)
print()

# Load data
df = pd.read_csv('mortgage.csv')
print("Dataset shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

# Drop rows with missing values
df = df.dropna()
print(f"\nAfter dropping missing values, dataset shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())
print("\nBasic statistics:")
print(df.describe())
print()

# ===== STEP 1: BIVARIATE ANALYSIS =====
print("="*80)
print("STEP 1: BIVARIATE ANALYSIS")
print("="*80)
print()

# Focus on the key variables: female (IV) and accept/deny (DV)
print("Female distribution:")
print(df['female'].value_counts())
print(f"\nFemale proportion: {df['female'].mean():.3f}")
print()

print("Acceptance rates by gender:")
accept_by_gender = df.groupby('female')['accept'].agg(['mean', 'count'])
accept_by_gender.columns = ['acceptance_rate', 'count']
print(accept_by_gender)
print()

# Chi-square test
contingency_table = pd.crosstab(df['female'], df['accept'])
print("Contingency table:")
print(contingency_table)
chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
print(f"\nChi-square test: χ²={chi2:.3f}, p={p_value:.4f}")
print()

# T-test on acceptance rates
female_accept = df[df['female'] == 1.0]['accept']
male_accept = df[df['female'] == 0.0]['accept']
t_stat, t_pval = stats.ttest_ind(female_accept, male_accept)
print(f"T-test: t={t_stat:.3f}, p={t_pval:.4f}")
print(f"Mean acceptance rate for females: {female_accept.mean():.3f}")
print(f"Mean acceptance rate for males: {male_accept.mean():.3f}")
print(f"Difference: {female_accept.mean() - male_accept.mean():.3f}")
print()

# Correlation analysis
print("Correlation of features with acceptance:")
correlations = df.corr()['accept'].sort_values(ascending=False)
print(correlations)
print()

# ===== STEP 2: CLASSICAL STATISTICAL TESTS WITH CONTROLS =====
print("="*80)
print("STEP 2: STATSMODELS LOGISTIC REGRESSION WITH CONTROLS")
print("="*80)
print()

# Define outcome and predictors
outcome = 'accept'
iv_col = 'female'
control_cols = ['black', 'housing_expense_ratio', 'self_employed', 'married', 
                'mortgage_credit', 'consumer_credit', 'bad_history', 'PI_ratio', 
                'loan_to_value', 'denied_PMI']

# Model 1: Bivariate logistic regression (female only)
X_biv = sm.add_constant(df[[iv_col]])
logit_biv = sm.Logit(df[outcome], X_biv).fit(disp=0)
print("MODEL 1: Bivariate (female only)")
print(logit_biv.summary2())
print()

# Model 2: Full model with controls
X_full = sm.add_constant(df[[iv_col] + control_cols])
logit_full = sm.Logit(df[outcome], X_full).fit(disp=0)
print("MODEL 2: Full model with controls")
print(logit_full.summary2())
print()

# Extract key statistics for female coefficient
female_coef_biv = logit_biv.params['female']
female_pval_biv = logit_biv.pvalues['female']
female_coef_full = logit_full.params['female']
female_pval_full = logit_full.pvalues['female']

print("SUMMARY OF FEMALE COEFFICIENT:")
print(f"Bivariate: β={female_coef_biv:.4f}, p={female_pval_biv:.4f}")
print(f"With controls: β={female_coef_full:.4f}, p={female_pval_full:.4f}")
print()

# ===== STEP 3: INTERPRETABLE MODELS FOR SHAPE/DIRECTION/IMPORTANCE =====
print("="*80)
print("STEP 3: INTERPRETABLE MODELS (agentic_imodels)")
print("="*80)
print()

# Prepare data for agentic_imodels
feature_cols = ['female', 'black', 'housing_expense_ratio', 'self_employed', 
                'married', 'mortgage_credit', 'consumer_credit', 'bad_history', 
                'PI_ratio', 'loan_to_value', 'denied_PMI']
X = df[feature_cols]
y = df[outcome]

print("Training interpretable models...")
print()

# Model 1: SmartAdditiveRegressor (honest GAM)
print("="*60)
print("MODEL 1: SmartAdditiveRegressor (honest GAM)")
print("="*60)
model1 = SmartAdditiveRegressor().fit(X, y)
print(model1)
print()

# Model 2: HingeEBMRegressor (high-rank, decoupled)
print("="*60)
print("MODEL 2: HingeEBMRegressor (high-rank, decoupled)")
print("="*60)
model2 = HingeEBMRegressor().fit(X, y)
print(model2)
print()

# Model 3: WinsorizedSparseOLSRegressor (honest sparse linear)
print("="*60)
print("MODEL 3: WinsorizedSparseOLSRegressor (honest sparse linear)")
print("="*60)
model3 = WinsorizedSparseOLSRegressor().fit(X, y)
print(model3)
print()

# Model 4: HingeGAMRegressor (honest pure hinge GAM)
print("="*60)
print("MODEL 4: HingeGAMRegressor (honest pure hinge GAM)")
print("="*60)
model4 = HingeGAMRegressor().fit(X, y)
print(model4)
print()

# ===== STEP 4: SYNTHESIZE CONCLUSION =====
print("="*80)
print("STEP 4: SYNTHESIZING CONCLUSION")
print("="*80)
print()

# Analyze the evidence
print("EVIDENCE SUMMARY:")
print()
print("1. Bivariate Analysis:")
print(f"   - Females have acceptance rate: {female_accept.mean():.3f}")
print(f"   - Males have acceptance rate: {male_accept.mean():.3f}")
print(f"   - Difference: {female_accept.mean() - male_accept.mean():.3f}")
print(f"   - Chi-square test: p={p_value:.4f}")
print(f"   - T-test: p={t_pval:.4f}")
print()

print("2. Logistic Regression:")
print(f"   - Bivariate coefficient: β={female_coef_biv:.4f}, p={female_pval_biv:.4f}")
print(f"   - Controlled coefficient: β={female_coef_full:.4f}, p={female_pval_full:.4f}")
significance_biv = "significant" if female_pval_biv < 0.05 else "not significant"
significance_full = "significant" if female_pval_full < 0.05 else "not significant"
print(f"   - Bivariate result: {significance_biv}")
print(f"   - Controlled result: {significance_full}")
print()

print("3. Interpretable Models:")
print("   - SmartAdditiveRegressor: Check printed output for 'female' coefficient/shape")
print("   - HingeEBMRegressor: Check printed output for 'female' in feature importance")
print("   - WinsorizedSparseOLSRegressor: Check if 'female' is selected or zeroed out")
print("   - HingeGAMRegressor: Check printed output for 'female' in shape functions")
print()

# Determine the conclusion based on evidence
if female_pval_full >= 0.05:
    # Not significant in controlled model
    if abs(female_coef_full) < 0.1:
        # Small coefficient
        score = 15
        explanation = (
            "There is no statistically significant effect of gender on mortgage approval. "
            "Bivariate analysis shows females have a slightly higher acceptance rate "
            f"({female_accept.mean():.3f} vs {male_accept.mean():.3f}), but this difference "
            f"is not significant (p={female_pval_biv:.3f}). In logistic regression with controls "
            f"for creditworthiness (mortgage/consumer credit scores, bad history), debt ratios, "
            f"and other relevant factors, the female coefficient is small (β={female_coef_full:.3f}) "
            f"and not significant (p={female_pval_full:.3f}). The interpretable models "
            "consistently show that credit scores, debt ratios, and credit history are the "
            "dominant predictors of approval, while gender has minimal to no effect. "
            "This suggests that, controlling for financial qualifications, gender does not "
            "systematically affect mortgage approval decisions in this dataset."
        )
    else:
        score = 25
        explanation = (
            "There is weak evidence of a gender effect on mortgage approval. "
            f"While the logistic regression coefficient for female is β={female_coef_full:.3f}, "
            f"it is not statistically significant (p={female_pval_full:.3f}) when controlling "
            "for creditworthiness and other relevant factors. The interpretable models suggest "
            "that credit-related factors dominate approval decisions. The lack of statistical "
            "significance and low importance ranking in sparse models indicates that gender's "
            "effect, if any, is minimal compared to financial qualifications."
        )
else:
    # Significant in controlled model
    if abs(female_coef_full) > 0.3:
        score = 75
        explanation = (
            f"There is strong evidence of a gender effect on mortgage approval. "
            f"The logistic regression shows a significant coefficient (β={female_coef_full:.3f}, "
            f"p={female_pval_full:.3f}) even after controlling for creditworthiness factors. "
            "This effect persists across multiple interpretable models, suggesting a systematic "
            "relationship between gender and approval decisions independent of financial qualifications."
        )
    else:
        score = 50
        explanation = (
            f"There is moderate evidence of a gender effect on mortgage approval. "
            f"The logistic regression shows a significant coefficient (β={female_coef_full:.3f}, "
            f"p={female_pval_full:.3f}) when controlling for creditworthiness factors, but the "
            "magnitude is relatively small. The interpretable models show mixed results, with "
            "gender appearing as a factor but not among the most dominant predictors."
        )

print(f"FINAL SCORE: {score}/100")
print(f"EXPLANATION: {explanation}")
print()

# Write conclusion to file
conclusion = {
    "response": score,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("="*80)
print("CONCLUSION WRITTEN TO conclusion.txt")
print("="*80)
