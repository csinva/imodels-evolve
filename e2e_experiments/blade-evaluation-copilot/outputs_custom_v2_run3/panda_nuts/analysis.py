import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from agentic_imodels import SmartAdditiveRegressor, HingeEBMRegressor, WinsorizedSparseOLSRegressor

# Load dataset
df = pd.read_csv('panda_nuts.csv')

print("=" * 80)
print("DATASET OVERVIEW")
print("=" * 80)
print(f"Shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nSummary statistics:\n{df.describe()}")
print(f"\nCategorical value counts:")
print(f"Sex:\n{df['sex'].value_counts()}")
print(f"Help:\n{df['help'].value_counts()}")
print(f"Hammer:\n{df['hammer'].value_counts()}")

# Research question: How do age, sex, and receiving help influence nut-cracking efficiency?
# Let's define efficiency as nuts_opened per second
df['efficiency'] = df['nuts_opened'] / df['seconds']

print(f"\n\nEfficiency (nuts/second) summary:\n{df['efficiency'].describe()}")

# Encode categorical variables
df['sex_male'] = (df['sex'] == 'm').astype(int)
df['help_yes'] = (df['help'] == 'y').astype(int)

# Create dummy variables for hammer type
hammer_dummies = pd.get_dummies(df['hammer'], prefix='hammer', drop_first=True)
df = pd.concat([df, hammer_dummies], axis=1)

print("\n" + "=" * 80)
print("BIVARIATE ANALYSIS")
print("=" * 80)

# Correlation analysis
print("\nCorrelations with efficiency:")
corr_cols = ['age', 'sex_male', 'help_yes', 'efficiency']
corr_matrix = df[corr_cols].corr()
print(corr_matrix['efficiency'].sort_values(ascending=False))

# Statistical tests
print("\n\nBivariate statistical tests:")

# Age vs efficiency (continuous vs continuous)
age_corr, age_pval = stats.pearsonr(df['age'], df['efficiency'])
print(f"Age vs Efficiency: r={age_corr:.4f}, p={age_pval:.4f}")

# Sex vs efficiency (categorical vs continuous)
male_eff = df[df['sex_male'] == 1]['efficiency']
female_eff = df[df['sex_male'] == 0]['efficiency']
sex_ttest = stats.ttest_ind(male_eff, female_eff)
print(f"Sex vs Efficiency: t={sex_ttest.statistic:.4f}, p={sex_ttest.pvalue:.4f}")
print(f"  Male mean efficiency: {male_eff.mean():.4f}")
print(f"  Female mean efficiency: {female_eff.mean():.4f}")

# Help vs efficiency (categorical vs continuous)
help_yes_eff = df[df['help_yes'] == 1]['efficiency']
help_no_eff = df[df['help_yes'] == 0]['efficiency']
help_ttest = stats.ttest_ind(help_yes_eff, help_no_eff)
print(f"Help vs Efficiency: t={help_ttest.statistic:.4f}, p={help_ttest.pvalue:.4f}")
print(f"  With help mean efficiency: {help_yes_eff.mean():.4f}")
print(f"  Without help mean efficiency: {help_no_eff.mean():.4f}")

print("\n" + "=" * 80)
print("CLASSICAL REGRESSION (OLS) WITH CONTROLS")
print("=" * 80)

# Prepare features for regression
# Key predictors: age, sex_male, help_yes
# Controls: hammer type, individual ID effects (random effects not easily done in OLS, so we'll use fixed effects or omit)
# Since we have repeated measures per individual, let's include hammer as control

X_cols = ['age', 'sex_male', 'help_yes'] + [col for col in df.columns if col.startswith('hammer_')]
X = df[X_cols].copy().astype(float)  # Ensure all numeric
y = df['efficiency'].copy().astype(float)

# Add constant for OLS
X_with_const = sm.add_constant(X)

# Fit OLS model
ols_model = sm.OLS(y, X_with_const).fit()
print(ols_model.summary())

print("\n\nKey findings from OLS:")
print(f"Age coefficient: {ols_model.params['age']:.6f}, p={ols_model.pvalues['age']:.4f}")
print(f"Sex (male) coefficient: {ols_model.params['sex_male']:.6f}, p={ols_model.pvalues['sex_male']:.4f}")
print(f"Help coefficient: {ols_model.params['help_yes']:.6f}, p={ols_model.pvalues['help_yes']:.4f}")

print("\n" + "=" * 80)
print("INTERPRETABLE MODELS - SHAPE, DIRECTION, IMPORTANCE")
print("=" * 80)

# Fit multiple interpretable models
X_for_imodels = df[X_cols].copy()
y_for_imodels = y.copy()

print("\n--- Model 1: SmartAdditiveRegressor (honest GAM) ---")
smart_model = SmartAdditiveRegressor()
smart_model.fit(X_for_imodels, y_for_imodels)
print(smart_model)
smart_r2 = smart_model.score(X_for_imodels, y_for_imodels)
print(f"\nR^2 on full data: {smart_r2:.4f}")

print("\n--- Model 2: HingeEBMRegressor (high-rank, decoupled) ---")
hinge_model = HingeEBMRegressor()
hinge_model.fit(X_for_imodels, y_for_imodels)
print(hinge_model)
hinge_r2 = hinge_model.score(X_for_imodels, y_for_imodels)
print(f"\nR^2 on full data: {hinge_r2:.4f}")

print("\n--- Model 3: WinsorizedSparseOLSRegressor (honest sparse linear) ---")
sparse_model = WinsorizedSparseOLSRegressor()
sparse_model.fit(X_for_imodels, y_for_imodels)
print(sparse_model)
sparse_r2 = sparse_model.score(X_for_imodels, y_for_imodels)
print(f"\nR^2 on full data: {sparse_r2:.4f}")

print("\n" + "=" * 80)
print("SYNTHESIS AND CONCLUSION")
print("=" * 80)

# Analyze the findings
age_significant = ols_model.pvalues['age'] < 0.05
age_coef = ols_model.params['age']
sex_significant = ols_model.pvalues['sex_male'] < 0.05
sex_coef = ols_model.params['sex_male']
help_significant = ols_model.pvalues['help_yes'] < 0.05
help_coef = ols_model.params['help_yes']

print("\nEvidence summary:")
print(f"1. AGE: coefficient={age_coef:.6f}, p={ols_model.pvalues['age']:.4f}, bivariate r={age_corr:.4f}")
print(f"   - Statistical significance: {'YES' if age_significant else 'NO'}")

print(f"\n2. SEX (male vs female): coefficient={sex_coef:.6f}, p={ols_model.pvalues['sex_male']:.4f}")
print(f"   - Statistical significance: {'YES' if sex_significant else 'NO'}")
print(f"   - Male-female difference: {male_eff.mean() - female_eff.mean():.6f} nuts/sec")

print(f"\n3. HELP: coefficient={help_coef:.6f}, p={ols_model.pvalues['help_yes']:.4f}")
print(f"   - Statistical significance: {'YES' if help_significant else 'NO'}")
print(f"   - Help vs no-help difference: {help_yes_eff.mean() - help_no_eff.mean():.6f} nuts/sec")

# Determine response score on 0-100 scale
# Research question: "How do age, sex, and receiving help influence nut-cracking efficiency?"
# This is asking whether these variables influence efficiency, not just one variable

# Count how many of the three variables show significant effects
significant_vars = sum([age_significant, sex_significant, help_significant])
effect_strengths = []

if age_significant:
    effect_strengths.append(abs(age_coef))
if sex_significant:
    effect_strengths.append(abs(sex_coef))
if help_significant:
    effect_strengths.append(abs(help_coef))

# Reasoning for score:
# - All 3 variables can influence efficiency (that's the question)
# - Score should reflect the strength and consistency of evidence
# - Strong evidence for multiple variables = high score
# - Mixed or weak evidence = moderate score
# - No evidence = low score

explanation = "Analysis of 84 observations of western chimpanzees cracking panda nuts:\n\n"

explanation += f"STATISTICAL TESTS (with controls for hammer type):\n"
explanation += f"- Age: coefficient={age_coef:.4f}, p={ols_model.pvalues['age']:.4f} ({'significant' if age_significant else 'not significant'})\n"
explanation += f"- Sex: coefficient={sex_coef:.4f}, p={ols_model.pvalues['sex_male']:.4f} ({'significant' if sex_significant else 'not significant'})\n"
explanation += f"- Help: coefficient={help_coef:.4f}, p={ols_model.pvalues['help_yes']:.4f} ({'significant' if help_significant else 'not significant'})\n\n"

explanation += f"INTERPRETABLE MODELS: Fitted {3} interpretable regressors (SmartAdditive, HingeEBM, WinsorizedSparseOLS) "
explanation += f"to assess feature importance and robustness. "

# Determine overall conclusion
if significant_vars == 3:
    response_score = 85  # Strong evidence for all three
    explanation += f"All three variables show statistically significant effects on efficiency. "
elif significant_vars == 2:
    response_score = 65  # Moderate-strong evidence
    explanation += f"Two of three variables show significant effects. "
elif significant_vars == 1:
    response_score = 45  # Moderate evidence
    explanation += f"One of three variables shows significant effect. "
else:
    response_score = 20  # Weak evidence
    explanation += f"None of the three variables show statistically significant effects at p<0.05. "

# Adjust based on effect sizes and consistency
if significant_vars > 0:
    avg_effect = np.mean(effect_strengths) if effect_strengths else 0
    # Efficiency scale is roughly 0-2 nuts/sec, so effects > 0.1 are substantial
    if avg_effect > 0.1:
        response_score = min(response_score + 10, 95)
        explanation += f"Effects are substantial in magnitude (avg effect size: {avg_effect:.4f}). "

explanation += f"\n\nCONCLUSION: The evidence "
if response_score >= 75:
    explanation += "strongly supports"
elif response_score >= 60:
    explanation += "moderately to strongly supports"
elif response_score >= 40:
    explanation += "moderately supports"
elif response_score >= 25:
    explanation += "weakly supports"
else:
    explanation += "does not clearly support"

explanation += " that age, sex, and receiving help influence nut-cracking efficiency in western chimpanzees."

print(f"\n\nFINAL ASSESSMENT:")
print(f"Response score: {response_score}/100")
print(f"\n{explanation}")

# Write conclusion.txt
conclusion = {
    "response": response_score,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n" + "=" * 80)
print("conclusion.txt written successfully!")
print("=" * 80)
