#!/usr/bin/env python3
"""
Analysis script for understanding how children's reliance on majority preference 
develops with age across different cultural contexts.
"""

import json
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

# Import agentic_imodels interpretable regressors
from agentic_imodels import (
    SmartAdditiveRegressor,
    HingeEBMRegressor,
    WinsorizedSparseOLSRegressor,
    HingeGAMRegressor,
)

print("=" * 80)
print("ANALYSIS: Children's Reliance on Majority Preference Development")
print("=" * 80)
print()

# ============================================================================
# Step 1: Load and Explore Data
# ============================================================================
print("STEP 1: DATA EXPLORATION")
print("-" * 80)

df = pd.read_csv('boxes.csv')
print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head(10))
print(f"\nDataset info:")
print(df.info())
print(f"\nSummary statistics:")
print(df.describe())

# Check for missing values
print(f"\nMissing values:")
print(df.isnull().sum())

# Research question: How does reliance on majority preference develop with age across cultures?
# The outcome variable y is: 1=unchosen, 2=majority, 3=minority
# We need to understand if children increasingly choose the majority option (y=2) as they age

# Create a binary outcome: chose_majority (1 if y==2, 0 otherwise)
df['chose_majority'] = (df['y'] == 2).astype(int)
df['chose_minority'] = (df['y'] == 3).astype(int)
df['chose_unchosen'] = (df['y'] == 1).astype(int)

print(f"\nOutcome distribution:")
print(f"  Unchosen option (y=1): {(df['y']==1).sum()} ({(df['y']==1).mean()*100:.1f}%)")
print(f"  Majority option (y=2): {(df['y']==2).sum()} ({(df['y']==2).mean()*100:.1f}%)")
print(f"  Minority option (y=3): {(df['y']==3).sum()} ({(df['y']==3).mean()*100:.1f}%)")

print(f"\nAge distribution:")
print(df['age'].value_counts().sort_index())

print(f"\nCulture distribution:")
print(df['culture'].value_counts().sort_index())

# Key correlations
print(f"\nCorrelation of chose_majority with key variables:")
print(f"  Age: {df[['age', 'chose_majority']].corr().iloc[0, 1]:.4f}")
print(f"  Gender: {df[['gender', 'chose_majority']].corr().iloc[0, 1]:.4f}")
print(f"  Majority_first: {df[['majority_first', 'chose_majority']].corr().iloc[0, 1]:.4f}")

print()

# ============================================================================
# Step 2: Bivariate Analysis (Age vs. Majority Choice)
# ============================================================================
print("STEP 2: BIVARIATE ANALYSIS")
print("-" * 80)

# Group by age and compute proportion choosing majority
age_majority = df.groupby('age')['chose_majority'].agg(['mean', 'count'])
print("Proportion choosing majority by age:")
print(age_majority)

# Statistical test: correlation between age and majority choice
corr_age_maj, p_corr = stats.pearsonr(df['age'], df['chose_majority'])
print(f"\nPearson correlation (age, chose_majority): r={corr_age_maj:.4f}, p={p_corr:.4f}")

# Chi-square test for age group and choice
age_groups = pd.cut(df['age'], bins=[3, 6, 9, 15], labels=['Young (4-6)', 'Middle (7-9)', 'Old (10-14)'])
chi2, p_chi2, dof, expected = stats.chi2_contingency(pd.crosstab(age_groups, df['chose_majority']))
print(f"Chi-square test (age group vs chose_majority): χ²={chi2:.4f}, p={p_chi2:.4f}")

print()

# ============================================================================
# Step 3: Classical Statistical Test (Logistic Regression with Controls)
# ============================================================================
print("STEP 3: CLASSICAL STATISTICAL TEST - LOGISTIC REGRESSION")
print("-" * 80)

# Model 1: Bivariate (age only)
X_bivariate = sm.add_constant(df['age'])
logit_bivariate = sm.Logit(df['chose_majority'], X_bivariate).fit(disp=0)
print("Model 1: Bivariate (age only)")
print(logit_bivariate.summary2().tables[1])
print(f"  Age coefficient: {logit_bivariate.params['age']:.4f} (p={logit_bivariate.pvalues['age']:.4f})")
print(f"  Pseudo R²: {logit_bivariate.prsquared:.4f}")

# Model 2: With controls (gender, majority_first, culture)
# Create dummy variables for culture
culture_dummies = pd.get_dummies(df['culture'], prefix='culture', drop_first=True).astype(float)
X_full_df = pd.concat([df[['age', 'gender', 'majority_first']].astype(float), culture_dummies], axis=1)
X_full = sm.add_constant(X_full_df.values)
logit_full = sm.Logit(df['chose_majority'].values, X_full).fit(disp=0)
print("\nModel 2: With controls (gender, majority_first, culture)")
# Create column names for reference
col_names = ['const', 'age', 'gender', 'majority_first'] + [f'culture_{i}' for i in range(2, 9)]
print(logit_full.summary2().tables[1].head(10))
print(f"\n  Age coefficient: {logit_full.params[1]:.4f} (p={logit_full.pvalues[1]:.4f})")
print(f"  Gender coefficient: {logit_full.params[2]:.4f} (p={logit_full.pvalues[2]:.4f})")
print(f"  Majority_first coefficient: {logit_full.params[3]:.4f} (p={logit_full.pvalues[3]:.4f})")
print(f"  Pseudo R²: {logit_full.prsquared:.4f}")

# Check if age effect persists with controls
age_sig_bivariate = logit_bivariate.pvalues['age'] < 0.05
age_sig_controlled = logit_full.pvalues[1] < 0.05
print(f"\nAge effect significance:")
print(f"  Bivariate: {'Significant' if age_sig_bivariate else 'Not significant'} (p={logit_bivariate.pvalues['age']:.4f})")
print(f"  With controls: {'Significant' if age_sig_controlled else 'Not significant'} (p={logit_full.pvalues[1]:.4f})")

print()

# ============================================================================
# Step 4: Interpretable Models for Shape, Direction, and Importance
# ============================================================================
print("STEP 4: INTERPRETABLE MODELS - SHAPE, DIRECTION, IMPORTANCE")
print("-" * 80)

# Prepare data for regression (predicting chose_majority as continuous)
# Include age, gender, majority_first, and culture as features
X_features = df[['age', 'gender', 'majority_first', 'culture']].copy()
y_outcome = df['chose_majority'].values

# Split data for model validation
X_train, X_test, y_train, y_test = train_test_split(X_features, y_outcome, test_size=0.2, random_state=42)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print()

# ============================================================================
# Model 4.1: SmartAdditiveRegressor (honest GAM, reveals nonlinear shapes)
# ============================================================================
print("=" * 80)
print("MODEL 4.1: SmartAdditiveRegressor (Honest GAM)")
print("=" * 80)

smart_additive = SmartAdditiveRegressor()
smart_additive.fit(X_train, y_train)
y_pred_smart = smart_additive.predict(X_test)
r2_smart = r2_score(y_test, y_pred_smart)
rmse_smart = np.sqrt(mean_squared_error(y_test, y_pred_smart))

print(f"Performance: R²={r2_smart:.4f}, RMSE={rmse_smart:.4f}")
print()
print("Model form:")
print(smart_additive)
print()

# ============================================================================
# Model 4.2: HingeEBMRegressor (best rank with good interpretability)
# ============================================================================
print("=" * 80)
print("MODEL 4.2: HingeEBMRegressor (High-rank, Decoupled)")
print("=" * 80)

hinge_ebm = HingeEBMRegressor()
hinge_ebm.fit(X_train, y_train)
y_pred_hinge = hinge_ebm.predict(X_test)
r2_hinge = r2_score(y_test, y_pred_hinge)
rmse_hinge = np.sqrt(mean_squared_error(y_test, y_pred_hinge))

print(f"Performance: R²={r2_hinge:.4f}, RMSE={rmse_hinge:.4f}")
print()
print("Model form:")
print(hinge_ebm)
print()

# ============================================================================
# Model 4.3: WinsorizedSparseOLSRegressor (honest sparse linear)
# ============================================================================
print("=" * 80)
print("MODEL 4.3: WinsorizedSparseOLSRegressor (Honest Sparse Linear)")
print("=" * 80)

winsorized_ols = WinsorizedSparseOLSRegressor()
winsorized_ols.fit(X_train, y_train)
y_pred_wins = winsorized_ols.predict(X_test)
r2_wins = r2_score(y_test, y_pred_wins)
rmse_wins = np.sqrt(mean_squared_error(y_test, y_pred_wins))

print(f"Performance: R²={r2_wins:.4f}, RMSE={rmse_wins:.4f}")
print()
print("Model form:")
print(winsorized_ols)
print()

# ============================================================================
# Step 5: Synthesis and Conclusion
# ============================================================================
print("=" * 80)
print("STEP 5: SYNTHESIS AND CONCLUSION")
print("=" * 80)

print("\nKey Findings:")
print("-" * 80)

print("\n1. STATISTICAL SIGNIFICANCE:")
print(f"   - Bivariate correlation (age, majority choice): r={corr_age_maj:.4f}, p={p_corr:.4f}")
print(f"   - Logistic regression (bivariate): β={logit_bivariate.params['age']:.4f}, p={logit_bivariate.pvalues['age']:.4f}")
print(f"   - Logistic regression (controlled): β={logit_full.params[1]:.4f}, p={logit_full.pvalues[1]:.4f}")
if age_sig_controlled:
    print(f"   → Age effect is SIGNIFICANT and persists after controlling for gender, order, and culture")
else:
    print(f"   → Age effect becomes non-significant after controls")

print("\n2. EFFECT DIRECTION AND MAGNITUDE:")
if logit_full.params[1] > 0:
    print(f"   - Positive age effect: older children are MORE likely to choose majority")
    print(f"   - Each additional year of age increases log-odds by {logit_full.params[1]:.4f}")
else:
    print(f"   - Negative age effect: older children are LESS likely to choose majority")

print("\n3. INTERPRETABLE MODEL INSIGHTS:")
print(f"   - All three models fitted successfully")
print(f"   - Models reveal the shape and importance of age effect")
print(f"   - Look at the printed model forms above to see if age is:")
print(f"     * Included with non-zero coefficient (importance)")
print(f"     * Linear or nonlinear in its effect")
print(f"     * Consistent across models (robustness)")

print("\n4. CONTROL VARIABLES:")
print(f"   - Gender effect: β={logit_full.params[2]:.4f}, p={logit_full.pvalues[2]:.4f}")
print(f"   - Majority_first effect: β={logit_full.params[3]:.4f}, p={logit_full.pvalues[3]:.4f}")
print(f"   - Culture effects: vary by site (see full model summary)")

print("\n5. OVERALL INTERPRETATION:")

# Determine the strength of evidence
evidence_score = 0
explanation_parts = []

# Criterion 1: Statistical significance (0-30 points)
if age_sig_controlled and logit_full.pvalues[1] < 0.01:
    evidence_score += 30
    explanation_parts.append(f"Age has a highly significant effect (p={logit_full.pvalues[1]:.4f}) that persists after controlling for gender, presentation order, and culture")
elif age_sig_controlled and logit_full.pvalues[1] < 0.05:
    evidence_score += 20
    explanation_parts.append(f"Age has a significant effect (p={logit_full.pvalues[1]:.4f}) after controls, though moderate")
elif age_sig_bivariate:
    evidence_score += 10
    explanation_parts.append(f"Age shows bivariate significance (p={logit_bivariate.pvalues['age']:.4f}) but weakens with controls (p={logit_full.pvalues[1]:.4f})")
else:
    evidence_score += 0
    explanation_parts.append(f"Age effect is not statistically significant (p={logit_full.pvalues[1]:.4f})")

# Criterion 2: Effect direction consistency (0-25 points)
if logit_full.params[1] > 0 and corr_age_maj > 0:
    evidence_score += 25
    explanation_parts.append(f"The effect direction is consistent: older children increasingly rely on majority preference (β={logit_full.params[1]:.4f}, r={corr_age_maj:.4f})")
elif logit_full.params[1] > 0 or corr_age_maj > 0:
    evidence_score += 15
    explanation_parts.append("Effect direction is somewhat consistent but not uniform across all analyses")
else:
    evidence_score += 5
    explanation_parts.append("Effect direction is unclear or negative (older children less likely to choose majority)")

# Criterion 3: Effect size (0-20 points)
if abs(corr_age_maj) > 0.15:
    evidence_score += 20
    explanation_parts.append(f"The correlation magnitude is moderate-to-large (r={abs(corr_age_maj):.4f})")
elif abs(corr_age_maj) > 0.10:
    evidence_score += 10
    explanation_parts.append(f"The correlation magnitude is small-to-moderate (r={abs(corr_age_maj):.4f})")
else:
    evidence_score += 5
    explanation_parts.append(f"The correlation magnitude is small (r={abs(corr_age_maj):.4f})")

# Criterion 4: Model consistency (0-25 points)
# Check R² values - if models perform reasonably, they captured the relationships
avg_r2 = (r2_smart + r2_hinge + r2_wins) / 3
if avg_r2 > 0.10:
    evidence_score += 25
    explanation_parts.append(f"The interpretable models show good predictive performance (avg R²={avg_r2:.3f}), confirming robust age-related patterns")
elif avg_r2 > 0.05:
    evidence_score += 15
    explanation_parts.append(f"The interpretable models show moderate predictive performance (avg R²={avg_r2:.3f})")
else:
    evidence_score += 5
    explanation_parts.append(f"The interpretable models show weak predictive performance (avg R²={avg_r2:.3f}), suggesting age effect may be weak")

# Final calibration
print(f"\n   Evidence Score: {evidence_score}/100")
print(f"\n   Based on the analysis:")
for i, part in enumerate(explanation_parts, 1):
    print(f"   {i}. {part}")

# Create final response
response_score = evidence_score
explanation = " ".join(explanation_parts)

# Add developmental interpretation
explanation += f" The research question asks about how reliance on majority preference develops with age across cultures. "
if response_score >= 70:
    explanation += "The data provides strong evidence that children's reliance on majority preference increases significantly with age, even after accounting for cultural context and other factors."
elif response_score >= 40:
    explanation += "The data provides moderate evidence that children's reliance on majority preference changes with age, though the effect is not as strong or consistent across all analytical approaches."
else:
    explanation += "The data provides weak evidence for a developmental trend in majority preference reliance with age. The effect, if present, is small and may not be robust across cultural contexts."

print(f"\n   Final Conclusion: {explanation[:200]}...")
print()

# ============================================================================
# Step 6: Write conclusion.txt
# ============================================================================
print("=" * 80)
print("WRITING CONCLUSION FILE")
print("=" * 80)

conclusion = {
    "response": response_score,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print(f"\nConclusion written to conclusion.txt")
print(f"Response score: {response_score}")
print(f"Explanation length: {len(explanation)} characters")
print()
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
