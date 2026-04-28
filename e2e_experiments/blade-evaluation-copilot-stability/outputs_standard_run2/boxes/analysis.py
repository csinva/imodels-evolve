import json
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.formula.api import ols
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('boxes.csv')

print("="*80)
print("DATA EXPLORATION")
print("="*80)
print(f"\nDataset shape: {df.shape}")
print("\nFirst few rows:")
print(df.head(10))
print("\nSummary statistics:")
print(df.describe())
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())

# Research Question: How do children's reliance on majority preference develop over growth in age across different cultural contexts?

# Create binary variable: did child choose majority option (y=2)?
df['chose_majority'] = (df['y'] == 2).astype(int)

print("\n" + "="*80)
print("ANALYZING MAJORITY PREFERENCE BY AGE")
print("="*80)

# Calculate proportion choosing majority by age
majority_by_age = df.groupby('age').agg({
    'chose_majority': ['mean', 'count']
}).reset_index()
majority_by_age.columns = ['age', 'proportion_majority', 'count']
print("\nProportion choosing majority by age:")
print(majority_by_age)

# Correlation between age and choosing majority
corr, p_value = stats.spearmanr(df['age'], df['chose_majority'])
print(f"\nSpearman correlation between age and choosing majority: r={corr:.4f}, p={p_value:.4f}")

# Logistic regression: age predicting majority choice
X_age = df[['age']].values
y_majority = df['chose_majority'].values

# Simple logistic regression
log_reg_simple = LogisticRegression(random_state=42, max_iter=1000)
log_reg_simple.fit(X_age, y_majority)
print(f"\nLogistic Regression (age -> majority choice):")
print(f"  Coefficient: {log_reg_simple.coef_[0][0]:.4f}")
print(f"  Intercept: {log_reg_simple.intercept_[0]:.4f}")

# Using statsmodels for p-values
X_age_sm = sm.add_constant(X_age)
logit_model = sm.Logit(y_majority, X_age_sm)
logit_result = logit_model.fit(disp=0)
print("\nLogistic Regression (statsmodels) - with p-values:")
print(logit_result.summary2().tables[1])

print("\n" + "="*80)
print("ANALYZING CULTURAL DIFFERENCES")
print("="*80)

# Proportion choosing majority by culture
majority_by_culture = df.groupby('culture').agg({
    'chose_majority': ['mean', 'count']
}).reset_index()
majority_by_culture.columns = ['culture', 'proportion_majority', 'count']
print("\nProportion choosing majority by culture:")
print(majority_by_culture)

# Chi-square test: culture and majority choice
contingency = pd.crosstab(df['culture'], df['chose_majority'])
chi2, p_chi, dof, expected = stats.chi2_contingency(contingency)
print(f"\nChi-square test (culture vs majority choice): χ²={chi2:.4f}, p={p_chi:.4f}")

print("\n" + "="*80)
print("AGE × CULTURE INTERACTION")
print("="*80)

# Logistic regression with interaction
df['age_centered'] = df['age'] - df['age'].mean()
df['culture_factor'] = df['culture'].astype(str)

# Create dummy variables for culture
culture_dummies = pd.get_dummies(df['culture'], prefix='culture', drop_first=True)
X_full = pd.concat([df[['age_centered']], culture_dummies], axis=1)

# Add interaction terms
for col in culture_dummies.columns:
    X_full[f'age_x_{col}'] = df['age_centered'] * culture_dummies[col]

# Fit model with interactions
X_full_sm = sm.add_constant(X_full).astype(float)
logit_full = sm.Logit(y_majority, X_full_sm)
logit_full_result = logit_full.fit(disp=0)

print("\nLogistic Regression with Age × Culture Interactions:")
print(logit_full_result.summary2().tables[1])

# Check for significant age effects
age_coef = logit_result.params[1]
age_pval = logit_result.pvalues[1]
print(f"\n\nMain effect of age: coefficient={age_coef:.4f}, p={age_pval:.4f}")

# Check if any interaction terms are significant
interaction_terms = [col for col in X_full.columns if 'age_x_' in col]
interaction_pvals = [logit_full_result.pvalues[col] for col in interaction_terms if col in logit_full_result.pvalues.index]
print(f"\nInteraction term p-values: {[f'{p:.4f}' for p in interaction_pvals]}")
min_interaction_p = min(interaction_pvals) if interaction_pvals else 1.0
print(f"Minimum interaction p-value: {min_interaction_p:.4f}")

print("\n" + "="*80)
print("DEVELOPMENTAL TREND WITHIN EACH CULTURE")
print("="*80)

# Analyze age effect within each culture
culture_age_effects = []
for culture_id in sorted(df['culture'].unique()):
    df_culture = df[df['culture'] == culture_id]
    if len(df_culture) > 10:  # Only analyze if sufficient data
        corr_c, p_c = stats.spearmanr(df_culture['age'], df_culture['chose_majority'])
        culture_age_effects.append({
            'culture': culture_id,
            'n': len(df_culture),
            'correlation': corr_c,
            'p_value': p_c
        })

culture_effects_df = pd.DataFrame(culture_age_effects)
print("\nAge-majority correlation within each culture:")
print(culture_effects_df)

# Count how many cultures show significant positive relationship
sig_positive = sum((culture_effects_df['correlation'] > 0) & (culture_effects_df['p_value'] < 0.05))
sig_negative = sum((culture_effects_df['correlation'] < 0) & (culture_effects_df['p_value'] < 0.05))
print(f"\nCultures with significant positive age effect: {sig_positive}/{len(culture_effects_df)}")
print(f"Cultures with significant negative age effect: {sig_negative}/{len(culture_effects_df)}")

print("\n" + "="*80)
print("FINAL INTERPRETATION")
print("="*80)

# Determine response based on statistical evidence
# The question asks: "How do children's reliance on majority preference develop over growth in age across different cultural contexts?"
# This is asking if there's development (change with age) across cultures

# Key evidence:
# 1. Overall age effect
# 2. Interaction effects (does it vary by culture?)
# 3. Within-culture trends

if age_pval < 0.05:
    print(f"\n✓ Significant main effect of age on majority preference (p={age_pval:.4f})")
    print(f"  Age coefficient: {age_coef:.4f} ({'positive' if age_coef > 0 else 'negative'})")
    
    if min_interaction_p < 0.05:
        print(f"\n✓ Significant Age × Culture interactions detected (min p={min_interaction_p:.4f})")
        print("  This suggests developmental trajectory varies across cultures")
        response_score = 85  # Strong yes - development varies by culture
        explanation = f"There is a significant main effect of age on majority preference (p={age_pval:.4f}, coef={age_coef:.4f}), and the developmental trajectory significantly varies across cultures (min interaction p={min_interaction_p:.4f}). {sig_positive}/{len(culture_effects_df)} cultures show significant positive age effects. This provides strong evidence that children's reliance on majority preference develops with age, but differently across cultural contexts."
    else:
        print(f"\n✗ No significant Age × Culture interactions (min p={min_interaction_p:.4f})")
        print("  Development pattern is similar across cultures")
        response_score = 75  # Yes - development exists but similar across cultures
        explanation = f"There is a significant main effect of age on majority preference (p={age_pval:.4f}, coef={age_coef:.4f}), indicating that children's reliance on majority preference develops with age. However, Age × Culture interactions are not significant (min p={min_interaction_p:.4f}), suggesting the developmental pattern is similar across cultural contexts. {sig_positive}/{len(culture_effects_df)} individual cultures show significant positive age effects."
else:
    print(f"\n✗ No significant main effect of age (p={age_pval:.4f})")
    
    if sig_positive >= len(culture_effects_df) / 2:
        print(f"  However, {sig_positive}/{len(culture_effects_df)} cultures individually show significant positive effects")
        response_score = 55  # Weak yes - mixed evidence
        explanation = f"While the overall main effect of age is not significant (p={age_pval:.4f}), {sig_positive}/{len(culture_effects_df)} individual cultures show significant positive correlations between age and majority preference. This suggests some developmental changes across cultures, but the effect is not uniform enough to be statistically significant overall."
    else:
        print(f"  Only {sig_positive}/{len(culture_effects_df)} cultures show significant positive effects")
        response_score = 25  # Mostly no
        explanation = f"There is no significant overall effect of age on majority preference (p={age_pval:.4f}), and only {sig_positive}/{len(culture_effects_df)} cultures show significant positive correlations individually. The evidence does not strongly support developmental changes in children's reliance on majority preference across cultural contexts."

print(f"\n{'='*80}")
print(f"RESPONSE SCORE: {response_score}/100")
print(f"EXPLANATION: {explanation}")
print(f"{'='*80}\n")

# Write conclusion
conclusion = {
    "response": response_score,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("✓ conclusion.txt written successfully")
