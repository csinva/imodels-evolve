#!/usr/bin/env python3
import json
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.formula.api import ols
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('boxes.csv')

print("=" * 80)
print("DATA EXPLORATION")
print("=" * 80)
print(f"\nDataset shape: {df.shape}")
print(f"\nColumn types:\n{df.dtypes}")
print(f"\nSummary statistics:\n{df.describe()}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# The research question asks about how children's reliance on majority preference
# develops with age across different cultural contexts
# y = 1 (unchosen), 2 (majority), 3 (minority)
# We'll focus on whether children choose the majority option (y=2)

# Create binary outcome: chose majority (1) vs not (0)
df['chose_majority'] = (df['y'] == 2).astype(int)

print("\n" + "=" * 80)
print("OUTCOME DISTRIBUTION")
print("=" * 80)
print(f"\nChoice distribution:")
print(df['y'].value_counts().sort_index())
print(f"\n1 = unchosen option: {(df['y'] == 1).sum()} ({(df['y'] == 1).mean()*100:.1f}%)")
print(f"2 = majority option: {(df['y'] == 2).sum()} ({(df['y'] == 2).mean()*100:.1f}%)")
print(f"3 = minority option: {(df['y'] == 3).sum()} ({(df['y'] == 3).mean()*100:.1f}%)")

print("\n" + "=" * 80)
print("AGE AND MAJORITY PREFERENCE ANALYSIS")
print("=" * 80)

# Analyze how majority preference changes with age
age_majority = df.groupby('age')['chose_majority'].agg(['mean', 'count'])
print("\nProportion choosing majority by age:")
print(age_majority)

# Test correlation between age and choosing majority
correlation, p_value = stats.spearmanr(df['age'], df['chose_majority'])
print(f"\nSpearman correlation between age and choosing majority: r={correlation:.4f}, p={p_value:.4f}")

# Logistic regression: age predicting majority choice
X_age = df[['age']].values
y_binary = df['chose_majority'].values

# Fit logistic regression
lr_age = LogisticRegression(random_state=42, max_iter=1000)
lr_age.fit(X_age, y_binary)
print(f"\nLogistic Regression - Age coefficient: {lr_age.coef_[0][0]:.4f}")
print(f"Intercept: {lr_age.intercept_[0]:.4f}")

# Use statsmodels for p-values
logit_model = sm.Logit(y_binary, sm.add_constant(X_age))
result = logit_model.fit(disp=0)
print(f"\nStatsmodels Logistic Regression Results:")
print(result.summary())

age_coef = result.params[1]
age_pval = result.pvalues[1]
print(f"\nAge coefficient: {age_coef:.4f}, p-value: {age_pval:.6f}")

print("\n" + "=" * 80)
print("CULTURE ANALYSIS")
print("=" * 80)

# Check how culture moderates the age effect
culture_majority = df.groupby('culture')['chose_majority'].agg(['mean', 'count'])
print("\nProportion choosing majority by culture:")
print(culture_majority)

# Age x Culture interaction
print("\nProportion choosing majority by age and culture:")
age_culture = df.groupby(['age', 'culture'])['chose_majority'].mean().unstack(fill_value=np.nan)
print(age_culture)

# Test for interaction using logistic regression
df_encoded = df.copy()
df_encoded['culture'] = df_encoded['culture'].astype('category')

# Full model with interaction
X_full = pd.get_dummies(df_encoded[['age', 'culture']], columns=['culture'], drop_first=True)
# Add interaction terms
for col in X_full.columns:
    if col.startswith('culture_'):
        X_full[f'age_x_{col}'] = df_encoded['age'].values * X_full[col].values

# Convert all columns to float
X_full = X_full.astype(float)

logit_full = sm.Logit(y_binary, sm.add_constant(X_full))
result_full = logit_full.fit(disp=0)
print(f"\nFull model with age x culture interactions:")
print(result_full.summary())

# Test if age effect differs across cultures using ANOVA-like approach
# Fit separate regressions per culture
print("\n" + "=" * 80)
print("AGE EFFECT BY CULTURE")
print("=" * 80)

age_effects = {}
for culture_id in df['culture'].unique():
    df_culture = df[df['culture'] == culture_id]
    if len(df_culture) > 10:  # Only if enough samples
        X_c = df_culture[['age']].values
        y_c = df_culture['chose_majority'].values
        
        try:
            logit_c = sm.Logit(y_c, sm.add_constant(X_c))
            result_c = logit_c.fit(disp=0)
            age_coef_c = result_c.params[1]
            age_pval_c = result_c.pvalues[1]
            age_effects[culture_id] = {
                'coef': age_coef_c,
                'pval': age_pval_c,
                'n': len(df_culture)
            }
            print(f"\nCulture {culture_id} (n={len(df_culture)}): age_coef={age_coef_c:.4f}, p={age_pval_c:.4f}")
        except:
            print(f"\nCulture {culture_id}: Could not fit model")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

# Determine the answer to the research question
# "How do children's reliance on majority preference develop over growth in age 
# across different cultural contexts?"

# Key findings:
# 1. Is there a significant age effect overall?
overall_age_significant = age_pval < 0.05
age_effect_positive = age_coef > 0

# 2. Does the effect vary by culture (interaction)?
# Check if interaction terms are significant
interaction_significant = False
if result_full is not None:
    interaction_pvals = []
    for param_name, pval in result_full.pvalues.items():
        if 'age_x_culture' in param_name:
            interaction_pvals.append(pval)
    if interaction_pvals and min(interaction_pvals) < 0.05:
        interaction_significant = True

# 3. Count how many cultures show significant age effects
n_cultures_sig = sum(1 for eff in age_effects.values() if eff['pval'] < 0.05)
n_cultures_pos = sum(1 for eff in age_effects.values() if eff['coef'] > 0)

print(f"\nOverall age effect: coef={age_coef:.4f}, p={age_pval:.6f}, significant={overall_age_significant}")
print(f"Age effect is positive: {age_effect_positive}")
print(f"Interaction with culture significant: {interaction_significant}")
print(f"Number of cultures with significant positive age effect: {n_cultures_sig}/{len(age_effects)}")
print(f"Number of cultures with positive age coefficient: {n_cultures_pos}/{len(age_effects)}")

# Build explanation
explanation_parts = []

if overall_age_significant and age_effect_positive:
    explanation_parts.append(f"There is a significant positive relationship between age and majority preference (coef={age_coef:.3f}, p={age_pval:.4f})")
    base_score = 75
elif overall_age_significant:
    explanation_parts.append(f"There is a significant age effect (p={age_pval:.4f}) but direction is negative")
    base_score = 50
else:
    explanation_parts.append(f"No significant overall age effect on majority preference (p={age_pval:.4f})")
    base_score = 30

if interaction_significant:
    explanation_parts.append("The age effect varies significantly across cultures, suggesting cultural context matters")
    base_score += 10
else:
    explanation_parts.append("The age effect is relatively consistent across cultures")

if n_cultures_sig > len(age_effects) * 0.5:
    explanation_parts.append(f"{n_cultures_sig}/{len(age_effects)} cultures show individually significant age effects")
    base_score += 5
else:
    explanation_parts.append(f"Only {n_cultures_sig}/{len(age_effects)} cultures show individually significant age effects")
    base_score -= 10

# Cap the score
response_score = min(max(base_score, 0), 100)

explanation = ". ".join(explanation_parts) + "."

print(f"\nFinal Response Score: {response_score}")
print(f"Explanation: {explanation}")

# Write conclusion
conclusion = {
    "response": response_score,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n" + "=" * 80)
print("Analysis complete! Conclusion written to conclusion.txt")
print("=" * 80)
