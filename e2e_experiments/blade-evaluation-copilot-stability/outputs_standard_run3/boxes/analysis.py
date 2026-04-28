import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import json

# Load the data
df = pd.read_csv('boxes.csv')

# Explore the data
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nSummary statistics:")
print(df.describe())
print("\nData types:")
print(df.dtypes)
print("\nValue counts for y (outcome):")
print(df['y'].value_counts().sort_index())

# Research question: How do children's reliance on majority preference develop over growth in age across different cultural contexts?
# y=1: unchosen option, y=2: majority option, y=3: minority option

# Create binary indicator for choosing majority (y=2)
df['chose_majority'] = (df['y'] == 2).astype(int)

print("\n" + "="*60)
print("ANALYSIS: Reliance on Majority Preference by Age and Culture")
print("="*60)

# 1. Overall relationship between age and majority choice
print("\n1. Correlation between age and majority choice:")
corr, p_value = stats.pearsonr(df['age'], df['chose_majority'])
print(f"   Pearson r = {corr:.4f}, p-value = {p_value:.4e}")

# 2. Logistic regression: majority choice ~ age
print("\n2. Logistic Regression: Majority Choice ~ Age")
X_age = sm.add_constant(df['age'])
logit_age = sm.Logit(df['chose_majority'], X_age)
result_age = logit_age.fit(disp=False)
print(result_age.summary())

# 3. Age effect across cultures
print("\n3. Analyzing Age Effect Across Cultures:")
cultures = sorted(df['culture'].unique())
culture_results = []

for culture in cultures:
    culture_df = df[df['culture'] == culture]
    n = len(culture_df)
    
    # Correlation for this culture
    if n > 10:  # Only if enough data
        corr_c, p_c = stats.pearsonr(culture_df['age'], culture_df['chose_majority'])
        culture_results.append({
            'culture': culture,
            'n': n,
            'corr': corr_c,
            'p_value': p_c
        })
        print(f"   Culture {culture} (n={n}): r={corr_c:.3f}, p={p_c:.4f}")

# 4. Interaction model: age * culture
print("\n4. Logistic Regression with Age x Culture Interaction:")
# Create culture dummies
df['culture_cat'] = df['culture'].astype('category')
culture_dummies = pd.get_dummies(df['culture'], prefix='culture', drop_first=True)

# Main effects + interaction
X_interaction = pd.DataFrame({
    'const': np.ones(len(df)),
    'age': df['age'].values,
})
# Add culture dummies
for col in culture_dummies.columns:
    X_interaction[col] = culture_dummies[col].values
    # Add interaction terms
    X_interaction[f'age_x_{col}'] = df['age'].values * culture_dummies[col].values

# Ensure all columns are numeric
X_interaction = X_interaction.astype(float)

logit_interaction = sm.Logit(df['chose_majority'], X_interaction)
result_interaction = logit_interaction.fit(disp=False)
print(result_interaction.summary())

# 5. Test if age effect varies by culture (likelihood ratio test)
# Compare model with and without interactions
X_no_interaction = X_interaction[[col for col in X_interaction.columns if 'age_x_' not in col]]
logit_no_interaction = sm.Logit(df['chose_majority'], X_no_interaction)
result_no_interaction = logit_no_interaction.fit(disp=False)

lr_stat = 2 * (result_interaction.llf - result_no_interaction.llf)
df_diff = result_interaction.df_model - result_no_interaction.df_model
lr_p = stats.chi2.sf(lr_stat, df_diff)

print(f"\n5. Likelihood Ratio Test for Age x Culture Interaction:")
print(f"   LR statistic = {lr_stat:.3f}, df = {df_diff}, p-value = {lr_p:.4f}")

# 6. Examine age trends within each culture more carefully
print("\n6. Age Trend Analysis by Culture:")
print("   (Testing if majority choice increases with age in each culture)")

culture_age_effects = []
for culture in cultures:
    culture_df = df[df['culture'] == culture]
    if len(culture_df) > 10:
        X_c = sm.add_constant(culture_df['age'])
        logit_c = sm.Logit(culture_df['chose_majority'], X_c)
        result_c = logit_c.fit(disp=False)
        age_coef = result_c.params['age']
        age_p = result_c.pvalues['age']
        culture_age_effects.append({
            'culture': culture,
            'age_coefficient': age_coef,
            'p_value': age_p,
            'significant': age_p < 0.05
        })
        sig_str = "*" if age_p < 0.05 else " "
        print(f"   Culture {culture}: age coef={age_coef:+.4f}, p={age_p:.4f} {sig_str}")

# Summary statistics by age group
print("\n7. Majority Choice Rate by Age Group:")
df['age_group'] = pd.cut(df['age'], bins=[3, 6, 9, 15], labels=['4-6', '7-9', '10-14'])
age_group_summary = df.groupby('age_group')['chose_majority'].agg(['mean', 'count'])
print(age_group_summary)

# ANOVA for age groups
print("\n8. ANOVA: Majority Choice across Age Groups")
age_groups = [df[df['age_group'] == ag]['chose_majority'].values for ag in ['4-6', '7-9', '10-14']]
f_stat, anova_p = stats.f_oneway(*age_groups)
print(f"   F-statistic = {f_stat:.3f}, p-value = {anova_p:.4e}")

# Summary by age and culture
print("\n9. Majority Choice Rate by Age Group and Culture:")
summary_table = df.groupby(['age_group', 'culture'])['chose_majority'].agg(['mean', 'count'])
print(summary_table)

# DETERMINE CONCLUSION
print("\n" + "="*60)
print("CONCLUSION")
print("="*60)

# The research question asks about development across age AND across cultures
# Key findings:
# - Overall age effect (from model 2)
# - Age x culture interaction (from model 4 and LR test)
# - Individual culture age effects (from model 6)

age_coef_overall = result_age.params['age']
age_p_overall = result_age.pvalues['age']

significant_cultures = sum([1 for c in culture_age_effects if c['significant']])
total_cultures = len(culture_age_effects)

interaction_significant = lr_p < 0.05

print(f"\nKey Findings:")
print(f"1. Overall age effect: coef={age_coef_overall:+.4f}, p={age_p_overall:.4e}")
print(f"2. Age x Culture interaction: p={lr_p:.4f}")
print(f"3. Significant age effects in {significant_cultures}/{total_cultures} cultures")

# Determine response score
if age_p_overall < 0.001 and age_coef_overall > 0:
    # Strong positive age effect overall
    if interaction_significant:
        # Development varies by culture
        response = 80  # Strong yes, but varies by culture
        explanation = f"Children's reliance on majority preference significantly increases with age (p={age_p_overall:.2e}, positive coefficient). However, this developmental pattern varies significantly across cultures (interaction p={lr_p:.3f}), with {significant_cultures} of {total_cultures} cultures showing significant age effects."
    else:
        # Consistent across cultures
        response = 90  # Very strong yes
        explanation = f"Children's reliance on majority preference significantly and consistently increases with age across cultures (p={age_p_overall:.2e}, positive coefficient). The age effect does not significantly vary by culture (interaction p={lr_p:.3f}), suggesting a universal developmental pattern."
elif age_p_overall < 0.05 and age_coef_overall > 0:
    # Moderate positive age effect
    if interaction_significant:
        response = 65
        explanation = f"There is a significant positive relationship between age and majority preference (p={age_p_overall:.3f}), but this pattern varies significantly across cultures (interaction p={lr_p:.3f}), indicating culture-specific developmental trajectories."
    else:
        response = 75
        explanation = f"Children's reliance on majority preference increases with age (p={age_p_overall:.3f}, positive coefficient), with consistent patterns across cultures (interaction p={lr_p:.3f})."
elif age_p_overall < 0.05 and age_coef_overall < 0:
    # Significant negative effect (unexpected)
    response = 30
    explanation = f"Surprisingly, majority preference shows a significant negative relationship with age (p={age_p_overall:.3f}), opposite to expected developmental patterns."
else:
    # Not significant overall
    if interaction_significant:
        response = 40
        explanation = f"While the overall age effect is not significant (p={age_p_overall:.3f}), there is significant variation across cultures (interaction p={lr_p:.3f}), suggesting culture-specific but not universal developmental patterns."
    else:
        response = 20
        explanation = f"No significant overall relationship between age and majority preference (p={age_p_overall:.3f}), and no significant variation across cultures (interaction p={lr_p:.3f})."

print(f"\nFinal Score: {response}/100")
print(f"Explanation: {explanation}")

# Write conclusion to file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\nConclusion written to conclusion.txt")
