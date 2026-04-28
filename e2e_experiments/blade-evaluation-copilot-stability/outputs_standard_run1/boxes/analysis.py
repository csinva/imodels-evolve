import pandas as pd
import numpy as np
import json
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('boxes.csv')

print("="*80)
print("DATA EXPLORATION")
print("="*80)
print(f"\nDataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head(10))
print(f"\nSummary statistics:")
print(df.describe())
print(f"\nMissing values:")
print(df.isnull().sum())

# Research question: How do children's reliance on majority preference develop 
# over growth in age across different cultural contexts?

# y=1: unchosen option (not following anyone)
# y=2: majority option (following majority)
# y=3: minority option (following minority)

print("\n" + "="*80)
print("ANALYZING MAJORITY PREFERENCE BY AGE")
print("="*80)

# Create binary variable: follows majority (y=2) vs not (y=1 or y=3)
df['follows_majority'] = (df['y'] == 2).astype(int)

print(f"\nOverall majority following rate: {df['follows_majority'].mean():.3f}")
print(f"\nMajority following by age:")
age_majority = df.groupby('age')['follows_majority'].agg(['mean', 'count'])
print(age_majority)

# Test correlation between age and majority following
correlation, p_value_corr = stats.pearsonr(df['age'], df['follows_majority'])
print(f"\nPearson correlation between age and following majority: r={correlation:.4f}, p={p_value_corr:.4e}")

# Logistic regression: Does age predict majority following?
X = df[['age']].values
y = df['follows_majority'].values

# Standardize age for better interpretation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit logistic regression
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_scaled, y)

print(f"\nLogistic regression coefficient for age (standardized): {log_reg.coef_[0][0]:.4f}")

# Use statsmodels for p-values
X_sm = sm.add_constant(df['age'])
logit_model = sm.Logit(df['follows_majority'], X_sm)
logit_result = logit_model.fit(disp=0)
print(f"\nStatsmodels Logistic Regression Results:")
print(logit_result.summary())

age_coef = logit_result.params['age']
age_pvalue = logit_result.pvalues['age']
print(f"\nAge coefficient: {age_coef:.4f}, p-value: {age_pvalue:.4e}")

print("\n" + "="*80)
print("ANALYZING CULTURAL CONTEXT EFFECTS")
print("="*80)

print(f"\nMajority following by culture:")
culture_majority = df.groupby('culture')['follows_majority'].agg(['mean', 'count'])
print(culture_majority)

# ANOVA: Does culture affect majority following?
cultures = [group['follows_majority'].values for name, group in df.groupby('culture')]
f_stat, p_value_anova = stats.f_oneway(*cultures)
print(f"\nANOVA test for culture effect: F={f_stat:.4f}, p={p_value_anova:.4e}")

# Test interaction: age × culture
print("\n" + "="*80)
print("TESTING AGE × CULTURE INTERACTION")
print("="*80)

# Create interaction term
df['age_x_culture'] = df['age'] * df['culture']

# Fit model with interaction
X_interaction = sm.add_constant(df[['age', 'culture', 'age_x_culture']])
logit_interaction = sm.Logit(df['follows_majority'], X_interaction)
result_interaction = logit_interaction.fit(disp=0)
print(result_interaction.summary())

interaction_coef = result_interaction.params['age_x_culture']
interaction_pvalue = result_interaction.pvalues['age_x_culture']
print(f"\nInteraction term (age × culture) coefficient: {interaction_coef:.4f}, p-value: {interaction_pvalue:.4e}")

# Additional analysis: Linear trend across age groups
print("\n" + "="*80)
print("LINEAR TREND ANALYSIS")
print("="*80)

# Compute mean majority following by age
age_means = df.groupby('age')['follows_majority'].mean()
ages = age_means.index.values.reshape(-1, 1)
means = age_means.values

# Linear regression on means
lr = LinearRegression()
lr.fit(ages, means)
slope = lr.coef_[0]
intercept = lr.intercept_
r2 = lr.score(ages, means)

print(f"Linear trend: slope={slope:.4f}, intercept={intercept:.4f}, R²={r2:.4f}")

# Spearman correlation (non-parametric)
spearman_corr, spearman_p = stats.spearmanr(df['age'], df['follows_majority'])
print(f"Spearman correlation: rho={spearman_corr:.4f}, p={spearman_p:.4e}")

print("\n" + "="*80)
print("FINAL INTERPRETATION")
print("="*80)

# Determine the answer to the research question
# "How do children's reliance on majority preference develop over growth in age 
# across different cultural contexts?"

# Key findings:
# 1. Is there a significant relationship between age and majority following?
# 2. Does culture moderate this relationship (interaction)?

age_significant = age_pvalue < 0.05
interaction_significant = interaction_pvalue < 0.05
age_positive = age_coef > 0

print(f"\nKey findings:")
print(f"1. Age effect on majority following: {'SIGNIFICANT' if age_significant else 'NOT SIGNIFICANT'} (p={age_pvalue:.4e})")
print(f"2. Direction of age effect: {'POSITIVE' if age_positive else 'NEGATIVE'} (coef={age_coef:.4f})")
print(f"3. Age × Culture interaction: {'SIGNIFICANT' if interaction_significant else 'NOT SIGNIFICANT'} (p={interaction_pvalue:.4e})")
print(f"4. Culture main effect: {'SIGNIFICANT' if p_value_anova < 0.05 else 'NOT SIGNIFICANT'} (p={p_value_anova:.4e})")

# Construct conclusion
if age_significant and interaction_significant:
    # Strong evidence for developmental change that varies by culture
    response = 85
    explanation = (
        f"Yes, there is strong evidence that children's reliance on majority preference "
        f"develops with age across cultural contexts. Age significantly predicts majority "
        f"following (p={age_pvalue:.4e}, coef={age_coef:.4f}), and this relationship is "
        f"significantly moderated by culture (interaction p={interaction_pvalue:.4e}). "
        f"This indicates developmental trajectories differ across cultural contexts."
    )
elif age_significant and not interaction_significant:
    # Moderate evidence: age effect exists but similar across cultures
    response = 70
    explanation = (
        f"Yes, children's reliance on majority preference develops with age "
        f"(p={age_pvalue:.4e}, coef={age_coef:.4f}), but this developmental pattern "
        f"appears relatively consistent across cultures (no significant interaction, "
        f"p={interaction_pvalue:.4e}). The age effect is present but not strongly "
        f"moderated by cultural context."
    )
elif not age_significant and interaction_significant:
    # Weak evidence: complex cultural differences but no clear main age effect
    response = 45
    explanation = (
        f"The relationship is complex. While there is no significant main effect of age "
        f"(p={age_pvalue:.4e}), there is a significant age × culture interaction "
        f"(p={interaction_pvalue:.4e}), suggesting developmental patterns vary by culture "
        f"but without a consistent overall trend."
    )
else:
    # No significant relationships found
    response = 20
    explanation = (
        f"No, there is insufficient evidence that children's reliance on majority preference "
        f"systematically develops with age across cultural contexts. Neither the main effect "
        f"of age (p={age_pvalue:.4e}) nor the age × culture interaction (p={interaction_pvalue:.4e}) "
        f"reached statistical significance."
    )

print(f"\nConclusion Score: {response}/100")
print(f"Explanation: {explanation}")

# Write conclusion to file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n" + "="*80)
print("Analysis complete! conclusion.txt has been created.")
print("="*80)
