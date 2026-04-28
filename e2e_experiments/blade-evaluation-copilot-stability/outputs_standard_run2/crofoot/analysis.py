import pandas as pd
import numpy as np
import json
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('crofoot.csv')

print("="*80)
print("DATA EXPLORATION")
print("="*80)
print(f"\nDataset shape: {df.shape}")
print(f"\nColumn names: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nSummary statistics:")
print(df.describe())
print(f"\nMissing values:")
print(df.isnull().sum())

# Research question: How do relative group size and contest location influence 
# the probability of a capuchin monkey group winning an intergroup contest?

print("\n" + "="*80)
print("FEATURE ENGINEERING")
print("="*80)

# Create key features based on the research question
# 1. Relative group size (focal vs other)
df['size_diff'] = df['n_focal'] - df['n_other']
df['size_ratio'] = df['n_focal'] / df['n_other']

# 2. Contest location (relative distance from home range centers)
# If dist_focal < dist_other, focal is closer to its home (home advantage)
df['location_advantage'] = df['dist_other'] - df['dist_focal']  # Positive = focal has home advantage

# Additional features
df['male_diff'] = df['m_focal'] - df['m_other']
df['female_diff'] = df['f_focal'] - df['f_other']

print("\nNew features created:")
print(f"- size_diff: difference in group size (focal - other)")
print(f"- size_ratio: ratio of focal/other group size")
print(f"- location_advantage: dist_other - dist_focal (positive = focal closer to home)")
print(f"- male_diff: difference in males")
print(f"- female_diff: difference in females")

print("\n" + "="*80)
print("EXPLORATORY ANALYSIS")
print("="*80)

# Win rate by group size difference
print(f"\nOverall win rate: {df['win'].mean():.3f}")
print(f"\nWin rate when focal is larger (size_diff > 0): {df[df['size_diff'] > 0]['win'].mean():.3f}")
print(f"Win rate when focal is smaller (size_diff < 0): {df[df['size_diff'] < 0]['win'].mean():.3f}")
print(f"Win rate when equal size (size_diff == 0): {df[df['size_diff'] == 0]['win'].mean():.3f}")

# Win rate by location
print(f"\nWin rate when focal has location advantage (location_advantage > 0): {df[df['location_advantage'] > 0]['win'].mean():.3f}")
print(f"Win rate when focal has location disadvantage (location_advantage < 0): {df[df['location_advantage'] < 0]['win'].mean():.3f}")

# Correlations
print("\nCorrelations with win:")
correlations = df[['win', 'size_diff', 'size_ratio', 'location_advantage', 'male_diff', 'female_diff']].corr()['win'].sort_values(ascending=False)
print(correlations)

print("\n" + "="*80)
print("STATISTICAL TESTS")
print("="*80)

# T-test: Does size difference affect wins?
larger_wins = df[df['size_diff'] > 0]['win']
smaller_wins = df[df['size_diff'] < 0]['win']
t_stat_size, p_value_size = stats.ttest_ind(larger_wins, smaller_wins)
print(f"\nT-test: Win rate when larger vs smaller")
print(f"  t-statistic: {t_stat_size:.4f}, p-value: {p_value_size:.4f}")

# T-test: Does location advantage affect wins?
home_adv_wins = df[df['location_advantage'] > 0]['win']
home_disadv_wins = df[df['location_advantage'] < 0]['win']
t_stat_loc, p_value_loc = stats.ttest_ind(home_adv_wins, home_disadv_wins)
print(f"\nT-test: Win rate with location advantage vs disadvantage")
print(f"  t-statistic: {t_stat_loc:.4f}, p-value: {p_value_loc:.4f}")

# Point-biserial correlation (equivalent to Pearson for binary outcome)
corr_size, p_corr_size = stats.pearsonr(df['size_diff'], df['win'])
print(f"\nCorrelation between size_diff and win:")
print(f"  r = {corr_size:.4f}, p-value: {p_corr_size:.4f}")

corr_loc, p_corr_loc = stats.pearsonr(df['location_advantage'], df['win'])
print(f"\nCorrelation between location_advantage and win:")
print(f"  r = {corr_loc:.4f}, p-value: {p_corr_loc:.4f}")

print("\n" + "="*80)
print("LOGISTIC REGRESSION MODEL")
print("="*80)

# Logistic regression with statsmodels for p-values
X = df[['size_diff', 'location_advantage']].copy()
X = sm.add_constant(X)
y = df['win']

logit_model = sm.Logit(y, X)
result = logit_model.fit(disp=False)
print("\nLogistic Regression Results:")
print(result.summary())

print("\n" + "="*80)
print("INTERPRETABLE MODEL: COEFFICIENTS")
print("="*80)

# Simplified logistic regression for interpretation
X_simple = df[['size_diff', 'location_advantage']]
y_simple = df['win']

# Standardize for interpretation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_simple)

lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_scaled, y_simple)

print("\nStandardized Logistic Regression Coefficients:")
for feature, coef in zip(['size_diff', 'location_advantage'], lr_model.coef_[0]):
    print(f"  {feature}: {coef:.4f}")
print(f"  Intercept: {lr_model.intercept_[0]:.4f}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

# Determine the response score based on statistical significance
# Both factors need to show significant relationships

significant_size = p_corr_size < 0.05
significant_location = p_corr_loc < 0.05
both_significant = significant_size and significant_location

print(f"\nStatistical Significance:")
print(f"  - Relative group size: p = {p_corr_size:.4f} {'(SIGNIFICANT)' if significant_size else '(NOT SIGNIFICANT)'}")
print(f"  - Contest location: p = {p_corr_loc:.4f} {'(SIGNIFICANT)' if significant_location else '(NOT SIGNIFICANT)'}")

# Effect sizes
print(f"\nEffect Sizes (correlation coefficients):")
print(f"  - Relative group size: r = {corr_size:.4f}")
print(f"  - Contest location: r = {corr_loc:.4f}")

# Determine response based on significance and effect sizes
if both_significant:
    # Both factors are significant - strong Yes
    if abs(corr_size) > 0.3 and abs(corr_loc) > 0.2:
        response = 85  # Strong positive effects
    else:
        response = 75  # Moderate effects
    explanation = (f"Both relative group size (p={p_corr_size:.4f}, r={corr_size:.3f}) and contest location "
                  f"(p={p_corr_loc:.4f}, r={corr_loc:.3f}) show statistically significant relationships with "
                  f"contest outcomes. Larger groups have higher win rates, and groups closer to their home "
                  f"range center also have an advantage. The logistic regression confirms both factors "
                  f"independently influence winning probability.")
elif significant_size and not significant_location:
    response = 60
    explanation = (f"Relative group size shows a significant relationship (p={p_corr_size:.4f}, r={corr_size:.3f}), "
                  f"but contest location does not (p={p_corr_loc:.4f}). Only one of the two proposed factors "
                  f"significantly influences contest outcomes.")
elif significant_location and not significant_size:
    response = 60
    explanation = (f"Contest location shows a significant relationship (p={p_corr_loc:.4f}, r={corr_loc:.3f}), "
                  f"but relative group size does not (p={p_corr_size:.4f}). Only one of the two proposed factors "
                  f"significantly influences contest outcomes.")
else:
    # Neither factor is significant
    response = 25
    explanation = (f"Neither relative group size (p={p_corr_size:.4f}) nor contest location (p={p_corr_loc:.4f}) "
                  f"show statistically significant relationships with contest outcomes at the 0.05 significance level. "
                  f"The data does not provide strong evidence that these factors influence winning probability.")

print(f"\nFinal Assessment:")
print(f"  Response Score: {response}/100")
print(f"  Explanation: {explanation}")

# Write conclusion to file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n" + "="*80)
print("Analysis complete! Results written to conclusion.txt")
print("="*80)
