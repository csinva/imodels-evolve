import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from datetime import datetime
import json

# Load the data
df = pd.read_csv('fertility.csv')

# Research question: What is the effect of hormonal fluctuations associated with fertility on women's religiosity?

# Parse dates
df['DateTesting'] = pd.to_datetime(df['DateTesting'], format='%m/%d/%y')
df['StartDateofLastPeriod'] = pd.to_datetime(df['StartDateofLastPeriod'], format='%m/%d/%y')
df['StartDateofPeriodBeforeLast'] = pd.to_datetime(df['StartDateofPeriodBeforeLast'], format='%m/%d/%y')

# Calculate cycle day (days since last period started)
df['CycleDay'] = (df['DateTesting'] - df['StartDateofLastPeriod']).dt.days

# Calculate computed cycle length from the two period dates
df['ComputedCycleLength'] = (df['StartDateofLastPeriod'] - df['StartDateofPeriodBeforeLast']).dt.days

# Create a fertility measure: ovulation typically occurs around day 14 of the cycle
# High fertility period is typically days 10-16 of the cycle
df['FertilityPhase'] = df['CycleDay'].apply(lambda x: 1 if 10 <= x <= 16 else 0)

# Create an average religiosity score
df['AvgReligiosity'] = df[['Rel1', 'Rel2', 'Rel3']].mean(axis=1)

# Remove rows with missing values in key columns
df_clean = df.dropna(subset=['AvgReligiosity', 'CycleDay', 'ReportedCycleLength'])

print("="*60)
print("DATA EXPLORATION")
print("="*60)
print(f"Total participants: {len(df_clean)}")
print(f"\nReligiosity statistics:")
print(df_clean['AvgReligiosity'].describe())
print(f"\nCycle day statistics:")
print(df_clean['CycleDay'].describe())
print(f"\nFertility phase distribution:")
print(df_clean['FertilityPhase'].value_counts())

# Calculate correlation between cycle day and religiosity
correlation, p_value_corr = stats.pearsonr(df_clean['CycleDay'], df_clean['AvgReligiosity'])
print(f"\nCorrelation between cycle day and religiosity: r = {correlation:.4f}, p = {p_value_corr:.4f}")

# Compare religiosity between high fertility and low fertility phases
high_fertility = df_clean[df_clean['FertilityPhase'] == 1]['AvgReligiosity']
low_fertility = df_clean[df_clean['FertilityPhase'] == 0]['AvgReligiosity']

t_stat, p_value_ttest = stats.ttest_ind(high_fertility, low_fertility)
print(f"\nT-test comparing high vs. low fertility phases:")
print(f"  High fertility mean: {high_fertility.mean():.3f} (SD={high_fertility.std():.3f}, n={len(high_fertility)})")
print(f"  Low fertility mean: {low_fertility.mean():.3f} (SD={low_fertility.std():.3f}, n={len(low_fertility)})")
print(f"  t = {t_stat:.4f}, p = {p_value_ttest:.4f}")

# Regression analysis controlling for relationship status
print("\n" + "="*60)
print("REGRESSION ANALYSIS")
print("="*60)

# Prepare data for regression
X = df_clean[['CycleDay', 'Relationship', 'ReportedCycleLength', 'Sure1', 'Sure2']].copy()
y = df_clean['AvgReligiosity'].copy()

# Handle missing values in ReportedCycleLength
X['ReportedCycleLength'].fillna(X['ReportedCycleLength'].median(), inplace=True)

# Statsmodels regression with p-values
X_sm = sm.add_constant(X)
model = sm.OLS(y, X_sm).fit()
print(model.summary())

# Extract key results
cycle_day_coef = model.params['CycleDay']
cycle_day_pval = model.pvalues['CycleDay']
print(f"\nCycle day coefficient: {cycle_day_coef:.4f}, p-value: {cycle_day_pval:.4f}")

# Additional analysis: quadratic relationship
# Ovulation effects might show a non-linear pattern
X_quad = df_clean[['CycleDay']].copy()
X_quad['CycleDaySquared'] = X_quad['CycleDay'] ** 2
X_quad = sm.add_constant(X_quad)
y_quad = df_clean['AvgReligiosity'].copy()
model_quad = sm.OLS(y_quad, X_quad).fit()

print("\n" + "="*60)
print("QUADRATIC MODEL (Non-linear effects)")
print("="*60)
print(f"CycleDay coefficient: {model_quad.params['CycleDay']:.4f}, p = {model_quad.pvalues['CycleDay']:.4f}")
print(f"CycleDay^2 coefficient: {model_quad.params['CycleDaySquared']:.4f}, p = {model_quad.pvalues['CycleDaySquared']:.4f}")
print(f"Model R-squared: {model_quad.rsquared:.4f}")

# Determine conclusion
print("\n" + "="*60)
print("CONCLUSION")
print("="*60)

# Key findings:
# 1. Direct correlation between cycle day and religiosity
# 2. T-test comparing high vs low fertility phases
# 3. Regression coefficient for cycle day controlling for other factors

significant_effects = []
if p_value_corr < 0.05:
    significant_effects.append(f"correlation (p={p_value_corr:.4f})")
if p_value_ttest < 0.05:
    significant_effects.append(f"fertility phase t-test (p={p_value_ttest:.4f})")
if cycle_day_pval < 0.05:
    significant_effects.append(f"regression coefficient (p={cycle_day_pval:.4f})")

if len(significant_effects) > 0:
    response = 75  # Yes, significant relationship found
    explanation = f"Significant relationship detected between hormonal fluctuations (fertility cycle) and religiosity. "
    explanation += f"Significant findings: {', '.join(significant_effects)}. "
    
    if p_value_ttest < 0.05:
        direction = "higher" if high_fertility.mean() > low_fertility.mean() else "lower"
        explanation += f"Women in high fertility phase showed {direction} religiosity (mean difference = {abs(high_fertility.mean() - low_fertility.mean()):.3f}). "
    
    if cycle_day_pval < 0.05:
        direction = "positive" if cycle_day_coef > 0 else "negative"
        explanation += f"Cycle day showed a {direction} effect (β={cycle_day_coef:.4f}) on religiosity when controlling for relationship status and cycle characteristics."
else:
    response = 25  # No significant relationship
    explanation = f"No significant relationship found between hormonal fluctuations (fertility cycle) and religiosity. "
    explanation += f"Correlation test p={p_value_corr:.4f}, t-test p={p_value_ttest:.4f}, regression p={cycle_day_pval:.4f}. "
    explanation += "All tests failed to reach statistical significance (p < 0.05), suggesting that fertility-related hormonal fluctuations do not have a detectable effect on women's religiosity in this dataset."

print(f"Response: {response}")
print(f"Explanation: {explanation}")

# Write conclusion to file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n✓ Analysis complete. Results written to conclusion.txt")
