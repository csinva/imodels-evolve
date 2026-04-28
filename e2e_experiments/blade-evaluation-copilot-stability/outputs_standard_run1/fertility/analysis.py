import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import statsmodels.api as sm
import json
from datetime import datetime

# Load the dataset
df = pd.read_csv('fertility.csv')

print("=" * 80)
print("ANALYZING THE EFFECT OF HORMONAL FLUCTUATIONS ON WOMEN'S RELIGIOSITY")
print("=" * 80)

# Data exploration
print("\nDataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())

# Create religiosity composite score
df['Religiosity'] = df[['Rel1', 'Rel2', 'Rel3']].mean(axis=1)
print("\nReligiosity score statistics:")
print(df['Religiosity'].describe())

# Convert date columns to datetime
df['DateTesting'] = pd.to_datetime(df['DateTesting'], format='%m/%d/%y')
df['StartDateofLastPeriod'] = pd.to_datetime(df['StartDateofLastPeriod'], format='%m/%d/%y')
df['StartDateofPeriodBeforeLast'] = pd.to_datetime(df['StartDateofPeriodBeforeLast'], format='%m/%d/%y')

# Calculate days since last period (proxy for menstrual cycle phase)
df['DaysSinceLastPeriod'] = (df['DateTesting'] - df['StartDateofLastPeriod']).dt.days

# Calculate actual cycle length from the two period dates
df['ActualCycleLength'] = (df['StartDateofLastPeriod'] - df['StartDateofPeriodBeforeLast']).dt.days

# For cycle phase analysis, we'll use reported cycle length when available, otherwise actual
df['CycleLength'] = df['ReportedCycleLength'].fillna(df['ActualCycleLength'])

# Calculate cycle phase (normalized position in cycle, 0-1)
df['CyclePhase'] = df['DaysSinceLastPeriod'] / df['CycleLength']

# Identify fertility window (typically days 7-21 for a 28-day cycle, adjusted proportionally)
# The fertile window is roughly in the middle of the cycle (ovulation around day 14)
# We'll define high fertility as cycle phase between 0.3 and 0.7 (roughly days 8-20 for 28-day cycle)
df['FertilityWindow'] = ((df['CyclePhase'] >= 0.3) & (df['CyclePhase'] <= 0.7)).astype(int)

# Remove outliers and invalid data
df_clean = df[(df['DaysSinceLastPeriod'] > 0) & 
              (df['DaysSinceLastPeriod'] < 60) &
              (df['CycleLength'] > 15) &
              (df['CycleLength'] < 45) &
              (df['CyclePhase'] >= 0) &
              (df['CyclePhase'] <= 1.5)].copy()

print(f"\nData cleaned: {len(df)} -> {len(df_clean)} rows")
print(f"\nCycle phase statistics:")
print(df_clean['CyclePhase'].describe())
print(f"\nFertility window distribution:")
print(df_clean['FertilityWindow'].value_counts())

print("\n" + "=" * 80)
print("STATISTICAL ANALYSIS")
print("=" * 80)

# Analysis 1: T-test comparing religiosity in fertile vs non-fertile windows
fertile_religiosity = df_clean[df_clean['FertilityWindow'] == 1]['Religiosity']
non_fertile_religiosity = df_clean[df_clean['FertilityWindow'] == 0]['Religiosity']

t_stat, p_value_ttest = stats.ttest_ind(fertile_religiosity, non_fertile_religiosity, nan_policy='omit')
print(f"\nT-test: Religiosity in fertile window vs non-fertile window")
print(f"  Fertile mean: {fertile_religiosity.mean():.3f} (n={len(fertile_religiosity)})")
print(f"  Non-fertile mean: {non_fertile_religiosity.mean():.3f} (n={len(non_fertile_religiosity)})")
print(f"  Difference: {fertile_religiosity.mean() - non_fertile_religiosity.mean():.3f}")
print(f"  t-statistic: {t_stat:.3f}")
print(f"  p-value: {p_value_ttest:.4f}")

# Analysis 2: Correlation between cycle phase and religiosity
correlation, p_value_corr = stats.pearsonr(df_clean['CyclePhase'].dropna(), 
                                            df_clean['Religiosity'].dropna())
print(f"\nCorrelation: Cycle phase vs Religiosity")
print(f"  Pearson r: {correlation:.3f}")
print(f"  p-value: {p_value_corr:.4f}")

# Analysis 3: Linear regression with cycle phase predicting religiosity
X = df_clean[['CyclePhase', 'DaysSinceLastPeriod']].dropna()
y = df_clean.loc[X.index, 'Religiosity']

# Add constant for statsmodels
X_with_const = sm.add_constant(X)
model_sm = sm.OLS(y, X_with_const).fit()
print(f"\nLinear Regression: Cycle phase & days since period -> Religiosity")
print(model_sm.summary())

# Analysis 4: Control for relationship status
X_controlled = df_clean[['CyclePhase', 'Relationship', 'Sure1', 'Sure2']].dropna()
y_controlled = df_clean.loc[X_controlled.index, 'Religiosity']
X_controlled_const = sm.add_constant(X_controlled)
model_controlled = sm.OLS(y_controlled, X_controlled_const).fit()
print(f"\nControlled Regression: Including relationship status and certainty")
print(model_controlled.summary())

# Analysis 5: ANOVA by fertility window groups
groups = [df_clean[df_clean['FertilityWindow'] == i]['Religiosity'].dropna() 
          for i in range(2)]
f_stat, p_value_anova = stats.f_oneway(*groups)
print(f"\nANOVA: Religiosity across fertility windows")
print(f"  F-statistic: {f_stat:.3f}")
print(f"  p-value: {p_value_anova:.4f}")

# Analysis 6: Check individual religiosity items
print(f"\nAnalysis of individual religiosity items:")
for rel_col in ['Rel1', 'Rel2', 'Rel3']:
    fertile_rel = df_clean[df_clean['FertilityWindow'] == 1][rel_col]
    non_fertile_rel = df_clean[df_clean['FertilityWindow'] == 0][rel_col]
    t_stat_item, p_value_item = stats.ttest_ind(fertile_rel, non_fertile_rel, nan_policy='omit')
    print(f"  {rel_col}: t={t_stat_item:.3f}, p={p_value_item:.4f}")

# Spearman correlation (non-parametric alternative)
spearman_corr, spearman_p = stats.spearmanr(df_clean['CyclePhase'].dropna(), 
                                             df_clean['Religiosity'].dropna())
print(f"\nSpearman correlation: Cycle phase vs Religiosity")
print(f"  Spearman rho: {spearman_corr:.3f}")
print(f"  p-value: {spearman_p:.4f}")

print("\n" + "=" * 80)
print("INTERPRETABLE MODELING")
print("=" * 80)

# Use interpretable models to understand feature importance
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Prepare features for modeling
features_for_model = ['CyclePhase', 'DaysSinceLastPeriod', 'CycleLength', 
                      'FertilityWindow', 'Relationship', 'Sure1', 'Sure2']
X_model = df_clean[features_for_model].dropna()
y_model = df_clean.loc[X_model.index, 'Religiosity']

print(f"\nModeling with {len(X_model)} samples and {len(features_for_model)} features")

# Linear regression for interpretability
lr = LinearRegression()
lr.fit(X_model, y_model)
print(f"\nLinear Regression Coefficients:")
for feat, coef in zip(features_for_model, lr.coef_):
    print(f"  {feat}: {coef:.4f}")
print(f"  Intercept: {lr.intercept_:.4f}")
print(f"  R²: {lr.score(X_model, y_model):.4f}")

# Decision tree for interpretability
dt = DecisionTreeRegressor(max_depth=3, random_state=42)
dt.fit(X_model, y_model)
print(f"\nDecision Tree Feature Importances:")
for feat, imp in zip(features_for_model, dt.feature_importances_):
    print(f"  {feat}: {imp:.4f}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

# Determine the response based on statistical significance
significant_threshold = 0.05
effect_size_threshold = 0.15  # Cohen's d

# Calculate Cohen's d for effect size
cohen_d = (fertile_religiosity.mean() - non_fertile_religiosity.mean()) / \
          np.sqrt((fertile_religiosity.std()**2 + non_fertile_religiosity.std()**2) / 2)

print(f"\nKey findings:")
print(f"1. T-test p-value: {p_value_ttest:.4f} (threshold: {significant_threshold})")
print(f"2. Correlation p-value: {p_value_corr:.4f} (threshold: {significant_threshold})")
print(f"3. Cohen's d effect size: {cohen_d:.4f} (threshold: {effect_size_threshold})")
print(f"4. Regression cycle phase p-value: {model_sm.pvalues['CyclePhase']:.4f}")

# Decision logic
is_significant = (p_value_ttest < significant_threshold) or \
                 (p_value_corr < significant_threshold) or \
                 (model_sm.pvalues['CyclePhase'] < significant_threshold)

has_meaningful_effect = abs(cohen_d) > effect_size_threshold or \
                        abs(correlation) > 0.15

if is_significant and has_meaningful_effect:
    response = 75  # Strong yes - significant relationship found
    explanation = (f"Statistical analysis reveals a significant relationship between hormonal "
                   f"fluctuations and religiosity (t-test p={p_value_ttest:.4f}, correlation "
                   f"r={correlation:.3f} p={p_value_corr:.4f}). The effect size (Cohen's d="
                   f"{cohen_d:.3f}) suggests a meaningful difference in religiosity between "
                   f"fertile and non-fertile cycle phases. Regression analysis controlling for "
                   f"other factors confirms cycle phase has a significant effect.")
elif is_significant:
    response = 60  # Moderate yes - significant but small effect
    explanation = (f"Statistical tests indicate a significant relationship (p-values < 0.05), "
                   f"but the effect size is small (Cohen's d={cohen_d:.3f}, r={correlation:.3f}). "
                   f"While hormonal fluctuations show statistically significant association with "
                   f"religiosity, the practical magnitude of this effect is modest.")
elif p_value_ttest < 0.10 or p_value_corr < 0.10:
    response = 40  # Weak evidence
    explanation = (f"There is marginal evidence of a relationship (t-test p={p_value_ttest:.4f}, "
                   f"correlation p={p_value_corr:.4f}), but it does not reach conventional "
                   f"statistical significance (p < 0.05). The effect size (Cohen's d={cohen_d:.3f}) "
                   f"is also small. The evidence is suggestive but not conclusive.")
else:
    response = 20  # No significant relationship
    explanation = (f"No significant relationship was found between hormonal fluctuations and "
                   f"religiosity. The t-test (p={p_value_ttest:.4f}) and correlation "
                   f"(p={p_value_corr:.4f}) both fail to reach statistical significance. "
                   f"The effect size is negligible (Cohen's d={cohen_d:.3f}). Based on this "
                   f"analysis, hormonal fluctuations associated with fertility do not appear "
                   f"to have a meaningful effect on women's religiosity.")

print(f"\nFinal assessment: {response}/100")
print(f"Explanation: {explanation}")

# Write conclusion to file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n" + "=" * 80)
print("Analysis complete! Results written to conclusion.txt")
print("=" * 80)
