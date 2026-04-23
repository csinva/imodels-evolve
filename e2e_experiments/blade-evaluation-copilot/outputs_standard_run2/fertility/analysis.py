import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from datetime import datetime
import json

# Load the dataset
df = pd.read_csv('fertility.csv')

print("=" * 80)
print("FERTILITY AND RELIGIOSITY ANALYSIS")
print("=" * 80)
print(f"\nDataset shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())

# Data exploration
print("\n" + "=" * 80)
print("DATA EXPLORATION")
print("=" * 80)
print("\nSummary statistics:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

# Calculate cycle day (fertility phase indicator)
# Convert dates to datetime
df['DateTesting'] = pd.to_datetime(df['DateTesting'])
df['StartDateofLastPeriod'] = pd.to_datetime(df['StartDateofLastPeriod'])
df['StartDateofPeriodBeforeLast'] = pd.to_datetime(df['StartDateofPeriodBeforeLast'])

# Calculate days since last period started (cycle day)
df['DaysSinceLastPeriod'] = (df['DateTesting'] - df['StartDateofLastPeriod']).dt.days

# Calculate actual cycle length from previous two periods
df['ActualPreviousCycleLength'] = (df['StartDateofLastPeriod'] - df['StartDateofPeriodBeforeLast']).dt.days

print("\n" + "=" * 80)
print("FERTILITY CYCLE PHASE CALCULATION")
print("=" * 80)
print(f"\nDays since last period - Mean: {df['DaysSinceLastPeriod'].mean():.2f}, Std: {df['DaysSinceLastPeriod'].std():.2f}")
print(f"Range: {df['DaysSinceLastPeriod'].min()} to {df['DaysSinceLastPeriod'].max()} days")

# Create fertility phase indicators
# Follicular phase: days 1-14 (pre-ovulation, low fertility)
# Ovulatory phase: days 10-17 (peak fertility)
# Luteal phase: days 15+ (post-ovulation, low fertility)
df['FertileWindow'] = ((df['DaysSinceLastPeriod'] >= 10) & (df['DaysSinceLastPeriod'] <= 17)).astype(int)
df['FollicularPhase'] = (df['DaysSinceLastPeriod'] <= 14).astype(int)
df['LutealPhase'] = (df['DaysSinceLastPeriod'] >= 15).astype(int)

print(f"\nFertile window (days 10-17): {df['FertileWindow'].sum()} participants")
print(f"Follicular phase (days 1-14): {df['FollicularPhase'].sum()} participants")
print(f"Luteal phase (days 15+): {df['LutealPhase'].sum()} participants")

# Create composite religiosity score (average of Rel1, Rel2, Rel3)
df['Religiosity'] = df[['Rel1', 'Rel2', 'Rel3']].mean(axis=1)

print("\n" + "=" * 80)
print("RELIGIOSITY MEASURES")
print("=" * 80)
print(f"\nReligiosity composite score - Mean: {df['Religiosity'].mean():.2f}, Std: {df['Religiosity'].std():.2f}")
print(f"\nCorrelations between religiosity items:")
print(df[['Rel1', 'Rel2', 'Rel3']].corr())

# Drop rows with missing religiosity data
df_analysis = df.dropna(subset=['Religiosity', 'DaysSinceLastPeriod'])
print(f"\nSample size after removing missing data: {len(df_analysis)}")

print("\n" + "=" * 80)
print("STATISTICAL ANALYSIS: FERTILITY PHASE AND RELIGIOSITY")
print("=" * 80)

# Analysis 1: Compare religiosity between fertile window and non-fertile periods
fertile_religiosity = df_analysis[df_analysis['FertileWindow'] == 1]['Religiosity']
non_fertile_religiosity = df_analysis[df_analysis['FertileWindow'] == 0]['Religiosity']

print(f"\nReligiosity during fertile window (n={len(fertile_religiosity)}): {fertile_religiosity.mean():.3f} ± {fertile_religiosity.std():.3f}")
print(f"Religiosity outside fertile window (n={len(non_fertile_religiosity)}): {non_fertile_religiosity.mean():.3f} ± {non_fertile_religiosity.std():.3f}")

t_stat, p_value_ttest = stats.ttest_ind(fertile_religiosity, non_fertile_religiosity)
print(f"\nT-test: t={t_stat:.3f}, p={p_value_ttest:.4f}")

# Analysis 2: Compare follicular vs luteal phase
follicular_religiosity = df_analysis[df_analysis['FollicularPhase'] == 1]['Religiosity']
luteal_religiosity = df_analysis[df_analysis['LutealPhase'] == 1]['Religiosity']

print(f"\nReligiosity during follicular phase (n={len(follicular_religiosity)}): {follicular_religiosity.mean():.3f} ± {follicular_religiosity.std():.3f}")
print(f"Religiosity during luteal phase (n={len(luteal_religiosity)}): {luteal_religiosity.mean():.3f} ± {luteal_religiosity.std():.3f}")

t_stat2, p_value_phase = stats.ttest_ind(follicular_religiosity, luteal_religiosity)
print(f"\nT-test (follicular vs luteal): t={t_stat2:.3f}, p={p_value_phase:.4f}")

# Analysis 3: Correlation between cycle day and religiosity
correlation, p_value_corr = stats.pearsonr(df_analysis['DaysSinceLastPeriod'], df_analysis['Religiosity'])
print(f"\nPearson correlation (cycle day vs religiosity): r={correlation:.3f}, p={p_value_corr:.4f}")

# Analysis 4: Linear regression with cycle day predicting religiosity
print("\n" + "=" * 80)
print("LINEAR REGRESSION: Cycle Day → Religiosity")
print("=" * 80)

X = df_analysis[['DaysSinceLastPeriod']].values
y = df_analysis['Religiosity'].values

X_with_const = sm.add_constant(X)
model_ols = sm.OLS(y, X_with_const).fit()
print(model_ols.summary())

# Analysis 5: Multiple regression controlling for relationship status
print("\n" + "=" * 80)
print("MULTIPLE REGRESSION: Controlling for Relationship Status")
print("=" * 80)

df_analysis_complete = df_analysis.dropna(subset=['Relationship'])
X_multi = df_analysis_complete[['DaysSinceLastPeriod', 'Relationship']].values
y_multi = df_analysis_complete['Religiosity'].values

X_multi_const = sm.add_constant(X_multi)
model_multi = sm.OLS(y_multi, X_multi_const).fit()
print(model_multi.summary())

# Analysis 6: Decision tree for interpretability
print("\n" + "=" * 80)
print("DECISION TREE ANALYSIS")
print("=" * 80)

from sklearn.tree import DecisionTreeRegressor
tree_model = DecisionTreeRegressor(max_depth=3, min_samples_leaf=20, random_state=42)
tree_model.fit(df_analysis[['DaysSinceLastPeriod']].values, df_analysis['Religiosity'].values)

print(f"\nDecision tree feature importance (DaysSinceLastPeriod): {tree_model.feature_importances_[0]:.4f}")
print(f"Tree score (R²): {tree_model.score(df_analysis[['DaysSinceLastPeriod']].values, df_analysis['Religiosity'].values):.4f}")

# Analysis 7: Test for each religiosity item separately
print("\n" + "=" * 80)
print("INDIVIDUAL RELIGIOSITY ITEMS")
print("=" * 80)

for rel_item in ['Rel1', 'Rel2', 'Rel3']:
    df_item = df_analysis.dropna(subset=[rel_item])
    corr_item, p_item = stats.pearsonr(df_item['DaysSinceLastPeriod'], df_item[rel_item])
    print(f"\n{rel_item}: correlation with cycle day: r={corr_item:.3f}, p={p_item:.4f}")

# CONCLUSION
print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

# Determine response based on statistical significance
significance_threshold = 0.05
significant_results = []

if p_value_ttest < significance_threshold:
    significant_results.append(f"T-test (fertile vs non-fertile): p={p_value_ttest:.4f}")
    
if p_value_phase < significance_threshold:
    significant_results.append(f"T-test (follicular vs luteal): p={p_value_phase:.4f}")
    
if p_value_corr < significance_threshold:
    significant_results.append(f"Correlation (cycle day vs religiosity): p={p_value_corr:.4f}")

# Check regression p-value for cycle day coefficient
cycle_day_pvalue = model_ols.pvalues[1]
if cycle_day_pvalue < significance_threshold:
    significant_results.append(f"Regression coefficient: p={cycle_day_pvalue:.4f}")

print(f"\nNumber of significant findings (p < {significance_threshold}): {len(significant_results)}")
if significant_results:
    for result in significant_results:
        print(f"  - {result}")

# Calculate response score (0-100)
# If we have multiple significant results, higher confidence
# Also consider effect size
if len(significant_results) >= 2:
    # Strong evidence
    if abs(correlation) > 0.15:
        response = 75  # Strong Yes - multiple significant tests and meaningful effect size
    else:
        response = 60  # Moderate Yes - multiple significant tests but small effect
elif len(significant_results) == 1:
    # Some evidence
    if abs(correlation) > 0.10:
        response = 55  # Lean Yes - one significant test with some effect
    else:
        response = 45  # Borderline - one significant test but very small effect
else:
    # No significant evidence
    if abs(correlation) < 0.05:
        response = 15  # Strong No - no significance and no relationship
    else:
        response = 30  # Lean No - no significance but small trend

explanation = f"The analysis examined whether hormonal fluctuations associated with fertility affect women's religiosity. "

if len(significant_results) > 0:
    explanation += f"We found {len(significant_results)} significant statistical relationship(s) (p < {significance_threshold}): "
    explanation += "; ".join(significant_results) + ". "
    explanation += f"The correlation between cycle day and religiosity was r={correlation:.3f}. "
    
    if response >= 70:
        explanation += "The multiple significant findings and meaningful effect size provide strong evidence of an effect."
    elif response >= 55:
        explanation += "The significant findings suggest a relationship exists, though the effect size is modest."
    else:
        explanation += "While statistically significant, the effect size is quite small."
else:
    explanation += f"We found no statistically significant relationships (all p-values > {significance_threshold}). "
    explanation += f"The correlation between cycle day and religiosity was only r={correlation:.3f} (p={p_value_corr:.3f}). "
    explanation += "The data does not support the hypothesis that fertility-related hormonal fluctuations affect religiosity."

print(f"\nFinal Assessment: {response}/100")
print(f"Explanation: {explanation}")

# Write conclusion to file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n" + "=" * 80)
print("Analysis complete. Conclusion written to conclusion.txt")
print("=" * 80)
