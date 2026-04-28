import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import json
from datetime import datetime

# Load the dataset
df = pd.read_csv('fertility.csv')

print("=" * 80)
print("RESEARCH QUESTION:")
print("What is the effect of hormonal fluctuations associated with fertility on women's religiosity?")
print("=" * 80)

# Data exploration
print("\nDATASET SHAPE:", df.shape)
print("\nCOLUMNS:", df.columns.tolist())
print("\nFIRST FEW ROWS:")
print(df.head())

# Convert dates to datetime
df['DateTesting'] = pd.to_datetime(df['DateTesting'], format='%m/%d/%y')
df['StartDateofLastPeriod'] = pd.to_datetime(df['StartDateofLastPeriod'], format='%m/%d/%y')
df['StartDateofPeriodBeforeLast'] = pd.to_datetime(df['StartDateofPeriodBeforeLast'], format='%m/%d/%y')

# Calculate cycle position - days since last period (proxy for fertility cycle phase)
df['DaysSinceLastPeriod'] = (df['DateTesting'] - df['StartDateofLastPeriod']).dt.days

# Calculate actual cycle length from the data
df['ActualCycleLength'] = (df['StartDateofLastPeriod'] - df['StartDateofPeriodBeforeLast']).dt.days

# Create fertility-related variables
# Normalized cycle position (0-1 scale where 0 is start, 1 is end of cycle)
df['CyclePosition'] = df['DaysSinceLastPeriod'] / df['ReportedCycleLength']

# Estimate fertility phase (ovulation typically occurs around day 14, mid-cycle)
# High fertility window: days 10-18 of typical 28-day cycle
# For variable cycle lengths, estimate as 0.35-0.65 of cycle
df['HighFertilityPhase'] = ((df['CyclePosition'] >= 0.35) & (df['CyclePosition'] <= 0.65)).astype(int)

# Create composite religiosity score (average of 3 religiosity items)
df['Religiosity'] = df[['Rel1', 'Rel2', 'Rel3']].mean(axis=1)

print("\n" + "=" * 80)
print("FERTILITY CYCLE VARIABLES:")
print("=" * 80)
print(f"\nDays Since Last Period - Mean: {df['DaysSinceLastPeriod'].mean():.2f}, Std: {df['DaysSinceLastPeriod'].std():.2f}")
print(f"Reported Cycle Length - Mean: {df['ReportedCycleLength'].mean():.2f}, Std: {df['ReportedCycleLength'].std():.2f}")
print(f"Cycle Position (0-1) - Mean: {df['CyclePosition'].mean():.2f}, Std: {df['CyclePosition'].std():.2f}")
print(f"High Fertility Phase: {df['HighFertilityPhase'].sum()} women ({df['HighFertilityPhase'].mean()*100:.1f}%)")

print("\n" + "=" * 80)
print("RELIGIOSITY SUMMARY:")
print("=" * 80)
print(df[['Rel1', 'Rel2', 'Rel3', 'Religiosity']].describe())

# Clean data - remove rows with missing values in key variables
analysis_df = df[['Religiosity', 'DaysSinceLastPeriod', 'CyclePosition', 'HighFertilityPhase', 
                   'ReportedCycleLength', 'Relationship', 'Sure1', 'Sure2']].dropna()

print(f"\nAnalysis sample size: {len(analysis_df)} (removed {len(df) - len(analysis_df)} rows with missing data)")

# ============================================================================
# STATISTICAL ANALYSIS 1: Compare religiosity between fertility phases
# ============================================================================
print("\n" + "=" * 80)
print("ANALYSIS 1: T-TEST - High vs Low Fertility Phase")
print("=" * 80)

high_fert = analysis_df[analysis_df['HighFertilityPhase'] == 1]['Religiosity']
low_fert = analysis_df[analysis_df['HighFertilityPhase'] == 0]['Religiosity']

print(f"\nHigh Fertility Phase (n={len(high_fert)}): Mean={high_fert.mean():.3f}, Std={high_fert.std():.3f}")
print(f"Low Fertility Phase (n={len(low_fert)}): Mean={low_fert.mean():.3f}, Std={low_fert.std():.3f}")
print(f"Difference: {high_fert.mean() - low_fert.mean():.3f}")

t_stat, p_value = stats.ttest_ind(high_fert, low_fert)
print(f"\nt-test: t={t_stat:.3f}, p={p_value:.4f}")

# ============================================================================
# ANALYSIS 2: Correlation between cycle position and religiosity
# ============================================================================
print("\n" + "=" * 80)
print("ANALYSIS 2: CORRELATION - Cycle Position vs Religiosity")
print("=" * 80)

corr_pearson, p_pearson = stats.pearsonr(analysis_df['CyclePosition'], analysis_df['Religiosity'])
corr_spearman, p_spearman = stats.spearmanr(analysis_df['CyclePosition'], analysis_df['Religiosity'])

print(f"\nPearson correlation: r={corr_pearson:.3f}, p={p_pearson:.4f}")
print(f"Spearman correlation: rho={corr_spearman:.3f}, p={p_spearman:.4f}")

corr_days, p_days = stats.pearsonr(analysis_df['DaysSinceLastPeriod'], analysis_df['Religiosity'])
print(f"\nDays Since Last Period correlation: r={corr_days:.3f}, p={p_days:.4f}")

# ============================================================================
# ANALYSIS 3: Linear Regression with statsmodels (for p-values)
# ============================================================================
print("\n" + "=" * 80)
print("ANALYSIS 3: REGRESSION MODEL - Fertility Cycle Effects on Religiosity")
print("=" * 80)

# Model 1: Simple model with cycle position
X1 = sm.add_constant(analysis_df[['CyclePosition']])
y = analysis_df['Religiosity']
model1 = sm.OLS(y, X1).fit()

print("\nModel 1: Religiosity ~ CyclePosition")
print(model1.summary())

# Model 2: Extended model with controls
X2 = sm.add_constant(analysis_df[['CyclePosition', 'HighFertilityPhase', 'Relationship']])
model2 = sm.OLS(y, X2).fit()

print("\nModel 2: Religiosity ~ CyclePosition + HighFertilityPhase + Relationship")
print(model2.summary())

# Model 3: Non-linear effects (quadratic cycle position)
analysis_df['CyclePosition_sq'] = analysis_df['CyclePosition'] ** 2
X3 = sm.add_constant(analysis_df[['CyclePosition', 'CyclePosition_sq', 'Relationship']])
model3 = sm.OLS(y, X3).fit()

print("\nModel 3: Religiosity ~ CyclePosition + CyclePosition^2 + Relationship")
print(model3.summary())

# ============================================================================
# ANALYSIS 4: Interpretable model using sklearn
# ============================================================================
print("\n" + "=" * 80)
print("ANALYSIS 4: INTERPRETABLE LINEAR MODEL (Ridge Regression)")
print("=" * 80)

# Prepare features
features = ['CyclePosition', 'DaysSinceLastPeriod', 'HighFertilityPhase', 
            'ReportedCycleLength', 'Relationship']
X = analysis_df[features]
y = analysis_df['Religiosity']

# Standardize for better interpretation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit Ridge regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_scaled, y)

print("\nRidge Regression Coefficients (standardized):")
for feat, coef in zip(features, ridge.coef_):
    print(f"  {feat:25s}: {coef:7.4f}")
print(f"  {'Intercept':25s}: {ridge.intercept_:7.4f}")

# ============================================================================
# SYNTHESIS AND CONCLUSION
# ============================================================================
print("\n" + "=" * 80)
print("SYNTHESIS OF RESULTS")
print("=" * 80)

# Collect evidence
evidence = []

# Evidence 1: T-test
if p_value < 0.05:
    evidence.append(f"T-test shows {'significant' if p_value < 0.05 else 'no significant'} difference (p={p_value:.4f})")
    sig_ttest = True
else:
    evidence.append(f"T-test shows no significant difference between fertility phases (p={p_value:.4f})")
    sig_ttest = False

# Evidence 2: Correlations
if p_pearson < 0.05 or p_days < 0.05:
    evidence.append(f"Correlation analysis shows {'weak but significant' if abs(corr_pearson) < 0.2 else 'significant'} relationship (p_pearson={p_pearson:.4f}, p_days={p_days:.4f})")
    sig_corr = True
else:
    evidence.append(f"Correlation analysis shows no significant relationship (p_pearson={p_pearson:.4f}, p_days={p_days:.4f})")
    sig_corr = False

# Evidence 3: Regression coefficients
cycle_p = model2.pvalues['CyclePosition']
fert_p = model2.pvalues['HighFertilityPhase']
if cycle_p < 0.05 or fert_p < 0.05:
    evidence.append(f"Regression model shows significant fertility effects (CyclePosition p={cycle_p:.4f}, HighFertilityPhase p={fert_p:.4f})")
    sig_reg = True
else:
    evidence.append(f"Regression model shows no significant fertility effects (CyclePosition p={cycle_p:.4f}, HighFertilityPhase p={fert_p:.4f})")
    sig_reg = False

# Evidence 4: Effect size
effect_size = abs(high_fert.mean() - low_fert.mean())
cohens_d = (high_fert.mean() - low_fert.mean()) / np.sqrt((high_fert.std()**2 + low_fert.std()**2) / 2)
evidence.append(f"Effect size: difference={effect_size:.3f}, Cohen's d={cohens_d:.3f}")

print("\nKEY FINDINGS:")
for i, e in enumerate(evidence, 1):
    print(f"{i}. {e}")

# Determine final response
# Count significant findings
sig_count = sum([sig_ttest, sig_corr, sig_reg])

print("\n" + "=" * 80)
print("FINAL CONCLUSION")
print("=" * 80)

if sig_count == 0:
    # No significant effects found
    response = 15
    explanation = (
        "No significant effect of hormonal fluctuations on religiosity was found. "
        f"T-test (p={p_value:.3f}), correlation (p={p_pearson:.3f}), and regression "
        f"(cycle position p={cycle_p:.3f}, fertility phase p={fert_p:.3f}) all failed to show "
        "significant relationships. The data does not support the hypothesis that fertility "
        "cycle affects women's religiosity."
    )
elif sig_count == 1 and not sig_ttest:
    # Weak evidence - only 1 test significant and it's not the main comparison
    response = 35
    explanation = (
        "Weak evidence for an effect. Only one statistical test showed significance, "
        f"but the primary comparison (t-test, p={p_value:.3f}) was not significant. "
        "The evidence is mixed and insufficient to conclude a clear relationship between "
        "fertility cycle and religiosity."
    )
elif sig_count == 1 and sig_ttest:
    # Moderate evidence - t-test significant
    response = 55
    explanation = (
        f"Moderate evidence for an effect. The t-test showed a significant difference (p={p_value:.3f}) "
        f"between high and low fertility phases, but other analyses (correlation p={p_pearson:.3f}, "
        f"regression p={cycle_p:.3f}) did not confirm this. The effect size is small (Cohen's d={cohens_d:.2f}). "
        "There is some evidence of a relationship but it is not robust across all analyses."
    )
elif sig_count == 2:
    # Good evidence
    response = 70
    explanation = (
        f"Good evidence for an effect. Multiple statistical tests showed significant relationships: "
        f"t-test p={p_value:.3f}, correlation p={p_pearson:.3f}, regression p={cycle_p:.3f}. "
        f"The effect size is {'small' if abs(cohens_d) < 0.3 else 'moderate'} (Cohen's d={cohens_d:.2f}). "
        "The data suggests that hormonal fluctuations associated with fertility do affect women's religiosity, "
        "though the effect magnitude is modest."
    )
else:
    # Strong evidence - all tests significant
    response = 85
    explanation = (
        f"Strong evidence for an effect. All statistical analyses showed significant relationships: "
        f"t-test p={p_value:.3f}, correlation p={p_pearson:.3f}, regression analyses p<0.05. "
        f"Effect size (Cohen's d={cohens_d:.2f}) is {'small but consistent' if abs(cohens_d) < 0.3 else 'meaningful'}. "
        "The data strongly supports that hormonal fluctuations associated with fertility affect women's religiosity."
    )

print(f"\nResponse Score: {response}/100")
print(f"Explanation: {explanation}")

# Write conclusion to file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n" + "=" * 80)
print("Analysis complete. Results written to conclusion.txt")
print("=" * 80)
