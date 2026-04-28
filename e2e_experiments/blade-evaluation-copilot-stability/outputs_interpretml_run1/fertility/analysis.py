import pandas as pd
import numpy as np
import json
from datetime import datetime
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
import statsmodels.api as sm
from interpret.glassbox import ExplainableBoostingRegressor

# Load data
df = pd.read_csv('fertility.csv')

print("=" * 80)
print("ANALYZING: Effect of hormonal fluctuations on women's religiosity")
print("=" * 80)
print(f"\nDataset shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())

# Create religiosity composite score
df['religiosity'] = df[['Rel1', 'Rel2', 'Rel3']].mean(axis=1)
print(f"\nReligiosity score created (mean of Rel1, Rel2, Rel3)")
print(f"Religiosity stats: mean={df['religiosity'].mean():.2f}, std={df['religiosity'].std():.2f}")

# Parse dates and calculate cycle phase
df['DateTesting'] = pd.to_datetime(df['DateTesting'], format='%m/%d/%y')
df['StartDateofLastPeriod'] = pd.to_datetime(df['StartDateofLastPeriod'], format='%m/%d/%y')
df['StartDateofPeriodBeforeLast'] = pd.to_datetime(df['StartDateofPeriodBeforeLast'], format='%m/%d/%y')

# Calculate days since last period (proxy for cycle phase)
df['days_since_period'] = (df['DateTesting'] - df['StartDateofLastPeriod']).dt.days

# Calculate actual cycle length from dates
df['actual_cycle_length'] = (df['StartDateofLastPeriod'] - df['StartDateofPeriodBeforeLast']).dt.days

# Calculate cycle phase percentage (0-100%)
df['cycle_phase_pct'] = (df['days_since_period'] / df['actual_cycle_length']) * 100

# Estimate fertility phase (ovulation typically occurs around day 14, or mid-cycle)
# High fertility: days 10-18 of cycle (or around 33-60% through cycle for varying lengths)
df['estimated_fertile_window'] = ((df['days_since_period'] >= 10) & (df['days_since_period'] <= 18)).astype(int)

print(f"\n--- Cycle Phase Analysis ---")
print(f"Days since period: mean={df['days_since_period'].mean():.1f}, std={df['days_since_period'].std():.1f}")
print(f"Actual cycle length: mean={df['actual_cycle_length'].mean():.1f}, std={df['actual_cycle_length'].std():.1f}")
print(f"Cycle phase percentage: mean={df['cycle_phase_pct'].mean():.1f}%, std={df['cycle_phase_pct'].std():.1f}%")
print(f"Women in fertile window (days 10-18): {df['estimated_fertile_window'].sum()} ({df['estimated_fertile_window'].mean()*100:.1f}%)")

# Clean data - remove rows with missing values for key variables
df_clean = df.dropna(subset=['religiosity', 'days_since_period', 'actual_cycle_length', 
                               'cycle_phase_pct', 'Relationship', 'Sure1', 'Sure2'])
print(f"\nClean dataset shape: {df_clean.shape}")

# ============================================
# ANALYSIS 1: Correlation Analysis
# ============================================
print("\n" + "=" * 80)
print("ANALYSIS 1: CORRELATION ANALYSIS")
print("=" * 80)

# Correlation between religiosity and cycle variables
corr_vars = ['days_since_period', 'cycle_phase_pct', 'actual_cycle_length', 'estimated_fertile_window']
for var in corr_vars:
    r, p = stats.pearsonr(df_clean['religiosity'], df_clean[var])
    print(f"Religiosity vs {var}: r={r:.3f}, p={p:.4f}")

# ============================================
# ANALYSIS 2: T-test for Fertile Window
# ============================================
print("\n" + "=" * 80)
print("ANALYSIS 2: T-TEST - FERTILE WINDOW vs NON-FERTILE")
print("=" * 80)

fertile = df_clean[df_clean['estimated_fertile_window'] == 1]['religiosity']
non_fertile = df_clean[df_clean['estimated_fertile_window'] == 0]['religiosity']

print(f"Religiosity in fertile window (n={len(fertile)}): mean={fertile.mean():.2f}, std={fertile.std():.2f}")
print(f"Religiosity outside fertile window (n={len(non_fertile)}): mean={non_fertile.mean():.2f}, std={non_fertile.std():.2f}")

t_stat, t_pval = stats.ttest_ind(fertile, non_fertile)
print(f"T-test: t={t_stat:.3f}, p={t_pval:.4f}")

# Effect size (Cohen's d)
cohens_d = (fertile.mean() - non_fertile.mean()) / np.sqrt(((len(fertile)-1)*fertile.std()**2 + (len(non_fertile)-1)*non_fertile.std()**2) / (len(fertile) + len(non_fertile) - 2))
print(f"Cohen's d effect size: {cohens_d:.3f}")

# ============================================
# ANALYSIS 3: Linear Regression with statsmodels
# ============================================
print("\n" + "=" * 80)
print("ANALYSIS 3: LINEAR REGRESSION (statsmodels OLS)")
print("=" * 80)

# Model 1: Simple regression with days since period
X1 = sm.add_constant(df_clean['days_since_period'])
model1 = sm.OLS(df_clean['religiosity'], X1).fit()
print("\nModel 1: Religiosity ~ days_since_period")
print(model1.summary())

# Model 2: Regression with cycle phase percentage
X2 = sm.add_constant(df_clean['cycle_phase_pct'])
model2 = sm.OLS(df_clean['religiosity'], X2).fit()
print("\nModel 2: Religiosity ~ cycle_phase_pct")
print(model2.summary())

# Model 3: Multiple regression with controls
X3 = df_clean[['days_since_period', 'Relationship', 'Sure1', 'Sure2', 'actual_cycle_length']]
X3 = sm.add_constant(X3)
model3 = sm.OLS(df_clean['religiosity'], X3).fit()
print("\nModel 3: Religiosity ~ days_since_period + controls")
print(model3.summary())

# ============================================
# ANALYSIS 4: Interpretable Model (EBM)
# ============================================
print("\n" + "=" * 80)
print("ANALYSIS 4: EXPLAINABLE BOOSTING MACHINE (EBM)")
print("=" * 80)

# Prepare features
feature_cols = ['days_since_period', 'cycle_phase_pct', 'Relationship', 'Sure1', 'Sure2', 'actual_cycle_length']
X_ebm = df_clean[feature_cols]
y_ebm = df_clean['religiosity']

# Fit EBM
ebm = ExplainableBoostingRegressor(random_state=42, interactions=0)
ebm.fit(X_ebm, y_ebm)

# Feature importance
print("\nEBM Feature Importance (absolute):")
for i, col in enumerate(feature_cols):
    importance = np.abs(ebm.term_scores_[i]).mean()
    print(f"  {col}: {importance:.4f}")

# ============================================
# ANALYSIS 5: Quadratic relationship (cycle follows a curve)
# ============================================
print("\n" + "=" * 80)
print("ANALYSIS 5: QUADRATIC REGRESSION")
print("=" * 80)

# Test if there's a U-shaped or inverted U-shaped relationship
df_clean['cycle_phase_squared'] = df_clean['cycle_phase_pct'] ** 2
X_quad = df_clean[['cycle_phase_pct', 'cycle_phase_squared']]
X_quad = sm.add_constant(X_quad)
model_quad = sm.OLS(df_clean['religiosity'], X_quad).fit()
print("\nQuadratic Model: Religiosity ~ cycle_phase_pct + cycle_phase_pct²")
print(model_quad.summary())

# ============================================
# CONCLUSION
# ============================================
print("\n" + "=" * 80)
print("SYNTHESIS AND CONCLUSION")
print("=" * 80)

# Gather evidence
evidence = {
    'correlations': {},
    't_test_pval': t_pval,
    't_test_effect_size': cohens_d,
    'ols_pvals': {
        'days_since_period': model1.pvalues['days_since_period'],
        'cycle_phase_pct': model2.pvalues['cycle_phase_pct'],
        'days_controlled': model3.pvalues['days_since_period']
    },
    'r_squared': {
        'model1': model1.rsquared,
        'model2': model2.rsquared,
        'model3': model3.rsquared
    }
}

for var in corr_vars:
    r, p = stats.pearsonr(df_clean['religiosity'], df_clean[var])
    evidence['correlations'][var] = {'r': r, 'p': p}

print("\nKey Statistical Results:")
print(f"1. T-test (fertile vs non-fertile): p={t_pval:.4f}, Cohen's d={cohens_d:.3f}")
print(f"2. OLS days_since_period: p={evidence['ols_pvals']['days_since_period']:.4f}")
print(f"3. OLS cycle_phase_pct: p={evidence['ols_pvals']['cycle_phase_pct']:.4f}")
print(f"4. OLS with controls: p={evidence['ols_pvals']['days_controlled']:.4f}")
print(f"5. Best R² achieved: {max(evidence['r_squared'].values()):.4f}")

# Decision logic: if p < 0.05 for any key test, there's evidence of a relationship
significant_results = [
    t_pval < 0.05,
    evidence['ols_pvals']['days_since_period'] < 0.05,
    evidence['ols_pvals']['cycle_phase_pct'] < 0.05,
    evidence['ols_pvals']['days_controlled'] < 0.05
]

any_significant = any(significant_results)
num_significant = sum(significant_results)

if any_significant:
    if num_significant >= 3:
        response_score = 75  # Strong evidence
        explanation = f"Strong evidence of relationship: {num_significant}/4 tests significant (p<0.05). "
    elif num_significant >= 2:
        response_score = 65  # Moderate-strong evidence
        explanation = f"Moderate evidence of relationship: {num_significant}/4 tests significant (p<0.05). "
    else:
        response_score = 55  # Weak-moderate evidence
        explanation = f"Weak evidence of relationship: {num_significant}/4 tests significant (p<0.05). "
    
    # Add details about the direction and magnitude
    if cohens_d != 0:
        direction = "higher" if cohens_d > 0 else "lower"
        explanation += f"Women in fertile window showed {direction} religiosity (d={cohens_d:.3f}). "
    
    explanation += f"However, effect sizes were small (best R²={max(evidence['r_squared'].values()):.3f}), "
    explanation += "suggesting hormonal fluctuations have limited practical impact on religiosity."
else:
    response_score = 20  # No evidence
    explanation = f"No significant relationship found: all p-values > 0.05. "
    explanation += f"T-test p={t_pval:.3f}, OLS p-values ranged {min(evidence['ols_pvals'].values()):.3f}-{max(evidence['ols_pvals'].values()):.3f}. "
    explanation += "The data does not support the hypothesis that hormonal fluctuations affect women's religiosity."

print(f"\nFinal Assessment:")
print(f"Response Score: {response_score}/100")
print(f"Explanation: {explanation}")

# Write conclusion to file
conclusion = {
    "response": response_score,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n" + "=" * 80)
print("Analysis complete. Results written to conclusion.txt")
print("=" * 80)
