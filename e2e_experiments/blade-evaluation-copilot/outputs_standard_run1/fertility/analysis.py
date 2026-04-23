import pandas as pd
import numpy as np
import json
from datetime import datetime
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import statsmodels.api as sm
from imodels import RuleFitRegressor, HSTreeRegressor

# Load data
df = pd.read_csv('fertility.csv')

print("=" * 80)
print("ANALYZING: Effect of hormonal fluctuations on women's religiosity")
print("=" * 80)

# Calculate cycle position (fertility indicator)
# Convert dates to datetime
df['DateTesting'] = pd.to_datetime(df['DateTesting'])
df['StartDateofLastPeriod'] = pd.to_datetime(df['StartDateofLastPeriod'])
df['StartDateofPeriodBeforeLast'] = pd.to_datetime(df['StartDateofPeriodBeforeLast'])

# Calculate days since last period (cycle day)
df['DaysSinceLastPeriod'] = (df['DateTesting'] - df['StartDateofLastPeriod']).dt.days

# Calculate computed cycle length
df['ComputedCycleLength'] = (df['StartDateofLastPeriod'] - df['StartDateofPeriodBeforeLast']).dt.days

# Use reported cycle length, fall back to computed if needed
df['CycleLength'] = df['ReportedCycleLength'].fillna(df['ComputedCycleLength'])

# Calculate cycle position (proportion through cycle)
df['CyclePosition'] = df['DaysSinceLastPeriod'] / df['CycleLength']

# Identify fertility window: fertile window is typically days 10-17 of cycle
# Ovulation occurs ~14 days before next period
df['FertileWindow'] = ((df['DaysSinceLastPeriod'] >= 10) & 
                        (df['DaysSinceLastPeriod'] <= 17)).astype(int)

# Alternative: calculate days until ovulation (assuming ovulation 14 days before end)
df['DaysUntilOvulation'] = df['CycleLength'] - 14 - df['DaysSinceLastPeriod']

# Create religiosity composite score
df['Religiosity'] = (df['Rel1'] + df['Rel2'] + df['Rel3']) / 3

# Filter valid data
valid_mask = (
    (df['DaysSinceLastPeriod'] >= 0) & 
    (df['DaysSinceLastPeriod'] <= 45) &
    (df['CycleLength'] >= 20) & 
    (df['CycleLength'] <= 40) &
    df['Religiosity'].notna()
)
df_clean = df[valid_mask].copy()

print(f"\nSample size: {len(df_clean)} participants")
print(f"Religiosity mean: {df_clean['Religiosity'].mean():.2f} (SD: {df_clean['Religiosity'].std():.2f})")
print(f"Mean cycle day: {df_clean['DaysSinceLastPeriod'].mean():.2f} (SD: {df_clean['DaysSinceLastPeriod'].std():.2f})")
print(f"Fertile window participants: {df_clean['FertileWindow'].sum()} ({100*df_clean['FertileWindow'].mean():.1f}%)")

# Analysis 1: Compare religiosity between fertile and non-fertile windows
print("\n" + "=" * 80)
print("ANALYSIS 1: T-test comparing fertile vs non-fertile windows")
print("=" * 80)
fertile_group = df_clean[df_clean['FertileWindow'] == 1]['Religiosity']
non_fertile_group = df_clean[df_clean['FertileWindow'] == 0]['Religiosity']

t_stat, p_value = stats.ttest_ind(fertile_group, non_fertile_group)
print(f"Fertile window (n={len(fertile_group)}): M={fertile_group.mean():.3f}, SD={fertile_group.std():.3f}")
print(f"Non-fertile (n={len(non_fertile_group)}): M={non_fertile_group.mean():.3f}, SD={non_fertile_group.std():.3f}")
print(f"T-statistic: {t_stat:.3f}")
print(f"P-value: {p_value:.4f}")
cohen_d = (fertile_group.mean() - non_fertile_group.mean()) / np.sqrt((fertile_group.std()**2 + non_fertile_group.std()**2) / 2)
print(f"Cohen's d (effect size): {cohen_d:.3f}")

# Analysis 2: Correlation between cycle position and religiosity
print("\n" + "=" * 80)
print("ANALYSIS 2: Correlation between cycle position and religiosity")
print("=" * 80)
r_cycle, p_cycle = stats.pearsonr(df_clean['CyclePosition'], df_clean['Religiosity'])
print(f"Pearson r: {r_cycle:.3f}, p-value: {p_cycle:.4f}")

r_day, p_day = stats.pearsonr(df_clean['DaysSinceLastPeriod'], df_clean['Religiosity'])
print(f"Days since period vs religiosity: r={r_day:.3f}, p={p_day:.4f}")

# Analysis 3: OLS regression with statsmodels (for p-values)
print("\n" + "=" * 80)
print("ANALYSIS 3: OLS Regression predicting religiosity")
print("=" * 80)

X_reg = df_clean[['DaysSinceLastPeriod', 'CycleLength', 'Relationship', 'Sure1', 'Sure2']].copy()
y_reg = df_clean['Religiosity']

# Add constant
X_reg_sm = sm.add_constant(X_reg)
model_ols = sm.OLS(y_reg, X_reg_sm).fit()
print(model_ols.summary())

# Analysis 4: Interpretable model - Linear regression with coefficients
print("\n" + "=" * 80)
print("ANALYSIS 4: Linear Regression - Feature Coefficients")
print("=" * 80)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reg)
lr = LinearRegression()
lr.fit(X_scaled, y_reg)

feature_names = X_reg.columns
for name, coef in zip(feature_names, lr.coef_):
    print(f"{name:30s}: {coef:7.3f}")

# Analysis 5: Rule-based interpretable model
print("\n" + "=" * 80)
print("ANALYSIS 5: Interpretable Rule-Based Model (HSTree)")
print("=" * 80)

try:
    hstree = HSTreeRegressor(max_depth=3, random_state=42)
    hstree.fit(X_reg, y_reg)
    print("Feature importances:")
    for name, imp in zip(feature_names, hstree.feature_importances_):
        print(f"{name:30s}: {imp:.3f}")
except Exception as e:
    print(f"HSTree failed: {e}")

# Analysis 6: Nonlinear relationship check
print("\n" + "=" * 80)
print("ANALYSIS 6: Quadratic relationship test")
print("=" * 80)

df_clean['CyclePosition_sq'] = df_clean['CyclePosition'] ** 2
X_quad = df_clean[['CyclePosition', 'CyclePosition_sq', 'Relationship']]
X_quad_sm = sm.add_constant(X_quad)
model_quad = sm.OLS(df_clean['Religiosity'], X_quad_sm).fit()
print(f"Cycle position coefficient: {model_quad.params['CyclePosition']:.3f}, p={model_quad.pvalues['CyclePosition']:.4f}")
print(f"Cycle position^2 coefficient: {model_quad.params['CyclePosition_sq']:.3f}, p={model_quad.pvalues['CyclePosition_sq']:.4f}")

# Summary and conclusion
print("\n" + "=" * 80)
print("SUMMARY OF FINDINGS")
print("=" * 80)

significant_findings = []
effect_direction = None

# Check each analysis for significance
if p_value < 0.05:
    significant_findings.append(f"Fertile window difference (p={p_value:.4f})")
    if fertile_group.mean() > non_fertile_group.mean():
        effect_direction = "higher religiosity during fertile window"
    else:
        effect_direction = "lower religiosity during fertile window"

if p_cycle < 0.05:
    significant_findings.append(f"Cycle position correlation (p={p_cycle:.4f})")
    if r_cycle > 0:
        effect_direction = effect_direction or "positive correlation with cycle position"
    else:
        effect_direction = effect_direction or "negative correlation with cycle position"

if p_day < 0.05:
    significant_findings.append(f"Days since period correlation (p={p_day:.4f})")

if model_ols.pvalues['DaysSinceLastPeriod'] < 0.05:
    significant_findings.append(f"Days since period in regression (p={model_ols.pvalues['DaysSinceLastPeriod']:.4f})")

print(f"Significant findings ({len(significant_findings)}):")
for finding in significant_findings:
    print(f"  - {finding}")

# Determine conclusion
if len(significant_findings) > 0:
    # At least some evidence of effect
    # Strength depends on p-values and consistency
    min_p = min(p_value, p_cycle, p_day, model_ols.pvalues['DaysSinceLastPeriod'])
    
    if min_p < 0.01 and abs(cohen_d) > 0.3:
        # Strong evidence
        response_score = 75
        explanation = f"Strong evidence of hormonal effects on religiosity. {effect_direction}. Smallest p-value: {min_p:.4f}, Cohen's d: {cohen_d:.3f}"
    elif min_p < 0.05:
        # Moderate evidence
        response_score = 60
        explanation = f"Moderate evidence of hormonal effects on religiosity. {effect_direction}. Smallest p-value: {min_p:.4f}"
    else:
        # Weak evidence
        response_score = 45
        explanation = f"Weak evidence of hormonal effects on religiosity. Effect direction unclear."
else:
    # No significant findings
    response_score = 25
    explanation = f"No statistically significant relationship found. T-test p={p_value:.3f}, correlation p={p_cycle:.3f}. Effect size is negligible (d={cohen_d:.3f})."

print("\n" + "=" * 80)
print(f"CONCLUSION SCORE: {response_score}/100")
print(f"EXPLANATION: {explanation}")
print("=" * 80)

# Write conclusion
conclusion = {
    "response": response_score,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n✓ conclusion.txt written successfully")
