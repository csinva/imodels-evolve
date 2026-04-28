import pandas as pd
import numpy as np
import json
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from interpret.glassbox import ExplainableBoostingRegressor
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('fertility.csv')

# Convert date columns to datetime
df['DateTesting'] = pd.to_datetime(df['DateTesting'], format='%m/%d/%y')
df['StartDateofLastPeriod'] = pd.to_datetime(df['StartDateofLastPeriod'], format='%m/%d/%y')
df['StartDateofPeriodBeforeLast'] = pd.to_datetime(df['StartDateofPeriodBeforeLast'], format='%m/%d/%y')

# Calculate cycle phase - days since last period started
df['DaysSinceLastPeriod'] = (df['DateTesting'] - df['StartDateofLastPeriod']).dt.days

# Calculate estimated cycle day (as a proportion of cycle)
df['CycleDay'] = df['DaysSinceLastPeriod']
df['CycleProportion'] = df['CycleDay'] / df['ReportedCycleLength']

# Create fertility phase indicators
# Follicular phase: days 1-14 (before ovulation)
# Ovulatory phase: days 12-16 (peak fertility)
# Luteal phase: days 15-28 (after ovulation)
df['FertileWindow'] = ((df['CycleDay'] >= 10) & (df['CycleDay'] <= 18)).astype(int)
df['LutealPhase'] = (df['CycleDay'] > 18).astype(int)
df['FollicularPhase'] = (df['CycleDay'] <= 10).astype(int)

# Create composite religiosity score
df['Religiosity'] = (df['Rel1'] + df['Rel2'] + df['Rel3']) / 3

# Filter valid data (reasonable cycle days and no missing religiosity)
df_clean = df[(df['CycleDay'] >= 0) & (df['CycleDay'] <= 45) & 
              (df['ReportedCycleLength'] >= 20) & 
              (df['Religiosity'].notna())].copy()

print("="*80)
print("DATA EXPLORATION")
print("="*80)
print(f"\nDataset shape: {df_clean.shape}")
print(f"\nReligiosity stats:\n{df_clean['Religiosity'].describe()}")
print(f"\nCycle day stats:\n{df_clean['CycleDay'].describe()}")
print(f"\nCycle length stats:\n{df_clean['ReportedCycleLength'].describe()}")

print("\n" + "="*80)
print("CORRELATION ANALYSIS")
print("="*80)
# Correlation between cycle variables and religiosity
corr_cycle_day, p_cycle_day = stats.pearsonr(df_clean['CycleDay'], df_clean['Religiosity'])
print(f"\nCycle Day vs Religiosity:")
print(f"  Correlation: {corr_cycle_day:.4f}, p-value: {p_cycle_day:.4f}")

corr_cycle_prop, p_cycle_prop = stats.pearsonr(df_clean['CycleProportion'], df_clean['Religiosity'])
print(f"\nCycle Proportion vs Religiosity:")
print(f"  Correlation: {corr_cycle_prop:.4f}, p-value: {p_cycle_prop:.4f}")

print("\n" + "="*80)
print("PHASE COMPARISON (ANOVA)")
print("="*80)
# Compare religiosity across cycle phases
fertile_group = df_clean[df_clean['FertileWindow'] == 1]['Religiosity']
luteal_group = df_clean[df_clean['LutealPhase'] == 1]['Religiosity']
follicular_group = df_clean[df_clean['FollicularPhase'] == 1]['Religiosity']

print(f"\nReligiosity by phase:")
print(f"  Follicular (n={len(follicular_group)}): mean={follicular_group.mean():.3f}, std={follicular_group.std():.3f}")
print(f"  Fertile window (n={len(fertile_group)}): mean={fertile_group.mean():.3f}, std={fertile_group.std():.3f}")
print(f"  Luteal (n={len(luteal_group)}): mean={luteal_group.mean():.3f}, std={luteal_group.std():.3f}")

# One-way ANOVA
phase_labels = []
for idx, row in df_clean.iterrows():
    if row['FollicularPhase'] == 1:
        phase_labels.append('Follicular')
    elif row['FertileWindow'] == 1:
        phase_labels.append('Fertile')
    else:
        phase_labels.append('Luteal')
df_clean['Phase'] = phase_labels

groups = [df_clean[df_clean['Phase'] == phase]['Religiosity'].values for phase in ['Follicular', 'Fertile', 'Luteal']]
f_stat, p_anova = stats.f_oneway(*groups)
print(f"\nOne-way ANOVA: F={f_stat:.4f}, p-value: {p_anova:.4f}")

# T-test: Fertile window vs non-fertile
fertile_vals = df_clean[df_clean['FertileWindow'] == 1]['Religiosity']
nonfertile_vals = df_clean[df_clean['FertileWindow'] == 0]['Religiosity']
t_stat, p_ttest = stats.ttest_ind(fertile_vals, nonfertile_vals)
print(f"\nT-test (Fertile vs Non-fertile):")
print(f"  t={t_stat:.4f}, p-value: {p_ttest:.4f}")

print("\n" + "="*80)
print("REGRESSION ANALYSIS (OLS)")
print("="*80)
# Multiple regression with statsmodels for detailed statistics
X_reg = df_clean[['CycleDay', 'ReportedCycleLength', 'Relationship', 'Sure1', 'Sure2']]
y_reg = df_clean['Religiosity']
X_reg_const = sm.add_constant(X_reg)

model_ols = sm.OLS(y_reg, X_reg_const).fit()
print(model_ols.summary())

print("\n" + "="*80)
print("INTERPRETABLE MODEL: EXPLAINABLE BOOSTING")
print("="*80)
# Use interpret library for interpretable model
X_ebm = df_clean[['CycleDay', 'CycleProportion', 'ReportedCycleLength', 'Relationship', 'Sure1', 'Sure2']]
y_ebm = df_clean['Religiosity']

ebm = ExplainableBoostingRegressor(random_state=42, interactions=0)
ebm.fit(X_ebm, y_ebm)

print("\nExplainable Boosting Model - Feature Importance:")
for i, (name, importance) in enumerate(zip(X_ebm.columns, ebm.term_importances())):
    print(f"  {name}: {importance:.4f}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

# Determine if there's a significant effect
significant_effects = []
effect_strengths = []

if p_cycle_day < 0.05:
    significant_effects.append(f"CycleDay correlation (p={p_cycle_day:.4f}, r={corr_cycle_day:.4f})")
    effect_strengths.append(abs(corr_cycle_day))

if p_anova < 0.05:
    significant_effects.append(f"Phase differences (p={p_anova:.4f})")
    # Calculate effect size (eta squared)
    grand_mean = df_clean['Religiosity'].mean()
    ss_between = sum([len(g) * (g.mean() - grand_mean)**2 for g in groups])
    ss_total = sum((df_clean['Religiosity'] - grand_mean)**2)
    eta_squared = ss_between / ss_total
    effect_strengths.append(eta_squared)
    
if p_ttest < 0.05:
    significant_effects.append(f"Fertile vs non-fertile (p={p_ttest:.4f})")
    # Cohen's d
    pooled_std = np.sqrt(((len(fertile_vals)-1)*fertile_vals.std()**2 + (len(nonfertile_vals)-1)*nonfertile_vals.std()**2) / (len(fertile_vals)+len(nonfertile_vals)-2))
    cohens_d = abs(fertile_vals.mean() - nonfertile_vals.mean()) / pooled_std
    effect_strengths.append(cohens_d)

# Check OLS p-value for CycleDay
cycle_day_pval = model_ols.pvalues['CycleDay']
if cycle_day_pval < 0.05:
    significant_effects.append(f"OLS CycleDay coefficient (p={cycle_day_pval:.4f})")

print(f"\nSignificant effects found: {len(significant_effects)}")
for effect in significant_effects:
    print(f"  - {effect}")

# Determine response score
if len(significant_effects) == 0:
    response = 10
    explanation = "No statistically significant relationship found between hormonal fluctuations (menstrual cycle phase) and religiosity. The correlation between cycle day and religiosity was not significant (p={:.3f}), ANOVA across phases showed no significant differences (p={:.3f}), and the comparison between fertile and non-fertile windows was not significant (p={:.3f}).".format(p_cycle_day, p_anova, p_ttest)
elif len(significant_effects) >= 2 and any(e > 0.1 for e in effect_strengths):
    response = 65
    explanation = "Multiple statistically significant effects detected: {}. While these suggest some relationship between menstrual cycle and religiosity, the effect sizes are relatively modest. The evidence supports a moderate relationship.".format('; '.join(significant_effects[:2]))
elif len(significant_effects) >= 1:
    response = 45
    explanation = "Found statistically significant evidence: {}. However, the relationship is relatively weak with small effect sizes, suggesting hormonal fluctuations have only a limited association with religiosity scores.".format(significant_effects[0])
else:
    response = 15
    explanation = "Very weak or no evidence of relationship between hormonal fluctuations and religiosity. Most statistical tests showed non-significant results."

print(f"\nFinal Response Score: {response}/100")
print(f"Explanation: {explanation}")

# Write conclusion to file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n" + "="*80)
print("Analysis complete. Results written to conclusion.txt")
print("="*80)
