import pandas as pd
import numpy as np
import json
from datetime import datetime
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from imodels import RuleFitRegressor, FIGSRegressor

# Load the data
df = pd.read_csv('fertility.csv')

# Parse dates
df['DateTesting'] = pd.to_datetime(df['DateTesting'], format='%m/%d/%y')
df['StartDateofLastPeriod'] = pd.to_datetime(df['StartDateofLastPeriod'], format='%m/%d/%y')
df['StartDateofPeriodBeforeLast'] = pd.to_datetime(df['StartDateofPeriodBeforeLast'], format='%m/%d/%y')

# Calculate days since last period (cycle phase)
df['DaysSinceLastPeriod'] = (df['DateTesting'] - df['StartDateofLastPeriod']).dt.days

# Calculate computed cycle length from actual dates
df['ComputedCycleLength'] = (df['StartDateofLastPeriod'] - df['StartDateofPeriodBeforeLast']).dt.days

# Create fertility/ovulation-related variables
# Ovulation typically occurs around day 14 in a 28-day cycle, or about cycle_length - 14
# Fertile window is typically days 10-18 of cycle, or cycle_length - 18 to cycle_length - 10
df['EstimatedOvulationDay'] = df['ReportedCycleLength'] - 14

# Calculate distance from ovulation (absolute value - closer to 0 means closer to ovulation)
df['DaysFromOvulation'] = df['DaysSinceLastPeriod'] - df['EstimatedOvulationDay']
df['AbsDaysFromOvulation'] = np.abs(df['DaysFromOvulation'])

# Create fertility phase indicators
# Fertile window (high fertility): typically cycle days 10-18 or ovulation +/- 4 days
df['InFertileWindow'] = ((df['DaysFromOvulation'] >= -4) & (df['DaysFromOvulation'] <= 4)).astype(int)

# Follicular phase (pre-ovulation): days 1 to ovulation
df['FollicularPhase'] = (df['DaysSinceLastPeriod'] <= df['EstimatedOvulationDay']).astype(int)

# Luteal phase (post-ovulation): after ovulation
df['LutealPhase'] = (df['DaysSinceLastPeriod'] > df['EstimatedOvulationDay']).astype(int)

# Create overall religiosity score (average of three religiosity questions)
df['ReligiosityScore'] = df[['Rel1', 'Rel2', 'Rel3']].mean(axis=1)

# Clean data - remove rows with missing key variables
df_clean = df.dropna(subset=['ReligiosityScore', 'DaysSinceLastPeriod', 'ReportedCycleLength'])

print("=" * 80)
print("DATA EXPLORATION")
print("=" * 80)
print(f"\nSample size: {len(df_clean)}")
print(f"\nReligiosity Score - Mean: {df_clean['ReligiosityScore'].mean():.2f}, SD: {df_clean['ReligiosityScore'].std():.2f}")
print(f"Days Since Last Period - Mean: {df_clean['DaysSinceLastPeriod'].mean():.2f}, SD: {df_clean['DaysSinceLastPeriod'].std():.2f}")
print(f"\nWomen in fertile window: {df_clean['InFertileWindow'].sum()} ({df_clean['InFertileWindow'].mean()*100:.1f}%)")
print(f"Women in follicular phase: {df_clean['FollicularPhase'].sum()} ({df_clean['FollicularPhase'].mean()*100:.1f}%)")
print(f"Women in luteal phase: {df_clean['LutealPhase'].sum()} ({df_clean['LutealPhase'].mean()*100:.1f}%)")

# Test 1: Compare religiosity between fertile window and non-fertile window
fertile_religiosity = df_clean[df_clean['InFertileWindow'] == 1]['ReligiosityScore']
non_fertile_religiosity = df_clean[df_clean['InFertileWindow'] == 0]['ReligiosityScore']

t_stat_fertile, p_value_fertile = stats.ttest_ind(fertile_religiosity, non_fertile_religiosity)

print("\n" + "=" * 80)
print("TEST 1: Fertile Window vs Non-Fertile Window")
print("=" * 80)
print(f"Fertile window (n={len(fertile_religiosity)}): Mean={fertile_religiosity.mean():.2f}, SD={fertile_religiosity.std():.2f}")
print(f"Non-fertile (n={len(non_fertile_religiosity)}): Mean={non_fertile_religiosity.mean():.2f}, SD={non_fertile_religiosity.std():.2f}")
print(f"t-statistic: {t_stat_fertile:.3f}, p-value: {p_value_fertile:.4f}")

# Test 2: Compare religiosity between follicular and luteal phases
follicular_religiosity = df_clean[df_clean['FollicularPhase'] == 1]['ReligiosityScore']
luteal_religiosity = df_clean[df_clean['LutealPhase'] == 1]['ReligiosityScore']

t_stat_phase, p_value_phase = stats.ttest_ind(follicular_religiosity, luteal_religiosity)

print("\n" + "=" * 80)
print("TEST 2: Follicular Phase vs Luteal Phase")
print("=" * 80)
print(f"Follicular phase (n={len(follicular_religiosity)}): Mean={follicular_religiosity.mean():.2f}, SD={follicular_religiosity.std():.2f}")
print(f"Luteal phase (n={len(luteal_religiosity)}): Mean={luteal_religiosity.mean():.2f}, SD={luteal_religiosity.std():.2f}")
print(f"t-statistic: {t_stat_phase:.3f}, p-value: {p_value_phase:.4f}")

# Test 3: Correlation between days from ovulation and religiosity
corr_ovulation, p_value_corr = stats.pearsonr(df_clean['AbsDaysFromOvulation'], df_clean['ReligiosityScore'])

print("\n" + "=" * 80)
print("TEST 3: Correlation - Distance from Ovulation vs Religiosity")
print("=" * 80)
print(f"Pearson correlation: {corr_ovulation:.3f}, p-value: {p_value_corr:.4f}")
print("(Negative would mean closer to ovulation = higher religiosity)")

# Test 4: Linear regression with statsmodels for detailed statistics
X = df_clean[['DaysSinceLastPeriod', 'InFertileWindow', 'AbsDaysFromOvulation', 'ReportedCycleLength', 'Relationship']]
y = df_clean['ReligiosityScore']

X_with_const = sm.add_constant(X)
model_sm = sm.OLS(y, X_with_const).fit()

print("\n" + "=" * 80)
print("TEST 4: Multiple Regression with Fertility Variables")
print("=" * 80)
print(model_sm.summary())

# Test 5: Interpretable model using imodels RuleFit
print("\n" + "=" * 80)
print("TEST 5: Interpretable RuleFit Model")
print("=" * 80)

X_rulefit = df_clean[['DaysSinceLastPeriod', 'InFertileWindow', 'AbsDaysFromOvulation', 
                       'FollicularPhase', 'ReportedCycleLength', 'Relationship', 'Sure1', 'Sure2']]
y_rulefit = df_clean['ReligiosityScore']

try:
    rf_model = RuleFitRegressor(max_rules=10, random_state=42)
    rf_model.fit(X_rulefit, y_rulefit)
    
    print("\nRuleFit Model Rules:")
    rules_df = rf_model.get_rules()
    if len(rules_df) > 0:
        print(rules_df[['rule', 'coef', 'importance']].head(10))
    else:
        print("No significant rules extracted")
except Exception as e:
    print(f"RuleFit failed: {e}")

# Test 6: Simple decision tree for interpretability
from sklearn.tree import DecisionTreeRegressor
dt_model = DecisionTreeRegressor(max_depth=3, min_samples_leaf=20, random_state=42)
dt_model.fit(X_rulefit, y_rulefit)

print("\n" + "=" * 80)
print("TEST 6: Decision Tree Feature Importances")
print("=" * 80)
feature_names = X_rulefit.columns
importances = pd.DataFrame({
    'feature': feature_names,
    'importance': dt_model.feature_importances_
}).sort_values('importance', ascending=False)
print(importances)

# Additional test: Relationship status as potential confound
print("\n" + "=" * 80)
print("TEST 7: ANOVA - Religiosity by Relationship Status")
print("=" * 80)
relationship_groups = [df_clean[df_clean['Relationship'] == i]['ReligiosityScore'].values 
                       for i in df_clean['Relationship'].unique() if not pd.isna(i)]
f_stat, p_value_anova = stats.f_oneway(*relationship_groups)
print(f"F-statistic: {f_stat:.3f}, p-value: {p_value_anova:.4f}")

for rel_status in sorted(df_clean['Relationship'].unique()):
    if not pd.isna(rel_status):
        group = df_clean[df_clean['Relationship'] == rel_status]['ReligiosityScore']
        print(f"  Relationship={int(rel_status)}: Mean={group.mean():.2f}, SD={group.std():.2f}, n={len(group)}")

# CONCLUSION
print("\n" + "=" * 80)
print("SYNTHESIS AND CONCLUSION")
print("=" * 80)

# Analyze all results
significant_results = []
if p_value_fertile < 0.05:
    significant_results.append(f"Fertile window effect (p={p_value_fertile:.4f})")
if p_value_phase < 0.05:
    significant_results.append(f"Cycle phase effect (p={p_value_phase:.4f})")
if p_value_corr < 0.05:
    significant_results.append(f"Ovulation distance correlation (p={p_value_corr:.4f})")

# Check regression coefficients
fertile_window_coef = model_sm.params.get('InFertileWindow', 0)
fertile_window_pval = model_sm.pvalues.get('InFertileWindow', 1)
days_from_ovulation_pval = model_sm.pvalues.get('AbsDaysFromOvulation', 1)

print(f"\nSignificant findings (p<0.05): {len(significant_results)}")
for result in significant_results:
    print(f"  - {result}")

print(f"\nRegression coefficient for InFertileWindow: {fertile_window_coef:.4f} (p={fertile_window_pval:.4f})")
print(f"Regression p-value for AbsDaysFromOvulation: {days_from_ovulation_pval:.4f}")

# Determine response score
explanation_parts = []

if p_value_fertile < 0.05 or p_value_phase < 0.05 or p_value_corr < 0.05:
    # At least one test is significant
    if p_value_fertile < 0.01 or p_value_phase < 0.01:
        response_score = 70
        explanation_parts.append("Strong statistical evidence")
    elif p_value_fertile < 0.05 or p_value_phase < 0.05:
        response_score = 60
        explanation_parts.append("Statistically significant effects found")
    else:
        response_score = 55
        explanation_parts.append("Marginal statistical evidence")
    
    if len(significant_results) > 1:
        response_score += 10
        explanation_parts.append(f"across {len(significant_results)} tests")
else:
    # No significant effects
    response_score = 20
    explanation_parts.append("No statistically significant relationship found")
    explanation_parts.append(f"All tests failed to reach significance (p-values: {p_value_fertile:.3f}, {p_value_phase:.3f}, {p_value_corr:.3f})")

explanation = "; ".join(explanation_parts) + f". Research question: hormonal fluctuations during menstrual cycle and religiosity."

# Write conclusion
conclusion = {
    "response": response_score,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print(f"\n{'='*80}")
print(f"FINAL SCORE: {response_score}/100")
print(f"EXPLANATION: {explanation}")
print(f"{'='*80}")
print("\nconclusion.txt has been written.")
