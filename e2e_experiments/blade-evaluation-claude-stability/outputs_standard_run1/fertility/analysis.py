import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('fertility.csv')

# Parse dates
for col in ['DateTesting', 'StartDateofLastPeriod', 'StartDateofPeriodBeforeLast']:
    df[col] = pd.to_datetime(df[col], format='%m/%d/%y', errors='coerce')

# Compute cycle length from dates (fallback to ReportedCycleLength)
df['ComputedCycleLength'] = (df['StartDateofLastPeriod'] - df['StartDateofPeriodBeforeLast']).dt.days
df['CycleLength'] = df['ReportedCycleLength'].fillna(df['ComputedCycleLength'])

# Days since last period at time of testing
df['DaysSinceLastPeriod'] = (df['DateTesting'] - df['StartDateofLastPeriod']).dt.days

# Fertility: proximity to ovulation (ovulation ~14 days before next period)
# Days until next expected period = CycleLength - DaysSinceLastPeriod
# Days to ovulation = DaysUntilNextPeriod - 14
df['DaysUntilNextPeriod'] = df['CycleLength'] - df['DaysSinceLastPeriod']
df['DaysToOvulation'] = df['DaysUntilNextPeriod'] - 14
# Fertility is highest when DaysToOvulation ~ 0; use negative absolute distance
df['FertilityProximity'] = -np.abs(df['DaysToOvulation'])

# Composite religiosity score
df['Religiosity'] = df[['Rel1', 'Rel2', 'Rel3']].mean(axis=1)

# Drop rows missing key variables
analysis_df = df.dropna(subset=['FertilityProximity', 'Religiosity', 'CycleLength', 'DaysSinceLastPeriod'])
print(f"N after dropping NAs: {len(analysis_df)}")
print(analysis_df[['FertilityProximity', 'Religiosity']].describe())

# --- Statistical tests ---

# 1. Pearson correlation: FertilityProximity vs Religiosity
r, p_pearson = stats.pearsonr(analysis_df['FertilityProximity'], analysis_df['Religiosity'])
print(f"\nPearson r(FertilityProximity, Religiosity) = {r:.4f}, p = {p_pearson:.4f}")

# 2. Also test raw days-to-ovulation vs religiosity
r2, p2 = stats.pearsonr(analysis_df['DaysToOvulation'].dropna(),
                         analysis_df.loc[analysis_df['DaysToOvulation'].notna(), 'Religiosity'])
print(f"Pearson r(DaysToOvulation, Religiosity) = {r2:.4f}, p = {p2:.4f}")

# 3. High vs Low fertility groups (fertile window = within 5 days of ovulation)
high_fert = analysis_df[analysis_df['DaysToOvulation'].abs() <= 5]['Religiosity']
low_fert = analysis_df[analysis_df['DaysToOvulation'].abs() > 5]['Religiosity']
print(f"\nHigh fertility (n={len(high_fert)}): mean Religiosity = {high_fert.mean():.3f}")
print(f"Low fertility  (n={len(low_fert)}): mean Religiosity = {low_fert.mean():.3f}")
t_stat, p_ttest = stats.ttest_ind(high_fert, low_fert)
print(f"t-test: t = {t_stat:.4f}, p = {p_ttest:.4f}")

# 4. OLS regression with controls
X = analysis_df[['FertilityProximity', 'Relationship', 'CycleLength']].dropna()
y = analysis_df.loc[X.index, 'Religiosity']
X_const = sm.add_constant(X)
model = sm.OLS(y, X_const).fit()
print("\nOLS regression summary:")
print(model.summary())

# 5. Individual religiosity items
for item in ['Rel1', 'Rel2', 'Rel3']:
    r_i, p_i = stats.pearsonr(analysis_df['FertilityProximity'], analysis_df[item])
    print(f"r({item}, FertilityProximity) = {r_i:.4f}, p = {p_i:.4f}")

# --- Decision ---
# Aggregate evidence: if p_pearson < 0.05 or p_ttest < 0.05, answer yes
significant = (p_pearson < 0.05) or (p_ttest < 0.05)
fertility_coef_p = model.pvalues.get('FertilityProximity', 1.0)

print(f"\nFertilityProximity OLS p-value: {fertility_coef_p:.4f}")
print(f"Overall significant evidence: {significant}")

# Score: base on strength of evidence
if fertility_coef_p < 0.01 and abs(r) > 0.15:
    response = 75
elif fertility_coef_p < 0.05 and abs(r) > 0.1:
    response = 65
elif fertility_coef_p < 0.05 or p_ttest < 0.05:
    response = 55
elif fertility_coef_p < 0.1 or p_pearson < 0.1:
    response = 35
else:
    response = 20

explanation = (
    f"Analysis of {len(analysis_df)} women: fertility proximity (|days to ovulation| inverted) "
    f"correlated with composite religiosity: r={r:.4f}, p={p_pearson:.4f}. "
    f"High vs low fertility t-test: t={t_stat:.4f}, p={p_ttest:.4f}. "
    f"OLS regression FertilityProximity coefficient p={fertility_coef_p:.4f}. "
    f"The statistical evidence {'supports' if significant else 'does not support'} an effect of "
    f"fertility-related hormonal fluctuations on religiosity."
)

result = {"response": response, "explanation": explanation}
print(f"\nResult: {result}")

with open('conclusion.txt', 'w') as f:
    json.dump(result, f)
print("conclusion.txt written.")
