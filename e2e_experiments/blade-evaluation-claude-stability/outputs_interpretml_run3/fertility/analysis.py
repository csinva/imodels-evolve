import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import json
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('fertility.csv')

# Parse dates
for col in ['DateTesting', 'StartDateofLastPeriod', 'StartDateofPeriodBeforeLast']:
    df[col] = pd.to_datetime(df[col], format='%m/%d/%y', errors='coerce')

# Compute cycle day: days since last period start
df['CycleDay'] = (df['DateTesting'] - df['StartDateofLastPeriod']).dt.days

# Use reported cycle length or computed cycle length as fallback
df['ComputedCycleLength'] = (df['StartDateofLastPeriod'] - df['StartDateofPeriodBeforeLast']).dt.days
df['CycleLength'] = df['ReportedCycleLength'].fillna(df['ComputedCycleLength'])
df['CycleLength'] = df['CycleLength'].where(df['CycleLength'].between(21, 42), 28)

# Estimate fertility: peak around day 14 of a 28-day cycle
# Fertile window: roughly days (CycleLength - 17) to (CycleLength - 10)
df['FertileWindowStart'] = df['CycleLength'] - 17
df['FertileWindowEnd'] = df['CycleLength'] - 10
df['InFertileWindow'] = ((df['CycleDay'] >= df['FertileWindowStart']) &
                          (df['CycleDay'] <= df['FertileWindowEnd'])).astype(int)

# Continuous fertility score: distance from ovulation day (CycleLength - 14)
df['OvulationDay'] = df['CycleLength'] - 14
df['DaysFromOvulation'] = np.abs(df['CycleDay'] - df['OvulationDay'])
# Fertility score: higher = closer to ovulation
df['FertilityScore'] = np.exp(-df['DaysFromOvulation'] / 5)

# Composite religiosity
df['Religiosity'] = df[['Rel1', 'Rel2', 'Rel3']].mean(axis=1)

# Drop rows with missing key values
df_clean = df.dropna(subset=['CycleDay', 'Religiosity', 'FertilityScore', 'InFertileWindow'])
print(f"Clean N={len(df_clean)}")

print("\n--- Summary ---")
print(df_clean[['Religiosity', 'FertilityScore', 'InFertileWindow', 'CycleDay']].describe())

# 1. Correlation: fertility score vs religiosity
r_cont, p_cont = stats.pearsonr(df_clean['FertilityScore'], df_clean['Religiosity'])
print(f"\nPearson r(FertilityScore, Religiosity) = {r_cont:.4f}, p = {p_cont:.4f}")

r_sp, p_sp = stats.spearmanr(df_clean['FertilityScore'], df_clean['Religiosity'])
print(f"Spearman r = {r_sp:.4f}, p = {p_sp:.4f}")

# 2. T-test: fertile window vs not
fertile = df_clean[df_clean['InFertileWindow'] == 1]['Religiosity']
non_fertile = df_clean[df_clean['InFertileWindow'] == 0]['Religiosity']
print(f"\nFertile window N={len(fertile)}, mean={fertile.mean():.3f}")
print(f"Non-fertile N={len(non_fertile)}, mean={non_fertile.mean():.3f}")
t_stat, p_ttest = stats.ttest_ind(fertile, non_fertile)
print(f"t-test t={t_stat:.4f}, p={p_ttest:.4f}")

# 3. OLS regression: fertility predicting religiosity
X = sm.add_constant(df_clean['FertilityScore'])
ols = sm.OLS(df_clean['Religiosity'], X).fit()
print("\nOLS summary:")
print(ols.summary())

# 4. Per-item correlations
for col in ['Rel1', 'Rel2', 'Rel3']:
    tmp = df_clean[['FertilityScore', col]].dropna()
    r, p = stats.pearsonr(tmp['FertilityScore'], tmp[col])
    print(f"  {col}: r={r:.4f}, p={p:.4f}")

# Determine conclusion
# Significance threshold p < 0.05
sig_continuous = p_cont < 0.05
sig_ttest = p_ttest < 0.05
ols_p = ols.pvalues['FertilityScore']
sig_ols = ols_p < 0.05

print(f"\nSignificant continuous correlation: {sig_continuous} (p={p_cont:.4f})")
print(f"Significant t-test (fertile vs not): {sig_ttest} (p={p_ttest:.4f})")
print(f"Significant OLS: {sig_ols} (p={ols_p:.4f})")

# Score: if no test is significant, low score; if some are, moderate; if all are, high
n_sig = sum([sig_continuous, sig_ttest, sig_ols])
effect_direction = "positive" if r_cont > 0 else "negative"

if n_sig == 0:
    response = 15
    explanation = (
        f"No significant relationship found between fertility (hormonal fluctuations) and religiosity. "
        f"Pearson r={r_cont:.4f} (p={p_cont:.4f}), t-test p={p_ttest:.4f}, OLS p={ols_p:.4f}. "
        f"Mean religiosity fertile={fertile.mean():.3f} vs non-fertile={non_fertile.mean():.3f}. "
        f"The data does not support that hormonal fluctuations associated with fertility affect women's religiosity."
    )
elif n_sig == 1:
    response = 35
    explanation = (
        f"Weak/borderline evidence for a {effect_direction} relationship between fertility and religiosity. "
        f"Pearson r={r_cont:.4f} (p={p_cont:.4f}), t-test p={p_ttest:.4f}, OLS p={ols_p:.4f}. "
        f"Only {n_sig}/3 tests reached significance; evidence is insufficient to confirm effect."
    )
elif n_sig == 2:
    response = 55
    explanation = (
        f"Moderate evidence for a {effect_direction} relationship between fertility and religiosity. "
        f"Pearson r={r_cont:.4f} (p={p_cont:.4f}), t-test p={p_ttest:.4f}, OLS p={ols_p:.4f}. "
        f"{n_sig}/3 tests significant. Some support for the hypothesis."
    )
else:
    response = 75
    explanation = (
        f"Consistent evidence for a {effect_direction} relationship between fertility and religiosity. "
        f"Pearson r={r_cont:.4f} (p={p_cont:.4f}), t-test p={p_ttest:.4f}, OLS p={ols_p:.4f}. "
        f"All 3 tests significant. Fertility fluctuations appear associated with religiosity changes."
    )

result = {"response": response, "explanation": explanation}
with open('conclusion.txt', 'w') as f:
    json.dump(result, f)

print(f"\nconclusion.txt written: response={response}")
