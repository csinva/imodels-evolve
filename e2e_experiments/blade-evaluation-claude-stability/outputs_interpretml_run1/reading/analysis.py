import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('reading.csv')
print(f"Shape: {df.shape}")
print(df[['reader_view', 'dyslexia', 'dyslexia_bin', 'speed']].describe())

# Focus on dyslexic individuals (dyslexia_bin == 1)
dyslexic = df[df['dyslexia_bin'] == 1].copy()
print(f"\nDyslexic individuals: {len(dyslexic)} rows, {dyslexic['uuid'].nunique()} unique participants")

# Reading speed by reader_view for dyslexic individuals
rv_on = dyslexic[dyslexic['reader_view'] == 1]['speed']
rv_off = dyslexic[dyslexic['reader_view'] == 0]['speed']

print(f"\nDyslexic - Reader View ON:  n={len(rv_on)}, mean={rv_on.mean():.2f}, median={rv_on.median():.2f}")
print(f"Dyslexic - Reader View OFF: n={len(rv_off)}, mean={rv_off.mean():.2f}, median={rv_off.median():.2f}")

# Mann-Whitney U test (non-parametric due to skewed speed distribution)
mwu_stat, mwu_p = stats.mannwhitneyu(rv_on, rv_off, alternative='two-sided')
print(f"\nMann-Whitney U test (dyslexic): stat={mwu_stat:.2f}, p={mwu_p:.4f}")

# T-test
tstat, tp = stats.ttest_ind(rv_on, rv_off)
print(f"T-test (dyslexic): stat={tstat:.4f}, p={tp:.4f}")

# Also check with log-transformed speed
log_rv_on = np.log1p(rv_on)
log_rv_off = np.log1p(rv_off)
tstat_log, tp_log = stats.ttest_ind(log_rv_on, log_rv_off)
print(f"T-test log-speed (dyslexic): stat={tstat_log:.4f}, p={tp_log:.4f}")

# OLS regression: speed ~ reader_view + dyslexia interaction, for dyslexic subset
X = sm.add_constant(dyslexic[['reader_view', 'dyslexia']])
model = sm.OLS(np.log1p(dyslexic['speed']), X).fit()
print("\nOLS (dyslexic subset, log speed ~ reader_view + dyslexia):")
print(model.summary().tables[1])

# Broader analysis: interaction between reader_view and dyslexia_bin in full data
df2 = df.dropna(subset=['reader_view', 'dyslexia_bin', 'speed']).copy()
df2['rv_x_dyslexia'] = df2['reader_view'] * df2['dyslexia_bin']
X2 = sm.add_constant(df2[['reader_view', 'dyslexia_bin', 'rv_x_dyslexia']])
model2 = sm.OLS(np.log1p(df2['speed']), X2).fit()
print("\nOLS (full data, log speed ~ reader_view * dyslexia_bin):")
print(model2.summary().tables[1])

rv_coef = model2.params['reader_view']
interaction_coef = model2.params['rv_x_dyslexia']
interaction_p = model2.pvalues['rv_x_dyslexia']
rv_p = model2.pvalues['reader_view']
print(f"\nreader_view coef={rv_coef:.4f}, p={rv_p:.4f}")
print(f"interaction coef={interaction_coef:.4f}, p={interaction_p:.4f}")

# Determine conclusion
# Key evidence: does reader_view significantly increase speed for dyslexic individuals?
main_effect_significant = tp < 0.05
log_effect_significant = tp_log < 0.05
mwu_significant = mwu_p < 0.05

mean_diff = rv_on.mean() - rv_off.mean()
direction_positive = mean_diff > 0

print(f"\nSummary:")
print(f"  Mean speed dyslexic + reader_view: {rv_on.mean():.2f}")
print(f"  Mean speed dyslexic - reader_view: {rv_off.mean():.2f}")
print(f"  Difference (ON - OFF): {mean_diff:.2f}")
print(f"  T-test p-value: {tp:.4f} (significant: {main_effect_significant})")
print(f"  Log T-test p-value: {tp_log:.4f} (significant: {log_effect_significant})")
print(f"  MWU p-value: {mwu_p:.4f} (significant: {mwu_significant})")

# Score: if significant improvement, high score; if not, low score
if log_effect_significant and direction_positive:
    score = 75
    explanation = (
        f"Reader View shows a statistically significant improvement in reading speed for individuals with dyslexia. "
        f"Log-transformed t-test: p={tp_log:.4f}. Mean speed with reader view: {rv_on.mean():.1f} vs without: {rv_off.mean():.1f} "
        f"(difference: {mean_diff:.1f}). Mann-Whitney U test p={mwu_p:.4f}. "
        f"The evidence supports that Reader View improves reading speed for dyslexic users."
    )
elif mwu_significant and direction_positive:
    score = 65
    explanation = (
        f"Reader View shows a statistically significant improvement in reading speed for individuals with dyslexia "
        f"(Mann-Whitney U p={mwu_p:.4f}). Mean speed with reader view: {rv_on.mean():.1f} vs without: {rv_off.mean():.1f}. "
        f"The non-parametric test supports improvement, though log t-test p={tp_log:.4f}."
    )
elif main_effect_significant and direction_positive:
    score = 60
    explanation = (
        f"Reader View shows a statistically significant improvement in reading speed for individuals with dyslexia "
        f"(t-test p={tp:.4f}). Mean speed with reader view: {rv_on.mean():.1f} vs without: {rv_off.mean():.1f}."
    )
elif not (main_effect_significant or log_effect_significant or mwu_significant):
    score = 25
    explanation = (
        f"Reader View does NOT show a statistically significant improvement in reading speed for individuals with dyslexia. "
        f"T-test p={tp:.4f}, log t-test p={tp_log:.4f}, MWU p={mwu_p:.4f}. "
        f"Mean speed with reader view: {rv_on.mean():.1f} vs without: {rv_off.mean():.1f} (difference: {mean_diff:.1f}). "
        f"Lack of significance suggests no reliable effect."
    )
else:
    score = 35
    explanation = (
        f"Mixed evidence: Reader View does not consistently improve reading speed for dyslexic individuals. "
        f"T-test p={tp:.4f}, log t-test p={tp_log:.4f}, MWU p={mwu_p:.4f}. "
        f"Mean speed difference: {mean_diff:.1f} ({rv_on.mean():.1f} vs {rv_off.mean():.1f})."
    )

result = {"response": score, "explanation": explanation}
with open('conclusion.txt', 'w') as f:
    json.dump(result, f)

print(f"\nConclusion written: score={score}")
print(f"Explanation: {explanation}")
