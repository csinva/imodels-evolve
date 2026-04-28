import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('reading.csv')

print("Shape:", df.shape)
print("\nSummary stats for speed:")
print(df['speed'].describe())

# Focus on dyslexic participants only
dyslexic = df[df['dyslexia_bin'] == 1]
non_dyslexic = df[df['dyslexia_bin'] == 0]

print(f"\nDyslexic participants rows: {len(dyslexic)}")
print(f"Non-dyslexic participants rows: {len(non_dyslexic)}")

# For dyslexic participants, compare speed with vs without reader view
dys_reader = dyslexic[dyslexic['reader_view'] == 1]['speed']
dys_no_reader = dyslexic[dyslexic['reader_view'] == 0]['speed']

print(f"\nDyslexic + Reader View ON  - mean speed: {dys_reader.mean():.2f}, n={len(dys_reader)}")
print(f"Dyslexic + Reader View OFF - mean speed: {dys_no_reader.mean():.2f}, n={len(dys_no_reader)}")

# t-test for dyslexic group
t_stat, p_val = stats.ttest_ind(dys_reader, dys_no_reader)
print(f"\nt-test (dyslexic group, reader view on vs off): t={t_stat:.4f}, p={p_val:.4f}")

# Also check with log-transformed speed (more normal distribution)
dys_reader_log = np.log1p(dys_reader)
dys_no_reader_log = np.log1p(dys_no_reader)
t_log, p_log = stats.ttest_ind(dys_reader_log, dys_no_reader_log)
print(f"t-test (log speed): t={t_log:.4f}, p={p_log:.4f}")

# Mann-Whitney U test (non-parametric)
u_stat, p_mw = stats.mannwhitneyu(dys_reader, dys_no_reader, alternative='two-sided')
print(f"Mann-Whitney U test: U={u_stat:.2f}, p={p_mw:.4f}")

# OLS regression: speed ~ reader_view * dyslexia_bin (interaction)
df_model = df[['speed', 'reader_view', 'dyslexia_bin']].dropna()
df_model['interaction'] = df_model['reader_view'] * df_model['dyslexia_bin']
X = sm.add_constant(df_model[['reader_view', 'dyslexia_bin', 'interaction']])
y = np.log1p(df_model['speed'])
model = sm.OLS(y, X).fit()
print("\nOLS regression (log speed ~ reader_view * dyslexia_bin):")
print(model.summary().tables[1])

# Effect size (Cohen's d) for dyslexic group
pooled_std = np.sqrt((dys_reader.var() + dys_no_reader.var()) / 2)
cohens_d = (dys_reader.mean() - dys_no_reader.mean()) / pooled_std
print(f"\nCohen's d (dyslexic group): {cohens_d:.4f}")

# Conclusion
# p_val from t-test on dyslexic subgroup is the primary metric
# Also consider direction: is speed higher with reader view?
speed_increase = dys_reader.mean() > dys_no_reader.mean()
print(f"\nReader view increases speed for dyslexic: {speed_increase}")
print(f"Primary p-value: {p_val:.4f}")
print(f"Mann-Whitney p-value: {p_mw:.4f}")

# Determine response score
# p < 0.05 and positive direction => moderate to strong yes
# p < 0.01 => strong yes
# p >= 0.05 => no
if p_val < 0.05 and p_mw < 0.05 and speed_increase:
    if p_val < 0.01:
        response = 75
    else:
        response = 65
    explanation = (
        f"Reader View significantly improves reading speed for individuals with dyslexia. "
        f"Dyslexic participants with Reader View ON had mean speed {dys_reader.mean():.1f} vs {dys_no_reader.mean():.1f} without it. "
        f"t-test: t={t_stat:.3f}, p={p_val:.4f}; Mann-Whitney: p={p_mw:.4f}. "
        f"Cohen's d={cohens_d:.3f}. OLS interaction term p={model.pvalues['interaction']:.4f}."
    )
elif p_val < 0.05 and speed_increase:
    response = 60
    explanation = (
        f"Reader View shows a statistically significant improvement in reading speed for dyslexic individuals. "
        f"Mean speed: {dys_reader.mean():.1f} (reader view) vs {dys_no_reader.mean():.1f} (no reader view). "
        f"t-test p={p_val:.4f}, Mann-Whitney p={p_mw:.4f}. Cohen's d={cohens_d:.3f}."
    )
else:
    response = 30
    explanation = (
        f"No statistically significant improvement in reading speed for dyslexic individuals with Reader View. "
        f"Mean speed: {dys_reader.mean():.1f} (reader view) vs {dys_no_reader.mean():.1f} (no reader view). "
        f"t-test: t={t_stat:.3f}, p={p_val:.4f}; Mann-Whitney: p={p_mw:.4f}. "
        f"Speed increase direction: {speed_increase}. Cohen's d={cohens_d:.3f}."
    )

result = {"response": response, "explanation": explanation}
with open('conclusion.txt', 'w') as f:
    json.dump(result, f)

print("\nconclusion.txt written.")
print(json.dumps(result, indent=2))
