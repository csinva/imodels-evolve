import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('reading.csv')

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nBasic stats for speed:")
print(df['speed'].describe())
print("\nReader view counts:", df['reader_view'].value_counts().to_dict())
print("Dyslexia_bin counts:", df['dyslexia_bin'].value_counts().to_dict())

# Focus on dyslexic individuals
dyslexic = df[df['dyslexia_bin'] == 1].copy()
print(f"\nDyslexic participants: {dyslexic['uuid'].nunique()} unique, {len(dyslexic)} rows")

# Compare reading speed with/without reader_view for dyslexic individuals
dyslexic_rv_on = dyslexic[dyslexic['reader_view'] == 1]['speed']
dyslexic_rv_off = dyslexic[dyslexic['reader_view'] == 0]['speed']

print(f"\nDyslexic + Reader View ON: n={len(dyslexic_rv_on)}, mean={dyslexic_rv_on.mean():.2f}, median={dyslexic_rv_on.median():.2f}")
print(f"Dyslexic + Reader View OFF: n={len(dyslexic_rv_off)}, mean={dyslexic_rv_off.mean():.2f}, median={dyslexic_rv_off.median():.2f}")

# t-test
t_stat, p_val = stats.ttest_ind(dyslexic_rv_on, dyslexic_rv_off)
print(f"\nT-test (dyslexic, RV on vs off): t={t_stat:.4f}, p={p_val:.4f}")

# Mann-Whitney U test (non-parametric, speed is skewed)
u_stat, p_mw = stats.mannwhitneyu(dyslexic_rv_on, dyslexic_rv_off, alternative='two-sided')
print(f"Mann-Whitney U (dyslexic, RV on vs off): U={u_stat:.1f}, p={p_mw:.4f}")

# Also check for non-dyslexic as comparison
non_dyslexic = df[df['dyslexia_bin'] == 0]
nd_rv_on = non_dyslexic[non_dyslexic['reader_view'] == 1]['speed']
nd_rv_off = non_dyslexic[non_dyslexic['reader_view'] == 0]['speed']
t2, p2 = stats.ttest_ind(nd_rv_on, nd_rv_off)
print(f"\nNon-dyslexic + Reader View ON: n={len(nd_rv_on)}, mean={nd_rv_on.mean():.2f}")
print(f"Non-dyslexic + Reader View OFF: n={len(nd_rv_off)}, mean={nd_rv_off.mean():.2f}")
print(f"T-test (non-dyslexic, RV on vs off): t={t2:.4f}, p={p2:.4f}")

# Regression with interaction term: speed ~ reader_view * dyslexia_bin + controls
df_model = df[['speed', 'reader_view', 'dyslexia_bin', 'age', 'num_words']].dropna().copy()
df_model['log_speed'] = np.log1p(df_model['speed'])
df_model['interaction'] = df_model['reader_view'] * df_model['dyslexia_bin']

X = df_model[['reader_view', 'dyslexia_bin', 'interaction', 'age', 'num_words']]
X = sm.add_constant(X)
y = df_model['log_speed']
model = sm.OLS(y, X).fit()
print("\n--- OLS Regression (log speed) ---")
print(model.summary().tables[1])

# Effect size for dyslexic group
effect_size = (dyslexic_rv_on.mean() - dyslexic_rv_off.mean()) / dyslexic['speed'].std()
print(f"\nEffect size (Cohen's d, dyslexic RV on vs off): {effect_size:.4f}")

# Determine response
# Reader view improves speed if: mean speed is higher with reader_view ON for dyslexics
# AND the difference is statistically significant
direction_positive = dyslexic_rv_on.mean() > dyslexic_rv_off.mean()
significant = p_mw < 0.05

print(f"\nDirection positive (higher speed with RV): {direction_positive}")
print(f"Statistically significant (Mann-Whitney p<0.05): {significant}")

interaction_coef = model.params['interaction']
interaction_pval = model.pvalues['interaction']
rv_coef = model.params['reader_view']
rv_pval = model.pvalues['reader_view']
print(f"OLS interaction coef: {interaction_coef:.4f}, p={interaction_pval:.4f}")
print(f"OLS reader_view coef: {rv_coef:.4f}, p={rv_pval:.4f}")

# Score: if positive direction AND significant => high score
if direction_positive and significant:
    response = 70
    explanation = (
        f"For dyslexic individuals, reading speed is {'higher' if direction_positive else 'lower'} "
        f"with Reader View ON (mean={dyslexic_rv_on.mean():.1f}) vs OFF (mean={dyslexic_rv_off.mean():.1f}). "
        f"Mann-Whitney U test: p={p_mw:.4f} (significant). "
        f"T-test: p={p_val:.4f}. Effect size (Cohen's d): {effect_size:.3f}. "
        f"OLS interaction term (reader_view*dyslexia): coef={interaction_coef:.4f}, p={interaction_pval:.4f}. "
        f"The evidence supports that Reader View improves reading speed for dyslexic individuals, though the effect size is modest."
    )
elif direction_positive and not significant:
    response = 35
    explanation = (
        f"For dyslexic individuals, reading speed is higher with Reader View ON (mean={dyslexic_rv_on.mean():.1f}) "
        f"vs OFF (mean={dyslexic_rv_off.mean():.1f}), but not statistically significant "
        f"(Mann-Whitney p={p_mw:.4f}, t-test p={p_val:.4f}). Effect size: {effect_size:.3f}. "
        f"OLS interaction coef={interaction_coef:.4f}, p={interaction_pval:.4f}. "
        f"Weak or no evidence for improvement."
    )
else:
    response = 20
    explanation = (
        f"For dyslexic individuals, reading speed is NOT higher with Reader View ON (mean={dyslexic_rv_on.mean():.1f}) "
        f"vs OFF (mean={dyslexic_rv_off.mean():.1f}). Mann-Whitney p={p_mw:.4f}, t-test p={p_val:.4f}. "
        f"Effect size: {effect_size:.3f}. No evidence that Reader View improves reading speed for dyslexics."
    )

result = {"response": response, "explanation": explanation}
with open('conclusion.txt', 'w') as f:
    json.dump(result, f)

print(f"\nConclusion written: response={response}")
print(f"Explanation: {explanation}")
