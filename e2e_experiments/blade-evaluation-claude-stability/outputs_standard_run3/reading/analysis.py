import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import json

df = pd.read_csv("reading.csv")

print("Shape:", df.shape)
print(df[['reader_view', 'dyslexia', 'dyslexia_bin', 'speed']].describe())

# Focus on dyslexic individuals only
dyslexic = df[df['dyslexia_bin'] == 1]
print(f"\nDyslexic participants rows: {len(dyslexic)}")

# Reading speed for dyslexic individuals: reader_view=1 vs reader_view=0
dys_rv1 = dyslexic[dyslexic['reader_view'] == 1]['speed'].dropna()
dys_rv0 = dyslexic[dyslexic['reader_view'] == 0]['speed'].dropna()

print(f"\nDyslexic + Reader View ON: n={len(dys_rv1)}, mean={dys_rv1.mean():.2f}, median={dys_rv1.median():.2f}")
print(f"Dyslexic + Reader View OFF: n={len(dys_rv0)}, mean={dys_rv0.mean():.2f}, median={dys_rv0.median():.2f}")

# t-test
t_stat, p_val = stats.ttest_ind(dys_rv1, dys_rv0, equal_var=False)
print(f"\nWelch t-test: t={t_stat:.4f}, p={p_val:.4f}")

# Mann-Whitney U (non-parametric, speed is skewed)
u_stat, p_mw = stats.mannwhitneyu(dys_rv1, dys_rv0, alternative='two-sided')
print(f"Mann-Whitney U: U={u_stat:.2f}, p={p_mw:.4f}")

# OLS regression: speed ~ reader_view * dyslexia_bin + controls
df_model = df[['speed', 'reader_view', 'dyslexia_bin', 'age', 'num_words']].dropna()
df_model['interaction'] = df_model['reader_view'] * df_model['dyslexia_bin']
X = sm.add_constant(df_model[['reader_view', 'dyslexia_bin', 'interaction', 'age', 'num_words']])
y = np.log1p(df_model['speed'])  # log-transform skewed speed
model = sm.OLS(y, X).fit()
print("\nOLS (log speed):")
print(model.summary().tables[1])

interaction_coef = model.params['interaction']
interaction_p = model.pvalues['interaction']
reader_view_coef = model.params['reader_view']
reader_view_p = model.pvalues['reader_view']

print(f"\nInteraction (reader_view x dyslexia_bin): coef={interaction_coef:.4f}, p={interaction_p:.4f}")
print(f"Reader view main effect: coef={reader_view_coef:.4f}, p={reader_view_p:.4f}")

# Decide score
# Primary question: does reader_view improve speed for dyslexic individuals?
mean_diff = dys_rv1.mean() - dys_rv0.mean()
direction_positive = mean_diff > 0

# Score logic:
# - If mean speed is higher with reader view AND statistically significant -> high score
# - If not significant -> low score
if p_mw < 0.05 and direction_positive:
    score = 75
    explanation = (
        f"Reader View is associated with higher reading speed for dyslexic individuals "
        f"(mean with RV={dys_rv1.mean():.1f} vs without={dys_rv0.mean():.1f} words/min). "
        f"Mann-Whitney U test: p={p_mw:.4f} (significant). "
        f"OLS interaction term (reader_view x dyslexia): coef={interaction_coef:.4f}, p={interaction_p:.4f}. "
        "Evidence supports that Reader View improves reading speed for individuals with dyslexia."
    )
elif p_mw < 0.05 and not direction_positive:
    score = 20
    explanation = (
        f"Reader View is associated with LOWER reading speed for dyslexic individuals "
        f"(mean with RV={dys_rv1.mean():.1f} vs without={dys_rv0.mean():.1f}). "
        f"Mann-Whitney U: p={p_mw:.4f}. Effect is significant but in the wrong direction."
    )
else:
    score = 30
    explanation = (
        f"No statistically significant difference in reading speed for dyslexic individuals "
        f"between Reader View ON (mean={dys_rv1.mean():.1f}) and OFF (mean={dys_rv0.mean():.1f}). "
        f"Mann-Whitney U: p={p_mw:.4f}, t-test p={p_val:.4f}. "
        f"OLS interaction coef={interaction_coef:.4f}, p={interaction_p:.4f}. "
        "Insufficient evidence that Reader View improves reading speed for individuals with dyslexia."
    )

result = {"response": score, "explanation": explanation}
with open("conclusion.txt", "w") as f:
    json.dump(result, f)

print(f"\nConclusion: score={score}")
print(f"Explanation: {explanation}")
print("conclusion.txt written.")
