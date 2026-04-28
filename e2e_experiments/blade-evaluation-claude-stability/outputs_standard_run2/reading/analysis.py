import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import json

df = pd.read_csv("reading.csv")

print("Shape:", df.shape)
print(df.head())
print(df.dtypes)
print(df[["reader_view", "dyslexia", "dyslexia_bin", "speed"]].describe())

# Focus on dyslexic readers only
dyslexic = df[df["dyslexia_bin"] == 1].copy()
print(f"\nDyslexic readers: {len(dyslexic)} rows")

reader_view_on = dyslexic[dyslexic["reader_view"] == 1]["speed"]
reader_view_off = dyslexic[dyslexic["reader_view"] == 0]["speed"]

print(f"Reader view ON (dyslexic): n={len(reader_view_on)}, mean={reader_view_on.mean():.2f}, median={reader_view_on.median():.2f}")
print(f"Reader view OFF (dyslexic): n={len(reader_view_off)}, mean={reader_view_off.mean():.2f}, median={reader_view_off.median():.2f}")

# t-test for dyslexic readers: reader_view vs speed
t_stat, p_val = stats.ttest_ind(reader_view_on, reader_view_off)
print(f"\nt-test (dyslexic only): t={t_stat:.4f}, p={p_val:.4f}")

# Mann-Whitney U test (non-parametric, robust to outliers)
u_stat, p_mw = stats.mannwhitneyu(reader_view_on, reader_view_off, alternative="two-sided")
print(f"Mann-Whitney U (dyslexic only): U={u_stat:.4f}, p={p_mw:.4f}")

# OLS regression: speed ~ reader_view * dyslexia_bin (interaction)
df_model = df.dropna(subset=["speed", "reader_view", "dyslexia_bin"]).copy()
df_model["interaction"] = df_model["reader_view"] * df_model["dyslexia_bin"]
X = sm.add_constant(df_model[["reader_view", "dyslexia_bin", "interaction"]])
y = df_model["speed"]
ols = sm.OLS(y, X).fit()
print("\nOLS regression summary:")
print(ols.summary())

# Direction check
diff_mean = reader_view_on.mean() - reader_view_off.mean()
diff_median = reader_view_on.median() - reader_view_off.median()
print(f"\nMean speed difference (ON - OFF) for dyslexic: {diff_mean:.2f}")
print(f"Median speed difference (ON - OFF) for dyslexic: {diff_median:.2f}")

# Determine response score
# Higher speed = faster reading = improvement
significant = p_mw < 0.05
improves = diff_mean > 0

if significant and improves:
    response = 75
    explanation = (
        f"Reader View significantly improves reading speed for individuals with dyslexia. "
        f"Dyslexic readers with Reader View ON had a mean speed of {reader_view_on.mean():.1f} words/min "
        f"vs {reader_view_off.mean():.1f} words/min without it (difference: {diff_mean:.1f}). "
        f"Mann-Whitney U test: p={p_mw:.4f} (significant at alpha=0.05). "
        f"The OLS interaction term (reader_view x dyslexia_bin) had coefficient "
        f"{ols.params.get('interaction', float('nan')):.2f} with p={ols.pvalues.get('interaction', float('nan')):.4f}."
    )
elif significant and not improves:
    response = 20
    explanation = (
        f"Reader View does NOT improve reading speed for dyslexic individuals — it appears to reduce it. "
        f"Mean speed ON: {reader_view_on.mean():.1f}, OFF: {reader_view_off.mean():.1f} (difference: {diff_mean:.1f}). "
        f"Mann-Whitney U test: p={p_mw:.4f} (significant). "
        f"OLS interaction p={ols.pvalues.get('interaction', float('nan')):.4f}."
    )
else:
    response = 40
    explanation = (
        f"No statistically significant improvement in reading speed for dyslexic individuals with Reader View. "
        f"Mean speed ON: {reader_view_on.mean():.1f}, OFF: {reader_view_off.mean():.1f} (difference: {diff_mean:.1f}). "
        f"Mann-Whitney U test: p={p_mw:.4f} (not significant at alpha=0.05). "
        f"OLS interaction p={ols.pvalues.get('interaction', float('nan')):.4f}."
    )

result = {"response": response, "explanation": explanation}
with open("conclusion.txt", "w") as f:
    json.dump(result, f)

print("\nconclusion.txt written.")
print(json.dumps(result, indent=2))
