import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('reading.csv')

print("Shape:", df.shape)
print("\nBasic stats for speed:")
print(df['speed'].describe())

# Focus on dyslexic individuals
dyslexic = df[df['dyslexia_bin'] == 1].copy()
print(f"\nDyslexic participants rows: {len(dyslexic)}")
print(f"Reader view distribution among dyslexic:\n{dyslexic['reader_view'].value_counts()}")

# Compare reading speed: reader_view=1 vs reader_view=0 for dyslexic individuals
speed_rv1 = dyslexic[dyslexic['reader_view'] == 1]['speed'].dropna()
speed_rv0 = dyslexic[dyslexic['reader_view'] == 0]['speed'].dropna()

print(f"\nDyslexic, Reader View ON  - mean speed: {speed_rv1.mean():.2f}, n={len(speed_rv1)}")
print(f"Dyslexic, Reader View OFF - mean speed: {speed_rv0.mean():.2f}, n={len(speed_rv0)}")

# Remove extreme outliers (beyond 3 std)
threshold = df['speed'].mean() + 3 * df['speed'].std()
dyslexic_clean = dyslexic[dyslexic['speed'] <= threshold].copy()
speed_rv1_c = dyslexic_clean[dyslexic_clean['reader_view'] == 1]['speed'].dropna()
speed_rv0_c = dyslexic_clean[dyslexic_clean['reader_view'] == 0]['speed'].dropna()

print(f"\nAfter outlier removal:")
print(f"Dyslexic, Reader View ON  - mean speed: {speed_rv1_c.mean():.2f}, n={len(speed_rv1_c)}")
print(f"Dyslexic, Reader View OFF - mean speed: {speed_rv0_c.mean():.2f}, n={len(speed_rv0_c)}")

# t-test
t_stat, p_val = stats.ttest_ind(speed_rv1_c, speed_rv0_c)
print(f"\nIndependent t-test (dyslexic only, cleaned): t={t_stat:.4f}, p={p_val:.4f}")

# Also test on raw dyslexic data
t_stat_raw, p_val_raw = stats.ttest_ind(speed_rv1, speed_rv0)
print(f"Independent t-test (dyslexic only, raw): t={t_stat_raw:.4f}, p={p_val_raw:.4f}")

# Mann-Whitney U (non-parametric, robust)
u_stat, p_mw = stats.mannwhitneyu(speed_rv1_c, speed_rv0_c, alternative='two-sided')
print(f"Mann-Whitney U test (cleaned): U={u_stat:.1f}, p={p_mw:.4f}")

# OLS regression on dyslexic individuals: speed ~ reader_view + controls
reg_data = dyslexic_clean[['speed', 'reader_view', 'age', 'num_words']].dropna()
X = sm.add_constant(reg_data[['reader_view', 'age', 'num_words']])
y = reg_data['speed']
model = sm.OLS(y, X).fit()
print("\nOLS Regression (dyslexic subset):")
print(model.summary().tables[1])

rv_coef = model.params['reader_view']
rv_pval = model.pvalues['reader_view']
print(f"\nreader_view coef: {rv_coef:.4f}, p={rv_pval:.4f}")

# Determine conclusion
# Use p-value from t-test and effect direction
effect_direction = speed_rv1_c.mean() > speed_rv0_c.mean()
p_threshold = 0.05

if p_mw < p_threshold and effect_direction:
    response = 75
    explanation = (
        f"Reader View significantly improves reading speed for dyslexic individuals. "
        f"Mean speed with Reader View: {speed_rv1_c.mean():.1f} vs without: {speed_rv0_c.mean():.1f} wpm. "
        f"Mann-Whitney U test p={p_mw:.4f} (significant). "
        f"OLS regression: reader_view coef={rv_coef:.2f}, p={rv_pval:.4f}."
    )
elif p_mw < p_threshold and not effect_direction:
    response = 25
    explanation = (
        f"Reader View significantly decreases reading speed for dyslexic individuals. "
        f"Mean speed with Reader View: {speed_rv1_c.mean():.1f} vs without: {speed_rv0_c.mean():.1f} wpm. "
        f"Mann-Whitney U test p={p_mw:.4f} (significant). "
        f"OLS regression: reader_view coef={rv_coef:.2f}, p={rv_pval:.4f}."
    )
else:
    response = 35
    explanation = (
        f"No statistically significant improvement in reading speed for dyslexic individuals with Reader View. "
        f"Mean speed with Reader View: {speed_rv1_c.mean():.1f} vs without: {speed_rv0_c.mean():.1f} wpm. "
        f"Mann-Whitney U test p={p_mw:.4f} (not significant at 0.05). "
        f"OLS regression: reader_view coef={rv_coef:.2f}, p={rv_pval:.4f}."
    )

print(f"\nFinal response: {response}")
print(f"Explanation: {explanation}")

with open('conclusion.txt', 'w') as f:
    json.dump({"response": response, "explanation": explanation}, f)

print("\nconclustion.txt written successfully.")
