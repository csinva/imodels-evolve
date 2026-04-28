import json
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('reading.csv')

# Research Question: Does 'Reader View' improve reading speed for individuals with dyslexia?

print("=" * 80)
print("RESEARCH QUESTION: Does 'Reader View' improve reading speed for individuals with dyslexia?")
print("=" * 80)

# Data exploration
print("\nDataset shape:", df.shape)
print("\nKey variables for analysis:")
print("- reader_view: 0 (off) or 1 (on)")
print("- dyslexia_bin: 0 (no dyslexia) or 1 (has dyslexia)")
print("- speed: reading speed (words per time unit)")

print("\nReader View distribution:")
print(df['reader_view'].value_counts())

print("\nDyslexia distribution:")
print(df['dyslexia_bin'].value_counts())

# Filter for individuals with dyslexia
df_dyslexia = df[df['dyslexia_bin'] == 1].copy()

print(f"\nSample size with dyslexia: {len(df_dyslexia)}")
print(f"  - Without Reader View: {(df_dyslexia['reader_view'] == 0).sum()}")
print(f"  - With Reader View: {(df_dyslexia['reader_view'] == 1).sum()}")

# Summary statistics for speed by reader_view (for dyslexic readers)
print("\n" + "=" * 80)
print("READING SPEED STATISTICS (Dyslexic Readers Only)")
print("=" * 80)

speed_no_rv = df_dyslexia[df_dyslexia['reader_view'] == 0]['speed']
speed_with_rv = df_dyslexia[df_dyslexia['reader_view'] == 1]['speed']

print("\nWithout Reader View:")
print(f"  Mean: {speed_no_rv.mean():.2f}")
print(f"  Median: {speed_no_rv.median():.2f}")
print(f"  Std: {speed_no_rv.std():.2f}")
print(f"  N: {len(speed_no_rv)}")

print("\nWith Reader View:")
print(f"  Mean: {speed_with_rv.mean():.2f}")
print(f"  Median: {speed_with_rv.median():.2f}")
print(f"  Std: {speed_with_rv.std():.2f}")
print(f"  N: {len(speed_with_rv)}")

print(f"\nDifference in means: {speed_with_rv.mean() - speed_no_rv.mean():.2f}")
print(f"Percent change: {((speed_with_rv.mean() - speed_no_rv.mean()) / speed_no_rv.mean() * 100):.2f}%")

# Statistical test: Independent t-test
print("\n" + "=" * 80)
print("STATISTICAL TEST: Independent Samples t-test")
print("=" * 80)

# Check for outliers (remove extreme values above 10,000 for cleaner analysis)
speed_no_rv_clean = speed_no_rv[speed_no_rv < 10000]
speed_with_rv_clean = speed_with_rv[speed_with_rv < 10000]

print(f"\nAfter removing outliers (speed > 10,000):")
print(f"  Without Reader View: N = {len(speed_no_rv_clean)}")
print(f"  With Reader View: N = {len(speed_with_rv_clean)}")

t_stat, p_value = stats.ttest_ind(speed_with_rv_clean, speed_no_rv_clean)

print(f"\nt-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Significance level: α = 0.05")

if p_value < 0.05:
    print(f"Result: SIGNIFICANT (p < 0.05)")
else:
    print(f"Result: NOT SIGNIFICANT (p >= 0.05)")

# Effect size (Cohen's d)
pooled_std = np.sqrt(((len(speed_with_rv_clean) - 1) * speed_with_rv_clean.std()**2 + 
                      (len(speed_no_rv_clean) - 1) * speed_no_rv_clean.std()**2) / 
                     (len(speed_with_rv_clean) + len(speed_no_rv_clean) - 2))
cohens_d = (speed_with_rv_clean.mean() - speed_no_rv_clean.mean()) / pooled_std

print(f"\nCohen's d (effect size): {cohens_d:.4f}")
if abs(cohens_d) < 0.2:
    effect_interpretation = "negligible"
elif abs(cohens_d) < 0.5:
    effect_interpretation = "small"
elif abs(cohens_d) < 0.8:
    effect_interpretation = "medium"
else:
    effect_interpretation = "large"
print(f"Effect size interpretation: {effect_interpretation}")

# Regression analysis with controls
print("\n" + "=" * 80)
print("REGRESSION ANALYSIS (Controlling for Covariates)")
print("=" * 80)

# Prepare data for regression
df_reg = df_dyslexia[df_dyslexia['speed'] < 10000].copy()
df_reg = df_reg.dropna(subset=['speed', 'reader_view', 'age', 'num_words'])

# Create regression model
X = df_reg[['reader_view', 'age', 'num_words', 'scrolling_time']]
y = df_reg['speed']

X_with_const = sm.add_constant(X)
model = sm.OLS(y, X_with_const).fit()

print(model.summary())

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)

reader_view_coef = model.params['reader_view']
reader_view_pval = model.pvalues['reader_view']

print(f"\nReader View coefficient: {reader_view_coef:.2f}")
print(f"Reader View p-value: {reader_view_pval:.4f}")

if reader_view_pval < 0.05:
    if reader_view_coef > 0:
        print("\nReader View has a SIGNIFICANT POSITIVE effect on reading speed for dyslexic readers.")
        direction = "increases"
    else:
        print("\nReader View has a SIGNIFICANT NEGATIVE effect on reading speed for dyslexic readers.")
        direction = "decreases"
    print(f"When Reader View is enabled, reading speed {direction} by {abs(reader_view_coef):.2f} units (p < 0.05).")
else:
    print("\nReader View does NOT have a statistically significant effect on reading speed for dyslexic readers.")
    print(f"The effect is not significant (p = {reader_view_pval:.4f} >= 0.05).")

# Determine final response
print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

# Score based on p-value and effect direction
if p_value < 0.001:
    # Very strong evidence
    if speed_with_rv_clean.mean() > speed_no_rv_clean.mean():
        response_score = 95
        explanation = f"Very strong evidence (p < 0.001) that Reader View improves reading speed for individuals with dyslexia. T-test shows significant difference (t={t_stat:.2f}, p={p_value:.4f}). Mean speed increases from {speed_no_rv_clean.mean():.1f} to {speed_with_rv_clean.mean():.1f} ({((speed_with_rv_clean.mean() - speed_no_rv_clean.mean()) / speed_no_rv_clean.mean() * 100):.1f}% increase). Effect size: {effect_interpretation} (d={cohens_d:.2f})."
    else:
        response_score = 5
        explanation = f"Very strong evidence (p < 0.001) that Reader View does NOT improve reading speed for dyslexic individuals; it actually decreases it. T-test: t={t_stat:.2f}, p={p_value:.4f}. Mean speed decreases from {speed_no_rv_clean.mean():.1f} to {speed_with_rv_clean.mean():.1f}."
elif p_value < 0.01:
    # Strong evidence
    if speed_with_rv_clean.mean() > speed_no_rv_clean.mean():
        response_score = 85
        explanation = f"Strong evidence (p < 0.01) that Reader View improves reading speed for individuals with dyslexia. T-test: t={t_stat:.2f}, p={p_value:.4f}. Mean speed increases from {speed_no_rv_clean.mean():.1f} to {speed_with_rv_clean.mean():.1f} ({((speed_with_rv_clean.mean() - speed_no_rv_clean.mean()) / speed_no_rv_clean.mean() * 100):.1f}% increase). Effect size: {effect_interpretation}."
    else:
        response_score = 15
        explanation = f"Strong evidence (p < 0.01) that Reader View does NOT improve reading speed; it decreases it. T-test: t={t_stat:.2f}, p={p_value:.4f}. Mean speed decreases from {speed_no_rv_clean.mean():.1f} to {speed_with_rv_clean.mean():.1f}."
elif p_value < 0.05:
    # Moderate evidence
    if speed_with_rv_clean.mean() > speed_no_rv_clean.mean():
        response_score = 70
        explanation = f"Moderate evidence (p < 0.05) that Reader View improves reading speed for individuals with dyslexia. T-test: t={t_stat:.2f}, p={p_value:.4f}. Mean speed increases from {speed_no_rv_clean.mean():.1f} to {speed_with_rv_clean.mean():.1f}. Effect size: {effect_interpretation}."
    else:
        response_score = 30
        explanation = f"Moderate evidence (p < 0.05) that Reader View does NOT improve reading speed; there's a significant decrease. T-test: t={t_stat:.2f}, p={p_value:.4f}."
else:
    # Not significant
    response_score = 50
    explanation = f"No statistically significant evidence that Reader View improves reading speed for individuals with dyslexia. T-test not significant (t={t_stat:.2f}, p={p_value:.4f} >= 0.05). While mean speed changed from {speed_no_rv_clean.mean():.1f} to {speed_with_rv_clean.mean():.1f}, this difference could be due to chance."

print(f"\nResponse Score: {response_score}/100")
print(f"\nExplanation: {explanation}")

# Write conclusion to file
conclusion = {
    "response": response_score,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE - conclusion.txt written")
print("=" * 80)
