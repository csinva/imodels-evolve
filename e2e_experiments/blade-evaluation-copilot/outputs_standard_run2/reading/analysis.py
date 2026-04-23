import pandas as pd
import numpy as np
import json
from scipy import stats
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Load the data
df = pd.read_csv('reading.csv')

print("="*80)
print("DATA EXPLORATION")
print("="*80)
print(f"\nDataset shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())

print(f"\n\nBasic statistics:")
print(df.describe())

print(f"\n\nMissing values:")
print(df.isnull().sum())

# Focus on the research question: Does Reader View improve reading speed for individuals with dyslexia?
print("\n" + "="*80)
print("RESEARCH QUESTION: Does Reader View improve reading speed for dyslexia individuals?")
print("="*80)

# Check the distribution of dyslexia and reader_view
print(f"\n\nDyslexia distribution:")
print(df['dyslexia_bin'].value_counts())
print(f"\n\nReader View distribution:")
print(df['reader_view'].value_counts())

# Filter to only individuals with dyslexia
df_dyslexia = df[df['dyslexia_bin'] == 1].copy()
print(f"\n\nNumber of records with dyslexia: {len(df_dyslexia)}")

# Compare reading speed with and without reader view for dyslexic individuals
with_reader_view = df_dyslexia[df_dyslexia['reader_view'] == 1]['speed']
without_reader_view = df_dyslexia[df_dyslexia['reader_view'] == 0]['speed']

print(f"\n\nDyslexic individuals WITH Reader View:")
print(f"  Count: {len(with_reader_view)}")
print(f"  Mean speed: {with_reader_view.mean():.2f}")
print(f"  Median speed: {with_reader_view.median():.2f}")
print(f"  Std: {with_reader_view.std():.2f}")

print(f"\nDyslexic individuals WITHOUT Reader View:")
print(f"  Count: {len(without_reader_view)}")
print(f"  Mean speed: {without_reader_view.mean():.2f}")
print(f"  Median speed: {without_reader_view.median():.2f}")
print(f"  Std: {without_reader_view.std():.2f}")

# Calculate the difference
speed_diff = with_reader_view.mean() - without_reader_view.mean()
print(f"\n\nDifference in mean speed (with - without): {speed_diff:.2f}")
if speed_diff > 0:
    print("  -> Reading speed is HIGHER with Reader View")
else:
    print("  -> Reading speed is LOWER with Reader View")

# Statistical test: Independent t-test
print("\n" + "="*80)
print("STATISTICAL TESTS")
print("="*80)

# Remove outliers for more robust analysis (speeds > 99th percentile)
speed_99 = df_dyslexia['speed'].quantile(0.99)
with_reader_view_clean = with_reader_view[with_reader_view < speed_99]
without_reader_view_clean = without_reader_view[without_reader_view < speed_99]

print(f"\n\nAfter removing outliers (> 99th percentile: {speed_99:.2f}):")
print(f"  WITH Reader View: n={len(with_reader_view_clean)}, mean={with_reader_view_clean.mean():.2f}")
print(f"  WITHOUT Reader View: n={len(without_reader_view_clean)}, mean={without_reader_view_clean.mean():.2f}")

# T-test on cleaned data
t_stat, p_value = stats.ttest_ind(with_reader_view_clean, without_reader_view_clean)
print(f"\n\nIndependent t-test (cleaned data):")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_value:.4f}")
print(f"  Significant at α=0.05? {p_value < 0.05}")

# Mann-Whitney U test (non-parametric alternative)
u_stat, p_value_mw = stats.mannwhitneyu(with_reader_view_clean, without_reader_view_clean, alternative='two-sided')
print(f"\n\nMann-Whitney U test (non-parametric):")
print(f"  U-statistic: {u_stat:.4f}")
print(f"  p-value: {p_value_mw:.4f}")
print(f"  Significant at α=0.05? {p_value_mw < 0.05}")

# Regression analysis controlling for confounders
print("\n" + "="*80)
print("REGRESSION ANALYSIS (controlling for confounders)")
print("="*80)

# Prepare data for regression - only dyslexic individuals, remove extreme outliers
df_reg = df_dyslexia[df_dyslexia['speed'] < speed_99].copy()

# Create a regression model
# Control for: age, device, education, num_words, correct_rate
model_formula = 'speed ~ reader_view + age + C(device) + C(education) + num_words + correct_rate'
model = ols(model_formula, data=df_reg).fit()
print(f"\n\nOLS Regression Results:")
print(model.summary())

reader_view_coef = model.params['reader_view']
reader_view_pval = model.pvalues['reader_view']
print(f"\n\nReader View coefficient: {reader_view_coef:.4f}")
print(f"Reader View p-value: {reader_view_pval:.4f}")
print(f"Significant at α=0.05? {reader_view_pval < 0.05}")

# Effect size (Cohen's d)
pooled_std = np.sqrt(((len(with_reader_view_clean)-1)*with_reader_view_clean.std()**2 + 
                       (len(without_reader_view_clean)-1)*without_reader_view_clean.std()**2) / 
                      (len(with_reader_view_clean) + len(without_reader_view_clean) - 2))
cohens_d = (with_reader_view_clean.mean() - without_reader_view_clean.mean()) / pooled_std
print(f"\n\nEffect size (Cohen's d): {cohens_d:.4f}")
print(f"  Interpretation: ", end="")
if abs(cohens_d) < 0.2:
    print("negligible effect")
elif abs(cohens_d) < 0.5:
    print("small effect")
elif abs(cohens_d) < 0.8:
    print("medium effect")
else:
    print("large effect")

# Interaction analysis: Does the effect differ by dyslexia severity?
print("\n" + "="*80)
print("INTERACTION ANALYSIS: Reader View × Dyslexia Severity")
print("="*80)

df_dyslexia_full = df[df['dyslexia'] > 0].copy()
df_dyslexia_full = df_dyslexia_full[df_dyslexia_full['speed'] < speed_99].copy()

# Separate by dyslexia severity (1 vs 2)
for severity in [1, 2]:
    subset = df_dyslexia_full[df_dyslexia_full['dyslexia'] == severity]
    with_rv = subset[subset['reader_view'] == 1]['speed']
    without_rv = subset[subset['reader_view'] == 0]['speed']
    
    if len(with_rv) > 0 and len(without_rv) > 0:
        t, p = stats.ttest_ind(with_rv, without_rv)
        print(f"\n\nDyslexia severity {severity}:")
        print(f"  WITH RV: n={len(with_rv)}, mean={with_rv.mean():.2f}")
        print(f"  WITHOUT RV: n={len(without_rv)}, mean={without_rv.mean():.2f}")
        print(f"  t-test p-value: {p:.4f}")

# Final conclusion
print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

# Determine response based on statistical evidence
# We're looking at whether Reader View IMPROVES reading speed
# Improvement means higher speed with Reader View

is_significant = (p_value < 0.05) or (reader_view_pval < 0.05)
effect_direction_positive = (with_reader_view_clean.mean() > without_reader_view_clean.mean()) and (reader_view_coef > 0)

print(f"\n\nStatistical significance: {is_significant}")
print(f"Effect direction (positive = improvement): {effect_direction_positive}")
print(f"Effect size: {cohens_d:.4f}")

# Scoring logic:
# - If not significant: score 0-30 (no clear evidence)
# - If significant but negative effect: score 0-20 (evidence against)
# - If significant and positive effect: score based on effect size
#   - Small effect (d < 0.5): 60-75
#   - Medium effect (0.5 <= d < 0.8): 75-85
#   - Large effect (d >= 0.8): 85-100

if not is_significant:
    response = 15
    explanation = (f"There is no statistically significant evidence that Reader View improves "
                   f"reading speed for individuals with dyslexia. The t-test yielded p={p_value:.3f} "
                   f"and the regression analysis p={reader_view_pval:.3f}, both non-significant at α=0.05. "
                   f"The observed mean difference was {speed_diff:.2f} words/min with Cohen's d={cohens_d:.3f}, "
                   f"indicating a negligible to small effect that could be due to chance.")
elif effect_direction_positive:
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        response = 55
        explanation = (f"There is weak statistical evidence (p={p_value:.3f}) suggesting Reader View "
                       f"may improve reading speed for dyslexic individuals, but the effect size is "
                       f"negligible (Cohen's d={cohens_d:.3f}). The mean improvement was {speed_diff:.2f} "
                       f"words/min, which is not practically meaningful.")
    elif abs_d < 0.5:
        response = 70
        explanation = (f"There is statistically significant evidence (p={p_value:.3f}) that Reader View "
                       f"improves reading speed for individuals with dyslexia. The effect size is small "
                       f"(Cohen's d={cohens_d:.3f}) with a mean improvement of {speed_diff:.2f} words/min. "
                       f"Regression analysis controlling for confounders supports this finding (p={reader_view_pval:.3f}).")
    elif abs_d < 0.8:
        response = 80
        explanation = (f"There is strong statistically significant evidence (p={p_value:.3f}) that Reader View "
                       f"improves reading speed for individuals with dyslexia. The effect size is medium "
                       f"(Cohen's d={cohens_d:.3f}) with a mean improvement of {speed_diff:.2f} words/min. "
                       f"This improvement is both statistically significant and practically meaningful.")
    else:
        response = 90
        explanation = (f"There is very strong statistically significant evidence (p={p_value:.3f}) that Reader View "
                       f"substantially improves reading speed for individuals with dyslexia. The effect size is large "
                       f"(Cohen's d={cohens_d:.3f}) with a mean improvement of {speed_diff:.2f} words/min. "
                       f"This represents a practically significant improvement.")
else:
    response = 10
    explanation = (f"The evidence suggests that Reader View does NOT improve reading speed for "
                   f"individuals with dyslexia. In fact, dyslexic readers had a mean reading speed "
                   f"that was {-speed_diff:.2f} words/min LOWER with Reader View enabled. "
                   f"Statistical tests showed p={p_value:.3f}, and the effect, while possibly "
                   f"significant, is in the opposite direction of improvement.")

print(f"\n\nFinal Assessment:")
print(f"  Response score (0-100): {response}")
print(f"  Explanation: {explanation}")

# Write conclusion to file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n" + "="*80)
print("Analysis complete! conclusion.txt has been created.")
print("="*80)
