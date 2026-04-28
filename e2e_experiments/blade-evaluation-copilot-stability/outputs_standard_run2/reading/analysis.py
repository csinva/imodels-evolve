import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import json

# Load the dataset
df = pd.read_csv('reading.csv')

# Initial exploration
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nDataset info:")
print(df.info())
print("\nSummary statistics:")
print(df.describe())

# Check the key variables
print("\n=== Key Variables ===")
print("Reader View distribution:")
print(df['reader_view'].value_counts())
print("\nDyslexia distribution:")
print(df['dyslexia_bin'].value_counts())
print("\nSpeed statistics:")
print(df['speed'].describe())

# Focus on the research question: Does Reader View improve reading speed for individuals with dyslexia?
# We'll analyze the interaction between reader_view and dyslexia_bin on speed

# Filter data for individuals with dyslexia
dyslexic_df = df[df['dyslexia_bin'] == 1].copy()
non_dyslexic_df = df[df['dyslexia_bin'] == 0].copy()

print("\n=== Analysis for Dyslexic Individuals ===")
print(f"Number of dyslexic individuals: {len(dyslexic_df)}")
print(f"Number of non-dyslexic individuals: {len(non_dyslexic_df)}")

# Compare reading speed with and without reader view for dyslexic individuals
dyslexic_with_rv = dyslexic_df[dyslexic_df['reader_view'] == 1]['speed']
dyslexic_without_rv = dyslexic_df[dyslexic_df['reader_view'] == 0]['speed']

print(f"\nDyslexic with Reader View: n={len(dyslexic_with_rv)}, mean={dyslexic_with_rv.mean():.2f}, std={dyslexic_with_rv.std():.2f}")
print(f"Dyslexic without Reader View: n={len(dyslexic_without_rv)}, mean={dyslexic_without_rv.mean():.2f}, std={dyslexic_without_rv.std():.2f}")

# Perform t-test for dyslexic individuals
if len(dyslexic_with_rv) > 0 and len(dyslexic_without_rv) > 0:
    t_stat_dyslexic, p_value_dyslexic = stats.ttest_ind(dyslexic_with_rv, dyslexic_without_rv)
    print(f"\nT-test for dyslexic individuals:")
    print(f"t-statistic: {t_stat_dyslexic:.4f}")
    print(f"p-value: {p_value_dyslexic:.4f}")
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(dyslexic_with_rv)-1)*dyslexic_with_rv.std()**2 + 
                          (len(dyslexic_without_rv)-1)*dyslexic_without_rv.std()**2) / 
                         (len(dyslexic_with_rv) + len(dyslexic_without_rv) - 2))
    cohens_d = (dyslexic_with_rv.mean() - dyslexic_without_rv.mean()) / pooled_std
    print(f"Cohen's d (effect size): {cohens_d:.4f}")

# For comparison, also analyze non-dyslexic individuals
non_dyslexic_with_rv = non_dyslexic_df[non_dyslexic_df['reader_view'] == 1]['speed']
non_dyslexic_without_rv = non_dyslexic_df[non_dyslexic_df['reader_view'] == 0]['speed']

print(f"\n=== For Comparison: Non-Dyslexic Individuals ===")
print(f"Non-dyslexic with Reader View: n={len(non_dyslexic_with_rv)}, mean={non_dyslexic_with_rv.mean():.2f}, std={non_dyslexic_with_rv.std():.2f}")
print(f"Non-dyslexic without Reader View: n={len(non_dyslexic_without_rv)}, mean={non_dyslexic_without_rv.mean():.2f}, std={non_dyslexic_without_rv.std():.2f}")

if len(non_dyslexic_with_rv) > 0 and len(non_dyslexic_without_rv) > 0:
    t_stat_non_dyslexic, p_value_non_dyslexic = stats.ttest_ind(non_dyslexic_with_rv, non_dyslexic_without_rv)
    print(f"\nT-test for non-dyslexic individuals:")
    print(f"t-statistic: {t_stat_non_dyslexic:.4f}")
    print(f"p-value: {p_value_non_dyslexic:.4f}")

# Regression analysis with interaction term
print("\n=== Regression Analysis with Interaction ===")
# Create interaction term
df['reader_dyslexia_interaction'] = df['reader_view'] * df['dyslexia_bin']

# Prepare data for regression
X = df[['reader_view', 'dyslexia_bin', 'reader_dyslexia_interaction']].copy()
y = df['speed'].copy()

# Remove any NaN values
mask = ~(X.isna().any(axis=1) | y.isna())
X = X[mask]
y = y[mask]

# Add constant for statsmodels
X_with_const = sm.add_constant(X)

# Fit OLS model
model = sm.OLS(y, X_with_const).fit()
print(model.summary())

# Extract key results
interaction_coef = model.params['reader_dyslexia_interaction']
interaction_pvalue = model.pvalues['reader_dyslexia_interaction']

print(f"\n=== Key Findings ===")
print(f"Interaction coefficient (reader_view × dyslexia_bin): {interaction_coef:.4f}")
print(f"Interaction p-value: {interaction_pvalue:.4f}")

# Determine the conclusion
# Higher speed values indicate faster reading
# If reader view improves speed for dyslexic individuals, we expect:
# - Positive effect on speed for dyslexic individuals with reader view
# - Significant t-test or significant interaction term

alpha = 0.05
significant = p_value_dyslexic < alpha if len(dyslexic_with_rv) > 0 and len(dyslexic_without_rv) > 0 else False
speed_improvement = dyslexic_with_rv.mean() > dyslexic_without_rv.mean() if len(dyslexic_with_rv) > 0 and len(dyslexic_without_rv) > 0 else False

print(f"\nStatistically significant difference: {significant} (p={p_value_dyslexic:.4f})")
print(f"Speed improvement with Reader View: {speed_improvement}")
print(f"Mean difference: {dyslexic_with_rv.mean() - dyslexic_without_rv.mean():.2f}")

# Determine Likert score
# 0 = strong "No", 100 = strong "Yes"
if significant and speed_improvement:
    # Significant improvement
    if abs(cohens_d) > 0.5:  # Medium to large effect
        likert_score = 80  # Strong Yes
        explanation = f"Yes, Reader View significantly improves reading speed for individuals with dyslexia (p={p_value_dyslexic:.4f}). Dyslexic individuals read {dyslexic_with_rv.mean():.1f} words/min with Reader View vs {dyslexic_without_rv.mean():.1f} words/min without it, showing a statistically significant improvement with Cohen's d={cohens_d:.2f}."
    else:  # Small effect
        likert_score = 65  # Moderate Yes
        explanation = f"Yes, Reader View shows a statistically significant but modest improvement in reading speed for individuals with dyslexia (p={p_value_dyslexic:.4f}). The effect size is small (Cohen's d={cohens_d:.2f})."
elif significant and not speed_improvement:
    # Significant but in wrong direction (slower)
    likert_score = 20  # Strong No
    explanation = f"No, Reader View does not improve reading speed for individuals with dyslexia. In fact, it significantly decreases speed (p={p_value_dyslexic:.4f}). Dyslexic individuals read faster without Reader View."
elif not significant and speed_improvement:
    # Not significant but shows improvement trend
    likert_score = 45  # Weak Yes/Neutral
    explanation = f"The data shows a trend toward improved reading speed with Reader View for dyslexic individuals (mean difference: {dyslexic_with_rv.mean() - dyslexic_without_rv.mean():.1f} words/min), but the difference is not statistically significant (p={p_value_dyslexic:.4f})."
else:
    # Not significant and no improvement
    likert_score = 30  # No
    explanation = f"No, the data does not support that Reader View improves reading speed for individuals with dyslexia. The difference is not statistically significant (p={p_value_dyslexic:.4f})."

print(f"\n=== CONCLUSION ===")
print(f"Likert Score: {likert_score}")
print(f"Explanation: {explanation}")

# Write conclusion to file
conclusion = {
    "response": likert_score,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\nConclusion written to conclusion.txt")
