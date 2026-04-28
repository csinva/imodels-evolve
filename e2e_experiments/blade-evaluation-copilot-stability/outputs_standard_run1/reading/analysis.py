#!/usr/bin/env python3
"""
Analysis script to determine if Reader View improves reading speed for individuals with dyslexia.
"""

import json
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Load the data
df = pd.read_csv('reading.csv')

print("="*80)
print("RESEARCH QUESTION: Does 'Reader View' improve reading speed for individuals with dyslexia?")
print("="*80)

# Data exploration
print("\n1. DATASET OVERVIEW")
print(f"Total observations: {len(df)}")
print(f"Total unique participants: {df['uuid'].nunique()}")
print(f"\nColumns: {df.columns.tolist()}")

# Check for missing values
print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

# Key variables summary
print("\n2. KEY VARIABLES SUMMARY")
print(f"\nReader View Distribution:")
print(df['reader_view'].value_counts())
print(f"\nDyslexia Status Distribution:")
print(df['dyslexia'].value_counts())
print(f"\nDyslexia Binary Distribution:")
print(df['dyslexia_bin'].value_counts())

# Speed statistics
print(f"\nReading Speed Statistics:")
print(df['speed'].describe())

# Filter data for analysis
# Focus on people with dyslexia (dyslexia_bin = 1)
df_dyslexia = df[df['dyslexia_bin'] == 1].copy()
df_no_dyslexia = df[df['dyslexia_bin'] == 0].copy()

print(f"\n3. SAMPLE SIZES")
print(f"Observations with dyslexia: {len(df_dyslexia)}")
print(f"Observations without dyslexia: {len(df_no_dyslexia)}")

# Analyze reading speed by reader view for dyslexic individuals
print("\n4. READING SPEED BY READER VIEW (DYSLEXIC INDIVIDUALS)")
dyslexia_no_reader = df_dyslexia[df_dyslexia['reader_view'] == 0]['speed']
dyslexia_with_reader = df_dyslexia[df_dyslexia['reader_view'] == 1]['speed']

print(f"\nWithout Reader View (n={len(dyslexia_no_reader)}):")
print(f"  Mean: {dyslexia_no_reader.mean():.2f} words/min")
print(f"  Median: {dyslexia_no_reader.median():.2f} words/min")
print(f"  Std: {dyslexia_no_reader.std():.2f}")

print(f"\nWith Reader View (n={len(dyslexia_with_reader)}):")
print(f"  Mean: {dyslexia_with_reader.mean():.2f} words/min")
print(f"  Median: {dyslexia_with_reader.median():.2f} words/min")
print(f"  Std: {dyslexia_with_reader.std():.2f}")

speed_diff = dyslexia_with_reader.mean() - dyslexia_no_reader.mean()
print(f"\nDifference in mean speed: {speed_diff:.2f} words/min")
print(f"Percentage change: {(speed_diff / dyslexia_no_reader.mean()) * 100:.2f}%")

# Statistical test: Paired t-test or independent t-test
# Check if data is paired (same participants with and without reader view)
print("\n5. STATISTICAL SIGNIFICANCE TEST")

# For dyslexic individuals: test if reader view affects speed
t_stat, p_value = stats.ttest_ind(dyslexia_with_reader, dyslexia_no_reader)
print(f"\nIndependent t-test for dyslexic individuals:")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_value:.4f}")
print(f"  Significant at α=0.05: {p_value < 0.05}")

# Also perform Mann-Whitney U test (non-parametric alternative)
u_stat, u_pvalue = stats.mannwhitneyu(dyslexia_with_reader, dyslexia_no_reader, alternative='two-sided')
print(f"\nMann-Whitney U test (non-parametric):")
print(f"  U-statistic: {u_stat:.4f}")
print(f"  p-value: {u_pvalue:.4f}")
print(f"  Significant at α=0.05: {u_pvalue < 0.05}")

# Effect size (Cohen's d)
pooled_std = np.sqrt(((len(dyslexia_no_reader) - 1) * dyslexia_no_reader.std()**2 + 
                       (len(dyslexia_with_reader) - 1) * dyslexia_with_reader.std()**2) / 
                      (len(dyslexia_no_reader) + len(dyslexia_with_reader) - 2))
cohens_d = speed_diff / pooled_std
print(f"\nEffect size (Cohen's d): {cohens_d:.4f}")

# Regression analysis controlling for other variables
print("\n6. REGRESSION ANALYSIS (Controlling for confounders)")

# Filter out rows with missing values for the regression
df_dyslexia_reg = df_dyslexia.dropna(subset=['speed', 'reader_view', 'age', 'num_words'])

# Linear regression with statsmodels for p-values
X = df_dyslexia_reg[['reader_view', 'age', 'num_words', 'correct_rate']]
X = sm.add_constant(X)
y = df_dyslexia_reg['speed']

model = sm.OLS(y, X).fit()
print("\nOLS Regression Results (Dyslexic individuals only):")
print(model.summary())

# Extract reader_view coefficient
reader_view_coef = model.params['reader_view']
reader_view_pvalue = model.pvalues['reader_view']
print(f"\nReader View coefficient: {reader_view_coef:.2f}")
print(f"Reader View p-value: {reader_view_pvalue:.4f}")
print(f"Significant at α=0.05: {reader_view_pvalue < 0.05}")

# Also check interaction effect between dyslexia and reader view
print("\n7. INTERACTION ANALYSIS (Dyslexia × Reader View)")
df_reg_full = df.dropna(subset=['speed', 'reader_view', 'dyslexia_bin', 'age', 'num_words'])
df_reg_full['dyslexia_x_reader'] = df_reg_full['dyslexia_bin'] * df_reg_full['reader_view']

X_full = df_reg_full[['reader_view', 'dyslexia_bin', 'dyslexia_x_reader', 'age', 'num_words', 'correct_rate']]
X_full = sm.add_constant(X_full)
y_full = df_reg_full['speed']

model_full = sm.OLS(y_full, X_full).fit()
print("\nOLS Regression with Interaction Term:")
print(model_full.summary())

interaction_coef = model_full.params['dyslexia_x_reader']
interaction_pvalue = model_full.pvalues['dyslexia_x_reader']
print(f"\nInteraction coefficient: {interaction_coef:.2f}")
print(f"Interaction p-value: {interaction_pvalue:.4f}")
print(f"Significant at α=0.05: {interaction_pvalue < 0.05}")

# CONCLUSION
print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

# Determine the response based on statistical significance
if p_value < 0.05 and speed_diff > 0:
    # Statistically significant positive effect
    if cohens_d > 0.5:
        response = 75  # Strong yes - large effect size
        explanation = f"Yes, Reader View significantly improves reading speed for dyslexic individuals. T-test p={p_value:.4f}, mean speed increased by {speed_diff:.1f} words/min ({(speed_diff / dyslexia_no_reader.mean()) * 100:.1f}%), Cohen's d={cohens_d:.2f} (medium-to-large effect)."
    else:
        response = 65  # Moderate yes - small-to-medium effect
        explanation = f"Yes, Reader View moderately improves reading speed for dyslexic individuals. T-test p={p_value:.4f}, mean speed increased by {speed_diff:.1f} words/min ({(speed_diff / dyslexia_no_reader.mean()) * 100:.1f}%), Cohen's d={cohens_d:.2f} (small-to-medium effect)."
elif p_value < 0.05 and speed_diff < 0:
    # Statistically significant negative effect
    response = 15  # Strong no - actually makes it worse
    explanation = f"No, Reader View actually decreases reading speed for dyslexic individuals. T-test p={p_value:.4f}, mean speed decreased by {abs(speed_diff):.1f} words/min ({abs(speed_diff / dyslexia_no_reader.mean()) * 100:.1f}%)."
else:
    # Not statistically significant
    if speed_diff > 0:
        response = 40  # Weak evidence, slight positive trend
        explanation = f"No strong evidence that Reader View improves reading speed for dyslexic individuals. Although mean speed increased by {speed_diff:.1f} words/min, the difference is not statistically significant (p={p_value:.4f})."
    else:
        response = 30  # Weak evidence, slight negative trend
        explanation = f"No strong evidence that Reader View improves reading speed for dyslexic individuals. Mean speed decreased by {abs(speed_diff):.1f} words/min, but the difference is not statistically significant (p={p_value:.4f})."

print(f"\nResponse (0-100): {response}")
print(f"Explanation: {explanation}")

# Write conclusion to file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n" + "="*80)
print("Analysis complete. Results written to conclusion.txt")
print("="*80)
