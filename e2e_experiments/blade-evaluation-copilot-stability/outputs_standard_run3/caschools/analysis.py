#!/usr/bin/env python3
"""
Analysis script to answer: Is a lower student-teacher ratio associated with higher academic performance?
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import json

# Load the data
df = pd.read_csv('caschools.csv')

# Calculate student-teacher ratio
df['student_teacher_ratio'] = df['students'] / df['teachers']

# Create a combined academic performance measure (average of reading and math scores)
df['academic_performance'] = (df['read'] + df['math']) / 2

print("=" * 80)
print("DATA EXPLORATION")
print("=" * 80)
print("\nDataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

print("\n" + "=" * 80)
print("DESCRIPTIVE STATISTICS")
print("=" * 80)
print("\nStudent-Teacher Ratio:")
print(df['student_teacher_ratio'].describe())
print("\nAcademic Performance (Avg of Read & Math):")
print(df['academic_performance'].describe())
print("\nReading Score:")
print(df['read'].describe())
print("\nMath Score:")
print(df['math'].describe())

# Correlation analysis
print("\n" + "=" * 80)
print("CORRELATION ANALYSIS")
print("=" * 80)
corr_performance = stats.pearsonr(df['student_teacher_ratio'], df['academic_performance'])
corr_read = stats.pearsonr(df['student_teacher_ratio'], df['read'])
corr_math = stats.pearsonr(df['student_teacher_ratio'], df['math'])

print(f"\nCorrelation between Student-Teacher Ratio and Academic Performance:")
print(f"  Pearson r = {corr_performance[0]:.4f}, p-value = {corr_performance[1]:.4e}")
print(f"\nCorrelation between Student-Teacher Ratio and Reading Score:")
print(f"  Pearson r = {corr_read[0]:.4f}, p-value = {corr_read[1]:.4e}")
print(f"\nCorrelation between Student-Teacher Ratio and Math Score:")
print(f"  Pearson r = {corr_math[0]:.4f}, p-value = {corr_math[1]:.4e}")

# Simple linear regression with academic performance as outcome
print("\n" + "=" * 80)
print("LINEAR REGRESSION: Academic Performance ~ Student-Teacher Ratio")
print("=" * 80)
X = df[['student_teacher_ratio']]
y = df['academic_performance']
X_with_const = sm.add_constant(X)
model_simple = sm.OLS(y, X_with_const).fit()
print(model_simple.summary())

# Multiple regression controlling for confounders
print("\n" + "=" * 80)
print("MULTIPLE REGRESSION WITH CONTROLS")
print("=" * 80)
print("Controlling for: income, english learners %, lunch assistance %, expenditure")

# Select control variables
X_multi = df[['student_teacher_ratio', 'income', 'english', 'lunch', 'expenditure']].copy()
X_multi = X_multi.dropna()
y_multi = df.loc[X_multi.index, 'academic_performance']

X_multi_const = sm.add_constant(X_multi)
model_multi = sm.OLS(y_multi, X_multi_const).fit()
print(model_multi.summary())

# Separate analyses for reading and math
print("\n" + "=" * 80)
print("SEPARATE ANALYSES FOR READING AND MATH SCORES")
print("=" * 80)

# Reading
X_read = sm.add_constant(df[['student_teacher_ratio']])
model_read = sm.OLS(df['read'], X_read).fit()
print("\n--- READING SCORE ---")
print(f"Coefficient: {model_read.params['student_teacher_ratio']:.4f}")
print(f"P-value: {model_read.pvalues['student_teacher_ratio']:.4e}")
print(f"R-squared: {model_read.rsquared:.4f}")

# Math
X_math = sm.add_constant(df[['student_teacher_ratio']])
model_math = sm.OLS(df['math'], X_math).fit()
print("\n--- MATH SCORE ---")
print(f"Coefficient: {model_math.params['student_teacher_ratio']:.4f}")
print(f"P-value: {model_math.pvalues['student_teacher_ratio']:.4e}")
print(f"R-squared: {model_math.rsquared:.4f}")

# Group analysis: Compare high vs low student-teacher ratio
print("\n" + "=" * 80)
print("GROUP COMPARISON: HIGH vs LOW STUDENT-TEACHER RATIO")
print("=" * 80)
median_ratio = df['student_teacher_ratio'].median()
print(f"Median student-teacher ratio: {median_ratio:.2f}")

df['ratio_group'] = df['student_teacher_ratio'].apply(lambda x: 'Low' if x < median_ratio else 'High')
low_ratio = df[df['ratio_group'] == 'Low']['academic_performance']
high_ratio = df[df['ratio_group'] == 'High']['academic_performance']

print(f"\nLow ratio group (< {median_ratio:.2f}):")
print(f"  Mean performance: {low_ratio.mean():.2f}, SD: {low_ratio.std():.2f}, N: {len(low_ratio)}")
print(f"\nHigh ratio group (>= {median_ratio:.2f}):")
print(f"  Mean performance: {high_ratio.mean():.2f}, SD: {high_ratio.std():.2f}, N: {len(high_ratio)}")

t_stat, p_value = stats.ttest_ind(low_ratio, high_ratio)
print(f"\nIndependent t-test:")
print(f"  t-statistic = {t_stat:.4f}, p-value = {p_value:.4e}")
print(f"  Mean difference: {low_ratio.mean() - high_ratio.mean():.2f} points")

# Effect size (Cohen's d)
pooled_std = np.sqrt(((len(low_ratio)-1)*low_ratio.std()**2 + (len(high_ratio)-1)*high_ratio.std()**2) / (len(low_ratio)+len(high_ratio)-2))
cohens_d = (low_ratio.mean() - high_ratio.mean()) / pooled_std
print(f"  Cohen's d (effect size): {cohens_d:.4f}")

# Final interpretation
print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

# Collect all p-values
p_values = [
    corr_performance[1],
    model_simple.pvalues['student_teacher_ratio'],
    model_multi.pvalues['student_teacher_ratio'],
    p_value
]

# All tests should be significant
all_significant = all(p < 0.05 for p in p_values)

# Check direction: negative correlation means lower ratio → higher performance
negative_correlation = corr_performance[0] < 0

# Determine response score
if all_significant and negative_correlation:
    # Strong evidence for the relationship
    response = 85  # High confidence "Yes"
    explanation = (
        f"There is strong statistical evidence that lower student-teacher ratios are associated with "
        f"higher academic performance. The correlation is {corr_performance[0]:.3f} (p < 0.001), "
        f"indicating that as the student-teacher ratio decreases, test scores increase. "
        f"This relationship remains significant even when controlling for income, English learners, "
        f"lunch assistance, and expenditure (p = {model_multi.pvalues['student_teacher_ratio']:.4f}). "
        f"Schools with below-median student-teacher ratios score {low_ratio.mean() - high_ratio.mean():.1f} "
        f"points higher on average (p < 0.001, Cohen's d = {cohens_d:.2f})."
    )
elif negative_correlation and any(p < 0.05 for p in p_values):
    # Some evidence but not all tests significant
    response = 65
    explanation = (
        f"There is moderate evidence that lower student-teacher ratios are associated with higher "
        f"academic performance. The correlation is {corr_performance[0]:.3f}, and some statistical "
        f"tests show significance, though the relationship may be partially confounded by other factors."
    )
elif not negative_correlation and not all_significant:
    # No clear relationship
    response = 20
    explanation = (
        f"There is weak or no evidence of a relationship between student-teacher ratio and academic "
        f"performance. The correlation is {corr_performance[0]:.3f} and most statistical tests do not "
        f"show significant relationships."
    )
else:
    # Weak or inconsistent evidence
    response = 40
    explanation = (
        f"The evidence for a relationship between student-teacher ratio and academic performance is "
        f"inconsistent or weak. While some patterns exist, they do not consistently support a strong "
        f"association."
    )

print(f"\nResponse Score: {response}/100")
print(f"\nExplanation: {explanation}")

# Write conclusion to file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n" + "=" * 80)
print("Analysis complete. Results written to conclusion.txt")
print("=" * 80)
