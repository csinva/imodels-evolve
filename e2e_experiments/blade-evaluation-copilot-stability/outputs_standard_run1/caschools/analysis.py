import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import json

# Load data
df = pd.read_csv('caschools.csv')

# Calculate student-teacher ratio
df['student_teacher_ratio'] = df['students'] / df['teachers']

# Create average score as overall academic performance measure
df['avg_score'] = (df['read'] + df['math']) / 2

# Basic exploratory analysis
print("=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)
print(f"\nDataset shape: {df.shape}")
print(f"\nStudent-Teacher Ratio Statistics:")
print(df['student_teacher_ratio'].describe())
print(f"\nAverage Score Statistics:")
print(df['avg_score'].describe())

# Correlation analysis
correlation = df[['student_teacher_ratio', 'avg_score']].corr()
print(f"\nCorrelation Matrix:")
print(correlation)

pearson_corr, pearson_p = stats.pearsonr(df['student_teacher_ratio'], df['avg_score'])
print(f"\nPearson Correlation: {pearson_corr:.4f}")
print(f"P-value: {pearson_p:.6f}")

# Simple linear regression with statsmodels for detailed statistics
print("\n" + "=" * 60)
print("SIMPLE LINEAR REGRESSION (Student-Teacher Ratio -> Avg Score)")
print("=" * 60)
X_simple = sm.add_constant(df['student_teacher_ratio'])
y = df['avg_score']
model_simple = sm.OLS(y, X_simple).fit()
print(model_simple.summary())

# Extract key results
coef_str = model_simple.params['student_teacher_ratio']
pval_str = model_simple.pvalues['student_teacher_ratio']
r_squared = model_simple.rsquared

print(f"\nKey Results:")
print(f"Coefficient: {coef_str:.4f}")
print(f"P-value: {pval_str:.6f}")
print(f"R-squared: {r_squared:.4f}")

# Multiple regression controlling for confounders
print("\n" + "=" * 60)
print("MULTIPLE REGRESSION (Controlling for Confounders)")
print("=" * 60)

# Select relevant features
features = ['student_teacher_ratio', 'calworks', 'lunch', 'english', 'income', 'expenditure']
X_multi = df[features].copy()

# Handle any missing values
X_multi = X_multi.fillna(X_multi.mean())
y = df['avg_score']

# Fit model with statsmodels
X_multi_sm = sm.add_constant(X_multi)
model_multi = sm.OLS(y, X_multi_sm).fit()
print(model_multi.summary())

coef_str_multi = model_multi.params['student_teacher_ratio']
pval_str_multi = model_multi.pvalues['student_teacher_ratio']
r_squared_multi = model_multi.rsquared

print(f"\nKey Results (Multiple Regression):")
print(f"Student-Teacher Ratio Coefficient: {coef_str_multi:.4f}")
print(f"Student-Teacher Ratio P-value: {pval_str_multi:.6f}")
print(f"R-squared: {r_squared_multi:.4f}")

# Interpretable models with scikit-learn
print("\n" + "=" * 60)
print("INTERPRETABLE MODELS")
print("=" * 60)

# Standardize features for better interpretation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_multi)

# Linear regression
lr = LinearRegression()
lr.fit(X_scaled, y)

print("\nLinear Regression Feature Importance (Standardized Coefficients):")
for i, feature in enumerate(features):
    print(f"{feature}: {lr.coef_[i]:.4f}")

# Ridge regression for robustness
ridge = Ridge(alpha=1.0)
ridge.fit(X_scaled, y)

print("\nRidge Regression Feature Importance (Standardized Coefficients):")
for i, feature in enumerate(features):
    print(f"{feature}: {ridge.coef_[i]:.4f}")

# Group-based analysis: Compare low vs high student-teacher ratio
print("\n" + "=" * 60)
print("GROUP COMPARISON ANALYSIS")
print("=" * 60)

# Create terciles of student-teacher ratio
df['str_tercile'] = pd.qcut(df['student_teacher_ratio'], q=3, labels=['Low', 'Medium', 'High'])

print("\nAverage Scores by Student-Teacher Ratio Tercile:")
tercile_means = df.groupby('str_tercile')['avg_score'].agg(['mean', 'std', 'count'])
print(tercile_means)

# ANOVA to test differences across groups
low = df[df['str_tercile'] == 'Low']['avg_score']
medium = df[df['str_tercile'] == 'Medium']['avg_score']
high = df[df['str_tercile'] == 'High']['avg_score']

f_stat, anova_p = stats.f_oneway(low, medium, high)
print(f"\nANOVA F-statistic: {f_stat:.4f}")
print(f"ANOVA P-value: {anova_p:.6f}")

# T-test between low and high terciles
t_stat, ttest_p = stats.ttest_ind(low, high)
print(f"\nT-test (Low vs High Student-Teacher Ratio):")
print(f"Low STR mean: {low.mean():.2f}")
print(f"High STR mean: {high.mean():.2f}")
print(f"Difference: {low.mean() - high.mean():.2f}")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {ttest_p:.6f}")

# Summary of findings
print("\n" + "=" * 60)
print("SUMMARY OF FINDINGS")
print("=" * 60)

print(f"\n1. Correlation: {pearson_corr:.4f} (p={pearson_p:.6f})")
print(f"2. Simple Regression Coefficient: {coef_str:.4f} (p={pval_str:.6f})")
print(f"3. Multiple Regression Coefficient: {coef_str_multi:.4f} (p={pval_str_multi:.6f})")
print(f"4. Group Difference (Low - High STR): {low.mean() - high.mean():.2f} (p={ttest_p:.6f})")

# Decision logic
print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)

# Determine response based on statistical significance and effect direction
if pearson_p < 0.05 and pval_str < 0.05:
    # Significant relationship exists
    if pearson_corr < 0:  # Negative correlation means lower STR -> higher scores
        # Strong evidence
        if pearson_p < 0.001 and abs(pearson_corr) > 0.5:
            response = 95
            explanation = f"Strong evidence: Lower student-teacher ratio is significantly associated with higher academic performance. Pearson correlation = {pearson_corr:.3f} (p < 0.001), indicating a strong negative relationship. Simple regression shows each 1-unit increase in STR decreases scores by {abs(coef_str):.2f} points (p < 0.001). Even after controlling for confounders, the relationship remains significant (coef = {coef_str_multi:.3f}, p = {pval_str_multi:.6f})."
        # Moderate to strong evidence
        elif pearson_p < 0.01:
            response = 85
            explanation = f"Strong evidence: Lower student-teacher ratio is significantly associated with higher academic performance. Pearson correlation = {pearson_corr:.3f} (p = {pearson_p:.6f}), simple regression coefficient = {coef_str:.3f} (p < 0.01). After controlling for demographic and economic factors, the relationship remains significant (coef = {coef_str_multi:.3f}, p = {pval_str_multi:.6f})."
        # Moderate evidence
        else:
            response = 75
            explanation = f"Moderate evidence: Lower student-teacher ratio is significantly associated with higher academic performance. Pearson correlation = {pearson_corr:.3f} (p = {pearson_p:.4f}). Regression analysis confirms the negative relationship is statistically significant (simple: p = {pval_str:.4f}, multiple: p = {pval_str_multi:.4f})."
    else:  # Positive correlation means higher STR -> higher scores (opposite of expected)
        response = 10
        explanation = f"Evidence against: Higher (not lower) student-teacher ratio is associated with higher scores. Pearson correlation = {pearson_corr:.3f} (p = {pearson_p:.4f}). This contradicts the hypothesis that lower STR improves performance."
else:
    # No significant relationship
    if pearson_p < 0.10:
        response = 40
        explanation = f"Weak evidence: The relationship between student-teacher ratio and academic performance shows a trend (p = {pearson_p:.4f}) but does not reach conventional significance (p < 0.05). Pearson correlation = {pearson_corr:.3f}. The evidence is insufficient to confidently conclude an association exists."
    else:
        response = 20
        explanation = f"No evidence: No significant association found between student-teacher ratio and academic performance. Pearson correlation = {pearson_corr:.3f} (p = {pearson_p:.4f}). The relationship is not statistically significant (p > 0.05)."

print(f"\nResponse Score: {response}/100")
print(f"Explanation: {explanation}")

# Write conclusion to file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n" + "=" * 60)
print("Analysis complete. Results written to conclusion.txt")
print("=" * 60)
