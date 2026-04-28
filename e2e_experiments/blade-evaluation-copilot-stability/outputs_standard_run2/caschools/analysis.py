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
df['str'] = df['students'] / df['teachers']

# Create overall academic performance measure (average of reading and math scores)
df['academic_performance'] = (df['read'] + df['math']) / 2

print("=" * 80)
print("DATA EXPLORATION")
print("=" * 80)

print("\nDataset Shape:", df.shape)
print("\nFirst few rows:")
print(df[['students', 'teachers', 'str', 'read', 'math', 'academic_performance']].head())

print("\nSummary Statistics:")
print(df[['str', 'read', 'math', 'academic_performance']].describe())

print("\nStudent-Teacher Ratio Statistics:")
print(f"Mean: {df['str'].mean():.2f}")
print(f"Median: {df['str'].median():.2f}")
print(f"Min: {df['str'].min():.2f}")
print(f"Max: {df['str'].max():.2f}")

print("\n" + "=" * 80)
print("CORRELATION ANALYSIS")
print("=" * 80)

# Correlation between student-teacher ratio and test scores
corr_str_read = df['str'].corr(df['read'])
corr_str_math = df['str'].corr(df['math'])
corr_str_performance = df['str'].corr(df['academic_performance'])

print(f"\nCorrelation between STR and Reading Score: {corr_str_read:.4f}")
print(f"Correlation between STR and Math Score: {corr_str_math:.4f}")
print(f"Correlation between STR and Academic Performance: {corr_str_performance:.4f}")

# Pearson correlation test for statistical significance
corr_coef_read, p_value_read = stats.pearsonr(df['str'], df['read'])
corr_coef_math, p_value_math = stats.pearsonr(df['str'], df['math'])
corr_coef_performance, p_value_performance = stats.pearsonr(df['str'], df['academic_performance'])

print(f"\nPearson correlation test (STR vs Reading):")
print(f"  Correlation: {corr_coef_read:.4f}, p-value: {p_value_read:.4e}")
print(f"  Significant: {p_value_read < 0.05}")

print(f"\nPearson correlation test (STR vs Math):")
print(f"  Correlation: {corr_coef_math:.4f}, p-value: {p_value_math:.4e}")
print(f"  Significant: {p_value_math < 0.05}")

print(f"\nPearson correlation test (STR vs Academic Performance):")
print(f"  Correlation: {corr_coef_performance:.4f}, p-value: {p_value_performance:.4e}")
print(f"  Significant: {p_value_performance < 0.05}")

print("\n" + "=" * 80)
print("LINEAR REGRESSION ANALYSIS (Simple)")
print("=" * 80)

# Simple linear regression: Academic Performance ~ Student-Teacher Ratio
X_simple = df[['str']].values
y = df['academic_performance'].values

# Fit simple linear regression
lr_simple = LinearRegression()
lr_simple.fit(X_simple, y)

print(f"\nSimple Linear Regression: Academic Performance ~ STR")
print(f"  Coefficient (STR): {lr_simple.coef_[0]:.4f}")
print(f"  Intercept: {lr_simple.intercept_:.4f}")
print(f"  R-squared: {lr_simple.score(X_simple, y):.4f}")

# Using statsmodels for p-values
X_sm = sm.add_constant(df['str'])
model_simple = sm.OLS(df['academic_performance'], X_sm).fit()
print("\nStatsmodels OLS Regression Summary (STR only):")
print(model_simple.summary())

print("\n" + "=" * 80)
print("CONTROLLING FOR CONFOUNDERS - MULTIPLE REGRESSION")
print("=" * 80)

# Multiple regression controlling for socioeconomic factors
# Include income, english learners, lunch (poverty indicator)
control_vars = ['str', 'income', 'english', 'lunch']
df_clean = df[control_vars + ['academic_performance']].dropna()

X_multi = sm.add_constant(df_clean[control_vars])
y_multi = df_clean['academic_performance']

model_multi = sm.OLS(y_multi, X_multi).fit()
print("\nMultiple Regression: Academic Performance ~ STR + Controls")
print(model_multi.summary())

print("\n" + "=" * 80)
print("GROUP COMPARISON: LOW vs HIGH STR")
print("=" * 80)

# Divide schools into low and high student-teacher ratio groups
str_median = df['str'].median()
df['str_group'] = df['str'].apply(lambda x: 'Low STR' if x < str_median else 'High STR')

low_str_performance = df[df['str_group'] == 'Low STR']['academic_performance']
high_str_performance = df[df['str_group'] == 'High STR']['academic_performance']

print(f"\nMedian STR: {str_median:.2f}")
print(f"\nLow STR Group (< {str_median:.2f}):")
print(f"  N = {len(low_str_performance)}")
print(f"  Mean Performance: {low_str_performance.mean():.2f}")
print(f"  Std: {low_str_performance.std():.2f}")

print(f"\nHigh STR Group (>= {str_median:.2f}):")
print(f"  N = {len(high_str_performance)}")
print(f"  Mean Performance: {high_str_performance.mean():.2f}")
print(f"  Std: {high_str_performance.std():.2f}")

# T-test comparing low vs high STR groups
t_stat, t_pvalue = stats.ttest_ind(low_str_performance, high_str_performance)
print(f"\nIndependent t-test:")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {t_pvalue:.4e}")
print(f"  Significant: {t_pvalue < 0.05}")
print(f"  Mean difference: {low_str_performance.mean() - high_str_performance.mean():.2f} points")

print("\n" + "=" * 80)
print("INTERPRETATION AND CONCLUSION")
print("=" * 80)

# Determine conclusion based on statistical evidence
evidence = []

# Check correlation
if p_value_performance < 0.05:
    evidence.append("significant_correlation")
    print(f"\n1. CORRELATION: Significant negative correlation (r={corr_coef_performance:.3f}, p={p_value_performance:.4e})")
    print("   Lower student-teacher ratio is associated with higher academic performance.")
else:
    print(f"\n1. CORRELATION: No significant correlation (p={p_value_performance:.4f})")

# Check simple regression
if model_simple.pvalues['str'] < 0.05:
    evidence.append("significant_simple_regression")
    print(f"\n2. SIMPLE REGRESSION: STR coefficient is significant (coef={model_simple.params['str']:.3f}, p={model_simple.pvalues['str']:.4e})")
    print("   Each additional student per teacher is associated with lower test scores.")
else:
    print(f"\n2. SIMPLE REGRESSION: STR coefficient not significant (p={model_simple.pvalues['str']:.4f})")

# Check multiple regression
if model_multi.pvalues['str'] < 0.05:
    evidence.append("significant_after_controls")
    print(f"\n3. MULTIPLE REGRESSION: STR remains significant after controlling for confounders")
    print(f"   (coef={model_multi.params['str']:.3f}, p={model_multi.pvalues['str']:.4e})")
    print("   The effect persists even when accounting for income, English learners, and poverty.")
else:
    print(f"\n3. MULTIPLE REGRESSION: STR not significant after controls (p={model_multi.pvalues['str']:.4f})")
    print("   The relationship may be driven by confounding factors.")

# Check group comparison
if t_pvalue < 0.05:
    evidence.append("significant_group_difference")
    print(f"\n4. GROUP COMPARISON: Significant difference between low and high STR groups")
    print(f"   (t={t_stat:.3f}, p={t_pvalue:.4e})")
    print(f"   Schools with lower STR have {abs(low_str_performance.mean() - high_str_performance.mean()):.1f} points higher scores on average.")
else:
    print(f"\n4. GROUP COMPARISON: No significant difference (p={t_pvalue:.4f})")

# Calculate response score (0-100 Likert scale)
# Strong evidence = 100, No evidence = 0
num_significant = len(evidence)

if num_significant == 0:
    response_score = 10  # Very weak "No"
    explanation = "No statistically significant relationship found between student-teacher ratio and academic performance in any of the tests conducted."
elif num_significant == 1:
    response_score = 40  # Weak "No" to neutral
    explanation = "Limited evidence of a relationship. Only one statistical test showed significance, suggesting the association is weak or inconsistent."
elif num_significant == 2:
    response_score = 65  # Moderate "Yes"
    explanation = "Moderate evidence supporting the relationship. Multiple statistical tests show significance, but not all methods converge."
elif num_significant == 3:
    response_score = 85  # Strong "Yes"
    explanation = f"Strong evidence: Lower student-teacher ratio is significantly associated with higher academic performance (correlation r={corr_coef_performance:.3f}, p<0.05). This relationship holds in simple regression, multiple regression, and group comparisons."
else:  # 4 significant tests
    response_score = 95  # Very strong "Yes"
    explanation = f"Very strong evidence: All statistical tests confirm a significant negative relationship between student-teacher ratio and academic performance. Lower STR is associated with higher test scores (r={corr_coef_performance:.3f}, p<0.001), and this effect persists after controlling for socioeconomic factors. Schools with lower STR score approximately {abs(low_str_performance.mean() - high_str_performance.mean()):.1f} points higher."

# Adjust based on effect size
effect_size = abs(corr_coef_performance)
if effect_size < 0.1:
    response_score = max(0, response_score - 30)  # Very weak effect
elif effect_size < 0.3:
    response_score = max(0, response_score - 10)  # Weak effect

print("\n" + "=" * 80)
print("FINAL CONCLUSION")
print("=" * 80)
print(f"\nResponse Score: {response_score}/100")
print(f"Explanation: {explanation}")

# Write conclusion to file
conclusion = {
    "response": response_score,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\nConclusion written to conclusion.txt")
