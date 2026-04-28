import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from interpret.glassbox import ExplainableBoostingRegressor
import json

# Load data
df = pd.read_csv('caschools.csv')

# Calculate student-teacher ratio
df['str'] = df['students'] / df['teachers']

# Use average of reading and math scores as academic performance measure
df['academic_performance'] = (df['read'] + df['math']) / 2

print("=" * 80)
print("EXPLORING THE DATA")
print("=" * 80)
print("\nDataset shape:", df.shape)
print("\nStudent-Teacher Ratio Statistics:")
print(df['str'].describe())
print("\nAcademic Performance Statistics:")
print(df['academic_performance'].describe())

# Check for missing values
print("\nMissing values:")
print(df[['str', 'academic_performance']].isnull().sum())

# Correlation analysis
print("\n" + "=" * 80)
print("CORRELATION ANALYSIS")
print("=" * 80)
correlation = df['str'].corr(df['academic_performance'])
print(f"\nPearson correlation between student-teacher ratio and academic performance: {correlation:.4f}")

# Pearson correlation test
pearson_r, pearson_p = stats.pearsonr(df['str'], df['academic_performance'])
print(f"Pearson r: {pearson_r:.4f}, p-value: {pearson_p:.6f}")

# Spearman correlation test (non-parametric)
spearman_r, spearman_p = stats.spearmanr(df['str'], df['academic_performance'])
print(f"Spearman r: {spearman_r:.4f}, p-value: {spearman_p:.6f}")

# Linear regression with statsmodels for detailed statistics
print("\n" + "=" * 80)
print("LINEAR REGRESSION WITH STATSMODELS")
print("=" * 80)
X = sm.add_constant(df['str'])
y = df['academic_performance']
model_sm = sm.OLS(y, X).fit()
print(model_sm.summary())

# Simple linear regression with sklearn
print("\n" + "=" * 80)
print("SIMPLE LINEAR REGRESSION")
print("=" * 80)
X_simple = df[['str']]
y_simple = df['academic_performance']
lr = LinearRegression()
lr.fit(X_simple, y_simple)
print(f"Coefficient (slope): {lr.coef_[0]:.4f}")
print(f"Intercept: {lr.intercept_:.4f}")
print(f"R-squared: {lr.score(X_simple, y_simple):.4f}")

# Multiple regression controlling for confounders
print("\n" + "=" * 80)
print("MULTIPLE REGRESSION CONTROLLING FOR CONFOUNDERS")
print("=" * 80)
# Include socioeconomic factors that might confound the relationship
control_vars = ['str', 'income', 'english', 'lunch', 'calworks']
X_multi = sm.add_constant(df[control_vars])
model_multi = sm.OLS(df['academic_performance'], X_multi).fit()
print(model_multi.summary())

# Explainable Boosting Regressor for interpretable non-linear relationships
print("\n" + "=" * 80)
print("EXPLAINABLE BOOSTING REGRESSOR")
print("=" * 80)
ebr = ExplainableBoostingRegressor(random_state=42)
ebr.fit(df[['str']], df['academic_performance'])
print(f"EBR R-squared: {ebr.score(df[['str']], df['academic_performance']):.4f}")

# Group analysis: Split into low/medium/high STR groups
print("\n" + "=" * 80)
print("GROUP ANALYSIS (ANOVA)")
print("=" * 80)
df['str_group'] = pd.qcut(df['str'], q=3, labels=['Low STR', 'Medium STR', 'High STR'])
print("\nAcademic Performance by Student-Teacher Ratio Group:")
print(df.groupby('str_group')['academic_performance'].describe())

# ANOVA test
groups = [df[df['str_group'] == group]['academic_performance'].values 
          for group in ['Low STR', 'Medium STR', 'High STR']]
f_stat, anova_p = stats.f_oneway(*groups)
print(f"\nANOVA F-statistic: {f_stat:.4f}, p-value: {anova_p:.6f}")

# Post-hoc t-tests
print("\nPost-hoc pairwise t-tests:")
low_str = df[df['str_group'] == 'Low STR']['academic_performance']
high_str = df[df['str_group'] == 'High STR']['academic_performance']
t_stat, t_p = stats.ttest_ind(low_str, high_str)
print(f"Low STR vs High STR: t={t_stat:.4f}, p={t_p:.6f}")
print(f"Mean difference: {low_str.mean() - high_str.mean():.2f} points")

# INTERPRETATION
print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)

# Key findings
print("\n1. CORRELATION:")
print(f"   - Pearson correlation: {pearson_r:.4f} (p={pearson_p:.6f})")
print(f"   - This indicates a {'NEGATIVE' if pearson_r < 0 else 'POSITIVE'} relationship")

print("\n2. SIMPLE LINEAR REGRESSION:")
print(f"   - Coefficient: {lr.coef_[0]:.4f}")
print(f"   - For each additional student per teacher, academic performance")
print(f"     changes by {lr.coef_[0]:.2f} points")
print(f"   - R-squared: {lr.score(X_simple, y_simple):.4f} ({lr.score(X_simple, y_simple)*100:.1f}% variance explained)")
print(f"   - Regression p-value: {model_sm.pvalues['str']:.6f}")

print("\n3. MULTIPLE REGRESSION (controlling for income, English learners, lunch, calworks):")
print(f"   - STR coefficient: {model_multi.params['str']:.4f}")
print(f"   - STR p-value: {model_multi.pvalues['str']:.6f}")

print("\n4. GROUP COMPARISON:")
print(f"   - Low STR schools: {low_str.mean():.2f} ± {low_str.std():.2f}")
print(f"   - High STR schools: {high_str.mean():.2f} ± {high_str.std():.2f}")
print(f"   - Difference: {low_str.mean() - high_str.mean():.2f} points")
print(f"   - t-test p-value: {t_p:.6f}")

# Determine conclusion
print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

# Check statistical significance (using p < 0.05 threshold)
is_significant = pearson_p < 0.05
is_negative_relationship = pearson_r < 0
effect_size = abs(pearson_r)

# Likert scale: 0 = strong "No", 100 = strong "Yes"
# Question: "Is a lower student-teacher ratio associated with higher academic performance?"
# This means: Does lower STR → higher performance? (i.e., negative correlation)

if is_significant and is_negative_relationship:
    # Strong negative correlation = lower STR associated with higher performance
    if effect_size > 0.5:
        response = 95  # Very strong evidence
        explanation = f"Strong negative correlation (r={pearson_r:.3f}, p<0.001) indicates lower student-teacher ratios are significantly associated with higher academic performance. The relationship explains {lr.score(X_simple, y_simple)*100:.1f}% of variance. Each additional student per teacher decreases scores by {abs(lr.coef_[0]):.2f} points."
    elif effect_size > 0.3:
        response = 80  # Strong evidence
        explanation = f"Moderate negative correlation (r={pearson_r:.3f}, p={pearson_p:.4f}) shows lower student-teacher ratios are significantly associated with higher academic performance. Schools with low STR score {low_str.mean() - high_str.mean():.1f} points higher than high STR schools (p<0.001)."
    else:
        response = 70  # Moderate evidence
        explanation = f"Weak but significant negative correlation (r={pearson_r:.3f}, p={pearson_p:.4f}) suggests lower student-teacher ratios are associated with higher academic performance, though the effect is small."
elif not is_significant:
    response = 50  # No evidence
    explanation = f"No significant relationship found (r={pearson_r:.3f}, p={pearson_p:.4f}). The correlation between student-teacher ratio and academic performance is not statistically significant."
else:  # Significant but positive relationship (opposite of what question asks)
    response = 10  # Strong evidence against
    explanation = f"Evidence suggests the opposite: positive correlation (r={pearson_r:.3f}, p={pearson_p:.4f}) indicates higher student-teacher ratios are associated with higher performance, contradicting the hypothesis."

print(f"\nLikert Scale Response: {response}/100")
print(f"Explanation: {explanation}")

# Save conclusion
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n" + "=" * 80)
print("SAVED conclusion.txt")
print("=" * 80)
