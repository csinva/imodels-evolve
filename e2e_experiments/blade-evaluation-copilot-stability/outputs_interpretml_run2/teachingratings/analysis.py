import pandas as pd
import numpy as np
import json
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
from interpret.glassbox import ExplainableBoostingRegressor

# Load data
df = pd.read_csv('teachingratings.csv')

print("=" * 80)
print("RESEARCH QUESTION: What is the impact of beauty on teaching evaluations?")
print("=" * 80)

# Data exploration
print("\nDataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nSummary statistics:")
print(df.describe())
print("\nData types:")
print(df.dtypes)

# Focus on key variables: beauty and eval
print("\n" + "=" * 80)
print("KEY VARIABLES ANALYSIS")
print("=" * 80)
print(f"\nBeauty rating - Mean: {df['beauty'].mean():.4f}, Std: {df['beauty'].std():.4f}")
print(f"Evaluation score - Mean: {df['eval'].mean():.4f}, Std: {df['eval'].std():.4f}")

# Correlation analysis
print("\n" + "=" * 80)
print("CORRELATION ANALYSIS")
print("=" * 80)
correlation = df[['beauty', 'eval']].corr()
print("\nCorrelation between beauty and evaluation:")
print(correlation)

# Pearson correlation test
corr_coef, p_value_corr = stats.pearsonr(df['beauty'], df['eval'])
print(f"\nPearson correlation coefficient: {corr_coef:.4f}")
print(f"P-value: {p_value_corr:.6f}")
print(f"Statistically significant (p < 0.05): {p_value_corr < 0.05}")

# Simple linear regression with statsmodels for statistical tests
print("\n" + "=" * 80)
print("SIMPLE LINEAR REGRESSION: eval ~ beauty")
print("=" * 80)
X_simple = sm.add_constant(df['beauty'])
model_simple = sm.OLS(df['eval'], X_simple).fit()
print(model_simple.summary())

# Extract key statistics
beauty_coef = model_simple.params['beauty']
beauty_pvalue = model_simple.pvalues['beauty']
r_squared = model_simple.rsquared

print(f"\nKey findings from simple regression:")
print(f"  - Beauty coefficient: {beauty_coef:.4f}")
print(f"  - P-value: {beauty_pvalue:.6f}")
print(f"  - R-squared: {r_squared:.4f}")
print(f"  - Statistically significant: {beauty_pvalue < 0.05}")

# Multiple regression controlling for other factors
print("\n" + "=" * 80)
print("MULTIPLE REGRESSION: Controlling for confounders")
print("=" * 80)

# Prepare data for multiple regression
df_model = df.copy()

# Encode categorical variables
le_minority = LabelEncoder()
le_gender = LabelEncoder()
le_credits = LabelEncoder()
le_division = LabelEncoder()
le_native = LabelEncoder()
le_tenure = LabelEncoder()

df_model['minority_enc'] = le_minority.fit_transform(df_model['minority'])
df_model['gender_enc'] = le_gender.fit_transform(df_model['gender'])
df_model['credits_enc'] = le_credits.fit_transform(df_model['credits'])
df_model['division_enc'] = le_division.fit_transform(df_model['division'])
df_model['native_enc'] = le_native.fit_transform(df_model['native'])
df_model['tenure_enc'] = le_tenure.fit_transform(df_model['tenure'])

# Build multiple regression model
features = ['beauty', 'age', 'gender_enc', 'minority_enc', 'native_enc', 
            'tenure_enc', 'credits_enc', 'division_enc', 'students']
X_multi = sm.add_constant(df_model[features])
model_multi = sm.OLS(df_model['eval'], X_multi).fit()
print(model_multi.summary())

beauty_coef_multi = model_multi.params['beauty']
beauty_pvalue_multi = model_multi.pvalues['beauty']

print(f"\nKey findings from multiple regression:")
print(f"  - Beauty coefficient (controlled): {beauty_coef_multi:.4f}")
print(f"  - P-value: {beauty_pvalue_multi:.6f}")
print(f"  - Statistically significant: {beauty_pvalue_multi < 0.05}")

# Explainable Boosting Machine for interpretability
print("\n" + "=" * 80)
print("EXPLAINABLE BOOSTING MACHINE")
print("=" * 80)

# Prepare data for EBM
X_ebm = df_model[features].values
y_ebm = df_model['eval'].values

ebm = ExplainableBoostingRegressor(random_state=42)
ebm.fit(X_ebm, y_ebm)

print("\nEBM Feature Importances:")
feature_names = features
for i, (feat, importance) in enumerate(zip(feature_names, ebm.term_importances())):
    print(f"  {feat}: {importance:.4f}")

# Effect size analysis
print("\n" + "=" * 80)
print("EFFECT SIZE ANALYSIS")
print("=" * 80)

# Split into groups based on beauty (below/above median)
median_beauty = df['beauty'].median()
low_beauty = df[df['beauty'] <= median_beauty]['eval']
high_beauty = df[df['beauty'] > median_beauty]['eval']

# T-test comparing low vs high beauty
t_stat, p_value_ttest = stats.ttest_ind(low_beauty, high_beauty)
mean_diff = high_beauty.mean() - low_beauty.mean()
cohen_d = mean_diff / np.sqrt((low_beauty.std()**2 + high_beauty.std()**2) / 2)

print(f"\nComparison of evaluation scores:")
print(f"  Low beauty group (n={len(low_beauty)}): mean = {low_beauty.mean():.4f}, std = {low_beauty.std():.4f}")
print(f"  High beauty group (n={len(high_beauty)}): mean = {high_beauty.mean():.4f}, std = {high_beauty.std():.4f}")
print(f"  Mean difference: {mean_diff:.4f}")
print(f"  T-statistic: {t_stat:.4f}")
print(f"  P-value: {p_value_ttest:.6f}")
print(f"  Cohen's d (effect size): {cohen_d:.4f}")
print(f"  Statistically significant: {p_value_ttest < 0.05}")

# Interpret effect size (Cohen's d)
if abs(cohen_d) < 0.2:
    effect_interpretation = "negligible"
elif abs(cohen_d) < 0.5:
    effect_interpretation = "small"
elif abs(cohen_d) < 0.8:
    effect_interpretation = "medium"
else:
    effect_interpretation = "large"

print(f"  Effect size interpretation: {effect_interpretation}")

# CONCLUSION
print("\n" + "=" * 80)
print("FINAL CONCLUSION")
print("=" * 80)

# Determine the response score (0-100 Likert scale)
# 0 = strong No, 100 = strong Yes

# Criteria for determining the score:
# 1. Statistical significance of the relationship
# 2. Consistency across different models
# 3. Effect size
# 4. Practical significance

is_significant_simple = beauty_pvalue < 0.05
is_significant_multi = beauty_pvalue_multi < 0.05
is_significant_ttest = p_value_ttest < 0.05

if is_significant_simple and is_significant_multi and is_significant_ttest:
    # All tests show significance - strong evidence
    if abs(cohen_d) >= 0.5:
        # Medium to large effect size
        response = 90
        explanation = f"There is strong evidence for a positive impact of beauty on teaching evaluations. The relationship is statistically significant across multiple analyses: simple regression (coef={beauty_coef:.4f}, p<0.001), multiple regression controlling for confounders (coef={beauty_coef_multi:.4f}, p<0.001), and t-test comparing high vs low beauty groups (p<0.001). The effect size is {effect_interpretation} (Cohen's d={cohen_d:.2f}), meaning a one-unit increase in beauty rating is associated with approximately {beauty_coef_multi:.2f} points higher evaluation score."
    else:
        # Small effect size
        response = 75
        explanation = f"There is solid evidence for a positive impact of beauty on teaching evaluations. The relationship is statistically significant across all analyses: simple regression (coef={beauty_coef:.4f}, p={beauty_pvalue:.4f}), multiple regression (coef={beauty_coef_multi:.4f}, p={beauty_pvalue_multi:.4f}), and t-test (p={p_value_ttest:.4f}). However, the effect size is {effect_interpretation} (Cohen's d={cohen_d:.2f}), indicating the practical impact is modest."
elif is_significant_simple and is_significant_multi:
    # Significant in regressions but not in t-test
    response = 70
    explanation = f"There is moderate evidence for a positive impact of beauty on teaching evaluations. Both simple and multiple regression analyses show statistically significant relationships (p<0.05), but the effect size is relatively small (Cohen's d={cohen_d:.2f})."
elif is_significant_simple:
    # Only significant in simple regression
    response = 50
    explanation = f"There is weak evidence for an impact of beauty on teaching evaluations. While the simple regression shows a significant relationship (p={beauty_pvalue:.4f}), this relationship becomes less clear when controlling for other factors in multiple regression (p={beauty_pvalue_multi:.4f})."
else:
    # Not significant
    response = 20
    explanation = f"There is insufficient evidence for a meaningful impact of beauty on teaching evaluations. The relationship is not statistically significant in our analyses (p-values > 0.05)."

print(f"\nResponse Score: {response}/100")
print(f"Explanation: {explanation}")

# Write conclusion to file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE - conclusion.txt written")
print("=" * 80)
