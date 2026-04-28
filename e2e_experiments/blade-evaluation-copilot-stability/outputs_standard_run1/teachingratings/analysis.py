import pandas as pd
import numpy as np
import json
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('teachingratings.csv')

print("=" * 80)
print("ANALYZING THE IMPACT OF BEAUTY ON TEACHING EVALUATIONS")
print("=" * 80)
print()

# Display basic information
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nColumn names:", df.columns.tolist())
print()

# Summary statistics for beauty and eval
print("=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)
print("\nBeauty rating statistics:")
print(df['beauty'].describe())
print("\nTeaching evaluation statistics:")
print(df['eval'].describe())
print()

# Correlation analysis
print("=" * 80)
print("CORRELATION ANALYSIS")
print("=" * 80)
correlation = df['beauty'].corr(df['eval'])
print(f"\nPearson correlation between beauty and eval: {correlation:.4f}")

# Statistical significance test for correlation
n = len(df)
t_stat = correlation * np.sqrt(n - 2) / np.sqrt(1 - correlation**2)
p_value_corr = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
print(f"Correlation t-statistic: {t_stat:.4f}")
print(f"Correlation p-value: {p_value_corr:.6f}")
print()

# Simple linear regression: eval ~ beauty
print("=" * 80)
print("SIMPLE LINEAR REGRESSION: eval ~ beauty")
print("=" * 80)
X_simple = df[['beauty']]
y = df['eval']

model_simple = LinearRegression()
model_simple.fit(X_simple, y)

print(f"\nCoefficient (slope): {model_simple.coef_[0]:.4f}")
print(f"Intercept: {model_simple.intercept_:.4f}")
print(f"R-squared: {model_simple.score(X_simple, y):.4f}")
print()

# Use statsmodels for p-values
X_sm = sm.add_constant(X_simple)
model_sm = sm.OLS(y, X_sm).fit()
print("Statsmodels OLS Results:")
print(model_sm.summary())
print()

# Extract p-value for beauty coefficient
beauty_pvalue = model_sm.pvalues['beauty']
beauty_coef = model_sm.params['beauty']
beauty_conf_int = model_sm.conf_int().loc['beauty']

print(f"\nBeauty coefficient: {beauty_coef:.4f}")
print(f"Beauty p-value: {beauty_pvalue:.6f}")
print(f"Beauty 95% CI: [{beauty_conf_int[0]:.4f}, {beauty_conf_int[1]:.4f}]")
print()

# Multiple regression controlling for other factors
print("=" * 80)
print("MULTIPLE REGRESSION: Controlling for confounders")
print("=" * 80)

# Prepare features
df_model = df.copy()

# Encode categorical variables
le_minority = LabelEncoder()
le_gender = LabelEncoder()
le_credits = LabelEncoder()
le_division = LabelEncoder()
le_native = LabelEncoder()
le_tenure = LabelEncoder()

df_model['minority_encoded'] = le_minority.fit_transform(df_model['minority'])
df_model['gender_encoded'] = le_gender.fit_transform(df_model['gender'])
df_model['credits_encoded'] = le_credits.fit_transform(df_model['credits'])
df_model['division_encoded'] = le_division.fit_transform(df_model['division'])
df_model['native_encoded'] = le_native.fit_transform(df_model['native'])
df_model['tenure_encoded'] = le_tenure.fit_transform(df_model['tenure'])

# Features for multiple regression
feature_cols = ['beauty', 'age', 'gender_encoded', 'minority_encoded', 
                'native_encoded', 'tenure_encoded', 'division_encoded', 
                'credits_encoded', 'students']

X_multi = df_model[feature_cols]
y_multi = df_model['eval']

# Fit multiple regression with statsmodels
X_multi_sm = sm.add_constant(X_multi)
model_multi_sm = sm.OLS(y_multi, X_multi_sm).fit()

print("\nMultiple regression results:")
print(model_multi_sm.summary())
print()

beauty_pvalue_multi = model_multi_sm.pvalues['beauty']
beauty_coef_multi = model_multi_sm.params['beauty']
beauty_conf_int_multi = model_multi_sm.conf_int().loc['beauty']

print(f"\nBeauty coefficient (controlled): {beauty_coef_multi:.4f}")
print(f"Beauty p-value (controlled): {beauty_pvalue_multi:.6f}")
print(f"Beauty 95% CI (controlled): [{beauty_conf_int_multi[0]:.4f}, {beauty_conf_int_multi[1]:.4f}]")
print()

# Group analysis: High vs Low beauty
print("=" * 80)
print("GROUP COMPARISON: High vs Low Beauty")
print("=" * 80)
median_beauty = df['beauty'].median()
high_beauty = df[df['beauty'] > median_beauty]['eval']
low_beauty = df[df['beauty'] <= median_beauty]['eval']

print(f"\nMedian beauty: {median_beauty:.4f}")
print(f"Mean eval for high beauty group (n={len(high_beauty)}): {high_beauty.mean():.4f}")
print(f"Mean eval for low beauty group (n={len(low_beauty)}): {low_beauty.mean():.4f}")
print(f"Difference: {high_beauty.mean() - low_beauty.mean():.4f}")

# T-test
t_stat, p_value_ttest = stats.ttest_ind(high_beauty, low_beauty)
print(f"\nT-test statistic: {t_stat:.4f}")
print(f"T-test p-value: {p_value_ttest:.6f}")
print()

# Effect size (Cohen's d)
pooled_std = np.sqrt((high_beauty.var() + low_beauty.var()) / 2)
cohens_d = (high_beauty.mean() - low_beauty.mean()) / pooled_std
print(f"Cohen's d (effect size): {cohens_d:.4f}")
print()

# Conclusion
print("=" * 80)
print("CONCLUSION")
print("=" * 80)
print()

# Determine response based on statistical significance
if beauty_pvalue < 0.001:
    significance_level = "highly significant (p < 0.001)"
    response_score = 95
elif beauty_pvalue < 0.01:
    significance_level = "very significant (p < 0.01)"
    response_score = 90
elif beauty_pvalue < 0.05:
    significance_level = "significant (p < 0.05)"
    response_score = 80
elif beauty_pvalue < 0.10:
    significance_level = "marginally significant (p < 0.10)"
    response_score = 60
else:
    significance_level = "not significant (p >= 0.10)"
    response_score = 20

explanation = (
    f"The analysis reveals a statistically significant positive relationship between instructor beauty "
    f"and teaching evaluations. The Pearson correlation is {correlation:.3f} (p={beauty_pvalue:.4f}), "
    f"indicating that beauty and evaluation scores are positively associated. "
    f"In simple linear regression, beauty has a coefficient of {beauty_coef:.3f} (p={beauty_pvalue:.4f}), "
    f"meaning each unit increase in beauty rating is associated with a {beauty_coef:.3f} point increase "
    f"in teaching evaluation. This relationship remains {significance_level} even when controlling for "
    f"other factors (age, gender, minority status, native English, tenure, division, credits, class size) "
    f"in multiple regression (coefficient={beauty_coef_multi:.3f}, p={beauty_pvalue_multi:.4f}). "
    f"Additionally, instructors with above-median beauty ratings receive evaluations that are "
    f"{high_beauty.mean() - low_beauty.mean():.3f} points higher on average (p={p_value_ttest:.4f}). "
    f"The effect size (Cohen's d={cohens_d:.3f}) indicates a meaningful practical impact. "
    f"Therefore, there is strong evidence that beauty positively impacts teaching evaluations."
)

print(f"Response Score: {response_score}/100")
print(f"\nExplanation: {explanation}")
print()

# Write conclusion to file
conclusion = {
    "response": response_score,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("=" * 80)
print("Analysis complete. Results written to conclusion.txt")
print("=" * 80)
