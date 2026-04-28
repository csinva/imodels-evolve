import pandas as pd
import numpy as np
import json
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

# Load the data
df = pd.read_csv('teachingratings.csv')

print("="*50)
print("ANALYZING: Impact of beauty on teaching evaluations")
print("="*50)

# Explore the data
print("\nData shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

print("\nSummary statistics for key variables:")
print(df[['beauty', 'eval']].describe())

print("\nCorrelation between beauty and eval:")
correlation = df['beauty'].corr(df['eval'])
print(f"Pearson correlation: {correlation:.4f}")

# Perform correlation test
corr_result = stats.pearsonr(df['beauty'], df['eval'])
print(f"Correlation p-value: {corr_result[1]:.6f}")

# Simple linear regression
print("\n" + "="*50)
print("SIMPLE LINEAR REGRESSION: eval ~ beauty")
print("="*50)
X = df[['beauty']]
y = df['eval']
X_sm = sm.add_constant(X)
model_simple = sm.OLS(y, X_sm).fit()
print(model_simple.summary())

# Extract key results from simple regression
beauty_coef = model_simple.params['beauty']
beauty_pval = model_simple.pvalues['beauty']
r_squared = model_simple.rsquared

print(f"\nKey finding from simple regression:")
print(f"  Beauty coefficient: {beauty_coef:.4f}")
print(f"  P-value: {beauty_pval:.6f}")
print(f"  R-squared: {r_squared:.4f}")

# Multiple regression controlling for other factors
print("\n" + "="*50)
print("MULTIPLE REGRESSION: eval ~ beauty + controls")
print("="*50)

# Prepare data for multiple regression
df_model = df.copy()

# Convert categorical variables to numeric
df_model['minority_num'] = (df_model['minority'] == 'yes').astype(int)
df_model['gender_num'] = (df_model['gender'] == 'female').astype(int)
df_model['credits_num'] = (df_model['credits'] == 'more').astype(int)
df_model['division_num'] = (df_model['division'] == 'upper').astype(int)
df_model['native_num'] = (df_model['native'] == 'yes').astype(int)
df_model['tenure_num'] = (df_model['tenure'] == 'yes').astype(int)

# Build multiple regression model
predictors = ['beauty', 'age', 'minority_num', 'gender_num', 'credits_num', 
              'division_num', 'native_num', 'tenure_num', 'students']
X_multi = df_model[predictors]
X_multi_sm = sm.add_constant(X_multi)
model_multi = sm.OLS(y, X_multi_sm).fit()
print(model_multi.summary())

# Extract key results from multiple regression
beauty_coef_multi = model_multi.params['beauty']
beauty_pval_multi = model_multi.pvalues['beauty']

print(f"\nKey finding from multiple regression:")
print(f"  Beauty coefficient (controlling for other factors): {beauty_coef_multi:.4f}")
print(f"  P-value: {beauty_pval_multi:.6f}")

# Test for different beauty groups
print("\n" + "="*50)
print("COMPARING HIGH vs LOW BEAUTY GROUPS")
print("="*50)
median_beauty = df['beauty'].median()
high_beauty = df[df['beauty'] > median_beauty]['eval']
low_beauty = df[df['beauty'] <= median_beauty]['eval']

print(f"High beauty group (n={len(high_beauty)}): mean eval = {high_beauty.mean():.3f}")
print(f"Low beauty group (n={len(low_beauty)}): mean eval = {low_beauty.mean():.3f}")

t_stat, t_pval = stats.ttest_ind(high_beauty, low_beauty)
print(f"T-test: t-statistic = {t_stat:.4f}, p-value = {t_pval:.6f}")

# Determine conclusion
print("\n" + "="*50)
print("CONCLUSION")
print("="*50)

# Decision logic based on statistical significance
alpha = 0.05

# Both simple and multiple regression show significance
if beauty_pval < alpha and beauty_pval_multi < alpha:
    if beauty_coef > 0 and beauty_coef_multi > 0:
        response_score = 95
        explanation = (
            f"Strong evidence that beauty positively impacts teaching evaluations. "
            f"Simple regression shows significant positive coefficient ({beauty_coef:.3f}, p={beauty_pval:.4f}). "
            f"Multiple regression controlling for age, gender, minority status, credits, division, "
            f"native language, tenure, and class size confirms the effect remains significant "
            f"({beauty_coef_multi:.3f}, p={beauty_pval_multi:.4f}). "
            f"T-test shows instructors with above-median beauty receive significantly higher "
            f"evaluations (p={t_pval:.4f}). The effect is robust and statistically significant."
        )
    else:
        response_score = 30
        explanation = (
            f"Beauty shows statistically significant but negative relationship with evaluations. "
            f"This is unexpected and warrants further investigation."
        )
elif beauty_pval < alpha or beauty_pval_multi < alpha:
    response_score = 70
    explanation = (
        f"Moderate evidence for beauty's impact. Significant in {'simple' if beauty_pval < alpha else 'multiple'} "
        f"regression (p={min(beauty_pval, beauty_pval_multi):.4f}), but effect weakens when controlling for other factors."
    )
else:
    response_score = 20
    explanation = (
        f"No statistically significant relationship between beauty and teaching evaluations. "
        f"Simple regression p-value: {beauty_pval:.4f}, Multiple regression p-value: {beauty_pval_multi:.4f}. "
        f"Both exceed significance threshold of 0.05."
    )

print(f"Response score: {response_score}")
print(f"Explanation: {explanation}")

# Write conclusion to file
conclusion = {
    "response": response_score,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\nconclusion.txt has been created successfully!")
