import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from interpret.glassbox import ExplainableBoostingRegressor
import json

# Load the dataset
df = pd.read_csv('teachingratings.csv')

print("="*80)
print("RESEARCH QUESTION: What is the impact of beauty on teaching evaluations?")
print("="*80)

# Explore the data
print("\nDataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

print("\nSummary statistics:")
print(df[['beauty', 'eval']].describe())

print("\nData types:")
print(df.dtypes)

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Correlation between beauty and eval
print("\n" + "="*80)
print("CORRELATION ANALYSIS")
print("="*80)
correlation = df['beauty'].corr(df['eval'])
print(f"\nPearson correlation between beauty and eval: {correlation:.4f}")

# Statistical test for correlation
n = len(df)
t_stat = correlation * np.sqrt(n - 2) / np.sqrt(1 - correlation**2)
p_value_corr = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value_corr:.6f}")

# Simple linear regression with beauty only
print("\n" + "="*80)
print("SIMPLE LINEAR REGRESSION: eval ~ beauty")
print("="*80)
X_simple = df[['beauty']]
y = df['eval']
X_simple_with_const = sm.add_constant(X_simple)
model_simple = sm.OLS(y, X_simple_with_const).fit()
print(model_simple.summary())

beauty_coef = model_simple.params['beauty']
beauty_pval = model_simple.pvalues['beauty']
beauty_conf = model_simple.conf_int().loc['beauty']
r_squared = model_simple.rsquared

print(f"\nBeauty coefficient: {beauty_coef:.4f}")
print(f"P-value: {beauty_pval:.6f}")
print(f"95% Confidence Interval: [{beauty_conf[0]:.4f}, {beauty_conf[1]:.4f}]")
print(f"R-squared: {r_squared:.4f}")

# Multiple regression with control variables
print("\n" + "="*80)
print("MULTIPLE REGRESSION WITH CONTROLS")
print("="*80)

# Prepare data for multiple regression
df_reg = df.copy()

# Convert categorical variables to dummy variables
df_reg['minority_yes'] = (df_reg['minority'] == 'yes').astype(int)
df_reg['gender_female'] = (df_reg['gender'] == 'female').astype(int)
df_reg['credits_more'] = (df_reg['credits'] == 'more').astype(int)
df_reg['division_upper'] = (df_reg['division'] == 'upper').astype(int)
df_reg['native_yes'] = (df_reg['native'] == 'yes').astype(int)
df_reg['tenure_yes'] = (df_reg['tenure'] == 'yes').astype(int)

# Select features for multiple regression
control_vars = ['beauty', 'age', 'minority_yes', 'gender_female', 'credits_more', 
                'division_upper', 'native_yes', 'tenure_yes', 'students']
X_multiple = df_reg[control_vars]
X_multiple_with_const = sm.add_constant(X_multiple)
model_multiple = sm.OLS(y, X_multiple_with_const).fit()
print(model_multiple.summary())

beauty_coef_control = model_multiple.params['beauty']
beauty_pval_control = model_multiple.pvalues['beauty']
beauty_conf_control = model_multiple.conf_int().loc['beauty']
r_squared_control = model_multiple.rsquared

print(f"\nBeauty coefficient (with controls): {beauty_coef_control:.4f}")
print(f"P-value: {beauty_pval_control:.6f}")
print(f"95% Confidence Interval: [{beauty_conf_control[0]:.4f}, {beauty_conf_control[1]:.4f}]")
print(f"R-squared: {r_squared_control:.4f}")

# Use Explainable Boosting Regressor for interpretable non-linear analysis
print("\n" + "="*80)
print("EXPLAINABLE BOOSTING REGRESSOR")
print("="*80)

ebm = ExplainableBoostingRegressor(random_state=42)
ebm.fit(X_multiple, y)
print(f"EBM R-squared: {ebm.score(X_multiple, y):.4f}")

# Get feature importances
feature_importances = ebm.term_importances()
print("\nFeature importances:")
for i, feat in enumerate(control_vars):
    print(f"{feat}: {feature_importances[i]:.4f}")

# Determine response based on statistical significance and effect size
print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

# Key findings
print(f"\n1. Correlation: r = {correlation:.4f}, p = {p_value_corr:.6f}")
print(f"2. Simple regression: β = {beauty_coef:.4f}, p = {beauty_pval:.6f}")
print(f"3. Multiple regression (with controls): β = {beauty_coef_control:.4f}, p = {beauty_pval_control:.6f}")

# Determine Likert score (0-100)
# Strong evidence: p < 0.001 → score near 90-100
# Good evidence: p < 0.01 → score near 70-85
# Moderate evidence: p < 0.05 → score near 55-70
# Weak/no evidence: p >= 0.05 → score near 0-45

if beauty_pval_control < 0.001:
    response = 95
    explanation = (f"There is very strong statistical evidence that beauty has a significant impact on teaching evaluations. "
                  f"The correlation is {correlation:.3f} (p<0.001). In multiple regression controlling for age, gender, minority status, "
                  f"tenure, native language, course type, division, and class size, beauty has a coefficient of {beauty_coef_control:.3f} "
                  f"(p={beauty_pval_control:.4f}), meaning each standard deviation increase in beauty rating is associated with "
                  f"a {beauty_coef_control:.3f}-point increase in teaching evaluation score. The effect is statistically significant "
                  f"and practically meaningful.")
elif beauty_pval_control < 0.01:
    response = 80
    explanation = (f"There is strong statistical evidence that beauty impacts teaching evaluations. "
                  f"After controlling for multiple confounding variables, beauty has a significant coefficient of {beauty_coef_control:.3f} "
                  f"(p={beauty_pval_control:.4f}), indicating a positive relationship between instructor attractiveness and evaluation scores.")
elif beauty_pval_control < 0.05:
    response = 65
    explanation = (f"There is moderate statistical evidence that beauty affects teaching evaluations. "
                  f"The effect (β={beauty_coef_control:.3f}, p={beauty_pval_control:.4f}) is statistically significant at the 0.05 level, "
                  f"though the evidence is less strong than at stricter significance thresholds.")
else:
    response = 20
    explanation = (f"There is insufficient statistical evidence that beauty has a significant impact on teaching evaluations when "
                  f"controlling for other factors. While the simple correlation is {correlation:.3f}, the multiple regression coefficient "
                  f"is {beauty_coef_control:.3f} with p={beauty_pval_control:.4f}, which is not statistically significant at conventional levels.")

print(f"\nLikert Score: {response}/100")
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
