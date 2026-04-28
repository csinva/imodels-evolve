import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('teachingratings.csv')

# Read research question
with open('info.json', 'r') as f:
    info = json.load(f)
research_question = info['research_questions'][0]

print("="*80)
print(f"Research Question: {research_question}")
print("="*80)

# Data exploration
print("\n1. DATA EXPLORATION")
print("-"*80)
print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

print("\n2. SUMMARY STATISTICS")
print("-"*80)
print("\nKey variables:")
print(df[['beauty', 'eval']].describe())

print("\n3. CORRELATION ANALYSIS")
print("-"*80)
correlation = df['beauty'].corr(df['eval'])
print(f"Correlation between beauty and eval: {correlation:.4f}")

# Perform Pearson correlation test
pearson_corr, pearson_pval = stats.pearsonr(df['beauty'], df['eval'])
print(f"Pearson correlation: {pearson_corr:.4f}, p-value: {pearson_pval:.6f}")

# Perform Spearman correlation test (non-parametric)
spearman_corr, spearman_pval = stats.spearmanr(df['beauty'], df['eval'])
print(f"Spearman correlation: {spearman_corr:.4f}, p-value: {spearman_pval:.6f}")

print("\n4. SIMPLE LINEAR REGRESSION (beauty -> eval)")
print("-"*80)
X_simple = df[['beauty']].values
y = df['eval'].values

# Simple linear regression
model_simple = LinearRegression()
model_simple.fit(X_simple, y)
print(f"Coefficient: {model_simple.coef_[0]:.4f}")
print(f"Intercept: {model_simple.intercept_:.4f}")
print(f"R-squared: {model_simple.score(X_simple, y):.4f}")

# Use statsmodels for p-values
X_sm = sm.add_constant(X_simple)
ols_model = sm.OLS(y, X_sm).fit()
print(f"\nStatsmodels OLS Summary:")
print(ols_model.summary())

print("\n5. MULTIPLE LINEAR REGRESSION (controlling for confounders)")
print("-"*80)

# Prepare features - encode categorical variables
df_encoded = df.copy()
categorical_cols = ['minority', 'gender', 'credits', 'division', 'native', 'tenure']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col + '_encoded'] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le

# Select features for multiple regression
feature_cols = ['beauty', 'age', 'minority_encoded', 'gender_encoded', 'credits_encoded', 
                'division_encoded', 'native_encoded', 'tenure_encoded', 'students']
X_multi = df_encoded[feature_cols].values
y = df_encoded['eval'].values

# Multiple linear regression
model_multi = LinearRegression()
model_multi.fit(X_multi, y)
print(f"R-squared: {model_multi.score(X_multi, y):.4f}")
print(f"\nCoefficients:")
for i, col in enumerate(feature_cols):
    print(f"  {col}: {model_multi.coef_[i]:.4f}")

# Use statsmodels for p-values
X_sm_multi = sm.add_constant(X_multi)
ols_model_multi = sm.OLS(y, X_sm_multi).fit()
print(f"\nStatsmodels OLS Summary (Multiple Regression):")
print(ols_model_multi.summary())

print("\n6. INTERPRETABLE MODEL: DECISION TREE")
print("-"*80)
from sklearn.tree import DecisionTreeRegressor

# Train decision tree with limited depth for interpretability
dt_model = DecisionTreeRegressor(max_depth=3, random_state=42)
dt_model.fit(X_multi, y)
print(f"Decision Tree R-squared: {dt_model.score(X_multi, y):.4f}")
print(f"\nFeature importances:")
for i, col in enumerate(feature_cols):
    print(f"  {col}: {dt_model.feature_importances_[i]:.4f}")

print("\n7. STATISTICAL SIGNIFICANCE ASSESSMENT")
print("-"*80)
print(f"Beauty coefficient p-value (simple regression): {ols_model.pvalues[1]:.6f}")
print(f"Beauty coefficient p-value (multiple regression): {ols_model_multi.pvalues[1]:.6f}")

# Determine significance level
alpha = 0.05
is_significant_simple = ols_model.pvalues[1] < alpha
is_significant_multi = ols_model_multi.pvalues[1] < alpha

print(f"\nIs beauty significant in simple regression (α={alpha})? {is_significant_simple}")
print(f"Is beauty significant in multiple regression (α={alpha})? {is_significant_multi}")

print("\n8. EFFECT SIZE ANALYSIS")
print("-"*80)
beauty_coef_simple = ols_model.params[1]
beauty_coef_multi = ols_model_multi.params[1]
beauty_std = df['beauty'].std()
eval_std = df['eval'].std()

# Standardized effect: how much does a 1 SD change in beauty affect eval?
standardized_effect_simple = beauty_coef_simple * beauty_std / eval_std
standardized_effect_multi = beauty_coef_multi * beauty_std / eval_std

print(f"Beauty coefficient (simple): {beauty_coef_simple:.4f}")
print(f"Beauty coefficient (multiple): {beauty_coef_multi:.4f}")
print(f"Standardized effect (simple): {standardized_effect_simple:.4f} SD")
print(f"Standardized effect (multiple): {standardized_effect_multi:.4f} SD")

# How much does eval change for a 1-unit increase in beauty?
print(f"\nInterpretation:")
print(f"  A 1-unit increase in beauty rating is associated with a {beauty_coef_simple:.4f} point")
print(f"  increase in teaching evaluation (simple regression).")
print(f"  After controlling for other factors: {beauty_coef_multi:.4f} point increase.")

print("\n9. CONCLUSION")
print("="*80)

# Determine response score and explanation
if is_significant_multi and beauty_coef_multi > 0:
    # Significant positive relationship
    if ols_model_multi.pvalues[1] < 0.001:
        response = 95  # Very strong evidence
        strength = "very strong"
    elif ols_model_multi.pvalues[1] < 0.01:
        response = 85  # Strong evidence
        strength = "strong"
    else:
        response = 75  # Moderate evidence
        strength = "moderate"
    
    explanation = (f"Yes, there is {strength} statistical evidence that beauty positively impacts "
                   f"teaching evaluations. The beauty coefficient is {beauty_coef_multi:.4f} "
                   f"(p={ols_model_multi.pvalues[1]:.4f}), meaning a 1-unit increase in beauty "
                   f"rating is associated with a {beauty_coef_multi:.4f} point increase in teaching "
                   f"evaluation score, even after controlling for age, gender, minority status, "
                   f"tenure, native English speaker status, course division, credits, and number of students. "
                   f"The correlation is {pearson_corr:.3f} and the effect remains robust in multiple regression.")
    
elif is_significant_simple and beauty_coef_simple > 0 and not is_significant_multi:
    # Significant in simple but not multiple - suggests confounding
    response = 60
    explanation = (f"There is some evidence of a relationship between beauty and teaching evaluations "
                   f"in simple correlation (r={pearson_corr:.3f}, p={pearson_pval:.4f}), but this relationship "
                   f"becomes non-significant (p={ols_model_multi.pvalues[1]:.4f}) when controlling for other "
                   f"factors like age, gender, and class characteristics. This suggests the apparent effect "
                   f"may be partially explained by confounding variables.")
    
elif not is_significant_simple and not is_significant_multi:
    # No significant relationship
    response = 10
    explanation = (f"No, there is no statistically significant impact of beauty on teaching evaluations. "
                   f"The p-values in both simple (p={ols_model.pvalues[1]:.4f}) and multiple "
                   f"(p={ols_model_multi.pvalues[1]:.4f}) regression exceed the significance threshold "
                   f"of 0.05. The correlation is weak (r={pearson_corr:.3f}) and not significant.")
else:
    # Edge cases
    response = 50
    explanation = (f"The relationship between beauty and teaching evaluations is ambiguous. "
                   f"Correlation: {pearson_corr:.3f} (p={pearson_pval:.4f}). "
                   f"Multiple regression p-value: {ols_model_multi.pvalues[1]:.4f}.")

print(f"Response Score: {response}")
print(f"Explanation: {explanation}")

# Write conclusion to file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n" + "="*80)
print("Analysis complete. Conclusion written to conclusion.txt")
print("="*80)
