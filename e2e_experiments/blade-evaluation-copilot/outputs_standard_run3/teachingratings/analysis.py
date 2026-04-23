#!/usr/bin/env python3
"""
Analysis script to answer: What is the impact of beauty on teaching evaluations?
"""

import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
from imodels import RuleFitRegressor, FIGSRegressor, HSTreeRegressor

# Load the data
df = pd.read_csv('teachingratings.csv')

print("="*80)
print("DATASET OVERVIEW")
print("="*80)
print(f"Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nData types:")
print(df.dtypes)
print(f"\nSummary statistics:")
print(df.describe())

# Check for missing values
print(f"\nMissing values:")
print(df.isnull().sum())

print("\n" + "="*80)
print("EXPLORATORY DATA ANALYSIS")
print("="*80)

# Focus on beauty and eval (teaching evaluation)
print(f"\nBeauty statistics:")
print(df['beauty'].describe())
print(f"\nEval (teaching evaluation) statistics:")
print(df['eval'].describe())

# Correlation between beauty and eval
correlation = df['beauty'].corr(df['eval'])
print(f"\nPearson correlation between beauty and eval: {correlation:.4f}")

# Simple statistical test - Pearson correlation significance
n = len(df)
t_stat = correlation * np.sqrt(n - 2) / np.sqrt(1 - correlation**2)
p_value_corr = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
print(f"Correlation p-value: {p_value_corr:.6f}")

print("\n" + "="*80)
print("LINEAR REGRESSION ANALYSIS")
print("="*80)

# Simple linear regression: eval ~ beauty
X_simple = df[['beauty']].values
y = df['eval'].values

lr_simple = LinearRegression()
lr_simple.fit(X_simple, y)
r2_simple = lr_simple.score(X_simple, y)

print(f"\nSimple Linear Regression: eval ~ beauty")
print(f"Coefficient (slope): {lr_simple.coef_[0]:.4f}")
print(f"Intercept: {lr_simple.intercept_:.4f}")
print(f"R-squared: {r2_simple:.4f}")

# Using statsmodels for p-values
X_sm = sm.add_constant(X_simple)
model_sm = sm.OLS(y, X_sm).fit()
print(f"\nStatsmodels OLS Results:")
print(model_sm.summary())

beauty_pvalue = model_sm.pvalues[1]
beauty_coef = model_sm.params[1]
beauty_conf_int = model_sm.conf_int()[1]

print(f"\n*** Beauty coefficient: {beauty_coef:.4f}")
print(f"*** Beauty p-value: {beauty_pvalue:.6f}")
print(f"*** Beauty 95% CI: [{beauty_conf_int[0]:.4f}, {beauty_conf_int[1]:.4f}]")

print("\n" + "="*80)
print("MULTIPLE REGRESSION ANALYSIS (CONTROLLING FOR CONFOUNDERS)")
print("="*80)

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

# Multiple regression with controls
features = ['beauty', 'minority_enc', 'age', 'gender_enc', 'credits_enc', 
            'division_enc', 'native_enc', 'tenure_enc', 'students']

X_full = df_model[features].values
X_full_sm = sm.add_constant(X_full)

model_full = sm.OLS(y, X_full_sm).fit()
print(f"\nMultiple Regression Results:")
print(model_full.summary())

beauty_coef_full = model_full.params[1]  # beauty is first feature (index 1 after constant)
beauty_pvalue_full = model_full.pvalues[1]
beauty_conf_int_full = model_full.conf_int()[1]

print(f"\n*** Beauty coefficient (with controls): {beauty_coef_full:.4f}")
print(f"*** Beauty p-value (with controls): {beauty_pvalue_full:.6f}")
print(f"*** Beauty 95% CI (with controls): [{beauty_conf_int_full[0]:.4f}, {beauty_conf_int_full[1]:.4f}]")

print("\n" + "="*80)
print("INTERPRETABLE MODELS")
print("="*80)

# Decision Tree for interpretability
dt = DecisionTreeRegressor(max_depth=4, min_samples_leaf=20, random_state=42)
dt.fit(X_full, y)
print(f"\nDecision Tree R-squared: {dt.score(X_full, y):.4f}")
print(f"Feature importances:")
for i, feat in enumerate(features):
    print(f"  {feat}: {dt.feature_importances_[i]:.4f}")

# Try HSTreeRegressor from imodels
try:
    hst = HSTreeRegressor(max_leaf_nodes=10, random_state=42)
    hst.fit(X_full, y)
    print(f"\nHSTreeRegressor R-squared: {hst.score(X_full, y):.4f}")
    print("Rules:")
    print(hst)
except Exception as e:
    print(f"\nHSTreeRegressor error: {e}")

# Try FIGSRegressor from imodels
try:
    figs = FIGSRegressor(max_rules=10, random_state=42)
    figs.fit(X_full, y)
    print(f"\nFIGSRegressor R-squared: {figs.score(X_full, y):.4f}")
except Exception as e:
    print(f"\nFIGSRegressor error: {e}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

# Determine the impact
is_significant = beauty_pvalue_full < 0.05
effect_size = abs(beauty_coef_full)
is_positive = beauty_coef_full > 0

print(f"\nKey findings:")
print(f"1. Correlation between beauty and eval: {correlation:.4f} (p={p_value_corr:.6f})")
print(f"2. Simple regression coefficient: {beauty_coef:.4f} (p={beauty_pvalue:.6f})")
print(f"3. Multiple regression coefficient (with controls): {beauty_coef_full:.4f} (p={beauty_pvalue_full:.6f})")
print(f"4. Statistical significance (p < 0.05): {is_significant}")
print(f"5. Direction of effect: {'Positive' if is_positive else 'Negative'}")

# Determine Likert scale response (0-100)
# 0 = strong "No" (no impact), 100 = strong "Yes" (strong impact)
if not is_significant:
    response = 20  # Low score - no significant impact
    explanation = f"Beauty does not have a statistically significant impact on teaching evaluations (p={beauty_pvalue_full:.4f} > 0.05 in multiple regression controlling for confounders). The coefficient is {beauty_coef_full:.4f}, but this is not significant."
elif effect_size < 0.1:
    response = 50  # Moderate score - significant but small effect
    explanation = f"Beauty has a statistically significant but small impact on teaching evaluations (p={beauty_pvalue_full:.4f} < 0.05). The coefficient is {beauty_coef_full:.4f}, indicating that a 1-unit increase in beauty is associated with a {beauty_coef_full:.4f}-point increase in evaluation scores. However, the effect size is relatively small."
elif effect_size < 0.2:
    response = 75  # High score - significant moderate effect
    explanation = f"Beauty has a statistically significant and moderate impact on teaching evaluations (p={beauty_pvalue_full:.4f} < 0.05). The coefficient is {beauty_coef_full:.4f}, indicating that a 1-unit increase in beauty (about 1 standard deviation) is associated with a {beauty_coef_full:.4f}-point increase in evaluation scores (on a 1-5 scale). This effect persists even when controlling for other factors."
else:
    response = 90  # Very high score - significant strong effect
    explanation = f"Beauty has a statistically significant and strong impact on teaching evaluations (p={beauty_pvalue_full:.4f} < 0.05). The coefficient is {beauty_coef_full:.4f}, indicating a substantial effect. A 1-unit increase in beauty is associated with a {beauty_coef_full:.4f}-point increase in evaluation scores. This is a notable effect that persists when controlling for confounders."

print(f"\nFinal assessment: {response}/100")
print(f"Explanation: {explanation}")

# Write conclusion to file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print(f"\n{'='*80}")
print("DONE - conclusion.txt written")
print(f"{'='*80}")
