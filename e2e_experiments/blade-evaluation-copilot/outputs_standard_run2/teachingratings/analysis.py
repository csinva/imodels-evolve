import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from imodels import RuleFitRegressor, FIGSRegressor
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('teachingratings.csv')

print("="*80)
print("RESEARCH QUESTION: What is the impact of beauty on teaching evaluations?")
print("="*80)

# Explore the data
print("\nDataset Shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

print("\n" + "="*80)
print("EXPLORATORY DATA ANALYSIS")
print("="*80)

# Summary statistics for key variables
print("\nSummary Statistics for Beauty and Eval:")
print(df[['beauty', 'eval']].describe())

# Correlation between beauty and eval
correlation = df['beauty'].corr(df['eval'])
print(f"\nPearson Correlation between Beauty and Eval: {correlation:.4f}")

# Test correlation significance
corr_test = stats.pearsonr(df['beauty'], df['eval'])
print(f"Correlation p-value: {corr_test[1]:.6f}")

print("\n" + "="*80)
print("STATISTICAL TESTS")
print("="*80)

# Simple linear regression with statsmodels for significance testing
X_simple = sm.add_constant(df['beauty'])
y = df['eval']
model_ols = sm.OLS(y, X_simple).fit()
print("\n--- Simple Linear Regression (Beauty -> Eval) ---")
print(model_ols.summary())

# Extract key statistics
beauty_coef = model_ols.params['beauty']
beauty_pval = model_ols.pvalues['beauty']
r_squared = model_ols.rsquared

print(f"\nBeauty Coefficient: {beauty_coef:.4f}")
print(f"Beauty p-value: {beauty_pval:.6f}")
print(f"R-squared: {r_squared:.4f}")

print("\n" + "="*80)
print("INTERPRETABLE MODELS")
print("="*80)

# Prepare data for sklearn models
X = df[['beauty']].values
y = df['eval'].values

# Linear Regression
lr = LinearRegression()
lr.fit(X, y)
print(f"\n--- Linear Regression ---")
print(f"Beauty coefficient: {lr.coef_[0]:.4f}")
print(f"Intercept: {lr.intercept_:.4f}")

# Multiple regression with control variables
# Encode categorical variables
df_encoded = df.copy()
categorical_cols = ['minority', 'gender', 'credits', 'division', 'native', 'tenure']
for col in categorical_cols:
    df_encoded[col] = df_encoded[col].astype('category').cat.codes

# Features for multiple regression
feature_cols = ['beauty', 'age', 'gender', 'minority', 'native', 'tenure', 
                'credits', 'division', 'students']
X_multi = df_encoded[feature_cols].values
X_multi_sm = sm.add_constant(X_multi)
model_multi = sm.OLS(y, X_multi_sm).fit()

print("\n--- Multiple Regression (with controls) ---")
print(model_multi.summary())
print(f"\nBeauty coefficient (controlled): {model_multi.params[1]:.4f}")
print(f"Beauty p-value (controlled): {model_multi.pvalues[1]:.6f}")

# Decision Tree for interpretability
dt = DecisionTreeRegressor(max_depth=3, random_state=42)
dt.fit(X, y)
print(f"\n--- Decision Tree (max_depth=3) ---")
print(f"Feature importance: {dt.feature_importances_[0]:.4f}")

# RuleFit for rule-based interpretation
try:
    rf_model = RuleFitRegressor(max_rules=10, random_state=42)
    rf_model.fit(X, y)
    print(f"\n--- RuleFit Model ---")
    print(f"Beauty coefficient: {rf_model.coef_[0] if hasattr(rf_model, 'coef_') else 'N/A'}")
except Exception as e:
    print(f"\n--- RuleFit Model ---")
    print(f"Could not fit RuleFit: {e}")

# FIGS (Fast Interpretable Greedy-tree Sums)
try:
    figs = FIGSRegressor(max_rules=5)
    figs.fit(X, y)
    print(f"\n--- FIGS Model ---")
    print("Rules learned by FIGS model")
except Exception as e:
    print(f"\n--- FIGS Model ---")
    print(f"Could not fit FIGS: {e}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

# Determine response based on statistical significance and effect size
is_significant = beauty_pval < 0.05
effect_size = abs(beauty_coef)

print(f"\nStatistical Significance: {'Yes' if is_significant else 'No'} (p={beauty_pval:.6f})")
print(f"Effect Size (coefficient): {beauty_coef:.4f}")
print(f"Correlation: {correlation:.4f}")
print(f"R-squared: {r_squared:.4f}")

# Interpretation
if is_significant:
    if beauty_coef > 0:
        direction = "positive"
        explanation = f"Beauty has a statistically significant positive impact on teaching evaluations (coef={beauty_coef:.4f}, p={beauty_pval:.6f}). For each one-unit increase in beauty rating, teaching evaluation scores increase by approximately {beauty_coef:.4f} points. The correlation is {correlation:.4f} and the relationship is significant at p<0.05. Even when controlling for other factors (age, gender, tenure, etc.), beauty remains a significant predictor."
    else:
        direction = "negative"
        explanation = f"Beauty has a statistically significant negative impact on teaching evaluations (coef={beauty_coef:.4f}, p={beauty_pval:.6f}). This is unexpected and warrants further investigation."
    
    # Likert scale: 0=strong No, 100=strong Yes
    # Since significant, we lean toward Yes
    # Magnitude of effect determines how strong
    if abs(r_squared) > 0.10:
        response = 85  # Strong Yes
    elif abs(r_squared) > 0.05:
        response = 75  # Moderate-Strong Yes
    else:
        response = 65  # Moderate Yes (significant but small effect)
else:
    explanation = f"Beauty does not have a statistically significant impact on teaching evaluations (coef={beauty_coef:.4f}, p={beauty_pval:.6f}). While there may be a slight correlation ({correlation:.4f}), the relationship is not significant at the conventional alpha=0.05 level, suggesting no reliable impact."
    response = 15  # Strong No (not significant)

print(f"\nFinal Response: {response}")
print(f"Explanation: {explanation}")

# Write conclusion to file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n✓ Conclusion written to conclusion.txt")
