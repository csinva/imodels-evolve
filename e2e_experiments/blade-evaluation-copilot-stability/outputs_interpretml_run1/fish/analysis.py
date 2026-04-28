#!/usr/bin/env python3
"""
Analysis of fishing data to understand factors influencing fish caught per hour.
Research question: What factors influence the number of fish caught by visitors 
to a national park and how can we estimate the rate of fish caught per hour?
"""

import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from interpret.glassbox import ExplainableBoostingRegressor

# Load the data
df = pd.read_csv('fish.csv')

print("=" * 80)
print("FISHING DATA ANALYSIS")
print("=" * 80)
print("\n1. DATA OVERVIEW")
print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head(10))
print(f"\nSummary statistics:")
print(df.describe())

# Create the target variable: fish caught per hour
# Need to handle cases where hours is very small to avoid division by zero
df['fish_per_hour'] = df.apply(
    lambda row: row['fish_caught'] / row['hours'] if row['hours'] > 0.01 else 0, 
    axis=1
)

print("\n2. TARGET VARIABLE: FISH PER HOUR")
print(f"Mean fish per hour: {df['fish_per_hour'].mean():.4f}")
print(f"Median fish per hour: {df['fish_per_hour'].median():.4f}")
print(f"Std fish per hour: {df['fish_per_hour'].std():.4f}")
print(f"Max fish per hour: {df['fish_per_hour'].max():.4f}")

# Explore correlations
print("\n3. CORRELATION ANALYSIS")
correlations = df[['fish_caught', 'livebait', 'camper', 'persons', 'child', 'hours', 'fish_per_hour']].corr()
print("\nCorrelation matrix:")
print(correlations)

# Focus on correlations with fish_per_hour and fish_caught
print("\nCorrelations with fish_per_hour:")
print(correlations['fish_per_hour'].sort_values(ascending=False))

# Statistical tests for each factor
print("\n4. STATISTICAL TESTS FOR INDIVIDUAL FACTORS")

# Livebait effect
livebait_yes = df[df['livebait'] == 1]['fish_per_hour']
livebait_no = df[df['livebait'] == 0]['fish_per_hour']
t_stat_livebait, p_val_livebait = stats.ttest_ind(livebait_yes, livebait_no)
print(f"\nLivebait effect (t-test):")
print(f"  Mean fish/hr with livebait: {livebait_yes.mean():.4f}")
print(f"  Mean fish/hr without livebait: {livebait_no.mean():.4f}")
print(f"  t-statistic: {t_stat_livebait:.4f}, p-value: {p_val_livebait:.4f}")

# Camper effect
camper_yes = df[df['camper'] == 1]['fish_per_hour']
camper_no = df[df['camper'] == 0]['fish_per_hour']
t_stat_camper, p_val_camper = stats.ttest_ind(camper_yes, camper_no)
print(f"\nCamper effect (t-test):")
print(f"  Mean fish/hr with camper: {camper_yes.mean():.4f}")
print(f"  Mean fish/hr without camper: {camper_no.mean():.4f}")
print(f"  t-statistic: {t_stat_camper:.4f}, p-value: {p_val_camper:.4f}")

# Correlation tests for continuous variables
corr_persons, p_persons = stats.pearsonr(df['persons'], df['fish_per_hour'])
print(f"\nPersons correlation: r={corr_persons:.4f}, p-value: {p_persons:.4f}")

corr_child, p_child = stats.pearsonr(df['child'], df['fish_per_hour'])
print(f"Children correlation: r={corr_child:.4f}, p-value: {p_child:.4f}")

corr_hours, p_hours = stats.pearsonr(df['hours'], df['fish_per_hour'])
print(f"Hours correlation: r={corr_hours:.4f}, p-value: {p_hours:.4f}")

# Build regression model with statsmodels for detailed statistics
print("\n5. MULTIPLE REGRESSION ANALYSIS (OLS)")
X = df[['livebait', 'camper', 'persons', 'child', 'hours']]
y = df['fish_per_hour']

# Add constant for intercept
X_with_const = sm.add_constant(X)
model_ols = sm.OLS(y, X_with_const).fit()
print("\nOLS Regression Results:")
print(model_ols.summary())

# Build interpretable models
print("\n6. INTERPRETABLE MACHINE LEARNING MODELS")

# Linear Regression with scikit-learn
lr_model = LinearRegression()
lr_model.fit(X, y)
print("\nLinear Regression Coefficients:")
for feature, coef in zip(X.columns, lr_model.coef_):
    print(f"  {feature}: {coef:.6f}")
print(f"  Intercept: {lr_model.intercept_:.6f}")

# Cross-validated R-squared
cv_scores = cross_val_score(lr_model, X, y, cv=5, scoring='r2')
print(f"\nCross-validated R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Explainable Boosting Machine
print("\n7. EXPLAINABLE BOOSTING REGRESSOR")
ebm = ExplainableBoostingRegressor(random_state=42)
ebm.fit(X, y)
print("\nEBM Term Importances:")
for feature, importance in zip(X.columns, ebm.term_importances()):
    print(f"  {feature}: {importance:.6f}")

# Also analyze raw fish_caught with hours as predictor
print("\n8. RATE ESTIMATION: PREDICTING FISH CAUGHT FROM HOURS")
X_rate = df[['hours', 'livebait', 'camper', 'persons', 'child']]
y_rate = df['fish_caught']
X_rate_const = sm.add_constant(X_rate)
model_rate = sm.OLS(y_rate, X_rate_const).fit()
print("\nOLS Model for fish_caught:")
print(model_rate.summary())

# Calculate average catch rate
total_fish = df['fish_caught'].sum()
total_hours = df['hours'].sum()
overall_rate = total_fish / total_hours
print(f"\n9. OVERALL STATISTICS")
print(f"Total fish caught: {total_fish}")
print(f"Total hours: {total_hours:.2f}")
print(f"Overall average: {overall_rate:.4f} fish per hour")

# Determine conclusion
print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

# Synthesize findings
significant_factors = []
if p_val_livebait < 0.05:
    significant_factors.append(f"livebait (p={p_val_livebait:.4f})")
if p_val_camper < 0.05:
    significant_factors.append(f"camper (p={p_val_camper:.4f})")
if p_persons < 0.05:
    significant_factors.append(f"persons (p={p_persons:.4f})")
if p_child < 0.05:
    significant_factors.append(f"children (p={p_child:.4f})")
if p_hours < 0.05:
    significant_factors.append(f"hours (p={p_hours:.4f})")

print(f"\nSignificant factors affecting fish caught per hour: {significant_factors}")
print(f"Overall average catch rate: {overall_rate:.4f} fish/hour")
print(f"OLS R-squared: {model_ols.rsquared:.4f}")
print(f"All predictors F-statistic p-value: {model_ols.f_pvalue:.6f}")

# Determine response score and explanation
# The research question asks about factors influencing fish caught and estimating rate per hour
# We found significant factors and can estimate the rate

# Check if we have statistically significant predictors
if model_ols.f_pvalue < 0.05 and len(significant_factors) > 0:
    # We have significant factors
    response_score = 85  # Strong yes
    explanation = (
        f"Yes, we can identify factors and estimate catch rates. "
        f"The overall rate is {overall_rate:.3f} fish/hour. "
        f"Significant factors include: {', '.join(significant_factors)}. "
        f"The regression model (R²={model_ols.rsquared:.3f}, p={model_ols.f_pvalue:.4f}) "
        f"shows these factors significantly influence catch rates."
    )
elif len(significant_factors) > 0:
    # Some factors are significant
    response_score = 70
    explanation = (
        f"Several factors show significant relationships: {', '.join(significant_factors)}. "
        f"The average catch rate is {overall_rate:.3f} fish/hour. "
        f"We can estimate rates though the overall model fit (R²={model_ols.rsquared:.3f}) is modest."
    )
else:
    # No significant factors
    response_score = 40
    explanation = (
        f"The overall average rate is {overall_rate:.3f} fish/hour, "
        f"but no individual factors show strong statistical significance (all p > 0.05). "
        f"The model has limited predictive power (R²={model_ols.rsquared:.3f})."
    )

print(f"\nResponse Score: {response_score}/100")
print(f"Explanation: {explanation}")

# Write conclusion to file
conclusion = {
    "response": response_score,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n" + "=" * 80)
print("Analysis complete! Results written to conclusion.txt")
print("=" * 80)
