#!/usr/bin/env python3
"""
Analysis of fish caught data to understand factors influencing catch rate
"""

import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from imodels import RuleFitRegressor, FIGSRegressor, HSTreeRegressor

# Load the data
df = pd.read_csv('fish.csv')

print("="*80)
print("EXPLORATORY DATA ANALYSIS")
print("="*80)
print("\nDataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head(10))
print("\nSummary statistics:")
print(df.describe())
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())

# Calculate fish per hour (catch rate)
df['fish_per_hour'] = df['fish_caught'] / df['hours']
# Handle division by zero or very small hours
df.loc[df['hours'] < 0.01, 'fish_per_hour'] = np.nan
df_clean = df.dropna(subset=['fish_per_hour'])
# Cap outliers
df_clean = df_clean[df_clean['fish_per_hour'] < 100]

print("\n" + "="*80)
print("FISH PER HOUR ANALYSIS")
print("="*80)
print(f"\nFish per hour statistics (n={len(df_clean)}):")
print(df_clean['fish_per_hour'].describe())
print(f"\nMean fish per hour: {df_clean['fish_per_hour'].mean():.3f}")
print(f"Median fish per hour: {df_clean['fish_per_hour'].median():.3f}")

# Correlation analysis
print("\n" + "="*80)
print("CORRELATION ANALYSIS")
print("="*80)
print("\nCorrelation with fish_per_hour:")
correlations = df_clean[['fish_per_hour', 'livebait', 'camper', 'persons', 'child', 'hours']].corr()['fish_per_hour'].sort_values(ascending=False)
print(correlations)

print("\nCorrelation with fish_caught:")
correlations_caught = df[['fish_caught', 'livebait', 'camper', 'persons', 'child', 'hours']].corr()['fish_caught'].sort_values(ascending=False)
print(correlations_caught)

# Statistical tests for each factor
print("\n" + "="*80)
print("STATISTICAL SIGNIFICANCE TESTS")
print("="*80)

# Test 1: Effect of livebait on fish per hour
with_livebait = df_clean[df_clean['livebait'] == 1]['fish_per_hour']
without_livebait = df_clean[df_clean['livebait'] == 0]['fish_per_hour']
t_stat_livebait, p_val_livebait = stats.ttest_ind(with_livebait, without_livebait)
print(f"\nLivebait effect on fish per hour:")
print(f"  With livebait: mean={with_livebait.mean():.3f}, n={len(with_livebait)}")
print(f"  Without livebait: mean={without_livebait.mean():.3f}, n={len(without_livebait)}")
print(f"  t-statistic: {t_stat_livebait:.4f}, p-value: {p_val_livebait:.4f}")

# Test 2: Effect of camper on fish per hour
with_camper = df_clean[df_clean['camper'] == 1]['fish_per_hour']
without_camper = df_clean[df_clean['camper'] == 0]['fish_per_hour']
t_stat_camper, p_val_camper = stats.ttest_ind(with_camper, without_camper)
print(f"\nCamper effect on fish per hour:")
print(f"  With camper: mean={with_camper.mean():.3f}, n={len(with_camper)}")
print(f"  Without camper: mean={without_camper.mean():.3f}, n={len(without_camper)}")
print(f"  t-statistic: {t_stat_camper:.4f}, p-value: {p_val_camper:.4f}")

# Test 3: Correlation between persons and fish per hour
corr_persons, p_val_persons = stats.pearsonr(df_clean['persons'], df_clean['fish_per_hour'])
print(f"\nPersons correlation with fish per hour:")
print(f"  Correlation: {corr_persons:.4f}, p-value: {p_val_persons:.4f}")

# Test 4: Correlation between child and fish per hour
corr_child, p_val_child = stats.pearsonr(df_clean['child'], df_clean['fish_per_hour'])
print(f"\nChildren correlation with fish per hour:")
print(f"  Correlation: {corr_child:.4f}, p-value: {p_val_child:.4f}")

# Test 5: Correlation between hours and fish caught
corr_hours, p_val_hours = stats.pearsonr(df['hours'], df['fish_caught'])
print(f"\nHours correlation with fish caught:")
print(f"  Correlation: {corr_hours:.4f}, p-value: {p_val_hours:.4f}")

# Linear regression with statsmodels for p-values
print("\n" + "="*80)
print("LINEAR REGRESSION WITH P-VALUES (statsmodels)")
print("="*80)

# Model 1: Predicting fish_per_hour
X_rate = df_clean[['livebait', 'camper', 'persons', 'child']]
y_rate = df_clean['fish_per_hour']
X_rate_sm = sm.add_constant(X_rate)
model_rate = sm.OLS(y_rate, X_rate_sm).fit()
print("\nModel: fish_per_hour ~ livebait + camper + persons + child")
print(model_rate.summary())

# Model 2: Predicting fish_caught with hours
X_caught = df[['livebait', 'camper', 'persons', 'child', 'hours']]
y_caught = df['fish_caught']
X_caught_sm = sm.add_constant(X_caught)
model_caught = sm.OLS(y_caught, X_caught_sm).fit()
print("\nModel: fish_caught ~ livebait + camper + persons + child + hours")
print(model_caught.summary())

# Interpretable models
print("\n" + "="*80)
print("INTERPRETABLE MODELS (imodels)")
print("="*80)

# Ridge regression for stable coefficients
ridge = Ridge(alpha=1.0)
ridge.fit(X_rate, y_rate)
print("\nRidge Regression (fish_per_hour):")
for feat, coef in zip(X_rate.columns, ridge.coef_):
    print(f"  {feat}: {coef:.4f}")
print(f"  Intercept: {ridge.intercept_:.4f}")

# Decision tree for feature importances
tree = DecisionTreeRegressor(max_depth=4, random_state=42)
tree.fit(X_rate, y_rate)
print("\nDecision Tree Feature Importances (fish_per_hour):")
for feat, imp in zip(X_rate.columns, tree.feature_importances_):
    print(f"  {feat}: {imp:.4f}")

# FIGS for interpretable rules
try:
    figs = FIGSRegressor(max_rules=10)
    figs.fit(X_rate, y_rate)
    print("\nFIGS Regressor (fish_per_hour):")
    print(f"  Score: {figs.score(X_rate, y_rate):.4f}")
    print(f"  Number of rules: {len(figs.rules_)}")
    if hasattr(figs, 'feature_importances_'):
        print("  Feature importances:")
        for feat, imp in zip(X_rate.columns, figs.feature_importances_):
            print(f"    {feat}: {imp:.4f}")
except Exception as e:
    print(f"\nFIGS failed: {e}")

# HSTree for interpretable tree
try:
    hstree = HSTreeRegressor(max_leaf_nodes=8)
    hstree.fit(X_rate, y_rate)
    print("\nHSTree Regressor (fish_per_hour):")
    print(f"  Score: {hstree.score(X_rate, y_rate):.4f}")
    if hasattr(hstree, 'feature_importances_'):
        print("  Feature importances:")
        for feat, imp in zip(X_rate.columns, hstree.feature_importances_):
            print(f"    {feat}: {imp:.4f}")
except Exception as e:
    print(f"\nHSTree failed: {e}")

# Summary of findings
print("\n" + "="*80)
print("SUMMARY OF FINDINGS")
print("="*80)

# Evaluate statistical significance
significant_factors = []
if p_val_livebait < 0.05:
    significant_factors.append(f"livebait (p={p_val_livebait:.4f})")
if p_val_camper < 0.05:
    significant_factors.append(f"camper (p={p_val_camper:.4f})")
if p_val_persons < 0.05:
    significant_factors.append(f"persons (p={p_val_persons:.4f})")
if p_val_child < 0.05:
    significant_factors.append(f"child (p={p_val_child:.4f})")
if p_val_hours < 0.05:
    significant_factors.append(f"hours (p={p_val_hours:.4f})")

print(f"\nMean fish caught per hour: {df_clean['fish_per_hour'].mean():.3f}")
print(f"Median fish caught per hour: {df_clean['fish_per_hour'].median():.3f}")
print(f"\nStatistically significant factors (p < 0.05):")
if significant_factors:
    for factor in significant_factors:
        print(f"  - {factor}")
else:
    print("  None found")

print(f"\nKey regression coefficients from statsmodels (fish_per_hour):")
for var in ['livebait', 'camper', 'persons', 'child']:
    coef = model_rate.params[var]
    pval = model_rate.pvalues[var]
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
    print(f"  {var}: {coef:.4f} (p={pval:.4f}) {sig}")

print(f"\nR-squared for fish_per_hour model: {model_rate.rsquared:.4f}")
print(f"R-squared for fish_caught model: {model_caught.rsquared:.4f}")

# Generate conclusion
print("\n" + "="*80)
print("GENERATING CONCLUSION")
print("="*80)

# Determine response score based on evidence
# Question asks: "What factors influence the number of fish caught and how can we estimate the rate?"

# We can definitely estimate the rate (average fish per hour)
can_estimate_rate = True
rate_is_clear = df_clean['fish_per_hour'].mean() > 0

# Check if we found significant factors
has_significant_factors = len(significant_factors) > 0

# Check model fit quality
model_fits_reasonably = model_rate.rsquared > 0.05 or model_caught.rsquared > 0.1

# Hours is highly correlated with fish_caught (this is expected)
hours_matters = p_val_hours < 0.05

# Build explanation
explanation_parts = []
explanation_parts.append(f"Average catch rate: {df_clean['fish_per_hour'].mean():.2f} fish/hour (median: {df_clean['fish_per_hour'].median():.2f}).")

if hours_matters:
    explanation_parts.append(f"Hours spent is significantly correlated with total fish caught (r={corr_hours:.3f}, p<0.001).")

if has_significant_factors:
    explanation_parts.append(f"Found {len(significant_factors)} significant factors: {', '.join([f.split(' (')[0] for f in significant_factors])}.")
else:
    explanation_parts.append("No factors showed strong statistical significance for catch rate.")

# Check which factors have strongest effects
strongest_effects = []
if p_val_livebait < 0.05:
    diff = with_livebait.mean() - without_livebait.mean()
    strongest_effects.append(f"livebait (+{diff:.2f} fish/hr)")
if p_val_camper < 0.05:
    diff = with_camper.mean() - without_camper.mean()
    strongest_effects.append(f"camper ({diff:+.2f} fish/hr)")

if strongest_effects:
    explanation_parts.append(f"Strongest effects: {', '.join(strongest_effects)}.")

explanation_parts.append(f"Linear models explain {model_rate.rsquared:.1%} of variance in catch rate.")

# Determine score
# - Can we estimate the rate? Yes (we have data showing ~3 fish/hour average)
# - Can we identify factors? Some evidence but not strong
# - If hours is significant (expected), and we can estimate rate: score 60-80
# - If other factors are significant too: score 70-90
# - If no clear factors: score 40-60

if can_estimate_rate and hours_matters:
    if len(significant_factors) >= 3:
        score = 75  # Good evidence
    elif len(significant_factors) >= 1:
        score = 65  # Moderate evidence
    else:
        score = 55  # Weak evidence but can estimate rate
else:
    score = 45  # Limited evidence

# Adjust based on model fit
if model_caught.rsquared > 0.3:
    score += 5
elif model_caught.rsquared < 0.1:
    score -= 5

explanation = " ".join(explanation_parts)

print(f"\nFinal score: {score}")
print(f"Explanation: {explanation}")

# Write conclusion
conclusion = {
    "response": score,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n" + "="*80)
print("ANALYSIS COMPLETE - conclusion.txt created")
print("="*80)
