#!/usr/bin/env python3
"""
Analysis script for capuchin monkey intergroup contest data.
Research question: How do relative group size and contest location influence 
the probability of a capuchin monkey group winning an intergroup contest?
"""

import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from interpret.glassbox import ExplainableBoostingClassifier

# Load the data
df = pd.read_csv('crofoot.csv')

print("="*80)
print("DATA EXPLORATION")
print("="*80)
print(f"\nDataset shape: {df.shape}")
print(f"\nColumn names: {df.columns.tolist()}")
print("\nFirst few rows:")
print(df.head())
print("\nSummary statistics:")
print(df.describe())
print("\nMissing values:")
print(df.isnull().sum())

# Create derived features that are relevant to the research question
# 1. Relative group size (focal vs other)
df['relative_size'] = df['n_focal'] - df['n_other']
df['size_ratio'] = df['n_focal'] / df['n_other']

# 2. Contest location (distance from home range center)
# Lower dist_focal means closer to focal's home (advantage for focal)
# Higher dist_other means farther from other's home (advantage for focal)
df['location_advantage'] = df['dist_other'] - df['dist_focal']

print("\n" + "="*80)
print("DERIVED FEATURES")
print("="*80)
print("\nRelative size (n_focal - n_other):")
print(df['relative_size'].describe())
print("\nLocation advantage (dist_other - dist_focal):")
print(df['location_advantage'].describe())

# Analyze win rate by different factors
print("\n" + "="*80)
print("WIN RATE ANALYSIS")
print("="*80)
print(f"\nOverall win rate for focal group: {df['win'].mean():.3f}")

# Win rate by relative group size
print("\n--- Win rate by relative group size ---")
for rel_size in sorted(df['relative_size'].unique()):
    wins = df[df['relative_size'] == rel_size]['win'].mean()
    count = len(df[df['relative_size'] == rel_size])
    print(f"Relative size {rel_size:+2d}: {wins:.3f} (n={count})")

# Win rate by location advantage
print("\n--- Win rate by location advantage quartiles ---")
df['location_quartile'] = pd.qcut(df['location_advantage'], q=4, labels=['Q1(low)', 'Q2', 'Q3', 'Q4(high)'])
for quartile in df['location_quartile'].cat.categories:
    wins = df[df['location_quartile'] == quartile]['win'].mean()
    count = len(df[df['location_quartile'] == quartile])
    print(f"{quartile}: {wins:.3f} (n={count})")

print("\n" + "="*80)
print("STATISTICAL TESTS")
print("="*80)

# Test 1: Correlation between relative size and winning
print("\n--- Correlation: Relative size vs Winning ---")
corr_size, p_size = stats.pointbiserialr(df['win'], df['relative_size'])
print(f"Point-biserial correlation: r = {corr_size:.4f}, p = {p_size:.4f}")

# Test 2: Correlation between location advantage and winning
print("\n--- Correlation: Location advantage vs Winning ---")
corr_loc, p_loc = stats.pointbiserialr(df['win'], df['location_advantage'])
print(f"Point-biserial correlation: r = {corr_loc:.4f}, p = {p_loc:.4f}")

# Test 3: Logistic regression with statsmodels for p-values
print("\n--- Logistic Regression (statsmodels) ---")
X = df[['relative_size', 'location_advantage']].copy()
X = sm.add_constant(X)
y = df['win']

logit_model = sm.Logit(y, X)
result = logit_model.fit(disp=0)
print(result.summary())

print("\n--- Key coefficients and p-values ---")
for var in ['relative_size', 'location_advantage']:
    coef = result.params[var]
    pval = result.pvalues[var]
    print(f"{var:20s}: coef={coef:8.4f}, p={pval:.4f}, significant={'YES' if pval < 0.05 else 'NO'}")

print("\n" + "="*80)
print("INTERPRETABLE MODELS")
print("="*80)

# Scikit-learn Logistic Regression for comparison
print("\n--- Scikit-learn Logistic Regression ---")
X_features = df[['relative_size', 'location_advantage']]
y = df['win']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

lr = LogisticRegression(random_state=42)
lr.fit(X_scaled, y)
print(f"Accuracy: {lr.score(X_scaled, y):.3f}")
print("Coefficients:")
for i, col in enumerate(X_features.columns):
    print(f"  {col:20s}: {lr.coef_[0][i]:.4f}")

# Explainable Boosting Classifier
print("\n--- Explainable Boosting Classifier ---")
ebc = ExplainableBoostingClassifier(random_state=42, max_bins=16)
ebc.fit(X_features, y)
print(f"Accuracy: {ebc.score(X_features, y):.3f}")
print("\nFeature importances:")
for i, col in enumerate(X_features.columns):
    importance = abs(ebc.term_importances_[i]) if hasattr(ebc, 'term_importances_') else 0
    print(f"  {col:20s}: {importance:.4f}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

# Determine response based on statistical significance
# Research question: Do relative group size AND contest location influence winning probability?

# Check significance at alpha = 0.05
size_significant = p_size < 0.05
location_significant = p_loc < 0.05

# From logistic regression
size_sig_logit = result.pvalues['relative_size'] < 0.05
location_sig_logit = result.pvalues['location_advantage'] < 0.05

print(f"\nRelative size correlation: p = {p_size:.4f} ({'significant' if size_significant else 'not significant'})")
print(f"Location advantage correlation: p = {p_loc:.4f} ({'significant' if location_significant else 'not significant'})")
print(f"\nLogistic regression p-values:")
print(f"  Relative size: p = {result.pvalues['relative_size']:.4f} ({'significant' if size_sig_logit else 'not significant'})")
print(f"  Location advantage: p = {result.pvalues['location_advantage']:.4f} ({'significant' if location_sig_logit else 'not significant'})")

# Determine response score
# Both factors significant: 100 (strong yes)
# One factor significant: 50-70 (moderate yes)
# Neither significant: 0-20 (strong no)

if size_sig_logit and location_sig_logit:
    response = 95
    explanation = (
        "Both relative group size and contest location significantly influence winning probability. "
        f"Logistic regression shows relative_size (p={result.pvalues['relative_size']:.3f}) and "
        f"location_advantage (p={result.pvalues['location_advantage']:.3f}) are both significant predictors. "
        f"Larger groups have higher win rates (correlation r={corr_size:.3f}), and groups fighting closer to "
        f"their home range center also have an advantage (correlation r={corr_loc:.3f})."
    )
elif size_sig_logit or location_sig_logit:
    response = 65
    sig_factor = "relative group size" if size_sig_logit else "contest location"
    explanation = (
        f"Only {sig_factor} shows a significant influence on winning probability "
        f"(p < 0.05 in logistic regression). "
        f"Relative size: p={result.pvalues['relative_size']:.3f}, "
        f"Location advantage: p={result.pvalues['location_advantage']:.3f}. "
        f"The evidence partially supports the hypothesis."
    )
else:
    response = 15
    explanation = (
        f"Neither relative group size (p={result.pvalues['relative_size']:.3f}) nor "
        f"contest location (p={result.pvalues['location_advantage']:.3f}) show "
        f"statistically significant influence on winning probability at alpha=0.05. "
        f"The data does not provide strong evidence for the hypothesis."
    )

print(f"\nResponse score: {response}/100")
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
