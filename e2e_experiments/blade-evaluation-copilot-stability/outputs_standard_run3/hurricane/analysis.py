import json
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('hurricane.csv')

print("=" * 80)
print("HURRICANE DATA ANALYSIS")
print("=" * 80)
print("\nResearch Question:")
print("Hurricanes with more feminine names are perceived as less threatening")
print("and hence lead to fewer precautionary measures by the general public.")
print("\nKey hypothesis: More feminine names -> More deaths (due to less precaution)")
print("=" * 80)

# Basic exploration
print("\n1. DATA EXPLORATION")
print("-" * 80)
print(f"Dataset shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print("\nFirst few rows:")
print(df.head())

print("\n2. SUMMARY STATISTICS")
print("-" * 80)
print(df[['masfem', 'alldeaths', 'category', 'ndam', 'min', 'wind']].describe())

# Key variables:
# masfem: masculinity-femininity index (higher = more feminine)
# alldeaths: total deaths caused by hurricane
# Other controls: category, min pressure, wind speed, ndam (damage)

print("\n3. CORRELATION ANALYSIS")
print("-" * 80)
correlations = df[['masfem', 'alldeaths', 'category', 'ndam', 'min', 'wind']].corr()
print("\nCorrelation with deaths:")
print(correlations['alldeaths'].sort_values(ascending=False))

print("\nCorrelation between masfem and alldeaths:", 
      df['masfem'].corr(df['alldeaths']))

# Simple statistical test: correlation between femininity and deaths
print("\n4. STATISTICAL TEST: Correlation between femininity and deaths")
print("-" * 80)
r, p_value = stats.pearsonr(df['masfem'], df['alldeaths'])
print(f"Pearson correlation coefficient: r = {r:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Significant at 0.05 level: {p_value < 0.05}")

# T-test: comparing male vs female named hurricanes
print("\n5. T-TEST: Male vs Female named hurricanes")
print("-" * 80)
male_deaths = df[df['gender_mf'] == 0]['alldeaths']
female_deaths = df[df['gender_mf'] == 1]['alldeaths']
print(f"Male named hurricanes - mean deaths: {male_deaths.mean():.2f} (n={len(male_deaths)})")
print(f"Female named hurricanes - mean deaths: {female_deaths.mean():.2f} (n={len(female_deaths)})")
t_stat, t_p_value = stats.ttest_ind(male_deaths, female_deaths)
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {t_p_value:.4f}")
print(f"Significant at 0.05 level: {t_p_value < 0.05}")

# Linear regression with statsmodels for p-values
print("\n6. LINEAR REGRESSION (Simple): deaths ~ masfem")
print("-" * 80)
X = sm.add_constant(df['masfem'])
y = df['alldeaths']
model = sm.OLS(y, X).fit()
print(model.summary())

# Multiple regression controlling for hurricane severity
print("\n7. MULTIPLE REGRESSION: Controlling for hurricane severity")
print("-" * 80)
# Use min pressure, wind, and category as controls
control_vars = ['masfem', 'min', 'wind', 'category']
X_controls = df[control_vars].dropna()
y_controls = df.loc[X_controls.index, 'alldeaths']
X_controls = sm.add_constant(X_controls)
model_controls = sm.OLS(y_controls, X_controls).fit()
print(model_controls.summary())

# Check for interaction effects
print("\n8. INTERACTION EFFECT: masfem × damage")
print("-" * 80)
df['masfem_x_ndam'] = df['masfem'] * df['ndam']
interaction_vars = ['masfem', 'ndam', 'masfem_x_ndam', 'min', 'category']
X_interact = df[interaction_vars].dropna()
y_interact = df.loc[X_interact.index, 'alldeaths']
X_interact = sm.add_constant(X_interact)
model_interact = sm.OLS(y_interact, X_interact).fit()
print(model_interact.summary())

# Interpretable model with sklearn
print("\n9. DECISION TREE (Interpretable)")
print("-" * 80)
features = ['masfem', 'min', 'wind', 'category', 'ndam']
X_tree = df[features].dropna()
y_tree = df.loc[X_tree.index, 'alldeaths']
tree = DecisionTreeRegressor(max_depth=3, random_state=42)
tree.fit(X_tree, y_tree)
print("Feature importances:")
for feat, imp in zip(features, tree.feature_importances_):
    print(f"  {feat}: {imp:.4f}")

# Try with imodels if available
try:
    from imodels import FIGSRegressor, HSTreeRegressor
    print("\n10. INTERPRETABLE MODEL (imodels FIGSRegressor)")
    print("-" * 80)
    figs = FIGSRegressor(max_rules=5)
    figs.fit(X_tree, y_tree)
    print("FIGS Rules:")
    print(figs)
    print("\nFeature importances:")
    if hasattr(figs, 'feature_importances_'):
        for feat, imp in zip(features, figs.feature_importances_):
            print(f"  {feat}: {imp:.4f}")
except ImportError:
    print("\n10. imodels not available, skipping")

# Final conclusion
print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

# Determine response based on statistical evidence
# Key findings:
# 1. Correlation test
# 2. T-test
# 3. Regression coefficient and p-value

significant_findings = []
evidence_score = 0

if p_value < 0.05:
    significant_findings.append(f"Correlation test significant (p={p_value:.4f}, r={r:.4f})")
    if r > 0:
        evidence_score += 30
        
if t_p_value < 0.05:
    significant_findings.append(f"T-test significant (p={t_p_value:.4f})")
    if female_deaths.mean() > male_deaths.mean():
        evidence_score += 30

# Check regression coefficient for masfem
masfem_coef = model.params['masfem']
masfem_pval = model.pvalues['masfem']
if masfem_pval < 0.05:
    significant_findings.append(f"Simple regression significant (p={masfem_pval:.4f}, coef={masfem_coef:.2f})")
    if masfem_coef > 0:
        evidence_score += 25

# Check controlled regression
masfem_coef_controlled = model_controls.params['masfem']
masfem_pval_controlled = model_controls.pvalues['masfem']
if masfem_pval_controlled < 0.05:
    significant_findings.append(f"Controlled regression significant (p={masfem_pval_controlled:.4f}, coef={masfem_coef_controlled:.2f})")
    if masfem_coef_controlled > 0:
        evidence_score += 15

print("\nSignificant findings:")
if significant_findings:
    for finding in significant_findings:
        print(f"  • {finding}")
else:
    print("  • No statistically significant relationship found")

# Build explanation
if evidence_score >= 60:
    response = min(90, 60 + evidence_score // 3)
    explanation = (
        f"Strong evidence supports the hypothesis. Femininity index (masfem) shows "
        f"statistically significant positive correlation with deaths (r={r:.3f}, p={p_value:.4f}). "
        f"Regression analysis confirms that more feminine hurricane names are associated with "
        f"higher death tolls (coef={masfem_coef:.2f}, p={masfem_pval:.4f}), even when controlling "
        f"for hurricane severity. This supports the claim that feminine names may lead to fewer "
        f"precautionary measures."
    )
elif evidence_score >= 30:
    response = min(70, 40 + evidence_score)
    explanation = (
        f"Moderate evidence supports the hypothesis. Some statistical tests show significance, "
        f"but the relationship is not consistently strong across all analyses. "
        f"Correlation: r={r:.3f} (p={p_value:.4f}). "
        f"The data suggests a possible relationship between feminine names and deaths, "
        f"but results are mixed when controlling for hurricane severity."
    )
elif evidence_score > 0:
    response = min(50, 20 + evidence_score)
    explanation = (
        f"Weak evidence for the hypothesis. While some tests show marginal significance, "
        f"the relationship is not robust. Correlation: r={r:.3f} (p={p_value:.4f}). "
        f"The effect of name femininity on deaths is unclear and may be confounded by "
        f"other factors."
    )
else:
    response = max(10, 30 - abs(int(r * 30)))
    explanation = (
        f"No significant statistical evidence supports the hypothesis. "
        f"Correlation between femininity and deaths is not significant (r={r:.3f}, p={p_value:.4f}). "
        f"T-test comparing male vs female named hurricanes shows no significant difference "
        f"(p={t_p_value:.4f}). The data does not support the claim that feminine hurricane names "
        f"lead to fewer precautionary measures and more deaths."
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

print("\n" + "=" * 80)
print("Analysis complete. Results written to conclusion.txt")
print("=" * 80)
