import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import statsmodels.api as sm
import json
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('hurricane.csv')

print("=" * 80)
print("HURRICANE NAME FEMININITY AND DEATHS ANALYSIS")
print("=" * 80)
print("\nResearch Question: Hurricanes with more feminine names are perceived as less")
print("threatening and hence lead to fewer precautionary measures by the general public.")
print("\nHypothesis: More feminine names → perceived as less threatening → fewer")
print("precautionary measures → MORE DEATHS")
print("=" * 80)

# Explore the data
print("\n1. DATA EXPLORATION")
print("-" * 80)
print(f"Dataset shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print("\nFirst few rows:")
print(df.head())

print("\n2. KEY VARIABLES SUMMARY")
print("-" * 80)
print("\nFemininity scores (masfem):")
print(df['masfem'].describe())
print("\nDeaths (alldeaths):")
print(df['alldeaths'].describe())
print("\nHurricane severity metrics:")
print(df[['category', 'wind', 'min', 'ndam']].describe())

# Check for correlations
print("\n3. CORRELATION ANALYSIS")
print("-" * 80)
print("\nCorrelation between femininity and deaths:")
corr_fem_deaths = df['masfem'].corr(df['alldeaths'])
print(f"Pearson correlation (masfem vs alldeaths): {corr_fem_deaths:.4f}")

# Spearman correlation (non-parametric, robust to outliers)
spearman_corr, spearman_p = stats.spearmanr(df['masfem'], df['alldeaths'])
print(f"Spearman correlation (masfem vs alldeaths): {spearman_corr:.4f}, p-value: {spearman_p:.4f}")

# Correlation with control variables
print("\nCorrelations with key variables:")
key_vars = ['masfem', 'alldeaths', 'category', 'wind', 'min', 'ndam']
corr_matrix = df[key_vars].corr()
print(corr_matrix[['masfem', 'alldeaths']])

# Test binary gender difference
print("\n4. BINARY GENDER COMPARISON")
print("-" * 80)
male_deaths = df[df['gender_mf'] == 0]['alldeaths']
female_deaths = df[df['gender_mf'] == 1]['alldeaths']
print(f"Male-named hurricanes: mean deaths = {male_deaths.mean():.2f}, median = {male_deaths.median():.2f}")
print(f"Female-named hurricanes: mean deaths = {female_deaths.mean():.2f}, median = {female_deaths.median():.2f}")

# T-test
t_stat, t_pval = stats.ttest_ind(female_deaths, male_deaths)
print(f"\nT-test (female vs male): t-statistic = {t_stat:.4f}, p-value = {t_pval:.4f}")

# Mann-Whitney U test (non-parametric, better for skewed distributions)
u_stat, u_pval = stats.mannwhitneyu(female_deaths, male_deaths, alternative='two-sided')
print(f"Mann-Whitney U test: U-statistic = {u_stat:.4f}, p-value = {u_pval:.4f}")

# 5. REGRESSION ANALYSIS - Controlling for hurricane severity
print("\n5. REGRESSION ANALYSIS (Controlling for Hurricane Severity)")
print("-" * 80)

# Simple linear regression: deaths ~ femininity
print("\n5a. Simple Linear Regression: deaths ~ femininity")
X_simple = df[['masfem']].values
y = df['alldeaths'].values

# Add constant for statsmodels
X_simple_sm = sm.add_constant(X_simple)
model_simple = sm.OLS(y, X_simple_sm).fit()
print(model_simple.summary())

print(f"\nKey results:")
print(f"  Coefficient for masfem: {model_simple.params[1]:.4f}")
print(f"  P-value: {model_simple.pvalues[1]:.4f}")
print(f"  R-squared: {model_simple.rsquared:.4f}")

# Multiple regression: deaths ~ femininity + severity controls
print("\n5b. Multiple Regression: deaths ~ femininity + severity controls")
control_vars = ['masfem', 'category', 'wind', 'min', 'ndam']
df_clean = df[control_vars + ['alldeaths']].dropna()

X_multi = df_clean[control_vars].values
y_multi = df_clean['alldeaths'].values

# Add constant
X_multi_sm = sm.add_constant(X_multi)
model_multi = sm.OLS(y_multi, X_multi_sm).fit()
print(model_multi.summary())

print(f"\nKey results:")
print(f"  Coefficient for masfem: {model_multi.params[1]:.4f}")
print(f"  P-value: {model_multi.pvalues[1]:.4f}")
print(f"  R-squared: {model_multi.rsquared:.4f}")

# 6. LOG-TRANSFORMED ANALYSIS (deaths are highly skewed)
print("\n6. LOG-TRANSFORMED ANALYSIS (Addressing Skewness)")
print("-" * 80)

# Add small constant to avoid log(0)
df['log_deaths'] = np.log(df['alldeaths'] + 1)

print("\n6a. Simple regression: log(deaths) ~ femininity")
X_log_simple = sm.add_constant(df[['masfem']].values)
y_log = df['log_deaths'].values
model_log_simple = sm.OLS(y_log, X_log_simple).fit()
print(model_log_simple.summary())

print(f"\nKey results:")
print(f"  Coefficient for masfem: {model_log_simple.params[1]:.4f}")
print(f"  P-value: {model_log_simple.pvalues[1]:.4f}")
print(f"  R-squared: {model_log_simple.rsquared:.4f}")

print("\n6b. Multiple regression: log(deaths) ~ femininity + controls")
df_log_clean = df[control_vars + ['log_deaths']].dropna()
X_log_multi = sm.add_constant(df_log_clean[control_vars].values)
y_log_multi = df_log_clean['log_deaths'].values
model_log_multi = sm.OLS(y_log_multi, X_log_multi).fit()
print(model_log_multi.summary())

print(f"\nKey results:")
print(f"  Coefficient for masfem: {model_log_multi.params[1]:.4f}")
print(f"  P-value: {model_log_multi.pvalues[1]:.4f}")
print(f"  R-squared: {model_log_multi.rsquared:.4f}")

# 7. INTERPRETABLE MODEL ANALYSIS
print("\n7. INTERPRETABLE MODEL WITH FEATURE IMPORTANCE")
print("-" * 80)

from interpret.glassbox import ExplainableBoostingRegressor

# Prepare data
features = ['masfem', 'category', 'wind', 'ndam']
df_ebm = df[features + ['alldeaths']].dropna()
X_ebm = df_ebm[features]
y_ebm = df_ebm['alldeaths']

# Train EBM
ebm = ExplainableBoostingRegressor(random_state=42)
ebm.fit(X_ebm, y_ebm)

print("\nExplainable Boosting Machine (EBM) Feature Importances:")
# EBM uses term_importances instead of feature_importances_
if hasattr(ebm, 'term_importances'):
    importances = ebm.term_importances()
    feature_importance = list(zip(features, importances))
    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
    for feat, importance in feature_importance:
        print(f"  {feat}: {importance:.4f}")
else:
    print("  EBM model trained successfully but importance values not directly accessible")

# 8. FINAL CONCLUSION
print("\n" + "=" * 80)
print("FINAL ANALYSIS AND CONCLUSION")
print("=" * 80)

# Collect evidence
evidence = []

# Correlation evidence
if abs(corr_fem_deaths) > 0.1 and spearman_p < 0.05:
    evidence.append(f"Significant correlation found (Spearman r={spearman_corr:.3f}, p={spearman_p:.4f})")
else:
    evidence.append(f"Weak/non-significant correlation (Spearman r={spearman_corr:.3f}, p={spearman_p:.4f})")

# T-test evidence
if t_pval < 0.05:
    evidence.append(f"Significant difference in deaths between genders (p={t_pval:.4f})")
else:
    evidence.append(f"No significant difference in deaths between genders (p={t_pval:.4f})")

# Regression evidence
if model_simple.pvalues[1] < 0.05:
    evidence.append(f"Simple regression significant (p={model_simple.pvalues[1]:.4f}, coef={model_simple.params[1]:.2f})")
else:
    evidence.append(f"Simple regression NOT significant (p={model_simple.pvalues[1]:.4f})")

if model_multi.pvalues[1] < 0.05:
    evidence.append(f"Multiple regression significant (p={model_multi.pvalues[1]:.4f}, coef={model_multi.params[1]:.2f})")
else:
    evidence.append(f"Multiple regression NOT significant when controlling for severity (p={model_multi.pvalues[1]:.4f})")

if model_log_simple.pvalues[1] < 0.05:
    evidence.append(f"Log-transformed simple regression significant (p={model_log_simple.pvalues[1]:.4f})")
else:
    evidence.append(f"Log-transformed simple regression NOT significant (p={model_log_simple.pvalues[1]:.4f})")

print("\nEvidence Summary:")
for i, ev in enumerate(evidence, 1):
    print(f"{i}. {ev}")

# Determine response score
# The hypothesis is that more feminine names lead to MORE deaths
# We need strong statistical evidence to support this

# Count significant results
sig_results = sum([
    spearman_p < 0.05 and spearman_corr > 0,  # Positive correlation
    t_pval < 0.05 and female_deaths.mean() > male_deaths.mean(),  # Female names have more deaths
    model_simple.pvalues[1] < 0.05 and model_simple.params[1] > 0,  # Positive coefficient
    model_log_simple.pvalues[1] < 0.05 and model_log_simple.params[1] > 0  # Positive log coefficient
])

# Check if effect persists when controlling for severity
effect_persists = model_multi.pvalues[1] < 0.05 and model_multi.params[1] > 0

# Scoring logic
if sig_results >= 3 and effect_persists:
    response = 85  # Strong evidence
    explanation = "Strong statistical evidence: multiple tests show significant positive relationship between femininity and deaths, even when controlling for hurricane severity."
elif sig_results >= 2 and not effect_persists:
    response = 55  # Moderate evidence, but doesn't hold when controlling
    explanation = "Moderate evidence: some tests show significant relationship, but effect weakens or disappears when controlling for hurricane severity, suggesting confounding."
elif sig_results >= 2:
    response = 70  # Moderate-strong evidence
    explanation = "Moderate to strong evidence: multiple significant tests support the hypothesis that more feminine names are associated with more deaths."
elif sig_results == 1:
    response = 40  # Weak evidence
    explanation = "Weak evidence: only one test shows significant relationship. Not enough consistent evidence to support the hypothesis."
else:
    response = 15  # Very weak/no evidence
    explanation = "Very weak or no evidence: tests show no significant relationship between name femininity and deaths."

# Adjust based on actual p-values
if spearman_p < 0.05:
    if spearman_corr > 0:
        if model_simple.pvalues[1] < 0.05 and model_simple.params[1] > 0:
            response = max(response, 65)
            explanation = "Significant positive correlation (Spearman p<0.05) and regression shows positive effect of femininity on deaths, supporting the hypothesis."
        else:
            response = max(response, 50)
            explanation = "Significant correlation found, but regression analysis shows inconsistent or weak effects. Mixed evidence."
    else:
        response = min(response, 30)
        explanation = "Significant negative correlation found, contradicting the hypothesis that feminine names lead to more deaths."
else:
    if model_simple.pvalues[1] >= 0.05:
        response = min(response, 25)
        explanation = "No significant correlation or regression effect found. Statistical tests do not support the hypothesis that feminine names lead to more deaths."

print(f"\n{'='*80}")
print(f"FINAL SCORE: {response}/100")
print(f"EXPLANATION: {explanation}")
print(f"{'='*80}")

# Write conclusion to file
conclusion = {
    "response": int(response),
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n✓ Conclusion written to conclusion.txt")
