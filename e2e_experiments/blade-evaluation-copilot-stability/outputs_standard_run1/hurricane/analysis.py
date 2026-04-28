import json
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# Load the research question
with open('info.json', 'r') as f:
    info = json.load(f)

research_question = info['research_questions'][0]
print(f"Research Question: {research_question}")
print("="*80)

# Load the dataset
df = pd.read_csv('hurricane.csv')

print(f"\nDataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

# Basic statistics
print("\n" + "="*80)
print("EXPLORATORY DATA ANALYSIS")
print("="*80)

print("\nKey variables summary:")
print(df[['masfem', 'alldeaths', 'category', 'min', 'wind', 'ndam']].describe())

print("\nCorrelation between femininity (masfem) and deaths (alldeaths):")
corr_masfem_deaths = df[['masfem', 'alldeaths']].corr()
print(corr_masfem_deaths)

# Check for missing values
print(f"\nMissing values:")
print(df[['masfem', 'alldeaths', 'category', 'min', 'wind']].isnull().sum())

# Remove rows with missing values for key variables
df_clean = df.dropna(subset=['masfem', 'alldeaths', 'min', 'wind'])
print(f"\nClean dataset shape: {df_clean.shape}")

print("\n" + "="*80)
print("STATISTICAL ANALYSIS")
print("="*80)

# 1. Simple correlation test between femininity and deaths
print("\n1. Pearson correlation between masfem and alldeaths:")
corr, p_value = stats.pearsonr(df_clean['masfem'], df_clean['alldeaths'])
print(f"   Correlation: {corr:.4f}, p-value: {p_value:.4f}")

# 2. Spearman correlation (non-parametric)
print("\n2. Spearman correlation between masfem and alldeaths:")
spearman_corr, spearman_p = stats.spearmanr(df_clean['masfem'], df_clean['alldeaths'])
print(f"   Correlation: {spearman_corr:.4f}, p-value: {spearman_p:.4f}")

# 3. Compare deaths between masculine (masfem < 5.5) and feminine (masfem > 5.5) names
median_masfem = df_clean['masfem'].median()
print(f"\n3. Comparing deaths: masculine vs feminine names (median split at {median_masfem:.2f}):")
masculine = df_clean[df_clean['masfem'] < median_masfem]['alldeaths']
feminine = df_clean[df_clean['masfem'] > median_masfem]['alldeaths']
print(f"   Masculine names: mean deaths = {masculine.mean():.2f}, median = {masculine.median():.2f}")
print(f"   Feminine names: mean deaths = {feminine.mean():.2f}, median = {feminine.median():.2f}")

# t-test
t_stat, t_pval = stats.ttest_ind(masculine, feminine)
print(f"   t-test: t={t_stat:.4f}, p-value: {t_pval:.4f}")

# Mann-Whitney U test (non-parametric)
u_stat, u_pval = stats.mannwhitneyu(masculine, feminine, alternative='two-sided')
print(f"   Mann-Whitney U test: U={u_stat:.4f}, p-value: {u_pval:.4f}")

print("\n" + "="*80)
print("REGRESSION ANALYSIS (Controlling for hurricane severity)")
print("="*80)

# 4. Linear regression: deaths ~ masfem (simple)
print("\n4. Simple Linear Regression: alldeaths ~ masfem")
X_simple = df_clean[['masfem']].values
y = df_clean['alldeaths'].values

X_simple_sm = sm.add_constant(X_simple)
model_simple = sm.OLS(y, X_simple_sm).fit()
print(model_simple.summary().tables[1])
print(f"\n   masfem coefficient: {model_simple.params[1]:.4f}")
print(f"   masfem p-value: {model_simple.pvalues[1]:.4f}")

# 5. Multiple regression: deaths ~ masfem + controls (category, min pressure, wind, damage)
print("\n5. Multiple Regression: alldeaths ~ masfem + category + min + wind")
# Use category, min (pressure), wind as controls for hurricane severity
X_multi = df_clean[['masfem', 'category', 'min', 'wind']].values
X_multi_sm = sm.add_constant(X_multi)
model_multi = sm.OLS(y, X_multi_sm).fit()
print(model_multi.summary().tables[1])
print(f"\n   masfem coefficient (controlled): {model_multi.params[1]:.4f}")
print(f"   masfem p-value (controlled): {model_multi.pvalues[1]:.4f}")

print("\n" + "="*80)
print("INTERPRETABLE MODELS")
print("="*80)

# 6. Decision Tree for interpretability
print("\n6. Decision Tree Regressor (max_depth=3):")
from sklearn.tree import export_text
dt = DecisionTreeRegressor(max_depth=3, random_state=42)
X_tree = df_clean[['masfem', 'category', 'min', 'wind']]
dt.fit(X_tree, y)
feature_names = ['masfem', 'category', 'min', 'wind']
tree_rules = export_text(dt, feature_names=feature_names, max_depth=3)
print(tree_rules)

# Feature importances
print("\n   Feature importances:")
for name, importance in zip(feature_names, dt.feature_importances_):
    print(f"   {name}: {importance:.4f}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

# Analyze the evidence
print("\nEvidence summary:")
print(f"1. Pearson correlation: r={corr:.4f}, p={p_value:.4f}")
print(f"2. Spearman correlation: r={spearman_corr:.4f}, p={spearman_p:.4f}")
print(f"3. t-test (masculine vs feminine): p={t_pval:.4f}")
print(f"4. Simple regression (masfem coef): β={model_simple.params[1]:.4f}, p={model_simple.pvalues[1]:.4f}")
print(f"5. Multiple regression (masfem coef): β={model_multi.params[1]:.4f}, p={model_multi.pvalues[1]:.4f}")

# Decision logic
# The research question asks if hurricanes with more feminine names lead to fewer precautionary measures
# and hence more deaths. This would mean a POSITIVE relationship between femininity and deaths.

alpha = 0.05  # significance threshold

# Check if there's a significant positive relationship
is_positive_corr = corr > 0
is_significant_pearson = p_value < alpha
is_significant_spearman = spearman_p < alpha
is_significant_simple_reg = model_simple.pvalues[1] < alpha and model_simple.params[1] > 0
is_significant_multi_reg = model_multi.pvalues[1] < alpha and model_multi.params[1] > 0

# Count significant positive findings
significant_count = sum([
    is_positive_corr and is_significant_pearson,
    is_positive_corr and is_significant_spearman,
    is_significant_simple_reg,
    is_significant_multi_reg
])

print(f"\nSignificant positive relationships found: {significant_count}/4")

# Determine response score
if significant_count >= 3:
    # Strong evidence for the hypothesis
    response = 85
    explanation = (
        f"Strong evidence supports the hypothesis. Femininity (masfem) shows a significant positive "
        f"correlation with deaths (r={corr:.3f}, p={p_value:.4f}). Simple regression shows masfem "
        f"coefficient β={model_simple.params[1]:.2f} (p={model_simple.pvalues[1]:.4f}), and this "
        f"relationship remains when controlling for hurricane severity (β={model_multi.params[1]:.2f}, "
        f"p={model_multi.pvalues[1]:.4f}). More feminine-named hurricanes are associated with higher deaths."
    )
elif significant_count >= 2:
    # Moderate evidence
    response = 70
    explanation = (
        f"Moderate evidence supports the hypothesis. Multiple statistical tests show a positive relationship "
        f"between femininity and deaths (correlation r={corr:.3f}, p={p_value:.4f}). "
        f"Simple regression β={model_simple.params[1]:.2f} (p={model_simple.pvalues[1]:.4f}). "
        f"However, when controlling for hurricane severity, the effect is {'significant' if is_significant_multi_reg else 'not significant'} "
        f"(p={model_multi.pvalues[1]:.4f})."
    )
elif significant_count == 1:
    # Weak evidence
    response = 55
    explanation = (
        f"Weak evidence for the hypothesis. Some tests show a positive trend between femininity and deaths "
        f"(r={corr:.3f}), but results are inconsistent or marginally significant (p={p_value:.4f}). "
        f"When controlling for hurricane severity, the relationship has p={model_multi.pvalues[1]:.4f}."
    )
else:
    # No evidence or negative evidence
    if corr < 0:
        response = 15
        explanation = (
            f"No support for the hypothesis. The correlation between femininity and deaths is "
            f"negative (r={corr:.3f}), opposite to what the hypothesis predicts. "
            f"None of the statistical tests show a significant positive relationship."
        )
    else:
        response = 30
        explanation = (
            f"Insufficient evidence for the hypothesis. While the correlation is positive (r={corr:.3f}), "
            f"it is not statistically significant (p={p_value:.4f}). Neither simple nor controlled "
            f"regression analyses show significant effects (p={model_simple.pvalues[1]:.4f}, "
            f"p={model_multi.pvalues[1]:.4f})."
        )

print(f"\nFinal Response: {response}/100")
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
