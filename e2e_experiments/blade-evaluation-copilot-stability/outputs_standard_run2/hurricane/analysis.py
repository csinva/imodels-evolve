import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import statsmodels.api as sm
import json

# Load the data
df = pd.read_csv('hurricane.csv')

print("=" * 80)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 80)
print("\nDataset shape:", df.shape)
print("\nColumn names:", df.columns.tolist())
print("\nBasic statistics:")
print(df.describe())

print("\n" + "=" * 80)
print("RESEARCH QUESTION ANALYSIS")
print("=" * 80)
print("\nResearch Question: Hurricanes with more feminine names are perceived as less")
print("threatening and hence lead to fewer precautionary measures by the general public.")
print("\nHypothesis: More feminine hurricane names → less perceived threat → fewer")
print("precautions → higher death toll")
print("\nKey variables:")
print("  - masfem: Masculinity-femininity index (1=very masculine, 11=very feminine)")
print("  - alldeaths: Total number of deaths caused by the hurricane")
print("  - Control variables: min (pressure), wind, category, ndam (damage)")

# Check for missing values
print("\n" + "=" * 80)
print("DATA QUALITY CHECK")
print("=" * 80)
print("\nMissing values:")
print(df.isnull().sum())

# Focus on key variables
key_vars = ['masfem', 'alldeaths', 'min', 'wind', 'category', 'ndam', 'gender_mf']
df_analysis = df[key_vars].dropna()
print(f"\nRows available for analysis: {len(df_analysis)} out of {len(df)}")

print("\n" + "=" * 80)
print("CORRELATION ANALYSIS")
print("=" * 80)
corr_matrix = df_analysis[['masfem', 'alldeaths', 'min', 'wind', 'category', 'ndam']].corr()
print("\nCorrelation matrix:")
print(corr_matrix)

# Correlation between femininity and deaths
corr_fem_deaths = df_analysis['masfem'].corr(df_analysis['alldeaths'])
print(f"\nCorrelation between masfem and alldeaths: {corr_fem_deaths:.4f}")

# Spearman correlation (non-parametric)
spearman_corr, spearman_pval = stats.spearmanr(df_analysis['masfem'], df_analysis['alldeaths'])
print(f"Spearman correlation: {spearman_corr:.4f}, p-value: {spearman_pval:.4f}")

print("\n" + "=" * 80)
print("SIMPLE LINEAR REGRESSION: DEATHS ~ FEMININITY")
print("=" * 80)

# Univariate regression: deaths ~ femininity
X_simple = sm.add_constant(df_analysis['masfem'])
y = df_analysis['alldeaths']
model_simple = sm.OLS(y, X_simple).fit()
print(model_simple.summary())

print("\nSimple regression interpretation:")
print(f"  - Coefficient for masfem: {model_simple.params['masfem']:.4f}")
print(f"  - P-value: {model_simple.pvalues['masfem']:.4f}")
print(f"  - R-squared: {model_simple.rsquared:.4f}")

if model_simple.pvalues['masfem'] < 0.05:
    print(f"  - Result: SIGNIFICANT positive relationship (p < 0.05)")
else:
    print(f"  - Result: NO significant relationship (p >= 0.05)")

print("\n" + "=" * 80)
print("MULTIPLE REGRESSION: CONTROLLING FOR HURRICANE SEVERITY")
print("=" * 80)

# Multiple regression controlling for hurricane characteristics
# Deaths might be influenced by both name femininity AND hurricane severity
# We need to control for confounds
X_multi = sm.add_constant(df_analysis[['masfem', 'min', 'wind', 'ndam']])
model_multi = sm.OLS(df_analysis['alldeaths'], X_multi).fit()
print(model_multi.summary())

print("\nMultiple regression interpretation (controlling for severity):")
print(f"  - Coefficient for masfem: {model_multi.params['masfem']:.4f}")
print(f"  - P-value: {model_multi.pvalues['masfem']:.4f}")
print(f"  - R-squared: {model_multi.rsquared:.4f}")

if model_multi.pvalues['masfem'] < 0.05:
    print(f"  - Result: SIGNIFICANT effect even after controlling for severity")
else:
    print(f"  - Result: NO significant effect after controlling for severity")

print("\n" + "=" * 80)
print("LOG-TRANSFORMED ANALYSIS (for skewed death data)")
print("=" * 80)

# Deaths are highly skewed - try log transformation
df_analysis['log_deaths'] = np.log1p(df_analysis['alldeaths'])
df_analysis['log_ndam'] = np.log1p(df_analysis['ndam'])

X_log = sm.add_constant(df_analysis[['masfem', 'min', 'wind', 'log_ndam']])
model_log = sm.OLS(df_analysis['log_deaths'], X_log).fit()
print(model_log.summary())

print("\nLog-transformed regression interpretation:")
print(f"  - Coefficient for masfem: {model_log.params['masfem']:.4f}")
print(f"  - P-value: {model_log.pvalues['masfem']:.4f}")
print(f"  - R-squared: {model_log.rsquared:.4f}")

print("\n" + "=" * 80)
print("INTERPRETABLE MODELS FROM IMODELS")
print("=" * 80)

from imodels import FIGSRegressor, HSTreeRegressor

# Prepare data for tree models
X_tree = df_analysis[['masfem', 'min', 'wind', 'ndam']].values
y_tree = df_analysis['alldeaths'].values

# FIGS - Fast Interpretable Greedy-tree Sums
print("\nFIGS Regressor (interpretable tree-based model):")
try:
    figs = FIGSRegressor(max_rules=5)
    figs.fit(X_tree, y_tree)
    print("FIGS model trained successfully")
    print(f"Number of rules: {len(figs.trees_)}")
    print("\nFeature importances:")
    feature_names = ['masfem', 'min', 'wind', 'ndam']
    if hasattr(figs, 'feature_importances_'):
        for i, imp in enumerate(figs.feature_importances_):
            print(f"  {feature_names[i]}: {imp:.4f}")
except Exception as e:
    print(f"FIGS model error: {e}")

# HSTree - Hierarchical Shrinkage Tree
print("\nHSTree Regressor (interpretable hierarchical tree):")
try:
    hstree = HSTreeRegressor(max_leaf_nodes=10)
    hstree.fit(X_tree, y_tree)
    print("HSTree model trained successfully")
    if hasattr(hstree, 'feature_importances_'):
        print("\nFeature importances:")
        for i, imp in enumerate(hstree.feature_importances_):
            print(f"  {feature_names[i]}: {imp:.4f}")
except Exception as e:
    print(f"HSTree model error: {e}")

print("\n" + "=" * 80)
print("COMPARING MASCULINE VS FEMININE NAMED HURRICANES")
print("=" * 80)

# Split into masculine vs feminine (using median or binary indicator)
median_masfem = df_analysis['masfem'].median()
df_analysis['is_feminine'] = (df_analysis['masfem'] > median_masfem).astype(int)

feminine_hurricanes = df_analysis[df_analysis['is_feminine'] == 1]
masculine_hurricanes = df_analysis[df_analysis['is_feminine'] == 0]

print(f"\nMedian masfem split point: {median_masfem:.2f}")
print(f"Feminine hurricanes (masfem > median): {len(feminine_hurricanes)}")
print(f"Masculine hurricanes (masfem <= median): {len(masculine_hurricanes)}")

print(f"\nMean deaths for feminine hurricanes: {feminine_hurricanes['alldeaths'].mean():.2f}")
print(f"Mean deaths for masculine hurricanes: {masculine_hurricanes['alldeaths'].mean():.2f}")

# T-test
t_stat, t_pval = stats.ttest_ind(feminine_hurricanes['alldeaths'], 
                                   masculine_hurricanes['alldeaths'])
print(f"\nT-test comparing deaths between feminine and masculine named hurricanes:")
print(f"  - T-statistic: {t_stat:.4f}")
print(f"  - P-value: {t_pval:.4f}")

if t_pval < 0.05:
    print(f"  - Result: SIGNIFICANT difference (p < 0.05)")
else:
    print(f"  - Result: NO significant difference (p >= 0.05)")

# Mann-Whitney U test (non-parametric alternative)
u_stat, u_pval = stats.mannwhitneyu(feminine_hurricanes['alldeaths'], 
                                     masculine_hurricanes['alldeaths'],
                                     alternative='two-sided')
print(f"\nMann-Whitney U test (non-parametric):")
print(f"  - U-statistic: {u_stat:.4f}")
print(f"  - P-value: {u_pval:.4f}")

print("\n" + "=" * 80)
print("FINAL CONCLUSION")
print("=" * 80)

# Synthesize all evidence
significant_tests = []
all_pvalues = []

# Collect p-values from various tests
tests_info = [
    ("Simple correlation (Spearman)", spearman_pval),
    ("Simple linear regression", model_simple.pvalues['masfem']),
    ("Multiple regression (controlled)", model_multi.pvalues['masfem']),
    ("Log-transformed regression", model_log.pvalues['masfem']),
    ("T-test (masculine vs feminine)", t_pval),
    ("Mann-Whitney U test", u_pval)
]

print("\nSummary of statistical tests:")
for test_name, pval in tests_info:
    all_pvalues.append(pval)
    is_sig = pval < 0.05
    significant_tests.append(is_sig)
    sig_str = "SIGNIFICANT" if is_sig else "NOT significant"
    print(f"  - {test_name}: p={pval:.4f} [{sig_str}]")

num_significant = sum(significant_tests)
proportion_significant = num_significant / len(significant_tests)

print(f"\nTests showing significance: {num_significant}/{len(significant_tests)} ({proportion_significant*100:.1f}%)")

# Check direction of effect
positive_relationship = model_simple.params['masfem'] > 0
print(f"\nDirection of relationship: {'POSITIVE' if positive_relationship else 'NEGATIVE'}")
print(f"  (Higher femininity score → {'more' if positive_relationship else 'fewer'} deaths)")

# Determine final response
print("\n" + "-" * 80)
print("FINAL ASSESSMENT:")
print("-" * 80)

# Calculate a response score based on evidence
response_score = 0
explanation_parts = []

if proportion_significant >= 0.5:
    # Majority of tests are significant
    if positive_relationship:
        # Positive relationship supports the hypothesis
        response_score = 60 + int(proportion_significant * 40)
        explanation_parts.append(
            f"{num_significant} out of {len(significant_tests)} statistical tests show "
            f"a significant positive relationship between hurricane name femininity and deaths."
        )
        if model_multi.pvalues['masfem'] < 0.05:
            explanation_parts.append(
                "The relationship remains significant even after controlling for hurricane severity "
                "(pressure, wind speed, damage)."
            )
        else:
            explanation_parts.append(
                "However, the relationship becomes non-significant when controlling for hurricane "
                "severity, suggesting confounding factors may play a role."
            )
    else:
        # Negative relationship contradicts hypothesis
        response_score = 20
        explanation_parts.append(
            "While some tests show significance, the relationship is in the opposite direction "
            "of what the hypothesis predicts (more feminine names associated with fewer deaths)."
        )
else:
    # Majority of tests are NOT significant
    response_score = 10 + int(proportion_significant * 30)
    explanation_parts.append(
        f"Only {num_significant} out of {len(significant_tests)} statistical tests show "
        f"a significant relationship between hurricane name femininity and deaths."
    )
    explanation_parts.append(
        "The lack of consistent statistical significance suggests that the proposed relationship "
        "is weak or may not exist in this dataset."
    )

# Additional context about effect size
if abs(corr_fem_deaths) < 0.2:
    explanation_parts.append(
        f"The correlation is weak (r={corr_fem_deaths:.3f}), indicating femininity of names "
        "explains little variance in death tolls."
    )
elif abs(corr_fem_deaths) >= 0.3:
    explanation_parts.append(
        f"The correlation is moderate (r={corr_fem_deaths:.3f}), suggesting some association "
        "between name femininity and deaths."
    )

explanation = " ".join(explanation_parts)

print(f"\nResponse score: {response_score}/100")
print(f"Explanation: {explanation}")

# Write conclusion to file
conclusion = {
    "response": response_score,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n" + "=" * 80)
print("Analysis complete! conclusion.txt has been created.")
print("=" * 80)
