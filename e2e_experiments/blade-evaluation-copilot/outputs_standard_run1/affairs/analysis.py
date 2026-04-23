import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from imodels import RuleFitRegressor, HSTreeRegressor
import json

# Load the dataset
df = pd.read_csv('affairs.csv')

print("="*80)
print("DATASET EXPLORATION")
print("="*80)
print(f"Dataset shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())
print("\nColumn data types:")
print(df.dtypes)
print("\nBasic statistics:")
print(df.describe())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Focus on the research question: Does having children decrease extramarital affairs?
print("\n" + "="*80)
print("RESEARCH QUESTION: Does having children decrease extramarital affairs?")
print("="*80)

# Convert children to binary
df['children_binary'] = (df['children'] == 'yes').astype(int)

# Group by children status
print("\nAffairs by children status:")
affairs_by_children = df.groupby('children')['affairs'].agg(['mean', 'median', 'std', 'count'])
print(affairs_by_children)

# Calculate the proportion of people who had affairs (affairs > 0)
print("\nProportion who had any affairs:")
for child_status in ['yes', 'no']:
    subset = df[df['children'] == child_status]
    prop_affairs = (subset['affairs'] > 0).mean()
    print(f"  {child_status} children: {prop_affairs:.3f} ({(subset['affairs'] > 0).sum()}/{len(subset)})")

# Statistical tests
print("\n" + "="*80)
print("STATISTICAL TESTS")
print("="*80)

# T-test comparing affairs between those with and without children
with_children = df[df['children'] == 'yes']['affairs']
without_children = df[df['children'] == 'no']['affairs']

t_stat, p_value_ttest = stats.ttest_ind(with_children, without_children)
print(f"\nIndependent t-test:")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_value_ttest:.4f}")
print(f"  Mean affairs (with children): {with_children.mean():.3f}")
print(f"  Mean affairs (without children): {without_children.mean():.3f}")
print(f"  Difference: {with_children.mean() - without_children.mean():.3f}")

# Mann-Whitney U test (non-parametric alternative)
u_stat, p_value_mannwhitney = stats.mannwhitneyu(with_children, without_children, alternative='two-sided')
print(f"\nMann-Whitney U test (non-parametric):")
print(f"  U-statistic: {u_stat:.4f}")
print(f"  p-value: {p_value_mannwhitney:.4f}")

# OLS Regression with statsmodels for p-values
print("\n" + "="*80)
print("LINEAR REGRESSION (statsmodels OLS)")
print("="*80)

# Simple regression: affairs ~ children
X_simple = sm.add_constant(df['children_binary'])
y = df['affairs']
model_simple = sm.OLS(y, X_simple).fit()
print("\nSimple regression: affairs ~ children")
print(model_simple.summary())

# Multiple regression controlling for other factors
print("\n" + "="*80)
print("MULTIPLE REGRESSION (controlling for confounds)")
print("="*80)

# Encode gender
df['gender_binary'] = (df['gender'] == 'male').astype(int)

# Create feature matrix
feature_cols = ['children_binary', 'gender_binary', 'age', 'yearsmarried', 
                'religiousness', 'education', 'occupation', 'rating']
X_multi = sm.add_constant(df[feature_cols])
model_multi = sm.OLS(y, X_multi).fit()
print(model_multi.summary())

# Interpretable models
print("\n" + "="*80)
print("INTERPRETABLE MODELS")
print("="*80)

# Decision Tree
print("\nDecision Tree Regressor:")
dt = DecisionTreeRegressor(max_depth=4, min_samples_split=20, random_state=42)
X_features = df[feature_cols]
dt.fit(X_features, y)
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': dt.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance)

# RuleFit model
print("\nRuleFit Model:")
try:
    rulefit = RuleFitRegressor(max_rules=10, random_state=42)
    rulefit.fit(X_features, y)
    print(f"Fitted RuleFit with {len(rulefit.rules_)} rules")
    if hasattr(rulefit, 'feature_importances_'):
        rf_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': rulefit.feature_importances_
        }).sort_values('importance', ascending=False)
        print(rf_importance)
except Exception as e:
    print(f"RuleFit error: {e}")

# HSTree (Hierarchical Shrinkage Tree)
print("\nHSTree Model:")
try:
    hstree = HSTreeRegressor(max_depth=4, random_state=42)
    hstree.fit(X_features, y)
    if hasattr(hstree, 'feature_importances_'):
        hst_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': hstree.feature_importances_
        }).sort_values('importance', ascending=False)
        print(hst_importance)
except Exception as e:
    print(f"HSTree error: {e}")

# Effect size calculation
print("\n" + "="*80)
print("EFFECT SIZE")
print("="*80)
cohen_d = (with_children.mean() - without_children.mean()) / np.sqrt(
    ((len(with_children)-1)*with_children.std()**2 + (len(without_children)-1)*without_children.std()**2) / 
    (len(with_children) + len(without_children) - 2)
)
print(f"Cohen's d: {cohen_d:.4f}")

# Correlation analysis
print("\n" + "="*80)
print("CORRELATION ANALYSIS")
print("="*80)
corr_children_affairs = df['children_binary'].corr(df['affairs'])
print(f"Pearson correlation (children, affairs): {corr_children_affairs:.4f}")
r, p_corr = stats.pearsonr(df['children_binary'], df['affairs'])
print(f"P-value: {p_corr:.4f}")

# CONCLUSION
print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

# Analyze the results
children_coef = model_simple.params['children_binary']
children_pval = model_simple.pvalues['children_binary']

print(f"\nKey findings:")
print(f"1. Mean affairs with children: {with_children.mean():.3f}")
print(f"2. Mean affairs without children: {without_children.mean():.3f}")
print(f"3. Difference: {with_children.mean() - without_children.mean():.3f}")
print(f"4. Simple regression coefficient: {children_coef:.4f} (p={children_pval:.4f})")
print(f"5. T-test p-value: {p_value_ttest:.4f}")
print(f"6. Mann-Whitney p-value: {p_value_mannwhitney:.4f}")

# Decision logic for response score
# The question asks: "Does having children decrease engagement in extramarital affairs?"
# A decrease would mean: with_children < without_children (negative coefficient)

if p_value_ttest < 0.05 and children_coef < 0:
    # Statistically significant decrease
    response_score = 85
    explanation = (
        f"Yes, having children significantly decreases extramarital affairs. "
        f"People with children have {with_children.mean():.2f} affairs on average vs "
        f"{without_children.mean():.2f} without children (difference: {with_children.mean() - without_children.mean():.2f}). "
        f"T-test p-value={p_value_ttest:.4f} and regression coefficient={children_coef:.3f} (p={children_pval:.4f}), "
        f"both indicating statistical significance. The effect size (Cohen's d={cohen_d:.3f}) shows a meaningful difference."
    )
elif p_value_ttest < 0.10 and children_coef < 0:
    # Marginally significant decrease
    response_score = 65
    explanation = (
        f"Having children shows a trend toward decreasing extramarital affairs, but with marginal significance. "
        f"People with children have {with_children.mean():.2f} affairs vs {without_children.mean():.2f} without children. "
        f"T-test p-value={p_value_ttest:.4f} (approaching significance) and regression coefficient={children_coef:.3f}. "
        f"The effect exists but is not strongly significant."
    )
elif children_coef < 0:
    # Non-significant decrease
    response_score = 30
    explanation = (
        f"While people with children have slightly fewer affairs ({with_children.mean():.2f} vs {without_children.mean():.2f}), "
        f"this difference is not statistically significant (t-test p={p_value_ttest:.4f}). "
        f"The regression coefficient is {children_coef:.3f} (p={children_pval:.4f}), suggesting no reliable decrease."
    )
elif p_value_ttest < 0.05 and children_coef > 0:
    # Statistically significant increase (opposite direction)
    response_score = 5
    explanation = (
        f"No, having children does not decrease affairs. In fact, the data shows people with children have "
        f"more affairs ({with_children.mean():.2f} vs {without_children.mean():.2f}), and this is statistically significant "
        f"(t-test p={p_value_ttest:.4f}, coefficient={children_coef:.3f})."
    )
else:
    # No clear effect
    response_score = 40
    explanation = (
        f"There is no clear evidence that having children decreases extramarital affairs. "
        f"Mean affairs: {with_children.mean():.2f} (with children) vs {without_children.mean():.2f} (without). "
        f"Statistical tests show no significant difference (t-test p={p_value_ttest:.4f})."
    )

print(f"\nResponse score: {response_score}")
print(f"Explanation: {explanation}")

# Write conclusion to file
conclusion = {
    "response": response_score,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\nConclusion written to conclusion.txt")
