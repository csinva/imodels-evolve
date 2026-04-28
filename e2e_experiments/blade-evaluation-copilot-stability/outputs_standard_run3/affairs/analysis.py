import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
import json

# Load data
df = pd.read_csv('affairs.csv')

print("=" * 80)
print("RESEARCH QUESTION: Does having children decrease (if at all) the engagement in extramarital affairs?")
print("=" * 80)

# Data exploration
print("\nDataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nData summary:")
print(df.describe())
print("\nColumn types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())

# Focus on the key variables: children and affairs
print("\n" + "=" * 80)
print("KEY VARIABLE ANALYSIS: CHILDREN AND AFFAIRS")
print("=" * 80)

# Distribution of children variable
print("\nDistribution of 'children' variable:")
print(df['children'].value_counts())

# Distribution of affairs
print("\nDistribution of 'affairs' (extramarital activity):")
print(df['affairs'].value_counts().sort_index())

# Create binary affair indicator for easier interpretation
df['had_affair'] = (df['affairs'] > 0).astype(int)

# Group statistics
print("\n" + "=" * 80)
print("COMPARISON: AFFAIRS BY CHILDREN STATUS")
print("=" * 80)

grouped = df.groupby('children').agg({
    'affairs': ['mean', 'median', 'std', 'count'],
    'had_affair': ['mean', 'sum']
})
print(grouped)

# Separate groups
with_children = df[df['children'] == 'yes']['affairs']
without_children = df[df['children'] == 'no']['affairs']

print(f"\nMean affairs with children: {with_children.mean():.3f}")
print(f"Mean affairs without children: {without_children.mean():.3f}")
print(f"Difference: {with_children.mean() - without_children.mean():.3f}")

# Binary affair rates
with_children_binary = df[df['children'] == 'yes']['had_affair']
without_children_binary = df[df['children'] == 'no']['had_affair']

print(f"\nAffair rate with children: {with_children_binary.mean():.3f} ({with_children_binary.sum()}/{len(with_children_binary)})")
print(f"Affair rate without children: {without_children_binary.mean():.3f} ({without_children_binary.sum()}/{len(without_children_binary)})")

# Statistical tests
print("\n" + "=" * 80)
print("STATISTICAL TESTS")
print("=" * 80)

# T-test for continuous affairs variable
t_stat, p_value_ttest = stats.ttest_ind(with_children, without_children)
print(f"\nT-test (affairs count):")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_value_ttest:.4f}")
print(f"  Significant at α=0.05? {p_value_ttest < 0.05}")

# Mann-Whitney U test (non-parametric alternative)
u_stat, p_value_mann = stats.mannwhitneyu(with_children, without_children, alternative='two-sided')
print(f"\nMann-Whitney U test (affairs count):")
print(f"  U-statistic: {u_stat:.4f}")
print(f"  p-value: {p_value_mann:.4f}")
print(f"  Significant at α=0.05? {p_value_mann < 0.05}")

# Chi-square test for binary affair indicator
contingency_table = pd.crosstab(df['children'], df['had_affair'])
print(f"\nContingency table (children vs had_affair):")
print(contingency_table)

chi2, p_value_chi2, dof, expected = stats.chi2_contingency(contingency_table)
print(f"\nChi-square test (binary affair indicator):")
print(f"  Chi-square statistic: {chi2:.4f}")
print(f"  p-value: {p_value_chi2:.4f}")
print(f"  Degrees of freedom: {dof}")
print(f"  Significant at α=0.05? {p_value_chi2 < 0.05}")

# Regression analysis with statsmodels for p-values
print("\n" + "=" * 80)
print("REGRESSION ANALYSIS")
print("=" * 80)

# Encode children as binary (1 = yes, 0 = no)
df['children_binary'] = (df['children'] == 'yes').astype(int)

# Simple linear regression: affairs ~ children
X_simple = sm.add_constant(df['children_binary'])
y = df['affairs']
model_simple = sm.OLS(y, X_simple).fit()
print("\nSimple Linear Regression: affairs ~ children")
print(model_simple.summary())

# Multiple regression controlling for other factors
print("\n" + "=" * 80)
print("MULTIPLE REGRESSION (controlling for confounders)")
print("=" * 80)

# Encode gender
df['gender_binary'] = (df['gender'] == 'male').astype(int)

# Build feature matrix
predictors = ['children_binary', 'gender_binary', 'age', 'yearsmarried', 
              'religiousness', 'education', 'occupation', 'rating']
X_multi = sm.add_constant(df[predictors])
model_multi = sm.OLS(y, X_multi).fit()
print("\nMultiple Linear Regression:")
print(model_multi.summary())

# Interpretable model using sklearn
print("\n" + "=" * 80)
print("INTERPRETABLE MODEL: Decision Tree")
print("=" * 80)

from sklearn.tree import DecisionTreeRegressor, export_text
X_features = df[predictors]
tree_model = DecisionTreeRegressor(max_depth=4, min_samples_leaf=20, random_state=42)
tree_model.fit(X_features, y)

print("\nDecision Tree Rules:")
tree_rules = export_text(tree_model, feature_names=predictors)
print(tree_rules)

print("\nFeature importances:")
for name, importance in sorted(zip(predictors, tree_model.feature_importances_), 
                               key=lambda x: x[1], reverse=True):
    print(f"  {name}: {importance:.4f}")

# Final conclusion
print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

# Synthesize results
children_coef = model_simple.params['children_binary']
children_pval = model_simple.pvalues['children_binary']
children_coef_multi = model_multi.params['children_binary']
children_pval_multi = model_multi.pvalues['children_binary']

print(f"\nSimple regression coefficient for children: {children_coef:.4f} (p={children_pval:.4f})")
print(f"Multiple regression coefficient for children: {children_coef_multi:.4f} (p={children_pval_multi:.4f})")
print(f"\nT-test p-value: {p_value_ttest:.4f}")
print(f"Mann-Whitney p-value: {p_value_mann:.4f}")
print(f"Chi-square p-value: {p_value_chi2:.4f}")

# Determine response
# The question asks if having children DECREASES affairs
# A negative coefficient with p < 0.05 would indicate YES (high score)
# A positive coefficient or non-significant result would indicate NO (low score)

is_significant = (children_pval_multi < 0.05)
is_negative = (children_coef_multi < 0)

if is_significant and is_negative:
    # Strong evidence that children decrease affairs
    response = 85
    explanation = (
        f"Yes, having children is associated with decreased extramarital affairs. "
        f"Multiple regression shows children decrease affairs by {abs(children_coef_multi):.3f} "
        f"(p={children_pval_multi:.4f}, significant at α=0.05). "
        f"This relationship holds even when controlling for age, marriage duration, religiousness, "
        f"education, occupation, and marital satisfaction. "
        f"Mean affairs: {with_children.mean():.2f} (with children) vs {without_children.mean():.2f} (without). "
        f"Multiple statistical tests (t-test p={p_value_ttest:.4f}, Mann-Whitney p={p_value_mann:.4f}) "
        f"confirm this relationship."
    )
elif is_significant and not is_negative:
    # Significant positive effect (children increase affairs) - opposite of question
    response = 5
    explanation = (
        f"No, the data shows the opposite: having children is associated with INCREASED affairs. "
        f"Regression coefficient: {children_coef_multi:.3f} (p={children_pval_multi:.4f}). "
        f"Mean affairs: {with_children.mean():.2f} (with children) vs {without_children.mean():.2f} (without)."
    )
elif not is_significant:
    # No significant relationship
    if abs(children_coef_multi) < 0.1:
        response = 50
        explanation = (
            f"The relationship is unclear. While the regression coefficient is {children_coef_multi:.3f}, "
            f"it is not statistically significant (p={children_pval_multi:.4f}). "
            f"Mean affairs: {with_children.mean():.2f} (with children) vs {without_children.mean():.2f} (without). "
            f"The evidence is insufficient to conclude that children decrease affairs."
        )
    else:
        response = 30
        explanation = (
            f"No significant relationship found. Regression coefficient: {children_coef_multi:.3f} "
            f"(p={children_pval_multi:.4f}, not significant). "
            f"Mean affairs: {with_children.mean():.2f} (with children) vs {without_children.mean():.2f} (without). "
            f"Cannot conclude that children decrease affairs."
        )

print(f"\n{'='*80}")
print(f"FINAL RESPONSE: {response}/100")
print(f"EXPLANATION: {explanation}")
print(f"{'='*80}")

# Write conclusion to file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\nConclusion written to conclusion.txt")
