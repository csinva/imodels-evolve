import pandas as pd
import numpy as np
import json
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('affairs.csv')

print("="*80)
print("RESEARCH QUESTION: Does having children decrease engagement in extramarital affairs?")
print("="*80)
print()

# Basic exploration
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nSummary statistics:")
print(df.describe())
print()

# Focus on the key variables: children (yes/no) and affairs (frequency)
print("="*80)
print("KEY VARIABLE ANALYSIS")
print("="*80)
print("\nAffairs distribution:")
print(df['affairs'].value_counts().sort_index())
print("\nChildren distribution:")
print(df['children'].value_counts())
print()

# Create binary indicator for children
df['has_children'] = (df['children'] == 'yes').astype(int)

# Compare affairs by children status
print("="*80)
print("DESCRIPTIVE STATISTICS: Affairs by Children Status")
print("="*80)
affairs_by_children = df.groupby('children')['affairs'].agg(['count', 'mean', 'median', 'std'])
print(affairs_by_children)
print()

# Percentage with any affair
print("Percentage with any affair (affairs > 0):")
df['any_affair'] = (df['affairs'] > 0).astype(int)
affair_rate = df.groupby('children')['any_affair'].agg(['mean', 'sum', 'count'])
print(affair_rate)
print()

# Statistical test 1: T-test comparing mean affairs
no_children_affairs = df[df['children'] == 'no']['affairs']
yes_children_affairs = df[df['children'] == 'yes']['affairs']

print("="*80)
print("STATISTICAL TEST 1: Independent Samples T-Test")
print("="*80)
print(f"Mean affairs (no children): {no_children_affairs.mean():.4f}")
print(f"Mean affairs (yes children): {yes_children_affairs.mean():.4f}")
print(f"Difference: {no_children_affairs.mean() - yes_children_affairs.mean():.4f}")

t_stat, p_value_ttest = stats.ttest_ind(no_children_affairs, yes_children_affairs)
print(f"\nT-statistic: {t_stat:.4f}")
print(f"P-value: {p_value_ttest:.6f}")
print(f"Significant at α=0.05: {p_value_ttest < 0.05}")
print()

# Statistical test 2: Mann-Whitney U test (non-parametric)
print("="*80)
print("STATISTICAL TEST 2: Mann-Whitney U Test (Non-parametric)")
print("="*80)
u_stat, p_value_mann = stats.mannwhitneyu(no_children_affairs, yes_children_affairs, alternative='two-sided')
print(f"U-statistic: {u_stat:.4f}")
print(f"P-value: {p_value_mann:.6f}")
print(f"Significant at α=0.05: {p_value_mann < 0.05}")
print()

# Statistical test 3: Chi-square test for any affair vs children
print("="*80)
print("STATISTICAL TEST 3: Chi-Square Test (Any Affair vs Children)")
print("="*80)
contingency_table = pd.crosstab(df['has_children'], df['any_affair'])
print("Contingency table:")
print(contingency_table)
chi2, p_value_chi, dof, expected = stats.chi2_contingency(contingency_table)
print(f"\nChi-square statistic: {chi2:.4f}")
print(f"P-value: {p_value_chi:.6f}")
print(f"Significant at α=0.05: {p_value_chi < 0.05}")
print()

# Regression analysis with statsmodels for p-values
print("="*80)
print("REGRESSION ANALYSIS 1: Simple Linear Regression (Affairs ~ Children)")
print("="*80)
X_simple = df[['has_children']]
y = df['affairs']
X_simple = sm.add_constant(X_simple)
model_simple = sm.OLS(y, X_simple).fit()
print(model_simple.summary())
print()

# Multiple regression controlling for confounders
print("="*80)
print("REGRESSION ANALYSIS 2: Multiple Regression (Controlling for Confounders)")
print("="*80)
# Encode gender
df['gender_male'] = (df['gender'] == 'male').astype(int)

# Build full model
X_full = df[['has_children', 'age', 'yearsmarried', 'gender_male', 
             'religiousness', 'education', 'occupation', 'rating']]
X_full = sm.add_constant(X_full)
model_full = sm.OLS(y, X_full).fit()
print(model_full.summary())
print()

# Interpretable model using sklearn
print("="*80)
print("INTERPRETABLE MODEL: Decision Tree")
print("="*80)
X_tree = df[['has_children', 'age', 'yearsmarried', 'gender_male', 
             'religiousness', 'education', 'occupation', 'rating']]
tree = DecisionTreeRegressor(max_depth=4, min_samples_leaf=20, random_state=42)
tree.fit(X_tree, y)
feature_importance = pd.DataFrame({
    'feature': X_tree.columns,
    'importance': tree.feature_importances_
}).sort_values('importance', ascending=False)
print("Feature importances:")
print(feature_importance)
print()

# Effect size calculation (Cohen's d)
print("="*80)
print("EFFECT SIZE: Cohen's d")
print("="*80)
pooled_std = np.sqrt(((len(no_children_affairs)-1)*no_children_affairs.std()**2 + 
                       (len(yes_children_affairs)-1)*yes_children_affairs.std()**2) / 
                      (len(no_children_affairs) + len(yes_children_affairs) - 2))
cohens_d = (no_children_affairs.mean() - yes_children_affairs.mean()) / pooled_std
print(f"Cohen's d: {cohens_d:.4f}")
print(f"Effect size interpretation: ", end="")
if abs(cohens_d) < 0.2:
    print("negligible")
elif abs(cohens_d) < 0.5:
    print("small")
elif abs(cohens_d) < 0.8:
    print("medium")
else:
    print("large")
print()

# Correlation analysis
print("="*80)
print("CORRELATION ANALYSIS")
print("="*80)
corr, p_val_corr = stats.pearsonr(df['has_children'], df['affairs'])
print(f"Pearson correlation (children vs affairs): {corr:.4f}")
print(f"P-value: {p_val_corr:.6f}")
print()

# Final interpretation and scoring
print("="*80)
print("FINAL INTERPRETATION")
print("="*80)

# Determine the response score based on evidence
significant_tests = 0
if p_value_ttest < 0.05:
    significant_tests += 1
    print(f"✓ T-test shows significant difference (p={p_value_ttest:.6f})")
else:
    print(f"✗ T-test does NOT show significant difference (p={p_value_ttest:.6f})")

if p_value_mann < 0.05:
    significant_tests += 1
    print(f"✓ Mann-Whitney U test shows significant difference (p={p_value_mann:.6f})")
else:
    print(f"✗ Mann-Whitney U test does NOT show significant difference (p={p_value_mann:.6f})")

if p_value_chi < 0.05:
    significant_tests += 1
    print(f"✓ Chi-square test shows significant association (p={p_value_chi:.6f})")
else:
    print(f"✗ Chi-square test does NOT show significant association (p={p_value_chi:.6f})")

# Check coefficient from regression
children_coef = model_simple.params['has_children']
children_pval = model_simple.pvalues['has_children']
children_coef_full = model_full.params['has_children']
children_pval_full = model_full.pvalues['has_children']

print(f"\nSimple regression coefficient for children: {children_coef:.4f} (p={children_pval:.6f})")
print(f"Multiple regression coefficient for children: {children_coef_full:.4f} (p={children_pval_full:.6f})")

print(f"\nMean difference: {no_children_affairs.mean() - yes_children_affairs.mean():.4f}")
print(f"Direction: {'Children associated with FEWER affairs' if children_coef < 0 else 'Children associated with MORE affairs'}")

# Determine Likert score
# 0 = strong "No", 100 = strong "Yes" (Yes = children decrease affairs)
# Since we're testing if children DECREASE affairs, a negative coefficient is what we want

if significant_tests == 0:
    # No significant evidence
    response = 10
    explanation = "No statistically significant relationship found between having children and extramarital affairs. T-test (p={:.3f}), Mann-Whitney (p={:.3f}), and Chi-square (p={:.3f}) all non-significant. Mean difference is only {:.2f}. The evidence does not support that having children decreases affairs.".format(
        p_value_ttest, p_value_mann, p_value_chi, no_children_affairs.mean() - yes_children_affairs.mean())
elif significant_tests >= 1 and children_coef < 0:
    # Some/all tests significant AND children associated with fewer affairs
    if significant_tests >= 2 and abs(cohens_d) > 0.2:
        # Strong evidence
        response = 75
        explanation = "Strong statistical evidence that having children is associated with FEWER extramarital affairs. {}/3 tests significant. Simple regression shows coefficient={:.3f} (p={:.4f}), indicating people with children have {:.2f} fewer affairs on average. Cohen's d={:.3f} indicates a {} effect size. This relationship holds even when controlling for confounders in multiple regression (coef={:.3f}, p={:.4f}).".format(
            significant_tests, children_coef, children_pval, 
            abs(children_coef), cohens_d, 
            "small" if abs(cohens_d) < 0.5 else "medium" if abs(cohens_d) < 0.8 else "large",
            children_coef_full, children_pval_full)
    else:
        # Moderate evidence
        response = 60
        explanation = "Moderate statistical evidence that having children is associated with fewer extramarital affairs. {}/3 tests significant. The relationship shows children associated with {:.2f} fewer affairs (p={:.4f}), though the effect size is relatively small (Cohen's d={:.3f}).".format(
            significant_tests, abs(children_coef), children_pval, cohens_d)
elif significant_tests >= 1 and children_coef > 0:
    # Significant but opposite direction (children increase affairs)
    response = 5
    explanation = "Statistical tests show a significant relationship, but in the OPPOSITE direction: having children is associated with MORE affairs, not fewer. Coefficient={:.3f} (p={:.4f}). This contradicts the hypothesis that children decrease affairs.".format(
        children_coef, children_pval)
else:
    # Edge case
    response = 15
    explanation = "Limited or mixed evidence. While {}/3 tests show significance, the effect size is small (Cohen's d={:.3f}) and the practical significance is questionable.".format(
        significant_tests, cohens_d)

print(f"\n{'='*80}")
print(f"CONCLUSION")
print(f"{'='*80}")
print(f"Response Score (0-100): {response}")
print(f"Explanation: {explanation}")
print()

# Write conclusion to file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("✓ Conclusion written to conclusion.txt")
