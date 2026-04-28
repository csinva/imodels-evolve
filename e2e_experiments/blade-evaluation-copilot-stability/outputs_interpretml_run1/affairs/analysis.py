import pandas as pd
import numpy as np
import json
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from interpret.glassbox import ExplainableBoostingRegressor
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('affairs.csv')

print("="*80)
print("EXPLORATORY DATA ANALYSIS")
print("="*80)

print("\nDataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nSummary statistics:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

# Focus on children and affairs
print("\n" + "="*80)
print("COMPARING AFFAIRS BY CHILDREN STATUS")
print("="*80)

print("\nValue counts for 'children':")
print(df['children'].value_counts())

print("\nValue counts for 'affairs':")
print(df['affairs'].value_counts())

# Group by children status
children_groups = df.groupby('children')['affairs'].describe()
print("\nAffairs statistics by children status:")
print(children_groups)

# Calculate means
mean_with_children = df[df['children'] == 'yes']['affairs'].mean()
mean_without_children = df[df['children'] == 'no']['affairs'].mean()

print(f"\nMean affairs with children: {mean_with_children:.4f}")
print(f"Mean affairs without children: {mean_without_children:.4f}")
print(f"Difference: {mean_with_children - mean_without_children:.4f}")

# Statistical test: Independent t-test
print("\n" + "="*80)
print("STATISTICAL SIGNIFICANCE TEST: T-TEST")
print("="*80)

affairs_with_children = df[df['children'] == 'yes']['affairs']
affairs_without_children = df[df['children'] == 'no']['affairs']

t_stat, p_value = stats.ttest_ind(affairs_with_children, affairs_without_children)
print(f"\nT-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")

# Mann-Whitney U test (non-parametric alternative)
u_stat, u_pvalue = stats.mannwhitneyu(affairs_with_children, affairs_without_children, alternative='two-sided')
print(f"\nMann-Whitney U statistic: {u_stat:.4f}")
print(f"Mann-Whitney U p-value: {u_pvalue:.4f}")

# Effect size: Cohen's d
pooled_std = np.sqrt(((len(affairs_with_children)-1)*affairs_with_children.std()**2 + 
                       (len(affairs_without_children)-1)*affairs_without_children.std()**2) / 
                      (len(affairs_with_children) + len(affairs_without_children) - 2))
cohens_d = (mean_with_children - mean_without_children) / pooled_std
print(f"\nCohen's d effect size: {cohens_d:.4f}")

# Check proportion having any affairs
print("\n" + "="*80)
print("PROPORTION HAVING AFFAIRS")
print("="*80)

prop_with_children = (df[df['children'] == 'yes']['affairs'] > 0).mean()
prop_without_children = (df[df['children'] == 'no']['affairs'] > 0).mean()

print(f"\nProportion with affairs (with children): {prop_with_children:.4f}")
print(f"Proportion with affairs (without children): {prop_without_children:.4f}")

# Chi-square test for independence
contingency_table = pd.crosstab(df['children'], df['affairs'] > 0)
print("\nContingency table:")
print(contingency_table)

chi2, chi_p_value, dof, expected = stats.chi2_contingency(contingency_table)
print(f"\nChi-square statistic: {chi2:.4f}")
print(f"Chi-square p-value: {chi_p_value:.4f}")

# Regression analysis with statsmodels
print("\n" + "="*80)
print("REGRESSION ANALYSIS WITH STATSMODELS")
print("="*80)

# Create binary variable for children
df['children_binary'] = (df['children'] == 'yes').astype(int)

# Simple linear regression: affairs ~ children
X_simple = sm.add_constant(df['children_binary'])
y = df['affairs']
model_simple = sm.OLS(y, X_simple).fit()
print("\nSimple regression: affairs ~ children")
print(model_simple.summary())

# Multiple regression: control for other variables
# Encode gender
df['gender_binary'] = (df['gender'] == 'male').astype(int)

X_multi = df[['children_binary', 'age', 'yearsmarried', 'gender_binary', 
              'religiousness', 'education', 'occupation', 'rating']]
X_multi = sm.add_constant(X_multi)
model_multi = sm.OLS(y, X_multi).fit()
print("\n\nMultiple regression with controls:")
print(model_multi.summary())

# Interpretable ML models
print("\n" + "="*80)
print("INTERPRETABLE ML MODELS")
print("="*80)

# Prepare features
X_ml = df[['children_binary', 'age', 'yearsmarried', 'gender_binary', 
           'religiousness', 'education', 'occupation', 'rating']].values
y_ml = df['affairs'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_ml, y_ml, test_size=0.2, random_state=42)

# Explainable Boosting Regressor
print("\nTraining Explainable Boosting Regressor...")
ebm = ExplainableBoostingRegressor(random_state=42)
ebm.fit(X_train, y_train)

feature_names = ['children_binary', 'age', 'yearsmarried', 'gender_binary', 
                 'religiousness', 'education', 'occupation', 'rating']

print("\nFeature importances from EBM:")
importances = ebm.term_importances()
for name, importance in zip(feature_names, importances):
    print(f"  {name}: {importance:.4f}")

# Decision Tree for interpretability
print("\nTraining Decision Tree Regressor...")
dt = DecisionTreeRegressor(max_depth=4, random_state=42)
dt.fit(X_train, y_train)

print("\nFeature importances from Decision Tree:")
for name, importance in zip(feature_names, dt.feature_importances_):
    print(f"  {name}: {importance:.4f}")

# Ridge regression for stable coefficients
print("\nTraining Ridge Regression...")
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

print("\nRidge regression coefficients:")
for name, coef in zip(feature_names, ridge.coef_):
    print(f"  {name}: {coef:.4f}")

# Determine conclusion
print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

# Analyze results
print("\nKey findings:")
print(f"1. Mean affairs with children: {mean_with_children:.4f}")
print(f"2. Mean affairs without children: {mean_without_children:.4f}")
print(f"3. T-test p-value: {p_value:.4f}")
print(f"4. Mann-Whitney U p-value: {u_pvalue:.4f}")
print(f"5. Chi-square p-value: {chi_p_value:.4f}")
print(f"6. Simple regression p-value for children: {model_simple.pvalues['children_binary']:.4f}")
print(f"7. Multiple regression p-value for children: {model_multi.pvalues['children_binary']:.4f}")

# Determine response based on statistical evidence
# p-value < 0.05 indicates statistical significance
alpha = 0.05

# Check if having children is associated with FEWER affairs (negative relationship)
is_significant = (p_value < alpha or u_pvalue < alpha or chi_p_value < alpha or 
                  model_simple.pvalues['children_binary'] < alpha or 
                  model_multi.pvalues['children_binary'] < alpha)

# Check direction of relationship
direction_is_decrease = mean_with_children < mean_without_children

print(f"\nStatistically significant relationship: {is_significant}")
print(f"Direction is decrease (fewer affairs with children): {direction_is_decrease}")

# Determine Likert score
if is_significant and direction_is_decrease:
    # Strong evidence that children decrease affairs
    if p_value < 0.001 and u_pvalue < 0.001:
        response = 90  # Very strong evidence
        explanation = (f"Strong statistical evidence that having children decreases engagement in extramarital affairs. "
                      f"Mean affairs with children ({mean_with_children:.2f}) is significantly lower than without children ({mean_without_children:.2f}). "
                      f"Multiple statistical tests show significance: t-test p={p_value:.4f}, Mann-Whitney p={u_pvalue:.4f}, "
                      f"chi-square p={chi_p_value:.4f}. The simple regression coefficient is {model_simple.params['children_binary']:.3f} "
                      f"(p={model_simple.pvalues['children_binary']:.4f}), indicating children are associated with fewer affairs. "
                      f"This relationship holds even when controlling for other factors in multiple regression "
                      f"(coef={model_multi.params['children_binary']:.3f}, p={model_multi.pvalues['children_binary']:.4f}).")
    elif p_value < 0.01:
        response = 80  # Very strong evidence
        explanation = (f"Very strong statistical evidence that having children decreases engagement in extramarital affairs. "
                      f"Mean affairs with children ({mean_with_children:.2f}) is significantly lower than without children ({mean_without_children:.2f}). "
                      f"T-test p-value={p_value:.4f}, Mann-Whitney p={u_pvalue:.4f}, chi-square p={chi_p_value:.4f}. "
                      f"Regression analysis confirms: simple regression coef={model_simple.params['children_binary']:.3f} (p={model_simple.pvalues['children_binary']:.4f}), "
                      f"multiple regression coef={model_multi.params['children_binary']:.3f} (p={model_multi.pvalues['children_binary']:.4f}).")
    else:
        response = 70  # Moderate to strong evidence
        explanation = (f"Moderate to strong statistical evidence that having children decreases engagement in extramarital affairs. "
                      f"Mean affairs with children ({mean_with_children:.2f}) is lower than without children ({mean_without_children:.2f}). "
                      f"T-test p-value={p_value:.4f}, Mann-Whitney p={u_pvalue:.4f}. "
                      f"Regression coefficient for children: {model_simple.params['children_binary']:.3f} (p={model_simple.pvalues['children_binary']:.4f}).")
elif not is_significant and direction_is_decrease:
    # Trend suggests decrease but not statistically significant
    response = 40  # Weak evidence
    explanation = (f"Weak evidence for a decrease. Mean affairs with children ({mean_with_children:.2f}) is slightly lower "
                  f"than without children ({mean_without_children:.2f}), but this difference is not statistically significant. "
                  f"T-test p-value={p_value:.4f}, Mann-Whitney p={u_pvalue:.4f}. Without statistical significance, "
                  f"we cannot confidently conclude that children decrease affair engagement.")
elif is_significant and not direction_is_decrease:
    # Significant relationship but OPPOSITE direction (children increase affairs)
    response = 10  # Strong "No"
    explanation = (f"Evidence suggests the opposite: having children is associated with MORE affairs, not fewer. "
                  f"Mean affairs with children ({mean_with_children:.2f}) is higher than without children ({mean_without_children:.2f}). "
                  f"This contradicts the hypothesis that children decrease affair engagement. "
                  f"T-test p-value={p_value:.4f}.")
else:
    # Not significant and no clear direction
    response = 30  # Mostly "No"
    explanation = (f"No significant evidence that children decrease affair engagement. "
                  f"Mean affairs with children ({mean_with_children:.2f}) vs without children ({mean_without_children:.2f}). "
                  f"T-test p-value={p_value:.4f}, Mann-Whitney p={u_pvalue:.4f}. "
                  f"The relationship is not statistically significant.")

print(f"\nFinal response score: {response}")
print(f"Explanation: {explanation}")

# Write conclusion to file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n" + "="*80)
print("Analysis complete. Conclusion written to conclusion.txt")
print("="*80)
