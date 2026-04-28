import pandas as pd
import numpy as np
import json
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score
import statsmodels.api as sm
from interpret.glassbox import ExplainableBoostingRegressor

# Load the data
df = pd.read_csv('panda_nuts.csv')

print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nData summary:")
print(df.describe())
print("\nData info:")
print(df.info())

# Create efficiency metric: nuts opened per second
df['efficiency'] = df['nuts_opened'] / df['seconds']

print("\nEfficiency statistics:")
print(df['efficiency'].describe())

# Explore correlations with efficiency
print("\n=== EXPLORATORY ANALYSIS ===")
print("\nCorrelation with efficiency:")
print("Age vs Efficiency:", stats.pearsonr(df['age'], df['efficiency']))
print("\nEfficiency by sex:")
print(df.groupby('sex')['efficiency'].describe())
print("\nEfficiency by help status:")
print(df.groupby('help')['efficiency'].describe())

# Statistical tests
print("\n=== STATISTICAL TESTS ===")

# 1. Age effect - Pearson correlation
age_corr, age_pval = stats.pearsonr(df['age'], df['efficiency'])
print(f"\n1. Age effect:")
print(f"   Correlation: {age_corr:.4f}, p-value: {age_pval:.4f}")
print(f"   Significant: {age_pval < 0.05}")

# 2. Sex effect - t-test
male_eff = df[df['sex'] == 'm']['efficiency']
female_eff = df[df['sex'] == 'f']['efficiency']
sex_tstat, sex_pval = stats.ttest_ind(male_eff, female_eff)
print(f"\n2. Sex effect:")
print(f"   Male mean: {male_eff.mean():.4f}, Female mean: {female_eff.mean():.4f}")
print(f"   t-statistic: {sex_tstat:.4f}, p-value: {sex_pval:.4f}")
print(f"   Significant: {sex_pval < 0.05}")

# 3. Help effect - t-test
help_yes = df[df['help'] == 'y']['efficiency']
help_no = df[df['help'] == 'N']['efficiency']
help_tstat, help_pval = stats.ttest_ind(help_yes, help_no)
print(f"\n3. Help effect:")
print(f"   With help mean: {help_yes.mean():.4f}, Without help mean: {help_no.mean():.4f}")
print(f"   t-statistic: {help_tstat:.4f}, p-value: {help_pval:.4f}")
print(f"   Significant: {help_pval < 0.05}")

# Prepare data for regression
print("\n=== INTERPRETABLE MODELING ===")

# Encode categorical variables
le_sex = LabelEncoder()
le_help = LabelEncoder()
df['sex_encoded'] = le_sex.fit_transform(df['sex'])
df['help_encoded'] = le_help.fit_transform(df['help'])

# Features for modeling
X = df[['age', 'sex_encoded', 'help_encoded']]
y = df['efficiency']

# 1. Linear Regression with statsmodels for p-values
X_with_const = sm.add_constant(X)
model_ols = sm.OLS(y, X_with_const).fit()
print("\n1. OLS Regression Summary:")
print(model_ols.summary())

# 2. Explainable Boosting Regressor from interpret
print("\n2. Explainable Boosting Regressor:")
ebm = ExplainableBoostingRegressor(random_state=42)
ebm.fit(X, y)

# Get feature importances
feature_names = ['age', 'sex', 'help']
print("\nFeature Importances (absolute scale):")
for name, importance in zip(feature_names, ebm.term_importances()):
    print(f"   {name}: {importance:.4f}")

# Cross-validation score
cv_scores = cross_val_score(ebm, X, y, cv=5, scoring='r2')
print(f"\nCross-validation R² scores: {cv_scores}")
print(f"Mean CV R²: {cv_scores.mean():.4f}")

# 3. Simple Linear Regression for coefficient interpretation
lr = LinearRegression()
lr.fit(X, y)
print("\n3. Linear Regression Coefficients:")
for name, coef in zip(feature_names, lr.coef_):
    print(f"   {name}: {coef:.4f}")
print(f"   Intercept: {lr.intercept_:.4f}")
print(f"   R² score: {lr.score(X, y):.4f}")

# ANOVA test for combined effect
print("\n=== ANOVA TEST ===")
# Create groups based on all three factors
df['group'] = df['age'].astype(str) + '_' + df['sex'] + '_' + df['help']
groups = [group['efficiency'].values for name, group in df.groupby('group') if len(group) >= 2]
if len(groups) >= 2:
    f_stat, anova_pval = stats.f_oneway(*groups)
    print(f"ANOVA F-statistic: {f_stat:.4f}, p-value: {anova_pval:.4f}")
    print(f"Overall significant effect: {anova_pval < 0.05}")

# Determine conclusion
print("\n=== CONCLUSION ===")

# Collect p-values
significant_factors = []
if age_pval < 0.05:
    significant_factors.append(f"age (p={age_pval:.4f}, r={age_corr:.3f})")
if sex_pval < 0.05:
    significant_factors.append(f"sex (p={sex_pval:.4f})")
if help_pval < 0.05:
    significant_factors.append(f"help (p={help_pval:.4f})")

# Calculate response score
num_significant = len(significant_factors)
total_factors = 3

# OLS p-values for confirmation
ols_pvals = model_ols.pvalues[1:]  # Exclude constant
ols_significant = sum(ols_pvals < 0.05)

# Weighted scoring based on:
# - Number of significant factors
# - Strength of relationships (effect sizes)
# - Model fit (R²)

if num_significant == 0:
    response = 10  # Very weak evidence
    explanation = "None of the three factors (age, sex, help) showed statistically significant relationships with nut-cracking efficiency in univariate tests."
elif num_significant == 1:
    response = 40  # Some evidence but limited
    explanation = f"Only one factor showed a significant relationship: {significant_factors[0]}. The other two factors did not significantly influence nut-cracking efficiency."
elif num_significant == 2:
    response = 70  # Strong evidence
    explanation = f"Two of the three factors showed significant relationships: {', '.join(significant_factors)}. This indicates that multiple factors influence nut-cracking efficiency."
else:  # All 3 significant
    response = 90  # Very strong evidence
    explanation = f"All three factors showed significant relationships with nut-cracking efficiency: {', '.join(significant_factors)}. The multivariate OLS model confirms these relationships."

# Adjust based on effect sizes and model fit
model_r2 = lr.score(X, y)
if model_r2 > 0.3:
    response = min(100, response + 5)
elif model_r2 < 0.1:
    response = max(0, response - 10)

# Add context about relationships
if num_significant > 0:
    explanation += f" The linear regression model explains {model_r2*100:.1f}% of variance in efficiency. "
    if age_pval < 0.05:
        explanation += f"Age shows a {'positive' if age_corr > 0 else 'negative'} correlation. "
    if sex_pval < 0.05:
        explanation += f"Sex differences are present. "
    if help_pval < 0.05:
        explanation += f"Receiving help {'increases' if help_tstat > 0 else 'decreases'} efficiency. "

print(f"\nResponse score: {response}")
print(f"Explanation: {explanation}")

# Write conclusion to file
conclusion = {
    "response": int(response),
    "explanation": explanation.strip()
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\nConclusion written to conclusion.txt")
