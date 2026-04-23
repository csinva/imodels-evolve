import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from imodels import RuleFitRegressor, FIGSRegressor
import json

# Load the dataset
df = pd.read_csv('mortgage.csv')

print("=" * 80)
print("DATASET OVERVIEW")
print("=" * 80)
print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nData types:")
print(df.dtypes)
print(f"\nMissing values:")
print(df.isnull().sum())

# Summary statistics
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)
print(df.describe())

# Focus on the research question: Does gender affect mortgage approval?
print("\n" + "=" * 80)
print("GENDER VS MORTGAGE APPROVAL - UNIVARIATE ANALYSIS")
print("=" * 80)

# Create cross-tabulation
gender_approval = pd.crosstab(df['female'], df['accept'], normalize='index')
print("\nApproval rates by gender (proportion):")
print(gender_approval)

# Raw counts
gender_approval_counts = pd.crosstab(df['female'], df['accept'])
print("\nApproval counts by gender:")
print(gender_approval_counts)

# Calculate approval rates
male_approval = df[df['female'] == 0]['accept'].mean()
female_approval = df[df['female'] == 1]['accept'].mean()
print(f"\nMale approval rate: {male_approval:.4f} ({male_approval*100:.2f}%)")
print(f"Female approval rate: {female_approval:.4f} ({female_approval*100:.2f}%)")
print(f"Difference: {(female_approval - male_approval)*100:.2f} percentage points")

# Chi-square test for independence
chi2, p_value_chi2, dof, expected = stats.chi2_contingency(gender_approval_counts)
print(f"\nChi-square test:")
print(f"  Chi-square statistic: {chi2:.4f}")
print(f"  P-value: {p_value_chi2:.6f}")
print(f"  Significant at α=0.05? {p_value_chi2 < 0.05}")

# Two-proportion z-test
male_approvals = df[df['female'] == 0]['accept'].sum()
male_total = (df['female'] == 0).sum()
female_approvals = df[df['female'] == 1]['accept'].sum()
female_total = (df['female'] == 1).sum()

# Calculate pooled proportion
p_pooled = (male_approvals + female_approvals) / (male_total + female_total)
se = np.sqrt(p_pooled * (1 - p_pooled) * (1/male_total + 1/female_total))
z_stat = (female_approval - male_approval) / se
p_value_z = 2 * (1 - stats.norm.cdf(abs(z_stat)))

print(f"\nTwo-proportion z-test:")
print(f"  Z-statistic: {z_stat:.4f}")
print(f"  P-value: {p_value_z:.6f}")
print(f"  Significant at α=0.05? {p_value_z < 0.05}")

# Multivariate analysis - control for confounders
print("\n" + "=" * 80)
print("MULTIVARIATE ANALYSIS - CONTROLLING FOR CONFOUNDERS")
print("=" * 80)

# Prepare data for modeling
# Features that might confound the relationship
feature_cols = ['female', 'black', 'housing_expense_ratio', 'self_employed', 
                'married', 'mortgage_credit', 'consumer_credit', 'bad_history',
                'PI_ratio', 'loan_to_value', 'denied_PMI']

X = df[feature_cols].copy()
y = df['accept'].copy()

# Check for missing values in features
print(f"\nMissing values in features:")
print(X.isnull().sum())

# Remove any rows with missing values
mask = ~(X.isnull().any(axis=1) | y.isnull())
X_clean = X[mask]
y_clean = y[mask]
print(f"\nCleaned dataset shape: {X_clean.shape}")

# 1. Logistic Regression with statsmodels for p-values
print("\n" + "-" * 80)
print("1. LOGISTIC REGRESSION (statsmodels)")
print("-" * 80)

X_sm = sm.add_constant(X_clean)
logit_model = sm.Logit(y_clean, X_sm)
logit_result = logit_model.fit(disp=0)

print("\nLogistic Regression Summary:")
print(logit_result.summary2())

female_coef = logit_result.params['female']
female_pval = logit_result.pvalues['female']
female_or = np.exp(female_coef)

print(f"\n*** KEY RESULT: Female coefficient ***")
print(f"  Coefficient: {female_coef:.4f}")
print(f"  P-value: {female_pval:.6f}")
print(f"  Odds Ratio: {female_or:.4f}")
print(f"  Significant at α=0.05? {female_pval < 0.05}")
print(f"  Interpretation: Being female is associated with {(female_or-1)*100:.2f}% change in odds of approval")

# 2. Interpretable tree-based model
print("\n" + "-" * 80)
print("2. DECISION TREE CLASSIFIER (Interpretable)")
print("-" * 80)

tree_model = DecisionTreeClassifier(max_depth=5, min_samples_split=100, random_state=42)
tree_model.fit(X_clean, y_clean)

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': tree_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importances from Decision Tree:")
print(feature_importance)

female_importance = feature_importance[feature_importance['feature'] == 'female']['importance'].values[0]
print(f"\n*** Female feature importance: {female_importance:.4f} ***")
print(f"Rank: {feature_importance[feature_importance['feature'] == 'female'].index[0] + 1} out of {len(feature_cols)}")

# 3. FIGS - Fast Interpretable Greedy-tree Sums
print("\n" + "-" * 80)
print("3. FIGS CLASSIFIER (imodels)")
print("-" * 80)

try:
    figs_model = FIGSRegressor(max_rules=10)
    figs_model.fit(X_clean, y_clean)
    
    print("\nFIGS Feature Importances:")
    if hasattr(figs_model, 'feature_importances_'):
        figs_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': figs_model.feature_importances_
        }).sort_values('importance', ascending=False)
        print(figs_importance)
        
        female_figs_importance = figs_importance[figs_importance['feature'] == 'female']['importance'].values[0]
        print(f"\n*** Female feature importance (FIGS): {female_figs_importance:.4f} ***")
except Exception as e:
    print(f"FIGS model error: {e}")

# 4. Additional analysis - stratified by confounders
print("\n" + "=" * 80)
print("STRATIFIED ANALYSIS")
print("=" * 80)

# Check if gender effect varies by other important factors
print("\nApproval rates by gender and race:")
stratified = df.groupby(['female', 'black'])['accept'].agg(['mean', 'count'])
print(stratified)

print("\nApproval rates by gender and marital status:")
stratified_married = df.groupby(['female', 'married'])['accept'].agg(['mean', 'count'])
print(stratified_married)

print("\nApproval rates by gender and bad credit history:")
stratified_credit = df.groupby(['female', 'bad_history'])['accept'].agg(['mean', 'count'])
print(stratified_credit)

# Final conclusion
print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

# Synthesize all evidence
evidence_summary = f"""
UNIVARIATE ANALYSIS:
- Male approval rate: {male_approval*100:.2f}%
- Female approval rate: {female_approval*100:.2f}%
- Difference: {(female_approval - male_approval)*100:.2f} percentage points
- Chi-square test p-value: {p_value_chi2:.6f} (significant: {p_value_chi2 < 0.05})
- Two-proportion z-test p-value: {p_value_z:.6f} (significant: {p_value_z < 0.05})

MULTIVARIATE ANALYSIS (controlling for confounders):
- Logistic regression coefficient for female: {female_coef:.4f}
- Logistic regression p-value: {female_pval:.6f} (significant: {female_pval < 0.05})
- Odds ratio: {female_or:.4f} ({(female_or-1)*100:.2f}% change in odds)
- Decision tree feature importance rank: {feature_importance[feature_importance['feature'] == 'female'].index[0] + 1} out of {len(feature_cols)}
"""

print(evidence_summary)

# Determine response and explanation
if p_value_chi2 < 0.05 and female_pval < 0.05:
    # Both univariate and multivariate tests are significant
    response = 85
    explanation = f"Yes, gender significantly affects mortgage approval. In univariate analysis, female applicants have a {(female_approval - male_approval)*100:.2f} percentage point {'lower' if female_approval < male_approval else 'higher'} approval rate (p={p_value_chi2:.4f}). After controlling for creditworthiness, income, and other factors via logistic regression, the gender effect remains statistically significant (coef={female_coef:.4f}, p={female_pval:.4f}, OR={female_or:.4f}), indicating gender has an independent effect on approval decisions."
elif p_value_chi2 < 0.05 and female_pval >= 0.05:
    # Univariate is significant but multivariate is not
    response = 40
    explanation = f"Gender shows a raw association with mortgage approval (p={p_value_chi2:.4f}), but this effect is not statistically significant when controlling for creditworthiness, financial ratios, and other confounders (multivariate p={female_pval:.4f}). This suggests the gender difference in approval rates is largely explained by other factors rather than gender itself having an independent effect."
elif p_value_chi2 >= 0.05:
    # Not even significant in univariate analysis
    response = 15
    explanation = f"No, gender does not significantly affect mortgage approval. The univariate chi-square test shows no significant relationship (p={p_value_chi2:.4f}). The approval rate difference between males and females is {abs((female_approval - male_approval)*100):.2f} percentage points, which is not statistically significant at the 0.05 level."
else:
    # Edge case - multivariate significant but not univariate (unusual)
    response = 50
    explanation = f"Mixed evidence. Univariate tests do not show a significant effect (p={p_value_chi2:.4f}), but multivariate analysis suggests gender may play a role when controlling for other factors (p={female_pval:.4f})."

# Write conclusion to file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n" + "=" * 80)
print("FINAL ANSWER")
print("=" * 80)
print(f"Response: {response}")
print(f"Explanation: {explanation}")
print("\nConclusion written to conclusion.txt")
