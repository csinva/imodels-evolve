import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from imodels import FIGSClassifier, HSTreeClassifier
import json
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('mortgage.csv')

# Drop rows with missing values in key columns
df = df.dropna()

# Explore the data
print("=" * 80)
print("DATASET OVERVIEW")
print("=" * 80)
print(f"Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
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

# Focus on the research question: How does gender affect mortgage approval?
print("\n" + "=" * 80)
print("GENDER AND MORTGAGE APPROVAL ANALYSIS")
print("=" * 80)

# Check distribution of gender
print(f"\nGender distribution:")
print(df['female'].value_counts())
print(f"Female proportion: {df['female'].mean():.2%}")

# Check approval rates by gender
print(f"\nApproval rates by gender:")
approval_by_gender = df.groupby('female')['accept'].agg(['mean', 'count'])
approval_by_gender.index = ['Male', 'Female']
print(approval_by_gender)

male_approval = df[df['female'] == 0]['accept'].mean()
female_approval = df[df['female'] == 1]['accept'].mean()
print(f"\nMale approval rate: {male_approval:.2%}")
print(f"Female approval rate: {female_approval:.2%}")
print(f"Difference: {(female_approval - male_approval):.2%}")

# Statistical test: Chi-square test for independence
print("\n" + "=" * 80)
print("CHI-SQUARE TEST: Gender vs Approval")
print("=" * 80)
contingency_table = pd.crosstab(df['female'], df['accept'])
chi2, p_value_chi, dof, expected = stats.chi2_contingency(contingency_table)
print(f"Contingency table:")
print(contingency_table)
print(f"\nChi-square statistic: {chi2:.4f}")
print(f"P-value: {p_value_chi:.4f}")
print(f"Degrees of freedom: {dof}")

# Two-sample t-test (alternative approach)
print("\n" + "=" * 80)
print("TWO-SAMPLE T-TEST: Approval rates by gender")
print("=" * 80)
male_approvals = df[df['female'] == 0]['accept']
female_approvals = df[df['female'] == 1]['accept']
t_stat, p_value_t = stats.ttest_ind(female_approvals, male_approvals)
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value_t:.4f}")

# Logistic regression with statsmodels for p-values
print("\n" + "=" * 80)
print("LOGISTIC REGRESSION (Univariate): Gender predicting approval")
print("=" * 80)
X_simple = sm.add_constant(df['female'])
y = df['accept']
logit_model = sm.Logit(y, X_simple).fit(disp=0)
print(logit_model.summary2())

# Controlled analysis: Account for confounding variables
print("\n" + "=" * 80)
print("MULTIVARIATE LOGISTIC REGRESSION: Controlling for other factors")
print("=" * 80)
# Include all relevant features
features = ['female', 'black', 'housing_expense_ratio', 'self_employed', 
            'married', 'mortgage_credit', 'consumer_credit', 'bad_history', 
            'PI_ratio', 'loan_to_value', 'denied_PMI']

X_multi = sm.add_constant(df[features])
logit_multi = sm.Logit(y, X_multi).fit(disp=0)
print(logit_multi.summary2())

# Extract gender coefficient and p-value from multivariate model
gender_coef = logit_multi.params['female']
gender_pval = logit_multi.pvalues['female']
gender_odds_ratio = np.exp(gender_coef)
print(f"\n*** GENDER EFFECT (controlling for other factors) ***")
print(f"Coefficient: {gender_coef:.4f}")
print(f"Odds ratio: {gender_odds_ratio:.4f}")
print(f"P-value: {gender_pval:.4f}")
print(f"95% CI: {logit_multi.conf_int().loc['female'].values}")

# Interpretable model: Decision Tree
print("\n" + "=" * 80)
print("INTERPRETABLE MODEL: Decision Tree")
print("=" * 80)
X_tree = df[features]
y_tree = df['accept']
tree_model = DecisionTreeClassifier(max_depth=5, random_state=42)
tree_model.fit(X_tree, y_tree)

feature_importance = pd.DataFrame({
    'feature': features,
    'importance': tree_model.feature_importances_
}).sort_values('importance', ascending=False)
print("Feature importances:")
print(feature_importance)
print(f"\nGender rank: {feature_importance[feature_importance['feature'] == 'female'].index[0] + 1} out of {len(features)}")

# Use imodels for interpretable tree
print("\n" + "=" * 80)
print("IMODELS: FIGS Classifier (Interpretable Tree)")
print("=" * 80)
figs_model = FIGSClassifier(max_rules=10)
figs_model.fit(X_tree, y_tree)
print(f"FIGS feature importances:")
figs_importance = pd.DataFrame({
    'feature': features,
    'importance': figs_model.feature_importances_
}).sort_values('importance', ascending=False)
print(figs_importance)

# Compute effect size (Cohen's h for proportions)
print("\n" + "=" * 80)
print("EFFECT SIZE")
print("=" * 80)
# Cohen's h for difference in proportions
h = 2 * (np.arcsin(np.sqrt(female_approval)) - np.arcsin(np.sqrt(male_approval)))
print(f"Cohen's h (effect size): {h:.4f}")
print(f"Interpretation: ", end="")
if abs(h) < 0.2:
    print("small effect")
elif abs(h) < 0.5:
    print("medium effect")
else:
    print("large effect")

# Correlation analysis
print("\n" + "=" * 80)
print("CORRELATION: Gender with Approval")
print("=" * 80)
corr, corr_pval = stats.pearsonr(df['female'], df['accept'])
print(f"Pearson correlation: {corr:.4f}")
print(f"P-value: {corr_pval:.4f}")

# Compare with other demographic variables
print("\n" + "=" * 80)
print("COMPARISON: Gender effect vs Race effect")
print("=" * 80)
black_approval = df[df['black'] == 1]['accept'].mean()
non_black_approval = df[df['black'] == 0]['accept'].mean()
print(f"Black approval rate: {black_approval:.2%}")
print(f"Non-Black approval rate: {non_black_approval:.2%}")
print(f"Difference: {(black_approval - non_black_approval):.2%}")
chi2_black, p_value_black, _, _ = stats.chi2_contingency(pd.crosstab(df['black'], df['accept']))
print(f"Chi-square p-value for race: {p_value_black:.4f}")

# CONCLUSION
print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

# Determine response based on statistical significance and effect size
# Key findings:
# 1. P-value from chi-square test
# 2. P-value from multivariate logistic regression (controlling for confounders)
# 3. Effect size
# 4. Practical significance

if gender_pval < 0.05:
    if abs(gender_coef) > 0.1:  # Meaningful effect
        response = 75
        explanation = f"Yes, gender has a statistically significant effect on mortgage approval (p={gender_pval:.4f} in multivariate regression controlling for creditworthiness, employment, and other factors). Female applicants have an odds ratio of {gender_odds_ratio:.2f} compared to males. The univariate chi-square test also shows significance (p={p_value_chi:.4f}). While the effect size is modest (Cohen's h={h:.3f}), the relationship is robust when controlling for confounding variables."
    else:
        response = 60
        explanation = f"Gender shows a statistically significant but small effect on mortgage approval (p={gender_pval:.4f}). The effect persists when controlling for other factors, with an odds ratio of {gender_odds_ratio:.2f}, but the practical significance is limited given the small coefficient ({gender_coef:.4f})."
elif gender_pval < 0.10:
    response = 45
    explanation = f"Gender shows a marginal effect on mortgage approval (p={gender_pval:.4f}), which approaches but does not reach conventional statistical significance. While the univariate test suggests a relationship (p={p_value_chi:.4f}), the effect becomes weaker when controlling for creditworthiness and other legitimate lending criteria."
else:
    response = 25
    explanation = f"Gender does not have a statistically significant effect on mortgage approval when controlling for relevant factors (p={gender_pval:.4f}). While the univariate chi-square test shows p={p_value_chi:.4f}, this apparent relationship is largely explained by differences in creditworthiness, employment status, and other legitimate lending criteria. The multivariate analysis indicates gender is not a significant predictor."

print(f"Response (0-100): {response}")
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
