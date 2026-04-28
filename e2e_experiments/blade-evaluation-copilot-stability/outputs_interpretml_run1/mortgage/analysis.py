import pandas as pd
import numpy as np
import json
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import statsmodels.api as sm
from interpret.glassbox import ExplainableBoostingClassifier

# Load the data
df = pd.read_csv('mortgage.csv')

print("="*80)
print("MORTGAGE APPROVAL ANALYSIS: GENDER EFFECTS")
print("="*80)
print(f"\nDataset shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")

# Handle missing values
print(f"\nMissing values per column:")
print(df.isnull().sum())

# Drop rows with missing values for analysis
df_clean = df.dropna()
print(f"\nDataset shape after removing missing values: {df_clean.shape}")
df = df_clean

# Basic exploration
print("\n" + "="*80)
print("DATA EXPLORATION")
print("="*80)

print("\nGender distribution:")
print(df['female'].value_counts())
print(f"\nProportion female: {df['female'].mean():.3f}")

print("\nAcceptance rate overall:")
print(df['accept'].value_counts())
print(f"Overall acceptance rate: {df['accept'].mean():.3f}")

# Key analysis: Acceptance rate by gender
print("\n" + "="*80)
print("ACCEPTANCE RATES BY GENDER")
print("="*80)

acceptance_by_gender = df.groupby('female')['accept'].agg(['mean', 'count', 'sum'])
acceptance_by_gender.index = ['Male', 'Female']
print(acceptance_by_gender)

male_acceptance = df[df['female'] == 0]['accept'].mean()
female_acceptance = df[df['female'] == 1]['accept'].mean()

print(f"\nMale acceptance rate: {male_acceptance:.4f}")
print(f"Female acceptance rate: {female_acceptance:.4f}")
print(f"Difference: {female_acceptance - male_acceptance:.4f}")

# Statistical test: Chi-square test for independence
print("\n" + "="*80)
print("CHI-SQUARE TEST: GENDER vs ACCEPTANCE")
print("="*80)

contingency_table = pd.crosstab(df['female'], df['accept'])
print("\nContingency table:")
print(contingency_table)

chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
print(f"\nChi-square statistic: {chi2:.4f}")
print(f"P-value: {p_value:.6f}")
print(f"Degrees of freedom: {dof}")

if p_value < 0.05:
    print(f"Result: SIGNIFICANT (p < 0.05) - Gender and acceptance are NOT independent")
else:
    print(f"Result: NOT SIGNIFICANT (p >= 0.05) - No evidence of relationship")

# T-test comparing acceptance rates
print("\n" + "="*80)
print("TWO-SAMPLE T-TEST: ACCEPTANCE BY GENDER")
print("="*80)

male_accepts = df[df['female'] == 0]['accept']
female_accepts = df[df['female'] == 1]['accept']

t_stat, t_pvalue = stats.ttest_ind(male_accepts, female_accepts)
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {t_pvalue:.6f}")

# Logistic regression - univariate (gender only)
print("\n" + "="*80)
print("LOGISTIC REGRESSION: GENDER ONLY")
print("="*80)

X_gender = df[['female']]
y = df['accept']

logreg_simple = LogisticRegression(random_state=42)
logreg_simple.fit(X_gender, y)

print(f"Gender coefficient: {logreg_simple.coef_[0][0]:.4f}")
print(f"Intercept: {logreg_simple.intercept_[0]:.4f}")

# Use statsmodels for p-values
X_gender_sm = sm.add_constant(X_gender)
logit_model = sm.Logit(y, X_gender_sm)
logit_result = logit_model.fit(disp=0)

print("\nStatsmodels Logistic Regression Summary:")
print(logit_result.summary2())

gender_pvalue = logit_result.pvalues['female']
gender_coef = logit_result.params['female']
print(f"\nGender coefficient: {gender_coef:.4f}")
print(f"Gender p-value: {gender_pvalue:.6f}")

# Multivariate analysis - controlling for other factors
print("\n" + "="*80)
print("MULTIVARIATE LOGISTIC REGRESSION")
print("="*80)

# Select relevant features for the model
feature_cols = ['female', 'black', 'housing_expense_ratio', 'self_employed', 
                'married', 'mortgage_credit', 'consumer_credit', 'bad_history',
                'PI_ratio', 'loan_to_value', 'denied_PMI']

X_multi = df[feature_cols]
y = df['accept']

# Statsmodels for p-values
X_multi_sm = sm.add_constant(X_multi)
logit_model_multi = sm.Logit(y, X_multi_sm)
logit_result_multi = logit_model_multi.fit(disp=0)

print("\nMultivariate Logistic Regression Summary:")
print(logit_result_multi.summary2())

gender_coef_multi = logit_result_multi.params['female']
gender_pvalue_multi = logit_result_multi.pvalues['female']

print(f"\nGender coefficient (controlling for other factors): {gender_coef_multi:.4f}")
print(f"Gender p-value (controlling for other factors): {gender_pvalue_multi:.6f}")

# Interpretable model: ExplainableBoostingClassifier
print("\n" + "="*80)
print("EXPLAINABLE BOOSTING CLASSIFIER")
print("="*80)

X_train, X_test, y_train, y_test = train_test_split(X_multi, y, test_size=0.2, random_state=42)

ebm = ExplainableBoostingClassifier(random_state=42)
ebm.fit(X_train, y_train)

# Get feature importances
feature_importance = ebm.term_importances()
feature_names = X_multi.columns

print("\nFeature Importances:")
for name, importance in sorted(zip(feature_names, feature_importance), key=lambda x: x[1], reverse=True):
    print(f"{name:25s}: {importance:.4f}")

female_importance = feature_importance[feature_names.tolist().index('female')]
female_rank = sorted(feature_importance, reverse=True).index(female_importance) + 1
print(f"\nGender importance rank: {female_rank} out of {len(feature_names)}")

# Summary statistics by gender for context
print("\n" + "="*80)
print("DESCRIPTIVE STATISTICS BY GENDER")
print("="*80)

for col in ['housing_expense_ratio', 'PI_ratio', 'loan_to_value', 
            'mortgage_credit', 'consumer_credit', 'bad_history']:
    print(f"\n{col}:")
    print(f"  Male mean: {df[df['female']==0][col].mean():.4f}")
    print(f"  Female mean: {df[df['female']==1][col].mean():.4f}")
    t_stat, p_val = stats.ttest_ind(df[df['female']==0][col], df[df['female']==1][col])
    print(f"  T-test p-value: {p_val:.6f}")

# CONCLUSION
print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

# Determine the response based on statistical evidence
explanation_parts = []

# 1. Univariate relationship
if p_value < 0.05:
    explanation_parts.append(f"There is a statistically significant relationship between gender and mortgage approval (chi-square p={p_value:.4f}).")
    if female_acceptance > male_acceptance:
        explanation_parts.append(f"Female applicants have a higher acceptance rate ({female_acceptance:.1%}) than male applicants ({male_acceptance:.1%}).")
    else:
        explanation_parts.append(f"Male applicants have a higher acceptance rate ({male_acceptance:.1%}) than female applicants ({female_acceptance:.1%}).")
else:
    explanation_parts.append(f"The chi-square test shows no significant relationship between gender and acceptance (p={p_value:.4f}).")

# 2. Multivariate analysis
if gender_pvalue_multi < 0.05:
    explanation_parts.append(f"Even after controlling for creditworthiness factors, gender remains a significant predictor (p={gender_pvalue_multi:.4f}).")
    response_score = 75  # Strong evidence
else:
    explanation_parts.append(f"After controlling for credit history, debt ratios, and other factors, gender is not a significant predictor (p={gender_pvalue_multi:.4f}).")
    if p_value < 0.05:
        explanation_parts.append("This suggests the gender difference in univariate analysis is explained by differences in financial characteristics between male and female applicants.")
        response_score = 30  # Weak relationship, explained by confounders
    else:
        response_score = 10  # No evidence

# 3. Effect size consideration
if abs(gender_coef_multi) > 0.1 and gender_pvalue_multi < 0.05:
    explanation_parts.append(f"The gender effect size is meaningful (coefficient={gender_coef_multi:.3f}).")

# 4. Feature importance
if female_rank <= 5 and p_value < 0.05:
    explanation_parts.append(f"In the interpretable model, gender ranks #{female_rank} in importance among all features.")

# Final scoring logic
if p_value < 0.05 and gender_pvalue_multi < 0.05:
    response_score = 80  # Strong evidence at both levels
elif p_value < 0.05 and gender_pvalue_multi < 0.10:
    response_score = 65  # Univariate significant, multivariate marginal
elif p_value < 0.05 and gender_pvalue_multi >= 0.10:
    response_score = 25  # Relationship exists but explained by confounders
elif p_value >= 0.05:
    response_score = 10  # No evidence of relationship

explanation = " ".join(explanation_parts)

print(f"\nResponse score: {response_score}")
print(f"\nExplanation: {explanation}")

# Write conclusion to file
conclusion = {
    "response": response_score,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n" + "="*80)
print("Analysis complete. Results written to conclusion.txt")
print("="*80)
