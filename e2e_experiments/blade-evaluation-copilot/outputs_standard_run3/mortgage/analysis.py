import json
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from imodels import RuleFitClassifier
import warnings
warnings.filterwarnings('ignore')

# Load research question and data
with open('info.json', 'r') as f:
    info = json.load(f)

research_question = info['research_questions'][0]
print(f"Research Question: {research_question}")

# Load dataset
df = pd.read_csv('mortgage.csv')
print(f"\nDataset shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")

# Explore data
print("\n=== Data Exploration ===")
print(df.describe())
print(f"\nMissing values:\n{df.isnull().sum()}")

# Drop rows with missing values in key variables
df_clean = df.dropna(subset=['female', 'accept', 'married', 'PI_ratio'])
print(f"\nDataset shape after dropping missing values: {df_clean.shape}")
df = df_clean

# Focus on gender (female) and acceptance (accept)
print("\n=== Gender Distribution ===")
print(df['female'].value_counts())
print(f"\nProportion female: {df['female'].mean():.3f}")

print("\n=== Acceptance Rate by Gender ===")
gender_acceptance = df.groupby('female')['accept'].agg(['mean', 'count', 'std'])
print(gender_acceptance)

male_acceptance = df[df['female'] == 0]['accept'].mean()
female_acceptance = df[df['female'] == 1]['accept'].mean()
print(f"\nMale acceptance rate: {male_acceptance:.3f}")
print(f"Female acceptance rate: {female_acceptance:.3f}")
print(f"Difference: {female_acceptance - male_acceptance:.3f}")

# Statistical Test 1: Chi-square test for independence
print("\n=== Chi-Square Test ===")
contingency_table = pd.crosstab(df['female'], df['accept'])
print("Contingency table:")
print(contingency_table)
chi2, p_value_chi2, dof, expected = stats.chi2_contingency(contingency_table)
print(f"Chi-square statistic: {chi2:.4f}")
print(f"P-value: {p_value_chi2:.6f}")

# Statistical Test 2: Two-sample t-test
print("\n=== Two-Sample T-Test ===")
male_accept = df[df['female'] == 0]['accept']
female_accept = df[df['female'] == 1]['accept']
t_stat, p_value_ttest = stats.ttest_ind(male_accept, female_accept)
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value_ttest:.6f}")

# Univariate Logistic Regression
print("\n=== Univariate Logistic Regression (accept ~ female) ===")
X_univariate = df[['female']]
y = df['accept']
model_univariate = LogisticRegression()
model_univariate.fit(X_univariate, y)
print(f"Coefficient for female: {model_univariate.coef_[0][0]:.4f}")

# Statsmodels for p-values
X_sm_uni = sm.add_constant(df['female'])
model_sm_uni = sm.Logit(y, X_sm_uni).fit(disp=0)
print(model_sm_uni.summary())
p_value_female_univariate = model_sm_uni.pvalues['female']
print(f"\nP-value for female coefficient: {p_value_female_univariate:.6f}")

# Multivariate Analysis: Control for other factors
print("\n=== Multivariate Logistic Regression ===")
# Select relevant features
control_vars = ['black', 'housing_expense_ratio', 'self_employed', 'married', 
                'mortgage_credit', 'consumer_credit', 'bad_history', 'PI_ratio', 
                'loan_to_value', 'denied_PMI']
X_multi = df[['female'] + control_vars].copy()

# Check for missing values
print(f"Missing values in features:\n{X_multi.isnull().sum()}")

# Fit multivariate model with statsmodels
X_sm_multi = sm.add_constant(X_multi)
model_sm_multi = sm.Logit(y, X_sm_multi).fit(disp=0)
print(model_sm_multi.summary())
p_value_female_multivariate = model_sm_multi.pvalues['female']
coef_female_multivariate = model_sm_multi.params['female']
print(f"\nFemale coefficient (controlling for other factors): {coef_female_multivariate:.4f}")
print(f"P-value for female coefficient: {p_value_female_multivariate:.6f}")

# Interpretable model: Decision Tree
print("\n=== Interpretable Model: RuleFit ===")
try:
    rulefit = RuleFitClassifier(max_rules=10, random_state=42)
    rulefit.fit(X_multi, y)
    
    # Get feature importances
    feature_importance = pd.DataFrame({
        'feature': X_multi.columns,
        'importance': rulefit.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nFeature importances from RuleFit:")
    print(feature_importance.head(10))
    
    female_importance_rank = feature_importance[feature_importance['feature'] == 'female'].index[0]
    print(f"\nFemale feature rank by importance: {female_importance_rank + 1} out of {len(feature_importance)}")
except Exception as e:
    print(f"RuleFit failed: {e}")

# Conclusion
print("\n=== CONCLUSION ===")
print(f"Research Question: {research_question}")
print(f"\nRaw acceptance rates:")
print(f"  Male: {male_acceptance:.3f} ({df[df['female']==0]['accept'].sum()}/{len(df[df['female']==0])})")
print(f"  Female: {female_acceptance:.3f} ({df[df['female']==1]['accept'].sum()}/{len(df[df['female']==1])})")
print(f"  Raw difference: {female_acceptance - male_acceptance:.3f}")
print(f"\nStatistical tests:")
print(f"  Chi-square p-value: {p_value_chi2:.6f}")
print(f"  T-test p-value: {p_value_ttest:.6f}")
print(f"  Univariate logistic regression p-value: {p_value_female_univariate:.6f}")
print(f"  Multivariate logistic regression p-value: {p_value_female_multivariate:.6f}")

# Determine response
alpha = 0.05
if p_value_female_multivariate < alpha:
    # Significant effect after controlling for confounders
    if coef_female_multivariate < 0:
        response = 80  # Strong evidence that gender affects approval negatively for females
        explanation = f"Gender has a statistically significant effect on mortgage approval (p={p_value_female_multivariate:.4f} < 0.05). After controlling for creditworthiness factors (credit scores, debt ratios, employment, etc.), being female is associated with lower approval odds (coefficient={coef_female_multivariate:.3f}). Raw acceptance rates show males at {male_acceptance:.1%} vs females at {female_acceptance:.1%}."
    else:
        response = 80  # Strong evidence that gender affects approval positively for females
        explanation = f"Gender has a statistically significant effect on mortgage approval (p={p_value_female_multivariate:.4f} < 0.05). After controlling for other factors, being female is associated with higher approval odds (coefficient={coef_female_multivariate:.3f})."
elif p_value_chi2 < alpha and abs(female_acceptance - male_acceptance) > 0.03:
    # Significant in univariate but not multivariate - likely confounded
    response = 40
    explanation = f"While raw acceptance rates differ between males ({male_acceptance:.1%}) and females ({female_acceptance:.1%}), this difference is not statistically significant after controlling for creditworthiness factors (multivariate p={p_value_female_multivariate:.3f}). The univariate relationship (p={p_value_chi2:.4f}) appears to be explained by other variables like credit scores, debt ratios, and employment status."
else:
    # No significant effect
    response = 15
    explanation = f"Gender does not have a statistically significant effect on mortgage approval. Both univariate (p={p_value_chi2:.3f}) and multivariate (p={p_value_female_multivariate:.3f}) analyses fail to show significance at α=0.05. Raw acceptance rates are similar: males {male_acceptance:.1%} vs females {female_acceptance:.1%}."

print(f"\nResponse score: {response}")
print(f"Explanation: {explanation}")

# Write conclusion to file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n✓ Analysis complete. conclusion.txt written.")
