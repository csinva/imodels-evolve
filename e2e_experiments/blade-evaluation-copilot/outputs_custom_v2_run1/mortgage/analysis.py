import json
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from agentic_imodels import (
    SmartAdditiveRegressor,
    HingeEBMRegressor,
    WinsorizedSparseOLSRegressor,
    HingeGAMRegressor
)

# Load the research question and data
with open('info.json', 'r') as f:
    info = json.load(f)

research_question = info['research_questions'][0]
print(f"Research Question: {research_question}")
print("=" * 80)

# Load the dataset
df = pd.read_csv('mortgage.csv')

print(f"\nDataset shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())

print("\nDataset summary statistics:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

# Handle missing values - drop rows with missing values in key columns
print("\nDropping rows with missing values...")
df = df.dropna()
print(f"Dataset shape after dropping missing values: {df.shape}")

# The outcome variable: accept (1 if accepted, 0 if denied)
# The predictor of interest: female (1 if female, 0 if male)
# Controls: black, housing_expense_ratio, self_employed, married, 
#           mortgage_credit, consumer_credit, bad_history, PI_ratio, 
#           loan_to_value, denied_PMI

print("\n" + "=" * 80)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 80)

# Check the distribution of key variables
print(f"\nFemale distribution: {df['female'].value_counts().to_dict()}")
print(f"Acceptance rate overall: {df['accept'].mean():.4f}")
print(f"Acceptance rate by gender:")
print(df.groupby('female')['accept'].mean())

# Bivariate analysis: female vs accept
female_accept = df[df['female'] == 1]['accept']
male_accept = df[df['female'] == 0]['accept']

print(f"\nMale acceptance rate: {male_accept.mean():.4f} (n={len(male_accept)})")
print(f"Female acceptance rate: {female_accept.mean():.4f} (n={len(female_accept)})")

# Two-sample t-test for difference in acceptance rates
t_stat, p_val = stats.ttest_ind(male_accept, female_accept)
print(f"T-test: t={t_stat:.4f}, p={p_val:.4f}")

# Chi-square test for independence
contingency_table = pd.crosstab(df['female'], df['accept'])
print(f"\nContingency table:")
print(contingency_table)
chi2, p_chi = stats.chi2_contingency(contingency_table)[:2]
print(f"Chi-square test: χ²={chi2:.4f}, p={p_chi:.4f}")

# Correlation analysis
print("\nCorrelation of 'female' with 'accept' and other features:")
correlations = df.corr()['accept'].sort_values(ascending=False)
print(correlations)

print("\n" + "=" * 80)
print("CLASSICAL STATISTICAL TESTS (OLS with controls)")
print("=" * 80)

# Prepare data for regression
# Drop the index column and redundant 'deny' column (inverse of accept)
feature_cols = ['female', 'black', 'housing_expense_ratio', 'self_employed', 
                'married', 'mortgage_credit', 'consumer_credit', 'bad_history', 
                'PI_ratio', 'loan_to_value', 'denied_PMI']
outcome_col = 'accept'

# 1. Bivariate regression: female -> accept
X_bivariate = sm.add_constant(df[['female']])
model_bivariate = sm.OLS(df[outcome_col], X_bivariate).fit()
print("\n1. Bivariate OLS (female only):")
print(model_bivariate.summary())

# 2. Multivariate regression: female + controls -> accept
X_controls = sm.add_constant(df[feature_cols])
model_controls = sm.OLS(df[outcome_col], X_controls).fit()
print("\n2. Multivariate OLS (female + controls):")
print(model_controls.summary())

# 3. Logistic regression (more appropriate for binary outcome)
print("\n3. Logistic Regression (bivariate):")
logit_bivariate = sm.Logit(df[outcome_col], X_bivariate).fit()
print(logit_bivariate.summary())

print("\n4. Logistic Regression (with controls):")
logit_controls = sm.Logit(df[outcome_col], X_controls).fit()
print(logit_controls.summary())

print("\n" + "=" * 80)
print("INTERPRETABLE MODELS (agentic_imodels)")
print("=" * 80)

# Prepare data for interpretable models
X = df[feature_cols]
y = df[outcome_col]

# Split for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit multiple interpretable models
models_to_fit = [
    SmartAdditiveRegressor(),
    HingeEBMRegressor(),
    WinsorizedSparseOLSRegressor(),
    HingeGAMRegressor()
]

model_results = []

for model in models_to_fit:
    model_name = model.__class__.__name__
    print(f"\n{'=' * 80}")
    print(f"Fitting {model_name}")
    print(f"{'=' * 80}")
    
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    print(f"\nPerformance:")
    print(f"  R² (train): {r2_train:.4f}")
    print(f"  R² (test): {r2_test:.4f}")
    print(f"  RMSE (test): {rmse_test:.4f}")
    
    print(f"\nModel form:")
    print(model)
    
    model_results.append({
        'name': model_name,
        'r2_train': r2_train,
        'r2_test': r2_test,
        'rmse_test': rmse_test,
        'model_str': str(model)
    })

print("\n" + "=" * 80)
print("SYNTHESIS AND CONCLUSION")
print("=" * 80)

# Analyze the results
print("\nKey findings:")

# From bivariate analysis
print(f"\n1. BIVARIATE ANALYSIS:")
print(f"   - Male acceptance rate: {male_accept.mean():.4f}")
print(f"   - Female acceptance rate: {female_accept.mean():.4f}")
print(f"   - Difference: {male_accept.mean() - female_accept.mean():.4f}")
print(f"   - T-test p-value: {p_val:.4f}")
print(f"   - Chi-square p-value: {p_chi:.4f}")

# From OLS
female_coef_bivariate = model_bivariate.params['female']
female_pval_bivariate = model_bivariate.pvalues['female']
female_coef_controls = model_controls.params['female']
female_pval_controls = model_controls.pvalues['female']

print(f"\n2. OLS REGRESSION:")
print(f"   - Bivariate: β={female_coef_bivariate:.4f}, p={female_pval_bivariate:.4f}")
print(f"   - With controls: β={female_coef_controls:.4f}, p={female_pval_controls:.4f}")

# From logistic regression
female_coef_logit_biv = logit_bivariate.params['female']
female_pval_logit_biv = logit_bivariate.pvalues['female']
female_coef_logit_ctrl = logit_controls.params['female']
female_pval_logit_ctrl = logit_controls.pvalues['female']

print(f"\n3. LOGISTIC REGRESSION:")
print(f"   - Bivariate: β={female_coef_logit_biv:.4f}, p={female_pval_logit_biv:.4f}")
print(f"   - With controls: β={female_coef_logit_ctrl:.4f}, p={female_pval_logit_ctrl:.4f}")

print(f"\n4. INTERPRETABLE MODELS:")
for result in model_results:
    print(f"   - {result['name']}: R²={result['r2_test']:.4f}, RMSE={result['rmse_test']:.4f}")

# Determine the response
# The question asks: "How does gender affect whether banks approve an individual's mortgage application?"

# Analysis of evidence:
# - Bivariate: there IS a difference in acceptance rates (male: ~0.87, female: ~0.82)
# - Statistical significance: The p-values from t-test and chi-square suggest some difference
# - BUT: When controlling for other factors (credit, income ratios, employment, etc.)
#   the effect of gender becomes much smaller or potentially non-significant
# - The interpretable models will show us if 'female' is important after accounting for other factors

# Check what the interpretable models say about 'female' variable importance
print("\n5. INTERPRETATION FROM INTERPRETABLE MODELS:")

# Look for evidence in the model strings
female_mentioned = []
for result in model_results:
    model_str = result['model_str']
    if 'female' in model_str.lower():
        female_mentioned.append(result['name'])
        print(f"   - {result['name']}: 'female' appears in the model")
        # Extract the relevant lines
        for line in model_str.split('\n'):
            if 'female' in line.lower():
                print(f"     {line.strip()}")
    else:
        print(f"   - {result['name']}: 'female' NOT selected/appears minimal")

# Synthesize conclusion
explanation_parts = []

# Evidence for an effect
if p_chi < 0.05 or p_val < 0.05:
    explanation_parts.append(f"Bivariate analysis shows a significant difference in acceptance rates (male: {male_accept.mean():.3f}, female: {female_accept.mean():.3f}, p={min(p_val, p_chi):.4f}).")

# Effect after controls
if female_pval_controls < 0.05 or abs(female_coef_controls) > 0.01:
    explanation_parts.append(f"In OLS with controls, the female coefficient is {female_coef_controls:.4f} (p={female_pval_controls:.4f}).")
else:
    explanation_parts.append(f"In OLS with controls, the female effect becomes very small (β={female_coef_controls:.4f}, p={female_pval_controls:.4f}).")

if female_pval_logit_ctrl < 0.05 or abs(female_coef_logit_ctrl) > 0.1:
    explanation_parts.append(f"Logistic regression with controls shows β={female_coef_logit_ctrl:.4f} (p={female_pval_logit_ctrl:.4f}).")
else:
    explanation_parts.append(f"Logistic regression with controls shows minimal effect (β={female_coef_logit_ctrl:.4f}, p={female_pval_logit_ctrl:.4f}).")

# Interpretable models evidence
if len(female_mentioned) >= 2:
    explanation_parts.append(f"{len(female_mentioned)} out of {len(model_results)} interpretable models include 'female' as a relevant feature, suggesting it has some predictive importance.")
elif len(female_mentioned) == 1:
    explanation_parts.append(f"Only {len(female_mentioned)} out of {len(model_results)} interpretable models include 'female', suggesting weak to moderate importance.")
else:
    explanation_parts.append(f"None of the interpretable models prominently feature 'female', suggesting other factors (credit scores, income ratios, etc.) are more important.")

# Key controls that matter more
top_controls = correlations.head(6)  # Top features correlated with acceptance
top_control_names = [c for c in top_controls.index if c not in ['accept', 'deny', 'Unnamed: 0']]
if top_control_names:
    explanation_parts.append(f"The strongest predictors of acceptance are: {', '.join(top_control_names[:3])}, with 'female' being less dominant.")

# Determine Likert score
# Scoring logic:
# - Strong significant effect that persists across models and is top-ranked → 75-100
# - Moderate/partially significant/mid-rank → 40-70
# - Weak, inconsistent, or marginal → 15-40
# - Zero coefficient in Lasso AND non-significant AND low importance → 0-15

# Check the strength of evidence:
bivariate_significant = (p_chi < 0.05) or (p_val < 0.05)
controls_significant = (female_pval_controls < 0.05) or (female_pval_logit_ctrl < 0.05)
models_support = len(female_mentioned) >= 2
effect_magnitude = abs(female_coef_controls)

if bivariate_significant and controls_significant and models_support and effect_magnitude > 0.03:
    # Strong evidence
    response = 70  # Moderate to strong
elif bivariate_significant and (controls_significant or models_support):
    # Moderate evidence
    response = 50
elif bivariate_significant and not controls_significant and not models_support:
    # Weak evidence - bivariate only, disappears with controls
    response = 25
else:
    # Very weak or no evidence
    response = 15

explanation = " ".join(explanation_parts)

print(f"\n{'=' * 80}")
print(f"FINAL RESPONSE: {response}/100")
print(f"EXPLANATION: {explanation}")
print(f"{'=' * 80}")

# Write conclusion to file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\nConclusion written to conclusion.txt")
