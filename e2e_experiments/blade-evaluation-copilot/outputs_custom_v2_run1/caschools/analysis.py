import json
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from agentic_imodels import SmartAdditiveRegressor, HingeEBMRegressor, WinsorizedSparseOLSRegressor

# Load data
df = pd.read_csv('caschools.csv')

print("=" * 80)
print("RESEARCH QUESTION:")
print("Is a lower student-teacher ratio associated with higher academic performance?")
print("=" * 80)

# Calculate student-teacher ratio
df['str_ratio'] = df['students'] / df['teachers']

# Use average of reading and math scores as academic performance measure
df['avg_score'] = (df['read'] + df['math']) / 2

print("\n" + "=" * 80)
print("DATA EXPLORATION")
print("=" * 80)

print(f"\nDataset shape: {df.shape}")
print(f"\nBasic statistics for key variables:")
print(df[['str_ratio', 'avg_score', 'read', 'math']].describe())

print(f"\n\nMissing values:")
print(df[['str_ratio', 'avg_score', 'income', 'english', 'lunch', 'calworks']].isnull().sum())

# Bivariate correlation
corr = stats.pearsonr(df['str_ratio'], df['avg_score'])
print(f"\n\nBivariate correlation (student-teacher ratio vs avg_score):")
print(f"  r = {corr[0]:.4f}, p-value = {corr[1]:.6f}")

print("\n\nCorrelation matrix:")
corr_cols = ['str_ratio', 'avg_score', 'income', 'english', 'lunch', 'calworks', 'expenditure']
print(df[corr_cols].corr()['avg_score'].sort_values(ascending=False))

print("\n" + "=" * 80)
print("CLASSICAL REGRESSION: OLS WITH CONTROLS")
print("=" * 80)

# Prepare data for regression
# Controls: income, english learners, lunch (proxy for poverty), calworks, expenditure
control_vars = ['income', 'english', 'lunch', 'calworks', 'expenditure']
all_vars = ['str_ratio'] + control_vars

# Remove any rows with missing values in these columns
df_clean = df[all_vars + ['avg_score']].dropna()

print(f"\nSample size after removing missing values: {len(df_clean)}")

# Bivariate OLS (no controls)
X_bivariate = sm.add_constant(df_clean[['str_ratio']])
model_bivariate = sm.OLS(df_clean['avg_score'], X_bivariate).fit()
print("\n--- Bivariate OLS (no controls) ---")
print(model_bivariate.summary())

# Full OLS with controls
X_full = sm.add_constant(df_clean[all_vars])
model_full = sm.OLS(df_clean['avg_score'], X_full).fit()
print("\n--- Full OLS with controls ---")
print(model_full.summary())

# Extract key results
str_coef_bivariate = model_bivariate.params['str_ratio']
str_pval_bivariate = model_bivariate.pvalues['str_ratio']
str_coef_full = model_full.params['str_ratio']
str_pval_full = model_full.pvalues['str_ratio']

print(f"\n\nKEY FINDINGS FROM OLS:")
print(f"  Bivariate: coef = {str_coef_bivariate:.4f}, p = {str_pval_bivariate:.6f}")
print(f"  With controls: coef = {str_coef_full:.4f}, p = {str_pval_full:.6f}")

print("\n" + "=" * 80)
print("INTERPRETABLE MODELS: SHAPE, DIRECTION, MAGNITUDE, ROBUSTNESS")
print("=" * 80)

# Prepare data for interpretable models
X = df_clean[all_vars]
y = df_clean['avg_score']

# Fit multiple interpretable models
models_to_fit = [
    ('SmartAdditiveRegressor', SmartAdditiveRegressor()),
    ('HingeEBMRegressor', HingeEBMRegressor()),
    ('WinsorizedSparseOLSRegressor', WinsorizedSparseOLSRegressor())
]

results = {}

for name, model in models_to_fit:
    print(f"\n\n{'=' * 40}")
    print(f"{name}")
    print('=' * 40)
    
    # Fit model
    model.fit(X, y)
    
    # Predictions
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    print(f"\nPerformance: R² = {r2:.4f}, RMSE = {rmse:.4f}")
    print(f"\nFitted model form:")
    print(model)
    
    results[name] = {
        'model': model,
        'r2': r2,
        'rmse': rmse
    }

print("\n" + "=" * 80)
print("INTERPRETATION AND CONCLUSION")
print("=" * 80)

# Analyze the results
print("\n1. STATISTICAL SIGNIFICANCE:")
print(f"   - Bivariate relationship: coef = {str_coef_bivariate:.4f}, p = {str_pval_bivariate:.6f}")
print(f"     {'SIGNIFICANT' if str_pval_bivariate < 0.05 else 'NOT SIGNIFICANT'} at α=0.05")
print(f"   - With controls: coef = {str_coef_full:.4f}, p = {str_pval_full:.6f}")
print(f"     {'SIGNIFICANT' if str_pval_full < 0.05 else 'NOT SIGNIFICANT'} at α=0.05")

print("\n2. DIRECTION AND MAGNITUDE:")
if str_coef_bivariate < 0:
    print("   - NEGATIVE relationship: higher student-teacher ratio → lower scores")
    print(f"   - Effect size: {abs(str_coef_bivariate):.2f} points per unit increase in STR")
elif str_coef_bivariate > 0:
    print("   - POSITIVE relationship: higher student-teacher ratio → higher scores")
    print(f"   - Effect size: {str_coef_bivariate:.2f} points per unit increase in STR")
else:
    print("   - NO relationship detected")

print("\n3. ROBUSTNESS:")
if str_pval_bivariate < 0.05 and str_pval_full < 0.05:
    print("   - Effect PERSISTS after controlling for income, English learners,")
    print("     lunch eligibility, CalWorks, and expenditure")
elif str_pval_bivariate < 0.05 and str_pval_full >= 0.05:
    print("   - Effect WEAKENS and becomes non-significant with controls")
    print("   - Suggests confounding by socioeconomic factors")
else:
    print("   - Relationship not significant even in bivariate analysis")

print("\n4. FEATURE IMPORTANCE FROM INTERPRETABLE MODELS:")
print("   - All three interpretable models were fitted")
print("   - Check the printed model forms above for:")
print("     • Whether str_ratio coefficient is non-zero")
print("     • Relative magnitude compared to other features")
print("     • Shape (linear vs. threshold vs. non-monotone)")

# Calculate Likert score
explanation_parts = []

# Determine Likert score based on evidence
if str_pval_bivariate < 0.001 and str_pval_full < 0.05 and str_coef_bivariate < 0:
    # Very strong negative relationship, persists with controls
    likert_score = 85
    explanation_parts.append("Strong statistical evidence: STR negatively affects scores")
    explanation_parts.append(f"Bivariate: coef={str_coef_bivariate:.3f}, p<0.001")
    explanation_parts.append(f"With controls: coef={str_coef_full:.3f}, p={str_pval_full:.4f}")
    explanation_parts.append("Effect persists after controlling for SES factors")
    
elif str_pval_bivariate < 0.05 and str_pval_full < 0.05 and str_coef_bivariate < 0:
    # Significant negative relationship, persists with controls
    likert_score = 75
    explanation_parts.append("Significant negative relationship between STR and scores")
    explanation_parts.append(f"Bivariate: coef={str_coef_bivariate:.3f}, p={str_pval_bivariate:.4f}")
    explanation_parts.append(f"With controls: coef={str_coef_full:.3f}, p={str_pval_full:.4f}")
    explanation_parts.append("Relationship robust to socioeconomic controls")
    
elif str_pval_bivariate < 0.05 and str_coef_bivariate < 0:
    # Significant bivariate but weakens with controls
    if str_pval_full < 0.10:
        likert_score = 60
        explanation_parts.append("Moderate evidence: STR negatively affects scores")
        explanation_parts.append(f"Bivariate: coef={str_coef_bivariate:.3f}, p={str_pval_bivariate:.4f}")
        explanation_parts.append(f"With controls: coef={str_coef_full:.3f}, p={str_pval_full:.4f} (marginally significant)")
    else:
        likert_score = 40
        explanation_parts.append("Weak evidence: bivariate relationship present but weakens substantially with controls")
        explanation_parts.append(f"Bivariate: coef={str_coef_bivariate:.3f}, p={str_pval_bivariate:.4f}")
        explanation_parts.append(f"With controls: coef={str_coef_full:.3f}, p={str_pval_full:.4f} (not significant)")
        explanation_parts.append("Effect likely confounded by socioeconomic factors")
        
else:
    # No significant relationship
    likert_score = 15
    explanation_parts.append("Little to no evidence of relationship")
    explanation_parts.append(f"Bivariate: coef={str_coef_bivariate:.3f}, p={str_pval_bivariate:.4f}")
    if str_pval_full is not None:
        explanation_parts.append(f"With controls: coef={str_coef_full:.3f}, p={str_pval_full:.4f}")

# Check interpretable models - add robustness note
explanation_parts.append("Interpretable models fitted for shape/magnitude analysis")

explanation = " | ".join(explanation_parts)

print("\n" + "=" * 80)
print("FINAL CONCLUSION")
print("=" * 80)
print(f"\nLikert Score: {likert_score}/100")
print(f"\nExplanation: {explanation}")

# Write conclusion to file
conclusion = {
    "response": likert_score,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n✓ conclusion.txt written successfully")
