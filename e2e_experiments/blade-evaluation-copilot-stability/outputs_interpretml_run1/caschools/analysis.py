import pandas as pd
import numpy as np
import json
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from interpret.glassbox import ExplainableBoostingRegressor

# Load the dataset
df = pd.read_csv('caschools.csv')

# Calculate student-teacher ratio
df['student_teacher_ratio'] = df['students'] / df['teachers']

# Create composite academic performance score (average of math and reading)
df['academic_performance'] = (df['math'] + df['read']) / 2

print("="*80)
print("EXPLORATORY DATA ANALYSIS")
print("="*80)
print("\nDataset shape:", df.shape)
print("\nBasic statistics for key variables:")
print(df[['student_teacher_ratio', 'academic_performance', 'math', 'read']].describe())

print("\n" + "="*80)
print("CORRELATION ANALYSIS")
print("="*80)
# Calculate correlation between student-teacher ratio and academic performance
corr_performance = stats.pearsonr(df['student_teacher_ratio'], df['academic_performance'])
corr_math = stats.pearsonr(df['student_teacher_ratio'], df['math'])
corr_read = stats.pearsonr(df['student_teacher_ratio'], df['read'])

print(f"\nCorrelation (student-teacher ratio vs academic performance): r={corr_performance[0]:.4f}, p={corr_performance[1]:.6f}")
print(f"Correlation (student-teacher ratio vs math scores): r={corr_math[0]:.4f}, p={corr_math[1]:.6f}")
print(f"Correlation (student-teacher ratio vs reading scores): r={corr_read[0]:.4f}, p={corr_read[1]:.6f}")

print("\n" + "="*80)
print("SIMPLE LINEAR REGRESSION (STATSMODELS)")
print("="*80)
# Simple linear regression with statsmodels for p-values
X_simple = sm.add_constant(df['student_teacher_ratio'])
y = df['academic_performance']
model_simple = sm.OLS(y, X_simple).fit()
print(model_simple.summary())

print("\n" + "="*80)
print("MULTIPLE REGRESSION (CONTROLLING FOR CONFOUNDERS)")
print("="*80)
# Multiple regression controlling for confounding variables
# Include socioeconomic factors that might affect both class size and performance
confounders = ['student_teacher_ratio', 'income', 'lunch', 'english', 'expenditure']
X_multi = sm.add_constant(df[confounders])
y = df['academic_performance']
model_multi = sm.OLS(y, X_multi).fit()
print(model_multi.summary())

print("\n" + "="*80)
print("INTERPRETABLE MODEL: EXPLAINABLE BOOSTING REGRESSOR")
print("="*80)
# Use EBM for interpretable analysis
X_ebm = df[confounders].copy()
y_ebm = df['academic_performance'].copy()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_ebm, y_ebm, test_size=0.2, random_state=42)

# Train EBM
ebm = ExplainableBoostingRegressor(random_state=42)
ebm.fit(X_train, y_train)

# Get feature importances
ebm_importances = ebm.term_importances()
feature_importance = list(zip(ebm.feature_names_in_, ebm_importances))
feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)

print("\nFeature Importances from EBM:")
for feature, importance in feature_importance:
    print(f"  {feature}: {importance:.4f}")

print("\n" + "="*80)
print("ANALYSIS AND CONCLUSION")
print("="*80)

# Extract key statistics
str_coefficient_simple = model_simple.params['student_teacher_ratio']
str_pvalue_simple = model_simple.pvalues['student_teacher_ratio']
str_coefficient_multi = model_multi.params['student_teacher_ratio']
str_pvalue_multi = model_multi.pvalues['student_teacher_ratio']

print(f"\nSimple regression coefficient: {str_coefficient_simple:.4f} (p={str_pvalue_simple:.6f})")
print(f"Multiple regression coefficient: {str_coefficient_multi:.4f} (p={str_pvalue_multi:.6f})")

# Interpretation
print("\nInterpretation:")
print(f"- Correlation is negative (r={corr_performance[0]:.4f}), indicating lower STR → higher performance")
print(f"- In simple regression, each unit increase in STR is associated with {str_coefficient_simple:.2f} point decrease")
print(f"- After controlling for income, lunch program %, English learners %, and expenditure:")
print(f"  Coefficient = {str_coefficient_multi:.4f}, p-value = {str_pvalue_multi:.6f}")

# Determine response based on statistical significance
# Using alpha = 0.05 as threshold
if str_pvalue_multi < 0.001:
    response_score = 95
    explanation = f"Strong evidence (p<0.001) that lower student-teacher ratio is associated with higher academic performance. After controlling for socioeconomic confounders (income, lunch program eligibility, English learners, expenditure), the relationship remains statistically significant (coefficient={str_coefficient_multi:.3f}, p={str_pvalue_multi:.6f}). The negative coefficient indicates that as student-teacher ratio decreases (more teachers per student), academic performance increases."
elif str_pvalue_multi < 0.01:
    response_score = 85
    explanation = f"Very strong evidence (p<0.01) that lower student-teacher ratio is associated with higher academic performance. After controlling for confounders, the relationship is highly significant (coefficient={str_coefficient_multi:.3f}, p={str_pvalue_multi:.6f})."
elif str_pvalue_multi < 0.05:
    response_score = 75
    explanation = f"Significant evidence (p<0.05) that lower student-teacher ratio is associated with higher academic performance. After controlling for confounders, the relationship is statistically significant (coefficient={str_coefficient_multi:.3f}, p={str_pvalue_multi:.6f})."
elif str_pvalue_multi < 0.10:
    response_score = 55
    explanation = f"Weak evidence (p<0.10) for association between student-teacher ratio and academic performance. After controlling for confounders, the relationship is marginally significant (coefficient={str_coefficient_multi:.3f}, p={str_pvalue_multi:.6f})."
else:
    response_score = 30
    explanation = f"Limited evidence for association between student-teacher ratio and academic performance after controlling for confounders. While simple correlation is significant, the relationship becomes non-significant (p={str_pvalue_multi:.6f}) when accounting for socioeconomic factors."

print(f"\nFinal Response Score: {response_score}/100")
print(f"Explanation: {explanation}")

# Write conclusion to file
conclusion = {
    "response": response_score,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n" + "="*80)
print("CONCLUSION WRITTEN TO conclusion.txt")
print("="*80)
