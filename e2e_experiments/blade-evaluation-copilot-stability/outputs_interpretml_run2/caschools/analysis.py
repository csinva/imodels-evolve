import pandas as pd
import numpy as np
import json
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from interpret.glassbox import ExplainableBoostingRegressor

# Load the dataset
df = pd.read_csv('caschools.csv')

# Calculate student-teacher ratio
df['student_teacher_ratio'] = df['students'] / df['teachers']

# Create a composite academic performance score (average of reading and math)
df['academic_performance'] = (df['read'] + df['math']) / 2

print("=" * 80)
print("DATA EXPLORATION")
print("=" * 80)
print("\nDataset shape:", df.shape)
print("\nStudent-Teacher Ratio Statistics:")
print(df['student_teacher_ratio'].describe())
print("\nAcademic Performance Statistics:")
print(df['academic_performance'].describe())

# Check for missing values
print("\nMissing values:")
print(df[['student_teacher_ratio', 'academic_performance', 'read', 'math']].isnull().sum())

# Remove any rows with missing values in key columns
df_clean = df[['student_teacher_ratio', 'academic_performance', 'read', 'math', 
               'income', 'english', 'lunch', 'calworks']].dropna()

print(f"\nClean dataset shape: {df_clean.shape}")

print("\n" + "=" * 80)
print("CORRELATION ANALYSIS")
print("=" * 80)

# Pearson correlation between student-teacher ratio and academic performance
pearson_corr, pearson_p = pearsonr(df_clean['student_teacher_ratio'], 
                                     df_clean['academic_performance'])
print(f"\nPearson correlation: {pearson_corr:.4f}")
print(f"P-value: {pearson_p:.6f}")

# Spearman correlation (non-parametric)
spearman_corr, spearman_p = spearmanr(df_clean['student_teacher_ratio'], 
                                       df_clean['academic_performance'])
print(f"\nSpearman correlation: {spearman_corr:.4f}")
print(f"P-value: {spearman_p:.6f}")

print("\n" + "=" * 80)
print("SIMPLE LINEAR REGRESSION")
print("=" * 80)

# Simple linear regression using statsmodels for statistical inference
X = sm.add_constant(df_clean['student_teacher_ratio'])
y = df_clean['academic_performance']

model_simple = sm.OLS(y, X).fit()
print(model_simple.summary())

print("\n" + "=" * 80)
print("MULTIPLE REGRESSION (CONTROLLING FOR CONFOUNDERS)")
print("=" * 80)

# Multiple regression controlling for socioeconomic factors
# These confounders could affect both class size and performance
X_multi = sm.add_constant(df_clean[['student_teacher_ratio', 'income', 
                                      'english', 'lunch', 'calworks']])
model_multi = sm.OLS(df_clean['academic_performance'], X_multi).fit()
print(model_multi.summary())

print("\n" + "=" * 80)
print("INTERPRETABLE MODEL: EXPLAINABLE BOOSTING MACHINE")
print("=" * 80)

# Use Explainable Boosting Regressor for interpretable model
X_ebm = df_clean[['student_teacher_ratio', 'income', 'english', 'lunch', 'calworks']]
y_ebm = df_clean['academic_performance']

ebm = ExplainableBoostingRegressor(random_state=42)
ebm.fit(X_ebm, y_ebm)

# Get feature importances
feature_importance = ebm.term_importances()
print("\nFeature Importances (EBM):")
for i, (name, importance) in enumerate(zip(X_ebm.columns, feature_importance)):
    print(f"{name}: {importance:.4f}")

print("\n" + "=" * 80)
print("ANALYSIS OF RESULTS")
print("=" * 80)

# Analyze the coefficient for student-teacher ratio from multiple regression
coef_str = model_multi.params['student_teacher_ratio']
pval_str = model_multi.pvalues['student_teacher_ratio']
conf_int = model_multi.conf_int().loc['student_teacher_ratio']

print(f"\nStudent-Teacher Ratio coefficient: {coef_str:.4f}")
print(f"95% Confidence Interval: [{conf_int[0]:.4f}, {conf_int[1]:.4f}]")
print(f"P-value: {pval_str:.6f}")

# Determine statistical significance
alpha = 0.05
is_significant = pval_str < alpha

print(f"\nStatistically significant at α={alpha}? {is_significant}")

# Interpret the direction of the relationship
if is_significant:
    if coef_str < 0:
        relationship_direction = "NEGATIVE (lower ratio → higher performance)"
    else:
        relationship_direction = "POSITIVE (higher ratio → lower performance)"
    print(f"Direction of relationship: {relationship_direction}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

# Build conclusion based on statistical evidence
explanation_parts = []

# Add correlation evidence
explanation_parts.append(
    f"Pearson correlation between student-teacher ratio and academic performance "
    f"is {pearson_corr:.3f} (p={pearson_p:.4f})"
)

# Add simple regression evidence
simple_coef = model_simple.params['student_teacher_ratio']
simple_pval = model_simple.pvalues['student_teacher_ratio']
explanation_parts.append(
    f"simple linear regression shows coefficient of {simple_coef:.3f} (p={simple_pval:.4f})"
)

# Add multiple regression evidence (most important - controls for confounders)
explanation_parts.append(
    f"multiple regression controlling for income, English learners, lunch eligibility, "
    f"and CalWorks shows coefficient of {coef_str:.3f} (p={pval_str:.4f})"
)

# Determine response score based on evidence
if pval_str < 0.001:
    # Very strong evidence
    if coef_str < 0:
        response_score = 95  # Strong yes - negative coefficient means lower ratio → higher performance
    else:
        response_score = 90  # Strong yes - but positive coefficient (unexpected)
    strength = "very strong"
elif pval_str < 0.01:
    # Strong evidence
    if coef_str < 0:
        response_score = 85
    else:
        response_score = 80
    strength = "strong"
elif pval_str < 0.05:
    # Moderate evidence
    if coef_str < 0:
        response_score = 70
    else:
        response_score = 65
    strength = "moderate"
elif pval_str < 0.10:
    # Weak evidence
    response_score = 45
    strength = "weak"
else:
    # No significant evidence
    response_score = 20
    strength = "no significant"

explanation = (
    f"There is {strength} statistical evidence of an association between student-teacher ratio "
    f"and academic performance. {'. '.join(explanation_parts)}. "
)

if is_significant and coef_str < 0:
    explanation += (
        f"The negative coefficient indicates that lower student-teacher ratios are associated "
        f"with higher academic performance, even after controlling for socioeconomic factors. "
        f"Each unit increase in student-teacher ratio is associated with a {abs(coef_str):.2f} "
        f"point decrease in academic performance."
    )
elif is_significant and coef_str > 0:
    explanation += (
        f"Interestingly, the positive coefficient suggests higher student-teacher ratios are "
        f"associated with higher performance, which may indicate confounding factors or "
        f"that other school characteristics dominate the relationship."
    )
else:
    explanation += (
        f"The relationship is not statistically significant, suggesting that student-teacher "
        f"ratio may not be a strong predictor of academic performance in this dataset, or that "
        f"the relationship is confounded by other factors."
    )

print(f"\nResponse Score: {response_score}/100")
print(f"\nExplanation: {explanation}")

# Write conclusion to file
conclusion = {
    "response": response_score,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n" + "=" * 80)
print("Conclusion written to conclusion.txt")
print("=" * 80)
