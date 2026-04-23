import pandas as pd
import numpy as np
import json
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('caschools.csv')

# Calculate student-teacher ratio
df['student_teacher_ratio'] = df['students'] / df['teachers']

# Create average test score as overall academic performance measure
df['avg_score'] = (df['read'] + df['math']) / 2

print("="*80)
print("ANALYZING: Is a lower student-teacher ratio associated with higher academic performance?")
print("="*80)

print("\n1. DESCRIPTIVE STATISTICS")
print("-"*80)
print(f"Student-Teacher Ratio: Mean={df['student_teacher_ratio'].mean():.2f}, Std={df['student_teacher_ratio'].std():.2f}")
print(f"Average Test Score: Mean={df['avg_score'].mean():.2f}, Std={df['avg_score'].std():.2f}")
print(f"Reading Score: Mean={df['read'].mean():.2f}, Std={df['read'].std():.2f}")
print(f"Math Score: Mean={df['math'].mean():.2f}, Std={df['math'].std():.2f}")

print("\n2. CORRELATION ANALYSIS")
print("-"*80)
# Correlation between student-teacher ratio and academic performance
corr_avg = df['student_teacher_ratio'].corr(df['avg_score'])
corr_read = df['student_teacher_ratio'].corr(df['read'])
corr_math = df['student_teacher_ratio'].corr(df['math'])

print(f"Correlation (Student-Teacher Ratio vs Average Score): {corr_avg:.4f}")
print(f"Correlation (Student-Teacher Ratio vs Reading): {corr_read:.4f}")
print(f"Correlation (Student-Teacher Ratio vs Math): {corr_math:.4f}")

# Pearson correlation test
r_avg, p_avg = stats.pearsonr(df['student_teacher_ratio'], df['avg_score'])
r_read, p_read = stats.pearsonr(df['student_teacher_ratio'], df['read'])
r_math, p_math = stats.pearsonr(df['student_teacher_ratio'], df['math'])

print(f"\nPearson test (Avg Score): r={r_avg:.4f}, p-value={p_avg:.4e}")
print(f"Pearson test (Reading): r={r_read:.4f}, p-value={p_read:.4e}")
print(f"Pearson test (Math): r={r_math:.4f}, p-value={p_math:.4e}")

print("\n3. LINEAR REGRESSION ANALYSIS (Simple)")
print("-"*80)
# Simple linear regression: avg_score ~ student_teacher_ratio
X = df[['student_teacher_ratio']].values
y = df['avg_score'].values

model = LinearRegression()
model.fit(X, y)
r2 = model.score(X, y)

print(f"Simple Regression: R² = {r2:.4f}")
print(f"Coefficient: {model.coef_[0]:.4f} (points change per unit increase in ratio)")
print(f"Intercept: {model.intercept_:.2f}")
print(f"Interpretation: Each 1-unit increase in student-teacher ratio is associated with {model.coef_[0]:.2f} point change in test scores")

print("\n4. STATISTICAL REGRESSION WITH P-VALUES")
print("-"*80)
# Using statsmodels for p-values
X_sm = sm.add_constant(df['student_teacher_ratio'])
model_sm = sm.OLS(df['avg_score'], X_sm).fit()
print(model_sm.summary().tables[1])

print("\n5. MULTIPLE REGRESSION (Controlling for confounders)")
print("-"*80)
# Control for socioeconomic factors
control_vars = ['student_teacher_ratio', 'income', 'english', 'lunch', 'calworks']
df_clean = df[control_vars + ['avg_score']].dropna()

X_multi = sm.add_constant(df_clean[control_vars])
model_multi = sm.OLS(df_clean['avg_score'], X_multi).fit()
print("Controlling for: income, english learners %, lunch %, calworks %")
print(model_multi.summary().tables[1])

print("\n6. COMPARING HIGH VS LOW STUDENT-TEACHER RATIO GROUPS")
print("-"*80)
# Split into quartiles
q1 = df['student_teacher_ratio'].quantile(0.25)
q3 = df['student_teacher_ratio'].quantile(0.75)

low_ratio = df[df['student_teacher_ratio'] <= q1]['avg_score']
high_ratio = df[df['student_teacher_ratio'] >= q3]['avg_score']

print(f"Low ratio group (≤{q1:.2f}): Mean score = {low_ratio.mean():.2f}, N = {len(low_ratio)}")
print(f"High ratio group (≥{q3:.2f}): Mean score = {high_ratio.mean():.2f}, N = {len(high_ratio)}")
print(f"Difference: {low_ratio.mean() - high_ratio.mean():.2f} points")

# T-test
t_stat, p_ttest = stats.ttest_ind(low_ratio, high_ratio)
print(f"\nT-test: t={t_stat:.4f}, p-value={p_ttest:.4e}")

print("\n7. INTERPRETING RESULTS")
print("-"*80)

# Determine conclusion based on statistical evidence
significant = p_avg < 0.05
negative_correlation = corr_avg < 0

print(f"Statistical Significance: {'YES' if significant else 'NO'} (p={p_avg:.4e})")
print(f"Direction of Relationship: {'NEGATIVE' if negative_correlation else 'POSITIVE'} (r={corr_avg:.4f})")
print(f"Effect Size: {'SMALL' if abs(corr_avg) < 0.3 else 'MODERATE' if abs(corr_avg) < 0.5 else 'LARGE'}")

# Summary
if significant and negative_correlation:
    conclusion = "YES - There is a statistically significant NEGATIVE relationship between student-teacher ratio and academic performance."
    conclusion += f"\nLower ratios are associated with HIGHER test scores (r={corr_avg:.3f}, p<0.001)."
    response_score = 85  # Strong yes
elif significant and not negative_correlation:
    conclusion = "OPPOSITE DIRECTION - Higher student-teacher ratios are associated with higher scores (unexpected)."
    response_score = 20  # This would be against the hypothesis
elif not significant:
    conclusion = "NO - No statistically significant relationship found."
    response_score = 15  # No evidence
else:
    conclusion = "INCONCLUSIVE"
    response_score = 50

print(f"\nCONCLUSION: {conclusion}")
print(f"RESPONSE SCORE: {response_score}/100")

# Additional context from multiple regression
coef_str = model_multi.params['student_teacher_ratio']
pval_str = model_multi.pvalues['student_teacher_ratio']
print(f"\nEven after controlling for socioeconomic factors:")
print(f"Student-Teacher Ratio coefficient: {coef_str:.4f}, p={pval_str:.4e}")
print(f"Significant after controls: {'YES' if pval_str < 0.05 else 'NO'}")

# Final determination
if pval_str < 0.05 and coef_str < 0:
    explanation = f"Strong evidence: Lower student-teacher ratios are significantly associated with higher academic performance. Simple correlation r={corr_avg:.3f} (p<0.001). Even controlling for income, English learners, and other socioeconomic factors, the relationship remains significant (coef={coef_str:.2f}, p={pval_str:.4f}). Schools with low ratios (≤{q1:.2f}) score {low_ratio.mean() - high_ratio.mean():.1f} points higher than high ratio schools (≥{q3:.2f})."
    response_score = 85
elif pval_str < 0.05:
    if coef_str > 0:
        explanation = f"Significant but opposite direction: Higher student-teacher ratios are associated with higher scores (coef={coef_str:.2f}, p={pval_str:.4f}). This contradicts the hypothesis. The relationship may be confounded by unmeasured factors."
        response_score = 20
    else:
        explanation = f"Significant negative relationship found: Lower ratios associated with higher scores (coef={coef_str:.2f}, p={pval_str:.4f})."
        response_score = 80
else:
    explanation = f"No significant relationship found. While simple correlation is r={corr_avg:.3f} (p={p_avg:.4f}), after controlling for socioeconomic factors, the effect is not significant (p={pval_str:.3f}). The apparent relationship may be due to confounding variables like income and English learner percentage."
    response_score = 25

print("\n" + "="*80)
print("FINAL ANSWER")
print("="*80)
print(f"Response Score: {response_score}")
print(f"Explanation: {explanation}")

# Write conclusion to file
result = {
    "response": response_score,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(result, f)

print("\n✓ conclusion.txt has been created")
