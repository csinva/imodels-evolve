import pandas as pd
import numpy as np
import json
from scipy import stats
import statsmodels.api as sm
from sklearn.tree import DecisionTreeClassifier

# Load the dataset
df = pd.read_csv('mortgage.csv')

print("=" * 80)
print("MORTGAGE APPLICATION ANALYSIS: GENDER EFFECT")
print("=" * 80)
print(f"\nDataset shape: {df.shape}")

# Check for missing values and clean
print(f"\nMissing values per column:")
print(df.isnull().sum())

df_clean = df.dropna()
print(f"\nDataset shape after cleaning: {df_clean.shape}")
df = df_clean

# Basic exploration
print("\n" + "=" * 80)
print("1. EXPLORATORY DATA ANALYSIS")
print("=" * 80)

print("\nGender distribution:")
print(df['female'].value_counts())
print(f"Female applicants: {df['female'].sum():.0f} ({df['female'].mean()*100:.1f}%)")

print("\nAcceptance rates by gender:")
acceptance_by_gender = df.groupby('female')['accept'].agg(['mean', 'sum', 'count'])
print(acceptance_by_gender)

female_acceptance = df[df['female'] == 1]['accept'].mean()
male_acceptance = df[df['female'] == 0]['accept'].mean()
print(f"\nFemale acceptance rate: {female_acceptance*100:.2f}%")
print(f"Male acceptance rate: {male_acceptance*100:.2f}%")
print(f"Difference: {(female_acceptance - male_acceptance)*100:.2f} percentage points")

# Chi-square test
print("\n" + "=" * 80)
print("2. CHI-SQUARE TEST")
print("=" * 80)
contingency_table = pd.crosstab(df['female'], df['accept'])
print(contingency_table)

chi2, p_value_chi, dof, expected = stats.chi2_contingency(contingency_table)
print(f"\nChi-square p-value: {p_value_chi:.6f}")
print(f"Significant: {'YES' if p_value_chi < 0.05 else 'NO'}")

# T-test
print("\n" + "=" * 80)
print("3. T-TEST")
print("=" * 80)
female_accepts = df[df['female'] == 1]['accept']
male_accepts = df[df['female'] == 0]['accept']
t_stat, p_value_t = stats.ttest_ind(female_accepts, male_accepts)
print(f"T-test p-value: {p_value_t:.6f}")
print(f"Significant: {'YES' if p_value_t < 0.05 else 'NO'}")

# Bivariate logistic regression
print("\n" + "=" * 80)
print("4. BIVARIATE LOGISTIC REGRESSION")
print("=" * 80)

X_simple = sm.add_constant(df['female'])
y = df['accept']
logit_simple = sm.Logit(y, X_simple).fit(disp=0)
print(f"Female coefficient p-value: {logit_simple.pvalues['female']:.6f}")
print(f"Odds Ratio: {np.exp(logit_simple.params['female']):.4f}")

# Multivariate logistic regression
print("\n" + "=" * 80)
print("5. MULTIVARIATE LOGISTIC REGRESSION")
print("=" * 80)

control_vars = ['black', 'housing_expense_ratio', 'self_employed', 'married', 
                'mortgage_credit', 'consumer_credit', 'bad_history', 'PI_ratio', 
                'loan_to_value', 'denied_PMI']

X_multi = df[['female'] + control_vars].copy()
X_multi = sm.add_constant(X_multi)
y_multi = df['accept']

logit_multi = sm.Logit(y_multi, X_multi).fit(disp=0)
print(f"Female coefficient: {logit_multi.params['female']:.4f}")
print(f"P-value: {logit_multi.pvalues['female']:.6f}")
print(f"Odds Ratio: {np.exp(logit_multi.params['female']):.4f}")
print(f"Significant: {'YES' if logit_multi.pvalues['female'] < 0.05 else 'NO'}")

# Effect size
print("\n" + "=" * 80)
print("6. EFFECT SIZE")
print("=" * 80)

p1 = female_acceptance
p2 = male_acceptance
cohens_h = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))
print(f"Cohen's h: {cohens_h:.4f} ({'negligible' if abs(cohens_h) < 0.05 else 'small' if abs(cohens_h) < 0.2 else 'medium'})")

# Decision Tree
print("\n" + "=" * 80)
print("7. DECISION TREE FEATURE IMPORTANCE")
print("=" * 80)

X_features = df[['female'] + control_vars].copy()
y_target = df['accept']

dt = DecisionTreeClassifier(max_depth=5, min_samples_leaf=50, random_state=42)
dt.fit(X_features, y_target)
feature_imp = pd.DataFrame({
    'feature': X_features.columns,
    'importance': dt.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_imp)

# Final conclusion
print("\n" + "=" * 80)
print("8. CONCLUSION")
print("=" * 80)

print(f"\nChi-square p-value: {p_value_chi:.6f}")
print(f"Multivariate logistic p-value: {logit_multi.pvalues['female']:.6f}")
print(f"Effect size (Cohen's h): {cohens_h:.4f}")

# Determine score
if p_value_chi >= 0.05 and logit_multi.pvalues['female'] >= 0.05:
    response_score = 10
    conclusion = "No statistically significant effect"
elif p_value_chi < 0.05 and logit_multi.pvalues['female'] >= 0.05:
    response_score = 35
    conclusion = "Bivariate effect but not after controlling for confounders"
else:
    response_score = 80
    conclusion = "Statistically significant effect"

# Adjust for tiny effect size
if abs(cohens_h) < 0.02:
    response_score = min(response_score, 15)
    conclusion = "No meaningful effect - virtually identical rates"

explanation = f"""Statistical tests reveal {'no statistically significant gender effect' if p_value_chi >= 0.05 else 'a statistically significant gender effect'} on mortgage approval.

Bivariate: Chi-square p={p_value_chi:.6f}, t-test p={p_value_t:.6f}
Multivariate: Logistic regression p={logit_multi.pvalues['female']:.6f} (controlling for race, credit, debt, etc.)
Effect size: Cohen's h={cohens_h:.4f} (virtually zero - approval rates differ by only {abs(p1-p2)*100:.2f} percentage points)

Female: {female_acceptance*100:.1f}% approved, Male: {male_acceptance*100:.1f}% approved

Conclusion: {conclusion}. The data shows essentially identical approval rates between genders."""

result = {"response": int(response_score), "explanation": explanation}

with open('conclusion.txt', 'w') as f:
    json.dump(result, f)

print(f"\nResponse score: {result['response']}")
print(f"\nExplanation: {result['explanation']}")
print("\n✓ conclusion.txt created")
