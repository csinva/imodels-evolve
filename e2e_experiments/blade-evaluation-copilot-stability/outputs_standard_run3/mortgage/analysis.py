import pandas as pd
import numpy as np
import json
from scipy import stats
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest

# Load the dataset
df = pd.read_csv('mortgage.csv')

print("=" * 80)
print("ANALYZING: How does gender affect whether banks approve mortgage applications?")
print("=" * 80)

# Explore the data
print("\n1. DATA OVERVIEW")
print(f"Total samples: {len(df)}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nBasic statistics:")
print(df.describe())

# Check for missing values
print(f"\nMissing values:")
print(df.isnull().sum())

# Distribution of key variables
print("\n2. GENDER DISTRIBUTION")
female_count = df['female'].sum()
male_count = len(df) - female_count
print(f"Female applicants: {female_count} ({female_count/len(df)*100:.1f}%)")
print(f"Male applicants: {male_count} ({male_count/len(df)*100:.1f}%)")

print("\n3. APPROVAL RATES BY GENDER")
# Calculate approval rates by gender
approval_by_gender = df.groupby('female')['accept'].agg(['sum', 'count', 'mean'])
approval_by_gender.index = ['Male', 'Female']
approval_by_gender.columns = ['Approved', 'Total', 'Approval_Rate']
print(approval_by_gender)

male_approval_rate = approval_by_gender.loc['Male', 'Approval_Rate']
female_approval_rate = approval_by_gender.loc['Female', 'Approval_Rate']
print(f"\nMale approval rate: {male_approval_rate*100:.2f}%")
print(f"Female approval rate: {female_approval_rate*100:.2f}%")
print(f"Difference: {(female_approval_rate - male_approval_rate)*100:.2f} percentage points")

# Statistical test: Two-proportion z-test
print("\n4. STATISTICAL SIGNIFICANCE TEST (Two-Proportion Z-Test)")
female_approved = approval_by_gender.loc['Female', 'Approved']
female_total = approval_by_gender.loc['Female', 'Total']
male_approved = approval_by_gender.loc['Male', 'Approved']
male_total = approval_by_gender.loc['Male', 'Total']

z_stat, p_value = proportions_ztest([female_approved, male_approved], 
                                     [female_total, male_total])
print(f"Z-statistic: {z_stat:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Significance at α=0.05: {'YES' if p_value < 0.05 else 'NO'}")

# Chi-square test
print("\n5. CHI-SQUARE TEST")
contingency_table = pd.crosstab(df['female'], df['accept'])
chi2, p_chi, dof, expected = stats.chi2_contingency(contingency_table)
print(f"Chi-square statistic: {chi2:.4f}")
print(f"P-value: {p_chi:.4f}")
print(f"Degrees of freedom: {dof}")

# Control for other variables using logistic regression
print("\n6. LOGISTIC REGRESSION (Controlling for Other Factors)")
# Select features for the model
features = ['female', 'black', 'housing_expense_ratio', 'self_employed', 
            'married', 'mortgage_credit', 'consumer_credit', 'bad_history', 
            'PI_ratio', 'loan_to_value', 'denied_PMI']

X = df[features].copy()
y = df['accept']

# Handle any missing values
X = X.fillna(X.mean())

# Fit logistic regression using statsmodels for better statistics
X_with_const = sm.add_constant(X)
logit_model = sm.Logit(y, X_with_const)
result = logit_model.fit(disp=0)

print("\nLogistic Regression Results Summary:")
print(result.summary())

# Extract gender coefficient
gender_coef = result.params['female']
gender_pvalue = result.pvalues['female']
gender_se = result.bse['female']
gender_ci = result.conf_int().loc['female']

print(f"\n*** GENDER EFFECT ***")
print(f"Coefficient for 'female': {gender_coef:.4f}")
print(f"Standard Error: {gender_se:.4f}")
print(f"P-value: {gender_pvalue:.4f}")
print(f"95% Confidence Interval: [{gender_ci[0]:.4f}, {gender_ci[1]:.4f}]")
print(f"Odds Ratio: {np.exp(gender_coef):.4f}")
print(f"Interpretation: Being female {'increases' if gender_coef > 0 else 'decreases'} odds of approval by {abs((np.exp(gender_coef) - 1) * 100):.2f}%")

# Compare approval rates while controlling for creditworthiness
print("\n7. COMPARING SIMILAR APPLICANTS")
# Look at applicants with similar credit profiles
median_mortgage_credit = df['mortgage_credit'].median()
median_consumer_credit = df['consumer_credit'].median()

similar_credit = df[(df['mortgage_credit'] == median_mortgage_credit) & 
                    (df['consumer_credit'] == median_consumer_credit) &
                    (df['bad_history'] == 0)]

if len(similar_credit) > 50:
    print(f"\nApplicants with median credit scores and no bad history (n={len(similar_credit)}):")
    similar_approval = similar_credit.groupby('female')['accept'].mean()
    print(f"Male approval rate: {similar_approval[0]*100:.2f}%")
    print(f"Female approval rate: {similar_approval[1]*100:.2f}%")

# Correlation analysis
print("\n8. CORRELATION ANALYSIS")
print("\nCorrelation between 'female' and 'accept':")
correlation = df['female'].corr(df['accept'])
print(f"Pearson correlation: {correlation:.4f}")

# Calculate effect size (Cohen's h for proportions)
print("\n9. EFFECT SIZE")
h = 2 * (np.arcsin(np.sqrt(female_approval_rate)) - np.arcsin(np.sqrt(male_approval_rate)))
print(f"Cohen's h: {h:.4f}")
print(f"Effect size interpretation: ", end="")
if abs(h) < 0.2:
    print("Small effect")
elif abs(h) < 0.5:
    print("Medium effect")
else:
    print("Large effect")

# CONCLUSION
print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

# Determine response based on statistical significance and effect
is_significant = p_value < 0.05
gender_coef_significant = gender_pvalue < 0.05

if gender_coef_significant:
    if gender_coef < 0:
        # Female has negative coefficient = lower approval rate
        response_score = 75  # Yes, gender affects approval (females disadvantaged)
        explanation = (
            f"Yes, gender significantly affects mortgage approval. "
            f"After controlling for creditworthiness and other factors using logistic regression, "
            f"being female is associated with a statistically significant decrease in approval odds "
            f"(coefficient: {gender_coef:.4f}, p={gender_pvalue:.4f}). "
            f"Female applicants have {abs((np.exp(gender_coef) - 1) * 100):.1f}% lower odds of approval. "
            f"The raw approval rates are {male_approval_rate*100:.1f}% for males vs {female_approval_rate*100:.1f}% for females. "
            f"This disparity remains significant even when controlling for financial factors."
        )
    else:
        # Female has positive coefficient = higher approval rate
        response_score = 70  # Yes, gender affects approval (females advantaged)
        explanation = (
            f"Yes, gender significantly affects mortgage approval. "
            f"After controlling for creditworthiness and other factors using logistic regression, "
            f"being female is associated with a statistically significant increase in approval odds "
            f"(coefficient: {gender_coef:.4f}, p={gender_pvalue:.4f}). "
            f"Female applicants have {abs((np.exp(gender_coef) - 1) * 100):.1f}% higher odds of approval. "
            f"The raw approval rates are {male_approval_rate*100:.1f}% for males vs {female_approval_rate*100:.1f}% for females."
        )
else:
    # Not significant after controlling for other factors
    response_score = 25  # No significant effect
    explanation = (
        f"No, gender does not significantly affect mortgage approval after controlling for other factors. "
        f"While raw approval rates differ slightly ({male_approval_rate*100:.1f}% for males vs {female_approval_rate*100:.1f}% for females), "
        f"logistic regression shows that after accounting for creditworthiness, financial ratios, and other factors, "
        f"the gender coefficient is not statistically significant (coefficient: {gender_coef:.4f}, p={gender_pvalue:.4f}). "
        f"This suggests the observed difference is explained by other factors rather than gender discrimination."
    )

print(f"\nResponse Score: {response_score}/100")
print(f"Explanation: {explanation}")

# Write conclusion to file
conclusion = {
    "response": response_score,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n✓ Analysis complete. Results written to conclusion.txt")
