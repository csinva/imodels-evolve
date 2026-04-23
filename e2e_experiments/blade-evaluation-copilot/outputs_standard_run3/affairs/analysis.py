import pandas as pd
import numpy as np
import json
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
from imodels import RuleFitRegressor, FIGSRegressor

# Load the data
df = pd.read_csv('affairs.csv')

print("=" * 80)
print("RESEARCH QUESTION: Does having children decrease engagement in extramarital affairs?")
print("=" * 80)

# Data exploration
print("\nDataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nSummary statistics:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

# Focus on the key variables: children (predictor) and affairs (outcome)
print("\n" + "=" * 80)
print("KEY VARIABLE ANALYSIS: Children vs Affairs")
print("=" * 80)

# Children distribution
print("\nDistribution of 'children':")
print(df['children'].value_counts())

# Affairs distribution
print("\nDistribution of 'affairs':")
print(df['affairs'].value_counts().sort_index())

# Group by children and analyze affairs
print("\n" + "=" * 80)
print("AFFAIRS BY CHILDREN STATUS")
print("=" * 80)
affairs_by_children = df.groupby('children')['affairs'].agg(['mean', 'median', 'std', 'count'])
print(affairs_by_children)

# Descriptive statistics
mean_affairs_with_children = df[df['children'] == 'yes']['affairs'].mean()
mean_affairs_no_children = df[df['children'] == 'no']['affairs'].mean()

print(f"\nMean affairs WITH children: {mean_affairs_with_children:.3f}")
print(f"Mean affairs WITHOUT children: {mean_affairs_no_children:.3f}")
print(f"Difference: {mean_affairs_no_children - mean_affairs_with_children:.3f}")

# Statistical test: Independent samples t-test
print("\n" + "=" * 80)
print("STATISTICAL TEST: Independent T-Test")
print("=" * 80)

affairs_with_children = df[df['children'] == 'yes']['affairs']
affairs_no_children = df[df['children'] == 'no']['affairs']

t_stat, p_value = stats.ttest_ind(affairs_no_children, affairs_with_children)
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Significance level: 0.05")
print(f"Significant at 0.05 level: {p_value < 0.05}")

# Mann-Whitney U test (non-parametric alternative)
print("\n" + "=" * 80)
print("STATISTICAL TEST: Mann-Whitney U Test (non-parametric)")
print("=" * 80)
u_stat, p_value_mw = stats.mannwhitneyu(affairs_no_children, affairs_with_children, alternative='two-sided')
print(f"U-statistic: {u_stat:.4f}")
print(f"P-value: {p_value_mw:.4f}")
print(f"Significant at 0.05 level: {p_value_mw < 0.05}")

# Linear regression with statsmodels for detailed statistics
print("\n" + "=" * 80)
print("LINEAR REGRESSION: Affairs ~ Children (with p-values)")
print("=" * 80)

# Encode children as binary
df['children_encoded'] = (df['children'] == 'yes').astype(int)

X = df[['children_encoded']]
y = df['affairs']
X_with_const = sm.add_constant(X)

model_ols = sm.OLS(y, X_with_const).fit()
print(model_ols.summary())

# Multiple regression controlling for confounders
print("\n" + "=" * 80)
print("MULTIPLE REGRESSION: Controlling for confounders")
print("=" * 80)

# Encode categorical variables
df['gender_encoded'] = (df['gender'] == 'male').astype(int)

# Select features for multiple regression
features = ['children_encoded', 'age', 'yearsmarried', 'gender_encoded', 
            'religiousness', 'education', 'occupation', 'rating']
X_multi = df[features]
y_multi = df['affairs']
X_multi_with_const = sm.add_constant(X_multi)

model_multi = sm.OLS(y_multi, X_multi_with_const).fit()
print(model_multi.summary())

# Interpretable model using imodels
print("\n" + "=" * 80)
print("INTERPRETABLE MODEL: RuleFit")
print("=" * 80)

try:
    rulefit = RuleFitRegressor(max_rules=10, random_state=42)
    rulefit.fit(X_multi, y_multi)
    
    print("Feature importances:")
    for i, feature in enumerate(features):
        print(f"  {feature}: {rulefit.feature_importances_[i]:.4f}")
    
    # Check coefficient for children
    children_idx = features.index('children_encoded')
    print(f"\nChildren coefficient from RuleFit: {rulefit.feature_importances_[children_idx]:.4f}")
except Exception as e:
    print(f"RuleFit error: {e}")

# Effect size: Cohen's d
print("\n" + "=" * 80)
print("EFFECT SIZE: Cohen's d")
print("=" * 80)

mean_diff = mean_affairs_no_children - mean_affairs_with_children
pooled_std = np.sqrt((affairs_no_children.var() + affairs_with_children.var()) / 2)
cohens_d = mean_diff / pooled_std
print(f"Cohen's d: {cohens_d:.4f}")
print(f"Interpretation: ", end="")
if abs(cohens_d) < 0.2:
    print("Negligible effect")
elif abs(cohens_d) < 0.5:
    print("Small effect")
elif abs(cohens_d) < 0.8:
    print("Medium effect")
else:
    print("Large effect")

# Correlation analysis
print("\n" + "=" * 80)
print("CORRELATION ANALYSIS")
print("=" * 80)

corr_children_affairs = df['children_encoded'].corr(df['affairs'])
print(f"Pearson correlation (children vs affairs): {corr_children_affairs:.4f}")

# Point-biserial correlation (same as Pearson for binary predictor)
r_pb, p_pb = stats.pointbiserialr(df['children_encoded'], df['affairs'])
print(f"Point-biserial correlation: {r_pb:.4f}")
print(f"P-value: {p_pb:.4f}")

# Final conclusion
print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

# Determine conclusion based on statistical evidence
# Key metrics:
# 1. Mean difference
# 2. P-value from t-test
# 3. Effect size
# 4. Regression coefficient

children_coef = model_multi.params['children_encoded']
children_pval = model_multi.pvalues['children_encoded']

print(f"\nSummary of Evidence:")
print(f"1. Mean affairs WITHOUT children: {mean_affairs_no_children:.3f}")
print(f"2. Mean affairs WITH children: {mean_affairs_with_children:.3f}")
print(f"3. Mean difference: {mean_diff:.3f} (positive = more affairs without children)")
print(f"4. T-test p-value: {p_value:.4f}")
print(f"5. Multiple regression coefficient: {children_coef:.4f} (controlling for confounders)")
print(f"6. Multiple regression p-value: {children_pval:.4f}")
print(f"7. Effect size (Cohen's d): {cohens_d:.4f}")

# Decision logic
if children_pval < 0.05:
    # Statistically significant relationship
    if children_coef < 0:
        # Negative coefficient means having children decreases affairs
        # Effect size determines strength of conclusion
        if abs(cohens_d) < 0.2:
            response = 60  # Weak but significant effect
            explanation = f"Having children is statistically significantly associated with fewer extramarital affairs (p={children_pval:.4f}, coef={children_coef:.3f}), but the effect size is small (Cohen's d={cohens_d:.3f}). The relationship exists but is weak."
        elif abs(cohens_d) < 0.5:
            response = 75  # Moderate effect
            explanation = f"Having children significantly decreases engagement in extramarital affairs (p={children_pval:.4f}, coef={children_coef:.3f}). The effect size is small to moderate (Cohen's d={cohens_d:.3f}), with those having children engaging in {mean_affairs_with_children:.2f} vs {mean_affairs_no_children:.2f} affairs on average."
        else:
            response = 90  # Strong effect
            explanation = f"Having children substantially decreases engagement in extramarital affairs (p={children_pval:.4f}, coef={children_coef:.3f}). The effect size is substantial (Cohen's d={cohens_d:.3f}), with clear differences between groups."
    else:
        # Positive coefficient means having children increases affairs
        response = 10  # Opposite of expected
        explanation = f"The data shows an unexpected result: having children is associated with slightly MORE extramarital affairs (p={children_pval:.4f}, coef={children_coef:.3f}), contradicting the hypothesis that children decrease affairs."
else:
    # Not statistically significant
    if abs(mean_diff) < 0.2:
        response = 5  # No meaningful relationship
        explanation = f"No significant relationship found between having children and extramarital affairs (p={children_pval:.4f}). The difference in means is negligible ({mean_diff:.3f}), suggesting children do not meaningfully affect affair engagement."
    else:
        response = 30  # Trend but not significant
        explanation = f"Although there is a numerical difference in mean affairs between those with and without children ({mean_diff:.3f}), the relationship is not statistically significant (p={children_pval:.4f}). We cannot confidently conclude that children decrease affairs."

print(f"\nFinal Response: {response}/100")
print(f"Explanation: {explanation}")

# Write conclusion to file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\nConclusion written to conclusion.txt")
