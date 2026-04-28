import pandas as pd
import numpy as np
import json
from scipy import stats
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('soccer.csv')

print("=" * 80)
print("RESEARCH QUESTION: Are soccer players with a dark skin tone more likely")
print("than those with a light skin tone to receive red cards from referees?")
print("=" * 80)

# Data exploration
print("\n1. DATA EXPLORATION")
print(f"Total records: {len(df)}")
print(f"\nDataset shape: {df.shape}")

# Create average skin tone rating
df['skinTone'] = (df['rater1'] + df['rater2']) / 2

# Check for missing values in key variables
print(f"\nMissing values in skin tone ratings: {df['skinTone'].isna().sum()}")
print(f"Missing values in red cards: {df['redCards'].isna().sum()}")

# Filter to records with skin tone ratings
df_with_skin = df[df['skinTone'].notna()].copy()
print(f"\nRecords with skin tone data: {len(df_with_skin)}")

# Summary statistics
print("\n2. SUMMARY STATISTICS")
print(f"\nSkin tone distribution:")
print(df_with_skin['skinTone'].describe())
print(f"\nRed cards distribution:")
print(df_with_skin['redCards'].describe())
print(f"Total red cards given: {df_with_skin['redCards'].sum()}")
print(f"Player-referee dyads with at least one red card: {(df_with_skin['redCards'] > 0).sum()}")

# Categorize skin tone into light (0-0.25), medium (0.25-0.75), and dark (0.75-1.0)
df_with_skin['skinToneCategory'] = pd.cut(df_with_skin['skinTone'], 
                                            bins=[0, 0.375, 0.625, 1.0],
                                            labels=['Light', 'Medium', 'Dark'],
                                            include_lowest=True)

print(f"\nSkin tone categories:")
print(df_with_skin['skinToneCategory'].value_counts())

# Calculate red card rates by skin tone category
print("\n3. RED CARD RATES BY SKIN TONE")
red_card_rates = df_with_skin.groupby('skinToneCategory').agg({
    'redCards': ['sum', 'mean', 'count']
}).round(4)
print(red_card_rates)

# Calculate rate per 1000 dyads for clarity
for category in ['Light', 'Medium', 'Dark']:
    cat_data = df_with_skin[df_with_skin['skinToneCategory'] == category]
    red_cards = cat_data['redCards'].sum()
    total_dyads = len(cat_data)
    rate_per_1000 = (red_cards / total_dyads) * 1000
    print(f"\n{category} skin tone: {red_cards} red cards in {total_dyads} dyads = {rate_per_1000:.2f} per 1000 dyads")

# 4. STATISTICAL TESTS
print("\n4. STATISTICAL ANALYSIS")

# Test 1: Correlation between skin tone (continuous) and red cards
correlation, p_value_corr = stats.spearmanr(df_with_skin['skinTone'], df_with_skin['redCards'])
print(f"\nSpearman correlation between skin tone and red cards: {correlation:.4f}")
print(f"P-value: {p_value_corr:.6f}")

# Test 2: Compare light vs dark skin tone groups
# Define light (0-0.5) vs dark (0.5-1.0)
df_with_skin['skinToneBinary'] = (df_with_skin['skinTone'] >= 0.5).astype(int)  # 0 = light, 1 = dark
light_skin = df_with_skin[df_with_skin['skinToneBinary'] == 0]
dark_skin = df_with_skin[df_with_skin['skinToneBinary'] == 1]

print(f"\nLight skin group (skinTone < 0.5): {len(light_skin)} dyads")
print(f"Dark skin group (skinTone >= 0.5): {len(dark_skin)} dyads")

# Mann-Whitney U test (non-parametric, appropriate for count data)
stat, p_value_mw = stats.mannwhitneyu(light_skin['redCards'], dark_skin['redCards'], alternative='two-sided')
print(f"\nMann-Whitney U test comparing light vs dark skin:")
print(f"U-statistic: {stat:.2f}, P-value: {p_value_mw:.6f}")

# Compare means
light_mean = light_skin['redCards'].mean()
dark_mean = dark_skin['redCards'].mean()
print(f"\nMean red cards (light skin): {light_mean:.6f}")
print(f"Mean red cards (dark skin): {dark_mean:.6f}")
print(f"Difference: {dark_mean - light_mean:.6f} ({((dark_mean - light_mean) / light_mean * 100):.2f}% increase)")

# Test 3: Logistic regression controlling for confounders
print("\n5. REGRESSION ANALYSIS (CONTROLLING FOR CONFOUNDERS)")

# Prepare data for regression - simpler version without position
df_reg = df_with_skin[['skinTone', 'redCards', 'games', 'yellowCards']].copy()
df_reg = df_reg.dropna()

# Create binary outcome: received at least one red card
df_reg['hasRedCard'] = (df_reg['redCards'] > 0).astype(int)

# Standardize continuous predictors
scaler = StandardScaler()
continuous_vars = ['skinTone', 'games', 'yellowCards']
df_reg[continuous_vars] = scaler.fit_transform(df_reg[continuous_vars])

# Logistic regression with statsmodels for p-values
X = df_reg[['skinTone', 'games', 'yellowCards']].values
y = df_reg['hasRedCard'].values

X_with_const = sm.add_constant(X)
logit_model = sm.Logit(y, X_with_const)
result = logit_model.fit(disp=0)

print("\nLogistic Regression Results (DV: Received at least one red card)")
print(f"Skin tone coefficient: {result.params[1]:.4f}")
print(f"P-value: {result.pvalues[1]:.6f}")
print(f"Odds ratio: {np.exp(result.params[1]):.4f}")
print(f"\nInterpretation: Each 1 SD increase in skin tone (darker) is associated with")
print(f"{(np.exp(result.params[1]) - 1) * 100:.2f}% change in odds of receiving a red card")

# Linear regression for count of red cards
print("\n\nLinear Regression Results (DV: Number of red cards)")
X_lin = df_reg[['skinTone', 'games', 'yellowCards']].values
X_lin = sm.add_constant(X_lin)
y_lin = df_reg['redCards'].values
ols_model = sm.OLS(y_lin, X_lin)
ols_result = ols_model.fit()

print(f"Skin tone coefficient: {ols_result.params[1]:.6f}")
print(f"P-value: {ols_result.pvalues[1]:.6f}")

# 6. INTERPRETABLE MODEL WITH IMODELS
print("\n6. INTERPRETABLE MODEL (Using Decision Tree)")
from sklearn.tree import DecisionTreeClassifier

# Simple decision tree for interpretability
X_simple = df_reg[['skinTone', 'games', 'yellowCards']]
y_simple = df_reg['hasRedCard']

tree_model = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_model.fit(X_simple, y_simple)

feature_importance = pd.DataFrame({
    'feature': X_simple.columns,
    'importance': tree_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nDecision Tree Feature Importances:")
print(feature_importance)

# CONCLUSION
print("\n" + "=" * 80)
print("7. CONCLUSION")
print("=" * 80)

# Determine response based on statistical evidence
# We have multiple lines of evidence:
# 1. Correlation between skin tone and red cards
# 2. Comparison of light vs dark skin groups
# 3. Regression analysis controlling for confounders

significant_correlation = p_value_corr < 0.05
significant_comparison = p_value_mw < 0.05
significant_regression = result.pvalues[1] < 0.05
positive_effect = correlation > 0 and dark_mean > light_mean

print(f"\nStatistical Evidence:")
print(f"- Correlation significant: {significant_correlation} (p={p_value_corr:.6f})")
print(f"- Group comparison significant: {significant_comparison} (p={p_value_mw:.6f})")
print(f"- Regression significant: {significant_regression} (p={result.pvalues[1]:.6f})")
print(f"- Effect direction positive: {positive_effect}")

# Calculate response score
if significant_correlation and significant_comparison and significant_regression and positive_effect:
    response = 90  # Strong Yes
    explanation = f"Strong statistical evidence shows that players with darker skin tones receive more red cards. Spearman correlation r={correlation:.4f} (p={p_value_corr:.6f}), Mann-Whitney U test p={p_value_mw:.6f}, and logistic regression controlling for games and yellow cards shows skin tone coefficient β={result.params[1]:.4f} (p={result.pvalues[1]:.6f}). Players with dark skin (≥0.5) received {dark_mean:.6f} red cards per dyad vs {light_mean:.6f} for light skin, a {((dark_mean - light_mean) / light_mean * 100):.2f}% increase."
elif (significant_correlation or significant_comparison or significant_regression) and positive_effect:
    response = 70  # Moderate Yes
    explanation = f"Moderate statistical evidence suggests that players with darker skin tones receive more red cards. Some tests show significance: correlation p={p_value_corr:.6f}, comparison p={p_value_mw:.6f}, regression p={result.pvalues[1]:.6f}. Dark skin players show {((dark_mean - light_mean) / light_mean * 100):.2f}% more red cards."
elif positive_effect and not (significant_correlation and significant_comparison):
    response = 40  # Weak evidence
    explanation = f"Weak evidence for the relationship. While dark skin players show {((dark_mean - light_mean) / light_mean * 100):.2f}% more red cards on average, the statistical tests are not consistently significant (correlation p={p_value_corr:.6f}, comparison p={p_value_mw:.6f}), suggesting the relationship may not be robust."
else:
    response = 20  # No strong evidence
    explanation = f"No strong statistical evidence that players with darker skin tones receive more red cards. Statistical tests did not show significant results (correlation p={p_value_corr:.6f}, comparison p={p_value_mw:.6f}, regression p={result.pvalues[1]:.6f})."

print(f"\nFinal Response Score: {response}/100")
print(f"Explanation: {explanation}")

# Write conclusion to file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n" + "=" * 80)
print("Analysis complete. Results written to conclusion.txt")
print("=" * 80)
