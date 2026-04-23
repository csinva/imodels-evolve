import pandas as pd
import numpy as np
import json
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('soccer.csv')

print("=" * 80)
print("SOCCER PLAYER RED CARDS AND SKIN TONE ANALYSIS")
print("=" * 80)
print(f"\nDataset shape: {df.shape}")

# Create average skin tone rating
df['skinTone'] = (df['rater1'] + df['rater2']) / 2

# Remove rows with missing skin tone data
df_analysis = df[df['skinTone'].notna()].copy()
print(f"Rows with skin tone data: {len(df_analysis)}")

# ============================================================================
# EXPLORATORY ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("1. EXPLORATORY DATA ANALYSIS")
print("=" * 80)

print("\nSkin Tone Distribution:")
print(df_analysis['skinTone'].describe())

print("\nRed Cards Distribution:")
print(df_analysis['redCards'].describe())
print(f"Total red cards: {df_analysis['redCards'].sum()}")

# Red card rate by skin tone category
df_analysis['skinTone_category'] = pd.cut(df_analysis['skinTone'], 
                                            bins=[0, 0.25, 0.5, 0.75, 1.0],
                                            labels=['Very Light', 'Light', 'Dark', 'Very Dark'])

print("\nRed Cards by Skin Tone Category:")
skin_summary = df_analysis.groupby('skinTone_category').agg({
    'redCards': ['sum', 'mean', 'count']
}).round(4)
print(skin_summary)

# Calculate red card rates per game
df_analysis['redCard_rate'] = df_analysis['redCards'] / df_analysis['games']

print("\nRed Card Rate (per game) by Skin Tone Category:")
rate_summary = df_analysis.groupby('skinTone_category')['redCard_rate'].agg(['mean', 'std', 'count']).round(6)
print(rate_summary)

# ============================================================================
# STATISTICAL TESTS
# ============================================================================
print("\n" + "=" * 80)
print("2. STATISTICAL TESTS")
print("=" * 80)

# Test 1: Correlation between skin tone and red cards
corr_coef, corr_pval = stats.spearmanr(df_analysis['skinTone'], df_analysis['redCards'])
print(f"\nSpearman Correlation (skinTone vs redCards):")
print(f"  Coefficient: {corr_coef:.4f}")
print(f"  P-value: {corr_pval:.6f}")

# Test 2: Compare light vs dark skin (median split)
median_skin = df_analysis['skinTone'].median()
light_skin = df_analysis[df_analysis['skinTone'] <= median_skin]['redCards']
dark_skin = df_analysis[df_analysis['skinTone'] > median_skin]['redCards']

u_stat, u_pval = stats.mannwhitneyu(dark_skin, light_skin, alternative='greater')
print(f"\nMann-Whitney U Test (Dark vs Light Skin):")
print(f"  Light skin: mean = {light_skin.mean():.4f}, n = {len(light_skin)}")
print(f"  Dark skin: mean = {dark_skin.mean():.4f}, n = {len(dark_skin)}")
print(f"  P-value (one-tailed): {u_pval:.6f}")

# Test 3: Poisson regression (appropriate for count outcomes)
print("\n" + "-" * 80)
print("Poisson Regression: Red Cards ~ Skin Tone")
print("-" * 80)

poisson_data = df_analysis[['redCards', 'skinTone', 'games']].copy()
X_poisson = sm.add_constant(poisson_data[['skinTone']].values)
y_poisson = poisson_data['redCards'].values

poisson_model = sm.GLM(y_poisson, X_poisson, 
                       family=sm.families.Poisson(),
                       offset=np.log(poisson_data['games'].values)).fit()

print(f"\nSkin Tone Coefficient: {poisson_model.params[1]:.4f}")
print(f"Skin Tone P-value: {poisson_model.pvalues[1]:.6f}")
print(f"Exp(coefficient) = {np.exp(poisson_model.params[1]):.4f}")
print("  (Rate ratio: increase in red card rate per unit skin tone)")

# Test 4: Multivariate Poisson with controls
print("\n" + "-" * 80)
print("Multivariate Poisson Regression (with controls)")
print("-" * 80)

# Prepare data with numeric controls only
controls_data = df_analysis[['skinTone', 'height', 'weight', 'yellowCards', 
                              'redCards', 'games']].copy()
controls_data = controls_data.dropna()

X_multi = sm.add_constant(controls_data[['skinTone', 'height', 'weight', 'yellowCards']].values)
y_multi = controls_data['redCards'].values

multi_model = sm.GLM(y_multi, X_multi, 
                     family=sm.families.Poisson(),
                     offset=np.log(controls_data['games'].values)).fit()

print(f"\nSkin Tone Coefficient (adjusted): {multi_model.params[1]:.4f}")
print(f"Skin Tone P-value (adjusted): {multi_model.pvalues[1]:.6f}")
print(f"Exp(coefficient) = {np.exp(multi_model.params[1]):.4f}")

# ============================================================================
# INTERPRETABLE MODELS
# ============================================================================
print("\n" + "=" * 80)
print("3. INTERPRETABLE MODELS")
print("=" * 80)

# Linear regression for red card rate
print("\nLinear Regression: Red Card Rate ~ Skin Tone")
X_linear = df_analysis[['skinTone']].values
y_rate = df_analysis['redCard_rate'].values

lr_model = LinearRegression()
lr_model.fit(X_linear, y_rate)

print(f"  Coefficient: {lr_model.coef_[0]:.6f}")
print(f"  Intercept: {lr_model.intercept_:.6f}")
print(f"  R²: {lr_model.score(X_linear, y_rate):.6f}")

# Statistical significance test
X_lr_sm = sm.add_constant(X_linear)
lr_sm = sm.OLS(y_rate, X_lr_sm).fit()
print(f"  P-value for skinTone: {lr_sm.pvalues[1]:.6f}")

# ============================================================================
# CONCLUSION
# ============================================================================
print("\n" + "=" * 80)
print("4. CONCLUSION")
print("=" * 80)

print("\nEvidence Summary:")
print(f"1. Spearman correlation: r = {corr_coef:.4f}, p = {corr_pval:.6f}")
print(f"2. Mann-Whitney U: p = {u_pval:.6f}")
print(f"3. Poisson (unadjusted): coef = {poisson_model.params[1]:.4f}, p = {poisson_model.pvalues[1]:.6f}")
print(f"4. Poisson (adjusted): coef = {multi_model.params[1]:.4f}, p = {multi_model.pvalues[1]:.6f}")
print(f"5. Linear regression: coef = {lr_model.coef_[0]:.6f}, p = {lr_sm.pvalues[1]:.6f}")

# Determine response based on statistical significance
significance_level = 0.05

significant_tests = sum([
    corr_pval < significance_level,
    u_pval < significance_level,
    poisson_model.pvalues[1] < significance_level,
    multi_model.pvalues[1] < significance_level,
    lr_sm.pvalues[1] < significance_level
])

print(f"\nSignificant tests (p < {significance_level}): {significant_tests}/5")

# Determine score and explanation
if significant_tests >= 4:
    if poisson_model.pvalues[1] < 0.001:
        response_score = 90
        explanation = f"Strong evidence: All statistical tests show significant positive relationships (p < 0.001) between darker skin tone and red cards. Poisson regression rate ratio = {np.exp(poisson_model.params[1]):.2f}, meaning players with maximum skin tone have {np.exp(poisson_model.params[1]):.2f}x the red card rate."
    else:
        response_score = 80
        explanation = f"Strong evidence: Multiple tests show significant relationships (p < 0.05) between darker skin tone and red cards. Poisson regression rate ratio = {np.exp(poisson_model.params[1]):.2f}."
elif significant_tests >= 3:
    response_score = 70
    explanation = f"Moderate evidence: Majority of tests show significant relationships. Poisson regression coef = {poisson_model.params[1]:.3f}, p = {poisson_model.pvalues[1]:.4f}."
elif significant_tests >= 2:
    response_score = 60
    explanation = f"Some evidence: Some tests show significant relationships, but results are mixed."
elif significant_tests >= 1:
    response_score = 40
    explanation = f"Weak evidence: Only one test shows significance."
else:
    response_score = 10
    explanation = f"No evidence: No tests show significant relationships (all p > {significance_level})."

print(f"\nFinal Score: {response_score}/100")
print(f"Explanation: {explanation}")

# Write conclusion
conclusion = {
    "response": response_score,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n" + "=" * 80)
print("Analysis complete. Results written to conclusion.txt")
print("=" * 80)
