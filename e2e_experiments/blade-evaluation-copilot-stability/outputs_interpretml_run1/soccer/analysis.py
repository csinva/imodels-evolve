import pandas as pd
import numpy as np
import json
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('soccer.csv')

print("="*80)
print("ANALYZING: Are soccer players with dark skin tone more likely to receive red cards?")
print("="*80)
print(f"\nDataset shape: {df.shape}")
print(f"Total observations: {len(df)}")

# Calculate average skin tone from rater1 and rater2
df['skin_tone'] = df[['rater1', 'rater2']].mean(axis=1)

# Remove rows with missing skin tone data
df_with_skin = df.dropna(subset=['skin_tone'])
print(f"Observations with skin tone data: {len(df_with_skin)}")

# Create binary variables for analysis
# Dark skin: skin_tone > 0.5 (above midpoint)
# Light skin: skin_tone < 0.5 (below midpoint)
df_with_skin['dark_skin'] = (df_with_skin['skin_tone'] > 0.5).astype(int)
df_with_skin['light_skin'] = (df_with_skin['skin_tone'] < 0.5).astype(int)

# Calculate red card rates
dark_skin_players = df_with_skin[df_with_skin['dark_skin'] == 1]
light_skin_players = df_with_skin[df_with_skin['light_skin'] == 1]

print("\n" + "="*80)
print("DESCRIPTIVE STATISTICS")
print("="*80)

# Red cards summary
print(f"\nDark skin players (n={len(dark_skin_players)}):")
print(f"  Total red cards: {dark_skin_players['redCards'].sum()}")
print(f"  Red card rate per dyad: {dark_skin_players['redCards'].mean():.6f}")
print(f"  Players with at least 1 red card: {(dark_skin_players['redCards'] > 0).sum()}")

print(f"\nLight skin players (n={len(light_skin_players)}):")
print(f"  Total red cards: {light_skin_players['redCards'].sum()}")
print(f"  Red card rate per dyad: {light_skin_players['redCards'].mean():.6f}")
print(f"  Players with at least 1 red card: {(light_skin_players['redCards'] > 0).sum()}")

# Distribution of skin tone
print(f"\nSkin tone distribution:")
print(f"  Mean: {df_with_skin['skin_tone'].mean():.3f}")
print(f"  Std: {df_with_skin['skin_tone'].std():.3f}")
print(f"  Min: {df_with_skin['skin_tone'].min():.3f}")
print(f"  Max: {df_with_skin['skin_tone'].max():.3f}")

print("\n" + "="*80)
print("STATISTICAL TESTS")
print("="*80)

# Test 1: T-test comparing red card rates between dark and light skin players
t_stat, p_value_ttest = stats.ttest_ind(
    dark_skin_players['redCards'],
    light_skin_players['redCards'],
    equal_var=False
)

print(f"\n1. Independent t-test (dark vs light skin):")
print(f"   t-statistic: {t_stat:.4f}")
print(f"   p-value: {p_value_ttest:.6f}")
print(f"   Significant at α=0.05: {'YES' if p_value_ttest < 0.05 else 'NO'}")

# Test 2: Mann-Whitney U test (non-parametric alternative)
u_stat, p_value_mann = stats.mannwhitneyu(
    dark_skin_players['redCards'],
    light_skin_players['redCards'],
    alternative='two-sided'
)

print(f"\n2. Mann-Whitney U test (non-parametric):")
print(f"   U-statistic: {u_stat:.4f}")
print(f"   p-value: {p_value_mann:.6f}")
print(f"   Significant at α=0.05: {'YES' if p_value_mann < 0.05 else 'NO'}")

# Test 3: Logistic Regression with statsmodels for detailed statistics
# Predict red card occurrence (binary: 0 or 1+)
df_with_skin['has_red_card'] = (df_with_skin['redCards'] > 0).astype(int)

# Simple model: skin_tone predicting red cards
X_simple = sm.add_constant(df_with_skin['skin_tone'])
y = df_with_skin['has_red_card']
logit_model_simple = sm.Logit(y, X_simple).fit(disp=0)

print(f"\n3. Logistic Regression (simple: skin_tone → red card):")
print(logit_model_simple.summary2().tables[1])

# Test 4: Multiple logistic regression controlling for confounders
# Control for: games (exposure), position, league country
print("\n4. Multiple Logistic Regression (controlling for confounders):")

# Prepare features
features = ['skin_tone', 'games', 'height', 'weight']
df_model = df_with_skin[features + ['has_red_card']].dropna()

# Fit model without position dummies for simplicity
X_multi = sm.add_constant(df_model[features].astype(float))
y_multi = df_model['has_red_card'].astype(float)
logit_model_multi = sm.Logit(y_multi, X_multi).fit(disp=0)

print("\nKey coefficient (skin_tone):")
skin_tone_coef = logit_model_multi.params['skin_tone']
skin_tone_pval = logit_model_multi.pvalues['skin_tone']
skin_tone_ci = logit_model_multi.conf_int().loc['skin_tone']

print(f"   Coefficient: {skin_tone_coef:.4f}")
print(f"   p-value: {skin_tone_pval:.6f}")
print(f"   95% CI: [{skin_tone_ci[0]:.4f}, {skin_tone_ci[1]:.4f}]")
print(f"   Odds Ratio: {np.exp(skin_tone_coef):.4f}")
print(f"   Significant at α=0.05: {'YES' if skin_tone_pval < 0.05 else 'NO'}")

# Test 5: Poisson regression for count data (number of red cards)
print("\n5. Poisson Regression (red card counts):")
X_poisson = sm.add_constant(df_with_skin[['skin_tone', 'games']])
y_poisson = df_with_skin['redCards']
poisson_model = sm.GLM(y_poisson, X_poisson, family=sm.families.Poisson()).fit()

print("\nKey coefficient (skin_tone):")
skin_coef_poisson = poisson_model.params['skin_tone']
skin_pval_poisson = poisson_model.pvalues['skin_tone']
print(f"   Coefficient: {skin_coef_poisson:.4f}")
print(f"   p-value: {skin_pval_poisson:.6f}")
print(f"   Rate Ratio: {np.exp(skin_coef_poisson):.4f}")
print(f"   Significant at α=0.05: {'YES' if skin_pval_poisson < 0.05 else 'NO'}")

# Test 6: Correlation analysis
print("\n6. Correlation Analysis:")
corr, p_corr = stats.pearsonr(df_with_skin['skin_tone'], df_with_skin['redCards'])
print(f"   Pearson correlation: {corr:.4f}")
print(f"   p-value: {p_corr:.6f}")
print(f"   Significant at α=0.05: {'YES' if p_corr < 0.05 else 'NO'}")

print("\n" + "="*80)
print("INTERPRETABLE MODEL: EXPLAINABLE BOOSTING")
print("="*80)

from interpret.glassbox import ExplainableBoostingClassifier

# Prepare data for EBM
features_ebm = ['skin_tone', 'games', 'height', 'weight', 'yellowCards']
df_ebm = df_with_skin[features_ebm + ['has_red_card']].dropna()

X_ebm = df_ebm[features_ebm]
y_ebm = df_ebm['has_red_card']

# Train EBM
ebm = ExplainableBoostingClassifier(random_state=42, max_rounds=1000)
ebm.fit(X_ebm, y_ebm)

# Get global importance
print("\nFeature Importance (EBM):")
importances = ebm.term_importances()
for i, (feature, importance) in enumerate(zip(ebm.feature_names_in_, importances)):
    print(f"   {i+1}. {feature}: {importance:.4f}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

# Determine response based on statistical evidence
significant_tests = 0
total_tests = 6

if p_value_ttest < 0.05:
    significant_tests += 1
if p_value_mann < 0.05:
    significant_tests += 1
if logit_model_simple.pvalues['skin_tone'] < 0.05:
    significant_tests += 1
if skin_tone_pval < 0.05:
    significant_tests += 1
if skin_pval_poisson < 0.05:
    significant_tests += 1
if p_corr < 0.05:
    significant_tests += 1

print(f"\nSignificant tests: {significant_tests}/{total_tests}")

# Calculate effect size (Cohen's d)
mean_diff = dark_skin_players['redCards'].mean() - light_skin_players['redCards'].mean()
pooled_std = np.sqrt((dark_skin_players['redCards'].var() + light_skin_players['redCards'].var()) / 2)
cohens_d = mean_diff / pooled_std

print(f"Effect size (Cohen's d): {cohens_d:.4f}")

# Decision logic
if significant_tests >= 4:
    if mean_diff > 0:
        response = 85  # Strong Yes
        explanation = (
            f"Strong evidence of relationship: {significant_tests}/6 statistical tests were significant (p<0.05). "
            f"Dark skin players have a {mean_diff:.6f} higher red card rate per dyad. "
            f"Multiple regression controlling for confounders shows skin_tone coefficient = {skin_tone_coef:.4f} (p={skin_tone_pval:.6f}, OR={np.exp(skin_tone_coef):.4f}). "
            f"Poisson regression confirms higher rate (RR={np.exp(skin_coef_poisson):.4f}, p={skin_pval_poisson:.6f}). "
            f"Effect size (Cohen's d={cohens_d:.4f}) indicates a meaningful practical difference."
        )
    else:
        response = 15  # Strong No (opposite direction)
        explanation = (
            f"Evidence suggests opposite relationship: {significant_tests}/6 tests significant but effect is negative. "
            f"Light skin players actually have higher red card rates."
        )
elif significant_tests >= 2:
    if mean_diff > 0:
        response = 65  # Moderate Yes
        explanation = (
            f"Moderate evidence of relationship: {significant_tests}/6 statistical tests were significant (p<0.05). "
            f"Dark skin players show {mean_diff:.6f} higher red card rate. "
            f"However, not all tests reached significance, suggesting relationship may be modest or context-dependent."
        )
    else:
        response = 35  # Moderate No
        explanation = (
            f"Mixed evidence: {significant_tests}/6 tests significant but with inconsistent direction. "
            f"Results do not strongly support the hypothesis."
        )
else:
    response = 20  # Strong No
    explanation = (
        f"Insufficient evidence of relationship: Only {significant_tests}/6 statistical tests were significant. "
        f"T-test p={p_value_ttest:.4f}, Mann-Whitney p={p_value_mann:.4f}, "
        f"Simple logistic regression p={logit_model_simple.pvalues['skin_tone']:.4f}. "
        f"Multiple tests failed to reach conventional significance threshold (p<0.05). "
        f"Effect size (Cohen's d={cohens_d:.4f}) is very small, indicating minimal practical difference."
    )

print(f"\nFinal Assessment:")
print(f"Response: {response}/100")
print(f"Explanation: {explanation}")

# Write conclusion to file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n" + "="*80)
print("Analysis complete. Results written to conclusion.txt")
print("="*80)
