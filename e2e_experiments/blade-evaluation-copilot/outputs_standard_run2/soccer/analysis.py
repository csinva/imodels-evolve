import pandas as pd
import numpy as np
import json
from scipy import stats
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('soccer.csv')

print("="*60)
print("ANALYSIS: Skin Tone and Red Card Likelihood")
print("="*60)
print(f"\nDataset shape: {df.shape}")
print(f"Total player-referee dyads: {len(df)}")

# Calculate average skin tone rating
df['skin_tone'] = (df['rater1'] + df['rater2']) / 2

# Filter to only rows with skin tone data
df_with_skin = df[df['skin_tone'].notna()].copy()
print(f"Dyads with skin tone data: {len(df_with_skin)}")

# Categorize skin tone into light (< 0.5) and dark (>= 0.5)
df_with_skin['dark_skin'] = (df_with_skin['skin_tone'] >= 0.5).astype(int)
df_with_skin['light_skin'] = (df_with_skin['skin_tone'] < 0.5).astype(int)

print(f"\nPlayers with light skin tone (<0.5): {df_with_skin['light_skin'].sum()} dyads")
print(f"Players with dark skin tone (>=0.5): {df_with_skin['dark_skin'].sum()} dyads")

# Calculate red card rates
print("\n" + "="*60)
print("RED CARD STATISTICS")
print("="*60)

light_skin_dyads = df_with_skin[df_with_skin['light_skin'] == 1]
dark_skin_dyads = df_with_skin[df_with_skin['dark_skin'] == 1]

# Red card rates
light_red_rate = light_skin_dyads['redCards'].sum() / len(light_skin_dyads)
dark_red_rate = dark_skin_dyads['redCards'].sum() / len(dark_skin_dyads)

print(f"\nLight skin red cards: {light_skin_dyads['redCards'].sum()} out of {len(light_skin_dyads)} dyads")
print(f"Light skin red card rate: {light_red_rate:.6f}")
print(f"\nDark skin red cards: {dark_skin_dyads['redCards'].sum()} out of {len(dark_skin_dyads)} dyads")
print(f"Dark skin red card rate: {dark_red_rate:.6f}")
print(f"\nRate ratio (dark/light): {dark_red_rate/light_red_rate:.3f}")

# Statistical test - Chi-square test
contingency_table = pd.crosstab(df_with_skin['dark_skin'], df_with_skin['redCards'] > 0)
chi2, p_chi2, dof, expected = stats.chi2_contingency(contingency_table)
print(f"\nChi-square test (dark vs light skin, any red card):")
print(f"  Chi-square statistic: {chi2:.4f}")
print(f"  P-value: {p_chi2:.6f}")

# Two-sample t-test on red card counts
light_red_cards = light_skin_dyads['redCards'].values
dark_red_cards = dark_skin_dyads['redCards'].values
t_stat, p_ttest = stats.ttest_ind(dark_red_cards, light_red_cards)
print(f"\nT-test (dark vs light skin red card counts):")
print(f"  T-statistic: {t_stat:.4f}")
print(f"  P-value: {p_ttest:.6f}")

# Mann-Whitney U test (non-parametric)
u_stat, p_mann = stats.mannwhitneyu(dark_red_cards, light_red_cards, alternative='greater')
print(f"\nMann-Whitney U test (dark > light):")
print(f"  U-statistic: {u_stat:.4f}")
print(f"  P-value: {p_mann:.6f}")

# Logistic regression controlling for confounders
print("\n" + "="*60)
print("LOGISTIC REGRESSION (RED CARD LIKELIHOOD)")
print("="*60)

# Create binary outcome: any red card
df_with_skin['any_red_card'] = (df_with_skin['redCards'] > 0).astype(int)

# Control variables: games, yellowCards, position, league, etc.
# Fill missing positions
df_with_skin['position'] = df_with_skin['position'].fillna('Unknown')

# Create feature matrix
features_for_model = df_with_skin[['dark_skin', 'games', 'yellowCards', 'height', 'weight']].copy()
features_for_model = features_for_model.fillna(features_for_model.mean())

# Simple logistic regression with skin tone only
X_simple = features_for_model[['dark_skin']].values
y = df_with_skin['any_red_card'].values

log_reg_simple = LogisticRegression(random_state=42, max_iter=1000)
log_reg_simple.fit(X_simple, y)

print(f"\nSimple model (skin tone only):")
print(f"  Dark skin coefficient: {log_reg_simple.coef_[0][0]:.4f}")
print(f"  Odds ratio: {np.exp(log_reg_simple.coef_[0][0]):.4f}")

# Logistic regression with controls
X_full = features_for_model.values
log_reg_full = LogisticRegression(random_state=42, max_iter=1000)
log_reg_full.fit(X_full, y)

print(f"\nFull model (with controls):")
print(f"  Dark skin coefficient: {log_reg_full.coef_[0][0]:.4f}")
print(f"  Odds ratio: {np.exp(log_reg_full.coef_[0][0]):.4f}")

# Statsmodels for p-values
X_sm = sm.add_constant(features_for_model)
logit_model = sm.Logit(y, X_sm)
result = logit_model.fit(disp=0)

print(f"\nStatsmodels Logistic Regression:")
print(f"  Dark skin coefficient: {result.params['dark_skin']:.4f}")
print(f"  Dark skin p-value: {result.pvalues['dark_skin']:.6f}")
print(f"  Dark skin odds ratio: {np.exp(result.params['dark_skin']):.4f}")

# Linear regression for red card count
print("\n" + "="*60)
print("LINEAR REGRESSION (RED CARD COUNT)")
print("="*60)

X_linear = sm.add_constant(features_for_model)
y_count = df_with_skin['redCards'].values

ols_model = sm.OLS(y_count, X_linear)
ols_result = ols_model.fit()

print(f"\nOLS Regression:")
print(f"  Dark skin coefficient: {ols_result.params['dark_skin']:.6f}")
print(f"  Dark skin p-value: {ols_result.pvalues['dark_skin']:.6f}")
print(f"  Dark skin 95% CI: [{ols_result.conf_int().loc['dark_skin', 0]:.6f}, {ols_result.conf_int().loc['dark_skin', 1]:.6f}]")

# Interpretable decision tree
print("\n" + "="*60)
print("DECISION TREE (INTERPRETABLE MODEL)")
print("="*60)

tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(features_for_model, y)

feature_names = ['dark_skin', 'games', 'yellowCards', 'height', 'weight']
importances = tree.feature_importances_

print("\nFeature importances:")
for name, imp in zip(feature_names, importances):
    print(f"  {name}: {imp:.4f}")

# Summary statistics by skin tone
print("\n" + "="*60)
print("SUMMARY STATISTICS BY SKIN TONE")
print("="*60)

print("\nLight skin players:")
print(f"  Mean red cards: {light_skin_dyads['redCards'].mean():.6f}")
print(f"  Mean yellow cards: {light_skin_dyads['yellowCards'].mean():.4f}")
print(f"  Mean games: {light_skin_dyads['games'].mean():.4f}")

print("\nDark skin players:")
print(f"  Mean red cards: {dark_skin_dyads['redCards'].mean():.6f}")
print(f"  Mean yellow cards: {dark_skin_dyads['yellowCards'].mean():.4f}")
print(f"  Mean games: {dark_skin_dyads['games'].mean():.4f}")

# Correlation analysis
print("\n" + "="*60)
print("CORRELATION ANALYSIS")
print("="*60)

corr_skin_red = df_with_skin[['skin_tone', 'redCards']].corr().iloc[0, 1]
print(f"\nPearson correlation (skin_tone, redCards): {corr_skin_red:.4f}")

# Spearman correlation (non-parametric)
spearman_corr, spearman_p = stats.spearmanr(df_with_skin['skin_tone'], df_with_skin['redCards'])
print(f"Spearman correlation: {spearman_corr:.4f}, p-value: {spearman_p:.6f}")

# Conclusion
print("\n" + "="*60)
print("CONCLUSION")
print("="*60)

# Determine response score based on statistical evidence
significance_count = 0
significant_tests = []

if p_chi2 < 0.05:
    significance_count += 1
    significant_tests.append("Chi-square test")
    
if p_ttest < 0.05:
    significance_count += 1
    significant_tests.append("T-test")
    
if p_mann < 0.05:
    significance_count += 1
    significant_tests.append("Mann-Whitney U test")
    
if result.pvalues['dark_skin'] < 0.05:
    significance_count += 1
    significant_tests.append("Logistic regression")
    
if ols_result.pvalues['dark_skin'] < 0.05:
    significance_count += 1
    significant_tests.append("Linear regression")

# Calculate response score
if significance_count >= 3:
    response_score = 75 + (significance_count - 3) * 5
elif significance_count == 2:
    response_score = 60
elif significance_count == 1:
    response_score = 45
else:
    response_score = 20

# Check effect direction and magnitude
odds_ratio = np.exp(result.params['dark_skin'])
rate_ratio = dark_red_rate / light_red_rate

print(f"\nStatistical significance summary:")
print(f"  Number of significant tests (p < 0.05): {significance_count}/5")
if significant_tests:
    print(f"  Significant tests: {', '.join(significant_tests)}")

print(f"\nEffect magnitude:")
print(f"  Rate ratio (dark/light): {rate_ratio:.3f}")
print(f"  Odds ratio (logistic regression): {odds_ratio:.3f}")

# Final determination
if significance_count >= 3 and rate_ratio > 1.0:
    response_score = max(75, response_score)
    explanation = f"Yes, there is strong statistical evidence that soccer players with dark skin tone are more likely to receive red cards. Multiple statistical tests show significance (p < 0.05): {', '.join(significant_tests)}. Dark skin players have {rate_ratio:.2f}x higher red card rate ({dark_red_rate:.4f}) compared to light skin players ({light_red_rate:.4f}). Logistic regression shows odds ratio of {odds_ratio:.3f}, controlling for confounders."
elif significance_count >= 1 and rate_ratio > 1.0:
    response_score = min(65, max(response_score, 50))
    explanation = f"There is moderate evidence suggesting players with dark skin tone may be more likely to receive red cards. {significance_count} statistical test(s) show significance. Dark skin players have {rate_ratio:.2f}x higher red card rate, but the evidence is not overwhelming across all tests."
else:
    response_score = 30
    explanation = f"There is limited or no significant statistical evidence that players with dark skin tone are more likely to receive red cards. Most statistical tests do not show significance (p >= 0.05), suggesting the relationship may not be robust."

print(f"\nFinal assessment: {response_score}/100")
print(f"\n{explanation}")

# Write conclusion to file
conclusion = {
    "response": response_score,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n" + "="*60)
print(f"Analysis complete. Conclusion written to conclusion.txt")
print("="*60)
