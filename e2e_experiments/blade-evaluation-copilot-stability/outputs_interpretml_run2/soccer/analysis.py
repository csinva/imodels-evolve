import pandas as pd
import numpy as np
import json
from scipy import stats
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor

# Load the data
df = pd.read_csv('soccer.csv')

print("="*80)
print("DATA EXPLORATION")
print("="*80)
print(f"\nDataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

# Create average skin tone rating
df['skin_tone'] = (df['rater1'] + df['rater2']) / 2

# Remove rows with missing skin tone data
df_complete = df[df['skin_tone'].notna()].copy()
print(f"\nRows with skin tone data: {len(df_complete)}")

# Summary statistics for skin tone
print(f"\nSkin Tone Summary Statistics:")
print(df_complete['skin_tone'].describe())

# Summary statistics for red cards
print(f"\nRed Cards Summary Statistics:")
print(df_complete['redCards'].describe())

# Categorize skin tone into light (< 0.5) and dark (>= 0.5)
df_complete['skin_category'] = df_complete['skin_tone'].apply(lambda x: 'dark' if x >= 0.5 else 'light')

print(f"\nSkin tone distribution:")
print(df_complete['skin_category'].value_counts())

# Calculate red card rates by skin category
print("\n" + "="*80)
print("RED CARD RATES BY SKIN TONE")
print("="*80)

grouped = df_complete.groupby('skin_category').agg({
    'redCards': ['sum', 'mean', 'count'],
    'games': 'sum'
})

print(grouped)

light_skin = df_complete[df_complete['skin_category'] == 'light']
dark_skin = df_complete[df_complete['skin_category'] == 'dark']

# Red cards per game (rate)
light_red_rate = light_skin['redCards'].sum() / light_skin['games'].sum()
dark_red_rate = dark_skin['redCards'].sum() / dark_skin['games'].sum()

print(f"\nLight skin players: {light_red_rate:.6f} red cards per game")
print(f"Dark skin players: {dark_red_rate:.6f} red cards per game")
print(f"Ratio (dark/light): {dark_red_rate/light_red_rate:.3f}")

# Statistical tests
print("\n" + "="*80)
print("STATISTICAL TESTS")
print("="*80)

# T-test on red cards per player-referee dyad
t_stat, p_value_ttest = stats.ttest_ind(dark_skin['redCards'], light_skin['redCards'])
print(f"\nT-test (independent samples):")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_value_ttest:.6f}")

# Mann-Whitney U test (non-parametric alternative)
u_stat, p_value_mann = stats.mannwhitneyu(dark_skin['redCards'], light_skin['redCards'], alternative='greater')
print(f"\nMann-Whitney U test (dark > light):")
print(f"  U-statistic: {u_stat:.4f}")
print(f"  p-value: {p_value_mann:.6f}")

# Logistic regression
print("\n" + "="*80)
print("LOGISTIC REGRESSION - INTERPRETABLE MODEL")
print("="*80)

# Create binary outcome: did player receive at least 1 red card?
df_complete['received_red'] = (df_complete['redCards'] > 0).astype(int)

print(f"\nPlayers who received at least 1 red card: {df_complete['received_red'].sum()} ({100*df_complete['received_red'].mean():.2f}%)")

# Prepare features for modeling
features_for_model = ['skin_tone', 'games', 'yellowCards', 'height', 'weight']
X = df_complete[features_for_model].copy()

# Handle any remaining missing values
X = X.fillna(X.mean())
y = df_complete['received_red']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=features_for_model, index=X.index)

# Logistic regression with statsmodels for p-values
X_with_const = sm.add_constant(X_scaled_df)
logit_model = sm.Logit(y, X_with_const)
logit_result = logit_model.fit(disp=0)

print("\nLogistic Regression Results:")
print(logit_result.summary2().tables[1])

# Extract skin_tone coefficient and p-value
skin_tone_coef = logit_result.params['skin_tone']
skin_tone_pval = logit_result.pvalues['skin_tone']
skin_tone_odds_ratio = np.exp(skin_tone_coef)

print(f"\n*** SKIN TONE EFFECT ***")
print(f"Coefficient: {skin_tone_coef:.4f}")
print(f"Odds Ratio: {skin_tone_odds_ratio:.4f}")
print(f"P-value: {skin_tone_pval:.6f}")
print(f"95% CI: {logit_result.conf_int().loc['skin_tone'].values}")

# Poisson regression for count data
print("\n" + "="*80)
print("POISSON REGRESSION - RED CARDS COUNT")
print("="*80)

# Use Poisson regression with exposure (number of games)
X_poisson = df_complete[['skin_tone', 'yellowCards', 'height', 'weight']].copy()
# Fill missing values with column-specific means
for col in X_poisson.columns:
    X_poisson.loc[:, col] = X_poisson[col].fillna(X_poisson[col].mean())

# Scale the features
scaler_poisson = StandardScaler()
X_poisson_scaled = scaler_poisson.fit_transform(X_poisson)
X_poisson_const = sm.add_constant(X_poisson_scaled)

# Check for any remaining NaNs or infs
if np.any(np.isnan(X_poisson_const)) or np.any(np.isinf(X_poisson_const)):
    print("Warning: NaNs or Infs detected after scaling, using simpler approach")
    # Use original data without scaling
    X_poisson_const = sm.add_constant(X_poisson)

poisson_model = sm.GLM(df_complete['redCards'], 
                       X_poisson_const,
                       family=sm.families.Poisson(),
                       exposure=df_complete['games'])
poisson_result = poisson_model.fit()

print(poisson_result.summary2().tables[1])

skin_tone_coef_poisson = poisson_result.params['x1']  # x1 is skin_tone
skin_tone_pval_poisson = poisson_result.pvalues['x1']
incident_rate_ratio = np.exp(skin_tone_coef_poisson)

print(f"\n*** SKIN TONE EFFECT (Poisson) ***")
print(f"Coefficient: {skin_tone_coef_poisson:.4f}")
print(f"Incident Rate Ratio: {incident_rate_ratio:.4f}")
print(f"P-value: {skin_tone_pval_poisson:.6f}")

# Explainable Boosting Classifier
print("\n" + "="*80)
print("EXPLAINABLE BOOSTING CLASSIFIER")
print("="*80)

ebc = ExplainableBoostingClassifier(random_state=42, max_rounds=100)
ebc.fit(X, y)

# Get feature importances
feature_importance = list(zip(features_for_model, ebc.term_importances()))
feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)

print("\nFeature Importances:")
for feat, imp in feature_importance:
    print(f"  {feat}: {imp:.4f}")

# Correlation analysis
print("\n" + "="*80)
print("CORRELATION ANALYSIS")
print("="*80)

corr_red_skin = df_complete[['redCards', 'skin_tone']].corr()
print("\nCorrelation matrix (redCards vs skin_tone):")
print(corr_red_skin)

# Pearson and Spearman correlations
pearson_r, pearson_p = stats.pearsonr(df_complete['skin_tone'], df_complete['redCards'])
spearman_r, spearman_p = stats.spearmanr(df_complete['skin_tone'], df_complete['redCards'])

print(f"\nPearson correlation: r={pearson_r:.4f}, p={pearson_p:.6f}")
print(f"Spearman correlation: rho={spearman_r:.4f}, p={spearman_p:.6f}")

# Final conclusion
print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

# Determine response based on statistical evidence
significant_results = []
if p_value_ttest < 0.05:
    significant_results.append(f"T-test (p={p_value_ttest:.6f})")
if p_value_mann < 0.05:
    significant_results.append(f"Mann-Whitney U (p={p_value_mann:.6f})")
if skin_tone_pval < 0.05:
    significant_results.append(f"Logistic regression (p={skin_tone_pval:.6f})")
if skin_tone_pval_poisson < 0.05:
    significant_results.append(f"Poisson regression (p={skin_tone_pval_poisson:.6f})")

# Calculate effect size
effect_magnitude = dark_red_rate / light_red_rate if light_red_rate > 0 else 1.0

if len(significant_results) >= 3 and effect_magnitude > 1.2:
    response_score = 85
    explanation = f"Strong evidence: Dark skin players receive red cards at {dark_red_rate:.4f} per game vs {light_red_rate:.4f} for light skin players (ratio: {effect_magnitude:.2f}). Significant results from {len(significant_results)}/4 tests: {', '.join(significant_results)}. Logistic regression shows odds ratio of {skin_tone_odds_ratio:.2f}, Poisson regression shows incident rate ratio of {incident_rate_ratio:.2f}."
elif len(significant_results) >= 2:
    response_score = 70
    explanation = f"Moderate evidence: Dark skin players have {dark_red_rate:.4f} red cards per game vs {light_red_rate:.4f} for light skin (ratio: {effect_magnitude:.2f}). Significant in {len(significant_results)}/4 tests: {', '.join(significant_results)}."
elif len(significant_results) == 1:
    response_score = 55
    explanation = f"Weak evidence: Some statistical significance ({', '.join(significant_results)}) but not consistent across tests. Rate ratio: {effect_magnitude:.2f}."
else:
    response_score = 25
    explanation = f"No clear evidence: None of the primary statistical tests showed significant differences (all p > 0.05). Rate ratio was {effect_magnitude:.2f}, but not statistically significant."

print(f"\nResponse Score: {response_score}/100")
print(f"Explanation: {explanation}")

# Write conclusion to file
conclusion = {
    "response": response_score,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\nConclusion written to conclusion.txt")
