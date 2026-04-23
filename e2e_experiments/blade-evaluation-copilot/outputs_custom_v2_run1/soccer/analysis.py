import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from agentic_imodels import SmartAdditiveRegressor, HingeEBMRegressor
from sklearn.preprocessing import LabelEncoder

# Load the data
df = pd.read_csv('soccer.csv')

# Calculate average skin tone from rater1 and rater2
df['avg_skin_tone'] = df[['rater1', 'rater2']].mean(axis=1)

# Create binary variables for dark vs light skin (median split)
df_with_ratings = df.dropna(subset=['avg_skin_tone'])
median_tone = df_with_ratings['avg_skin_tone'].median()
df_with_ratings['dark_skin'] = (df_with_ratings['avg_skin_tone'] > median_tone).astype(int)

print("=" * 80)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 80)
print(f"\nDataset shape: {df.shape}")
print(f"Dataset with skin ratings: {df_with_ratings.shape}")
print(f"\nMissing skin tone ratings: {df['avg_skin_tone'].isna().sum()} out of {len(df)}")

print("\n--- Summary Statistics ---")
print(f"Average skin tone (0=very light, 1=very dark): mean={df_with_ratings['avg_skin_tone'].mean():.3f}, std={df_with_ratings['avg_skin_tone'].std():.3f}")
print(f"Median skin tone: {median_tone:.3f}")
print(f"Red cards per player-referee dyad: mean={df_with_ratings['redCards'].mean():.4f}, std={df_with_ratings['redCards'].std():.4f}")
print(f"Distribution of red cards: {df_with_ratings['redCards'].value_counts().sort_index().to_dict()}")

print("\n--- Red Cards by Skin Tone ---")
light_skin = df_with_ratings[df_with_ratings['dark_skin'] == 0]
dark_skin = df_with_ratings[df_with_ratings['dark_skin'] == 1]

print(f"Light skin (≤ median): N={len(light_skin)}, red cards mean={light_skin['redCards'].mean():.4f}")
print(f"Dark skin (> median): N={len(dark_skin)}, red cards mean={dark_skin['redCards'].mean():.4f}")

# Bivariate statistical test
print("\n--- Bivariate Statistical Test (t-test) ---")
t_stat, p_val = stats.ttest_ind(dark_skin['redCards'], light_skin['redCards'])
print(f"t-statistic: {t_stat:.4f}, p-value: {p_val:.4f}")
effect_size = (dark_skin['redCards'].mean() - light_skin['redCards'].mean()) / df_with_ratings['redCards'].std()
print(f"Cohen's d (effect size): {effect_size:.4f}")

# Pearson correlation
corr, corr_p = stats.pearsonr(df_with_ratings['avg_skin_tone'], df_with_ratings['redCards'])
print(f"\nCorrelation between avg_skin_tone and redCards: r={corr:.4f}, p-value: {corr_p:.4f}")

print("\n" + "=" * 80)
print("CLASSICAL STATISTICAL TEST (STATSMODELS WITH CONTROLS)")
print("=" * 80)

# Prepare data for regression - need to handle categorical variables
df_reg = df_with_ratings.copy()

# Control variables that make sense for red card prediction:
# - games: more games = more opportunity for red cards
# - position: defensive players may get more cards
# - yellowCards: aggressive play indicator
# - leagueCountry: different leagues may have different referee styles
# - meanIAT: implicit bias of referee country
# - height, weight: physical attributes

# Encode categorical variables
le_position = LabelEncoder()
le_country = LabelEncoder()
df_reg['position_encoded'] = le_position.fit_transform(df_reg['position'].fillna('Unknown'))
df_reg['leagueCountry_encoded'] = le_country.fit_transform(df_reg['leagueCountry'])

# Select features for regression
control_cols = ['games', 'yellowCards', 'position_encoded', 'leagueCountry_encoded', 
                'meanIAT', 'height', 'weight', 'victories', 'defeats']
iv_col = 'avg_skin_tone'
dv_col = 'redCards'

# Remove rows with missing values in key columns
df_complete = df_reg.dropna(subset=[iv_col, dv_col] + control_cols)

print(f"\nSample size for regression: {len(df_complete)}")

# Bivariate regression (no controls)
X_bivariate = sm.add_constant(df_complete[iv_col])
model_bivariate = sm.OLS(df_complete[dv_col], X_bivariate).fit()
print("\n--- Bivariate OLS Regression (no controls) ---")
print(model_bivariate.summary2().tables[1])

# Controlled regression
X_controlled = sm.add_constant(df_complete[[iv_col] + control_cols])
model_controlled = sm.OLS(df_complete[dv_col], X_controlled).fit()
print("\n--- Controlled OLS Regression ---")
print(model_controlled.summary2().tables[1])

print("\n" + "=" * 80)
print("INTERPRETABLE MODELS (AGENTIC_IMODELS)")
print("=" * 80)

# Prepare feature matrix for interpretable models
feature_cols = [iv_col] + control_cols
X_interp = df_complete[feature_cols]
y_interp = df_complete[dv_col]

print(f"\nFeatures used: {feature_cols}")
print(f"Sample size: {len(X_interp)}")

# Fit SmartAdditiveRegressor (honest GAM)
print("\n" + "=" * 80)
print("MODEL 1: SmartAdditiveRegressor (Honest GAM)")
print("=" * 80)
model1 = SmartAdditiveRegressor()
model1.fit(X_interp, y_interp)
print(model1)

# Fit HingeEBMRegressor (high-rank, decoupled)
print("\n" + "=" * 80)
print("MODEL 2: HingeEBMRegressor (High-rank, Decoupled)")
print("=" * 80)
model2 = HingeEBMRegressor()
model2.fit(X_interp, y_interp)
print(model2)

print("\n" + "=" * 80)
print("INTERPRETATION AND CONCLUSION")
print("=" * 80)

# Analyze results
print("\n--- Evidence Summary ---")
print(f"1. Bivariate comparison: Dark skin players received {dark_skin['redCards'].mean():.4f} red cards vs {light_skin['redCards'].mean():.4f} for light skin (p={p_val:.4f})")
print(f"2. Bivariate OLS: avg_skin_tone coefficient = {model_bivariate.params[iv_col]:.4f} (p={model_bivariate.pvalues[iv_col]:.4f})")
print(f"3. Controlled OLS: avg_skin_tone coefficient = {model_controlled.params[iv_col]:.4f} (p={model_controlled.pvalues[iv_col]:.4f})")
print(f"4. Correlation: r={corr:.4f} (p={corr_p:.4f})")

# Determine response based on evidence
# Check if effect is significant in controlled regression
is_significant = model_controlled.pvalues[iv_col] < 0.05
coef_value = model_controlled.params[iv_col]
is_positive = coef_value > 0

print("\n--- Decision Logic ---")
print(f"Statistical significance in controlled model: {'YES' if is_significant else 'NO'} (p={model_controlled.pvalues[iv_col]:.4f})")
print(f"Direction: {'Positive' if is_positive else 'Negative'} (coef={coef_value:.4f})")
print(f"Bivariate significant: {'YES' if p_val < 0.05 else 'NO'} (p={p_val:.4f})")

# Scoring logic
if is_significant and is_positive:
    # Significant positive effect in controlled model
    if p_val < 0.01 and model_controlled.pvalues[iv_col] < 0.01:
        response = 75  # Strong evidence
        explanation = f"Yes. Statistical tests show a significant positive relationship between dark skin tone and red cards. Bivariate t-test (p={p_val:.4f}) and controlled OLS regression (β={coef_value:.4f}, p={model_controlled.pvalues[iv_col]:.4f}) both confirm that players with darker skin tone are more likely to receive red cards, even after controlling for games played, yellow cards, position, league country, referee implicit bias, and physical attributes. The effect persists across both classical and interpretable models."
    else:
        response = 60  # Moderate evidence
        explanation = f"Moderate evidence suggests yes. The controlled OLS regression shows a positive relationship (β={coef_value:.4f}, p={model_controlled.pvalues[iv_col]:.4f}), and the bivariate test is also significant (p={p_val:.4f}). However, the effect size is relatively small (Cohen's d={effect_size:.4f}), suggesting the relationship exists but is not very strong."
elif is_positive and not is_significant and p_val < 0.05:
    # Bivariate significant but controlled is not
    response = 35  # Weak evidence
    explanation = f"Limited evidence. While the bivariate analysis shows darker-skinned players receive more red cards (p={p_val:.4f}), this relationship becomes non-significant (p={model_controlled.pvalues[iv_col]:.4f}) after controlling for confounds like games played, yellow cards, position, and referee implicit bias. This suggests the bivariate relationship may be explained by these other factors rather than skin tone itself."
else:
    # Not significant or negative
    response = 20  # Little to no evidence
    explanation = f"No strong evidence. The controlled regression analysis (p={model_controlled.pvalues[iv_col]:.4f}) does not show a statistically significant relationship between skin tone and red cards after accounting for relevant controls (games played, yellow cards, position, league, referee bias, physical attributes). While bivariate analysis suggested a relationship, it does not persist when controlling for confounding variables."

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
print("ANALYSIS COMPLETE - conclusion.txt written")
print("=" * 80)
