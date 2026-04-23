import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import json

# Load the data
df = pd.read_csv('soccer.csv')

print("=" * 80)
print("DATASET OVERVIEW")
print("=" * 80)
print(f"Total observations: {len(df)}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nData shape: {df.shape}")

# Key variables for the research question
print("\n" + "=" * 80)
print("KEY VARIABLES FOR RESEARCH QUESTION")
print("=" * 80)

# Calculate average skin tone from the two raters
df['skinTone'] = (df['rater1'] + df['rater2']) / 2
print(f"\nSkin tone statistics (0=very light, 1=very dark):")
print(df['skinTone'].describe())

# Red cards statistics
print(f"\nRed cards statistics:")
print(df['redCards'].describe())
print(f"\nRed cards distribution:")
print(df['redCards'].value_counts().sort_index())

# Create binary indicators for skin tone categories
# Light skin: skinTone < 0.4
# Dark skin: skinTone > 0.6
df['lightSkin'] = (df['skinTone'] < 0.4).astype(int)
df['darkSkin'] = (df['skinTone'] > 0.6).astype(int)

print(f"\nLight skin tone players (skinTone < 0.4): {df['lightSkin'].sum()}")
print(f"Dark skin tone players (skinTone > 0.6): {df['darkSkin'].sum()}")

# Calculate red card rates per player
# Since data is player-referee dyads, aggregate to player level
player_data = df.groupby('playerShort').agg({
    'redCards': 'sum',
    'games': 'sum',
    'skinTone': 'first',  # Same for each player
    'yellowCards': 'sum',
    'yellowReds': 'sum',
    'height': 'first',
    'weight': 'first',
    'position': 'first'
}).reset_index()

player_data['redCardRate'] = player_data['redCards'] / player_data['games']
player_data['lightSkin'] = (player_data['skinTone'] < 0.4).astype(int)
player_data['darkSkin'] = (player_data['skinTone'] > 0.6).astype(int)

# Remove players with missing skin tone data
player_data_clean = player_data[player_data['skinTone'].notna()].copy()

print("\n" + "=" * 80)
print("PLAYER-LEVEL ANALYSIS")
print("=" * 80)
print(f"Total unique players with skin tone data: {len(player_data_clean)}")

# Compare red card rates between light and dark skin players
light_players = player_data_clean[player_data_clean['lightSkin'] == 1]
dark_players = player_data_clean[player_data_clean['darkSkin'] == 1]

print(f"\nLight skin players: {len(light_players)}")
print(f"Dark skin players: {len(dark_players)}")

# Red card statistics by skin tone
print("\n" + "=" * 80)
print("RED CARD RATES BY SKIN TONE")
print("=" * 80)

light_red_card_rate = light_players['redCardRate'].mean()
dark_red_card_rate = dark_players['redCardRate'].mean()

print(f"\nLight skin players - Average red card rate: {light_red_card_rate:.6f}")
print(f"Dark skin players - Average red card rate: {dark_red_card_rate:.6f}")
print(f"Difference: {dark_red_card_rate - light_red_card_rate:.6f}")
print(f"Ratio (dark/light): {dark_red_card_rate / light_red_card_rate:.3f}")

# Statistical test: t-test comparing red card rates
t_stat, p_value = stats.ttest_ind(dark_players['redCardRate'], light_players['redCardRate'])
print(f"\nT-test: t={t_stat:.4f}, p-value={p_value:.6f}")

# Also test at dyad level
dyad_data_clean = df[df['skinTone'].notna()].copy()
light_dyads = dyad_data_clean[dyad_data_clean['skinTone'] < 0.4]
dark_dyads = dyad_data_clean[dyad_data_clean['skinTone'] > 0.6]

light_red_rate_dyad = light_dyads['redCards'].sum() / light_dyads['games'].sum()
dark_red_rate_dyad = dark_dyads['redCards'].sum() / dark_dyads['games'].sum()

print(f"\nDyad-level analysis:")
print(f"Light skin - Red cards per game: {light_red_rate_dyad:.6f}")
print(f"Dark skin - Red cards per game: {dark_red_rate_dyad:.6f}")

# Chi-square test on actual red card counts
light_red_cards = light_dyads['redCards'].sum()
light_total_games = light_dyads['games'].sum()
dark_red_cards = dark_dyads['redCards'].sum()
dark_total_games = dark_dyads['games'].sum()

# Contingency table: [got red card, no red card] x [light, dark]
contingency = np.array([
    [light_red_cards, light_total_games - light_red_cards],
    [dark_red_cards, dark_total_games - dark_red_cards]
])
chi2, p_chi2, dof, expected = stats.chi2_contingency(contingency)
print(f"\nChi-square test: χ²={chi2:.4f}, p-value={p_chi2:.6f}")

# Logistic regression controlling for confounders
print("\n" + "=" * 80)
print("LOGISTIC REGRESSION (Player-level)")
print("=" * 80)

# Create binary outcome: did player receive any red card?
player_data_clean['hasRedCard'] = (player_data_clean['redCards'] > 0).astype(int)

# Prepare data for logistic regression
X_cols = ['skinTone', 'games', 'yellowCards']
y = player_data_clean['hasRedCard']
X = player_data_clean[X_cols].copy()
X = X.dropna()
y = y[X.index]

# Add constant for statsmodels
X_sm = sm.add_constant(X)

# Logistic regression
logit_model = sm.Logit(y, X_sm)
result = logit_model.fit(disp=0)
print(result.summary())

print("\n" + "=" * 80)
print("POISSON REGRESSION (Dyad-level)")
print("=" * 80)

# Poisson regression for count data - controlling for exposure (games)
dyad_model_data = dyad_data_clean[['redCards', 'skinTone', 'games', 'yellowCards']].dropna()
y_dyad = dyad_model_data['redCards']
X_dyad = dyad_model_data[['skinTone', 'yellowCards']]
X_dyad_sm = sm.add_constant(X_dyad)

# Add log(games) as exposure
poisson_model = sm.GLM(y_dyad, X_dyad_sm, family=sm.families.Poisson(), 
                       offset=np.log(dyad_model_data['games']))
poisson_result = poisson_model.fit()
print(poisson_result.summary())

# Correlation analysis
print("\n" + "=" * 80)
print("CORRELATION ANALYSIS")
print("=" * 80)

# Correlation between skin tone and red cards
corr, p_corr = stats.pearsonr(player_data_clean['skinTone'], player_data_clean['redCards'])
print(f"Pearson correlation (skin tone vs red cards): r={corr:.4f}, p={p_corr:.6f}")

corr_rate, p_corr_rate = stats.pearsonr(player_data_clean['skinTone'], player_data_clean['redCardRate'])
print(f"Pearson correlation (skin tone vs red card rate): r={corr_rate:.4f}, p={p_corr_rate:.6f}")

# Linear regression
print("\n" + "=" * 80)
print("LINEAR REGRESSION (Red card rate ~ Skin tone)")
print("=" * 80)

X_lr = player_data_clean[['skinTone']].copy()
y_lr = player_data_clean['redCardRate']
X_lr_sm = sm.add_constant(X_lr)

lr_model = sm.OLS(y_lr, X_lr_sm)
lr_result = lr_model.fit()
print(lr_result.summary())

# Conclusion
print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

# Determine response based on statistical significance and effect size
# The research question is: Are dark skin players MORE likely to receive red cards?

significant_tests = []
effect_direction = []

# Test 1: T-test
if p_value < 0.05:
    significant_tests.append("T-test")
    if dark_red_card_rate > light_red_card_rate:
        effect_direction.append("positive")
    else:
        effect_direction.append("negative")

# Test 2: Chi-square
if p_chi2 < 0.05:
    significant_tests.append("Chi-square")
    if dark_red_rate_dyad > light_red_rate_dyad:
        effect_direction.append("positive")
    else:
        effect_direction.append("negative")

# Test 3: Logistic regression skin tone coefficient
skin_tone_pvalue = result.pvalues['skinTone']
skin_tone_coef = result.params['skinTone']
if skin_tone_pvalue < 0.05:
    significant_tests.append("Logistic regression")
    if skin_tone_coef > 0:
        effect_direction.append("positive")
    else:
        effect_direction.append("negative")

# Test 4: Poisson regression
poisson_pvalue = poisson_result.pvalues['skinTone']
poisson_coef = poisson_result.params['skinTone']
if poisson_pvalue < 0.05:
    significant_tests.append("Poisson regression")
    if poisson_coef > 0:
        effect_direction.append("positive")
    else:
        effect_direction.append("negative")

# Test 5: Correlation
if p_corr_rate < 0.05:
    significant_tests.append("Correlation")
    if corr_rate > 0:
        effect_direction.append("positive")
    else:
        effect_direction.append("negative")

print(f"\nSignificant tests (p < 0.05): {significant_tests}")
print(f"Effect directions: {effect_direction}")

# Calculate effect size
relative_increase = ((dark_red_card_rate - light_red_card_rate) / light_red_card_rate) * 100
print(f"\nEffect size: Dark skin players have {relative_increase:.1f}% higher red card rate")

# Determine Likert score
if len(significant_tests) == 0:
    # No significant relationship found
    response = 10
    explanation = (f"No statistically significant relationship found. "
                  f"Dark skin players had {relative_increase:.1f}% higher red card rate, "
                  f"but none of the statistical tests (t-test p={p_value:.3f}, "
                  f"chi-square p={p_chi2:.3f}, logistic regression p={skin_tone_pvalue:.3f}, "
                  f"Poisson regression p={poisson_pvalue:.3f}, correlation p={p_corr_rate:.3f}) "
                  f"reached significance at p<0.05.")
elif all(d == "positive" for d in effect_direction):
    # All significant tests show positive relationship
    if len(significant_tests) >= 3:
        response = 85
        explanation = (f"Strong evidence: {len(significant_tests)} out of 5 statistical tests "
                      f"showed significant positive relationship (p<0.05). Dark skin players "
                      f"received {relative_increase:.1f}% more red cards than light skin players. "
                      f"Significant tests: {', '.join(significant_tests)}.")
    else:
        response = 70
        explanation = (f"Moderate evidence: {len(significant_tests)} statistical test(s) "
                      f"showed significant positive relationship (p<0.05). Dark skin players "
                      f"received {relative_increase:.1f}% more red cards than light skin players. "
                      f"Significant tests: {', '.join(significant_tests)}.")
else:
    # Mixed or negative results
    response = 30
    explanation = (f"Weak or inconsistent evidence. Some tests significant but results mixed. "
                  f"Significant tests: {', '.join(significant_tests)} with directions: {effect_direction}.")

print(f"\nFinal Likert score: {response}/100")
print(f"Explanation: {explanation}")

# Write conclusion to file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\nConclusion written to conclusion.txt")
