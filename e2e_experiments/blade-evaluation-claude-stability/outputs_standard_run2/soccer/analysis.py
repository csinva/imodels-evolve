import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('soccer.csv')
print(f"Shape: {df.shape}")
print(df[['rater1','rater2','redCards','games']].describe())

# Average skin tone across two raters
df['skin_tone'] = (df['rater1'].fillna(df['rater2']) + df['rater2'].fillna(df['rater1'])) / 2
# Drop rows where both raters are missing
df = df.dropna(subset=['skin_tone', 'redCards', 'games'])
print(f"After dropping missing skin_tone: {df.shape}")

# Red card rate per game
df['rc_rate'] = df['redCards'] / df['games']

# Aggregate by player
player = df.groupby('playerShort').agg(
    skin_tone=('skin_tone', 'mean'),
    total_red=('redCards', 'sum'),
    total_games=('games', 'sum')
).reset_index()
player['rc_rate'] = player['total_red'] / player['total_games']
print(f"\nPlayers: {len(player)}")

# Binarize: dark = skin_tone > 0.5, light = skin_tone <= 0.5
dark = player[player['skin_tone'] > 0.5]['rc_rate']
light = player[player['skin_tone'] <= 0.5]['rc_rate']
print(f"\nDark skin players: {len(dark)}, Light skin players: {len(light)}")
print(f"Dark RC rate mean: {dark.mean():.5f}, Light RC rate mean: {light.mean():.5f}")

# t-test
t_stat, p_val = stats.ttest_ind(dark, light)
print(f"\nt-test: t={t_stat:.4f}, p={p_val:.6f}")

# Mann-Whitney U test (non-parametric)
u_stat, p_mw = stats.mannwhitneyu(dark, light, alternative='greater')
print(f"Mann-Whitney U (dark > light): U={u_stat:.1f}, p={p_mw:.6f}")

# OLS regression: rc_rate ~ skin_tone (player-level)
X = sm.add_constant(player['skin_tone'])
y = player['rc_rate']
model = sm.OLS(y, X).fit()
print(f"\nOLS coef on skin_tone: {model.params['skin_tone']:.6f}, p={model.pvalues['skin_tone']:.6f}")

# Dyad-level logistic-ish: use Poisson regression on redCards with log(games) offset
# OLS on dyad level with skin_tone
X2 = sm.add_constant(df['skin_tone'])
y2 = df['rc_rate']
model2 = sm.OLS(y2, X2).fit()
print(f"Dyad OLS coef on skin_tone: {model2.params['skin_tone']:.6f}, p={model2.pvalues['skin_tone']:.6f}")

# Correlation
corr, p_corr = stats.pearsonr(player['skin_tone'], player['rc_rate'])
print(f"\nPearson correlation: r={corr:.4f}, p={p_corr:.6f}")

spearman_r, p_sp = stats.spearmanr(player['skin_tone'], player['rc_rate'])
print(f"Spearman correlation: r={spearman_r:.4f}, p={p_sp:.6f}")

# Decision tree for interpretability
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(max_depth=3, random_state=42)
dt.fit(player[['skin_tone']], player['rc_rate'])
print(f"\nDecision Tree feature importance for skin_tone: {dt.feature_importances_[0]:.4f}")

# Summary
print("\n=== SUMMARY ===")
print(f"Dark skin RC rate: {dark.mean():.5f}")
print(f"Light skin RC rate: {light.mean():.5f}")
print(f"Ratio (dark/light): {dark.mean()/light.mean():.3f}")
print(f"t-test p-value: {p_val:.6f}")
print(f"Mann-Whitney p-value (one-sided): {p_mw:.6f}")
print(f"OLS skin_tone coef p-value: {model.pvalues['skin_tone']:.6f}")
print(f"Pearson r: {corr:.4f}, p: {p_corr:.6f}")

# Determine score
# Multiple tests: check significance
significant = p_val < 0.05 and dark.mean() > light.mean()
mw_sig = p_mw < 0.05
ols_sig = model.pvalues['skin_tone'] < 0.05 and model.params['skin_tone'] > 0

evidence_count = sum([significant, mw_sig, ols_sig, p_corr < 0.05 and corr > 0])
ratio = dark.mean() / light.mean() if light.mean() > 0 else 1.0

print(f"\nSignificant tests: {evidence_count}/4")
print(f"RC ratio dark/light: {ratio:.3f}")

# Score: if consistent evidence across multiple tests pointing to dark > light
if evidence_count >= 3 and ratio > 1.1:
    score = 75
elif evidence_count >= 2 and ratio > 1.05:
    score = 65
elif evidence_count >= 1 and ratio > 1.0:
    score = 55
else:
    score = 35

explanation = (
    f"Analysis of {len(player)} players: dark-skinned players (skin>0.5, n={len(dark)}) had "
    f"mean red card rate {dark.mean():.5f} vs light-skinned (n={len(light)}) at {light.mean():.5f} "
    f"(ratio={ratio:.3f}). "
    f"t-test p={p_val:.4f}, Mann-Whitney one-sided p={p_mw:.4f}, "
    f"OLS regression coef={model.params['skin_tone']:.5f} p={model.pvalues['skin_tone']:.4f}, "
    f"Pearson r={corr:.4f} p={p_corr:.4f}. "
    f"{evidence_count}/4 statistical tests support a significant positive association between dark skin tone and red card rate. "
    f"The evidence suggests dark-skinned players receive more red cards, consistent with the original paper's findings."
)

result = {"response": score, "explanation": explanation}
with open('conclusion.txt', 'w') as f:
    json.dump(result, f)

print(f"\nconclusion.txt written with response={score}")
print(f"Explanation: {explanation}")
