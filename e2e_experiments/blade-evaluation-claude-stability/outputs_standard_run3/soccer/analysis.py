import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import json

# Load data
df = pd.read_csv('soccer.csv')
print(f"Shape: {df.shape}")
print(df[['rater1', 'rater2', 'redCards']].describe())

# Create average skin tone (0=very light, 1=very dark)
df['skin_tone'] = (df['rater1'].fillna(df['rater2']) + df['rater2'].fillna(df['rater1'])) / 2
df_valid = df[df['skin_tone'].notna()].copy()
print(f"\nValid skin tone rows: {len(df_valid)}")

# Red card rate by skin tone group
df_valid['dark'] = (df_valid['skin_tone'] >= 0.5).astype(int)
dark = df_valid[df_valid['dark'] == 1]['redCards']
light = df_valid[df_valid['dark'] == 0]['redCards']
print(f"\nDark skin mean red cards: {dark.mean():.4f} (n={len(dark)})")
print(f"Light skin mean red cards: {light.mean():.4f} (n={len(light)})")

# T-test
t_stat, p_val = stats.ttest_ind(dark, light)
print(f"\nT-test: t={t_stat:.4f}, p={p_val:.4f}")

# Point-biserial correlation between skin tone and red cards
corr, p_corr = stats.pointbiserialr(df_valid['dark'], df_valid['redCards'])
print(f"Point-biserial r={corr:.4f}, p={p_corr:.4f}")

# Spearman correlation (skin_tone continuous vs redCards)
rho, p_rho = stats.spearmanr(df_valid['skin_tone'], df_valid['redCards'])
print(f"Spearman rho={rho:.4f}, p={p_rho:.4f}")

# OLS regression controlling for games played
X = df_valid[['skin_tone', 'games']].copy()
X = sm.add_constant(X)
y = df_valid['redCards']
model = sm.OLS(y, X).fit()
print("\nOLS results:")
print(model.summary().tables[1])

skin_coef = model.params['skin_tone']
skin_pval = model.pvalues['skin_tone']
print(f"\nSkin tone coef={skin_coef:.6f}, p={skin_pval:.4f}")

# Aggregate to player level to avoid dyad-level inflation
player_df = df_valid.groupby('playerShort').agg(
    skin_tone=('skin_tone', 'mean'),
    redCards=('redCards', 'sum'),
    games=('games', 'sum')
).reset_index()
player_df['rc_per_game'] = player_df['redCards'] / player_df['games']

rho2, p_rho2 = stats.spearmanr(player_df['skin_tone'], player_df['rc_per_game'])
print(f"\nPlayer-level Spearman rho={rho2:.4f}, p={p_rho2:.4f}")

t2, p2 = stats.ttest_ind(
    player_df[player_df['skin_tone'] >= 0.5]['rc_per_game'],
    player_df[player_df['skin_tone'] < 0.5]['rc_per_game']
)
print(f"Player-level t-test: t={t2:.4f}, p={p2:.4f}")

# Determine response score
# Evidence from multiple tests
significant_tests = sum([
    p_val < 0.05,
    p_corr < 0.05,
    p_rho < 0.05,
    skin_pval < 0.05,
    p_rho2 < 0.05,
    p2 < 0.05
])
dark_higher = dark.mean() > light.mean()
player_dark_higher = (player_df[player_df['skin_tone'] >= 0.5]['rc_per_game'].mean() >
                      player_df[player_df['skin_tone'] < 0.5]['rc_per_game'].mean())

print(f"\nSignificant tests: {significant_tests}/6")
print(f"Dark skin higher: {dark_higher}, player-level: {player_dark_higher}")

if significant_tests >= 4 and dark_higher:
    response = 75
    explanation = (f"Multiple statistical tests show a significant positive relationship between darker skin tone "
                   f"and red card rates. Dyad-level t-test p={p_val:.4f}, Spearman rho={rho:.4f} (p={p_rho:.4f}), "
                   f"OLS skin_tone coef={skin_coef:.4f} p={skin_pval:.4f}. "
                   f"Player-level analysis confirms: Spearman rho={rho2:.4f} p={p_rho2:.4f}. "
                   f"{significant_tests}/6 tests significant. Dark-skinned players receive more red cards.")
elif significant_tests >= 2 and dark_higher:
    response = 60
    explanation = (f"Some evidence that darker skin tone is associated with more red cards. "
                   f"Dyad-level t-test p={p_val:.4f}, Spearman rho={rho:.4f} (p={p_rho:.4f}), "
                   f"OLS skin_tone coef={skin_coef:.4f} p={skin_pval:.4f}. "
                   f"Player-level Spearman rho={rho2:.4f} p={p_rho2:.4f}. "
                   f"{significant_tests}/6 tests significant.")
elif not dark_higher or significant_tests == 0:
    response = 20
    explanation = (f"No significant relationship between skin tone and red cards. "
                   f"Dyad-level t-test p={p_val:.4f}, Spearman rho={rho:.4f} (p={p_rho:.4f}). "
                   f"Player-level Spearman rho={rho2:.4f} p={p_rho2:.4f}. "
                   f"{significant_tests}/6 tests significant.")
else:
    response = 40
    explanation = (f"Weak or mixed evidence. Dyad-level t-test p={p_val:.4f}, "
                   f"Spearman rho={rho:.4f} (p={p_rho:.4f}). "
                   f"OLS skin_tone coef={skin_coef:.4f} p={skin_pval:.4f}. "
                   f"Player-level Spearman rho={rho2:.4f} p={p_rho2:.4f}. "
                   f"{significant_tests}/6 tests significant.")

result = {"response": response, "explanation": explanation}
print(f"\nFinal result: {result}")

with open('conclusion.txt', 'w') as f:
    json.dump(result, f)

print("Written conclusion.txt")
