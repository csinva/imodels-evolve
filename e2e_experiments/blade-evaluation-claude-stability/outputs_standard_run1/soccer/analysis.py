import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import json
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('soccer.csv')
print(f"Shape: {df.shape}")
print(df[['rater1', 'rater2', 'redCards']].describe())

# Average skin tone from both raters
df['skin_tone'] = (df['rater1'] + df['rater2']) / 2

# Drop rows where skin tone is missing
df_skin = df.dropna(subset=['skin_tone', 'redCards'])
print(f"\nRows with skin tone data: {len(df_skin)}")

# Binary: dark skin (skin_tone > 0.5) vs light skin (skin_tone <= 0.5)
dark = df_skin[df_skin['skin_tone'] > 0.5]['redCards']
light = df_skin[df_skin['skin_tone'] <= 0.5]['redCards']

print(f"\nLight skin dyads: {len(light)}, mean red cards: {light.mean():.4f}")
print(f"Dark skin dyads: {len(dark)}, mean red cards: {dark.mean():.4f}")

# Mann-Whitney U test (non-parametric, since redCards is count/skewed)
u_stat, p_val_mw = stats.mannwhitneyu(dark, light, alternative='greater')
print(f"\nMann-Whitney U (dark > light): U={u_stat:.1f}, p={p_val_mw:.4f}")

# t-test
t_stat, p_val_t = stats.ttest_ind(dark, light)
print(f"Independent t-test: t={t_stat:.4f}, p={p_val_t:.4f}")

# Red card rate (any red card in dyad)
dark_rate = (dark > 0).mean()
light_rate = (light > 0).mean()
print(f"\nLight skin red card rate: {light_rate:.4f}")
print(f"Dark skin red card rate: {dark_rate:.4f}")

# Chi-square test on having any red card
from scipy.stats import chi2_contingency
dark_rc = (dark > 0).sum()
dark_no_rc = (dark == 0).sum()
light_rc = (light > 0).sum()
light_no_rc = (light == 0).sum()
ct = np.array([[dark_rc, dark_no_rc], [light_rc, light_no_rc]])
chi2, p_chi2, dof, expected = chi2_contingency(ct)
print(f"\nChi-square test: chi2={chi2:.4f}, p={p_chi2:.4f}")

# OLS regression: redCards ~ skin_tone (controlling for games played)
df_reg = df_skin.dropna(subset=['games'])
X = sm.add_constant(df_reg[['skin_tone', 'games']])
y = df_reg['redCards']
model = sm.OLS(y, X).fit()
print("\nOLS regression summary:")
print(model.summary().tables[1])

skin_coef = model.params['skin_tone']
skin_pval = model.pvalues['skin_tone']
print(f"\nSkin tone coefficient: {skin_coef:.6f}, p-value: {skin_pval:.4f}")

# Aggregate by player (avoid dyad-level duplication)
player_agg = df_skin.groupby('playerShort').agg(
    skin_tone=('skin_tone', 'mean'),
    total_redCards=('redCards', 'sum'),
    total_games=('games', 'sum')
).dropna()
player_agg['redCard_rate'] = player_agg['total_redCards'] / player_agg['total_games']

print(f"\nUnique players with skin data: {len(player_agg)}")

# Correlation at player level
r, p_r = stats.pearsonr(player_agg['skin_tone'], player_agg['redCard_rate'])
print(f"Pearson r (skin_tone vs red_card_rate per player): r={r:.4f}, p={p_r:.4f}")

sp_r, sp_p = stats.spearmanr(player_agg['skin_tone'], player_agg['redCard_rate'])
print(f"Spearman r (skin_tone vs red_card_rate per player): r={sp_r:.4f}, p={sp_p:.4f}")

# Determine conclusion
# Use multiple evidence points
evidence_yes = 0
evidence_total = 0

# Mann-Whitney
evidence_total += 1
if p_val_mw < 0.05 and dark.mean() > light.mean():
    evidence_yes += 1

# t-test (one-sided)
evidence_total += 1
if p_val_t < 0.05 and dark.mean() > light.mean():
    evidence_yes += 1

# Chi-square
evidence_total += 1
if p_chi2 < 0.05 and dark_rate > light_rate:
    evidence_yes += 1

# OLS
evidence_total += 1
if skin_pval < 0.05 and skin_coef > 0:
    evidence_yes += 1

# Player-level correlation
evidence_total += 1
if p_r < 0.05 and r > 0:
    evidence_yes += 1

print(f"\nEvidence score: {evidence_yes}/{evidence_total}")

# Score: weight the evidence
# Key stats for conclusion
mean_diff = dark.mean() - light.mean()
print(f"Mean red card difference (dark - light): {mean_diff:.4f}")

# Final score determination
if evidence_yes >= 4:
    score = 80
elif evidence_yes == 3:
    score = 65
elif evidence_yes == 2:
    score = 50
elif evidence_yes == 1:
    score = 35
else:
    score = 20

# Adjust based on effect size
# The effect size is likely small given the base rates are low
if abs(mean_diff) < 0.001:
    score = max(score - 10, 0)

explanation = (
    f"Analysis of {len(df_skin)} player-referee dyads with skin tone data. "
    f"Dark skin players (skin_tone>0.5, n={len(dark)}) averaged {dark.mean():.4f} red cards per dyad vs "
    f"light skin players (n={len(light)}) averaging {light.mean():.4f}. "
    f"Mann-Whitney U test (dark>light): p={p_val_mw:.4f}. "
    f"Chi-square on any red card: chi2={chi2:.4f}, p={p_chi2:.4f}. "
    f"OLS regression coefficient for skin_tone (controlling for games): {skin_coef:.6f}, p={skin_pval:.4f}. "
    f"Player-level Pearson correlation: r={r:.4f}, p={p_r:.4f}. "
    f"Evidence score: {evidence_yes}/{evidence_total} tests significant in expected direction. "
    f"Overall, there is {'a statistically significant' if evidence_yes >= 3 else 'weak or no'} association "
    f"between darker skin tone and receiving more red cards."
)

result = {"response": score, "explanation": explanation}
print(f"\nFinal result: {result}")

with open('conclusion.txt', 'w') as f:
    json.dump(result, f)

print("conclusion.txt written.")
