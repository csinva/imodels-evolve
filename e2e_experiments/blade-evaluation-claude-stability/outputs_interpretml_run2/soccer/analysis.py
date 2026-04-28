import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv('soccer.csv')

# Create skin tone variable (average of two raters)
df['skin_tone'] = (df['rater1'] + df['rater2']) / 2
df_valid = df.dropna(subset=['skin_tone', 'redCards'])

# Binary outcome: received at least one red card
df_valid = df_valid.copy()
df_valid['got_red'] = (df_valid['redCards'] > 0).astype(int)

# Split into light (skin_tone <= 0.5) and dark (skin_tone > 0.5)
light = df_valid[df_valid['skin_tone'] <= 0.5]
dark = df_valid[df_valid['skin_tone'] > 0.5]

print(f"Total dyads with skin data: {len(df_valid)}")
print(f"Light skin dyads: {len(light)}, red card rate: {light['got_red'].mean():.4f}")
print(f"Dark skin dyads: {len(dark)}, red card rate: {dark['got_red'].mean():.4f}")

# Chi-square test on binary red card outcome
ct = pd.crosstab(df_valid['got_red'], pd.cut(df_valid['skin_tone'], bins=[-0.01, 0.5, 1.01], labels=['light', 'dark']))
chi2, p_chi2, dof, expected = stats.chi2_contingency(ct)
print(f"\nChi-square test: chi2={chi2:.4f}, p={p_chi2:.6f}")

# Point-biserial correlation between skin tone and red cards
corr, p_corr = stats.pointbiserialr(df_valid['skin_tone'], df_valid['redCards'])
print(f"Correlation (skin_tone vs redCards): r={corr:.4f}, p={p_corr:.6f}")

# Aggregate by player to avoid dyad-level inflation
player_agg = df_valid.groupby('playerShort').agg(
    skin_tone=('skin_tone', 'mean'),
    total_redCards=('redCards', 'sum'),
    total_games=('games', 'sum')
).reset_index()
player_agg['red_rate'] = player_agg['total_redCards'] / player_agg['total_games']

light_p = player_agg[player_agg['skin_tone'] <= 0.5]['red_rate']
dark_p = player_agg[player_agg['skin_tone'] > 0.5]['red_rate']

t_stat, p_t = stats.ttest_ind(dark_p, light_p)
print(f"\nPlayer-level t-test (dark vs light red rate): t={t_stat:.4f}, p={p_t:.6f}")
print(f"Dark mean rate: {dark_p.mean():.4f}, Light mean rate: {light_p.mean():.4f}")

# OLS regression at player level
X = sm.add_constant(player_agg['skin_tone'])
model = sm.OLS(player_agg['red_rate'], X).fit()
print(f"\nOLS: skin_tone coef={model.params['skin_tone']:.4f}, p={model.pvalues['skin_tone']:.6f}")

# Logistic regression at dyad level controlling for games
X_log = df_valid[['skin_tone', 'games']].fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_log)
lr = LogisticRegression(max_iter=1000)
lr.fit(X_scaled, df_valid['got_red'])
print(f"\nLogistic coefs (skin_tone, games): {lr.coef_[0]}")

# Summary
significant = p_chi2 < 0.05 and p_corr < 0.05
print(f"\nSignificant association: {significant}")
print(f"Dark skin players have higher red card rate: {dark_p.mean() > light_p.mean()}")

# Determine response score
# Use chi2 p-value, correlation, t-test results
# Both p values significant → moderate to strong yes
if p_chi2 < 0.001 and p_t < 0.05:
    response = 72
    explanation = (
        f"Statistical tests show a significant positive association between darker skin tone and red card rates. "
        f"Chi-square test: chi2={chi2:.2f}, p={p_chi2:.4f}. "
        f"Player-level t-test comparing dark vs light skin players: t={t_stat:.3f}, p={p_t:.4f}. "
        f"Dark-skinned players received red cards at rate {dark_p.mean():.4f} vs {light_p.mean():.4f} for light-skinned. "
        f"OLS regression skin_tone coefficient={model.params['skin_tone']:.4f} (p={model.pvalues['skin_tone']:.4f}). "
        f"The evidence suggests dark skin tone is associated with higher red card probability, supporting a 'Yes' answer."
    )
elif p_chi2 < 0.05 or p_t < 0.05:
    response = 60
    explanation = (
        f"Some statistical tests show a significant association between darker skin tone and red card rates. "
        f"Chi-square test: chi2={chi2:.2f}, p={p_chi2:.4f}. "
        f"Player-level t-test: t={t_stat:.3f}, p={p_t:.4f}. "
        f"Dark-skinned players received red cards at rate {dark_p.mean():.4f} vs {light_p.mean():.4f} for light-skinned."
    )
else:
    response = 35
    explanation = (
        f"Statistical tests do not show a significant association between skin tone and red cards. "
        f"Chi-square test: chi2={chi2:.2f}, p={p_chi2:.4f}. "
        f"Player-level t-test: t={t_stat:.3f}, p={p_t:.4f}. "
        f"Dark-skinned players received red cards at rate {dark_p.mean():.4f} vs {light_p.mean():.4f} for light-skinned."
    )

result = {"response": response, "explanation": explanation}
with open('conclusion.txt', 'w') as f:
    json.dump(result, f)

print(f"\nWritten conclusion.txt with response={response}")
