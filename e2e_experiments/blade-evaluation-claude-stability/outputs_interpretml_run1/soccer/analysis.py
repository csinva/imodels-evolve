import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('soccer.csv')
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Compute mean skin tone from two raters
df['skin_tone'] = (df['rater1'].fillna(0) + df['rater2'].fillna(0)) / 2
# Only keep rows where both raters provided ratings
df_rated = df.dropna(subset=['rater1', 'rater2']).copy()
df_rated['skin_tone'] = (df_rated['rater1'] + df_rated['rater2']) / 2

print(f"\nRated rows: {len(df_rated)}")
print(f"Skin tone distribution:\n{df_rated['skin_tone'].value_counts().sort_index()}")
print(f"\nRed card stats:\n{df_rated['redCards'].describe()}")

# Split into dark vs light skin tone (median split at midpoint 0.5)
# Scale: 0=very light, 1=very dark
dark = df_rated[df_rated['skin_tone'] >= 0.5]
light = df_rated[df_rated['skin_tone'] < 0.5]

print(f"\nDark skin players (tone >= 0.5): {len(dark)}, mean redCards: {dark['redCards'].mean():.4f}")
print(f"Light skin players (tone < 0.5): {len(light)}, mean redCards: {light['redCards'].mean():.4f}")

# Mann-Whitney U test (non-parametric, data is count/skewed)
stat_mw, p_mw = stats.mannwhitneyu(dark['redCards'], light['redCards'], alternative='greater')
print(f"\nMann-Whitney U (dark > light): stat={stat_mw:.2f}, p={p_mw:.6f}")

# T-test
stat_t, p_t = stats.ttest_ind(dark['redCards'], light['redCards'])
print(f"T-test: stat={stat_t:.4f}, p={p_t:.6f}")

# Pearson correlation
corr, p_corr = stats.pearsonr(df_rated['skin_tone'], df_rated['redCards'])
print(f"Pearson correlation (skin_tone vs redCards): r={corr:.4f}, p={p_corr:.6f}")

# Spearman correlation
rho, p_sp = stats.spearmanr(df_rated['skin_tone'], df_rated['redCards'])
print(f"Spearman correlation: rho={rho:.4f}, p={p_sp:.6f}")

# OLS regression controlling for games played
df_rated['has_red'] = (df_rated['redCards'] > 0).astype(int)
X = sm.add_constant(df_rated[['skin_tone', 'games']])
y = df_rated['redCards']
model = sm.OLS(y, X).fit()
print(f"\nOLS regression results:")
print(model.summary().tables[1])

# Logistic regression: probability of receiving any red card
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

X_lr = df_rated[['skin_tone', 'games']].values
y_lr = df_rated['has_red'].values
scaler = StandardScaler()
X_lr_scaled = scaler.fit_transform(X_lr)
lr = LogisticRegression()
lr.fit(X_lr_scaled, y_lr)
print(f"\nLogistic Regression coefficients (skin_tone, games): {lr.coef_[0]}")

# Red card rates by skin tone quartile
df_rated['skin_quartile'] = pd.qcut(df_rated['skin_tone'], q=4, duplicates='drop')
rate_by_quartile = df_rated.groupby('skin_quartile')['redCards'].mean()
print(f"\nMean red cards by skin tone quartile:\n{rate_by_quartile}")

# EBM from interpret
try:
    from interpret.glassbox import ExplainableBoostingRegressor
    feature_cols = ['skin_tone', 'games', 'yellowCards', 'goals']
    df_ebm = df_rated[feature_cols + ['redCards']].dropna()
    ebm = ExplainableBoostingRegressor(random_state=42, max_rounds=100)
    ebm.fit(df_ebm[feature_cols], df_ebm['redCards'])
    importances = dict(zip(feature_cols, ebm.term_importances()))
    print(f"\nEBM feature importances: {importances}")
except Exception as e:
    print(f"EBM failed: {e}")

# Summary
print("\n=== SUMMARY ===")
print(f"Dark skin mean red cards: {dark['redCards'].mean():.4f}")
print(f"Light skin mean red cards: {light['redCards'].mean():.4f}")
print(f"Ratio dark/light: {dark['redCards'].mean() / max(light['redCards'].mean(), 1e-9):.4f}")
print(f"Correlation r={corr:.4f}, p={p_corr:.6f}")
print(f"Mann-Whitney p={p_mw:.6f}")
print(f"T-test p={p_t:.6f}")

# Determine response score
# Use multiple lines of evidence
significant = p_corr < 0.05 or p_mw < 0.05 or p_t < 0.05
positive_direction = corr > 0 and dark['redCards'].mean() > light['redCards'].mean()

if significant and positive_direction:
    response = 75
    explanation = (
        f"Statistical analysis of {len(df_rated)} player-referee dyads with skin tone ratings shows "
        f"dark-skinned players (tone>=0.5) receive {dark['redCards'].mean():.4f} red cards on average vs "
        f"{light['redCards'].mean():.4f} for light-skinned players. "
        f"Pearson correlation r={corr:.4f} (p={p_corr:.4f}), Mann-Whitney U p={p_mw:.4f}, "
        f"T-test p={p_t:.4f}. The relationship is statistically significant and in the positive direction, "
        f"indicating darker-skinned players are more likely to receive red cards."
    )
elif significant and not positive_direction:
    response = 25
    explanation = (
        f"Statistically significant result but in the opposite direction: light-skinned players receive more red cards. "
        f"Correlation r={corr:.4f} (p={p_corr:.4f})."
    )
else:
    response = 30
    explanation = (
        f"No statistically significant relationship found between skin tone and red cards. "
        f"Correlation r={corr:.4f} (p={p_corr:.4f}), Mann-Whitney p={p_mw:.4f}."
    )

conclusion = {"response": response, "explanation": explanation}
with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)
print(f"\nConclusion written: {conclusion}")
