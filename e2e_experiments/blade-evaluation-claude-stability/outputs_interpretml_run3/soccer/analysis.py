import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv("soccer.csv")

# Create average skin tone from both raters
df["skin_tone"] = (df["rater1"].fillna(df["rater2"]) + df["rater2"].fillna(df["rater1"])) / 2
# If both raters available, average them; else use whichever is available
mask_both = df["rater1"].notna() & df["rater2"].notna()
df.loc[mask_both, "skin_tone"] = (df.loc[mask_both, "rater1"] + df.loc[mask_both, "rater2"]) / 2
df.loc[~mask_both & df["rater1"].notna(), "skin_tone"] = df.loc[~mask_both & df["rater1"].notna(), "rater1"]
df.loc[~mask_both & df["rater2"].notna(), "skin_tone"] = df.loc[~mask_both & df["rater2"].notna(), "rater2"]

# Drop rows without skin tone data
df_clean = df.dropna(subset=["skin_tone", "redCards", "games"]).copy()
df_clean = df_clean[df_clean["games"] > 0]

# Red card rate per game
df_clean["redcard_rate"] = df_clean["redCards"] / df_clean["games"]

# Binary: dark vs light (median split or threshold)
median_tone = df_clean["skin_tone"].median()
df_clean["dark_skin"] = (df_clean["skin_tone"] > median_tone).astype(int)

print(f"Dataset: {len(df_clean)} dyads with skin tone data")
print(f"Skin tone median: {median_tone:.3f}")
print(f"Dark skin dyads: {df_clean['dark_skin'].sum()}, Light skin: {(df_clean['dark_skin']==0).sum()}")
print(f"\nOverall red card rate: {df_clean['redcard_rate'].mean():.4f}")

# Group statistics
dark = df_clean[df_clean["dark_skin"] == 1]
light = df_clean[df_clean["dark_skin"] == 0]
print(f"\nDark skin red card rate: {dark['redcard_rate'].mean():.4f} (n={len(dark)})")
print(f"Light skin red card rate: {light['redcard_rate'].mean():.4f} (n={len(light)})")

# Total red cards per group (aggregate by player)
player_agg = df_clean.groupby(["playerShort", "dark_skin"]).agg(
    total_redcards=("redCards", "sum"),
    total_games=("games", "sum")
).reset_index()
player_agg["rc_rate"] = player_agg["total_redcards"] / player_agg["total_games"]

dark_players = player_agg[player_agg["dark_skin"] == 1]["rc_rate"]
light_players = player_agg[player_agg["dark_skin"] == 0]["rc_rate"]
print(f"\nPlayer-level dark skin mean rc_rate: {dark_players.mean():.4f} (n={len(dark_players)})")
print(f"Player-level light skin mean rc_rate: {light_players.mean():.4f} (n={len(light_players)})")

# T-test at player level
t_stat, p_value = stats.ttest_ind(dark_players, light_players)
print(f"\nT-test (player-level): t={t_stat:.4f}, p={p_value:.4f}")

# Mann-Whitney U test (non-parametric)
u_stat, p_mwu = stats.mannwhitneyu(dark_players, light_players, alternative="greater")
print(f"Mann-Whitney U (dark > light): U={u_stat:.1f}, p={p_mwu:.4f}")

# OLS regression: redcard_rate ~ skin_tone (dyad level)
X = sm.add_constant(df_clean["skin_tone"])
model_ols = sm.OLS(df_clean["redcard_rate"], X).fit()
print(f"\nOLS regression (dyad level):")
print(f"  skin_tone coef: {model_ols.params['skin_tone']:.6f}, p={model_ols.pvalues['skin_tone']:.4f}")

# OLS with games as weight
model_wls = sm.WLS(df_clean["redcard_rate"], X, weights=df_clean["games"]).fit()
print(f"WLS regression (weighted by games):")
print(f"  skin_tone coef: {model_wls.params['skin_tone']:.6f}, p={model_wls.pvalues['skin_tone']:.4f}")

# Logistic regression: P(any red card) ~ skin_tone
df_clean["got_redcard"] = (df_clean["redCards"] > 0).astype(int)
X2 = sm.add_constant(df_clean["skin_tone"])
logit_model = sm.Logit(df_clean["got_redcard"], X2).fit(disp=0)
print(f"\nLogistic regression (any red card ~ skin_tone):")
print(f"  skin_tone coef: {logit_model.params['skin_tone']:.4f}, p={logit_model.pvalues['skin_tone']:.4f}")
print(f"  Odds ratio: {np.exp(logit_model.params['skin_tone']):.4f}")

# Poisson regression on player-level totals
player_agg2 = df_clean.groupby(["playerShort", "dark_skin", "skin_tone"]).agg(
    total_rc=("redCards", "sum"),
    total_games=("games", "sum")
).reset_index()
X3 = sm.add_constant(player_agg2["skin_tone"])
poisson_model = sm.GLM(
    player_agg2["total_rc"],
    X3,
    family=sm.families.Poisson(),
    exposure=player_agg2["total_games"]
).fit()
print(f"\nPoisson regression (player-level, offset=log(games)):")
print(f"  skin_tone coef: {poisson_model.params['skin_tone']:.4f}, p={poisson_model.pvalues['skin_tone']:.4f}")
print(f"  IRR: {np.exp(poisson_model.params['skin_tone']):.4f}")

# Summary of findings
p_ttest = p_value
p_logit = logit_model.pvalues["skin_tone"]
p_poisson = poisson_model.pvalues["skin_tone"]
p_mwu_val = p_mwu
or_logit = np.exp(logit_model.params["skin_tone"])
irr_poisson = np.exp(poisson_model.params["skin_tone"])

sig_count = sum([p_ttest < 0.05, p_mwu_val < 0.05, p_logit < 0.05, p_poisson < 0.05])
print(f"\nSignificant tests (p<0.05): {sig_count}/4")

# Determine response score
# Multiple tests show direction and significance
if sig_count >= 3 and or_logit > 1 and irr_poisson > 1:
    response = 75
    explanation = (
        f"Multiple statistical tests consistently show that darker-skinned players receive more red cards. "
        f"Logistic regression: OR={or_logit:.3f}, p={p_logit:.4f}; "
        f"Poisson regression: IRR={irr_poisson:.3f}, p={p_poisson:.4f}; "
        f"Mann-Whitney p={p_mwu_val:.4f}; T-test p={p_ttest:.4f}. "
        f"{sig_count}/4 tests significant. The effect is modest but consistent across methods, "
        f"supporting that darker-skinned players are somewhat more likely to receive red cards."
    )
elif sig_count >= 2 and or_logit > 1:
    response = 65
    explanation = (
        f"Some statistical tests show darker-skinned players receive more red cards. "
        f"Logistic regression: OR={or_logit:.3f}, p={p_logit:.4f}; "
        f"Poisson regression: IRR={irr_poisson:.3f}, p={p_poisson:.4f}; "
        f"Mann-Whitney p={p_mwu_val:.4f}; T-test p={p_ttest:.4f}. "
        f"{sig_count}/4 tests significant. Evidence suggests a positive but modest relationship."
    )
elif sig_count >= 1 and or_logit > 1:
    response = 55
    explanation = (
        f"Weak evidence that darker-skinned players receive more red cards. "
        f"Logistic regression: OR={or_logit:.3f}, p={p_logit:.4f}; "
        f"Poisson regression: IRR={irr_poisson:.3f}, p={p_poisson:.4f}. "
        f"Only {sig_count}/4 tests significant. Effect direction is positive but inconsistent."
    )
else:
    response = 35
    explanation = (
        f"Insufficient evidence to conclude darker-skinned players receive more red cards. "
        f"Logistic regression: OR={or_logit:.3f}, p={p_logit:.4f}; "
        f"Poisson regression: IRR={irr_poisson:.3f}, p={p_poisson:.4f}. "
        f"Only {sig_count}/4 tests significant."
    )

result = {"response": response, "explanation": explanation}
print(f"\nConclusion: {json.dumps(result, indent=2)}")

with open("conclusion.txt", "w") as f:
    json.dump(result, f)

print("\nconclusion.txt written.")
