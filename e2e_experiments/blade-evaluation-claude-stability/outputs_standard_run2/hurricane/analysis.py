import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor

# Load data
df = pd.read_csv("hurricane.csv")

print("Shape:", df.shape)
print(df.describe())
print("\nCorrelation with alldeaths:")
print(df[["masfem", "gender_mf", "alldeaths", "category", "wind", "ndam", "min"]].corr()["alldeaths"])

# Research question: Do more feminine-named hurricanes cause more deaths?
# Key variables: masfem (femininity index), alldeaths

# 1. Correlation test
r, p_corr = stats.pearsonr(df["masfem"], df["alldeaths"])
print(f"\nPearson r(masfem, alldeaths) = {r:.4f}, p = {p_corr:.4f}")

r_spear, p_spear = stats.spearmanr(df["masfem"], df["alldeaths"])
print(f"Spearman r(masfem, alldeaths) = {r_spear:.4f}, p = {p_spear:.4f}")

# 2. Binary gender t-test
male = df[df["gender_mf"] == 0]["alldeaths"]
female = df[df["gender_mf"] == 1]["alldeaths"]
t_stat, p_ttest = stats.ttest_ind(female, male)
print(f"\nMean deaths male: {male.mean():.2f}, female: {female.mean():.2f}")
print(f"t-test: t={t_stat:.4f}, p={p_ttest:.4f}")

# 3. OLS regression controlling for storm severity
cols = ["masfem", "category", "wind", "ndam", "min"]
df_clean = df[cols + ["alldeaths"]].dropna()
X = df_clean[cols].copy()
X = sm.add_constant(X)
y = df_clean["alldeaths"]
model = sm.OLS(y, X).fit()
print("\nOLS regression results:")
print(model.summary())

# 4. Log-transformed deaths (deaths are highly skewed)
df_clean["log_deaths"] = np.log1p(df_clean["alldeaths"])
r_log, p_log = stats.pearsonr(df_clean["masfem"], df_clean["log_deaths"])
print(f"\nPearson r(masfem, log(1+alldeaths)) = {r_log:.4f}, p = {p_log:.4f}")

# OLS with log deaths
X2 = df_clean[cols].copy()
X2 = sm.add_constant(X2)
y2 = df_clean["log_deaths"]
model2 = sm.OLS(y2, X2).fit()
print("\nOLS with log deaths:")
print(model2.summary())
masfem_coef = model2.params["masfem"]
masfem_pval = model2.pvalues["masfem"]
print(f"\nmasfem coef (controlled): {masfem_coef:.4f}, p={masfem_pval:.4f}")

# 5. Decision tree for feature importance
from sklearn.preprocessing import StandardScaler
feats = ["masfem", "category", "wind", "ndam", "min", "gender_mf"]
Xdt = df[feats].fillna(df[feats].median())
ydt = df["alldeaths"]
dt = DecisionTreeRegressor(max_depth=3, random_state=42)
dt.fit(Xdt, ydt)
for feat, imp in sorted(zip(feats, dt.feature_importances_), key=lambda x: -x[1]):
    print(f"  {feat}: {imp:.4f}")

# Summarize evidence
# The original paper (Jung et al. 2014) claimed feminine names -> more deaths
# But reanalysis showed this was driven by a few high-death storms and may be spurious
# Simonsohn's specification curve showed results not robust

# Key finding: correlation of masfem with alldeaths
# p-value from controlled regression for masfem
evidence_for = []
evidence_against = []

if p_corr < 0.05 and r > 0:
    evidence_for.append(f"Positive Pearson correlation (r={r:.3f}, p={p_corr:.3f})")
else:
    evidence_against.append(f"Pearson correlation not significant (r={r:.3f}, p={p_corr:.3f})")

if p_spear < 0.05 and r_spear > 0:
    evidence_for.append(f"Positive Spearman correlation (r={r_spear:.3f}, p={p_spear:.3f})")
else:
    evidence_against.append(f"Spearman correlation not significant (r={r_spear:.3f}, p={p_spear:.3f})")

if p_ttest < 0.05 and female.mean() > male.mean():
    evidence_for.append(f"Female hurricanes have more deaths (t-test p={p_ttest:.3f})")
else:
    evidence_against.append(f"Binary gender difference not significant (p={p_ttest:.3f})")

if masfem_pval < 0.05 and masfem_coef > 0:
    evidence_for.append(f"masfem significant in controlled OLS (coef={masfem_coef:.3f}, p={masfem_pval:.3f})")
else:
    evidence_against.append(f"masfem NOT significant when controlling for storm severity (coef={masfem_coef:.3f}, p={masfem_pval:.3f})")

print("\nEvidence FOR:", evidence_for)
print("Evidence AGAINST:", evidence_against)

# Score: the claim lacks robust support when controlling for severity
# Raw correlation may exist but disappears when controlling for storm intensity
# Score around 25-35: weak/not robust evidence
n_for = len(evidence_for)
n_against = len(evidence_against)

# The key test is the controlled regression; if masfem is not significant there,
# the claim is not supported
if masfem_pval >= 0.05:
    score = 20  # not significant when controlling for confounders
    explanation = (
        f"The research question asks whether more feminine hurricane names lead to fewer "
        f"precautionary measures (proxied by higher death tolls). "
        f"Raw Pearson correlation between femininity (masfem) and deaths: r={r:.3f}, p={p_corr:.3f}. "
        f"Binary gender t-test: female mean deaths={female.mean():.1f} vs male={male.mean():.1f}, p={p_ttest:.3f}. "
        f"However, when controlling for storm severity (category, wind, pressure, damage) via OLS regression on log-deaths, "
        f"masfem coefficient={masfem_coef:.3f} with p={masfem_pval:.3f} — NOT statistically significant. "
        f"Storm severity dominates death tolls. The femininity-deaths relationship is not robust to "
        f"controlling for confounders, consistent with Simonsohn et al.'s specification curve reanalysis "
        f"showing the original Jung et al. (2014) finding was not robust. Score: 20 (weak/no support)."
    )
else:
    score = 65
    explanation = (
        f"masfem is significant (coef={masfem_coef:.3f}, p={masfem_pval:.3f}) even after controlling "
        f"for storm severity. Raw Pearson r={r:.3f} (p={p_corr:.3f}). Some support for the claim."
    )

print(f"\nFinal score: {score}")
print(f"Explanation: {explanation}")

result = {"response": score, "explanation": explanation}
with open("conclusion.txt", "w") as f:
    json.dump(result, f)

print("\nconclusion.txt written.")
