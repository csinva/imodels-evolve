import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("hurricane.csv")
print("Shape:", df.shape)
print(df.describe())
print("\nCorrelations with alldeaths:")
print(df[["masfem", "gender_mf", "alldeaths", "category", "ndam", "wind", "min"]].corr()["alldeaths"])

# Basic correlation: femininity vs deaths
r_masfem, p_masfem = stats.pearsonr(df["masfem"].dropna(), df.loc[df["masfem"].notna(), "alldeaths"])
print(f"\nPearson r(masfem, alldeaths) = {r_masfem:.4f}, p = {p_masfem:.4f}")

r_mturk, p_mturk = stats.pearsonr(df["masfem_mturk"].dropna(), df.loc[df["masfem_mturk"].notna(), "alldeaths"])
print(f"Pearson r(masfem_mturk, alldeaths) = {r_mturk:.4f}, p = {p_mturk:.4f}")

# t-test: male vs female named hurricanes
male = df[df["gender_mf"] == 0]["alldeaths"]
female = df[df["gender_mf"] == 1]["alldeaths"]
t_stat, p_ttest = stats.ttest_ind(male, female)
print(f"\nt-test (male vs female deaths): t={t_stat:.4f}, p={p_ttest:.4f}")
print(f"Male mean={male.mean():.2f}, Female mean={female.mean():.2f}")

# OLS controlling for storm severity
cols = ["masfem", "category", "wind", "min", "ndam", "alldeaths"]
df_clean = df[cols].dropna()
X = df_clean[["masfem", "category", "wind", "min", "ndam"]]
X = sm.add_constant(X)
y = df_clean["alldeaths"]
model = sm.OLS(y, X).fit()
print("\nOLS controlling for storm severity:")
print(model.summary())

# Log transform deaths (heavy skew)
df["log_deaths"] = np.log1p(df["alldeaths"])
df_clean2 = df[["masfem", "category", "wind", "min", "ndam", "log_deaths"]].dropna()
X2 = sm.add_constant(df_clean2[["masfem", "category", "wind", "min", "ndam"]])
y2 = df_clean2["log_deaths"]
model2 = sm.OLS(y2, X2).fit()
print("\nOLS with log(deaths+1):")
print(model2.summary())

masfem_coef = model2.params["masfem"]
masfem_p = model2.pvalues["masfem"]
print(f"\nmasfem coef in log model: {masfem_coef:.4f}, p={masfem_p:.4f}")

# Spearman correlation (robust to outliers)
rho, p_spearman = stats.spearmanr(df["masfem"].dropna(), df.loc[df["masfem"].notna(), "alldeaths"])
print(f"\nSpearman r(masfem, alldeaths) = {rho:.4f}, p = {p_spearman:.4f}")

# Summarize evidence
# Positive correlation (more feminine -> more deaths) was the original claim but controversial
# We check if the effect is statistically significant after controlling for severity
evidence_for = []
evidence_against = []

if p_masfem < 0.05:
    if r_masfem > 0:
        evidence_for.append(f"Positive Pearson correlation r={r_masfem:.3f}, p={p_masfem:.4f}")
    else:
        evidence_against.append(f"Negative Pearson correlation r={r_masfem:.3f}, p={p_masfem:.4f}")
else:
    evidence_against.append(f"No significant Pearson correlation (r={r_masfem:.3f}, p={p_masfem:.4f})")

if masfem_p < 0.05:
    if masfem_coef > 0:
        evidence_for.append(f"Significant positive effect in controlled OLS (coef={masfem_coef:.3f}, p={masfem_p:.4f})")
    else:
        evidence_against.append(f"Significant negative effect in controlled OLS (coef={masfem_coef:.3f}, p={masfem_p:.4f})")
else:
    evidence_against.append(f"No significant masfem effect after controlling for severity (p={masfem_p:.4f})")

print("\nEvidence for hypothesis:", evidence_for)
print("Evidence against hypothesis:", evidence_against)

# Score: the original paper's finding is controversial; most replications find no effect after proper controls
# We base score on the controlled analysis (log deaths OLS with severity controls)
if masfem_p < 0.05 and masfem_coef > 0:
    score = 65
    explanation = (
        f"After controlling for storm severity (category, wind, pressure, damage), masfem shows a "
        f"significant positive association with log(deaths+1) (coef={masfem_coef:.3f}, p={masfem_p:.4f}). "
        f"Raw Pearson r={r_masfem:.3f} (p={p_masfem:.4f}). This supports the hypothesis that more feminine "
        f"hurricane names are associated with more deaths, consistent with reduced precautionary behavior. "
        f"However, effect size is modest and the finding is controversial in the literature."
    )
elif masfem_p < 0.05 and masfem_coef < 0:
    score = 20
    explanation = (
        f"Controlled OLS shows a significant negative effect of femininity on deaths (coef={masfem_coef:.3f}, p={masfem_p:.4f}), "
        f"opposite to the hypothesis. Raw correlation r={r_masfem:.3f} (p={p_masfem:.4f})."
    )
else:
    score = 25
    explanation = (
        f"After controlling for storm severity, there is no statistically significant effect of hurricane name femininity on deaths "
        f"(masfem coef={masfem_coef:.3f}, p={masfem_p:.4f}). Raw Pearson r={r_masfem:.3f} (p={p_masfem:.4f}). "
        f"t-test male vs female: t={t_stat:.3f}, p={p_ttest:.4f} (male mean={male.mean():.1f}, female mean={female.mean():.1f}). "
        f"The evidence does not support the hypothesis that more feminine hurricane names lead to fewer precautionary measures resulting in more deaths."
    )

result = {"response": score, "explanation": explanation}
with open("conclusion.txt", "w") as f:
    json.dump(result, f)

print(f"\nconclusion.txt written with score={score}")
print(explanation)
