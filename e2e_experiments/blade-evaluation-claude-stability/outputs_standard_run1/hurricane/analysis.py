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
print(df[["masfem", "gender_mf", "category", "wind", "min", "ndam", "alldeaths"]].corr()["alldeaths"])

# Basic correlation between femininity and deaths
r, p = stats.pearsonr(df["masfem"], df["alldeaths"])
print(f"\nPearson r(masfem, alldeaths) = {r:.4f}, p = {p:.4f}")

r2, p2 = stats.pointbiserialr(df["gender_mf"], df["alldeaths"])
print(f"Point-biserial r(gender_mf, alldeaths) = {r2:.4f}, p = {p2:.4f}")

# t-test: male vs female named hurricanes
male_deaths = df[df["gender_mf"] == 0]["alldeaths"]
female_deaths = df[df["gender_mf"] == 1]["alldeaths"]
t, pt = stats.ttest_ind(female_deaths, male_deaths)
print(f"\nt-test female vs male deaths: t={t:.4f}, p={pt:.4f}")
print(f"Female mean={female_deaths.mean():.2f}, Male mean={male_deaths.mean():.2f}")

# OLS regression: deaths ~ masfem controlling for storm intensity
cols = ["masfem", "wind", "min", "ndam", "alldeaths"]
dfc = df[cols].dropna()
X = dfc[["masfem", "wind", "min", "ndam"]].copy()
X = sm.add_constant(X)
y = dfc["alldeaths"]
ols = sm.OLS(y, X).fit()
print("\nOLS (deaths ~ masfem + wind + min + ndam):")
print(ols.summary().tables[1])

# Log deaths (common approach for heavy-tailed fatality data)
dfc["log_deaths"] = np.log1p(dfc["alldeaths"])
X2 = dfc[["masfem", "wind", "min", "ndam"]].copy()
X2 = sm.add_constant(X2)
ols2 = sm.OLS(dfc["log_deaths"], X2).fit()
print("\nOLS log(deaths+1) ~ masfem + wind + min + ndam:")
print(ols2.summary().tables[1])
masfem_pval = ols2.pvalues["masfem"]
masfem_coef = ols2.params["masfem"]
print(f"\nmasfem coef={masfem_coef:.4f}, p={masfem_pval:.4f}")

# Spearman correlation
rsp, psp = stats.spearmanr(df["masfem"], df["alldeaths"])
print(f"\nSpearman r(masfem, alldeaths) = {rsp:.4f}, p = {psp:.4f}")

# Decision: is the relationship statistically significant?
# We look at multiple evidence points
evidence_scores = []

# Raw Pearson
if p < 0.05 and r > 0:
    evidence_scores.append(70)
elif p < 0.10 and r > 0:
    evidence_scores.append(50)
else:
    evidence_scores.append(20)

# Controlled OLS (log deaths)
if masfem_pval < 0.05 and masfem_coef > 0:
    evidence_scores.append(75)
elif masfem_pval < 0.10 and masfem_coef > 0:
    evidence_scores.append(55)
else:
    evidence_scores.append(20)

# Spearman
if psp < 0.05 and rsp > 0:
    evidence_scores.append(70)
elif psp < 0.10 and rsp > 0:
    evidence_scores.append(50)
else:
    evidence_scores.append(20)

response = int(np.mean(evidence_scores))
print(f"\nEvidence scores: {evidence_scores}, mean={response}")

# Build explanation
explanation = (
    f"Analysis of 94 US landfalling hurricanes (1950-2012). "
    f"Pearson correlation between femininity (masfem) and deaths: r={r:.3f}, p={p:.3f}. "
    f"Spearman correlation: r={rsp:.3f}, p={psp:.3f}. "
    f"OLS regression (log deaths ~ masfem + wind + min + ndam): masfem coef={masfem_coef:.3f}, p={masfem_pval:.3f}. "
    f"t-test female vs male named hurricanes: female mean deaths={female_deaths.mean():.1f}, male mean={male_deaths.mean():.1f}, p={pt:.3f}. "
    f"The evidence {'supports' if response >= 50 else 'does not support'} the hypothesis that more feminine-named hurricanes cause more deaths, "
    f"suggesting people perceive them as less threatening and take fewer precautions. "
    f"However, the effect is sensitive to model specification and the original finding has been contested in the literature."
)

result = {"response": response, "explanation": explanation}
print("\nResult:", json.dumps(result, indent=2))

with open("conclusion.txt", "w") as f:
    json.dump(result, f)
print("conclusion.txt written.")
