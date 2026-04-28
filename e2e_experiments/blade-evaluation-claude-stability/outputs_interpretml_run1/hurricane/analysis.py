import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("hurricane.csv")

print("Shape:", df.shape)
print(df[["masfem", "gender_mf", "alldeaths", "category", "ndam", "wind", "min"]].describe())

# Correlation between femininity (masfem) and deaths
corr, pval = stats.pearsonr(df["masfem"], df["alldeaths"])
print(f"\nPearson corr masfem vs alldeaths: r={corr:.4f}, p={pval:.4f}")

# Spearman (robust to outliers)
scorr, spval = stats.spearmanr(df["masfem"], df["alldeaths"])
print(f"Spearman corr masfem vs alldeaths: r={scorr:.4f}, p={spval:.4f}")

# Binary gender comparison
female = df[df["gender_mf"] == 1]["alldeaths"]
male = df[df["gender_mf"] == 0]["alldeaths"]
t_stat, t_pval = stats.ttest_ind(female, male)
print(f"\nT-test female vs male deaths: t={t_stat:.4f}, p={t_pval:.4f}")
print(f"Female mean deaths: {female.mean():.2f}, Male mean deaths: {male.mean():.2f}")

# OLS regression: deaths ~ masfem controlling for storm severity
cols = ["masfem", "category", "wind", "min", "ndam", "alldeaths"]
df_clean = df[cols].dropna()
X = df_clean[["masfem", "category", "wind", "min", "ndam"]].copy()
X = sm.add_constant(X)
y = df_clean["alldeaths"]
model = sm.OLS(y, X).fit()
print("\nOLS summary (deaths ~ masfem + controls):")
print(model.summary())

# Log-transform deaths (highly skewed)
df_clean["log_deaths"] = np.log1p(df_clean["alldeaths"])
X2 = df_clean[["masfem", "category", "wind", "min", "ndam"]].copy()
X2 = sm.add_constant(X2)
model2 = sm.OLS(df_clean["log_deaths"], X2).fit()
print("\nOLS summary (log_deaths ~ masfem + controls):")
print(model2.summary())

masfem_coef = model2.params["masfem"]
masfem_pval = model2.pvalues["masfem"]
print(f"\nmasfem coef={masfem_coef:.4f}, p={masfem_pval:.4f}")

# Simple regression without controls
X_simple = sm.add_constant(df_clean[["masfem"]])
model_simple = sm.OLS(df_clean["log_deaths"], X_simple).fit()
simple_coef = model_simple.params["masfem"]
simple_pval = model_simple.pvalues["masfem"]
print(f"Simple regression: masfem coef={simple_coef:.4f}, p={simple_pval:.4f}")

# Decide score
# Main claim: more feminine names -> more deaths (less precaution)
# Check controlled regression p-value
significant = masfem_pval < 0.05
positive_effect = masfem_coef > 0

print(f"\nSignificant effect with controls: {significant}")
print(f"Positive effect (more feminine -> more deaths): {positive_effect}")

# The original paper (Jung et al 2014) claimed yes, but replications and
# Simonsohn's specification curve found the result is not robust.
# We evaluate empirically from the data.

if significant and positive_effect:
    score = 70
    explanation = (
        f"The controlled regression shows a positive, statistically significant effect of "
        f"femininity (masfem) on deaths (coef={masfem_coef:.4f}, p={masfem_pval:.4f}), "
        "supporting the hypothesis that more feminine hurricane names lead to more deaths "
        "due to fewer precautionary measures."
    )
elif positive_effect and not significant:
    score = 35
    explanation = (
        f"The controlled regression shows a positive but NOT statistically significant effect of "
        f"femininity on deaths (coef={masfem_coef:.4f}, p={masfem_pval:.4f}). "
        f"Simple correlation is also weak (r={corr:.3f}, p={pval:.3f}). "
        "The direction is consistent with the hypothesis but the evidence is not robust."
    )
else:
    score = 20
    explanation = (
        f"The controlled regression does not support the hypothesis "
        f"(coef={masfem_coef:.4f}, p={masfem_pval:.4f}). "
        f"Pearson r={corr:.3f} (p={pval:.3f}). "
        "There is insufficient evidence that more feminine hurricane names lead to more deaths."
    )

result = {"response": score, "explanation": explanation}
with open("conclusion.txt", "w") as f:
    json.dump(result, f)

print("\nconclusion.txt written:")
print(json.dumps(result, indent=2))
