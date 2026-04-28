import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv("hurricane.csv")
print("Shape:", df.shape)
print(df.describe())

# Research question: Do hurricanes with more feminine names lead to more deaths
# (because public perceives them as less threatening and takes fewer precautions)?

# Key variables: masfem (femininity index, higher = more feminine), alldeaths
print("\nCorrelation masfem vs alldeaths:", df["masfem"].corr(df["alldeaths"]))
print("Correlation gender_mf vs alldeaths:", df["gender_mf"].corr(df["alldeaths"]))

# Simple correlation test
r, p = stats.pearsonr(df["masfem"], df["alldeaths"])
print(f"\nPearson r={r:.4f}, p={p:.4f}")

# t-test: male vs female named hurricanes
male = df[df["gender_mf"] == 0]["alldeaths"]
female = df[df["gender_mf"] == 1]["alldeaths"]
t, p_t = stats.ttest_ind(female, male)
print(f"t-test female vs male deaths: t={t:.4f}, p={p_t:.4f}")
print(f"Mean deaths female={female.mean():.2f}, male={male.mean():.2f}")

# OLS regression controlling for storm severity
# Key confounders: category (severity), ndam (damage proxy for severity), wind speed
cols = ["masfem", "category", "wind", "ndam", "alldeaths"]
df_clean = df[cols].dropna()
X = df_clean[["masfem", "category", "wind", "ndam"]]
y = df_clean["alldeaths"]

X_const = sm.add_constant(X)
model = sm.OLS(y, X_const).fit()
print("\n--- OLS with controls ---")
print(model.summary())

# Log-transform deaths (heavily skewed)
df_clean["log_deaths"] = np.log1p(df_clean["alldeaths"])
r_log, p_log = stats.pearsonr(df_clean["masfem"], df_clean["log_deaths"])
print(f"\nPearson r (log deaths) r={r_log:.4f}, p={p_log:.4f}")

X2 = df_clean[["masfem", "category", "wind", "ndam"]]
y2 = df_clean["log_deaths"]
X2_const = sm.add_constant(X2)
model2 = sm.OLS(y2, X2_const).fit()
print("\n--- OLS log(deaths) with controls ---")
print(model2.summary())

masfem_coef = model2.params["masfem"]
masfem_pval = model2.pvalues["masfem"]
print(f"\nmasfem coef={masfem_coef:.4f}, p={masfem_pval:.4f}")

# Determine response score
# The research question asks if feminine names -> less precaution -> more deaths
# We look at whether masfem positively predicts deaths
# Evidence: is masfem coefficient positive and significant?

if masfem_pval < 0.05 and masfem_coef > 0:
    response = 75
    explanation = (
        f"After controlling for storm severity (category, wind, damage), masfem has a positive "
        f"coefficient ({masfem_coef:.3f}) predicting log(deaths) with p={masfem_pval:.3f} (<0.05). "
        f"This supports the claim that more feminine hurricane names are associated with more deaths, "
        f"consistent with the hypothesis that they are perceived as less threatening. "
        f"Raw correlation r={r:.3f} (p={p:.3f}), binary t-test p={p_t:.3f}."
    )
elif masfem_pval < 0.10 and masfem_coef > 0:
    response = 55
    explanation = (
        f"masfem coefficient is positive ({masfem_coef:.3f}) in controlled regression but only "
        f"marginally significant (p={masfem_pval:.3f}). Weak evidence supporting the hypothesis."
    )
else:
    response = 25
    explanation = (
        f"After controlling for storm severity, the relationship between femininity of name (masfem) "
        f"and deaths is not statistically significant (coef={masfem_coef:.3f}, p={masfem_pval:.3f}). "
        f"The raw correlation r={r:.3f} (p={p:.3f}). Binary gender t-test p={p_t:.3f}. "
        f"The evidence does not strongly support that feminine hurricane names lead to more deaths via reduced precaution."
    )

result = {"response": response, "explanation": explanation}
print("\nConclusion:", result)

with open("conclusion.txt", "w") as f:
    json.dump(result, f)

print("Written conclusion.txt")
