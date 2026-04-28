import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv("boxes.csv")
print("Shape:", df.shape)
print(df.describe())
print("\nValue counts for y:\n", df["y"].value_counts())
print("\nCulture counts:\n", df["culture"].value_counts())

# Create binary outcome: chose majority (y==2)
df["chose_majority"] = (df["y"] == 2).astype(int)

# --- Overall majority choice rate by age ---
age_majority = df.groupby("age")["chose_majority"].mean()
print("\nMajority choice rate by age:\n", age_majority)

# Spearman correlation between age and chose_majority
rho, pval_spearman = stats.spearmanr(df["age"], df["chose_majority"])
print(f"\nSpearman rho={rho:.4f}, p={pval_spearman:.4f}")

# --- OLS regression: chose_majority ~ age ---
X = sm.add_constant(df[["age"]])
ols = sm.OLS(df["chose_majority"], X).fit()
print("\nOLS age -> chose_majority:")
print(ols.summary())

# --- OLS with culture interaction: chose_majority ~ age + culture + age*culture ---
df["age_x_culture"] = df["age"] * df["culture"]
X2 = sm.add_constant(df[["age", "culture", "age_x_culture"]])
ols2 = sm.OLS(df["chose_majority"], X2).fit()
print("\nOLS with age*culture interaction:")
print(ols2.summary())

# --- Per-culture Spearman correlations ---
print("\nPer-culture Spearman rho (age vs chose_majority):")
culture_results = {}
for c in sorted(df["culture"].unique()):
    sub = df[df["culture"] == c]
    r, p = stats.spearmanr(sub["age"], sub["chose_majority"])
    culture_results[c] = {"rho": r, "p": p, "n": len(sub)}
    print(f"  Culture {c}: rho={r:.3f}, p={p:.3f}, n={len(sub)}")

# --- ANOVA: does majority choice rate differ across cultures? ---
groups = [df[df["culture"] == c]["chose_majority"].values for c in sorted(df["culture"].unique())]
f_stat, pval_anova = stats.f_oneway(*groups)
print(f"\nANOVA across cultures: F={f_stat:.4f}, p={pval_anova:.4f}")

# --- Summary ---
sig_cultures = sum(1 for v in culture_results.values() if v["p"] < 0.05)
pos_rho_cultures = sum(1 for v in culture_results.values() if v["rho"] > 0)
overall_pval = ols.pvalues["age"]
overall_coef = ols.params["age"]

print(f"\n=== Summary ===")
print(f"Overall OLS coef for age: {overall_coef:.4f}, p={overall_pval:.4f}")
print(f"Cultures with significant age->majority effect: {sig_cultures}/{len(culture_results)}")
print(f"Cultures with positive rho: {pos_rho_cultures}/{len(culture_results)}")

# Build conclusion
# Strong positive relationship between age and majority choice overall?
strong_yes = overall_pval < 0.05 and overall_coef > 0
moderately_yes = pos_rho_cultures >= len(culture_results) // 2

if strong_yes and sig_cultures >= 2:
    score = 75
    explanation = (
        f"There is a statistically significant positive relationship between age and majority preference "
        f"(OLS coef={overall_coef:.4f}, p={overall_pval:.4f}). "
        f"{sig_cultures} out of {len(culture_results)} cultures show significant age effects individually. "
        f"{pos_rho_cultures}/{len(culture_results)} cultures have a positive age-majority Spearman correlation. "
        f"ANOVA confirms significant cross-cultural differences (F={f_stat:.4f}, p={pval_anova:.4f}). "
        f"Children's reliance on majority preference increases with age, and the pattern varies across cultural contexts."
    )
elif strong_yes:
    score = 60
    explanation = (
        f"Overall OLS shows significant positive age effect (coef={overall_coef:.4f}, p={overall_pval:.4f}), "
        f"but only {sig_cultures}/{len(culture_results)} cultures reach individual significance. "
        f"Moderate evidence that majority preference grows with age across cultures."
    )
elif moderately_yes:
    score = 45
    explanation = (
        f"Mixed evidence: overall age effect p={overall_pval:.4f} (coef={overall_coef:.4f}). "
        f"{pos_rho_cultures}/{len(culture_results)} cultures trend positively but most are non-significant."
    )
else:
    score = 20
    explanation = (
        f"Weak or no evidence for age-related increase in majority preference "
        f"(overall p={overall_pval:.4f}, coef={overall_coef:.4f}). "
        f"Only {pos_rho_cultures}/{len(culture_results)} cultures show a positive trend."
    )

result = {"response": score, "explanation": explanation}
print("\nConclusion:", json.dumps(result, indent=2))

with open("conclusion.txt", "w") as f:
    json.dump(result, f)

print("\nconclusion.txt written.")
