import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("amtl.csv")
print("Shape:", df.shape)
print(df.head())
print(df.dtypes)
print("\nGenus counts:\n", df["genus"].value_counts())

# Compute AMTL rate per row
df["amtl_rate"] = df["num_amtl"] / df["sockets"]

# Summary stats by genus
print("\nAMTL rate by genus:")
summary = df.groupby("genus")["amtl_rate"].agg(["mean", "median", "std", "count"])
print(summary)

# Is Homo sapiens the human genus?
df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

# --- Binomial GLM (statsmodels) ---
# Encode categoricals
df["tooth_anterior"] = (df["tooth_class"] == "Anterior").astype(int)
df["tooth_posterior"] = (df["tooth_class"] == "Posterior").astype(int)
# Premolar is reference

# Encode genus dummies (Homo sapiens vs others)
genus_dummies = pd.get_dummies(df["genus"], drop_first=False)
print("\nGenus dummies:\n", genus_dummies.columns.tolist())
# Use Pan as reference (alphabetically first after drop)

# Build design matrix
X_cols = ["is_human", "age", "prob_male", "tooth_anterior", "tooth_posterior"]
X = df[X_cols].copy()
X = sm.add_constant(X)

# Binomial GLM: response is (num_amtl, sockets - num_amtl)
endog = np.column_stack([df["num_amtl"], df["sockets"] - df["num_amtl"]])

glm_model = sm.GLM(endog, X, family=sm.families.Binomial())
glm_result = glm_model.fit()
print("\n--- Binomial GLM (is_human + age + sex + tooth_class) ---")
print(glm_result.summary())

is_human_coef = glm_result.params["is_human"]
is_human_pval = glm_result.pvalues["is_human"]
is_human_ci_low, is_human_ci_high = glm_result.conf_int().loc["is_human"]
is_human_or = np.exp(is_human_coef)  # odds ratio

print(f"\nHomo sapiens coefficient: {is_human_coef:.4f}")
print(f"Odds ratio: {is_human_or:.4f}")
print(f"95% CI: ({np.exp(is_human_ci_low):.4f}, {np.exp(is_human_ci_high):.4f})")
print(f"p-value: {is_human_pval:.6f}")

# --- Full genus model (all four genera) ---
df_full = pd.get_dummies(df[["genus"]], drop_first=True).astype(float)
X2 = pd.concat([df[["age", "prob_male", "tooth_anterior", "tooth_posterior"]].astype(float), df_full], axis=1)
X2 = sm.add_constant(X2)

glm2 = sm.GLM(endog, X2, family=sm.families.Binomial())
glm2_result = glm2.fit()
print("\n--- Full genus model ---")
print(glm2_result.summary())

# --- Simple t-test comparing human vs non-human AMTL rates ---
human_rates = df.loc[df["is_human"] == 1, "amtl_rate"]
nonhuman_rates = df.loc[df["is_human"] == 0, "amtl_rate"]
t_stat, t_pval = stats.ttest_ind(human_rates, nonhuman_rates)
print(f"\nT-test: human mean={human_rates.mean():.4f}, non-human mean={nonhuman_rates.mean():.4f}")
print(f"t={t_stat:.4f}, p={t_pval:.6f}")

# --- Decision: score based on GLM result ---
# High odds ratio > 1 and low p-value = Yes (humans have higher AMTL)
significant = is_human_pval < 0.05
positive_direction = is_human_coef > 0

print(f"\nSignificant: {significant}, Positive direction: {positive_direction}")

if significant and positive_direction:
    or_val = is_human_or
    # Score based on effect size
    if or_val > 5:
        score = 95
    elif or_val > 3:
        score = 88
    elif or_val > 2:
        score = 80
    elif or_val > 1.5:
        score = 72
    else:
        score = 65
    explanation = (
        f"Binomial GLM shows Homo sapiens have significantly higher AMTL rates compared to "
        f"non-human primates after accounting for age, sex, and tooth class. "
        f"Coefficient for is_human: {is_human_coef:.4f} (odds ratio={is_human_or:.2f}, "
        f"95% CI [{np.exp(is_human_ci_low):.2f}, {np.exp(is_human_ci_high):.2f}], "
        f"p={is_human_pval:.2e}). Raw AMTL rates: humans={human_rates.mean():.3f} vs "
        f"non-humans={nonhuman_rates.mean():.3f}."
    )
elif significant and not positive_direction:
    score = 10
    explanation = (
        f"Binomial GLM shows Homo sapiens have significantly LOWER AMTL rates than non-human primates "
        f"(coefficient={is_human_coef:.4f}, OR={is_human_or:.2f}, p={is_human_pval:.2e})."
    )
else:
    score = 30
    explanation = (
        f"No significant difference found (p={is_human_pval:.4f}). Coefficient for is_human: "
        f"{is_human_coef:.4f} (OR={is_human_or:.2f}). Cannot conclude humans have higher AMTL."
    )

result = {"response": score, "explanation": explanation}
print("\nConclusion:", result)

with open("conclusion.txt", "w") as f:
    json.dump(result, f)

print("conclusion.txt written.")
