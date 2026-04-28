import json
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv("amtl.csv")
print("Shape:", df.shape)
print(df.head())
print(df.describe())
print("\nGenus counts:\n", df["genus"].value_counts())
print("\nTooth class counts:\n", df["tooth_class"].value_counts())

# Compute AMTL rate
df["amtl_rate"] = df["num_amtl"] / df["sockets"]

# Summary by genus
print("\nAMTL rate by genus:")
print(df.groupby("genus")["amtl_rate"].describe())

# Is Homo sapiens higher?
homo = df[df["genus"] == "Homo sapiens"]["amtl_rate"]
nonhomo = df[df["genus"] != "Homo sapiens"]["amtl_rate"]
t_stat, p_val = stats.ttest_ind(homo, nonhomo)
print(f"\nt-test Homo vs non-human primates: t={t_stat:.4f}, p={p_val:.6f}")

# ANOVA across all genera
groups = [df[df["genus"] == g]["amtl_rate"].values for g in df["genus"].unique()]
f_stat, p_anova = stats.f_oneway(*groups)
print(f"One-way ANOVA across genera: F={f_stat:.4f}, p={p_anova:.6f}")

# Binomial GLM: num_amtl ~ genus + age + prob_male + tooth_class
# Encode categoricals
df["is_homo"] = (df["genus"] == "Homo sapiens").astype(int)
tooth_dummies = pd.get_dummies(df["tooth_class"], drop_first=True, prefix="tooth")

X = pd.concat([
    df[["age", "prob_male", "is_homo"]],
    tooth_dummies
], axis=1)
X = sm.add_constant(X)

# Binomial GLM with [successes, failures]
endog = np.column_stack([df["num_amtl"], df["sockets"] - df["num_amtl"]])
glm = sm.GLM(endog, X.astype(float), family=sm.families.Binomial())
result = glm.fit()
print("\nBinomial GLM summary:")
print(result.summary())

is_homo_coef = result.params["is_homo"]
is_homo_pval = result.pvalues["is_homo"]
print(f"\nis_homo coefficient: {is_homo_coef:.4f}, p-value: {is_homo_pval:.6e}")

# Mean AMTL rates
homo_mean = homo.mean()
nonhomo_mean = nonhomo.mean()
print(f"\nHomo sapiens mean AMTL rate: {homo_mean:.4f}")
print(f"Non-human primates mean AMTL rate: {nonhomo_mean:.4f}")

# Conclusion
significant = is_homo_pval < 0.05
positive_effect = is_homo_coef > 0

if significant and positive_effect:
    response = 90
    explanation = (
        f"Binomial GLM controlling for age, sex (prob_male), and tooth class shows "
        f"Homo sapiens have significantly higher AMTL rates than non-human primates "
        f"(coefficient={is_homo_coef:.4f}, p={is_homo_pval:.2e}). "
        f"Mean AMTL rate: Homo={homo_mean:.4f} vs non-human primates={nonhomo_mean:.4f}. "
        f"Raw t-test also confirms significant difference (p={p_val:.4f}). "
        f"The answer is Yes: modern humans have higher AMTL frequencies after accounting for age, sex, and tooth class."
    )
elif significant and not positive_effect:
    response = 15
    explanation = (
        f"Binomial GLM shows Homo sapiens have significantly LOWER AMTL rates "
        f"(coefficient={is_homo_coef:.4f}, p={is_homo_pval:.2e}). "
        f"Mean AMTL rate: Homo={homo_mean:.4f} vs non-human primates={nonhomo_mean:.4f}."
    )
else:
    response = 30
    explanation = (
        f"No significant difference in AMTL rates between Homo sapiens and non-human primates "
        f"after controlling for age, sex, and tooth class "
        f"(coefficient={is_homo_coef:.4f}, p={is_homo_pval:.2e}). "
        f"Mean AMTL rate: Homo={homo_mean:.4f} vs non-human primates={nonhomo_mean:.4f}."
    )

output = {"response": response, "explanation": explanation}
with open("conclusion.txt", "w") as f:
    json.dump(output, f)
print("\nconclusion.txt written:")
print(json.dumps(output, indent=2))
