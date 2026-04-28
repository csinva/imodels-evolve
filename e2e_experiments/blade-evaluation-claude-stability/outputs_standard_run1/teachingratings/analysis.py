import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge
import json

# Load dataset
df = pd.read_csv("teachingratings.csv")
print("Shape:", df.shape)
print(df.head())
print(df.describe())

# Basic correlation
corr, pval = stats.pearsonr(df["beauty"], df["eval"])
print(f"\nPearson correlation (beauty vs eval): r={corr:.4f}, p={pval:.4f}")

# OLS regression: beauty -> eval (simple)
X_simple = sm.add_constant(df["beauty"])
model_simple = sm.OLS(df["eval"], X_simple).fit()
print("\n--- Simple OLS: eval ~ beauty ---")
print(model_simple.summary())

# OLS with controls
df_enc = df.copy()
df_enc["minority_num"] = (df["minority"] == "yes").astype(int)
df_enc["gender_num"] = (df["gender"] == "female").astype(int)
df_enc["credits_num"] = (df["credits"] == "single").astype(int)
df_enc["native_num"] = (df["native"] == "yes").astype(int)
df_enc["tenure_num"] = (df["tenure"] == "yes").astype(int)
df_enc["division_num"] = (df["division"] == "upper").astype(int)

features = ["beauty", "age", "minority_num", "gender_num", "credits_num",
            "native_num", "tenure_num", "division_num"]
X_full = sm.add_constant(df_enc[features])
model_full = sm.OLS(df_enc["eval"], X_full).fit()
print("\n--- Full OLS: eval ~ beauty + controls ---")
print(model_full.summary())

beauty_coef = model_full.params["beauty"]
beauty_pval = model_full.pvalues["beauty"]
print(f"\nBeauty coefficient: {beauty_coef:.4f}, p-value: {beauty_pval:.4f}")

# Interpret result
significant = beauty_pval < 0.05
positive = beauty_coef > 0

print(f"\nSignificant: {significant}, Positive effect: {positive}")
print(f"Simple correlation: r={corr:.4f}, p={pval:.4f}")

# Score: beauty has a positive, statistically significant effect -> high score
if significant and positive:
    response = 82
    explanation = (
        f"Beauty has a statistically significant positive impact on teaching evaluations. "
        f"Simple Pearson r={corr:.3f} (p={pval:.4f}). "
        f"In a multiple regression controlling for age, gender, minority status, credits, native English, tenure, and division, "
        f"the beauty coefficient is {beauty_coef:.4f} (p={beauty_pval:.4f}), confirming a robust positive effect. "
        f"Higher-rated instructors on physical appearance receive meaningfully higher teaching evaluation scores."
    )
elif significant and not positive:
    response = 18
    explanation = (
        f"Beauty has a statistically significant but negative effect on evaluations (coef={beauty_coef:.4f}, p={beauty_pval:.4f}). "
        f"Simple correlation r={corr:.3f} (p={pval:.4f})."
    )
else:
    response = 25
    explanation = (
        f"No statistically significant impact of beauty on teaching evaluations was found. "
        f"Simple correlation r={corr:.3f} (p={pval:.4f}). "
        f"Full model beauty coef={beauty_coef:.4f} (p={beauty_pval:.4f})."
    )

result = {"response": response, "explanation": explanation}
with open("conclusion.txt", "w") as f:
    json.dump(result, f)

print("\nWritten conclusion.txt:")
print(json.dumps(result, indent=2))
