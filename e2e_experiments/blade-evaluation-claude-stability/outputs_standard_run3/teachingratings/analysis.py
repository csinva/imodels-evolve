import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv("teachingratings.csv")
print("Shape:", df.shape)
print(df.describe())

# Encode categorical variables
cat_cols = ["minority", "gender", "credits", "division", "native", "tenure"]
df_enc = df.copy()
le = LabelEncoder()
for col in cat_cols:
    df_enc[col] = le.fit_transform(df_enc[col].astype(str))

# Pearson correlation between beauty and eval
r, p_corr = stats.pearsonr(df["beauty"], df["eval"])
print(f"\nPearson r(beauty, eval) = {r:.4f}, p = {p_corr:.4e}")

# Simple OLS: eval ~ beauty
X_simple = sm.add_constant(df["beauty"])
ols_simple = sm.OLS(df["eval"], X_simple).fit()
print("\n--- Simple OLS: eval ~ beauty ---")
print(ols_simple.summary())

# Multiple OLS controlling for covariates
feature_cols = ["beauty", "age", "gender", "minority", "native", "tenure",
                "credits", "division", "students", "allstudents"]
X_multi = sm.add_constant(df_enc[feature_cols])
ols_multi = sm.OLS(df_enc["eval"], X_multi).fit()
print("\n--- Multiple OLS: eval ~ beauty + covariates ---")
print(ols_multi.summary())

beauty_coef = ols_multi.params["beauty"]
beauty_pval = ols_multi.pvalues["beauty"]
beauty_ci = ols_multi.conf_int().loc["beauty"]
print(f"\nBeauty coefficient: {beauty_coef:.4f}, p={beauty_pval:.4e}, 95% CI: [{beauty_ci[0]:.4f}, {beauty_ci[1]:.4f}]")

# Determine score
# Significant positive effect -> high score
if beauty_pval < 0.05 and beauty_coef > 0:
    response = 85
    explanation = (
        f"Beauty has a statistically significant positive effect on teaching evaluations. "
        f"Simple Pearson correlation: r={r:.3f} (p={p_corr:.4e}). "
        f"In a multiple regression controlling for age, gender, minority status, native English speaker, "
        f"tenure, credits, division, and class size, the beauty coefficient is {beauty_coef:.3f} "
        f"(p={beauty_pval:.4e}, 95% CI [{beauty_ci[0]:.3f}, {beauty_ci[1]:.3f}]). "
        f"Higher beauty ratings are associated with higher teaching evaluation scores, even after controlling for confounders."
    )
elif beauty_pval < 0.05 and beauty_coef < 0:
    response = 15
    explanation = (
        f"Beauty has a statistically significant negative effect on teaching evaluations. "
        f"Simple Pearson correlation: r={r:.3f} (p={p_corr:.4e}). "
        f"Multiple regression beauty coefficient: {beauty_coef:.3f} (p={beauty_pval:.4e})."
    )
else:
    response = 20
    explanation = (
        f"Beauty does not have a statistically significant effect on teaching evaluations. "
        f"Simple Pearson correlation: r={r:.3f} (p={p_corr:.4e}). "
        f"Multiple regression beauty coefficient: {beauty_coef:.3f} (p={beauty_pval:.4e})."
    )

result = {"response": response, "explanation": explanation}
with open("conclusion.txt", "w") as f:
    json.dump(result, f)

print("\nconclusion.txt written.")
print(json.dumps(result, indent=2))
