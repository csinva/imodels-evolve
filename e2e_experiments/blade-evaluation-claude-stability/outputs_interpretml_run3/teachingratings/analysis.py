import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge
from interpret.glassbox import ExplainableBoostingRegressor

df = pd.read_csv("teachingratings.csv")
print("Shape:", df.shape)
print(df.describe())
print("\nCorrelation beauty vs eval:", df["beauty"].corr(df["eval"]))

# Simple correlation test
r, p_pearson = stats.pearsonr(df["beauty"], df["eval"])
print(f"\nPearson r={r:.4f}, p={p_pearson:.6f}")

# OLS regression: beauty -> eval (simple)
X_simple = sm.add_constant(df["beauty"])
ols_simple = sm.OLS(df["eval"], X_simple).fit()
print("\n--- Simple OLS (beauty -> eval) ---")
print(ols_simple.summary())

# Multiple regression controlling for covariates
df_model = df.copy()
df_model["minority"] = (df_model["minority"] == "yes").astype(int)
df_model["gender_male"] = (df_model["gender"] == "male").astype(int)
df_model["credits_single"] = (df_model["credits"] == "single").astype(int)
df_model["division_upper"] = (df_model["division"] == "upper").astype(int)
df_model["native_yes"] = (df_model["native"] == "yes").astype(int)
df_model["tenure_yes"] = (df_model["tenure"] == "yes").astype(int)

covars = ["beauty", "age", "minority", "gender_male", "credits_single",
          "division_upper", "native_yes", "tenure_yes", "students"]
X_multi = sm.add_constant(df_model[covars])
ols_multi = sm.OLS(df_model["eval"], X_multi).fit()
print("\n--- Multiple OLS (with covariates) ---")
print(ols_multi.summary())

# EBM for non-linear relationships
ebm = ExplainableBoostingRegressor(random_state=42)
ebm.fit(df_model[covars], df_model["eval"])
importances = dict(zip(covars, ebm.term_importances()))
print("\nEBM feature importances:", importances)

# Extract key results
beauty_coef_simple = ols_simple.params["beauty"]
beauty_p_simple = ols_simple.pvalues["beauty"]
beauty_coef_multi = ols_multi.params["beauty"]
beauty_p_multi = ols_multi.pvalues["beauty"]

print(f"\nSummary: beauty coef (simple)={beauty_coef_simple:.4f}, p={beauty_p_simple:.6f}")
print(f"Summary: beauty coef (multi)={beauty_coef_multi:.4f}, p={beauty_p_multi:.6f}")

# Determine response score
# Strong positive: significant in both simple and multiple regression
significant = (beauty_p_simple < 0.05) and (beauty_p_multi < 0.05)
positive_effect = beauty_coef_multi > 0

if significant and positive_effect:
    response = 82
    explanation = (
        f"Beauty has a statistically significant positive impact on teaching evaluations. "
        f"Simple OLS: beta={beauty_coef_simple:.4f}, p={beauty_p_simple:.4f}. "
        f"Multiple OLS (controlling for age, gender, minority, credits, division, native, tenure, students): "
        f"beta={beauty_coef_multi:.4f}, p={beauty_p_multi:.4f}. "
        f"Pearson r={r:.4f}. EBM importance for beauty={importances.get('beauty', 'N/A'):.4f}. "
        f"Higher beauty ratings are significantly associated with higher teaching evaluation scores, "
        f"even after controlling for other instructor and course characteristics."
    )
elif significant:
    response = 30
    explanation = (
        f"Beauty is statistically significant but has a negative effect: "
        f"beta={beauty_coef_multi:.4f}, p={beauty_p_multi:.4f}."
    )
else:
    response = 20
    explanation = (
        f"Beauty does not have a statistically significant impact on teaching evaluations "
        f"after controlling for other factors: beta={beauty_coef_multi:.4f}, p={beauty_p_multi:.4f}."
    )

conclusion = {"response": response, "explanation": explanation}
with open("conclusion.txt", "w") as f:
    json.dump(conclusion, f)

print("\nconclusion.txt written.")
print(json.dumps(conclusion, indent=2))
