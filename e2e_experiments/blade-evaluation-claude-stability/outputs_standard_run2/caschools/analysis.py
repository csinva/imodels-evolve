import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("caschools.csv")

# Compute student-teacher ratio and composite test score
df["str"] = df["students"] / df["teachers"]
df["score"] = (df["read"] + df["math"]) / 2

# --- Correlation ---
r, p_corr = stats.pearsonr(df["str"], df["score"])
print(f"Pearson r(str, score) = {r:.4f}, p = {p_corr:.4e}")

# --- OLS with str only ---
X_simple = sm.add_constant(df["str"])
ols_simple = sm.OLS(df["score"], X_simple).fit()
print(ols_simple.summary())

# --- Multivariate OLS controlling for confounders ---
features = ["str", "lunch", "calworks", "english", "income", "expenditure"]
X_multi = sm.add_constant(df[features])
ols_multi = sm.OLS(df["score"], X_multi).fit()
print(ols_multi.summary())

coef_str = ols_multi.params["str"]
pval_str = ols_multi.pvalues["str"]
print(f"\nMultivariate OLS: coef(str) = {coef_str:.4f}, p = {pval_str:.4e}")

# --- Ridge for robustness check ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])
ridge = Ridge(alpha=1.0)
ridge.fit(X_scaled, df["score"])
coef_names = features
ridge_coefs = dict(zip(coef_names, ridge.coef_))
print("\nRidge standardized coefficients:", ridge_coefs)

# --- Median split t-test ---
median_str = df["str"].median()
low_str = df.loc[df["str"] <= median_str, "score"]
high_str = df.loc[df["str"] > median_str, "score"]
t_stat, p_ttest = stats.ttest_ind(low_str, high_str)
print(f"\nMedian split t-test: t = {t_stat:.4f}, p = {p_ttest:.4e}")
print(f"Mean score low STR: {low_str.mean():.2f}, high STR: {high_str.mean():.2f}")

# --- Conclusion ---
# Simple correlation is negative (higher STR -> lower scores), significant.
# But after controlling for socioeconomic confounders the str effect may shrink.
simple_sig = p_corr < 0.05
multi_sig = pval_str < 0.05
simple_negative = r < 0
multi_negative = coef_str < 0

# Bivariate evidence is strong; multivariate effect direction and significance determine final score.
# If still negative and significant after controls -> Yes (high score).
# If not significant after controls -> moderate/low score.
if multi_sig and multi_negative:
    response = 72
    explanation = (
        f"Both bivariate and multivariate analyses support that a lower student-teacher ratio (STR) "
        f"is associated with higher academic performance. Pearson r = {r:.3f} (p = {p_corr:.2e}) shows "
        f"a significant negative correlation between STR and test scores. In a multivariate OLS controlling "
        f"for lunch eligibility, CalWorks, English learners, income, and expenditure, STR coefficient = "
        f"{coef_str:.3f} (p = {pval_str:.2e}), still statistically significant and negative (higher STR -> "
        f"lower scores). The median-split t-test confirms that districts with lower STR score "
        f"{low_str.mean():.1f} vs {high_str.mean():.1f} for higher STR (p = {p_ttest:.2e}). "
        f"Overall, evidence consistently supports the association, though the effect size is modest after "
        f"controlling for socioeconomic factors."
    )
elif simple_sig and simple_negative and not multi_sig:
    response = 40
    explanation = (
        f"The bivariate correlation between STR and test scores is significant (r = {r:.3f}, p = {p_corr:.2e}), "
        f"but after controlling for socioeconomic confounders the STR effect is no longer statistically "
        f"significant (coef = {coef_str:.3f}, p = {pval_str:.2e}). This suggests the observed association is "
        f"largely driven by confounding variables rather than a direct relationship."
    )
else:
    response = 25
    explanation = (
        f"No strong consistent evidence that lower STR is associated with higher scores. "
        f"Pearson r = {r:.3f} (p = {p_corr:.2e}); multivariate coef = {coef_str:.3f} (p = {pval_str:.2e})."
    )

result = {"response": response, "explanation": explanation}
with open("conclusion.txt", "w") as f:
    json.dump(result, f)

print("\nconclusion.txt written.")
print(json.dumps(result, indent=2))
