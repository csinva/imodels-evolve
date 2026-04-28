import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

df = pd.read_csv("affairs.csv")

# Encode children as binary
df["children_bin"] = (df["children"] == "yes").astype(int)

# Group stats
no_children = df[df["children"] == "no"]["affairs"]
with_children = df[df["children"] == "yes"]["affairs"]

print("Mean affairs (no children):", no_children.mean())
print("Mean affairs (with children):", with_children.mean())

# Mann-Whitney U test (non-parametric, affairs is discrete/skewed)
u_stat, p_mw = stats.mannwhitneyu(no_children, with_children, alternative="greater")
print(f"Mann-Whitney U (no_children > with_children): U={u_stat:.1f}, p={p_mw:.4f}")

# t-test
t_stat, p_t = stats.ttest_ind(no_children, with_children, alternative="greater")
print(f"t-test (no_children > with_children): t={t_stat:.4f}, p={p_t:.4f}")

# OLS regression controlling for confounders
df_model = df.copy()
df_model["gender_bin"] = (df_model["gender"] == "male").astype(int)
X_cols = ["children_bin", "age", "yearsmarried", "religiousness", "education", "occupation", "rating", "gender_bin"]
X = sm.add_constant(df_model[X_cols])
y = df_model["affairs"]
ols = sm.OLS(y, X).fit()
print(ols.summary())

children_coef = ols.params["children_bin"]
children_pval = ols.pvalues["children_bin"]
print(f"\nOLS children_bin coef={children_coef:.4f}, p={children_pval:.4f}")

# Decide response
# children_bin coef > 0 means having children INCREASES affairs (i.e., does NOT decrease)
# We're asking: does having children DECREASE affairs?
# "decrease" = children_bin coef < 0 and significant
decreases = (children_coef < 0) and (children_pval < 0.05)
mw_decreases = (p_mw < 0.05)  # no_children > with_children means with_children lower

if decreases and mw_decreases:
    response = 70
    explanation = (
        f"Both OLS regression (coef={children_coef:.3f}, p={children_pval:.4f}) and "
        f"Mann-Whitney U test (p={p_mw:.4f}) suggest having children is associated with "
        "fewer affairs. However, effect size is modest."
    )
elif decreases:
    response = 55
    explanation = (
        f"OLS regression (coef={children_coef:.3f}, p={children_pval:.4f}) suggests "
        "having children is negatively associated with affairs when controlling for "
        f"confounders, but Mann-Whitney (p={p_mw:.4f}) is not significant in the same direction."
    )
elif mw_decreases:
    response = 45
    explanation = (
        f"Raw comparison (Mann-Whitney p={p_mw:.4f}) suggests fewer affairs for those with children, "
        f"but after controlling for confounders OLS coef={children_coef:.3f} (p={children_pval:.4f}) "
        "is not significantly negative, indicating confounders explain the difference."
    )
else:
    # children_bin coef > 0 or not significant
    if children_pval < 0.05 and children_coef > 0:
        response = 15
        explanation = (
            f"OLS regression shows having children is positively associated with affairs "
            f"(coef={children_coef:.3f}, p={children_pval:.4f}), suggesting children do NOT decrease "
            "extramarital affairs. Mann-Whitney also does not support a decrease."
        )
    else:
        response = 25
        explanation = (
            f"No statistically significant evidence that having children decreases extramarital affairs. "
            f"OLS coef for children={children_coef:.3f} (p={children_pval:.4f}); "
            f"Mann-Whitney p={p_mw:.4f}. The relationship is not significant or goes in the opposite direction."
        )

conclusion = {"response": response, "explanation": explanation}
with open("conclusion.txt", "w") as f:
    json.dump(conclusion, f)

print("\nConclusion:", conclusion)
