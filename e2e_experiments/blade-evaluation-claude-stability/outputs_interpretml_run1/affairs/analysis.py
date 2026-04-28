import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import Ridge
from interpret.glassbox import ExplainableBoostingRegressor

df = pd.read_csv("affairs.csv")

# Encode children as binary
df["children_bin"] = (df["children"] == "yes").astype(int)
df["gender_bin"] = (df["gender"] == "male").astype(int)

# Summary stats by children
print("=== Affairs by children status ===")
print(df.groupby("children")["affairs"].describe())

no_children = df[df["children"] == "no"]["affairs"]
yes_children = df[df["children"] == "yes"]["affairs"]
print(f"\nMean affairs (no children): {no_children.mean():.4f}")
print(f"Mean affairs (yes children): {yes_children.mean():.4f}")

# Mann-Whitney U test (distribution is heavily zero-inflated)
mw_stat, mw_p = stats.mannwhitneyu(no_children, yes_children, alternative="two-sided")
print(f"\nMann-Whitney U: stat={mw_stat:.2f}, p={mw_p:.4f}")

# t-test
t_stat, t_p = stats.ttest_ind(no_children, yes_children)
print(f"t-test: stat={t_stat:.4f}, p={t_p:.4f}")

# OLS regression with children only
X_simple = sm.add_constant(df["children_bin"])
ols_simple = sm.OLS(df["affairs"], X_simple).fit()
print("\n=== OLS: affairs ~ children ===")
print(ols_simple.summary())

# OLS with controls
controls = ["children_bin", "gender_bin", "age", "yearsmarried",
            "religiousness", "education", "occupation", "rating"]
X_full = sm.add_constant(df[controls])
ols_full = sm.OLS(df["affairs"], X_full).fit()
print("\n=== OLS: affairs ~ children + controls ===")
print(ols_full.summary())

# EBM for feature importance
ebm = ExplainableBoostingRegressor(random_state=42)
X_ebm = df[controls]
ebm.fit(X_ebm, df["affairs"])
importances = dict(zip(controls, ebm.term_importances()))
print("\n=== EBM feature importances ===")
for k, v in sorted(importances.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v:.4f}")

# Determine conclusion
children_coef_simple = ols_simple.params["children_bin"]
children_pval_simple = ols_simple.pvalues["children_bin"]
children_coef_full = ols_full.params["children_bin"]
children_pval_full = ols_full.pvalues["children_bin"]

print(f"\nChildren coef (simple OLS): {children_coef_simple:.4f}, p={children_pval_simple:.4f}")
print(f"Children coef (full OLS):   {children_coef_full:.4f}, p={children_pval_full:.4f}")
print(f"MW p-value: {mw_p:.4f}")

# Score: positive coef means children → MORE affairs (not decrease)
# Negative coef means children → FEWER affairs (decrease = Yes)
# We check: does having children DECREASE affairs?
# If coef is negative and significant → Yes (high score)
# If coef is positive or not significant → No (low score)

sig = children_pval_full < 0.05
direction_decrease = children_coef_full < 0
mw_sig = mw_p < 0.05

print(f"\nSignificant in full model: {sig}")
print(f"Direction is decrease: {direction_decrease}")

if sig and direction_decrease:
    response = 30  # significant but INCREASES affairs (children associated with more affairs) → "No"
    explanation = (
        f"The full OLS regression shows children coefficient = {children_coef_full:.3f} "
        f"(p={children_pval_full:.4f}), indicating having children is associated with MORE "
        f"affairs, not fewer. The raw means also confirm: those without children have "
        f"{no_children.mean():.2f} affairs on average vs {yes_children.mean():.2f} for those with children. "
        f"Mann-Whitney U test p={mw_p:.4f}. Having children does NOT decrease extramarital affairs."
    )
elif sig and not direction_decrease:
    response = 70
    explanation = (
        f"The full OLS regression shows children coefficient = {children_coef_full:.3f} "
        f"(p={children_pval_full:.4f}), indicating having children is significantly associated with "
        f"FEWER affairs. Mann-Whitney U p={mw_p:.4f}. Having children appears to decrease affairs."
    )
else:
    response = 20
    explanation = (
        f"No significant effect found. Children coefficient in full model = {children_coef_full:.3f} "
        f"(p={children_pval_full:.4f}). Mann-Whitney U p={mw_p:.4f}. "
        f"Mean affairs: no children={no_children.mean():.2f}, with children={yes_children.mean():.2f}. "
        f"There is no statistically significant evidence that having children decreases extramarital affairs."
    )

# Re-check direction for proper scoring
# coef > 0 means children=yes → more affairs → does NOT decrease
# coef < 0 means children=yes → fewer affairs → DOES decrease
# Rewrite response properly
if children_coef_full > 0 and sig:
    # children associated with MORE affairs — significant increase, not decrease
    response = 25
    explanation = (
        f"Having children is significantly associated with MORE extramarital affairs (not fewer). "
        f"Full OLS: children coef = {children_coef_full:.3f}, p={children_pval_full:.4f}. "
        f"Mean affairs: no children={no_children.mean():.2f} vs with children={yes_children.mean():.2f}. "
        f"Mann-Whitney U p={mw_p:.4f}. The evidence does not support the hypothesis that children decrease affairs."
    )
elif children_coef_full < 0 and sig:
    response = 72
    explanation = (
        f"Having children is significantly associated with FEWER extramarital affairs. "
        f"Full OLS: children coef = {children_coef_full:.3f}, p={children_pval_full:.4f}. "
        f"Mean affairs: no children={no_children.mean():.2f} vs with children={yes_children.mean():.2f}. "
        f"Mann-Whitney U p={mw_p:.4f}. Evidence supports that children are associated with decreased affairs."
    )
else:
    response = 20
    explanation = (
        f"No significant relationship found. Full OLS: children coef = {children_coef_full:.3f}, "
        f"p={children_pval_full:.4f}. Mean affairs: no children={no_children.mean():.2f} vs "
        f"with children={yes_children.mean():.2f}. Mann-Whitney U p={mw_p:.4f}. "
        f"Having children does not significantly decrease extramarital affairs."
    )

conclusion = {"response": response, "explanation": explanation}
with open("conclusion.txt", "w") as f:
    json.dump(conclusion, f)

print(f"\n=== CONCLUSION ===")
print(json.dumps(conclusion, indent=2))
