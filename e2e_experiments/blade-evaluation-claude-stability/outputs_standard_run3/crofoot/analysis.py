import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv("crofoot.csv")
print("Shape:", df.shape)
print(df.describe())
print("\nWin rate:", df["win"].mean())

# Feature engineering
df["rel_size"] = df["n_focal"] / df["n_other"]         # relative group size
df["size_diff"] = df["n_focal"] - df["n_other"]         # absolute group size difference
df["loc_advantage"] = df["dist_other"] - df["dist_focal"]  # positive = focal closer to home
df["rel_males"] = df["m_focal"] / df["m_other"]

print("\n--- Correlation with win ---")
corr_cols = ["rel_size", "size_diff", "loc_advantage", "dist_focal", "dist_other"]
for col in corr_cols:
    r, p = stats.pointbiserialr(df[col], df["win"])
    print(f"{col}: r={r:.3f}, p={p:.4f}")

# Split wins vs losses
wins = df[df["win"] == 1]
losses = df[df["win"] == 0]

print("\n--- T-tests: wins vs losses ---")
for col in ["rel_size", "size_diff", "loc_advantage", "dist_focal", "dist_other"]:
    t, p = stats.ttest_ind(wins[col], losses[col])
    print(f"{col}: win_mean={wins[col].mean():.3f}, loss_mean={losses[col].mean():.3f}, t={t:.3f}, p={p:.4f}")

# Logistic regression
X = df[["size_diff", "loc_advantage"]].copy()
X = sm.add_constant(X)
y = df["win"]

logit_model = sm.Logit(y, X).fit(disp=0)
print("\n--- Logistic Regression (statsmodels) ---")
print(logit_model.summary())

# Separate tests for each variable
print("\n--- Logistic Regression: size_diff only ---")
X_size = sm.add_constant(df[["size_diff"]])
m_size = sm.Logit(y, X_size).fit(disp=0)
print(m_size.summary())

print("\n--- Logistic Regression: loc_advantage only ---")
X_loc = sm.add_constant(df[["loc_advantage"]])
m_loc = sm.Logit(y, X_loc).fit(disp=0)
print(m_loc.summary())

# Extract key p-values
size_pval = logit_model.pvalues["size_diff"]
loc_pval = logit_model.pvalues["loc_advantage"]
size_pval_uni = m_size.pvalues["size_diff"]
loc_pval_uni = m_loc.pvalues["loc_advantage"]

print(f"\nMultivariate: size_diff p={size_pval:.4f}, loc_advantage p={loc_pval:.4f}")
print(f"Univariate:   size_diff p={size_pval_uni:.4f}, loc_advantage p={loc_pval_uni:.4f}")

# Both factors significant?
both_sig = (size_pval < 0.05) and (loc_pval < 0.05)
either_sig_uni = (size_pval_uni < 0.05) or (loc_pval_uni < 0.05)
both_sig_uni = (size_pval_uni < 0.05) and (loc_pval_uni < 0.05)

print(f"\nBoth significant (multivariate): {both_sig}")
print(f"Both significant (univariate): {both_sig_uni}")

# Determine response score
# The question asks how BOTH factors influence win probability
# Strong yes = both are significant predictors
if both_sig:
    score = 85
    explanation = (
        f"Both relative group size (size_diff p={size_pval:.4f}) and contest location "
        f"(loc_advantage p={loc_pval:.4f}) are statistically significant predictors of winning "
        f"in multivariate logistic regression. Groups closer to their home range center and larger "
        f"groups are more likely to win, confirming both factors influence win probability."
    )
elif both_sig_uni:
    score = 75
    explanation = (
        f"Both relative group size (size_diff univariate p={size_pval_uni:.4f}) and contest location "
        f"(loc_advantage univariate p={loc_pval_uni:.4f}) are significant in univariate logistic "
        f"regression. In the multivariate model, size_diff p={size_pval:.4f}, loc p={loc_pval:.4f}. "
        f"Both factors meaningfully influence win probability."
    )
elif (size_pval_uni < 0.05) or (loc_pval_uni < 0.05):
    score = 55
    explanation = (
        f"Only one factor is clearly significant: size_diff univariate p={size_pval_uni:.4f}, "
        f"loc_advantage univariate p={loc_pval_uni:.4f}. Partial support that both factors matter."
    )
else:
    score = 20
    explanation = (
        f"Neither relative group size (p={size_pval_uni:.4f}) nor contest location "
        f"(p={loc_pval_uni:.4f}) is statistically significant. No clear evidence both factors "
        f"influence win probability."
    )

print(f"\nFinal score: {score}")
print(f"Explanation: {explanation}")

result = {"response": score, "explanation": explanation}
with open("conclusion.txt", "w") as f:
    json.dump(result, f)

print("\nconclusion.txt written.")
