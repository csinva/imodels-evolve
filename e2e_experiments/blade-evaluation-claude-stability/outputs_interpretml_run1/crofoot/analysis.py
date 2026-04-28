import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("crofoot.csv")

print("Shape:", df.shape)
print(df.describe())
print("\nWin rate:", df["win"].mean())

# Derived features
df["rel_size"] = df["n_focal"] / df["n_other"]          # relative group size (>1 = focal bigger)
df["rel_dist"] = df["dist_focal"] - df["dist_other"]    # positive = focal farther from home range
df["size_diff"] = df["n_focal"] - df["n_other"]

# --- 1. Correlation with win ---
for col in ["rel_size", "rel_dist", "size_diff", "dist_focal", "dist_other", "n_focal", "n_other"]:
    r, p = stats.pointbiserialr(df["win"], df[col])
    print(f"{col:20s}  r={r:.3f}  p={p:.4f}")

# --- 2. Logistic regression with statsmodels ---
X_vars = ["rel_size", "rel_dist"]
X = sm.add_constant(df[X_vars])
y = df["win"]
logit_model = sm.Logit(y, X).fit(disp=False)
print("\nLogit summary:")
print(logit_model.summary())

# --- 3. Separate t-tests: winners vs losers ---
wins = df[df["win"] == 1]
losses = df[df["win"] == 0]

t_size, p_size = stats.ttest_ind(wins["rel_size"], losses["rel_size"])
t_dist, p_dist = stats.ttest_ind(wins["rel_dist"], losses["rel_dist"])
print(f"\nrel_size  t={t_size:.3f}  p={p_size:.4f}")
print(f"rel_dist  t={t_dist:.3f}  p={p_dist:.4f}")

# --- 4. EBM (interpretable) ---
try:
    from interpret.glassbox import ExplainableBoostingClassifier
    ebm = ExplainableBoostingClassifier(random_state=42)
    ebm.fit(df[["rel_size", "rel_dist"]], df["win"])
    print("\nEBM feature importances:", dict(zip(["rel_size", "rel_dist"], ebm.term_importances())))
except Exception as e:
    print("EBM skipped:", e)

# --- Conclusion ---
# Both relative size (p_size) and relative distance (p_dist) are the key signals
size_sig = p_size < 0.05
dist_sig = p_dist < 0.05
logit_size_p = logit_model.pvalues.get("rel_size", 1.0)
logit_dist_p = logit_model.pvalues.get("rel_dist", 1.0)

print(f"\nLogit p-values — rel_size: {logit_size_p:.4f}  rel_dist: {logit_dist_p:.4f}")

# Scoring: question asks about BOTH factors; assess joint evidence
# Use logistic regression p-values as primary signal
both_sig = logit_size_p < 0.05 and logit_dist_p < 0.05
one_sig = logit_size_p < 0.05 or logit_dist_p < 0.05

if both_sig:
    score = 85
    explanation = (
        f"Both relative group size (logit p={logit_size_p:.4f}) and contest location "
        f"(relative distance from home range, logit p={logit_dist_p:.4f}) significantly "
        f"influence the probability of winning. Larger focal groups relative to opponents "
        f"and contests held closer to the focal group's home range are associated with higher "
        f"win probability. t-test p-values: size p={p_size:.4f}, location p={p_dist:.4f}."
    )
elif one_sig:
    # Determine which one
    if logit_size_p < logit_dist_p:
        score = 65
        explanation = (
            f"Relative group size (logit p={logit_size_p:.4f}) significantly influences win "
            f"probability, while contest location (logit p={logit_dist_p:.4f}) shows weaker "
            f"evidence. Overall, there is partial support that both factors matter."
        )
    else:
        score = 65
        explanation = (
            f"Contest location (logit p={logit_dist_p:.4f}) significantly influences win "
            f"probability, while relative group size (logit p={logit_size_p:.4f}) shows weaker "
            f"evidence. Overall, there is partial support that both factors matter."
        )
else:
    score = 35
    explanation = (
        f"Neither relative group size (logit p={logit_size_p:.4f}) nor contest location "
        f"(logit p={logit_dist_p:.4f}) reaches conventional significance in this small sample "
        f"(n={len(df)}). Trends exist but evidence is insufficient to firmly conclude both "
        f"factors influence win probability."
    )

result = {"response": score, "explanation": explanation}
print("\nResult:", result)

with open("conclusion.txt", "w") as f:
    json.dump(result, f)

print("conclusion.txt written.")
