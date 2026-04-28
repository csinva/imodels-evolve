import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv("affairs.csv")

# Encode children as binary
df["children_bin"] = (df["children"] == "yes").astype(int)
df["gender_bin"] = (df["gender"] == "male").astype(int)

# Summary stats by children group
children_yes = df[df["children"] == "yes"]["affairs"]
children_no = df[df["children"] == "no"]["affairs"]

print("=== Summary Statistics ===")
print(f"Children=Yes  n={len(children_yes)}  mean={children_yes.mean():.3f}  std={children_yes.std():.3f}")
print(f"Children=No   n={len(children_no)}  mean={children_no.mean():.3f}  std={children_no.std():.3f}")

# Mann-Whitney U test (data is non-normal, ordinal-ish)
u_stat, p_mw = stats.mannwhitneyu(children_yes, children_no, alternative="two-sided")
print(f"\nMann-Whitney U: U={u_stat:.1f}, p={p_mw:.4f}")

# t-test
t_stat, p_t = stats.ttest_ind(children_yes, children_no)
print(f"Independent t-test: t={t_stat:.3f}, p={p_t:.4f}")

# OLS regression: affairs ~ children + controls
X = df[["children_bin", "age", "yearsmarried", "religiousness", "education", "occupation", "rating", "gender_bin"]]
X = sm.add_constant(X)
y = df["affairs"]
model = sm.OLS(y, X).fit()
print("\n=== OLS Regression (with controls) ===")
print(model.summary())

children_coef = model.params["children_bin"]
children_pval = model.pvalues["children_bin"]
print(f"\nChildren coefficient: {children_coef:.4f}, p-value: {children_pval:.4f}")

# Decision tree for feature importance
from sklearn.preprocessing import LabelEncoder
tree = DecisionTreeRegressor(max_depth=4, random_state=42)
feature_cols = ["children_bin", "age", "yearsmarried", "religiousness", "education", "occupation", "rating", "gender_bin"]
tree.fit(df[feature_cols], df["affairs"])
importances = dict(zip(feature_cols, tree.feature_importances_))
print("\n=== Decision Tree Feature Importances ===")
for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
    print(f"  {feat}: {imp:.4f}")

# Determine conclusion
# Children=Yes has HIGHER mean affairs; coefficient in OLS with controls may differ
# Key: does having children DECREASE affairs?
# Raw means: children=yes has higher affairs (confounded by years married / age)
# With controls, check sign and significance of children_bin

print(f"\nRaw means: yes={children_yes.mean():.3f}, no={children_no.mean():.3f}")
print(f"OLS children coef (controlling for confounders): {children_coef:.4f}, p={children_pval:.4f}")

# If children_coef > 0 (with controls), children INCREASE (not decrease) affairs
# If not significant or positive, answer is "No, children do not decrease affairs"

if children_pval < 0.05 and children_coef < 0:
    response = 70
    explanation = (
        f"Having children is associated with significantly fewer extramarital affairs "
        f"(OLS coef={children_coef:.3f}, p={children_pval:.4f}), controlling for age, years married, "
        f"religiousness, education, occupation, marriage rating, and gender. "
        f"Raw means: yes={children_yes.mean():.3f}, no={children_no.mean():.3f}. "
        f"Mann-Whitney p={p_mw:.4f}."
    )
elif children_pval < 0.05 and children_coef > 0:
    response = 15
    explanation = (
        f"Having children is associated with significantly MORE extramarital affairs, not fewer "
        f"(OLS coef={children_coef:.3f}, p={children_pval:.4f}), controlling for confounders. "
        f"Raw means: yes={children_yes.mean():.3f}, no={children_no.mean():.3f}. "
        f"The evidence contradicts the hypothesis that children decrease affairs."
    )
else:
    response = 20
    explanation = (
        f"No statistically significant relationship found between having children and extramarital affairs "
        f"after controlling for confounders (OLS coef={children_coef:.3f}, p={children_pval:.4f}). "
        f"Raw means: yes={children_yes.mean():.3f}, no={children_no.mean():.3f} (Mann-Whitney p={p_mw:.4f}). "
        f"Children do not appear to significantly decrease engagement in extramarital affairs."
    )

conclusion = {"response": response, "explanation": explanation}
with open("conclusion.txt", "w") as f:
    json.dump(conclusion, f)

print(f"\n=== Conclusion written ===")
print(json.dumps(conclusion, indent=2))
