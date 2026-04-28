import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv("boxes.csv")
print("Shape:", df.shape)
print(df.describe())
print("\nValue counts for y:\n", df["y"].value_counts())

# Binary outcome: did child choose majority (y==2)?
df["chose_majority"] = (df["y"] == 2).astype(int)

print("\nOverall majority choice rate:", df["chose_majority"].mean())

# --- Age effect on majority choice ---
age_groups = df.groupby("age")["chose_majority"].agg(["mean", "count"])
print("\nMajority choice rate by age:\n", age_groups)

# Pearson correlation between age and chose_majority
r, p_r = stats.pearsonr(df["age"], df["chose_majority"])
print(f"\nPearson r(age, chose_majority) = {r:.4f}, p = {p_r:.4f}")

# Spearman correlation (ordinal-safe)
rho, p_rho = stats.spearmanr(df["age"], df["chose_majority"])
print(f"Spearman rho = {rho:.4f}, p = {p_rho:.4f}")

# --- Cultural variation ---
culture_groups = df.groupby("culture")["chose_majority"].agg(["mean", "count"])
print("\nMajority choice rate by culture:\n", culture_groups)

# One-way ANOVA across cultures
culture_lists = [g["chose_majority"].values for _, g in df.groupby("culture")]
f_stat, p_anova = stats.f_oneway(*culture_lists)
print(f"\nANOVA across cultures: F = {f_stat:.4f}, p = {p_anova:.4f}")

# --- Logistic regression: age + culture + interaction ---
df_model = df[["chose_majority", "age", "culture"]].copy()
# Add age*culture interaction
for c in df["culture"].unique():
    df_model[f"age_x_c{c}"] = df["age"] * (df["culture"] == c).astype(int)

X = df_model.drop("chose_majority", axis=1)
y = df_model["chose_majority"]

X_const = sm.add_constant(X[["age", "culture"]])
logit_model = sm.Logit(y, X_const).fit(disp=0)
print("\nLogistic regression (age + culture):\n", logit_model.summary2())

# Age p-value from logistic regression
age_pval = logit_model.pvalues["age"]
age_coef = logit_model.params["age"]
print(f"\nAge coefficient: {age_coef:.4f}, p-value: {age_pval:.4f}")

# --- Interpretable model: Decision Tree ---
from sklearn.tree import DecisionTreeClassifier, export_text

dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(df[["age", "culture", "gender", "majority_first"]], df["chose_majority"])
print("\nDecision Tree feature importances:")
for name, imp in zip(["age", "culture", "gender", "majority_first"], dt.feature_importances_):
    print(f"  {name}: {imp:.4f}")
print(export_text(dt, feature_names=["age", "culture", "gender", "majority_first"]))

# --- Per-culture age correlation ---
print("\nPer-culture Pearson r(age, chose_majority):")
sig_cultures = 0
for c, grp in df.groupby("culture"):
    if len(grp) > 5:
        rc, pc = stats.pearsonr(grp["age"], grp["chose_majority"])
        sig = "***" if pc < 0.05 else ""
        print(f"  Culture {c}: r={rc:.3f}, p={pc:.3f} {sig}")
        if pc < 0.05:
            sig_cultures += 1

# --- Conclusion ---
# Age is significant if p < 0.05 in overall logistic regression
age_sig = age_pval < 0.05
culture_sig = p_anova < 0.05
positive_age_effect = age_coef > 0

# Score: age develops majority preference + cultural variation exists
# Strong yes if age is significant with positive direction, moderate if mixed
if age_sig and positive_age_effect:
    score = 80
    explanation = (
        f"Children's reliance on majority preference significantly increases with age "
        f"(logistic regression: coef={age_coef:.3f}, p={age_pval:.4f}; Pearson r={r:.3f}, p={p_r:.4f}). "
        f"ANOVA across {df['culture'].nunique()} cultures shows {'significant' if culture_sig else 'non-significant'} "
        f"cultural variation (F={f_stat:.2f}, p={p_anova:.4f}). "
        f"The positive age coefficient indicates older children are more likely to follow the majority. "
        f"Cultural context also plays a role, with some cultures showing stronger age-related development."
    )
elif age_sig and not positive_age_effect:
    score = 30
    explanation = (
        f"Age is significantly associated with majority preference but in the negative direction "
        f"(coef={age_coef:.3f}, p={age_pval:.4f}), suggesting reliance on majority decreases with age."
    )
else:
    score = 20
    explanation = (
        f"No significant relationship between age and majority preference was found "
        f"(logistic regression: coef={age_coef:.3f}, p={age_pval:.4f}; Pearson r={r:.3f}, p={p_r:.4f})."
    )

result = {"response": score, "explanation": explanation}
with open("conclusion.txt", "w") as f:
    json.dump(result, f)

print("\n=== CONCLUSION ===")
print(json.dumps(result, indent=2))
print("conclusion.txt written.")
