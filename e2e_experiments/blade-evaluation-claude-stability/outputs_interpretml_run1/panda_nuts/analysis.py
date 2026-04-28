import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
from interpret.glassbox import ExplainableBoostingRegressor

# Load data
df = pd.read_csv("panda_nuts.csv")
print("Shape:", df.shape)
print(df.describe())
print(df.head())

# Compute efficiency (nuts per second)
df["efficiency"] = df["nuts_opened"] / df["seconds"]

# Encode categorical variables
df["sex_enc"] = (df["sex"] == "m").astype(int)
df["help_enc"] = (df["help"] == "y").astype(int)

print("\nEfficiency summary:")
print(df["efficiency"].describe())

# --- Statistical tests ---

# 1. Age vs efficiency (Pearson correlation)
r_age, p_age = stats.pearsonr(df["age"], df["efficiency"])
print(f"\nAge-efficiency correlation: r={r_age:.3f}, p={p_age:.4f}")

# 2. Sex vs efficiency (t-test)
male_eff = df[df["sex"] == "m"]["efficiency"]
female_eff = df[df["sex"] == "f"]["efficiency"]
t_sex, p_sex = stats.ttest_ind(male_eff, female_eff)
print(f"Sex t-test: t={t_sex:.3f}, p={p_sex:.4f}")
print(f"  Male mean={male_eff.mean():.3f}, Female mean={female_eff.mean():.3f}")

# 3. Help vs efficiency (t-test)
help_y = df[df["help"] == "y"]["efficiency"]
help_n = df[df["help"] == "N"]["efficiency"]
t_help, p_help = stats.ttest_ind(help_y, help_n)
print(f"Help t-test: t={t_help:.3f}, p={p_help:.4f}")
print(f"  Help=y mean={help_y.mean():.3f}, Help=N mean={help_n.mean():.3f}")

# --- OLS regression ---
X_ols = df[["age", "sex_enc", "help_enc"]].copy()
X_ols = sm.add_constant(X_ols)
model_ols = sm.OLS(df["efficiency"], X_ols).fit()
print("\nOLS Results:")
print(model_ols.summary())

# --- EBM (interpretable boosting) ---
X = df[["age", "sex_enc", "help_enc"]].values
y = df["efficiency"].values
ebm = ExplainableBoostingRegressor(random_state=42)
ebm.fit(X, y)
print("\nEBM feature importances:")
for name, imp in zip(["age", "sex", "help"], ebm.term_importances()):
    print(f"  {name}: {imp:.4f}")

# --- Summarize findings ---
findings = {
    "age_r": r_age,
    "age_p": p_age,
    "sex_p": p_sex,
    "help_p": p_help,
    "ols_age_coef": model_ols.params["age"],
    "ols_age_p": model_ols.pvalues["age"],
    "ols_sex_p": model_ols.pvalues["sex_enc"],
    "ols_help_p": model_ols.pvalues["help_enc"],
}
print("\nSummary:", findings)

# Determine response score
# All three factors together influence efficiency — age is likely significant (older = better),
# sex may differ, help may boost efficiency.
sig_count = sum([
    p_age < 0.05,
    p_sex < 0.05,
    p_help < 0.05,
])
ols_sig_count = sum([
    model_ols.pvalues["age"] < 0.05,
    model_ols.pvalues["sex_enc"] < 0.05,
    model_ols.pvalues["help_enc"] < 0.05,
])

# Build explanation
parts = []
if p_age < 0.05:
    parts.append(f"age is significantly correlated with efficiency (r={r_age:.2f}, p={p_age:.4f})")
else:
    parts.append(f"age is not significantly correlated with efficiency (r={r_age:.2f}, p={p_age:.4f})")

if p_sex < 0.05:
    parts.append(f"sex significantly predicts efficiency (p={p_sex:.4f})")
else:
    parts.append(f"sex does not significantly predict efficiency (p={p_sex:.4f})")

if p_help < 0.05:
    parts.append(f"receiving help significantly predicts efficiency (p={p_help:.4f})")
else:
    parts.append(f"receiving help does not significantly predict efficiency (p={p_help:.4f})")

# Response: question asks HOW they influence (not yes/no for each), so we assess overall influence.
# If at least age is significant, there IS a combined influence story; score higher.
# Score based on proportion of significant predictors.
if sig_count == 3:
    response = 85
elif sig_count == 2:
    response = 70
elif sig_count == 1:
    response = 55
else:
    response = 25

explanation = (
    f"Analysis of nut-cracking efficiency (nuts/second): {'; '.join(parts)}. "
    f"{sig_count}/3 predictors are individually significant. "
    f"OLS regression confirms {ols_sig_count}/3 predictors significant. "
    f"Overall, age, sex, and help collectively {'do' if sig_count > 0 else 'do not'} "
    f"influence efficiency, with age being the primary driver if significant."
)

result = {"response": response, "explanation": explanation}
print("\nConclusion:", result)

with open("conclusion.txt", "w") as f:
    json.dump(result, f)

print("conclusion.txt written.")
