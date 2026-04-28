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
print(df.head())
print(df.describe())
print(df.dtypes)

# Define efficiency as nuts per second
df["efficiency"] = df["nuts_opened"] / df["seconds"]

# Encode categorical variables
df["sex_bin"] = (df["sex"] == "m").astype(int)
df["help_bin"] = (df["help"].str.lower() == "y").astype(int)

print("\nEfficiency summary:")
print(df["efficiency"].describe())

# --- Statistical tests ---

# Age vs efficiency (Pearson correlation)
r_age, p_age = stats.pearsonr(df["age"], df["efficiency"])
print(f"\nAge vs efficiency: r={r_age:.3f}, p={p_age:.4f}")

# Sex vs efficiency (t-test)
eff_m = df.loc[df["sex"] == "m", "efficiency"]
eff_f = df.loc[df["sex"] == "f", "efficiency"]
t_sex, p_sex = stats.ttest_ind(eff_m, eff_f)
print(f"Sex vs efficiency: t={t_sex:.3f}, p={p_sex:.4f}")
print(f"  Male mean={eff_m.mean():.4f}, Female mean={eff_f.mean():.4f}")

# Help vs efficiency (t-test)
eff_help_y = df.loc[df["help"].str.lower() == "y", "efficiency"]
eff_help_n = df.loc[df["help"].str.lower() == "n", "efficiency"]
t_help, p_help = stats.ttest_ind(eff_help_y, eff_help_n)
print(f"Help vs efficiency: t={t_help:.3f}, p={p_help:.4f}")
print(f"  Help=Y mean={eff_help_y.mean():.4f}, Help=N mean={eff_help_n.mean():.4f}")

# --- OLS regression ---
X_ols = sm.add_constant(df[["age", "sex_bin", "help_bin"]])
model_ols = sm.OLS(df["efficiency"], X_ols).fit()
print("\nOLS regression summary:")
print(model_ols.summary())

# --- EBM ---
X = df[["age", "sex_bin", "help_bin"]].values
y = df["efficiency"].values
ebm = ExplainableBoostingRegressor(random_state=42)
ebm.fit(X, y)
importances = ebm.term_importances()
print("\nEBM feature importances:")
for name, imp in zip(["age", "sex_bin", "help_bin"], importances):
    print(f"  {name}: {imp:.4f}")

# --- Ridge regression ---
ridge = Ridge(alpha=1.0)
ridge.fit(X, y)
print("\nRidge coefficients:")
for name, coef in zip(["age", "sex_bin", "help_bin"], ridge.coef_):
    print(f"  {name}: {coef:.4f}")

# --- Synthesize results ---
# Significance threshold
alpha = 0.05

age_sig = p_age < alpha
sex_sig = p_sex < alpha
help_sig = p_help < alpha

# OLS p-values
ols_pvals = model_ols.pvalues
age_ols_p = ols_pvals.get("age", 1.0)
sex_ols_p = ols_pvals.get("sex_bin", 1.0)
help_ols_p = ols_pvals.get("help_bin", 1.0)

print(f"\nSignificance summary:")
print(f"  Age: r={r_age:.3f}, p={p_age:.4f}, sig={age_sig}, OLS p={age_ols_p:.4f}")
print(f"  Sex: t={t_sex:.3f}, p={p_sex:.4f}, sig={sex_sig}, OLS p={sex_ols_p:.4f}")
print(f"  Help: t={t_help:.3f}, p={p_help:.4f}, sig={help_sig}, OLS p={help_ols_p:.4f}")

# Score: how many of the three factors are significant and in the expected direction
# Expected: age+ (older = more efficient), help+ (help = more efficient)
n_sig = sum([age_sig, sex_sig, help_sig])
age_positive = r_age > 0

# Compute overall response score
# Strong influence of all three -> high score; none significant -> low score
# Scale: each sig factor adds ~25 pts; if age is positive (expected), add bonus
score = 0
if age_sig and age_positive:
    score += 35
elif age_sig:
    score += 20
if sex_sig:
    score += 25
if help_sig:
    score += 25

# Cap at 100
score = min(score, 100)

explanation = (
    f"Analysis of nut-cracking efficiency (nuts/second) against age, sex, and help. "
    f"Age: Pearson r={r_age:.3f}, p={p_age:.4f} ({'significant' if age_sig else 'not significant'}), "
    f"OLS p={age_ols_p:.4f}. "
    f"Sex: t={t_sex:.3f}, p={p_sex:.4f} ({'significant' if sex_sig else 'not significant'}), "
    f"OLS p={sex_ols_p:.4f}. "
    f"Help: t={t_help:.3f}, p={p_help:.4f} ({'significant' if help_sig else 'not significant'}), "
    f"OLS p={help_ols_p:.4f}. "
    f"EBM importances: age={importances[0]:.4f}, sex={importances[1]:.4f}, help={importances[2]:.4f}. "
    f"Overall, {n_sig} of 3 factors show statistically significant influence on efficiency. "
    f"Age shows a {'positive' if age_positive else 'negative'} relationship with efficiency."
)

result = {"response": score, "explanation": explanation}
print(f"\nFinal result: {result}")

with open("conclusion.txt", "w") as f:
    json.dump(result, f)

print("conclusion.txt written.")
