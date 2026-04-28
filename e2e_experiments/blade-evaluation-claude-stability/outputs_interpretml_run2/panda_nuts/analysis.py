import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv("panda_nuts.csv")
print("Shape:", df.shape)
print(df.head())
print(df.describe())
print(df.dtypes)

# Compute efficiency: nuts per second
df["efficiency"] = df["nuts_opened"] / df["seconds"]

print("\nEfficiency stats:")
print(df["efficiency"].describe())

# Encode categorical variables
df["sex_enc"] = (df["sex"] == "m").astype(int)  # 1=male, 0=female
df["help_enc"] = (df["help"] == "y").astype(int)  # 1=yes, 0=no

print("\nSex distribution:")
print(df.groupby("sex")["efficiency"].describe())

print("\nHelp distribution:")
print(df.groupby("help")["efficiency"].describe())

# --- Statistical tests ---

# 1. Age vs efficiency: Pearson correlation
age_corr, age_p = stats.pearsonr(df["age"], df["efficiency"])
print(f"\nAge-efficiency correlation: r={age_corr:.4f}, p={age_p:.4f}")

# 2. Sex vs efficiency: t-test
male_eff = df[df["sex"] == "m"]["efficiency"]
female_eff = df[df["sex"] == "f"]["efficiency"]
t_sex, p_sex = stats.ttest_ind(male_eff, female_eff)
print(f"Sex t-test: t={t_sex:.4f}, p={p_sex:.4f}")
print(f"  Male mean={male_eff.mean():.4f}, Female mean={female_eff.mean():.4f}")

# 3. Help vs efficiency: t-test
help_yes = df[df["help"] == "y"]["efficiency"]
help_no = df[df["help"] == "N"]["efficiency"]
t_help, p_help = stats.ttest_ind(help_yes, help_no)
print(f"Help t-test: t={t_help:.4f}, p={p_help:.4f}")
print(f"  Help=yes mean={help_yes.mean():.4f}, Help=no mean={help_no.mean():.4f}")

# --- OLS regression ---
X = df[["age", "sex_enc", "help_enc"]].copy()
X = sm.add_constant(X)
y = df["efficiency"]

model = sm.OLS(y, X).fit()
print("\nOLS summary:")
print(model.summary())

# --- Interpretable model: EBM ---
try:
    from interpret.glassbox import ExplainableBoostingRegressor
    ebm = ExplainableBoostingRegressor(random_state=42)
    ebm.fit(df[["age", "sex_enc", "help_enc"]], df["efficiency"])
    importances = ebm.term_importances()
    print("\nEBM feature importances:")
    for name, imp in zip(["age", "sex_enc", "help_enc"], importances):
        print(f"  {name}: {imp:.4f}")
except Exception as e:
    print(f"EBM failed: {e}")

# --- Summarize ---
significant_factors = []
if age_p < 0.05:
    significant_factors.append(f"age (r={age_corr:.3f}, p={age_p:.4f})")
if p_sex < 0.05:
    significant_factors.append(f"sex (p={p_sex:.4f})")
if p_help < 0.05:
    significant_factors.append(f"help (p={p_help:.4f})")

# Score: based on whether any/all factors are significant
# Research question asks HOW all three influence efficiency
# Use a composite score reflecting number of significant factors
n_sig = sum([age_p < 0.05, p_sex < 0.05, p_help < 0.05])
# If age is significant (main driver), give high score
# The question is about INFLUENCE, not just yes/no
# Score based on strength of age effect (main predictor) and significance

if age_p < 0.01:
    response = 80
elif age_p < 0.05:
    response = 65
else:
    response = 40

# Adjust for sex and help significance
response += 5 * (p_sex < 0.05)
response += 5 * (p_help < 0.05)
response = min(response, 95)

explanation = (
    f"Analysis of {len(df)} observations. "
    f"Efficiency (nuts/sec) as outcome. "
    f"Age-efficiency correlation: r={age_corr:.3f}, p={age_p:.4f}. "
    f"Sex difference: male_mean={male_eff.mean():.3f} vs female_mean={female_eff.mean():.3f}, p={p_sex:.4f}. "
    f"Help effect: yes_mean={help_yes.mean():.3f} vs no_mean={help_no.mean():.3f}, p={p_help:.4f}. "
    f"OLS: age coef={model.params['age']:.4f} (p={model.pvalues['age']:.4f}), "
    f"sex coef={model.params['sex_enc']:.4f} (p={model.pvalues['sex_enc']:.4f}), "
    f"help coef={model.params['help_enc']:.4f} (p={model.pvalues['help_enc']:.4f}). "
    f"Significant factors: {significant_factors if significant_factors else 'none at p<0.05'}."
)

result = {"response": int(response), "explanation": explanation}
print("\nResult:", result)

with open("conclusion.txt", "w") as f:
    json.dump(result, f)

print("conclusion.txt written.")
