import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import json

# Load dataset
df = pd.read_csv("panda_nuts.csv")
print("Shape:", df.shape)
print(df.head())
print(df.describe())
print(df.dtypes)

# Create efficiency metric: nuts per second
df["efficiency"] = df["nuts_opened"] / df["seconds"]
print("\nEfficiency stats:")
print(df["efficiency"].describe())

# Encode categorical variables
df["sex_num"] = (df["sex"] == "m").astype(int)
df["help_num"] = (df["help"] == "y").astype(int)

# --- Age vs efficiency ---
r_age, p_age = stats.pearsonr(df["age"], df["efficiency"])
print(f"\nAge-efficiency correlation: r={r_age:.3f}, p={p_age:.4f}")

# --- Sex vs efficiency (t-test) ---
male_eff = df[df["sex"] == "m"]["efficiency"]
female_eff = df[df["sex"] == "f"]["efficiency"]
t_sex, p_sex = stats.ttest_ind(male_eff, female_eff)
print(f"\nSex t-test: t={t_sex:.3f}, p={p_sex:.4f}")
print(f"Male mean={male_eff.mean():.3f}, Female mean={female_eff.mean():.3f}")

# --- Help vs efficiency (t-test) ---
help_y = df[df["help"] == "y"]["efficiency"]
help_n = df[df["help"] == "N"]["efficiency"]
t_help, p_help = stats.ttest_ind(help_y, help_n)
print(f"\nHelp t-test: t={t_help:.3f}, p={p_help:.4f}")
print(f"Help=yes mean={help_y.mean():.3f}, Help=no mean={help_n.mean():.3f}")

# --- OLS regression with all three predictors ---
X = df[["age", "sex_num", "help_num"]]
X = sm.add_constant(X)
model = sm.OLS(df["efficiency"], X).fit()
print("\nOLS Regression Summary:")
print(model.summary())

# --- Sklearn linear regression for feature importance ---
from sklearn.linear_model import Ridge
ridge = Ridge()
ridge.fit(df[["age", "sex_num", "help_num"]], df["efficiency"])
print("\nRidge coefficients:")
for name, coef in zip(["age", "sex_num", "help_num"], ridge.coef_):
    print(f"  {name}: {coef:.4f}")

# Build conclusion
sig_age = p_age < 0.05
sig_sex = p_sex < 0.05
sig_help = p_help < 0.05

print(f"\nSignificant: age={sig_age}, sex={sig_sex}, help={sig_help}")

# Score: 0-100 on how well these factors collectively predict/influence efficiency
# All three significant -> high score, none -> low
sig_count = sum([sig_age, sig_sex, sig_help])
# Also look at model R-squared
r2 = model.rsquared
print(f"R-squared: {r2:.3f}")

# Research question asks "how do age, sex, and help influence efficiency"
# If all factors are significant, score is high (they do influence it)
# Use p-values and direction of effects
if sig_count == 3:
    score = 85
elif sig_count == 2:
    score = 70
elif sig_count == 1:
    score = 50
else:
    score = 20

# Adjust based on R-squared
score = min(100, max(0, int(score + (r2 - 0.2) * 50)))

explanation = (
    f"Research question: How do age, sex, and help influence nut-cracking efficiency (nuts/sec)? "
    f"Age-efficiency correlation: r={r_age:.3f} (p={p_age:.4f}, sig={sig_age}). "
    f"Sex effect (male vs female): mean {male_eff.mean():.3f} vs {female_eff.mean():.3f} (p={p_sex:.4f}, sig={sig_sex}). "
    f"Help effect (yes vs no): mean {help_y.mean():.3f} vs {help_n.mean():.3f} (p={p_help:.4f}, sig={sig_help}). "
    f"OLS model R2={r2:.3f}. "
    f"{sig_count}/3 predictors significant. "
    f"Conclusion: {'All three factors significantly influence efficiency.' if sig_count==3 else f'{sig_count} factor(s) show significant influence.'}"
)

result = {"response": score, "explanation": explanation}
with open("conclusion.txt", "w") as f:
    json.dump(result, f)

print("\nconclusion.txt written.")
print(result)
