import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
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
df["sex_enc"] = (df["sex"] == "m").astype(int)
df["help_enc"] = (df["help"] == "y").astype(int)

# --- Statistical tests ---

# 1. Age vs efficiency (Pearson correlation)
age_corr, age_pval = stats.pearsonr(df["age"], df["efficiency"])
print(f"\nAge vs efficiency: r={age_corr:.3f}, p={age_pval:.4f}")

# 2. Sex vs efficiency (t-test)
male_eff = df[df["sex"] == "m"]["efficiency"]
female_eff = df[df["sex"] == "f"]["efficiency"]
t_sex, p_sex = stats.ttest_ind(male_eff, female_eff)
print(f"Sex vs efficiency: t={t_sex:.3f}, p={p_sex:.4f}")
print(f"  Male mean: {male_eff.mean():.4f}, Female mean: {female_eff.mean():.4f}")

# 3. Help vs efficiency (t-test)
help_y = df[df["help"] == "y"]["efficiency"]
help_n = df[df["help"] == "N"]["efficiency"]
t_help, p_help = stats.ttest_ind(help_y, help_n)
print(f"Help vs efficiency: t={t_help:.3f}, p={p_help:.4f}")
print(f"  Help=yes mean: {help_y.mean():.4f}, Help=no mean: {help_n.mean():.4f}")

# --- Multiple regression (OLS with statsmodels) ---
X = df[["age", "sex_enc", "help_enc"]].copy()
X = sm.add_constant(X)
y = df["efficiency"]
model = sm.OLS(y, X).fit()
print("\nOLS Regression Summary:")
print(model.summary())

# --- Interpretable model: Decision Tree ---
X_sklearn = df[["age", "sex_enc", "help_enc"]].values
tree = DecisionTreeRegressor(max_depth=3, random_state=42)
tree.fit(X_sklearn, y)
feat_names = ["age", "sex", "help"]
print("\nDecision Tree feature importances:")
for name, imp in zip(feat_names, tree.feature_importances_):
    print(f"  {name}: {imp:.3f}")

# --- Summarize findings ---
findings = {
    "age_correlation": age_corr,
    "age_pvalue": age_pval,
    "sex_t": t_sex,
    "sex_pvalue": p_sex,
    "help_t": t_help,
    "help_pvalue": p_help,
    "ols_age_pvalue": model.pvalues["age"],
    "ols_sex_pvalue": model.pvalues["sex_enc"],
    "ols_help_pvalue": model.pvalues["help_enc"],
}
print("\nFindings:", findings)

# Determine Likert score
# The question asks HOW these factors influence efficiency —
# we check if at least some are significant
sig_count = sum([
    age_pval < 0.05,
    p_sex < 0.05,
    p_help < 0.05,
])
print(f"\nSignificant predictors: {sig_count}/3")

# Construct response score:
# If all 3 significant -> strong yes (85-95)
# If 2 significant -> moderate yes (65-75)
# If 1 significant -> weak yes (45-55)
# If 0 significant -> no (15-25)
if sig_count == 3:
    response = 90
    explanation = (
        f"All three factors — age, sex, and receiving help — significantly influence nut-cracking efficiency. "
        f"Age correlation r={age_corr:.3f} (p={age_pval:.4f}), sex t-test p={p_sex:.4f}, help t-test p={p_help:.4f}. "
        f"OLS confirms: age p={model.pvalues['age']:.4f}, sex p={model.pvalues['sex_enc']:.4f}, help p={model.pvalues['help_enc']:.4f}. "
        f"The Decision Tree assigns importances: age={tree.feature_importances_[0]:.3f}, sex={tree.feature_importances_[1]:.3f}, help={tree.feature_importances_[2]:.3f}. "
        f"Together these variables strongly explain variation in nut-cracking efficiency."
    )
elif sig_count == 2:
    response = 70
    sig_vars = [name for name, p in [("age", age_pval), ("sex", p_sex), ("help", p_help)] if p < 0.05]
    explanation = (
        f"Two of three factors are statistically significant: {', '.join(sig_vars)}. "
        f"Age: r={age_corr:.3f} (p={age_pval:.4f}), sex: p={p_sex:.4f}, help: p={p_help:.4f}. "
        f"OLS p-values: age={model.pvalues['age']:.4f}, sex={model.pvalues['sex_enc']:.4f}, help={model.pvalues['help_enc']:.4f}. "
        f"These factors meaningfully influence nut-cracking efficiency."
    )
elif sig_count == 1:
    sig_vars = [name for name, p in [("age", age_pval), ("sex", p_sex), ("help", p_help)] if p < 0.05]
    response = 50
    explanation = (
        f"Only one factor is statistically significant: {', '.join(sig_vars)}. "
        f"Age: r={age_corr:.3f} (p={age_pval:.4f}), sex: p={p_sex:.4f}, help: p={p_help:.4f}. "
        f"OLS p-values: age={model.pvalues['age']:.4f}, sex={model.pvalues['sex_enc']:.4f}, help={model.pvalues['help_enc']:.4f}. "
        f"The influence of these factors on efficiency is limited."
    )
else:
    response = 20
    explanation = (
        f"None of the three factors show statistically significant influence on nut-cracking efficiency. "
        f"Age: r={age_corr:.3f} (p={age_pval:.4f}), sex: p={p_sex:.4f}, help: p={p_help:.4f}. "
        f"OLS p-values: age={model.pvalues['age']:.4f}, sex={model.pvalues['sex_enc']:.4f}, help={model.pvalues['help_enc']:.4f}. "
        f"The data does not support a strong influence of these variables on efficiency."
    )

conclusion = {"response": response, "explanation": explanation}
print("\nConclusion:", conclusion)

with open("conclusion.txt", "w") as f:
    json.dump(conclusion, f)

print("conclusion.txt written.")
