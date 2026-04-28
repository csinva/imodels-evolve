import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv("mortgage.csv")
print("Shape:", df.shape)
print(df.head())
print(df.describe())

# Research question: How does gender affect mortgage approval?
# female=1 means female applicant, deny=1 means denied

print("\n--- Denial rates by gender ---")
denial_by_gender = df.groupby("female")["deny"].agg(["mean", "count", "sum"])
denial_by_gender.index = ["Male", "Female"]
print(denial_by_gender)

male_denials = df[df["female"] == 0]["deny"]
female_denials = df[df["female"] == 1]["deny"]

print(f"\nMale denial rate: {male_denials.mean():.4f} (n={len(male_denials)})")
print(f"Female denial rate: {female_denials.mean():.4f} (n={len(female_denials)})")

# Chi-square test
contingency = pd.crosstab(df["female"], df["deny"])
chi2, p_chi2, dof, expected = stats.chi2_contingency(contingency)
print(f"\nChi-square test: chi2={chi2:.4f}, p={p_chi2:.4f}")

# t-test on deny (binary outcome)
t_stat, p_ttest = stats.ttest_ind(male_denials, female_denials)
print(f"t-test: t={t_stat:.4f}, p={p_ttest:.4f}")

# Logistic regression - simple (female only)
df_simple = df[["female", "deny"]].dropna()
X_simple = sm.add_constant(df_simple[["female"]])
logit_simple = sm.Logit(df_simple["deny"], X_simple).fit(disp=0)
print("\n--- Simple logistic regression (female -> deny) ---")
print(logit_simple.summary())

# Logistic regression - controlled for confounders
controls = ["female", "black", "housing_expense_ratio", "self_employed", "married",
            "mortgage_credit", "consumer_credit", "bad_history", "PI_ratio", "loan_to_value", "denied_PMI"]
df_clean = df[controls + ["deny"]].dropna()
X_full = sm.add_constant(df_clean[controls])
logit_full = sm.Logit(df_clean["deny"], X_full).fit(disp=0)
print("\n--- Full logistic regression (controlled) ---")
print(logit_full.summary())

female_coef = logit_full.params["female"]
female_pval = logit_full.pvalues["female"]
print(f"\nFemale coefficient: {female_coef:.4f}, p-value: {female_pval:.4f}")

# Decision tree for feature importances
from sklearn.tree import DecisionTreeClassifier
X_tree = df_clean[controls]
y_tree = df_clean["deny"]
dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_tree, y_tree)
importances = pd.Series(dt.feature_importances_, index=controls).sort_values(ascending=False)
print("\n--- Decision Tree Feature Importances ---")
print(importances)

# Summary
simple_p = p_chi2
controlled_p = female_pval
simple_diff = female_denials.mean() - male_denials.mean()

print(f"\n=== SUMMARY ===")
print(f"Raw denial rate difference (Female - Male): {simple_diff:.4f}")
print(f"Chi-square p-value (unadjusted): {simple_p:.4f}")
print(f"Logistic regression p-value for female (controlled): {controlled_p:.4f}")
print(f"Female coefficient in full model (log-odds): {female_coef:.4f}")

# Determine response score
# A significant effect of gender = "Yes" (higher score)
# If female is significant after controls, gender matters
if controlled_p < 0.05:
    if female_coef < 0:  # negative coef means female -> lower denial
        response = 70
        explanation = (f"Gender (female) has a statistically significant effect on mortgage denial "
                       f"(p={controlled_p:.4f}) even after controlling for creditworthiness variables. "
                       f"The negative coefficient ({female_coef:.4f}) indicates female applicants are "
                       f"less likely to be denied (i.e., more likely to be approved) than male applicants. "
                       f"Raw denial rates: Male={male_denials.mean():.3f}, Female={female_denials.mean():.3f}. "
                       f"Chi-square p={simple_p:.4f}. Gender does significantly affect approval, though "
                       f"the direction favors women.")
    else:
        response = 70
        explanation = (f"Gender (female) has a statistically significant effect on mortgage denial "
                       f"(p={controlled_p:.4f}) even after controlling for creditworthiness. "
                       f"The positive coefficient ({female_coef:.4f}) indicates female applicants are "
                       f"more likely to be denied. Raw denial rates: Male={male_denials.mean():.3f}, "
                       f"Female={female_denials.mean():.3f}. Chi-square p={simple_p:.4f}.")
else:
    response = 35
    explanation = (f"After controlling for creditworthiness variables (credit scores, debt ratios, etc.), "
                   f"gender (female) does NOT have a statistically significant effect on mortgage denial "
                   f"(p={controlled_p:.4f}, coef={female_coef:.4f}). "
                   f"Raw denial rates differ slightly: Male={male_denials.mean():.3f}, Female={female_denials.mean():.3f} "
                   f"(chi-square p={simple_p:.4f}), but this disappears when controlling for other factors. "
                   f"The most important predictors are creditworthiness variables, not gender.")

result = {"response": response, "explanation": explanation}
print(f"\nFinal result: {result}")

with open("conclusion.txt", "w") as f:
    json.dump(result, f)

print("conclusion.txt written.")
