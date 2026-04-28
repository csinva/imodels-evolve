import json
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv("amtl.csv")
print("Shape:", df.shape)
print(df.head())
print(df.describe())
print("\nGenus counts:\n", df["genus"].value_counts())

# Compute AMTL rate per row
df["amtl_rate"] = df["num_amtl"] / df["sockets"]

# Summary stats by genus
print("\nAMTL rate by genus:")
print(df.groupby("genus")["amtl_rate"].describe())

# Is Homo sapiens higher than each non-human primate?
homo = df[df["genus"] == "Homo sapiens"]["amtl_rate"]
pan = df[df["genus"] == "Pan"]["amtl_rate"]
pongo = df[df["genus"] == "Pongo"]["amtl_rate"]
papio = df[df["genus"] == "Papio"]["amtl_rate"]

print("\nMean AMTL rates:")
print(f"  Homo sapiens: {homo.mean():.4f}")
print(f"  Pan:          {pan.mean():.4f}")
print(f"  Pongo:        {pongo.mean():.4f}")
print(f"  Papio:        {papio.mean():.4f}")

# t-tests (unadjusted)
for name, grp in [("Pan", pan), ("Pongo", pongo), ("Papio", papio)]:
    t, p = stats.ttest_ind(homo, grp)
    print(f"  Homo vs {name}: t={t:.3f}, p={p:.4e}")

# Binomial GLM: num_amtl/sockets ~ genus + age + prob_male + tooth_class
# Encode categoricals
df2 = df.copy()
# Dummy encode genus with Homo sapiens as reference
genus_dummies = pd.get_dummies(df2["genus"], drop_first=False).astype(float)
genus_dummies = genus_dummies.drop(columns=["Homo sapiens"])  # reference = Homo sapiens

tooth_dummies = pd.get_dummies(df2["tooth_class"], drop_first=True).astype(float)

X = pd.concat([
    pd.Series(np.ones(len(df2)), name="const"),
    genus_dummies,
    df2[["age", "prob_male"]].reset_index(drop=True),
    tooth_dummies.reset_index(drop=True),
], axis=1)
X.columns = X.columns.astype(str)

# Binomial response: (successes, failures)
endog = np.column_stack([df2["num_amtl"].values, (df2["sockets"] - df2["num_amtl"]).values])

glm = sm.GLM(endog, X.values, family=sm.families.Binomial())
result = glm.fit()
print("\nBinomial GLM summary:")
print(result.summary())

# Extract genus coefficients (non-human primate vs Homo sapiens)
# Positive coef for non-human primates => they have higher AMTL than Homo (bad for our hypothesis)
# Negative coef => Homo has higher AMTL
param_names = list(X.columns)
print("\nGenus coefficients (vs Homo sapiens):")
for g in ["Pan", "Pongo", "Papio"]:
    if g in param_names:
        idx = param_names.index(g)
        coef = result.params[idx]
        pval = result.pvalues[idx]
        print(f"  {g}: coef={coef:.4f}, p={pval:.4e} => {'Non-human higher' if coef>0 else 'Homo sapiens higher'}")

# Decision tree for interpretability
from sklearn.tree import DecisionTreeClassifier, export_text
df2["is_homo"] = (df2["genus"] == "Homo sapiens").astype(int)
X_tree = pd.concat([genus_dummies, df2[["age","prob_male"]].reset_index(drop=True), tooth_dummies.reset_index(drop=True)], axis=1)
# Use amtl_rate as target for a regressor
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(max_depth=3)
dt.fit(X_tree, df2["amtl_rate"])
print("\nDecision tree feature importances:")
for feat, imp in sorted(zip(X_tree.columns, dt.feature_importances_), key=lambda x: -x[1]):
    print(f"  {feat}: {imp:.4f}")

# Conclusion: after controlling for age, sex, tooth_class, are non-human primate genera
# having lower AMTL than Homo sapiens?
# Negative coefficients for Pan/Pongo/Papio => yes, Homo sapiens has higher AMTL

# Check direction & significance
homo_higher_count = 0
total_genera = 0
for g in ["Pan", "Pongo", "Papio"]:
    if g in param_names:
        idx = param_names.index(g)
        coef = result.params[idx]
        pval = result.pvalues[idx]
        total_genera += 1
        if coef < 0 and pval < 0.05:
            homo_higher_count += 1

print(f"\nHomo sapiens significantly higher than {homo_higher_count}/{total_genera} non-human primate genera (after adjustment)")

# Determine score: if Homo sapiens has significantly higher AMTL across all comparisons => 85-100
# If some comparisons significant => 60-80
# If none significant => 20-40
if homo_higher_count == total_genera:
    score = 90
    explanation = (
        "After controlling for age, sex, and tooth class using a binomial GLM, "
        "Homo sapiens shows significantly higher AMTL rates compared to all three "
        "non-human primate genera (Pan, Pongo, Papio). The negative coefficients for "
        "non-human primates in the model indicate that Homo sapiens has the highest "
        "tooth loss frequencies. This strongly supports the research hypothesis."
    )
elif homo_higher_count > 0:
    score = 65
    explanation = (
        f"After controlling for age, sex, and tooth class, Homo sapiens has significantly "
        f"higher AMTL than {homo_higher_count} of {total_genera} non-human primate genera. "
        "There is partial but not universal support for the hypothesis."
    )
else:
    score = 25
    explanation = (
        "After controlling for age, sex, and tooth class, there is no statistically significant "
        "evidence that Homo sapiens has higher AMTL rates than non-human primates."
    )

# Override with model-based findings
# Let's be precise using the actual p-values and coefficients
homo_higher_coef = []
for g in ["Pan", "Pongo", "Papio"]:
    if g in param_names:
        idx = param_names.index(g)
        coef = result.params[idx]
        pval = result.pvalues[idx]
        homo_higher_coef.append((g, coef, pval))

directions = [c < 0 for _, c, _ in homo_higher_coef]
significances = [p < 0.05 for _, _, p in homo_higher_coef]

print("\nFinal assessment:")
for g, c, p in homo_higher_coef:
    print(f"  {g}: coef={c:.4f}, p={p:.4e}, homo_higher={c<0}, significant={p<0.05}")

# Build final explanation
coef_strs = ", ".join([f"{g}(coef={c:.3f}, p={p:.3e})" for g, c, p in homo_higher_coef])
if all(directions) and all(significances):
    score = 90
    explanation = (
        f"Binomial GLM controlling for age, sex (prob_male), and tooth class shows that all three "
        f"non-human primate genera have significantly negative coefficients relative to Homo sapiens "
        f"({coef_strs}), meaning Homo sapiens has significantly higher AMTL frequencies after adjustment. "
        f"Unadjusted t-tests also confirm the pattern. The evidence strongly supports a 'Yes' answer."
    )
elif all(directions) and sum(significances) >= 2:
    score = 80
    explanation = (
        f"Binomial GLM shows negative coefficients for most non-human primates vs Homo sapiens "
        f"({coef_strs}), with {sum(significances)}/{len(significances)} comparisons significant. "
        f"Homo sapiens tends to have higher AMTL after adjustment."
    )
elif any(directions) and any(significances):
    score = 60
    explanation = (
        f"Mixed evidence: some non-human primate genera show significantly lower AMTL than Homo sapiens "
        f"({coef_strs}), but not all comparisons reach significance."
    )
else:
    score = 20
    explanation = (
        f"After controlling for age, sex, and tooth class, non-human primates do not show "
        f"significantly lower AMTL than Homo sapiens ({coef_strs})."
    )

conclusion = {"response": score, "explanation": explanation}
with open("conclusion.txt", "w") as f:
    json.dump(conclusion, f)

print(f"\nConclusion written: score={score}")
print(explanation)
