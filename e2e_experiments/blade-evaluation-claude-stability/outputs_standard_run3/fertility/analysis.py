import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor

# Load data
df = pd.read_csv("fertility.csv")
print("Shape:", df.shape)
print(df.head())
print(df.describe())

# Parse dates
for col in ["DateTesting", "StartDateofLastPeriod", "StartDateofPeriodBeforeLast"]:
    df[col] = pd.to_datetime(df[col])

# Compute cycle phase / fertility proxy
# Days since last period
df["DaysSinceLastPeriod"] = (df["DateTesting"] - df["StartDateofLastPeriod"]).dt.days

# Estimated day in cycle
df["CycleDay"] = df["DaysSinceLastPeriod"]

# Fertility window: ovulation ~ cycle_length - 14 days from start
# High fertility ~days 10-17 (rough estimate for 28-day cycle)
df["CycleLength"] = df["ReportedCycleLength"].fillna(28)
df["OvulationDay"] = df["CycleLength"] - 14
df["DaysFromOvulation"] = (df["CycleDay"] - df["OvulationDay"]).abs()

# Fertility score: closer to ovulation = higher fertility
df["FertilityScore"] = -df["DaysFromOvulation"]  # higher = closer to ovulation

# Compute composite religiosity score
df["Religiosity"] = df[["Rel1", "Rel2", "Rel3"]].mean(axis=1)

print("\nFertilityScore stats:", df["FertilityScore"].describe())
print("Religiosity stats:", df["Religiosity"].describe())

# Filter to rows with valid cycle data
valid = df.dropna(subset=["DaysSinceLastPeriod", "CycleLength", "Religiosity"])
print(f"\nValid rows: {len(valid)}")

# Correlation between fertility score and religiosity
r, p = stats.pearsonr(valid["FertilityScore"], valid["Religiosity"])
print(f"\nPearson r (FertilityScore vs Religiosity): r={r:.4f}, p={p:.4f}")

r_sp, p_sp = stats.spearmanr(valid["FertilityScore"], valid["Religiosity"])
print(f"Spearman r: r={r_sp:.4f}, p={p_sp:.4f}")

# Divide into high-fertility (near ovulation) vs low-fertility groups
median_days = valid["DaysFromOvulation"].median()
high_fert = valid[valid["DaysFromOvulation"] <= median_days]["Religiosity"]
low_fert = valid[valid["DaysFromOvulation"] > median_days]["Religiosity"]

t_stat, t_p = stats.ttest_ind(high_fert, low_fert)
print(f"\nT-test (high vs low fertility religiosity): t={t_stat:.4f}, p={t_p:.4f}")
print(f"High fertility mean: {high_fert.mean():.4f}, Low fertility mean: {low_fert.mean():.4f}")

# OLS regression
X = sm.add_constant(valid[["FertilityScore", "CycleDay", "CycleLength"]])
model = sm.OLS(valid["Religiosity"], X).fit()
print("\nOLS summary:")
print(model.summary())

# Ridge regression
ridge = Ridge(alpha=1.0)
features = ["FertilityScore", "CycleDay", "CycleLength", "Sure1", "Sure2", "Relationship"]
X_ridge = valid[features].fillna(valid[features].mean())
ridge.fit(X_ridge, valid["Religiosity"])
print("\nRidge coefficients:")
for feat, coef in zip(features, ridge.coef_):
    print(f"  {feat}: {coef:.4f}")

# Decision tree for interpretability
dt = DecisionTreeRegressor(max_depth=3, random_state=42)
dt.fit(X_ridge, valid["Religiosity"])
print("\nDecision tree feature importances:")
for feat, imp in sorted(zip(features, dt.feature_importances_), key=lambda x: -x[1]):
    print(f"  {feat}: {imp:.4f}")

# Summary
fertility_p = p  # pearson p-value
fertility_effect = abs(r)

# Determine response score
# p < 0.05 = statistically significant
# Effect size interpretation
if fertility_p < 0.05:
    base_score = 65 + int(fertility_effect * 30)
else:
    base_score = max(10, 30 - int((1 - fertility_effect) * 20))

response_score = min(100, max(0, base_score))

explanation = (
    f"Analysis of 275 women's menstrual cycle data and religiosity scores. "
    f"Fertility was estimated by proximity to ovulation (cycle_length - 14 days). "
    f"Pearson correlation between fertility score and composite religiosity: r={r:.4f}, p={p:.4f}. "
    f"Spearman correlation: r={r_sp:.4f}, p={p_sp:.4f}. "
    f"T-test comparing high vs low fertility groups: t={t_stat:.4f}, p={t_p:.4f}. "
    f"High-fertility mean religiosity={high_fert.mean():.2f}, Low-fertility mean={low_fert.mean():.2f}. "
    f"OLS regression p-value for FertilityScore: {model.pvalues.get('FertilityScore', 'N/A')}. "
    f"The evidence {'supports' if fertility_p < 0.05 else 'does not support'} a statistically significant "
    f"relationship between hormonal fluctuations (fertility) and religiosity in this sample."
)

result = {"response": response_score, "explanation": explanation}
print("\nResult:", result)

with open("conclusion.txt", "w") as f:
    json.dump(result, f)

print("conclusion.txt written.")
