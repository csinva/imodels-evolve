import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge
from interpret.glassbox import ExplainableBoostingRegressor

# Load data
df = pd.read_csv("fertility.csv")
print("Shape:", df.shape)
print(df.head())
print(df.describe())

# Parse dates
for col in ["DateTesting", "StartDateofLastPeriod", "StartDateofPeriodBeforeLast"]:
    df[col] = pd.to_datetime(df[col])

# Compute cycle length from dates
df["ComputedCycleLength"] = (df["StartDateofLastPeriod"] - df["StartDateofPeriodBeforeLast"]).dt.days

# Days since last period start (proxy for cycle phase)
df["DaysSinceLast"] = (df["DateTesting"] - df["StartDateofLastPeriod"]).dt.days

# Cycle length to use: prefer computed if reasonable
cycle_len = df["ReportedCycleLength"].fillna(df["ComputedCycleLength"])
cycle_len = cycle_len.clip(21, 38)

# Estimate day of cycle
df["DayOfCycle"] = df["DaysSinceLast"].clip(0, None)

# Fertility window: days 10-17 of cycle typically
# Compute normalized day position (0=start, 1=end of cycle)
df["CyclePos"] = df["DayOfCycle"] / cycle_len

# High-fertility flag: roughly days 10-17 of a 28-day cycle -> fraction ~0.36-0.61
df["HighFertility"] = ((df["DayOfCycle"] >= 10) & (df["DayOfCycle"] <= 17)).astype(int)

# Composite religiosity score
df["Religiosity"] = df[["Rel1", "Rel2", "Rel3"]].mean(axis=1)

print("\nFertility group sizes:")
print(df["HighFertility"].value_counts())

# t-test: high vs low fertility religiosity
hi = df[df["HighFertility"] == 1]["Religiosity"]
lo = df[df["HighFertility"] == 0]["Religiosity"]
t_stat, p_val = stats.ttest_ind(hi, lo)
print(f"\nHigh fertility religiosity: mean={hi.mean():.3f}, n={len(hi)}")
print(f"Low fertility religiosity:  mean={lo.mean():.3f}, n={len(lo)}")
print(f"t-test: t={t_stat:.3f}, p={p_val:.4f}")

# Pearson correlation between cycle position and religiosity
mask = df["CyclePos"].between(0, 1)
r, p_r = stats.pearsonr(df.loc[mask, "CyclePos"], df.loc[mask, "Religiosity"])
print(f"\nPearson r(CyclePos, Religiosity) = {r:.3f}, p={p_r:.4f}")

# OLS regression
X = sm.add_constant(df.loc[mask, "CyclePos"])
ols = sm.OLS(df.loc[mask, "Religiosity"], X).fit()
print(ols.summary())

# EBM model
features = ["DayOfCycle", "CyclePos", "HighFertility", "ReportedCycleLength"]
sub = df[features + ["Religiosity"]].dropna()
ebm = ExplainableBoostingRegressor(random_state=42)
ebm.fit(sub[features], sub["Religiosity"])
importances = dict(zip(features, ebm.term_importances()))
print("\nEBM feature importances:", importances)

# Summary judgement
# Research question: Does hormonal fluctuations (fertility) affect religiosity?
# Use p-value from t-test and correlation as primary evidence
significant = p_val < 0.05 or p_r < 0.05

# Compute response score: if not significant -> low score
if p_val < 0.01:
    response = 75
elif p_val < 0.05:
    response = 60
else:
    # Also check correlation p-value
    if p_r < 0.05:
        response = 55
    else:
        response = 20  # no significant effect found

explanation = (
    f"The research question asks whether hormonal fluctuations associated with fertility "
    f"affect women's religiosity. Using a composite religiosity score (mean of Rel1/Rel2/Rel3), "
    f"women were classified as high-fertility (days 10-17 of cycle, n={len(hi)}) vs low-fertility "
    f"(n={len(lo)}). T-test: t={t_stat:.3f}, p={p_val:.4f}. Pearson correlation between normalized "
    f"cycle position and religiosity: r={r:.3f}, p={p_r:.4f}. "
    f"High-fertility mean religiosity={hi.mean():.3f}, Low-fertility mean={lo.mean():.3f}. "
    f"EBM importances: {importances}. "
    f"{'Statistically significant effect found.' if significant else 'No statistically significant effect found.'}"
)

result = {"response": response, "explanation": explanation}
with open("conclusion.txt", "w") as f:
    json.dump(result, f)

print("\nconclusion.txt written.")
print(json.dumps(result, indent=2))
