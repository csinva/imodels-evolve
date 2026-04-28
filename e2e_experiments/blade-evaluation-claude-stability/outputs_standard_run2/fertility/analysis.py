import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import json

df = pd.read_csv("fertility.csv")

# Parse dates
for col in ["DateTesting", "StartDateofLastPeriod", "StartDateofPeriodBeforeLast"]:
    df[col] = pd.to_datetime(df[col], format="%m/%d/%y", errors="coerce")

# Compute cycle length from actual dates (as a check)
df["ActualCycleLength"] = (df["StartDateofLastPeriod"] - df["StartDateofPeriodBeforeLast"]).dt.days

# Days since last period (proxy for cycle phase)
df["DaysSinceLastPeriod"] = (df["DateTesting"] - df["StartDateofLastPeriod"]).dt.days

# Use reported cycle length to estimate fertility window
# Ovulation typically occurs ~14 days before next period
df["CycleLength"] = df["ReportedCycleLength"].fillna(df["ActualCycleLength"])
df["DaysUntilNextPeriod"] = df["CycleLength"] - df["DaysSinceLastPeriod"]
df["DaysUntilOvulation"] = df["DaysUntilNextPeriod"] - 14

# Fertile window: within ~5 days of ovulation
df["NearOvulation"] = df["DaysUntilOvulation"].abs() <= 5

# Create composite religiosity score
df["Religiosity"] = df[["Rel1", "Rel2", "Rel3"]].mean(axis=1)

print("Dataset shape:", df.shape)
print("\nBasic stats:")
print(df[["Religiosity", "DaysSinceLastPeriod", "DaysUntilOvulation", "NearOvulation"]].describe())

print("\nMean religiosity by ovulation proximity:")
print(df.groupby("NearOvulation")["Religiosity"].agg(["mean", "std", "count"]))

# T-test: near ovulation vs not
near = df[df["NearOvulation"] == True]["Religiosity"].dropna()
far = df[df["NearOvulation"] == False]["Religiosity"].dropna()
t_stat, p_val = stats.ttest_ind(near, far)
print(f"\nT-test (near vs far from ovulation): t={t_stat:.4f}, p={p_val:.4f}")

# Correlation: days until ovulation vs religiosity
valid = df[["DaysUntilOvulation", "Religiosity"]].dropna()
r, p_corr = stats.pearsonr(valid["DaysUntilOvulation"], valid["Religiosity"])
print(f"Pearson correlation (DaysUntilOvulation vs Religiosity): r={r:.4f}, p={p_corr:.4f}")

# OLS regression controlling for relationship status
reg_df = df[["Religiosity", "DaysUntilOvulation", "Relationship", "ReportedCycleLength"]].dropna()
X = sm.add_constant(reg_df[["DaysUntilOvulation", "Relationship", "ReportedCycleLength"]])
y = reg_df["Religiosity"]
model = sm.OLS(y, X).fit()
print("\nOLS regression summary:")
print(model.summary())

# Also test each religiosity item separately
for item in ["Rel1", "Rel2", "Rel3"]:
    near_item = df[df["NearOvulation"] == True][item].dropna()
    far_item = df[df["NearOvulation"] == False][item].dropna()
    t, p = stats.ttest_ind(near_item, far_item)
    print(f"T-test {item}: t={t:.4f}, p={p:.4f}, mean_near={near_item.mean():.3f}, mean_far={far_item.mean():.3f}")

# Determine conclusion
# Collect evidence
p_values = [p_val, p_corr]
ols_p = model.pvalues.get("DaysUntilOvulation", 1.0)
p_values.append(ols_p)

min_p = min(p_values)
print(f"\nMin p-value across tests: {min_p:.4f}")
print(f"OLS p-value for DaysUntilOvulation: {ols_p:.4f}")

# Determine response score (0-100)
# Strong effect = p < 0.01 -> high score
# Marginal = 0.01-0.05 -> moderate
# Not significant = p > 0.05 -> low score
if min_p < 0.01:
    response = 75
    explanation = (
        f"Statistical analysis shows a significant relationship between fertility-related hormonal "
        f"fluctuations (proximity to ovulation) and religiosity. T-test p={p_val:.4f}, "
        f"Pearson r={r:.4f} (p={p_corr:.4f}), OLS regression p={ols_p:.4f} for ovulation timing. "
        f"Mean religiosity near ovulation: {near.mean():.3f}, far: {far.mean():.3f}."
    )
elif min_p < 0.05:
    response = 55
    explanation = (
        f"Marginally significant relationship detected between fertility hormonal fluctuations "
        f"(ovulation proximity) and religiosity. T-test p={p_val:.4f}, Pearson r={r:.4f} "
        f"(p={p_corr:.4f}), OLS p={ols_p:.4f}. Effect is modest and borderline significant."
    )
else:
    response = 20
    explanation = (
        f"No significant relationship found between fertility-related hormonal fluctuations "
        f"(ovulation proximity) and religiosity. T-test p={p_val:.4f}, Pearson r={r:.4f} "
        f"(p={p_corr:.4f}), OLS regression p={ols_p:.4f} for ovulation timing. "
        f"Mean religiosity near ovulation: {near.mean():.3f}, far: {far.mean():.3f}. "
        f"None of the tests reached conventional significance thresholds."
    )

result = {"response": response, "explanation": explanation}
with open("conclusion.txt", "w") as f:
    json.dump(result, f)

print(f"\nConclusion written: response={response}")
print(f"Explanation: {explanation}")
