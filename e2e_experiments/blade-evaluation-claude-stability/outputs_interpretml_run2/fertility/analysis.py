import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import Ridge
from interpret.glassbox import ExplainableBoostingRegressor

# Load data
df = pd.read_csv("fertility.csv")

# Parse dates
for col in ["DateTesting", "StartDateofLastPeriod", "StartDateofPeriodBeforeLast"]:
    df[col] = pd.to_datetime(df[col], format="%m/%d/%y", errors="coerce")

# Compute days since last period started (cycle day)
df["DaysSinceLastPeriod"] = (df["DateTesting"] - df["StartDateofLastPeriod"]).dt.days

# Compute actual cycle length from the two period start dates
df["ActualCycleLength"] = (df["StartDateofLastPeriod"] - df["StartDateofPeriodBeforeLast"]).dt.days

# Use reported cycle length if actual is missing/implausible
cycle_len = df["ActualCycleLength"].where(
    df["ActualCycleLength"].between(20, 45), other=df["ReportedCycleLength"]
)
df["CycleLength"] = cycle_len

# Estimate days until next ovulation: ovulation ~ cycle_length - 14
# Fertility window: ~6 days ending on ovulation day
# Days until ovulation from testing date
df["DaysUntilOvulation"] = df["CycleLength"] - 14 - df["DaysSinceLastPeriod"]

# Fertility score: higher = closer to ovulation
# Standard approach: fertile if DaysUntilOvulation in [-1, 5] (Gangestad & Thornhill)
df["InFertileWindow"] = df["DaysUntilOvulation"].between(-1, 5).astype(int)

# Continuous fertility: proximity to ovulation (inverted distance)
df["FertilityProximity"] = -df["DaysUntilOvulation"].abs()

# Religiosity composite
df["Religiosity"] = df[["Rel1", "Rel2", "Rel3"]].mean(axis=1)

# Drop rows missing key variables
df_clean = df.dropna(subset=["DaysSinceLastPeriod", "CycleLength", "Religiosity"]).copy()
df_clean = df_clean[df_clean["DaysSinceLastPeriod"] >= 0]
df_clean = df_clean[df_clean["CycleLength"].between(20, 45)]

print(f"Clean sample size: {len(df_clean)}")
print(f"\nReligiosity stats:\n{df_clean['Religiosity'].describe()}")
print(f"\nFertile window N={df_clean['InFertileWindow'].sum()}, Non-fertile N={(df_clean['InFertileWindow']==0).sum()}")

# --- Statistical tests ---

# 1. t-test: fertile vs non-fertile religiosity
fertile = df_clean[df_clean["InFertileWindow"] == 1]["Religiosity"]
non_fertile = df_clean[df_clean["InFertileWindow"] == 0]["Religiosity"]
t_stat, p_val = stats.ttest_ind(fertile, non_fertile)
print(f"\nT-test (fertile vs non-fertile):")
print(f"  Fertile mean={fertile.mean():.3f}, Non-fertile mean={non_fertile.mean():.3f}")
print(f"  t={t_stat:.3f}, p={p_val:.4f}")

# 2. Correlation between fertility proximity and religiosity
corr, p_corr = stats.pearsonr(
    df_clean["FertilityProximity"].dropna(),
    df_clean.loc[df_clean["FertilityProximity"].notna(), "Religiosity"]
)
print(f"\nPearson correlation (fertility proximity vs religiosity): r={corr:.3f}, p={p_corr:.4f}")

# 3. OLS regression controlling for relationship status
X = df_clean[["InFertileWindow", "Relationship"]].dropna()
y = df_clean.loc[X.index, "Religiosity"]
X_sm = sm.add_constant(X)
ols = sm.OLS(y, X_sm).fit()
print(f"\nOLS regression:\n{ols.summary()}")

# 4. Ridge regression with more features
features = ["InFertileWindow", "FertilityProximity", "Relationship", "CycleLength"]
X_ridge = df_clean[features].dropna()
y_ridge = df_clean.loc[X_ridge.index, "Religiosity"]
ridge = Ridge(alpha=1.0)
ridge.fit(X_ridge, y_ridge)
print(f"\nRidge coefficients:")
for feat, coef in zip(features, ridge.coef_):
    print(f"  {feat}: {coef:.4f}")

# 5. EBM for nonlinear effects
ebm = ExplainableBoostingRegressor(random_state=42)
ebm.fit(X_ridge, y_ridge)
print(f"\nEBM feature importances:")
for feat, imp in zip(features, ebm.term_importances()):
    print(f"  {feat}: {imp:.4f}")

# Determine conclusion
# Primary test: t-test p-value for fertile vs non-fertile
# Secondary: OLS p-value for InFertileWindow
fertile_coef_pval = ols.pvalues.get("InFertileWindow", 1.0)
print(f"\nOLS p-value for InFertileWindow: {fertile_coef_pval:.4f}")

# Score based on evidence strength
if p_val < 0.05 or fertile_coef_pval < 0.05:
    response = 70
    explanation = (
        f"Significant effect found: fertile-window women show different religiosity "
        f"(t={t_stat:.3f}, p={p_val:.4f}; OLS p={fertile_coef_pval:.4f}). "
        f"Fertile mean={fertile.mean():.3f} vs non-fertile mean={non_fertile.mean():.3f}. "
        f"Fertility fluctuations associated with the menstrual cycle appear to have a statistically "
        f"significant effect on religiosity."
    )
elif p_val < 0.10 or fertile_coef_pval < 0.10:
    response = 45
    explanation = (
        f"Marginal trend: fertile-window women t={t_stat:.3f}, p={p_val:.4f}; "
        f"OLS p={fertile_coef_pval:.4f}. Effect is suggestive but not significant at alpha=0.05. "
        f"Weak evidence that hormonal fluctuations affect religiosity."
    )
else:
    response = 20
    explanation = (
        f"No significant effect: t={t_stat:.3f}, p={p_val:.4f}; "
        f"OLS p-value for InFertileWindow={fertile_coef_pval:.4f}. "
        f"Fertile mean={fertile.mean():.3f} vs non-fertile mean={non_fertile.mean():.3f}. "
        f"Hormonal fluctuations associated with fertility do not appear to significantly affect "
        f"women's religiosity in this dataset."
    )

conclusion = {"response": response, "explanation": explanation}
with open("conclusion.txt", "w") as f:
    json.dump(conclusion, f)

print(f"\nConclusion: {conclusion}")
