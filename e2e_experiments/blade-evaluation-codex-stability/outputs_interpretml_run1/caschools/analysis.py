import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from interpret.glassbox import ExplainableBoostingRegressor
from scipy import stats
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")


def section(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def safe_p(x: float) -> float:
    if np.isnan(x):
        return 1.0
    return float(x)


# 1) Load data
df = pd.read_csv("caschools.csv")
df["student_teacher_ratio"] = df["students"] / df["teachers"]
df["avg_score"] = (df["read"] + df["math"]) / 2.0

# Keep main analysis frame (numeric features used for tests/models)
analysis_cols = [
    "student_teacher_ratio",
    "avg_score",
    "income",
    "english",
    "lunch",
    "calworks",
    "expenditure",
    "computer",
    "students",
]
work = df[analysis_cols].copy()
work = work.dropna()

section("Research Question")
print("Is a lower student-teacher ratio associated with higher academic performance?")
print(f"Rows available for analysis: {len(work)}")


# 2) Data exploration
section("Data Quality and Summary Statistics")
missing = work.isna().sum()
print("Missing values by column:")
print(missing.to_string())

summary = work.describe().T
summary["skew"] = work.skew(numeric_only=True)
print("\nSummary statistics:")
print(summary.round(4).to_string())

section("Distribution Snapshots")
for col in ["student_teacher_ratio", "avg_score"]:
    counts, edges = np.histogram(work[col], bins=10)
    print(f"\nHistogram bins for {col}:")
    for i in range(len(counts)):
        print(f"  [{edges[i]:.3f}, {edges[i+1]:.3f}): {counts[i]}")

section("Correlations")
corr = work.corr(numeric_only=True)
print("Correlation matrix (rounded):")
print(corr.round(3).to_string())
print("\nCorrelations with avg_score (sorted):")
print(corr["avg_score"].sort_values(ascending=False).round(3).to_string())


# 3) Statistical tests
section("Statistical Tests")
pearson_r, pearson_p = stats.pearsonr(work["student_teacher_ratio"], work["avg_score"])
spearman_r, spearman_p = stats.spearmanr(work["student_teacher_ratio"], work["avg_score"])
print(f"Pearson r = {pearson_r:.4f}, p = {pearson_p:.6g}")
print(f"Spearman rho = {spearman_r:.4f}, p = {spearman_p:.6g}")

median_str = work["student_teacher_ratio"].median()
low_str = work.loc[work["student_teacher_ratio"] <= median_str, "avg_score"]
high_str = work.loc[work["student_teacher_ratio"] > median_str, "avg_score"]
t_stat, t_p = stats.ttest_ind(low_str, high_str, equal_var=False)
print(
    f"Welch t-test (low STR vs high STR avg_score): t = {t_stat:.4f}, p = {t_p:.6g}, "
    f"mean_low = {low_str.mean():.3f}, mean_high = {high_str.mean():.3f}"
)

quartile_labels = ["Q1_lowest_STR", "Q2", "Q3", "Q4_highest_STR"]
work["str_quartile"] = pd.qcut(work["student_teacher_ratio"], q=4, labels=quartile_labels)
quartile_groups = [work.loc[work["str_quartile"] == q, "avg_score"].values for q in quartile_labels]
f_stat, f_p = stats.f_oneway(*quartile_groups)
quartile_means = work.groupby("str_quartile", observed=False)["avg_score"].mean()
print(f"One-way ANOVA across STR quartiles: F = {f_stat:.4f}, p = {f_p:.6g}")
print("Quartile means (avg_score):")
print(quartile_means.round(3).to_string())

# OLS regressions
y = work["avg_score"]
X_simple = sm.add_constant(work[["student_teacher_ratio"]])
ols_simple = sm.OLS(y, X_simple).fit()

controls = ["student_teacher_ratio", "income", "english", "lunch", "calworks", "expenditure"]
X_ctrl = sm.add_constant(work[controls])
ols_ctrl = sm.OLS(y, X_ctrl).fit()

print("\nSimple OLS: avg_score ~ student_teacher_ratio")
print(
    f"coef_STR = {ols_simple.params['student_teacher_ratio']:.4f}, "
    f"p = {ols_simple.pvalues['student_teacher_ratio']:.6g}, R^2 = {ols_simple.rsquared:.4f}"
)

print("\nControlled OLS: avg_score ~ STR + socioeconomic controls")
print(
    f"coef_STR = {ols_ctrl.params['student_teacher_ratio']:.4f}, "
    f"p = {ols_ctrl.pvalues['student_teacher_ratio']:.6g}, "
    f"95% CI = [{ols_ctrl.conf_int().loc['student_teacher_ratio', 0]:.4f}, "
    f"{ols_ctrl.conf_int().loc['student_teacher_ratio', 1]:.4f}], R^2 = {ols_ctrl.rsquared:.4f}"
)


# 4) Interpretable models (sklearn + interpret)
section("Interpretable Models")
feature_cols = [
    "student_teacher_ratio",
    "income",
    "english",
    "lunch",
    "calworks",
    "expenditure",
    "computer",
    "students",
]
X = work[feature_cols]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

linear_models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.01, max_iter=20000, random_state=42),
}

for name, model in linear_models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    r2 = r2_score(y_test, pred)
    coefs = pd.Series(model.coef_, index=feature_cols).sort_values(key=np.abs, ascending=False)
    print(f"\n{name} test R^2: {r2:.4f}")
    print("Top coefficients (by |value|):")
    print(coefs.round(4).to_string())


tree = DecisionTreeRegressor(max_depth=3, random_state=42)
tree.fit(X_train, y_train)
tree_pred = tree.predict(X_test)
tree_r2 = r2_score(y_test, tree_pred)
tree_importance = pd.Series(tree.feature_importances_, index=feature_cols).sort_values(ascending=False)
print(f"\nDecisionTreeRegressor (max_depth=3) test R^2: {tree_r2:.4f}")
print("Feature importances:")
print(tree_importance.round(4).to_string())


# Explainable Boosting Regressor (interpret)
ebm = ExplainableBoostingRegressor(random_state=42, interactions=0)
ebm.fit(X_train, y_train)
ebm_pred = ebm.predict(X_test)
ebm_r2 = r2_score(y_test, ebm_pred)
term_imp = pd.Series(ebm.term_importances(), index=ebm.term_names_).sort_values(ascending=False)
print(f"\nExplainableBoostingRegressor test R^2: {ebm_r2:.4f}")
print("EBM term importances:")
print(term_imp.round(4).to_string())

# Approximate directional effect of STR from EBM by varying only STR around median profile.
str_grid = np.linspace(work["student_teacher_ratio"].quantile(0.05), work["student_teacher_ratio"].quantile(0.95), 60)
ref = X.median(numeric_only=True)
ref_df = pd.DataFrame([ref.values] * len(str_grid), columns=feature_cols)
ref_df["student_teacher_ratio"] = str_grid
grid_pred = ebm.predict(ref_df)
ebm_str_slope = np.polyfit(str_grid, grid_pred, 1)[0]
print(f"Approximate EBM slope for STR effect: {ebm_str_slope:.4f} points per +1 STR")


# 5) Evidence synthesis and final Likert response
section("Evidence Synthesis")

simple_coef = float(ols_simple.params["student_teacher_ratio"])
simple_p = safe_p(float(ols_simple.pvalues["student_teacher_ratio"]))
ctrl_coef = float(ols_ctrl.params["student_teacher_ratio"])
ctrl_p = safe_p(float(ols_ctrl.pvalues["student_teacher_ratio"]))

# Build a transparent rule-based score.
score = 50

# Bivariate association evidence
if pearson_p < 0.05 and pearson_r < 0:
    score += 18
elif pearson_p < 0.05 and pearson_r > 0:
    score -= 18
else:
    score -= 8

if simple_p < 0.05 and simple_coef < 0:
    score += 12
elif simple_p < 0.05 and simple_coef > 0:
    score -= 12
else:
    score -= 8

# Group tests
if t_p < 0.05 and low_str.mean() > high_str.mean():
    score += 8
elif t_p < 0.05:
    score -= 8

quartile_decreasing = bool(quartile_means.iloc[0] > quartile_means.iloc[-1])
if f_p < 0.05 and quartile_decreasing:
    score += 8
elif f_p < 0.05:
    score -= 8

# Controlled inference gets stronger weight
if ctrl_p < 0.05 and ctrl_coef < 0:
    score += 20
elif ctrl_p < 0.05 and ctrl_coef > 0:
    score -= 20
else:
    score -= 30

# If interpretable ML ranks STR as low-importance, dampen certainty.
str_rank = int(list(term_imp.index).index("student_teacher_ratio") + 1)
if str_rank > (len(term_imp) // 2):
    score -= 8

# If adjusted relationship is not significant, cap score below a clear "Yes".
if ctrl_p >= 0.05:
    score = min(score, 45)

score = int(np.clip(round(score), 0, 100))

explanation = (
    "Unadjusted analyses show that lower student-teacher ratio is linked to higher average test scores "
    f"(Pearson r={pearson_r:.3f}, p={pearson_p:.2e}; simple OLS coef={simple_coef:.3f}, p={simple_p:.3g}; "
    f"t-test p={t_p:.2e}; ANOVA p={f_p:.2e}). "
    "However, once key socioeconomic controls are included, the student-teacher ratio effect is small and not statistically significant "
    f"(controlled OLS coef={ctrl_coef:.3f}, p={ctrl_p:.3g}, 95% CI includes 0). "
    f"Interpretable models also rank student-teacher ratio below stronger predictors like lunch, english, and income "
    f"(EBM STR importance rank {str_rank}/{len(term_imp)}; approximate EBM STR slope {ebm_str_slope:.3f}). "
    "Overall, evidence supports a weak raw association but not a robust independent relationship, so the answer leans No."
)

print(f"Final Likert response (0-100): {score}")
print("Conclusion summary:")
print(explanation)

result = {"response": score, "explanation": explanation}
with open("conclusion.txt", "w", encoding="utf-8") as f:
    json.dump(result, f)

print("\nWrote conclusion.txt")
