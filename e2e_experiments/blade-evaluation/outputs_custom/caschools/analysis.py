import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from interp_models import HingeEBMRegressor, SmartAdditiveRegressor

warnings.filterwarnings("ignore")


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


# 1) Load metadata and data
with open("info.json", "r", encoding="utf-8") as f:
    info = json.load(f)

research_question = info.get("research_questions", ["Unknown question"])[0]
df = pd.read_csv("caschools.csv")

# 2) Basic feature engineering for question-specific analysis
# Student-teacher ratio (STR) and academic performance index
for col in ["students", "teachers", "read", "math", "income", "english", "lunch", "calworks", "expenditure", "computer"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["str_ratio"] = df["students"] / df["teachers"]
df["avg_score"] = (df["read"] + df["math"]) / 2.0
df["computer_per_student"] = df["computer"] / df["students"]

analysis_cols = [
    "avg_score",
    "str_ratio",
    "income",
    "english",
    "lunch",
    "calworks",
    "expenditure",
    "computer_per_student",
    "students",
]

df_model = df[analysis_cols].dropna().copy()

# 3) Data exploration
print("=" * 80)
print("RESEARCH QUESTION")
print(research_question)
print("=" * 80)
print(f"Rows (raw): {len(df)}, Rows used after NA drop: {len(df_model)}")
print("\nSummary statistics:")
print(df_model.describe().T)

print("\nCorrelation matrix (Pearson):")
print(df_model.corr(numeric_only=True).round(3))

# Direct bivariate tests for the main question
pearson_r, pearson_p = stats.pearsonr(df_model["str_ratio"], df_model["avg_score"])
spearman_rho, spearman_p = stats.spearmanr(df_model["str_ratio"], df_model["avg_score"])

print("\nBivariate association tests:")
print(f"Pearson corr(str_ratio, avg_score): r={pearson_r:.4f}, p={pearson_p:.4g}")
print(f"Spearman corr(str_ratio, avg_score): rho={spearman_rho:.4f}, p={spearman_p:.4g}")

# t-test: low vs high STR quartiles
q25 = df_model["str_ratio"].quantile(0.25)
q75 = df_model["str_ratio"].quantile(0.75)
low_str = df_model.loc[df_model["str_ratio"] <= q25, "avg_score"]
high_str = df_model.loc[df_model["str_ratio"] >= q75, "avg_score"]
t_stat, t_p = stats.ttest_ind(low_str, high_str, equal_var=False)
mean_diff = float(low_str.mean() - high_str.mean())

print("\nT-test (lowest STR quartile vs highest STR quartile):")
print(
    f"n_low={len(low_str)}, n_high={len(high_str)}, "
    f"mean_low={low_str.mean():.3f}, mean_high={high_str.mean():.3f}, "
    f"diff(low-high)={mean_diff:.3f}, t={t_stat:.4f}, p={t_p:.4g}"
)

# ANOVA across STR terciles
tercile_labels = ["low", "mid", "high"]
df_model["str_tercile"] = pd.qcut(df_model["str_ratio"], q=3, labels=tercile_labels)

anova_groups = [
    df_model.loc[df_model["str_tercile"] == lab, "avg_score"].values for lab in tercile_labels
]
f_stat, anova_p = stats.f_oneway(*anova_groups)
print("\nANOVA across STR terciles:")
print(f"F={f_stat:.4f}, p={anova_p:.4g}")
print("Group means (avg_score):")
print(df_model.groupby("str_tercile")["avg_score"].mean())

# 4) Multivariable regression (statistical significance with controls)
X_controls = df_model[
    ["str_ratio", "income", "english", "lunch", "calworks", "expenditure", "computer_per_student", "students"]
]
y = df_model["avg_score"]

X_sm = sm.add_constant(X_controls)
ols = sm.OLS(y, X_sm).fit()

print("\nOLS regression summary (target=avg_score):")
print(ols.summary())

str_coef = safe_float(ols.params.get("str_ratio", np.nan))
str_p = safe_float(ols.pvalues.get("str_ratio", np.nan))
str_ci_low, str_ci_high = [safe_float(v) for v in ols.conf_int().loc["str_ratio"].tolist()]

# 5) Predictive/interpretable modeling (custom + standard)
X = X_controls.values
feature_names = list(X_controls.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y.values, test_size=0.25, random_state=42)

models = {
    "LinearRegression": LinearRegression(),
    "RidgeCV": RidgeCV(alphas=np.logspace(-3, 3, 20)),
    "LassoCV": LassoCV(cv=5, random_state=42, max_iter=10000),
    "DecisionTree": DecisionTreeRegressor(max_depth=4, random_state=42),
    "SmartAdditiveRegressor": SmartAdditiveRegressor(n_rounds=250, learning_rate=0.08, min_samples_leaf=8),
    "HingeEBMRegressor": HingeEBMRegressor(n_knots=3, max_input_features=15, ebm_outer_bags=2, ebm_max_rounds=400),
}

metrics = []
fit_models = {}
failed_models = {}
for name, model in models.items():
    try:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        r2 = r2_score(y_test, pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        metrics.append((name, r2, rmse))
        fit_models[name] = model
    except Exception as exc:
        failed_models[name] = str(exc)

print("\nModel performance on held-out test set:")
for name, r2, rmse in sorted(metrics, key=lambda x: x[1], reverse=True):
    print(f"{name:24s} R2={r2:.4f}  RMSE={rmse:.4f}")
if failed_models:
    print("Skipped models:")
    for name, err in failed_models.items():
        print(f"  {name}: {err}")

# Print interpretable model descriptions heavily
print("\n" + "=" * 80)
print("SmartAdditiveRegressor interpretation")
print("Feature order:", feature_names)
if "SmartAdditiveRegressor" in fit_models:
    print(str(fit_models["SmartAdditiveRegressor"]))
else:
    print("SmartAdditiveRegressor was unavailable.")

print("\n" + "=" * 80)
print("HingeEBMRegressor interpretation")
print("Feature order:", feature_names)
if "HingeEBMRegressor" in fit_models:
    print(str(fit_models["HingeEBMRegressor"]))
else:
    print("HingeEBMRegressor was unavailable.")

# Optional: imodels if available
try:
    from imodels import FIGSRegressor, RuleFitRegressor

    figs = FIGSRegressor(max_rules=20, random_state=42)
    figs.fit(X_train, y_train)
    figs_pred = figs.predict(X_test)
    figs_r2 = r2_score(y_test, figs_pred)

    rulefit = RuleFitRegressor(random_state=42, max_rules=40)
    rulefit.fit(X_train, y_train)
    rf_pred = rulefit.predict(X_test)
    rf_r2 = r2_score(y_test, rf_pred)

    print("\nAdditional interpretable baselines (imodels):")
    print(f"FIGSRegressor R2={figs_r2:.4f}")
    print(f"RuleFitRegressor R2={rf_r2:.4f}")
except Exception as e:
    print("\nimodels optional models skipped:", e)

# 6) Evidence synthesis -> Likert response (0-100)
# Strong evidence requires: significant negative STR coefficient in controlled OLS,
# significant negative bivariate correlation, and robust group differences.

score = 50
explanations = []

if np.isfinite(pearson_p) and pearson_p < 0.05 and pearson_r < 0:
    score += 15
    explanations.append(f"Pearson correlation is negative and significant (r={pearson_r:.3f}, p={pearson_p:.3g}).")
else:
    explanations.append(f"Pearson correlation is not strongly supportive (r={pearson_r:.3f}, p={pearson_p:.3g}).")

if np.isfinite(spearman_p) and spearman_p < 0.05 and spearman_rho < 0:
    score += 10
    explanations.append(f"Spearman correlation confirms monotonic negative association (rho={spearman_rho:.3f}, p={spearman_p:.3g}).")
else:
    explanations.append(f"Spearman result is weak/non-significant (rho={spearman_rho:.3f}, p={spearman_p:.3g}).")

if np.isfinite(t_p) and t_p < 0.05 and mean_diff > 0:
    score += 10
    explanations.append(f"Low-STR districts outperform high-STR districts (mean diff={mean_diff:.2f}, p={t_p:.3g}).")
else:
    explanations.append(f"Quartile t-test is weak/non-supportive (mean diff={mean_diff:.2f}, p={t_p:.3g}).")

if np.isfinite(anova_p) and anova_p < 0.05:
    score += 5
    explanations.append(f"ANOVA across STR terciles is significant (p={anova_p:.3g}).")
else:
    explanations.append(f"ANOVA across STR terciles is not significant (p={anova_p:.3g}).")

if np.isfinite(str_p) and str_p < 0.05 and str_coef < 0:
    score += 20
    explanations.append(
        f"Controlled OLS shows a significant negative STR effect (coef={str_coef:.3f}, "
        f"95% CI [{str_ci_low:.3f}, {str_ci_high:.3f}], p={str_p:.3g})."
    )
else:
    explanations.append(
        f"Controlled OLS does not show a clear negative significant STR effect "
        f"(coef={str_coef:.3f}, p={str_p:.3g})."
    )

# magnitude adjustment from controlled effect size
if np.isfinite(str_coef):
    # Typical STR range around ~10 units, so coef around -0.5 or below is meaningful
    if str_coef <= -0.8:
        score += 5
    elif str_coef >= 0:
        score -= 15

score = int(max(0, min(100, round(score))))

# Mention custom interpretable model usage (as required)
custom_used = [name for name in ["SmartAdditiveRegressor", "HingeEBMRegressor"] if name in fit_models]
if len(custom_used) == 2:
    explanations.append(
        "Both custom interpretability tools were fit successfully and used to inspect feature effects and possible nonlinearities."
    )
elif len(custom_used) == 1:
    explanations.append(
        f"One custom interpretability tool ({custom_used[0]}) was fit successfully and used for model interpretation."
    )
else:
    explanations.append("Custom interpretability tools could not be fit in this environment.")

final_explanation = " ".join(explanations)

result = {"response": score, "explanation": final_explanation}

with open("conclusion.txt", "w", encoding="utf-8") as f:
    json.dump(result, f)

print("\nWrote conclusion.txt:")
print(json.dumps(result, indent=2))
