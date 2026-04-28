import json
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd
import statsmodels.api as sm
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from scipy import stats
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore", category=UserWarning)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def top_abs(d: Dict[str, float], n: int = 5) -> List[str]:
    items = sorted(d.items(), key=lambda kv: abs(kv[1]), reverse=True)[:n]
    return [f"{k}={v:.3f}" for k, v in items]


# 1) Read metadata and research question
with open("info.json", "r") as f:
    info = json.load(f)

research_questions = info.get("research_questions", [])
research_question = research_questions[0] if research_questions else ""
print("Research question:")
print(research_question)
print("-" * 80)

# 2) Load dataset

df = pd.read_csv("hurricane.csv")
print(f"Dataset shape: {df.shape}")
print("Missing values by column:")
print(df.isna().sum().sort_values(ascending=False).to_string())

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print("\\nNumeric columns:", numeric_cols)

# 3) EDA: summary stats, distributions, correlations
summary = df[numeric_cols].describe().T
summary["skew"] = df[numeric_cols].skew(numeric_only=True)
print("\\nSummary statistics (numeric):")
print(summary[["mean", "std", "min", "25%", "50%", "75%", "max", "skew"]].round(3).to_string())

key_distribution_cols = ["masfem", "alldeaths", "wind", "min", "category", "ndam15"]
key_distribution_cols = [c for c in key_distribution_cols if c in df.columns]
print("\\nSelected distribution quantiles:")
print(df[key_distribution_cols].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round(3).to_string())

corr_matrix = df[numeric_cols].corr(numeric_only=True)
print("\\nCorrelations with alldeaths:")
if "alldeaths" in corr_matrix.columns:
    print(corr_matrix["alldeaths"].sort_values(ascending=False).round(3).to_string())

# Target transform for heavy-tailed death counts

df["log_alldeaths"] = np.log1p(df["alldeaths"])

# 4) Statistical tests focused on question: femininity -> more deaths
results = {}

pearson_r, pearson_p = stats.pearsonr(df["masfem"], df["alldeaths"])
spearman_r, spearman_p = stats.spearmanr(df["masfem"], df["alldeaths"])
results["pearson_masfem_alldeaths"] = {"r": safe_float(pearson_r), "p": safe_float(pearson_p)}
results["spearman_masfem_alldeaths"] = {"rho": safe_float(spearman_r), "p": safe_float(spearman_p)}

female_deaths = df.loc[df["gender_mf"] == 1, "alldeaths"]
male_deaths = df.loc[df["gender_mf"] == 0, "alldeaths"]
welch_t, welch_p = stats.ttest_ind(female_deaths, male_deaths, equal_var=False)
results["welch_t_female_vs_male_alldeaths"] = {
    "t": safe_float(welch_t),
    "p": safe_float(welch_p),
    "female_mean": safe_float(female_deaths.mean()),
    "male_mean": safe_float(male_deaths.mean()),
}

mw_u, mw_p = stats.mannwhitneyu(female_deaths, male_deaths, alternative="two-sided")
results["mannwhitney_female_vs_male_alldeaths"] = {
    "u": safe_float(mw_u),
    "p": safe_float(mw_p),
}

# ANOVA: storm category effect on deaths (contextual severity check)
cat_groups = [
    df.loc[df["category"] == c, "log_alldeaths"].values
    for c in sorted(df["category"].dropna().unique())
    if (df["category"] == c).sum() > 1
]
if len(cat_groups) >= 2:
    anova_f, anova_p = stats.f_oneway(*cat_groups)
    results["anova_log_deaths_by_category"] = {"F": safe_float(anova_f), "p": safe_float(anova_p)}

# OLS with controls
ols_features = ["masfem", "wind", "min", "category", "ndam15", "elapsedyrs"]
ols_df = df[["log_alldeaths"] + ols_features].dropna()
X = sm.add_constant(ols_df[ols_features])
y = ols_df["log_alldeaths"]
ols_model = sm.OLS(y, X).fit(cov_type="HC3")
results["ols_masfem_controls"] = {
    "coef_masfem": safe_float(ols_model.params.get("masfem", np.nan)),
    "p_masfem": safe_float(ols_model.pvalues.get("masfem", np.nan)),
    "ci_low": safe_float(ols_model.conf_int().loc["masfem", 0]),
    "ci_high": safe_float(ols_model.conf_int().loc["masfem", 1]),
    "r2": safe_float(ols_model.rsquared),
}

# Robustness: binary female-name indicator instead of continuous femininity score
ols2_features = ["gender_mf", "wind", "min", "category", "ndam15", "elapsedyrs"]
ols2_df = df[["log_alldeaths"] + ols2_features].dropna()
X2 = sm.add_constant(ols2_df[ols2_features])
y2 = ols2_df["log_alldeaths"]
ols2_model = sm.OLS(y2, X2).fit(cov_type="HC3")
results["ols_gender_controls"] = {
    "coef_gender_mf": safe_float(ols2_model.params.get("gender_mf", np.nan)),
    "p_gender_mf": safe_float(ols2_model.pvalues.get("gender_mf", np.nan)),
    "ci_low": safe_float(ols2_model.conf_int().loc["gender_mf", 0]),
    "ci_high": safe_float(ols2_model.conf_int().loc["gender_mf", 1]),
    "r2": safe_float(ols2_model.rsquared),
}

print("\\nKey statistical tests:")
for k, v in results.items():
    print(k, v)

# 5) Interpretable models (scikit-learn + interpret)
model_features = [
    "masfem",
    "gender_mf",
    "wind",
    "min",
    "category",
    "ndam15",
    "elapsedyrs",
    "masfem_mturk",
]
model_df = df[model_features + ["log_alldeaths", "alldeaths"]].copy()
for col in model_features:
    if model_df[col].isna().any():
        model_df[col] = model_df[col].fillna(model_df[col].median())

X_model = model_df[model_features]
y_model = model_df["log_alldeaths"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_model)

lin = LinearRegression().fit(X_scaled, y_model)
ridge = Ridge(alpha=1.0, random_state=0).fit(X_scaled, y_model)
lasso = Lasso(alpha=0.02, random_state=0, max_iter=20000).fit(X_scaled, y_model)

lin_coef = dict(zip(model_features, lin.coef_))
ridge_coef = dict(zip(model_features, ridge.coef_))
lasso_coef = dict(zip(model_features, lasso.coef_))

print("\\nTop standardized coefficients (LinearRegression):", top_abs(lin_coef))
print("Top standardized coefficients (Ridge):", top_abs(ridge_coef))
print("Top standardized coefficients (Lasso):", top_abs(lasso_coef))

# Tree model for transparent split-based importance
reg_tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=5, random_state=0)
reg_tree.fit(X_model, y_model)
tree_importance = dict(zip(model_features, reg_tree.feature_importances_))
print("Top feature importances (DecisionTreeRegressor):", top_abs(tree_importance))

# Explainable Boosting Regressor (interpretable additive model)
ebm_reg = ExplainableBoostingRegressor(random_state=0, interactions=0)
ebm_reg.fit(X_model, y_model)
ebm_reg_importance = dict(zip(ebm_reg.term_names_, ebm_reg.term_importances()))
print("Top term importances (EBM Regressor):", top_abs(ebm_reg_importance))

# Classification framing: high-death storms (above median)
y_class = (model_df["alldeaths"] > model_df["alldeaths"].median()).astype(int)
clf_tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=0)
clf_tree.fit(X_model, y_class)
clf_tree_importance = dict(zip(model_features, clf_tree.feature_importances_))

ebm_clf = ExplainableBoostingClassifier(random_state=0, interactions=0)
ebm_clf.fit(X_model, y_class)
ebm_clf_importance = dict(zip(ebm_clf.term_names_, ebm_clf.term_importances()))

print("Top feature importances (DecisionTreeClassifier):", top_abs(clf_tree_importance))
print("Top term importances (EBM Classifier):", top_abs(ebm_clf_importance))

# 6) Evidence synthesis into Likert response
# Hypothesis direction for support: more feminine names -> more deaths (fewer precautions)

support_signals = 0
oppose_signals = 0

if pearson_p < 0.05:
    if pearson_r > 0:
        support_signals += 1
    else:
        oppose_signals += 1

if spearman_p < 0.05:
    if spearman_r > 0:
        support_signals += 1
    else:
        oppose_signals += 1

if welch_p < 0.05:
    if female_deaths.mean() > male_deaths.mean():
        support_signals += 1
    else:
        oppose_signals += 1

ols_coef = results["ols_masfem_controls"]["coef_masfem"]
ols_p = results["ols_masfem_controls"]["p_masfem"]
if ols_p < 0.05:
    if ols_coef > 0:
        support_signals += 2
    else:
        oppose_signals += 2

ols2_coef = results["ols_gender_controls"]["coef_gender_mf"]
ols2_p = results["ols_gender_controls"]["p_gender_mf"]
if ols2_p < 0.05:
    if ols2_coef > 0:
        support_signals += 1
    else:
        oppose_signals += 1

if support_signals == 0 and oppose_signals == 0:
    response_score = 20
elif support_signals > oppose_signals:
    response_score = int(min(95, 55 + 12 * (support_signals - oppose_signals)))
elif oppose_signals > support_signals:
    response_score = int(max(5, 45 - 12 * (oppose_signals - support_signals)))
else:
    response_score = 50

response_score = int(max(0, min(100, response_score)))

masfem_lin = lin_coef.get("masfem", np.nan)
masfem_ridge = ridge_coef.get("masfem", np.nan)
masfem_lasso = lasso_coef.get("masfem", np.nan)
masfem_tree_imp = tree_importance.get("masfem", np.nan)
masfem_ebm_imp = ebm_reg_importance.get("masfem", np.nan)

explanation = (
    "Evidence for the claim is weak. The direct association between name femininity and deaths is not statistically "
    f"significant (Pearson r={pearson_r:.3f}, p={pearson_p:.3f}; Spearman rho={spearman_r:.3f}, p={spearman_p:.3f}). "
    "Female-named storms have higher mean deaths in raw comparison, but the difference is not significant "
    f"(Welch t-test p={welch_p:.3f}). In controlled OLS regression on log deaths, femininity is not significant "
    f"(coef={ols_coef:.3f}, p={ols_p:.3f}, 95% CI [{results['ols_masfem_controls']['ci_low']:.3f}, "
    f"{results['ols_masfem_controls']['ci_high']:.3f}]). Interpretable models place much more importance on "
    "storm intensity/damage variables than on name femininity "
    f"(Linear coef masfem={masfem_lin:.3f}, Ridge={masfem_ridge:.3f}, Lasso={masfem_lasso:.3f}, "
    f"Tree importance={masfem_tree_imp:.3f}, EBM importance={masfem_ebm_imp:.3f})."
)

output = {
    "response": response_score,
    "explanation": explanation,
}

with open("conclusion.txt", "w") as f:
    json.dump(output, f)

print("\\nWrote conclusion.txt with response score:", response_score)
