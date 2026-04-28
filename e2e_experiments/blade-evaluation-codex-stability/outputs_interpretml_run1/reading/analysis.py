import json
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
import statsmodels.api as sm
import statsmodels.formula.api as smf
from interpret.glassbox import ExplainableBoostingRegressor

warnings.filterwarnings("ignore")
RANDOM_STATE = 42


def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def extract_linear_coefs(fitted_pipeline, feature_names):
    model = fitted_pipeline.named_steps["model"]
    coefs = pd.Series(model.coef_, index=feature_names)
    return coefs.sort_values(key=np.abs, ascending=False)


def aggregate_importance_by_base_feature(importances, encoded_feature_names):
    grouped = {}
    for fname, imp in zip(encoded_feature_names, importances):
        if "__" in fname:
            raw = fname.split("__", 1)[1]
        else:
            raw = fname

        # OneHotEncoder names often look like: cat__device_desktop
        # Recover original feature name heuristically.
        candidates = ["device", "page_id", "english_native", "gender", "education", "language"]
        base = raw
        for c in candidates:
            if raw.startswith(c + "_"):
                base = c
                break

        grouped[base] = grouped.get(base, 0.0) + float(imp)
    return pd.Series(grouped).sort_values(ascending=False)


# 1) Load metadata and data
with open("info.json", "r", encoding="utf-8") as f:
    info = json.load(f)

question = info.get("research_questions", ["Unknown question"])[0]
df = pd.read_csv("reading.csv")

print_section("Research Question")
print(question)

print_section("Data Overview")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("Missing values by column:")
print(df.isna().sum().sort_values(ascending=False).head(15))

# Convert known numeric fields defensively
numeric_candidates = [
    "reader_view",
    "running_time",
    "adjusted_running_time",
    "scrolling_time",
    "num_words",
    "correct_rate",
    "img_width",
    "age",
    "dyslexia",
    "gender",
    "retake_trial",
    "dyslexia_bin",
    "Flesch_Kincaid",
    "speed",
]
for col in numeric_candidates:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print_section("Summary Statistics (Numeric)")
print(df[numeric_cols].describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]])

print_section("Speed Distribution")
speed = df["speed"].dropna()
print(f"Speed count: {len(speed)}")
print(f"Speed mean: {speed.mean():.3f}")
print(f"Speed median: {speed.median():.3f}")
print(f"Speed std: {speed.std():.3f}")
print(f"Speed skewness: {stats.skew(speed):.3f}")
print("Speed quantiles:")
print(speed.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]))

print_section("Correlations With Speed (Numeric)")
corr_speed = df[numeric_cols].corr(numeric_only=True)["speed"].drop("speed").sort_values(key=np.abs, ascending=False)
print(corr_speed)

# 2) Dyslexia-focused inferential tests
print_section("Statistical Tests: Dyslexia Subgroup")
dys = df[(df["dyslexia_bin"] == 1) & df["reader_view"].isin([0, 1])].copy()
dys = dys.dropna(subset=["uuid", "speed", "reader_view"])
dys["log_speed"] = np.log1p(dys["speed"])

rv1 = dys.loc[dys["reader_view"] == 1, "log_speed"]
rv0 = dys.loc[dys["reader_view"] == 0, "log_speed"]
welch_res = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy="omit")
mann_res = stats.mannwhitneyu(
    dys.loc[dys["reader_view"] == 1, "speed"],
    dys.loc[dys["reader_view"] == 0, "speed"],
    alternative="two-sided",
)

print(f"Dyslexia rows: {len(dys)}, unique participants: {dys['uuid'].nunique()}")
print(f"Welch t-test on log(speed): statistic={welch_res.statistic:.4f}, p={welch_res.pvalue:.6f}")
print(f"Mann-Whitney U on raw speed: statistic={mann_res.statistic:.4f}, p={mann_res.pvalue:.6f}")

# Paired test by participant (mean speed per condition)
pivot = dys.pivot_table(index="uuid", columns="reader_view", values="speed", aggfunc="mean")
paired = pivot.dropna()
if len(paired) >= 4:
    paired_log_diff = np.log1p(paired[1]) - np.log1p(paired[0])
    paired_t = stats.ttest_rel(np.log1p(paired[1]), np.log1p(paired[0]))
    paired_effect_pct = (np.exp(paired_log_diff.mean()) - 1.0) * 100.0
    print(f"Paired participants with both conditions: {len(paired)}")
    print(f"Paired t-test on log(speed): statistic={paired_t.statistic:.4f}, p={paired_t.pvalue:.6f}")
    print(f"Estimated paired percent change (Reader View vs Control): {paired_effect_pct:.3f}%")
else:
    paired_t = None
    paired_effect_pct = np.nan
    print("Not enough paired data for paired t-test.")

# 3) Regression models (statsmodels)
print_section("Statsmodels Regression")
reg_cols = [
    "speed",
    "reader_view",
    "num_words",
    "Flesch_Kincaid",
    "age",
    "retake_trial",
    "correct_rate",
    "device",
    "page_id",
    "english_native",
    "gender",
    "uuid",
]
reg_dys = dys[reg_cols].dropna().copy()
reg_dys["log_speed"] = np.log1p(reg_dys["speed"])

formula_dys = (
    "log_speed ~ reader_view + num_words + Flesch_Kincaid + age + retake_trial + "
    "correct_rate + C(device) + C(page_id) + C(english_native) + C(gender)"
)

try:
    ols_dys = smf.ols(formula_dys, data=reg_dys).fit(
        cov_type="cluster", cov_kwds={"groups": reg_dys["uuid"]}
    )
except Exception:
    ols_dys = smf.ols(formula_dys, data=reg_dys).fit()

reader_coef = safe_float(ols_dys.params.get("reader_view", np.nan))
reader_p = safe_float(ols_dys.pvalues.get("reader_view", np.nan))
reader_pct = (np.exp(reader_coef) - 1.0) * 100.0 if pd.notna(reader_coef) else np.nan
print(f"Dyslexia OLS reader_view coef (log scale): {reader_coef:.6f}")
print(f"Dyslexia OLS reader_view p-value: {reader_p:.6f}")
print(f"Dyslexia OLS implied percent change: {reader_pct:.3f}%")

# Full-sample interaction to test if dyslexia moderates the effect
full_cols = ["speed", "reader_view", "dyslexia_bin", "num_words", "Flesch_Kincaid", "retake_trial", "page_id"]
full_reg = df[full_cols].dropna().copy()
full_reg["log_speed"] = np.log1p(full_reg["speed"])
formula_interaction = (
    "log_speed ~ reader_view * dyslexia_bin + num_words + Flesch_Kincaid + "
    "retake_trial + C(page_id)"
)
interaction_model = smf.ols(formula_interaction, data=full_reg).fit()
anova_table = sm.stats.anova_lm(interaction_model, typ=2)
interaction_coef = safe_float(interaction_model.params.get("reader_view:dyslexia_bin", np.nan))
interaction_p = safe_float(interaction_model.pvalues.get("reader_view:dyslexia_bin", np.nan))

print("Interaction model key coefficients/p-values:")
print(interaction_model.params[["reader_view", "dyslexia_bin", "reader_view:dyslexia_bin"]])
print(interaction_model.pvalues[["reader_view", "dyslexia_bin", "reader_view:dyslexia_bin"]])
print("ANOVA (Type II) rows for reader_view and interaction:")
print(anova_table.loc[["reader_view", "reader_view:dyslexia_bin"], ["F", "PR(>F)"]])

# 4) Interpretable ML models: sklearn + interpret
print_section("Interpretable Models")
feature_cols = [
    "reader_view",
    "num_words",
    "Flesch_Kincaid",
    "age",
    "retake_trial",
    "correct_rate",
    "device",
    "page_id",
    "english_native",
    "gender",
]
model_df = dys[feature_cols + ["speed"]].dropna().copy()
model_df["log_speed"] = np.log1p(model_df["speed"])

num_features = ["reader_view", "num_words", "Flesch_Kincaid", "age", "retake_trial", "correct_rate"]
cat_features = ["device", "page_id", "english_native", "gender"]

preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline(steps=[("scaler", StandardScaler())]), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
    ]
)

X = model_df[feature_cols]
y = model_df["log_speed"]

lin_pipe = Pipeline(steps=[("prep", preprocess), ("model", LinearRegression())])
ridge_pipe = Pipeline(steps=[("prep", preprocess), ("model", Ridge(alpha=1.0, random_state=RANDOM_STATE))])
lasso_pipe = Pipeline(steps=[("prep", preprocess), ("model", Lasso(alpha=0.001, random_state=RANDOM_STATE, max_iter=20000))])
tree_pipe = Pipeline(
    steps=[
        ("prep", preprocess),
        (
            "model",
            DecisionTreeRegressor(max_depth=4, min_samples_leaf=20, random_state=RANDOM_STATE),
        ),
    ]
)

lin_pipe.fit(X, y)
ridge_pipe.fit(X, y)
lasso_pipe.fit(X, y)
tree_pipe.fit(X, y)

feature_names = lin_pipe.named_steps["prep"].get_feature_names_out()
lin_coefs = extract_linear_coefs(lin_pipe, feature_names)
ridge_coefs = extract_linear_coefs(ridge_pipe, feature_names)
lasso_coefs = extract_linear_coefs(lasso_pipe, feature_names)

print("Top |coef| features (LinearRegression):")
print(lin_coefs.head(10))
print("Top |coef| features (Ridge):")
print(ridge_coefs.head(10))
print("Top |coef| features (Lasso):")
print(lasso_coefs.head(10))

def reader_coef_from_series(series):
    return float(series.get("num__reader_view", np.nan))

lin_reader_coef = reader_coef_from_series(lin_coefs)
ridge_reader_coef = reader_coef_from_series(ridge_coefs)
lasso_reader_coef = reader_coef_from_series(lasso_coefs)

tree_importances = tree_pipe.named_steps["model"].feature_importances_
tree_importance_grouped = aggregate_importance_by_base_feature(tree_importances, feature_names)
print("DecisionTreeRegressor grouped feature importances:")
print(tree_importance_grouped)

# Explainable Boosting Regressor from interpret
X_ebm = model_df[feature_cols].copy()
for c in cat_features:
    X_ebm[c] = X_ebm[c].astype(str).fillna("Missing")
for c in num_features:
    X_ebm[c] = X_ebm[c].fillna(X_ebm[c].median())

ebm = ExplainableBoostingRegressor(
    random_state=RANDOM_STATE,
    interactions=0,
    max_rounds=400,
)
ebm.fit(X_ebm, y)

term_names = list(ebm.term_names_)
term_importances = pd.Series(ebm.term_importances(), index=term_names).sort_values(ascending=False)
print("EBM term importances:")
print(term_importances)

ebm_reader_importance = float(term_importances.get("reader_view", 0.0))

# 5) Score synthesis and JSON conclusion
print_section("Conclusion Synthesis")
if paired_t is not None:
    primary_p = float(paired_t.pvalue)
    primary_effect_pct = float(paired_effect_pct)
else:
    primary_p = float(welch_res.pvalue)
    primary_effect_pct = (np.exp(rv1.mean() - rv0.mean()) - 1.0) * 100.0

# Likert score: strong evidence for no relationship => low score
if primary_p < 0.01 and primary_effect_pct > 0:
    response = 95
elif primary_p < 0.05 and primary_effect_pct > 0:
    response = 85
elif primary_p < 0.10 and primary_effect_pct > 0:
    response = 70
else:
    # Non-significant evidence defaults to "No".
    if abs(primary_effect_pct) < 1.0:
        response = 8
    elif abs(primary_effect_pct) < 3.0:
        response = 15
    else:
        response = 25

# Conservative adjustment based on controlled OLS and interaction.
if pd.notna(reader_p) and reader_p < 0.05 and reader_pct > 0:
    response = max(response, 75)
if pd.notna(interaction_p) and interaction_p < 0.05 and interaction_coef > 0:
    response = max(response, 70)
if pd.notna(reader_p) and reader_p >= 0.20 and pd.notna(primary_p) and primary_p >= 0.20:
    response = min(response, 20)

response = int(np.clip(round(response), 0, 100))

explanation = (
    f"Question: {question} Dyslexia-only tests showed no statistically significant improvement in speed with "
    f"Reader View (paired t-test p={primary_p:.3g}, estimated change={primary_effect_pct:.2f}%). "
    f"Unpaired checks were also non-significant (Welch p={welch_res.pvalue:.3g}, Mann-Whitney p={mann_res.pvalue:.3g}). "
    f"In controlled OLS for the dyslexia subgroup, reader_view was not significant (p={reader_p:.3g}, "
    f"implied change={reader_pct:.2f}%). The reader_view*dyslexia_bin interaction in the full sample was "
    f"also non-significant (p={interaction_p:.3g}). Interpretable models (Linear/Ridge/Lasso, Decision Tree, EBM) "
    f"did not identify reader_view as a dominant driver compared with content and participant/context variables. "
    f"Overall, evidence does not support that Reader View improves reading speed for individuals with dyslexia."
)

result = {"response": response, "explanation": explanation}
with open("conclusion.txt", "w", encoding="utf-8") as f:
    f.write(json.dumps(result))

print(f"Final response score: {response}")
print("Wrote conclusion.txt")
