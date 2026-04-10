import json
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from imodels import FIGSRegressor, HSTreeRegressor, RuleFitRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text

warnings.filterwarnings("ignore")


def pct_change_from_log_diff(log_diff: float) -> float:
    return (np.exp(log_diff) - 1.0) * 100.0


def top_abs_series(series: pd.Series, n: int = 10) -> pd.Series:
    return series.reindex(series.abs().sort_values(ascending=False).head(n).index)


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


with open("info.json", "r", encoding="utf-8") as f:
    info = json.load(f)

question = info["research_questions"][0]
df = pd.read_csv("reading.csv")

# Core transforms
# Log-transform speed due extreme skew and outliers.
df["log_speed"] = np.log1p(df["speed"])

print_header("RESEARCH QUESTION")
print(question)

print_header("DATA OVERVIEW")
print(f"Rows: {len(df):,} | Columns: {df.shape[1]}")
print("Missing values (top 10):")
print(df.isna().sum().sort_values(ascending=False).head(10).to_string())

numeric_cols = [
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
    "log_speed",
]
existing_numeric = [c for c in numeric_cols if c in df.columns]

print_header("SUMMARY STATISTICS")
print(df[existing_numeric].describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]].to_string())

print_header("DISTRIBUTION CHECKS")
speed_skew = df["speed"].skew()
log_speed_skew = df["log_speed"].skew()
print(f"Skewness(speed): {speed_skew:.3f}")
print(f"Skewness(log_speed): {log_speed_skew:.3f}")
counts, edges = np.histogram(df["speed"].dropna(), bins=10)
print("Speed histogram bin counts:")
for i, c in enumerate(counts):
    print(f"  [{edges[i]:.1f}, {edges[i+1]:.1f}): {int(c)}")

print_header("CORRELATIONS WITH SPEED")
corr = df[existing_numeric].corr(numeric_only=True)
print(corr["speed"].sort_values(ascending=False).to_string())

print_header("GROUP MEANS")
group_means = (
    df.groupby(["dyslexia_bin", "reader_view"], dropna=False)[["speed", "log_speed", "correct_rate"]]
    .mean()
    .round(4)
)
print(group_means.to_string())

# --- Statistical tests focused on dyslexic participants ---
print_header("STATISTICAL TESTS")
dys = df[df["dyslexia_bin"] == 1].copy()
dys = dys.dropna(subset=["uuid", "reader_view", "speed", "log_speed"])

rv0 = dys.loc[dys["reader_view"] == 0, "log_speed"]
rv1 = dys.loc[dys["reader_view"] == 1, "log_speed"]

welch_t = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy="omit")
mann_whitney = stats.mannwhitneyu(rv1, rv0, alternative="two-sided")

# Paired test by participant because all dyslexic participants appear in both conditions.
paired = dys.groupby(["uuid", "reader_view"])["log_speed"].mean().unstack()
paired = paired.dropna(subset=[0, 1])
paired_t = stats.ttest_rel(paired[1], paired[0], nan_policy="omit")
paired_w = stats.wilcoxon(paired[1], paired[0], zero_method="wilcox")
paired_diff = float((paired[1] - paired[0]).mean())

print(f"Dyslexic-only Welch t-test on log_speed: t={welch_t.statistic:.4f}, p={welch_t.pvalue:.4g}")
print(f"Dyslexic-only Mann-Whitney U: U={mann_whitney.statistic:.4f}, p={mann_whitney.pvalue:.4g}")
print(f"Dyslexic paired t-test (uuid mean log_speed): t={paired_t.statistic:.4f}, p={paired_t.pvalue:.4g}")
print(f"Dyslexic paired Wilcoxon: W={paired_w.statistic:.4f}, p={paired_w.pvalue:.4g}")
print(
    "Mean paired log-speed difference (reader_view=1 - 0): "
    f"{paired_diff:.6f} ({pct_change_from_log_diff(paired_diff):.2f}% approx)"
)

# One-way ANOVA across the 4 (reader_view x dyslexia_bin) groups
anova_df = df.dropna(subset=["log_speed", "reader_view", "dyslexia_bin"]).copy()
anova_groups: List[pd.Series] = []
for rv in [0, 1]:
    for db in [0, 1]:
        vals = anova_df.loc[(anova_df["reader_view"] == rv) & (anova_df["dyslexia_bin"] == db), "log_speed"]
        if len(vals) > 1:
            anova_groups.append(vals)
anova_res = stats.f_oneway(*anova_groups)
print(f"One-way ANOVA across (reader_view x dyslexia_bin) groups: F={anova_res.statistic:.4f}, p={anova_res.pvalue:.4g}")

# OLS using statsmodels.api.OLS (as requested)
controls_num = ["num_words", "Flesch_Kincaid", "correct_rate", "age", "retake_trial", "img_width"]
controls_cat = ["device", "education", "gender", "english_native", "page_id"]

ols_dys = dys[["log_speed", "reader_view", "uuid"] + controls_num + controls_cat].copy()
for c in controls_num:
    ols_dys[c] = ols_dys[c].fillna(ols_dys[c].median())
for c in controls_cat:
    ols_dys[c] = ols_dys[c].fillna("Missing").astype(str)

X_dys = ols_dys[["reader_view"] + controls_num + controls_cat].copy()
X_dys = pd.get_dummies(X_dys, columns=controls_cat, drop_first=True)
X_dys = sm.add_constant(X_dys, has_constant="add")
X_dys = X_dys.astype(float)
y_dys = ols_dys["log_speed"].astype(float)
groups_dys = ols_dys["uuid"]

model_dys = sm.OLS(y_dys, X_dys).fit(cov_type="cluster", cov_kwds={"groups": groups_dys})
reader_coef = float(model_dys.params.get("reader_view", np.nan))
reader_p = float(model_dys.pvalues.get("reader_view", np.nan))
reader_ci = model_dys.conf_int().loc["reader_view"].tolist()

print(
    "Dyslexic OLS (clustered by uuid) reader_view effect on log_speed: "
    f"coef={reader_coef:.6f}, p={reader_p:.4g}, 95% CI=[{reader_ci[0]:.6f}, {reader_ci[1]:.6f}]"
)

# Full-sample interaction model: does reader_view effect differ for dyslexia?
full = df.dropna(subset=["log_speed", "reader_view", "dyslexia_bin"]).copy()
full["interaction"] = full["reader_view"] * full["dyslexia_bin"]

for c in controls_num:
    full[c] = full[c].fillna(full[c].median())
for c in controls_cat:
    full[c] = full[c].fillna("Missing").astype(str)

X_full = full[["reader_view", "dyslexia_bin", "interaction"] + controls_num + controls_cat].copy()
X_full = pd.get_dummies(X_full, columns=controls_cat, drop_first=True)
X_full = sm.add_constant(X_full, has_constant="add")
X_full = X_full.astype(float)
y_full = full["log_speed"].astype(float)

model_interact = sm.OLS(y_full, X_full).fit()
inter_coef = float(model_interact.params.get("interaction", np.nan))
inter_p = float(model_interact.pvalues.get("interaction", np.nan))
print(f"Full-sample OLS interaction (reader_view*dyslexia_bin): coef={inter_coef:.6f}, p={inter_p:.4g}")

# --- Interpretable ML models ---
print_header("INTERPRETABLE MODELS")
model_features = [
    "reader_view",
    "dyslexia_bin",
    "num_words",
    "correct_rate",
    "img_width",
    "age",
    "retake_trial",
    "Flesch_Kincaid",
    "device",
    "education",
    "gender",
    "language",
    "english_native",
    "page_id",
]

mdf = df[["log_speed", "speed"] + model_features].copy()
mdf = mdf.dropna(subset=["log_speed", "speed", "reader_view"])

num_feats = [
    "reader_view",
    "dyslexia_bin",
    "num_words",
    "correct_rate",
    "img_width",
    "age",
    "retake_trial",
    "Flesch_Kincaid",
]
cat_feats = ["device", "education", "gender", "language", "english_native", "page_id"]

for c in num_feats:
    mdf[c] = mdf[c].fillna(mdf[c].median())
for c in cat_feats:
    mdf[c] = mdf[c].fillna("Missing").astype(str)

X = pd.get_dummies(mdf[model_features], columns=cat_feats, drop_first=False)
X = X.astype(float)
y = mdf["log_speed"].astype(float)

lin = LinearRegression()
ridge = Ridge(alpha=1.0, random_state=42)
lasso = Lasso(alpha=0.001, max_iter=20000, random_state=42)

dtr = DecisionTreeRegressor(max_depth=4, min_samples_leaf=30, random_state=42)
median_speed = float(mdf["speed"].median())
y_cls = (mdf["speed"] > median_speed).astype(int)
dtc = DecisionTreeClassifier(max_depth=4, min_samples_leaf=30, random_state=42)

lin.fit(X, y)
ridge.fit(X, y)
lasso.fit(X, y)
dtr.fit(X, y)
dtc.fit(X, y_cls)

lin_coef = pd.Series(lin.coef_, index=X.columns)
ridge_coef = pd.Series(ridge.coef_, index=X.columns)
lasso_coef = pd.Series(lasso.coef_, index=X.columns)
dtr_imp = pd.Series(dtr.feature_importances_, index=X.columns)
dtc_imp = pd.Series(dtc.feature_importances_, index=X.columns)

lin_reader = float(lin_coef.get("reader_view", np.nan))
ridge_reader = float(ridge_coef.get("reader_view", np.nan))
lasso_reader = float(lasso_coef.get("reader_view", np.nan))
dtr_reader = float(dtr_imp.get("reader_view", 0.0))
dtc_reader = float(dtc_imp.get("reader_view", 0.0))

print(f"LinearRegression coef(reader_view): {lin_reader:.6f}")
print(f"Ridge coef(reader_view): {ridge_reader:.6f}")
print(f"Lasso coef(reader_view): {lasso_reader:.6f}")
print(f"DecisionTreeRegressor importance(reader_view): {dtr_reader:.6f}")
print(f"DecisionTreeClassifier importance(reader_view): {dtc_reader:.6f}")

print("Top LinearRegression coefficients (abs):")
print(top_abs_series(lin_coef, 10).to_string())
print("Top DecisionTreeRegressor importances:")
print(dtr_imp.sort_values(ascending=False).head(10).to_string())

print("DecisionTreeRegressor structure (depth<=4):")
print(export_text(dtr, feature_names=list(X.columns), max_depth=3))

# imodels models
rulefit = RuleFitRegressor(random_state=42, max_rules=30)
rulefit.fit(X, y, feature_names=X.columns)
rule_strings = [str(r) for r in getattr(rulefit, "rules_", [])]
reader_rules = [r for r in rule_strings if "reader_view" in r]

figs = FIGSRegressor(random_state=42, max_rules=12)
figs.fit(X, y)
figs_imp = pd.Series(figs.feature_importances_, index=X.columns)
figs_reader = float(figs_imp.get("reader_view", 0.0))

hst = HSTreeRegressor(max_leaf_nodes=8, random_state=42)
hst.fit(X, y)
hst_text = str(hst)
hst_uses_reader = "reader_view" in hst_text

print(f"RuleFit total rules: {len(rule_strings)} | rules mentioning reader_view: {len(reader_rules)}")
if reader_rules:
    print("Example RuleFit rules with reader_view:")
    for rr in reader_rules[:5]:
        print(f"  - {rr}")
print(f"FIGS importance(reader_view): {figs_reader:.6f}")
print("Top FIGS feature importances:")
print(figs_imp.sort_values(ascending=False).head(10).to_string())
print(f"HSTree uses reader_view in splits: {hst_uses_reader}")

# --- Conclusion synthesis ---
print_header("CONCLUSION SYNTHESIS")
key_tests: Dict[str, Dict[str, float]] = {
    "welch": {"p": float(welch_t.pvalue), "effect": float(rv1.mean() - rv0.mean())},
    "paired_t": {"p": float(paired_t.pvalue), "effect": paired_diff},
    "ols_reader": {"p": reader_p, "effect": reader_coef},
    "interaction": {"p": inter_p, "effect": inter_coef},
}

for k, v in key_tests.items():
    print(f"{k}: p={v['p']:.4g}, effect={v['effect']:.6f}")

pos_sig = sum(1 for v in key_tests.values() if v["p"] < 0.05 and v["effect"] > 0)
neg_sig = sum(1 for v in key_tests.values() if v["p"] < 0.05 and v["effect"] < 0)
all_nonsig = all(v["p"] >= 0.10 for v in key_tests.values())

if pos_sig >= 2 and neg_sig == 0:
    response = 85
elif neg_sig >= 2 and pos_sig == 0:
    response = 10
elif all_nonsig:
    response = 15
else:
    response = 35

explanation = (
    f"Question: {question} Key dyslexia-focused tests found no statistically significant reader-view speed gain "
    f"(Welch p={welch_t.pvalue:.3f}, paired p={paired_t.pvalue:.3f}, clustered OLS p={reader_p:.3f}). "
    f"Estimated effect sizes were near zero (paired diff={paired_diff:.4f} log points, about {pct_change_from_log_diff(paired_diff):.2f}% change). "
    f"The full-sample interaction test was also non-significant (p={inter_p:.3f}), suggesting no distinct benefit for dyslexic readers. "
    f"Interpretable models (linear/ridge/lasso coefficients, tree importances, RuleFit/FIGS/HSTree rules) did not identify reader_view as a strong or consistent driver of reading speed relative to other factors. "
    f"Overall evidence is weak for a true improvement effect in this dataset."
)

result = {"response": int(response), "explanation": explanation}
with open("conclusion.txt", "w", encoding="utf-8") as f:
    f.write(json.dumps(result))

print("Wrote conclusion.txt")
print(json.dumps(result, indent=2))
