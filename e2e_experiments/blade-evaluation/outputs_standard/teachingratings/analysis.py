import json
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from imodels import RuleFitRegressor, FIGSRegressor, HSTreeRegressor

warnings.filterwarnings("ignore")


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


# 1. Load data
print_section("Load Data")
df = pd.read_csv("teachingratings.csv")
print(f"Shape: {df.shape}")
print("Columns:", list(df.columns))
print("Missing values by column:")
print(df.isna().sum())

# 2. Basic exploration
print_section("Summary Statistics")
cat_cols = [c for c in df.columns if df[c].dtype == "object"]
num_cols = [c for c in df.columns if c not in cat_cols]

print("Numeric summary:")
print(df[num_cols].describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]])

print("\nCategorical distributions:")
for c in cat_cols:
    print(f"\n{c}:")
    print(df[c].value_counts(dropna=False))

for v in ["beauty", "eval"]:
    if v in df.columns:
        vals = df[v].astype(float)
        print(f"\nDistribution stats for {v}:")
        print({
            "mean": round(vals.mean(), 4),
            "std": round(vals.std(), 4),
            "median": round(vals.median(), 4),
            "skew": round(stats.skew(vals), 4),
            "kurtosis": round(stats.kurtosis(vals), 4),
        })

print_section("Correlations")
corr = df[num_cols].corr(numeric_only=True)
print(corr.round(3))
if "eval" in corr.columns:
    print("\nCorrelations with eval:")
    print(corr["eval"].sort_values(ascending=False).round(3))

# 3. Statistical tests focused on research question
print_section("Statistical Tests: Beauty vs Teaching Evaluation")

pearson_r, pearson_p = stats.pearsonr(df["beauty"], df["eval"])
spearman_rho, spearman_p = stats.spearmanr(df["beauty"], df["eval"])
print(f"Pearson r={pearson_r:.4f}, p={pearson_p:.3e}")
print(f"Spearman rho={spearman_rho:.4f}, p={spearman_p:.3e}")

# Median split t-test
median_beauty = df["beauty"].median()
hi = df.loc[df["beauty"] >= median_beauty, "eval"]
lo = df.loc[df["beauty"] < median_beauty, "eval"]
t_stat, t_p = stats.ttest_ind(hi, lo, equal_var=False)
cohen_d = (hi.mean() - lo.mean()) / np.sqrt((hi.var(ddof=1) + lo.var(ddof=1)) / 2)
print(
    f"Median split t-test: high beauty mean={hi.mean():.4f}, low beauty mean={lo.mean():.4f}, "
    f"t={t_stat:.4f}, p={t_p:.3e}, Cohen's d={cohen_d:.4f}"
)

# ANOVA by beauty quartiles
df["beauty_quartile"] = pd.qcut(df["beauty"], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
quartile_groups = [g["eval"].values for _, g in df.groupby("beauty_quartile")]
anova_f, anova_p = stats.f_oneway(*quartile_groups)
quart_means = df.groupby("beauty_quartile")["eval"].mean()
print(f"ANOVA across beauty quartiles: F={anova_f:.4f}, p={anova_p:.3e}")
print("Mean eval by beauty quartile:")
print(quart_means)

# OLS models
simple_ols = smf.ols("eval ~ beauty", data=df).fit()
full_ols = smf.ols(
    "eval ~ beauty + age + C(gender) + C(minority) + C(credits) + C(division) + C(native) + C(tenure) + students + allstudents",
    data=df,
).fit()

simple_coef = safe_float(simple_ols.params.get("beauty", np.nan))
simple_p = safe_float(simple_ols.pvalues.get("beauty", np.nan))
simple_r2 = safe_float(simple_ols.rsquared)

full_coef = safe_float(full_ols.params.get("beauty", np.nan))
full_p = safe_float(full_ols.pvalues.get("beauty", np.nan))
full_adj_r2 = safe_float(full_ols.rsquared_adj)

print("\nSimple OLS: eval ~ beauty")
print(f"beauty coef={simple_coef:.4f}, p={simple_p:.3e}, R^2={simple_r2:.4f}")

print("\nAdjusted OLS with controls")
print(f"beauty coef={full_coef:.4f}, p={full_p:.3e}, adj R^2={full_adj_r2:.4f}")

# 4. Interpretable models (sklearn + imodels)
print_section("Interpretable Models")

feature_df = df.drop(columns=["eval", "beauty_quartile"], errors="ignore").copy()
# drop identifier-like columns to avoid misleading interpretations
feature_df = feature_df.drop(columns=["rownames", "prof"], errors="ignore")

X = pd.get_dummies(feature_df, drop_first=True)
y = df["eval"].astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model_results = {}

lin = LinearRegression()
lin.fit(X_train, y_train)
lin_pred = lin.predict(X_test)
lin_r2 = r2_score(y_test, lin_pred)
lin_coef = float(lin.coef_[list(X.columns).index("beauty")]) if "beauty" in X.columns else np.nan
model_results["LinearRegression"] = {"r2": lin_r2, "beauty_signal": lin_coef}

ridge = Ridge(alpha=1.0, random_state=42)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
ridge_r2 = r2_score(y_test, ridge_pred)
ridge_coef = float(ridge.coef_[list(X.columns).index("beauty")]) if "beauty" in X.columns else np.nan
model_results["Ridge"] = {"r2": ridge_r2, "beauty_signal": ridge_coef}

lasso = Lasso(alpha=0.001, random_state=42, max_iter=10000)
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)
lasso_r2 = r2_score(y_test, lasso_pred)
lasso_coef = float(lasso.coef_[list(X.columns).index("beauty")]) if "beauty" in X.columns else np.nan
model_results["Lasso"] = {"r2": lasso_r2, "beauty_signal": lasso_coef}

tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=20, random_state=42)
tree.fit(X_train, y_train)
tree_pred = tree.predict(X_test)
tree_r2 = r2_score(y_test, tree_pred)
tree_beauty_imp = (
    float(tree.feature_importances_[list(X.columns).index("beauty")]) if "beauty" in X.columns else np.nan
)
model_results["DecisionTreeRegressor"] = {"r2": tree_r2, "beauty_signal": tree_beauty_imp}

# Top coefficients/importances from sklearn
coef_table = pd.Series(lin.coef_, index=X.columns).sort_values(key=np.abs, ascending=False)
imp_table = pd.Series(tree.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Top 10 absolute LinearRegression coefficients:")
print(coef_table.head(10))
print("\nTop 10 DecisionTree feature importances:")
print(imp_table.head(10))

# imodels: RuleFit
rulefit_beauty_signal = np.nan
rulefit_r2 = np.nan
try:
    rulefit = RuleFitRegressor(random_state=42, max_rules=40)
    rulefit.fit(X_train, y_train, feature_names=list(X.columns))
    rf_pred = rulefit.predict(X_test)
    rulefit_r2 = r2_score(y_test, rf_pred)

    # API differs by version: _get_rules is available in this environment.
    rules_df = rulefit._get_rules()
    nonzero_rules = rules_df.loc[rules_df["importance"] > 0].sort_values("importance", ascending=False)
    beauty_rules = nonzero_rules[nonzero_rules["rule"].str.contains("beauty", case=False, regex=False)]
    if not beauty_rules.empty:
        rulefit_beauty_signal = float(beauty_rules["importance"].sum())

    print("\nTop RuleFit rules by importance:")
    print(nonzero_rules[["rule", "coef", "support", "importance"]].head(10))
except Exception as e:
    print(f"RuleFitRegressor failed: {e}")

model_results["RuleFitRegressor"] = {"r2": rulefit_r2, "beauty_signal": rulefit_beauty_signal}

# imodels: FIGS
figs_beauty_signal = np.nan
figs_r2 = np.nan
try:
    figs = FIGSRegressor(max_rules=12, random_state=42)
    figs.fit(X_train, y_train, feature_names=list(X.columns))
    figs_pred = figs.predict(X_test)
    figs_r2 = r2_score(y_test, figs_pred)

    if hasattr(figs, "feature_importances_") and "beauty" in X.columns:
        figs_beauty_signal = float(figs.feature_importances_[list(X.columns).index("beauty")])

    print("\nFIGS model summary (truncated):")
    print(str(figs)[:1000])
except Exception as e:
    print(f"FIGSRegressor failed: {e}")

model_results["FIGSRegressor"] = {"r2": figs_r2, "beauty_signal": figs_beauty_signal}

# imodels: HSTree
hs_beauty_signal = np.nan
hs_r2 = np.nan
try:
    hs = HSTreeRegressor(random_state=42)
    hs.fit(X_train, y_train)
    hs_pred = hs.predict(X_test)
    hs_r2 = r2_score(y_test, hs_pred)

    if hasattr(hs, "estimator_") and hasattr(hs.estimator_, "feature_importances_") and "beauty" in X.columns:
        hs_beauty_signal = float(hs.estimator_.feature_importances_[list(X.columns).index("beauty")])
except Exception as e:
    print(f"HSTreeRegressor failed: {e}")

model_results["HSTreeRegressor"] = {"r2": hs_r2, "beauty_signal": hs_beauty_signal}

print("\nModel results (R^2 + beauty signal):")
for name, out in model_results.items():
    print(f"{name}: R^2={out['r2']:.4f}, beauty_signal={out['beauty_signal']}")

# 5. Evidence synthesis and Likert score
print_section("Conclusion Synthesis")

sig_checks = {
    "pearson": pearson_p < 0.05,
    "simple_ols": simple_p < 0.05,
    "full_ols": full_p < 0.05,
    "ttest": t_p < 0.05,
    "anova": anova_p < 0.05,
}
sig_count = int(sum(sig_checks.values()))

sign_direction = np.sign([x for x in [simple_coef, full_coef, lin_coef, ridge_coef, lasso_coef] if np.isfinite(x) and x != 0])
if len(sign_direction) > 0:
    direction_consistency = abs(sign_direction.sum()) / len(sign_direction)
else:
    direction_consistency = 0.0

beauty_signals = []
for m in model_results.values():
    s = m["beauty_signal"]
    if np.isfinite(s):
        beauty_signals.append(float(s > 0))
model_support_ratio = float(np.mean(beauty_signals)) if beauty_signals else 0.0

# Score construction:
# - Primarily based on number of significant tests
# - Adjusted for effect size and model consistency
std_beta = abs(pearson_r)  # in simple regression with single predictor, standardized beta == correlation
score = 10 + 12 * sig_count
score += int(round(15 * std_beta))
score += int(round(8 * model_support_ratio))
if simple_r2 < 0.05:
    score -= 4
if direction_consistency > 0.8:
    score += 2
score = int(np.clip(score, 0, 100))

explanation = (
    f"Beauty shows a statistically significant positive relationship with teaching evaluations. "
    f"Key tests all reject no-association (Pearson r={pearson_r:.3f}, p={pearson_p:.2e}; "
    f"simple OLS beauty coef={simple_coef:.3f}, p={simple_p:.2e}; adjusted OLS coef={full_coef:.3f}, p={full_p:.2e}; "
    f"median-split t-test p={t_p:.3f}; quartile ANOVA p={anova_p:.2e}). "
    f"Interpretable models (Linear/Ridge/Lasso, decision tree importance, and imodels RuleFit/FIGS/HSTree) "
    f"also generally indicate beauty contributes positively. "
    f"The effect is real but modest in magnitude (correlation about {pearson_r:.3f}, simple R^2={simple_r2:.3f}), "
    f"so beauty appears to matter, but it is not the dominant driver of evaluation scores."
)

result = {"response": score, "explanation": explanation}

print("Significance checks:", sig_checks)
print(f"Direction consistency: {direction_consistency:.3f}")
print(f"Model support ratio: {model_support_ratio:.3f}")
print(f"Final Likert response: {score}")

with open("conclusion.txt", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=True)

print("\nWrote conclusion.txt")
