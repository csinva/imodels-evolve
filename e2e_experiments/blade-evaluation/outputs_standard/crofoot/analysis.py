import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text

warnings.filterwarnings("ignore")


def safe_num(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "nan"
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


# 1) Load metadata and data
info = json.loads(Path("info.json").read_text())
question = info.get("research_questions", ["Unknown question"])[0].strip()
df = pd.read_csv("crofoot.csv")

# 2) Feature engineering focused on the question
# Relative group size and location advantage for focal group.
df["rel_group_size"] = df["n_focal"] - df["n_other"]
df["rel_male"] = df["m_focal"] - df["m_other"]
df["rel_female"] = df["f_focal"] - df["f_other"]
df["location_advantage"] = df["dist_other"] - df["dist_focal"]
df["dist_ratio"] = df["dist_focal"] / (df["dist_other"] + 1e-9)

print("Research question:")
print(question)
print("\nData shape:", df.shape)
print("Missing values by column:")
print(df.isna().sum())

# 3) Exploration: summary statistics, distributions, correlations
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print("\nSummary statistics:")
print(df[numeric_cols].describe().T)

print("\nWin distribution:")
print(df["win"].value_counts(dropna=False).sort_index())
print("Win rate:", fmt(df["win"].mean()))

corr_cols = [
    "win",
    "rel_group_size",
    "location_advantage",
    "dist_focal",
    "dist_other",
    "n_focal",
    "n_other",
    "rel_male",
    "rel_female",
]
corr_cols = [c for c in corr_cols if c in df.columns]
print("\nCorrelation matrix (Pearson):")
print(df[corr_cols].corr().round(3))

# Point-biserial correlations with binary outcome
pb_results = {}
for feat in ["rel_group_size", "location_advantage"]:
    r, p = stats.pointbiserialr(df["win"], df[feat])
    pb_results[feat] = {"r": r, "p": p}
print("\nPoint-biserial correlations with win:")
for feat, res in pb_results.items():
    print(f"{feat}: r={fmt(res['r'])}, p={fmt(res['p'])}")

# 4) Statistical tests
wins = df[df["win"] == 1]
losses = df[df["win"] == 0]

# Welch t-tests
test_rel = stats.ttest_ind(wins["rel_group_size"], losses["rel_group_size"], equal_var=False)
test_loc = stats.ttest_ind(wins["location_advantage"], losses["location_advantage"], equal_var=False)

print("\nWelch t-tests (wins vs losses):")
print(f"rel_group_size: t={fmt(test_rel.statistic)}, p={fmt(test_rel.pvalue)}")
print(f"location_advantage: t={fmt(test_loc.statistic)}, p={fmt(test_loc.pvalue)}")

# ANOVA by relative size category (smaller/equal/larger)
size_cat = pd.cut(
    df["rel_group_size"],
    bins=[-np.inf, -0.5, 0.5, np.inf],
    labels=["focal_smaller", "equal", "focal_larger"],
)
size_groups = [df.loc[size_cat == c, "win"].values for c in size_cat.cat.categories]
size_groups = [g for g in size_groups if len(g) > 1]
if len(size_groups) >= 2:
    anova_size = stats.f_oneway(*size_groups)
else:
    anova_size = None

# ANOVA by location-advantage tertiles
loc_bins = pd.qcut(df["location_advantage"], q=3, labels=["low", "mid", "high"], duplicates="drop")
loc_groups = [df.loc[loc_bins == c, "win"].values for c in loc_bins.cat.categories]
loc_groups = [g for g in loc_groups if len(g) > 1]
if len(loc_groups) >= 2:
    anova_loc = stats.f_oneway(*loc_groups)
else:
    anova_loc = None

print("\nANOVA results:")
if anova_size is not None:
    print(f"win ~ size_category: F={fmt(anova_size.statistic)}, p={fmt(anova_size.pvalue)}")
else:
    print("win ~ size_category: insufficient data")
if anova_loc is not None:
    print(f"win ~ location_advantage_tertile: F={fmt(anova_loc.statistic)}, p={fmt(anova_loc.pvalue)}")
else:
    print("win ~ location_advantage_tertile: insufficient data")

# 5) Interpretable regression/classification models
primary_features = ["rel_group_size", "location_advantage"]
all_features = ["rel_group_size", "location_advantage", "rel_male", "rel_female"]

X_primary = df[primary_features]
X_all = df[all_features]
y = df["win"].astype(float)

# statsmodels OLS (requested) for inferential statistics
X_ols = sm.add_constant(X_primary)
ols_model = sm.OLS(y, X_ols).fit()
print("\nOLS summary (win as probability approximation):")
print(ols_model.summary())

# statsmodels logistic regression (better for binary outcome)
logit_model = None
try:
    logit_model = sm.Logit(y, X_ols).fit(disp=False)
    print("\nLogit coefficients:")
    print(logit_model.params)
    print("Logit p-values:")
    print(logit_model.pvalues)
except Exception as e:
    print("\nLogit model failed:", e)

# scikit-learn linear models
lin = LinearRegression().fit(X_all, y)
ridge = Ridge(alpha=1.0).fit(X_all, y)
lasso = Lasso(alpha=0.01, max_iter=10000).fit(X_all, y)

print("\nLinear model coefficients (unscaled features):")
for name, model in [("LinearRegression", lin), ("Ridge", ridge), ("Lasso", lasso)]:
    print(name)
    for f, c in zip(all_features, model.coef_):
        print(f"  {f}: {fmt(c)}")

# Also show standardized coefficients for direct effect-size comparability
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)
lin_scaled = LinearRegression().fit(X_scaled, y)
print("\nStandardized LinearRegression coefficients:")
for f, c in zip(all_features, lin_scaled.coef_):
    print(f"  {f}: {fmt(c)}")

# Interpretable trees
dtr = DecisionTreeRegressor(max_depth=3, random_state=0).fit(X_all, y)
dtc = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X_all, y.astype(int))

print("\nDecisionTreeRegressor feature importances:")
for f, imp in zip(all_features, dtr.feature_importances_):
    print(f"  {f}: {fmt(imp)}")

print("\nDecisionTreeClassifier feature importances:")
for f, imp in zip(all_features, dtc.feature_importances_):
    print(f"  {f}: {fmt(imp)}")

print("\nDecisionTreeClassifier rules:")
print(export_text(dtc, feature_names=all_features, max_depth=3))

# imodels interpretable models
imodels_results = {}
try:
    from imodels import FIGSRegressor, HSTreeRegressor, RuleFitRegressor

    rf = RuleFitRegressor()
    rf.fit(X_all.values, y.values, feature_names=all_features)
    rf_score = safe_num(rf.score(X_all.values, y.values), default=np.nan)
    imodels_results["RuleFitRegressor_r2"] = rf_score

    rules_preview = ""
    if hasattr(rf, "get_rules"):
        try:
            rules_df = rf.get_rules()
            if isinstance(rules_df, pd.DataFrame) and not rules_df.empty:
                rules_df = rules_df.copy()
                if "coef" in rules_df.columns:
                    rules_df = rules_df.loc[rules_df["coef"].abs() > 1e-8]
                    rules_df = rules_df.reindex(rules_df["coef"].abs().sort_values(ascending=False).index)
                rules_preview = rules_df.head(5).to_string(index=False)
        except Exception:
            rules_preview = ""

    figs = FIGSRegressor()
    figs.fit(X_all.values, y.values, feature_names=all_features)
    figs_score = safe_num(figs.score(X_all.values, y.values), default=np.nan)
    imodels_results["FIGSRegressor_r2"] = figs_score

    hs = HSTreeRegressor()
    hs.fit(X_all.values, y.values, feature_names=all_features)
    hs_score = safe_num(hs.score(X_all.values, y.values), default=np.nan)
    imodels_results["HSTreeRegressor_r2"] = hs_score

    print("\nimodels fit metrics:")
    for k, v in imodels_results.items():
        print(f"  {k}: {fmt(v)}")

    if rules_preview:
        print("\nTop RuleFit rules:")
        print(rules_preview)

    # Optional feature importance extraction when available
    for model_name, model in [("FIGS", figs), ("HSTree", hs)]:
        if hasattr(model, "feature_importances_"):
            print(f"\n{model_name} feature importances:")
            for f, imp in zip(all_features, model.feature_importances_):
                print(f"  {f}: {fmt(imp)}")

except Exception as e:
    print("\nimodels not available or failed:", e)

# 6) Build final conclusion score (0-100 Likert)
coef_rel = safe_num(ols_model.params.get("rel_group_size", np.nan))
coef_loc = safe_num(ols_model.params.get("location_advantage", np.nan))
p_rel = safe_num(ols_model.pvalues.get("rel_group_size", np.nan))
p_loc = safe_num(ols_model.pvalues.get("location_advantage", np.nan))

# Evidence synthesis: significance is the primary criterion.
# Per instructions, non-significant relationships should map to a low score.
sig_rel = np.isfinite(p_rel) and p_rel < 0.05
sig_loc = np.isfinite(p_loc) and p_loc < 0.05
dir_rel = np.isfinite(coef_rel) and coef_rel > 0
dir_loc = np.isfinite(coef_loc) and coef_loc > 0

score = 10

if sig_rel and dir_rel:
    score += 35
elif sig_rel:
    score += 15
elif dir_rel:
    score += 8
else:
    score += 2

if sig_loc and dir_loc:
    score += 35
elif sig_loc:
    score += 15
elif dir_loc:
    score += 8
else:
    score += 2

# Add only small incremental evidence from auxiliary tests/models.
if test_rel.pvalue < 0.05 and wins["rel_group_size"].mean() > losses["rel_group_size"].mean():
    score += 5
if test_loc.pvalue < 0.05 and wins["location_advantage"].mean() > losses["location_advantage"].mean():
    score += 5

if logit_model is not None:
    lp = logit_model.pvalues
    lc = logit_model.params
    if "rel_group_size" in lp.index and lp["rel_group_size"] < 0.05 and lc["rel_group_size"] > 0:
        score += 4
    if "location_advantage" in lp.index and lp["location_advantage"] < 0.05 and lc["location_advantage"] > 0:
        score += 4

score = int(np.clip(round(score), 0, 100))

if sig_rel or sig_loc:
    interpretation = (
        "There is at least partial statistically significant evidence that relative group size and/or "
        "contest location influence win probability."
    )
else:
    interpretation = (
        "The tested relationships were not statistically significant, so evidence is weak for a clear "
        "influence in this sample."
    )

explanation = (
    f"Using 58 contests, OLS gave rel_group_size={fmt(coef_rel,3)} (p={fmt(p_rel,4)}) and "
    f"location_advantage={fmt(coef_loc,4)} (p={fmt(p_loc,4)}). "
    f"Welch t-tests were p={fmt(test_rel.pvalue,4)} for relative group size and p={fmt(test_loc.pvalue,4)} "
    f"for location advantage. "
    f"{interpretation} "
    f"Interpretable models (decision trees and imodels) suggested possible directional patterns, but "
    f"significance-based inference remains limited."
)

result = {"response": score, "explanation": explanation}
Path("conclusion.txt").write_text(json.dumps(result))

print("\nFinal conclusion JSON written to conclusion.txt:")
print(json.dumps(result, indent=2))
