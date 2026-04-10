import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from imodels import FIGSRegressor, HSTreeRegressor, RuleFitRegressor


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def extract_child_coef(feature_names, coef_array):
    coef_map = dict(zip(feature_names, coef_array))
    for name, coef in coef_map.items():
        if "children_yes" in name:
            return safe_float(coef)
    return np.nan


def top_feature_importances(feature_names, importances, top_k=5):
    pairs = list(zip(feature_names, importances))
    pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
    return pairs[:top_k]


def main():
    info_path = Path("info.json")
    data_path = Path("affairs.csv")
    out_path = Path("conclusion.txt")

    info = json.loads(info_path.read_text())
    research_question = info.get("research_questions", ["Unknown question"])[0]

    df = pd.read_csv(data_path)

    # Basic setup for analysis
    df["has_affair"] = (df["affairs"] > 0).astype(int)
    df["children_yes"] = (df["children"].astype(str).str.lower() == "yes").astype(int)
    df["gender_male"] = (df["gender"].astype(str).str.lower() == "male").astype(int)

    print("Research question:", research_question)
    print("\nData shape:", df.shape)
    print("Columns:", list(df.columns))

    # 1) EDA: summary statistics, distributions, correlations
    print("\n=== Summary statistics ===")
    print(df.describe(include="all").transpose())

    print("\n=== Affairs distribution (counts) ===")
    print(df["affairs"].value_counts(dropna=False).sort_index())

    print("\n=== Affairs distribution (proportions) ===")
    print(df["affairs"].value_counts(normalize=True, dropna=False).sort_index())

    group_stats = (
        df.groupby("children", observed=False)["affairs"]
        .agg(["count", "mean", "median", "std"])
        .sort_index()
    )
    print("\n=== Group stats by children ===")
    print(group_stats)

    corr_df = df[[
        "affairs",
        "children_yes",
        "gender_male",
        "age",
        "yearsmarried",
        "religiousness",
        "education",
        "occupation",
        "rating",
    ]].copy()
    corr = corr_df.corr(numeric_only=True)
    print("\n=== Correlation with affairs ===")
    print(corr["affairs"].sort_values(ascending=False))

    # 2) Statistical tests focused on children -> affairs relationship
    yes = df.loc[df["children_yes"] == 1, "affairs"]
    no = df.loc[df["children_yes"] == 0, "affairs"]

    t_res = stats.ttest_ind(yes, no, equal_var=False)
    mw_res = stats.mannwhitneyu(yes, no, alternative="two-sided")
    f_res = stats.f_oneway(yes, no)

    contingency = pd.crosstab(df["children"], df["has_affair"])
    chi2_res = stats.chi2_contingency(contingency)

    print("\n=== Statistical tests ===")
    print("Welch t-test (mean affairs children yes vs no):", t_res)
    print("Mann-Whitney U:", mw_res)
    print("One-way ANOVA:", f_res)
    print("Chi-square (children vs any affair):")
    print("contingency table:\n", contingency)
    print("chi2, pvalue:", chi2_res[0], chi2_res[1])

    # OLS with controls for confounding
    ols_formula = (
        "affairs ~ C(children) + age + yearsmarried + religiousness + "
        "education + occupation + rating + C(gender)"
    )
    ols_model = smf.ols(ols_formula, data=df).fit()
    ols_hc3 = ols_model.get_robustcov_results(cov_type="HC3")

    coef_key = "C(children)[T.yes]"
    coef_idx = list(ols_model.params.index).index(coef_key)
    ols_child_coef = safe_float(ols_hc3.params[coef_idx])
    ols_child_p = safe_float(ols_hc3.pvalues[coef_idx])
    ols_ci_low, ols_ci_high = [safe_float(x) for x in ols_hc3.conf_int()[coef_idx]]

    print("\n=== OLS (HC3 robust) ===")
    print("children coef:", ols_child_coef)
    print("children p-value:", ols_child_p)
    print("children 95% CI:", (ols_ci_low, ols_ci_high))

    # 3) Interpretable models (sklearn + imodels)
    X = df.drop(columns=["affairs", "has_affair", "children_yes", "gender_male", "rownames"])
    y_reg = df["affairs"].values
    y_clf = df["has_affair"].values

    cat_cols = ["gender", "children"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                cat_cols,
            ),
            ("num", "passthrough", num_cols),
        ]
    )

    X_enc = preprocessor.fit_transform(X)
    feature_names = list(preprocessor.get_feature_names_out())

    lin = LinearRegression().fit(X_enc, y_reg)
    ridge = Ridge(alpha=1.0, random_state=0).fit(X_enc, y_reg)
    lasso = Lasso(alpha=0.01, max_iter=20000, random_state=0).fit(X_enc, y_reg)

    lin_child_coef = extract_child_coef(feature_names, lin.coef_)
    ridge_child_coef = extract_child_coef(feature_names, ridge.coef_)
    lasso_child_coef = extract_child_coef(feature_names, lasso.coef_)

    tree_reg = DecisionTreeRegressor(max_depth=3, random_state=0)
    tree_reg.fit(X_enc, y_reg)
    tree_reg_top = top_feature_importances(feature_names, tree_reg.feature_importances_, top_k=5)

    tree_clf = DecisionTreeClassifier(max_depth=3, random_state=0)
    tree_clf.fit(X_enc, y_clf)
    tree_clf_top = top_feature_importances(feature_names, tree_clf.feature_importances_, top_k=5)

    print("\n=== Interpretable sklearn models ===")
    print("Linear child coef:", lin_child_coef)
    print("Ridge child coef:", ridge_child_coef)
    print("Lasso child coef:", lasso_child_coef)
    print("DecisionTreeRegressor top importances:", tree_reg_top)
    print("DecisionTreeClassifier top importances:", tree_clf_top)

    # imodels
    rulefit_child_rule_count = 0
    rulefit_child_rule_coef_mean = np.nan
    rulefit_child_rule_coef_max_abs = np.nan
    figs_child_importance = np.nan
    hstree_uses_child = False

    print("\n=== imodels ===")

    try:
        rulefit = RuleFitRegressor(random_state=0, n_estimators=200, tree_size=4)
        rulefit.fit(X_enc, y_reg, feature_names=feature_names)
        if hasattr(rulefit, "get_rules"):
            rules = rulefit.get_rules()
        else:
            rules = rulefit._get_rules(exclude_zero_coef=True)

        if "rule" in rules.columns and "coef" in rules.columns:
            child_rules = rules[rules["rule"].astype(str).str.contains("children", case=False, na=False)]
            rulefit_child_rule_count = int(child_rules.shape[0])
            if rulefit_child_rule_count > 0:
                rulefit_child_rule_coef_mean = safe_float(child_rules["coef"].mean())
                rulefit_child_rule_coef_max_abs = safe_float(child_rules["coef"].abs().max())

        print("RuleFit child-related rule count:", rulefit_child_rule_count)
        print("RuleFit child rule mean coef:", rulefit_child_rule_coef_mean)
        print("RuleFit child rule max |coef|:", rulefit_child_rule_coef_max_abs)
    except Exception as exc:
        print("RuleFitRegressor failed:", repr(exc))

    try:
        figs = FIGSRegressor(random_state=0, max_rules=12)
        figs.fit(X_enc, y_reg, feature_names=feature_names)
        figs_importances = getattr(figs, "feature_importances_", None)
        if figs_importances is not None:
            figs_map = dict(zip(feature_names, figs_importances))
            for k, v in figs_map.items():
                if "children_yes" in k:
                    figs_child_importance = safe_float(v)
                    break
        print("FIGS child importance:", figs_child_importance)
    except Exception as exc:
        print("FIGSRegressor failed:", repr(exc))

    try:
        hstree = HSTreeRegressor(random_state=0, max_leaf_nodes=8)
        hstree.fit(X_enc, y_reg)
        tree_text = str(hstree)
        # In this encoding, X1 corresponds to cat__children_yes
        hstree_uses_child = ("X1" in tree_text)
        print("HSTree uses children split (X1 present):", hstree_uses_child)
    except Exception as exc:
        print("HSTreeRegressor failed:", repr(exc))

    # 4) Convert evidence into Likert score for the question
    mean_yes = safe_float(yes.mean())
    mean_no = safe_float(no.mean())
    affair_rate_yes = safe_float(df.loc[df["children_yes"] == 1, "has_affair"].mean())
    affair_rate_no = safe_float(df.loc[df["children_yes"] == 0, "has_affair"].mean())

    p_t = safe_float(t_res.pvalue)
    p_mw = safe_float(mw_res.pvalue)
    p_chi2 = safe_float(chi2_res[1])

    score = 50

    # Non-significant adjusted relationship should push toward "No"
    if ols_child_p < 0.05:
        if ols_child_coef < 0:
            score += 25
        else:
            score -= 25
    else:
        score -= 20

    # Unadjusted differences (direction + significance)
    if p_t < 0.05:
        if mean_yes < mean_no:
            score += 20
        else:
            score -= 20

    if p_mw < 0.05:
        if mean_yes < mean_no:
            score += 10
        else:
            score -= 10

    if p_chi2 < 0.05:
        if affair_rate_yes < affair_rate_no:
            score += 15
        else:
            score -= 15

    # Weak model-based directional adjustment
    child_coefs = [lin_child_coef, ridge_child_coef, lasso_child_coef]
    child_coefs = [c for c in child_coefs if np.isfinite(c)]
    if child_coefs:
        avg_child_coef = float(np.mean(child_coefs))
        if avg_child_coef < 0:
            score += 5
        elif avg_child_coef > 0:
            score -= 5
    else:
        avg_child_coef = np.nan

    if np.isfinite(figs_child_importance) and figs_child_importance < 0.01:
        score -= 3

    if rulefit_child_rule_count == 0:
        score -= 2

    score = int(np.clip(round(score), 0, 100))

    explanation = (
        f"Question: {research_question.strip()} Unadjusted evidence does not support a decrease: "
        f"mean affairs is higher with children ({mean_yes:.3f}) than without ({mean_no:.3f}); "
        f"Welch t-test p={p_t:.4g}, Mann-Whitney p={p_mw:.4g}, and chi-square on any affair p={p_chi2:.4g} "
        f"all indicate significant differences in the opposite direction (higher reported affairs when children=yes). "
        f"In adjusted OLS, the children coefficient is negative ({ols_child_coef:.3f}) but not significant "
        f"(p={ols_child_p:.4g}, 95% CI [{ols_ci_low:.3f}, {ols_ci_high:.3f}]). "
        f"Interpretable linear models give small negative child coefficients (avg {avg_child_coef:.3f}), "
        f"while tree/rule models do not show children as a dominant driver. Overall, evidence for a true "
        f"decrease is weak to absent."
    )

    result = {"response": score, "explanation": explanation}
    out_path.write_text(json.dumps(result, ensure_ascii=True))

    print("\nWrote conclusion.txt with response:", score)


if __name__ == "__main__":
    main()
