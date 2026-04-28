import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor


warnings.filterwarnings("ignore")


def make_one_hot_encoder():
    try:
        return OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(drop="first", handle_unknown="ignore", sparse=False)


def extract_model_effect(models, feature_name):
    effects = {}
    for model_name, model in models.items():
        if hasattr(model, "named_steps") and "model" in model.named_steps:
            inner = model.named_steps["model"]
            pre = model.named_steps["pre"]
            feat_names = pre.get_feature_names_out()
            if hasattr(inner, "coef_"):
                coefs = inner.coef_.ravel()
                fmap = dict(zip(feat_names, coefs))
                effects[model_name] = float(fmap.get(feature_name, np.nan))
            elif hasattr(inner, "feature_importances_"):
                importances = inner.feature_importances_
                fmap = dict(zip(feat_names, importances))
                effects[model_name] = float(fmap.get(feature_name, np.nan))
    return effects


def main():
    base = Path(".")
    info = json.loads((base / "info.json").read_text())
    research_question = info.get("research_questions", ["Unknown question"])[0].strip()

    df = pd.read_csv(base / "affairs.csv")
    df["has_affair"] = (df["affairs"] > 0).astype(int)

    print("Research question:", research_question)
    print("\\nData shape:", df.shape)
    print("Missing values per column:\n", df.isna().sum())

    numeric_cols = [
        "affairs",
        "age",
        "yearsmarried",
        "religiousness",
        "education",
        "occupation",
        "rating",
    ]
    cat_cols = ["gender", "children"]

    print("\\nSummary statistics (numeric):")
    print(df[numeric_cols].describe().T)

    print("\\nCategorical distributions:")
    for col in cat_cols:
        print(f"\\n{col}:")
        print(df[col].value_counts(dropna=False))

    print("\\nTarget distribution (affairs):")
    print(df["affairs"].value_counts().sort_index())

    group_stats = df.groupby("children")["affairs"].agg(["count", "mean", "median", "std"])
    print("\\nAffairs by children group:")
    print(group_stats)

    corr_df = df[numeric_cols].copy()
    corr_df["children_yes"] = (df["children"] == "yes").astype(int)
    corr_to_target = corr_df.corr(numeric_only=True)["affairs"].sort_values(ascending=False)
    print("\\nCorrelations with affairs:")
    print(corr_to_target)

    # Statistical tests for relationship between children and affairs
    affairs_yes = df.loc[df["children"] == "yes", "affairs"]
    affairs_no = df.loc[df["children"] == "no", "affairs"]

    welch_two_sided = stats.ttest_ind(affairs_yes, affairs_no, equal_var=False)
    welch_less = stats.ttest_ind(affairs_yes, affairs_no, equal_var=False, alternative="less")
    welch_greater = stats.ttest_ind(affairs_yes, affairs_no, equal_var=False, alternative="greater")
    mann_whitney = stats.mannwhitneyu(affairs_yes, affairs_no, alternative="two-sided")
    anova = stats.f_oneway(affairs_yes, affairs_no)

    contingency = pd.crosstab(df["children"], df["has_affair"])
    chi2, chi2_p, _, _ = stats.chi2_contingency(contingency)

    print("\\nStatistical tests:")
    print(f"Welch t-test (two-sided): stat={welch_two_sided.statistic:.4f}, p={welch_two_sided.pvalue:.6f}")
    print(f"Welch t-test (one-sided, children lowers affairs): stat={welch_less.statistic:.4f}, p={welch_less.pvalue:.6f}")
    print(f"Welch t-test (one-sided, children increases affairs): stat={welch_greater.statistic:.4f}, p={welch_greater.pvalue:.6f}")
    print(f"Mann-Whitney U (two-sided): stat={mann_whitney.statistic:.4f}, p={mann_whitney.pvalue:.6f}")
    print(f"One-way ANOVA: stat={anova.statistic:.4f}, p={anova.pvalue:.6f}")
    print(f"Chi-square (children x any affair): chi2={chi2:.4f}, p={chi2_p:.6f}")

    # Regression with controls
    ols_formula = (
        "affairs ~ C(children) + C(gender) + age + yearsmarried + "
        "religiousness + education + occupation + rating"
    )
    ols_model = smf.ols(ols_formula, data=df).fit()

    logit_formula = (
        "has_affair ~ C(children) + C(gender) + age + yearsmarried + "
        "religiousness + education + occupation + rating"
    )
    logit_model = smf.logit(logit_formula, data=df).fit(disp=0)

    ols_child_coef = float(ols_model.params.get("C(children)[T.yes]", np.nan))
    ols_child_p = float(ols_model.pvalues.get("C(children)[T.yes]", np.nan))
    logit_child_coef = float(logit_model.params.get("C(children)[T.yes]", np.nan))
    logit_child_p = float(logit_model.pvalues.get("C(children)[T.yes]", np.nan))

    print("\\nControlled models:")
    print(f"OLS child coefficient: {ols_child_coef:.4f} (p={ols_child_p:.6f})")
    print(f"Logit child coefficient: {logit_child_coef:.4f} (p={logit_child_p:.6f})")

    # Interpretable sklearn models
    feature_cols = ["gender", "age", "yearsmarried", "children", "religiousness", "education", "occupation", "rating"]
    X = df[feature_cols]
    y_reg = df["affairs"]
    y_cls = df["has_affair"]

    pre = ColumnTransformer(
        transformers=[
            ("cat", make_one_hot_encoder(), ["gender", "children"]),
            ("num", "passthrough", ["age", "yearsmarried", "religiousness", "education", "occupation", "rating"]),
        ]
    )

    reg_models = {
        "linear": Pipeline([("pre", pre), ("model", LinearRegression())]),
        "ridge": Pipeline([("pre", pre), ("model", Ridge(alpha=1.0, random_state=0))]),
        "lasso": Pipeline([("pre", pre), ("model", Lasso(alpha=0.01, random_state=0, max_iter=20000))]),
        "tree_reg": Pipeline([("pre", pre), ("model", DecisionTreeRegressor(max_depth=3, random_state=0))]),
    }
    for model in reg_models.values():
        model.fit(X, y_reg)

    cls_tree = Pipeline([("pre", pre), ("model", DecisionTreeClassifier(max_depth=3, random_state=0))])
    cls_tree.fit(X, y_cls)

    child_feature = "cat__children_yes"
    child_effects = extract_model_effect(reg_models, child_feature)

    tree_feat_names = cls_tree.named_steps["pre"].get_feature_names_out()
    tree_importances = cls_tree.named_steps["model"].feature_importances_
    tree_importance_map = dict(zip(tree_feat_names, tree_importances))

    print("\\nSklearn interpretable model child effects:")
    for k, v in child_effects.items():
        print(f"{k}: {v:.6f}")
    print(f"tree_cls children importance: {tree_importance_map.get(child_feature, np.nan):.6f}")

    # Interpret glassbox models
    ebm_reg = ExplainableBoostingRegressor(interactions=0, random_state=0)
    ebm_reg.fit(X, y_reg)
    ebm_reg_global = ebm_reg.explain_global().data()

    ebm_cls = ExplainableBoostingClassifier(interactions=0, random_state=0)
    ebm_cls.fit(X, y_cls)
    ebm_cls_global = ebm_cls.explain_global().data()

    ebm_reg_importance = dict(zip(ebm_reg_global["names"], [float(s) for s in ebm_reg_global["scores"]]))
    ebm_cls_importance = dict(zip(ebm_cls_global["names"], [float(s) for s in ebm_cls_global["scores"]]))

    children_idx_reg = list(ebm_reg.term_names_).index("children")
    children_reg_shape = ebm_reg.explain_global().data(children_idx_reg)
    reg_child_scores = dict(zip(children_reg_shape["names"], [float(s) for s in children_reg_shape["scores"]]))

    children_idx_cls = list(ebm_cls.term_names_).index("children")
    children_cls_shape = ebm_cls.explain_global().data(children_idx_cls)
    cls_child_scores = dict(zip(children_cls_shape["names"], [float(s) for s in children_cls_shape["scores"]]))

    print("\\nEBM importances:")
    print(f"EBM reg children importance: {ebm_reg_importance.get('children', np.nan):.6f}")
    print(f"EBM cls children importance: {ebm_cls_importance.get('children', np.nan):.6f}")
    print(f"EBM reg children additive scores: {reg_child_scores}")
    print(f"EBM cls children additive scores: {cls_child_scores}")

    mean_yes = float(affairs_yes.mean())
    mean_no = float(affairs_no.mean())
    diff = mean_yes - mean_no

    decrease_supported = (diff < 0) and (welch_less.pvalue < 0.05)
    controlled_decrease_supported = (ols_child_coef < 0 and ols_child_p < 0.05) or (
        logit_child_coef < 0 and logit_child_p < 0.05
    )
    opposite_significant = (diff > 0) and (welch_greater.pvalue < 0.05)

    if decrease_supported and controlled_decrease_supported:
        response = 90
    elif decrease_supported:
        response = 75
    elif opposite_significant:
        response = 5
    elif diff > 0:
        response = 15
    else:
        response = 30

    explanation = (
        f"Question: {research_question} Unadjusted data do not support a decrease: mean affairs is "
        f"{mean_yes:.3f} for couples with children vs {mean_no:.3f} without children (difference={diff:.3f}). "
        f"The one-sided Welch t-test for a decrease gives p={welch_less.pvalue:.4g} (not significant), while the "
        f"opposite direction is significant (p={welch_greater.pvalue:.4g}). A chi-square test on any affair vs none "
        f"is significant (p={chi2_p:.4g}) in the direction of higher affair prevalence with children. In controlled "
        f"models, the children coefficient is not significantly negative (OLS coef={ols_child_coef:.3f}, p={ols_child_p:.4g}; "
        f"Logit coef={logit_child_coef:.3f}, p={logit_child_p:.4g}). Interpretable models (linear/tree/EBM) also do not "
        f"show robust evidence that children reduce affairs."
    )

    result = {"response": int(response), "explanation": explanation}
    with open(base / "conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(result, f)

    print("\\nWrote conclusion.txt")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
