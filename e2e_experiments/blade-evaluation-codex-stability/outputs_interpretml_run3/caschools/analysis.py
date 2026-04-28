import json
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from interpret.glassbox import ExplainableBoostingRegressor

warnings.filterwarnings("ignore")


def main() -> None:
    # Load metadata and data
    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    question = info.get("research_questions", ["Unknown question"])[0]
    df = pd.read_csv("caschools.csv")

    # Feature engineering
    df["avg_score"] = (df["read"] + df["math"]) / 2.0
    df["str"] = df["students"] / df["teachers"]
    df["computer_per_student"] = df["computer"] / df["students"].replace(0, np.nan)

    # Data exploration
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    missing_counts = df.isna().sum().to_dict()
    summary_stats = df[numeric_cols].describe().T
    skewness = df[numeric_cols].skew(numeric_only=True).sort_values(ascending=False)
    corr_to_target = (
        df[numeric_cols]
        .corr(numeric_only=True)["avg_score"]
        .drop("avg_score")
        .sort_values(ascending=False)
    )

    print("Research question:", question)
    print("\nRows, Columns:", df.shape)
    print("\nMissing values by column:")
    print(pd.Series(missing_counts)[pd.Series(missing_counts) > 0])
    print("\nSummary statistics (selected columns):")
    print(summary_stats.loc[["avg_score", "str", "income", "english", "lunch", "expenditure"]])
    print("\nTop skewed numeric variables:")
    print(skewness.head(8))
    print("\nCorrelation with avg_score:")
    print(corr_to_target.head(10))
    print(corr_to_target.tail(10))

    # Statistical tests directly tied to the research question
    valid = df[["str", "avg_score"]].dropna()
    pearson_r, pearson_p = stats.pearsonr(valid["str"], valid["avg_score"])
    spearman_rho, spearman_p = stats.spearmanr(valid["str"], valid["avg_score"])

    median_str = valid["str"].median()
    low_str = valid.loc[valid["str"] <= median_str, "avg_score"]
    high_str = valid.loc[valid["str"] > median_str, "avg_score"]
    t_stat, t_p = stats.ttest_ind(low_str, high_str, equal_var=False)

    # ANOVA across student-teacher ratio tertiles
    tertiles = pd.qcut(valid["str"], q=3, labels=["low", "mid", "high"])
    anova_df = valid.assign(str_group=tertiles)
    g_low = anova_df.loc[anova_df["str_group"] == "low", "avg_score"]
    g_mid = anova_df.loc[anova_df["str_group"] == "mid", "avg_score"]
    g_high = anova_df.loc[anova_df["str_group"] == "high", "avg_score"]
    anova_f, anova_p = stats.f_oneway(g_low, g_mid, g_high)
    tertile_means = {
        "low": float(g_low.mean()),
        "mid": float(g_mid.mean()),
        "high": float(g_high.mean()),
    }

    # OLS models with and without controls
    model_df = df[
        [
            "avg_score",
            "str",
            "calworks",
            "lunch",
            "english",
            "income",
            "expenditure",
            "computer_per_student",
            "students",
        ]
    ].dropna()

    X_simple = sm.add_constant(model_df[["str"]])
    y = model_df["avg_score"]
    ols_simple = sm.OLS(y, X_simple).fit()

    X_controls = sm.add_constant(
        model_df[
            [
                "str",
                "calworks",
                "lunch",
                "english",
                "income",
                "expenditure",
                "computer_per_student",
                "students",
            ]
        ]
    )
    ols_controls = sm.OLS(y, X_controls).fit()

    # Interpretable machine learning models
    feature_cols = [
        "str",
        "calworks",
        "lunch",
        "english",
        "income",
        "expenditure",
        "computer_per_student",
        "students",
    ]
    X = model_df[feature_cols]
    y = model_df["avg_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    lin = LinearRegression().fit(X_train, y_train)
    ridge = Ridge(alpha=1.0, random_state=42).fit(X_train, y_train)
    lasso = Lasso(alpha=0.05, random_state=42, max_iter=10000).fit(X_train, y_train)
    tree = DecisionTreeRegressor(max_depth=3, random_state=42).fit(X_train, y_train)
    ebm = ExplainableBoostingRegressor(random_state=42, interactions=0)
    ebm.fit(X_train, y_train)

    model_results = {
        "LinearRegression": r2_score(y_test, lin.predict(X_test)),
        "Ridge": r2_score(y_test, ridge.predict(X_test)),
        "Lasso": r2_score(y_test, lasso.predict(X_test)),
        "DecisionTree": r2_score(y_test, tree.predict(X_test)),
        "EBM": r2_score(y_test, ebm.predict(X_test)),
    }

    lin_coef = dict(zip(feature_cols, lin.coef_))
    ridge_coef = dict(zip(feature_cols, ridge.coef_))
    lasso_coef = dict(zip(feature_cols, lasso.coef_))
    tree_importances = dict(zip(feature_cols, tree.feature_importances_))

    ebm_terms = list(ebm.term_names_)
    ebm_scores = list(ebm.term_importances())
    ebm_importances = dict(zip(ebm_terms, ebm_scores))
    ebm_str_importance = float(ebm_importances.get("str", 0.0))

    print("\nStatistical tests:")
    print(f"Pearson r(str, avg_score) = {pearson_r:.4f}, p = {pearson_p:.4g}")
    print(f"Spearman rho(str, avg_score) = {spearman_rho:.4f}, p = {spearman_p:.4g}")
    print(
        f"T-test avg_score low-STR vs high-STR: t = {t_stat:.4f}, p = {t_p:.4g}, "
        f"mean_low = {low_str.mean():.2f}, mean_high = {high_str.mean():.2f}"
    )
    print(f"ANOVA across STR tertiles: F = {anova_f:.4f}, p = {anova_p:.4g}")
    print("STR tertile means:", tertile_means)

    print("\nOLS (simple):")
    print(ols_simple.summary().tables[1])
    print("\nOLS (with controls):")
    print(ols_controls.summary().tables[1])

    print("\nModel test R^2:")
    for k, v in model_results.items():
        print(f"{k}: {v:.4f}")

    print("\nKey interpretable model effects for STR:")
    print(f"LinearRegression coef[str] = {lin_coef['str']:.4f}")
    print(f"Ridge coef[str] = {ridge_coef['str']:.4f}")
    print(f"Lasso coef[str] = {lasso_coef['str']:.4f}")
    print(f"DecisionTree importance[str] = {tree_importances['str']:.4f}")
    print(f"EBM importance[str] = {ebm_str_importance:.4f}")

    # Score synthesis for the required Likert-scale response
    str_coef = float(ols_controls.params["str"])
    str_p = float(ols_controls.pvalues["str"])

    if str_coef < 0 and str_p < 0.001:
        score = 95
    elif str_coef < 0 and str_p < 0.01:
        score = 90
    elif str_coef < 0 and str_p < 0.05:
        score = 82
    elif str_coef < 0 and str_p < 0.10:
        score = 68
    elif str_coef < 0:
        score = 45
    elif str_coef > 0 and str_p < 0.05:
        score = 15
    elif str_coef > 0:
        score = 30
    else:
        score = 50

    consistency = 0.0
    if pearson_p < 0.05:
        consistency += 1.0 if pearson_r < 0 else -1.0
    if t_p < 0.05:
        consistency += 1.0 if low_str.mean() > high_str.mean() else -1.0
    if anova_p < 0.05:
        if tertile_means["low"] > tertile_means["mid"] > tertile_means["high"]:
            consistency += 1.0
        elif tertile_means["low"] < tertile_means["mid"] < tertile_means["high"]:
            consistency -= 1.0

    consistency += 0.5 if lin_coef["str"] < 0 else -0.5
    consistency += 0.5 if ridge_coef["str"] < 0 else -0.5
    consistency += 0.5 if lasso_coef["str"] < 0 else -0.5

    tree_rank = sorted(tree_importances.items(), key=lambda kv: kv[1], reverse=True)
    top3_tree = [name for name, _ in tree_rank[:3]]
    consistency += 0.5 if "str" in top3_tree else 0.0

    score = int(np.clip(round(score + 4 * consistency), 0, 100))

    explanation = (
        f"Question: {question} "
        f"Controls-adjusted OLS estimates STR coefficient {str_coef:.3f} (p={str_p:.3g}), "
        f"indicating that higher student-teacher ratios are associated with lower average test scores when controlling for demographics/resources. "
        f"Bivariate evidence is consistent: Pearson r={pearson_r:.3f} (p={pearson_p:.3g}), "
        f"median-split t-test p={t_p:.3g} with higher scores in lower-STR districts, and ANOVA p={anova_p:.3g} across STR tertiles. "
        f"Interpretable models align: linear/ridge/lasso coefficients on STR are "
        f"{lin_coef['str']:.3f}/{ridge_coef['str']:.3f}/{lasso_coef['str']:.3f}, "
        f"tree importance for STR={tree_importances['str']:.3f}, and EBM STR importance={ebm_str_importance:.3f}. "
        f"Overall evidence supports a meaningful negative association between student-teacher ratio and academic performance."
    )

    output = {"response": score, "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(output, f)


if __name__ == "__main__":
    main()
