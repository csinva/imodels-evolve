import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingRegressor
from scipy import stats
import statsmodels.formula.api as smf
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def safe_feature_names(preprocessor: ColumnTransformer) -> list:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        names = []
        for name, transformer, cols in preprocessor.transformers_:
            if name == "remainder":
                continue
            if hasattr(transformer, "get_feature_names_out"):
                try:
                    out = transformer.get_feature_names_out(cols)
                except Exception:
                    out = transformer.get_feature_names_out()
                names.extend([f"{name}__{x}" for x in out])
            else:
                names.extend([f"{name}__{c}" for c in cols])
        return names


def main() -> None:
    info_path = Path("info.json")
    data_path = Path("teachingratings.csv")

    info = json.loads(info_path.read_text())
    research_question = info.get("research_questions", ["Unknown research question"])[0]

    df = pd.read_csv(data_path)

    print_section("Research Question")
    print(research_question)

    print_section("Data Overview")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print("Missing values per column:")
    print(df.isna().sum().to_string())

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    print_section("Summary Statistics (Numeric)")
    print(df[numeric_cols].describe().T.to_string())

    print_section("Category Distributions")
    for c in categorical_cols:
        vc = df[c].value_counts(dropna=False)
        frac = (vc / len(df)).round(3)
        out = pd.DataFrame({"count": vc, "proportion": frac})
        print(f"\n{c}:")
        print(out.to_string())

    print_section("Correlations with Teaching Evaluation (eval)")
    corr_eval = df[numeric_cols].corr(numeric_only=True)["eval"].sort_values(ascending=False)
    print(corr_eval.to_string())

    beauty = df["beauty"]
    eval_scores = df["eval"]

    print_section("Statistical Tests for Beauty -> Evaluation")
    pearson_r, pearson_p = stats.pearsonr(beauty, eval_scores)
    spearman_rho, spearman_p = stats.spearmanr(beauty, eval_scores)
    print(f"Pearson correlation: r={pearson_r:.4f}, p={pearson_p:.4g}")
    print(f"Spearman correlation: rho={spearman_rho:.4f}, p={spearman_p:.4g}")

    median_beauty = beauty.median()
    low_group = eval_scores[beauty <= median_beauty]
    high_group = eval_scores[beauty > median_beauty]
    t_stat, t_p = stats.ttest_ind(high_group, low_group, equal_var=False)
    mean_diff = high_group.mean() - low_group.mean()
    print(
        "Median split t-test (high beauty vs low beauty): "
        f"t={t_stat:.4f}, p={t_p:.4g}, mean_diff={mean_diff:.4f}"
    )

    quartiles = pd.qcut(beauty, 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    groups = [eval_scores[quartiles == q] for q in quartiles.cat.categories]
    f_stat, anova_p = stats.f_oneway(*groups)
    print(f"ANOVA across beauty quartiles: F={f_stat:.4f}, p={anova_p:.4g}")

    ols_simple = smf.ols("eval ~ beauty", data=df).fit(cov_type="HC3")
    ols_adjusted = smf.ols(
        "eval ~ beauty + age + students + allstudents + "
        "C(minority) + C(gender) + C(credits) + C(division) + C(native) + C(tenure)",
        data=df,
    ).fit(cov_type="HC3")

    print("\nOLS (simple, HC3 robust):")
    print(
        f"beauty coef={ols_simple.params['beauty']:.4f}, "
        f"p={ols_simple.pvalues['beauty']:.4g}, "
        f"95% CI=[{ols_simple.conf_int().loc['beauty', 0]:.4f}, "
        f"{ols_simple.conf_int().loc['beauty', 1]:.4f}], "
        f"R^2={ols_simple.rsquared:.4f}"
    )

    print("\nOLS (adjusted controls, HC3 robust):")
    print(
        f"beauty coef={ols_adjusted.params['beauty']:.4f}, "
        f"p={ols_adjusted.pvalues['beauty']:.4g}, "
        f"95% CI=[{ols_adjusted.conf_int().loc['beauty', 0]:.4f}, "
        f"{ols_adjusted.conf_int().loc['beauty', 1]:.4f}], "
        f"R^2={ols_adjusted.rsquared:.4f}"
    )

    print_section("Interpretable Predictive Models")
    feature_cols = [
        "minority",
        "age",
        "gender",
        "credits",
        "beauty",
        "division",
        "native",
        "tenure",
        "students",
        "allstudents",
    ]

    X = df[feature_cols].copy()
    y = df["eval"].copy()

    num_features = ["age", "beauty", "students", "allstudents"]
    cat_features = [c for c in feature_cols if c not in num_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_features),
        ]
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "linear": LinearRegression(),
        "ridge": Ridge(alpha=1.0, random_state=42),
        "lasso": Lasso(alpha=0.01, max_iter=20000, random_state=42),
    }

    cv_results = {}
    for name, model in models.items():
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        rmse = -cross_val_score(
            pipe,
            X,
            y,
            cv=cv,
            scoring="neg_root_mean_squared_error",
        ).mean()
        cv_results[name] = rmse

    for name, rmse in cv_results.items():
        print(f"{name} CV RMSE: {rmse:.4f}")

    linear_pipe = Pipeline([("prep", preprocessor), ("model", LinearRegression())])
    linear_pipe.fit(X, y)
    feat_names = safe_feature_names(linear_pipe.named_steps["prep"])
    coefs = pd.Series(linear_pipe.named_steps["model"].coef_, index=feat_names)
    top_linear = coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(10)
    print("\nTop linear model coefficients by |magnitude|:")
    print(top_linear.to_string())

    standardized_beauty_coef = float(coefs.get("num__beauty", np.nan))

    tree_pipe = Pipeline(
        [
            ("prep", preprocessor),
            ("model", DecisionTreeRegressor(max_depth=3, min_samples_leaf=20, random_state=42)),
        ]
    )
    tree_pipe.fit(X, y)
    tree_importances = pd.Series(tree_pipe.named_steps["model"].feature_importances_, index=feat_names)
    top_tree = tree_importances.sort_values(ascending=False).head(10)
    print("\nDecision tree feature importances (top 10):")
    print(top_tree.to_string())

    ebm = ExplainableBoostingRegressor(random_state=42, interactions=0)
    ebm.fit(X, y)
    ebm_global = ebm.explain_global().data()
    ebm_importance = pd.Series(ebm_global["scores"], index=ebm_global["names"]).sort_values(ascending=False)
    print("\nExplainable Boosting feature importance:")
    print(ebm_importance.to_string())

    if "beauty" in ebm_importance.index:
        beauty_ebm_rank = int((ebm_importance.index == "beauty").argmax() + 1)
        beauty_ebm_score = float(ebm_importance.loc["beauty"])
    else:
        beauty_ebm_rank = -1
        beauty_ebm_score = float("nan")

    print_section("Conclusion Scoring")
    beauty_coef_adj = float(ols_adjusted.params["beauty"])
    beauty_p_adj = float(ols_adjusted.pvalues["beauty"])

    significant_positive = (beauty_coef_adj > 0) and (beauty_p_adj < 0.05)

    if significant_positive:
        response = 85
    elif beauty_coef_adj > 0 and beauty_p_adj < 0.10:
        response = 65
    elif beauty_coef_adj > 0:
        response = 45
    else:
        response = 20

    explanation = (
        f"Question: {research_question} Pearson r={pearson_r:.3f} (p={pearson_p:.3g}) indicates a positive "
        f"bivariate association. In robust OLS with controls, beauty coefficient={beauty_coef_adj:.3f} "
        f"(p={beauty_p_adj:.3g}, 95% CI {ols_adjusted.conf_int().loc['beauty', 0]:.3f} to "
        f"{ols_adjusted.conf_int().loc['beauty', 1]:.3f}), supporting a statistically significant positive effect "
        f"on teaching evaluations. Median-split t-test p={t_p:.3g} and quartile ANOVA p={anova_p:.3g} are consistent. "
        f"Interpretable models (linear coefficient for standardized beauty={standardized_beauty_coef:.3f}; "
        f"EBM beauty importance rank={beauty_ebm_rank}, score={beauty_ebm_score:.3f}) also identify beauty as an "
        f"important positive predictor. Overall evidence supports a meaningful positive impact, though effect size is moderate."
    )

    result = {"response": int(response), "explanation": explanation}

    Path("conclusion.txt").write_text(json.dumps(result, ensure_ascii=True))

    print(f"Likert response: {response}")
    print("Wrote conclusion.txt")


if __name__ == "__main__":
    main()
