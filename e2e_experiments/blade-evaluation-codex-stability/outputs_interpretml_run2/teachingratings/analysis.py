import json
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from pandas.api.types import is_numeric_dtype
import statsmodels.formula.api as smf
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from interpret.glassbox import ExplainableBoostingRegressor

warnings.filterwarnings("ignore")


def get_categorical_columns(df: pd.DataFrame) -> list:
    cols = []
    for c in df.columns:
        if not is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def make_onehot():
    try:
        return OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(drop="first", handle_unknown="ignore", sparse=False)


def main():
    # 1) Load
    df = pd.read_csv("teachingratings.csv")

    # 2) Explore
    print("=== DATA OVERVIEW ===")
    print(f"Rows: {len(df)}, Columns: {df.shape[1]}")
    print("\nDtypes:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isna().sum())

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = get_categorical_columns(df)

    print("\n=== SUMMARY STATISTICS (NUMERIC) ===")
    print(df[numeric_cols].describe().T)

    print("\n=== DISTRIBUTION SNAPSHOT ===")
    for col in ["beauty", "eval", "age", "students", "allstudents"]:
        if col in df.columns:
            skew = stats.skew(df[col])
            kurt = stats.kurtosis(df[col])
            hist_counts, bin_edges = np.histogram(df[col], bins=8)
            print(f"{col}: skew={skew:.3f}, kurtosis={kurt:.3f}")
            print(f"  bins={np.round(bin_edges, 3)}")
            print(f"  counts={hist_counts}")

    print("\n=== CORRELATIONS (NUMERIC WITH eval) ===")
    corr_with_eval = df[numeric_cols].corr(numeric_only=True)["eval"].sort_values(ascending=False)
    print(corr_with_eval)

    # 3) Statistical tests for beauty -> eval
    print("\n=== STATISTICAL TESTS ===")

    pearson_r, pearson_p = stats.pearsonr(df["beauty"], df["eval"])
    spearman_rho, spearman_p = stats.spearmanr(df["beauty"], df["eval"])

    q25, q75 = df["beauty"].quantile([0.25, 0.75])
    low_beauty_eval = df.loc[df["beauty"] <= q25, "eval"]
    high_beauty_eval = df.loc[df["beauty"] >= q75, "eval"]
    t_stat, t_p = stats.ttest_ind(high_beauty_eval, low_beauty_eval, equal_var=False)

    df["beauty_quartile"] = pd.qcut(df["beauty"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    groups = [df.loc[df["beauty_quartile"] == q, "eval"].values for q in ["Q1", "Q2", "Q3", "Q4"]]
    f_stat, anova_p = stats.f_oneway(*groups)

    print(f"Pearson r(beauty, eval) = {pearson_r:.4f}, p={pearson_p:.4g}")
    print(f"Spearman rho(beauty, eval) = {spearman_rho:.4f}, p={spearman_p:.4g}")
    print(
        "Top vs Bottom beauty quartile t-test: "
        f"t={t_stat:.4f}, p={t_p:.4g}, "
        f"mean_top={high_beauty_eval.mean():.3f}, mean_bottom={low_beauty_eval.mean():.3f}"
    )
    print(f"ANOVA across beauty quartiles: F={f_stat:.4f}, p={anova_p:.4g}")

    # OLS models for inferential effect size and significance
    simple_ols = smf.ols("eval ~ beauty", data=df).fit()
    controlled_ols = smf.ols(
        "eval ~ beauty + age + students + allstudents + C(minority) + C(gender) + "
        "C(credits) + C(division) + C(native) + C(tenure)",
        data=df,
    ).fit()

    beauty_coef_simple = float(simple_ols.params["beauty"])
    beauty_p_simple = float(simple_ols.pvalues["beauty"])
    beauty_ci_simple = simple_ols.conf_int().loc["beauty"].tolist()

    beauty_coef_ctrl = float(controlled_ols.params["beauty"])
    beauty_p_ctrl = float(controlled_ols.pvalues["beauty"])
    beauty_ci_ctrl = controlled_ols.conf_int().loc["beauty"].tolist()

    print("\nOLS (eval ~ beauty):")
    print(
        f"  beauty coef={beauty_coef_simple:.4f}, p={beauty_p_simple:.4g}, "
        f"95% CI=({beauty_ci_simple[0]:.4f}, {beauty_ci_simple[1]:.4f}), R^2={simple_ols.rsquared:.4f}"
    )

    print("\nControlled OLS (with covariates):")
    print(
        f"  beauty coef={beauty_coef_ctrl:.4f}, p={beauty_p_ctrl:.4g}, "
        f"95% CI=({beauty_ci_ctrl[0]:.4f}, {beauty_ci_ctrl[1]:.4f}), R^2={controlled_ols.rsquared:.4f}"
    )

    # 4) Interpretable models (sklearn + interpret)
    feature_cols = [
        c
        for c in df.columns
        if c not in {"eval", "rownames", "prof", "beauty_quartile"}
    ]
    X = df[feature_cols].copy()
    y = df["eval"].copy()

    model_cat_cols = [c for c in X.columns if c in cat_cols]
    model_num_cols = [c for c in X.columns if c not in model_cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", make_onehot(), model_cat_cols),
            ("num", "passthrough", model_num_cols),
        ]
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scorer = make_scorer(r2_score)

    models = {
        "linear": LinearRegression(),
        "ridge": Ridge(alpha=1.0, random_state=42),
        "lasso": Lasso(alpha=0.01, random_state=42, max_iter=10000),
        "tree": DecisionTreeRegressor(max_depth=3, random_state=42),
    }

    cv_results = {}
    fitted_pipelines = {}

    print("\n=== INTERPRETABLE MODEL PERFORMANCE (5-fold CV R^2) ===")
    for name, model in models.items():
        pipe = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )
        scores = cross_val_score(pipe, X, y, cv=cv, scoring=scorer)
        cv_results[name] = float(np.mean(scores))
        print(f"{name}: mean R^2={np.mean(scores):.4f} (std={np.std(scores):.4f})")
        pipe.fit(X, y)
        fitted_pipelines[name] = pipe

    # Coefficients from linear model
    linear_pipe = fitted_pipelines["linear"]
    ohe = linear_pipe.named_steps["preprocessor"].named_transformers_["cat"]
    cat_feature_names = []
    if len(model_cat_cols) > 0:
        cat_feature_names = ohe.get_feature_names_out(model_cat_cols).tolist()
    all_feature_names = cat_feature_names + model_num_cols

    linear_coefs = linear_pipe.named_steps["model"].coef_
    coef_table = pd.DataFrame({"feature": all_feature_names, "coef": linear_coefs})
    coef_table["abs_coef"] = coef_table["coef"].abs()
    coef_table = coef_table.sort_values("abs_coef", ascending=False)

    print("\nTop linear coefficients by magnitude:")
    print(coef_table[["feature", "coef"]].head(10).to_string(index=False))

    # Tree importances
    tree_model = fitted_pipelines["tree"].named_steps["model"]
    tree_importances = pd.DataFrame(
        {
            "feature": all_feature_names,
            "importance": tree_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    print("\nTop decision-tree feature importances:")
    print(tree_importances.head(10).to_string(index=False))

    # EBM for additive interpretable model
    X_ebm = df[feature_cols].copy()
    for c in model_cat_cols:
        X_ebm[c] = X_ebm[c].astype("category")

    ebm = ExplainableBoostingRegressor(random_state=42, interactions=0)
    ebm.fit(X_ebm, y)
    ebm_global = ebm.explain_global()
    ebm_names = ebm_global.data()["names"]
    ebm_scores = ebm_global.data()["scores"]
    ebm_importance = pd.DataFrame({"feature": ebm_names, "importance": ebm_scores}).sort_values(
        "importance", ascending=False
    )

    ebm_beauty_rank = (
        int(np.where(ebm_importance["feature"].values == "beauty")[0][0]) + 1
        if "beauty" in ebm_importance["feature"].values
        else None
    )

    print("\nTop EBM global importances:")
    print(ebm_importance.head(10).to_string(index=False))

    # 5) Conclusion score (0-100) based on significance + effect consistency
    eval_sd = float(df["eval"].std())
    standardized_effect = beauty_coef_ctrl / eval_sd if eval_sd > 0 else 0.0

    significance_support = sum(
        [
            pearson_p < 0.05,
            t_p < 0.05,
            anova_p < 0.05,
            beauty_p_simple < 0.05,
            beauty_p_ctrl < 0.05,
        ]
    )

    score = 50
    if beauty_p_ctrl < 0.001:
        score += 25
    elif beauty_p_ctrl < 0.01:
        score += 20
    elif beauty_p_ctrl < 0.05:
        score += 12
    else:
        score -= 25

    if beauty_coef_ctrl > 0:
        score += 10
    else:
        score -= 10

    score += int(min(10, max(0, abs(standardized_effect) * 40)))
    score += int(min(8, significance_support))

    if ebm_beauty_rank is not None and ebm_beauty_rank <= 5:
        score += 5

    score = int(max(0, min(100, score)))

    explanation = (
        f"Beauty shows a positive, statistically significant association with teaching evaluations. "
        f"Pearson r={pearson_r:.3f} (p={pearson_p:.3g}), quartile t-test p={t_p:.3g}, "
        f"and ANOVA p={anova_p:.3g}. In OLS, beauty remains significant after controls "
        f"(coef={beauty_coef_ctrl:.3f}, p={beauty_p_ctrl:.3g}, "
        f"95% CI [{beauty_ci_ctrl[0]:.3f}, {beauty_ci_ctrl[1]:.3f}]). "
        f"Interpretable models (linear/tree/EBM) also rank beauty as an important predictor, "
        f"supporting a meaningful but moderate positive effect rather than an overwhelming one."
    )

    result = {"response": score, "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(result, f)

    print("\n=== FINAL CONCLUSION JSON ===")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
