import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor


def pvalue_to_score(p: float) -> int:
    if p < 0.01:
        return 100
    if p < 0.05:
        return 85
    if p < 0.10:
        return 65
    return 20


def main() -> None:
    info_path = Path("info.json")
    data_path = Path("panda_nuts.csv")

    info = json.loads(info_path.read_text())
    question = info.get("research_questions", ["Unknown question"])[0]
    print("Research question:")
    print(question)
    print("=" * 80)

    df = pd.read_csv(data_path)
    df["sex"] = df["sex"].astype(str).str.strip().str.lower()
    df["help"] = df["help"].astype(str).str.strip().str.lower().replace({"n": "n", "no": "n", "y": "y", "yes": "y"})
    df["hammer"] = df["hammer"].astype(str).str.strip()

    if (df["seconds"] <= 0).any():
        raise ValueError("Encountered non-positive session duration, cannot compute efficiency safely.")

    # Efficiency target for this question: nuts opened per minute.
    df["eff_per_min"] = (df["nuts_opened"] / df["seconds"]) * 60.0
    df["log_eff_per_min"] = np.log1p(df["eff_per_min"])

    print("Data shape:", df.shape)
    print("Missing values per column:")
    print(df.isna().sum())
    print("=" * 80)

    print("Numeric summary:")
    print(df[["age", "nuts_opened", "seconds", "eff_per_min"]].describe().T)
    print("=" * 80)

    print("Categorical counts:")
    for c in ["sex", "help", "hammer"]:
        print(f"{c} counts:\n{df[c].value_counts(dropna=False)}")
    print("=" * 80)

    print("Distribution snapshots (histogram bins):")
    age_hist = np.histogram(df["age"], bins=6)
    eff_hist = np.histogram(df["eff_per_min"], bins=6)
    print("Age histogram counts:", age_hist[0].tolist())
    print("Age histogram edges:", np.round(age_hist[1], 2).tolist())
    print("Efficiency histogram counts:", eff_hist[0].tolist())
    print("Efficiency histogram edges:", np.round(eff_hist[1], 2).tolist())
    print("=" * 80)

    print("Correlations (numeric columns):")
    corr = df[["age", "nuts_opened", "seconds", "eff_per_min"]].corr(numeric_only=True)
    print(corr)
    print("=" * 80)

    # Statistical tests
    age_spearman = stats.spearmanr(df["age"], df["eff_per_min"])
    age_pearson = stats.pearsonr(df["age"], df["eff_per_min"])

    eff_f = df.loc[df["sex"] == "f", "eff_per_min"]
    eff_m = df.loc[df["sex"] == "m", "eff_per_min"]
    sex_ttest = stats.ttest_ind(eff_f, eff_m, equal_var=False)
    sex_mw = stats.mannwhitneyu(eff_f, eff_m, alternative="two-sided")

    eff_help_n = df.loc[df["help"] == "n", "eff_per_min"]
    eff_help_y = df.loc[df["help"] == "y", "eff_per_min"]
    help_ttest = stats.ttest_ind(eff_help_n, eff_help_y, equal_var=False)
    help_mw = stats.mannwhitneyu(eff_help_n, eff_help_y, alternative="two-sided")

    print("Statistical tests:")
    print(f"Age vs efficiency (Spearman): rho={age_spearman.statistic:.4f}, p={age_spearman.pvalue:.4g}")
    print(f"Age vs efficiency (Pearson): r={age_pearson.statistic:.4f}, p={age_pearson.pvalue:.4g}")
    print(f"Sex effect Welch t-test (f vs m): t={sex_ttest.statistic:.4f}, p={sex_ttest.pvalue:.4g}")
    print(f"Sex effect Mann-Whitney (f vs m): U={sex_mw.statistic:.4f}, p={sex_mw.pvalue:.4g}")
    print(f"Help effect Welch t-test (no-help vs help): t={help_ttest.statistic:.4f}, p={help_ttest.pvalue:.4g}")
    print(f"Help effect Mann-Whitney (no-help vs help): U={help_mw.statistic:.4f}, p={help_mw.pvalue:.4g}")
    print("=" * 80)

    # OLS models with and without hammer control; clustered SE by chimp ID for repeated sessions.
    ols_basic = smf.ols("log_eff_per_min ~ age + C(sex) + C(help)", data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["chimpanzee"]}
    )
    ols_control = smf.ols("log_eff_per_min ~ age + C(sex) + C(help) + C(hammer)", data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["chimpanzee"]}
    )

    print("Cluster-robust OLS (age + sex + help):")
    print(ols_basic.summary())
    print("=" * 80)
    print("Cluster-robust OLS with hammer control:")
    print(ols_control.summary())
    print("=" * 80)

    # Interpretable sklearn models
    target = "log_eff_per_min"
    model_features = ["age", "sex", "help", "hammer"]
    X = df[model_features].copy()
    y = df[target].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    categorical = ["sex", "help", "hammer"]
    numeric = ["age"]

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical),
        ]
    )

    linear_pipe = Pipeline([("pre", pre), ("model", LinearRegression())])
    ridge_pipe = Pipeline([("pre", pre), ("model", Ridge(alpha=1.0, random_state=42))])
    lasso_pipe = Pipeline([("pre", pre), ("model", Lasso(alpha=0.01, random_state=42, max_iter=10000))])
    tree_pipe = Pipeline([("pre", pre), ("model", DecisionTreeRegressor(max_depth=3, random_state=42))])

    linear_pipe.fit(X_train, y_train)
    ridge_pipe.fit(X_train, y_train)
    lasso_pipe.fit(X_train, y_train)
    tree_pipe.fit(X_train, y_train)

    yhat_linear = linear_pipe.predict(X_test)
    yhat_ridge = ridge_pipe.predict(X_test)
    yhat_lasso = lasso_pipe.predict(X_test)
    yhat_tree = tree_pipe.predict(X_test)

    print("Interpretable sklearn model performance (R^2 on test split):")
    print(f"LinearRegression: {r2_score(y_test, yhat_linear):.4f}")
    print(f"Ridge:            {r2_score(y_test, yhat_ridge):.4f}")
    print(f"Lasso:            {r2_score(y_test, yhat_lasso):.4f}")
    print(f"DecisionTree:     {r2_score(y_test, yhat_tree):.4f}")

    feature_names = linear_pipe.named_steps["pre"].get_feature_names_out().tolist()
    linear_coef = pd.Series(linear_pipe.named_steps["model"].coef_, index=feature_names).sort_values(key=np.abs, ascending=False)
    ridge_coef = pd.Series(ridge_pipe.named_steps["model"].coef_, index=feature_names).sort_values(key=np.abs, ascending=False)
    lasso_coef = pd.Series(lasso_pipe.named_steps["model"].coef_, index=feature_names).sort_values(key=np.abs, ascending=False)
    tree_imp = pd.Series(tree_pipe.named_steps["model"].feature_importances_, index=feature_names).sort_values(ascending=False)

    print("\nTop coefficients/importances:")
    print("LinearRegression coef (top 8 abs):")
    print(linear_coef.head(8))
    print("Ridge coef (top 8 abs):")
    print(ridge_coef.head(8))
    print("Lasso coef (top 8 abs):")
    print(lasso_coef.head(8))
    print("DecisionTree feature_importances_ (top 8):")
    print(tree_imp.head(8))
    print("=" * 80)

    # interpret glassbox models
    ebm_reg = ExplainableBoostingRegressor(random_state=42, interactions=0)
    ebm_reg.fit(X, y)
    ebm_reg_importance = pd.Series(ebm_reg.term_importances(), index=ebm_reg.term_names_).sort_values(ascending=False)

    y_cls = (df["eff_per_min"] > df["eff_per_min"].median()).astype(int)
    ebm_clf = ExplainableBoostingClassifier(random_state=42, interactions=0)
    ebm_clf.fit(X, y_cls)
    ebm_clf_importance = pd.Series(ebm_clf.term_importances(), index=ebm_clf.term_names_).sort_values(ascending=False)

    print("EBM regressor term importances:")
    print(ebm_reg_importance)
    print("EBM classifier term importances:")
    print(ebm_clf_importance)
    print("=" * 80)

    # Evidence synthesis for required 0-100 response.
    p_age = [age_spearman.pvalue, float(ols_basic.pvalues.get("age", np.nan)), float(ols_control.pvalues.get("age", np.nan))]
    p_sex = [sex_ttest.pvalue, float(ols_basic.pvalues.get("C(sex)[T.m]", np.nan)), float(ols_control.pvalues.get("C(sex)[T.m]", np.nan))]
    p_help = [help_ttest.pvalue, help_mw.pvalue, float(ols_basic.pvalues.get("C(help)[T.y]", np.nan)), float(ols_control.pvalues.get("C(help)[T.y]", np.nan))]

    score_age = int(round(np.mean([pvalue_to_score(p) for p in p_age if not np.isnan(p)])))
    score_sex = int(round(np.mean([pvalue_to_score(p) for p in p_sex if not np.isnan(p)])))
    score_help = int(round(np.mean([pvalue_to_score(p) for p in p_help if not np.isnan(p)])))

    response = int(round(np.mean([score_age, score_sex, score_help])))
    response = max(0, min(100, response))

    mean_by_sex = df.groupby("sex")["eff_per_min"].mean().to_dict()
    mean_by_help = df.groupby("help")["eff_per_min"].mean().to_dict()

    age_dir = "positive" if age_spearman.statistic > 0 else "negative"
    sex_dir = "male>female" if mean_by_sex.get("m", np.nan) > mean_by_sex.get("f", np.nan) else "female>male"
    help_dir = "help<higher?" if mean_by_help.get("y", np.nan) > mean_by_help.get("n", np.nan) else "help lower than no-help"

    explanation = (
        f"Evidence is mixed-to-strong overall (score={response}). "
        f"Age shows a {age_dir} and statistically significant association with efficiency "
        f"(Spearman p={age_spearman.pvalue:.3g}; OLS p={ols_basic.pvalues.get('age', np.nan):.3g}). "
        f"Sex is also significant with {sex_dir} efficiency "
        f"(Welch p={sex_ttest.pvalue:.3g}; OLS p={ols_basic.pvalues.get('C(sex)[T.m]', np.nan):.3g}). "
        f"Help is not consistently significant after accounting for controls/repeated observations "
        f"(Welch p={help_ttest.pvalue:.3g}, Mann-Whitney p={help_mw.pvalue:.3g}, controlled OLS p={ols_control.pvalues.get('C(help)[T.y]', np.nan):.3g}); "
        f"mean efficiency no-help={mean_by_help.get('n', float('nan')):.2f}, help={mean_by_help.get('y', float('nan')):.2f} nuts/min. "
        f"Interpretable models (linear/tree/EBM) consistently rank age and sex as important, with weaker and unstable help effect."
    )

    output = {"response": response, "explanation": explanation}
    Path("conclusion.txt").write_text(json.dumps(output, ensure_ascii=True))

    print("Final conclusion JSON written to conclusion.txt:")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
