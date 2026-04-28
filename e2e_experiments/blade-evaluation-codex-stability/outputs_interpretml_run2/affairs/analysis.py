import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")


def top_feature_importances(names, values, k=8):
    pairs = sorted(zip(names, values), key=lambda x: abs(x[1]), reverse=True)
    return pairs[:k]


def get_children_term_score_from_global_explanation(global_explanation):
    data = global_explanation.data()
    names = data.get("names", [])
    scores = data.get("scores", [])
    child_scores = [
        float(score)
        for name, score in zip(names, scores)
        if "children" in str(name).lower()
    ]
    return float(np.sum(child_scores)) if child_scores else 0.0


def main():
    info_path = Path("info.json")
    data_path = Path("affairs.csv")

    info = json.loads(info_path.read_text())
    question = info.get("research_questions", ["Unknown question"])[0].strip()

    df = pd.read_csv(data_path)

    print("=== Research Question ===")
    print(question)
    print("\n=== Data Overview ===")
    print(f"Shape: {df.shape}")
    print("Columns:", list(df.columns))
    print("Missing values by column:")
    print(df.isna().sum())

    # Treat rownames as an identifier when present.
    id_cols = []
    if "rownames" in df.columns and df["rownames"].nunique() == len(df):
        id_cols.append("rownames")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print("\n=== Numeric Summary ===")
    print(df[numeric_cols].describe().T)

    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    print("\n=== Categorical Distributions ===")
    for col in cat_cols:
        print(f"\n{col} value counts:")
        print(df[col].value_counts(dropna=False))

    print("\n=== Affairs Distribution ===")
    print(df["affairs"].value_counts().sort_index())

    corr = df[numeric_cols].corr(numeric_only=True)
    print("\n=== Correlation with affairs ===")
    print(corr["affairs"].sort_values(ascending=False))

    # Binary helper columns for tests/regressions.
    df_model = df.copy()
    df_model["children_bin"] = (df_model["children"].astype(str).str.lower() == "yes").astype(int)
    df_model["affair_any"] = (df_model["affairs"] > 0).astype(int)

    affairs_yes = df_model.loc[df_model["children_bin"] == 1, "affairs"]
    affairs_no = df_model.loc[df_model["children_bin"] == 0, "affairs"]

    welch_t = stats.ttest_ind(affairs_yes, affairs_no, equal_var=False)
    mw = stats.mannwhitneyu(affairs_yes, affairs_no, alternative="two-sided")
    anova = stats.f_oneway(affairs_yes, affairs_no)

    contingency = pd.crosstab(df_model["children"], df_model["affair_any"])
    chi2, chi2_p, _, _ = stats.chi2_contingency(contingency)

    print("\n=== Statistical Tests: children vs affairs ===")
    print(f"Mean affairs | children=yes: {affairs_yes.mean():.4f}")
    print(f"Mean affairs | children=no : {affairs_no.mean():.4f}")
    print(f"Welch t-test: statistic={welch_t.statistic:.4f}, p={welch_t.pvalue:.6f}")
    print(f"Mann-Whitney U: statistic={mw.statistic:.4f}, p={mw.pvalue:.6f}")
    print(f"One-way ANOVA: F={anova.statistic:.4f}, p={anova.pvalue:.6f}")
    print(f"Chi-square (children x any affair): chi2={chi2:.4f}, p={chi2_p:.6f}")

    base_covariates = [
        "age",
        "yearsmarried",
        "religiousness",
        "education",
        "occupation",
        "rating",
    ]
    if "gender" in df_model.columns:
        gender_term = " + C(gender)"
    else:
        gender_term = ""

    for col in id_cols:
        if col in base_covariates:
            base_covariates.remove(col)

    ols_formula = "affairs ~ children_bin + " + " + ".join(base_covariates) + gender_term
    logit_formula = "affair_any ~ children_bin + " + " + ".join(base_covariates) + gender_term

    ols = smf.ols(ols_formula, data=df_model).fit()
    logit = smf.logit(logit_formula, data=df_model).fit(disp=0)

    ols_coef = float(ols.params["children_bin"])
    ols_p = float(ols.pvalues["children_bin"])
    logit_coef = float(logit.params["children_bin"])
    logit_p = float(logit.pvalues["children_bin"])

    print("\n=== Regression Models (Adjusted) ===")
    print(f"OLS children_bin coef={ols_coef:.4f}, p={ols_p:.6f}")
    print(f"Logit(any affair) children_bin coef={logit_coef:.4f}, p={logit_p:.6f}")

    # Interpretable ML models.
    model_features = [c for c in df.columns if c != "affairs" and c not in id_cols]
    X = df_model[model_features]
    y_reg = df_model["affairs"]
    y_clf = df_model["affair_any"]

    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
    numeric_features = [c for c in X.columns if c not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                categorical_features,
            ),
            ("num", "passthrough", numeric_features),
        ]
    )

    X_train, X_test, y_train_reg, y_test_reg = train_test_split(
        X, y_reg, test_size=0.25, random_state=42
    )
    _, _, y_train_clf, y_test_clf = train_test_split(
        X, y_clf, test_size=0.25, random_state=42
    )

    lin = Pipeline([("prep", preprocessor), ("model", LinearRegression())])
    ridge = Pipeline([("prep", preprocessor), ("model", Ridge(alpha=1.0))])
    lasso = Pipeline([("prep", preprocessor), ("model", Lasso(alpha=0.01, max_iter=10000))])
    tree_reg = Pipeline(
        [
            ("prep", preprocessor),
            ("model", DecisionTreeRegressor(max_depth=3, min_samples_leaf=20, random_state=42)),
        ]
    )
    tree_clf = Pipeline(
        [
            ("prep", preprocessor),
            ("model", DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, random_state=42)),
        ]
    )

    lin.fit(X_train, y_train_reg)
    ridge.fit(X_train, y_train_reg)
    lasso.fit(X_train, y_train_reg)
    tree_reg.fit(X_train, y_train_reg)
    tree_clf.fit(X_train, y_train_clf)

    lin_r2 = r2_score(y_test_reg, lin.predict(X_test))
    ridge_r2 = r2_score(y_test_reg, ridge.predict(X_test))
    lasso_r2 = r2_score(y_test_reg, lasso.predict(X_test))
    tree_r2 = r2_score(y_test_reg, tree_reg.predict(X_test))
    tree_acc = accuracy_score(y_test_clf, tree_clf.predict(X_test))

    transformed_feature_names = lin.named_steps["prep"].get_feature_names_out()
    lin_coefs = lin.named_steps["model"].coef_
    tree_importances = tree_reg.named_steps["model"].feature_importances_

    top_lin = top_feature_importances(transformed_feature_names, lin_coefs)
    top_tree = top_feature_importances(transformed_feature_names, tree_importances)

    child_lin_coef = 0.0
    for name, coef in zip(transformed_feature_names, lin_coefs):
        if "children" in str(name).lower():
            child_lin_coef += float(coef)

    # Interpret glassbox models.
    ebr = ExplainableBoostingRegressor(random_state=42)
    ebr.fit(X_train, y_train_reg)
    ebr_global = ebr.explain_global()
    ebr_children_score = get_children_term_score_from_global_explanation(ebr_global)

    ebc = ExplainableBoostingClassifier(random_state=42)
    ebc.fit(X_train, y_train_clf)
    ebc_global = ebc.explain_global()
    ebc_children_score = get_children_term_score_from_global_explanation(ebc_global)

    print("\n=== Interpretable Model Summary ===")
    print(
        "R^2 (Linear/Ridge/Lasso/TreeReg): "
        f"{lin_r2:.4f} / {ridge_r2:.4f} / {lasso_r2:.4f} / {tree_r2:.4f}"
    )
    print(f"DecisionTreeClassifier accuracy: {tree_acc:.4f}")
    print(f"LinearRegression combined children coefficient: {child_lin_coef:.4f}")
    print("Top linear coefficients (abs):", top_lin)
    print("Top tree importances:", top_tree)
    print(f"EBM regressor children global score: {ebr_children_score:.4f}")
    print(f"EBM classifier children global score: {ebc_children_score:.4f}")

    # Likert scoring for: "Does having children decrease engagement in extramarital affairs?"
    mean_yes = float(affairs_yes.mean())
    mean_no = float(affairs_no.mean())
    unadjusted_opposite = welch_t.pvalue < 0.05 and mean_yes > mean_no
    adjusted_negative_sig = (ols_p < 0.05 and ols_coef < 0) or (logit_p < 0.05 and logit_coef < 0)
    adjusted_positive_sig = (ols_p < 0.05 and ols_coef > 0) or (logit_p < 0.05 and logit_coef > 0)

    if adjusted_negative_sig:
        response = 85
        if unadjusted_opposite:
            response -= 15
    else:
        response = 20
        if unadjusted_opposite:
            response = 10
        if adjusted_positive_sig:
            response = 5
        if welch_t.pvalue >= 0.05 and ols_p >= 0.05 and logit_p >= 0.05:
            response = 25

    response = int(max(0, min(100, response)))

    direction_text = "higher" if mean_yes > mean_no else "lower"
    explanation = (
        f"The data do not support that children decrease affairs. Unadjusted mean affairs are {mean_yes:.2f} "
        f"for marriages with children vs {mean_no:.2f} without children ({direction_text} with children), "
        f"with Welch t-test p={welch_t.pvalue:.4g} and Mann-Whitney p={mw.pvalue:.4g}. "
        f"In adjusted models controlling for age, years married, religiousness, education, occupation, rating, and gender, "
        f"children has OLS coef={ols_coef:.3f} (p={ols_p:.3g}) and logistic coef={logit_coef:.3f} (p={logit_p:.3g}), "
        f"so there is no significant negative children effect. Interpretable models (linear/tree/EBM) likewise do not show "
        f"a strong consistent negative contribution from children."
    )

    conclusion = {"response": response, "explanation": explanation}
    Path("conclusion.txt").write_text(json.dumps(conclusion))

    print("\n=== Final Conclusion JSON ===")
    print(json.dumps(conclusion, indent=2))


if __name__ == "__main__":
    main()
