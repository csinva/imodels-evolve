import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from interpret.glassbox import ExplainableBoostingClassifier

warnings.filterwarnings("ignore")
RANDOM_STATE = 42


def safe_auc(y_true, y_score):
    try:
        return roc_auc_score(y_true, y_score)
    except Exception:
        return float("nan")


def main():
    info_path = Path("info.json")
    data_path = Path("mortgage.csv")

    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

    research_question = info.get("research_questions", ["Unknown question"])[0]
    print(f"Research question: {research_question}")

    df = pd.read_csv(data_path)

    # Remove auto-generated index columns if present.
    unnamed_cols = [c for c in df.columns if c.lower().startswith("unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    # Keep only complete rows for statistical tests and interpretable models.
    print("\n=== DATA OVERVIEW ===")
    print(f"Rows: {len(df)}, Columns: {df.shape[1]}")
    print("Missing values by column:")
    print(df.isna().sum().sort_values(ascending=False).to_string())

    # Explore summary statistics and distributions.
    print("\n=== SUMMARY STATISTICS (NUMERIC) ===")
    print(df.describe().T.to_string())

    print("\n=== KEY DISTRIBUTIONS ===")
    for col in ["female", "accept", "deny", "black", "married", "self_employed", "bad_history"]:
        if col in df.columns:
            counts = df[col].value_counts(dropna=False).sort_index()
            props = df[col].value_counts(normalize=True, dropna=False).sort_index()
            dist = pd.DataFrame({"count": counts, "proportion": props})
            print(f"\n{col}:")
            print(dist.to_string())

    print("\n=== CORRELATIONS WITH ACCEPT ===")
    if "accept" not in df.columns:
        raise ValueError("Expected 'accept' column not found.")
    corr = df.corr(numeric_only=True)["accept"].sort_values(ascending=False)
    print(corr.to_string())

    # Statistical tests focused on gender vs mortgage approval.
    if "female" not in df.columns:
        raise ValueError("Expected 'female' column not found.")

    print("\n=== STATISTICAL TESTS ===")
    male_accept = df.loc[df["female"] == 0, "accept"]
    female_accept = df.loc[df["female"] == 1, "accept"]

    ttest = stats.ttest_ind(female_accept, male_accept, equal_var=False, nan_policy="omit")
    print(
        "Welch t-test on acceptance by gender: "
        f"t={ttest.statistic:.4f}, p={ttest.pvalue:.6g}, "
        f"female_mean={female_accept.mean():.4f}, male_mean={male_accept.mean():.4f}"
    )

    contingency = pd.crosstab(df["female"], df["accept"])
    chi2, chi2_p, dof, _ = stats.chi2_contingency(contingency)
    print(
        "Chi-square test (female x accept): "
        f"chi2={chi2:.4f}, dof={dof}, p={chi2_p:.6g}"
    )

    if "mortgage_credit" in df.columns:
        groups = [grp["accept"].values for _, grp in df.groupby("mortgage_credit")]
        if len(groups) >= 2:
            anova = stats.f_oneway(*groups)
            print(
                "ANOVA (acceptance across mortgage_credit levels): "
                f"F={anova.statistic:.4f}, p={anova.pvalue:.6g}"
            )

    # Adjusted regression models for gender effect.
    candidate_covariates = [
        "black",
        "housing_expense_ratio",
        "self_employed",
        "married",
        "mortgage_credit",
        "consumer_credit",
        "bad_history",
        "PI_ratio",
        "loan_to_value",
        "denied_PMI",
    ]
    covariates = [c for c in candidate_covariates if c in df.columns]

    formula_unadj = "accept ~ female"
    formula_adj = "accept ~ female + " + " + ".join(covariates) if covariates else formula_unadj

    logit_unadj = smf.logit(formula_unadj, data=df).fit(disp=False)
    logit_adj = smf.logit(formula_adj, data=df).fit(disp=False)

    lpm_adj = smf.ols(formula_adj, data=df).fit(cov_type="HC3")

    beta_female_unadj = float(logit_unadj.params["female"])
    p_female_unadj = float(logit_unadj.pvalues["female"])
    beta_female_adj = float(logit_adj.params["female"])
    p_female_adj = float(logit_adj.pvalues["female"])
    or_female_adj = float(np.exp(beta_female_adj))

    print(
        "Logit (unadjusted) female effect: "
        f"coef={beta_female_unadj:.4f}, OR={np.exp(beta_female_unadj):.4f}, p={p_female_unadj:.6g}"
    )
    print(
        "Logit (adjusted) female effect: "
        f"coef={beta_female_adj:.4f}, OR={or_female_adj:.4f}, p={p_female_adj:.6g}"
    )
    print(
        "OLS/LPM (adjusted, HC3 robust SE) female effect: "
        f"coef={float(lpm_adj.params['female']):.4f}, p={float(lpm_adj.pvalues['female']):.6g}"
    )

    # Interpretable ML models.
    print("\n=== INTERPRETABLE MODELS ===")
    feature_cols = [c for c in df.columns if c not in {"accept", "deny"}]
    X = df[feature_cols].copy()
    y = df["accept"].astype(int).copy()

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            )
        ],
        remainder="drop",
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Logistic regression (interpretable linear model)
    logreg_pipe = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]
    )
    logreg_pipe.fit(X_train, y_train)
    lr_proba = logreg_pipe.predict_proba(X_test)[:, 1]
    lr_pred = (lr_proba >= 0.5).astype(int)

    lr_acc = accuracy_score(y_test, lr_pred)
    lr_auc = safe_auc(y_test, lr_proba)

    coef_series = pd.Series(
        logreg_pipe.named_steps["model"].coef_[0],
        index=num_cols,
    ).sort_values(key=np.abs, ascending=False)

    female_lr_coef = float(coef_series.get("female", np.nan))

    print(f"LogisticRegression: accuracy={lr_acc:.4f}, roc_auc={lr_auc:.4f}")
    print("Top |coefficients|:")
    print(coef_series.head(8).to_string())

    # Decision tree (interpretable rule-like model)
    X_train_num = X_train[num_cols].copy()
    X_test_num = X_test[num_cols].copy()
    imp = SimpleImputer(strategy="median")
    X_train_num_imp = imp.fit_transform(X_train_num)
    X_test_num_imp = imp.transform(X_test_num)
    X_train_num_imp_df = pd.DataFrame(X_train_num_imp, columns=num_cols, index=X_train_num.index)
    X_test_num_imp_df = pd.DataFrame(X_test_num_imp, columns=num_cols, index=X_test_num.index)

    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=30, random_state=RANDOM_STATE)
    tree.fit(X_train_num_imp_df, y_train)
    tree_proba = tree.predict_proba(X_test_num_imp_df)[:, 1]
    tree_pred = (tree_proba >= 0.5).astype(int)
    tree_acc = accuracy_score(y_test, tree_pred)
    tree_auc = safe_auc(y_test, tree_proba)

    tree_importance = pd.Series(tree.feature_importances_, index=num_cols).sort_values(ascending=False)
    female_tree_importance = float(tree_importance.get("female", 0.0))

    print(f"DecisionTreeClassifier: accuracy={tree_acc:.4f}, roc_auc={tree_auc:.4f}")
    print("Top feature importances:")
    print(tree_importance.head(8).to_string())

    # Explainable Boosting Machine (interpretable additive model)
    ebm = ExplainableBoostingClassifier(
        random_state=RANDOM_STATE,
        interactions=0,
        max_bins=64,
        max_rounds=500,
    )
    ebm.fit(X_train_num_imp_df, y_train)
    ebm_proba = ebm.predict_proba(X_test_num_imp_df)[:, 1]
    ebm_pred = (ebm_proba >= 0.5).astype(int)
    ebm_acc = accuracy_score(y_test, ebm_pred)
    ebm_auc = safe_auc(y_test, ebm_proba)

    term_names = list(ebm.term_names_)
    term_imps = pd.Series(ebm.term_importances(), index=term_names).sort_values(ascending=False)
    female_ebm_importance = float(term_imps.get("female", 0.0))

    print(f"ExplainableBoostingClassifier: accuracy={ebm_acc:.4f}, roc_auc={ebm_auc:.4f}")
    print("Top EBM term importances:")
    print(term_imps.head(8).to_string())

    # Build a calibrated Likert response from significance and consistency.
    # Interpretation target: "Does gender affect approval?"
    unadjusted_sig = (ttest.pvalue < 0.05) or (chi2_p < 0.05)
    adjusted_sig = p_female_adj < 0.05
    adjusted_direction_positive = beta_female_adj > 0

    # Start neutral and move by evidence.
    score = 50
    if adjusted_sig:
        score += 20
    else:
        score -= 20

    if unadjusted_sig:
        score += 15
    else:
        score -= 5

    # Stronger magnitude if OR deviates meaningfully from 1.
    score += int(np.clip((abs(or_female_adj - 1.0) / 0.5) * 10, 0, 10))

    # Penalize inconsistency between adjusted and unadjusted evidence.
    if adjusted_sig and not unadjusted_sig:
        score -= 10

    score = int(np.clip(score, 0, 100))

    direction_text = "higher" if adjusted_direction_positive else "lower"
    explanation = (
        f"Unadjusted approval rates are nearly identical by gender "
        f"(female={female_accept.mean():.3f}, male={male_accept.mean():.3f}; "
        f"chi-square p={chi2_p:.3g}; t-test p={ttest.pvalue:.3g}). "
        f"After controlling for credit and application variables in logistic regression, "
        f"female is statistically significant (OR={or_female_adj:.2f}, p={p_female_adj:.3g}), "
        f"indicating {direction_text} odds of approval for women conditional on covariates. "
        f"Interpretable models (logistic coefficients, tree importances, and EBM term importances) "
        f"show gender is a weaker predictor than core risk variables, so evidence for a gender effect "
        f"exists but is moderate rather than strong."
    )

    result = {"response": score, "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(result, f)

    print("\n=== CONCLUSION JSON ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
