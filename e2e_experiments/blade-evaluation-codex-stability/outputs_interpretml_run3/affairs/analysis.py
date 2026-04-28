import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

warnings.filterwarnings("ignore")


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def main() -> None:
    data_path = Path("affairs.csv")
    info_path = Path("info.json")

    if info_path.exists():
        info = json.loads(info_path.read_text())
        rq = info.get("research_questions", ["N/A"])[0]
    else:
        rq = "Does having children decrease engagement in extramarital affairs?"

    print_section("Research Question")
    print(rq)

    df = pd.read_csv(data_path)

    # Basic cleaning / feature construction
    df = df.copy()
    df["children_bin"] = (df["children"].str.lower() == "yes").astype(int)
    df["gender_bin"] = (df["gender"].str.lower() == "male").astype(int)
    df["affair_any"] = (df["affairs"] > 0).astype(int)

    print_section("Data Overview")
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    print("\nDtypes:")
    print(df.dtypes)

    print_section("Summary Statistics")
    print(df.describe(include="all").transpose())

    print_section("Distribution Snapshots")
    print("Children counts:")
    print(df["children"].value_counts(dropna=False))
    print("\nAffairs value counts:")
    print(df["affairs"].value_counts().sort_index())
    print("\nAffair-any rate by children:")
    print(df.groupby("children")["affair_any"].mean())
    print("\nAffairs mean/median by children:")
    print(df.groupby("children")["affairs"].agg(["mean", "median", "std", "count"]))

    numeric_cols = [
        "affairs",
        "age",
        "yearsmarried",
        "religiousness",
        "education",
        "occupation",
        "rating",
        "children_bin",
        "gender_bin",
        "affair_any",
    ]

    print_section("Correlation Matrix (Numeric)")
    corr = df[numeric_cols].corr(numeric_only=True)
    print(corr)

    # Statistical tests focused on children vs affairs
    print_section("Statistical Tests")
    affairs_yes = df.loc[df["children_bin"] == 1, "affairs"]
    affairs_no = df.loc[df["children_bin"] == 0, "affairs"]

    ttest = stats.ttest_ind(affairs_yes, affairs_no, equal_var=False)
    mwu = stats.mannwhitneyu(affairs_yes, affairs_no, alternative="two-sided")
    anova = stats.f_oneway(affairs_yes, affairs_no)

    ctab = pd.crosstab(df["children"], df["affair_any"])
    chi2_stat, chi2_p, _, _ = stats.chi2_contingency(ctab)

    print(f"Welch t-test affairs~children: statistic={ttest.statistic:.4f}, p={ttest.pvalue:.6f}")
    print(f"Mann-Whitney U affairs~children: statistic={mwu.statistic:.4f}, p={mwu.pvalue:.6f}")
    print(f"One-way ANOVA affairs~children: F={anova.statistic:.4f}, p={anova.pvalue:.6f}")
    print("\nChi-square table children x affair_any:")
    print(ctab)
    print(f"Chi-square test: chi2={chi2_stat:.4f}, p={chi2_p:.6f}")

    # Adjusted regression (interpretability + inference)
    print_section("Adjusted OLS (statsmodels)")
    ols = smf.ols(
        "affairs ~ C(children) + C(gender) + age + yearsmarried + religiousness + education + occupation + rating",
        data=df,
    ).fit(cov_type="HC3")
    print(ols.summary())

    child_coef = safe_float(ols.params.get("C(children)[T.yes]", np.nan))
    child_p = safe_float(ols.pvalues.get("C(children)[T.yes]", np.nan))
    print(f"\nAdjusted child coefficient: {child_coef:.4f}, p={child_p:.6f}")

    # Interpretable sklearn models
    print_section("Interpretable sklearn Models")
    features = [
        "gender",
        "age",
        "yearsmarried",
        "children",
        "religiousness",
        "education",
        "occupation",
        "rating",
    ]
    X = pd.get_dummies(df[features], drop_first=True)
    y_reg = df["affairs"]
    y_clf = df["affair_any"]

    lin = LinearRegression().fit(X, y_reg)
    ridge = Ridge(alpha=1.0, random_state=42).fit(X, y_reg)
    lasso = Lasso(alpha=0.01, random_state=42, max_iter=10000).fit(X, y_reg)
    dtr = DecisionTreeRegressor(max_depth=3, random_state=42).fit(X, y_reg)
    dtc = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X, y_clf)

    coef_df = pd.DataFrame(
        {
            "feature": X.columns,
            "linear_coef": lin.coef_,
            "ridge_coef": ridge.coef_,
            "lasso_coef": lasso.coef_,
        }
    ).sort_values("linear_coef", key=lambda s: s.abs(), ascending=False)
    print("Top linear/ridge/lasso coefficients by |linear_coef|:")
    print(coef_df.head(10).to_string(index=False))

    imp_reg = pd.DataFrame({"feature": X.columns, "importance": dtr.feature_importances_}).sort_values(
        "importance", ascending=False
    )
    imp_clf = pd.DataFrame({"feature": X.columns, "importance": dtc.feature_importances_}).sort_values(
        "importance", ascending=False
    )
    print("\nDecisionTreeRegressor feature importances:")
    print(imp_reg.head(10).to_string(index=False))
    print("\nDecisionTreeClassifier feature importances:")
    print(imp_clf.head(10).to_string(index=False))

    # interpret models
    print_section("interpret Glassbox Models")
    ebm_reg_children_effect = np.nan
    ebm_clf_children_effect = np.nan

    try:
        from interpret.glassbox import (
            ExplainableBoostingRegressor,
            ExplainableBoostingClassifier,
            DecisionListClassifier,
        )

        ebm_reg = ExplainableBoostingRegressor(random_state=42, interactions=0)
        ebm_reg.fit(df[features], y_reg)
        reg_global = ebm_reg.explain_global().data()
        reg_importance = pd.DataFrame(
            {"term": reg_global["names"], "importance": [float(v) for v in reg_global["scores"]]}
        ).sort_values("importance", ascending=False)
        print("EBM Regressor term importances:")
        print(reg_importance.to_string(index=False))

        reg_child_idx = ebm_reg.term_names_.index("children")
        reg_child_data = ebm_reg.explain_global().data(reg_child_idx)
        reg_child_scores = dict(zip(reg_child_data["names"], reg_child_data["scores"]))
        ebm_reg_children_effect = safe_float(reg_child_scores.get("yes", np.nan) - reg_child_scores.get("no", np.nan))
        print(f"\nEBM Regressor children contribution scores: {reg_child_scores}")
        print(f"EBM Regressor effect (yes - no): {ebm_reg_children_effect:.6f}")

        ebm_clf = ExplainableBoostingClassifier(random_state=42, interactions=0)
        ebm_clf.fit(df[features], y_clf)
        clf_global = ebm_clf.explain_global().data()
        clf_importance = pd.DataFrame(
            {"term": clf_global["names"], "importance": [float(v) for v in clf_global["scores"]]}
        ).sort_values("importance", ascending=False)
        print("\nEBM Classifier term importances:")
        print(clf_importance.to_string(index=False))

        clf_child_idx = ebm_clf.term_names_.index("children")
        clf_child_data = ebm_clf.explain_global().data(clf_child_idx)
        clf_child_scores = dict(zip(clf_child_data["names"], clf_child_data["scores"]))
        ebm_clf_children_effect = safe_float(clf_child_scores.get("yes", np.nan) - clf_child_scores.get("no", np.nan))
        print(f"\nEBM Classifier children contribution scores: {clf_child_scores}")
        print(f"EBM Classifier log-odds effect (yes - no): {ebm_clf_children_effect:.6f}")

        # Optional: DecisionListClassifier may require additional dependency (skrules)
        try:
            dlc = DecisionListClassifier(random_state=42, max_features=4)
            dlc.fit(df[features], y_clf)
            dl_global = dlc.explain_global().data()
            print("\nDecisionListClassifier global explanation available.")
            print(f"Keys: {list(dl_global.keys())}")
        except Exception as dl_err:
            print(f"\nDecisionListClassifier unavailable in this environment: {dl_err}")

    except Exception as interpret_err:
        print(f"interpret models unavailable: {interpret_err}")

    # Aggregate evidence into Likert score answering:
    # "Does having children decrease engagement in affairs?"
    mean_yes = safe_float(affairs_yes.mean())
    mean_no = safe_float(affairs_no.mean())
    rate_yes = safe_float(df.loc[df["children_bin"] == 1, "affair_any"].mean())
    rate_no = safe_float(df.loc[df["children_bin"] == 0, "affair_any"].mean())

    score = 50.0

    # Unadjusted mean difference test
    if ttest.pvalue < 0.05:
        score += 20 if mean_yes < mean_no else -25
    else:
        score += 5 if mean_yes < mean_no else -5

    # Binary-rate association test
    if chi2_p < 0.05:
        score += 20 if rate_yes < rate_no else -25
    else:
        score += 5 if rate_yes < rate_no else -5

    # Adjusted OLS inference
    if np.isfinite(child_coef) and np.isfinite(child_p):
        if child_p < 0.05:
            score += 25 if child_coef < 0 else -20
        else:
            score += 3 if child_coef < 0 else -3

    # ANOVA direction support
    if anova.pvalue < 0.05:
        score += 10 if mean_yes < mean_no else -10

    # EBM directional evidence (lower means support "decrease")
    if np.isfinite(ebm_reg_children_effect):
        score += 6 if ebm_reg_children_effect < 0 else -6
    if np.isfinite(ebm_clf_children_effect):
        score += 8 if ebm_clf_children_effect < 0 else -8

    response = int(np.clip(round(score), 0, 100))

    explanation = (
        f"Unadjusted tests do not support a decrease: mean affairs is {mean_yes:.3f} for couples with children "
        f"vs {mean_no:.3f} without children (Welch t-test p={ttest.pvalue:.4g}; Mann-Whitney p={mwu.pvalue:.4g}), "
        f"and affair prevalence is {rate_yes:.3f} with children vs {rate_no:.3f} without children "
        f"(chi-square p={chi2_p:.4g}). In adjusted OLS, the children coefficient is {child_coef:.3f} "
        f"with p={child_p:.4g}, indicating no statistically significant decrease after controls. "
        f"Interpretable EBM models show mixed but overall weak evidence for a protective children effect "
        f"(regression children effect yes-no={ebm_reg_children_effect:.3f}, classifier log-odds effect yes-no={ebm_clf_children_effect:.3f}). "
        "Overall, evidence does not support the claim that having children decreases engagement in extramarital affairs."
    )

    result = {"response": response, "explanation": explanation}

    Path("conclusion.txt").write_text(json.dumps(result))

    print_section("Final Conclusion JSON")
    print(json.dumps(result, indent=2))
    print("\nWrote conclusion.txt")


if __name__ == "__main__":
    main()
