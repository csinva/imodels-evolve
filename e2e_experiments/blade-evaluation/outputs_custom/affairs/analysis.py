import json
import re
import warnings

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from interp_models import HingeEBMRegressor, SmartAdditiveRegressor

warnings.filterwarnings("ignore")


def print_section(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def parse_custom_model_child_effect(model_text: str, child_index: int = 0):
    child_token = f"x{child_index}"

    # If model explicitly excludes x0, that is direct evidence of no independent effect.
    excluded_match = re.search(r"Features with zero coefficients \(excluded\):\s*(.*)", model_text)
    if excluded_match:
        excluded_text = excluded_match.group(1)
        excluded_set = {s.strip() for s in excluded_text.split(",")}
        if child_token in excluded_set:
            return {"status": "excluded", "coefficient": 0.0}

    # Otherwise, try to pull a linear coefficient term for x0.
    linear_match = re.search(rf"([+-]?\d+\.\d+)\*{child_token}(?!\d)", model_text)
    if linear_match:
        coef = float(linear_match.group(1))
        return {"status": "linear_term", "coefficient": coef}

    # Could still be nonlinear only; return unknown if not explicit.
    return {"status": "not_explicit", "coefficient": np.nan}


def main() -> None:
    print_section("1) LOAD RESEARCH CONTEXT")
    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    research_question = info["research_questions"][0].strip()
    print("Research question:")
    print(f"- {research_question}")

    df = pd.read_csv("affairs.csv")
    print(f"\nDataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")

    print_section("2) BASIC EDA")
    print("Missing values per column:")
    print(df.isna().sum().to_string())

    numeric_cols = [
        "affairs",
        "age",
        "yearsmarried",
        "religiousness",
        "education",
        "occupation",
        "rating",
    ]
    print("\nSummary statistics (numeric columns):")
    print(df[numeric_cols].describe().T.to_string(float_format=lambda x: f"{x:0.3f}"))

    print("\nCategorical distributions:")
    for col in ["children", "gender"]:
        print(f"\n{col}:")
        print(df[col].value_counts(dropna=False).to_string())

    print("\nTarget distribution (`affairs`):")
    print(df["affairs"].value_counts().sort_index().to_string())
    zero_rate = (df["affairs"] == 0).mean()
    print(f"Proportion with zero reported affairs: {zero_rate:0.3f}")

    # Modeling-ready encoded dataframe.
    dfe = df.copy()
    dfe["children_yes"] = (dfe["children"] == "yes").astype(int)
    dfe["gender_male"] = (dfe["gender"] == "male").astype(int)
    dfe["has_affair"] = (dfe["affairs"] > 0).astype(int)

    corr_cols = [
        "affairs",
        "children_yes",
        "gender_male",
        "age",
        "yearsmarried",
        "religiousness",
        "education",
        "occupation",
        "rating",
    ]
    corr = dfe[corr_cols].corr(numeric_only=True)
    print("\nCorrelation with `affairs`:")
    print(corr["affairs"].sort_values(ascending=False).to_string(float_format=lambda x: f"{x:0.3f}"))

    print_section("3) STATISTICAL TESTS")

    with_children = dfe.loc[dfe["children_yes"] == 1, "affairs"]
    without_children = dfe.loc[dfe["children_yes"] == 0, "affairs"]
    mean_yes = with_children.mean()
    mean_no = without_children.mean()

    print("Affairs by children status:")
    print(dfe.groupby("children")["affairs"].agg(["count", "mean", "std"]).to_string(float_format=lambda x: f"{x:0.3f}"))

    welch = stats.ttest_ind(with_children, without_children, equal_var=False)
    print(
        f"\nWelch t-test (mean difference): t={welch.statistic:0.3f}, p={welch.pvalue:0.5f}"
    )

    mann = stats.mannwhitneyu(with_children, without_children, alternative="two-sided")
    print(
        f"Mann-Whitney U test (distribution shift): U={mann.statistic:0.1f}, p={mann.pvalue:0.5f}"
    )

    # ANOVA (equivalent one-way group comparison).
    anova_model = smf.ols("affairs ~ C(children)", data=dfe).fit()
    anova_table = sm.stats.anova_lm(anova_model, typ=2)
    anova_p = float(anova_table.loc["C(children)", "PR(>F)"])
    print("\nOne-way ANOVA for `affairs ~ children`:")
    print(anova_table.to_string(float_format=lambda x: f"{x:0.5f}"))

    # Chi-square for having any affair.
    contingency = pd.crosstab(dfe["children"], dfe["has_affair"])
    chi2, chi2_p, _, _ = stats.chi2_contingency(contingency)
    a = contingency.loc["yes", 1]
    b = contingency.loc["yes", 0]
    c = contingency.loc["no", 1]
    d = contingency.loc["no", 0]
    odds_ratio = (a * d) / (b * c) if all(v > 0 for v in [a, b, c, d]) else np.nan
    print("\nContingency table: children vs any affair")
    print(contingency.to_string())
    print(f"Chi-square test: chi2={chi2:0.3f}, p={chi2_p:0.5f}, odds_ratio_yes_vs_no={odds_ratio:0.3f}")

    # Multivariable OLS with robust SE.
    controls_formula = (
        "affairs ~ children_yes + age + yearsmarried + religiousness + "
        "education + occupation + rating + C(gender)"
    )
    ols = smf.ols(controls_formula, data=dfe).fit(cov_type="HC3")
    ols_child_coef = float(ols.params["children_yes"])
    ols_child_p = float(ols.pvalues["children_yes"])
    print("\nOLS with controls (HC3 robust SE):")
    print(ols.summary().tables[1])

    # Poisson GLM as count-data robustness check.
    pois = smf.glm(
        controls_formula,
        data=dfe,
        family=sm.families.Poisson(),
    ).fit(cov_type="HC3")
    pois_child_coef = float(pois.params["children_yes"])
    pois_child_p = float(pois.pvalues["children_yes"])
    print("\nPoisson GLM with controls (HC3 robust SE):")
    print(pois.summary().tables[1])

    print_section("4) INTERPRETABLE MODELS (CUSTOM + STANDARD)")

    feature_cols = [
        "children_yes",
        "gender_male",
        "age",
        "yearsmarried",
        "religiousness",
        "education",
        "occupation",
        "rating",
    ]
    X = dfe[feature_cols].astype(float).values
    y = dfe["affairs"].astype(float).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    print("Feature index mapping for custom model output:")
    for i, feat in enumerate(feature_cols):
        print(f"  x{i} -> {feat}")

    models = {
        "SmartAdditiveRegressor": SmartAdditiveRegressor(
            n_rounds=300, learning_rate=0.05, min_samples_leaf=10
        ),
        "HingeEBMRegressor": HingeEBMRegressor(
            n_knots=3, max_input_features=8, ebm_outer_bags=2, ebm_max_rounds=300
        ),
        "LinearRegression": LinearRegression(),
        "LassoCV": LassoCV(cv=5, random_state=42, max_iter=10000),
        "DecisionTreeRegressor": DecisionTreeRegressor(
            max_depth=3, min_samples_leaf=20, random_state=42
        ),
    }

    # Optional imodels model (available per prompt).
    try:
        from imodels import RuleFitRegressor

        models["RuleFitRegressor"] = RuleFitRegressor(random_state=42)
    except Exception as e:
        print(f"RuleFitRegressor unavailable or failed to import: {e}")

    model_results = {}
    custom_child_readouts = {}

    for name, model in models.items():
        print(f"\n--- {name} ---")
        try:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            r2 = r2_score(y_test, pred)
            mae = mean_absolute_error(y_test, pred)
            model_results[name] = {"r2": float(r2), "mae": float(mae)}
            print(f"Test R^2: {r2:0.4f}")
            print(f"Test MAE: {mae:0.4f}")

            if hasattr(model, "coef_"):
                coef = np.ravel(model.coef_)
                if coef.shape[0] >= 1:
                    print(f"children_yes coefficient (if linear): {coef[0]:0.4f}")
                    model_results[name]["child_coef"] = float(coef[0])

            if hasattr(model, "feature_importances_"):
                fi = np.ravel(model.feature_importances_)
                if fi.shape[0] >= 1:
                    print(f"children_yes feature importance: {fi[0]:0.4f}")
                    model_results[name]["child_importance"] = float(fi[0])

            if name in {"SmartAdditiveRegressor", "HingeEBMRegressor"}:
                model_text = str(model)
                print("Model interpretation:")
                print(model_text)
                readout = parse_custom_model_child_effect(model_text, child_index=0)
                custom_child_readouts[name] = readout
                print(f"children_yes custom readout: {readout}")

        except Exception as e:
            print(f"Model {name} failed: {e}")
            model_results[name] = {"error": str(e)}

    print("\nModel summary table:")
    print(pd.DataFrame(model_results).T.to_string(float_format=lambda x: f"{x:0.4f}"))

    print_section("5) CONCLUSION FOR RESEARCH QUESTION")

    evidence_for_decrease = 0.0
    evidence_against_decrease = 0.0

    # Unadjusted mean comparison
    if welch.pvalue < 0.05:
        if mean_yes < mean_no:
            evidence_for_decrease += 2.0
        else:
            evidence_against_decrease += 2.0

    if mann.pvalue < 0.05:
        if mean_yes < mean_no:
            evidence_for_decrease += 1.0
        else:
            evidence_against_decrease += 1.0

    if anova_p < 0.05:
        if mean_yes < mean_no:
            evidence_for_decrease += 1.0
        else:
            evidence_against_decrease += 1.0

    if chi2_p < 0.05:
        if odds_ratio < 1.0:
            evidence_for_decrease += 2.0
        else:
            evidence_against_decrease += 2.0

    # Adjusted regression evidence.
    if ols_child_p < 0.05:
        if ols_child_coef < 0:
            evidence_for_decrease += 2.0
        else:
            evidence_against_decrease += 2.0

    if pois_child_p < 0.05:
        if pois_child_coef < 0:
            evidence_for_decrease += 2.0
        else:
            evidence_against_decrease += 2.0

    # Custom model readouts.
    for readout in custom_child_readouts.values():
        status = readout.get("status")
        coef = readout.get("coefficient")
        if status == "linear_term" and np.isfinite(coef):
            if coef < 0:
                evidence_for_decrease += 1.0
            elif coef > 0:
                evidence_against_decrease += 1.0
        elif status == "excluded":
            # Excluded suggests no meaningful independent effect; not support for decrease.
            evidence_against_decrease += 0.5

    # Convert evidence to 0-100 Likert score where higher means stronger YES.
    # 50 = neutral; lower means evidence against "decrease".
    raw_score = 50.0 + 10.0 * (evidence_for_decrease - evidence_against_decrease)
    response_score = int(np.clip(np.round(raw_score), 0, 100))

    explanation = (
        f"The data do not support that having children decreases extramarital affairs. "
        f"Unadjusted comparisons show higher affair engagement among those with children "
        f"(mean {mean_yes:0.2f} vs {mean_no:0.2f}; Welch t-test p={welch.pvalue:0.5f}; "
        f"Mann-Whitney p={mann.pvalue:0.5f}; ANOVA p={anova_p:0.5f}; chi-square p={chi2_p:0.5f}, "
        f"odds ratio={odds_ratio:0.2f}). After adjustment for age, years married, religiousness, "
        f"education, occupation, rating, and gender, the children effect is near zero and not significant "
        f"(OLS coef={ols_child_coef:0.3f}, p={ols_child_p:0.3f}; Poisson coef={pois_child_coef:0.3f}, "
        f"p={pois_child_p:0.3f}). In the custom interpretable models, children is excluded or not an active "
        f"driver relative to stronger predictors, so independent evidence for a decreasing effect is weak."
    )

    result = {
        "response": response_score,
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=True)

    print(f"Likert response (0-100): {response_score}")
    print("Wrote conclusion.txt")


if __name__ == "__main__":
    main()
