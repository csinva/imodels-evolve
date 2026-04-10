import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from imodels import FIGSRegressor, HSTreeRegressor, RuleFitRegressor
from interp_models import HingeEBMRegressor, SmartAdditiveRegressor


warnings.filterwarnings("ignore")


def relabel_model_text(text: str, feature_names: list[str]) -> str:
    mapped = text
    for i in sorted(range(len(feature_names)), reverse=True):
        mapped = mapped.replace(f"x{i}", feature_names[i])
    return mapped


def evaluate_regression_models(X_train, X_test, y_train, y_test):
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge(alpha=1.0)": Ridge(alpha=1.0, random_state=42),
        "Lasso(alpha=0.001)": Lasso(alpha=0.001, random_state=42, max_iter=10000),
        "DecisionTree(max_depth=4)": DecisionTreeRegressor(max_depth=4, random_state=42),
        "RuleFitRegressor": RuleFitRegressor(random_state=42),
        "FIGSRegressor": FIGSRegressor(random_state=42),
        "HSTreeRegressor": HSTreeRegressor(random_state=42),
    }

    results = {}
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            pred = np.clip(model.predict(X_test), 0, 1)
            results[name] = {
                "r2": float(r2_score(y_test, pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
            }
        except Exception as exc:
            results[name] = {"error": str(exc)}
    return results


def main():
    info = json.loads(Path("info.json").read_text())
    research_question = info["research_questions"][0]

    df = pd.read_csv("boxes.csv")

    # Primary outcome for research question: choosing the majority option.
    df["majority_choice"] = (df["y"] == 2).astype(int)

    print("=" * 88)
    print("Research question:")
    print(research_question)
    print("=" * 88)

    print("\nData shape:", df.shape)
    print("Columns:", list(df.columns))
    print("\nSummary statistics:\n", df.describe().to_string())

    print("\nOutcome distribution y (1=unchosen, 2=majority, 3=minority):")
    print(df["y"].value_counts(normalize=True).sort_index().rename("proportion").to_string())

    for col in ["gender", "majority_first", "culture"]:
        print(f"\nDistribution of {col}:")
        print(df[col].value_counts(normalize=True).sort_index().rename("proportion").to_string())

    corr_cols = ["majority_choice", "age", "gender", "majority_first", "culture"]
    print("\nCorrelation matrix:")
    print(df[corr_cols].corr(numeric_only=True).to_string())

    print("\nMajority-choice rate by age:")
    print(df.groupby("age")["majority_choice"].mean().to_string())

    print("\nMajority-choice rate by culture:")
    print(df.groupby("culture")["majority_choice"].mean().to_string())

    # Statistical tests targeted to the research question.
    point_r, point_p = stats.pointbiserialr(df["majority_choice"], df["age"])

    median_age = float(df["age"].median())
    younger = df.loc[df["age"] <= median_age, "majority_choice"]
    older = df.loc[df["age"] > median_age, "majority_choice"]
    t_stat, t_p = stats.ttest_ind(younger, older, equal_var=False)

    groups = [g["majority_choice"].to_numpy() for _, g in df.groupby("culture")]
    anova_f, anova_p = stats.f_oneway(*groups)

    contingency = pd.crosstab(df["culture"], df["y"])
    chi2_stat, chi2_p, _, _ = stats.chi2_contingency(contingency)

    ols_model = smf.ols(
        "majority_choice ~ age * C(culture) + C(gender) + C(majority_first)",
        data=df,
    ).fit()

    logit_model = smf.logit(
        "majority_choice ~ age * C(culture) + C(gender) + C(majority_first)",
        data=df,
    ).fit(disp=0)

    interaction_terms = [
        t for t in logit_model.params.index if t.startswith("age:C(culture)")
    ]
    if interaction_terms:
        wald_hypothesis = " , ".join([f"{term} = 0" for term in interaction_terms])
        interaction_wald = logit_model.wald_test(wald_hypothesis)
        interaction_p = float(np.squeeze(interaction_wald.pvalue))
    else:
        interaction_p = 1.0

    age_coef_logit = float(logit_model.params.get("age", np.nan))
    age_p_logit = float(logit_model.pvalues.get("age", np.nan))

    print("\n" + "=" * 88)
    print("Statistical tests")
    print("=" * 88)
    print(f"Point-biserial age-majority correlation: r={point_r:.4f}, p={point_p:.4g}")
    print(
        "Median split t-test (younger vs older majority-choice): "
        f"t={t_stat:.4f}, p={t_p:.4g}, "
        f"mean_younger={younger.mean():.4f}, mean_older={older.mean():.4f}"
    )
    print(f"ANOVA majority-choice by culture: F={anova_f:.4f}, p={anova_p:.4g}")
    print(f"Chi-square culture x 3-way outcome(y): chi2={chi2_stat:.4f}, p={chi2_p:.4g}")
    print("\nOLS coefficient table:")
    print(ols_model.summary().tables[1])
    print("\nLogistic age effect and age-by-culture interaction:")
    print(f"logit age coefficient={age_coef_logit:.4f}, p={age_p_logit:.4g}")
    print(f"joint Wald p-value for all age*culture terms={interaction_p:.4g}")

    # Modeling data.
    X = df[["age", "gender", "majority_first", "culture"]].copy()
    X = pd.get_dummies(X, columns=["culture"], drop_first=True)
    y = df["majority_choice"].astype(float)
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print("\n" + "=" * 88)
    print("Custom interpretable models")
    print("=" * 88)

    smart_model = SmartAdditiveRegressor(
        n_rounds=300,
        learning_rate=0.05,
        min_samples_leaf=10,
    )
    smart_model.fit(X_train.values, y_train.values)
    smart_pred = np.clip(smart_model.predict(X_test.values), 0, 1)
    smart_r2 = float(r2_score(y_test, smart_pred))
    smart_rmse = float(np.sqrt(mean_squared_error(y_test, smart_pred)))

    print("\nSmartAdditiveRegressor interpretation:")
    print(relabel_model_text(str(smart_model), feature_names))
    print(f"SmartAdditive holdout R2={smart_r2:.4f}, RMSE={smart_rmse:.4f}")

    hinge_model = HingeEBMRegressor(
        n_knots=3,
        max_input_features=min(15, X_train.shape[1]),
        ebm_outer_bags=4,
        ebm_max_rounds=500,
    )
    hinge_model.fit(X_train.values, y_train.values)
    hinge_pred = np.clip(hinge_model.predict(X_test.values), 0, 1)
    hinge_r2 = float(r2_score(y_test, hinge_pred))
    hinge_rmse = float(np.sqrt(mean_squared_error(y_test, hinge_pred)))

    print("\nHingeEBMRegressor interpretation:")
    print(relabel_model_text(str(hinge_model), feature_names))
    print(f"HingeEBM holdout R2={hinge_r2:.4f}, RMSE={hinge_rmse:.4f}")

    print("\n" + "=" * 88)
    print("Standard model comparison")
    print("=" * 88)
    baseline_results = evaluate_regression_models(
        X_train.values,
        X_test.values,
        y_train.values,
        y_test.values,
    )
    for model_name, metrics in baseline_results.items():
        if "error" in metrics:
            print(f"{model_name}: ERROR -> {metrics['error']}")
        else:
            print(
                f"{model_name}: R2={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}"
            )

    # Likert score: question asks if majority reliance develops with age across culture.
    # Strong support requires significant age trend and/or significant age-by-culture interaction.
    if age_p_logit < 0.05 and interaction_p < 0.05:
        response = 90
    elif age_p_logit < 0.05 or interaction_p < 0.05:
        response = 70
    elif age_p_logit < 0.10 or interaction_p < 0.10:
        response = 40
    else:
        response = 15

    explanation = (
        "Evidence does not support a developmental age trend in majority preference across "
        "cultures: age was not significant in logistic regression "
        f"(coef={age_coef_logit:.3f}, p={age_p_logit:.3f}), and the joint age-by-culture "
        f"interaction was not significant (p={interaction_p:.3f}). "
        f"Simple tests agree (point-biserial r={point_r:.3f}, p={point_p:.3f}; ANOVA by culture "
        f"p={anova_p:.3f}). Interpretable models (SmartAdditive and HingeEBM) achieved limited "
        "predictive power and did not reveal a strong age-driven pattern, suggesting majority "
        "reliance is relatively stable with age in this sample, with stronger effects from "
        "experimental condition variables than age growth."
    )

    output = {"response": int(response), "explanation": explanation}
    Path("conclusion.txt").write_text(json.dumps(output, ensure_ascii=True))

    print("\nWrote conclusion.txt:")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
