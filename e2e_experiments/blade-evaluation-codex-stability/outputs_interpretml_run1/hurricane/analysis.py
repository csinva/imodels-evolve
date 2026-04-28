import json
from pathlib import Path

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import statsmodels.api as sm
import statsmodels.formula.api as smf


def to_float(value):
    try:
        return float(value)
    except Exception:
        return float("nan")


def clamp_int(x, lo=0, hi=100):
    return int(max(lo, min(hi, round(x))))


def main():
    info = json.loads(Path("info.json").read_text())
    research_question = info.get("research_questions", [""])[0]

    df = pd.read_csv("hurricane.csv")

    # Basic feature engineering for skewed outcomes and damage.
    df["log_alldeaths"] = np.log1p(df["alldeaths"])
    df["log_ndam15"] = np.log1p(df["ndam15"])
    df["high_death"] = (df["alldeaths"] >= df["alldeaths"].quantile(0.75)).astype(int)

    key_numeric = [
        "masfem",
        "gender_mf",
        "min",
        "wind",
        "category",
        "alldeaths",
        "log_alldeaths",
        "ndam15",
        "log_ndam15",
        "year",
        "elapsedyrs",
        "masfem_mturk",
    ]

    exploration = {
        "shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
        "research_question": research_question,
        "missing_values": df[key_numeric].isna().sum().to_dict(),
        "summary_statistics": df[key_numeric].describe().to_dict(),
        "distribution_stats": {
            col: {
                "mean": to_float(df[col].mean()),
                "median": to_float(df[col].median()),
                "std": to_float(df[col].std()),
                "skew": to_float(stats.skew(df[col].dropna())),
            }
            for col in ["masfem", "alldeaths", "log_alldeaths", "ndam15", "log_ndam15"]
        },
        "pearson_correlation": df[key_numeric].corr(method="pearson").to_dict(),
        "spearman_correlation": df[key_numeric].corr(method="spearman").to_dict(),
    }

    # Statistical tests on the core hypothesis.
    pearson_log = stats.pearsonr(df["masfem"], df["log_alldeaths"])
    spearman_log = stats.spearmanr(df["masfem"], df["log_alldeaths"])

    female = df.loc[df["gender_mf"] == 1, "log_alldeaths"]
    male = df.loc[df["gender_mf"] == 0, "log_alldeaths"]
    ttest_gender = stats.ttest_ind(female, male, equal_var=False, nan_policy="omit")

    anova_model = smf.ols("log_alldeaths ~ C(category)", data=df).fit()
    anova_table = sm.stats.anova_lm(anova_model, typ=2)

    ols_main = smf.ols(
        "log_alldeaths ~ masfem + min + wind + category + log_ndam15 + year",
        data=df,
    ).fit()

    ols_gender = smf.ols(
        "log_alldeaths ~ gender_mf + min + wind + category + log_ndam15 + year",
        data=df,
    ).fit()

    ols_interaction = smf.ols(
        "log_alldeaths ~ masfem * category + min + wind + log_ndam15 + year",
        data=df,
    ).fit()

    nb_main = smf.glm(
        "alldeaths ~ masfem + min + wind + category + log_ndam15 + year",
        data=df,
        family=sm.families.NegativeBinomial(),
    ).fit()

    nb_interaction = smf.glm(
        "alldeaths ~ masfem * category + min + wind + log_ndam15 + year",
        data=df,
        family=sm.families.NegativeBinomial(),
    ).fit()

    stats_results = {
        "pearson_masfem_logdeaths": {
            "r": to_float(pearson_log.statistic),
            "p_value": to_float(pearson_log.pvalue),
        },
        "spearman_masfem_logdeaths": {
            "rho": to_float(spearman_log.statistic),
            "p_value": to_float(spearman_log.pvalue),
        },
        "ttest_female_vs_male_logdeaths": {
            "t_stat": to_float(ttest_gender.statistic),
            "p_value": to_float(ttest_gender.pvalue),
            "female_mean": to_float(female.mean()),
            "male_mean": to_float(male.mean()),
        },
        "anova_category_on_logdeaths": anova_table.to_dict(),
        "ols_main": {
            "r_squared": to_float(ols_main.rsquared),
            "coef_masfem": to_float(ols_main.params.get("masfem", np.nan)),
            "p_masfem": to_float(ols_main.pvalues.get("masfem", np.nan)),
        },
        "ols_gender": {
            "r_squared": to_float(ols_gender.rsquared),
            "coef_gender_mf": to_float(ols_gender.params.get("gender_mf", np.nan)),
            "p_gender_mf": to_float(ols_gender.pvalues.get("gender_mf", np.nan)),
        },
        "ols_interaction": {
            "coef_masfem_category": to_float(ols_interaction.params.get("masfem:category", np.nan)),
            "p_masfem_category": to_float(ols_interaction.pvalues.get("masfem:category", np.nan)),
        },
        "nb_main": {
            "coef_masfem": to_float(nb_main.params.get("masfem", np.nan)),
            "p_masfem": to_float(nb_main.pvalues.get("masfem", np.nan)),
        },
        "nb_interaction": {
            "coef_masfem_category": to_float(nb_interaction.params.get("masfem:category", np.nan)),
            "p_masfem_category": to_float(nb_interaction.pvalues.get("masfem:category", np.nan)),
        },
    }

    # Interpretable ML models.
    features = [
        "masfem",
        "gender_mf",
        "min",
        "wind",
        "category",
        "log_ndam15",
        "year",
        "elapsedyrs",
        "masfem_mturk",
    ]

    X = df[features].copy()
    y_reg = df["log_alldeaths"].copy()
    y_cls = df["high_death"].copy()

    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=features)

    X_train, X_test, y_train, y_test = train_test_split(
        X_imp, y_reg, test_size=0.30, random_state=42
    )

    reg_models = {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=1.0, random_state=42),
        "lasso": Lasso(alpha=0.001, random_state=42, max_iter=20000),
        "decision_tree_regressor": DecisionTreeRegressor(max_depth=3, random_state=42),
        "explainable_boosting_regressor": ExplainableBoostingRegressor(
            random_state=42, interactions=0, max_rounds=300
        ),
    }

    reg_results = {}
    for name, model in reg_models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        item = {"test_r2": to_float(r2_score(y_test, pred))}

        if hasattr(model, "coef_"):
            item["coefficients"] = {
                feat: to_float(val) for feat, val in zip(features, model.coef_)
            }
        if hasattr(model, "feature_importances_"):
            item["feature_importances"] = {
                feat: to_float(val)
                for feat, val in zip(features, model.feature_importances_)
            }
        if name == "explainable_boosting_regressor":
            gexp = model.explain_global().data()
            item["global_importance"] = {
                n: to_float(s) for n, s in zip(gexp["names"], gexp["scores"])
            }

        reg_results[name] = item

    # Interpretable classifiers on high-death indicator.
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        X_imp, y_cls, test_size=0.30, random_state=42, stratify=y_cls
    )

    clf_models = {
        "decision_tree_classifier": DecisionTreeClassifier(max_depth=3, random_state=42),
        "explainable_boosting_classifier": ExplainableBoostingClassifier(
            random_state=42, interactions=0, max_rounds=300
        ),
    }

    clf_results = {}
    for name, model in clf_models.items():
        model.fit(Xc_train, yc_train)
        pred = model.predict(Xc_test)
        item = {"test_accuracy": to_float(accuracy_score(yc_test, pred))}

        if hasattr(model, "feature_importances_"):
            item["feature_importances"] = {
                feat: to_float(val)
                for feat, val in zip(features, model.feature_importances_)
            }
        if name == "explainable_boosting_classifier":
            gexp = model.explain_global().data()
            item["global_importance"] = {
                n: to_float(s) for n, s in zip(gexp["names"], gexp["scores"])
            }

        clf_results[name] = item

    # Hypothesis scoring: emphasize significance of direct femininity effect.
    direct_pvalues = [
        stats_results["pearson_masfem_logdeaths"]["p_value"],
        stats_results["ttest_female_vs_male_logdeaths"]["p_value"],
        stats_results["ols_main"]["p_masfem"],
        stats_results["nb_main"]["p_masfem"],
    ]
    significant_direct = sum(p < 0.05 for p in direct_pvalues)
    borderline_direct = sum((p >= 0.05) and (p < 0.10) for p in direct_pvalues)

    interaction_pvalues = [
        stats_results["ols_interaction"]["p_masfem_category"],
        stats_results["nb_interaction"]["p_masfem_category"],
    ]
    significant_interactions = sum(p < 0.05 for p in interaction_pvalues)

    ebm_imp = reg_results["explainable_boosting_regressor"].get("global_importance", {})
    sorted_imp = sorted(ebm_imp.items(), key=lambda x: x[1], reverse=True)
    masfem_rank = next(
        (idx + 1 for idx, (name, _) in enumerate(sorted_imp) if name == "masfem"),
        len(sorted_imp) + 1,
    )

    score = 25 + 15 * significant_direct + 5 * borderline_direct + 8 * significant_interactions
    if significant_direct == 0 and significant_interactions == 0:
        score -= 10
    if masfem_rank > 5:
        score -= 5
    response_score = clamp_int(score)

    explanation = (
        f"Question: {research_question} "
        f"Direct tests do not show a statistically significant main effect of name femininity on fatalities "
        f"(Pearson p={stats_results['pearson_masfem_logdeaths']['p_value']:.3f}, "
        f"female-vs-male t-test p={stats_results['ttest_female_vs_male_logdeaths']['p_value']:.3f}, "
        f"OLS adjusted p={stats_results['ols_main']['p_masfem']:.3f}, "
        f"Negative Binomial adjusted p={stats_results['nb_main']['p_masfem']:.3f}). "
        f"There is some interaction evidence with storm category in the Negative Binomial model "
        f"(p={stats_results['nb_interaction']['p_masfem_category']:.3f}), but this is not a stable overall main effect. "
        f"Interpretable models prioritize damage/intensity features (e.g., log_ndam15, pressure) over masfem, "
        f"so the dataset provides weak and model-dependent support for the hypothesis."
    )

    # Optional full report for reproducibility.
    report = {
        "exploration": exploration,
        "statistical_tests": stats_results,
        "interpretable_regression_models": reg_results,
        "interpretable_classification_models": clf_results,
        "masfem_importance_rank_in_ebm": masfem_rank,
        "response": response_score,
        "explanation": explanation,
    }
    Path("analysis_report.json").write_text(json.dumps(report, indent=2))

    # Required output format.
    conclusion = {"response": response_score, "explanation": explanation}
    Path("conclusion.txt").write_text(json.dumps(conclusion))

    print("Analysis complete.")
    print(f"response={response_score}")


if __name__ == "__main__":
    main()
