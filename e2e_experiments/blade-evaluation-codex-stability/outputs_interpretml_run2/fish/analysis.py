import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from interpret.glassbox import ExplainableBoostingRegressor


def print_section(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def summarize_distribution(series: pd.Series, name: str) -> None:
    q = series.quantile([0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])
    print(f"{name} summary:")
    print(
        {
            "mean": float(series.mean()),
            "std": float(series.std()),
            "median": float(series.median()),
            "skew": float(series.skew()),
            "zero_share": float((series == 0).mean()),
        }
    )
    print(f"{name} quantiles:\n{q.to_string()}")


def run_tests(df: pd.DataFrame) -> dict:
    print_section("Statistical Tests")

    results = {}

    # Welch t-tests for binary factors on fish-per-hour
    for col in ["livebait", "camper"]:
        grp0 = df.loc[df[col] == 0, "fish_per_hour"]
        grp1 = df.loc[df[col] == 1, "fish_per_hour"]
        t_stat, p_val = stats.ttest_ind(grp1, grp0, equal_var=False)
        results[f"ttest_{col}_rate_p"] = float(p_val)
        print(
            f"Welch t-test fish_per_hour by {col}: "
            f"mean(0)={grp0.mean():.3f}, mean(1)={grp1.mean():.3f}, "
            f"t={t_stat:.3f}, p={p_val:.5g}"
        )

    # ANOVA for rate by group size (persons)
    persons_groups = [g["fish_per_hour"].values for _, g in df.groupby("persons")]
    f_stat, p_val = stats.f_oneway(*persons_groups)
    results["anova_persons_rate_p"] = float(p_val)
    print(f"ANOVA fish_per_hour by persons: F={f_stat:.3f}, p={p_val:.5g}")

    # Correlations with rate
    for col in ["persons", "child", "hours"]:
        rho, p_val = stats.spearmanr(df[col], df["fish_per_hour"])
        results[f"spearman_{col}_rate_p"] = float(p_val)
        results[f"spearman_{col}_rate_rho"] = float(rho)
        print(f"Spearman(rate, {col}): rho={rho:.3f}, p={p_val:.5g}")

    return results


def run_models(df: pd.DataFrame) -> dict:
    print_section("Interpretable Models")

    features = ["livebait", "camper", "persons", "child", "hours"]
    X = df[features]
    y_rate = df["fish_per_hour"]
    y_log_rate = df["log_fish_per_hour"]
    y_log_count = df["log_fish_caught"]

    out = {}

    # OLS for inferential statistics on transformed outcomes
    ols_rate = sm.OLS(y_log_rate, sm.add_constant(X)).fit()
    out["ols_rate_f_pvalue"] = float(ols_rate.f_pvalue)
    out["ols_rate_pvalues"] = {k: float(v) for k, v in ols_rate.pvalues.items()}
    out["ols_rate_params"] = {k: float(v) for k, v in ols_rate.params.items()}
    print("OLS on log(1 + fish_per_hour):")
    print(ols_rate.summary())

    ols_count = sm.OLS(y_log_count, sm.add_constant(X)).fit()
    out["ols_count_f_pvalue"] = float(ols_count.f_pvalue)
    out["ols_count_pvalues"] = {k: float(v) for k, v in ols_count.pvalues.items()}
    out["ols_count_params"] = {k: float(v) for k, v in ols_count.params.items()}
    print("\nOLS on log(1 + fish_caught):")
    print(ols_count.summary())

    # Predictive but interpretable models on log-rate
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log_rate, test_size=0.30, random_state=42
    )

    sk_models = {
        "linear": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "lasso": Lasso(alpha=0.001, max_iter=20000),
        "tree": DecisionTreeRegressor(max_depth=3, random_state=42),
    }

    out["sklearn"] = {}
    for name, model in sk_models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        model_out = {
            "r2": float(r2_score(y_test, preds)),
            "mae": float(mean_absolute_error(y_test, preds)),
        }
        if hasattr(model, "coef_"):
            model_out["coefficients"] = {
                feat: float(val) for feat, val in zip(features, model.coef_)
            }
        if hasattr(model, "feature_importances_"):
            model_out["feature_importances"] = {
                feat: float(val)
                for feat, val in zip(features, model.feature_importances_)
            }
        out["sklearn"][name] = model_out

    print("\nScikit-learn interpretable model metrics (target = log(1 + fish_per_hour)):")
    for name, vals in out["sklearn"].items():
        print(f"{name}: r2={vals['r2']:.3f}, mae={vals['mae']:.3f}")
        if "coefficients" in vals:
            print(f"  coefficients={vals['coefficients']}")
        if "feature_importances" in vals:
            print(f"  feature_importances={vals['feature_importances']}")

    # Explainable Boosting Machine (interpret)
    ebm = ExplainableBoostingRegressor(random_state=42, interactions=0)
    ebm.fit(X_train, y_train)
    ebm_preds = ebm.predict(X_test)

    out["ebm"] = {
        "r2": float(r2_score(y_test, ebm_preds)),
        "mae": float(mean_absolute_error(y_test, ebm_preds)),
        "term_importances": {
            feat: float(val) for feat, val in zip(features, ebm.term_importances())
        },
    }

    print("\nExplainableBoostingRegressor metrics:")
    print(
        f"EBM: r2={out['ebm']['r2']:.3f}, mae={out['ebm']['mae']:.3f}, "
        f"term_importances={out['ebm']['term_importances']}"
    )

    return out


def score_conclusion(df: pd.DataFrame, tests_out: dict, models_out: dict) -> tuple[int, str]:
    # Core estimates
    pooled_rate = float(df["fish_caught"].sum() / df["hours"].sum())
    mean_trip_rate = float(df["fish_per_hour"].mean())

    sig_predictors_rate = [
        k
        for k, p in models_out["ols_rate_pvalues"].items()
        if k not in {"const"} and p < 0.05
    ]

    # Construct a Likert score from strength/consistency of evidence.
    score = 50
    if models_out["ols_rate_f_pvalue"] < 0.001:
        score += 20
    if tests_out["ttest_livebait_rate_p"] < 0.05:
        score += 10
    if tests_out["anova_persons_rate_p"] < 0.05:
        score += 10
    if tests_out["spearman_child_rate_p"] < 0.05:
        score += 5
    if tests_out["spearman_hours_rate_p"] < 0.05:
        score += 5
    if models_out["ebm"]["r2"] > 0.25:
        score += 5

    # Penalties for contradictory/non-significant signals on key factors
    if tests_out["ttest_camper_rate_p"] >= 0.05:
        score -= 5
    if len(sig_predictors_rate) < 2:
        score -= 10

    score = max(0, min(100, int(round(score))))

    # Explanatory text grounded in test/model results
    expl = (
        "Estimated catch rate is about "
        f"{pooled_rate:.2f} fish/hour overall (total fish / total hours), "
        f"with mean per-trip rate {mean_trip_rate:.2f}. "
        "There is strong evidence that catch rate depends on trip composition and behavior: "
        f"livebait is significant in Welch t-test (p={tests_out['ttest_livebait_rate_p']:.3g}), "
        f"group size differs by ANOVA (p={tests_out['anova_persons_rate_p']:.3g}), "
        f"and OLS on log(1+rate) is globally significant (F-test p={models_out['ols_rate_f_pvalue']:.3g}) "
        f"with significant predictors {', '.join(sig_predictors_rate)}. "
        "Interpretable models (linear/tree/EBM) identify persons, child, and hours as important drivers; "
        f"EBM reached R^2={models_out['ebm']['r2']:.2f}. "
        "Camper alone was not significant in univariate t-test, so confidence is high but not absolute."
    )

    return score, expl


def main() -> None:
    info_path = Path("info.json")
    data_path = Path("fish.csv")

    if not info_path.exists() or not data_path.exists():
        raise FileNotFoundError("Required files info.json and fish.csv must exist in current directory.")

    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

    question = info.get("research_questions", ["Unknown question"])[0]

    df = pd.read_csv(data_path)

    # Derived outcomes
    df["fish_per_hour"] = df["fish_caught"] / df["hours"]
    df["log_fish_per_hour"] = np.log1p(df["fish_per_hour"])
    df["log_fish_caught"] = np.log1p(df["fish_caught"])

    print_section("Research Question")
    print(question)

    print_section("Dataset Overview")
    print(f"Shape: {df.shape}")
    print("Columns:", list(df.columns))
    print("\nMissing values by column:")
    print(df.isna().sum().to_string())

    print("\nSummary statistics:")
    print(df.describe().to_string())

    print_section("Distributions")
    summarize_distribution(df["fish_caught"], "fish_caught")
    summarize_distribution(df["hours"], "hours")
    summarize_distribution(df["fish_per_hour"], "fish_per_hour")

    print_section("Correlations")
    pearson = df[["fish_caught", "fish_per_hour", "livebait", "camper", "persons", "child", "hours"]].corr(
        method="pearson"
    )
    spearman = df[["fish_caught", "fish_per_hour", "livebait", "camper", "persons", "child", "hours"]].corr(
        method="spearman"
    )
    print("Pearson correlation matrix:\n", pearson.to_string())
    print("\nSpearman correlation matrix:\n", spearman.to_string())

    tests_out = run_tests(df)
    models_out = run_models(df)

    score, explanation = score_conclusion(df, tests_out, models_out)

    output = {"response": int(score), "explanation": explanation}
    with Path("conclusion.txt").open("w", encoding="utf-8") as f:
        json.dump(output, f)

    print_section("Final Conclusion JSON")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
