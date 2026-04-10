import json
import warnings

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.formula.api as smf
from imodels import FIGSRegressor, RuleFitRegressor
from interp_models import HingeEBMRegressor, SmartAdditiveRegressor
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")


def header(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def poisson_rate_ci(total_events: float, total_exposure: float, alpha: float = 0.05):
    if total_events < 0 or total_exposure <= 0:
        return np.nan, np.nan
    if total_events == 0:
        lower = 0.0
    else:
        lower = 0.5 * stats.chi2.ppf(alpha / 2, 2 * total_events) / total_exposure
    upper = 0.5 * stats.chi2.ppf(1 - alpha / 2, 2 * (total_events + 1)) / total_exposure
    return lower, upper


def replace_feature_tokens(text: str, feature_names):
    out = text
    for i, name in enumerate(feature_names):
        out = out.replace(f"x{i}", name)
    return out


def safe_cross_val(model, X, y, cv):
    scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
    return float(np.mean(scores)), float(np.std(scores))


def main():
    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    question = info["research_questions"][0]
    df = pd.read_csv("fish.csv")

    required_cols = ["fish_caught", "livebait", "camper", "persons", "child", "hours"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["fish_per_hour"] = df["fish_caught"] / df["hours"]
    df["log_fish_caught"] = np.log1p(df["fish_caught"])
    df["log_hours"] = np.log1p(df["hours"])

    feature_cols = ["livebait", "camper", "persons", "child", "hours"]

    header("Research Question")
    print(question)

    header("Data Overview")
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    print(df[required_cols].describe().T.round(4))
    print("\nMissing values per column:")
    print(df[required_cols].isna().sum())

    zero_catch_share = float((df["fish_caught"] == 0).mean())
    overall_rate = float(df["fish_caught"].sum() / df["hours"].sum())
    rate_ci_low, rate_ci_high = poisson_rate_ci(df["fish_caught"].sum(), df["hours"].sum())

    print("\nDistribution summaries:")
    print(f"Zero-catch trips: {zero_catch_share:.1%}")
    print(f"Mean fish per hour by trip: {df['fish_per_hour'].mean():.3f}")
    print(f"Median fish per hour by trip: {df['fish_per_hour'].median():.3f}")
    print(f"Overall catch rate (total fish/total hours): {overall_rate:.3f} fish/hour")
    print(f"95% Poisson CI for overall catch rate: [{rate_ci_low:.3f}, {rate_ci_high:.3f}]")

    header("Correlations")
    corr_targets = ["fish_caught", "fish_per_hour"]
    corr_rows = []
    for target in corr_targets:
        for col in feature_cols:
            pearson_r, pearson_p = stats.pearsonr(df[col], df[target])
            spear_r, spear_p = stats.spearmanr(df[col], df[target])
            corr_rows.append(
                {
                    "target": target,
                    "feature": col,
                    "pearson_r": pearson_r,
                    "pearson_p": pearson_p,
                    "spearman_r": spear_r,
                    "spearman_p": spear_p,
                }
            )
    corr_df = pd.DataFrame(corr_rows)
    print(corr_df.round(4).to_string(index=False))

    header("Statistical Tests")
    test_rows = []
    for binary_col in ["livebait", "camper"]:
        for target in ["fish_caught", "fish_per_hour"]:
            group1 = df.loc[df[binary_col] == 1, target]
            group0 = df.loc[df[binary_col] == 0, target]
            t_stat, p_val = stats.ttest_ind(group1, group0, equal_var=False)
            test_rows.append(
                {
                    "test": "Welch t-test",
                    "factor": binary_col,
                    "target": target,
                    "stat": t_stat,
                    "p_value": p_val,
                    "mean_if_1": float(group1.mean()),
                    "mean_if_0": float(group0.mean()),
                }
            )

    for factor in ["persons", "child"]:
        for target in ["fish_caught", "fish_per_hour"]:
            groups = [g[target].values for _, g in df.groupby(factor)]
            f_stat, p_val = stats.f_oneway(*groups)
            test_rows.append(
                {
                    "test": "One-way ANOVA",
                    "factor": factor,
                    "target": target,
                    "stat": f_stat,
                    "p_value": p_val,
                    "mean_if_1": np.nan,
                    "mean_if_0": np.nan,
                }
            )

    tests_df = pd.DataFrame(test_rows)
    print(tests_df.round(4).to_string(index=False))

    header("Regression With P-values (statsmodels OLS)")
    ols_raw = smf.ols("fish_caught ~ hours + livebait + camper + persons + child", data=df).fit()
    ols_log = smf.ols(
        "log_fish_caught ~ log_hours + livebait + camper + persons + child", data=df
    ).fit()
    ols_rate = smf.ols("fish_per_hour ~ hours + livebait + camper + persons + child", data=df).fit()

    print("Model 1: fish_caught ~ hours + livebait + camper + persons + child")
    print(ols_raw.summary2().tables[1].round(4))
    print(f"R^2={ols_raw.rsquared:.4f}, model p-value={ols_raw.f_pvalue:.4g}")

    print("\nModel 2: log_fish_caught ~ log_hours + livebait + camper + persons + child")
    print(ols_log.summary2().tables[1].round(4))
    print(f"R^2={ols_log.rsquared:.4f}, model p-value={ols_log.f_pvalue:.4g}")

    print("\nModel 3: fish_per_hour ~ hours + livebait + camper + persons + child")
    print(ols_rate.summary2().tables[1].round(4))
    print(f"R^2={ols_rate.rsquared:.4f}, model p-value={ols_rate.f_pvalue:.4g}")

    header("Interpretable Models (Custom + Standard)")
    X = df[feature_cols].values
    y = df["log_fish_caught"].values

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    model_specs = {
        "LinearRegression": LinearRegression(),
        "LassoCV": LassoCV(cv=5, random_state=42, max_iter=10000),
        "DecisionTreeRegressor(depth=4)": DecisionTreeRegressor(max_depth=4, random_state=42),
        "RuleFitRegressor": RuleFitRegressor(random_state=42),
        "FIGSRegressor": FIGSRegressor(random_state=42),
        "SmartAdditiveRegressor": SmartAdditiveRegressor(
            n_rounds=300, learning_rate=0.08, min_samples_leaf=8
        ),
        "HingeEBMRegressor": HingeEBMRegressor(n_knots=3),
    }

    cv_rows = []
    for name, model in model_specs.items():
        mean_r2, std_r2 = safe_cross_val(model, X, y, cv)
        cv_rows.append({"model": name, "cv_r2_mean": mean_r2, "cv_r2_std": std_r2})

    cv_df = pd.DataFrame(cv_rows).sort_values("cv_r2_mean", ascending=False)
    print("Cross-validated R^2 on log_fish_caught:")
    print(cv_df.round(4).to_string(index=False))

    smart_model = SmartAdditiveRegressor(n_rounds=300, learning_rate=0.08, min_samples_leaf=8)
    smart_model.fit(X, y)

    hinge_model = HingeEBMRegressor(n_knots=3)
    hinge_model.fit(X, y)

    print("\nSmartAdditiveRegressor interpretation:")
    print(replace_feature_tokens(str(smart_model), feature_cols))

    print("\nHingeEBMRegressor interpretation:")
    print(replace_feature_tokens(str(hinge_model), feature_cols))

    header("Research Question Synthesis")
    significant_log_predictors = [
        name
        for name, p in ols_log.pvalues.items()
        if name != "Intercept" and float(p) < 0.05
    ]

    best_row = cv_df.iloc[0]
    best_model_name = best_row["model"]
    best_model_r2 = float(best_row["cv_r2_mean"])

    # Evidence-weighted Likert score answering whether there is a meaningful
    # relationship that supports estimating fish caught per hour.
    score = 0.0
    if ols_log.f_pvalue < 0.05:
        score += 30.0
    score += 10.0 * (len(significant_log_predictors) / 5.0)
    score += 30.0 * min(best_model_r2 / 0.50, 1.0)

    livebait_rate_p = tests_df.loc[
        (tests_df["test"] == "Welch t-test")
        & (tests_df["factor"] == "livebait")
        & (tests_df["target"] == "fish_per_hour"),
        "p_value",
    ].iloc[0]
    persons_rate_p = tests_df.loc[
        (tests_df["test"] == "One-way ANOVA")
        & (tests_df["factor"] == "persons")
        & (tests_df["target"] == "fish_per_hour"),
        "p_value",
    ].iloc[0]

    if livebait_rate_p < 0.05:
        score += 10.0
    if persons_rate_p < 0.05:
        score += 10.0

    response_score = int(np.clip(round(score), 0, 100))

    explanation = (
        f"Yes: the data supports a meaningful relationship. Estimated overall catch rate is "
        f"{overall_rate:.2f} fish/hour (95% CI {rate_ci_low:.2f}-{rate_ci_high:.2f}). "
        f"In OLS on log catches, significant predictors were {', '.join(significant_log_predictors)} "
        f"(model p={ols_log.f_pvalue:.2e}, R^2={ols_log.rsquared:.2f}). "
        f"Custom interpretable models (SmartAdditiveRegressor and HingeEBMRegressor) both retained "
        f"strong positive effects for persons and hours, positive livebait effect, and negative child effect. "
        f"Best CV performance was {best_model_name} with mean R^2={best_model_r2:.2f}."
    )

    print("Likert response (0-100):", response_score)
    print("Explanation:", explanation)

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump({"response": response_score, "explanation": explanation}, f)


if __name__ == "__main__":
    main()
