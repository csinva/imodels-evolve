import json
import re
import warnings

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from interp_models import HingeEBMRegressor, SmartAdditiveRegressor

warnings.filterwarnings("ignore")


def p_to_strength(p_value: float) -> float:
    if p_value < 1e-6:
        return 1.0
    if p_value < 1e-4:
        return 0.95
    if p_value < 1e-3:
        return 0.85
    if p_value < 1e-2:
        return 0.75
    if p_value < 0.05:
        return 0.6
    if p_value < 0.1:
        return 0.4
    return 0.2


def map_model_text(text: str, feature_names: list[str]) -> str:
    mapped = text
    # Replace longer indices first (x12 before x1).
    for idx in sorted(range(len(feature_names)), key=lambda k: -len(str(k))):
        mapped = re.sub(rf"\\bx{idx}\\b", feature_names[idx], mapped)
    return mapped


def main() -> None:
    print("Loading data...")
    df = pd.read_csv("soccer.csv")

    # Skin-tone engineering
    df["skin_tone"] = df[["rater1", "rater2"]].mean(axis=1)
    df["has_skin"] = df["skin_tone"].notna().astype(int)
    df["dark_skin"] = (df["skin_tone"] > 0.5).astype(int)
    df["light_skin"] = (df["skin_tone"] < 0.5).astype(int)
    df["neutral_skin"] = (df["skin_tone"] == 0.5).astype(int)

    # Rate outcome to account for exposure (games)
    df["red_rate"] = df["redCards"] / df["games"].replace(0, np.nan)
    df["any_red"] = (df["redCards"] > 0).astype(int)

    # Keep only rows with available skin ratings
    dfm = df[df["has_skin"] == 1].copy()

    print("\n=== Basic Data Checks ===")
    print(f"Rows total: {len(df):,}")
    print(f"Rows with skin-tone ratings: {len(dfm):,}")
    print("Missingness (selected columns):")
    print(df[["redCards", "games", "rater1", "rater2", "meanIAT", "meanExp"]].isna().mean())

    print("\n=== Summary Statistics ===")
    summary_cols = [
        "redCards",
        "red_rate",
        "games",
        "skin_tone",
        "yellowCards",
        "yellowReds",
        "goals",
        "meanIAT",
        "meanExp",
    ]
    print(dfm[summary_cols].describe().T)

    print("\n=== Distribution Checks ===")
    print("redCards value counts:")
    print(dfm["redCards"].value_counts(normalize=True).sort_index())
    print("\nskin_tone value counts:")
    print(dfm["skin_tone"].value_counts(normalize=True).sort_index())

    print("\n=== Correlations (Pearson) ===")
    corr_cols = [
        "redCards",
        "red_rate",
        "skin_tone",
        "games",
        "yellowCards",
        "yellowReds",
        "goals",
        "meanIAT",
        "meanExp",
    ]
    print(dfm[corr_cols].corr().round(3))

    print("\n=== Hypothesis Tests: Dark vs Light Skin ===")
    group = dfm[(dfm["dark_skin"] == 1) | (dfm["light_skin"] == 1)].copy()
    dark_rate = group.loc[group["dark_skin"] == 1, "red_rate"]
    light_rate = group.loc[group["light_skin"] == 1, "red_rate"]

    # Welch t-test on red-card rate
    t_stat, p_ttest = stats.ttest_ind(dark_rate, light_rate, equal_var=False, nan_policy="omit")

    # Nonparametric robustness test
    u_stat, p_mw = stats.mannwhitneyu(dark_rate, light_rate, alternative="two-sided")

    # Chi-square on probability of any red card
    contingency = pd.crosstab(group["dark_skin"], group["any_red"])
    chi2_stat, p_chi2, _, _ = stats.chi2_contingency(contingency)

    print(f"Dark mean red rate:  {dark_rate.mean():.6f}")
    print(f"Light mean red rate: {light_rate.mean():.6f}")
    print(f"Mean difference (dark-light): {(dark_rate.mean() - light_rate.mean()):.6f}")
    print(f"Welch t-test: t={t_stat:.4f}, p={p_ttest:.3g}")
    print(f"Mann-Whitney U: U={u_stat:.4f}, p={p_mw:.3g}")
    print(f"Chi-square any_red: chi2={chi2_stat:.4f}, p={p_chi2:.3g}")

    print("\n=== Multivariable Regression ===")
    regression_cols = [
        "redCards",
        "red_rate",
        "games",
        "dark_skin",
        "skin_tone",
        "yellowCards",
        "yellowReds",
        "goals",
        "victories",
        "ties",
        "defeats",
        "height",
        "weight",
        "meanIAT",
        "meanExp",
        "position",
        "leagueCountry",
    ]
    reg = dfm[regression_cols].dropna().copy()

    # OLS on rate with robust SE
    y_ols = reg["red_rate"]
    X_num = reg[
        [
            "dark_skin",
            "games",
            "yellowCards",
            "yellowReds",
            "goals",
            "victories",
            "ties",
            "defeats",
            "height",
            "weight",
            "meanIAT",
            "meanExp",
        ]
    ]
    X_cat = pd.get_dummies(reg[["position", "leagueCountry"]], drop_first=True, dtype=float)
    X_ols = pd.concat([X_num, X_cat], axis=1)
    X_ols = sm.add_constant(X_ols, has_constant="add")
    ols_res = sm.OLS(y_ols, X_ols).fit(cov_type="HC3")

    dark_coef_ols = float(ols_res.params["dark_skin"])
    dark_p_ols = float(ols_res.pvalues["dark_skin"])

    print(f"OLS dark_skin coef (rate scale): {dark_coef_ols:.6f}")
    print(f"OLS dark_skin p-value: {dark_p_ols:.3g}")
    print(f"OLS R^2: {ols_res.rsquared:.4f}")

    # Poisson on counts with games offset
    y_pois = reg["redCards"]
    X_pois = pd.concat(
        [
            reg[
                [
                    "dark_skin",
                    "yellowCards",
                    "yellowReds",
                    "goals",
                    "victories",
                    "ties",
                    "defeats",
                    "height",
                    "weight",
                    "meanIAT",
                    "meanExp",
                ]
            ],
            X_cat,
        ],
        axis=1,
    )
    X_pois = sm.add_constant(X_pois, has_constant="add")
    poisson_res = sm.GLM(
        y_pois,
        X_pois,
        family=sm.families.Poisson(),
        offset=np.log(reg["games"].clip(lower=1)),
    ).fit(cov_type="HC3")

    dark_coef_pois = float(poisson_res.params["dark_skin"])
    dark_p_pois = float(poisson_res.pvalues["dark_skin"])
    dark_irr = float(np.exp(dark_coef_pois))

    print(f"Poisson dark_skin log-coef: {dark_coef_pois:.6f}")
    print(f"Poisson dark_skin IRR: {dark_irr:.4f}")
    print(f"Poisson dark_skin p-value: {dark_p_pois:.3g}")

    print("\n=== Custom Interpretable Models (interp_models.py) ===")
    feature_names = [
        "skin_tone",
        "dark_skin",
        "games",
        "yellowCards",
        "yellowReds",
        "goals",
        "victories",
        "ties",
        "defeats",
        "height",
        "weight",
        "meanIAT",
        "meanExp",
    ]

    model_df = reg[["red_rate"] + feature_names].copy()
    sample_n = min(25000, len(model_df))
    model_df = model_df.sample(n=sample_n, random_state=42)

    X_custom = model_df[feature_names].to_numpy(dtype=float)
    y_custom = model_df["red_rate"].to_numpy(dtype=float)

    smart = SmartAdditiveRegressor(n_rounds=120, learning_rate=0.08, min_samples_leaf=40)
    smart.fit(X_custom, y_custom)
    smart_text = map_model_text(str(smart), feature_names)
    print("\nSmartAdditiveRegressor interpretation:")
    print(smart_text)

    hinge = HingeEBMRegressor(n_knots=3, max_input_features=13, ebm_outer_bags=2, ebm_max_rounds=250)
    hinge.fit(X_custom, y_custom)
    hinge_text = map_model_text(str(hinge), feature_names)
    print("\nHingeEBMRegressor interpretation:")
    print(hinge_text)

    # Standard benchmark models
    print("\n=== Standard Model Benchmarks ===")
    X_train, X_test, y_train, y_test = train_test_split(
        model_df[feature_names], model_df["red_rate"], test_size=0.25, random_state=42
    )

    lin = LinearRegression()
    lin.fit(X_train, y_train)
    lin_pred = lin.predict(X_test)

    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train, y_train)
    ridge_pred = ridge.predict(X_test)

    tree = DecisionTreeRegressor(max_depth=4, min_samples_leaf=200, random_state=42)
    tree.fit(X_train, y_train)
    tree_pred = tree.predict(X_test)

    def metrics(name, y_true, y_hat):
        print(
            f"{name}: R2={r2_score(y_true, y_hat):.4f}, "
            f"RMSE={np.sqrt(mean_squared_error(y_true, y_hat)):.6f}"
        )

    metrics("LinearRegression", y_test, lin_pred)
    metrics("Ridge", y_test, ridge_pred)
    metrics("DecisionTree", y_test, tree_pred)

    lin_coef = pd.Series(lin.coef_, index=feature_names).sort_values(key=np.abs, ascending=False)
    print("Top LinearRegression coefficients (abs):")
    print(lin_coef.head(8))

    # Evidence synthesis into Likert score
    dark_direction_unadj = float(dark_rate.mean() - light_rate.mean())
    tests = {
        "ttest": (p_ttest, dark_direction_unadj > 0),
        "mannwhitney": (p_mw, dark_direction_unadj > 0),
        "chi2": (p_chi2, dark_direction_unadj > 0),
        "ols": (dark_p_ols, dark_coef_ols > 0),
        "poisson": (dark_p_pois, dark_coef_pois > 0),
    }

    strengths = []
    for p_val, is_positive in tests.values():
        signed_strength = p_to_strength(float(p_val)) * (1 if is_positive else -1)
        strengths.append(signed_strength)
    evidence_score = float(np.mean(strengths))

    # Map [-1, 1] to [0, 100], with extra penalty for non-significant adjusted effects.
    raw_likert = 50 + 50 * evidence_score
    if dark_p_ols >= 0.05 and dark_p_pois >= 0.05:
        raw_likert = min(raw_likert, 35)
    if dark_coef_ols <= 0 and dark_coef_pois <= 0 and (dark_p_ols < 0.05 or dark_p_pois < 0.05):
        raw_likert = min(raw_likert, 15)

    response = int(np.clip(np.round(raw_likert), 0, 100))

    explanation = (
        "Research question: Are darker-skinned players more likely to receive red cards? "
        f"Unadjusted comparison (dark vs light) red-card rate difference = {dark_direction_unadj:.6f}; "
        f"Welch t-test p={p_ttest:.3g}, Mann-Whitney p={p_mw:.3g}, chi-square p={p_chi2:.3g}. "
        f"Adjusted OLS (rate) dark_skin coef={dark_coef_ols:.6f}, p={dark_p_ols:.3g}. "
        f"Adjusted Poisson with games offset IRR={dark_irr:.4f}, p={dark_p_pois:.3g}. "
        "Custom interpretable models (SmartAdditiveRegressor and HingeEBMRegressor) were fit to inspect feature effects alongside standard models. "
        "The Likert score reflects combined direction and significance, emphasizing adjusted models."
    )

    out = {"response": response, "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(out, f)

    print("\nWrote conclusion.txt")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
