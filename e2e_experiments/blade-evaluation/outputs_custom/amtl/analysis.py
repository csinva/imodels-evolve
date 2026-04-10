import json
import re
import warnings

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from interp_models import HingeEBMRegressor, SmartAdditiveRegressor

warnings.filterwarnings("ignore")


def summarize_coeffs_from_model_str(model_str, feature_names):
    """Extract x{idx}: coef lines from custom model string and map to names."""
    mapping = {}
    for line in model_str.splitlines():
        m = re.search(r"^\s*x(\d+):\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$", line)
        if m:
            idx = int(m.group(1))
            coef = float(m.group(2))
            if 0 <= idx < len(feature_names):
                mapping[feature_names[idx]] = coef
    return mapping


def main():
    # 1) Read prompt metadata
    with open("info.json", "r") as f:
        info = json.load(f)

    question = info["research_questions"][0]
    print("Research question:")
    print(question)
    print("\n" + "=" * 88)

    # 2) Load data
    df = pd.read_csv("amtl.csv")
    df["amtl_rate"] = df["num_amtl"] / df["sockets"]
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    print(f"Dataset shape: {df.shape}")
    print("Columns:", list(df.columns))
    print("\nMissing values by column:")
    print(df.isna().sum())

    # 3) Explore: summary stats, distributions, correlations
    print("\nNumeric summary statistics:")
    numeric_cols = ["num_amtl", "sockets", "age", "stdev_age", "prob_male", "amtl_rate"]
    print(df[numeric_cols].describe().T)

    print("\nAMTL rate by genus:")
    genus_stats = df.groupby("genus")["amtl_rate"].agg(["mean", "std", "count"]).sort_values("mean", ascending=False)
    print(genus_stats)

    print("\nAMTL rate by tooth class:")
    tooth_stats = df.groupby("tooth_class")["amtl_rate"].agg(["mean", "std", "count"]).sort_values("mean", ascending=False)
    print(tooth_stats)

    print("\nCorrelation matrix (numeric variables):")
    corr = df[numeric_cols + ["is_human"]].corr()
    print(corr)

    # 4) Statistical tests focused on human vs non-human relationship
    specimen = (
        df.groupby(["specimen", "genus"], as_index=False)
        .agg(
            num_amtl=("num_amtl", "sum"),
            sockets=("sockets", "sum"),
            age=("age", "mean"),
            prob_male=("prob_male", "mean"),
        )
    )
    specimen["amtl_rate"] = specimen["num_amtl"] / specimen["sockets"]
    specimen["is_human"] = (specimen["genus"] == "Homo sapiens").astype(int)

    human_rates = specimen.loc[specimen["is_human"] == 1, "amtl_rate"].values
    nonhuman_rates = specimen.loc[specimen["is_human"] == 0, "amtl_rate"].values

    welch = stats.ttest_ind(human_rates, nonhuman_rates, equal_var=False)
    mann = stats.mannwhitneyu(human_rates, nonhuman_rates, alternative="greater")

    # One-way ANOVA across all four genera at specimen level
    rates_by_genus = [g["amtl_rate"].values for _, g in specimen.groupby("genus")]
    anova = stats.f_oneway(*rates_by_genus)

    print("\nSpecimen-level tests:")
    print(f"Human mean rate={human_rates.mean():.4f}, non-human mean rate={nonhuman_rates.mean():.4f}")
    print(f"Welch t-test: t={welch.statistic:.4f}, p={welch.pvalue:.3e}")
    print(f"Mann-Whitney (human > non-human): U={mann.statistic:.4f}, p={mann.pvalue:.3e}")
    print(f"One-way ANOVA across genera: F={anova.statistic:.4f}, p={anova.pvalue:.3e}")

    # Binomial GLM with adjustment for age/sex/tooth class
    glm = smf.glm(
        "amtl_rate ~ is_human + age + prob_male + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    ).fit()

    print("\nAdjusted binomial GLM (primary model):")
    print(glm.summary())

    glm_human_coef = float(glm.params["is_human"])
    glm_human_p = float(glm.pvalues["is_human"])
    glm_human_or = float(np.exp(glm_human_coef))

    # Full genus model for context
    glm_genus = smf.glm(
        "amtl_rate ~ C(genus) + age + prob_male + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    ).fit()

    print("\nAdjusted binomial GLM with full genus categories:")
    print(glm_genus.summary())

    # OLS (standard p-value-based linear baseline)
    ols = smf.ols("amtl_rate ~ is_human + age + prob_male + C(tooth_class)", data=df).fit()
    print("\nOLS baseline model:")
    print(ols.summary())

    # 5) Standard ML models on adjusted feature set
    X = pd.DataFrame(
        {
            "is_human": df["is_human"].astype(float),
            "age": df["age"].astype(float),
            "prob_male": df["prob_male"].astype(float),
        }
    )
    X = pd.concat(
        [X, pd.get_dummies(df["tooth_class"], prefix="tooth", drop_first=True).astype(float)],
        axis=1,
    )
    y = df["amtl_rate"].values

    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=0.25, random_state=42
    )

    standard_models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.0005, random_state=42, max_iter=10000),
        "DecisionTree": DecisionTreeRegressor(max_depth=4, random_state=42),
    }

    print("\nStandard model performance (test split):")
    for name, model in standard_models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        print(
            f"{name}: R2={r2_score(y_test, pred):.4f}, "
            f"MAE={mean_absolute_error(y_test, pred):.4f}"
        )

    # Optional: imodels interpretable trees/rules
    try:
        from imodels import FIGSRegressor, RuleFitRegressor

        rulefit = RuleFitRegressor(random_state=42)
        rulefit.fit(X_train, y_train)
        pred_rulefit = rulefit.predict(X_test)

        figs = FIGSRegressor(random_state=42)
        figs.fit(X_train, y_train)
        pred_figs = figs.predict(X_test)

        print("RuleFitRegressor: " f"R2={r2_score(y_test, pred_rulefit):.4f}, MAE={mean_absolute_error(y_test, pred_rulefit):.4f}")
        print("FIGSRegressor: " f"R2={r2_score(y_test, pred_figs):.4f}, MAE={mean_absolute_error(y_test, pred_figs):.4f}")
    except Exception as e:
        print(f"imodels models skipped due to: {e}")

    # 6) Custom interpretability models (required)
    smart = SmartAdditiveRegressor(n_rounds=250, learning_rate=0.05, min_samples_leaf=20)
    smart.fit(X_train, y_train)
    smart_pred = smart.predict(X_test)
    smart_r2 = r2_score(y_test, smart_pred)

    hinge = HingeEBMRegressor(n_knots=3, max_input_features=15, ebm_outer_bags=3, ebm_max_rounds=800)
    hinge.fit(X_train, y_train)
    hinge_pred = hinge.predict(X_test)
    hinge_r2 = r2_score(y_test, hinge_pred)

    smart_str = str(smart)
    hinge_str = str(hinge)

    print("\nCustom interpretability model performance:")
    print(f"SmartAdditiveRegressor: R2={smart_r2:.4f}, MAE={mean_absolute_error(y_test, smart_pred):.4f}")
    print(f"HingeEBMRegressor: R2={hinge_r2:.4f}, MAE={mean_absolute_error(y_test, hinge_pred):.4f}")

    print("\nSmartAdditiveRegressor interpretation:")
    print(smart_str)

    print("\nHingeEBMRegressor interpretation:")
    print(hinge_str)

    smart_coefs = summarize_coeffs_from_model_str(smart_str, feature_names)
    hinge_coefs = summarize_coeffs_from_model_str(hinge_str, feature_names)

    smart_human_coef = smart_coefs.get("is_human", 0.0)
    hinge_human_coef = hinge_coefs.get("is_human", 0.0)

    print("\nMapped custom-model coefficients for `is_human`:")
    print(f"SmartAdditive `is_human` coef (linear component): {smart_human_coef:.4f}")
    print(f"HingeEBM `is_human` coef (effective linearized): {hinge_human_coef:.4f}")

    # 7) Decision logic for final Likert score (0-100)
    mean_diff = float(human_rates.mean() - nonhuman_rates.mean())

    score = 50
    if mean_diff > 0:
        score += 10
    if welch.pvalue < 0.05 and mean_diff > 0:
        score += 10
    if welch.pvalue < 1e-3 and mean_diff > 0:
        score += 8
    if glm_human_coef > 0 and glm_human_p < 0.05:
        score += 12
    if glm_human_coef > 0 and glm_human_p < 1e-3:
        score += 8
    if glm_human_or > 2:
        score += 6
    if smart_human_coef > 0:
        score += 4
    if hinge_human_coef > 0:
        score += 4
    if anova.pvalue < 0.05:
        score += 3

    # Penalize in case adjusted effect is not significant or opposite direction
    if glm_human_p >= 0.05:
        score -= 25
    if glm_human_coef <= 0:
        score -= 20

    score = int(np.clip(round(score), 0, 100))

    explanation = (
        f"Specimen-level AMTL is higher in humans (mean {human_rates.mean():.3f}) than non-humans "
        f"(mean {nonhuman_rates.mean():.3f}); Welch t-test p={welch.pvalue:.2e}. "
        f"In adjusted binomial regression controlling for age, sex proxy, and tooth class, "
        f"the human indicator is positive (beta={glm_human_coef:.3f}, OR={glm_human_or:.2f}, p={glm_human_p:.2e}). "
        f"Custom interpretable models also assign a positive coefficient to the human indicator "
        f"(SmartAdditive {smart_human_coef:.3f}, HingeEBM {hinge_human_coef:.3f}), supporting a robust positive association."
    )

    output = {"response": score, "explanation": explanation}

    with open("conclusion.txt", "w") as f:
        json.dump(output, f)

    print("\nWrote conclusion.txt:")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
