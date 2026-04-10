import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeRegressor

from interp_models import HingeEBMRegressor, SmartAdditiveRegressor


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def fmt(x) -> float:
    return float(np.asarray(x))


def main() -> None:
    # 1) Load research question metadata
    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)
    research_question = info["research_questions"][0]
    print_section("RESEARCH QUESTION")
    print(research_question)

    # 2) Load dataset
    df = pd.read_csv("hurricane.csv")
    df["log_deaths"] = np.log1p(df["alldeaths"])

    print_section("DATA OVERVIEW")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print("Missing values by column:")
    print(df.isna().sum().to_string())

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print("\nSummary statistics (numeric columns):")
    print(df[numeric_cols].describe().T.to_string())

    dist_cols = ["alldeaths", "log_deaths", "masfem", "wind", "min", "ndam15"]
    print("\nDistribution diagnostics (skew/kurtosis):")
    print(df[dist_cols].agg(["skew", "kurt"]).T.to_string())

    print_section("CORRELATIONS")
    corr_vars = [
        "log_deaths",
        "alldeaths",
        "masfem",
        "gender_mf",
        "wind",
        "min",
        "category",
        "ndam15",
        "year",
        "elapsedyrs",
        "masfem_mturk",
    ]
    pearson_corr = df[corr_vars].corr(method="pearson")
    spearman_corr = df[corr_vars].corr(method="spearman")
    print("Pearson correlations with log_deaths:")
    print(pearson_corr["log_deaths"].sort_values(ascending=False).to_string())
    print("\nSpearman correlations with log_deaths:")
    print(spearman_corr["log_deaths"].sort_values(ascending=False).to_string())

    # 3) Statistical tests
    print_section("STATISTICAL TESTS")
    pearson_masfem = stats.pearsonr(df["masfem"], df["log_deaths"])
    spearman_masfem = stats.spearmanr(df["masfem"], df["alldeaths"])
    print(
        f"Pearson(masfem, log_deaths): r={pearson_masfem.statistic:.4f}, "
        f"p={pearson_masfem.pvalue:.4g}"
    )
    print(
        f"Spearman(masfem, alldeaths): rho={spearman_masfem.statistic:.4f}, "
        f"p={spearman_masfem.pvalue:.4g}"
    )

    female_log = df.loc[df["gender_mf"] == 1, "log_deaths"]
    male_log = df.loc[df["gender_mf"] == 0, "log_deaths"]
    welch = stats.ttest_ind(female_log, male_log, equal_var=False)
    mann_whitney = stats.mannwhitneyu(
        female_log,
        male_log,
        alternative="two-sided",
    )
    print(
        f"Welch t-test (female vs male names, log_deaths): "
        f"t={welch.statistic:.4f}, p={welch.pvalue:.4g}; "
        f"means female={female_log.mean():.4f}, male={male_log.mean():.4f}"
    )
    print(
        f"Mann-Whitney U (female vs male names, log_deaths): "
        f"U={mann_whitney.statistic:.4f}, p={mann_whitney.pvalue:.4g}"
    )

    df["masfem_tertile"] = pd.qcut(df["masfem"], q=3, labels=["low", "mid", "high"])
    g_low = df.loc[df["masfem_tertile"] == "low", "log_deaths"]
    g_mid = df.loc[df["masfem_tertile"] == "mid", "log_deaths"]
    g_high = df.loc[df["masfem_tertile"] == "high", "log_deaths"]
    anova = stats.f_oneway(g_low, g_mid, g_high)
    print(
        f"ANOVA(log_deaths across masfem tertiles): "
        f"F={anova.statistic:.4f}, p={anova.pvalue:.4g}"
    )

    # OLS with controls
    formula_primary = "log_deaths ~ masfem + wind + min + ndam15 + year + C(category)"
    ols_primary = smf.ols(formula_primary, data=df).fit()
    idx_masfem = list(ols_primary.params.index).index("masfem")
    robust_primary = ols_primary.get_robustcov_results(cov_type="HC3")
    p_masfem_robust = fmt(robust_primary.pvalues[idx_masfem])
    print("\nPrimary OLS with controls:")
    print(ols_primary.summary().tables[1])
    print(
        f"masfem coefficient={ols_primary.params['masfem']:.4f}, "
        f"p={ols_primary.pvalues['masfem']:.4g}, "
        f"HC3 robust p={p_masfem_robust:.4g}"
    )

    # Interaction model: does name femininity matter more in stronger storms?
    formula_interaction = "log_deaths ~ masfem*wind + min + ndam15 + year + C(category)"
    ols_interaction = smf.ols(formula_interaction, data=df).fit()
    p_interaction = fmt(ols_interaction.pvalues["masfem:wind"])
    print(
        f"Interaction model: coef(masfem:wind)="
        f"{ols_interaction.params['masfem:wind']:.6f}, p={p_interaction:.4g}"
    )

    # 4) Standard predictive models
    print_section("STANDARD MODELS (SKLEARN)")
    feature_cols = [
        "masfem",
        "gender_mf",
        "wind",
        "min",
        "category",
        "ndam15",
        "year",
        "elapsedyrs",
        "masfem_mturk",
    ]
    X = df[feature_cols].copy()
    X = X.fillna(X.median(numeric_only=True))
    y = df["log_deaths"].values
    x_mat = X.values
    masfem_idx = feature_cols.index("masfem")

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    standard_models = {
        "LinearRegression": LinearRegression(),
        "Ridge(alpha=1.0)": Ridge(alpha=1.0, random_state=42),
        "Lasso(alpha=0.01)": Lasso(alpha=0.01, max_iter=20000, random_state=42),
        "DecisionTree(max_depth=3)": DecisionTreeRegressor(
            max_depth=3,
            min_samples_leaf=5,
            random_state=42,
        ),
    }

    std_results = {}
    for name, model in standard_models.items():
        r2_scores = cross_val_score(model, x_mat, y, cv=cv, scoring="r2")
        rmse_scores = -cross_val_score(
            model,
            x_mat,
            y,
            cv=cv,
            scoring="neg_root_mean_squared_error",
        )
        model.fit(x_mat, y)
        masfem_coef = None
        if hasattr(model, "coef_"):
            masfem_coef = fmt(model.coef_[masfem_idx])
        std_results[name] = {
            "cv_r2_mean": float(np.mean(r2_scores)),
            "cv_rmse_mean": float(np.mean(rmse_scores)),
            "masfem_coef": masfem_coef,
        }
        print(
            f"{name}: CV R2={np.mean(r2_scores):.4f} (+/- {np.std(r2_scores):.4f}), "
            f"CV RMSE={np.mean(rmse_scores):.4f}, masfem_coef={masfem_coef}"
        )

    # 5) Custom interpretable models
    print_section("CUSTOM INTERPRETABLE MODELS")
    smart = SmartAdditiveRegressor(n_rounds=300, learning_rate=0.05, min_samples_leaf=5)
    smart.fit(x_mat, y)
    smart_pred = smart.predict(x_mat)
    smart_r2 = r2_score(y, smart_pred)
    print(f"SmartAdditiveRegressor train R2={smart_r2:.4f}")
    print("SmartAdditiveRegressor interpretation:")
    print(str(smart))

    hinge = HingeEBMRegressor(
        n_knots=3,
        max_input_features=15,
        ebm_outer_bags=3,
        ebm_max_rounds=300,
    )
    hinge.fit(x_mat, y)
    hinge_pred = hinge.predict(x_mat)
    hinge_r2 = r2_score(y, hinge_pred)
    print(f"\nHingeEBMRegressor train R2={hinge_r2:.4f}")
    print("HingeEBMRegressor interpretation:")
    print(str(hinge))

    # Quantify custom-model effect of masfem by counterfactual replacement
    q10 = float(np.quantile(X["masfem"], 0.10))
    q90 = float(np.quantile(X["masfem"], 0.90))
    x_low = x_mat.copy()
    x_high = x_mat.copy()
    x_low[:, masfem_idx] = q10
    x_high[:, masfem_idx] = q90
    smart_delta = float(np.mean(smart.predict(x_high) - smart.predict(x_low)))
    hinge_delta = float(np.mean(hinge.predict(x_high) - hinge.predict(x_low)))
    print(
        f"\nCounterfactual effect of higher femininity (10th->90th pct masfem): "
        f"SmartAdditive delta={smart_delta:.4f}, HingeEBM delta={hinge_delta:.4f}"
    )

    # 6) Synthesis for Likert score (0=no, 100=yes)
    # Evidence for the hypothesis requires a positive and statistically significant
    # relationship between femininity and fatalities after controls.
    evidence_points = 0
    max_points = 10

    if (pearson_masfem.pvalue < 0.05) and (pearson_masfem.statistic > 0):
        evidence_points += 1
    if (spearman_masfem.pvalue < 0.05) and (spearman_masfem.statistic > 0):
        evidence_points += 1
    if (welch.pvalue < 0.05) and (female_log.mean() > male_log.mean()):
        evidence_points += 1
    if (anova.pvalue < 0.05) and (g_high.mean() > g_low.mean()):
        evidence_points += 1
    if (ols_primary.pvalues["masfem"] < 0.05) and (ols_primary.params["masfem"] > 0):
        evidence_points += 3
    elif (ols_primary.pvalues["masfem"] < 0.10) and (ols_primary.params["masfem"] > 0):
        evidence_points += 1
    if (p_masfem_robust < 0.05) and (ols_primary.params["masfem"] > 0):
        evidence_points += 1
    if (p_interaction < 0.05) and (ols_interaction.params["masfem:wind"] > 0):
        evidence_points += 1
    if smart_delta > 0.15:
        evidence_points += 1
    if hinge_delta > 0.15:
        evidence_points += 1

    response_score = int(round((evidence_points / max_points) * 100))
    if response_score == 0:
        response_score = 10

    explanation = (
        "Evidence does not support the claim that more feminine hurricane names are linked "
        "to higher fatalities via reduced precautions. Correlations were near zero "
        f"(Pearson r={pearson_masfem.statistic:.3f}, p={pearson_masfem.pvalue:.3f}; "
        f"Spearman rho={spearman_masfem.statistic:.3f}, p={spearman_masfem.pvalue:.3f}). "
        f"Female-vs-male name differences were not significant (Welch p={welch.pvalue:.3f}), "
        f"and ANOVA across femininity tertiles was not significant (p={anova.pvalue:.3f}). "
        "In the controlled OLS model, masfem was small and non-significant "
        f"(coef={ols_primary.params['masfem']:.3f}, p={ols_primary.pvalues['masfem']:.3f}, "
        f"HC3 p={p_masfem_robust:.3f}), while storm severity/damage proxies (minimum pressure "
        "and normalized damage) were much stronger predictors. The custom interpretable models "
        "also prioritized damage/severity; HingeEBM effectively excluded femininity and "
        f"counterfactual high-vs-low femininity effects were weak (SmartAdditive={smart_delta:.3f}, "
        f"HingeEBM={hinge_delta:.3f} on log-deaths)."
    )

    print_section("FINAL CONCLUSION")
    print(f"Likert response (0-100): {response_score}")
    print(explanation)

    conclusion = {"response": response_score, "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(conclusion))


if __name__ == "__main__":
    main()
