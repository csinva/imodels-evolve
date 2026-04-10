import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from imodels import FIGSRegressor, HSTreeRegressor, RuleFitRegressor
from scipy import stats
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


def fmt(x, digits=4):
    return float(np.round(float(x), digits))


def print_header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def top_abs_coefs(feature_names, coefs, top_k=7):
    pairs = sorted(zip(feature_names, coefs), key=lambda z: abs(z[1]), reverse=True)
    return [{"feature": f, "coef": fmt(c)} for f, c in pairs[:top_k]]


def compute_score(evidence):
    score = 50

    # Core claim requires a positive and significant femininity effect on deaths.
    if evidence["ols_controls_coef"] > 0 and evidence["ols_controls_p"] < 0.05:
        score += 35
    else:
        score -= 25

    # Secondary checks.
    if evidence["pearson_r"] > 0 and evidence["pearson_p"] < 0.05:
        score += 12
    else:
        score -= 10

    if evidence["interaction_coef"] > 0 and evidence["interaction_p"] < 0.05:
        score += 15
    else:
        score -= 8

    if evidence["ttest_diff_female_minus_male"] > 0 and evidence["ttest_p"] < 0.05:
        score += 8
    else:
        score -= 5

    # Interpretability models should at least keep femininity among major predictors.
    if evidence["tree_importance_masfem"] >= 0.1 or evidence["figs_importance_masfem"] >= 0.1:
        score += 5
    else:
        score -= 5

    return int(np.clip(round(score), 0, 100))


def main():
    base = Path(".")
    info_path = base / "info.json"
    data_path = base / "hurricane.csv"
    out_path = base / "conclusion.txt"

    info = json.loads(info_path.read_text())
    question = info.get("research_questions", [""])[0]

    df = pd.read_csv(data_path)

    print_header("Research Question")
    print(question)

    print_header("Data Overview")
    print(f"Rows: {len(df)}, Columns: {df.shape[1]}")
    print("Columns:", list(df.columns))
    print("\nMissing values:")
    print(df.isna().sum().to_string())

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print("\nNumeric summary statistics:")
    print(df[numeric_cols].describe().T.to_string())

    # Target transformation for heavily right-skewed death counts.
    df["log_deaths"] = np.log1p(df["alldeaths"])

    print_header("Distribution and Correlation Checks")
    print(f"Skewness alldeaths: {fmt(stats.skew(df['alldeaths'], bias=False))}")
    print(f"Skewness log_deaths: {fmt(stats.skew(df['log_deaths'], bias=False))}")
    corr_cols = ["masfem", "gender_mf", "wind", "min", "category", "ndam15", "year", "alldeaths", "log_deaths"]
    corr = df[corr_cols].corr(numeric_only=True)
    print("\nCorrelation matrix (subset):")
    print(corr.to_string())

    print_header("Statistical Tests")
    clean = df.dropna(
        subset=["masfem", "gender_mf", "alldeaths", "wind", "min", "category", "year", "ndam15", "log_deaths"]
    ).copy()

    pearson_r, pearson_p = stats.pearsonr(clean["masfem"], clean["log_deaths"])
    spearman_rho, spearman_p = stats.spearmanr(clean["masfem"], clean["log_deaths"])
    print(f"Pearson masfem vs log_deaths: r={fmt(pearson_r)}, p={fmt(pearson_p)}")
    print(f"Spearman masfem vs log_deaths: rho={fmt(spearman_rho)}, p={fmt(spearman_p)}")

    female = clean.loc[clean["gender_mf"] == 1, "log_deaths"]
    male = clean.loc[clean["gender_mf"] == 0, "log_deaths"]
    t_stat, t_p = stats.ttest_ind(female, male, equal_var=False)
    t_diff = float(female.mean() - male.mean())
    print(
        f"T-test female vs male names (log_deaths): t={fmt(t_stat)}, p={fmt(t_p)}, "
        f"mean_diff={fmt(t_diff)}"
    )

    anova_groups = [g["log_deaths"].values for _, g in clean.groupby("category")]
    f_stat, f_p = stats.f_oneway(*anova_groups)
    print(f"ANOVA category -> log_deaths: F={fmt(f_stat)}, p={fmt(f_p)}")

    model_1 = smf.ols("log_deaths ~ masfem", data=clean).fit()
    model_2 = smf.ols("log_deaths ~ masfem + wind + min + C(category) + year", data=clean).fit()
    model_3 = smf.ols("log_deaths ~ masfem * wind + min + C(category) + year", data=clean).fit()
    model_4 = smf.ols("log_deaths ~ gender_mf + wind + min + C(category) + year", data=clean).fit()
    model_5 = smf.ols("log_deaths ~ masfem + wind + min + C(category) + year + np.log1p(ndam15)", data=clean).fit()

    print("\nOLS model summaries (key terms):")
    print(
        f"M1 masfem coef={fmt(model_1.params['masfem'])}, p={fmt(model_1.pvalues['masfem'])}, "
        f"R2={fmt(model_1.rsquared)}"
    )
    print(
        f"M2 masfem coef={fmt(model_2.params['masfem'])}, p={fmt(model_2.pvalues['masfem'])}, "
        f"R2={fmt(model_2.rsquared)}"
    )
    print(
        f"M3 masfem coef={fmt(model_3.params['masfem'])}, p={fmt(model_3.pvalues['masfem'])}; "
        f"interaction coef={fmt(model_3.params['masfem:wind'])}, p={fmt(model_3.pvalues['masfem:wind'])}"
    )
    print(
        f"M4 gender_mf coef={fmt(model_4.params['gender_mf'])}, p={fmt(model_4.pvalues['gender_mf'])}, "
        f"R2={fmt(model_4.rsquared)}"
    )
    print(
        f"M5 (adds damage control) masfem coef={fmt(model_5.params['masfem'])}, "
        f"p={fmt(model_5.pvalues['masfem'])}, R2={fmt(model_5.rsquared)}"
    )

    print_header("Interpretable Models")
    features = ["masfem", "gender_mf", "wind", "min", "category", "ndam15", "year"]
    X = clean[features].copy()
    y = clean["log_deaths"].values

    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    lin = LinearRegression().fit(Xz, y)
    ridge = Ridge(alpha=1.0).fit(Xz, y)
    lasso = Lasso(alpha=0.02, max_iter=20000).fit(Xz, y)
    tree = DecisionTreeRegressor(max_depth=3, random_state=0).fit(X, y)

    print("LinearRegression standardized coefficients:")
    print(top_abs_coefs(features, lin.coef_))
    print("Ridge standardized coefficients:")
    print(top_abs_coefs(features, ridge.coef_))
    print("Lasso standardized coefficients:")
    print(top_abs_coefs(features, lasso.coef_))
    print(
        "DecisionTreeRegressor feature importances:",
        {f: fmt(v) for f, v in sorted(zip(features, tree.feature_importances_), key=lambda t: t[1], reverse=True)},
    )

    rulefit = RuleFitRegressor(random_state=0, max_rules=40)
    rulefit.fit(X.values, y, feature_names=features)
    if hasattr(rulefit, "get_rules"):
        rules_df = rulefit.get_rules()
    else:
        rules_df = rulefit._get_rules(exclude_zero_coef=True)
    if "importance" in rules_df.columns:
        rules_top = (
            rules_df.sort_values("importance", ascending=False)
            .head(10)[["rule", "type", "coef", "importance"]]
            .to_dict(orient="records")
        )
    else:
        rules_top = (
            rules_df.head(10)[[c for c in rules_df.columns if c in {"rule", "type", "coef"}]].to_dict(orient="records")
        )
    print("RuleFit top terms/rules:")
    print(rules_top)

    figs = FIGSRegressor(random_state=0, max_rules=12)
    figs.fit(X.values, y, feature_names=features)
    figs_importances = getattr(figs, "feature_importances_", np.zeros(len(features)))
    print(
        "FIGS feature importances:",
        {f: fmt(v) for f, v in sorted(zip(features, figs_importances), key=lambda t: t[1], reverse=True)},
    )

    hstree = HSTreeRegressor(random_state=0, max_leaf_nodes=8)
    hstree.fit(X.values, y, feature_names=features)
    hstree_text = str(hstree)
    print("HSTree structure:")
    print(hstree_text)

    evidence = {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
        "ttest_p": float(t_p),
        "ttest_diff_female_minus_male": float(t_diff),
        "anova_p": float(f_p),
        "ols_controls_coef": float(model_2.params["masfem"]),
        "ols_controls_p": float(model_2.pvalues["masfem"]),
        "interaction_coef": float(model_3.params["masfem:wind"]),
        "interaction_p": float(model_3.pvalues["masfem:wind"]),
        "tree_importance_masfem": float(tree.feature_importances_[features.index("masfem")]),
        "figs_importance_masfem": float(figs_importances[features.index("masfem")]),
        "masfem_in_hstree_text": ("masfem" in hstree_text),
    }

    score = compute_score(evidence)

    explanation = (
        f"Evidence does not support the claim that more feminine hurricane names cause higher fatalities: "
        f"masfem-log_deaths correlation is weak/non-significant (Pearson r={fmt(pearson_r)}, p={fmt(pearson_p)}; "
        f"Spearman rho={fmt(spearman_rho)}, p={fmt(spearman_p)}), female-vs-male mean difference is non-significant "
        f"(p={fmt(t_p)}), and the controlled OLS effect of masfem is non-significant "
        f"(coef={fmt(model_2.params['masfem'])}, p={fmt(model_2.pvalues['masfem'])}) with non-significant "
        f"masfem*wind interaction (p={fmt(model_3.pvalues['masfem:wind'])}). Interpretable models mostly prioritize "
        f"storm intensity/damage variables (especially ndam15, wind, min) over name femininity."
    )

    result = {"response": score, "explanation": explanation}
    out_path.write_text(json.dumps(result))

    print_header("Final Likert Output")
    print(json.dumps(result, indent=2))
    print(f"\nWrote {out_path.resolve()}")


if __name__ == "__main__":
    main()
