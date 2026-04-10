import json
import warnings

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor, export_text

warnings.filterwarnings("ignore")


def section(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def main() -> None:
    # 1) Load and prepare data
    df = pd.read_csv("caschools.csv")
    df["student_teacher_ratio"] = df["students"] / df["teachers"]
    df["avg_score"] = (df["read"] + df["math"]) / 2.0
    df["computer_per_student"] = df["computer"] / df["students"]

    analysis_cols = [
        "avg_score",
        "student_teacher_ratio",
        "calworks",
        "lunch",
        "english",
        "income",
        "expenditure",
        "computer_per_student",
    ]
    data = df[analysis_cols].replace([np.inf, -np.inf], np.nan).dropna().copy()

    section("Data Overview")
    print(f"Rows used for analysis: {len(data)}")
    print(f"Columns used: {analysis_cols}")
    print("\nMissing values (before dropna):")
    print(df[analysis_cols].isna().sum().to_string())

    section("Summary Statistics")
    print(data.describe().T.to_string(float_format=lambda x: f"{x:,.4f}"))

    section("Distribution Diagnostics")
    distribution = pd.DataFrame(
        {
            "mean": data.mean(),
            "median": data.median(),
            "std": data.std(),
            "skew": data.skew(),
            "min": data.min(),
            "q25": data.quantile(0.25),
            "q75": data.quantile(0.75),
            "max": data.max(),
        }
    )
    print(distribution.to_string(float_format=lambda x: f"{x:,.4f}"))

    section("Correlations")
    corr = data.corr(numeric_only=True)
    print(corr.to_string(float_format=lambda x: f"{x:,.4f}"))
    print("\nCorrelations with avg_score:")
    print(corr["avg_score"].sort_values(ascending=False).to_string(float_format=lambda x: f"{x:,.4f}"))

    # 2) Statistical tests focused on research question
    str_vals = data["student_teacher_ratio"]
    y = data["avg_score"]

    pearson_r, pearson_p = stats.pearsonr(str_vals, y)
    spearman_r, spearman_p = stats.spearmanr(str_vals, y)

    median_str = str_vals.median()
    low_str_scores = y[str_vals <= median_str]
    high_str_scores = y[str_vals > median_str]
    t_stat, t_p = stats.ttest_ind(low_str_scores, high_str_scores, equal_var=False)

    quartile = pd.qcut(str_vals, q=4, labels=False, duplicates="drop")
    quartile_groups = [y[quartile == i] for i in sorted(pd.Series(quartile).dropna().unique())]
    anova_f, anova_p = stats.f_oneway(*quartile_groups)

    X_simple = sm.add_constant(data[["student_teacher_ratio"]])
    ols_simple = sm.OLS(y, X_simple).fit()

    features = [
        "student_teacher_ratio",
        "calworks",
        "lunch",
        "english",
        "income",
        "expenditure",
        "computer_per_student",
    ]
    X_multi = sm.add_constant(data[features])
    ols_multi = sm.OLS(y, X_multi).fit()

    simple_coef = float(ols_simple.params["student_teacher_ratio"])
    simple_p = float(ols_simple.pvalues["student_teacher_ratio"])
    simple_ci = ols_simple.conf_int().loc["student_teacher_ratio"].tolist()

    multi_coef = float(ols_multi.params["student_teacher_ratio"])
    multi_p = float(ols_multi.pvalues["student_teacher_ratio"])
    multi_ci = ols_multi.conf_int().loc["student_teacher_ratio"].tolist()

    section("Statistical Tests")
    print(
        f"Pearson correlation (student_teacher_ratio vs avg_score): r={pearson_r:.4f}, p={pearson_p:.6g}"
    )
    print(
        f"Spearman correlation (student_teacher_ratio vs avg_score): r={spearman_r:.4f}, p={spearman_p:.6g}"
    )
    print(
        f"Median split Welch t-test: t={t_stat:.4f}, p={t_p:.6g}, "
        f"mean(low_ratio)={low_str_scores.mean():.3f}, mean(high_ratio)={high_str_scores.mean():.3f}"
    )
    print(f"Quartile ANOVA: F={anova_f:.4f}, p={anova_p:.6g}")
    print(
        "Simple OLS (avg_score ~ student_teacher_ratio): "
        f"coef={simple_coef:.4f}, p={simple_p:.6g}, 95%CI=[{simple_ci[0]:.4f}, {simple_ci[1]:.4f}], "
        f"R^2={ols_simple.rsquared:.4f}"
    )
    print(
        "Controlled OLS (with socioeconomic controls): "
        f"coef={multi_coef:.4f}, p={multi_p:.6g}, 95%CI=[{multi_ci[0]:.4f}, {multi_ci[1]:.4f}], "
        f"R^2={ols_multi.rsquared:.4f}"
    )

    section("Controlled OLS Coefficients")
    print(ols_multi.params.to_string(float_format=lambda x: f"{x:,.4f}"))
    print("\nControlled OLS p-values")
    print(ols_multi.pvalues.to_string(float_format=lambda x: f"{x:,.6g}"))

    # 3) Interpretable sklearn models
    X = data[features]

    lin = LinearRegression().fit(X, y)
    ridge = Ridge(alpha=1.0).fit(X, y)
    lasso = Lasso(alpha=0.05, max_iter=20000).fit(X, y)
    tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=20, random_state=42).fit(X, y)

    section("Interpretable Models: scikit-learn")
    lin_coef = pd.Series(lin.coef_, index=features).sort_values(key=lambda s: s.abs(), ascending=False)
    ridge_coef = pd.Series(ridge.coef_, index=features).sort_values(key=lambda s: s.abs(), ascending=False)
    lasso_coef = pd.Series(lasso.coef_, index=features).sort_values(key=lambda s: s.abs(), ascending=False)
    tree_imp = pd.Series(tree.feature_importances_, index=features).sort_values(ascending=False)

    print("LinearRegression coefficients (sorted by |coef|):")
    print(lin_coef.to_string(float_format=lambda x: f"{x:,.4f}"))

    print("\nRidge coefficients (sorted by |coef|):")
    print(ridge_coef.to_string(float_format=lambda x: f"{x:,.4f}"))

    print("\nLasso coefficients (sorted by |coef|):")
    print(lasso_coef.to_string(float_format=lambda x: f"{x:,.4f}"))

    print("\nDecisionTreeRegressor feature importances:")
    print(tree_imp.to_string(float_format=lambda x: f"{x:,.4f}"))

    print("\nDecisionTreeRegressor rules:")
    print(export_text(tree, feature_names=features, max_depth=3))

    # 4) Interpretable imodels models
    section("Interpretable Models: imodels")
    imodels_success = True
    try:
        from imodels import FIGSRegressor, HSTreeRegressor, RuleFitRegressor

        rulefit = RuleFitRegressor(
            n_estimators=100,
            max_rules=25,
            include_linear=True,
            random_state=42,
            cv=True,
        )
        rulefit.fit(X, y, feature_names=features)

        linear_part = pd.Series([float(v) for v in rulefit.coef[: len(features)]], index=features)
        linear_part = linear_part.sort_values(key=lambda s: s.abs(), ascending=False)
        print("RuleFit linear coefficients (sorted by |coef|):")
        print(linear_part.to_string(float_format=lambda x: f"{x:,.4f}"))

        rule_coefs = [float(v) for v in rulefit.coef[len(features) :]]
        rule_strings = [str(r) for r in getattr(rulefit, "rules_", [])]
        if len(rule_coefs) == len(rule_strings) and len(rule_strings) > 0:
            rules_df = pd.DataFrame({"rule": rule_strings, "coef": rule_coefs})
            rules_df["abs_coef"] = rules_df["coef"].abs()
            top_rules = rules_df.sort_values("abs_coef", ascending=False).head(10)
            print("\nTop RuleFit rules by |coefficient|:")
            for _, row in top_rules.iterrows():
                print(f"coef={row['coef']:.4f} | rule: {row['rule']}")
        else:
            print("\nRuleFit rules could not be aligned with coefficients in this version.")

        figs = FIGSRegressor(max_rules=12, random_state=42)
        figs.fit(X, y, feature_names=features)
        figs_imp = pd.Series(figs.feature_importances_, index=features).sort_values(ascending=False)
        print("\nFIGS feature importances:")
        print(figs_imp.to_string(float_format=lambda x: f"{x:,.4f}"))

        figs_text = str(figs).splitlines()
        print("\nFIGS model text (first 35 lines):")
        for line in figs_text[:35]:
            print(line)

        hst = HSTreeRegressor(max_leaf_nodes=12, random_state=42)
        hst.fit(X, y)
        hst_est = getattr(hst, "estimator_", None)
        if hst_est is not None and hasattr(hst_est, "feature_importances_"):
            hst_imp = pd.Series(hst_est.feature_importances_, index=features).sort_values(ascending=False)
            print("\nHSTree feature importances:")
            print(hst_imp.to_string(float_format=lambda x: f"{x:,.4f}"))
            print("\nHSTree tree structure:")
            print(export_text(hst_est, feature_names=features, max_depth=3))
        else:
            print("\nHSTree feature importances are unavailable in this version.")

    except Exception as exc:
        imodels_success = False
        print(f"imodels analysis skipped due to error: {exc}")

    # 5) Evidence synthesis -> Likert score
    score = 50
    evidence = []

    if pearson_p < 0.05 and pearson_r < 0:
        score += 20
        evidence.append("significant negative Pearson correlation")
    elif pearson_p < 0.05 and pearson_r > 0:
        score -= 20
        evidence.append("significant positive Pearson correlation (opposite direction)")
    else:
        score -= 15
        evidence.append("no significant Pearson correlation")

    if t_p < 0.05 and low_str_scores.mean() > high_str_scores.mean():
        score += 10
        evidence.append("lower-ratio districts have significantly higher mean scores")
    elif t_p < 0.05:
        score -= 10
        evidence.append("group mean difference is significant but opposite direction")
    else:
        score -= 5
        evidence.append("median-split t-test not significant")

    if anova_p < 0.05:
        score += 10
        evidence.append("ANOVA across ratio quartiles is significant")
    else:
        score -= 5
        evidence.append("ANOVA across ratio quartiles is not significant")

    if multi_p < 0.05 and multi_coef < 0:
        score += 20
        evidence.append("negative association remains significant after controls")
    elif multi_p < 0.05 and multi_coef > 0:
        score -= 25
        evidence.append("controlled association is significant but opposite direction")
    else:
        score -= 25
        evidence.append("association is not significant after socioeconomic controls")

    if not imodels_success:
        score -= 5

    score = int(np.clip(round(score), 0, 100))

    explanation = (
        f"Bivariate tests support the hypothesis: Pearson r={pearson_r:.3f} (p={pearson_p:.2e}), "
        f"Welch t-test p={t_p:.2e}, and quartile ANOVA p={anova_p:.2e}, with lower student-teacher ratio "
        f"districts scoring higher on average. However, in controlled OLS the student-teacher ratio coefficient "
        f"is {multi_coef:.3f} with p={multi_p:.3f} (95% CI [{multi_ci[0]:.3f}, {multi_ci[1]:.3f}]), indicating "
        f"the direct relationship weakens and is not statistically significant after adjusting for confounders "
        f"(notably lunch/poverty and income). Interpretable tree/rule models also prioritize socioeconomic variables "
        f"over student-teacher ratio. Evidence summary: {', '.join(evidence)}."
    )

    conclusion = {"response": score, "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f)

    section("Final Conclusion")
    print(json.dumps(conclusion, indent=2))
    print("\nSaved to conclusion.txt")


if __name__ == "__main__":
    main()
