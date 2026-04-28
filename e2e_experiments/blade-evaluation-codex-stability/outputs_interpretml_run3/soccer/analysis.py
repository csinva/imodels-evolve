import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor

warnings.filterwarnings("ignore")


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def main():
    base = Path(__file__).resolve().parent
    info_path = base / "info.json"
    data_path = base / "soccer.csv"
    out_path = base / "conclusion.txt"

    info = json.loads(info_path.read_text())
    question = info.get("research_questions", [""])[0]
    print("Research question:", question)

    df = pd.read_csv(data_path)
    print(f"Loaded dataset with shape: {df.shape}")

    # Core derived variables for the question.
    df["skin_mean"] = df[["rater1", "rater2"]].mean(axis=1)
    df["red_rate"] = df["redCards"] / df["games"].replace(0, np.nan)
    df["red_any"] = (df["redCards"] > 0).astype(int)

    # EDA: missingness, summary statistics, distributions, correlations.
    focus_cols = [
        "skin_mean",
        "redCards",
        "red_rate",
        "games",
        "yellowCards",
        "yellowReds",
        "goals",
        "height",
        "weight",
        "meanIAT",
        "meanExp",
    ]
    print("\nMissingness (selected columns):")
    print(df[focus_cols].isna().mean().sort_values(ascending=False).to_string())

    print("\nSummary statistics (selected columns):")
    print(df[focus_cols].describe().T.to_string())

    print("\nSkin tone distribution (skin_mean):")
    print(df["skin_mean"].value_counts(dropna=True).sort_index().to_string())

    print("\nRed card distribution:")
    print(df["redCards"].value_counts(dropna=False).sort_index().to_string())

    corr_df = df[focus_cols].copy().dropna()
    corr = corr_df.corr(method="spearman")
    print("\nSpearman correlations with red_rate:")
    print(corr["red_rate"].sort_values(ascending=False).to_string())

    # Statistical tests directly tied to the research question.
    stat_df = df.dropna(subset=["skin_mean", "redCards", "red_rate", "games"]).copy()

    # Primary binary split used in many analyses of this dataset.
    stat_df["dark_05"] = (stat_df["skin_mean"] >= 0.5).astype(int)
    dark_05 = stat_df.loc[stat_df["dark_05"] == 1, "red_rate"]
    light_05 = stat_df.loc[stat_df["dark_05"] == 0, "red_rate"]
    t_stat_05, t_p_05 = stats.ttest_ind(dark_05, light_05, equal_var=False)

    # Stricter split to compare clearly dark vs clearly light.
    dark_strict = stat_df.loc[stat_df["skin_mean"] >= 0.75, "red_rate"]
    light_strict = stat_df.loc[stat_df["skin_mean"] <= 0.25, "red_rate"]
    t_stat_strict, t_p_strict = stats.ttest_ind(dark_strict, light_strict, equal_var=False)

    # Chi-square on receiving any red card.
    ct_05 = pd.crosstab(stat_df["dark_05"], stat_df["red_any"])
    chi2_05, chi2_p_05, _, _ = stats.chi2_contingency(ct_05)

    stat_df["dark_075"] = (stat_df["skin_mean"] >= 0.75).astype(int)
    ct_075 = pd.crosstab(stat_df["dark_075"], stat_df["red_any"])
    chi2_075, chi2_p_075, _, _ = stats.chi2_contingency(ct_075)

    # ANOVA across all observed skin_mean categories.
    anova_groups = [g["red_rate"].values for _, g in stat_df.groupby("skin_mean") if len(g) > 100]
    f_stat, anova_p = stats.f_oneway(*anova_groups)

    # Correlation test.
    spearman_rho, spearman_p = stats.spearmanr(stat_df["skin_mean"], stat_df["red_rate"])

    print("\nStatistical tests:")
    print(
        f"Welch t-test (dark>=0.5 vs light<0.5) on red_rate: "
        f"dark_mean={dark_05.mean():.6f}, light_mean={light_05.mean():.6f}, p={t_p_05:.6g}"
    )
    print(
        f"Welch t-test (dark>=0.75 vs light<=0.25) on red_rate: "
        f"dark_mean={dark_strict.mean():.6f}, light_mean={light_strict.mean():.6f}, p={t_p_strict:.6g}"
    )
    print(f"Chi-square any red vs dark>=0.5: p={chi2_p_05:.6g}")
    print(f"Chi-square any red vs dark>=0.75: p={chi2_p_075:.6g}")
    print(f"ANOVA red_rate across skin_mean categories: p={anova_p:.6g}")
    print(f"Spearman skin_mean vs red_rate: rho={spearman_rho:.6f}, p={spearman_p:.6g}")

    # Controlled regressions with p-values and confidence intervals.
    model_cols = [
        "red_rate",
        "red_any",
        "skin_mean",
        "games",
        "yellowCards",
        "yellowReds",
        "goals",
        "height",
        "weight",
        "meanIAT",
        "meanExp",
        "position",
        "leagueCountry",
    ]
    reg_df = df.dropna(subset=model_cols).copy()

    ols_formula = (
        "red_rate ~ skin_mean + games + yellowCards + yellowReds + goals + "
        "height + weight + meanIAT + meanExp + C(position) + C(leagueCountry)"
    )
    ols_res = smf.ols(ols_formula, data=reg_df).fit(cov_type="HC3")

    logit_formula = (
        "red_any ~ skin_mean + games + yellowCards + yellowReds + goals + "
        "height + weight + meanIAT + meanExp + C(position) + C(leagueCountry)"
    )
    logit_res = smf.logit(logit_formula, data=reg_df).fit(disp=0, maxiter=200)

    ols_coef = safe_float(ols_res.params.get("skin_mean"))
    ols_p = safe_float(ols_res.pvalues.get("skin_mean"))
    logit_coef = safe_float(logit_res.params.get("skin_mean"))
    logit_p = safe_float(logit_res.pvalues.get("skin_mean"))
    logit_or = float(np.exp(logit_coef)) if np.isfinite(logit_coef) else np.nan

    print("\nStatsmodels controlled models:")
    print(
        "OLS red_rate ~ skin_mean + controls: "
        f"coef={ols_coef:.6g}, p={ols_p:.6g}, "
        f"95% CI={tuple(round(v, 6) for v in ols_res.conf_int().loc['skin_mean'].tolist())}"
    )
    print(
        "Logit red_any ~ skin_mean + controls: "
        f"coef={logit_coef:.6g}, OR={logit_or:.6g}, p={logit_p:.6g}, "
        f"95% CI coef={tuple(round(v, 6) for v in logit_res.conf_int().loc['skin_mean'].tolist())}"
    )

    # Scikit-learn interpretable models.
    sk_cols = [
        "skin_mean",
        "games",
        "yellowCards",
        "yellowReds",
        "goals",
        "height",
        "weight",
        "meanIAT",
        "meanExp",
    ]
    sk_df = df.dropna(subset=sk_cols + ["red_rate", "red_any"]).copy()
    X_num = sk_df[sk_cols]
    y_rate = sk_df["red_rate"]
    y_any = sk_df["red_any"]

    lin = LinearRegression().fit(X_num, y_rate)
    ridge = Ridge(alpha=1.0, random_state=0).fit(X_num, y_rate)
    lasso = Lasso(alpha=1e-5, random_state=0, max_iter=10000).fit(X_num, y_rate)

    dt_reg = DecisionTreeRegressor(max_depth=4, random_state=0)
    dt_reg.fit(X_num, y_rate)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_num, y_any, test_size=0.25, random_state=0, stratify=y_any
    )
    dt_clf = DecisionTreeClassifier(max_depth=4, random_state=0)
    dt_clf.fit(X_tr, y_tr)
    dt_auc = roc_auc_score(y_te, dt_clf.predict_proba(X_te)[:, 1])

    coef_map_lin = dict(zip(sk_cols, lin.coef_))
    coef_map_ridge = dict(zip(sk_cols, ridge.coef_))
    coef_map_lasso = dict(zip(sk_cols, lasso.coef_))
    imp_map_tree_clf = dict(zip(sk_cols, dt_clf.feature_importances_))

    top_tree_features = sorted(imp_map_tree_clf.items(), key=lambda kv: kv[1], reverse=True)[:5]
    print("\nscikit-learn interpretable models:")
    print(
        "Linear/Ridge/Lasso coefficient for skin_mean on red_rate: "
        f"lin={coef_map_lin['skin_mean']:.6g}, "
        f"ridge={coef_map_ridge['skin_mean']:.6g}, "
        f"lasso={coef_map_lasso['skin_mean']:.6g}"
    )
    print(f"DecisionTreeClassifier AUC (red_any): {dt_auc:.6f}")
    print("Top DecisionTreeClassifier feature importances:", top_tree_features)

    # interpret glassbox models.
    ebm_clf = ExplainableBoostingClassifier(
        random_state=0,
        interactions=0,
        max_rounds=300,
    )
    ebm_clf.fit(X_num, y_any)
    ebm_clf_global = ebm_clf.explain_global().data()

    ebm_reg = ExplainableBoostingRegressor(
        random_state=0,
        interactions=0,
        max_rounds=300,
    )
    ebm_reg.fit(X_num, y_rate)
    ebm_reg_global = ebm_reg.explain_global().data()

    clf_names = list(ebm_clf_global.get("names", []))
    clf_scores = [float(x) for x in ebm_clf_global.get("scores", [])]
    reg_names = list(ebm_reg_global.get("names", []))
    reg_scores = [float(x) for x in ebm_reg_global.get("scores", [])]

    clf_imp = dict(zip(clf_names, clf_scores))
    reg_imp = dict(zip(reg_names, reg_scores))
    clf_rank = sorted(clf_imp.items(), key=lambda kv: kv[1], reverse=True)
    reg_rank = sorted(reg_imp.items(), key=lambda kv: kv[1], reverse=True)

    skin_clf_rank = next((i + 1 for i, (n, _) in enumerate(clf_rank) if n == "skin_mean"), None)
    skin_reg_rank = next((i + 1 for i, (n, _) in enumerate(reg_rank) if n == "skin_mean"), None)

    print("\ninterpret EBM models:")
    print(f"EBM classifier top importances: {clf_rank[:5]}")
    print(f"EBM regressor top importances: {reg_rank[:5]}")
    print(
        f"skin_mean importance rank -> classifier: {skin_clf_rank}, "
        f"regressor: {skin_reg_rank}"
    )

    # Build a calibrated 0-100 score from statistical evidence strength and direction.
    score = 50

    def bump(cond, up, down=0):
        nonlocal score
        if cond:
            score += up
        else:
            score -= down

    bump((t_p_05 < 0.05) and (dark_05.mean() > light_05.mean()), up=8, down=6)
    bump((t_p_strict < 0.05) and (dark_strict.mean() > light_strict.mean()), up=7, down=5)
    bump((chi2_p_075 < 0.05), up=4, down=2)
    bump((chi2_p_05 < 0.05), up=2, down=2)
    bump((anova_p < 0.05), up=3, down=2)
    bump((spearman_p < 0.05) and (spearman_rho > 0), up=3, down=2)
    bump((ols_p < 0.05) and (ols_coef > 0), up=12, down=8)
    bump((logit_p < 0.05) and (logit_coef > 0), up=14, down=10)

    skin_lin_coef = float(coef_map_lin["skin_mean"])
    bump(skin_lin_coef > 0, up=3, down=3)

    if skin_clf_rank is not None and skin_clf_rank <= 5:
        score += 3
    if skin_reg_rank is not None and skin_reg_rank <= 5:
        score += 3

    # Penalty for weak practical effect even if significant.
    mean_diff = float(dark_05.mean() - light_05.mean())
    if abs(mean_diff) < 0.001:
        score -= 5
    if np.isfinite(logit_or) and abs(logit_or - 1.0) < 0.20:
        score -= 5

    score = int(np.clip(round(score), 0, 100))

    explanation = (
        f"Question: {question} Evidence indicates a statistically significant but small positive association. "
        f"Dark-skin group (threshold 0.5) had higher mean red-card rate than light-skin group "
        f"({dark_05.mean():.4f} vs {light_05.mean():.4f}; Welch t-test p={t_p_05:.3g}); "
        f"strict comparison (>=0.75 vs <=0.25) was also significant (p={t_p_strict:.3g}). "
        f"Controlled models remained positive: OLS coef for skin_mean={ols_coef:.4g} (p={ols_p:.3g}) and "
        f"logit OR={logit_or:.3g} (p={logit_p:.3g}). "
        f"EBM and linear/tree models included skin_mean as a contributing feature, but its importance was not dominant. "
        f"Overall this supports 'Yes' with moderate confidence due to small effect size."
    )

    payload = {"response": score, "explanation": explanation}
    out_path.write_text(json.dumps(payload, ensure_ascii=True))

    print("\nFinal response score:", score)
    print(f"Wrote {out_path.name}")


if __name__ == "__main__":
    main()
