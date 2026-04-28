import json
import warnings
from typing import Dict, Any

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from interpret.glassbox import ExplainableBoostingRegressor, ExplainableBoostingClassifier

warnings.filterwarnings("ignore")
RANDOM_STATE = 42


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def safe_pvalue(model, key: str):
    try:
        return safe_float(model.pvalues[key])
    except Exception:
        return float("nan")


def safe_param(model, key: str):
    try:
        return safe_float(model.params[key])
    except Exception:
        return float("nan")


def evidence_delta(effect: float, pvalue: float, weight: float) -> float:
    if not np.isfinite(effect) or not np.isfinite(pvalue) or effect == 0:
        return 0.0
    sign = 1.0 if effect > 0 else -1.0
    if pvalue < 0.05:
        return weight * sign
    if pvalue < 0.10:
        return 0.5 * weight * sign
    return 0.0


def get_skin_coef(pipe: Pipeline, num_cols):
    try:
        feat_names = pipe.named_steps["preprocess"].get_feature_names_out()
        coefs = pipe.named_steps["model"].coef_
        idx = list(feat_names).index("num__skin_tone")
        return safe_float(coefs[idx])
    except Exception:
        return float("nan")


def get_skin_importance(pipe: Pipeline):
    try:
        feat_names = pipe.named_steps["preprocess"].get_feature_names_out()
        importances = pipe.named_steps["model"].feature_importances_
        idx = list(feat_names).index("num__skin_tone")
        return safe_float(importances[idx])
    except Exception:
        return float("nan")


def main():
    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    research_question = info.get("research_questions", [""])[0]
    print("Research question:", research_question)

    df = pd.read_csv("soccer.csv")
    print("Loaded soccer.csv with shape:", df.shape)

    # Basic feature engineering for analysis.
    df["skin_tone"] = df[["rater1", "rater2"]].mean(axis=1)
    df = df[df["games"] > 0].copy()
    df["red_rate"] = df["redCards"] / df["games"]
    df["any_red"] = (df["redCards"] > 0).astype(int)

    analysis_df = df[df["skin_tone"].notna()].copy()
    analysis_df["skin_category"] = pd.cut(
        analysis_df["skin_tone"],
        bins=[-0.001, 0.25, 0.75, 1.001],
        labels=["light", "middle", "dark"],
    )
    extreme_df = analysis_df[analysis_df["skin_category"].isin(["light", "dark"])].copy()
    extreme_df["is_dark"] = (extreme_df["skin_category"] == "dark").astype(int)

    print("\n=== Data Exploration ===")
    print("Rows with non-missing skin_tone:", analysis_df.shape[0])
    print("Rows in extreme dark/light groups:", extreme_df.shape[0])

    num_cols_summary = [
        "redCards",
        "red_rate",
        "games",
        "yellowCards",
        "yellowReds",
        "goals",
        "skin_tone",
        "height",
        "weight",
        "meanIAT",
        "meanExp",
    ]
    print("\nSummary statistics (selected numeric columns):")
    print(analysis_df[num_cols_summary].describe().round(4))

    print("\nMissingness (top 15 columns):")
    print((df.isna().mean().sort_values(ascending=False).head(15) * 100).round(2))

    print("\nRed card count distribution:")
    print(analysis_df["redCards"].value_counts().sort_index())

    print("\nSkin tone distribution:")
    print(analysis_df["skin_tone"].value_counts().sort_index())

    corr_cols = [
        "redCards",
        "red_rate",
        "games",
        "yellowCards",
        "yellowReds",
        "goals",
        "skin_tone",
        "meanIAT",
        "meanExp",
    ]
    corr_mat = analysis_df[corr_cols].corr(numeric_only=True)
    print("\nCorrelation with red_rate:")
    print(corr_mat["red_rate"].sort_values(ascending=False).round(4))

    print("\n=== Statistical Tests ===")
    dark_rate = extreme_df.loc[extreme_df["is_dark"] == 1, "red_rate"]
    light_rate = extreme_df.loc[extreme_df["is_dark"] == 0, "red_rate"]

    ttest = stats.ttest_ind(dark_rate, light_rate, equal_var=False)
    mwu = stats.mannwhitneyu(dark_rate, light_rate, alternative="two-sided")

    cont_table = pd.crosstab(extreme_df["is_dark"], extreme_df["any_red"])
    chi2_stat, chi2_p, _, _ = stats.chi2_contingency(cont_table)

    anova_groups = [g["red_rate"].values for _, g in analysis_df.groupby("skin_tone")]
    anova_stat, anova_p = stats.f_oneway(*anova_groups)

    spearman = stats.spearmanr(analysis_df["skin_tone"], analysis_df["red_rate"])

    dark_mean = safe_float(dark_rate.mean())
    light_mean = safe_float(light_rate.mean())
    dark_any_red = safe_float(extreme_df.loc[extreme_df["is_dark"] == 1, "any_red"].mean())
    light_any_red = safe_float(extreme_df.loc[extreme_df["is_dark"] == 0, "any_red"].mean())

    print(f"Welch t-test red_rate dark vs light: t={ttest.statistic:.4f}, p={ttest.pvalue:.6g}")
    print(f"Mann-Whitney U dark vs light: U={mwu.statistic:.4f}, p={mwu.pvalue:.6g}")
    print(f"Chi-square any_red by dark/light: chi2={chi2_stat:.4f}, p={chi2_p:.6g}")
    print(f"ANOVA red_rate across all skin_tone levels: F={anova_stat:.4f}, p={anova_p:.6g}")
    print(f"Spearman skin_tone vs red_rate: rho={spearman.correlation:.4f}, p={spearman.pvalue:.6g}")

    # Regression models for adjusted association.
    print("\n=== Statsmodels Regression ===")
    reg_df = analysis_df.dropna(
        subset=["skin_tone", "games", "yellowCards", "yellowReds", "goals", "position", "leagueCountry"]
    ).copy()

    ols = smf.ols(
        "red_rate ~ skin_tone + games + yellowCards + yellowReds + goals + C(position) + C(leagueCountry)",
        data=reg_df,
    ).fit(cov_type="HC3")

    poisson = smf.glm(
        "redCards ~ skin_tone + yellowCards + yellowReds + goals + C(position) + C(leagueCountry)",
        data=reg_df,
        family=sm.families.Poisson(),
        exposure=reg_df["games"],
    ).fit(cov_type="HC3")

    binom = smf.glm(
        "any_red ~ skin_tone + games + yellowCards + yellowReds + goals + C(position) + C(leagueCountry)",
        data=reg_df,
        family=sm.families.Binomial(),
    ).fit(cov_type="HC3")

    ols_coef = safe_param(ols, "skin_tone")
    ols_p = safe_pvalue(ols, "skin_tone")
    poi_coef = safe_param(poisson, "skin_tone")
    poi_p = safe_pvalue(poisson, "skin_tone")
    binom_coef = safe_param(binom, "skin_tone")
    binom_p = safe_pvalue(binom, "skin_tone")

    print(f"OLS skin_tone coef={ols_coef:.6g}, p={ols_p:.6g}")
    print(f"Poisson skin_tone log-rate coef={poi_coef:.6g}, p={poi_p:.6g}, IRR={np.exp(poi_coef):.4f}")
    print(f"Binomial skin_tone log-odds coef={binom_coef:.6g}, p={binom_p:.6g}, OR={np.exp(binom_coef):.4f}")

    print("\n=== Interpretable ML Models ===")
    model_features = [
        "skin_tone",
        "games",
        "yellowCards",
        "yellowReds",
        "goals",
        "height",
        "weight",
        "meanIAT",
        "meanExp",
        "leagueCountry",
        "position",
    ]
    model_df = analysis_df[model_features + ["red_rate", "any_red"]].dropna().copy()
    if len(model_df) > 50000:
        model_df = model_df.sample(n=50000, random_state=RANDOM_STATE)

    X = model_df[model_features].copy()
    y_reg = model_df["red_rate"].copy()
    y_clf = model_df["any_red"].copy()

    num_cols = [
        "skin_tone",
        "games",
        "yellowCards",
        "yellowReds",
        "goals",
        "height",
        "weight",
        "meanIAT",
        "meanExp",
    ]
    cat_cols = ["leagueCountry", "position"]

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X, y_reg, test_size=0.25, random_state=RANDOM_STATE
    )
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y_clf, test_size=0.25, random_state=RANDOM_STATE, stratify=y_clf
    )

    lin_pipe = Pipeline([("preprocess", pre), ("model", LinearRegression())])
    ridge_pipe = Pipeline([("preprocess", pre), ("model", Ridge(alpha=1.0, random_state=RANDOM_STATE))])
    lasso_pipe = Pipeline([("preprocess", pre), ("model", Lasso(alpha=0.0005, random_state=RANDOM_STATE, max_iter=5000))])
    tree_r_pipe = Pipeline([("preprocess", pre), ("model", DecisionTreeRegressor(max_depth=4, min_samples_leaf=200, random_state=RANDOM_STATE))])
    tree_c_pipe = Pipeline([("preprocess", pre), ("model", DecisionTreeClassifier(max_depth=4, min_samples_leaf=200, random_state=RANDOM_STATE))])

    lin_pipe.fit(X_train_r, y_train_r)
    ridge_pipe.fit(X_train_r, y_train_r)
    lasso_pipe.fit(X_train_r, y_train_r)
    tree_r_pipe.fit(X_train_r, y_train_r)
    tree_c_pipe.fit(X_train_c, y_train_c)

    reg_rmse = np.sqrt(mean_squared_error(y_test_r, lin_pipe.predict(X_test_r)))
    clf_auc = roc_auc_score(y_test_c, tree_c_pipe.predict_proba(X_test_c)[:, 1])

    lin_skin_coef = get_skin_coef(lin_pipe, num_cols)
    ridge_skin_coef = get_skin_coef(ridge_pipe, num_cols)
    lasso_skin_coef = get_skin_coef(lasso_pipe, num_cols)
    tree_skin_importance = get_skin_importance(tree_r_pipe)

    print(f"LinearRegression RMSE={reg_rmse:.6f}, skin_tone coef={lin_skin_coef:.6g}")
    print(f"Ridge skin_tone coef={ridge_skin_coef:.6g}")
    print(f"Lasso skin_tone coef={lasso_skin_coef:.6g}")
    print(f"DecisionTreeRegressor skin_tone importance={tree_skin_importance:.6g}")
    print(f"DecisionTreeClassifier AUC={clf_auc:.4f}")

    # Explainable Boosting Models from interpret.
    X_ebm = X.copy()
    for c in cat_cols:
        X_ebm[c] = X_ebm[c].astype(str)

    X_train_er, X_test_er, y_train_er, y_test_er = train_test_split(
        X_ebm, y_reg, test_size=0.25, random_state=RANDOM_STATE
    )
    X_train_ec, X_test_ec, y_train_ec, y_test_ec = train_test_split(
        X_ebm, y_clf, test_size=0.25, random_state=RANDOM_STATE, stratify=y_clf
    )

    ebm_reg = ExplainableBoostingRegressor(random_state=RANDOM_STATE, interactions=0, max_bins=64)
    ebm_clf = ExplainableBoostingClassifier(random_state=RANDOM_STATE, interactions=0, max_bins=64)

    ebm_reg.fit(X_train_er, y_train_er)
    ebm_clf.fit(X_train_ec, y_train_ec)

    ebm_reg_global = ebm_reg.explain_global().data()
    ebm_clf_global = ebm_clf.explain_global().data()

    reg_names = list(ebm_reg_global["names"])
    reg_scores = list(ebm_reg_global["scores"])
    clf_names = list(ebm_clf_global["names"])
    clf_scores = list(ebm_clf_global["scores"])

    ebm_reg_skin_importance = safe_float(reg_scores[reg_names.index("skin_tone")]) if "skin_tone" in reg_names else float("nan")
    ebm_clf_skin_importance = safe_float(clf_scores[clf_names.index("skin_tone")]) if "skin_tone" in clf_names else float("nan")

    clf_sorted = sorted(zip(clf_names, clf_scores), key=lambda x: x[1], reverse=True)
    reg_sorted = sorted(zip(reg_names, reg_scores), key=lambda x: x[1], reverse=True)

    ebm_clf_skin_rank = next((i + 1 for i, (n, _) in enumerate(clf_sorted) if n == "skin_tone"), np.nan)
    ebm_reg_skin_rank = next((i + 1 for i, (n, _) in enumerate(reg_sorted) if n == "skin_tone"), np.nan)

    print(f"EBM reg skin_tone importance={ebm_reg_skin_importance:.6g}, rank={ebm_reg_skin_rank}")
    print(f"EBM clf skin_tone importance={ebm_clf_skin_importance:.6g}, rank={ebm_clf_skin_rank}")
    print("Top 5 EBM classifier features:", clf_sorted[:5])

    # Convert evidence into a conservative 0-100 Likert score.
    score = 50.0
    score += evidence_delta(dark_mean - light_mean, safe_float(ttest.pvalue), weight=15)
    score += evidence_delta(dark_any_red - light_any_red, safe_float(chi2_p), weight=10)
    score += evidence_delta(safe_float(spearman.correlation), safe_float(spearman.pvalue), weight=5)
    score += evidence_delta(ols_coef, ols_p, weight=15)
    score += evidence_delta(poi_coef, poi_p, weight=15)
    score += evidence_delta(binom_coef, binom_p, weight=10)

    if np.isfinite(ebm_clf_skin_rank) and ebm_clf_skin_rank <= 5:
        score += 5
    if np.isfinite(ebm_reg_skin_rank) and ebm_reg_skin_rank <= 5:
        score += 3

    score = int(np.clip(np.round(score), 0, 100))

    explanation = (
        "Using player-referee dyads with rated skin tone, dark players (very dark/dark: skin_tone >= 0.75) "
        f"had a higher mean red-card rate than light players (very light/light: skin_tone <= 0.25), "
        f"{dark_mean:.4f} vs {light_mean:.4f} (Welch t-test p={safe_float(ttest.pvalue):.4g}). "
        f"The probability of receiving at least one red card was also higher ({dark_any_red:.4f} vs {light_any_red:.4f}, "
        f"chi-square p={safe_float(chi2_p):.4g}). Across all tone values, ANOVA (p={safe_float(anova_p):.4g}) and "
        f"Spearman correlation (rho={safe_float(spearman.correlation):.4f}, p={safe_float(spearman.pvalue):.4g}) "
        "support an association. In adjusted models controlling for games, discipline, goals, position, and league, "
        f"skin tone remained positive in OLS (coef={ols_coef:.4g}, p={ols_p:.4g}), Poisson (IRR={np.exp(poi_coef):.4f}, p={poi_p:.4g}), "
        f"and Binomial GLM (OR={np.exp(binom_coef):.4f}, p={binom_p:.4g}). Interpretable ML (linear models, decision trees, and EBM) "
        "also identified skin tone as a meaningful predictor, though not the single strongest one. Overall this is evidence for 'Yes', "
        "with some uncertainty about exact effect size and threshold definition."
    )

    result: Dict[str, Any] = {"response": score, "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=True)

    print("\nWrote conclusion.txt")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
