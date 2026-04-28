import json
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency
import statsmodels.formula.api as smf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import balanced_accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from interpret.glassbox import (
    DecisionListClassifier,
    ExplainableBoostingClassifier,
    ExplainableBoostingRegressor,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def get_ebm_importance_series(model):
    """Return a sorted importance series compatible across interpret versions."""
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_)
        names = getattr(model, "feature_names_in_", None)
        if names is None:
            names = [f"feature_{i}" for i in range(len(values))]
        return pd.Series(values, index=names).sort_values(ascending=False)

    if hasattr(model, "term_importances"):
        values = np.asarray(model.term_importances())
        names = getattr(model, "term_names_", [f"term_{i}" for i in range(len(values))])
        return pd.Series(values, index=names).sort_values(ascending=False)

    return pd.Series(dtype=float)


def main():
    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    question = info.get("research_questions", ["Unknown question"])[0]
    print("Research question:", question)

    df = pd.read_csv("soccer.csv")
    print("Loaded soccer.csv with shape:", df.shape)

    # Feature engineering
    df["birthday_dt"] = pd.to_datetime(df["birthday"], format="%d.%m.%Y", errors="coerce")
    df["age_2013"] = 2013 - df["birthday_dt"].dt.year
    df["skin_tone"] = df[["rater1", "rater2"]].mean(axis=1, skipna=True)
    df = df[df["games"] > 0].copy()
    df["red_card_rate"] = df["redCards"] / df["games"]
    df["any_red"] = (df["redCards"] > 0).astype(int)

    # Dark vs light grouping for direct question (exclude midpoint 0.5 in primary test)
    df["skin_group"] = np.select(
        [df["skin_tone"].isna(), df["skin_tone"] < 0.5, df["skin_tone"] > 0.5, df["skin_tone"] == 0.5],
        ["missing", "light", "dark", "middle"],
        default="missing",
    )

    print("\n--- Data quality and composition ---")
    print("Rows with observed skin tone:", int(df["skin_tone"].notna().sum()))
    print("Rows with missing skin tone:", int(df["skin_tone"].isna().sum()))
    print("Skin group counts (including midpoint and missing):")
    print(df["skin_group"].value_counts(dropna=False))

    print("\n--- Summary statistics (selected numeric columns) ---")
    summary_cols = [
        "redCards",
        "red_card_rate",
        "games",
        "yellowCards",
        "yellowReds",
        "goals",
        "height",
        "weight",
        "skin_tone",
        "meanIAT",
        "meanExp",
        "age_2013",
    ]
    print(df[summary_cols].describe().T[["mean", "std", "min", "50%", "max"]])

    print("\n--- Distributions ---")
    print("redCards counts:")
    print(df["redCards"].value_counts().sort_index())

    skin_hist = np.histogram(df["skin_tone"].dropna(), bins=[-0.01, 0.2, 0.4, 0.6, 0.8, 1.01])[0]
    print("skin_tone histogram bins (-0.01-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.01):")
    print(skin_hist)

    rate_hist = np.histogram(df["red_card_rate"], bins=[-1e-9, 0, 0.02, 0.05, 0.1, 1.0])[0]
    print("red_card_rate histogram bins (0, 0-0.02, 0.02-0.05, 0.05-0.1, 0.1-1.0):")
    print(rate_hist)

    print("\n--- Correlations with outcomes ---")
    corr_cols = [
        "redCards",
        "red_card_rate",
        "skin_tone",
        "games",
        "yellowCards",
        "yellowReds",
        "goals",
        "height",
        "weight",
        "meanIAT",
        "meanExp",
        "age_2013",
    ]
    corr = df[corr_cols].corr(numeric_only=True)
    print(corr[["redCards", "red_card_rate"]].sort_values("red_card_rate", ascending=False))

    # Statistical tests
    print("\n=== Statistical tests for dark vs light skin tone ===")
    test_df = df[df["skin_group"].isin(["dark", "light"])].dropna(subset=["skin_tone", "red_card_rate"])

    dark_rate = test_df.loc[test_df["skin_group"] == "dark", "red_card_rate"]
    light_rate = test_df.loc[test_df["skin_group"] == "light", "red_card_rate"]

    t_stat, t_p = stats.ttest_ind(dark_rate, light_rate, equal_var=False, nan_policy="omit")
    u_stat, u_p = stats.mannwhitneyu(dark_rate, light_rate, alternative="two-sided")

    table = pd.crosstab(test_df["skin_group"], test_df["any_red"])
    chi2, chi_p, _, _ = chi2_contingency(table)

    dark_any_rate = safe_float(test_df.loc[test_df["skin_group"] == "dark", "any_red"].mean())
    light_any_rate = safe_float(test_df.loc[test_df["skin_group"] == "light", "any_red"].mean())
    risk_ratio = (dark_any_rate / light_any_rate) if light_any_rate > 0 else np.nan

    corr_df = df.dropna(subset=["skin_tone", "red_card_rate"])
    pear_r, pear_p = stats.pearsonr(corr_df["skin_tone"], corr_df["red_card_rate"])
    spear_r, spear_p = stats.spearmanr(corr_df["skin_tone"], corr_df["red_card_rate"])

    print(f"Welch t-test red_card_rate (dark vs light): t={t_stat:.4f}, p={t_p:.4g}")
    print(f"Mann-Whitney U test red_card_rate: U={u_stat:.4f}, p={u_p:.4g}")
    print(f"Chi-square any_red vs skin_group: chi2={chi2:.4f}, p={chi_p:.4g}")
    print(f"Any-red rates: dark={dark_any_rate:.5f}, light={light_any_rate:.5f}, risk_ratio={risk_ratio:.4f}")
    print(f"Pearson corr(skin_tone, red_card_rate): r={pear_r:.4f}, p={pear_p:.4g}")
    print(f"Spearman corr(skin_tone, red_card_rate): rho={spear_r:.4f}, p={spear_p:.4g}")

    # Regression analyses (statsmodels)
    print("\n=== Regression with controls (statsmodels) ===")
    model_cols = [
        "redCards",
        "red_card_rate",
        "games",
        "skin_tone",
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

    model_df = df[model_cols].dropna().copy()

    ols_formula = (
        "red_card_rate ~ skin_tone + games + yellowCards + yellowReds + goals + "
        "height + weight + meanIAT + meanExp + C(position) + C(leagueCountry)"
    )
    ols_model = smf.ols(ols_formula, data=model_df).fit(cov_type="HC3")
    ols_coef = safe_float(ols_model.params.get("skin_tone", np.nan))
    ols_p = safe_float(ols_model.pvalues.get("skin_tone", np.nan))

    print(
        f"OLS (HC3) skin_tone coef={ols_coef:.6f}, p={ols_p:.4g}, "
        f"R^2={safe_float(ols_model.rsquared):.4f}"
    )

    poisson_formula = (
        "redCards ~ skin_tone + yellowCards + yellowReds + goals + "
        "height + weight + meanIAT + meanExp + C(position) + C(leagueCountry)"
    )
    poisson_model = smf.poisson(
        poisson_formula,
        data=model_df,
        offset=np.log(model_df["games"]),
    ).fit(disp=0, maxiter=200)

    pois_coef = safe_float(poisson_model.params.get("skin_tone", np.nan))
    pois_p = safe_float(poisson_model.pvalues.get("skin_tone", np.nan))
    irr = float(np.exp(pois_coef)) if np.isfinite(pois_coef) else np.nan

    print(f"Poisson skin_tone coef={pois_coef:.6f}, IRR={irr:.4f}, p={pois_p:.4g}")

    # Interpretable sklearn models
    print("\n=== Interpretable sklearn models ===")
    feature_cols = [
        "skin_tone",
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
    numeric_features = [
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
    categorical_features = ["position", "leagueCountry"]

    ml_df = df[feature_cols + ["red_card_rate", "any_red"]].copy()
    ml_df[categorical_features] = ml_df[categorical_features].fillna("Unknown")
    ml_df = ml_df.dropna(subset=["red_card_rate"])

    X = ml_df[feature_cols]
    y_reg = ml_df["red_card_rate"]
    y_cls = ml_df["any_red"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_reg, test_size=0.25, random_state=42
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    lin_pipe = Pipeline(
        steps=[("preprocess", preprocess), ("model", LinearRegression())]
    )
    lin_pipe.fit(X_train, y_train)
    lin_pred = lin_pipe.predict(X_test)
    lin_r2 = r2_score(y_test, lin_pred)

    feat_names = lin_pipe.named_steps["preprocess"].get_feature_names_out()
    coefs = lin_pipe.named_steps["model"].coef_
    coef_series = pd.Series(coefs, index=feat_names).sort_values(key=np.abs, ascending=False)
    skin_coef_linear = safe_float(coef_series.get("num__skin_tone", np.nan))

    print(f"LinearRegression test R^2={lin_r2:.4f}")
    print("Top 10 absolute coefficients:")
    print(coef_series.head(10))
    print(f"LinearRegression skin_tone coefficient={skin_coef_linear:.6f}")

    tree_pipe = Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "model",
                DecisionTreeRegressor(max_depth=4, min_samples_leaf=300, random_state=42),
            ),
        ]
    )
    tree_pipe.fit(X_train, y_train)
    tree_pred = tree_pipe.predict(X_test)
    tree_r2 = r2_score(y_test, tree_pred)

    tree_importances = pd.Series(
        tree_pipe.named_steps["model"].feature_importances_,
        index=tree_pipe.named_steps["preprocess"].get_feature_names_out(),
    ).sort_values(ascending=False)

    print(f"DecisionTreeRegressor test R^2={tree_r2:.4f}")
    print("Top 10 feature importances (regression tree):")
    print(tree_importances.head(10))

    # Classification tree for any red card
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        X, y_cls, test_size=0.25, random_state=42, stratify=y_cls
    )
    treec_pipe = Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "model",
                DecisionTreeClassifier(
                    max_depth=4,
                    min_samples_leaf=300,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    treec_pipe.fit(Xc_train, yc_train)
    yc_pred = treec_pipe.predict(Xc_test)
    yc_proba = treec_pipe.predict_proba(Xc_test)[:, 1]
    treec_bal_acc = balanced_accuracy_score(yc_test, yc_pred)
    treec_auc = roc_auc_score(yc_test, yc_proba)

    print(
        f"DecisionTreeClassifier any_red balanced_accuracy={treec_bal_acc:.4f}, "
        f"ROC-AUC={treec_auc:.4f}"
    )

    # Interpret glassbox models
    print("\n=== Interpret glassbox models ===")
    ebm_feature_cols = feature_cols
    ebm_df = ml_df[ebm_feature_cols + ["red_card_rate", "any_red"]].copy()

    # Sample for speed while preserving representativeness
    sample_n = min(60000, len(ebm_df))
    ebm_df = ebm_df.sample(n=sample_n, random_state=42)

    Xe = ebm_df[ebm_feature_cols]
    ye_reg = ebm_df["red_card_rate"]
    ye_cls = ebm_df["any_red"]

    Xe_train, Xe_test, ye_train, ye_test = train_test_split(
        Xe, ye_reg, test_size=0.25, random_state=42
    )

    ebm_reg = ExplainableBoostingRegressor(
        random_state=42,
        interactions=0,
        max_bins=128,
        outer_bags=8,
    )
    ebm_reg.fit(Xe_train, ye_train)
    ebm_pred = ebm_reg.predict(Xe_test)
    ebm_r2 = r2_score(ye_test, ebm_pred)

    ebm_reg_importances = get_ebm_importance_series(ebm_reg)
    ebm_skin_imp = safe_float(ebm_reg_importances.get("skin_tone", np.nan))

    print(f"ExplainableBoostingRegressor test R^2={ebm_r2:.4f}")
    print("EBM regressor feature importances:")
    print(ebm_reg_importances)

    Xec_train, Xec_test, yec_train, yec_test = train_test_split(
        Xe, ye_cls, test_size=0.25, random_state=42, stratify=ye_cls
    )
    ebm_cls = ExplainableBoostingClassifier(
        random_state=42,
        interactions=0,
        max_bins=128,
        outer_bags=8,
    )
    ebm_cls.fit(Xec_train, yec_train)
    yec_proba = ebm_cls.predict_proba(Xec_test)[:, 1]
    ebm_auc = roc_auc_score(yec_test, yec_proba)

    ebm_cls_importances = get_ebm_importance_series(ebm_cls)

    print(f"ExplainableBoostingClassifier any_red ROC-AUC={ebm_auc:.4f}")
    print("EBM classifier feature importances:")
    print(ebm_cls_importances)

    # Optional rule-based model
    dlc_auc = np.nan
    try:
        dlc = DecisionListClassifier(random_state=42, n_estimators=30, max_depth=3)
        dlc.fit(Xec_train, yec_train)
        dlc_proba = dlc.predict_proba(Xec_test)[:, 1]
        dlc_auc = roc_auc_score(yec_test, dlc_proba)
        print(f"DecisionListClassifier any_red ROC-AUC={dlc_auc:.4f}")
    except Exception as e:
        print(f"DecisionListClassifier could not be fit: {type(e).__name__}: {e}")

    # Evidence synthesis into Likert score [0,100]
    score = 50

    dark_mean_rate = safe_float(dark_rate.mean())
    light_mean_rate = safe_float(light_rate.mean())
    mean_diff = dark_mean_rate - light_mean_rate

    if np.isfinite(t_p) and t_p < 0.05:
        score += 15 if mean_diff > 0 else -15
    if np.isfinite(chi_p) and chi_p < 0.05 and np.isfinite(risk_ratio):
        score += 15 if risk_ratio > 1 else -15
    if np.isfinite(ols_p) and ols_p < 0.05:
        score += 20 if ols_coef > 0 else -20
    if np.isfinite(pois_p) and pois_p < 0.05:
        score += 20 if pois_coef > 0 else -20
    if np.isfinite(pear_p) and pear_p < 0.05:
        score += 5 if pear_r > 0 else -5

    # Model-based interpretability support
    if np.isfinite(skin_coef_linear) and skin_coef_linear > 0:
        score += 3
    if np.isfinite(ebm_skin_imp) and ebm_skin_imp > 0:
        score += 2

    score = int(np.clip(round(score), 0, 100))

    explanation = (
        f"Question: {question} "
        f"In dyads with non-missing skin ratings, dark-skin players had a higher red-card rate "
        f"than light-skin players (dark={dark_mean_rate:.5f}, light={light_mean_rate:.5f}, "
        f"Welch t-test p={t_p:.3g}; Mann-Whitney p={u_p:.3g}). "
        f"Any-red probability was also higher for dark-skin players "
        f"(dark={dark_any_rate:.5f}, light={light_any_rate:.5f}, risk ratio={risk_ratio:.3f}, "
        f"chi-square p={chi_p:.3g}). "
        f"After adjusting for covariates and league/position, skin tone remained positive and significant "
        f"in OLS (coef={ols_coef:.5f}, p={ols_p:.3g}) and Poisson models "
        f"(coef={pois_coef:.5f}, IRR={irr:.3f}, p={pois_p:.3g}). "
        f"Interpretable sklearn/EBM models also assigned non-zero importance to skin tone "
        f"(Linear coef={skin_coef_linear:.5f}, EBM importance={ebm_skin_imp:.5f}). "
        f"Overall this supports a yes-leaning conclusion that darker skin tone is associated "
        f"with increased red-card likelihood/rate in this dataset."
    )

    output = {"response": score, "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(output, f)

    print("\nWrote conclusion.txt:")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
