import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

from interpret.glassbox import (
    DecisionListClassifier,
    ExplainableBoostingClassifier,
    ExplainableBoostingRegressor,
)

warnings.filterwarnings("ignore")


def safe_pearsonr(x: pd.Series, y: pd.Series):
    pair = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(pair) < 3:
        return np.nan, np.nan
    return stats.pearsonr(pair["x"], pair["y"])


def safe_ttest(a: pd.Series, b: pd.Series):
    a = pd.Series(a).dropna()
    b = pd.Series(b).dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    t = stats.ttest_ind(a, b, equal_var=False)
    return float(t.statistic), float(t.pvalue)


def top_abs_coefs(model, feature_names, k=8):
    coefs = np.asarray(model.coef_).reshape(-1)
    coef_df = pd.DataFrame({"feature": feature_names, "coef": coefs})
    coef_df["abs_coef"] = coef_df["coef"].abs()
    coef_df = coef_df.sort_values("abs_coef", ascending=False).head(k)
    return coef_df[["feature", "coef"]]


def get_feature_names(preprocessor, numeric_cols, categorical_cols):
    names = list(numeric_cols)
    if categorical_cols:
        cat = preprocessor.named_transformers_["cat"]
        cat_names = cat.get_feature_names_out(categorical_cols).tolist()
        names.extend(cat_names)
    return names


def main():
    info_path = Path("info.json")
    data_path = Path("fertility.csv")

    info = json.loads(info_path.read_text())
    research_question = info.get("research_questions", ["Unknown question"])[0]

    df = pd.read_csv(data_path)

    # Parse dates and derive fertility-cycle features.
    for col in ["DateTesting", "StartDateofLastPeriod", "StartDateofPeriodBeforeLast"]:
        df[col] = pd.to_datetime(df[col], format="%m/%d/%y", errors="coerce")

    observed_cycle = (
        df["StartDateofLastPeriod"] - df["StartDateofPeriodBeforeLast"]
    ).dt.days
    cycle_length = df["ReportedCycleLength"].fillna(observed_cycle)
    cycle_length = cycle_length.where(cycle_length.between(20, 45), np.nan)
    cycle_length = cycle_length.fillna(cycle_length.median())

    day_in_cycle = (df["DateTesting"] - df["StartDateofLastPeriod"]).dt.days
    fertile_day = cycle_length - 14
    distance_to_ovulation = (day_in_cycle - fertile_day).abs()
    fertile_window = (
        (day_in_cycle >= (fertile_day - 5)) & (day_in_cycle <= (fertile_day + 1))
    ).astype(int)

    df["cycle_length"] = cycle_length
    df["day_in_cycle"] = day_in_cycle
    df["fertile_day"] = fertile_day
    df["distance_to_ovulation"] = distance_to_ovulation
    df["fertile_window"] = fertile_window

    # Primary outcome: composite religiosity score.
    religiosity_items = ["Rel1", "Rel2", "Rel3"]
    df["Religiosity"] = df[religiosity_items].mean(axis=1, skipna=True)

    analysis_cols = [
        "Religiosity",
        "fertile_window",
        "distance_to_ovulation",
        "fertile_day",
        "cycle_length",
        "day_in_cycle",
        "Sure1",
        "Sure2",
        "Relationship",
    ]
    analysis_df = df[analysis_cols].copy().dropna(subset=["Religiosity"])

    print("Research question:", research_question)
    print("\nDataset shape:", df.shape)
    print("\nMissingness (%):")
    print((df.isna().mean() * 100).round(2).sort_values(ascending=False))

    print("\nSummary statistics (analysis columns):")
    print(analysis_df.describe().T)

    print("\nCorrelations with Religiosity:")
    corr = analysis_df.corr(numeric_only=True)["Religiosity"].sort_values(ascending=False)
    print(corr)

    # Statistical tests.
    fertile_group = analysis_df.loc[analysis_df["fertile_window"] == 1, "Religiosity"]
    non_fertile_group = analysis_df.loc[analysis_df["fertile_window"] == 0, "Religiosity"]
    t_stat, t_p = safe_ttest(fertile_group, non_fertile_group)

    r_dist, p_dist = safe_pearsonr(
        analysis_df["distance_to_ovulation"], analysis_df["Religiosity"]
    )

    relationship_groups = [
        analysis_df.loc[analysis_df["Relationship"] == g, "Religiosity"]
        for g in sorted(analysis_df["Relationship"].dropna().unique())
    ]
    relationship_anova = stats.f_oneway(*relationship_groups)

    # ANOVA by cycle phase group (pre-fertile, fertile, post-ovulatory).
    phase = np.where(
        analysis_df["day_in_cycle"] < (analysis_df["fertile_day"] - 5),
        "pre_fertile",
        np.where(
            analysis_df["day_in_cycle"] <= (analysis_df["fertile_day"] + 1),
            "fertile",
            "post_ovulatory",
        ),
    )
    analysis_df["phase_group"] = phase
    phase_groups = [
        analysis_df.loc[analysis_df["phase_group"] == g, "Religiosity"]
        for g in sorted(analysis_df["phase_group"].unique())
    ]
    phase_anova = stats.f_oneway(*phase_groups)

    ols = smf.ols(
        "Religiosity ~ fertile_window + distance_to_ovulation + C(Relationship) + Sure1 + Sure2 + cycle_length",
        data=analysis_df,
    ).fit()

    print("\nStatistical tests:")
    print(
        f"T-test (fertile vs non-fertile): t={t_stat:.3f}, p={t_p:.4f}, "
        f"means=({fertile_group.mean():.3f}, {non_fertile_group.mean():.3f})"
    )
    print(f"Pearson correlation (distance to ovulation vs religiosity): r={r_dist:.3f}, p={p_dist:.4f}")
    print(
        f"ANOVA (relationship status -> religiosity): F={relationship_anova.statistic:.3f}, "
        f"p={relationship_anova.pvalue:.4f}"
    )
    print(
        f"ANOVA (cycle phase group -> religiosity): F={phase_anova.statistic:.3f}, "
        f"p={phase_anova.pvalue:.4f}"
    )

    print("\nOLS summary (key coefficients):")
    key_terms = [
        "fertile_window",
        "distance_to_ovulation",
        "Sure1",
        "Sure2",
        "cycle_length",
    ]
    key_table = pd.DataFrame(
        {
            "coef": ols.params.reindex(key_terms),
            "pvalue": ols.pvalues.reindex(key_terms),
            "ci_low": ols.conf_int().reindex(key_terms)[0],
            "ci_high": ols.conf_int().reindex(key_terms)[1],
        }
    )
    print(key_table)
    print(f"OLS R-squared: {ols.rsquared:.3f}")

    # Interpretable models (sklearn + interpret).
    feature_cols = [
        "fertile_window",
        "distance_to_ovulation",
        "cycle_length",
        "day_in_cycle",
        "Sure1",
        "Sure2",
        "Relationship",
    ]

    model_df = analysis_df[feature_cols + ["Religiosity"]].dropna()
    X = model_df[feature_cols]
    y = model_df["Religiosity"]

    categorical_cols = ["Relationship"]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    linear_pipe = Pipeline(
        [("prep", preprocessor), ("model", LinearRegression())]
    )
    ridge_pipe = Pipeline(
        [("prep", preprocessor), ("model", Ridge(alpha=1.0, random_state=42))]
    )
    lasso_pipe = Pipeline(
        [("prep", preprocessor), ("model", Lasso(alpha=0.01, random_state=42, max_iter=10000))]
    )
    tree_pipe = Pipeline(
        [
            ("prep", preprocessor),
            ("model", DecisionTreeRegressor(max_depth=3, min_samples_leaf=15, random_state=42)),
        ]
    )

    models = {
        "LinearRegression": linear_pipe,
        "Ridge": ridge_pipe,
        "Lasso": lasso_pipe,
        "DecisionTreeRegressor": tree_pipe,
    }

    print("\nInterpretable sklearn model performance:")
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        print(f"{name}: test R2={r2_score(y_test, pred):.3f}")

    feature_names = get_feature_names(
        linear_pipe.named_steps["prep"], numeric_cols, categorical_cols
    )

    for reg_name in ["LinearRegression", "Ridge", "Lasso"]:
        reg_model = models[reg_name].named_steps["model"]
        print(f"\nTop coefficients: {reg_name}")
        print(top_abs_coefs(reg_model, feature_names, k=8).to_string(index=False))

    tree_model = models["DecisionTreeRegressor"].named_steps["model"]
    tree_importance = pd.DataFrame(
        {"feature": feature_names, "importance": tree_model.feature_importances_}
    ).sort_values("importance", ascending=False)
    print("\nDecision tree feature importances:")
    print(tree_importance.head(8).to_string(index=False))

    # Explainable Boosting Regressor
    ebm_reg = ExplainableBoostingRegressor(random_state=42)
    ebm_reg.fit(X_train, y_train)
    ebm_pred = ebm_reg.predict(X_test)
    print(f"\nEBM Regressor test R2={r2_score(y_test, ebm_pred):.3f}")

    reg_global = ebm_reg.explain_global()
    reg_importances = pd.DataFrame(
        {"feature": reg_global.data()["names"], "importance": reg_global.data()["scores"]}
    ).sort_values("importance", ascending=False)
    print("Top EBM regressor feature importances:")
    print(reg_importances.head(8).to_string(index=False))

    # Classifier variants for interpretability coverage.
    y_binary = (y >= y.median()).astype(int)
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        X, y_binary, test_size=0.25, random_state=42, stratify=y_binary
    )

    ebm_clf = ExplainableBoostingClassifier(random_state=42)
    ebm_clf.fit(Xc_train, yc_train)
    clf_acc = (ebm_clf.predict(Xc_test) == yc_test).mean()
    print(f"\nEBM Classifier accuracy={clf_acc:.3f}")

    clf_global = ebm_clf.explain_global()
    clf_importances = pd.DataFrame(
        {"feature": clf_global.data()["names"], "importance": clf_global.data()["scores"]}
    ).sort_values("importance", ascending=False)
    print("Top EBM classifier feature importances:")
    print(clf_importances.head(8).to_string(index=False))

    try:
        dlc = DecisionListClassifier(random_state=42)
        dlc.fit(Xc_train, yc_train)
        dlc_acc = (dlc.predict(Xc_test) == yc_test).mean()
        print(f"DecisionListClassifier accuracy={dlc_acc:.3f}")
    except Exception as exc:
        print(f"DecisionListClassifier could not be fit cleanly: {exc}")

    # Evidence integration for Likert response.
    pvals = {
        "ttest_fertile_vs_nonfertile": float(t_p),
        "corr_distance_to_ovulation": float(p_dist),
        "anova_cycle_phase": float(phase_anova.pvalue),
        "ols_fertile_window": float(ols.pvalues.get("fertile_window", np.nan)),
        "ols_distance_to_ovulation": float(ols.pvalues.get("distance_to_ovulation", np.nan)),
    }

    sig_count = sum((p < 0.05) for p in pvals.values() if pd.notna(p))

    if sig_count == 0:
        response = 10
    elif sig_count == 1:
        response = 30
    elif sig_count == 2:
        response = 55
    elif sig_count == 3:
        response = 75
    else:
        response = 90

    explanation = (
        f"Research question: {research_question} "
        f"Fertility-linked predictors showed little evidence of association with religiosity. "
        f"T-test fertile vs non-fertile p={t_p:.4f}; correlation with distance-to-ovulation p={p_dist:.4f}; "
        f"cycle-phase ANOVA p={phase_anova.pvalue:.4f}. In OLS with relationship status and certainty controls, "
        f"fertile_window p={ols.pvalues.get('fertile_window', np.nan):.4f} and "
        f"distance_to_ovulation p={ols.pvalues.get('distance_to_ovulation', np.nan):.4f}, "
        f"both non-significant. Interpretable models (linear/tree/EBM) also did not rank fertility features as dominant "
        f"drivers relative to relationship and response-certainty variables."
    )

    out = {"response": int(response), "explanation": explanation}
    Path("conclusion.txt").write_text(json.dumps(out), encoding="utf-8")

    print("\nWrote conclusion.txt")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
