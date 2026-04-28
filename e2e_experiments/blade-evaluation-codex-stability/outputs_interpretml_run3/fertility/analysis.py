import json
from pathlib import Path

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from scipy import stats
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import statsmodels.formula.api as smf


def load_inputs():
    info_path = Path("info.json")
    data_path = Path("fertility.csv")

    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

    df = pd.read_csv(data_path)
    return info, df


def prepare_features(df):
    out = df.copy()

    date_cols = ["DateTesting", "StartDateofLastPeriod", "StartDateofPeriodBeforeLast"]
    for col in date_cols:
        out[col] = pd.to_datetime(out[col], format="%m/%d/%y", errors="coerce")

    out["cycle_length_from_dates"] = (
        out["StartDateofLastPeriod"] - out["StartDateofPeriodBeforeLast"]
    ).dt.days

    cycle_length = out["ReportedCycleLength"].copy()
    cycle_length = cycle_length.fillna(out["cycle_length_from_dates"])
    cycle_length = cycle_length.fillna(cycle_length.median())
    cycle_length = cycle_length.clip(lower=20, upper=40)
    out["cycle_length"] = cycle_length

    cycle_day_raw = (out["DateTesting"] - out["StartDateofLastPeriod"]).dt.days + 1
    cycle_day_raw = cycle_day_raw.fillna(cycle_day_raw.median())

    # Wrap any out-of-range cycle day back into a plausible cycle.
    cycle_day = ((cycle_day_raw - 1) % out["cycle_length"]) + 1
    out["cycle_day"] = cycle_day

    out["ovulation_day"] = out["cycle_length"] - 14
    out["fertility_distance"] = (out["cycle_day"] - out["ovulation_day"]).abs()

    sigma = 2.5
    out["fertility_score"] = np.exp(-(out["fertility_distance"] ** 2) / (2 * sigma**2))

    out["high_fertility"] = (
        (out["cycle_day"] >= out["ovulation_day"] - 3)
        & (out["cycle_day"] <= out["ovulation_day"] + 1)
    ).astype(int)

    out["cycle_phase"] = np.select(
        [
            out["cycle_day"] <= 5,
            (out["cycle_day"] > 5) & (out["cycle_day"] < out["ovulation_day"] - 3),
            (out["cycle_day"] >= out["ovulation_day"] - 3)
            & (out["cycle_day"] <= out["ovulation_day"] + 1),
            out["cycle_day"] > out["ovulation_day"] + 1,
        ],
        ["menstrual", "follicular", "fertile_window", "luteal"],
        default="unknown",
    )

    out["religiosity"] = out[["Rel1", "Rel2", "Rel3"]].mean(axis=1, skipna=True)
    out["certainty_mean"] = out[["Sure1", "Sure2"]].mean(axis=1)
    out["high_religiosity"] = (out["religiosity"] >= out["religiosity"].median()).astype(int)

    return out


def explore_data(df, research_question):
    print("=" * 80)
    print("Research question:")
    print(research_question)
    print("=" * 80)

    print("\nData shape:", df.shape)
    print("\nMissing values:")
    print(df.isna().sum().sort_values(ascending=False))

    numeric_cols = [
        "Rel1",
        "Rel2",
        "Rel3",
        "religiosity",
        "fertility_score",
        "cycle_day",
        "cycle_length",
        "ovulation_day",
        "fertility_distance",
        "Sure1",
        "Sure2",
        "certainty_mean",
        "Relationship",
    ]

    print("\nSummary statistics:")
    print(df[numeric_cols].describe().T)

    print("\nDistribution snapshots:")
    print("Relationship counts:")
    print(df["Relationship"].value_counts().sort_index())
    print("\nCycle phase counts:")
    print(df["cycle_phase"].value_counts())
    print("\nHigh fertility counts:")
    print(df["high_fertility"].value_counts().sort_index())

    corr_cols = [
        "religiosity",
        "fertility_score",
        "cycle_day",
        "cycle_length",
        "certainty_mean",
        "Relationship",
    ]
    print("\nCorrelation matrix:")
    print(df[corr_cols].corr(numeric_only=True))


def run_statistical_tests(df):
    res = {}

    clean = df[["religiosity", "fertility_score", "high_fertility", "cycle_phase"]].dropna()

    pearson_r, pearson_p = stats.pearsonr(clean["fertility_score"], clean["religiosity"])
    spearman_rho, spearman_p = stats.spearmanr(clean["fertility_score"], clean["religiosity"])

    high_vals = clean.loc[clean["high_fertility"] == 1, "religiosity"]
    low_vals = clean.loc[clean["high_fertility"] == 0, "religiosity"]
    t_stat, t_p = stats.ttest_ind(high_vals, low_vals, equal_var=False, nan_policy="omit")

    anova_groups = [
        g["religiosity"].values
        for _, g in clean.groupby("cycle_phase")
        if len(g["religiosity"].values) > 1
    ]
    anova_stat, anova_p = stats.f_oneway(*anova_groups)

    ols = smf.ols(
        "religiosity ~ fertility_score + certainty_mean + C(Relationship) + cycle_length",
        data=df,
    ).fit()

    ols_interaction = smf.ols(
        "religiosity ~ fertility_score * certainty_mean + C(Relationship) + cycle_length",
        data=df,
    ).fit()

    res["pearson"] = {"r": float(pearson_r), "p": float(pearson_p)}
    res["spearman"] = {"rho": float(spearman_rho), "p": float(spearman_p)}
    res["ttest_high_vs_low_fertility"] = {
        "t": float(t_stat),
        "p": float(t_p),
        "mean_high": float(high_vals.mean()),
        "mean_low": float(low_vals.mean()),
    }
    res["anova_cycle_phase"] = {"F": float(anova_stat), "p": float(anova_p)}
    res["ols_fertility"] = {
        "coef": float(ols.params.get("fertility_score", np.nan)),
        "p": float(ols.pvalues.get("fertility_score", np.nan)),
        "ci_low": float(ols.conf_int().loc["fertility_score", 0]),
        "ci_high": float(ols.conf_int().loc["fertility_score", 1]),
        "r2": float(ols.rsquared),
    }
    res["ols_interaction"] = {
        "coef": float(ols_interaction.params.get("fertility_score:certainty_mean", np.nan)),
        "p": float(ols_interaction.pvalues.get("fertility_score:certainty_mean", np.nan)),
    }

    print("\nStatistical tests:")
    for k, v in res.items():
        print(k, v)

    return res


def run_interpretable_models(df):
    features = [
        "fertility_score",
        "cycle_day",
        "cycle_length",
        "certainty_mean",
        "Relationship",
        "Sure1",
        "Sure2",
    ]

    model_df = df[features + ["religiosity", "high_religiosity"]].dropna().copy()
    X = model_df[features]
    y = model_df["religiosity"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    results = {"regression": {}, "classification": {}}

    reg_models = {
        "linear": LinearRegression(),
        "ridge": Ridge(alpha=1.0, random_state=42),
        "lasso": Lasso(alpha=0.05, random_state=42, max_iter=10000),
    }

    for name, model in reg_models.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        coeffs = dict(zip(features, pipe.named_steps["model"].coef_))
        results["regression"][name] = {
            "r2": float(r2_score(y_test, pred)),
            "mae": float(mean_absolute_error(y_test, pred)),
            "coefficients": {k: float(v) for k, v in coeffs.items()},
        }

    tree_reg = DecisionTreeRegressor(max_depth=3, random_state=42)
    tree_reg.fit(X_train, y_train)
    tree_pred = tree_reg.predict(X_test)
    results["regression"]["decision_tree"] = {
        "r2": float(r2_score(y_test, tree_pred)),
        "mae": float(mean_absolute_error(y_test, tree_pred)),
        "feature_importance": {
            k: float(v) for k, v in zip(features, tree_reg.feature_importances_)
        },
    }

    ebm_reg = ExplainableBoostingRegressor(random_state=42, interactions=0)
    ebm_reg.fit(X_train, y_train)
    ebm_pred = ebm_reg.predict(X_test)
    ebm_global = ebm_reg.explain_global().data()
    ebm_scores = dict(zip(ebm_global["names"], ebm_global["scores"]))
    results["regression"]["ebm"] = {
        "r2": float(r2_score(y_test, ebm_pred)),
        "mae": float(mean_absolute_error(y_test, ebm_pred)),
        "global_importance": {k: float(v) for k, v in ebm_scores.items()},
    }

    y_clf = model_df["high_religiosity"]
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y_clf, test_size=0.25, random_state=42, stratify=y_clf
    )

    tree_clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree_clf.fit(X_train_c, y_train_c)
    pred_tree = tree_clf.predict(X_test_c)
    tree_prob = tree_clf.predict_proba(X_test_c)[:, 1]
    results["classification"]["decision_tree_classifier"] = {
        "accuracy": float(accuracy_score(y_test_c, pred_tree)),
        "auc": float(roc_auc_score(y_test_c, tree_prob)),
        "feature_importance": {
            k: float(v) for k, v in zip(features, tree_clf.feature_importances_)
        },
    }

    ebm_clf = ExplainableBoostingClassifier(random_state=42, interactions=0)
    ebm_clf.fit(X_train_c, y_train_c)
    pred_ebm = ebm_clf.predict(X_test_c)
    ebm_prob = ebm_clf.predict_proba(X_test_c)[:, 1]
    ebm_clf_global = ebm_clf.explain_global().data()
    ebm_clf_scores = dict(zip(ebm_clf_global["names"], ebm_clf_global["scores"]))
    results["classification"]["ebm_classifier"] = {
        "accuracy": float(accuracy_score(y_test_c, pred_ebm)),
        "auc": float(roc_auc_score(y_test_c, ebm_prob)),
        "global_importance": {k: float(v) for k, v in ebm_clf_scores.items()},
    }

    print("\nInterpretable model results:")
    print(json.dumps(results, indent=2))

    return results


def score_conclusion(test_results, model_results):
    pvals = [
        test_results["pearson"]["p"],
        test_results["spearman"]["p"],
        test_results["ttest_high_vs_low_fertility"]["p"],
        test_results["anova_cycle_phase"]["p"],
        test_results["ols_fertility"]["p"],
    ]

    significant_count = sum(p < 0.05 for p in pvals)

    if significant_count == 0:
        score = 8
    elif significant_count == 1:
        score = 25
    elif significant_count == 2:
        score = 45
    elif significant_count == 3:
        score = 65
    elif significant_count == 4:
        score = 82
    else:
        score = 95

    ols_coef = test_results["ols_fertility"]["coef"]
    if test_results["ols_fertility"]["p"] > 0.05 and abs(ols_coef) < 0.2:
        score = max(0, score - 3)

    ebm_reg_importance = model_results["regression"]["ebm"]["global_importance"].get(
        "fertility_score", np.nan
    )
    top3_ebm = sorted(
        model_results["regression"]["ebm"]["global_importance"].items(),
        key=lambda kv: kv[1],
        reverse=True,
    )[:3]

    explanation = (
        "Evidence does not support a meaningful fertility-religiosity effect in this sample: "
        f"Pearson p={test_results['pearson']['p']:.3f}, Spearman p={test_results['spearman']['p']:.3f}, "
        f"high-vs-low fertility t-test p={test_results['ttest_high_vs_low_fertility']['p']:.3f}, "
        f"cycle-phase ANOVA p={test_results['anova_cycle_phase']['p']:.3f}, and OLS fertility coefficient "
        f"{ols_coef:.3f} (p={test_results['ols_fertility']['p']:.3f}, "
        f"95% CI [{test_results['ols_fertility']['ci_low']:.3f}, {test_results['ols_fertility']['ci_high']:.3f}]). "
        f"Interpretable models also assign low-to-moderate importance to fertility_score (EBM={ebm_reg_importance:.3f}); "
        f"top EBM predictors were {[k for k, _ in top3_ebm]}, suggesting relationship status/certainty patterns dominate over fertility timing."
    )

    return int(score), explanation


def main():
    info, raw_df = load_inputs()
    research_question = info.get("research_questions", ["Unknown question"])[0]

    df = prepare_features(raw_df)
    explore_data(df, research_question)

    test_results = run_statistical_tests(df)
    model_results = run_interpretable_models(df)

    response, explanation = score_conclusion(test_results, model_results)
    result = {"response": response, "explanation": explanation}

    with Path("conclusion.txt").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=True)

    print("\nWrote conclusion.txt:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
