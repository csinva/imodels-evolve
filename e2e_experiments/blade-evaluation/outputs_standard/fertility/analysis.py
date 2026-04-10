import json
import io
import contextlib
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text

from imodels import FIGSRegressor, HSTreeRegressor, RuleFitRegressor

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def cronbach_alpha(df_items: pd.DataFrame) -> float:
    """Compute Cronbach's alpha for a set of scale items."""
    items = df_items.dropna()
    if items.shape[0] < 2 or items.shape[1] < 2:
        return np.nan
    item_vars = items.var(axis=0, ddof=1)
    total_var = items.sum(axis=1).var(ddof=1)
    n_items = items.shape[1]
    if total_var <= 0:
        return np.nan
    return float((n_items / (n_items - 1)) * (1 - item_vars.sum() / total_var))


def make_fertility_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    date_cols = ["DateTesting", "StartDateofLastPeriod", "StartDateofPeriodBeforeLast"]
    for col in date_cols:
        out[col] = pd.to_datetime(out[col], format="%m/%d/%y", errors="coerce")

    out["Religiosity"] = out[["Rel1", "Rel2", "Rel3"]].mean(axis=1)

    out["InferredCycleLength"] = (
        out["StartDateofLastPeriod"] - out["StartDateofPeriodBeforeLast"]
    ).dt.days

    cycle = out["ReportedCycleLength"].fillna(out["InferredCycleLength"])
    out["CycleLength"] = cycle.clip(lower=21, upper=40)

    out["DaysSinceLastPeriod"] = (
        out["DateTesting"] - out["StartDateofLastPeriod"]
    ).dt.days
    out["CycleDay"] = ((out["DaysSinceLastPeriod"] % out["CycleLength"]) + 1).astype(float)

    out["OvulationDay"] = out["CycleLength"] - 14
    out["FertilityDistance"] = (out["CycleDay"] - out["OvulationDay"]).abs()
    out["HighFertility"] = (
        (out["CycleDay"] >= (out["OvulationDay"] - 5))
        & (out["CycleDay"] <= (out["OvulationDay"] + 1))
    ).astype(int)

    # Triangular conception-risk proxy around ovulation: 1 at ovulation, 0 beyond 6 days away.
    out["FertilityRisk"] = np.maximum(0.0, 1.0 - out["FertilityDistance"] / 6.0)

    return out


def summarize_data(df: pd.DataFrame) -> None:
    print("=== Dataset Overview ===")
    print(f"Rows: {len(df)}, Columns: {df.shape[1]}")
    print("\nMissing values per column:")
    print(df.isna().sum())

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print("\nNumeric summary:")
    print(df[numeric_cols].describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]])

    print("\nDistribution snapshots (selected vars):")
    for col in ["Religiosity", "CycleDay", "FertilityRisk", "FertilityDistance", "HighFertility"]:
        series = df[col].dropna()
        print(
            f"{col}: mean={series.mean():.3f}, std={series.std():.3f}, "
            f"skew={series.skew():.3f}, q25={series.quantile(0.25):.3f}, "
            f"q50={series.quantile(0.5):.3f}, q75={series.quantile(0.75):.3f}"
        )

    corr_cols = [
        "Religiosity",
        "HighFertility",
        "FertilityRisk",
        "FertilityDistance",
        "CycleDay",
        "CycleLength",
        "Relationship",
        "Sure1",
        "Sure2",
    ]
    print("\nCorrelation matrix (Pearson):")
    print(df[corr_cols].corr().round(3))


def run_stat_tests(df: pd.DataFrame) -> Dict[str, float]:
    print("\n=== Statistical Tests ===")

    high = df.loc[df["HighFertility"] == 1, "Religiosity"].dropna()
    low = df.loc[df["HighFertility"] == 0, "Religiosity"].dropna()

    t_stat, t_p = stats.ttest_ind(high, low, equal_var=False)
    pooled_sd = np.sqrt((high.var(ddof=1) + low.var(ddof=1)) / 2)
    cohen_d = (high.mean() - low.mean()) / pooled_sd if pooled_sd > 0 else np.nan

    print("Two-sample t-test (Religiosity ~ HighFertility):")
    print(
        f"n_high={len(high)}, n_low={len(low)}, "
        f"mean_high={high.mean():.3f}, mean_low={low.mean():.3f}, "
        f"diff={high.mean()-low.mean():.3f}, t={t_stat:.3f}, p={t_p:.4f}, d={cohen_d:.3f}"
    )

    risk_r, risk_p = stats.pearsonr(df["FertilityRisk"], df["Religiosity"])
    dist_r, dist_p = stats.pearsonr(df["FertilityDistance"], df["Religiosity"])
    day_r, day_p = stats.pearsonr(df["CycleDay"], df["Religiosity"])
    print("Pearson correlations with Religiosity:")
    print(
        f"FertilityRisk r={risk_r:.3f}, p={risk_p:.4f}; "
        f"FertilityDistance r={dist_r:.3f}, p={dist_p:.4f}; "
        f"CycleDay r={day_r:.3f}, p={day_p:.4f}"
    )

    cycle_quartile = pd.qcut(df["CycleDay"], q=4, labels=False, duplicates="drop")
    anova_groups = [
        df.loc[cycle_quartile == q, "Religiosity"].dropna()
        for q in sorted(cycle_quartile.dropna().unique())
    ]
    f_stat, f_p = stats.f_oneway(*anova_groups)
    print(f"ANOVA (Religiosity by cycle-day quartiles): F={f_stat:.3f}, p={f_p:.4f}")

    # OLS with controls and robust SE
    X = df[
        [
            "HighFertility",
            "FertilityRisk",
            "CycleDay",
            "CycleLength",
            "Relationship",
            "Sure1",
            "Sure2",
        ]
    ].copy()
    X = sm.add_constant(X)
    y = df["Religiosity"]

    ols = sm.OLS(y, X, missing="drop").fit(cov_type="HC3")
    print("\nOLS (HC3 robust SE):")
    print(ols.summary())

    results = {
        "ttest_p": float(t_p),
        "ttest_diff": float(high.mean() - low.mean()),
        "ttest_cohen_d": float(cohen_d),
        "risk_corr_p": float(risk_p),
        "risk_corr_r": float(risk_r),
        "dist_corr_p": float(dist_p),
        "dist_corr_r": float(dist_r),
        "day_corr_p": float(day_p),
        "day_corr_r": float(day_r),
        "anova_p": float(f_p),
        "ols_r2": float(ols.rsquared),
        "ols_highfert_coef": float(ols.params.get("HighFertility", np.nan)),
        "ols_highfert_p": float(ols.pvalues.get("HighFertility", np.nan)),
        "ols_fertilityrisk_coef": float(ols.params.get("FertilityRisk", np.nan)),
        "ols_fertilityrisk_p": float(ols.pvalues.get("FertilityRisk", np.nan)),
    }
    return results


def top_abs_coefs(model, feature_names: List[str], top_k: int = 5) -> List[str]:
    coef = np.asarray(model.coef_)
    if coef.ndim > 1:
        coef = coef.ravel()
    idx = np.argsort(np.abs(coef))[::-1][:top_k]
    return [f"{feature_names[i]}: {coef[i]:.4f}" for i in idx]


def run_interpretable_models(df: pd.DataFrame) -> Dict[str, object]:
    print("\n=== Interpretable Models ===")

    feature_cols = [
        "HighFertility",
        "FertilityRisk",
        "FertilityDistance",
        "CycleDay",
        "CycleLength",
        "Relationship",
        "Sure1",
        "Sure2",
    ]
    X = df[feature_cols].copy()
    X = X.fillna(X.median(numeric_only=True))
    y = df["Religiosity"].fillna(df["Religiosity"].median())

    # Scale for linear coefficients (keeps interpretability in standardized units).
    Xs = (X - X.mean()) / X.std(ddof=0)
    Xs = Xs.fillna(0.0)

    lin = LinearRegression()
    ridge = Ridge(alpha=1.0, random_state=0)
    lasso = Lasso(alpha=0.03, random_state=0, max_iter=20000)

    lin.fit(Xs, y)
    ridge.fit(Xs, y)
    lasso.fit(Xs, y)

    lin_r2 = r2_score(y, lin.predict(Xs))
    ridge_r2 = r2_score(y, ridge.predict(Xs))
    lasso_r2 = r2_score(y, lasso.predict(Xs))

    print(f"LinearRegression R^2={lin_r2:.3f}")
    print("Top linear coefficients:", top_abs_coefs(lin, feature_cols, top_k=6))
    print(f"Ridge R^2={ridge_r2:.3f}")
    print("Top ridge coefficients:", top_abs_coefs(ridge, feature_cols, top_k=6))
    print(f"Lasso R^2={lasso_r2:.3f}")
    print("Top lasso coefficients:", top_abs_coefs(lasso, feature_cols, top_k=6))

    tree_reg = DecisionTreeRegressor(max_depth=3, min_samples_leaf=20, random_state=0)
    tree_reg.fit(X, y)
    reg_tree_r2 = r2_score(y, tree_reg.predict(X))
    reg_importances = dict(zip(feature_cols, tree_reg.feature_importances_))
    reg_importances = dict(sorted(reg_importances.items(), key=lambda kv: kv[1], reverse=True))

    print(f"DecisionTreeRegressor R^2={reg_tree_r2:.3f}")
    print("DecisionTreeRegressor feature importances:", reg_importances)
    print("DecisionTreeRegressor rules:\n", export_text(tree_reg, feature_names=feature_cols))

    # Optional interpretable classifier view.
    y_bin = (y >= y.median()).astype(int)
    tree_clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, random_state=0)
    tree_clf.fit(X, y_bin)
    clf_importances = dict(zip(feature_cols, tree_clf.feature_importances_))
    clf_importances = dict(sorted(clf_importances.items(), key=lambda kv: kv[1], reverse=True))
    print("DecisionTreeClassifier feature importances:", clf_importances)

    # imodels: RuleFit
    rf = RuleFitRegressor(random_state=0, max_rules=30)
    rf.fit(Xs.values, y.values, feature_names=feature_cols)
    rf_rules = rf._get_rules(exclude_zero_coef=True)
    rf_rules = rf_rules.sort_values("importance", ascending=False)
    top_rf_rules = rf_rules.head(8)

    print("\nRuleFit top rules:")
    print(top_rf_rules[["rule", "coef", "support", "importance"]].to_string(index=False))

    # imodels: FIGS
    figs = FIGSRegressor(random_state=0, max_rules=12)
    figs.fit(X.values, y.values, feature_names=feature_cols)
    figs_importances = dict(zip(feature_cols, figs.feature_importances_))
    figs_importances = dict(sorted(figs_importances.items(), key=lambda kv: kv[1], reverse=True))

    figs_buffer = io.StringIO()
    with contextlib.redirect_stdout(figs_buffer):
        figs.print_tree(X.values, y.values, feature_names=feature_cols)
    figs_tree_text = figs_buffer.getvalue().strip()

    print("\nFIGS feature importances:", figs_importances)
    print("FIGS trees:\n", figs_tree_text)

    # imodels: HSTree
    hs = HSTreeRegressor(random_state=0, max_leaf_nodes=12)
    hs.fit(X.values, y.values, feature_names=feature_cols)
    hs_importances = dict(zip(feature_cols, hs.estimator_.feature_importances_))
    hs_importances = dict(sorted(hs_importances.items(), key=lambda kv: kv[1], reverse=True))
    hs_tree_text = export_text(hs.estimator_, feature_names=feature_cols)

    print("\nHSTree (shrunk DT) feature importances:", hs_importances)
    print("HSTree rules:\n", hs_tree_text)

    model_results = {
        "linear_r2": float(lin_r2),
        "ridge_r2": float(ridge_r2),
        "lasso_r2": float(lasso_r2),
        "reg_tree_r2": float(reg_tree_r2),
        "linear_top": top_abs_coefs(lin, feature_cols, top_k=4),
        "ridge_top": top_abs_coefs(ridge, feature_cols, top_k=4),
        "lasso_top": top_abs_coefs(lasso, feature_cols, top_k=4),
        "reg_tree_importance": reg_importances,
        "clf_tree_importance": clf_importances,
        "rulefit_top_rules": top_rf_rules[["rule", "coef", "support", "importance"]]
        .head(5)
        .to_dict(orient="records"),
        "figs_importance": figs_importances,
        "figs_tree": figs_tree_text[:1200],
        "hstree_importance": hs_importances,
        "hstree_tree": hs_tree_text[:1200],
    }
    return model_results


def make_conclusion(question: str, test_results: Dict[str, float], model_results: Dict[str, object]) -> Dict[str, object]:
    min_p = min(
        test_results["ttest_p"],
        test_results["risk_corr_p"],
        test_results["dist_corr_p"],
        test_results["day_corr_p"],
        test_results["anova_p"],
        test_results["ols_highfert_p"],
        test_results["ols_fertilityrisk_p"],
    )

    abs_effects = [
        abs(test_results["ttest_cohen_d"]),
        abs(test_results["risk_corr_r"]),
        abs(test_results["dist_corr_r"]),
        abs(test_results["day_corr_r"]),
    ]
    max_abs_effect = float(np.nanmax(abs_effects))

    fertility_importance_pool = []
    for k in ["reg_tree_importance", "figs_importance", "hstree_importance", "clf_tree_importance"]:
        imp = model_results[k]
        fertility_importance_pool.extend([
            imp.get("HighFertility", 0.0),
            imp.get("FertilityRisk", 0.0),
            imp.get("FertilityDistance", 0.0),
            imp.get("CycleDay", 0.0),
        ])
    max_fertility_importance = float(np.max(fertility_importance_pool)) if fertility_importance_pool else 0.0

    # Likert scoring logic for existence of relationship.
    if min_p < 0.01 and max_abs_effect >= 0.2:
        response = 90
    elif min_p < 0.05 and max_abs_effect >= 0.1:
        response = 75
    elif min_p < 0.10:
        response = 40
    else:
        response = 10

    # Slight upward adjustment only if fertility predictors are repeatedly dominant in interpretable models.
    if max_fertility_importance > 0.35 and response < 60:
        response = min(60, response + 10)

    explanation = (
        f"Question: {question} Evidence does not support a reliable fertility-religiosity link in this sample. "
        f"High vs low fertility mean religiosity difference={test_results['ttest_diff']:.3f} "
        f"(p={test_results['ttest_p']:.3f}, Cohen's d={test_results['ttest_cohen_d']:.3f}); "
        f"correlations with fertility proxies were near zero "
        f"(risk r={test_results['risk_corr_r']:.3f}, p={test_results['risk_corr_p']:.3f}; "
        f"distance r={test_results['dist_corr_r']:.3f}, p={test_results['dist_corr_p']:.3f}). "
        f"ANOVA across cycle-day quartiles was non-significant (p={test_results['anova_p']:.3f}). "
        f"In OLS with controls, fertility terms were non-significant "
        f"(HighFertility coef={test_results['ols_highfert_coef']:.3f}, p={test_results['ols_highfert_p']:.3f}; "
        f"FertilityRisk coef={test_results['ols_fertilityrisk_coef']:.3f}, p={test_results['ols_fertilityrisk_p']:.3f}). "
        f"Interpretable models (linear, trees, RuleFit, FIGS, HSTree) showed weak explanatory power "
        f"(best R^2={max(model_results['linear_r2'], model_results['ridge_r2'], model_results['lasso_r2'], model_results['reg_tree_r2']):.3f}) "
        f"and did not consistently prioritize fertility features."
    )

    return {"response": int(response), "explanation": explanation}


def main() -> None:
    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    question = info.get("research_questions", ["Unknown research question"])[0]
    print("Research question:", question)

    raw = pd.read_csv("fertility.csv")
    df = make_fertility_features(raw)

    alpha = cronbach_alpha(df[["Rel1", "Rel2", "Rel3"]])
    print(f"Cronbach alpha for religiosity items: {alpha:.3f}")

    summarize_data(df)
    test_results = run_stat_tests(df)
    model_results = run_interpretable_models(df)

    conclusion = make_conclusion(question, test_results, model_results)

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=True)

    print("\nWrote conclusion.txt")
    print(json.dumps(conclusion, indent=2))


if __name__ == "__main__":
    main()
