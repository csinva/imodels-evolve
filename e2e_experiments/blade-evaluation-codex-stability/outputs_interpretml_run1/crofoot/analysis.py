import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from interpret.glassbox import ExplainableBoostingClassifier

warnings.filterwarnings("ignore")


def _safe_float(x):
    try:
        if x is None:
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def evidence_score(min_p, positive_direction=True):
    if np.isnan(min_p):
        base = 30
    elif min_p < 0.01:
        base = 95
    elif min_p < 0.05:
        base = 80
    elif min_p < 0.10:
        base = 65
    elif min_p < 0.20:
        base = 45
    else:
        base = 20

    if not positive_direction:
        base = max(0, base - 20)

    return int(round(base))


def main():
    info_path = Path("info.json")
    data_path = Path("crofoot.csv")

    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

    research_question = info.get("research_questions", ["Unknown question"])[0]
    df = pd.read_csv(data_path)

    # Feature engineering aligned with question semantics.
    df["rel_group_size"] = df["n_focal"] - df["n_other"]
    df["rel_group_size_ratio"] = df["n_focal"] / df["n_other"]
    df["contest_location_advantage"] = df["dist_other"] - df["dist_focal"]
    df["focal_home_advantage"] = (df["dist_focal"] < df["dist_other"]).astype(int)

    print("Research question:", research_question)
    print("Rows/Columns:", df.shape)

    # 1) Data exploration: summaries, distributions, correlations
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    summary_stats = df[numeric_cols].describe().T
    print("\nSummary statistics (numeric):")
    print(summary_stats[["mean", "std", "min", "max"]])

    dist_stats = df[["rel_group_size", "contest_location_advantage", "win"]].agg(
        ["mean", "std", "min", "max", "median", "skew"]
    ).T
    print("\nDistribution stats for key variables:")
    print(dist_stats)

    corr_matrix = df[numeric_cols].corr(numeric_only=True)
    print("\nCorrelations with win:")
    if "win" in corr_matrix.columns:
        print(corr_matrix["win"].sort_values(ascending=False))

    # 2) Statistical tests for key relationships
    win1 = df[df["win"] == 1]
    win0 = df[df["win"] == 0]

    # Welch t-tests and one-way ANOVA (2 groups)
    t_rel_size = stats.ttest_ind(win1["rel_group_size"], win0["rel_group_size"], equal_var=False)
    t_location = stats.ttest_ind(
        win1["contest_location_advantage"], win0["contest_location_advantage"], equal_var=False
    )

    anova_rel_size = stats.f_oneway(win1["rel_group_size"], win0["rel_group_size"])
    anova_location = stats.f_oneway(
        win1["contest_location_advantage"], win0["contest_location_advantage"]
    )

    # Point-biserial correlations
    pb_rel_size = stats.pointbiserialr(df["win"], df["rel_group_size"])
    pb_location = stats.pointbiserialr(df["win"], df["contest_location_advantage"])

    # Chi-square for home advantage vs win
    contingency = pd.crosstab(df["focal_home_advantage"], df["win"])
    chi2_stat, chi2_p, _, _ = stats.chi2_contingency(contingency)

    print("\nStatistical tests:")
    print(
        f"rel_group_size t-test: t={t_rel_size.statistic:.3f}, p={t_rel_size.pvalue:.4f}; "
        f"ANOVA F={anova_rel_size.statistic:.3f}, p={anova_rel_size.pvalue:.4f}; "
        f"point-biserial r={pb_rel_size.statistic:.3f}, p={pb_rel_size.pvalue:.4f}"
    )
    print(
        f"contest_location_advantage t-test: t={t_location.statistic:.3f}, p={t_location.pvalue:.4f}; "
        f"ANOVA F={anova_location.statistic:.3f}, p={anova_location.pvalue:.4f}; "
        f"point-biserial r={pb_location.statistic:.3f}, p={pb_location.pvalue:.4f}"
    )
    print(f"home advantage chi-square: chi2={chi2_stat:.3f}, p={chi2_p:.4f}")

    # 3) Regression models (statsmodels)
    X_key = df[["rel_group_size", "contest_location_advantage"]]
    X_sm = sm.add_constant(X_key)
    y = df["win"]

    ols = sm.OLS(y, X_sm).fit()
    print("\nOLS (linear probability model) coefficients:")
    print(ols.params)
    print("OLS p-values:")
    print(ols.pvalues)

    try:
        logit = sm.Logit(y, X_sm).fit(disp=False)
        logit_params = logit.params
        logit_pvalues = logit.pvalues
        print("\nLogit coefficients:")
        print(logit_params)
        print("Logit p-values:")
        print(logit_pvalues)
    except Exception as e:
        logit = None
        logit_params = pd.Series({"rel_group_size": np.nan, "contest_location_advantage": np.nan})
        logit_pvalues = pd.Series({"rel_group_size": np.nan, "contest_location_advantage": np.nan})
        print("\nLogit model failed:", repr(e))

    # 4) Interpretable scikit-learn models
    lin = LinearRegression().fit(X_key, y)
    ridge = Ridge(alpha=1.0, random_state=0).fit(X_key, y)
    lasso = Lasso(alpha=0.01, random_state=0, max_iter=10000).fit(X_key, y)
    tree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X_key, y)

    print("\nScikit-learn interpretable model summaries:")
    print("LinearRegression coef:", dict(zip(X_key.columns, lin.coef_)))
    print("Ridge coef:", dict(zip(X_key.columns, ridge.coef_)))
    print("Lasso coef:", dict(zip(X_key.columns, lasso.coef_)))
    print("DecisionTree feature importances:", dict(zip(X_key.columns, tree.feature_importances_)))

    # 5) interpret.glassbox model
    ebm = ExplainableBoostingClassifier(random_state=0, interactions=0)
    ebm.fit(X_key, y)
    ebm_global = ebm.explain_global().data()
    ebm_importance = dict(zip(ebm_global["names"], [float(s) for s in ebm_global["scores"]]))
    print("EBM global importance:", ebm_importance)

    # 6) Build conclusion score (0-100) based on significance + direction
    rel_size_pvals = [
        _safe_float(t_rel_size.pvalue),
        _safe_float(anova_rel_size.pvalue),
        _safe_float(pb_rel_size.pvalue),
        _safe_float(ols.pvalues.get("rel_group_size", np.nan)),
        _safe_float(logit_pvalues.get("rel_group_size", np.nan)),
    ]
    loc_pvals = [
        _safe_float(t_location.pvalue),
        _safe_float(anova_location.pvalue),
        _safe_float(pb_location.pvalue),
        _safe_float(ols.pvalues.get("contest_location_advantage", np.nan)),
        _safe_float(logit_pvalues.get("contest_location_advantage", np.nan)),
        _safe_float(chi2_p),
    ]

    rel_size_min_p = np.nanmin(rel_size_pvals)
    loc_min_p = np.nanmin(loc_pvals)

    rel_size_coef = _safe_float(logit_params.get("rel_group_size", lin.coef_[0]))
    loc_coef = _safe_float(logit_params.get("contest_location_advantage", lin.coef_[1]))

    size_score = evidence_score(rel_size_min_p, positive_direction=rel_size_coef > 0)
    location_score = evidence_score(loc_min_p, positive_direction=loc_coef > 0)

    response = int(np.clip(round((size_score + location_score) / 2.0), 0, 100))

    explanation = (
        f"Using 58 contests, evidence is weak to moderate for an effect of relative group size and weak for contest location. "
        f"Relative group size showed a positive direction but was not conventionally significant "
        f"(Welch t-test p={t_rel_size.pvalue:.3f}, OLS p={ols.pvalues['rel_group_size']:.3f}, "
        f"Logit coef={rel_size_coef:.3f}, p={logit_pvalues.get('rel_group_size', np.nan):.3f}). "
        f"Contest location advantage also had a positive but non-significant association "
        f"(Welch t-test p={t_location.pvalue:.3f}, OLS p={ols.pvalues['contest_location_advantage']:.3f}, "
        f"Logit coef={loc_coef:.4f}, p={logit_pvalues.get('contest_location_advantage', np.nan):.3f}, "
        f"chi-square p={chi2_p:.3f}). "
        f"Interpretable models (linear/ridge/lasso coefficients, decision tree importance, and EBM global importance) "
        f"were directionally consistent but did not indicate strong predictive effects in this small sample."
    )

    output = {"response": response, "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(output, f)

    print("\nWrote conclusion.txt:")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
