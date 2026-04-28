import json
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from interpret.glassbox import ExplainableBoostingClassifier

warnings.filterwarnings("ignore")


def safe_qcut(series, q=3):
    """Create quantile bins with duplicate handling."""
    return pd.qcut(series, q=q, duplicates="drop")


def main():
    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    research_question = info["research_questions"][0]
    df = pd.read_csv("crofoot.csv")

    # Feature engineering aligned with question:
    # - Relative group size advantage
    # - Relative contest location advantage (positive = closer to focal group's center)
    df["size_diff"] = df["n_focal"] - df["n_other"]
    df["size_ratio"] = df["n_focal"] / df["n_other"]
    df["location_adv"] = df["dist_other"] - df["dist_focal"]
    df["male_diff"] = df["m_focal"] - df["m_other"]
    df["female_diff"] = df["f_focal"] - df["f_other"]

    print("=== RESEARCH QUESTION ===")
    print(research_question)
    print()

    print("=== DATA OVERVIEW ===")
    print(f"Shape: {df.shape}")
    print("Missing values by column:")
    print(df.isna().sum())
    print()

    print("=== SUMMARY STATISTICS ===")
    print(df.describe().T)
    print()

    print("=== CLASS BALANCE (win) ===")
    print(df["win"].value_counts(dropna=False).sort_index())
    print(f"Win rate: {df['win'].mean():.3f}")
    print()

    print("=== CORRELATIONS WITH win ===")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_with_win = df[numeric_cols].corr(numeric_only=True)["win"].sort_values(ascending=False)
    print(corr_with_win)
    print()

    # Statistical tests
    print("=== WELCH T-TESTS (win=1 vs win=0) ===")
    key_features = ["size_diff", "size_ratio", "location_adv", "dist_focal", "dist_other"]
    ttest_results = {}
    for col in key_features:
        win_vals = df.loc[df["win"] == 1, col]
        lose_vals = df.loc[df["win"] == 0, col]
        t_stat, p_val = stats.ttest_ind(win_vals, lose_vals, equal_var=False)
        ttest_results[col] = {"t_stat": float(t_stat), "p_value": float(p_val)}
        print(
            f"{col}: mean(win=1)={win_vals.mean():.3f}, mean(win=0)={lose_vals.mean():.3f}, "
            f"t={t_stat:.3f}, p={p_val:.4f}"
        )
    print()

    print("=== POINT-BISERIAL CORRELATIONS WITH win ===")
    pb_results = {}
    for col in key_features:
        r, p_val = stats.pointbiserialr(df["win"], df[col])
        pb_results[col] = {"r": float(r), "p_value": float(p_val)}
        print(f"{col}: r={r:.3f}, p={p_val:.4f}")
    print()

    print("=== ANOVA ON win ACROSS TERTILES ===")
    anova_results = {}
    for col in ["size_diff", "location_adv"]:
        bins = safe_qcut(df[col], q=3)
        groups = [g["win"].values for _, g in df.groupby(bins, observed=True)]
        if len(groups) >= 2:
            f_stat, p_val = stats.f_oneway(*groups)
            anova_results[col] = {"f_stat": float(f_stat), "p_value": float(p_val)}
            print(f"{col}: F={f_stat:.3f}, p={p_val:.4f}")
    print()

    print("=== CHI-SQUARE TESTS (tertiles vs win) ===")
    chi2_results = {}
    for col in ["size_diff", "location_adv"]:
        bins = safe_qcut(df[col], q=3)
        contingency = pd.crosstab(bins, df["win"])
        chi2, p_val, dof, _ = stats.chi2_contingency(contingency)
        chi2_results[col] = {"chi2": float(chi2), "p_value": float(p_val), "dof": int(dof)}
        print(f"{col}: chi2={chi2:.3f}, dof={dof}, p={p_val:.4f}")
    print()

    print("=== STATSMODELS LOGISTIC REGRESSION ===")
    logit_model = smf.logit("win ~ size_diff + location_adv", data=df).fit(disp=False)
    print(logit_model.summary())
    print()

    print("=== STATSMODELS OLS (LINEAR PROBABILITY) ===")
    X_ols = sm.add_constant(df[["size_diff", "location_adv"]])
    ols_model = sm.OLS(df["win"], X_ols).fit()
    print(ols_model.summary())
    print()

    print("=== INTERPRETABLE SCIKIT-LEARN MODELS ===")
    model_features = [
        "size_diff",
        "location_adv",
        "dist_focal",
        "dist_other",
        "male_diff",
        "female_diff",
    ]
    X = df[model_features]
    y = df["win"]

    lin_model = LinearRegression()
    lin_model.fit(X, y)
    lin_coefs = dict(zip(model_features, lin_model.coef_))
    print("LinearRegression coefficients:")
    for k, v in sorted(lin_coefs.items(), key=lambda kv: abs(kv[1]), reverse=True):
        print(f"  {k}: {v:.4f}")
    print(f"  intercept: {lin_model.intercept_:.4f}")
    print()

    tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree.fit(X, y)
    tree_importances = dict(zip(model_features, tree.feature_importances_))
    print("DecisionTreeClassifier feature_importances_:")
    for k, v in sorted(tree_importances.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {k}: {v:.4f}")
    print()

    print("=== INTERPRET GLASSBOX MODEL ===")
    ebm = ExplainableBoostingClassifier(random_state=42, interactions=0)
    ebm.fit(X, y)
    ebm_importances = dict(zip(ebm.term_names_, ebm.term_importances()))
    print("ExplainableBoostingClassifier term importances:")
    for k, v in sorted(ebm_importances.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {k}: {v:.4f}")
    print()

    # Evidence synthesis for final Likert score
    p_size = min(
        ttest_results["size_diff"]["p_value"],
        pb_results["size_diff"]["p_value"],
        float(logit_model.pvalues.get("size_diff", np.nan)),
    )
    p_location = min(
        ttest_results["location_adv"]["p_value"],
        pb_results["location_adv"]["p_value"],
        float(logit_model.pvalues.get("location_adv", np.nan)),
        ttest_results["dist_focal"]["p_value"],
    )

    if p_size < 0.05 and p_location < 0.05:
        score = 90
    elif p_size < 0.10 and p_location < 0.10:
        score = 65
    elif (p_size < 0.05) ^ (p_location < 0.05):
        score = 35
    elif (p_size < 0.10) ^ (p_location < 0.10):
        score = 30
    else:
        score = 15

    explanation = (
        f"Using 58 contests, evidence is mixed and overall weak for the combined claim. "
        f"Relative group size is not statistically significant (logit p={float(logit_model.pvalues.get('size_diff', np.nan)):.3f}; "
        f"Welch t-test p={ttest_results['size_diff']['p_value']:.3f}). "
        f"Location shows limited support: relative location advantage is not significant "
        f"(logit p={float(logit_model.pvalues.get('location_adv', np.nan)):.3f}; t-test p={ttest_results['location_adv']['p_value']:.3f}), "
        f"but focal distance is borderline (t-test p={ttest_results['dist_focal']['p_value']:.3f}). "
        f"Interpretable models (linear/tree/EBM) indicate location and composition features contribute, "
        f"yet inferential tests do not provide strong evidence that both relative group size and contest location robustly "
        f"increase win probability in this sample."
    )

    output = {"response": int(score), "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(output, f)

    print("=== FINAL CONCLUSION JSON ===")
    print(json.dumps(output))


if __name__ == "__main__":
    main()
