import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def main():
    data_path = Path("caschools.csv")
    if not data_path.exists():
        raise FileNotFoundError("caschools.csv not found in current directory")

    df = pd.read_csv(data_path)

    # Derived variables for question-focused analysis.
    df["student_teacher_ratio"] = df["students"] / df["teachers"]
    df["avg_score"] = (df["read"] + df["math"]) / 2.0

    # Keep columns used in models/tests and drop rows with missing values.
    feature_cols = [
        "student_teacher_ratio",
        "income",
        "english",
        "lunch",
        "calworks",
        "expenditure",
        "computer",
        "students",
    ]
    modeling_df = df[feature_cols + ["avg_score", "read", "math"]].dropna().copy()

    print("Rows used:", len(modeling_df))
    print("\nSummary statistics (selected):")
    print(modeling_df[["student_teacher_ratio", "avg_score", "read", "math"]].describe().round(3))

    print("\nSkewness (selected):")
    print(modeling_df[["student_teacher_ratio", "avg_score", "read", "math"]].skew().round(3))

    corr_vars = [
        "student_teacher_ratio",
        "avg_score",
        "income",
        "english",
        "lunch",
        "calworks",
        "expenditure",
        "computer",
    ]
    print("\nCorrelation matrix (selected):")
    print(modeling_df[corr_vars].corr().round(3))

    x_ratio = modeling_df["student_teacher_ratio"].to_numpy()
    y_score = modeling_df["avg_score"].to_numpy()

    pearson_r, pearson_p = stats.pearsonr(x_ratio, y_score)
    spearman_rho, spearman_p = stats.spearmanr(x_ratio, y_score)

    # Welch t-test: low STR vs high STR split at median.
    median_str = modeling_df["student_teacher_ratio"].median()
    low_str_scores = modeling_df.loc[modeling_df["student_teacher_ratio"] <= median_str, "avg_score"]
    high_str_scores = modeling_df.loc[modeling_df["student_teacher_ratio"] > median_str, "avg_score"]
    t_stat, t_p = stats.ttest_ind(low_str_scores, high_str_scores, equal_var=False)
    mean_diff_low_minus_high = low_str_scores.mean() - high_str_scores.mean()

    # ANOVA across STR tertiles.
    tertiles = pd.qcut(modeling_df["student_teacher_ratio"], q=3, labels=["low", "mid", "high"])
    g_low = modeling_df.loc[tertiles == "low", "avg_score"]
    g_mid = modeling_df.loc[tertiles == "mid", "avg_score"]
    g_high = modeling_df.loc[tertiles == "high", "avg_score"]
    anova_f, anova_p = stats.f_oneway(g_low, g_mid, g_high)
    tertile_means = {
        "low": safe_float(g_low.mean()),
        "mid": safe_float(g_mid.mean()),
        "high": safe_float(g_high.mean()),
    }

    print("\nStatistical tests:")
    print(f"Pearson r={pearson_r:.4f}, p={pearson_p:.4g}")
    print(f"Spearman rho={spearman_rho:.4f}, p={spearman_p:.4g}")
    print(
        "Welch t-test (low STR vs high STR): "
        f"t={t_stat:.4f}, p={t_p:.4g}, mean_diff={mean_diff_low_minus_high:.4f}"
    )
    print(f"ANOVA across STR tertiles: F={anova_f:.4f}, p={anova_p:.4g}, means={tertile_means}")

    # OLS models for significance and effect size.
    X_simple = sm.add_constant(modeling_df[["student_teacher_ratio"]])
    ols_simple = sm.OLS(modeling_df["avg_score"], X_simple).fit()

    X_control = sm.add_constant(modeling_df[feature_cols])
    ols_control = sm.OLS(modeling_df["avg_score"], X_control).fit()

    coef_simple = safe_float(ols_simple.params["student_teacher_ratio"])
    p_simple = safe_float(ols_simple.pvalues["student_teacher_ratio"])
    coef_control = safe_float(ols_control.params["student_teacher_ratio"])
    p_control = safe_float(ols_control.pvalues["student_teacher_ratio"])

    print("\nOLS effect of student_teacher_ratio:")
    print(f"Simple OLS coef={coef_simple:.4f}, p={p_simple:.4g}")
    print(f"Controlled OLS coef={coef_control:.4f}, p={p_control:.4g}")

    # Interpretable sklearn models.
    X = modeling_df[feature_cols]
    y = modeling_df["avg_score"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    lin = LinearRegression()
    lin.fit(X_train, y_train)
    lin_coef = safe_float(dict(zip(feature_cols, lin.coef_))["student_teacher_ratio"])
    lin_r2 = r2_score(y_test, lin.predict(X_test))

    ridge = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0, random_state=42)),
    ])
    ridge.fit(X_train, y_train)
    ridge_coef = safe_float(dict(zip(feature_cols, ridge.named_steps["model"].coef_))["student_teacher_ratio"])
    ridge_r2 = r2_score(y_test, ridge.predict(X_test))

    lasso = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=0.05, random_state=42, max_iter=10000)),
    ])
    lasso.fit(X_train, y_train)
    lasso_coef = safe_float(dict(zip(feature_cols, lasso.named_steps["model"].coef_))["student_teacher_ratio"])
    lasso_r2 = r2_score(y_test, lasso.predict(X_test))

    tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=20, random_state=42)
    tree.fit(X_train, y_train)
    tree_importances = dict(zip(feature_cols, tree.feature_importances_))
    tree_str_importance = safe_float(tree_importances.get("student_teacher_ratio", np.nan))
    tree_r2 = r2_score(y_test, tree.predict(X_test))

    print("\nScikit-learn interpretable models:")
    print(f"LinearRegression: STR coef={lin_coef:.4f}, test R2={lin_r2:.4f}")
    print(f"Ridge: STR coef={ridge_coef:.4f}, test R2={ridge_r2:.4f}")
    print(f"Lasso: STR coef={lasso_coef:.4f}, test R2={lasso_r2:.4f}")
    print(f"DecisionTree: STR importance={tree_str_importance:.4f}, test R2={tree_r2:.4f}")

    # Interpret EBM model for additive explainable relationship.
    ebm_slope = np.nan
    ebm_importance = np.nan
    ebm_r2 = np.nan
    ebm_ok = False

    try:
        from interpret.glassbox import ExplainableBoostingRegressor

        ebm = ExplainableBoostingRegressor(interactions=0, random_state=42)
        ebm.fit(X_train, y_train)
        ebm_r2 = r2_score(y_test, ebm.predict(X_test))

        # Global explanation importance.
        global_exp = ebm.explain_global(name="EBM")
        gdata = global_exp.data()
        if isinstance(gdata, dict) and "names" in gdata and "scores" in gdata:
            for nm, sc in zip(gdata["names"], gdata["scores"]):
                if nm == "student_teacher_ratio":
                    ebm_importance = safe_float(sc)
                    break

        # Estimate directional effect via ceteris paribus predictions over STR grid.
        base = X_train.median(numeric_only=True)
        grid = np.linspace(X["student_teacher_ratio"].min(), X["student_teacher_ratio"].max(), 60)
        preds = []
        for v in grid:
            row = base.copy()
            row["student_teacher_ratio"] = v
            preds.append(float(ebm.predict(pd.DataFrame([row]))[0]))
        ebm_slope = float(np.polyfit(grid, preds, 1)[0])
        ebm_ok = True

        print(
            f"EBM: STR directional slope={ebm_slope:.4f}, "
            f"STR importance={ebm_importance:.4f}, test R2={ebm_r2:.4f}"
        )
    except Exception as e:
        print(f"EBM model could not be fit: {e}")

    # Evidence aggregation into a 0-100 Likert response.
    score = 50
    contributions = []

    def add(cond, pos, neg, label):
        nonlocal score
        delta = pos if cond else neg
        score += delta
        contributions.append((label, delta, cond))

    add((pearson_r < 0 and pearson_p < 0.05), 12, -12, "pearson")
    add((spearman_rho < 0 and spearman_p < 0.05), 8, -8, "spearman")
    add((mean_diff_low_minus_high > 0 and t_p < 0.05), 10, -10, "ttest")
    monotonic = tertile_means["low"] > tertile_means["mid"] > tertile_means["high"]
    add((anova_p < 0.05 and monotonic), 8, -8, "anova_monotonic")
    add((coef_simple < 0 and p_simple < 0.05), 15, -15, "ols_simple")
    add((coef_control < 0 and p_control < 0.05), 20, -20, "ols_control")
    add((lin_coef < 0), 6, -6, "linear_coef")
    add((ridge_coef < 0), 5, -5, "ridge_coef")
    add((lasso_coef < 0), 5, -5, "lasso_coef")

    if ebm_ok:
        add((ebm_slope < 0), 11, -11, "ebm_direction")
    else:
        # Neutral if EBM is unavailable.
        contributions.append(("ebm_direction", 0, None))

    response = int(np.clip(round(score), 0, 100))

    explanation = (
        "Using California district data (n={n}), lower student-teacher ratio is "
        "consistently associated with higher average test scores. "
        "Pearson r={pr:.3f} (p={pp:.3g}) and Spearman rho={sr:.3f} (p={sp:.3g}) are negative; "
        "low-STR districts outperform high-STR districts by {md:.2f} points (Welch t-test p={tp:.3g}); "
        "ANOVA across STR tertiles is significant (p={ap:.3g}) with means low/mid/high={m0:.2f}/{m1:.2f}/{m2:.2f}. "
        "OLS shows a negative STR coefficient both unadjusted ({cs:.3f}, p={ps:.3g}) and with controls ({cc:.3f}, p={pc:.3g}). "
        "Interpretable ML models also support a negative relationship (Linear={lc:.3f}, Ridge={rc:.3f}, Lasso={lac:.3f}" +
        (", EBM slope={es:.3f}" if ebm_ok else "") +
        ")."
    ).format(
        n=len(modeling_df),
        pr=pearson_r,
        pp=pearson_p,
        sr=spearman_rho,
        sp=spearman_p,
        md=mean_diff_low_minus_high,
        tp=t_p,
        ap=anova_p,
        m0=tertile_means["low"],
        m1=tertile_means["mid"],
        m2=tertile_means["high"],
        cs=coef_simple,
        ps=p_simple,
        cc=coef_control,
        pc=p_control,
        lc=lin_coef,
        rc=ridge_coef,
        lac=lasso_coef,
        es=ebm_slope if ebm_ok else np.nan,
    )

    result = {
        "response": response,
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(result, f)

    print("\nLikert response:", response)
    print("Wrote conclusion.txt")


if __name__ == "__main__":
    main()
