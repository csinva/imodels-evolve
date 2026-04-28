import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from interpret.glassbox import ExplainableBoostingClassifier

warnings.filterwarnings("ignore")


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def histogram_table(series: pd.Series, bins: int = 8) -> pd.DataFrame:
    clean = series.dropna().to_numpy()
    counts, edges = np.histogram(clean, bins=bins)
    labels = [f"[{edges[i]:.2f}, {edges[i + 1]:.2f})" for i in range(len(edges) - 1)]
    return pd.DataFrame({"bin": labels, "count": counts})


def fit_logit(y: pd.Series, x: pd.DataFrame):
    x_const = sm.add_constant(x, has_constant="add")
    return sm.Logit(y, x_const).fit(disp=False)


def fit_ols(y: pd.Series, x: pd.DataFrame):
    x_const = sm.add_constant(x, has_constant="add")
    return sm.OLS(y, x_const).fit()


def main() -> None:
    info_path = Path("info.json")
    data_path = Path("crofoot.csv")

    info = json.loads(info_path.read_text())
    research_question = info["research_questions"][0].strip()

    df = pd.read_csv(data_path)

    # Feature engineering aligned to the research question.
    df["size_diff"] = df["n_focal"] - df["n_other"]
    df["size_ratio"] = df["n_focal"] / df["n_other"]
    df["loc_advantage"] = df["dist_other"] - df["dist_focal"]
    df["loc_ratio"] = df["dist_focal"] / (df["dist_focal"] + df["dist_other"])
    df["male_diff"] = df["m_focal"] - df["m_other"]
    df["female_diff"] = df["f_focal"] - df["f_other"]

    y = df["win"].astype(int)

    print_section("Research Question")
    print(research_question)

    print_section("Dataset Overview")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print("Missing values by column:")
    print(df.isna().sum())
    print("\\nOutcome balance (win):")
    print(df["win"].value_counts(dropna=False).sort_index())

    print_section("Summary Statistics")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(df[numeric_cols].describe().T)

    print_section("Distributions (Selected Variables)")
    for col in ["size_diff", "size_ratio", "loc_advantage", "dist_focal", "dist_other"]:
        print(f"\\nHistogram for {col}:")
        print(histogram_table(df[col], bins=8).to_string(index=False))

    print_section("Correlations")
    corr_cols = [
        "win",
        "size_diff",
        "size_ratio",
        "loc_advantage",
        "loc_ratio",
        "n_focal",
        "n_other",
        "dist_focal",
        "dist_other",
    ]
    corr_matrix = df[corr_cols].corr(numeric_only=True)
    print("Pearson correlation matrix:")
    print(corr_matrix.to_string())

    print("\\nPoint-biserial correlations with win:")
    point_biserial_results = {}
    for col in ["size_diff", "size_ratio", "loc_advantage", "loc_ratio", "dist_focal", "dist_other"]:
        r, p = stats.pointbiserialr(y, df[col])
        point_biserial_results[col] = {"r": float(r), "p": float(p)}
        print(f"{col:>14}: r={r: .4f}, p={p: .4f}")

    print_section("Statistical Tests")
    winners = df[df["win"] == 1]
    losers = df[df["win"] == 0]

    t_size = stats.ttest_ind(winners["size_diff"], losers["size_diff"], equal_var=False)
    t_loc_adv = stats.ttest_ind(winners["loc_advantage"], losers["loc_advantage"], equal_var=False)
    print(
        f"Welch t-test size_diff by outcome: t={t_size.statistic:.4f}, p={t_size.pvalue:.4f}"
    )
    print(
        f"Welch t-test loc_advantage by outcome: t={t_loc_adv.statistic:.4f}, p={t_loc_adv.pvalue:.4f}"
    )

    size_category = pd.cut(
        df["size_diff"],
        bins=[-np.inf, -0.5, 0.5, np.inf],
        labels=["focal_smaller", "equal", "focal_larger"],
    )
    size_ct = pd.crosstab(size_category, df["win"])
    chi2, chi2_p, _, _ = stats.chi2_contingency(size_ct)
    print(f"Chi-square for size category vs win: chi2={chi2:.4f}, p={chi2_p:.4f}")

    loc_tertiles = pd.qcut(df["loc_advantage"], q=3, duplicates="drop")
    loc_groups = [group["win"].values for _, group in df.groupby(loc_tertiles, observed=False)]
    loc_groups = [g for g in loc_groups if len(g) > 1]
    if len(loc_groups) >= 2:
        anova_loc = stats.f_oneway(*loc_groups)
        print(
            f"ANOVA of win across location-advantage tertiles: F={anova_loc.statistic:.4f}, p={anova_loc.pvalue:.4f}"
        )
    else:
        anova_loc = None
        print("ANOVA of win across location-advantage tertiles: not enough groups")

    primary_x = df[["size_diff", "loc_advantage"]]
    primary_x_interact = primary_x.copy()
    primary_x_interact["size_x_location"] = (
        primary_x_interact["size_diff"] * primary_x_interact["loc_advantage"]
    )

    logit_primary = fit_logit(y, primary_x)
    logit_interaction = fit_logit(y, primary_x_interact)
    ols_primary = fit_ols(y, primary_x)
    location_only = fit_logit(y, df[["dist_focal"]])

    print("\\nLogit model (win ~ size_diff + loc_advantage):")
    print(logit_primary.summary())
    print("\\nLogit model with interaction (win ~ size_diff * loc_advantage):")
    print(logit_interaction.summary())
    print("\\nOLS linear probability model (win ~ size_diff + loc_advantage):")
    print(ols_primary.summary())
    print("\\nLocation-only logit (win ~ dist_focal):")
    print(location_only.summary())

    print_section("Interpretable Models")
    model_features = [
        "size_diff",
        "loc_advantage",
        "n_focal",
        "n_other",
        "dist_focal",
        "dist_other",
    ]
    X = df[model_features]

    lin = LinearRegression()
    lin.fit(X, y)
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=0))
    ridge.fit(X, y)
    lasso = make_pipeline(StandardScaler(), Lasso(alpha=0.05, random_state=0, max_iter=10000))
    lasso.fit(X, y)

    tree_clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=0)
    tree_clf.fit(X, y)
    tree_reg = DecisionTreeRegressor(max_depth=3, min_samples_leaf=5, random_state=0)
    tree_reg.fit(X, y)

    print("LinearRegression coefficients:")
    print(dict(zip(model_features, lin.coef_)))

    ridge_coef = ridge.named_steps["ridge"].coef_
    lasso_coef = lasso.named_steps["lasso"].coef_
    print("Ridge coefficients (standardized feature space):")
    print(dict(zip(model_features, ridge_coef)))
    print("Lasso coefficients (standardized feature space):")
    print(dict(zip(model_features, lasso_coef)))

    print("DecisionTreeClassifier feature importances:")
    print(dict(zip(model_features, tree_clf.feature_importances_)))
    print("DecisionTreeRegressor feature importances:")
    print(dict(zip(model_features, tree_reg.feature_importances_)))

    ebm = ExplainableBoostingClassifier(random_state=0, interactions=0)
    ebm.fit(X, y)
    ebm_importances = dict(zip(model_features, ebm.term_importances()))
    ebm_probs = ebm.predict_proba(X)[:, 1]
    ebm_auc = roc_auc_score(y, ebm_probs)

    print("ExplainableBoostingClassifier term importances:")
    print(ebm_importances)
    print(f"ExplainableBoostingClassifier in-sample ROC AUC: {ebm_auc:.4f}")

    print_section("Conclusion Scoring")
    size_p = float(logit_primary.pvalues.get("size_diff", np.nan))
    location_p = float(logit_primary.pvalues.get("loc_advantage", np.nan))
    joint_p = float(logit_primary.llr_pvalue)
    focal_dist_p = float(location_only.pvalues.get("dist_focal", np.nan))

    score = 50.0

    # Relative group size evidence.
    if size_p < 0.05:
        score += 20
    elif size_p < 0.10:
        score += 8
    else:
        score -= 18

    # Contest location evidence.
    if location_p < 0.05:
        score += 20
    elif location_p < 0.10:
        score += 8
    else:
        score -= 18

    # Joint model evidence.
    if joint_p < 0.05:
        score += 10
    elif joint_p < 0.10:
        score += 4
    else:
        score -= 6

    # Small credit if an alternative location-only model is suggestive.
    if focal_dist_p < 0.05:
        score += 6
    elif focal_dist_p < 0.10:
        score += 3

    score = int(np.clip(round(score), 0, 100))

    explanation = (
        "Primary tests do not support a strong effect of relative group size and contest location on "
        "winning probability in this sample. In logistic regression, size_diff (p={:.3f}) and "
        "loc_advantage (p={:.3f}) are not statistically significant, and the joint model is not "
        "significant (LLR p={:.3f}). Welch t-tests also show non-significant differences for "
        "size_diff (p={:.3f}) and loc_advantage (p={:.3f}) between wins and losses. Interpretable "
        "models (tree and EBM) indicate distance-based variables can matter, and a location-only "
        "model using dist_focal is only borderline (p={:.3f}), so evidence for location is suggestive "
        "but weak. Overall this supports a mostly 'No' conclusion rather than a clear 'Yes'."
    ).format(
        size_p,
        location_p,
        joint_p,
        float(t_size.pvalue),
        float(t_loc_adv.pvalue),
        focal_dist_p,
    )

    conclusion = {"response": score, "explanation": explanation}
    Path("conclusion.txt").write_text(json.dumps(conclusion))

    print(f"Likert response: {score}")
    print("Wrote conclusion.txt")


if __name__ == "__main__":
    main()
