import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, export_text
from interpret.glassbox import ExplainableBoostingRegressor, ExplainableBoostingClassifier


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def main():
    info_path = Path("info.json")
    data_path = Path("fish.csv")

    info = json.loads(info_path.read_text())
    question = info.get("research_questions", [""])[0]

    df = pd.read_csv(data_path)

    print("Research question:")
    print(question)
    print("\nData shape:", df.shape)
    print("Columns:", list(df.columns))
    print("\nMissing values by column:")
    print(df.isna().sum())

    # Exploration
    print("\nSummary statistics:")
    print(df.describe().T)

    print("\nSkewness:")
    print(df.skew(numeric_only=True))

    corr = df.corr(numeric_only=True)
    print("\nCorrelation matrix:")
    print(corr)

    df["fish_per_hour"] = np.where(df["hours"] > 0, df["fish_caught"] / df["hours"], np.nan)
    weighted_rate = df["fish_caught"].sum() / df["hours"].sum()
    mean_group_rate = df["fish_per_hour"].replace([np.inf, -np.inf], np.nan).dropna().mean()
    zero_catch_frac = (df["fish_caught"] == 0).mean()

    print("\nRate estimates:")
    print(f"Weighted catch rate (total fish / total hours): {weighted_rate:.4f} fish/hour")
    print(f"Mean group-level catch rate: {mean_group_rate:.4f} fish/hour")
    print(f"Zero-catch fraction: {zero_catch_frac:.3f}")

    # Statistical tests
    pearson_r, pearson_p = stats.pearsonr(df["hours"], df["fish_caught"])
    spearman_r, spearman_p = stats.spearmanr(df["hours"], df["fish_caught"])

    t_livebait = stats.ttest_ind(
        df.loc[df["livebait"] == 1, "fish_caught"],
        df.loc[df["livebait"] == 0, "fish_caught"],
        equal_var=False,
    )
    t_camper = stats.ttest_ind(
        df.loc[df["camper"] == 1, "fish_caught"],
        df.loc[df["camper"] == 0, "fish_caught"],
        equal_var=False,
    )

    anova_persons = stats.f_oneway(*[g["fish_caught"].values for _, g in df.groupby("persons")])
    anova_child = stats.f_oneway(*[g["fish_caught"].values for _, g in df.groupby("child")])

    print("\nStatistical tests:")
    print(f"Pearson(hours, fish_caught): r={pearson_r:.4f}, p={pearson_p:.4g}")
    print(f"Spearman(hours, fish_caught): rho={spearman_r:.4f}, p={spearman_p:.4g}")
    print(f"t-test fish_caught by livebait: t={t_livebait.statistic:.4f}, p={t_livebait.pvalue:.4g}")
    print(f"t-test fish_caught by camper: t={t_camper.statistic:.4f}, p={t_camper.pvalue:.4g}")
    print(f"ANOVA fish_caught by persons: F={anova_persons.statistic:.4f}, p={anova_persons.pvalue:.4g}")
    print(f"ANOVA fish_caught by child: F={anova_child.statistic:.4f}, p={anova_child.pvalue:.4g}")

    # OLS with p-values and CIs
    features = ["livebait", "camper", "persons", "child", "hours"]
    X = df[features]
    y = df["fish_caught"]

    X_ols = sm.add_constant(X)
    ols = sm.OLS(y, X_ols).fit()
    print("\nOLS summary:")
    print(ols.summary())

    # Train/test for model interpretability + simple predictive check
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=42),
        "Lasso": Lasso(alpha=0.01, random_state=42, max_iter=20000),
        "DecisionTreeRegressor": DecisionTreeRegressor(
            max_depth=3, min_samples_leaf=10, random_state=42
        ),
    }

    model_metrics = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        model_metrics[name] = {
            "r2": safe_float(r2_score(y_test, preds)),
            "mae": safe_float(mean_absolute_error(y_test, preds)),
        }

    print("\nscikit-learn model metrics (test set):")
    for name, metrics in model_metrics.items():
        print(name, metrics)

    lr = models["LinearRegression"]
    ridge = models["Ridge"]
    lasso = models["Lasso"]
    tree = models["DecisionTreeRegressor"]

    print("\nLinear coefficients:")
    print(dict(zip(features, lr.coef_)))
    print("Ridge coefficients:")
    print(dict(zip(features, ridge.coef_)))
    print("Lasso coefficients:")
    print(dict(zip(features, lasso.coef_)))

    print("\nDecision tree feature importances:")
    print(dict(zip(features, tree.feature_importances_)))
    print("Decision tree rules:")
    print(export_text(tree, feature_names=features))

    # InterpretML models
    ebm_reg = ExplainableBoostingRegressor(random_state=42, interactions=0)
    ebm_reg.fit(X_train, y_train)
    ebm_preds = ebm_reg.predict(X_test)
    ebm_r2 = safe_float(r2_score(y_test, ebm_preds))
    ebm_mae = safe_float(mean_absolute_error(y_test, ebm_preds))

    ebm_importances = dict(zip(ebm_reg.term_names_, ebm_reg.term_importances()))

    print("\nEBM regressor metrics:")
    print({"r2": ebm_r2, "mae": ebm_mae})
    print("EBM term importances:")
    print(ebm_importances)

    y_binary = (y > 0).astype(int)
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )

    ebm_clf = ExplainableBoostingClassifier(random_state=42, interactions=0)
    ebm_clf.fit(X_train_c, y_train_c)
    y_pred_c = ebm_clf.predict(X_test_c)
    clf_acc = safe_float(accuracy_score(y_test_c, y_pred_c))
    clf_importances = dict(zip(ebm_clf.term_names_, ebm_clf.term_importances()))

    print("\nEBM classifier (catch any fish) accuracy:")
    print(clf_acc)
    print("EBM classifier term importances:")
    print(clf_importances)

    # Build evidence-driven score
    sig_predictors = [
        col
        for col in features
        if col in ols.pvalues.index and float(ols.pvalues[col]) < 0.05
    ]

    categorical_sig_tests = sum(
        [
            t_livebait.pvalue < 0.05,
            t_camper.pvalue < 0.05,
            anova_persons.pvalue < 0.05,
            anova_child.pvalue < 0.05,
        ]
    )

    best_r2 = max([m["r2"] for m in model_metrics.values()] + [ebm_r2])

    score = 50
    score += 20 if ols.f_pvalue < 0.05 else -20
    score += min(24, 8 * len(sig_predictors))
    score += 10 if (pearson_p < 0.05 or spearman_p < 0.05) else -10
    score += 10 if categorical_sig_tests >= 2 else (5 if categorical_sig_tests == 1 else -5)
    if best_r2 >= 0.25:
        score += 10
    elif best_r2 >= 0.10:
        score += 5
    elif best_r2 < 0:
        score -= 10

    if zero_catch_frac > 0.5:
        score -= 5

    score = int(max(0, min(100, round(score))))

    sorted_ebm = sorted(ebm_importances.items(), key=lambda kv: kv[1], reverse=True)
    top_ebm = [k for k, _ in sorted_ebm[:3]]

    explanation = (
        f"Estimated average catch rate is {weighted_rate:.2f} fish/hour (total fish divided by total hours; "
        f"mean group-level rate {mean_group_rate:.2f}). Evidence of relationships is statistically significant "
        f"overall (OLS F-test p={ols.f_pvalue:.3g}). Significant predictors in multivariable OLS: "
        f"{', '.join(sig_predictors) if sig_predictors else 'none'}. "
        f"Hours is associated with catch in correlation tests (Pearson p={pearson_p:.3g}, Spearman p={spearman_p:.3g}). "
        f"Group comparisons also show differences (livebait p={t_livebait.pvalue:.3g}, camper p={t_camper.pvalue:.3g}, "
        f"persons ANOVA p={anova_persons.pvalue:.3g}, child ANOVA p={anova_child.pvalue:.3g}). "
        f"Interpretable models rank key factors as {', '.join(top_ebm)} (EBM), supporting that catch outcomes are "
        f"influenced by group composition and trip characteristics, though high zero-catch frequency makes prediction noisy."
    )

    result = {"response": score, "explanation": explanation}
    Path("conclusion.txt").write_text(json.dumps(result, ensure_ascii=True))

    print("\nWrote conclusion.txt:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
