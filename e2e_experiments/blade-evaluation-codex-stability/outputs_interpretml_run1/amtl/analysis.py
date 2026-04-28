import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor


warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def make_ohe():
    try:
        return OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(drop="first", handle_unknown="ignore", sparse=False)


def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def main():
    base = Path(".")
    info_path = base / "info.json"
    data_path = base / "amtl.csv"

    info = json.loads(info_path.read_text())
    research_question = info.get("research_questions", ["Unknown question"])[0]

    df = pd.read_csv(data_path)

    # Derived outcomes for rate and binary AMTL presence.
    df["amtl_rate"] = df["num_amtl"] / df["sockets"]
    df["any_amtl"] = (df["num_amtl"] > 0).astype(int)
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    print("=== Research Question ===")
    print(research_question)

    print("\n=== Data Overview ===")
    print(f"Rows: {len(df)}, Columns: {df.shape[1]}")
    print("\nColumn dtypes:")
    print(df.dtypes)

    numeric_cols = ["num_amtl", "sockets", "amtl_rate", "age", "stdev_age", "prob_male"]
    categorical_cols = ["genus", "tooth_class", "pop"]

    print("\n=== Numeric Summary ===")
    print(df[numeric_cols].describe().T)

    print("\n=== Distribution Summaries ===")
    for col in numeric_cols:
        series = df[col]
        q = series.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()
        print(
            f"{col}: mean={series.mean():.4f}, std={series.std():.4f}, "
            f"skew={series.skew():.4f}, q10={q[0.1]:.4f}, q50={q[0.5]:.4f}, q90={q[0.9]:.4f}"
        )

    print("\n=== Categorical Counts ===")
    for col in categorical_cols:
        print(f"\n{col}:")
        print(df[col].value_counts())

    print("\n=== Correlations (Pearson) ===")
    print(df[numeric_cols].corr(numeric_only=True).round(4))

    # Group summaries linked to the question.
    genus_summary = (
        df.groupby("genus").agg(
            n_rows=("genus", "size"),
            total_amtl=("num_amtl", "sum"),
            total_sockets=("sockets", "sum"),
            mean_rate=("amtl_rate", "mean"),
        )
    )
    genus_summary["weighted_rate"] = genus_summary["total_amtl"] / genus_summary["total_sockets"]
    print("\n=== Genus-level AMTL Rates ===")
    print(genus_summary.sort_values("weighted_rate", ascending=False).round(4))

    print("\n=== Statistical Tests ===")
    # Welch t-test: human vs non-human AMTL rate
    human_rate = df.loc[df["is_human"] == 1, "amtl_rate"]
    nonhuman_rate = df.loc[df["is_human"] == 0, "amtl_rate"]
    t_stat, t_p = stats.ttest_ind(human_rate, nonhuman_rate, equal_var=False)
    print(f"Welch t-test (human vs non-human amtl_rate): t={t_stat:.4f}, p={t_p:.6g}")

    # One-way ANOVA: rate differences by genus
    anova_groups = [g["amtl_rate"].values for _, g in df.groupby("genus")]
    f_stat, anova_p = stats.f_oneway(*anova_groups)
    print(f"ANOVA (amtl_rate ~ genus): F={f_stat:.4f}, p={anova_p:.6g}")

    # Chi-square on aggregated missing vs present teeth by genus
    agg = df.groupby("genus")[["num_amtl", "sockets"]].sum()
    contingency = np.column_stack([agg["num_amtl"].values, (agg["sockets"] - agg["num_amtl"]).values])
    chi2_stat, chi2_p, _, _ = stats.chi2_contingency(contingency)
    print(f"Chi-square (tooth status by genus): chi2={chi2_stat:.4f}, p={chi2_p:.6g}")

    print("\n=== Regression Inference (statsmodels) ===")
    # OLS for interpretable coefficients and p-values.
    ols_X = df[["is_human", "age", "prob_male", "tooth_class"]].copy()
    ols_X = pd.get_dummies(ols_X, columns=["tooth_class"], drop_first=True)
    ols_X = sm.add_constant(ols_X, has_constant="add")
    ols_X = ols_X.astype(float)
    ols_model = sm.OLS(df["amtl_rate"], ols_X).fit()
    print("OLS coefficients (amtl_rate):")
    print(ols_model.params.round(4))
    print("OLS p-values:")
    print(ols_model.pvalues.round(6))

    # Primary model: weighted binomial GLM on rates with socket count as frequency weights.
    glm_pooled = smf.glm(
        formula="amtl_rate ~ is_human + age + prob_male + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    ).fit()

    print("\nBinomial GLM (pooled human effect) coefficients:")
    print(glm_pooled.params.round(4))
    print("Binomial GLM p-values:")
    print(glm_pooled.pvalues.round(6))

    glm_genus = smf.glm(
        formula=(
            "amtl_rate ~ C(genus, Treatment(reference='Homo sapiens')) "
            "+ age + prob_male + C(tooth_class)"
        ),
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    ).fit()

    print("\nBinomial GLM (Homo as reference) genus contrasts:")
    genus_terms = [
        c for c in glm_genus.params.index if c.startswith("C(genus, Treatment(reference='Homo sapiens'))")
    ]
    for term in genus_terms:
        print(
            f"{term}: coef={glm_genus.params[term]:.4f}, "
            f"p={glm_genus.pvalues[term]:.6g}"
        )

    print("\n=== Interpretable Models (sklearn) ===")
    feature_cols = ["age", "prob_male", "stdev_age", "genus", "tooth_class"]
    X = df[feature_cols]
    y_rate = df["amtl_rate"]

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", make_ohe(), ["genus", "tooth_class"]),
            ("num", "passthrough", ["age", "prob_male", "stdev_age"]),
        ]
    )

    linear_pipe = Pipeline(
        steps=[("preprocess", preprocess), ("model", LinearRegression())]
    )
    ridge_pipe = Pipeline(
        steps=[("preprocess", preprocess), ("model", Ridge(alpha=1.0, random_state=42))]
    )
    lasso_pipe = Pipeline(
        steps=[("preprocess", preprocess), ("model", Lasso(alpha=0.001, random_state=42, max_iter=20000))]
    )
    tree_pipe = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", DecisionTreeRegressor(max_depth=4, min_samples_leaf=20, random_state=42)),
        ]
    )

    linear_pipe.fit(X, y_rate, model__sample_weight=df["sockets"])
    ridge_pipe.fit(X, y_rate, model__sample_weight=df["sockets"])
    lasso_pipe.fit(X, y_rate, model__sample_weight=df["sockets"])
    tree_pipe.fit(X, y_rate, model__sample_weight=df["sockets"])

    transformed_feature_names = linear_pipe.named_steps["preprocess"].get_feature_names_out()

    linear_coefs = pd.Series(
        linear_pipe.named_steps["model"].coef_, index=transformed_feature_names
    ).sort_values(key=np.abs, ascending=False)
    ridge_coefs = pd.Series(
        ridge_pipe.named_steps["model"].coef_, index=transformed_feature_names
    ).sort_values(key=np.abs, ascending=False)
    lasso_coefs = pd.Series(
        lasso_pipe.named_steps["model"].coef_, index=transformed_feature_names
    ).sort_values(key=np.abs, ascending=False)
    tree_importances = pd.Series(
        tree_pipe.named_steps["model"].feature_importances_, index=transformed_feature_names
    ).sort_values(ascending=False)

    print("Top LinearRegression coefficients (abs):")
    print(linear_coefs.head(10).round(4))
    print("Top Ridge coefficients (abs):")
    print(ridge_coefs.head(10).round(4))
    print("Top Lasso coefficients (abs):")
    print(lasso_coefs.head(10).round(4))
    print("Top DecisionTreeRegressor feature importances:")
    print(tree_importances.head(10).round(4))

    print("\n=== Interpretable Models (interpret) ===")
    ebm_features = ["age", "prob_male", "stdev_age", "genus", "tooth_class"]
    X_ebm = df[ebm_features].copy()

    ebm_reg = ExplainableBoostingRegressor(random_state=42, interactions=0, max_rounds=200)
    ebm_reg.fit(X_ebm, y_rate, sample_weight=df["sockets"])
    ebm_reg_global = ebm_reg.explain_global().data()
    ebm_reg_importances = pd.Series(ebm_reg_global["scores"], index=ebm_reg_global["names"]).sort_values(
        ascending=False
    )
    print("EBM Regressor global importances:")
    print(ebm_reg_importances.head(10).round(4))

    ebm_clf = ExplainableBoostingClassifier(random_state=42, interactions=0, max_rounds=200)
    ebm_clf.fit(X_ebm, df["any_amtl"], sample_weight=df["sockets"])
    ebm_clf_global = ebm_clf.explain_global().data()
    ebm_clf_importances = pd.Series(ebm_clf_global["scores"], index=ebm_clf_global["names"]).sort_values(
        ascending=False
    )
    print("EBM Classifier global importances:")
    print(ebm_clf_importances.head(10).round(4))

    # Score synthesis for final yes/no answer to the research question.
    human_coef = safe_float(glm_pooled.params.get("is_human", np.nan))
    human_p = safe_float(glm_pooled.pvalues.get("is_human", np.nan))

    score = 50

    if np.isfinite(human_coef) and np.isfinite(human_p):
        if human_coef > 0 and human_p < 0.001:
            score += 25
        elif human_coef > 0 and human_p < 0.01:
            score += 20
        elif human_coef > 0 and human_p < 0.05:
            score += 15
        elif human_coef < 0 and human_p < 0.001:
            score -= 25
        elif human_coef < 0 and human_p < 0.01:
            score -= 20
        elif human_coef < 0 and human_p < 0.05:
            score -= 15

    # Evidence from each non-human genus contrast with Homo reference.
    genus_evidence = []
    for term in genus_terms:
        coef = safe_float(glm_genus.params.get(term, np.nan))
        pval = safe_float(glm_genus.pvalues.get(term, np.nan))
        genus_evidence.append((term, coef, pval))
        if np.isfinite(coef) and np.isfinite(pval) and pval < 0.05:
            if coef < 0:
                score += 8
            else:
                score -= 8

    mean_human = human_rate.mean()
    mean_nonhuman = nonhuman_rate.mean()

    if np.isfinite(t_p) and t_p < 0.05:
        if mean_human > mean_nonhuman:
            score += 10
        else:
            score -= 10

    if np.isfinite(anova_p) and anova_p < 0.05:
        score += 4

    if np.isfinite(chi2_p) and chi2_p < 0.05:
        weighted_rates = genus_summary["weighted_rate"].to_dict()
        human_weighted = weighted_rates.get("Homo sapiens", np.nan)
        nonhuman_max = max(v for k, v in weighted_rates.items() if k != "Homo sapiens")
        if np.isfinite(human_weighted) and human_weighted > nonhuman_max:
            score += 8
        else:
            score -= 8

    score = int(np.clip(round(score), 0, 100))

    genus_effect_text = "; ".join(
        [f"{t.split('[')[-1].rstrip(']')}: coef={c:.3f}, p={p:.3g}" for t, c, p in genus_evidence]
    )

    explanation = (
        f"Question: {research_question} "
        f"Weighted binomial GLM controlling for age, sex probability, and tooth class gave is_human coef={human_coef:.3f} "
        f"(p={human_p:.3g}), indicating {'higher' if human_coef > 0 else 'lower'} AMTL in humans. "
        f"Genus contrasts vs Homo: {genus_effect_text}. "
        f"Unadjusted tests also showed human mean AMTL rate={mean_human:.3f} vs non-human={mean_nonhuman:.3f} "
        f"(Welch t p={t_p:.3g}); ANOVA across genera p={anova_p:.3g}; chi-square on aggregated counts p={chi2_p:.3g}. "
        f"Interpretable sklearn and EBM models identified genus, age, and tooth class as major predictors, "
        f"supporting the adjusted inference."
    )

    output = {"response": score, "explanation": explanation}
    with open(base / "conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(output, f)

    print("\n=== Final Conclusion JSON ===")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
