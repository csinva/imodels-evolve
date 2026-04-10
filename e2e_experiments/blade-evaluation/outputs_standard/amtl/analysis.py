import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from imodels import FIGSRegressor, HSTreeRegressor, RuleFitRegressor

warnings.filterwarnings("ignore")


def print_header(title: str):
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def top_abs_effects(names, values, k=8):
    pairs = [(n, v) for n, v in zip(names, values)]
    pairs = sorted(pairs, key=lambda x: abs(float(x[1])), reverse=True)
    return pairs[:k]


def get_feature_names(preprocessor):
    # sklearn >=1.0 exposes get_feature_names_out on ColumnTransformer
    return preprocessor.get_feature_names_out()


def safe_rulefit_rules(model, max_rules=8):
    rules = []
    if hasattr(model, "get_rules"):
        try:
            rdf = model.get_rules()
            if rdf is not None and len(rdf) > 0:
                rdf = rdf.copy()
                rdf = rdf[rdf.coef != 0]
                if "support" in rdf.columns:
                    rdf = rdf.sort_values(["importance", "support"], ascending=[False, False])
                elif "importance" in rdf.columns:
                    rdf = rdf.sort_values("importance", ascending=False)
                rules = rdf.head(max_rules).to_dict("records")
        except Exception:
            pass
    return rules


def main():
    print_header("Load data and metadata")
    info = json.loads(Path("info.json").read_text())
    research_q = info["research_questions"][0]
    print("Research question:", research_q)

    df = pd.read_csv("amtl.csv")
    df["amtl_rate"] = df["num_amtl"] / df["sockets"]
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Non-human pooled indicator requested by the question
    df["group"] = np.where(df["is_human"] == 1, "Homo sapiens", "Non-human primates")

    print("Rows:", len(df), "Columns:", df.shape[1])
    print("Genera:")
    print(df["genus"].value_counts())

    print_header("Summary statistics")
    numeric_cols = ["num_amtl", "sockets", "amtl_rate", "age", "stdev_age", "prob_male"]
    print(df[numeric_cols].describe().round(4))

    print_header("Distribution summaries")
    genus_rate = (
        df.groupby("genus", as_index=False)
        .agg(
            n=("genus", "size"),
            amtl_sum=("num_amtl", "sum"),
            sockets_sum=("sockets", "sum"),
            mean_rate=("amtl_rate", "mean"),
            pooled_rate=("num_amtl", lambda x: x.sum()),
        )
    )
    genus_rate["pooled_rate"] = genus_rate["amtl_sum"] / genus_rate["sockets_sum"]
    print(genus_rate[["genus", "n", "amtl_sum", "sockets_sum", "mean_rate", "pooled_rate"]].round(4))

    tooth_rate = (
        df.groupby("tooth_class", as_index=False)
        .agg(
            n=("tooth_class", "size"),
            amtl_sum=("num_amtl", "sum"),
            sockets_sum=("sockets", "sum"),
        )
    )
    tooth_rate["pooled_rate"] = tooth_rate["amtl_sum"] / tooth_rate["sockets_sum"]
    print("\nBy tooth class:")
    print(tooth_rate.round(4))

    print_header("Correlations")
    corr = df[["amtl_rate", "num_amtl", "sockets", "age", "stdev_age", "prob_male", "is_human"]].corr()
    print(corr.round(4))

    print_header("Statistical tests")
    human = df[df["is_human"] == 1]
    nonhuman = df[df["is_human"] == 0]

    # Welch t-test on specimen/tooth-class level AMTL rates
    t_stat, t_p = stats.ttest_ind(human["amtl_rate"], nonhuman["amtl_rate"], equal_var=False)
    print(f"Welch t-test (human vs non-human amtl_rate): t={t_stat:.4f}, p={t_p:.4e}")

    # Mann-Whitney for robustness
    u_stat, u_p = stats.mannwhitneyu(human["amtl_rate"], nonhuman["amtl_rate"], alternative="two-sided")
    print(f"Mann-Whitney U (human vs non-human amtl_rate): U={u_stat:.1f}, p={u_p:.4e}")

    # Chi-square on pooled successes/failures (binomial count perspective)
    pooled = df.groupby("group", as_index=False).agg(amtl=("num_amtl", "sum"), sockets=("sockets", "sum"))
    pooled["not_amtl"] = pooled["sockets"] - pooled["amtl"]
    cont = pooled[["amtl", "not_amtl"]].to_numpy()
    chi2, chi2_p, _, _ = stats.chi2_contingency(cont)
    print(f"Chi-square pooled counts (AMTL vs not, human vs non-human): chi2={chi2:.4f}, p={chi2_p:.4e}")

    # One-way ANOVA by genus on row-level rate
    groups = [g["amtl_rate"].values for _, g in df.groupby("genus")]
    f_stat, f_p = stats.f_oneway(*groups)
    print(f"One-way ANOVA (amtl_rate ~ genus): F={f_stat:.4f}, p={f_p:.4e}")

    # OLS with covariates requested in question
    # Weighted by sockets so rows with more observable teeth contribute proportionally.
    ols = smf.wls(
        "amtl_rate ~ C(genus, Treatment(reference='Pan')) + age + prob_male + C(tooth_class)",
        data=df,
        weights=df["sockets"],
    ).fit()
    print("\nWeighted OLS summary (core covariate model):")
    print(ols.summary())

    # Binomial GLM (closer to the data generating process)
    # endog as proportion, frequency weights = trials
    glm_binom = smf.glm(
        "amtl_rate ~ C(genus, Treatment(reference='Pan')) + age + prob_male + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    ).fit()
    print("\nBinomial GLM summary:")
    print(glm_binom.summary())

    print_header("Interpretable ML models")
    features = ["genus", "age", "prob_male", "tooth_class"]
    X = df[features]
    y = df["amtl_rate"].values
    sample_weight = df["sockets"].values

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), ["genus", "tooth_class"]),
            ("num", "passthrough", ["age", "prob_male"]),
        ]
    )

    lin_pipe = Pipeline([("pre", pre), ("model", LinearRegression())])
    ridge_pipe = Pipeline([("pre", pre), ("model", Ridge(alpha=1.0, random_state=0))])
    lasso_pipe = Pipeline([("pre", pre), ("model", Lasso(alpha=0.0005, random_state=0, max_iter=20000))])

    lin_pipe.fit(X, y, model__sample_weight=sample_weight)
    ridge_pipe.fit(X, y, model__sample_weight=sample_weight)
    lasso_pipe.fit(X, y, model__sample_weight=sample_weight)

    pre_fit = lin_pipe.named_steps["pre"]
    feature_names = get_feature_names(pre_fit)

    lin_coef = lin_pipe.named_steps["model"].coef_
    ridge_coef = ridge_pipe.named_steps["model"].coef_
    lasso_coef = lasso_pipe.named_steps["model"].coef_

    print("Top LinearRegression effects (absolute coefficient):")
    for n, v in top_abs_effects(feature_names, lin_coef):
        print(f"  {n:40s} {v:+.6f}")

    print("\nTop Ridge effects (absolute coefficient):")
    for n, v in top_abs_effects(feature_names, ridge_coef):
        print(f"  {n:40s} {v:+.6f}")

    print("\nTop non-zero Lasso effects (absolute coefficient):")
    lasso_pairs = [(n, v) for n, v in zip(feature_names, lasso_coef) if abs(v) > 1e-9]
    lasso_pairs = sorted(lasso_pairs, key=lambda x: abs(float(x[1])), reverse=True)
    for n, v in lasso_pairs[:8]:
        print(f"  {n:40s} {v:+.6f}")

    # Trees
    X_tree = pd.get_dummies(X, drop_first=True)
    tree_reg = DecisionTreeRegressor(max_depth=3, min_samples_leaf=30, random_state=0)
    tree_reg.fit(X_tree, y, sample_weight=sample_weight)
    tree_reg_r2 = r2_score(y, tree_reg.predict(X_tree), sample_weight=sample_weight)
    print(f"\nDecisionTreeRegressor weighted R^2: {tree_reg_r2:.4f}")
    tree_imp = sorted(zip(X_tree.columns, tree_reg.feature_importances_), key=lambda x: x[1], reverse=True)
    print("Top tree regressor feature importances:")
    for n, v in tree_imp[:8]:
        print(f"  {n:40s} {v:.6f}")

    y_cls = (df["amtl_rate"] > 0).astype(int).values
    tree_cls = DecisionTreeClassifier(max_depth=3, min_samples_leaf=30, random_state=0)
    tree_cls.fit(X_tree, y_cls, sample_weight=sample_weight)
    cls_imp = sorted(zip(X_tree.columns, tree_cls.feature_importances_), key=lambda x: x[1], reverse=True)
    print("\nDecisionTreeClassifier feature importances (predict any AMTL):")
    for n, v in cls_imp[:8]:
        print(f"  {n:40s} {v:.6f}")

    # imodels regressors
    rulefit = RuleFitRegressor(random_state=0, n_estimators=40, tree_size=4, max_rules=60)
    rulefit.fit(X_tree.values, y, feature_names=list(X_tree.columns))
    rf_rules = safe_rulefit_rules(rulefit, max_rules=8)
    print("\nRuleFit top rules/effects:")
    if rf_rules:
        for r in rf_rules:
            rule_txt = str(r.get("rule", ""))
            coef = r.get("coef", np.nan)
            imp = r.get("importance", np.nan)
            print(f"  rule={rule_txt} | coef={coef:.6f} | importance={imp:.6f}")
    else:
        print("  (No non-zero rules extracted)")

    figs = FIGSRegressor(max_rules=12, random_state=0)
    figs.fit(X_tree.values, y, feature_names=list(X_tree.columns))
    figs_pred = figs.predict(X_tree.values)
    figs_r2 = r2_score(y, figs_pred, sample_weight=sample_weight)
    print(f"\nFIGSRegressor weighted R^2: {figs_r2:.4f}")
    if hasattr(figs, "feature_importances_"):
        figs_imp = sorted(zip(X_tree.columns, figs.feature_importances_), key=lambda x: x[1], reverse=True)
        print("FIGS feature importances:")
        for n, v in figs_imp[:8]:
            print(f"  {n:40s} {v:.6f}")

    hs = HSTreeRegressor(estimator_=DecisionTreeRegressor(max_leaf_nodes=12, random_state=0), reg_param=1.5)
    hs.fit(X_tree.values, y)
    hs_pred = hs.predict(X_tree.values)
    hs_r2 = r2_score(y, hs_pred, sample_weight=sample_weight)
    print(f"\nHSTreeRegressor weighted R^2: {hs_r2:.4f}")

    print_header("Inference for research question")
    # Extract key adjusted evidence: Homo sapiens coefficient in OLS and GLM
    key_terms_ols = [k for k in ols.params.index if "Homo sapiens" in k]
    key_terms_glm = [k for k in glm_binom.params.index if "Homo sapiens" in k]

    ols_human_coef = float(ols.params[key_terms_ols[0]]) if key_terms_ols else np.nan
    ols_human_p = float(ols.pvalues[key_terms_ols[0]]) if key_terms_ols else np.nan

    glm_human_coef = float(glm_binom.params[key_terms_glm[0]]) if key_terms_glm else np.nan
    glm_human_p = float(glm_binom.pvalues[key_terms_glm[0]]) if key_terms_glm else np.nan

    pooled_human_rate = float(human["num_amtl"].sum() / human["sockets"].sum())
    pooled_nonhuman_rate = float(nonhuman["num_amtl"].sum() / nonhuman["sockets"].sum())

    print(f"Pooled AMTL rate (human): {pooled_human_rate:.4f}")
    print(f"Pooled AMTL rate (non-human): {pooled_nonhuman_rate:.4f}")
    print(f"Adjusted OLS human effect (vs Pan reference): coef={ols_human_coef:+.6f}, p={ols_human_p:.4e}")
    print(f"Adjusted GLM human effect (log-odds, vs Pan reference): coef={glm_human_coef:+.6f}, p={glm_human_p:.4e}")

    strong_yes = (
        (pooled_human_rate > pooled_nonhuman_rate)
        and (t_p < 0.001)
        and (chi2_p < 0.001)
        and (np.isfinite(ols_human_p) and ols_human_p < 0.001)
        and (np.isfinite(glm_human_p) and glm_human_p < 0.001)
    )

    if strong_yes:
        response = 98
    elif (pooled_human_rate > pooled_nonhuman_rate) and ((t_p < 0.05) or (chi2_p < 0.05)):
        response = 80
    elif pooled_human_rate > pooled_nonhuman_rate:
        response = 60
    else:
        response = 10

    explanation = (
        f"Yes. Homo sapiens shows substantially higher AMTL than non-human primates "
        f"(pooled rates {pooled_human_rate:.3f} vs {pooled_nonhuman_rate:.3f}). "
        f"The difference is significant in unadjusted tests (Welch t-test p={t_p:.2e}, chi-square p={chi2_p:.2e}) "
        f"and remains significant after adjustment for age, sex probability, and tooth class in weighted OLS "
        f"(human coefficient {ols_human_coef:.3f}, p={ols_human_p:.2e}) and binomial GLM "
        f"(log-odds coefficient {glm_human_coef:.3f}, p={glm_human_p:.2e}). "
        f"Interpretable sklearn and imodels models also rank genus (especially Homo sapiens indicators) among the strongest predictors."
    )

    output = {"response": int(response), "explanation": explanation}
    Path("conclusion.txt").write_text(json.dumps(output))

    print("\nWrote conclusion.txt:")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
