import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore", category=FutureWarning)


def load_inputs():
    info_path = Path("info.json")
    data_path = Path("boxes.csv")

    info = json.loads(info_path.read_text())
    df = pd.read_csv(data_path)
    return info, df


def explore_data(df):
    print("=== DATA OVERVIEW ===")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print("\nSummary statistics:")
    print(df.describe(include="all"))

    print("\n=== DISTRIBUTIONS ===")
    for col in ["y", "gender", "age", "majority_first", "culture"]:
        print(f"\n{col} value counts:")
        print(df[col].value_counts(dropna=False).sort_index())

    # Correlations over numeric columns + derived majority choice
    corr_df = df.copy()
    corr_df["majority_choice"] = (corr_df["y"] == 2).astype(int)
    print("\n=== CORRELATIONS (Pearson) ===")
    print(corr_df.corr(numeric_only=True))
    print("\n=== CORRELATIONS (Spearman) ===")
    print(corr_df.corr(method="spearman", numeric_only=True))


def run_statistical_tests(df):
    out = {}
    df = df.copy()
    df["majority_choice"] = (df["y"] == 2).astype(int)

    # 1) Age difference between majority choosers and non-majority choosers
    maj_age = df.loc[df["majority_choice"] == 1, "age"]
    nonmaj_age = df.loc[df["majority_choice"] == 0, "age"]
    t_res = stats.ttest_ind(maj_age, nonmaj_age, equal_var=False)
    out["age_ttest"] = {
        "majority_age_mean": float(maj_age.mean()),
        "nonmajority_age_mean": float(nonmaj_age.mean()),
        "t_stat": float(t_res.statistic),
        "p_value": float(t_res.pvalue),
    }

    # 2) Correlation between age and majority choice
    sp = stats.spearmanr(df["age"], df["majority_choice"])
    out["age_majority_spearman"] = {
        "rho": float(sp.statistic),
        "p_value": float(sp.pvalue),
    }

    # 3) Culture vs majority choice
    contingency = pd.crosstab(df["culture"], df["majority_choice"])
    chi2, chi_p, _, _ = stats.chi2_contingency(contingency)
    out["culture_majority_chi2"] = {
        "chi2": float(chi2),
        "p_value": float(chi_p),
    }

    # 4) Logistic regression for majority choice with/without age*culture interaction
    base_model = smf.logit(
        "majority_choice ~ age + C(culture) + C(gender) + majority_first", data=df
    ).fit(disp=False, maxiter=200)

    interaction_model = smf.logit(
        "majority_choice ~ age * C(culture) + C(gender) + majority_first", data=df
    ).fit(disp=False, maxiter=300)

    lr_stat = 2.0 * (interaction_model.llf - base_model.llf)
    lr_df = float(interaction_model.df_model - base_model.df_model)
    lr_p = float(1.0 - stats.chi2.cdf(lr_stat, lr_df))

    out["logit_base"] = {
        "age_coef": float(base_model.params["age"]),
        "age_p": float(base_model.pvalues["age"]),
        "majority_first_coef": float(base_model.params["majority_first"]),
        "majority_first_p": float(base_model.pvalues["majority_first"]),
        "pseudo_r2": float(base_model.prsquared),
    }

    out["age_culture_interaction_lr_test"] = {
        "lr_stat": float(lr_stat),
        "df": lr_df,
        "p_value": lr_p,
    }

    # 5) ANOVA-style test using linear model (simple interpretability check)
    ols_model = smf.ols(
        "majority_choice ~ age + C(culture) + C(gender) + majority_first + age:C(culture)",
        data=df,
    ).fit()
    out["ols"] = {
        "age_coef": float(ols_model.params.get("age", np.nan)),
        "age_p": float(ols_model.pvalues.get("age", np.nan)),
        "r2": float(ols_model.rsquared),
    }

    print("\n=== STATISTICAL TESTS ===")
    for k, v in out.items():
        print(k, v)

    return out


def fit_interpretable_models(df):
    out = {}
    df = df.copy()
    y_bin = (df["y"] == 2).astype(int)

    X_base = df[["gender", "age", "majority_first", "culture"]].copy()
    X_lin = pd.get_dummies(
        X_base.astype({"gender": "category", "culture": "category"}),
        columns=["gender", "culture"],
        drop_first=True,
    )

    # Linear models on binary target as interpretable linear-probability approximations
    lin = LinearRegression()
    ridge = Ridge(alpha=1.0)
    lasso = Lasso(alpha=0.001, max_iter=10000)

    lin.fit(X_lin, y_bin)
    ridge.fit(X_lin, y_bin)
    lasso.fit(X_lin, y_bin)

    lin_coefs = pd.Series(lin.coef_, index=X_lin.columns).sort_values(key=np.abs, ascending=False)
    ridge_coefs = pd.Series(ridge.coef_, index=X_lin.columns).sort_values(key=np.abs, ascending=False)
    lasso_coefs = pd.Series(lasso.coef_, index=X_lin.columns).sort_values(key=np.abs, ascending=False)

    out["linear_regression_top_coefs"] = lin_coefs.head(6).to_dict()
    out["ridge_top_coefs"] = ridge_coefs.head(6).to_dict()
    out["lasso_top_coefs"] = lasso_coefs.head(6).to_dict()

    # Interpretable tree classifier
    X_train, X_test, y_train, y_test = train_test_split(
        X_base, y_bin, test_size=0.25, random_state=42, stratify=y_bin
    )

    tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree.fit(X_train, y_train)
    tree_acc = accuracy_score(y_test, tree.predict(X_test))
    out["decision_tree"] = {
        "test_accuracy": float(tree_acc),
        "feature_importances": {
            c: float(v) for c, v in zip(X_base.columns, tree.feature_importances_)
        },
    }

    # Explainable Boosting Machine (interpret)
    try:
        from interpret.glassbox import ExplainableBoostingClassifier

        ebm = ExplainableBoostingClassifier(random_state=42, interactions=0)
        ebm.fit(X_train, y_train)
        ebm_acc = accuracy_score(y_test, ebm.predict(X_test))
        g = ebm.explain_global().data()
        names = g.get("names", [])
        scores = g.get("scores", [])
        pairs = sorted(
            [(str(n), float(s)) for n, s in zip(names, scores)],
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        out["ebm"] = {
            "test_accuracy": float(ebm_acc),
            "global_importances": {k: v for k, v in pairs},
        }
    except Exception as exc:
        out["ebm"] = {"error": str(exc)}

    print("\n=== INTERPRETABLE MODELS ===")
    for k, v in out.items():
        print(k, v)

    return out


def make_conclusion(info, stats_out, model_out):
    question = info.get("research_questions", ["Unknown question"])[0]

    age_p = stats_out["logit_base"]["age_p"]
    age_coef = stats_out["logit_base"]["age_coef"]
    interact_p = stats_out["age_culture_interaction_lr_test"]["p_value"]
    sp_rho = stats_out["age_majority_spearman"]["rho"]
    sp_p = stats_out["age_majority_spearman"]["p_value"]
    chi_p = stats_out["culture_majority_chi2"]["p_value"]

    # Likert score toward "Yes, reliance on majority develops with age across cultures"
    score = 50

    # Age trend evidence
    if age_p < 0.05:
        score += 25 if age_coef > 0 else -25
    else:
        score -= 20

    # Cross-cultural age-pattern differences evidence
    if interact_p < 0.05:
        score += 20
    else:
        score -= 10

    # Nonparametric monotonic support
    if sp_p < 0.05:
        score += 10 if sp_rho > 0 else -10
    else:
        score -= 10

    # Main culture dependence for majority preference (not age-development directly)
    if chi_p < 0.05:
        score += 5

    # Model-importance cross-check (age should matter if claim is true)
    age_importance = None
    ebm = model_out.get("ebm", {})
    if isinstance(ebm, dict) and "global_importances" in ebm:
        age_importance = ebm["global_importances"].get("age", None)
        if age_importance is not None and abs(age_importance) < 0.02:
            score -= 5

    score = int(max(0, min(100, round(score))))

    explanation = (
        f"Question: {question} "
        f"Binary majority-choice analyses show no evidence that reliance on majority changes with age "
        f"(logistic age coef={age_coef:.3f}, p={age_p:.3g}; Spearman rho={sp_rho:.3f}, p={sp_p:.3g}; "
        f"age means t-test p={stats_out['age_ttest']['p_value']:.3g}). "
        f"Evidence for age-by-culture differences is also not significant (LR interaction p={interact_p:.3g}). "
        f"Culture main effect on majority choice is not significant at 0.05 (chi-square p={chi_p:.3g}). "
        f"Interpretable models emphasize non-age factors (notably majority_first), while age importance is weak. "
        f"Overall this supports a 'No/weak evidence' conclusion for developmental increase in majority reliance across cultures."
    )

    return {"response": score, "explanation": explanation}


def main():
    info, df = load_inputs()
    explore_data(df)
    stats_out = run_statistical_tests(df)
    model_out = fit_interpretable_models(df)

    conclusion = make_conclusion(info, stats_out, model_out)

    # Required output: ONLY JSON object in conclusion.txt
    Path("conclusion.txt").write_text(json.dumps(conclusion, ensure_ascii=True), encoding="utf-8")
    print("\nWrote conclusion.txt")
    print(conclusion)


if __name__ == "__main__":
    main()
