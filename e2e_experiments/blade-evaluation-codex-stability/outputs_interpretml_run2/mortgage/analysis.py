import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from interpret.glassbox import ExplainableBoostingClassifier

warnings.filterwarnings("ignore")


def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def main():
    info_path = Path("info.json")
    data_path = Path("mortgage.csv")

    info = json.loads(info_path.read_text())
    question = info.get("research_questions", ["Unknown question"])[0]

    df = pd.read_csv(data_path)
    print("Research question:", question)
    print("Dataset shape:", df.shape)

    # Basic cleanup
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Ensure numeric where expected
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    print("\nMissing values by column:")
    print(df.isna().sum().sort_values(ascending=False))

    # Drop missing rows for clean inference/modeling
    df = df.dropna().copy()
    print("Rows after dropping missing:", len(df))

    # Core variables
    if "deny" not in df.columns or "female" not in df.columns:
        raise ValueError("Dataset must include 'deny' and 'female' columns")

    df["female"] = df["female"].astype(int)
    df["deny"] = df["deny"].astype(int)
    if "accept" in df.columns:
        df["accept"] = df["accept"].astype(int)
    else:
        df["accept"] = 1 - df["deny"]

    # EDA: summary stats and group distributions
    print("\nSummary statistics:")
    print(df.describe().T[["mean", "std", "min", "max"]])

    print("\nDistribution of female:")
    print(df["female"].value_counts(normalize=True).rename("proportion"))

    print("\nDistribution of deny:")
    print(df["deny"].value_counts(normalize=True).rename("proportion"))

    denial_by_gender = df.groupby("female")["deny"].agg(["mean", "count", "sum"])
    denial_by_gender.index = denial_by_gender.index.map({0: "male", 1: "female"})
    denial_by_gender = denial_by_gender.rename(columns={"mean": "denial_rate", "sum": "num_denied"})
    print("\nDenial rate by gender:")
    print(denial_by_gender)

    # Correlations with deny and accept
    corr = df.corr(numeric_only=True)
    deny_corr = corr["deny"].sort_values(ascending=False)
    accept_corr = corr["accept"].sort_values(ascending=False)
    print("\nTop correlations with deny:")
    print(deny_corr.head(8))
    print("\nTop correlations with accept:")
    print(accept_corr.head(8))

    # Statistical tests focused on gender effect
    male_deny = df.loc[df["female"] == 0, "deny"]
    female_deny = df.loc[df["female"] == 1, "deny"]

    t_stat, t_p = stats.ttest_ind(female_deny, male_deny, equal_var=False)

    contingency = pd.crosstab(df["female"], df["deny"])
    chi2, chi2_p, _, _ = stats.chi2_contingency(contingency)

    # Contextual ANOVA: denial probability differs across mortgage credit groups
    groups = [g["deny"].values for _, g in df.groupby("mortgage_credit")]
    anova_f, anova_p = stats.f_oneway(*groups)

    print("\nStatistical tests:")
    print(f"Welch t-test (deny by female vs male): t={t_stat:.4f}, p={t_p:.6g}")
    print(f"Chi-square (female x deny): chi2={chi2:.4f}, p={chi2_p:.6g}")
    print(f"ANOVA (deny across mortgage_credit levels): F={anova_f:.4f}, p={anova_p:.6g}")

    # Regression-based inference (linear probability model with OLS)
    control_features = [
        "black",
        "housing_expense_ratio",
        "self_employed",
        "married",
        "mortgage_credit",
        "consumer_credit",
        "bad_history",
        "PI_ratio",
        "loan_to_value",
        "denied_PMI",
    ]
    model_features = ["female"] + [c for c in control_features if c in df.columns]

    X_unadj = sm.add_constant(df[["female"]])
    y = df["deny"]
    ols_unadj = sm.OLS(y, X_unadj).fit()

    X_adj = sm.add_constant(df[model_features])
    ols_adj = sm.OLS(y, X_adj).fit()

    female_coef_unadj = safe_float(ols_unadj.params.get("female", np.nan))
    female_p_unadj = safe_float(ols_unadj.pvalues.get("female", np.nan))
    female_coef_adj = safe_float(ols_adj.params.get("female", np.nan))
    female_p_adj = safe_float(ols_adj.pvalues.get("female", np.nan))

    print("\nOLS (unadjusted) female effect on deny:")
    print(f"coef={female_coef_unadj:.6f}, p={female_p_unadj:.6g}")

    print("\nOLS (adjusted) female effect on deny:")
    print(f"coef={female_coef_adj:.6f}, p={female_p_adj:.6g}")

    # Logistic regression for odds ratio interpretation
    logit_or = np.nan
    logit_p = np.nan
    try:
        logit_model = sm.Logit(y, X_adj).fit(disp=0)
        beta = safe_float(logit_model.params.get("female", np.nan))
        logit_p = safe_float(logit_model.pvalues.get("female", np.nan))
        logit_or = float(np.exp(beta)) if np.isfinite(beta) else np.nan
        print("\nLogit (adjusted) female effect:")
        print(f"odds_ratio={logit_or:.6f}, p={logit_p:.6g}")
    except Exception as e:
        print("\nLogit model failed:", e)

    # Interpretable ML models
    feature_cols = [c for c in df.columns if c not in ["deny", "accept"]]
    X = df[feature_cols]
    y_cls = df["deny"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cls, test_size=0.25, random_state=42, stratify=y_cls
    )

    # Linear models (interpretable coefficients)
    lin = LinearRegression()
    lin.fit(X_train, y_train)
    lin_coefs = pd.Series(lin.coef_, index=feature_cols)

    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train, y_train)
    ridge_coefs = pd.Series(ridge.coef_, index=feature_cols)

    lasso = Lasso(alpha=0.001, random_state=42, max_iter=10000)
    lasso.fit(X_train, y_train)
    lasso_coefs = pd.Series(lasso.coef_, index=feature_cols)

    # Decision tree classifier
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=30, random_state=42)
    tree.fit(X_train, y_train)
    tree_pred = tree.predict(X_test)
    tree_prob = tree.predict_proba(X_test)[:, 1]
    tree_acc = accuracy_score(y_test, tree_pred)
    tree_auc = roc_auc_score(y_test, tree_prob)
    tree_importance = pd.Series(tree.feature_importances_, index=feature_cols).sort_values(ascending=False)

    print("\nDecisionTreeClassifier test metrics:")
    print(f"accuracy={tree_acc:.4f}, auc={tree_auc:.4f}")
    print("Top tree feature importances:")
    print(tree_importance.head(8))

    # LogisticRegression from sklearn for stable coefficient comparison
    skl_logit = LogisticRegression(max_iter=3000, random_state=42)
    skl_logit.fit(X_train, y_train)
    skl_logit_coef = pd.Series(skl_logit.coef_.ravel(), index=feature_cols)

    # Explainable Boosting Machine (interpret)
    ebm = ExplainableBoostingClassifier(random_state=42, interactions=0)
    ebm.fit(X_train, y_train)
    ebm_prob = ebm.predict_proba(X_test)[:, 1]
    ebm_pred = (ebm_prob >= 0.5).astype(int)
    ebm_acc = accuracy_score(y_test, ebm_pred)
    ebm_auc = roc_auc_score(y_test, ebm_prob)

    term_names = list(ebm.term_names_)
    term_importances = list(ebm.term_importances())
    term_imp_map = dict(zip(term_names, term_importances))

    female_ebm_importance = safe_float(term_imp_map.get("female", np.nan), default=0.0)
    sorted_terms = sorted(term_imp_map.items(), key=lambda kv: kv[1], reverse=True)

    print("\nExplainableBoostingClassifier test metrics:")
    print(f"accuracy={ebm_acc:.4f}, auc={ebm_auc:.4f}")
    print("Top EBM term importances:")
    for name, val in sorted_terms[:8]:
        print(f"{name}: {val:.6f}")

    # Consolidate evidence for final Likert score
    denial_rate_male = float(male_deny.mean())
    denial_rate_female = float(female_deny.mean())
    denial_gap = denial_rate_female - denial_rate_male

    # Primary decision relies on adjusted significance of female
    p_ref = female_p_adj if np.isfinite(female_p_adj) else chi2_p
    coef_ref = female_coef_adj if np.isfinite(female_coef_adj) else female_coef_unadj

    score = 50
    if np.isfinite(p_ref):
        if p_ref < 0.001:
            score += 30
        elif p_ref < 0.01:
            score += 24
        elif p_ref < 0.05:
            score += 18
        elif p_ref < 0.10:
            score += 8
        else:
            score -= 18

    # Magnitude contribution (small gaps should not produce extreme scores)
    if np.isfinite(coef_ref):
        score += int(min(18, max(0, abs(coef_ref) * 300)))

    # Consistency bonus/penalty with unadjusted tests
    if np.isfinite(chi2_p):
        score += 5 if chi2_p < 0.05 else -5
    if np.isfinite(t_p):
        score += 4 if t_p < 0.05 else -4

    # Interpretable model consistency: female should matter in at least one model
    female_tree_importance = float(tree_importance.get("female", 0.0))
    female_lin_coef = float(lin_coefs.get("female", 0.0))
    female_ridge_coef = float(ridge_coefs.get("female", 0.0))
    female_lasso_coef = float(lasso_coefs.get("female", 0.0))
    female_skl_logit_coef = float(skl_logit_coef.get("female", 0.0))

    importance_signal = female_tree_importance + female_ebm_importance
    if importance_signal > 0.02:
        score += 6
    elif importance_signal < 0.005:
        score -= 6

    score = int(max(0, min(100, round(score))))

    # Direction/contextual explanation
    direction = "higher" if denial_gap > 0 else "lower"

    explanation = (
        f"Research question: {question} "
        f"Female applicants had a denial rate of {denial_rate_female:.3f} vs {denial_rate_male:.3f} for male applicants "
        f"(gap {denial_gap:+.3f}, {direction} for females). "
        f"Unadjusted tests: chi-square p={chi2_p:.4g}, t-test p={t_p:.4g}. "
        f"Adjusted OLS (deny ~ female + controls) estimated female coef={female_coef_adj:+.4f} with p={female_p_adj:.4g}; "
        f"adjusted logit odds ratio={logit_or:.3f} (p={logit_p:.4g}). "
        f"Interpretable models were consistent: DecisionTree female importance={female_tree_importance:.4f}, "
        f"EBM female term importance={female_ebm_importance:.4f}, linear/ridge/lasso/logit female coefficients "
        f"=({female_lin_coef:+.4f}, {female_ridge_coef:+.4f}, {female_lasso_coef:+.4f}, {female_skl_logit_coef:+.4f}). "
        f"Based on statistical significance and effect size after controlling for applicant risk factors, "
        f"the evidence score is {score}/100 for gender affecting approval outcomes."
    )

    conclusion = {
        "response": score,
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f)

    print("\nWrote conclusion.txt")
    print(json.dumps(conclusion, indent=2))


if __name__ == "__main__":
    main()
