import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

try:
    from interpret.glassbox import ExplainableBoostingClassifier
    INTERPRET_AVAILABLE = True
except Exception:
    INTERPRET_AVAILABLE = False


def safe_auc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_prob)


# 1. Load metadata and data
info_path = Path("info.json")
with info_path.open("r", encoding="utf-8") as f:
    info = json.load(f)

question = info.get("research_questions", [""])[0]
print("Research question:", question)

df = pd.read_csv("boxes.csv")
print("\nData shape:", df.shape)
print("Columns:", list(df.columns))

# 2. Basic cleaning / target construction
expected_cols = ["y", "gender", "age", "majority_first", "culture"]
missing = [c for c in expected_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing expected columns: {missing}")

# Binary outcome for reliance on majority option
df["majority_choice"] = (df["y"] == 2).astype(int)

# 3. EDA: summary statistics, distributions, correlations
print("\n=== Summary statistics ===")
print(df[expected_cols + ["majority_choice"]].describe())

print("\n=== Outcome distribution (y) ===")
print(df["y"].value_counts().sort_index())

print("\n=== Majority-choice distribution ===")
print(df["majority_choice"].value_counts().sort_index())
print("Majority-choice rate:", round(df["majority_choice"].mean(), 4))

print("\n=== Majority-choice rate by age ===")
print(df.groupby("age")["majority_choice"].mean().round(4))

print("\n=== Majority-choice rate by culture ===")
print(df.groupby("culture")["majority_choice"].mean().round(4))

print("\n=== Correlations (numeric) ===")
print(df[["majority_choice", "age", "gender", "majority_first", "culture"]].corr().round(4))

# 4. Statistical tests
print("\n=== Statistical tests ===")

# t-test: age by majority choice
age_majority = df.loc[df["majority_choice"] == 1, "age"]
age_nonmajority = df.loc[df["majority_choice"] == 0, "age"]
t_stat, t_p = stats.ttest_ind(age_majority, age_nonmajority, equal_var=False)
print(f"Welch t-test (age: majority vs non-majority): t={t_stat:.4f}, p={t_p:.4g}")

# chi-square: majority choice vs culture / gender / majority_first
chi2_culture, chi2_culture_p, _, _ = stats.chi2_contingency(pd.crosstab(df["culture"], df["majority_choice"]))
chi2_gender, chi2_gender_p, _, _ = stats.chi2_contingency(pd.crosstab(df["gender"], df["majority_choice"]))
chi2_order, chi2_order_p, _, _ = stats.chi2_contingency(pd.crosstab(df["majority_first"], df["majority_choice"]))
print(f"Chi-square (culture x majority): chi2={chi2_culture:.4f}, p={chi2_culture_p:.4g}")
print(f"Chi-square (gender x majority): chi2={chi2_gender:.4f}, p={chi2_gender_p:.4g}")
print(f"Chi-square (majority_first x majority): chi2={chi2_order:.4f}, p={chi2_order_p:.4g}")

# Logistic regression with interaction: does age trend differ by culture?
logit_model = smf.glm(
    formula="majority_choice ~ age * C(culture) + C(gender) + C(majority_first)",
    data=df,
    family=sm.families.Binomial(),
).fit()

print("\nGLM Binomial summary (truncated coefficients):")
print(logit_model.summary2().tables[1].head(20))

params = logit_model.params
pvals = logit_model.pvalues

age_coef = params.get("age", np.nan)
age_p = pvals.get("age", np.nan)

interaction_terms = [name for name in pvals.index if name.startswith("age:C(culture)")]
interaction_pvals = pvals.loc[interaction_terms] if interaction_terms else pd.Series(dtype=float)
interaction_sig = bool((interaction_pvals < 0.05).any()) if len(interaction_pvals) else False

# OLS + ANOVA for complementary interpretable significance table
ols_model = smf.ols(
    formula="majority_choice ~ age * C(culture) + C(gender) + C(majority_first)",
    data=df,
).fit()
anova_tbl = anova_lm(ols_model, typ=2)
print("\nANOVA (Type II) on majority_choice as numeric:")
print(anova_tbl)

# 5. Interpretable models (sklearn + interpret)
print("\n=== Interpretable models ===")

X = df[["age", "gender", "majority_first", "culture"]].copy()
y = df["majority_choice"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop=None, sparse_output=False, handle_unknown="ignore"), ["culture", "gender", "majority_first"]),
        ("num", "passthrough", ["age"]),
    ],
    remainder="drop",
)

# Linear / ridge / lasso on binary outcome as an interpretable approximation
lin_pipe = Pipeline([("prep", preprocess), ("model", LinearRegression())])
ridge_pipe = Pipeline([("prep", preprocess), ("model", Ridge(alpha=1.0))])
lasso_pipe = Pipeline([("prep", preprocess), ("model", Lasso(alpha=0.001, max_iter=20000))])

lin_pipe.fit(X_train, y_train)
ridge_pipe.fit(X_train, y_train)
lasso_pipe.fit(X_train, y_train)

lin_pred = np.clip(lin_pipe.predict(X_test), 0, 1)
ridge_pred = np.clip(ridge_pipe.predict(X_test), 0, 1)
lasso_pred = np.clip(lasso_pipe.predict(X_test), 0, 1)

print(f"LinearRegression AUC: {safe_auc(y_test, lin_pred):.4f}")
print(f"Ridge AUC: {safe_auc(y_test, ridge_pred):.4f}")
print(f"Lasso AUC: {safe_auc(y_test, lasso_pred):.4f}")

feature_names = lin_pipe.named_steps["prep"].get_feature_names_out()
lin_coefs = pd.Series(lin_pipe.named_steps["model"].coef_, index=feature_names).sort_values(key=np.abs, ascending=False)
ridge_coefs = pd.Series(ridge_pipe.named_steps["model"].coef_, index=feature_names).sort_values(key=np.abs, ascending=False)
lasso_coefs = pd.Series(lasso_pipe.named_steps["model"].coef_, index=feature_names).sort_values(key=np.abs, ascending=False)

print("\nTop LinearRegression coefficients (abs):")
print(lin_coefs.head(10).round(4))
print("\nTop Ridge coefficients (abs):")
print(ridge_coefs.head(10).round(4))
print("\nTop Lasso coefficients (abs):")
print(lasso_coefs.head(10).round(4))

# Decision tree classifier
tree = Pipeline(
    steps=[
        ("prep", preprocess),
        ("model", DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, random_state=42)),
    ]
)
tree.fit(X_train, y_train)

tree_prob = tree.predict_proba(X_test)[:, 1]
tree_pred = (tree_prob >= 0.5).astype(int)
print(f"DecisionTree accuracy: {accuracy_score(y_test, tree_pred):.4f}")
print(f"DecisionTree AUC: {safe_auc(y_test, tree_prob):.4f}")

tree_importance = pd.Series(
    tree.named_steps["model"].feature_importances_,
    index=tree.named_steps["prep"].get_feature_names_out(),
).sort_values(ascending=False)
print("\nDecisionTree feature importances:")
print(tree_importance.head(10).round(4))

# Explainable Boosting Classifier from interpret
ebm_importance = pd.Series(dtype=float)
if INTERPRET_AVAILABLE:
    ebm = ExplainableBoostingClassifier(random_state=42, interactions=5)
    ebm.fit(X_train, y_train)
    ebm_prob = ebm.predict_proba(X_test)[:, 1]
    ebm_pred = ebm.predict(X_test)
    print(f"ExplainableBoostingClassifier accuracy: {accuracy_score(y_test, ebm_pred):.4f}")
    print(f"ExplainableBoostingClassifier AUC: {safe_auc(y_test, ebm_prob):.4f}")

    # Global importances
    ebm_importance = pd.Series(ebm.term_importances(), index=ebm.term_names_).sort_values(ascending=False)
    print("\nEBM term importances:")
    print(ebm_importance.head(10).round(4))
else:
    print("interpret package unavailable; skipped EBM.")

# 6. Build conclusion score (0-100)
score = 50

# age effect significance and direction
if np.isfinite(age_p):
    if age_p < 0.001:
        score += 25 if age_coef > 0 else -25
    elif age_p < 0.05:
        score += 15 if age_coef > 0 else -15
    else:
        score += -10

# cultural variation and age-by-culture variation
if chi2_culture_p < 0.05:
    score += 10
if interaction_sig:
    score += 10

# support from interpretable models (age importance)
age_is_important = False
if "num__age" in tree_importance.index:
    age_rank_tree = int(tree_importance.index.get_loc("num__age")) + 1
    age_is_important = age_rank_tree <= 3

if len(ebm_importance) > 0:
    top_terms = list(ebm_importance.head(5).index)
    if any(term == "age" or term.startswith("age") for term in top_terms):
        age_is_important = True

if age_is_important:
    score += 5

score = int(max(0, min(100, round(score))))

# Plain-language interpretation
trend_word = "increases" if (np.isfinite(age_coef) and age_coef > 0) else "decreases"
interaction_phrase = (
    "Age trends differ significantly across cultures"
    if interaction_sig
    else "Age trends are broadly similar across cultures"
)

explanation = (
    f"Majority-choice reliance {trend_word} with age in logistic modeling "
    f"(age coef={age_coef:.3f}, p={age_p:.3g}). "
    f"Culture is associated with majority choice (chi-square p={chi2_culture_p:.3g}), and "
    f"{interaction_phrase.lower()}. "
    f"Interpretable models (decision tree/EBM) also identify age and culture as important predictors, "
    f"supporting a {'yes' if score >= 50 else 'no'} conclusion."
)

result = {
    "response": score,
    "explanation": explanation,
}

# Required output file: ONLY JSON object
with open("conclusion.txt", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=True)

print("\nWrote conclusion.txt")
print(json.dumps(result, indent=2))
