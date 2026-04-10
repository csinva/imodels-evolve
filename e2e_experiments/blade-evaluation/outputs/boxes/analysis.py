import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression


INFO_PATH = Path("info.json")
DATA_PATH = Path("boxes.csv")
CONCLUSION_PATH = Path("conclusion.txt")


def fit_logistic_and_loglik(X: pd.DataFrame, y: np.ndarray):
    """Fit logistic regression and return fitted model, log-likelihood, and parameter count."""
    try:
        model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=10000)
    except TypeError:
        # Backward compatibility with older sklearn versions.
        model = LogisticRegression(penalty="none", solver="lbfgs", max_iter=10000)

    model.fit(X, y)
    p = model.predict_proba(X)[:, 1]
    p = np.clip(p, 1e-12, 1 - 1e-12)
    ll = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    n_params = X.shape[1] + 1  # + intercept
    return model, ll, n_params


def lrt(ll_reduced: float, ll_full: float, df_reduced: int, df_full: int):
    """Likelihood-ratio test between nested models."""
    stat = 2.0 * (ll_full - ll_reduced)
    df = df_full - df_reduced
    p_value = float(chi2.sf(stat, df)) if df > 0 else np.nan
    return float(stat), int(df), p_value


def build_design_matrices(df: pd.DataFrame):
    """
    Build nested logistic-regression design matrices:
    M0: controls + culture
    M1: controls + culture + age
    M2: controls + culture + age + age:culture interactions
    """
    controls = df[["gender", "majority_first"]].astype(float).reset_index(drop=True)
    age = df[["age"]].astype(float).reset_index(drop=True)
    culture = pd.get_dummies(
        df["culture"].astype("category"),
        prefix="culture",
        drop_first=True,
        dtype=float,
    ).reset_index(drop=True)

    X0 = pd.concat([controls, culture], axis=1)
    X1 = pd.concat([controls, culture, age], axis=1)

    interaction_cols = {}
    for col in culture.columns:
        interaction_cols[f"age_x_{col}"] = age["age"].values * culture[col].values
    interactions = pd.DataFrame(interaction_cols)

    X2 = pd.concat([controls, culture, age, interactions], axis=1)

    return X0, X1, X2


def main():
    info = json.loads(INFO_PATH.read_text())
    question = info.get("research_questions", ["Unknown question"])[0]

    df = pd.read_csv(DATA_PATH)

    print("Research question:")
    print(question)
    print("\nData shape:", df.shape)
    print("Columns:", list(df.columns))
    print("Missing values by column:")
    print(df.isna().sum())

    # Outcome for majority preference reliance: chose majority option (y==2).
    df["majority_choice"] = (df["y"] == 2).astype(int)

    print("\nOutcome distribution (y):")
    print(df["y"].value_counts().sort_index())
    print("Majority-choice rate overall:", round(df["majority_choice"].mean(), 4))

    age_majority_rate = df.groupby("age")["majority_choice"].mean().sort_index()
    print("\nMajority-choice rate by age:")
    print(age_majority_rate)

    X0, X1, X2 = build_design_matrices(df)
    y = df["majority_choice"].to_numpy(dtype=int)

    m0, ll0, k0 = fit_logistic_and_loglik(X0, y)
    m1, ll1, k1 = fit_logistic_and_loglik(X1, y)
    m2, ll2, k2 = fit_logistic_and_loglik(X2, y)

    age_stat, age_df, p_age = lrt(ll0, ll1, k0, k1)
    int_stat, int_df, p_interaction = lrt(ll1, ll2, k1, k2)

    age_coef = float(m1.coef_[0][X1.columns.get_loc("age")])
    age_or = float(np.exp(age_coef))

    print("\nModel comparison results:")
    print(f"Age effect LRT: chi2({age_df})={age_stat:.4f}, p={p_age:.6g}")
    print(
        f"Age x culture interaction LRT: chi2({int_df})={int_stat:.4f}, "
        f"p={p_interaction:.6g}"
    )
    print(f"Estimated age coefficient (log-odds/year): {age_coef:.4f}")
    print(f"Estimated age odds ratio per year: {age_or:.4f}")

    # Convert evidence to 0-100 Likert score (0=strong No, 100=strong Yes).
    if p_age >= 0.05 and p_interaction >= 0.05:
        score = 10
    else:
        score = 35
        if p_age < 0.05:
            score += 35
            if p_age < 0.01:
                score += 5
            if age_coef > 0:
                score += 5
        if p_interaction < 0.05:
            score += 15
            if p_interaction < 0.01:
                score += 5

    score = int(max(0, min(100, round(score))))

    age_sig = "significant" if p_age < 0.05 else "not significant"
    int_sig = "significant" if p_interaction < 0.05 else "not significant"

    explanation = (
        "Majority-choice reliance (y=2) was analyzed with nested logistic models "
        "including controls (gender, majority_first), culture indicators, age, and "
        "age-by-culture interactions. "
        f"The global age effect was {age_sig} (LRT p={p_age:.3g}; "
        f"OR per year={age_or:.2f}), and the age-by-culture interaction was {int_sig} "
        f"(LRT p={p_interaction:.3g}). "
        "This indicates how strongly evidence supports age-related development in "
        "majority reliance across cultural contexts."
    )

    output = {"response": score, "explanation": explanation}
    CONCLUSION_PATH.write_text(json.dumps(output, ensure_ascii=True))

    print("\nWrote conclusion to", CONCLUSION_PATH)


if __name__ == "__main__":
    main()
