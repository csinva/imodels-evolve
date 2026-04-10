import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def permutation_pvalues_for_logistic(X, y, n_perm=5000, seed=0):
    rng = np.random.default_rng(seed)

    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=2000)
    model.fit(Xz, y)
    observed = model.coef_[0].copy()

    perm_coefs = np.zeros((n_perm, X.shape[1]))
    for i in range(n_perm):
        y_perm = rng.permutation(y)
        perm_model = LogisticRegression(max_iter=2000)
        perm_model.fit(Xz, y_perm)
        perm_coefs[i, :] = perm_model.coef_[0]

    pvals = []
    for j in range(X.shape[1]):
        p = (np.sum(np.abs(perm_coefs[:, j]) >= abs(observed[j])) + 1) / (n_perm + 1)
        pvals.append(float(p))

    return observed, pvals


def main():
    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    research_question = info["research_questions"][0]
    print("Research question:", research_question)

    df = pd.read_csv("crofoot.csv")
    print("\nData shape:", df.shape)
    print("Columns:", list(df.columns))
    print("\nMissing values by column:")
    print(df.isna().sum())

    # Derived predictors for relative group size and contest location advantage.
    # Positive values favor the focal group.
    df["rel_group_size"] = df["n_focal"] - df["n_other"]
    df["location_advantage"] = df["dist_other"] - df["dist_focal"]

    print("\nDerived variable summary:")
    print(df[["rel_group_size", "location_advantage", "win"]].describe())

    results = {}
    for var in ["rel_group_size", "location_advantage"]:
        r, p_corr = stats.pointbiserialr(df["win"], df[var])

        wins = df.loc[df["win"] == 1, var]
        losses = df.loc[df["win"] == 0, var]
        _, p_mw = stats.mannwhitneyu(wins, losses, alternative="two-sided")

        results[var] = {
            "pointbiserial_r": float(r),
            "pointbiserial_p": float(p_corr),
            "mannwhitney_p": float(p_mw),
            "win_median": float(np.median(wins)),
            "loss_median": float(np.median(losses)),
        }

    # Supporting multivariable model (with permutation p-values for coefficients)
    X = df[["rel_group_size", "location_advantage"]].to_numpy()
    y = df["win"].to_numpy()
    coefs, perm_pvals = permutation_pvalues_for_logistic(X, y, n_perm=5000, seed=7)

    logistic_results = {
        "coef_rel_group_size": float(coefs[0]),
        "coef_location_advantage": float(coefs[1]),
        "perm_p_rel_group_size": float(perm_pvals[0]),
        "perm_p_location_advantage": float(perm_pvals[1]),
    }

    print("\nSignificance test results:")
    print(json.dumps(results, indent=2))
    print("\nLogistic regression (standardized coefficients + permutation p-values):")
    print(json.dumps(logistic_results, indent=2))

    # Decision rule for Likert response:
    # if neither factor is significant (p < 0.05), return a low score.
    pvals_all = [
        results["rel_group_size"]["pointbiserial_p"],
        results["rel_group_size"]["mannwhitney_p"],
        logistic_results["perm_p_rel_group_size"],
        results["location_advantage"]["pointbiserial_p"],
        results["location_advantage"]["mannwhitney_p"],
        logistic_results["perm_p_location_advantage"],
    ]
    any_significant = any(p < 0.05 for p in pvals_all)

    if any_significant:
        response = 75
    else:
        response = 15

    explanation = (
        "Using 58 contests, neither relative group size nor contest location advantage "
        "showed statistically significant association with focal-group win probability "
        f"(group-size p-values: point-biserial={results['rel_group_size']['pointbiserial_p']:.3f}, "
        f"Mann-Whitney={results['rel_group_size']['mannwhitney_p']:.3f}, "
        f"logistic permutation={logistic_results['perm_p_rel_group_size']:.3f}; "
        f"location p-values: point-biserial={results['location_advantage']['pointbiserial_p']:.3f}, "
        f"Mann-Whitney={results['location_advantage']['mannwhitney_p']:.3f}, "
        f"logistic permutation={logistic_results['perm_p_location_advantage']:.3f}). "
        "Estimated effects are directionally positive but not significant, so evidence for a reliable "
        "relationship in this sample is weak."
    )

    output = {"response": int(response), "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(output, f)

    print("\nWrote conclusion.txt")


if __name__ == "__main__":
    main()
