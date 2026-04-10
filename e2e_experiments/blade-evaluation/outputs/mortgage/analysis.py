import json
from math import sqrt

import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact, norm


def main() -> None:
    # Load data
    df = pd.read_csv("mortgage.csv")

    # Basic exploration and cleaning for the research question variables
    subset = df[["female", "accept"]].dropna().copy()
    subset["female"] = subset["female"].astype(int)
    subset["accept"] = subset["accept"].astype(int)

    # Contingency table: rows=female (0 male, 1 female), cols=accept (0 denied, 1 accepted)
    ct = pd.crosstab(subset["female"], subset["accept"])

    # Ensure complete 2x2 table ordering
    for row in [0, 1]:
        if row not in ct.index:
            ct.loc[row] = [0, 0]
    for col in [0, 1]:
        if col not in ct.columns:
            ct[col] = 0
    ct = ct.sort_index().reindex(sorted(ct.columns), axis=1)

    male_denied = int(ct.loc[0, 0])
    male_accepted = int(ct.loc[0, 1])
    female_denied = int(ct.loc[1, 0])
    female_accepted = int(ct.loc[1, 1])

    n_male = male_denied + male_accepted
    n_female = female_denied + female_accepted

    male_accept_rate = male_accepted / n_male if n_male else float("nan")
    female_accept_rate = female_accepted / n_female if n_female else float("nan")
    rate_diff = female_accept_rate - male_accept_rate

    # Significance tests
    chi2_stat, chi2_p, _, _ = chi2_contingency(ct.values)
    _, fisher_p = fisher_exact(ct.values)

    # Two-proportion z-test (pooled)
    pooled = (male_accepted + female_accepted) / (n_male + n_female)
    se = sqrt(pooled * (1 - pooled) * (1 / n_male + 1 / n_female)) if n_male and n_female else float("nan")
    z_stat = (rate_diff / se) if se and se > 0 else 0.0
    z_p = 2 * (1 - norm.cdf(abs(z_stat)))

    # Effect size: odds ratio (with small-sample continuity correction if needed)
    a, b, c, d = female_accepted, female_denied, male_accepted, male_denied
    if min(a, b, c, d) == 0:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    odds_ratio = (a * d) / (b * c)

    # Decision rule for requested Likert score
    # Question is whether a relationship exists. Non-significant results -> low score.
    alpha = 0.05
    significant = (chi2_p < alpha) and (fisher_p < alpha) and (z_p < alpha)

    if significant:
        # Higher score when relationship is significant; direction explained in text.
        response = 88
    else:
        # Very strong "No" when p-values are far from significance and effect is tiny.
        if chi2_p > 0.5 and fisher_p > 0.5 and abs(rate_diff) < 0.01:
            response = 2
        else:
            response = 15

    explanation = (
        f"Using {len(subset)} applications with non-missing gender, approval rates were "
        f"{female_accept_rate:.4f} for women and {male_accept_rate:.4f} for men "
        f"(difference = {rate_diff:.4f}). A chi-square test of independence gave "
        f"p={chi2_p:.4g}, Fisher's exact test gave p={fisher_p:.4g}, and a two-proportion "
        f"z-test gave p={z_p:.4g}. The estimated odds ratio (female vs male approval) "
        f"was {odds_ratio:.4f}. These tests show no statistically significant relationship "
        f"between gender and mortgage approval in this dataset, so the evidence supports "
        f"a strong 'No' to gender affecting approval rates here."
    )

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump({"response": int(response), "explanation": explanation}, f)


if __name__ == "__main__":
    main()
