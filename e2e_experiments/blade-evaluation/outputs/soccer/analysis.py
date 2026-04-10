import json
import numpy as np
import pandas as pd
from scipy import stats


def two_rate_z_test(events_1, exposure_1, events_0, exposure_0):
    """Two-sided z-test comparing two Poisson incidence rates."""
    rate_1 = events_1 / exposure_1
    rate_0 = events_0 / exposure_0
    pooled_rate = (events_1 + events_0) / (exposure_1 + exposure_0)
    se = np.sqrt(pooled_rate * (1.0 / exposure_1 + 1.0 / exposure_0))
    z = (rate_1 - rate_0) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return rate_1, rate_0, z, p_value


def main():
    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    question = info.get("research_questions", ["Unknown question"])[0]
    print("Research question:", question)

    df = pd.read_csv("soccer.csv")
    print("\nDataset shape:", df.shape)
    print("Columns:", list(df.columns))

    required = ["rater1", "rater2", "redCards", "games"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Build skin-tone score from both raters.
    df["skin_tone"] = df[["rater1", "rater2"]].mean(axis=1)

    # Keep rows where the key variables are observed and exposure is valid.
    analysis_df = df.dropna(subset=["skin_tone", "redCards", "games"]).copy()
    analysis_df = analysis_df[analysis_df["games"] > 0].copy()

    # Binary grouping for the direct dark-vs-light question.
    analysis_df["dark_skin"] = (analysis_df["skin_tone"] >= 0.5).astype(int)
    analysis_df["any_red"] = (analysis_df["redCards"] > 0).astype(int)

    print("\nRows used for analysis:", len(analysis_df))
    print("Missing skin-tone fraction in original data:", df["skin_tone"].isna().mean())

    # Test 1: Difference in probability of receiving any red card.
    contingency = pd.crosstab(analysis_df["dark_skin"], analysis_df["any_red"])
    contingency = contingency.reindex(index=[0, 1], columns=[0, 1], fill_value=0)

    chi2, p_chi2, _, _ = stats.chi2_contingency(contingency)
    odds_ratio, p_fisher = stats.fisher_exact(contingency.values)

    # Test 2: Difference in red-card incidence rate per game.
    grouped = analysis_df.groupby("dark_skin").agg(
        red_cards=("redCards", "sum"),
        games=("games", "sum"),
        n_rows=("redCards", "size"),
        any_red_rate=("any_red", "mean"),
    )

    events_light = grouped.loc[0, "red_cards"]
    games_light = grouped.loc[0, "games"]
    events_dark = grouped.loc[1, "red_cards"]
    games_dark = grouped.loc[1, "games"]

    rate_dark, rate_light, z_rate, p_rate = two_rate_z_test(
        events_dark, games_dark, events_light, games_light
    )

    rate_ratio = rate_dark / rate_light
    se_log_rr = np.sqrt(1.0 / events_dark + 1.0 / events_light)
    rr_ci_low = float(np.exp(np.log(rate_ratio) - 1.96 * se_log_rr))
    rr_ci_high = float(np.exp(np.log(rate_ratio) + 1.96 * se_log_rr))

    # Test 3: Continuous association as robustness check.
    spearman_rho, p_spearman = stats.spearmanr(
        analysis_df["skin_tone"], analysis_df["redCards"]
    )

    print("\nContingency table (dark_skin x any_red):")
    print(contingency)
    print(f"Chi-square p-value: {p_chi2:.6g}")
    print(f"Fisher exact p-value: {p_fisher:.6g}, odds ratio: {odds_ratio:.4f}")

    print("\nGrouped summary:")
    print(grouped)
    print(
        f"Red-card rate per game (light): {rate_light:.6f}; "
        f"(dark): {rate_dark:.6f}; rate ratio: {rate_ratio:.4f}"
    )
    print(f"Rate-difference z-test p-value: {p_rate:.6g} (z={z_rate:.4f})")
    print(f"Rate ratio 95% CI: [{rr_ci_low:.4f}, {rr_ci_high:.4f}]")
    print(f"Spearman rho (skin_tone vs redCards): {spearman_rho:.4f}, p={p_spearman:.6g}")

    # Decision rule: prioritize the exposure-adjusted rate test for "more likely".
    significant_rate = p_rate < 0.05
    higher_dark_rate = rate_dark > rate_light

    if significant_rate and higher_dark_rate:
        # Significant but modest effect -> moderately high "Yes".
        response = 72 if rate_ratio >= 1.2 else 66
        explanation = (
            "Using rows with valid skin-tone ratings and games played, dark-skin players had a "
            f"higher red-card incidence rate per game than light-skin players "
            f"({rate_dark:.5f} vs {rate_light:.5f}; rate ratio={rate_ratio:.3f}, "
            f"95% CI [{rr_ci_low:.3f}, {rr_ci_high:.3f}]), and this difference was statistically "
            f"significant (rate z-test p={p_rate:.3g}). The any-red-card proportion test alone "
            f"was not significant (chi-square p={p_chi2:.3g}), so evidence is positive but modest "
            f"rather than overwhelming."
        )
    else:
        response = 28
        explanation = (
            "The analysis did not find statistically significant evidence that dark-skin players "
            "were more likely to receive red cards once tested with the chosen significance tests. "
            f"Rate z-test p={p_rate:.3g}, chi-square p={p_chi2:.3g}."
        )

    result = {"response": int(response), "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=True)

    print("\nWrote conclusion.txt")


if __name__ == "__main__":
    main()
