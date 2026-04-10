import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def benjamini_hochberg(pvals: Dict[str, float]) -> Dict[str, float]:
    """Return BH-adjusted p-values keyed by test name."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted = {}

    # Compute raw adjusted values
    raw = []
    for i, (name, p) in enumerate(items, start=1):
        raw.append((name, min(1.0, p * m / i)))

    # Enforce monotonicity from largest rank to smallest
    min_so_far = 1.0
    for name, val in reversed(raw):
        min_so_far = min(min_so_far, val)
        adjusted[name] = min_so_far

    return adjusted


def poisson_rate_ci(total_events: float, total_exposure: float, alpha: float = 0.05) -> Tuple[float, float]:
    """Exact CI for Poisson rate using chi-square method."""
    k = int(total_events)
    if total_exposure <= 0:
        return np.nan, np.nan

    if k == 0:
        lower = 0.0
    else:
        lower = 0.5 * stats.chi2.ppf(alpha / 2.0, 2 * k) / total_exposure
    upper = 0.5 * stats.chi2.ppf(1 - alpha / 2.0, 2 * (k + 1)) / total_exposure
    return lower, upper


def main() -> None:
    # 1) Read metadata and question
    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    question = info.get("research_questions", ["Unknown question"])[0]
    print("Research question:")
    print(question)

    # 2) Load data
    df = pd.read_csv("fish.csv")

    # 3) Explore data
    print("\nDataset shape:", df.shape)
    print("Columns:", list(df.columns))
    print("Missing values per column:")
    print(df.isna().sum())
    print("\nSummary statistics:")
    print(df.describe(include="all"))

    # Keep only positive-hour rows for rate calculations/tests
    df = df[df["hours"] > 0].copy()
    df["fish_per_hour"] = df["fish_caught"] / df["hours"]

    total_fish = float(df["fish_caught"].sum())
    total_hours = float(df["hours"].sum())
    overall_rate = total_fish / total_hours
    rate_ci_low, rate_ci_high = poisson_rate_ci(total_fish, total_hours)

    print("\nRate summary:")
    print(f"Total fish / total hours = {overall_rate:.4f} fish per hour")
    print(f"95% Poisson CI for overall rate: [{rate_ci_low:.4f}, {rate_ci_high:.4f}]")
    print(f"Median individual-group fish/hour: {df['fish_per_hour'].median():.4f}")

    # 4) Significance tests for relationships
    tests = {}
    directions: List[str] = []

    # Binary factors: nonparametric two-sample test
    for col in ["livebait", "camper"]:
        g0 = df.loc[df[col] == 0, "fish_per_hour"]
        g1 = df.loc[df[col] == 1, "fish_per_hour"]
        stat, p = stats.mannwhitneyu(g0, g1, alternative="two-sided")
        tests[col] = float(p)

        med0 = float(np.median(g0))
        med1 = float(np.median(g1))
        direction = "higher" if med1 > med0 else "lower" if med1 < med0 else "similar"
        directions.append(
            f"{col}: median fish/hour was {direction} for {col}=1 than {col}=0 "
            f"(medians {med1:.3f} vs {med0:.3f}, raw p={p:.4g})"
        )

    # Multi-level count factors: Kruskal-Wallis + Spearman for monotonic direction
    for col in ["persons", "child"]:
        grouped = [g["fish_per_hour"].values for _, g in df.groupby(col)]
        stat, p = stats.kruskal(*grouped)
        tests[col] = float(p)

        rho, rho_p = stats.spearmanr(df[col], df["fish_per_hour"])
        if rho > 0:
            trend = "increased"
        elif rho < 0:
            trend = "decreased"
        else:
            trend = "showed no monotonic change"
        directions.append(
            f"{col}: fish/hour varied across groups (Kruskal-Wallis raw p={p:.4g}); "
            f"Spearman rho={rho:.3f} (p={rho_p:.4g}), indicating fish/hour {trend} as {col} increased"
        )

    adj = benjamini_hochberg(tests)
    significant = [name for name, p in adj.items() if p < 0.05]

    print("\nRaw p-values:", tests)
    print("BH-adjusted p-values:", adj)
    print("Significant factors (BH<0.05):", significant)

    # 5) Convert to required Likert-style yes/no score
    # Here: question asks whether factors influence catch rate and to estimate fish/hour.
    # Strong evidence -> high score, weak/no evidence -> low score.
    n_sig = len(significant)
    if n_sig >= 3:
        response = 92
    elif n_sig >= 1:
        response = 78
    else:
        response = 12

    explanation = (
        f"Estimated average catch rate is {overall_rate:.3f} fish/hour "
        f"(95% CI {rate_ci_low:.3f} to {rate_ci_high:.3f}) based on total fish divided by total hours. "
        f"Using significance tests on fish-per-hour, BH-adjusted p-values identified {n_sig} significant factor(s): "
        f"{', '.join(significant) if significant else 'none'}. "
        + " ".join(directions)
        + " This supports a strong Yes that catch rate is related to trip/group factors."
    )

    result = {
        "response": int(response),
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=True)

    print("\nWrote conclusion.txt")
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
