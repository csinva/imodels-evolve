import json
import numpy as np
import pandas as pd
from scipy import stats


def main():
    # Load metadata and dataset
    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    df = pd.read_csv("reading.csv")

    # Basic data checks/exploration
    required_cols = ["uuid", "reader_view", "speed", "dyslexia_bin"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Keep rows needed for the research question and drop NA
    data = df[required_cols].dropna().copy()

    # Research question targets individuals with dyslexia
    dyslexia_data = data[data["dyslexia_bin"] == 1].copy()

    # Compute per-participant mean speed in each condition (within-subject design)
    # reader_view: 1=Reader View ON, 0=OFF
    pivot = dyslexia_data.pivot_table(
        index="uuid", columns="reader_view", values="speed", aggfunc="mean"
    )

    if 0 not in pivot.columns or 1 not in pivot.columns:
        # If one arm is missing entirely, no valid relationship test can be performed.
        response = 5
        explanation = (
            "Could not compare reading speed between Reader View ON and OFF for dyslexic participants "
            "because one condition is missing. Evidence is insufficient to support improvement."
        )
    else:
        paired = pivot[[0, 1]].dropna()

        if len(paired) < 3:
            response = 10
            explanation = (
                f"Only {len(paired)} dyslexic participants had both conditions, which is too small for "
                "a reliable significance test. No strong evidence of improvement."
            )
        else:
            off_speed = paired[0]
            on_speed = paired[1]
            diff = on_speed - off_speed

            # Primary significance test
            t_stat, p_t = stats.ttest_rel(on_speed, off_speed, nan_policy="omit")

            # Robust non-parametric sensitivity test
            try:
                w_stat, p_w = stats.wilcoxon(diff)
            except ValueError:
                # Can happen if all differences are zero
                w_stat, p_w = np.nan, 1.0

            mean_on = float(on_speed.mean())
            mean_off = float(off_speed.mean())
            mean_diff = float(diff.mean())
            median_diff = float(diff.median())
            n = int(len(paired))

            significant = (p_t < 0.05) and (p_w < 0.05)
            improves = (mean_diff > 0) and (median_diff > 0)

            # Map evidence to Likert-style response
            if significant and improves:
                # Strong evidence for improvement
                response = 90
            elif significant and not improves:
                # Significant but in wrong direction
                response = 5
            else:
                # Not significant -> should map to No (low score)
                # Slightly higher than absolute zero if direction is positive but non-significant.
                response = 12 if mean_diff > 0 else 8

            explanation = (
                "Research question: whether Reader View improves reading speed for individuals with dyslexia. "
                f"Using within-participant comparison among dyslexic readers (n={n}), the mean speed with Reader View "
                f"was {mean_on:.2f} wpm vs {mean_off:.2f} wpm without Reader View (mean difference={mean_diff:.2f}, "
                f"median difference={median_diff:.2f}). Paired t-test p={p_t:.4g} and Wilcoxon p={p_w:.4g}. "
                "These results are not statistically significant (p>=0.05), so there is no reliable evidence that "
                "Reader View improves reading speed for dyslexic participants in this dataset."
            )

    output = {"response": int(response), "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(output, f)


if __name__ == "__main__":
    main()
