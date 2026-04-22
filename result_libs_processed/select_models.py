"""Identify Pareto-optimal models and print the candidate pool for
hand-selecting the ~10 diverse finalists.

We rank on two axes:
  - mean_rank_global (lower is better; global rank averaged over 65 dev datasets)
  - test_interp_score (higher is better; fraction passed on 157-test held-out set)

Models without a valid held-out test interpretability score are excluded.
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
COMBINED = ROOT / "result_libs" / "combined_results.csv"


def pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows not dominated on (mean_rank_global low, test_interp_score high)."""
    df = df.sort_values(["mean_rank_global", "test_interp_score"], ascending=[True, False])
    keep = []
    best_interp = -1.0
    for _, row in df.iterrows():
        if row["test_interp_score"] > best_interp:
            keep.append(row)
            best_interp = row["test_interp_score"]
    return pd.DataFrame(keep)


def main() -> None:
    df = pd.read_csv(COMBINED)
    print(f"Loaded {len(df)} rows from combined_results.csv")

    # Keep only models with a valid held-out interpretability score > 0 and with a
    # python file on disk (so we can actually package them).
    valid = df[
        (df["test_interp_score"].notna())
        & (df["test_interp_score"] > 0)
    ].copy()
    baselines = valid[valid["commit"] == "baseline"].copy()
    evolved = valid[valid["commit"] != "baseline"].copy()
    evolved = evolved[evolved["model_file"].fillna("") != ""]

    print(f"After filtering: {len(evolved)} evolved, {len(baselines)} baselines")

    # Baseline Pareto (for comparison).
    print("\n=== Baseline Pareto front ===")
    b_pareto = pareto_front(baselines)
    print(b_pareto[["model_name", "mean_rank_global", "test_interp_score"]].to_string(index=False))

    # Evolved Pareto front.
    evolved_pareto = pareto_front(evolved)
    print(f"\n=== Evolved Pareto front ({len(evolved_pareto)} models) ===")
    print(
        evolved_pareto[
            ["experiment", "commit", "model_name", "mean_rank_global", "dev_interp_score", "test_interp_score", "description"]
        ].to_string(index=False)
    )

    # Combined Pareto (evolved + baselines), for context of what is Pareto-optimal overall.
    all_pareto = pareto_front(pd.concat([evolved, baselines]))
    print(f"\n=== Combined Pareto front ({len(all_pareto)} models) ===")
    print(
        all_pareto[
            ["experiment", "commit", "model_name", "mean_rank_global", "test_interp_score"]
        ].to_string(index=False)
    )

    # Also: relaxed candidate pool = evolved that beat any baseline on both
    # axes (i.e., not strictly dominated by any baseline).
    b_min_rank = baselines["mean_rank_global"].min()
    b_max_interp = baselines["test_interp_score"].max()
    print(
        f"\nBaseline extremes: min_rank={b_min_rank:.2f}, max_test_interp={b_max_interp:.3f}"
    )

    def beats_any_baseline(row):
        # Beats if there's no baseline that dominates it (lower rank AND higher interp).
        for _, b in baselines.iterrows():
            if b["mean_rank_global"] <= row["mean_rank_global"] and b["test_interp_score"] >= row["test_interp_score"]:
                # Strictly dominated if strictly better on at least one axis.
                if b["mean_rank_global"] < row["mean_rank_global"] or b["test_interp_score"] > row["test_interp_score"]:
                    return False
        return True

    non_dominated = evolved[evolved.apply(beats_any_baseline, axis=1)].copy()
    non_dominated = non_dominated.sort_values("test_interp_score", ascending=False)
    print(f"\n=== Evolved non-dominated vs baselines ({len(non_dominated)}) ===")
    print(
        non_dominated[
            ["experiment", "commit", "model_name", "mean_rank_global", "dev_interp_score", "test_interp_score"]
        ].head(40).to_string(index=False)
    )

    # Save pareto + non-dominated to disk.
    evolved_pareto.to_csv(ROOT / "result_libs" / "pareto_evolved.csv", index=False)
    non_dominated.to_csv(ROOT / "result_libs" / "non_dominated_evolved.csv", index=False)
    print("\nSaved pareto_evolved.csv and non_dominated_evolved.csv")


if __name__ == "__main__":
    main()
