"""Aggregate all experiments under ../result_libs/ into a single
combined_results.csv with globally-ranked predictive performance and
per-category interpretability scores (dev + held-out).
"""

from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULT_LIBS = ROOT / "result_libs"
OUTPUT = RESULT_LIBS / "combined_results.csv"

sys.path.insert(0, str(ROOT / "evolve" / "src"))
from interp_eval import CATEGORY_TESTS  # noqa: E402


def _csv_test_name(fn_name: str) -> str:
    if fn_name.startswith("test_"):
        return fn_name[5:]
    if fn_name.startswith("hard_test_"):
        return "hard_" + fn_name[10:]
    if fn_name.startswith("discrim_test_"):
        return "discrim_" + fn_name[13:]
    return fn_name


TEST_TO_CATEGORY = {
    _csv_test_name(fn.__name__): cat for cat, tests in CATEGORY_TESTS for fn in tests
}
CATEGORIES = [cat for cat, _ in CATEGORY_TESTS]


def list_experiments() -> list[Path]:
    return sorted(d for d in RESULT_LIBS.iterdir() if d.is_dir())


def load_overall(exp: Path, fname: str) -> pd.DataFrame:
    path = exp / fname
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["experiment"] = exp.name
    return df


def load_perf(exp: Path) -> pd.DataFrame:
    path = exp / "performance_results.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["experiment"] = exp.name
    return df


def unique_model_id(experiment: str, commit: str, model_name: str) -> str:
    if commit == "baseline":
        return f"baseline::{model_name}"
    return f"{experiment}::{commit}::{model_name}"


def find_model_file(experiment: str, commit: str) -> str:
    if commit == "baseline":
        return ""
    base = RESULT_LIBS / experiment / "interpretable_regressors_lib"
    for sub in ("success", "failure"):
        d = base / sub
        if not d.exists():
            continue
        matches = list(d.glob(f"interpretable_regressor_{commit}_*.py"))
        if matches:
            return str(matches[0].relative_to(ROOT))
    return ""


def compute_category_scores(df: pd.DataFrame, category_source: str) -> pd.DataFrame:
    """Return per-model, per-category fraction-passed table.

    category_source: either 'suite_to_category' (use TEST_TO_CATEGORY lookup)
    or 'column' (use the 'category' column directly).
    """
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    if category_source == "suite_to_category":
        df["category"] = df["test"].map(TEST_TO_CATEGORY)
        # Drop any tests we can't categorize.
        df = df.dropna(subset=["category"])
    # passed is boolean-ish; coerce.
    df["passed"] = df["passed"].astype(bool)
    grouped = df.groupby(["experiment", "model", "category"])["passed"].mean().reset_index()
    pivoted = grouped.pivot_table(
        index=["experiment", "model"], columns="category", values="passed"
    ).reset_index()
    # Ensure all 6 categories appear.
    for cat in CATEGORIES:
        if cat not in pivoted.columns:
            pivoted[cat] = float("nan")
    return pivoted


def main() -> None:
    experiments = list_experiments()
    print(f"Found {len(experiments)} experiments")

    # ---- Collect per-experiment tables ----
    overall_dev_frames, overall_test_frames = [], []
    perf_frames = []
    interp_dev_frames, interp_test_frames = [], []

    for exp in experiments:
        overall_dev_frames.append(load_overall(exp, "overall_results.csv"))
        overall_test_frames.append(load_overall(exp, "overall_results_test.csv"))
        perf_frames.append(load_perf(exp))
        for fname, target in [
            ("interpretability_results.csv", interp_dev_frames),
            ("interpretability_results_test.csv", interp_test_frames),
        ]:
            p = exp / fname
            if p.exists():
                df = pd.read_csv(p)
                df["experiment"] = exp.name
                target.append(df)

    overall_dev = pd.concat(overall_dev_frames, ignore_index=True)
    overall_test = pd.concat(overall_test_frames, ignore_index=True)
    perf = pd.concat(perf_frames, ignore_index=True)
    interp_dev = pd.concat(interp_dev_frames, ignore_index=True) if interp_dev_frames else pd.DataFrame()
    interp_test = pd.concat(interp_test_frames, ignore_index=True) if interp_test_frames else pd.DataFrame()

    # ---- Build a stable unique id per (experiment, model_name) ----
    # For baselines (commit == 'baseline'), collapse across experiments -- keep
    # one canonical copy per baseline model_name.
    overall_dev["model_id"] = [
        unique_model_id(exp, commit, name)
        for exp, commit, name in zip(
            overall_dev["experiment"], overall_dev["commit"], overall_dev["model_name"]
        )
    ]
    overall_test["model_id"] = [
        unique_model_id(exp, commit, name)
        for exp, commit, name in zip(
            overall_test["experiment"], overall_test["commit"], overall_test["model_name"]
        )
    ]

    # ---- Compute global mean rank from performance_results pooled across
    # experiments. Deduplicate baselines so PyGAM/etc are only counted once
    # per dataset. For duplicate baselines we take the mean RMSE across runs.
    perf["model_id"] = perf.apply(
        lambda r: (
            f"baseline::{r['model']}"
            if r["model"] in set(overall_dev.loc[overall_dev["commit"] == "baseline", "model_name"])
            else f"{r['experiment']}::{r['model']}"
        ),
        axis=1,
    )
    # Map the non-baseline model_id back to (experiment, commit, model_name) via overall_dev.
    # Build a lookup on (experiment, model_name) -> commit so we can match unique ids.
    exp_model_to_commit = {
        (r["experiment"], r["model_name"]): r["commit"]
        for _, r in overall_dev.iterrows()
        if r["commit"] != "baseline"
    }

    def canonical_model_id(row):
        if row["model"] in {n for n in overall_dev.loc[overall_dev["commit"] == "baseline", "model_name"]}:
            return f"baseline::{row['model']}"
        commit = exp_model_to_commit.get((row["experiment"], row["model"]))
        if commit is None:
            return f"{row['experiment']}::unknown::{row['model']}"
        return f"{row['experiment']}::{commit}::{row['model']}"

    perf["model_id"] = perf.apply(canonical_model_id, axis=1)

    # Collapse duplicate baseline rows per (dataset, model_id) by averaging.
    perf_agg = (
        perf.groupby(["dataset", "model_id"], as_index=False)["rmse"].mean()
    )
    perf_agg["rank_global"] = perf_agg.groupby("dataset")["rmse"].rank(method="average")
    n_datasets = perf_agg["dataset"].nunique()
    # Only models with rank on every dataset get a mean rank (to avoid biased means).
    counts = perf_agg.groupby("model_id")["dataset"].nunique()
    full_models = counts[counts == n_datasets].index
    mean_rank_global = (
        perf_agg[perf_agg["model_id"].isin(full_models)]
        .groupby("model_id")["rank_global"]
        .mean()
        .rename("mean_rank_global")
    )

    # ---- Per-category dev interpretability (need test->category via TEST_TO_CATEGORY) ----
    dev_cat = compute_category_scores(interp_dev, category_source="suite_to_category")
    test_cat = compute_category_scores(interp_test, category_source="column")

    # ---- Assemble final rows ----
    # Starting from overall_dev. Drop the 'status' column per instructions.
    base = overall_dev.rename(
        columns={
            "mean_rank": "mean_rank_local",
            "frac_interpretability_tests_passed": "dev_interp_score",
        }
    ).copy()
    base = base.drop(columns=["status"], errors="ignore")

    # Merge global mean rank (key=model_id).
    base = base.merge(mean_rank_global, left_on="model_id", right_index=True, how="left")

    # Merge test-set interpretability score. overall_results_test.csv does not
    # carry the commit hash for evolved rows, so match on (experiment, model_name)
    # and collapse baseline duplicates by taking the mean score.
    if not overall_test.empty:
        test_slim = overall_test.rename(
            columns={"frac_interpretability_tests_passed": "test_interp_score"}
        )[["experiment", "model_name", "test_interp_score"]]
        test_slim = (
            test_slim.groupby(["experiment", "model_name"], as_index=False)[
                "test_interp_score"
            ].mean()
        )
        base = base.merge(test_slim, on=["experiment", "model_name"], how="left")
    else:
        base["test_interp_score"] = float("nan")

    # Merge per-category scores (dev).
    if not dev_cat.empty:
        dev_cat = dev_cat.rename(
            columns={c: f"dev_{c}" for c in CATEGORIES if c in dev_cat.columns}
        )
        base = base.merge(
            dev_cat,
            left_on=["experiment", "model_name"],
            right_on=["experiment", "model"],
            how="left",
        ).drop(columns=["model"], errors="ignore")
    else:
        for cat in CATEGORIES:
            base[f"dev_{cat}"] = float("nan")

    if not test_cat.empty:
        test_cat = test_cat.rename(
            columns={c: f"test_{c}" for c in CATEGORIES if c in test_cat.columns}
        )
        base = base.merge(
            test_cat,
            left_on=["experiment", "model_name"],
            right_on=["experiment", "model"],
            how="left",
        ).drop(columns=["model"], errors="ignore")
    else:
        for cat in CATEGORIES:
            base[f"test_{cat}"] = float("nan")

    # Path to python source.
    base["model_file"] = [
        find_model_file(exp, commit)
        for exp, commit in zip(base["experiment"], base["commit"])
    ]

    # Drop rows with missing mean_rank_global (models that failed / no perf).
    base = base.dropna(subset=["mean_rank_global"])

    # Drop duplicate baseline rows (keep first).
    baseline_mask = base["commit"] == "baseline"
    baselines = base[baseline_mask].drop_duplicates(subset=["model_name"], keep="first")
    evolved = base[~baseline_mask]
    combined = pd.concat([baselines, evolved], ignore_index=True)

    # Reorder columns for readability.
    front = [
        "experiment",
        "commit",
        "model_name",
        "description",
        "mean_rank_global",
        "mean_rank_local",
        "dev_interp_score",
        "test_interp_score",
    ]
    category_cols = [f"dev_{c}" for c in CATEGORIES] + [f"test_{c}" for c in CATEGORIES]
    tail = ["model_file", "model_id"]
    ordered = front + category_cols + tail
    existing = [c for c in ordered if c in combined.columns]
    remainder = [c for c in combined.columns if c not in existing]
    combined = combined[existing + remainder]

    # Sort by global rank ascending.
    combined = combined.sort_values("mean_rank_global").reset_index(drop=True)

    combined.to_csv(OUTPUT, index=False)
    print(f"Wrote {OUTPUT} with {len(combined)} rows")
    print(combined.head(20).to_string())


if __name__ == "__main__":
    main()
