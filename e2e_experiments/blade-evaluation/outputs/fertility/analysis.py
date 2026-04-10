import json
import numpy as np
import pandas as pd
from scipy import stats


def compute_fertility_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Parse dates
    for col in ["DateTesting", "StartDateofLastPeriod", "StartDateofPeriodBeforeLast"]:
        out[col] = pd.to_datetime(out[col], format="%m/%d/%y", errors="coerce")

    # Composite religiosity score
    out["religiosity_mean"] = out[["Rel1", "Rel2", "Rel3"]].mean(axis=1)

    # Estimate cycle length from reported and observed dates
    observed_cycle = (
        out["StartDateofLastPeriod"] - out["StartDateofPeriodBeforeLast"]
    ).dt.days
    observed_cycle = observed_cycle.where(observed_cycle.between(15, 60))

    reported_cycle = pd.to_numeric(out["ReportedCycleLength"], errors="coerce")
    reported_cycle = reported_cycle.where(reported_cycle.between(15, 60))

    out["cycle_length_est"] = np.where(
        reported_cycle.notna() & observed_cycle.notna(),
        (reported_cycle + observed_cycle) / 2.0,
        np.where(reported_cycle.notna(), reported_cycle, observed_cycle),
    )

    # Cycle day at testing (1-indexed)
    out["cycle_day"] = (
        out["DateTesting"] - out["StartDateofLastPeriod"]
    ).dt.days + 1

    # Keep plausible cycle days
    out.loc[~out["cycle_day"].between(1, 60), "cycle_day"] = np.nan

    # Ovulation approximation: luteal phase ~14 days
    out["ovulation_day"] = out["cycle_length_est"] - 14

    # High fertility window: 5 days before ovulation through ovulation day
    out["fertile_window_start"] = out["ovulation_day"] - 5
    out["fertile_window_end"] = out["ovulation_day"]

    out["high_fertility"] = (
        out["cycle_day"].ge(out["fertile_window_start"])
        & out["cycle_day"].le(out["fertile_window_end"])
    ).astype(float)

    # Mark rows usable for analysis
    valid = (
        out["cycle_length_est"].notna()
        & out["cycle_day"].notna()
        & out["religiosity_mean"].notna()
        & out["Relationship"].notna()
    )
    out["valid_for_analysis"] = valid

    return out


def ols_with_pvalues(x: np.ndarray, y: np.ndarray):
    """Return OLS coefficients and two-sided p-values."""
    n, k = x.shape
    xtx_inv = np.linalg.pinv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    residuals = y - x @ beta
    dof = n - k

    if dof <= 0:
        pvals = np.full(k, np.nan)
        return beta, pvals

    sigma2 = (residuals @ residuals) / dof
    cov_beta = sigma2 * xtx_inv
    se = np.sqrt(np.maximum(np.diag(cov_beta), 0))

    with np.errstate(divide="ignore", invalid="ignore"):
        t_stats = beta / se
    pvals = 2 * stats.t.sf(np.abs(t_stats), df=dof)
    return beta, pvals


def main() -> None:
    df = pd.read_csv("fertility.csv")
    data = compute_fertility_features(df)

    n_total = len(data)
    analysis = data[data["valid_for_analysis"]].copy()
    n_analysis = len(analysis)

    fertile_group = analysis.loc[analysis["high_fertility"] == 1.0, "religiosity_mean"]
    nonfertile_group = analysis.loc[analysis["high_fertility"] == 0.0, "religiosity_mean"]

    # Welch's t-test (primary group comparison)
    t_stat, p_ttest = stats.ttest_ind(
        fertile_group,
        nonfertile_group,
        equal_var=False,
        nan_policy="omit",
    )

    # Point-biserial correlation as supplementary association test
    r_pb, p_pb = stats.pointbiserialr(
        analysis["high_fertility"].to_numpy(),
        analysis["religiosity_mean"].to_numpy(),
    )

    # OLS with relationship-status controls using dummy variables
    rel_dummies = pd.get_dummies(
        analysis["Relationship"].astype(int),
        prefix="rel",
        drop_first=True,
    )
    x = pd.concat(
        [
            pd.Series(1.0, index=analysis.index, name="intercept"),
            analysis["high_fertility"],
            rel_dummies,
        ],
        axis=1,
    ).astype(float)
    y = analysis["religiosity_mean"].astype(float).to_numpy()

    beta, pvals = ols_with_pvalues(x.to_numpy(), y)
    hf_idx = list(x.columns).index("high_fertility")
    coef_hf = float(beta[hf_idx])
    p_ols_hf = float(pvals[hf_idx])

    mean_fertile = float(fertile_group.mean()) if len(fertile_group) else np.nan
    mean_nonfertile = float(nonfertile_group.mean()) if len(nonfertile_group) else np.nan

    # Significance-driven answer
    significant = (
        (np.isfinite(p_ttest) and p_ttest < 0.05)
        or (np.isfinite(p_ols_hf) and p_ols_hf < 0.05)
    )

    if significant:
        response = 85
        verdict = "evidence of an association"
    else:
        response = 12
        verdict = "no statistically significant evidence of an association"

    explanation = (
        f"Using {n_analysis} of {n_total} participants with usable cycle/date data, fertility status was "
        f"estimated from cycle timing (high-fertility window = 5 days before ovulation through ovulation). "
        f"Religiosity was the mean of Rel1-Rel3. Welch t-test: t={t_stat:.3f}, p={p_ttest:.4f}; means "
        f"high-fertility={mean_fertile:.3f}, non-high-fertility={mean_nonfertile:.3f}. Supplementary "
        f"point-biserial correlation: r={r_pb:.3f}, p={p_pb:.4f}. OLS with relationship-status controls: "
        f"beta_high_fertility={coef_hf:.3f}, p={p_ols_hf:.4f}. Conclusion: {verdict} for an effect of "
        f"fertility-linked hormonal fluctuations on religiosity in this dataset."
    )

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps({"response": int(response), "explanation": explanation}))

    print("Rows total:", n_total)
    print("Rows analyzed:", n_analysis)
    print("Welch p:", p_ttest)
    print("OLS high_fertility p:", p_ols_hf)
    print("Wrote conclusion.txt")


if __name__ == "__main__":
    main()
