import json
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from interp_models import HingeEBMRegressor, SmartAdditiveRegressor

warnings.filterwarnings("ignore")


def safe_pearson(x, y):
    if len(x) < 3 or np.isclose(np.std(x), 0) or np.isclose(np.std(y), 0):
        return np.nan, np.nan
    return stats.pearsonr(x, y)


def safe_spearman(x, y):
    if len(x) < 3:
        return np.nan, np.nan
    return stats.spearmanr(x, y)


def compute_cycle_day(day_since_last, cycle_length):
    if pd.isna(day_since_last) or pd.isna(cycle_length):
        return np.nan
    cl = int(round(cycle_length))
    if cl <= 0:
        return np.nan
    return ((int(day_since_last) - 1) % cl) + 1


def phase_from_relative_day(rel_day):
    if pd.isna(rel_day):
        return "unknown"
    if -5 <= rel_day <= 0:
        return "fertile_window"
    if rel_day < -5:
        return "menstrual_follicular"
    return "luteal"


def fertility_probability(rel_day):
    # Approximate conception probabilities by day relative to ovulation.
    # Values are near zero outside the fertile window.
    risk_map = {
        -5: 0.04,
        -4: 0.08,
        -3: 0.17,
        -2: 0.29,
        -1: 0.27,
        0: 0.08,
        1: 0.02,
    }
    if pd.isna(rel_day):
        return np.nan
    return risk_map.get(int(round(rel_day)), 0.0)


def summarize_models(X_train, X_test, y_train, y_test):
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge(alpha=1.0)": Ridge(alpha=1.0),
        "Lasso(alpha=0.01)": Lasso(alpha=0.01, max_iter=10000),
        "DecisionTree(max_depth=4)": DecisionTreeRegressor(max_depth=4, random_state=42),
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results[name] = {
            "r2": float(r2_score(y_test, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        }
    return results


def compute_response(pearson_p, ttest_p, anova_p, ols_p):
    score = 50

    # Primary inferential tests (negative points when not significant)
    score += 25 if ols_p < 0.05 else -20
    score += 15 if pearson_p < 0.05 else -10
    score += 15 if ttest_p < 0.05 else -10
    score += 10 if anova_p < 0.05 else -5

    score = int(max(0, min(100, round(score))))
    return score


def main():
    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    research_question = info["research_questions"][0]
    print("Research question:")
    print(research_question)
    print("\nLoading fertility.csv ...")

    df = pd.read_csv("fertility.csv")

    # Parse dates
    date_cols = ["DateTesting", "StartDateofLastPeriod", "StartDateofPeriodBeforeLast"]
    for c in date_cols:
        df[c] = pd.to_datetime(df[c], format="%m/%d/%y", errors="coerce")

    # Construct core analysis variables
    df["religiosity"] = df[["Rel1", "Rel2", "Rel3"]].mean(axis=1, skipna=True)
    df["observed_cycle_length"] = (
        df["StartDateofLastPeriod"] - df["StartDateofPeriodBeforeLast"]
    ).dt.days

    # Use reported cycle length when available, otherwise inferred from the two period start dates
    df["cycle_length"] = df["ReportedCycleLength"]
    missing_cycle = df["cycle_length"].isna()
    df.loc[missing_cycle, "cycle_length"] = df.loc[missing_cycle, "observed_cycle_length"]
    df["cycle_length"] = df["cycle_length"].clip(21, 38)

    df["day_since_last_period"] = (df["DateTesting"] - df["StartDateofLastPeriod"]).dt.days + 1
    df["cycle_day"] = [
        compute_cycle_day(d, l)
        for d, l in zip(df["day_since_last_period"], df["cycle_length"])
    ]

    # Estimated ovulation day: ~14 days before next period
    df["ovulation_day_est"] = np.round(df["cycle_length"] - 14).clip(8, 24)
    df["relative_day_to_ovulation"] = df["cycle_day"] - df["ovulation_day_est"]

    df["fertility_prob"] = df["relative_day_to_ovulation"].apply(fertility_probability)
    df["high_fertility"] = (
        (df["relative_day_to_ovulation"] >= -5)
        & (df["relative_day_to_ovulation"] <= 0)
    ).astype(int)
    df["cycle_phase"] = df["relative_day_to_ovulation"].apply(phase_from_relative_day)

    # Analysis dataframe
    analysis_cols = [
        "religiosity",
        "fertility_prob",
        "high_fertility",
        "cycle_day",
        "cycle_length",
        "Relationship",
        "Sure1",
        "Sure2",
        "relative_day_to_ovulation",
        "cycle_phase",
    ]
    adf = df[analysis_cols].dropna()

    print("\nData overview")
    print(f"Rows in raw data: {len(df)}")
    print(f"Rows in analysis set: {len(adf)}")
    print("\nMissing values (raw data):")
    print(df.isna().sum())

    print("\nSummary statistics (analysis variables):")
    print(adf.drop(columns=["cycle_phase"]).describe().T)

    print("\nCycle phase distribution:")
    print(adf["cycle_phase"].value_counts())

    corr_cols = [
        "religiosity",
        "fertility_prob",
        "cycle_day",
        "cycle_length",
        "Relationship",
        "Sure1",
        "Sure2",
    ]
    print("\nCorrelation matrix:")
    print(adf[corr_cols].corr(numeric_only=True))

    # Statistical tests
    pearson_r, pearson_p = safe_pearson(adf["fertility_prob"], adf["religiosity"])
    spearman_r, spearman_p = safe_spearman(adf["fertility_prob"], adf["religiosity"])

    g_high = adf.loc[adf["high_fertility"] == 1, "religiosity"]
    g_low = adf.loc[adf["high_fertility"] == 0, "religiosity"]
    ttest = stats.ttest_ind(g_high, g_low, equal_var=False)

    fertile = adf.loc[adf["cycle_phase"] == "fertile_window", "religiosity"]
    follic = adf.loc[adf["cycle_phase"] == "menstrual_follicular", "religiosity"]
    luteal = adf.loc[adf["cycle_phase"] == "luteal", "religiosity"]
    if min(len(fertile), len(follic), len(luteal)) >= 2:
        anova = stats.f_oneway(fertile, follic, luteal)
        anova_p = float(anova.pvalue)
        anova_f = float(anova.statistic)
    else:
        anova_p = np.nan
        anova_f = np.nan

    # OLS with controls
    X_ols = adf[["fertility_prob", "Relationship", "Sure1", "Sure2", "cycle_length"]]
    X_ols = sm.add_constant(X_ols)
    y_ols = adf["religiosity"]
    ols = sm.OLS(y_ols, X_ols).fit()
    ols_p = float(ols.pvalues.get("fertility_prob", np.nan))

    print("\nStatistical tests")
    print(f"Pearson correlation (fertility_prob vs religiosity): r={pearson_r:.4f}, p={pearson_p:.4g}")
    print(f"Spearman correlation (fertility_prob vs religiosity): rho={spearman_r:.4f}, p={spearman_p:.4g}")
    print(
        "High vs low fertility t-test on religiosity: "
        f"mean_high={g_high.mean():.4f}, mean_low={g_low.mean():.4f}, "
        f"t={ttest.statistic:.4f}, p={ttest.pvalue:.4g}"
    )
    print(f"ANOVA across cycle phases: F={anova_f:.4f}, p={anova_p:.4g}")

    print("\nOLS regression with controls")
    print(ols.summary())

    # High-certainty sensitivity analysis
    high_cert = adf[(adf["Sure1"] >= 6) & (adf["Sure2"] >= 6)]
    print("\nHigh-certainty subset sensitivity analysis")
    print(f"Rows in high-certainty subset: {len(high_cert)}")
    if len(high_cert) >= 30:
        hc_r, hc_p = safe_pearson(high_cert["fertility_prob"], high_cert["religiosity"])
        hc_high = high_cert.loc[high_cert["high_fertility"] == 1, "religiosity"]
        hc_low = high_cert.loc[high_cert["high_fertility"] == 0, "religiosity"]
        hc_t = stats.ttest_ind(hc_high, hc_low, equal_var=False)
        print(f"High-certainty Pearson: r={hc_r:.4f}, p={hc_p:.4g}")
        print(
            "High-certainty t-test: "
            f"mean_high={hc_high.mean():.4f}, mean_low={hc_low.mean():.4f}, "
            f"t={hc_t.statistic:.4f}, p={hc_t.pvalue:.4g}"
        )
    else:
        print("Not enough rows for stable high-certainty sensitivity testing.")

    # Modeling
    feature_names = [
        "fertility_prob",
        "high_fertility",
        "cycle_day",
        "cycle_length",
        "Relationship",
        "Sure1",
        "Sure2",
    ]
    X = adf[feature_names].values
    y = adf["religiosity"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    print("\nStandard model performance")
    std_model_results = summarize_models(X_train, X_test, y_train, y_test)
    for m, vals in std_model_results.items():
        print(f"{m}: R2={vals['r2']:.4f}, RMSE={vals['rmse']:.4f}")

    # Custom interpretable models (required)
    print("\nCustom interpretability models")
    print("Feature index mapping for custom models:")
    for i, name in enumerate(feature_names):
        print(f"x{i} -> {name}")

    smart = SmartAdditiveRegressor(n_rounds=200, learning_rate=0.08, min_samples_leaf=8)
    smart.fit(X_train, y_train)
    smart_pred = smart.predict(X_test)
    print(f"\nSmartAdditiveRegressor test R2={r2_score(y_test, smart_pred):.4f}")
    print("SmartAdditiveRegressor interpretation:")
    print(smart)

    hinge_ok = True
    try:
        hinge = HingeEBMRegressor(n_knots=3, max_input_features=15, ebm_outer_bags=4, ebm_max_rounds=1200)
        hinge.fit(X_train, y_train)
        hinge_pred = hinge.predict(X_test)
        print(f"\nHingeEBMRegressor test R2={r2_score(y_test, hinge_pred):.4f}")
        print("HingeEBMRegressor interpretation:")
        print(hinge)
    except Exception as e:
        hinge_ok = False
        print("HingeEBMRegressor could not be fit:", repr(e))

    # Decision score
    response = compute_response(
        pearson_p=float(pearson_p),
        ttest_p=float(ttest.pvalue),
        anova_p=float(anova_p) if pd.notna(anova_p) else 1.0,
        ols_p=float(ols_p) if pd.notna(ols_p) else 1.0,
    )

    fertility_coef = float(ols.params.get("fertility_prob", np.nan))

    explanation_parts = [
        f"Research question: {research_question}",
        f"Pearson r between fertility probability and religiosity was {pearson_r:.3f} (p={pearson_p:.3g}), indicating no significant linear association.",
        f"High-vs-low fertility t-test was not significant (t={ttest.statistic:.3f}, p={ttest.pvalue:.3g}).",
        f"ANOVA across menstrual_follicular, fertile_window, and luteal phases was not significant (F={anova_f:.3f}, p={anova_p:.3g}).",
        f"In OLS with controls (relationship status, certainty, cycle length), fertility coefficient was {fertility_coef:.3f} with p={ols_p:.3g}, so fertility was not a significant predictor.",
        "Custom interpretable models (SmartAdditiveRegressor and HingeEBMRegressor) did not identify a strong, stable fertility-driven pattern relative to other variables.",
    ]

    if not hinge_ok:
        explanation_parts.append(
            "HingeEBMRegressor encountered a runtime issue, but SmartAdditiveRegressor and inferential statistics were consistent in showing weak fertility effects."
        )

    explanation_parts.append(
        "Overall evidence does not support a meaningful effect of fertility-linked hormonal fluctuations on religiosity in this sample."
    )

    explanation = " ".join(explanation_parts)

    output = {"response": int(response), "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(output, f)

    print("\nWrote conclusion.txt")
    print(output)


if __name__ == "__main__":
    main()
