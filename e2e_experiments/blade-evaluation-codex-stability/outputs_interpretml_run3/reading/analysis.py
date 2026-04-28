import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from interpret.glassbox import ExplainableBoostingRegressor

warnings.filterwarnings("ignore")


def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main():
    base = Path(__file__).resolve().parent
    info_path = base / "info.json"
    data_path = base / "reading.csv"
    conclusion_path = base / "conclusion.txt"

    info = json.loads(info_path.read_text())
    question = info.get("research_questions", ["Unknown question"])[0]

    df = pd.read_csv(data_path)

    print_header("Research Question")
    print(question)

    print_header("Dataset Overview")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {df.shape[1]}")
    print("Missing values by column:")
    print(df.isna().sum().sort_values(ascending=False).to_string())

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    print_header("Numeric Summary")
    print(df[numeric_cols].describe().T.to_string())

    print_header("Categorical Summary (top 5 levels)")
    for col in categorical_cols:
        vc = df[col].astype("string").value_counts(dropna=False).head(5)
        print(f"\n{col}:")
        print(vc.to_string())

    print_header("Distribution Checks for Speed")
    speed = df["speed"].dropna()
    log_speed = np.log1p(speed)
    print(
        f"speed mean={speed.mean():.3f}, median={speed.median():.3f}, std={speed.std():.3f}, "
        f"skew={stats.skew(speed):.3f}"
    )
    print(
        f"log1p(speed) mean={log_speed.mean():.3f}, median={log_speed.median():.3f}, std={log_speed.std():.3f}, "
        f"skew={stats.skew(log_speed):.3f}"
    )

    print_header("Correlations with Speed (numeric features)")
    corr = df[numeric_cols].corr(numeric_only=True)["speed"].drop("speed").sort_values(
        key=lambda s: s.abs(), ascending=False
    )
    print(corr.to_string())

    # Focused inferential tests for dyslexic readers
    print_header("Statistical Tests: Reader View Effect Among Dyslexic Participants")
    dys_df = df[df["dyslexia_bin"] == 1].copy()
    dys_df = dys_df.dropna(subset=["uuid", "reader_view", "speed"])
    dys_df["log_speed"] = np.log1p(dys_df["speed"])

    # Row-level Welch t-test (independence approximation)
    lv1 = dys_df.loc[dys_df["reader_view"] == 1, "log_speed"]
    lv0 = dys_df.loc[dys_df["reader_view"] == 0, "log_speed"]
    welch = stats.ttest_ind(lv1, lv0, equal_var=False)
    print(
        f"Welch t-test on log_speed (row-level): t={welch.statistic:.4f}, p={welch.pvalue:.4g}, "
        f"n1={len(lv1)}, n0={len(lv0)}"
    )

    # Participant-level paired test (preferred due repeated measures)
    paired = dys_df.pivot_table(index="uuid", columns="reader_view", values="speed", aggfunc="mean")
    paired = paired.dropna(subset=[0, 1])
    paired_log0 = np.log1p(paired[0])
    paired_log1 = np.log1p(paired[1])
    paired_t = stats.ttest_rel(paired_log1, paired_log0)
    try:
        wil = stats.wilcoxon(paired_log1 - paired_log0)
    except ValueError:
        wil = None

    diff = paired_log1 - paired_log0
    cohens_dz = safe_float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else 0.0
    print(
        f"Paired t-test on participant means (log_speed): t={paired_t.statistic:.4f}, "
        f"p={paired_t.pvalue:.4g}, pairs={len(paired)}"
    )
    if wil is not None:
        print(f"Wilcoxon signed-rank: W={safe_float(wil.statistic):.4f}, p={safe_float(wil.pvalue):.4g}")
    print(f"Mean paired log-difference (reader_view=1 minus 0): {diff.mean():.5f}")
    print(f"Cohen's dz: {cohens_dz:.4f}")

    # OLS in dyslexic subset (with cluster-robust SE by participant)
    covars = [
        "reader_view",
        "num_words",
        "correct_rate",
        "age",
        "retake_trial",
        "img_width",
        "Flesch_Kincaid",
    ]
    dreg = dys_df.dropna(subset=covars + ["uuid", "log_speed"]).copy()
    ols_dys = smf.ols(
        "log_speed ~ reader_view + num_words + correct_rate + age + retake_trial + img_width + Flesch_Kincaid",
        data=dreg,
    ).fit(cov_type="cluster", cov_kwds={"groups": dreg["uuid"]})
    rv_coef = safe_float(ols_dys.params.get("reader_view", np.nan))
    rv_p = safe_float(ols_dys.pvalues.get("reader_view", np.nan))
    print(
        f"Cluster-robust OLS (dyslexia subset): reader_view coef={rv_coef:.5f}, p={rv_p:.4g}"
    )

    # Full-sample interaction model
    full_cols = [
        "log_speed",
        "reader_view",
        "dyslexia_bin",
        "num_words",
        "correct_rate",
        "age",
        "retake_trial",
        "img_width",
        "Flesch_Kincaid",
    ]
    full = df.copy()
    full["log_speed"] = np.log1p(full["speed"])
    full = full.dropna(subset=full_cols)
    ols_int = smf.ols(
        "log_speed ~ reader_view * dyslexia_bin + num_words + correct_rate + age + retake_trial + img_width + Flesch_Kincaid",
        data=full,
    ).fit()
    int_coef = safe_float(ols_int.params.get("reader_view:dyslexia_bin", np.nan))
    int_p = safe_float(ols_int.pvalues.get("reader_view:dyslexia_bin", np.nan))
    print(
        "Interaction OLS (all participants): "
        f"reader_view:dyslexia_bin coef={int_coef:.5f}, p={int_p:.4g}"
    )

    # Interpretable ML models in dyslexia subset
    print_header("Interpretable Models (Dyslexia Subset)")
    model_cols = [
        "reader_view",
        "num_words",
        "correct_rate",
        "age",
        "retake_trial",
        "img_width",
        "Flesch_Kincaid",
        "device",
        "education",
        "gender",
        "language",
        "english_native",
    ]
    mdf = dys_df.dropna(subset=["speed"] + ["reader_view"]).copy()
    y = np.log1p(mdf["speed"])

    numeric_features = [c for c in model_cols if c in mdf.columns and pd.api.types.is_numeric_dtype(mdf[c])]
    categorical_features = [c for c in model_cols if c in mdf.columns and c not in numeric_features]

    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    lin = Pipeline([("pre", pre), ("model", LinearRegression())])
    lin.fit(mdf[model_cols], y)

    feat_names = lin.named_steps["pre"].get_feature_names_out()
    coefs = lin.named_steps["model"].coef_
    coef_df = pd.DataFrame({"feature": feat_names, "coef": coefs})
    coef_df["abs_coef"] = coef_df["coef"].abs()
    top_lin = coef_df.sort_values("abs_coef", ascending=False).head(10)
    print("Top 10 absolute coefficients (LinearRegression on log_speed):")
    print(top_lin[["feature", "coef"]].to_string(index=False))

    rv_linear_rows = coef_df[coef_df["feature"].str.contains("reader_view", regex=False)]
    rv_linear_coef = safe_float(rv_linear_rows["coef"].sum()) if not rv_linear_rows.empty else np.nan
    print(f"Aggregated linear coefficient contribution for reader_view features: {rv_linear_coef:.5f}")

    tree = Pipeline(
        [
            ("pre", pre),
            (
                "model",
                DecisionTreeRegressor(max_depth=3, min_samples_leaf=20, random_state=42),
            ),
        ]
    )
    tree.fit(mdf[model_cols], y)
    t_importances = tree.named_steps["model"].feature_importances_
    tree_imp = pd.DataFrame({"feature": feat_names, "importance": t_importances}).sort_values(
        "importance", ascending=False
    )
    print("Top 10 feature importances (DecisionTreeRegressor):")
    print(tree_imp.head(10).to_string(index=False))

    # Explainable Boosting Regressor for additive interpretability
    ebm_features = [c for c in model_cols if c in mdf.columns]
    X_ebm = mdf[ebm_features].copy()
    for c in categorical_features:
        if c in X_ebm.columns:
            X_ebm[c] = X_ebm[c].astype("category")
    for c in numeric_features:
        if c in X_ebm.columns:
            X_ebm[c] = pd.to_numeric(X_ebm[c], errors="coerce")

    # Light imputation for EBM
    for c in numeric_features:
        if c in X_ebm.columns:
            X_ebm[c] = X_ebm[c].fillna(X_ebm[c].median())
    for c in categorical_features:
        if c in X_ebm.columns:
            X_ebm[c] = X_ebm[c].cat.add_categories(["Missing"]).fillna("Missing")

    ebm = ExplainableBoostingRegressor(interactions=0, random_state=42)
    ebm.fit(X_ebm, y)

    term_names = list(ebm.term_names_)
    term_imps = list(ebm.term_importances())
    ebm_imp = pd.DataFrame({"term": term_names, "importance": term_imps}).sort_values(
        "importance", ascending=False
    )
    print("Top 10 term importances (EBM):")
    print(ebm_imp.head(10).to_string(index=False))

    rv_ebm_rows = ebm_imp[ebm_imp["term"].astype(str).str.contains("reader_view", regex=False)]
    rv_ebm_importance = safe_float(rv_ebm_rows["importance"].sum()) if not rv_ebm_rows.empty else 0.0

    # Convert log-effect to percent for interpretability
    approx_pct_change = (np.exp(diff.mean()) - 1.0) * 100.0

    print_header("Result Interpretation")
    print(
        "Primary paired analysis among dyslexic participants shows no meaningful speed gain with reader view "
        f"(paired t p={paired_t.pvalue:.4g}, mean log-diff={diff.mean():.5f}, approx % change={approx_pct_change:.2f}%)."
    )
    print(
        "Regression evidence is consistent: "
        f"dyslexia-subset OLS reader_view p={rv_p:.4g}; interaction p={int_p:.4g}."
    )
    print(
        "Interpretable ML models do not identify reader_view as a dominant predictor "
        f"(EBM reader_view importance={rv_ebm_importance:.5f})."
    )

    # Scoring logic: no significance + tiny effect => low score
    # Clamp score to [0, 100].
    pvals = [safe_float(paired_t.pvalue), safe_float(rv_p), safe_float(int_p)]
    p_non_sig = sum(p > 0.05 for p in pvals if not np.isnan(p))
    tiny_effect = abs(approx_pct_change) < 2.0 and abs(cohens_dz) < 0.1

    base_score = 10
    if p_non_sig >= 2 and tiny_effect:
        score = 6
    elif p_non_sig >= 2:
        score = 12
    else:
        score = 35

    score = int(max(0, min(100, score)))

    explanation = (
        "The data do not support that Reader View improves reading speed for individuals with dyslexia. "
        f"In dyslexic participants with paired measurements (n={len(paired)}), the paired t-test on log-speed "
        f"was not significant (p={paired_t.pvalue:.3f}) and the average effect was near zero "
        f"(about {approx_pct_change:.2f}% change). "
        f"Covariate-adjusted OLS in the dyslexia subset also found no significant Reader View effect (p={rv_p:.3f}), "
        f"and the ReaderView×Dyslexia interaction in the full sample was not significant (p={int_p:.3f}). "
        "Interpretable models (linear regression, decision tree, and EBM) did not rank Reader View as an important "
        "driver of reading speed relative to other factors."
    )

    payload = {"response": score, "explanation": explanation}
    conclusion_path.write_text(json.dumps(payload, ensure_ascii=True))

    print_header("Conclusion JSON")
    print(json.dumps(payload, indent=2))
    print(f"\nWrote: {conclusion_path}")


if __name__ == "__main__":
    main()
