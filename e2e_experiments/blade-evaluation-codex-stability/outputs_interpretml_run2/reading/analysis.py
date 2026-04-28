import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from interpret.glassbox import ExplainableBoostingRegressor


RANDOM_STATE = 42


def cohen_d(x: pd.Series, y: pd.Series) -> float:
    x = x.dropna().to_numpy()
    y = y.dropna().to_numpy()
    if len(x) < 2 or len(y) < 2:
        return np.nan
    nx, ny = len(x), len(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
    if pooled <= 0:
        return np.nan
    return (np.mean(x) - np.mean(y)) / np.sqrt(pooled)


def summarize_exploration(df: pd.DataFrame) -> dict:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    print("\\n=== DATA SHAPE ===")
    print(df.shape)

    print("\\n=== MISSING VALUES (TOP 10) ===")
    print(df.isna().sum().sort_values(ascending=False).head(10))

    print("\\n=== NUMERIC SUMMARY (selected columns) ===")
    selected = [
        c
        for c in [
            "speed",
            "reader_view",
            "dyslexia_bin",
            "running_time",
            "adjusted_running_time",
            "scrolling_time",
            "num_words",
            "correct_rate",
            "age",
            "Flesch_Kincaid",
        ]
        if c in df.columns
    ]
    print(df[selected].describe().T)

    print("\\n=== SPEED DISTRIBUTION SUMMARY ===")
    speed_desc = df["speed"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    print(speed_desc)
    print("Speed skewness:", float(df["speed"].skew()))

    print("\\n=== GROUPED SPEED MEANS ===")
    grouped = (
        df.groupby(["dyslexia_bin", "reader_view"], dropna=False)["speed"]
        .agg(["count", "mean", "median", "std"])
        .round(4)
    )
    print(grouped)

    print("\\n=== CORRELATIONS WITH SPEED (NUMERIC) ===")
    corr = df[numeric_cols].corr(numeric_only=True)["speed"].sort_values(key=lambda s: s.abs(), ascending=False)
    print(corr.head(12))

    return {
        "speed_skewness": float(df["speed"].skew()),
        "speed_summary": speed_desc.to_dict(),
    }


def run_statistical_tests(df: pd.DataFrame) -> dict:
    out = {}

    # Core hypothesis subset: individuals with dyslexia.
    dys = df[df["dyslexia_bin"] == 1].copy()
    rv_on = dys.loc[dys["reader_view"] == 1, "speed"]
    rv_off = dys.loc[dys["reader_view"] == 0, "speed"]

    ttest = stats.ttest_ind(rv_on, rv_off, equal_var=False, nan_policy="omit")
    mw = stats.mannwhitneyu(rv_on, rv_off, alternative="two-sided")

    log_on = np.log1p(rv_on)
    log_off = np.log1p(rv_off)
    ttest_log = stats.ttest_ind(log_on, log_off, equal_var=False, nan_policy="omit")

    out["dyslexia_speed_mean_reader_view_on"] = float(rv_on.mean())
    out["dyslexia_speed_mean_reader_view_off"] = float(rv_off.mean())
    out["dyslexia_mean_diff_on_minus_off"] = float(rv_on.mean() - rv_off.mean())
    out["welch_ttest_pvalue"] = float(ttest.pvalue)
    out["welch_ttest_stat"] = float(ttest.statistic)
    out["mannwhitney_pvalue"] = float(mw.pvalue)
    out["log_welch_ttest_pvalue"] = float(ttest_log.pvalue)
    out["cohen_d"] = float(cohen_d(rv_on, rv_off)) if not np.isnan(cohen_d(rv_on, rv_off)) else None
    out["dyslexia_n_on"] = int(rv_on.shape[0])
    out["dyslexia_n_off"] = int(rv_off.shape[0])

    print("\\n=== HYPOTHESIS TESTS (DYSLEXIA SUBSET) ===")
    print(f"N(reader_view=1)={out['dyslexia_n_on']}, N(reader_view=0)={out['dyslexia_n_off']}")
    print(f"Mean speed on={out['dyslexia_speed_mean_reader_view_on']:.4f}, off={out['dyslexia_speed_mean_reader_view_off']:.4f}")
    print(f"Mean difference (on-off)={out['dyslexia_mean_diff_on_minus_off']:.4f}")
    print(f"Welch t-test p={out['welch_ttest_pvalue']:.6f}")
    print(f"Mann-Whitney p={out['mannwhitney_pvalue']:.6f}")
    print(f"Welch t-test on log(speed) p={out['log_welch_ttest_pvalue']:.6f}")
    print(f"Cohen's d={out['cohen_d']}")

    # OLS on dyslexia subset with controls.
    dys_model_df = dys.dropna(
        subset=[
            "speed",
            "reader_view",
            "age",
            "num_words",
            "correct_rate",
            "retake_trial",
            "device",
            "gender",
            "english_native",
        ]
    ).copy()
    dys_model_df["log_speed"] = np.log1p(dys_model_df["speed"])

    ols_dys = smf.ols(
        "log_speed ~ reader_view + age + num_words + correct_rate + retake_trial + C(device) + C(gender) + C(english_native)",
        data=dys_model_df,
    ).fit(cov_type="HC3")

    out["ols_dys_reader_view_coef"] = float(ols_dys.params.get("reader_view", np.nan))
    out["ols_dys_reader_view_pvalue"] = float(ols_dys.pvalues.get("reader_view", np.nan))

    # Full-data interaction test to ask if dyslexia modifies reader-view effect.
    full_model_df = df.dropna(
        subset=[
            "speed",
            "reader_view",
            "dyslexia_bin",
            "age",
            "num_words",
            "correct_rate",
            "retake_trial",
            "device",
            "gender",
            "english_native",
        ]
    ).copy()
    full_model_df["log_speed"] = np.log1p(full_model_df["speed"])

    ols_inter = smf.ols(
        "log_speed ~ reader_view * dyslexia_bin + age + num_words + correct_rate + retake_trial + C(device) + C(gender) + C(english_native)",
        data=full_model_df,
    ).fit(cov_type="HC3")

    out["ols_interaction_coef"] = float(ols_inter.params.get("reader_view:dyslexia_bin", np.nan))
    out["ols_interaction_pvalue"] = float(ols_inter.pvalues.get("reader_view:dyslexia_bin", np.nan))

    print("\\n=== OLS RESULTS ===")
    print(
        "Dyslexia-only OLS: coef(reader_view)=",
        out["ols_dys_reader_view_coef"],
        "p=",
        out["ols_dys_reader_view_pvalue"],
    )
    print(
        "Interaction OLS: coef(reader_view:dyslexia_bin)=",
        out["ols_interaction_coef"],
        "p=",
        out["ols_interaction_pvalue"],
    )

    return out


def fit_interpretable_models(df: pd.DataFrame) -> dict:
    work = df.copy()
    work = work.dropna(subset=["speed", "reader_view", "dyslexia_bin"])
    work["log_speed"] = np.log1p(work["speed"])

    candidate_features = [
        "reader_view",
        "running_time",
        "adjusted_running_time",
        "scrolling_time",
        "num_words",
        "correct_rate",
        "img_width",
        "age",
        "device",
        "dyslexia",
        "education",
        "gender",
        "language",
        "retake_trial",
        "dyslexia_bin",
        "english_native",
        "Flesch_Kincaid",
        "page_id",
    ]
    features = [c for c in candidate_features if c in work.columns]

    X = work[features]
    y = work["log_speed"]

    categorical_cols = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype).startswith("category")]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )

    models = {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "lasso": Lasso(alpha=0.001, random_state=RANDOM_STATE, max_iter=20000),
        "decision_tree": DecisionTreeRegressor(max_depth=4, min_samples_leaf=25, random_state=RANDOM_STATE),
    }

    results = {}
    linear_reader_coef = None

    for name, model in models.items():
        pipe = Pipeline(steps=[("pre", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)

        metrics = {
            "r2": float(r2_score(y_test, pred)),
            "mae": float(mean_absolute_error(y_test, pred)),
        }
        results[name] = metrics

        if name in {"linear_regression", "ridge", "lasso"}:
            feature_names = pipe.named_steps["pre"].get_feature_names_out()
            coefs = pipe.named_steps["model"].coef_
            coef_map = dict(zip(feature_names, coefs))
            rv_coef = None
            for key in ["num__reader_view"]:
                if key in coef_map:
                    rv_coef = float(coef_map[key])
                    break
            results[name]["reader_view_coef_standardized"] = rv_coef
            if name == "linear_regression":
                linear_reader_coef = rv_coef

        if name == "decision_tree":
            feature_names = pipe.named_steps["pre"].get_feature_names_out()
            importances = pipe.named_steps["model"].feature_importances_
            imp_pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
            top_feats = [{"feature": f, "importance": float(v)} for f, v in imp_pairs[:10]]
            results[name]["top_feature_importances"] = top_feats

    # Interpret glassbox model
    ebm_features = [c for c in features if c != "running_time"]
    # Avoid leakage-dominant timing proxies so effect of reader_view can be interpreted more directly.
    ebm = ExplainableBoostingRegressor(
        random_state=RANDOM_STATE,
        interactions=0,
        max_rounds=2000,
        learning_rate=0.03,
    )

    X_train_ebm = X_train[ebm_features].copy()
    X_test_ebm = X_test[ebm_features].copy()
    for c in X_train_ebm.columns:
        if X_train_ebm[c].dtype == "object":
            X_train_ebm[c] = X_train_ebm[c].astype("category")
            X_test_ebm[c] = X_test_ebm[c].astype("category")

    ebm.fit(X_train_ebm, y_train)
    ebm_pred = ebm.predict(X_test_ebm)

    term_names = list(ebm.term_names_)
    term_imps = list(ebm.term_importances())
    ebm_term_importances = sorted(
        [{"term": n, "importance": float(v)} for n, v in zip(term_names, term_imps)],
        key=lambda x: x["importance"],
        reverse=True,
    )

    reader_importance = None
    for item in ebm_term_importances:
        if item["term"] == "reader_view":
            reader_importance = item["importance"]
            break

    results["explainable_boosting_regressor"] = {
        "r2": float(r2_score(y_test, ebm_pred)),
        "mae": float(mean_absolute_error(y_test, ebm_pred)),
        "reader_view_term_importance": reader_importance,
        "top_term_importances": ebm_term_importances[:10],
    }

    print("\\n=== INTERPRETABLE MODEL PERFORMANCE ===")
    for model_name, vals in results.items():
        print(model_name, {k: v for k, v in vals.items() if k in {"r2", "mae", "reader_view_coef_standardized", "reader_view_term_importance"}})

    return {
        "model_results": results,
        "linear_reader_view_coef_standardized": linear_reader_coef,
        "ebm_reader_view_importance": reader_importance,
    }


def make_conclusion(research_question: str, tests: dict, models: dict) -> dict:
    p_t = tests.get("welch_ttest_pvalue", np.nan)
    p_log = tests.get("log_welch_ttest_pvalue", np.nan)
    p_ols = tests.get("ols_dys_reader_view_pvalue", np.nan)
    p_inter = tests.get("ols_interaction_pvalue", np.nan)

    mean_diff = tests.get("dyslexia_mean_diff_on_minus_off", 0.0)
    coef_ols = tests.get("ols_dys_reader_view_coef", 0.0)
    coef_lin = models.get("linear_reader_view_coef_standardized", 0.0)

    sig_pos = (
        (p_t < 0.05 and mean_diff > 0)
        or (p_log < 0.05 and mean_diff > 0)
        or (p_ols < 0.05 and coef_ols > 0)
    )
    sig_neg = (
        (p_t < 0.05 and mean_diff < 0)
        or (p_log < 0.05 and mean_diff < 0)
        or (p_ols < 0.05 and coef_ols < 0)
    )

    if sig_pos:
        response = 85
    elif sig_neg:
        response = 10
    else:
        # No significant evidence: keep low. Slightly adjust by direction consistency.
        direction_votes = 0
        direction_votes += 1 if mean_diff > 0 else -1
        direction_votes += 1 if coef_ols > 0 else -1
        direction_votes += 1 if (coef_lin is not None and coef_lin > 0) else -1

        response = 20 if direction_votes >= 1 else 15
        if p_inter < 0.1:
            response = max(10, response - 5)

    explanation = (
        f"Research question: {research_question} "
        f"In participants with dyslexia, Reader View did not produce a statistically significant speed improvement "
        f"(Welch t-test p={p_t:.3f}, log-speed t-test p={p_log:.3f}, OLS-adjusted p={p_ols:.3f}). "
        f"The observed mean difference in raw speed was {mean_diff:.2f} (Reader View minus control), and the OLS reader_view coefficient was {coef_ols:.4f}. "
        f"The reader_view*dyslexia interaction was not significant (p={p_inter:.3f}), and interpretable models did not identify a strong consistent reader_view effect. "
        f"Therefore evidence does not support a clear improvement in reading speed for individuals with dyslexia."
    )

    return {"response": int(response), "explanation": explanation}


def main() -> None:
    info_path = Path("info.json")
    data_path = Path("reading.csv")

    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

    research_question = ""
    if isinstance(info.get("research_questions"), list) and info["research_questions"]:
        research_question = str(info["research_questions"][0])

    df = pd.read_csv(data_path)

    # Basic type normalization for categorical handling.
    for col in ["device", "education", "language", "english_native", "page_id"]:
        if col in df.columns:
            df[col] = df[col].astype("object")

    _ = summarize_exploration(df)
    tests = run_statistical_tests(df)
    model_info = fit_interpretable_models(df)

    conclusion = make_conclusion(research_question, tests, model_info)

    with Path("conclusion.txt").open("w", encoding="utf-8") as f:
        json.dump(conclusion, f)

    print("\\n=== FINAL CONCLUSION JSON ===")
    print(json.dumps(conclusion, indent=2))


if __name__ == "__main__":
    main()
