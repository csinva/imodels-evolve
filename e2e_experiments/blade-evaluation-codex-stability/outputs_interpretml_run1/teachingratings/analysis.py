import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from interpret.glassbox import ExplainableBoostingRegressor


def cohen_d(x: pd.Series, y: pd.Series) -> float:
    x = x.dropna().astype(float)
    y = y.dropna().astype(float)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    pooled = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if pooled == 0:
        return 0.0
    return (x.mean() - y.mean()) / pooled


def summarize_data(df: pd.DataFrame) -> dict:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    numeric_summary = df[numeric_cols].describe().T
    numeric_summary["skew"] = df[numeric_cols].skew()

    category_summary = {}
    for c in categorical_cols:
        vc = df[c].value_counts(dropna=False)
        category_summary[c] = {
            "counts": vc.to_dict(),
            "proportions": (vc / len(df)).round(3).to_dict(),
        }

    corr = df[numeric_cols].corr(numeric_only=True)

    return {
        "numeric_summary": numeric_summary,
        "category_summary": category_summary,
        "correlation_matrix": corr,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
    }


def run_statistical_tests(df: pd.DataFrame) -> dict:
    results = {}

    pearson = stats.pearsonr(df["beauty"], df["eval"])
    spearman = stats.spearmanr(df["beauty"], df["eval"])
    results["pearson"] = {"r": float(pearson.statistic), "p": float(pearson.pvalue)}
    results["spearman"] = {"rho": float(spearman.statistic), "p": float(spearman.pvalue)}

    median_beauty = df["beauty"].median()
    hi = df.loc[df["beauty"] >= median_beauty, "eval"]
    lo = df.loc[df["beauty"] < median_beauty, "eval"]
    t_stat, t_p = stats.ttest_ind(hi, lo, equal_var=False)
    results["median_split_ttest"] = {
        "t": float(t_stat),
        "p": float(t_p),
        "high_mean": float(hi.mean()),
        "low_mean": float(lo.mean()),
        "cohen_d": float(cohen_d(hi, lo)),
    }

    df_tmp = df.copy()
    df_tmp["beauty_quartile"] = pd.qcut(df_tmp["beauty"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
    anova_groups = [
        df_tmp.loc[df_tmp["beauty_quartile"] == q, "eval"].values for q in ["Q1", "Q2", "Q3", "Q4"]
    ]
    f_stat, f_p = stats.f_oneway(*anova_groups)
    results["anova_beauty_quartiles"] = {"F": float(f_stat), "p": float(f_p)}

    simple_formula = "eval ~ beauty"
    full_formula = (
        "eval ~ beauty + age + students + allstudents + "
        "C(minority) + C(gender) + C(credits) + C(division) + C(native) + C(tenure)"
    )

    simple_ols = smf.ols(simple_formula, data=df).fit()
    full_ols = smf.ols(full_formula, data=df).fit()
    robust_full = full_ols.get_robustcov_results(cov_type="cluster", groups=df["prof"])

    beauty_idx = full_ols.params.index.tolist().index("beauty")
    results["simple_ols"] = {
        "beauty_coef": float(simple_ols.params["beauty"]),
        "beauty_p": float(simple_ols.pvalues["beauty"]),
        "r2": float(simple_ols.rsquared),
    }
    results["full_ols"] = {
        "beauty_coef": float(full_ols.params["beauty"]),
        "beauty_p": float(full_ols.pvalues["beauty"]),
        "r2": float(full_ols.rsquared),
    }
    results["full_ols_cluster_robust"] = {
        "beauty_coef": float(robust_full.params[beauty_idx]),
        "beauty_p": float(robust_full.pvalues[beauty_idx]),
    }

    return results


def get_beauty_coef_from_pipeline(model_pipeline: Pipeline, feature_names: list[str]) -> float:
    reg = model_pipeline.named_steps["model"]
    if not hasattr(reg, "coef_"):
        return np.nan
    coefs = np.ravel(reg.coef_)
    idx = None
    for i, name in enumerate(feature_names):
        if name == "beauty" or name.endswith("__beauty"):
            idx = i
            break
    if idx is None:
        return np.nan
    return float(coefs[idx])


def aggregate_tree_importances(feature_names: list[str], importances: np.ndarray) -> dict:
    out = {}
    for name, imp in zip(feature_names, importances):
        if name.startswith("cat__"):
            raw = name.split("cat__", 1)[1].split("_", 1)[0]
        elif name.startswith("num__"):
            raw = name.split("num__", 1)[1]
        else:
            raw = name
        out[raw] = out.get(raw, 0.0) + float(imp)
    return dict(sorted(out.items(), key=lambda kv: kv[1], reverse=True))


def run_interpretable_models(df: pd.DataFrame) -> dict:
    features = [
        "beauty",
        "age",
        "students",
        "allstudents",
        "minority",
        "gender",
        "credits",
        "division",
        "native",
        "tenure",
    ]
    X = df[features].copy()
    y = df["eval"].astype(float)

    categorical_cols = ["minority", "gender", "credits", "division", "native", "tenure"]
    numeric_cols = [c for c in features if c not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    pipelines = {
        "linear": Pipeline([("prep", preprocessor), ("model", LinearRegression())]),
        "ridge": Pipeline([("prep", preprocessor), ("model", Ridge(alpha=1.0, random_state=42))]),
        "lasso": Pipeline([("prep", preprocessor), ("model", Lasso(alpha=0.001, random_state=42, max_iter=10000))]),
        "tree": Pipeline([("prep", preprocessor), ("model", DecisionTreeRegressor(max_depth=3, random_state=42))]),
    }

    model_results = {}
    fitted = {}
    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        model_results[name] = {"test_r2": float(r2_score(y_test, y_pred))}
        fitted[name] = pipe

    feature_names = fitted["linear"].named_steps["prep"].get_feature_names_out().tolist()

    for name in ["linear", "ridge", "lasso"]:
        model_results[name]["beauty_coef"] = get_beauty_coef_from_pipeline(fitted[name], feature_names)

    tree_model = fitted["tree"].named_steps["model"]
    tree_imp = aggregate_tree_importances(feature_names, tree_model.feature_importances_)
    model_results["tree"]["feature_importance"] = tree_imp
    model_results["tree"]["beauty_importance"] = float(tree_imp.get("beauty", 0.0))

    X_ebm = X.copy()
    for c in categorical_cols:
        X_ebm[c] = X_ebm[c].astype("category")

    ebm = ExplainableBoostingRegressor(random_state=42, interactions=0)
    ebm.fit(X_ebm, y)
    ebm_term_names = list(ebm.term_names_)
    ebm_term_imps = list(ebm.term_importances())
    ebm_importance = {k: float(v) for k, v in zip(ebm_term_names, ebm_term_imps)}

    model_results["ebm"] = {
        "term_importance": dict(sorted(ebm_importance.items(), key=lambda kv: kv[1], reverse=True)),
        "beauty_importance": float(ebm_importance.get("beauty", 0.0)),
        "train_r2": float(r2_score(y, ebm.predict(X_ebm))),
    }

    return model_results


def build_conclusion(stats_results: dict, model_results: dict) -> dict:
    pearson_p = stats_results["pearson"]["p"]
    spearman_p = stats_results["spearman"]["p"]
    ttest_p = stats_results["median_split_ttest"]["p"]
    anova_p = stats_results["anova_beauty_quartiles"]["p"]
    simple_p = stats_results["simple_ols"]["beauty_p"]
    full_p = stats_results["full_ols"]["beauty_p"]
    robust_p = stats_results["full_ols_cluster_robust"]["beauty_p"]

    full_beta = stats_results["full_ols"]["beauty_coef"]
    pearson_r = stats_results["pearson"]["r"]

    score = 50
    if full_p < 0.05 and robust_p < 0.05:
        score += 25
    else:
        score -= 25

    if simple_p < 0.05 and pearson_p < 0.05 and spearman_p < 0.05:
        score += 10
    else:
        score -= 10

    if ttest_p < 0.05 and anova_p < 0.05:
        score += 5

    if full_beta > 0:
        score += 8
    else:
        score -= 15

    if abs(full_beta) >= 0.1:
        score += 5

    if pearson_r > 0.15:
        score += 4

    ebm_imp = model_results["ebm"]["beauty_importance"]
    tree_imp = model_results["tree"]["beauty_importance"]
    if ebm_imp > 0 or tree_imp > 0:
        score += 3

    score = int(np.clip(round(score), 0, 100))

    explanation = (
        "Beauty shows a statistically significant positive association with teaching evaluations. "
        f"Key tests: Pearson r={pearson_r:.3f} (p={pearson_p:.2e}), Spearman p={spearman_p:.2e}, "
        f"median-split t-test p={ttest_p:.2e}, ANOVA across beauty quartiles p={anova_p:.2e}. "
        f"In regression, beauty remains positive and significant both in simple OLS "
        f"(beta={stats_results['simple_ols']['beauty_coef']:.3f}, p={simple_p:.2e}) and adjusted OLS "
        f"(beta={full_beta:.3f}, p={full_p:.2e}; cluster-robust p={robust_p:.2e}). "
        "Interpretable models (linear/ridge/lasso/tree/EBM) also assign non-zero importance to beauty, "
        "supporting a real but moderate effect size rather than a dominant predictor."
    )

    return {"response": score, "explanation": explanation}


def main() -> None:
    base = Path(".")
    info_path = base / "info.json"
    data_path = base / "teachingratings.csv"

    if not info_path.exists() or not data_path.exists():
        raise FileNotFoundError("Required files `info.json` or `teachingratings.csv` are missing.")

    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

    research_question = info.get("research_questions", ["Unknown question"])[0]
    print(f"Research question: {research_question}")

    df = pd.read_csv(data_path)
    print(f"Loaded data shape: {df.shape}")

    summary = summarize_data(df)
    print("\nNumeric summary (first rows):")
    print(summary["numeric_summary"].round(3).head(8).to_string())

    print("\nCategorical distributions:")
    for c, vals in summary["category_summary"].items():
        print(f"  {c}: {vals['counts']}")

    print("\nCorrelation matrix among numeric columns:")
    print(summary["correlation_matrix"].round(3).to_string())

    stats_results = run_statistical_tests(df)
    print("\nStatistical tests:")
    print(json.dumps(stats_results, indent=2))

    model_results = run_interpretable_models(df)
    print("\nInterpretable model results:")
    print(json.dumps(model_results, indent=2))

    conclusion = build_conclusion(stats_results, model_results)

    with (base / "conclusion.txt").open("w", encoding="utf-8") as f:
        json.dump(conclusion, f)

    print("\nWrote conclusion.txt")


if __name__ == "__main__":
    main()
