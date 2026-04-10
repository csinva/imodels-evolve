import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from interp_models import HingeEBMRegressor, SmartAdditiveRegressor

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx = len(x)
    ny = len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
    if pooled <= 0:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / np.sqrt(pooled))


def permutation_r2_drop(model, x_test: np.ndarray, y_test: np.ndarray, col_idx: int, random_state: int = 42) -> float:
    rng = np.random.default_rng(random_state)
    base = r2_score(y_test, model.predict(x_test))
    x_perm = x_test.copy()
    x_perm[:, col_idx] = rng.permutation(x_perm[:, col_idx])
    perm = r2_score(y_test, model.predict(x_perm))
    return float(base - perm)


def delta_effect(model, x_ref: np.ndarray, col_idx: int, delta: float) -> float:
    x_hi = x_ref.copy()
    x_hi[:, col_idx] = x_hi[:, col_idx] + delta
    pred_hi = model.predict(x_hi)
    pred_lo = model.predict(x_ref)
    return float(np.mean(pred_hi - pred_lo))


def main() -> None:
    info_path = Path("info.json")
    data_path = Path("teachingratings.csv")

    info = json.loads(info_path.read_text())
    question = info["research_questions"][0]

    df = pd.read_csv(data_path)

    section("Research Question")
    print(question)

    section("Data Overview")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print(f"Columns: {list(df.columns)}")
    print("Missing values per column:")
    print(df.isna().sum())

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    section("Summary Statistics")
    print(df[numeric_cols].describe().T)

    section("Selected Distributions")
    for col in ["beauty", "eval", "age", "students", "allstudents"]:
        q = df[col].quantile([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
        print(f"\n{col} quantiles:")
        print(q)
        counts, edges = np.histogram(df[col], bins=8)
        print(f"{col} histogram counts: {counts.tolist()}")
        print(f"{col} histogram edges: {[round(v, 3) for v in edges.tolist()]}")

    section("Categorical Distributions")
    for col in categorical_cols:
        print(f"\n{col} value counts:")
        print(df[col].value_counts(dropna=False))

    section("Correlations")
    corr_eval = df[numeric_cols].corr(numeric_only=True)["eval"].sort_values(ascending=False)
    print("Correlation with eval:")
    print(corr_eval)

    section("Hypothesis Tests: Beauty vs Teaching Evaluations")
    pearson_r, pearson_p = stats.pearsonr(df["beauty"], df["eval"])
    spearman_rho, spearman_p = stats.spearmanr(df["beauty"], df["eval"])
    print(f"Pearson r = {pearson_r:.4f}, p = {pearson_p:.3g}")
    print(f"Spearman rho = {spearman_rho:.4f}, p = {spearman_p:.3g}")

    ols_simple = smf.ols("eval ~ beauty", data=df).fit()
    beauty_coef_simple = float(ols_simple.params["beauty"])
    beauty_p_simple = float(ols_simple.pvalues["beauty"])
    ci_simple = ols_simple.conf_int().loc["beauty"].tolist()
    print("\nSimple OLS: eval ~ beauty")
    print(ols_simple.summary().tables[1])

    formula_adj = (
        "eval ~ beauty + age + students + allstudents + "
        "C(minority) + C(gender) + C(credits) + C(division) + C(native) + C(tenure)"
    )
    ols_adj = smf.ols(formula_adj, data=df).fit(cov_type="HC3")
    beauty_coef_adj = float(ols_adj.params["beauty"])
    beauty_p_adj = float(ols_adj.pvalues["beauty"])
    ci_adj = ols_adj.conf_int().loc["beauty"].tolist()
    print("\nAdjusted OLS with robust (HC3) SE")
    print(ols_adj.summary().tables[1])

    df_tests = df.copy()
    df_tests["beauty_q"] = pd.qcut(df_tests["beauty"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
    anova_model = smf.ols("eval ~ C(beauty_q)", data=df_tests).fit()
    anova_table = sm.stats.anova_lm(anova_model, typ=2)
    anova_p = float(anova_table.loc["C(beauty_q)", "PR(>F)"])
    print("\nANOVA across beauty quartiles")
    print(anova_table)

    low = df_tests.loc[df_tests["beauty_q"] == "Q1", "eval"].to_numpy()
    high = df_tests.loc[df_tests["beauty_q"] == "Q4", "eval"].to_numpy()
    t_res = stats.ttest_ind(high, low, equal_var=False)
    mean_diff_top_bottom = float(np.mean(high) - np.mean(low))
    d_top_bottom = cohen_d(high, low)
    print("\nWelch t-test: Top beauty quartile vs bottom quartile")
    print(f"t = {t_res.statistic:.4f}, p = {t_res.pvalue:.3g}")
    print(f"Mean difference (Q4 - Q1) = {mean_diff_top_bottom:.4f}")
    print(f"Cohen's d (Q4 vs Q1) = {d_top_bottom:.4f}")

    section("Modeling Setup")
    y = df["eval"].to_numpy(dtype=float)
    x_raw = df.drop(columns=["eval", "rownames", "prof"])
    x_df = pd.get_dummies(x_raw, drop_first=True)
    feature_names = ["beauty"] + [c for c in x_df.columns if c != "beauty"]
    x_df = x_df[feature_names]
    x = x_df.to_numpy(dtype=float)
    beauty_idx = feature_names.index("beauty")

    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(indices, test_size=0.25, random_state=42)
    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    beauty_sd = float(df["beauty"].std())

    print(f"Encoded features ({len(feature_names)}): {feature_names}")
    print("Feature map for custom model strings:")
    print(", ".join([f"x{i}={name}" for i, name in enumerate(feature_names)]))

    section("Standard Models")
    lin = LinearRegression()
    lin.fit(x_train, y_train)
    lin_r2 = r2_score(y_test, lin.predict(x_test))
    lin_beauty_coef = float(lin.coef_[beauty_idx])
    print(f"LinearRegression test R^2: {lin_r2:.4f}")
    print(f"LinearRegression beauty coefficient: {lin_beauty_coef:.4f}")

    lasso_pipe = make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=42, max_iter=20000))
    lasso_pipe.fit(x_train, y_train)
    lasso_r2 = r2_score(y_test, lasso_pipe.predict(x_test))
    lasso_model = lasso_pipe.named_steps["lassocv"]
    lasso_beauty_coef = float(lasso_model.coef_[beauty_idx])
    print(f"LassoCV (scaled) test R^2: {lasso_r2:.4f}")
    print(f"LassoCV beauty coefficient (scaled space): {lasso_beauty_coef:.4f}")

    tree = DecisionTreeRegressor(max_depth=3, random_state=42)
    tree.fit(x_train, y_train)
    tree_r2 = r2_score(y_test, tree.predict(x_test))
    tree_beauty_imp = float(tree.feature_importances_[beauty_idx])
    print(f"DecisionTreeRegressor(max_depth=3) test R^2: {tree_r2:.4f}")
    print(f"DecisionTree beauty feature importance: {tree_beauty_imp:.4f}")

    section("Custom Interpretable Models (Full Features)")
    smart_full = SmartAdditiveRegressor(n_rounds=300, learning_rate=0.05, min_samples_leaf=8)
    smart_full.fit(x_train, y_train)
    smart_full_r2 = r2_score(y_test, smart_full.predict(x_test))
    smart_full_perm = permutation_r2_drop(smart_full, x_test, y_test, beauty_idx)
    smart_full_delta = delta_effect(smart_full, x_test, beauty_idx, beauty_sd)
    smart_full_imp = float(smart_full.feature_importances_[beauty_idx])
    if beauty_idx in smart_full.linear_approx_:
        smart_slope, _, smart_r2_linear = smart_full.linear_approx_[beauty_idx]
    else:
        smart_slope, smart_r2_linear = 0.0, 0.0

    print(f"SmartAdditiveRegressor test R^2: {smart_full_r2:.4f}")
    print(f"Smart beauty importance(range): {smart_full_imp:.4f}")
    print(f"Smart beauty linear approx slope: {smart_slope:.4f}, linear-R^2: {smart_r2_linear:.4f}")
    print(f"Smart beauty +1 SD avg effect on eval: {smart_full_delta:.4f}")
    print(f"Smart beauty permutation R^2 drop: {smart_full_perm:.4f}")
    print("\nSmartAdditiveRegressor human-readable model:")
    print(smart_full)

    hinge_full = HingeEBMRegressor(
        n_knots=3,
        max_input_features=min(15, x_train.shape[1]),
        ebm_outer_bags=4,
        ebm_max_rounds=400,
    )
    hinge_full.fit(x_train, y_train)
    hinge_full_r2 = r2_score(y_test, hinge_full.predict(x_test))
    hinge_full_perm = permutation_r2_drop(hinge_full, x_test, y_test, beauty_idx)
    hinge_full_delta = delta_effect(hinge_full, x_test, beauty_idx, beauty_sd)

    print(f"HingeEBMRegressor test R^2: {hinge_full_r2:.4f}")
    print(f"Hinge beauty +1 SD avg effect on eval: {hinge_full_delta:.4f}")
    print(f"Hinge beauty permutation R^2 drop: {hinge_full_perm:.4f}")
    print("\nHingeEBMRegressor human-readable model:")
    print(hinge_full)

    section("Custom Interpretable Models (Beauty-Only)")
    x_beauty = df[["beauty"]].to_numpy(dtype=float)
    xb_train, xb_test = x_beauty[train_idx], x_beauty[test_idx]

    smart_beauty = SmartAdditiveRegressor(n_rounds=300, learning_rate=0.05, min_samples_leaf=8)
    smart_beauty.fit(xb_train, y_train)
    smart_beauty_r2 = r2_score(y_test, smart_beauty.predict(xb_test))
    smart_beauty_delta = delta_effect(smart_beauty, xb_test, 0, beauty_sd)
    print(f"SmartAdditive (beauty-only) test R^2: {smart_beauty_r2:.4f}")
    print(f"SmartAdditive (beauty-only) +1 SD avg effect: {smart_beauty_delta:.4f}")
    print("SmartAdditive (beauty-only) model:")
    print(smart_beauty)

    hinge_beauty = HingeEBMRegressor(n_knots=3, max_input_features=1, ebm_outer_bags=4, ebm_max_rounds=400)
    hinge_beauty.fit(xb_train, y_train)
    hinge_beauty_r2 = r2_score(y_test, hinge_beauty.predict(xb_test))
    hinge_beauty_delta = delta_effect(hinge_beauty, xb_test, 0, beauty_sd)
    print(f"HingeEBM (beauty-only) test R^2: {hinge_beauty_r2:.4f}")
    print(f"HingeEBM (beauty-only) +1 SD avg effect: {hinge_beauty_delta:.4f}")
    print("HingeEBM (beauty-only) model:")
    print(hinge_beauty)

    section("Evidence Synthesis")
    score = 50.0

    if beauty_coef_adj > 0:
        score += 12
    else:
        score -= 25

    if beauty_p_adj < 0.001:
        score += 22
    elif beauty_p_adj < 0.01:
        score += 16
    elif beauty_p_adj < 0.05:
        score += 10
    else:
        score -= 15

    if pearson_p < 0.05 and pearson_r > 0:
        score += 12
    elif pearson_p < 0.05:
        score -= 6
    else:
        score -= 10

    if anova_p < 0.05:
        score += 8
    else:
        score -= 6

    if t_res.pvalue < 0.05 and mean_diff_top_bottom > 0:
        score += 10
    elif t_res.pvalue < 0.05:
        score -= 8
    else:
        score -= 8

    if lin_beauty_coef > 0:
        score += 6
    else:
        score -= 6

    if lasso_beauty_coef > 0:
        score += 4
    else:
        score -= 4

    if smart_full_delta > 0:
        score += 6
    else:
        score -= 6

    if hinge_full_delta > 0:
        score += 6
    else:
        score -= 6

    if smart_full_perm > 0:
        score += 4
    else:
        score -= 4

    if hinge_full_perm > 0:
        score += 4
    else:
        score -= 4

    score = int(np.clip(round(score), 0, 100))

    explanation = (
        f"Beauty shows a statistically significant positive relationship with teaching evaluations: "
        f"Pearson r={pearson_r:.3f} (p={pearson_p:.2e}), simple OLS beauty coef={beauty_coef_simple:.3f} "
        f"(95% CI [{ci_simple[0]:.3f}, {ci_simple[1]:.3f}], p={beauty_p_simple:.2e}), and adjusted OLS beauty "
        f"coef={beauty_coef_adj:.3f} (95% CI [{ci_adj[0]:.3f}, {ci_adj[1]:.3f}], p={beauty_p_adj:.2e}). "
        f"Group tests agree (ANOVA across beauty quartiles p={anova_p:.2e}; top-bottom quartile eval difference="
        f"{mean_diff_top_bottom:.3f}, Welch p={t_res.pvalue:.2e}, d={d_top_bottom:.3f}). "
        f"Custom interpretable models also assign a positive beauty effect (Smart +1SD effect={smart_full_delta:.3f}, "
        f"Hinge +1SD effect={hinge_full_delta:.3f}) with positive permutation importance drops "
        f"(Smart {smart_full_perm:.3f}, Hinge {hinge_full_perm:.3f}), consistent with standard linear models."
    )

    conclusion_obj = {"response": score, "explanation": explanation}
    Path("conclusion.txt").write_text(json.dumps(conclusion_obj))

    print(f"Final Likert response (0-100): {score}")
    print("Wrote conclusion.txt")


if __name__ == "__main__":
    main()
