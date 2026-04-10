import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from statsmodels.stats.proportion import proportions_ztest

from interp_models import HingeEBMRegressor, SmartAdditiveRegressor

warnings.filterwarnings("ignore")
RANDOM_SEED = 42


def section(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def avg_toggle_effect(model, X: np.ndarray, feature_idx: int) -> float:
    x0 = X.copy()
    x1 = X.copy()
    x0[:, feature_idx] = 0.0
    x1[:, feature_idx] = 1.0
    return float(np.mean(model.predict(x1) - model.predict(x0)))


def main() -> None:
    # 1) Load data
    section("Load Dataset")
    df = pd.read_csv("mortgage.csv")
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    print(f"Rows: {len(df)}, Columns: {df.shape[1]}")
    print("Columns:", list(df.columns))

    # 2) Feature setup
    target = "deny"
    key_feature = "female"
    controls = [
        "black",
        "housing_expense_ratio",
        "self_employed",
        "married",
        "mortgage_credit",
        "consumer_credit",
        "bad_history",
        "PI_ratio",
        "loan_to_value",
    ]
    feature_cols = [key_feature] + controls

    # 3) Data exploration
    section("Missingness + Summary Statistics")
    missing = df[feature_cols + [target]].isna().sum()
    print("Missing values by column:")
    print(missing.to_string())

    print("\nSummary stats (numeric):")
    print(df[feature_cols + [target]].describe().T.to_string())

    section("Distribution Checks")
    for col in ["female", "black", "self_employed", "married", "bad_history", "deny"]:
        vc = df[col].value_counts(dropna=False).sort_index()
        print(f"\n{col} value counts:")
        print(vc.to_string())

    for col in ["housing_expense_ratio", "PI_ratio", "loan_to_value"]:
        vals = df[col].dropna().values
        hist, edges = np.histogram(vals, bins=10)
        print(f"\nHistogram bins for {col}:")
        for i in range(len(hist)):
            print(f"  [{edges[i]:.3f}, {edges[i+1]:.3f}): {hist[i]}")

    section("Correlation Structure")
    corr = df[feature_cols + [target]].corr(numeric_only=True)
    print("Correlations with deny:")
    print(corr[target].sort_values(ascending=False).to_string())

    # 4) Statistical tests for relationship female -> deny
    section("Statistical Tests: Female vs Mortgage Denial")
    infer_df = df[feature_cols + [target]].dropna().copy()
    male = infer_df.loc[infer_df[key_feature] == 0, target]
    female = infer_df.loc[infer_df[key_feature] == 1, target]

    male_rate = float(male.mean())
    female_rate = float(female.mean())
    raw_diff = female_rate - male_rate

    print(f"Complete-case sample size: {len(infer_df)}")
    print(f"Raw denial rate (male):   {male_rate:.4f}")
    print(f"Raw denial rate (female): {female_rate:.4f}")
    print(f"Raw difference (female - male): {raw_diff:.4f}")

    t_stat, t_p = stats.ttest_ind(female, male, equal_var=False)
    print(f"Welch t-test p-value: {t_p:.6f} (t={t_stat:.4f})")

    contingency = pd.crosstab(infer_df[key_feature], infer_df[target])
    chi2, chi2_p, _, _ = stats.chi2_contingency(contingency)
    print(f"Chi-square test p-value: {chi2_p:.6f} (chi2={chi2:.4f})")

    count = np.array([female.sum(), male.sum()])
    nobs = np.array([len(female), len(male)])
    z_stat, z_p = proportions_ztest(count=count, nobs=nobs)
    print(f"Two-proportion z-test p-value: {z_p:.6f} (z={z_stat:.4f})")

    r_pb, r_pb_p = stats.pointbiserialr(infer_df[key_feature], infer_df[target])
    print(f"Point-biserial correlation female~deny: r={r_pb:.4f}, p={r_pb_p:.6f}")

    X_inf = sm.add_constant(infer_df[feature_cols])
    y_inf = infer_df[target]

    logit = sm.Logit(y_inf, X_inf).fit(disp=0, maxiter=200)
    female_coef = float(logit.params[key_feature])
    female_p = float(logit.pvalues[key_feature])
    female_or = float(np.exp(female_coef))
    ci_low, ci_high = np.exp(logit.conf_int().loc[key_feature])
    print("\nAdjusted logistic regression (deny ~ female + controls):")
    print(f"female coefficient: {female_coef:.4f}")
    print(f"female odds ratio: {female_or:.4f}")
    print(f"female OR 95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"female p-value: {female_p:.6f}")

    ols = sm.OLS(y_inf, X_inf).fit(cov_type="HC3")
    female_ols_coef = float(ols.params[key_feature])
    female_ols_p = float(ols.pvalues[key_feature])
    print("\nAdjusted OLS linear probability model (HC3 robust SE):")
    print(f"female coefficient: {female_ols_coef:.4f}, p-value: {female_ols_p:.6f}")

    anova_model = smf.ols("deny ~ C(female) + C(black) + C(female):C(black)", data=infer_df).fit()
    anova_table = sm.stats.anova_lm(anova_model, typ=2)
    anova_female_p = float(anova_table.loc["C(female)", "PR(>F)"])
    anova_inter_p = float(anova_table.loc["C(female):C(black)", "PR(>F)"])
    print("\nTwo-way ANOVA (deny by female, black, interaction):")
    print(anova_table.to_string())

    # 5) Predictive interpretable modeling with custom tools + standard baselines
    section("Modeling Setup")
    X_all = df[feature_cols].copy()
    y_all = df[target].values.astype(float)

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_all)

    idx_female = feature_cols.index("female")
    print("Feature index map used by custom model strings:")
    for i, name in enumerate(feature_cols):
        print(f"  x{i} -> {name}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_imp,
        y_all,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=y_all,
    )

    section("Custom Interpretable Model: SmartAdditiveRegressor")
    smart = SmartAdditiveRegressor(n_rounds=300, learning_rate=0.08, min_samples_leaf=10)
    smart.fit(X_train, y_train)
    smart_pred = smart.predict(X_test)
    smart_r2 = r2_score(y_test, smart_pred)
    smart_rmse = float(np.sqrt(mean_squared_error(y_test, smart_pred)))
    smart_female_effect = avg_toggle_effect(smart, X_test, idx_female)

    print(f"Test R^2: {smart_r2:.4f}")
    print(f"Test RMSE: {smart_rmse:.4f}")
    print(f"Average denial change when toggling female 0->1: {smart_female_effect:.4f}")
    print("Model interpretation:")
    print(str(smart))

    section("Custom Interpretable Model: HingeEBMRegressor")
    hinge = None
    hinge_female_effect = np.nan
    hinge_r2 = np.nan
    hinge_rmse = np.nan
    try:
        hinge = HingeEBMRegressor(n_knots=3, max_input_features=15, ebm_outer_bags=3, ebm_max_rounds=1000)
        hinge.fit(X_train, y_train)
        hinge_pred = hinge.predict(X_test)
        hinge_r2 = r2_score(y_test, hinge_pred)
        hinge_rmse = float(np.sqrt(mean_squared_error(y_test, hinge_pred)))
        hinge_female_effect = avg_toggle_effect(hinge, X_test, idx_female)

        print(f"Test R^2: {hinge_r2:.4f}")
        print(f"Test RMSE: {hinge_rmse:.4f}")
        print(f"Average denial change when toggling female 0->1: {hinge_female_effect:.4f}")
        print("Model interpretation:")
        print(str(hinge))
    except Exception as exc:
        print(f"HingeEBMRegressor unavailable or failed: {exc}")

    section("Standard Baseline Models")
    lin = LinearRegression()
    lin.fit(X_train, y_train)
    lin_pred = lin.predict(X_test)
    lin_r2 = r2_score(y_test, lin_pred)
    lin_rmse = float(np.sqrt(mean_squared_error(y_test, lin_pred)))
    lin_female_coef = float(lin.coef_[idx_female])

    lasso = LassoCV(cv=5, random_state=RANDOM_SEED, max_iter=10000)
    lasso.fit(X_train, y_train)
    lasso_pred = lasso.predict(X_test)
    lasso_r2 = r2_score(y_test, lasso_pred)
    lasso_rmse = float(np.sqrt(mean_squared_error(y_test, lasso_pred)))
    lasso_female_coef = float(lasso.coef_[idx_female])

    tree = DecisionTreeRegressor(max_depth=3, random_state=RANDOM_SEED)
    tree.fit(X_train, y_train)
    tree_pred = tree.predict(X_test)
    tree_r2 = r2_score(y_test, tree_pred)
    tree_rmse = float(np.sqrt(mean_squared_error(y_test, tree_pred)))

    print(f"LinearRegression  -> R^2: {lin_r2:.4f}, RMSE: {lin_rmse:.4f}, female coef: {lin_female_coef:.4f}")
    print(f"LassoCV           -> R^2: {lasso_r2:.4f}, RMSE: {lasso_rmse:.4f}, female coef: {lasso_female_coef:.4f}")
    print(f"DecisionTreeReg   -> R^2: {tree_r2:.4f}, RMSE: {tree_rmse:.4f}")

    # 6) Synthesize evidence -> Likert score
    section("Synthesis")
    # Base score from strongest inferential test (adjusted logistic p-value)
    if female_p < 0.001:
        score = 90
    elif female_p < 0.01:
        score = 82
    elif female_p < 0.05:
        score = 72
    elif female_p < 0.10:
        score = 58
    else:
        score = 25

    # Penalize if unadjusted relationship is not significant
    if z_p > 0.10 and chi2_p > 0.10:
        score -= 10

    # Direction consistency across adjusted models
    direction_votes = [
        np.sign(female_coef),
        np.sign(female_ols_coef),
        np.sign(lin_female_coef),
        np.sign(lasso_female_coef),
    ]
    if np.isfinite(hinge_female_effect) and abs(hinge_female_effect) > 1e-6:
        direction_votes.append(np.sign(hinge_female_effect))

    nonzero_votes = [v for v in direction_votes if v != 0]
    if len(nonzero_votes) >= 3:
        same_direction_share = max(
            nonzero_votes.count(1) / len(nonzero_votes),
            nonzero_votes.count(-1) / len(nonzero_votes),
        )
        if same_direction_share >= 0.8:
            score += 3
        elif same_direction_share < 0.6:
            score -= 5

    score = int(np.clip(round(score), 0, 100))

    direction_text = "lower" if female_coef < 0 else "higher"
    explanation = (
        f"Unadjusted denial rates were nearly identical by gender (female={female_rate:.3f}, male={male_rate:.3f}; "
        f"z-test p={z_p:.3f}, chi-square p={chi2_p:.3f}), but adjusted models found a statistically significant gender effect. "
        f"In logistic regression controlling for credit and debt factors, female had coefficient {female_coef:.3f} "
        f"(OR={female_or:.3f}, 95% CI [{ci_low:.3f}, {ci_high:.3f}], p={female_p:.3f}), implying {direction_text} denial odds. "
        f"Linear probability and regularized linear models also gave negative female coefficients "
        f"(OLS={female_ols_coef:.3f}, LinearRegression={lin_female_coef:.3f}, Lasso={lasso_female_coef:.3f}). "
        f"Custom interpretable models were mixed: SmartAdditive estimated ~{smart_female_effect:.3f} average toggle effect "
        f"(essentially no additional nonlinear gender effect), while HingeEBM estimated {hinge_female_effect:.3f} (if available). "
        f"Overall evidence supports a real but modest gender association after adjustment."
    )

    result = {"response": score, "explanation": explanation}
    print("Final Likert score:", result["response"])
    print("Explanation:", result["explanation"])

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
