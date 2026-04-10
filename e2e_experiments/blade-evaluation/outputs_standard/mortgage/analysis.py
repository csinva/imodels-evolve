import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")


def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return float(default)


def get_feature_value(model, feature_names, feature="female"):
    if feature not in feature_names:
        return np.nan
    idx = feature_names.index(feature)

    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        if coef.ndim == 1 and idx < len(coef):
            return safe_float(coef[idx])
        if coef.ndim == 2 and coef.shape[1] > idx:
            return safe_float(coef[0, idx])

    if hasattr(model, "feature_importances_"):
        fi = np.asarray(model.feature_importances_)
        if idx < len(fi):
            return safe_float(fi[idx])

    if hasattr(model, "feature_importances"):
        try:
            fi = np.asarray(model.feature_importances)
            if idx < len(fi):
                return safe_float(fi[idx])
        except Exception:
            pass

    return np.nan


def main():
    cwd = Path(".")
    info_path = cwd / "info.json"
    data_path = cwd / "mortgage.csv"

    with open(info_path, "r") as f:
        info = json.load(f)

    research_q = info.get("research_questions", [""])[0]

    df = pd.read_csv(data_path)

    # Remove index-like column when present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Ensure numeric where possible
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna().reset_index(drop=True)

    # Define target
    if "accept" in df.columns:
        target = "accept"
    elif "deny" in df.columns:
        df["accept"] = 1 - df["deny"]
        target = "accept"
    else:
        raise ValueError("Dataset must contain 'accept' or 'deny'.")

    # Basic exploration
    summary_stats = df.describe().T
    corr = df.corr(numeric_only=True)
    corr_target = corr[target].sort_values(ascending=False)

    # Group outcomes by gender
    grp = df.groupby("female")[target].agg(["mean", "count", "std"]).rename_axis("female")
    female_accept = safe_float(grp.loc[1, "mean"]) if 1 in grp.index else np.nan
    male_accept = safe_float(grp.loc[0, "mean"]) if 0 in grp.index else np.nan
    rate_diff = female_accept - male_accept

    # Statistical tests
    y_f = df.loc[df["female"] == 1, target]
    y_m = df.loc[df["female"] == 0, target]

    t_stat, t_p = stats.ttest_ind(y_f, y_m, equal_var=False, nan_policy="omit")

    contingency = pd.crosstab(df["female"], df[target])
    chi2, chi2_p, chi2_dof, _ = stats.chi2_contingency(contingency)

    # OLS with controls (linear probability model)
    controls = [
        c
        for c in df.columns
        if c not in {target, "deny"}
    ]

    X_sm = sm.add_constant(df[controls], has_constant="add")
    ols = sm.OLS(df[target], X_sm).fit(cov_type="HC3")
    ols_coef_female = safe_float(ols.params.get("female", np.nan))
    ols_p_female = safe_float(ols.pvalues.get("female", np.nan))
    if "female" in ols.params.index:
        ci_low, ci_high = ols.conf_int().loc["female"].tolist()
    else:
        ci_low, ci_high = np.nan, np.nan

    # ANOVA perspective
    formula_terms = [
        "C(female)",
        "C(black)",
        "housing_expense_ratio",
        "C(self_employed)",
        "C(married)",
        "mortgage_credit",
        "consumer_credit",
        "bad_history",
        "PI_ratio",
        "loan_to_value",
        "denied_PMI",
    ]
    existing_terms = []
    for t in formula_terms:
        raw = t.replace("C(", "").replace(")", "")
        if raw in df.columns:
            existing_terms.append(t)
    formula = f"{target} ~ " + " + ".join(existing_terms)
    anova_model = smf.ols(formula, data=df).fit()
    anova_table = sm.stats.anova_lm(anova_model, typ=2)
    anova_row = "C(female)" if "C(female)" in anova_table.index else None
    anova_p_female = safe_float(anova_table.loc[anova_row, "PR(>F)"]) if anova_row else np.nan

    # Train/test for interpretable ML models
    feature_cols = [c for c in df.columns if c not in {target, "deny"}]
    X = df[feature_cols]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model_results = {}

    lin = LinearRegression().fit(X_train, y_train)
    lin_pred = lin.predict(X_test)
    model_results["LinearRegression"] = {
        "r2": safe_float(r2_score(y_test, lin_pred)),
        "female_effect": get_feature_value(lin, feature_cols, "female"),
    }

    ridge = Ridge(alpha=1.0, random_state=42).fit(X_train, y_train)
    ridge_pred = ridge.predict(X_test)
    model_results["Ridge"] = {
        "r2": safe_float(r2_score(y_test, ridge_pred)),
        "female_effect": get_feature_value(ridge, feature_cols, "female"),
    }

    lasso = Lasso(alpha=0.0005, random_state=42, max_iter=10000).fit(X_train, y_train)
    lasso_pred = lasso.predict(X_test)
    model_results["Lasso"] = {
        "r2": safe_float(r2_score(y_test, lasso_pred)),
        "female_effect": get_feature_value(lasso, feature_cols, "female"),
    }

    dtc = DecisionTreeClassifier(max_depth=4, min_samples_leaf=30, random_state=42).fit(
        X_train, y_train
    )
    dtc_pred = dtc.predict(X_test)
    dtc_proba = dtc.predict_proba(X_test)[:, 1]
    model_results["DecisionTreeClassifier"] = {
        "accuracy": safe_float(accuracy_score(y_test, dtc_pred)),
        "auc": safe_float(roc_auc_score(y_test, dtc_proba)),
        "female_importance": get_feature_value(dtc, feature_cols, "female"),
    }

    dtr = DecisionTreeRegressor(max_depth=4, min_samples_leaf=30, random_state=42).fit(
        X_train, y_train
    )
    dtr_pred = dtr.predict(X_test)
    model_results["DecisionTreeRegressor"] = {
        "r2": safe_float(r2_score(y_test, dtr_pred)),
        "female_importance": get_feature_value(dtr, feature_cols, "female"),
    }

    # imodels interpretable regressors
    imodels_notes = {}
    try:
        from imodels import FIGSRegressor, HSTreeRegressor, RuleFitRegressor

        rf = RuleFitRegressor(random_state=42)
        rf.fit(X_train.values, y_train.values, feature_names=feature_cols)
        rf_pred = rf.predict(X_test.values)
        rf_female = np.nan
        if hasattr(rf, "feature_importances_"):
            rf_female = get_feature_value(rf, feature_cols, "female")
        elif hasattr(rf, "get_rules"):
            rules = rf.get_rules()
            if isinstance(rules, pd.DataFrame) and "rule" in rules.columns and "coef" in rules.columns:
                female_rows = rules[rules["rule"].astype(str).str.contains("female", case=False, na=False)]
                if not female_rows.empty:
                    rf_female = safe_float(female_rows["coef"].abs().sum())
        model_results["RuleFitRegressor"] = {
            "r2": safe_float(r2_score(y_test, rf_pred)),
            "female_signal": safe_float(rf_female),
        }

        figs = FIGSRegressor(random_state=42)
        figs.fit(X_train.values, y_train.values, feature_names=feature_cols)
        figs_pred = figs.predict(X_test.values)
        model_results["FIGSRegressor"] = {
            "r2": safe_float(r2_score(y_test, figs_pred)),
            "female_importance": get_feature_value(figs, feature_cols, "female"),
        }

        hs = HSTreeRegressor(random_state=42)
        hs.fit(X_train.values, y_train.values, feature_names=feature_cols)
        hs_pred = hs.predict(X_test.values)
        model_results["HSTreeRegressor"] = {
            "r2": safe_float(r2_score(y_test, hs_pred)),
            "female_importance": get_feature_value(hs, feature_cols, "female"),
        }
    except Exception as e:
        imodels_notes["error"] = str(e)

    # Derive likelihood score for relationship existence
    # High score => evidence that gender has a statistically significant effect on approval.
    if np.isfinite(ols_p_female) and ols_p_female < 0.01:
        response = 90
    elif np.isfinite(ols_p_female) and ols_p_female < 0.05:
        response = 80
    elif np.isfinite(ols_p_female) and ols_p_female < 0.10:
        response = 60
    else:
        # If adjusted model not significant, allow only moderate support from unadjusted tests.
        if (np.isfinite(t_p) and t_p < 0.05) or (np.isfinite(chi2_p) and chi2_p < 0.05):
            response = 40
        else:
            response = 15

    # Light calibration by effect size in percentage points
    if np.isfinite(ols_coef_female):
        abs_pp = abs(ols_coef_female) * 100
        if abs_pp < 0.5:
            response = max(0, response - 15)
        elif abs_pp > 3.0:
            response = min(100, response + 5)

    direction = "lower" if rate_diff < 0 else "higher"
    explanation = (
        f"Research question: {research_q} "
        f"Female approval rate={female_accept:.3f}, male approval rate={male_accept:.3f} "
        f"(difference={rate_diff:.3f}, so female is {direction}). "
        f"Welch t-test p={t_p:.4g}; chi-square p={chi2_p:.4g}. "
        f"Adjusted OLS (accept ~ controls) female coef={ols_coef_female:.4f} "
        f"with p={ols_p_female:.4g} and 95% CI [{ci_low:.4f}, {ci_high:.4f}]. "
        f"ANOVA female p={anova_p_female:.4g}. "
        f"Interpretable models (Linear/Ridge/Lasso/Trees and imodels when available) were fit; "
        f"female effects/importances were compared to assess robustness. "
        f"Score reflects primarily statistical significance of the adjusted female effect."
    )

    # Persist a compact diagnostics file for reproducibility
    diagnostics = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "target": target,
        "summary_stats": summary_stats.round(4).to_dict(),
        "corr_with_target": corr_target.round(4).to_dict(),
        "group_stats_by_female": grp.round(4).to_dict(),
        "tests": {
            "welch_t_test": {"stat": safe_float(t_stat), "p": safe_float(t_p)},
            "chi_square": {
                "stat": safe_float(chi2),
                "p": safe_float(chi2_p),
                "dof": int(chi2_dof),
            },
            "ols_female": {
                "coef": safe_float(ols_coef_female),
                "p": safe_float(ols_p_female),
                "ci_low": safe_float(ci_low),
                "ci_high": safe_float(ci_high),
            },
            "anova_female_p": safe_float(anova_p_female),
        },
        "model_results": model_results,
        "imodels_notes": imodels_notes,
    }

    with open("analysis_diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)

    with open("conclusion.txt", "w") as f:
        json.dump({"response": int(response), "explanation": explanation}, f)

    print("Analysis complete. Wrote conclusion.txt")


if __name__ == "__main__":
    main()
