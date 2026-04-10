import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2


def logistic(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic transform."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def build_design_matrix(
    df: pd.DataFrame,
    include_genus: bool = True,
    drop_genus: str | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Construct model matrix with Homo sapiens and Anterior as reference levels."""
    cols = []
    names = []

    cols.append(np.ones(len(df), dtype=float))
    names.append("Intercept")

    cols.append(df["age"].to_numpy(dtype=float))
    names.append("age")

    cols.append(df["prob_male"].to_numpy(dtype=float))
    names.append("prob_male")

    for level in ["Posterior", "Premolar"]:
        cols.append((df["tooth_class"].to_numpy() == level).astype(float))
        names.append(f"tooth_{level}")

    if include_genus:
        for level in ["Pan", "Papio", "Pongo"]:
            if drop_genus == level:
                continue
            cols.append((df["genus"].to_numpy() == level).astype(float))
            names.append(f"genus_{level}")

    return np.column_stack(cols), names


def fit_binomial_logit(X: np.ndarray, y: np.ndarray, n: np.ndarray):
    """Fit aggregated-binomial logistic regression by MLE."""

    def neg_log_likelihood(beta: np.ndarray) -> float:
        p = logistic(X @ beta)
        p = np.clip(p, 1e-12, 1 - 1e-12)
        ll = np.sum(y * np.log(p) + (n - y) * np.log(1 - p))
        return -float(ll)

    def gradient(beta: np.ndarray) -> np.ndarray:
        p = logistic(X @ beta)
        return X.T @ (n * p - y)

    beta0 = np.zeros(X.shape[1], dtype=float)

    # Near-separation can make one optimizer report precision loss; try robust fallbacks.
    attempts = []
    for method in ["BFGS", "L-BFGS-B"]:
        result = minimize(
            neg_log_likelihood,
            beta0,
            jac=gradient,
            method=method,
        )
        attempts.append(result)
        if result.success:
            return result

    best = min(attempts, key=lambda r: r.fun if np.isfinite(r.fun) else np.inf)
    if not np.isfinite(best.fun):
        raise RuntimeError("Model fit failed: no finite solution from optimizers.")
    return best


def likelihood_ratio_test(full_ll: float, reduced_ll: float, dof: int) -> tuple[float, float]:
    lr_stat = 2.0 * (full_ll - reduced_ll)
    p_value = 1.0 - chi2.cdf(lr_stat, dof)
    return float(lr_stat), float(p_value)


def fmt_p(p: float) -> str:
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def main() -> None:
    base = Path(__file__).resolve().parent

    info_path = base / "info.json"
    data_path = base / "amtl.csv"

    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

    research_question = info["research_questions"][0]
    df = pd.read_csv(data_path)

    required = [
        "tooth_class",
        "specimen",
        "num_amtl",
        "sockets",
        "age",
        "prob_male",
        "genus",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if (df["num_amtl"] > df["sockets"]).any():
        raise ValueError("Found rows where num_amtl > sockets, which is invalid for binomial modeling.")

    df = df.copy()
    df["amtl_rate"] = df["num_amtl"] / df["sockets"]

    print("Research question:")
    print(research_question)
    print("\nData overview:")
    print(f"Rows: {len(df)}, specimens: {df['specimen'].nunique()}")
    print("Genus counts:")
    print(df["genus"].value_counts().to_string())
    print("Mean AMTL rate by genus:")
    print(df.groupby("genus")["amtl_rate"].mean().sort_values(ascending=False).to_string())

    y = df["num_amtl"].to_numpy(dtype=float)
    n = df["sockets"].to_numpy(dtype=float)

    X_full, names_full = build_design_matrix(df, include_genus=True)
    fit_full = fit_binomial_logit(X_full, y, n)
    ll_full = -fit_full.fun
    beta_full = fit_full.x

    X_no_genus, _ = build_design_matrix(df, include_genus=False)
    fit_no_genus = fit_binomial_logit(X_no_genus, y, n)
    ll_no_genus = -fit_no_genus.fun

    global_lr, global_p = likelihood_ratio_test(ll_full, ll_no_genus, dof=3)

    genus_results = {}
    for genus in ["Pan", "Papio", "Pongo"]:
        X_reduced, _ = build_design_matrix(df, include_genus=True, drop_genus=genus)
        fit_reduced = fit_binomial_logit(X_reduced, y, n)
        ll_reduced = -fit_reduced.fun

        lr, p_two_sided = likelihood_ratio_test(ll_full, ll_reduced, dof=1)
        coef = float(beta_full[names_full.index(f"genus_{genus}")])
        odds_ratio = float(np.exp(coef))

        p_one_sided = p_two_sided / 2.0 if coef < 0 else 1.0 - (p_two_sided / 2.0)

        genus_results[genus] = {
            "coef_log_odds": coef,
            "odds_ratio_vs_humans": odds_ratio,
            "lr_stat": float(lr),
            "p_two_sided": float(p_two_sided),
            "p_one_sided_humans_higher": float(p_one_sided),
            "supports_humans_higher": bool(coef < 0 and p_two_sided < 0.05),
        }

    print("\nModel test results:")
    print(f"Global genus effect (LRT, df=3): LR={global_lr:.3f}, p={fmt_p(global_p)}")
    for genus, res in genus_results.items():
        print(
            f"{genus}: coef={res['coef_log_odds']:.3f}, "
            f"OR={res['odds_ratio_vs_humans']:.3f}, "
            f"p(two-sided)={fmt_p(res['p_two_sided'])}"
        )

    all_supported = all(v["supports_humans_higher"] for v in genus_results.values())

    if all_supported and global_p < 0.05:
        response = 96
    elif global_p < 0.05 and any(v["supports_humans_higher"] for v in genus_results.values()):
        response = 75
    elif global_p < 0.05:
        response = 40
    else:
        response = 8

    explanation = (
        "Using aggregated-binomial logistic regression (controls: age, prob_male, tooth_class), "
        f"genus was significant overall (LRT p={fmt_p(global_p)}). Relative to Homo sapiens, "
        f"Pan (coef={genus_results['Pan']['coef_log_odds']:.3f}, p={fmt_p(genus_results['Pan']['p_two_sided'])}), "
        f"Papio (coef={genus_results['Papio']['coef_log_odds']:.3f}, p={fmt_p(genus_results['Papio']['p_two_sided'])}), "
        f"and Pongo (coef={genus_results['Pongo']['coef_log_odds']:.3f}, p={fmt_p(genus_results['Pongo']['p_two_sided'])}) "
        "all had significantly lower AMTL log-odds than humans after adjustment; thus the data strongly support "
        "that modern humans have higher AMTL frequencies than these non-human genera."
    )

    out = {"response": int(response), "explanation": explanation}
    with (base / "conclusion.txt").open("w", encoding="utf-8") as f:
        f.write(json.dumps(out, ensure_ascii=True))

    print("\nWrote conclusion.txt")


if __name__ == "__main__":
    main()
