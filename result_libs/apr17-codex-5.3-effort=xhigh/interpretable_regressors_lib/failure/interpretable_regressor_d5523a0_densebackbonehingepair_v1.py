"""
Interpretable regressor autoresearch script.
Defines a scikit-learn compatible interpretable regressor and evaluates it
on interpretability tests and TabArena regression datasets (same suite used
for baselines in run_baselines.py).

Usage: uv run model.py
"""

import csv
import os
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class DenseBackboneHingePairRegressor(BaseEstimator, RegressorMixin):
    """Dense ridge backbone plus at most a few validation-selected hinge terms."""

    def __init__(
        self,
        ridge_alpha=1.2,
        val_fraction=0.2,
        hinge_quantiles=(0.15, 0.3, 0.5, 0.7, 0.85),
        top_hinge_features=4,
        max_hinge_terms=2,
        min_rel_gain=0.004,
        negligible_coef_eps=0.03,
        random_state=0,
    ):
        self.ridge_alpha = ridge_alpha
        self.val_fraction = val_fraction
        self.hinge_quantiles = hinge_quantiles
        self.top_hinge_features = top_hinge_features
        self.max_hinge_terms = max_hinge_terms
        self.min_rel_gain = min_rel_gain
        self.negligible_coef_eps = negligible_coef_eps
        self.random_state = random_state

    def _fit_ridge(self, Z, y, alpha):
        z_mean = np.mean(Z, axis=0)
        z_std = np.std(Z, axis=0)
        z_std[z_std < 1e-12] = 1.0
        Zs = (Z - z_mean) / z_std

        y_mean = float(np.mean(y))
        y_centered = y - y_mean

        p = Zs.shape[1]
        gram = Zs.T @ Zs
        rhs = Zs.T @ y_centered
        coef_std = np.linalg.solve(gram + float(alpha) * np.eye(p), rhs)

        coef = coef_std / z_std
        intercept = float(y_mean - np.dot(coef, z_mean))
        return intercept, coef

    def _design_matrix(self, X, hinge_terms):
        cols = [X]
        for term in hinge_terms:
            z = self._hinge_values(X[:, term["feature"]], term["knot"], term["direction"])
            cols.append(z.reshape(-1, 1))
        if len(cols) == 1:
            return X
        return np.column_stack(cols)

    def _mse_from_params(self, X, y, intercept, linear_coef, hinge_terms, hinge_coefs):
        pred = intercept + X @ linear_coef
        for coef, term in zip(hinge_coefs, hinge_terms):
            pred += float(coef) * self._hinge_values(X[:, term["feature"]], term["knot"], term["direction"])
        return float(np.mean((y - pred) ** 2))

    def _hinge_values(self, x_col, knot, direction):
        if direction == "right":
            return np.maximum(0.0, x_col - knot)
        return np.maximum(0.0, knot - x_col)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        rng = np.random.RandomState(self.random_state)
        idx = np.arange(n_samples)
        rng.shuffle(idx)
        n_val = int(max(1, min(n_samples - 1, round(self.val_fraction * n_samples))))
        val_idx = idx[:n_val]
        tr_idx = idx[n_val:]
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        current_terms = []
        used_terms = set()

        for _ in range(max(0, int(self.max_hinge_terms)) + 1):
            Z_tr = self._design_matrix(X_tr, current_terms)
            curr_intercept, curr_coef = self._fit_ridge(Z_tr, y_tr, alpha=self.ridge_alpha)
            curr_linear = curr_coef[:n_features]
            curr_hinge_coefs = curr_coef[n_features:]
            curr_mse = self._mse_from_params(
                X_val, y_val, curr_intercept, curr_linear, current_terms, curr_hinge_coefs
            )

            if len(current_terms) >= self.max_hinge_terms:
                break

            order = np.argsort(np.abs(curr_linear))[::-1]
            top_feats = order[: min(self.top_hinge_features, n_features)]

            best_candidate = None
            best_candidate_mse = curr_mse
            best_candidate_intercept = curr_intercept
            best_candidate_linear = curr_linear
            best_candidate_hinge_coefs = curr_hinge_coefs

            for feat in top_feats:
                x_col = X_tr[:, int(feat)]
                knots = np.unique(np.round(np.quantile(x_col, self.hinge_quantiles), 8))
                for knot in knots:
                    for direction in ("right", "left"):
                        key = (int(feat), float(knot), direction)
                        if key in used_terms:
                            continue
                        z = self._hinge_values(x_col, float(knot), direction)
                        if float(np.std(z)) < 1e-10:
                            continue
                        cand_terms = current_terms + [{
                            "feature": int(feat),
                            "knot": float(knot),
                            "direction": direction,
                        }]
                        Z_cand = self._design_matrix(X_tr, cand_terms)
                        cand_intercept, cand_coef = self._fit_ridge(Z_cand, y_tr, alpha=self.ridge_alpha)
                        cand_linear = cand_coef[:n_features]
                        cand_hinge_coefs = cand_coef[n_features:]
                        cand_mse = self._mse_from_params(
                            X_val, y_val, cand_intercept, cand_linear, cand_terms, cand_hinge_coefs
                        )
                        if cand_mse < best_candidate_mse:
                            best_candidate_mse = cand_mse
                            best_candidate = key
                            best_candidate_intercept = cand_intercept
                            best_candidate_linear = cand_linear
                            best_candidate_hinge_coefs = cand_hinge_coefs

            rel_gain = (curr_mse - best_candidate_mse) / (abs(curr_mse) + 1e-12)
            if best_candidate is None or rel_gain < self.min_rel_gain:
                self._val_intercept_ = curr_intercept
                self._val_linear_coef_ = curr_linear
                self._val_hinge_coefs_ = curr_hinge_coefs
                self._val_mse_ = curr_mse
                break

            feat, knot, direction = best_candidate
            current_terms.append({"feature": feat, "knot": knot, "direction": direction})
            used_terms.add(best_candidate)
            self._val_intercept_ = best_candidate_intercept
            self._val_linear_coef_ = best_candidate_linear
            self._val_hinge_coefs_ = best_candidate_hinge_coefs
            self._val_mse_ = best_candidate_mse

        self.hinge_terms_ = current_terms

        Z_full = self._design_matrix(X, self.hinge_terms_)
        self.intercept_, coef_full = self._fit_ridge(Z_full, y, alpha=self.ridge_alpha)
        self.linear_coef_ = coef_full[:n_features]
        self.hinge_coefs_ = coef_full[n_features:]

        self.feature_importances_ = np.abs(self.linear_coef_).copy()
        for coef, term in zip(self.hinge_coefs_, self.hinge_terms_):
            self.feature_importances_[term["feature"]] += abs(float(coef))

        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "linear_coef_", "hinge_terms_", "hinge_coefs_", "feature_importances_"])
        X = np.asarray(X, dtype=float)
        pred = self.intercept_ + X @ self.linear_coef_
        for coef, term in zip(self.hinge_coefs_, self.hinge_terms_):
            z = self._hinge_values(X[:, term["feature"]], term["knot"], term["direction"])
            pred += float(coef) * z
        return pred

    def _equation_terms(self):
        terms = [f"{self.intercept_:+.6f}"]
        for j in range(self.n_features_in_):
            terms.append(f"{self.linear_coef_[j]:+.6f}*x{int(j)}")
        for coef, term in zip(self.hinge_coefs_, self.hinge_terms_):
            if abs(float(coef)) <= self.negligible_coef_eps:
                continue
            feat = term["feature"]
            knot = term["knot"]
            if term["direction"] == "right":
                terms.append(f"{float(coef):+.6f}*max(0, x{feat}-{knot:.6f})")
            else:
                terms.append(f"{float(coef):+.6f}*max(0, {knot:.6f}-x{feat})")
        return terms

    def __str__(self):
        check_is_fitted(self, ["intercept_", "linear_coef_", "feature_importances_"])
        active = np.where(np.abs(self.linear_coef_) > self.negligible_coef_eps)[0]
        inactive = [f"x{j}" for j in range(self.n_features_in_) if abs(self.linear_coef_[j]) <= self.negligible_coef_eps]
        top = np.argsort(self.feature_importances_)[::-1]
        top_txt = ", ".join(
            f"x{int(j)}:{self.feature_importances_[j]:.4f}" for j in top[: min(6, self.n_features_in_)]
        )

        lines = [
            "Dense Ridge + Few Hinges Regressor",
            "Exact prediction equation (use directly for manual calculation):",
            "  y = " + " ".join(self._equation_terms()),
            "",
            f"Linear features with non-negligible coefficients ({len(active)}): "
            + (", ".join(f"x{int(j)}" for j in active) if len(active) else "none"),
        ]

        lines.append("Linear coefficient table (all features):")
        for j in range(self.n_features_in_):
            lines.append(f"  x{j}: {self.linear_coef_[j]:+.6f}")

        if len(self.hinge_terms_) == 0:
            lines.append("Hinge corrections: none")
        else:
            lines.append("Hinge corrections:")
            for i, (coef, term) in enumerate(zip(self.hinge_coefs_, self.hinge_terms_), 1):
                feat = term["feature"]
                knot = term["knot"]
                direction = term["direction"]
                if direction == "right":
                    hinge_txt = f"max(0, x{feat}-{knot:.6f})"
                else:
                    hinge_txt = f"max(0, {knot:.6f}-x{feat})"
                lines.append(f"  h{i}: {float(coef):+.6f} * {hinge_txt}")

        lines.extend([
            "Features with near-zero linear effect (candidate irrelevant features): "
            + (", ".join(inactive) if inactive else "none"),
            "Top feature influence magnitudes: " + top_txt,
        ])
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
DenseBackboneHingePairRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "DenseBackboneHingePair_v1"
model_description = "Dense ridge backbone over all features with validation-selected up-to-two hinge corrections and explicit full equation table"
model_defs = [(model_shorthand_name, DenseBackboneHingePairRegressor())]


# ---------------------------------------------------------------------------
# Evaluation (do not edit anything below this line)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()

    # Interpretability tests
    interp_results = run_all_interp_tests(model_defs)
    n_passed = sum(r["passed"] for r in interp_results)
    total = len(interp_results)

    # prediction performance (RMSE)
    dataset_rmses = evaluate_all_regressors(model_defs)

    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        git_hash = ""

    # --- Upsert interpretability_results.csv ---
    model_name = model_defs[0][0]
    interp_csv = os.path.join(RESULTS_DIR, "interpretability_results.csv")
    interp_fields = ["model", "test", "suite", "passed", "ground_truth", "response"]

    def _suite(test_name):
        if test_name.startswith("insight_"): return "insight"
        if test_name.startswith("hard_"):    return "hard"
        return "standard"

    # Load existing rows, dropping old rows for this model
    existing_interp = []
    if os.path.exists(interp_csv):
        with open(interp_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("model") != model_name:
                    existing_interp.append(row)

    new_interp = [{
        "model": r["model"],
        "test": r["test"],
        "suite": _suite(r["test"]),
        "passed": r["passed"],
        "ground_truth": r.get("ground_truth", ""),
        "response": r.get("response", ""),
    } for r in interp_results]

    with open(interp_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=interp_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing_interp + new_interp)
    print(f"Interpretability results saved → {interp_csv}")

    # --- Upsert performance_results.csv and recompute ranks ---
    perf_csv = os.path.join(RESULTS_DIR, "performance_results.csv")
    perf_fields = ["dataset", "model", "rmse", "rank"]

    # Load existing rows, dropping old rows for this model
    existing_perf = []
    if os.path.exists(perf_csv):
        with open(perf_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("model") != model_name:
                    existing_perf.append(row)

    # Add new rows (without rank for now)
    for ds_name, model_rmses in dataset_rmses.items():
        rmse_val = model_rmses.get(model_name, float("nan"))
        existing_perf.append({
            "dataset": ds_name,
            "model": model_name,
            "rmse": "" if np.isnan(rmse_val) else f"{rmse_val:.6f}",
            "rank": "",
        })

    # Recompute ranks per dataset
    by_dataset = defaultdict(list)
    for row in existing_perf:
        by_dataset[row["dataset"]].append(row)

    for ds_name, rows in by_dataset.items():
        valid = [(r, float(r["rmse"])) for r in rows if r["rmse"] not in ("", None)]
        valid.sort(key=lambda x: x[1])
        for rank_idx, (r, _) in enumerate(valid, 1):
            r["rank"] = rank_idx
        # Leave rank empty for rows with no RMSE
        for r in rows:
            if r["rmse"] in ("", None):
                r["rank"] = ""

    with open(perf_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=perf_fields)
        writer.writeheader()
        for ds_name in by_dataset:
            for row in by_dataset[ds_name]:
                writer.writerow(row)
    print(f"Performance results saved → {perf_csv}")

    # --- Recompute global rank summary from updated performance_results.csv ---
    # Build dataset -> {model: rmse}
    perf_table = defaultdict(dict)
    with open(perf_csv, newline="") as f:
        for row in csv.DictReader(f):
            ds = row["dataset"]
            m = row["model"]
            rmse_s = row.get("rmse", "")
            if rmse_s in ("", None):
                perf_table[ds][m] = float("nan")
            else:
                try:
                    perf_table[ds][m] = float(rmse_s)
                except ValueError:
                    perf_table[ds][m] = float("nan")

    avg_rank, _ = compute_rank_scores(perf_table)
    mean_rank = avg_rank.get(model_name, float("nan"))

    # --- Upsert overall_results.csv ---
    overall_rows = [{
        "commit": git_hash,
        "mean_rank": f"{mean_rank:.2f}" if np.isfinite(mean_rank) else "",
        "frac_interpretability_tests_passed": f"{(n_passed / total):.4f}" if total else "",
        "status": "",  # fill manually after reviewing
        "model_name": model_name,
        "description": model_description,
    }]
    upsert_overall_results(overall_rows, RESULTS_DIR)

    # --- Plot update ---
    overall_csv = os.path.join(RESULTS_DIR, "overall_results.csv")
    plot_interp_vs_performance(
        overall_csv,
        os.path.join(RESULTS_DIR, "interpretability_vs_performance.png"),
    )

    # Print compact summary
    std_names = {t.__name__ for t in ALL_TESTS}
    hard_names = {t.__name__ for t in HARD_TESTS}
    ins_names = {t.__name__ for t in INSIGHT_TESTS}
    n_std = sum(r["passed"] for r in interp_results if r["test"] in std_names)
    n_hard = sum(r["passed"] for r in interp_results if r["test"] in hard_names)
    n_ins = sum(r["passed"] for r in interp_results if r["test"] in ins_names)

    print("\n---")
    print(f"tests_passed:  {n_passed}/{total} ({(n_passed/total):.2%})  "
          f"[std {n_std}/{len(std_names)}  hard {n_hard}/{len(hard_names)}  insight {n_ins}/{len(ins_names)}]")
    print(f"total_seconds: {time.time() - t0:.1f}s")
