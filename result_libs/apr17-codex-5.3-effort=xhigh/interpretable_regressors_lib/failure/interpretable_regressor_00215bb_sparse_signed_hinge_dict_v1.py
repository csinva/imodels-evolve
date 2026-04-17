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


class SparseSignedHingeDictionaryRegressor(BaseEstimator, RegressorMixin):
    """Screened additive dictionary model with linear, signed-hinge, and quadratic atoms."""

    def __init__(
        self,
        ridge_alpha=0.8,
        max_active_features=8,
        keep_terms=14,
        coef_threshold=0.015,
        include_interaction=True,
        negligible_coef_eps=1e-6,
    ):
        self.ridge_alpha = ridge_alpha
        self.max_active_features = max_active_features
        self.keep_terms = keep_terms
        self.coef_threshold = coef_threshold
        self.include_interaction = include_interaction
        self.negligible_coef_eps = negligible_coef_eps

    @staticmethod
    def _safe_corr(a, b):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        a0 = a - np.mean(a)
        b0 = b - np.mean(b)
        den = np.std(a0) * np.std(b0)
        if den <= 1e-12:
            return 0.0
        return float(np.mean(a0 * b0) / den)

    @staticmethod
    def _ridge_fit(X, y, alpha):
        p = X.shape[1]
        return np.linalg.solve(X.T @ X + float(alpha) * np.eye(p), X.T @ y)

    @staticmethod
    def _ols_fit(X, y):
        return np.linalg.lstsq(X, y, rcond=None)[0]

    def _build_basis(self, Xs, selected_global_features):
        n_samples = Xs.shape[0]
        cols = [np.ones(n_samples, dtype=float)]
        info = [("bias", None)]

        for feat in selected_global_features:
            x = Xs[:, feat]
            cols.append(x)
            info.append(("linear", int(feat)))

            cols.append(np.maximum(0.0, x))
            info.append(("hinge_pos", int(feat)))

            cols.append(np.maximum(0.0, -x))
            info.append(("hinge_neg", int(feat)))

            cols.append(x * x)
            info.append(("quad", int(feat)))

        if self.include_interaction and len(selected_global_features) >= 2:
            pair_scores = []
            for i in range(len(selected_global_features)):
                fi = selected_global_features[i]
                xi = Xs[:, fi]
                for j in range(i + 1, len(selected_global_features)):
                    fj = selected_global_features[j]
                    xj = Xs[:, fj]
                    pair_scores.append((abs(self._safe_corr(xi * xj, self.y_centered_)), fi, fj))
            if pair_scores:
                _, fi, fj = max(pair_scores, key=lambda t: t[0])
                cols.append(Xs[:, fi] * Xs[:, fj])
                info.append(("interaction", (int(fi), int(fj))))

        return np.column_stack(cols), info

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        self.x_mean_ = np.mean(X, axis=0)
        self.x_scale_ = np.std(X, axis=0)
        self.x_scale_[self.x_scale_ < 1e-12] = 1.0
        Xs = (X - self.x_mean_) / self.x_scale_

        self.intercept_ = float(np.mean(y))
        self.y_centered_ = y - self.intercept_

        corr = np.array([abs(self._safe_corr(Xs[:, j], self.y_centered_)) for j in range(n_features)], dtype=float)
        order = np.argsort(corr)[::-1]
        n_keep = max(1, min(int(self.max_active_features), n_features))
        self.active_features_ = np.array([int(j) for j in order[:n_keep]], dtype=int)

        design_full, basis_info_full = self._build_basis(Xs, self.active_features_)
        coef_full = self._ridge_fit(design_full, self.y_centered_, self.ridge_alpha)

        nonbias = np.abs(coef_full[1:])
        keep_idx = [0]
        if nonbias.size > 0:
            ranked = np.argsort(nonbias)[::-1]
            for ridx in ranked:
                if len(keep_idx) >= int(self.keep_terms):
                    break
                coef_val = abs(float(coef_full[1 + ridx]))
                if coef_val < float(self.coef_threshold):
                    continue
                keep_idx.append(1 + int(ridx))

        if len(keep_idx) == 1 and design_full.shape[1] > 1:
            keep_idx.extend([1, 2][: max(0, design_full.shape[1] - 1)])

        keep_idx = np.array(sorted(set(keep_idx)), dtype=int)
        self.selected_basis_info_ = [basis_info_full[i] for i in keep_idx]

        design_small = design_full[:, keep_idx]
        coef_small = self._ols_fit(design_small, self.y_centered_)

        coef_small[np.abs(coef_small) < self.negligible_coef_eps] = 0.0

        self.basis_keep_indices_ = keep_idx
        self.coef_small_ = coef_small

        full_linear = np.zeros(n_features, dtype=float)
        full_magnitude = np.zeros(n_features, dtype=float)

        for c, (term_type, term_meta) in zip(self.coef_small_, self.selected_basis_info_):
            mag = abs(float(c))
            if term_type in {"linear", "hinge_pos", "hinge_neg", "quad"}:
                feat = int(term_meta)
                full_magnitude[feat] += mag
                if term_type == "linear":
                    full_linear[feat] += float(c) / self.x_scale_[feat]
            elif term_type == "interaction":
                fi, fj = term_meta
                full_magnitude[fi] += 0.5 * mag
                full_magnitude[fj] += 0.5 * mag

        if np.max(full_magnitude) > 0:
            self.feature_importances_ = full_magnitude / np.max(full_magnitude)
        else:
            self.feature_importances_ = full_magnitude

        self.linear_coef_full_ = full_linear

        del self.y_centered_
        return self

    def _eval_basis_term(self, Xs, term_type, term_meta):
        if term_type == "bias":
            return np.ones(Xs.shape[0], dtype=float)
        if term_type == "linear":
            return Xs[:, term_meta]
        if term_type == "hinge_pos":
            return np.maximum(0.0, Xs[:, term_meta])
        if term_type == "hinge_neg":
            return np.maximum(0.0, -Xs[:, term_meta])
        if term_type == "quad":
            x = Xs[:, term_meta]
            return x * x
        if term_type == "interaction":
            fi, fj = term_meta
            return Xs[:, fi] * Xs[:, fj]
        raise ValueError(f"Unknown basis term: {term_type}")

    def predict(self, X):
        check_is_fitted(
            self,
            [
                "intercept_",
                "x_mean_",
                "x_scale_",
                "selected_basis_info_",
                "coef_small_",
            ],
        )
        X = np.asarray(X, dtype=float)
        Xs = (X - self.x_mean_) / self.x_scale_

        pred = np.full(X.shape[0], self.intercept_, dtype=float)
        for coef, (term_type, term_meta) in zip(self.coef_small_, self.selected_basis_info_):
            pred += float(coef) * self._eval_basis_term(Xs, term_type, term_meta)
        return pred

    def __str__(self):
        check_is_fitted(
            self,
            [
                "intercept_",
                "n_features_in_",
                "active_features_",
                "feature_importances_",
                "linear_coef_full_",
                "selected_basis_info_",
                "coef_small_",
                "x_mean_",
                "x_scale_",
            ],
        )

        ranked = np.argsort(self.feature_importances_)[::-1]
        important = [int(j) for j in ranked if self.feature_importances_[j] > 0]
        inactive = [f"x{i}" for i in range(self.n_features_in_) if i not in set(self.active_features_)]

        eq_terms = [f"{self.intercept_:+.4f}"]
        term_lines = []

        for coef, (term_type, meta) in zip(self.coef_small_, self.selected_basis_info_):
            c = float(coef)
            if abs(c) < self.negligible_coef_eps:
                continue
            if term_type == "bias":
                continue
            if term_type == "linear":
                j = int(meta)
                expr = f"((x{j}-{self.x_mean_[j]:+.4f})/{self.x_scale_[j]:.4f})"
                eq_terms.append(f"{c:+.4f}*{expr}")
            elif term_type == "hinge_pos":
                j = int(meta)
                expr = f"max(0, (x{j}-{self.x_mean_[j]:+.4f})/{self.x_scale_[j]:.4f})"
                eq_terms.append(f"{c:+.4f}*{expr}")
            elif term_type == "hinge_neg":
                j = int(meta)
                expr = f"max(0, -(x{j}-{self.x_mean_[j]:+.4f})/{self.x_scale_[j]:.4f})"
                eq_terms.append(f"{c:+.4f}*{expr}")
            elif term_type == "quad":
                j = int(meta)
                expr = f"((x{j}-{self.x_mean_[j]:+.4f})/{self.x_scale_[j]:.4f})^2"
                eq_terms.append(f"{c:+.4f}*{expr}")
            elif term_type == "interaction":
                j, k = meta
                expr = (
                    f"((x{j}-{self.x_mean_[j]:+.4f})/{self.x_scale_[j]:.4f})*"
                    f"((x{k}-{self.x_mean_[k]:+.4f})/{self.x_scale_[k]:.4f})"
                )
                eq_terms.append(f"{c:+.4f}*{expr}")

        for j in important[:10]:
            term_lines.append(
                f"x{j}: importance={self.feature_importances_[j]:.3f}, "
                f"linear_raw_effect_per_unit={self.linear_coef_full_[j]:+.4f}"
            )

        lines = [
            "Sparse Signed-Hinge Dictionary Regressor",
            "Prediction equation (raw features):",
            "  y = " + " ".join(eq_terms),
            "",
            "Most influential features:",
            *("  " + t for t in (term_lines if term_lines else ["none"])),
            "Inactive screened-out features: " + (", ".join(inactive) if inactive else "none"),
        ]
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys

_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])

# Update the model shorthand name and description below to reflect the class above and any changes you make.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
SparseSignedHingeDictionaryRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "SparseSignedHingeDict_v1"
model_description = "Correlation-screened additive dictionary with signed zero-knot hinges, quadratic atoms, one optional interaction, and hard-pruned OLS refit"
model_defs = [(model_shorthand_name, SparseSignedHingeDictionaryRegressor())]


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
