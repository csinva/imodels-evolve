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


class DenseResidualSymbolicRidgeRegressor(BaseEstimator, RegressorMixin):
    """
    Dense ridge backbone plus a tiny set of validation-gated symbolic atoms.

    Prediction form:
      y = intercept + sum_j w_j*x_j + sum_k v_k*phi_k(x)
    where each phi_k is one of:
      x_j^2, |x_j|, max(0, x_j), max(0, -x_j), x_i*x_j
    """

    def __init__(
        self,
        ridge_alphas=(1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0, 100.0),
        max_atom_features=4,
        max_atoms=2,
        candidate_corr_min=0.02,
        min_val_gain=0.0015,
        max_display_linear_terms=20,
        linear_coef_threshold=1e-7,
        random_state=42,
    ):
        self.ridge_alphas = ridge_alphas
        self.max_atom_features = max_atom_features
        self.max_atoms = max_atoms
        self.candidate_corr_min = candidate_corr_min
        self.min_val_gain = min_val_gain
        self.max_display_linear_terms = max_display_linear_terms
        self.linear_coef_threshold = linear_coef_threshold
        self.random_state = random_state

    @staticmethod
    def _safe_std(x):
        s = np.std(x, axis=0)
        s[s < 1e-12] = 1.0
        return s

    def _impute(self, X):
        X = np.asarray(X, dtype=float).copy()
        bad = ~np.isfinite(X)
        if bad.any():
            X[bad] = np.take(self.feature_medians_, np.where(bad)[1])
        return X

    def _fit_ridge_design(self, A, y):
        from sklearn.linear_model import RidgeCV

        m = np.mean(A, axis=0)
        s = self._safe_std(A)
        As = (A - m) / s
        reg = RidgeCV(alphas=np.asarray(self.ridge_alphas, dtype=float), cv=3)
        reg.fit(As, y)
        return reg, m, s

    @staticmethod
    def _predict_ridge_design(reg, A, m, s):
        return reg.predict((A - m) / s)

    @staticmethod
    def _atom_values(X, atom):
        kind = atom[0]
        if kind == "sq":
            j = atom[1]
            return X[:, j] ** 2
        if kind == "abs":
            j = atom[1]
            return np.abs(X[:, j])
        if kind == "hinge_pos":
            j = atom[1]
            return np.maximum(0.0, X[:, j])
        if kind == "hinge_neg":
            j = atom[1]
            return np.maximum(0.0, -X[:, j])
        i, j = atom[1], atom[2]
        return X[:, i] * X[:, j]

    @staticmethod
    def _atom_label(atom):
        kind = atom[0]
        if kind == "sq":
            return f"(x{atom[1]}^2)"
        if kind == "abs":
            return f"|x{atom[1]}|"
        if kind == "hinge_pos":
            return f"max(0, x{atom[1]})"
        if kind == "hinge_neg":
            return f"max(0, -x{atom[1]})"
        return f"(x{atom[1]}*x{atom[2]})"

    def _build_design(self, X, atoms):
        if not atoms:
            return X
        cols = [X]
        for atom in atoms:
            cols.append(self._atom_values(X, atom)[:, None])
        return np.concatenate(cols, axis=1)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape
        self.n_features_in_ = p

        self.feature_medians_ = np.nanmedian(X, axis=0)
        self.feature_medians_ = np.where(np.isfinite(self.feature_medians_), self.feature_medians_, 0.0)
        X = self._impute(X)

        rng = np.random.RandomState(self.random_state)
        idx = np.arange(n)
        rng.shuffle(idx)
        n_val = max(40, int(0.2 * n))
        n_val = min(n_val, max(1, n - 1))
        val_idx = idx[:n_val]
        tr_idx = idx[n_val:]
        if len(tr_idx) < max(20, int(0.55 * n)):
            tr_idx = idx
            val_idx = idx[: min(80, n)]

        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xva, yva = X[val_idx], y[val_idx]

        reg_base, dmean_base, dstd_base = self._fit_ridge_design(Xtr, ytr)
        pred_tr = self._predict_ridge_design(reg_base, Xtr, dmean_base, dstd_base)
        pred_va = self._predict_ridge_design(reg_base, Xva, dmean_base, dstd_base)
        base_val_mse = float(np.mean((yva - pred_va) ** 2))
        resid_tr = ytr - pred_tr

        # Candidate atoms over strongest linear features from the dense backbone.
        coef_main_raw = reg_base.coef_ / dstd_base
        top_main = np.argsort(np.abs(coef_main_raw))[::-1][: min(max(2, int(self.max_atom_features)), p)]

        centered_resid = resid_tr - np.mean(resid_tr)
        resid_norm = np.sqrt(np.dot(centered_resid, centered_resid)) + 1e-12
        candidate_atoms = []

        for j in top_main:
            candidate_atoms.extend([
                ("sq", int(j)),
                ("abs", int(j)),
                ("hinge_pos", int(j)),
                ("hinge_neg", int(j)),
            ])
        for a in range(len(top_main)):
            for b in range(a + 1, len(top_main)):
                candidate_atoms.append(("inter", int(top_main[a]), int(top_main[b])))

        # Remove duplicates while preserving order.
        uniq_atoms = list(dict.fromkeys(candidate_atoms))

        scored_atoms = []
        for atom in uniq_atoms:
            z = self._atom_values(Xtr, atom)
            zc = z - np.mean(z)
            z_norm = np.sqrt(np.dot(zc, zc)) + 1e-12
            corr = abs(float(np.dot(zc, centered_resid) / (z_norm * resid_norm)))
            if corr >= float(self.candidate_corr_min):
                scored_atoms.append((atom, corr))
        scored_atoms.sort(key=lambda t: t[1], reverse=True)

        selected_atoms = []
        current_val_mse = base_val_mse
        max_pool = min(len(scored_atoms), max(0, int(self.max_atoms)) * 8)
        for atom, _ in scored_atoms[:max_pool]:
            if len(selected_atoms) >= int(self.max_atoms):
                break
            trial_atoms = selected_atoms + [atom]
            Atr = self._build_design(Xtr, trial_atoms)
            Ava = self._build_design(Xva, trial_atoms)
            reg_trial, dmean_trial, dstd_trial = self._fit_ridge_design(Atr, ytr)
            pred_trial = self._predict_ridge_design(reg_trial, Ava, dmean_trial, dstd_trial)
            trial_val_mse = float(np.mean((yva - pred_trial) ** 2))
            if trial_val_mse + 1e-12 < current_val_mse:
                selected_atoms = trial_atoms
                current_val_mse = trial_val_mse

        use_atoms = (base_val_mse - current_val_mse) >= float(self.min_val_gain) and len(selected_atoms) > 0
        self.selected_atoms_ = selected_atoms if use_atoms else []

        Afinal = self._build_design(X, self.selected_atoms_)
        reg_final, dmean_final, dstd_final = self._fit_ridge_design(Afinal, y)

        self._reg_ = reg_final
        self._design_mean_ = dmean_final
        self._design_std_ = dstd_final
        self.alpha_ = float(reg_final.alpha_)

        coef_std = reg_final.coef_.astype(float)
        coef_design_raw = coef_std / dstd_final
        self.intercept_ = float(reg_final.intercept_ - np.dot(coef_design_raw, dmean_final))
        self.coef_ = coef_design_raw[:p].astype(float)
        self.atom_coefs_ = [float(coef_design_raw[p + k]) for k in range(len(self.selected_atoms_))]

        imp = np.abs(self.coef_).copy()
        for atom, c in zip(self.selected_atoms_, self.atom_coefs_):
            if atom[0] == "inter":
                imp[atom[1]] += 0.5 * abs(c)
                imp[atom[2]] += 0.5 * abs(c)
            else:
                imp[atom[1]] += abs(c)
        self.feature_importance_ = imp
        self.feature_rank_ = np.argsort(imp)[::-1]
        return self

    def predict(self, X):
        check_is_fitted(self, ["_reg_", "_design_mean_", "_design_std_"])
        X = self._impute(X)
        A = self._build_design(X, self.selected_atoms_)
        return self._predict_ridge_design(self._reg_, A, self._design_mean_, self._design_std_)

    def __str__(self):
        check_is_fitted(self, ["coef_", "intercept_", "feature_rank_"])
        lines = [
            "DenseResidualSymbolicRidgeRegressor",
            f"Backbone: RidgeCV over all raw features (alpha={self.alpha_:.4g})",
            f"Residual symbolic atoms used: {len(self.selected_atoms_)}",
            "",
            "Exact prediction equation:",
        ]

        abs_c = np.abs(self.coef_)
        if self.n_features_in_ <= int(self.max_display_linear_terms):
            linear_idx = np.arange(self.n_features_in_, dtype=int)
        else:
            keep = np.where(abs_c >= float(self.linear_coef_threshold))[0]
            if keep.size == 0:
                keep = np.array([int(np.argmax(abs_c))], dtype=int)
            keep = keep[np.argsort(abs_c[keep])[::-1][: int(self.max_display_linear_terms)]]
            linear_idx = np.array(sorted(int(k) for k in keep.tolist()), dtype=int)

        eq_terms = [f"{self.intercept_:+.6f}"]
        for j in linear_idx:
            eq_terms.append(f"{self.coef_[int(j)]:+.6f}*x{int(j)}")
        for atom, c in zip(self.selected_atoms_, self.atom_coefs_):
            eq_terms.append(f"{c:+.6f}*{self._atom_label(atom)}")
        lines.append("  y = " + " ".join(eq_terms))

        hidden = self.n_features_in_ - len(linear_idx)
        if hidden > 0:
            lines.append(f"  (+ {hidden} small linear terms omitted for readability)")

        if self.selected_atoms_:
            lines.extend(["", "Symbolic atom definitions:"])
            for atom, c in zip(self.selected_atoms_, self.atom_coefs_):
                lines.append(f"  {self._atom_label(atom)} with coefficient {c:+.6f}")

        lines.extend(["", "Feature importance (top 12):"])
        for j in self.feature_rank_[: min(12, self.n_features_in_)]:
            lines.append(f"  x{int(j)}: {self.feature_importance_[int(j)]:.6f}")

        near_zero = [f"x{j}" for j, v in enumerate(abs_c) if v < 1e-5]
        if near_zero:
            lines.append("Near-zero linear coefficients: " + ", ".join(near_zero[:20]))
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys

_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
DenseResidualSymbolicRidgeRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "DenseResSymbolicRidgeV1"
model_description = "Dense RidgeCV backbone with holdout-gated residual symbolic atoms (square/abs/hinge/interaction) for compact nonlinear corrections"
model_defs = [(model_shorthand_name, DenseResidualSymbolicRidgeRegressor())]


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

    # --- Compute mean_rank from the updated performance_results.csv ---
    # Build dataset_rmses dict with all models from the CSV for ranking
    all_dataset_rmses = defaultdict(dict)
    for row in existing_perf:
        rmse_str = row.get("rmse", "")
        if rmse_str not in ("", None):
            all_dataset_rmses[row["dataset"]][row["model"]] = float(rmse_str)
        else:
            all_dataset_rmses[row["dataset"]][row["model"]] = float("nan")
    avg_rank, _ = compute_rank_scores(dict(all_dataset_rmses))
    mean_rank = avg_rank.get(model_shorthand_name, float("nan"))

    upsert_overall_results([{
        "commit":                             git_hash,
        "mean_rank":                          f"{mean_rank:.2f}" if not np.isnan(mean_rank) else "nan",
        "frac_interpretability_tests_passed": f"{n_passed / total:.4f}" if total > 0 else "nan",
        "status":                             "",
        "model_name":                         model_shorthand_name,
        "description":                        model_description,
    }], RESULTS_DIR)

    # --- Plot ---
    overall_csv = os.path.join(RESULTS_DIR, "overall_results.csv")
    plot_interp_vs_performance(
        overall_csv,
        os.path.join(RESULTS_DIR, "interpretability_vs_performance.png"),
    )

    print()
    print("---")
    print(f"tests_passed:  {n_passed}/{total}" + (f" ({n_passed/total:.2%})" if total > 0 else ""))
    print(f"mean_rank:     {mean_rank:.2f}" if not np.isnan(mean_rank) else "mean_rank:     nan")
    print(f"total_seconds: {time.time() - t0:.1f}s")
