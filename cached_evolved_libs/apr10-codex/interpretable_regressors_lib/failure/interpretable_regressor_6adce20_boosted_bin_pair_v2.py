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


class BoostedBinPairRegressor(BaseEstimator, RegressorMixin):
    """
    Compact boosted additive model with one optional pairwise table:
      1) Winsorize each feature (5th/95th percentiles).
      2) Build per-feature quantile bins and fit stagewise additive updates.
      3) Add one data-driven pair interaction table if it helps.
      4) Apply global shrinkage for stability.
    """

    def __init__(
        self,
        n_bins=8,
        n_rounds=20,
        learning_rate=0.18,
        l2_bin=1.0,
        add_pair=True,
        pair_bins=5,
        max_display_terms=10,
    ):
        self.n_bins = n_bins
        self.n_rounds = n_rounds
        self.learning_rate = learning_rate
        self.l2_bin = l2_bin
        self.add_pair = add_pair
        self.pair_bins = pair_bins
        self.max_display_terms = max_display_terms

    def _winsorize(self, X):
        return np.clip(X, self.low_, self.high_)

    def _impute_non_finite(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        bad = ~np.isfinite(X)
        if np.any(bad):
            X = X.copy()
            X[bad] = np.take(self.impute_values_, np.where(bad)[1])
        return X

    def _make_edges(self, x, n_bins):
        q = np.linspace(0.0, 1.0, int(max(2, n_bins)) + 1)
        edges = np.quantile(x, q)
        edges[0] = -np.inf
        edges[-1] = np.inf
        edges = np.unique(edges)
        if edges.size < 3:
            edges = np.array([-np.inf, np.inf], dtype=float)
        return edges

    def _digitize(self, x, edges):
        if edges.size == 2:
            return np.zeros(x.shape[0], dtype=int), 1
        b = np.searchsorted(edges[1:-1], x, side="right")
        return b.astype(int), edges.size - 1

    def _fit_bins_to_residual(self, bins, residual, n_states):
        sum_r = np.bincount(bins, weights=residual, minlength=n_states)
        cnt = np.bincount(bins, minlength=n_states).astype(float)
        vals = sum_r / (cnt + float(self.l2_bin))
        vals = vals - np.average(vals, weights=np.maximum(cnt, 1.0))
        return vals

    def _score_feature(self, bins, residual, n_states):
        vals = self._fit_bins_to_residual(bins, residual, n_states)
        pred = vals[bins]
        gain = float(np.dot(pred, pred))
        return gain, vals

    def _fit_pair_table(self, bi, bj, residual, ni, nj):
        key = bi * nj + bj
        n_cells = ni * nj
        sum_r = np.bincount(key, weights=residual, minlength=n_cells)
        cnt = np.bincount(key, minlength=n_cells).astype(float)
        table = sum_r / (cnt + float(self.l2_bin))
        table = table.reshape(ni, nj)

        wi = np.maximum(cnt.reshape(ni, nj).sum(axis=1), 1.0)
        wj = np.maximum(cnt.reshape(ni, nj).sum(axis=0), 1.0)
        table -= np.average(table, axis=1, weights=wj)[..., None]
        table -= np.average(table, axis=0, weights=wi)[None, ...]
        return table

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape

        self.n_features_in_ = int(p)
        self.intercept_ = float(np.mean(y)) if y.size else 0.0
        self.feature_names_ = [f"x{j}" for j in range(p)]

        if p > 0:
            self.impute_values_ = np.nanmedian(X, axis=0)
            self.impute_values_ = np.where(np.isfinite(self.impute_values_), self.impute_values_, 0.0)
            X = self._impute_non_finite(X)
        else:
            self.impute_values_ = np.zeros(0, dtype=float)

        try:
            if n == 0 or p == 0:
                self.low_ = np.zeros(p, dtype=float)
                self.high_ = np.zeros(p, dtype=float)
                self.edges_ = [np.array([-np.inf, np.inf], dtype=float) for _ in range(p)]
                self.bin_values_ = [np.zeros(1, dtype=float) for _ in range(p)]
                self.selected_features_ = []
                self.pair_ = None
                self.training_mse_ = 0.0
                self.is_fitted_ = True
                return self

            self.low_ = np.quantile(X, 0.05, axis=0)
            self.high_ = np.quantile(X, 0.95, axis=0)
            Xw = self._winsorize(X)

            self.edges_ = [self._make_edges(Xw[:, j], self.n_bins) for j in range(p)]
            binned = []
            n_states = []
            for j in range(p):
                bj, nj = self._digitize(Xw[:, j], self.edges_[j])
                binned.append(bj)
                n_states.append(nj)

            self.bin_values_ = [np.zeros(ns, dtype=float) for ns in n_states]
            selected = set()

            pred = np.full(n, self.intercept_, dtype=float)
            y_center = y - self.intercept_
            residual = y_center.copy()

            rounds = int(max(1, self.n_rounds))
            lr = float(np.clip(self.learning_rate, 0.01, 1.0))

            for _ in range(rounds):
                best_j = None
                best_gain = -np.inf
                best_vals = None
                for j in range(p):
                    gain, vals = self._score_feature(binned[j], residual, n_states[j])
                    if gain > best_gain:
                        best_gain = gain
                        best_j = j
                        best_vals = vals
                if best_j is None or best_gain <= 1e-12:
                    break
                self.bin_values_[best_j] += lr * best_vals
                pred += lr * best_vals[binned[best_j]]
                residual = y_center - (pred - self.intercept_)
                selected.add(int(best_j))

            self.selected_features_ = sorted(selected)
            self.pair_ = None

            if self.add_pair and p >= 2:
                feature_scores = []
                for j in range(p):
                    strength = float(np.mean(np.abs(self.bin_values_[j])))
                    feature_scores.append((strength, j))
                feature_scores.sort(reverse=True)
                cands = [j for _, j in feature_scores[: min(6, p)]]

                best_pair_gain = 0.0
                best_pair = None
                for a in range(len(cands)):
                    for b in range(a + 1, len(cands)):
                        i, j = cands[a], cands[b]
                        ei = self._make_edges(Xw[:, i], self.pair_bins)
                        ej = self._make_edges(Xw[:, j], self.pair_bins)
                        bi, ni = self._digitize(Xw[:, i], ei)
                        bj, nj = self._digitize(Xw[:, j], ej)
                        table = self._fit_pair_table(bi, bj, residual, ni, nj)
                        pair_pred = table[bi, bj]
                        gain = float(np.dot(pair_pred, pair_pred))
                        if gain > best_pair_gain:
                            best_pair_gain = gain
                            best_pair = (i, j, ei, ej, table)

                if best_pair is not None and best_pair_gain > 1e-10:
                    i, j, ei, ej, table = best_pair
                    bi, _ = self._digitize(Xw[:, i], ei)
                    bj, _ = self._digitize(Xw[:, j], ej)
                    pred += 0.6 * lr * table[bi, bj]
                    self.pair_ = {
                        "i": int(i),
                        "j": int(j),
                        "edges_i": ei,
                        "edges_j": ej,
                        "table": 0.6 * lr * table,
                    }

            shrink = 0.9
            for j in range(p):
                self.bin_values_[j] *= shrink
            if self.pair_ is not None:
                self.pair_["table"] *= shrink

            self.is_fitted_ = True
            final_pred = self.predict(X)
            self.training_mse_ = float(np.mean((y - final_pred) ** 2))
            return self

        except Exception:
            self.low_ = np.full(p, -np.inf, dtype=float)
            self.high_ = np.full(p, np.inf, dtype=float)
            self.edges_ = [np.array([-np.inf, np.inf], dtype=float) for _ in range(p)]
            self.bin_values_ = [np.zeros(1, dtype=float) for _ in range(p)]
            self.selected_features_ = []
            self.pair_ = None
            self.training_mse_ = float(np.mean((y - self.intercept_) ** 2)) if n > 0 else 0.0
            self.is_fitted_ = True
            return self

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        X = self._impute_non_finite(X)
        Xw = self._winsorize(X)
        n = Xw.shape[0]
        yhat = np.full(n, self.intercept_, dtype=float)

        for j in self.selected_features_:
            bj, _ = self._digitize(Xw[:, j], self.edges_[j])
            vals = self.bin_values_[j]
            bj = np.clip(bj, 0, vals.size - 1)
            yhat += vals[bj]

        if self.pair_ is not None:
            pi = self.pair_
            bi, _ = self._digitize(Xw[:, pi["i"]], pi["edges_i"])
            bj, _ = self._digitize(Xw[:, pi["j"]], pi["edges_j"])
            bi = np.clip(bi, 0, pi["table"].shape[0] - 1)
            bj = np.clip(bj, 0, pi["table"].shape[1] - 1)
            yhat += pi["table"][bi, bj]
        return yhat

    def __str__(self):
        check_is_fitted(self, "is_fitted_")
        lines = ["Boosted Bin + Pair Equation Regressor:"]
        lines.append("  prediction = intercept + sum_j bin_effect_j(x_j) + optional_pair_table(x_i, x_j)")
        lines.append(f"  rounds: {int(self.n_rounds)}, lr: {float(self.learning_rate):.3f}, bins: {int(self.n_bins)}")
        lines.append(f"  selected additive features: {len(self.selected_features_)}/{self.n_features_in_}")
        lines.append(f"  training MSE: {self.training_mse_:.6f}")
        lines.append(f"  intercept: {self.intercept_:+.6f}")

        scores = []
        for j in self.selected_features_:
            vals = self.bin_values_[j]
            scores.append((float(np.mean(np.abs(vals))), j, vals))
        scores.sort(reverse=True)

        lines.append("  strongest additive bin-shapes:")
        for _, j, vals in scores[: int(max(1, self.max_display_terms))]:
            vals_txt = ", ".join(f"{v:+.3f}" for v in vals[: min(8, vals.size)])
            if vals.size > 8:
                vals_txt += ", ..."
            lines.append(f"    x{j}: bin_values=[{vals_txt}]")

        if self.pair_ is not None:
            pi = self.pair_
            table = pi["table"]
            lines.append(
                f"  pair interaction: x{pi['i']} x x{pi['j']} table {table.shape[0]}x{table.shape[1]}"
            )
            row_summ = ", ".join(f"{v:+.3f}" for v in table.mean(axis=1)[: min(6, table.shape[0])])
            col_summ = ", ".join(f"{v:+.3f}" for v in table.mean(axis=0)[: min(6, table.shape[1])])
            lines.append(f"    row means: [{row_summ}]")
            lines.append(f"    col means: [{col_summ}]")

        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
BoostedBinPairRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "BoostedBinPair_v2"
model_description = "Winsorized quantile-bin boosted additive model with one optional pairwise interaction table and explicit equation-style summaries"
model_defs = [(model_shorthand_name, BoostedBinPairRegressor())]


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

    std_tests = {t.__name__ for t in ALL_TESTS}
    hard_tests = {t.__name__ for t in HARD_TESTS}
    insight_tests = {t.__name__ for t in INSIGHT_TESTS}
    std_passed = sum(r["passed"] for r in interp_results if r["test"] in std_tests)
    hard_passed = sum(r["passed"] for r in interp_results if r["test"] in hard_tests)
    insight_passed = sum(r["passed"] for r in interp_results if r["test"] in insight_tests)
    print(f"[std {std_passed}/{len(std_tests)}  hard {hard_passed}/{len(hard_tests)}  insight {insight_passed}/{len(insight_tests)}]")
    print(f"total_seconds: {time.time() - t0:.1f}s")
