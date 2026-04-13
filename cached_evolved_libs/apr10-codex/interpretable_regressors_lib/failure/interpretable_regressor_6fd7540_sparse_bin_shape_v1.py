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


class SparseBinShapeRegressor(BaseEstimator, RegressorMixin):
    """
    Greedy sparse additive model that mixes linear terms and single-feature
    quantile-bin shape functions.
    """

    def __init__(
        self,
        screen_features=12,
        max_terms=7,
        n_bins=5,
        shape_shrink=20.0,
        complexity_penalty=0.004,
        min_improvement=1e-4,
        coef_quant_step=0.05,
        min_abs_coef=0.05,
    ):
        self.screen_features = screen_features
        self.max_terms = max_terms
        self.n_bins = n_bins
        self.shape_shrink = shape_shrink
        self.complexity_penalty = complexity_penalty
        self.min_improvement = min_improvement
        self.coef_quant_step = coef_quant_step
        self.min_abs_coef = min_abs_coef

    @staticmethod
    def _safe_corr(x, y):
        xc = x - np.mean(x)
        yc = y - np.mean(y)
        denom = (np.linalg.norm(xc) + 1e-12) * (np.linalg.norm(yc) + 1e-12)
        return float(np.dot(xc, yc) / denom)

    @staticmethod
    def _quantize(v, step):
        if step <= 0:
            return float(v)
        return float(np.round(v / step) * step)

    @staticmethod
    def _fit_ols(D, y):
        beta, _, _, _ = np.linalg.lstsq(D, y, rcond=None)
        return beta

    def _make_edges(self, x):
        qs = np.linspace(0.0, 1.0, int(max(2, self.n_bins + 1)))
        edges = np.quantile(x, qs)
        edges = np.asarray(edges, dtype=float)
        edges[0] = -np.inf
        edges[-1] = np.inf
        for i in range(1, len(edges) - 1):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + 1e-8
        return edges

    @staticmethod
    def _digitize(x, edges):
        return np.digitize(x, edges[1:-1], right=False).astype(int)

    def _build_shape(self, x, residual):
        edges = self._make_edges(x)
        bins = self._digitize(x, edges)
        nb = len(edges) - 1

        vals = np.zeros(nb, dtype=float)
        counts = np.zeros(nb, dtype=float)
        for b in range(nb):
            mask = bins == b
            c = int(np.sum(mask))
            if c > 0:
                vals[b] = float(np.mean(residual[mask]))
                counts[b] = float(c)

        lam = float(max(1e-8, self.shape_shrink))
        vals = vals * (counts / (counts + lam))
        if np.sum(counts) > 0:
            vals = vals - float(np.sum(vals * counts) / (np.sum(counts) + 1e-12))

        phi = vals[bins]
        return edges, vals, phi

    def _eval_spec(self, X, spec):
        kind = spec["kind"]
        j = int(spec["feature"])
        xj = X[:, j]
        if kind == "lin":
            return xj
        if kind == "shape":
            edges = np.asarray(spec["edges"], dtype=float)
            vals = np.asarray(spec["values"], dtype=float)
            bins = self._digitize(xj, edges)
            return vals[bins]
        raise ValueError(f"Unknown term kind: {kind}")

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape

        corr = np.array([abs(self._safe_corr(X[:, j], y)) for j in range(p)], dtype=float)
        k = int(max(1, min(int(self.screen_features), p)))
        feature_pool = np.argsort(-corr)[:k].astype(int).tolist()

        selected_specs = []
        selected_cols = []
        used_keys = set()

        intercept = float(np.mean(y))
        pred = np.full(n, intercept, dtype=float)
        best_obj = float(np.mean((y - pred) ** 2))

        max_terms = int(max(1, self.max_terms))
        for _ in range(max_terms):
            residual = y - pred
            best_next = None

            for j in feature_pool:
                key_lin = ("lin", int(j))
                if key_lin not in used_keys:
                    col = X[:, j]
                    D = np.column_stack([np.ones(n, dtype=float)] + selected_cols + [col])
                    beta = self._fit_ols(D, y)
                    pred_try = D @ beta
                    mse = float(np.mean((y - pred_try) ** 2))
                    obj = mse + float(self.complexity_penalty) * (len(selected_cols) + 1)
                    spec = {"kind": "lin", "feature": int(j)}
                    cand = (obj, key_lin, spec, col)
                    if (best_next is None) or (cand[0] < best_next[0]):
                        best_next = cand

                key_shape = ("shape", int(j))
                if key_shape not in used_keys:
                    edges, vals, phi = self._build_shape(X[:, j], residual)
                    D = np.column_stack([np.ones(n, dtype=float)] + selected_cols + [phi])
                    beta = self._fit_ols(D, y)
                    pred_try = D @ beta
                    mse = float(np.mean((y - pred_try) ** 2))
                    obj = mse + float(self.complexity_penalty) * (len(selected_cols) + 1)
                    spec = {
                        "kind": "shape",
                        "feature": int(j),
                        "edges": edges.tolist(),
                        "values": vals.tolist(),
                    }
                    cand = (obj, key_shape, spec, phi)
                    if (best_next is None) or (cand[0] < best_next[0]):
                        best_next = cand

            if best_next is None:
                break

            gain = best_obj - best_next[0]
            if gain < float(self.min_improvement):
                break

            best_obj = best_next[0]
            used_keys.add(best_next[1])
            selected_specs.append(best_next[2])
            selected_cols.append(best_next[3])

            D_curr = np.column_stack([np.ones(n, dtype=float)] + selected_cols)
            beta_curr = self._fit_ols(D_curr, y)
            pred = D_curr @ beta_curr

        if not selected_cols:
            j = int(np.argmax(corr))
            selected_specs = [{"kind": "lin", "feature": int(j)}]
            selected_cols = [X[:, j]]

        D_final = np.column_stack([np.ones(n, dtype=float)] + selected_cols)
        beta = self._fit_ols(D_final, y)

        raw_w = np.asarray(beta[1:], dtype=float)
        q_w = np.array([self._quantize(w, self.coef_quant_step) for w in raw_w], dtype=float)

        active_idx = [i for i, w in enumerate(q_w) if abs(float(w)) >= float(self.min_abs_coef)]
        if not active_idx:
            j = int(np.argmax(np.abs(raw_w)))
            active_idx = [j]
            q_w[j] = float(np.sign(raw_w[j]) * max(abs(q_w[j]), float(self.min_abs_coef)))

        self.term_specs_ = [selected_specs[i] for i in active_idx]
        self.weights_ = np.array([q_w[i] for i in active_idx], dtype=float)

        contrib = np.zeros(n, dtype=float)
        for w, spec in zip(self.weights_, self.term_specs_):
            contrib += float(w) * self._eval_spec(X, spec)
        self.intercept_ = float(np.mean(y - contrib))

        self.n_features_in_ = int(p)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        X = np.asarray(X, dtype=float)
        pred = np.full(X.shape[0], self.intercept_, dtype=float)
        for w, spec in zip(self.weights_, self.term_specs_):
            pred += float(w) * self._eval_spec(X, spec)
        return pred

    def __str__(self):
        check_is_fitted(self, "is_fitted_")
        lines = ["Sparse Bin-Shape Regressor:"]
        lines.append("  prediction = intercept + sum_k w_k * phi_k(x)")
        lines.append(f"  intercept: {self.intercept_:+.4f}")

        if len(self.weights_) == 0:
            lines.append("  (no active terms)")
            return "\n".join(lines)

        order = np.argsort(-np.abs(self.weights_))
        lines.append("  terms (sorted by |weight|):")
        for rank, idx in enumerate(order, 1):
            w = float(self.weights_[idx])
            spec = self.term_specs_[idx]
            j = int(spec["feature"])
            if spec["kind"] == "lin":
                lines.append(f"  {rank}. {w:+.4f} * x{j}")
            else:
                vals = [float(v) for v in spec["values"]]
                lo = [float(v) for v in spec["edges"][:-1]]
                hi = [float(v) for v in spec["edges"][1:]]
                piece = []
                for a, b, v in zip(lo, hi, vals):
                    left = "-inf" if not np.isfinite(a) else f"{a:.2f}"
                    right = "+inf" if not np.isfinite(b) else f"{b:.2f}"
                    piece.append(f"[{left},{right}):{v:+.2f}")
                lines.append(f"  {rank}. {w:+.4f} * shape(x{j}) with bins " + "; ".join(piece))

        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparseBinShapeRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SparseBinShape_v1"
model_description = "Greedy sparse additive equation mixing linear and quantile-bin shape terms with quantized coefficients"
model_defs = [(model_shorthand_name, SparseBinShapeRegressor())]


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
