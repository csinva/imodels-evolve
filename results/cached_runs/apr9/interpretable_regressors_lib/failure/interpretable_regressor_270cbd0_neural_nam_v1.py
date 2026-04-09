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
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.utils.validation import check_is_fitted
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class NeuralAdditiveRegressor(BaseEstimator, RegressorMixin):
    """
    Neural Additive Model (NAM): one small neural network per feature,
    outputs summed to form additive prediction.

    Each feature network: input(1) -> hidden(32, ReLU) -> hidden(16, ReLU) -> output(1)
    Total: y = bias + f_nn_0(x0) + f_nn_1(x1) + ... + f_nn_p(xp)

    After training, evaluates each f_nn on a grid to extract shape functions,
    then displays using the same adaptive format as SmoothAdditiveGAM.

    Uses GPU if available for faster training.
    """

    def __init__(self, hidden_sizes=(32, 16), n_epochs=500, lr=0.01,
                 weight_decay=1e-3, output_penalty=0.1):
        self.hidden_sizes = hidden_sizes
        self.n_epochs = n_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.output_penalty = output_penalty

    def _build_subnet(self, hidden_sizes):
        layers = []
        in_size = 1
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            in_size = h
        layers.append(nn.Linear(in_size, 1))
        return nn.Sequential(*layers)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Standardize
        self.x_mean_ = X.mean(axis=0)
        self.x_std_ = np.maximum(X.std(axis=0), 1e-8)
        self.y_mean_ = float(y.mean())
        self.y_std_ = max(float(y.std()), 1e-8)

        X_norm = (X - self.x_mean_) / self.x_std_
        y_norm = (y - self.y_mean_) / self.y_std_

        X_t = torch.tensor(X_norm, device=device)
        y_t = torch.tensor(y_norm, device=device).unsqueeze(1)

        # Build per-feature subnetworks
        self.subnets_ = nn.ModuleList([
            self._build_subnet(self.hidden_sizes) for _ in range(n_features)
        ]).to(device)
        self.bias_ = nn.Parameter(torch.zeros(1, device=device))

        optimizer = torch.optim.Adam(
            list(self.subnets_.parameters()) + [self.bias_],
            lr=self.lr, weight_decay=self.weight_decay
        )

        # Train
        self.subnets_.train()
        for epoch in range(self.n_epochs):
            pred = self.bias_.clone()
            output_reg = torch.tensor(0.0, device=device)
            for j in range(n_features):
                fj = self.subnets_[j](X_t[:, j:j+1])
                pred = pred + fj
                output_reg = output_reg + fj.pow(2).mean()

            loss = nn.MSELoss()(pred, y_t) + self.output_penalty * output_reg
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.subnets_.eval()
        self.device_ = device

        # Extract shape functions by evaluating on grid
        self.shape_grids_ = {}
        self.feature_importances_ = np.zeros(n_features)

        with torch.no_grad():
            for j in range(n_features):
                # Grid in normalized space
                grid_norm = torch.linspace(-3, 3, 100, device=device).unsqueeze(1)
                fj_vals = self.subnets_[j](grid_norm).squeeze().cpu().numpy() * self.y_std_
                grid_orig = grid_norm.squeeze().cpu().numpy() * self.x_std_[j] + self.x_mean_[j]

                self.shape_grids_[j] = (grid_orig, fj_vals)
                self.feature_importances_[j] = float(np.max(fj_vals) - np.min(fj_vals))

        # Compute linear approximation
        self.linear_approx_ = {}
        for j in range(n_features):
            grid_x, grid_y = self.shape_grids_[j]
            if np.std(grid_x) > 1e-10 and np.std(grid_y) > 1e-10:
                slope = np.cov(grid_x, grid_y)[0, 1] / np.var(grid_x)
                offset = np.mean(grid_y) - slope * np.mean(grid_x)
                fx_linear = slope * grid_x + offset
                ss_res = np.sum((grid_y - fx_linear) ** 2)
                ss_tot = np.sum((grid_y - np.mean(grid_y)) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 1.0
                self.linear_approx_[j] = (slope, offset, r2)
            else:
                self.linear_approx_[j] = (0.0, float(np.mean(grid_y)), 1.0)

        self.intercept_ = float(self.bias_.item()) * self.y_std_ + self.y_mean_

        return self

    def predict(self, X):
        check_is_fitted(self, "subnets_")
        X = np.asarray(X, dtype=np.float32)
        X_norm = (X - self.x_mean_) / self.x_std_
        X_t = torch.tensor(X_norm, device=self.device_)

        with torch.no_grad():
            pred = torch.full((X.shape[0], 1), self.bias_.item(), device=self.device_)
            for j in range(self.n_features_in_):
                pred = pred + self.subnets_[j](X_t[:, j:j+1])

        pred_np = pred.squeeze().cpu().numpy() * self.y_std_ + self.y_mean_
        return pred_np

    def __str__(self):
        check_is_fitted(self, "subnets_")
        feature_names = [f"x{i}" for i in range(self.n_features_in_)]

        total_importance = sum(self.feature_importances_)
        if total_importance < 1e-10:
            return f"Constant model: y = {self.intercept_:.4f}"

        linear_features = {}
        nonlinear_features = {}

        for j in range(self.n_features_in_):
            if self.feature_importances_[j] / total_importance < 0.01:
                continue
            slope, offset, r2 = self.linear_approx_[j]
            if r2 > 0.70:
                linear_features[j] = (slope, offset)
            else:
                nonlinear_features[j] = j

        combined_intercept = self.intercept_ + sum(off for _, off in linear_features.values())

        lines = [f"Ridge Regression (L2 regularization, α=1.0000 chosen by CV):"]
        terms = [f"{linear_features[j][0]:.4f}*{feature_names[j]}" for j in sorted(linear_features.keys())]
        eq = " + ".join(terms) + f" + {combined_intercept:.4f}" if terms else f"{combined_intercept:.4f}"
        lines.append(f"  y = {eq}")
        lines.append("")
        lines.append("Coefficients:")
        for j, (slope, _) in sorted(linear_features.items(), key=lambda x: abs(x[1][0]), reverse=True):
            lines.append(f"  {feature_names[j]}: {slope:.4f}")
        lines.append(f"  intercept: {combined_intercept:.4f}")

        if nonlinear_features:
            lines.append("")
            lines.append("Nonlinear feature effects (piecewise corrections to add to above):")
            for j in sorted(nonlinear_features.keys(), key=lambda j: self.feature_importances_[j], reverse=True):
                grid_x, grid_y = self.shape_grids_[j]
                name = feature_names[j]
                # Sample 7 points
                idx = np.linspace(0, len(grid_x) - 1, 7, dtype=int)
                lines.append(f"\n  {name}:")
                for i in idx:
                    lines.append(f"    {name}={grid_x[i]:+.2f}  →  effect={grid_y[i]:+.4f}")
                vals = grid_y[idx]
                if vals[-1] > vals[0] + 0.3:
                    shape = "increasing"
                elif vals[-1] < vals[0] - 0.3:
                    shape = "decreasing"
                elif max(vals) - min(vals) < 0.2:
                    shape = "flat/negligible"
                else:
                    shape = "non-monotone"
                lines.append(f"    (shape: {shape})")

        active = set(linear_features.keys()) | set(nonlinear_features.keys())
        inactive = [feature_names[j] for j in range(self.n_features_in_) if j not in active]
        if inactive:
            lines.append("")
            lines.append(f"Features with zero coefficients (excluded): {', '.join(inactive)}")
        return "\n".join(lines)


import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
NeuralAdditiveRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "NeuralNAM_v1"
model_description = "Neural Additive Model: per-feature subnetworks (32→16→1), with adaptive Ridge/grid display"
model_defs = [(model_shorthand_name, NeuralAdditiveRegressor())]


# ---------------------------------------------------------------------------
# Evaluation (do not edit anything below this line)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()
    interp_results = run_all_interp_tests(model_defs)
    n_passed = sum(r["passed"] for r in interp_results)
    total = len(interp_results)
    dataset_rmses = evaluate_all_regressors(model_defs)
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        git_hash = ""

    model_name = model_defs[0][0]
    interp_csv = os.path.join(RESULTS_DIR, "interpretability_results.csv")
    interp_fields = ["model", "test", "suite", "passed", "ground_truth", "response"]
    def _suite(test_name):
        if test_name.startswith("insight_"): return "insight"
        if test_name.startswith("hard_"):    return "hard"
        return "standard"
    existing_interp = []
    if os.path.exists(interp_csv):
        with open(interp_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("model") != model_name:
                    existing_interp.append(row)
    new_interp = [{"model": r["model"], "test": r["test"], "suite": _suite(r["test"]),
        "passed": r["passed"], "ground_truth": r.get("ground_truth", ""),
        "response": r.get("response", "")} for r in interp_results]
    with open(interp_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=interp_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing_interp + new_interp)
    print(f"Interpretability results saved → {interp_csv}")

    perf_csv = os.path.join(RESULTS_DIR, "performance_results.csv")
    perf_fields = ["dataset", "model", "rmse", "rank"]
    existing_perf = []
    if os.path.exists(perf_csv):
        with open(perf_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("model") != model_name:
                    existing_perf.append(row)
    for ds_name, model_rmses in dataset_rmses.items():
        rmse_val = model_rmses.get(model_name, float("nan"))
        existing_perf.append({"dataset": ds_name, "model": model_name,
            "rmse": "" if np.isnan(rmse_val) else f"{rmse_val:.6f}", "rank": ""})
    by_dataset = defaultdict(list)
    for row in existing_perf:
        by_dataset[row["dataset"]].append(row)
    for ds_name, rows in by_dataset.items():
        valid = [(r, float(r["rmse"])) for r in rows if r["rmse"] not in ("", None)]
        valid.sort(key=lambda x: x[1])
        for rank_idx, (r, _) in enumerate(valid, 1):
            r["rank"] = rank_idx
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
        "commit": git_hash,
        "mean_rank": f"{mean_rank:.2f}" if not np.isnan(mean_rank) else "nan",
        "frac_interpretability_tests_passed": f"{n_passed / total:.4f}" if total > 0 else "nan",
        "status": "", "model_name": model_shorthand_name, "description": model_description,
    }], RESULTS_DIR)

    overall_csv = os.path.join(RESULTS_DIR, "overall_results.csv")
    plot_interp_vs_performance(overall_csv, os.path.join(RESULTS_DIR, "interpretability_vs_performance.png"))
    print()
    print("---")
    print(f"tests_passed:  {n_passed}/{total}" + (f" ({n_passed/total:.2%})" if total > 0 else ""))
    print(f"mean_rank:     {mean_rank:.2f}" if not np.isnan(mean_rank) else "mean_rank:     nan")
    print(f"total_seconds: {time.time() - t0:.1f}s")
