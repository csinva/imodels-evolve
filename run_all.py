"""
Run ALL interpretability tests + TabArena classification evaluation on a unified model set.
Produces a scatter plot of interpretability score vs mean performance rank.

Usage: uv run run_all.py
"""

import json
import time
from copy import deepcopy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import imodelsx.llm
from interpretability import (
    ALL_TESTS, HARD_TESTS, INSIGHT_TESTS, _HAS_IMODELS, _HAS_PYGAM,
    _run_test_list,
)
from prepare import get_all_datasets

# ---------------------------------------------------------------------------
# Model registry: (name, regressor_for_interp, classifier_for_tabarena | None)
# ---------------------------------------------------------------------------

MODEL_DEFS = []

if _HAS_PYGAM:
    from pygam import LinearGAM
    MODEL_DEFS.append(("GAM", LinearGAM(n_splines=10), None))  # no classifier

MODEL_DEFS += [
    ("DT_shallow",
     DecisionTreeRegressor(max_depth=2, random_state=42),
     DecisionTreeClassifier(max_depth=2, random_state=42)),
    ("DT_deep",
     DecisionTreeRegressor(max_depth=8, random_state=42),
     DecisionTreeClassifier(max_depth=8, random_state=42)),
    ("OLS",
     LinearRegression(),
     Pipeline([("scaler", StandardScaler()),
               ("clf", LogisticRegression(max_iter=200, random_state=42))])),
    ("LASSO",
     Lasso(alpha=0.1),
     Pipeline([("scaler", StandardScaler()),
               ("clf", LogisticRegression(penalty="l1", solver="saga", C=10.0,
                                          max_iter=200, random_state=42))])),
    ("RF",
     RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42),
     RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)),
    ("GBM",
     GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
     GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)),
    ("MLP",
     MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=1000,
                  random_state=42, learning_rate_init=0.01),
     MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000,
                   random_state=42, learning_rate_init=0.01)),
]

if _HAS_IMODELS:
    from imodels import (
        FIGSClassifier, FIGSRegressor,
        HSTreeClassifier, HSTreeRegressor,
        RuleFitClassifier, RuleFitRegressor,
        TreeGAMClassifier, TreeGAMRegressor,
    )
    MODEL_DEFS += [
        ("FIGS",
         FIGSRegressor(max_rules=12),
         FIGSClassifier(max_rules=12)),
        ("RuleFit",
         RuleFitRegressor(max_rules=20, random_state=42),
         RuleFitClassifier(max_rules=20, random_state=42)),
        ("HSTree",
         HSTreeRegressor(max_leaf_nodes=16, random_state=42),
         HSTreeClassifier(max_leaf_nodes=16, random_state=42)),
        ("TreeGAM",
         TreeGAMRegressor(n_boosting_rounds=5, max_leaf_nodes=4, random_state=42),
         TreeGAMClassifier(n_boosting_rounds=5, max_leaf_nodes=4, random_state=42)),
    ]


# ---------------------------------------------------------------------------
# Step 1 — interpretability
# ---------------------------------------------------------------------------

def run_all_interp_tests(model_defs, llm):
    """Run standard + hard + insight tests on all regressors."""
    all_results = []
    for name, reg, _ in model_defs:
        print(f"\n{'='*60}")
        print(f"  Model (interp): {name}")
        print("=" * 60)
        for test_list, label in [
            (ALL_TESTS,     "standard"),
            (HARD_TESTS,    "hard"),
            (INSIGHT_TESTS, "insight"),
        ]:
            results = _run_test_list(reg, name, llm, test_list, label)
            all_results.extend(results)
    return all_results


# ---------------------------------------------------------------------------
# Step 2 — TabArena classification
# ---------------------------------------------------------------------------

def evaluate_all_classifiers(model_defs):
    """Evaluate all classifiers on every TabArena dataset.

    Returns:
        dataset_aucs : {dataset_name: {model_name: auc}}
    """
    classifier_pairs = [(name, clf) for name, _, clf in model_defs if clf is not None]
    dataset_aucs = {}

    for ds_name, X_train, X_test, y_train, y_test in get_all_datasets():
        print(f"\n  Dataset: {ds_name} — {X_train.shape[1]} features, "
              f"{len(np.unique(y_train))} classes")
        n_classes = len(np.unique(y_train))
        dataset_aucs[ds_name] = {}

        for name, clf in classifier_pairs:
            try:
                m = deepcopy(clf)
                m.fit(X_train, y_train)
                if n_classes == 2:
                    proba = m.predict_proba(X_test)[:, 1]
                    auc = roc_auc_score(y_test, proba)
                else:
                    proba = m.predict_proba(X_test)
                    auc = roc_auc_score(y_test, proba,
                                        multi_class="ovr", average="macro")
                dataset_aucs[ds_name][name] = float(auc)
                print(f"    {name:<15}: {auc:.4f}")
            except Exception as e:
                print(f"    {name:<15}: ERROR — {e}")
                dataset_aucs[ds_name][name] = float("nan")

    return dataset_aucs


def compute_rank_scores(dataset_aucs):
    """For each dataset rank models by AUC (1 = best), then average ranks."""
    all_model_names = set()
    for d in dataset_aucs.values():
        all_model_names.update(d.keys())

    ranks_per_model = {n: [] for n in all_model_names}
    mean_auc_per_model = {n: [] for n in all_model_names}

    for ds_name, model_aucs in dataset_aucs.items():
        valid = [(n, v) for n, v in model_aucs.items() if not np.isnan(v)]
        sorted_models = sorted(valid, key=lambda x: x[1], reverse=True)
        rank_map = {n: r + 1 for r, (n, _) in enumerate(sorted_models)}
        for name in all_model_names:
            if name in model_aucs and not np.isnan(model_aucs[name]):
                ranks_per_model[name].append(rank_map[name])
                mean_auc_per_model[name].append(model_aucs[name])

    avg_rank = {n: float(np.mean(v)) for n, v in ranks_per_model.items() if v}
    avg_auc  = {n: float(np.mean(v)) for n, v in mean_auc_per_model.items() if v}
    return avg_rank, avg_auc


# ---------------------------------------------------------------------------
# Step 3 — Plot
# ---------------------------------------------------------------------------

MODEL_GROUPS = {
    "black-box":   {"RF", "GBM", "MLP"},
    "imodels":     {"FIGS", "RuleFit", "HSTree", "TreeGAM"},
    "linear":      {"OLS", "LASSO"},
    "tree":        {"DT_shallow", "DT_deep"},
    "gam":         {"GAM"},
}
GROUP_COLORS = {
    "black-box": "#e74c3c",
    "imodels":   "#27ae60",
    "linear":    "#2980b9",
    "tree":      "#e67e22",
    "gam":       "#8e44ad",
}


def _model_color(name):
    for group, members in MODEL_GROUPS.items():
        if name in members:
            return GROUP_COLORS[group]
    return "#7f8c8d"


def plot_interp_vs_rank(interp_scores, avg_rank, avg_auc, out_path):
    """Scatter: x=avg performance rank (lower=better), y=interpretability score."""
    names = [n for n in interp_scores if n in avg_rank]
    x = [avg_rank[n] for n in names]       # lower rank = better performance
    y = [interp_scores[n] for n in names]  # fraction of 18 tests passed
    colors = [_model_color(n) for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # --- Left: interpretability vs rank ---
    ax = axes[0]
    ax.scatter(x, y, c=colors, s=140, zorder=5,
               edgecolors="white", linewidths=0.8)
    for name, xi, yi in zip(names, x, y):
        ax.annotate(name, (xi, yi), textcoords="offset points",
                    xytext=(5, 4), fontsize=8.5)
    ax.set_xlabel("Mean Performance Rank on TabArena\n(lower = better classification)", fontsize=10)
    ax.set_ylabel("Interpretability Score\n(fraction of 18 tests passed)", fontsize=10)
    ax.set_title("Interpretability vs. Performance Rank", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()  # leftmost = best rank

    # --- Right: interpretability vs AUC ---
    names2 = [n for n in interp_scores if n in avg_auc]
    x2 = [avg_auc[n] for n in names2]
    y2 = [interp_scores[n] for n in names2]
    c2 = [_model_color(n) for n in names2]
    ax2 = axes[1]
    ax2.scatter(x2, y2, c=c2, s=140, zorder=5,
                edgecolors="white", linewidths=0.8)
    for name, xi, yi in zip(names2, x2, y2):
        ax2.annotate(name, (xi, yi), textcoords="offset points",
                     xytext=(5, 4), fontsize=8.5)
    ax2.set_xlabel("Mean AUC on TabArena\n(higher = better)", fontsize=10)
    ax2.set_ylabel("Interpretability Score\n(fraction of 18 tests passed)", fontsize=10)
    ax2.set_title("Interpretability vs. Mean AUC", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    # Shared legend
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=GROUP_COLORS[g], markersize=9,
               label=g.replace("-", " ").title())
        for g in GROUP_COLORS
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=5, fontsize=9, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Interpretability–Performance Trade-off", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()

    # ── Step 1: interpretability ──────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 1 — Interpretability tests (all 18 per model)")
    print("=" * 65)
    llm = imodelsx.llm.get_llm("gpt-4o")
    interp_results = run_all_interp_tests(MODEL_DEFS, llm)

    model_names_interp = list(dict.fromkeys(r["model"] for r in interp_results))
    interp_scores = {}
    for mname in model_names_interp:
        subset = [r for r in interp_results if r["model"] == mname]
        n_passed = sum(r["passed"] for r in subset)
        interp_scores[mname] = n_passed / len(subset)

    print("\n\nInterpretability scores (all tests):")
    for name, score in sorted(interp_scores.items(), key=lambda x: -x[1]):
        n = sum(r["passed"] for r in interp_results if r["model"] == name)
        total = sum(1 for r in interp_results if r["model"] == name)
        std  = sum(r["passed"] for r in interp_results
                   if r["model"] == name and r["test"] in
                   {t.__name__ for t in ALL_TESTS})
        hard = sum(r["passed"] for r in interp_results
                   if r["model"] == name and r["test"] in
                   {t.__name__ for t in HARD_TESTS})
        ins  = sum(r["passed"] for r in interp_results
                   if r["model"] == name and r["test"] in
                   {t.__name__ for t in INSIGHT_TESTS})
        print(f"  {name:<15}: {n:2d}/{total} ({score:.2%})  "
              f"[std {std}/8  hard {hard}/5  insight {ins}/5]")

    # ── Step 2: TabArena ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 2 — TabArena classification evaluation")
    print("=" * 65)
    dataset_aucs = evaluate_all_classifiers(MODEL_DEFS)
    avg_rank, avg_auc = compute_rank_scores(dataset_aucs)

    print("\n\nTabArena summary (sorted by avg rank):")
    for name, rank in sorted(avg_rank.items(), key=lambda x: x[1]):
        print(f"  {name:<15}: avg_rank={rank:.2f}  mean_auc={avg_auc.get(name, float('nan')):.4f}")

    # ── Save results ─────────────────────────────────────────────────────
    all_scores = {
        "interpretability": interp_scores,
        "tabarena_avg_rank": avg_rank,
        "tabarena_mean_auc": avg_auc,
        "tabarena_per_dataset": dataset_aucs,
    }
    with open("all_scores.json", "w") as f:
        json.dump(all_scores, f, indent=2)
    print("\nScores saved → all_scores.json")

    # ── Step 3: Plot ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 3 — Plotting")
    print("=" * 65)
    plot_interp_vs_rank(interp_scores, avg_rank, avg_auc,
                        "interpretability_vs_performance.png")

    print(f"\nTotal time: {time.time() - t0:.1f}s")
