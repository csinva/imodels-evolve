# agentic-imodels

Interpretable tabular regressors discovered by agentic loops (Claude Code /
OpenAI Codex) — a sibling of the human-designed
[`imodels`](https://github.com/csinva/imodels) library.

Each model in this package was produced by `evolve`, an autonomous loop that
repeatedly rewrites a scikit-learn-compatible `InterpretableRegressor` class
to improve two metrics:

- **Predictive performance** — RMSE rank averaged across 65 tabular
  regression datasets (TabArena + PMLB).
- **Interpretability** — fraction of LLM-graded tests passed, covering
  feature attribution, point simulation, sensitivity, counterfactual,
  structural, and complex-function questions.

From the ~700 evolved regressors that survived evaluation, we curated
**10 models** that span different architectural ideas and different points
on the interpretability–performance trade-off.

## Installation

```bash
uv add agentic-imodels
# or
pip install agentic-imodels
```

The package depends on `numpy`, `scikit-learn`, and `interpret` (for the
EBM-backed models).

## Quick start

Every model is a scikit-learn regressor: `fit` on `(X, y)`, call `predict`,
and `print` / `str` it to see its interpretable form.

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from agentic_imodels import HingeEBMRegressor

X, y = fetch_california_housing(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0)

model = HingeEBMRegressor()
model.fit(X_tr, y_tr)

print(model)          # human-readable equation / tree / rule set
y_hat = model.predict(X_te)
```

All estimators follow the sklearn `BaseEstimator + RegressorMixin` contract,
so they work inside `Pipeline`, `cross_val_score`, `GridSearchCV`, etc.

## Available models

Metrics below: **rank** = mean global RMSE rank across 65 dev datasets
(lower is better). **dev interp** = fraction passed on the 43-test
development suite. **test interp** = fraction passed on the held-out
157-test generalization suite. For reference, the most predictive baseline
(TabPFN) has rank 94.5 / test interp 0.17, and the most interpretable
baseline (OLS) has rank 354.5 / test interp 0.69.

| Class | Shorthand | Rank ↓ | Dev interp ↑ | Test interp ↑ | Idea |
| --- | --- | ---: | ---: | ---: | --- |
| `HingeEBMRegressor` | HingeEBM_5bag | 108.2 | 0.651 | 0.707 | Lasso on hinge basis + hidden EBM on residuals. Display is a sparse linear equation. |
| `DistilledTreeBlendAtlasRegressor` | DistilledTreeBlendAtlas_v1 | 139.7 | 1.000 | 0.707 | Ridge student distilled from GBM+RF teachers, shown with a probe-answer "atlas" card. |
| `DualPathSparseSymbolicRegressor` | DualPathSparseSymbolic_v2 | 163.5 | 0.698 | 0.713 | Blended GBM/RF/Ridge teacher for batch `predict`, sparse forward-selected symbolic equation for display. |
| `HybridGAM` | HybridGAM_v9 | 163.8 | 0.721 | 0.675 | SmartAdditiveGAM display + hidden RF residual corrector (shrinkage 0.7). |
| `TeacherStudentRuleSplineRegressor` | TeacherStudentRuleSpline_v1 | 204.0 | 0.605 | **0.803** | GBM teacher for batch predict + sparse symbolic student over linear/square/abs/hinge/step/interaction/gated terms. |
| `SparseSignedBasisPursuitRegressor` | SparseSignedBasisPursuit_v1 | 272.7 | 0.674 | 0.758 | Forward-selected signed basis (linear/hinge/square/interaction) + ridge refit + coefficient rounding. |
| `HingeGAMRegressor` | HingeGAM_10bp | 280.2 | 0.558 | 0.783 | Pure Lasso on a 10-breakpoint hinge basis with an adaptive display; predict = display (no hidden corrector). |
| `WinsorizedSparseOLSRegressor` | WinsorizedSparseOLS | 326.9 | 0.651 | 0.726 | Clip features to `[p1, p99]`, LassoCV select top-8, OLS refit — honest sparse linear. |
| `TinyDTDepth2Regressor` | TinyDTDepth2_v1 | 334.0 | 0.674 | 0.713 | Depth-2 decision tree (4 leaves) — simplest truly-tree-based model. |
| `SmartAdditiveRegressor` | SmoothGAM_msl3 | 354.3 | 0.744 | 0.733 | Adaptive-linearization GAM: Laplacian-smoothed boosted stumps per feature, rendered as a linear coefficient when the shape is almost linear, else as a short piecewise table. |

Where to look:

- `HingeEBM` / `HybridGAM` / `DistilledTreeBlendAtlas` / `DualPathSparseSymbolic` /
  `TeacherStudentRuleSpline` use **display-predict decoupling**: a hidden
  corrector (EBM / RF / GBM) improves the numeric prediction while the
  `__str__` output remains a simple, human-readable formula.
- `SmoothGAM` / `HingeGAM` / `WinsorizedSparseOLS` / `SparseSignedBasisPursuit` /
  `TinyDT` are **honest**: `predict` and `__str__` agree — no silent corrector.

Pick a decoupled model when you need the lowest predictive rank; pick an
honest model when the interpretable formula must actually be what the model
computes.

## Provenance

All classes were produced by the `evolve` loop over Apr 9–20, 2026, using
Claude Opus 4.6/4.7 and OpenAI Codex GPT-5.3 as the coding agents. The
exact source file (with commit hash, git lineage, and original `evolve`
harness code) is referenced at the top of each module under
`agentic_imodels/` — you can chase down the original `interpretable_regressor`
script in the parent repo under `result_libs/`.

## Citation

If you use this library, please cite the evolve paper (see
`../paper-imodels-agentic/main.pdf` in the source tree):

> Interpretable Tabular Regressors Discovered by an Agentic Coding Loop.
> 2026.

## License

MIT.
