"""Smoke test: fit/predict/str each model on a small synthetic dataset."""

from __future__ import annotations

import time
import traceback

import numpy as np
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import agentic_imodels as ai


def run(cls):
    X, y = make_regression(n_samples=200, n_features=6, n_informative=4,
                            noise=0.3, random_state=0)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0)
    t0 = time.time()
    model = cls()
    model.fit(X_tr, y_tr)
    yhat = model.predict(X_te)
    s = str(model)
    r2 = r2_score(y_te, yhat)
    return r2, len(s.splitlines()), time.time() - t0


def main() -> None:
    for name in ai.__all__:
        cls = getattr(ai, name)
        try:
            r2, nlines, dt = run(cls)
            print(f"OK   {name:40s}  R^2={r2:+.3f}  str={nlines:3d} lines  {dt:5.1f}s")
        except Exception as exc:
            print(f"FAIL {name:40s}  {type(exc).__name__}: {exc}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
