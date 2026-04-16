"""Run just the 2 newly added simulatability tests and append to existing results."""
import csv
import sys
import os
import time

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "evolve", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from evaluate_new_generalization import (
    new_simulatability_triple_interaction,
    new_simulatability_quadratic_counterfactual,
    load_all_models,
    BASELINE_DEFS,
    BASELINE_DESCRIPTIONS,
    _suite,
    _run_one_new_test,
)
from pathlib import Path

NEW_RESULTS_DIR = Path(__file__).parent / "new_results"

def main():
    t0 = time.time()

    # Load models
    print("Loading models...")
    model_defs_full = load_all_models()
    for bname, bmodel in BASELINE_DEFS:
        model_defs_full.append((bname, bmodel, BASELINE_DESCRIPTIONS.get(bname, bname)))
    model_defs_simple = [(name, model) for name, model, _ in model_defs_full]
    print(f"Total models: {len(model_defs_simple)}")

    # Run only the 2 new tests
    new_tests = [
        new_simulatability_triple_interaction,
        new_simulatability_quadratic_counterfactual,
    ]

    print(f"\nRunning {len(new_tests)} tests on {len(model_defs_simple)} models...")
    results = []
    for name, reg in model_defs_simple:
        for test_fn in new_tests:
            print(f"  {name} / {test_fn.__name__}...", end=" ", flush=True)
            r = _run_one_new_test(name, test_fn.__name__, reg)
            status = "PASS" if r["passed"] else "FAIL"
            print(status)
            results.append(r)

    # Append to existing interpretability_results.csv
    interp_csv = str(NEW_RESULTS_DIR / "interpretability_results.csv")
    interp_fields = ["model", "test", "suite", "passed", "ground_truth", "response"]

    # Read existing rows and filter out any old runs of these 2 tests
    new_test_names = {t.__name__ for t in new_tests}
    existing = []
    if os.path.exists(interp_csv):
        with open(interp_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row["test"] not in new_test_names:
                    existing.append(row)

    # Add new results
    new_rows = [{
        "model": r["model"],
        "test": r["test"],
        "suite": _suite(r["test"]),
        "passed": r["passed"],
        "ground_truth": r.get("ground_truth", ""),
        "response": r.get("response", ""),
    } for r in results]

    with open(interp_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=interp_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing + new_rows)

    # Recompute overall_results.csv
    # Read all interp results
    all_interp = {}
    with open(interp_csv, newline="") as f:
        for row in csv.DictReader(f):
            model = row["model"]
            if model not in all_interp:
                all_interp[model] = {"passed": 0, "total": 0}
            all_interp[model]["total"] += 1
            if row["passed"] == "True":
                all_interp[model]["passed"] += 1

    # Read existing overall_results.csv and update interp scores
    overall_csv = str(NEW_RESULTS_DIR / "overall_results.csv")
    overall_rows = []
    with open(overall_csv, newline="") as f:
        for row in csv.DictReader(f):
            model = row["model_name"]
            if model in all_interp:
                d = all_interp[model]
                row["frac_interpretability_tests_passed"] = f"{d['passed'] / d['total']:.4f}"
            overall_rows.append(row)

    overall_fields = list(overall_rows[0].keys())
    with open(overall_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=overall_fields)
        writer.writeheader()
        writer.writerows(overall_rows)

    # Print summary
    n_pass = sum(1 for r in results if r["passed"])
    print(f"\n{'='*40}")
    print(f"New tests: {n_pass}/{len(results)} passed")
    print(f"Updated {interp_csv}")
    print(f"Updated {overall_csv}")
    print(f"Time: {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()
