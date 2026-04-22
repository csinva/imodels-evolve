"""Extract the curated regressor classes from ../result_libs/* into a
clean package at ../agentic-imodels/agentic_imodels/.

For each selected model we:
  * strip evaluation-harness imports (interp_eval/performance_eval/visualize)
  * drop the `sys.path.insert(...)` that points at the evolve src directory
  * truncate before the pickling hack / model_shorthand_name / __main__ block
  * rewrite the module preamble so the class is importable as a library

The public class name is kept identical to the in-file class so users can
look up the original source if needed.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_PKG = ROOT / "agentic-imodels" / "agentic_imodels"
OUT_PKG.mkdir(parents=True, exist_ok=True)


SELECTIONS = [
    # (module_filename, source_relpath, public_class_name, shorthand, rank, dev_interp, test_interp)
    (
        "hinge_ebm",
        "result_libs/apr9-claude-effort=medium-main-result/interpretable_regressors_lib/success/interpretable_regressor_9ad67ad_hinge_ebm_5bag.py",
        "HingeEBMRegressor",
        "HingeEBM_5bag",
        108.21, 0.6512, 0.7070,
    ),
    (
        "hybrid_gam",
        "result_libs/apr20-claude-4.7-effort=medium-rerun4/interpretable_regressors_lib/failure/interpretable_regressor_156c349_HybridGAM_v9.py",
        "HybridGAM",
        "HybridGAM_v9",
        163.83, 0.7209, 0.6752,
    ),
    (
        "smooth_additive_gam",
        "result_libs/apr9-claude-effort=medium-main-result/interpretable_regressors_lib/failure/interpretable_regressor_61e149a_msl3.py",
        "SmartAdditiveRegressor",
        "SmoothGAM_msl3",
        354.32, 0.7442, 0.7325,
    ),
    (
        "hinge_gam",
        "result_libs/apr9-claude-effort=medium-main-result/interpretable_regressors_lib/failure/interpretable_regressor_d551a55_hinge_gam_10bp.py",
        "HingeGAMRegressor",
        "HingeGAM_10bp",
        280.18, 0.5581, 0.7834,
    ),
    (
        "teacher_student_rule_spline",
        "result_libs/apr17-codex-5.3-effort=high/interpretable_regressors_lib/failure/interpretable_regressor_c2b5db4_TeacherStudentRuleSpline_v1.py",
        "TeacherStudentRuleSplineRegressor",
        "TeacherStudentRuleSpline_v1",
        204.03, 0.6047, 0.8025,
    ),
    (
        "dual_path_sparse_symbolic",
        "result_libs/apr17-codex-5.3-effort=high/interpretable_regressors_lib/failure/interpretable_regressor_4c8b421_dualpathsparsesymbolic_v2.py",
        "DualPathSparseSymbolicRegressor",
        "DualPathSparseSymbolic_v2",
        163.50, 0.6977, 0.7134,
    ),
    (
        "sparse_signed_basis_pursuit",
        "result_libs/apr17-codex-5.3-effort=high/interpretable_regressors_lib/success/interpretable_regressor_029630d_sparse_signed_basis_pursuit.py",
        "SparseSignedBasisPursuitRegressor",
        "SparseSignedBasisPursuit_v1",
        272.70, 0.6744, 0.7580,
    ),
    (
        "distilled_tree_blend_atlas",
        "result_libs/apr19-codex-5.3-effort=xhigh/interpretable_regressors_lib/success/interpretable_regressor_d34b7ed_distilledtreeblendatlasapr18aa.py",
        "DistilledTreeBlendAtlasRegressor",
        "DistilledTreeBlendAtlas_v1",
        139.69, 1.0000, 0.7070,
    ),
    (
        "winsorized_sparse_ols",
        "result_libs/apr19-claude-4.7-effort=medium-rerun2/interpretable_regressors_lib/failure/interpretable_regressor_f53ad88_WinsorizedSparseOLS.py",
        "WinsorizedSparseOLSRegressor",
        "WinsorizedSparseOLS",
        326.95, 0.6512, 0.7261,
    ),
    (
        "tiny_dt",
        "result_libs/apr19-claude-4.7-effort=medium-rerun3/interpretable_regressors_lib/failure/interpretable_regressor_fa6e001_TinyDTDepth2_v1.py",
        "TinyDTDepth2Regressor",
        "TinyDTDepth2_v1",
        334.01, 0.6744, 0.7134,
    ),
]

# Lines that should be dropped entirely from the raw source.
SKIP_LINE_PATTERNS = [
    re.compile(r"^\s*sys\.path\.insert\(0,\s*os\.path\.join\(os\.path\.dirname\(__file__\)"),
    re.compile(r"^\s*sys\.path\.insert\(0,"),
    re.compile(r"^\s*from\s+interp_eval\s"),
    re.compile(r"^\s*from\s+performance_eval\s"),
    re.compile(r"^\s*from\s+visualize\s"),
]

# Lines that mark the end of the class definition region. Everything from the
# first matching line to EOF is discarded.
END_PATTERNS = [
    re.compile(r"^# Make class picklable"),
    re.compile(r"^import sys as _sys"),
    re.compile(r"^model_shorthand_name\s*="),
    re.compile(r"^if __name__ ==\s*[\"']__main__[\"']"),
]


def clean_source(src: str) -> str:
    lines = src.splitlines()
    out = []
    for line in lines:
        if any(p.search(line) for p in END_PATTERNS):
            break
        if any(p.search(line) for p in SKIP_LINE_PATTERNS):
            continue
        out.append(line)
    # Trim trailing blank lines.
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out) + "\n"


HEADER_TEMPLATE = '''"""{module_name} — {class_name} from the agentic-imodels library.

Generated from: {src_path}

Shorthand: {shorthand}
Mean global rank (lower is better): {rank:.2f}   (pooled 65 dev datasets)
Interpretability (fraction passed, higher is better):
    dev  (43 tests):  {dev_interp:.3f}
    test (157 tests): {test_interp:.3f}
"""

'''


def build_module(entry: tuple) -> None:
    module_name, src_rel, cls_name, shorthand, rank, dev_i, test_i = entry
    src_path = ROOT / src_rel
    raw = src_path.read_text()
    cleaned = clean_source(raw)

    # Remove the original docstring at the top (lines starting with a triple
    # quote and ending at the next triple quote).
    if cleaned.lstrip().startswith('"""'):
        quote = '"""'
        first_idx = cleaned.index(quote)
        close_idx = cleaned.index(quote, first_idx + 3)
        cleaned = cleaned[close_idx + 3 :].lstrip("\n")

    header = HEADER_TEMPLATE.format(
        module_name=module_name,
        class_name=cls_name,
        src_path=src_rel,
        shorthand=shorthand,
        rank=rank,
        dev_interp=dev_i,
        test_interp=test_i,
    )
    content = header + cleaned
    dest = OUT_PKG / f"{module_name}.py"
    dest.write_text(content)
    print(f"Wrote {dest.relative_to(ROOT)}  (class={cls_name})")


def write_init() -> None:
    lines = ['"""agentic-imodels — interpretable tabular regressors discovered via an agentic loop."""', ""]
    lines.append("from importlib.metadata import version as _pkg_version")
    lines.append("")
    lines.append("__all__ = [")
    for _, _, cls_name, *_ in SELECTIONS:
        lines.append(f'    "{cls_name}",')
    lines.append("]")
    lines.append("")
    for mod, _, cls_name, *_ in SELECTIONS:
        lines.append(f"from .{mod} import {cls_name}")
    lines.append("")
    lines.append("try:")
    lines.append("    __version__ = _pkg_version(\"agentic-imodels\")")
    lines.append("except Exception:")
    lines.append("    __version__ = \"0.0.0\"")
    lines.append("")
    (OUT_PKG / "__init__.py").write_text("\n".join(lines))
    print("Wrote", (OUT_PKG / "__init__.py").relative_to(ROOT))


def main() -> None:
    for entry in SELECTIONS:
        build_module(entry)
    write_init()


if __name__ == "__main__":
    main()
