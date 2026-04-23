"""agentic-imodels — interpretable tabular regressors discovered via an agentic loop."""

from importlib.metadata import version as _pkg_version

__all__ = [
    "HingeEBMRegressor",
    "HybridGAM",
    "SmartAdditiveRegressor",
    "HingeGAMRegressor",
    "TeacherStudentRuleSplineRegressor",
    "DualPathSparseSymbolicRegressor",
    "SparseSignedBasisPursuitRegressor",
    "DistilledTreeBlendAtlasRegressor",
    "WinsorizedSparseOLSRegressor",
    "TinyDTDepth2Regressor",
]

from .hinge_ebm import HingeEBMRegressor
from .hybrid_gam import HybridGAM
from .smooth_additive_gam import SmartAdditiveRegressor
from .hinge_gam import HingeGAMRegressor
from .teacher_student_rule_spline import TeacherStudentRuleSplineRegressor
from .dual_path_sparse_symbolic import DualPathSparseSymbolicRegressor
from .sparse_signed_basis_pursuit import SparseSignedBasisPursuitRegressor
from .distilled_tree_blend_atlas import DistilledTreeBlendAtlasRegressor
from .winsorized_sparse_ols import WinsorizedSparseOLSRegressor
from .tiny_dt import TinyDTDepth2Regressor

try:
    __version__ = _pkg_version("agentic-imodels")
except Exception:
    __version__ = "0.0.0"
