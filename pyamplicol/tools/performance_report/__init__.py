# SPDX-License-Identifier: 0BSD
"""Three-mode performance-report support.

The package deliberately keeps measurement, storage, scheduling, and rendering
separate. ``src/pyamplicol/_profiling_campaign/result_tables.py`` is the
canonical command entry point.
"""

from .catalog import MADGRAPH_FULL_COMPARISON_VIEWS, REPORT_CATALOG, ReportCatalog
from .models import (
    Accuracy,
    ArtifactPolicy,
    CellSpec,
    ExecutionMode,
    MatrixComparisonView,
    ModelKey,
    ResultStatus,
    Workload,
)

__all__ = [
    "MADGRAPH_FULL_COMPARISON_VIEWS",
    "REPORT_CATALOG",
    "Accuracy",
    "ArtifactPolicy",
    "CellSpec",
    "ExecutionMode",
    "MatrixComparisonView",
    "ModelKey",
    "ReportCatalog",
    "ResultStatus",
    "Workload",
]
