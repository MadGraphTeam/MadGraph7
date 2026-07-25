# SPDX-License-Identifier: 0BSD
"""Three-mode performance-report support.

The package deliberately keeps measurement, storage, scheduling, and rendering
separate.  ``docs/result_tables.py`` remains the stable command entry point.
"""

from .catalog import REPORT_CATALOG, ReportCatalog
from .models import (
    Accuracy,
    ArtifactPolicy,
    CellSpec,
    ExecutionMode,
    ModelKey,
    ResultStatus,
    Workload,
)

__all__ = [
    "REPORT_CATALOG",
    "Accuracy",
    "ArtifactPolicy",
    "CellSpec",
    "ExecutionMode",
    "ModelKey",
    "ReportCatalog",
    "ResultStatus",
    "Workload",
]
