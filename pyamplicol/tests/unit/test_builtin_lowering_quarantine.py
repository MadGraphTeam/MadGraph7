# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"


def _assert_isolated_import_does_not_load_builtin(statement: str) -> None:
    script = "\n".join(
        (
            "import sys",
            f"sys.path.insert(0, {str(SOURCE_ROOT)!r})",
            statement,
            "loaded = sorted(name for name in sys.modules ",
            "    if name == 'pyamplicol.models.builtin' ",
            "    or name.startswith('pyamplicol.models.builtin.'))",
            "assert loaded == [], loaded",
        )
    )
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    "statement",
    (
        "import pyamplicol.generation",
        "import pyamplicol.models.base",
        "import pyamplicol.models.expressions",
    ),
)
def test_generic_lowering_imports_do_not_load_builtin_physics(statement: str) -> None:
    _assert_isolated_import_does_not_load_builtin(statement)


def test_builtin_lowering_diagnostics_are_not_generic_model_contracts() -> None:
    from pyamplicol.models import base, expressions
    from pyamplicol.models.builtin import lowering_types

    assert hasattr(lowering_types, "SymbolicLoweringReport")
    assert not hasattr(base.Model, "build_tensor_library")
    assert not hasattr(base.Model, "vertex_lowering_coverage")
    assert not hasattr(base, "VertexLoweringCoverageEntry")
    assert not hasattr(base, "VertexLoweringCoverageReport")
    assert not hasattr(expressions, "_flat_index")
    assert not hasattr(expressions, "_index_chirality")
    assert not hasattr(expressions, "_expr_vector_slash_terms")


def test_builtin_auxiliary_tensor_probe_executes() -> None:
    from pyamplicol.models import BuiltinSMModel
    from pyamplicol.models.builtin.lowering_tensor import (
        _build_auxiliary_tensor_probe,
    )

    probe = _build_auxiliary_tensor_probe(BuiltinSMModel())

    assert probe.engine == "spenso"
    assert probe.output_rank == 2
    assert probe.output_size == 16
    assert probe.nonzero_entries == 4
    assert probe.max_abs_entry == pytest.approx(1.5)
    assert probe.weighted_checksum == pytest.approx((0.0, 48.0))
