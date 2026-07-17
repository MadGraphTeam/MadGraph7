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
        "import pyamplicol.generation.lowering",
        "import pyamplicol.generation.lowering_reports",
        "import pyamplicol.generation.lowering_tensor",
        (
            "from pyamplicol.generation.lowering import "
            "ColorAlgebraProbe, RecursionLoweringPlan"
        ),
    ),
)
def test_generic_lowering_imports_do_not_load_builtin_physics(statement: str) -> None:
    _assert_isolated_import_does_not_load_builtin(statement)


def test_generation_lowering_exports_resolve_to_builtin_implementations() -> None:
    from pyamplicol.generation import lowering as lowering_facade
    from pyamplicol.models.builtin import lowering_reports, lowering_tensor

    assert (
        lowering_facade.build_symbolic_lowering_report
        is lowering_reports.build_symbolic_lowering_report
    )
    assert (
        lowering_facade.build_tensor_network_scalar_bundle
        is lowering_tensor.build_tensor_network_scalar_bundle
    )
    assert (
        lowering_facade.build_interleaved_tensor_network_scalar_bundle
        is lowering_tensor.build_interleaved_tensor_network_scalar_bundle
    )
    assert (
        lowering_facade._GraphTensorExpressionBuilder
        is lowering_tensor._GraphTensorExpressionBuilder
    )


def test_legacy_lowering_modules_delegate_public_and_private_names() -> None:
    from pyamplicol.generation import lowering_reports as reports_facade
    from pyamplicol.generation import lowering_tensor as tensor_facade
    from pyamplicol.models.builtin import lowering_reports, lowering_tensor

    assert (
        reports_facade.build_symbolic_lowering_report
        is lowering_reports.build_symbolic_lowering_report
    )
    assert (
        tensor_facade.build_tensor_network_scalar_bundle
        is lowering_tensor.build_tensor_network_scalar_bundle
    )
    assert (
        tensor_facade._build_auxiliary_tensor_probe
        is lowering_tensor._build_auxiliary_tensor_probe
    )


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
