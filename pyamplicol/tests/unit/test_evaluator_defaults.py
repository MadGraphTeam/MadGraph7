# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import subprocess
import sys

import pytest

from pyamplicol.evaluators import SymbolicaEvaluatorSettings


def test_evaluator_defaults_preserve_production_optimization_policy() -> None:
    settings = SymbolicaEvaluatorSettings()
    assert settings.backend == "jit"
    assert settings.iterations == 10
    assert settings.cpe_iterations is None
    assert settings.jit_optimization_level == 3
    assert settings.jit_direct_translation is False
    assert settings.max_horner_scheme_variables == 1000
    assert settings.max_common_pair_cache_entries == 5_000_000
    assert settings.max_common_pair_distance == 1000


def test_evaluator_module_import_is_symbolica_lazy() -> None:
    code = """
import sys
assert 'symbolica' not in sys.modules
from pyamplicol.evaluators import SymbolicaEvaluatorSettings
SymbolicaEvaluatorSettings()
assert 'symbolica' not in sys.modules
    """
    subprocess.run([sys.executable, "-c", code], check=True)


def test_removed_simd_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="compiled-complex"):
        SymbolicaEvaluatorSettings(backend="compiled-complex-4x")


def test_compiled_flags_do_not_probe_checkout_dependencies() -> None:
    settings = SymbolicaEvaluatorSettings(
        backend="compiled-complex",
        compiler_flags=("-DPYAMPLICOL_TEST_FLAG=1",),
    )

    payload = settings.to_json_dict()
    assert payload["effective_compiler_flags"] == ["-DPYAMPLICOL_TEST_FLAG=1"]
