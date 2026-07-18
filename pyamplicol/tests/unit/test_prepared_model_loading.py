# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from pathlib import Path

import pytest

from pyamplicol import ModelSource
from pyamplicol.api.errors import ModelError
from pyamplicol.api.models import _compiled_model_payload
from pyamplicol.models.loading import compile_model_source, load_compiled_model
from pyamplicol.models.prepared import (
    PreparedKernelPack,
    PreparedKernelRecord,
    write_prepared_model_bundle,
)


def _prepared_builtin_sm(tmp_path: Path) -> Path:
    compiled = compile_model_source("built-in-sm", use_cache=False)
    kernel = PreparedKernelRecord(
        kernel_id=0,
        contract_kind="vertex",
        canonical_signature="test:prepared-loading",
        input_arity=1,
        output_arity=1,
        input_layout=("input",),
        output_layout=("output",),
        exact_evaluator_state_path="kernels/0/exact.evaluator.bin",
        f64_evaluator_manifest={
            "kind": "symjit-application-evaluator",
            "input_len": 1,
            "output_len": 1,
            "application_path": "kernels/0/application.symjit",
            "evaluator_state_path": "kernels/0/exact.evaluator.bin",
        },
    )
    pack = PreparedKernelPack(
        backend="jit",
        optimization_settings={"optimization_level": 3},
        producer={"distribution": "pyamplicol", "version": "test"},
        dependency_abis={"symjit_application": "test-v1"},
        provenance={"compiled_model": "test"},
        target={
            "portable": False,
            "word_bits": 64,
            "endianness": "little",
            "target_triple": "test-target",
            "cpu_features": [],
        },
        kernels=(kernel,),
    )
    return write_prepared_model_bundle(
        tmp_path / "builtin",
        compiled_model=compiled.to_dict(),
        kernel_pack=pack,
        payloads={
            "kernels/0/exact.evaluator.bin": b"exact",
            "kernels/0/application.symjit": b"jit",
        },
    )


def test_model_source_loads_prepared_bundle_without_changing_provenance(
    tmp_path: Path,
) -> None:
    bundle_path = _prepared_builtin_sm(tmp_path)

    source = ModelSource.from_path(bundle_path)
    compiled = source.compile(use_cache=False)
    payload = _compiled_model_payload(compiled)

    assert source.kind == "prepared"
    assert compiled.name == "built-in-sm"
    assert compiled.source.kind == "built-in-sm"
    assert compiled.is_prepared
    assert compiled.prepared_backend == "jit"
    assert payload.prepared_bundle is not None
    assert payload.prepared_bundle.path == bundle_path


def test_internal_compiled_model_loader_accepts_prepared_bundle(tmp_path: Path) -> None:
    bundle_path = _prepared_builtin_sm(tmp_path)

    compiled = load_compiled_model(bundle_path)

    assert compiled.name == "built-in-sm"
    assert compiled.prepared_backend == "jit"
    assert compiled.prepared_bundle is not None


def test_prepared_model_rejects_compilation_options(tmp_path: Path) -> None:
    bundle_path = _prepared_builtin_sm(tmp_path)

    with pytest.raises(ModelError, match="already compiled"):
        ModelSource.from_path(bundle_path, simplify=False).compile(use_cache=False)
