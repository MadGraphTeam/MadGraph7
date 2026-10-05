# SPDX-License-Identifier: 0BSD
"""Bounded native-provider checks; reuse the already staged Rusticol SDK."""

from __future__ import annotations

import ctypes as ct
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest

from pyamplicol import Generator, ModelSource, ProcessSet, Runtime
from pyamplicol.assets.prepared_models import (
    BUILTIN_SM_JIT_O2,
    packaged_prepared_model_path,
)
from pyamplicol.config import (
    ColorConfig,
    EvaluatorConfig,
    EvaluatorOptimizationConfig,
    GenerationConfig,
    GenerationRelationDiscoveryConfig,
    GenerationValidationConfig,
    JITConfig,
    RunConfig,
)


def _unavailable(reason):
    if os.environ.get("PYAMPLICOL_REQUIRE_NATIVE_TESTS") == "1":
        pytest.fail(reason)
    pytest.skip(reason)


def _config_executable():
    configured = os.environ.get("RUSTICOL_CONFIG")
    candidates = (
        configured,
        str(Path(sys.executable).parent / "rusticol-config"),
        shutil.which("rusticol-config"),
    )
    for candidate in candidates:
        if candidate and (shutil.which(candidate) or Path(candidate).is_file()):
            return candidate
    _unavailable(
        "UMAMI native integration requires the staged rusticol-config executable"
    )


def _library(path):
    library = ct.CDLL(str(path))
    library.umami_initialize.argtypes = (ct.POINTER(ct.c_void_p), ct.c_char_p)
    library.umami_free.argtypes = (ct.c_void_p,)
    library.umami_get_meta.argtypes = (ct.c_int, ct.c_void_p)
    for name in ("supported_inputs", "required_inputs", "supported_outputs"):
        getattr(library, "umami_" + name).argtypes = (
            ct.POINTER(ct.POINTER(ct.c_bool)),
            ct.POINTER(ct.c_int),
        )
    library.umami_get_parameter.argtypes = (
        ct.c_void_p,
        ct.c_char_p,
        ct.POINTER(ct.c_double),
        ct.POINTER(ct.c_double),
    )
    library.umami_set_parameter.argtypes = (
        ct.c_void_p,
        ct.c_char_p,
        ct.c_double,
        ct.c_double,
    )
    library.umami_matrix_element.argtypes = (
        ct.c_void_p,
        ct.c_size_t,
        ct.c_size_t,
        ct.c_size_t,
        ct.c_size_t,
        ct.POINTER(ct.c_int),
        ct.POINTER(ct.c_void_p),
        ct.c_size_t,
        ct.POINTER(ct.c_int),
        ct.POINTER(ct.c_void_p),
    )
    return library


@dataclass
class Provider:
    artifact: Path
    build: Path
    data: dict
    library: object

    @contextmanager
    def handle(self):
        value = ct.c_void_p()
        assert (
            self.library.umami_initialize(ct.byref(value), os.fsencode(self.artifact))
            == 0
        )
        assert value.value is not None
        try:
            yield value
        finally:
            assert self.library.umami_free(value) == 0


@pytest.fixture(scope="module", params=("lc", "full"))
def providers(request, tmp_path_factory):
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        _unavailable("the Rusticol extension has not been built")
    if shutil.which("make") is None or shutil.which(os.environ.get("CC", "cc")) is None:
        _unavailable("UMAMI integration requires make and a C compiler")
    config_executable = _config_executable()
    root = tmp_path_factory.mktemp("umami-" + request.param)
    artifact, build = root / "artifact", root / "build"
    config = RunConfig(
        action="generate",
        color=ColorConfig(accuracy=request.param),
        generation=GenerationConfig(
            workers=1,
            emit_api_bundle=True,
            relation_discovery=GenerationRelationDiscoveryConfig(mode="off"),
            validation=GenerationValidationConfig(
                enabled=False, post_build_validation=False
            ),
        ),
        evaluator=EvaluatorConfig(
            execution_mode="recurrence",
            optimization=EvaluatorOptimizationConfig(cores=1),
            jit=JITConfig(optimization_level=2),
        ),
    )
    # The massive colour singlet forces a separate fixed-mass provider, while
    # retaining quark colour flow and generic UFO parameter declarations.
    expressions = (
        ("g g > g g",) if request.param == "lc" else ("g g > g g", "d d~ > z g")
    )
    names = ("gg",) if request.param == "lc" else ("gg", "dd_zg")
    with packaged_prepared_model_path(BUILTIN_SM_JIT_O2) as prepared_model:
        Generator(config).generate(
            ProcessSet.from_expressions(expressions, names=names),
            artifact,
            model=ModelSource.from_path(prepared_model),
        )
    completed = subprocess.run(
        [
            "make",
            "-C",
            str(artifact / "API/umami"),
            f"BUILD_DIR={build}",
            f"RUSTICOL_CONFIG={config_executable}",
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    root_metadata = json.loads((artifact / "API/umami/metadata.json").read_text())
    paths = sorted((artifact / "API/umami/providers").glob("*/metadata.json"))
    documents = (
        [json.loads(path.read_text()) for path in paths] if paths else [root_metadata]
    )
    assert len(documents) == len(expressions)
    return [
        Provider(artifact, build, data, _library(build / data["provider"]["library"]))
        for data in documents
    ]


def _entry(provider, channel=0, flavour=0):
    return provider.data["channels"][channel]["processes"][flavour]["runtime"]


def _point(provider, entry):
    runtime = Runtime.load(provider.artifact, process=entry["process_id"])
    points = runtime.validation_momenta()
    assert points
    # The provider maps physical slots to representative runtime slots.
    physical = [None] * len(points[0])
    for runtime_slot, physical_slot in enumerate(entry["permutation"]):
        physical[physical_slot] = points[0][runtime_slot]
    return tuple(physical)


def _expected(provider, entry, point, parameters=None):
    runtime = Runtime.load(provider.artifact, process=entry["process_id"])
    if parameters:
        runtime.set_model_parameters(parameters)
    runtime_point = tuple(point[index] for index in entry["permutation"])
    selected = None if entry["color_id"] is None else (entry["color_id"],)
    return (
        float(complex(runtime.evaluate((runtime_point,), color_flows=selected)[0]).real)
        * entry["factor"]
    )


def _batch(
    provider,
    handle,
    points,
    *,
    channel=0,
    flavour=0,
    alpha_s=None,
    offset=0,
    count=None,
):
    stride, n = len(points), len(points[0])
    momenta = (ct.c_double * (4 * n * stride))(
        *(
            float(points[event][leg][component])
            for component in range(4)
            for leg in range(n)
            for event in range(stride)
        )
    )
    inputs = {
        0: momenta,
        2: (ct.c_int * stride)(*[flavour] * stride),
        4: (ct.c_double * stride)(*[0.5] * stride),
        8: (ct.c_uint * stride)(*[channel] * stride),
    }
    if alpha_s is not None:
        inputs[1] = (ct.c_double * stride)(*alpha_s)
    outputs = {
        0: (ct.c_double * stride)(*[-123.0] * stride),
        2: (ct.c_int * stride)(*[-123] * stride),
        3: (ct.c_int * stride)(*[-123] * stride),
    }
    status = provider.library.umami_matrix_element(
        handle,
        stride - offset if count is None else count,
        stride,
        offset,
        len(inputs),
        (ct.c_int * len(inputs))(*inputs),
        (ct.c_void_p * len(inputs))(
            *(ct.cast(value, ct.c_void_p) for value in inputs.values())
        ),
        len(outputs),
        (ct.c_int * len(outputs))(*outputs),
        (ct.c_void_p * len(outputs))(
            *(ct.cast(value, ct.c_void_p) for value in outputs.values())
        ),
    )
    return status, {key: list(value) for key, value in outputs.items()}


def _parameter(provider, handle, name):
    real, imaginary = ct.c_double(), ct.c_double()
    assert (
        provider.library.umami_get_parameter(
            handle, name.encode(), ct.byref(real), ct.byref(imaginary)
        )
        == 0
    )
    return complex(real.value, imaginary.value)


def test_metadata_capabilities_and_hidden_native_symbols(providers):
    for provider in providers:
        with pytest.raises(AttributeError):
            _ = provider.library.rusticol_runtime_load
        count = ct.c_int()
        assert provider.library.umami_get_meta(1, ct.byref(count)) == 0
        assert count.value == provider.data["provider"]["particle_count"]
        masses = (ct.c_double * count.value)()
        assert provider.library.umami_get_meta(5, masses) == 0
        assert list(masses) == provider.data["provider"]["masses"]
        assert provider.library.umami_get_meta(2, ct.byref(count)) == 6
        for operation in ("supported_inputs", "required_inputs", "supported_outputs"):
            values, length = ct.POINTER(ct.c_bool)(), ct.c_int()
            assert (
                getattr(provider.library, "umami_" + operation)(
                    ct.byref(values), ct.byref(length)
                )
                == 0
            )
            flags = list(values[: length.value])
            if operation == "required_inputs":
                assert flags == [
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                ]
            elif operation == "supported_inputs":
                assert flags[1] == provider.data["provider"]["supports_alpha_s"]
                assert not flags[3] and not flags[5] and not flags[6] and not flags[7]
            else:
                assert flags == [True, False, True, True, False, False]


def test_all_exported_flavours_match_native_and_preserve_stride(providers):
    for provider in providers:
        with provider.handle() as handle:
            for channel, description in enumerate(provider.data["channels"]):
                for flavour in range(len(description["processes"])):
                    entry = _entry(provider, channel, flavour)
                    point = _point(provider, entry)
                    expected = _expected(provider, entry, point)
                    status, outputs = _batch(
                        provider,
                        handle,
                        (point,) * 3,
                        channel=channel,
                        flavour=flavour,
                        offset=1,
                        count=1,
                    )
                    assert status == 0
                    assert outputs[0] == pytest.approx(
                        [-123.0, expected, -123.0], rel=2e-11, abs=1e-15
                    )
                    assert outputs[2] == [-123, entry["color_index"], -123]
                    process = provider.data["runtime_processes"][entry["process_index"]]
                    assert outputs[3] == [-123, len(process["helicity_ids"]) // 2, -123]


def test_parameters_event_override_and_independent_concurrent_handles(providers):
    for provider in providers:
        assert provider.data["provider"]["supports_alpha_s"]
        entry = _entry(provider)
        name = provider.data["runtime_processes"][entry["process_index"]][
            "alpha_s_parameter"
        ]
        point = _point(provider, entry)
        with provider.handle() as first, provider.handle() as second:
            original = _parameter(provider, first, name)
            assert original.imag == 0 and original.real > 0
            changed = original.real * 1.2
            assert (
                provider.library.umami_set_parameter(first, name.encode(), changed, 0.0)
                == 0
            )
            assert _parameter(provider, first, name) == changed
            assert _parameter(provider, second, name) == original
            event_values = (original.real * 0.8, original.real * 1.1)
            status, outputs = _batch(
                provider, first, (point,) * 2, alpha_s=event_values
            )
            assert status == 0
            expected = [
                _expected(provider, entry, point, {name: value})
                for value in event_values
            ]
            assert outputs[0] == pytest.approx(expected, rel=2e-11)
            assert _parameter(provider, first, name) == changed
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [
                    pool.submit(_batch, provider, handle, (point,))
                    for handle in (first, second)
                ]
                values = [future.result() for future in futures]
            for (status, outputs), value in zip(
                values, (changed, original.real), strict=True
            ):
                assert status == 0
                assert outputs[0][0] == pytest.approx(
                    _expected(provider, entry, point, {name: value}), rel=2e-11
                )
            assert (
                provider.library.umami_set_parameter(first, b"does-not-exist", 1.0, 0.0)
                == 1
            )


def test_error_contract_and_zero_count(providers):
    for provider in providers:
        invalid = ct.c_void_p(123)
        assert (
            provider.library.umami_initialize(ct.byref(invalid), b"relative/artifact")
            == 1
        )
        assert invalid.value is None
        with provider.handle() as handle:
            evaluate = provider.library.umami_matrix_element
            assert evaluate(handle, 0, 0, 0, 0, None, None, 0, None, None) == 0
            assert evaluate(handle, 1, 1, 0, 0, None, None, 0, None, None) == 7
            assert evaluate(handle, 1, 1, 1, 0, None, None, 0, None, None) == 1
            selector = ct.c_int(0)
            pointer = (ct.c_void_p * 1)(ct.cast(ct.byref(selector), ct.c_void_p))
            assert (
                evaluate(handle, 0, 0, 0, 1, (ct.c_int * 1)(6), pointer, 0, None, None)
                == 3
            )
            assert (
                evaluate(handle, 0, 0, 0, 0, None, None, 1, (ct.c_int * 1)(1), pointer)
                == 4
            )
            point = _point(provider, _entry(provider))
            assert _batch(provider, handle, (point,), channel=999)[0] == 1
            assert _batch(provider, handle, (point,), flavour=-1)[0] == 1


def test_linked_driver_survives_relocation(providers, tmp_path):
    provider = providers[0]
    relocated = tmp_path / "relocated files with spaces"
    artifact, build = relocated / "artifact", relocated / "build"
    shutil.copytree(provider.artifact, artifact)
    build.mkdir(parents=True)
    library_name = provider.data["provider"]["library"]
    driver_name = (
        "umami_driver"
        if library_name == "libumami.so"
        else "umami_driver_" + provider.data["provider"]["id"]
    )
    shutil.copy2(provider.build / library_name, build / library_name)
    shutil.copy2(provider.build / driver_name, build / driver_name)
    completed = subprocess.run(
        [str(build / driver_name), str(artifact)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout)
    entry = _entry(provider)
    assert result["matrix_element"] == pytest.approx(
        _expected(provider, entry, _point(provider, entry)), rel=2e-11
    )
