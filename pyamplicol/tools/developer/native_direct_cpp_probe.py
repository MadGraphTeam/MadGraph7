#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Build and measure one genuine compiled-C++ DirectApplication leaf.

The input must be a real schema-v3 compiled-C++ process artifact.  The probe
loads its retained Symbolica evaluator state, selects the largest compatible
leaf that consumes a model scalar, emits the optimized instruction stream
directly against split planes, and compiles a direct-only native library.  The
original dense library is loaded separately as a parity/timing oracle.

Run this script under ``tools/ci/memory_watchdog.py --limit-gib 30``.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from symbolica import CompiledComplexEvaluator, Evaluator

from pyamplicol.evaluators.native_direct_cpp import (
    NativeDirectCppCompiler,
    NativeDirectCppParameterKind,
    NativeDirectCppSpec,
    compile_native_direct_cpp,
    render_native_direct_cpp,
)
from pyamplicol.generation.evaluator_container import PacbinReader


@dataclass(frozen=True, slots=True)
class RetainedLeaf:
    process_id: str
    stage_index: int
    chunk_index: int
    function_name: str
    input_len: int
    output_len: int
    parameter_kinds: tuple[NativeDirectCppParameterKind, ...]
    source_path: Path
    library_path: Path
    evaluator: Any
    instruction_count: int


class InputPlane(ctypes.Structure):
    _fields_ = [("values", ctypes.POINTER(ctypes.c_double))]


class OutputPlane(ctypes.Structure):
    _fields_ = [("values", ctypes.POINTER(ctypes.c_double))]


class Scalar(ctypes.Structure):
    _fields_ = [("value", ctypes.POINTER(ctypes.c_double))]


class Metadata(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("input_plane_count", ctypes.c_uint32),
        ("scalar_input_count", ctypes.c_uint32),
        ("output_plane_count", ctypes.c_uint32),
        ("simd_lane_width", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


@dataclass(slots=True)
class BoundDirectCall:
    function: Any
    input_descriptors: Any
    scalar_descriptors: Any
    output_descriptors: Any
    input_planes: list[np.ndarray]
    scalar_values: list[ctypes.c_double]
    output_planes: list[np.ndarray]
    input_count: int
    scalar_count: int
    output_count: int

    def evaluate(self, point_start: int, point_count: int) -> int:
        return int(
            self.function(
                self.input_descriptors,
                self.input_count,
                self.scalar_descriptors,
                self.scalar_count,
                self.output_descriptors,
                self.output_count,
                point_start,
                point_count,
            )
        )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument(
        "--process",
        help="stable process id; defaults to the artifact default process",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/private/tmp/pyamplicol-native-direct-cpp-probe"),
    )
    parser.add_argument("--compiler", default="c++")
    parser.add_argument("--optimization-level", type=int, default=3)
    parser.add_argument("--simd-lane-width", type=int, default=2)
    parser.add_argument("--points", type=int, default=129)
    parser.add_argument("--stride", type=int, default=136)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=2_000)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"could not read JSON document {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON document is not an object: {path}")
    return value


def _process_id(outer: dict[str, Any], requested: str | None) -> str:
    if requested is not None:
        return requested
    default = outer.get("default_process_id")
    if not isinstance(default, str) or not default:
        raise RuntimeError("artifact has no default process id")
    return default


def _parameter_kind(component: dict[str, Any]) -> NativeDirectCppParameterKind:
    storage_is_scalar = component.get("kind") == "model_parameter"
    real_valued = component.get("real_valued") is True
    if storage_is_scalar:
        return (
            NativeDirectCppParameterKind.REAL_SCALAR
            if real_valued
            else NativeDirectCppParameterKind.COMPLEX_SCALAR
        )
    return (
        NativeDirectCppParameterKind.REAL_PLANE
        if real_valued
        else NativeDirectCppParameterKind.COMPLEX_PLANE
    )


def _load_evaluator_state(
    reader: PacbinReader,
    logical_path: str,
) -> Any:
    member = reader.member(logical_path)
    payload = reader.read_member(logical_path, length=member.length)
    return Evaluator.load(payload)


def _select_retained_leaf(
    artifact: Path,
    process_id: str,
    target_triple: str,
    cpu_features: tuple[str, ...],
    lane_width: int,
) -> RetainedLeaf:
    process_root = artifact / "processes" / process_id
    execution = _read_json(process_root / "execution.json")
    compiled = execution.get("compiled")
    if not isinstance(compiled, dict):
        raise RuntimeError("process has no compiled execution blueprint")
    stage_evaluators = compiled.get("stage_evaluators")
    if not isinstance(stage_evaluators, dict):
        raise RuntimeError("process has no compiled stage evaluators")
    stages = stage_evaluators.get("stages")
    if not isinstance(stages, list):
        raise RuntimeError("compiled stage evaluator list is absent")

    candidates: list[RetainedLeaf] = []
    with PacbinReader.open(
        artifact / "evaluators.pacbin",
        verify_payloads=True,
    ) as reader:
        for stage_ordinal, raw_stage in enumerate(stages):
            if not isinstance(raw_stage, dict):
                continue
            raw_components = raw_stage.get("input_components")
            evaluator = raw_stage.get("evaluator")
            if not isinstance(raw_components, list) or not isinstance(evaluator, dict):
                continue
            components: dict[int, dict[str, Any]] = {}
            for raw_component in raw_components:
                if not isinstance(raw_component, dict):
                    continue
                parameter_index = raw_component.get("parameter_index")
                if isinstance(parameter_index, int) and not isinstance(
                    parameter_index, bool
                ):
                    components[parameter_index] = raw_component
            chunks = evaluator.get("chunks")
            maps = evaluator.get("chunk_input_indices")
            if not isinstance(chunks, list) or not isinstance(maps, list):
                continue
            for chunk_index, (raw_chunk, raw_map) in enumerate(
                zip(chunks, maps, strict=True)
            ):
                if not isinstance(raw_chunk, dict) or not isinstance(raw_map, list):
                    continue
                try:
                    parent_indices = tuple(int(index) for index in raw_map)
                    kinds = tuple(
                        _parameter_kind(components[index]) for index in parent_indices
                    )
                except (KeyError, TypeError, ValueError):
                    continue
                if not any(
                    kind
                    in {
                        NativeDirectCppParameterKind.REAL_SCALAR,
                        NativeDirectCppParameterKind.COMPLEX_SCALAR,
                    }
                    for kind in kinds
                ):
                    continue
                function_name = raw_chunk.get("function_name")
                state_path = raw_chunk.get("evaluator_state_path")
                source_path = raw_chunk.get("source_path")
                library_path = raw_chunk.get("library_path")
                input_len = raw_chunk.get("input_len")
                output_len = raw_chunk.get("output_len")
                if not isinstance(source_path, str) and isinstance(library_path, str):
                    # Repackaged manifests deliberately omit the source path
                    # key, but retain it beside the native library.
                    source_path = str(
                        Path(library_path).parent
                        / (
                            f"{function_name}.cpp"
                            if isinstance(function_name, str)
                            else ""
                        )
                    )
                if (
                    not isinstance(function_name, str)
                    or not isinstance(state_path, str)
                    or not isinstance(source_path, str)
                    or not isinstance(library_path, str)
                    or not isinstance(input_len, int)
                    or not isinstance(output_len, int)
                    or len(kinds) != input_len
                ):
                    continue
                logical_state_path = f"processes/{process_id}/{state_path}"
                try:
                    source_evaluator = _load_evaluator_state(
                        reader,
                        logical_state_path,
                    )
                    spec = NativeDirectCppSpec(
                        function_name=function_name,
                        parameter_kinds=kinds,
                        output_count=output_len,
                        target_triple=target_triple,
                        cpu_features=cpu_features,
                        simd_lane_width=lane_width,
                    )
                    rendered = render_native_direct_cpp(source_evaluator, spec)
                except Exception:
                    continue
                candidates.append(
                    RetainedLeaf(
                        process_id=process_id,
                        stage_index=int(raw_stage.get("stage_index", stage_ordinal)),
                        chunk_index=chunk_index,
                        function_name=function_name,
                        input_len=input_len,
                        output_len=output_len,
                        parameter_kinds=kinds,
                        source_path=process_root / Path(source_path),
                        library_path=process_root / Path(library_path),
                        evaluator=source_evaluator,
                        instruction_count=rendered.instruction_count,
                    )
                )
    if not candidates:
        raise RuntimeError(
            "no genuine compiled leaf with a model scalar fits the fail-closed "
            "native DirectApplication instruction subset"
        )
    return max(
        candidates,
        key=lambda candidate: (
            candidate.instruction_count,
            candidate.output_len,
            -candidate.stage_index,
            -candidate.chunk_index,
        ),
    )


def _pointer(array: np.ndarray) -> ctypes.POINTER(ctypes.c_double):
    if array.dtype != np.float64 or not array.flags.c_contiguous:
        raise RuntimeError("direct plane is not contiguous float64")
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def _bind_direct(
    library: Any,
    leaf: RetainedLeaf,
    dense_inputs: np.ndarray,
    stride: int,
) -> BoundDirectCall:
    input_planes: list[np.ndarray] = []
    scalar_values: list[ctypes.c_double] = []
    for parameter, kind in enumerate(leaf.parameter_kinds):
        values = dense_inputs[:, parameter]
        if kind is NativeDirectCppParameterKind.COMPLEX_PLANE:
            input_planes.append(np.ascontiguousarray(values.real))
            input_planes.append(np.ascontiguousarray(values.imag))
        elif kind is NativeDirectCppParameterKind.REAL_PLANE:
            input_planes.append(np.ascontiguousarray(values.real))
        elif kind is NativeDirectCppParameterKind.COMPLEX_SCALAR:
            scalar_values.append(ctypes.c_double(float(values[0].real)))
            scalar_values.append(ctypes.c_double(float(values[0].imag)))
        else:
            scalar_values.append(ctypes.c_double(float(values[0].real)))
    input_descriptors = (InputPlane * len(input_planes))(
        *(InputPlane(_pointer(values)) for values in input_planes)
    )
    scalar_descriptors = (Scalar * len(scalar_values))(
        *(Scalar(ctypes.pointer(value)) for value in scalar_values)
    )
    output_planes = [
        np.full(stride, np.nan, dtype=np.float64) for _ in range(2 * leaf.output_len)
    ]
    output_descriptors = (OutputPlane * len(output_planes))(
        *(OutputPlane(_pointer(values)) for values in output_planes)
    )
    symbol = getattr(library, f"{leaf.function_name}_direct_application_v1")
    symbol.argtypes = [
        ctypes.POINTER(InputPlane),
        ctypes.c_uint32,
        ctypes.POINTER(Scalar),
        ctypes.c_uint32,
        ctypes.POINTER(OutputPlane),
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
    ]
    symbol.restype = ctypes.c_int
    return BoundDirectCall(
        function=symbol,
        input_descriptors=input_descriptors,
        scalar_descriptors=scalar_descriptors,
        output_descriptors=output_descriptors,
        input_planes=input_planes,
        scalar_values=scalar_values,
        output_planes=output_planes,
        input_count=len(input_planes),
        scalar_count=len(scalar_values),
        output_count=len(output_planes),
    )


def _metadata(
    library: Any,
    leaf: RetainedLeaf,
) -> tuple[Metadata, dict[str, object]]:
    prefix = f"{leaf.function_name}_direct_application_v1"
    metadata_function = getattr(library, f"{prefix}_metadata")
    metadata_function.argtypes = []
    metadata_function.restype = Metadata
    metadata = metadata_function()
    strings: dict[str, object] = {}
    for suffix in ("target_triple", "cpu_features", "source_sha256"):
        function = getattr(library, f"{prefix}_{suffix}")
        function.argtypes = []
        function.restype = ctypes.c_char_p
        raw = function()
        if raw is None:
            raise RuntimeError(f"direct metadata string {suffix} is null")
        strings[suffix] = raw.decode("ascii")
    stack_function = getattr(library, f"{prefix}_logical_stack_bytes")
    stack_function.argtypes = []
    stack_function.restype = ctypes.c_uint32
    strings["logical_stack_bytes"] = int(stack_function())
    return metadata, strings


def _deterministic_inputs(
    leaf: RetainedLeaf,
    stride: int,
) -> np.ndarray:
    rng = np.random.default_rng(0xD1EC7A)
    values = (
        rng.normal(0.0, 0.25, size=(stride, leaf.input_len))
        + 1j * rng.normal(0.0, 0.25, size=(stride, leaf.input_len))
    ).astype(np.complex128)
    for parameter, kind in enumerate(leaf.parameter_kinds):
        if kind in {
            NativeDirectCppParameterKind.REAL_PLANE,
            NativeDirectCppParameterKind.REAL_SCALAR,
        }:
            values[:, parameter].imag = 0.0
        if kind in {
            NativeDirectCppParameterKind.REAL_SCALAR,
            NativeDirectCppParameterKind.COMPLEX_SCALAR,
        }:
            scalar = complex(values[0, parameter])
            if kind is NativeDirectCppParameterKind.REAL_SCALAR:
                scalar = complex(0.375, 0.0)
            values[:, parameter] = scalar
    values[0, 0] = complex(values[0, 0].real, -0.0)
    return values


def _resolved_direct_outputs(
    bound: BoundDirectCall,
    start: int,
    count: int,
    output_len: int,
) -> np.ndarray:
    stop = start + count
    return np.column_stack(
        [
            bound.output_planes[2 * output][start:stop]
            + 1j * bound.output_planes[2 * output + 1][start:stop]
            for output in range(output_len)
        ]
    )


def _test_parity_and_tails(
    dense: Any,
    dense_inputs: np.ndarray,
    bound: BoundDirectCall,
    leaf: RetainedLeaf,
    points: int,
    stride: int,
) -> dict[str, object]:
    expected = np.asarray(dense.evaluate(dense_inputs), dtype=np.complex128)
    if expected.shape != (stride, leaf.output_len):
        raise RuntimeError(f"dense oracle returned unexpected shape {expected.shape}")
    tail_ranges = (
        (0, 1),
        (0, 2),
        (0, 3),
        (0, min(127, points)),
        (1, min(127, points - 1)),
        (0, min(128, points)),
        (0, points),
    )
    maximum_error = 0.0
    for start, count in tail_ranges:
        if count < 1:
            continue
        for plane in bound.output_planes:
            plane.fill(np.nan)
        status = bound.evaluate(start, count)
        if status != 0:
            raise RuntimeError(
                f"direct leaf returned status {status} for range {start}+{count}"
            )
        actual = _resolved_direct_outputs(
            bound,
            start,
            count,
            leaf.output_len,
        )
        reference = expected[start : start + count]
        np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-15)
        maximum_error = max(
            maximum_error,
            float(np.max(np.abs(actual - reference), initial=0.0)),
        )
        for plane in bound.output_planes:
            if (
                not np.isnan(plane[:start]).all()
                or not np.isnan(plane[start + count :]).all()
            ):
                raise RuntimeError(
                    "direct odd-tail call overwrote an inactive sentinel"
                )

    bound.output_descriptors[0].values = bound.input_descriptors[0].values
    alias_status = bound.evaluate(0, 1)
    bound.output_descriptors[0].values = _pointer(bound.output_planes[0])
    if alias_status != 4:
        raise RuntimeError(
            f"native direct input/output alias returned {alias_status}, expected 4"
        )
    count_status = int(
        bound.function(
            bound.input_descriptors,
            bound.input_count - 1,
            bound.scalar_descriptors,
            bound.scalar_count,
            bound.output_descriptors,
            bound.output_count,
            0,
            1,
        )
    )
    if count_status != 2:
        raise RuntimeError(
            f"native direct descriptor mismatch returned {count_status}, expected 2"
        )
    return {
        "ranges": [list(item) for item in tail_ranges],
        "maximum_absolute_error": maximum_error,
        "rtol": 1e-12,
        "atol": 1e-15,
        "alias_status": alias_status,
        "descriptor_count_status": count_status,
    }


def _median_mad(values: list[int]) -> tuple[float, float]:
    median = float(statistics.median(values))
    mad = float(statistics.median(abs(value - median) for value in values))
    return median, mad


def _sample(
    function: Any,
    *,
    samples: int,
    repeats: int,
) -> list[int]:
    output: list[int] = []
    for _sample in range(samples):
        started = time.perf_counter_ns()
        for _repeat in range(repeats):
            function()
        output.append((time.perf_counter_ns() - started) // repeats)
    return output


def _timings(
    dense: Any,
    dense_inputs: np.ndarray,
    bound: BoundDirectCall,
    leaf: RetainedLeaf,
    points: int,
    samples: int,
    repeats: int,
) -> dict[str, object]:
    packed = np.empty((points, leaf.input_len), dtype=np.complex128)
    scattered = np.empty((points, leaf.output_len), dtype=np.complex128)
    dense_view = dense_inputs[:points]

    def direct_call() -> None:
        status = bound.evaluate(0, points)
        if status != 0:
            raise RuntimeError(f"timed direct leaf returned status {status}")

    def dense_prepacked() -> None:
        result = dense.evaluate(dense_view)
        if np.asarray(result).shape != scattered.shape:
            raise RuntimeError("timed dense oracle changed shape")

    def dense_gather_call_scatter() -> None:
        packed[:, :] = dense_view
        result = np.asarray(dense.evaluate(packed), dtype=np.complex128)
        scattered[:, :] = result

    direct_call()
    dense_prepacked()
    dense_gather_call_scatter()
    direct_samples: list[int] = []
    raw_samples: list[int] = []
    full_samples: list[int] = []
    for sample in range(samples):
        paths = (
            (
                ("direct", direct_call),
                ("raw_dense", dense_prepacked),
                ("full_dense", dense_gather_call_scatter),
            )
            if sample % 2 == 0
            else (
                ("full_dense", dense_gather_call_scatter),
                ("raw_dense", dense_prepacked),
                ("direct", direct_call),
            )
        )
        for label, function in paths:
            values = _sample(function, samples=1, repeats=repeats)
            {
                "direct": direct_samples,
                "raw_dense": raw_samples,
                "full_dense": full_samples,
            }[label].extend(values)
    direct_median, direct_mad = _median_mad(direct_samples)
    raw_median, raw_mad = _median_mad(raw_samples)
    full_median, full_mad = _median_mad(full_samples)
    return {
        "boundary": (
            "Python FFI probe: raw_dense is prepacked CompiledComplexEvaluator; "
            "full_dense adds NumPy gather/scatter; direct is one prebound native call"
        ),
        "points": points,
        "samples": samples,
        "repeats": repeats,
        "direct_ns_per_call": direct_samples,
        "raw_dense_ns_per_call": raw_samples,
        "full_dense_ns_per_call": full_samples,
        "direct_median_ns_per_call": direct_median,
        "direct_mad_ns_per_call": direct_mad,
        "raw_dense_median_ns_per_call": raw_median,
        "raw_dense_mad_ns_per_call": raw_mad,
        "full_dense_median_ns_per_call": full_median,
        "full_dense_mad_ns_per_call": full_mad,
        "direct_ns_per_point": direct_median / points,
        "raw_dense_ns_per_point": raw_median / points,
        "full_dense_ns_per_point": full_median / points,
        "raw_dense_over_direct": raw_median / direct_median,
        "full_dense_over_direct": full_median / direct_median,
    }


def main() -> int:
    args = _arguments()
    if args.points < 3 or args.stride < args.points:
        raise RuntimeError("probe stride must cover at least three active points")
    if args.samples < 3 or args.samples % 2 == 0 or args.repeats < 1:
        raise RuntimeError("probe needs an odd sample count >= 3 and positive repeats")
    artifact = args.artifact.resolve()
    outer = _read_json(artifact / "artifact.json")
    process_id = _process_id(outer, args.process)
    producer = outer.get("producer")
    target = producer.get("target") if isinstance(producer, dict) else None
    if not isinstance(target, dict):
        raise RuntimeError("artifact producer target metadata is absent")
    target_triple = target.get("triple")
    raw_cpu_features = target.get("cpu_features")
    if not isinstance(target_triple, str) or not isinstance(raw_cpu_features, list):
        raise RuntimeError("artifact producer target metadata is invalid")
    cpu_features = tuple(str(feature) for feature in raw_cpu_features)
    leaf = _select_retained_leaf(
        artifact,
        process_id,
        target_triple,
        cpu_features,
        args.simd_lane_width,
    )
    spec = NativeDirectCppSpec(
        function_name=leaf.function_name,
        parameter_kinds=leaf.parameter_kinds,
        output_count=leaf.output_len,
        target_triple=target_triple,
        cpu_features=cpu_features,
        simd_lane_width=args.simd_lane_width,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    direct_source = args.output_dir / f"{leaf.function_name}.direct.cpp"
    direct_library = args.output_dir / f"lib{leaf.function_name}.direct"
    compiler = NativeDirectCppCompiler(
        executable=args.compiler,
        optimization_level=args.optimization_level,
        native_arch=bool(cpu_features),
        extra_flags=(),
    )
    built = compile_native_direct_cpp(
        leaf.evaluator,
        spec=spec,
        compiler=compiler,
        output_source_path=direct_source,
        output_library_path=direct_library,
    )
    library = ctypes.CDLL(str(built.library_path))
    metadata, auxiliary_metadata = _metadata(library, leaf)
    if metadata.abi_version != 1 or metadata.struct_size != ctypes.sizeof(Metadata):
        raise RuntimeError("native direct primary metadata is incompatible")
    if metadata.flags != 0x3F or metadata.reserved != 0:
        raise RuntimeError("native direct primary metadata flags are incompatible")
    if auxiliary_metadata["target_triple"] != target_triple:
        raise RuntimeError("native direct target triple metadata changed")
    if auxiliary_metadata["cpu_features"] != ",".join(cpu_features):
        raise RuntimeError("native direct CPU feature metadata changed")
    if auxiliary_metadata["source_sha256"] != built.source.evaluator_state_sha256:
        raise RuntimeError("native direct source-state digest metadata changed")

    dense_inputs = _deterministic_inputs(leaf, args.stride)
    dense = CompiledComplexEvaluator.load(
        str(leaf.library_path),
        leaf.function_name,
        leaf.input_len,
        leaf.output_len,
    )
    bound = _bind_direct(library, leaf, dense_inputs, args.stride)
    if (
        metadata.input_plane_count != bound.input_count
        or metadata.scalar_input_count != bound.scalar_count
        or metadata.output_plane_count != bound.output_count
        or metadata.simd_lane_width != args.simd_lane_width
    ):
        raise RuntimeError("native direct descriptor metadata disagrees with bindings")
    parity = _test_parity_and_tails(
        dense,
        dense_inputs,
        bound,
        leaf,
        args.points,
        args.stride,
    )
    timings = _timings(
        dense,
        dense_inputs,
        bound,
        leaf,
        args.points,
        args.samples,
        args.repeats,
    )
    dense_source_bytes = leaf.source_path.stat().st_size
    dense_library_bytes = leaf.library_path.stat().st_size
    direct_source_bytes = built.source_path.stat().st_size
    direct_library_bytes = built.library_path.stat().st_size
    artifact_bytes = sum(
        path.stat().st_size for path in artifact.rglob("*") if path.is_file()
    )
    payload_delta = (
        direct_source_bytes
        - dense_source_bytes
        + direct_library_bytes
        - dense_library_bytes
    )
    result = {
        "schema_version": 1,
        "status": "pass",
        "artifact": str(artifact),
        "artifact_id": outer.get("artifact_id"),
        "process_id": process_id,
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
        },
        "leaf": {
            "stage_index": leaf.stage_index,
            "chunk_index": leaf.chunk_index,
            "function_name": leaf.function_name,
            "input_len": leaf.input_len,
            "output_len": leaf.output_len,
            "parameter_kinds": [kind.value for kind in leaf.parameter_kinds],
            "instruction_count": built.source.instruction_count,
            "temporary_count": built.source.temporary_count,
            "input_plane_count": built.source.input_plane_count,
            "scalar_input_count": built.source.scalar_input_count,
            "output_plane_count": built.source.output_plane_count,
            "logical_stack_bytes": built.source.logical_stack_bytes,
            "evaluator_state_sha256": built.source.evaluator_state_sha256,
        },
        "producer": {
            "abi": "pyamplicol-native-compiled-direct-application-v1",
            "target_triple": target_triple,
            "cpu_features": list(cpu_features),
            "simd_lane_width": args.simd_lane_width,
            "source_path": str(built.source_path),
            "library_path": str(built.library_path),
            "compiler_command": list(built.compiler_command),
            "compile_seconds": built.compile_seconds,
            "direct_source_calls_dense_symbol": (
                f"{leaf.function_name}_complexf64(" in built.source.source
            ),
        },
        "parity": parity,
        "timings": timings,
        "payload": {
            "original_artifact_bytes": artifact_bytes,
            "dense_source_bytes": dense_source_bytes,
            "direct_source_bytes": direct_source_bytes,
            "dense_library_bytes": dense_library_bytes,
            "direct_library_bytes": direct_library_bytes,
            "replacement_delta_bytes": payload_delta,
            "replacement_delta_fraction_of_artifact": payload_delta / artifact_bytes,
        },
    }
    if result["producer"]["direct_source_calls_dense_symbol"]:
        raise RuntimeError("generated direct entry contains a forbidden dense call")
    serialized = json.dumps(result, indent=2, sort_keys=True)
    print(serialized)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(f"{serialized}\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
