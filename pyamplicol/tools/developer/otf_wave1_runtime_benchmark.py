#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Measure the two fixed warmed OTF LC Wave-1 workloads with the native timer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ARTIFACT_ID = "81de5672cafb4cedd4a8cb4064e4b5e6caefb25f031e6b5fbb22990f82a073fc"
PROCESS_ID = "catalog_dd_tt_jets_n4"
PROCESS = "d d~ > t t~ g g"
CASE_ID = "catalog:dd_tt_jets:n4"
FLOW_ID = "flow:2,5,6,4,3,1"
HELICITY_ID = "h:-1,+1,-1,+1,-1,+1"
POINT_SHA256 = "111d277e8eb86e8d183adba74c0c83a3e4d20a6317a10bdb7c63893f5e5017c0"
BATCH_SHA256 = "a875706b2f87d9105b483b2d5d33285005902ed29485a8934601598852016c71"
BATCH_SIZE = 128
REPETITIONS = 4
WARMUPS = 3
SAMPLES = 11


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _point(fixture: Path) -> tuple[tuple[float, ...], ...]:
    payload = json.loads(fixture.read_text(encoding="utf-8"))
    case = next(
        (
            item
            for item in payload.get("catalog_cases", ())
            if item.get("id") == CASE_ID
        ),
        None,
    )
    if not isinstance(case, dict) or case.get("process") != PROCESS:
        raise RuntimeError(f"fixture omits the fixed case {CASE_ID!r}")
    point = tuple(tuple(float(value) for value in row) for row in case["momenta"])
    if _canonical_sha256((point,)) != POINT_SHA256:
        raise RuntimeError("fixed acceptance point identity changed")
    return point


def _installed_identity(package_root: Path, version: str) -> dict[str, object]:
    build_info_path = package_root / "_build_info.json"
    build_info = json.loads(build_info_path.read_text(encoding="utf-8"))
    extensions = tuple(package_root.glob("_rusticol*.so"))
    if len(extensions) != 1:
        raise RuntimeError("installed native extension is absent or ambiguous")
    return {
        "package_root": str(package_root),
        "package_version": version,
        "source_revision": build_info.get("source_revision"),
        "native_build_inputs_sha256": build_info.get("native_build_inputs_sha256"),
        "native_extension_sha256": _file_sha256(extensions[0]),
    }


def _value_record(value: object) -> dict[str, object]:
    resolved = complex(value)  # type: ignore[arg-type]
    if not math.isfinite(resolved.real) or not math.isfinite(resolved.imag):
        raise RuntimeError("selector smoke evaluation is not finite")
    return {
        "real": resolved.real,
        "imag": resolved.imag,
        "real_hex": resolved.real.hex(),
        "imag_hex": resolved.imag.hex(),
    }


def _summary(samples: list[float]) -> dict[str, object]:
    median = statistics.median(samples)
    return {
        "seconds_per_point": samples,
        "median_seconds_per_point": median,
        "mad_seconds_per_point": statistics.median(
            abs(value - median) for value in samples
        ),
        "minimum_seconds_per_point": min(samples),
        "maximum_seconds_per_point": max(samples),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    artifact = arguments.artifact.resolve(strict=True)
    fixture = arguments.fixture.resolve(strict=True)
    destination = arguments.output.resolve()
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"refusing to replace benchmark evidence: {destination}")

    import pyamplicol
    from pyamplicol import Runtime

    point = _point(fixture)
    batch = tuple(point for _ in range(BATCH_SIZE))
    if _canonical_sha256(batch) != BATCH_SHA256:
        raise RuntimeError("fixed benchmark batch identity changed")

    runtime = Runtime.load(artifact, process=PROCESS_ID)
    if runtime.artifact_id != ARTIFACT_ID:
        raise RuntimeError("benchmark artifact identity changed")
    if runtime.execution_mode != "on-the-fly":
        raise RuntimeError(
            f"expected on-the-fly runtime, got {runtime.execution_mode!r}"
        )
    backend = getattr(runtime, "_backend", None)
    timer = getattr(backend, "_benchmark_f64_wall_time", None)
    if not callable(timer):
        raise RuntimeError("native f64 wall timer is unavailable")

    workloads: tuple[tuple[str, dict[str, tuple[str, ...]]], ...] = (
        ("selected_flow_helicity_sum", {"color_flows": (FLOW_ID,)}),
        ("all_flow_single_helicity", {"helicities": (HELICITY_ID,)}),
    )
    smoke_values = {
        name: _value_record(runtime.evaluate((point,), precision=16, **selectors)[0])
        for name, selectors in workloads
    }
    for warmup in range(WARMUPS):
        ordered = workloads if warmup % 2 == 0 else tuple(reversed(workloads))
        for _, selectors in ordered:
            elapsed = float(timer(batch, 1, precision=16, **selectors))
            if not math.isfinite(elapsed) or elapsed <= 0.0:
                raise RuntimeError("native wall warmup returned an invalid duration")

    raw: dict[str, list[float]] = {name: [] for name, _ in workloads}
    divisor = REPETITIONS * BATCH_SIZE
    for sample in range(SAMPLES):
        ordered = workloads if sample % 2 == 0 else tuple(reversed(workloads))
        for name, selectors in ordered:
            # Only the last selector family is retained. Prime the exact route
            # after every order switch so the measured block is wholly warm.
            prime_elapsed = float(timer(batch, 1, precision=16, **selectors))
            if not math.isfinite(prime_elapsed) or prime_elapsed <= 0.0:
                raise RuntimeError("native selector prime returned an invalid duration")
            elapsed = float(timer(batch, REPETITIONS, precision=16, **selectors))
            if not math.isfinite(elapsed) or elapsed <= 0.0:
                raise RuntimeError("native wall sample returned an invalid duration")
            raw[name].append(elapsed / divisor)

    package_root = Path(pyamplicol.__file__).resolve().parent
    result: dict[str, Any] = {
        "kind": "pyamplicol-otf-wave1-native-wall-benchmark",
        "schema_version": 1,
        "label": arguments.label,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "driver_sha256": _file_sha256(Path(__file__).resolve()),
        "installation": _installed_identity(package_root, pyamplicol.__version__),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "contract": {
            "timer": "Runtime._benchmark_f64_wall_time",
            "artifact": str(artifact),
            "artifact_id": ARTIFACT_ID,
            "process_id": PROCESS_ID,
            "process": PROCESS,
            "point_case_id": CASE_ID,
            "point_sha256": POINT_SHA256,
            "batch_sha256": BATCH_SHA256,
            "batch_size": BATCH_SIZE,
            "repetitions": REPETITIONS,
            "warmups": WARMUPS,
            "samples": SAMPLES,
            "precision": 16,
            "per_sample_selector_prime_repetitions": 1,
            "selector_prime_timing": "excluded",
            "sample_order": "ABBA",
            "selectors": {
                "selected_flow_helicity_sum": {"color_flows": [FLOW_ID]},
                "all_flow_single_helicity": {"helicities": [HELICITY_ID]},
            },
        },
        "smoke_values": smoke_values,
        "workloads": {name: _summary(raw[name]) for name, _ in workloads},
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
