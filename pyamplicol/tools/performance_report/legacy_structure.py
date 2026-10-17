# SPDX-License-Identifier: 0BSD
"""Authenticated structural evidence for original-AmpliCol measurements.

The performance campaign times an unmodified physics path.  Structural
evidence is collected separately:

* generated libraries are parsed from their persisted ``amp*_lib.f03`` files;
* direct imode-2 and contracted-library probes execute one additional point
  with diagnostic-only Fortran output enabled.

The pinned legacy checkout may be shared by manually steered cells.  This
module serializes its temporary instrumentation per checkout, adds diagnostics
to the two probe sources, snapshots the exact instrumented sources, and
restores the checkout byte-for-byte after the attempt.  The diagnostics do not
alter generation, evaluation, filtering, or contraction.  No structural count
is inferred from timing or filled with a fallback value.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .artifacts import _filesystem_lock
from .models import CellSpec, Workload

LEGACY_PROOF_SCHEMA = "pyamplicol-legacy-final-structural-proof-v1"
LEGACY_SCOPE_REASON = "original-amplicol-open-quark-line-limit"
STRUCTURAL_PROBE_ENVIRONMENT = "AMPICOL_STRUCTURAL_PROBE"
_EVIDENCE_DIRECTORY = "legacy-structural-evidence"
_INDEX_SCHEMA = "pyamplicol-legacy-structural-index-v1"
_SOURCE_CONTRACT_SCHEMA = "pyamplicol-legacy-structural-source-contract-v1"
_INSTRUMENTATION_ABI = "pyamplicol-legacy-probe-structural-diagnostics-v1"
_MAPPING_ABI = "pyamplicol-legacy-structural-object-mapping-v1"
_LOCK_DIRECTORY = "pyamplicol-performance-report-locks"

_MODULE_NAME = re.compile(r"amp(\d+)_(\d+)_lib\.f03$")
_MODULE_CURRENT = re.compile(
    r"dimension\s*\(\s*1\s*:\s*\d+\s*,\s*(\d+)\s*\)\s*::\s*val_c\b",
    re.IGNORECASE,
)
_MODULE_INTERACTION = re.compile(
    r"dimension\s*\(\s*1\s*:\s*\d+\s*,\s*(\d+)\s*\)\s*::\s*int_c\b",
    re.IGNORECASE,
)
_MODULE_AMPLITUDE = re.compile(
    r"complex\s*\(\s*kind\s*=\s*8\s*\)\s*,\s*dimension\s*\(\s*(\d+)\s*\)"
    r"\s*,\s*intent\s*\(\s*out\s*\)\s*::\s*amps\b",
    re.IGNORECASE,
)
_SUBROUTINE = re.compile(
    r"^\s*subroutine\s+(\w+)\b(.*?)^\s*end\s+subroutine\s+\1\b",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)
_EXTERNAL_CURRENT = re.compile(
    r"call\s+ext_\w+\s*\(.*?val_c\s*\(\s*1\s*,\s*(\d+)\s*\).*?\)",
    re.IGNORECASE | re.DOTALL,
)
_INT1_ARRAY = re.compile(
    r"integer\s*,\s*parameter\s*,\s*dimension\s*\(([^)]*)\)\s*::\s*"
    r"int1\s*=\s*(?:reshape\s*\(\s*)?\[\s*&?(.*?)\]"
    r"(?:\s*,\s*shape\s*=\s*\[[^]]*\]\s*\))?",
    re.IGNORECASE | re.DOTALL,
)
_COMBINE_SHAPE = re.compile(r"0:(\d+),(\d+)")
_AMPLITUDE_ASSIGNMENT = re.compile(
    r"amps\s*\(\s*(\d+)\s*\)\s*=\s*sum\s*\(\s*"
    r"val_c\s*\([^,]+,\s*(\d+)\s*\)\s*\*\s*"
    r"val_c\s*\([^,]+,\s*(\d+)\s*\)\s*\)",
    re.IGNORECASE,
)
_PROCESS_ROW = re.compile(r"group:(\d+):integral:(\d+)")

_DIRECT_CURRENT = re.compile(
    r"^AMPICOL_STRUCTURAL_CURRENT\s+"
    r"(\d+)\s+([01])\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*$",
    re.MULTILINE,
)
_DIRECT_KERNEL = re.compile(
    r"^AMPICOL_STRUCTURAL_KERNEL\s+"
    r"(\d+)\s+(-?\d+)\s+(-?\d+)\s+(\d+)\s+(\d+)\s+"
    r"(\S+)\s+(\S+)\s*$",
    re.MULTILINE,
)
_DIRECT_SINGLET = re.compile(
    r"^AMPICOL_STRUCTURAL_KERNEL_SINGLET\s+(\d+)\s+(.*?)\s*$",
    re.MULTILINE,
)
_DIRECT_ATTACHMENT = re.compile(
    r"^AMPICOL_STRUCTURAL_ATTACHMENT\s+"
    r"(\d+)\s+(\d+)\s+(\d+)\s+(-?1)\s*$",
    re.MULTILINE,
)
_DIRECT_DESTINATION = re.compile(
    r"^AMPICOL_STRUCTURAL_DESTINATION\s+"
    r"(\d+)\s+(\d+)\s+(\d+)\s*$",
    re.MULTILINE,
)
_DIRECT_SUMMARY = {
    "current": re.compile(r"^AMPICOL_COLOR_PROBE_CURRENTS\s+(\d+)\s*$", re.M),
    "kernel": re.compile(r"^AMPICOL_COLOR_PROBE_VERTICES\s+(\d+)\s*$", re.M),
    "destination": re.compile(
        r"^AMPICOL_COLOR_PROBE_AMPLITUDES\s+(\d+)\s*$", re.M
    ),
    "accuracy": re.compile(r"^color_accuracy\s+(\S+)\s*$", re.M),
}
_LIBRARY_CALL = re.compile(
    r"^AMPICOL_COLOR_PROBE_LIBRARY_CALLS\s+(\d+)\s+(\d+)\s+(\d+)\s*$",
    re.MULTILINE,
)
_LIBRARY_ROW = re.compile(
    r"^AMPICOL_COLOR_PROBE_LIBRARY_ROW\s+"
    r"(\d+)\s+(\d+)\s+(\d+)\s+(-?1)\s*$",
    re.MULTILINE,
)


class LegacyStructuralProofError(RuntimeError):
    """The real legacy generation/probe evidence is absent or inconsistent."""


@dataclass(frozen=True, slots=True)
class InstrumentationRecord:
    """Exact source identities for one temporary diagnostic installation."""

    abi: str
    sources: tuple[Mapping[str, str], ...]
    evidence_paths: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class _ModuleStructure:
    path: Path
    group: int
    integral: int
    declared_current_count: int
    declared_kernel_count: int
    source_current_ids: tuple[int, ...]
    produced_current_ids: tuple[int, ...]
    kernel_term_ids: tuple[int, ...]
    combine_routes: tuple[tuple[int, tuple[int, ...]], ...]
    amplitude_destinations: tuple[tuple[int, int, int], ...]

    @property
    def counts(self) -> dict[str, int]:
        return {
            "source_current_count": len(self.source_current_ids),
            "produced_current_count": len(self.produced_current_ids),
            "kernel_evaluation_count": len(self.kernel_term_ids),
            "attachment_count": sum(
                len(kernel_ids) for _current, kernel_ids in self.combine_routes
            ),
            "amplitude_destination_count": len(self.amplitude_destinations),
        }


@dataclass(frozen=True, slots=True)
class _DirectStructure:
    accuracy: str
    current_map: tuple[tuple[int, int, int, int, int], ...]
    kernel_map: tuple[tuple[int, int, int, int, int, str, str], ...]
    singlet_map: tuple[tuple[int, tuple[int, ...]], ...]
    attachment_map: tuple[tuple[int, int, int, int], ...]
    destination_map: tuple[tuple[int, int, int], ...]

    @property
    def counts(self) -> dict[str, int]:
        return {
            "source_current_count": sum(row[1] for row in self.current_map),
            "produced_current_count": sum(1 - row[1] for row in self.current_map),
            "kernel_evaluation_count": len(self.kernel_map),
            "attachment_count": len(self.attachment_map),
            "amplitude_destination_count": len(self.destination_map),
        }


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _canonical_sha256(value: object) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, value: object) -> None:
    _atomic_write(
        path,
        json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n",
    )


def _replace_once(text: str, old: str, new: str, *, label: str) -> str:
    if text.count(old) != 1:
        raise LegacyStructuralProofError(
            f"legacy probe instrumentation anchor {label!r} occurs "
            f"{text.count(old)} times, expected exactly once"
        )
    return text.replace(old, new, 1)


_DIRECT_SUBROUTINE = r"""
  subroutine print_structural_proof()
    implicit none
    integer :: current_id, kernel_id, route_index, destination_id
    integer :: source_flag, route_sign
    do current_id=1,amp%n_cur
       source_flag = 0
       if (popcnt(amp%current_list(current_id)%bin).eq.1) source_flag = 1
       write (*,'(a,5(1x,i0))') 'AMPICOL_STRUCTURAL_CURRENT',&
            current_id,source_flag,amp%current_list(current_id)%type,&
            amp%current_list(current_id)%chirality,&
            amp%current_list(current_id)%bin
       do route_index=1,amp%current_list(current_id)%n_vert
          route_sign = 1
          if (amp%current_list(current_id)%vertex_sign(route_index)) route_sign = -1
          write (*,'(a,4(1x,i0))') 'AMPICOL_STRUCTURAL_ATTACHMENT',&
               current_id,route_index,&
               amp%current_list(current_id)%vertices(route_index),route_sign
       enddo
    enddo
    do kernel_id=1,amp%n_vert
       write (*,'(a,5(1x,i0),2(1x,es24.16))') 'AMPICOL_STRUCTURAL_KERNEL',&
            kernel_id,amp%interaction_list(kernel_id)%type,&
            amp%interaction_list(kernel_id)%chirality,&
            amp%interaction_list(kernel_id)%currents(1),&
            amp%interaction_list(kernel_id)%currents(2),&
            amp%interaction_list(kernel_id)%coupl(1),&
            amp%interaction_list(kernel_id)%coupl(2)
       if (allocated(amp%interaction_list(kernel_id)%singlet_mv)) then
          write (*,'(a,1x,i0,1x,*(i0,1x))')&
               'AMPICOL_STRUCTURAL_KERNEL_SINGLET',kernel_id,&
               amp%interaction_list(kernel_id)%singlet_mv(&
               0:amp%interaction_list(kernel_id)%singlet_mv(0))
       else
          write (*,'(a,2(1x,i0))')&
               'AMPICOL_STRUCTURAL_KERNEL_SINGLET',kernel_id,0
       endif
    enddo
    do destination_id=1,amp%n_amps
       write (*,'(a,3(1x,i0))') 'AMPICOL_STRUCTURAL_DESTINATION',&
            destination_id,amp%curr2amp(1,destination_id),&
            amp%curr2amp(2,destination_id)
    enddo
  end subroutine print_structural_proof

"""


_LIBRARY_SUBROUTINE = r"""
  subroutine print_library_call_histogram()
    implicit none
    integer :: row, previous, call_count
    logical :: first_occurrence
    do row=1,colour_amp%nColOrd
       first_occurrence = .true.
       do previous=1,row-1
          if (row_to_group(previous).eq.row_to_group(row) .and.&
               row_to_integral(previous).eq.row_to_integral(row)) then
             first_occurrence = .false.
             exit
          endif
       enddo
       if (.not.first_occurrence) cycle
       call_count = count(row_to_group.eq.row_to_group(row) .and.&
            row_to_integral.eq.row_to_integral(row))
       write (*,'(a,3(1x,i0))') 'AMPICOL_COLOR_PROBE_LIBRARY_CALLS',&
            row_to_group(row),row_to_integral(row),call_count
       write (99,'(a,3(1x,i0))') 'AMPICOL_COLOR_PROBE_LIBRARY_CALLS',&
            row_to_group(row),row_to_integral(row),call_count
    enddo
    do row=1,colour_amp%nColOrd
       write (*,'(a,4(1x,i0))') 'AMPICOL_COLOR_PROBE_LIBRARY_ROW',&
            row,row_to_group(row),row_to_integral(row),row_sign(row)
       write (99,'(a,4(1x,i0))') 'AMPICOL_COLOR_PROBE_LIBRARY_ROW',&
            row,row_to_group(row),row_to_integral(row),row_sign(row)
    enddo
  end subroutine print_library_call_histogram

"""


def _instrument_direct_source(text: str) -> str:
    if "subroutine print_structural_proof()" in text:
        raise LegacyStructuralProofError("direct probe is already instrumented")
    text = _replace_once(
        text,
        "  logical :: print_matrix, fixed_helicity\n",
        "  logical :: print_matrix, fixed_helicity, print_structure\n",
        label="direct logical declaration",
    )
    text = _replace_once(
        text,
        "  print_matrix = .false.\n  fixed_helicity = .false.\n",
        "  print_matrix = .false.\n"
        "  fixed_helicity = .false.\n"
        "  print_structure = .false.\n",
        label="direct logical initialization",
    )
    text = _replace_once(
        text,
        "  if (trim(adjustl(env_value)).eq.'1') print_matrix = .true.\n"
        "  env_value = ''\n",
        "  if (trim(adjustl(env_value)).eq.'1') print_matrix = .true.\n"
        "  env_value = ''\n"
        "  call get_environment_variable('AMPICOL_STRUCTURAL_PROBE',env_value)\n"
        "  if (trim(adjustl(env_value)).eq.'1') print_structure = .true.\n"
        "  env_value = ''\n",
        label="direct structural environment",
    )
    text = _replace_once(
        text,
        "  call print_recursion_counts()\n",
        "  call print_recursion_counts()\n"
        "  if (print_structure) call print_structural_proof()\n",
        label="direct structural call",
    )
    return _replace_once(
        text,
        "  subroutine parse_color_accuracy()\n",
        _DIRECT_SUBROUTINE + "  subroutine parse_color_accuracy()\n",
        label="direct structural subroutine",
    )


def _instrument_library_source(text: str) -> str:
    if "subroutine print_library_call_histogram()" in text:
        raise LegacyStructuralProofError("library probe is already instrumented")
    text = _replace_once(
        text,
        "  logical :: print_matrix\n",
        "  logical :: print_matrix, print_structure\n",
        label="library logical declaration",
    )
    text = _replace_once(
        text,
        "  print_matrix = .false.\n"
        "  call get_environment_variable('AMPICOL_COLOR_PROBE_MATRIX',env_value)\n",
        "  print_matrix = .false.\n"
        "  print_structure = .false.\n"
        "  call get_environment_variable('AMPICOL_COLOR_PROBE_MATRIX',env_value)\n",
        label="library logical initialization",
    )
    text = _replace_once(
        text,
        "  if (trim(adjustl(env_value)).eq.'1') print_matrix = .true.\n\n"
        "  argc = command_argument_count()\n",
        "  if (trim(adjustl(env_value)).eq.'1') print_matrix = .true.\n"
        "  env_value = ''\n"
        "  call get_environment_variable('AMPICOL_STRUCTURAL_PROBE',env_value)\n"
        "  if (trim(adjustl(env_value)).eq.'1') print_structure = .true.\n\n"
        "  argc = command_argument_count()\n",
        label="library structural environment",
    )
    text = _replace_once(
        text,
        "  call build_row_to_integral()\n",
        "  call build_row_to_integral()\n"
        "  if (print_structure) call print_library_call_histogram()\n",
        label="library structural call",
    )
    return _replace_once(
        text,
        "  integer function colour_order_match_sign(jgroup,jint,row,pass,leg_map)\n",
        _LIBRARY_SUBROUTINE
        + "  integer function colour_order_match_sign(jgroup,jint,row,pass,leg_map)\n",
        label="library structural subroutine",
    )


def _legacy_structural_probe_lock_path(repository: Path) -> Path:
    resolved = repository.expanduser().resolve(strict=False)
    repository_id = hashlib.sha256(os.fsencode(resolved)).hexdigest()
    return (
        Path(tempfile.gettempdir())
        / f"{_LOCK_DIRECTORY}-{os.getuid()}"
        / f"{repository_id}.lock"
    )


@contextmanager
def legacy_structural_probe_lock(repository: Path) -> Iterator[None]:
    """Serialize temporary probe-source edits for one shared legacy checkout."""

    with _filesystem_lock(
        _legacy_structural_probe_lock_path(repository),
        timeout=None,
        poll_interval=0.05,
    ):
        yield


@contextmanager
def instrument_legacy_structural_probes(
    repository: Path,
    artifact_path: Path,
) -> Iterator[InstrumentationRecord]:
    """Install diagnostic-only probe output and restore exact source bytes."""

    with (
        legacy_structural_probe_lock(repository),
        _instrument_legacy_structural_probes(
            repository,
            artifact_path,
        ) as instrumentation,
    ):
        yield instrumentation


@contextmanager
def _instrument_legacy_structural_probes(
    repository: Path,
    artifact_path: Path,
) -> Iterator[InstrumentationRecord]:
    evidence_root = artifact_path / _EVIDENCE_DIRECTORY
    transformers = {
        "amplicol_color_probe.f03": _instrument_direct_source,
        "amplicol_color_library_probe.f03": _instrument_library_source,
    }
    originals: dict[Path, tuple[bytes, os.stat_result]] = {}
    records: list[Mapping[str, str]] = []
    evidence_paths: list[Path] = []
    try:
        for name, transform in transformers.items():
            source = repository / name
            if not source.is_file():
                raise LegacyStructuralProofError(
                    f"pinned legacy probe source is absent: {source}"
                )
            original = source.read_bytes()
            stat = source.stat()
            originals[source] = (original, stat)
            try:
                patched = transform(original.decode("utf-8")).encode("utf-8")
            except UnicodeDecodeError as error:
                raise LegacyStructuralProofError(
                    f"legacy probe source is not UTF-8: {source}"
                ) from error
            snapshot = evidence_root / f"instrumented-{name}"
            _atomic_write(snapshot, patched)
            _atomic_write(source, patched)
            os.chmod(source, stat.st_mode)
            records.append(
                {
                    "path": name,
                    "original_sha256": _sha256_bytes(original),
                    "instrumented_sha256": _sha256_bytes(patched),
                    "instrumented_evidence_path": snapshot.relative_to(
                        artifact_path
                    ).as_posix(),
                }
            )
            evidence_paths.append(snapshot)
        yield InstrumentationRecord(
            abi=_INSTRUMENTATION_ABI,
            sources=tuple(records),
            evidence_paths=tuple(evidence_paths),
        )
    finally:
        for source, (payload, stat) in originals.items():
            _atomic_write(source, payload)
            os.chmod(source, stat.st_mode)
            os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))


def _required_match(pattern: re.Pattern[str], text: str, label: str) -> re.Match[str]:
    match = pattern.search(text)
    if match is None:
        raise LegacyStructuralProofError(f"legacy structural evidence lacks {label}")
    return match


def _parse_module(path: Path) -> _ModuleStructure:
    name = _MODULE_NAME.fullmatch(path.name)
    if name is None:
        raise LegacyStructuralProofError(f"invalid generated module name: {path}")
    text = path.read_text(encoding="utf-8")
    declared_current_count = int(
        _required_match(_MODULE_CURRENT, text, "module current dimension").group(1)
    )
    declared_kernel_count = int(
        _required_match(_MODULE_INTERACTION, text, "module kernel dimension").group(1)
    )
    source_ids = {int(value) for value in _EXTERNAL_CURRENT.findall(text)}
    produced_ids: set[int] = set()
    kernel_ids: set[int] = set()
    referenced_kernel_ids: set[int] = set()
    routes: list[tuple[int, tuple[int, ...]]] = []
    for match in _SUBROUTINE.finditer(text):
        routine = match.group(1).lower()
        array = _INT1_ARRAY.search(match.group(2))
        if array is None:
            continue
        values = [int(value) for value in re.findall(r"\d+", array.group(2))]
        if routine.startswith("vertex_"):
            kernel_ids.update(values)
            continue
        if not routine.startswith("combine_currents"):
            continue
        shape = _COMBINE_SHAPE.fullmatch(array.group(1).replace(" ", ""))
        if shape is None:
            raise LegacyStructuralProofError(
                f"unsupported combine-route shape in {path}"
            )
        stride = int(shape.group(1)) + 1
        column_count = int(shape.group(2))
        if len(values) != stride * column_count:
            raise LegacyStructuralProofError(
                f"combine-route payload has the wrong size in {path}"
            )
        for start in range(0, len(values), stride):
            current = values[start]
            contributions = tuple(values[start + 1 : start + stride])
            produced_ids.add(current)
            referenced_kernel_ids.update(contributions)
            routes.append((current, contributions))
    destinations = tuple(
        (int(index), int(left), int(right))
        for index, left, right in _AMPLITUDE_ASSIGNMENT.findall(text)
    )
    amplitude_count = int(
        _required_match(_MODULE_AMPLITUDE, text, "module amplitude dimension").group(
            1
        )
    )
    expected_currents = set(range(1, declared_current_count + 1))
    expected_kernels = set(range(1, declared_kernel_count + 1))
    if (
        source_ids & produced_ids
        or source_ids | produced_ids != expected_currents
        or kernel_ids != expected_kernels
        or referenced_kernel_ids != expected_kernels
        or [item[0] for item in destinations] != list(range(1, amplitude_count + 1))
        or any(
            left not in expected_currents or right not in expected_currents
            for _index, left, right in destinations
        )
    ):
        raise LegacyStructuralProofError(
            f"generated module has sparse/dangling structural objects: {path}"
        )
    return _ModuleStructure(
        path=path,
        group=int(name.group(1)),
        integral=int(name.group(2)),
        declared_current_count=declared_current_count,
        declared_kernel_count=declared_kernel_count,
        source_current_ids=tuple(sorted(source_ids)),
        produced_current_ids=tuple(sorted(produced_ids)),
        kernel_term_ids=tuple(sorted(kernel_ids)),
        combine_routes=tuple(routes),
        amplitude_destinations=destinations,
    )


def _parse_direct(path: Path, expected_accuracy: str) -> _DirectStructure:
    text = path.read_text(encoding="utf-8")
    current_map = tuple(
        tuple(int(value) for value in match)
        for match in _DIRECT_CURRENT.findall(text)
    )
    kernel_map = tuple(
        (
            int(identifier),
            int(kind),
            int(chirality),
            int(left),
            int(right),
            coupling_left,
            coupling_right,
        )
        for (
            identifier,
            kind,
            chirality,
            left,
            right,
            coupling_left,
            coupling_right,
        ) in _DIRECT_KERNEL.findall(text)
    )
    singlet_map = tuple(
        (
            int(identifier),
            tuple(int(value) for value in values.split()),
        )
        for identifier, values in _DIRECT_SINGLET.findall(text)
    )
    attachment_map = tuple(
        tuple(int(value) for value in match)
        for match in _DIRECT_ATTACHMENT.findall(text)
    )
    destination_map = tuple(
        tuple(int(value) for value in match)
        for match in _DIRECT_DESTINATION.findall(text)
    )
    counts = {
        label: int(_required_match(pattern, text, label).group(1))
        for label, pattern in _DIRECT_SUMMARY.items()
        if label != "accuracy"
    }
    accuracy = _required_match(
        _DIRECT_SUMMARY["accuracy"], text, "color accuracy"
    ).group(1)
    if accuracy != expected_accuracy:
        raise LegacyStructuralProofError(
            f"direct structural probe accuracy is {accuracy}, "
            f"expected {expected_accuracy}"
        )
    current_ids = [row[0] for row in current_map]
    kernel_ids = [row[0] for row in kernel_map]
    singlet_ids = [row[0] for row in singlet_map]
    destination_ids = [row[0] for row in destination_map]
    current_id_set = set(current_ids)
    kernel_id_set = set(kernel_ids)
    if (
        current_ids != list(range(1, counts["current"] + 1))
        or kernel_ids != list(range(1, counts["kernel"] + 1))
        or singlet_ids != kernel_ids
        or destination_ids != list(range(1, counts["destination"] + 1))
        or any(
            left not in current_id_set or right not in current_id_set
            for _identifier, _kind, _chirality, left, right, _coupling_a, _coupling_b
            in kernel_map
        )
        or any(
            current not in current_id_set
            or kernel not in kernel_id_set
            or position < 1
            for current, position, kernel, _sign in attachment_map
        )
        or any(
            left not in current_id_set or right not in current_id_set
            for _destination, left, right in destination_map
        )
        or set(range(1, counts["kernel"] + 1))
        != {row[2] for row in attachment_map}
    ):
        raise LegacyStructuralProofError(
            f"direct structural probe has sparse/dangling objects: {path}"
        )
    positions: Counter[int] = Counter()
    for current, position, _kernel, _sign in attachment_map:
        positions[current] += 1
        if positions[current] != position:
            raise LegacyStructuralProofError(
                f"direct structural attachment positions are not contiguous: {path}"
            )
    return _DirectStructure(
        accuracy=accuracy,
        current_map=current_map,
        kernel_map=kernel_map,
        singlet_map=singlet_map,
        attachment_map=attachment_map,
        destination_map=destination_map,
    )


def _parse_library_histogram(
    path: Path,
    modules: Mapping[tuple[int, int], _ModuleStructure],
) -> tuple[tuple[tuple[int, int, int], ...], tuple[tuple[int, int, int, int], ...]]:
    text = path.read_text(encoding="utf-8")
    histogram = tuple(
        tuple(int(value) for value in match)
        for match in _LIBRARY_CALL.findall(text)
    )
    rows = tuple(
        tuple(int(value) for value in match)
        for match in _LIBRARY_ROW.findall(text)
    )
    if not histogram or not rows:
        raise LegacyStructuralProofError(
            f"contracted probe lacks exact row-to-module calls: {path}"
        )
    row_ids = [row[0] for row in rows]
    if row_ids != list(range(1, len(rows) + 1)):
        raise LegacyStructuralProofError(
            f"contracted probe row axis is not contiguous: {path}"
        )
    observed = Counter((group, integral) for _row, group, integral, _sign in rows)
    declared = Counter(
        {(group, integral): count for group, integral, count in histogram}
    )
    if (
        observed != declared
        or len(declared) != len(histogram)
        or any(key not in modules for key in declared)
    ):
        raise LegacyStructuralProofError(
            f"contracted probe row histogram is incomplete or stale: {path}"
        )
    return histogram, rows


def _sum_counts(
    modules: Mapping[tuple[int, int], _ModuleStructure],
    multiplicities: Mapping[tuple[int, int], int],
) -> dict[str, int]:
    fields = (
        "source_current_count",
        "produced_current_count",
        "kernel_evaluation_count",
        "attachment_count",
        "amplitude_destination_count",
    )
    return {
        field: sum(
            module.counts[field] * multiplicities.get(key, 0)
            for key, module in modules.items()
        )
        for field in fields
    }


def _mapping_payload(
    modules: Mapping[tuple[int, int], _ModuleStructure],
) -> dict[str, object]:
    ordered = [modules[key] for key in sorted(modules)]
    return {
        "current_map": [
            {
                "module": [module.group, module.integral],
                "source_ids": list(module.source_current_ids),
                "produced_ids": list(module.produced_current_ids),
            }
            for module in ordered
        ],
        "kernel_map": [
            {
                "module": [module.group, module.integral],
                "kernel_term_ids": list(module.kernel_term_ids),
            }
            for module in ordered
        ],
        "combine_route_map": [
            {
                "module": [module.group, module.integral],
                "routes": [
                    [current, list(kernel_ids)]
                    for current, kernel_ids in module.combine_routes
                ],
            }
            for module in ordered
        ],
        "destination_map": [
            {
                "module": [module.group, module.integral],
                "destinations": [list(item) for item in module.amplitude_destinations],
            }
            for module in ordered
        ],
    }


def _direct_mapping_payload(structure: _DirectStructure) -> dict[str, object]:
    return {
        "current_map": [list(item) for item in structure.current_map],
        "kernel_map": [list(item) for item in structure.kernel_map],
        "kernel_singlet_map": [
            [identifier, list(values)] for identifier, values in structure.singlet_map
        ],
        "combine_route_map": [list(item) for item in structure.attachment_map],
        "destination_map": [list(item) for item in structure.destination_map],
    }


def _copy_evidence(source: Path, destination: Path) -> Path:
    if not source.is_file():
        raise LegacyStructuralProofError(f"legacy evidence file is absent: {source}")
    _atomic_write(destination, source.read_bytes())
    return destination


def write_legacy_structural_probe_output(
    path: Path,
    *,
    stdout: str,
    stderr: str,
) -> None:
    """Atomically preserve one dedicated diagnostic probe transcript."""

    payload = stdout
    if payload and not payload.endswith("\n"):
        payload += "\n"
    payload += stderr
    if payload and not payload.endswith("\n"):
        payload += "\n"
    _atomic_write(path, payload.encode("utf-8"))


def _evidence_inventory(
    artifact_path: Path,
    paths: set[Path],
) -> list[dict[str, str]]:
    root = artifact_path.resolve()
    inventory: list[dict[str, str]] = []
    for path in sorted(paths):
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(root)
        except ValueError as error:
            raise LegacyStructuralProofError(
                f"legacy evidence escapes artifact: {path}"
            ) from error
        if not resolved.is_file():
            raise LegacyStructuralProofError(
                f"legacy evidence disappeared before publication: {path}"
            )
        inventory.append(
            {"path": relative.as_posix(), "sha256": _sha256(resolved)}
        )
    if not inventory:
        raise LegacyStructuralProofError("legacy evidence inventory is empty")
    return inventory


def emit_legacy_structural_proof(
    cell: CellSpec,
    *,
    artifact_path: Path,
    process_row: str,
    source_revision: str,
    repository: Path,
    instrumentation: InstrumentationRecord,
) -> Path:
    """Validate real structural evidence and atomically publish its sidecar."""

    evidence_root = artifact_path / _EVIDENCE_DIRECTORY
    evidence_paths = set(instrumentation.evidence_paths)
    process_path = artifact_path / "processes.txt"
    log_path = artifact_path / "legacy.log"
    evidence_paths.update((process_path, log_path))
    modules_by_key: dict[tuple[int, int], _ModuleStructure] = {}
    for path in sorted(artifact_path.rglob("amp*_lib.f03")):
        module = _parse_module(path)
        key = (module.group, module.integral)
        if key in modules_by_key:
            raise LegacyStructuralProofError(
                f"duplicate generated module identity {key} in {artifact_path}"
            )
        modules_by_key[key] = module
        evidence_paths.add(path)

    direct_path = evidence_root / "direct-structural-probe.stdout"
    contracted_path = evidence_root / "contracted-structural-probe.stdout"
    mapping: dict[str, object]
    histogram: tuple[tuple[int, int, int], ...]
    row_map: tuple[tuple[int, int, int, int], ...]
    if direct_path.is_file():
        if modules_by_key:
            raise LegacyStructuralProofError(
                "direct structural evidence unexpectedly contains generated modules"
            )
        direct = _parse_direct(direct_path, cell.measurement.accuracy.value)
        mapping = _direct_mapping_payload(direct)
        active = direct.counts
        static = direct.counts
        row = _PROCESS_ROW.fullmatch(process_row)
        if row is None:
            raise LegacyStructuralProofError(
                f"invalid direct-probe process row: {process_row!r}"
            )
        histogram = ((int(row.group(1)), int(row.group(2)), 1),)
        row_map = (
            (1, int(row.group(1)), int(row.group(2)), 1),
        )
        evidence_paths.add(direct_path)
        executable = _copy_evidence(
            repository / "amplicol_color_probe",
            evidence_root / "amplicol_color_probe",
        )
        evidence_paths.add(executable)
        evidence_kind = "direct-imode2-exact-object-map"
    else:
        if not modules_by_key:
            raise LegacyStructuralProofError(
                "generated-library structural proof has no modules"
            )
        mapping = _mapping_payload(modules_by_key)
        static = _sum_counts(
            modules_by_key,
            {key: 1 for key in modules_by_key},
        )
        if cell.workload is Workload.SELECTED_FLOW:
            row = _PROCESS_ROW.fullmatch(process_row)
            if row is None:
                raise LegacyStructuralProofError(
                    f"invalid selected-flow process row: {process_row!r}"
                )
            key = (int(row.group(1)), int(row.group(2)))
            if key not in modules_by_key:
                raise LegacyStructuralProofError(
                    f"selected-flow generated module is absent: {key}"
                )
            histogram = ((key[0], key[1], 1),)
            row_map = ((1, key[0], key[1], 1),)
            active = _sum_counts(modules_by_key, {key: 1})
            executable = _copy_evidence(
                artifact_path
                / "selected-flow-generated-library"
                / "amplicol_library_benchmark",
                evidence_root / "amplicol_library_benchmark",
            )
            evidence_paths.add(executable)
            evidence_kind = "generated-library-selected-row-exact-object-map"
        elif cell.workload is Workload.CONTRACTED:
            histogram, row_map = _parse_library_histogram(
                contracted_path,
                modules_by_key,
            )
            active = _sum_counts(
                modules_by_key,
                {(group, integral): count for group, integral, count in histogram},
            )
            evidence_paths.add(contracted_path)
            output = (
                artifact_path
                / "contracted-generated-library"
                / "amplicol_color_library_probe.output"
            )
            evidence_paths.add(output)
            executable = (
                artifact_path
                / "contracted-generated-library"
                / "amplicol_color_library_probe"
            )
            evidence_paths.add(executable)
            evidence_kind = "generated-library-contracted-exact-call-map"
        else:
            raise LegacyStructuralProofError(
                "generated-library evidence cannot certify this workload"
            )

    mapping_sections = {
        "current_object_map_sha256": _canonical_sha256(mapping["current_map"]),
        "kernel_term_map_sha256": _canonical_sha256(mapping["kernel_map"]),
        "combine_route_map_sha256": _canonical_sha256(
            mapping["combine_route_map"]
        ),
        "amplitude_destination_map_sha256": _canonical_sha256(
            mapping["destination_map"]
        ),
    }
    source_contract = {
        "schema": _SOURCE_CONTRACT_SCHEMA,
        "cell_id": cell.cell_id,
        "process_key": cell.process_key,
        "process": cell.process,
        "n_final": cell.n_final,
        "accuracy": cell.measurement.accuracy.value,
        "workload": cell.workload.value,
        "process_row": process_row,
        "source_revision": source_revision,
        "instrumentation": {
            "abi": instrumentation.abi,
            "sources": list(instrumentation.sources),
        },
        "scope": "available",
    }
    source_contract_path = evidence_root / "source-contract.json"
    _atomic_json(source_contract_path, source_contract)
    evidence_paths.add(source_contract_path)
    source_contract_sha256 = _canonical_sha256(source_contract)
    histogram_payload = {
        "histogram": [list(item) for item in histogram],
        "row_map": [list(item) for item in row_map],
    }
    index = {
        "schema": _INDEX_SCHEMA,
        "evidence_kind": evidence_kind,
        "active": active,
        "static": static,
        "object_mapping": {
            **mapping_sections,
            "source_contract_sha256": source_contract_sha256,
        },
        "row_multiplicity": histogram_payload,
        "module_count": len(modules_by_key),
    }
    index_path = evidence_root / "legacy-structural-index.json"
    _atomic_json(index_path, index)
    evidence_paths.add(index_path)
    proof = {
        "schema": LEGACY_PROOF_SCHEMA,
        "cell_id": cell.cell_id,
        "source_revision": source_revision,
        "scope": "available",
        "accuracy": cell.measurement.accuracy.value,
        "workload": cell.workload.value,
        "active": active,
        "static": static,
        "object_mapping": {
            "status": "exact",
            "abi": _MAPPING_ABI,
            "accuracy": cell.measurement.accuracy.value,
            **mapping_sections,
            "source_contract_sha256": source_contract_sha256,
        },
        "row_multiplicity": {
            "status": "exact",
            "scope": cell.workload.value,
            "call_count": sum(count for _group, _integral, count in histogram),
            "histogram_sha256": _canonical_sha256(histogram_payload),
        },
        "evidence_files": _evidence_inventory(artifact_path, evidence_paths),
    }
    proof_path = artifact_path / "legacy-structural-proof.json"
    _atomic_json(proof_path, proof)
    return proof_path


def emit_legacy_scope_unavailable_proof(
    cell: CellSpec,
    *,
    artifact_path: Path,
    source_revision: str,
    maximum_open_quark_lines: int,
    observed_open_quark_lines: int,
) -> Path:
    """Publish the exact original-AmpliCol scope boundary, with no fake counts."""

    contract = {
        "schema": _SOURCE_CONTRACT_SCHEMA,
        "cell_id": cell.cell_id,
        "process_key": cell.process_key,
        "process": cell.process,
        "n_final": cell.n_final,
        "accuracy": cell.measurement.accuracy.value,
        "workload": cell.workload.value,
        "source_revision": source_revision,
        "scope": "unavailable",
        "reason": LEGACY_SCOPE_REASON,
        "maximum_open_quark_lines": maximum_open_quark_lines,
        "observed_open_quark_lines": observed_open_quark_lines,
    }
    contract_path = artifact_path / _EVIDENCE_DIRECTORY / "source-contract.json"
    _atomic_json(contract_path, contract)
    proof = {
        "schema": LEGACY_PROOF_SCHEMA,
        "cell_id": cell.cell_id,
        "source_revision": source_revision,
        "scope": "unavailable",
        "reason": LEGACY_SCOPE_REASON,
        "maximum_open_quark_lines": maximum_open_quark_lines,
        "observed_open_quark_lines": observed_open_quark_lines,
        "source_contract_sha256": _canonical_sha256(contract),
        "evidence_files": _evidence_inventory(artifact_path, {contract_path}),
    }
    proof_path = artifact_path / "legacy-structural-proof.json"
    _atomic_json(proof_path, proof)
    return proof_path


__all__ = [
    "LEGACY_PROOF_SCHEMA",
    "LEGACY_SCOPE_REASON",
    "STRUCTURAL_PROBE_ENVIRONMENT",
    "InstrumentationRecord",
    "LegacyStructuralProofError",
    "emit_legacy_scope_unavailable_proof",
    "emit_legacy_structural_proof",
    "instrument_legacy_structural_probes",
    "legacy_structural_probe_lock",
    "write_legacy_structural_probe_output",
]
