#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Portability acceptance of prepared pyAmpliCol artifacts (requirements addendum G).

A prepared artifact (tree or plan v4 loop process, SymJIT -O2 by default) must
evaluate on another machine, user and path: no absolute path, user or host
name in it, no load-time regeneration, nothing tied to the build host's CPU
features, and bitwise-identical binary64 results after relocation with the
same native build. This script checks those properties on one host and
produces the records to compare across hosts. It needs only the installed
pyAmpliCol (and, for ``sdk-build``, a C compiler and the SDK).

Subcommands (``--help`` on each):

``host``
    The host's CPU class as SymJIT selects its code path (x86-64 with AVX: the
    AVX code, which contracts complex products with FMA; without AVX: SSE;
    AArch64), and the facts a cross-host comparison must quote.
``scan ARTIFACT``
    Every file and every PACBIN member (zlib members decompressed) for
    absolute paths, the generating host's home, working and temporary
    directories, user and host names; the PACBIN member kinds (a portable
    artifact holds no native library, and its SymJIT applications are
    storage-v4 IR that the loader recompiles); the manifest's target (a
    portable artifact declares ``portable-64le`` without CPU features).
``points --legs N --count K --out FILE``
    Massless phase-space points (binary64, stored as hexadecimal literals)
    and one mu_R^2 per point, the input of ``evaluate`` and ``sdk``.
``evaluate ARTIFACT --process ID --points FILE --out FILE``
    Loads and evaluates through the public Python API (``pyamplicol.Runtime``)
    under a generation sentinel: every call of a generation entry point
    (``GENERATION_CODE``, ``GENERATION_FILES``) is counted, from before the
    first pyAmpliCol import to the end of the evaluation. Records load and
    evaluation times and every binary64 result as a hexadecimal literal.
``sdk-build --out EXE``
    Compiles the C client below (``CLIENT_SOURCE``) against the loop SDK
    (``rusticol_virtual.h``): from a wheel's ``rusticol-config`` or from a
    source checkout and a native build directory.
``sdk ARTIFACT --process ID --points FILE --client EXE --out FILE [--poles P]``
    The same evaluation through the C SDK (``rusticol_virtual_open``; with
    ``--poles`` the pole selection, ``rusticol_virtual_set_poles``),
    optionally without ``SYMBOLICA_LICENSE`` and with concurrent instances:
    the f64 SDK must need no Symbolica licence (the tree engine's rule,
    ``docs/user/symbolica-and-licensing.md``), so a restricted-Symbolica
    banner or an aborted concurrent instance is a finding.
    With ``--hold-permit`` the instances run while a plain Symbolica process
    holds this user's restricted thread permit (``RestrictedPermit``), so any
    Symbolica state access aborts deterministically.
``compare A B``
    Bitwise comparison of two ``evaluate``/``sdk``/``tree-capi`` records.
``check ARTIFACT --process ID --scratch DIR``
    All of it on one host: scan, a copy at a fresh path, ``evaluate`` of the
    original and of the copy in fresh interpreters with other working,
    home and temporary directories and another user name (and, with
    ``--no-licence``, without ``SYMBOLICA_LICENSE``), the comparison, with
    ``--hold-permit`` ``--instances`` concurrent evaluations of the copy while
    the user's restricted permit is held, and, with ``--client``, the SDK at
    both paths. ``--arithmetic``/``--precision`` select the loop carrier of
    the Python evaluations. Prints one JSON verdict.
``capi-build ARTIFACT --out EXE (--archive librusticol_capi.a | --rusticol-config C)``
    Compiles a tree artifact's own C example (``API/c/check_standalone.c`` of
    its API bundle) against the tree C API: a wheel's ``rusticol-config`` or a
    static archive built with ``cargo rustc --release -p rusticol-capi
    --crate-type staticlib -- --print native-static-libs`` (``--libs``: the
    printed native libraries).
``tree-capi ARTIFACT --client EXE --scratch DIR``
    The tree C API at the original and at a relocated copy, by default
    without ``SYMBOLICA_LICENSE`` in two concurrent processes while the
    permit is held, compared bitwise with each other and with the bundle's
    Python example (the public ``Runtime``) at the stored validation momenta.

Cross-host use: run ``points`` and ``evaluate`` on the build host, copy the
artifact directory, the points file and this script to the other host (or
container), run ``evaluate`` (and ``sdk``) there with the same pyAmpliCol
build, and ``compare`` the records. Bitwise equality is required for the same
native build and the same SymJIT code-path class (``host``); across classes
(AVX vs SSE, or x86-64 vs AArch64) SymJIT-compiled tree kernels may differ in
the last bits (FMA contraction), which ``compare`` reports with the
relative difference.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import zlib
from collections import Counter
from pathlib import Path
from typing import Any

SCHEMA = 1

# ---------------------------------------------------------------- sentinel

# Generation entry points: a load of a prepared artifact must call none of
# them. (path suffix, qualified name) of single functions, and whole modules
# that only generation uses.
GENERATION_CODE = frozenset(
    {
        ("pyamplicol/loops/v2/artifact.py", "generate_process"),
        ("pyamplicol/loops/v2/assemble.py", "LoopSlice.generate"),
        ("pyamplicol/loops/v2/plan_export.py", "export_plan"),
        ("pyamplicol/models/loading.py", "compile_model_source"),
        ("pyamplicol/generation/service.py", "GenerationBackend.generate"),
        ("pyamplicol/api/services.py", "Generator.generate"),
        # The native-trees export: the tree, counterterm and parameter
        # programs and their SymJIT plane applications.
        ("pyamplicol/loops/v2/native_trees.py", "export_trees"),
        ("pyamplicol/loops/v2/native_trees.py", "_plane_applications"),
        ("pyamplicol/loops/v2/native_trees.py", "_interaction_program"),
        ("pyamplicol/loops/v2/native_trees.py", "_counterterm_program"),
        ("pyamplicol/loops/v2/native_trees.py", "_parameter_program"),
        # The observables' slice and column plans, and the export of the
        # stored observable columns.
        ("pyamplicol/loops/v2/observables.py", "ObservableSlice.generate"),
        ("pyamplicol/loops/v2/observables.py", "ColumnPlan.__init__"),
        ("pyamplicol/loops/v2/observables.py", "NativeObservables.__init__"),
        ("pyamplicol/loops/v2/native_observables.py", "StoredObservables.export"),
        # The slice's own stages (also reachable outside LoopSlice.generate):
        # the certified-scaleless companion and its seeds, the LC/NLC colour
        # prune, the seed tables (Python and native) and the native colour
        # cells' table and projection stages.
        ("pyamplicol/loops/v2/assemble.py", "LoopSlice._scaleless_companion"),
        ("pyamplicol/loops/v2/assemble.py", "LoopSlice._companion_seeds"),
        ("pyamplicol/loops/v2/assemble.py", "LoopSlice._prune_colour"),
        ("pyamplicol/loops/v2/colour.py", "build_seed_table"),
        ("pyamplicol/loops/v2/colour.py", "extend_seed_table"),
        ("pyamplicol/loops/v2/native_seeds.py", "build_seed_table"),
        ("pyamplicol/loops/v2/native_seeds.py", "extend_seed_table"),
        ("pyamplicol/loops/v2/native_seeds.py", "NativeSeedStage.__init__"),
        ("pyamplicol/loops/v2/native_cells.py", "cell_table"),
        ("pyamplicol/loops/v2/native_cells.py", "project_records"),
        ("pyamplicol/loops/v2/native_cells.py", "directed_table"),
    }
)
GENERATION_FILES = (
    "pyamplicol/generation/dag_compiler.py",
    # The native generator's and the native DAG's inputs (they call
    # `_loop.V2Generation` and `_loop.V2Dag`).
    "pyamplicol/loops/v2/generator_input.py",
    "pyamplicol/loops/v2/dag_input.py",
    "pyamplicol/loops/v2/blocks.py",
    "pyamplicol/loops/v2/skeletons.py",
    "pyamplicol/loops/v2/schedules.py",
    "pyamplicol/loops/v2/ct_closure.py",
    "pyamplicol/loops/v2/ct_recursion.py",
    "pyamplicol/loops/v2/ct_sites.py",
)
CO_OPTIMIZED = 0x0001  # inspect.CO_OPTIMIZED: set on function code only


class GenerationSentinel:
    """Counts calls of generation code (``sys.setprofile`` on this thread and
    every thread started while installed)."""

    def __init__(self) -> None:
        self.calls: Counter[str] = Counter()
        self._known: dict[int, str | None] = {}

    def _classify(self, code: Any) -> str | None:
        key = id(code)
        if key not in self._known:
            filename = code.co_filename.replace(os.sep, "/")
            qualname = getattr(code, "co_qualname", code.co_name)
            label = None
            if not code.co_flags & CO_OPTIMIZED:
                # Module and class bodies run on import, which is not
                # generation; only function calls count.
                self._known[key] = None
                return None
            for suffix, name in GENERATION_CODE:
                if filename.endswith(suffix) and qualname == name:
                    label = f"{suffix}:{name}"
            if label is None and filename.endswith(GENERATION_FILES):
                label = f"{filename.rsplit('pyamplicol/', 1)[-1]}:{qualname}"
            self._known[key] = label
        return self._known[key]

    def _profile(self, frame: Any, event: str, arg: Any) -> None:
        if event == "call":
            label = self._classify(frame.f_code)
            caller = frame.f_back
            # A call from a module body is import-time set-up (for example a
            # policy closure built at module level), not generation.
            if label is not None and not (
                caller is not None and caller.f_code.co_name == "<module>"
            ):
                self.calls[label] += 1

    def __enter__(self) -> GenerationSentinel:
        import threading

        threading.setprofile(self._profile)
        sys.setprofile(self._profile)
        return self

    def __exit__(self, *exc: object) -> None:
        import threading

        sys.setprofile(None)
        threading.setprofile(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------- host


def _cpuid_flags() -> set[str] | None:
    """The CPU's own SymJIT-relevant feature bits, from the native extension
    (``_rusticol._symjit_host_cpu``: CPUID with the OS's XSAVE state, as
    SymJIT and the loaders' CPU gate read them); ``/proc/cpuinfo`` shows the
    physical host's under user-mode emulation (qemu-user). None on another
    architecture or without the native extension."""

    if platform.machine().lower() not in ("x86_64", "amd64"):
        return None
    try:
        from pyamplicol import _rusticol

        features, _supported = _rusticol._symjit_host_cpu()
    except Exception:
        return None
    return set(features)


def _cpu_flags() -> set[str]:
    flags = _cpuid_flags()
    if flags is not None:
        return flags
    try:
        text = Path("/proc/cpuinfo").read_text()
    except OSError:
        return set()
    for line in text.splitlines():
        if line.startswith(("flags", "Features")):
            return set(line.split(":", 1)[1].split())
    return set()


def symjit_class(machine: str | None = None, flags: set[str] | None = None) -> str:
    """The code path SymJIT 2.27 compiles for on this host
    (``Config::has_avx``: x86-64 with AVX takes the AVX code, which also
    emits FMA3 instructions; without AVX the SSE code)."""

    machine = (machine or platform.machine()).lower()
    flags = _cpu_flags() if flags is None else flags
    if machine in ("x86_64", "amd64"):
        if "avx" not in flags:
            return "x86-64-sse"
        if "fma" not in flags:
            return "x86-64-avx-without-fma"
        return "x86-64-avx-fma"
    if machine in ("aarch64", "arm64"):
        return "aarch64"
    return machine


def host() -> dict[str, Any]:
    flags = _cpu_flags()
    interesting = ("sse2", "sse4_2", "avx", "avx2", "fma", "avx512f", "asimd")
    model = ""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    record: dict[str, Any] = {
        "machine": platform.machine(),
        "system": platform.system(),
        "cpu": model,
        "cpu_flags": sorted(f for f in interesting if f in flags),
        "symjit_class": symjit_class(flags=flags),
        "python": platform.python_version(),
    }
    try:
        from pyamplicol import __version__

        record["pyamplicol"] = __version__
    except Exception as error:
        record["pyamplicol"] = f"unavailable: {error}"
    try:
        from pyamplicol._internal.versions import active_native_build_inputs_sha256

        record["native_build_inputs_sha256"] = active_native_build_inputs_sha256()
    except Exception:
        pass
    return record


# ---------------------------------------------------------------- scan

# Any absolute POSIX path under a common root, or a Windows drive path.
ABSOLUTE = re.compile(
    rb"(?<![\w.])/(?:tmp|var|home|shared|nix|usr|opt|root|Users|private|mnt"
    rb"|scratch|srv|data|run|etc|proc|lib|lib64|gpfs|cvmfs|afs|work)/"
    rb"|\b[A-Za-z]:[\\/](?:Users|Windows|tmp)\b"
)
PORTABLE_TARGET = "portable-64le"
PORTABLE_SHEBANG = b"#!/usr/bin/env "
NATIVE_BINARY_MAGIC = (
    b"\x7fELF",
    b"\xcf\xfa\xed\xfe",
    b"\xce\xfa\xed\xfe",
    b"\xca\xfe\xba\xbe",
    b"MZ",
)
# PACBIN member kinds (``generation.evaluator_container.PacbinMemberKind``).
NATIVE_LIBRARY_KIND = 3
SYMJIT_APPLICATION_KIND = 1
# SymJIT ``Application`` storage magic (little endian), at the start of the
# serialized application inside a storage-v4 member.
SYMJIT_APPLICATION_MAGIC = (0x40568795410D08E9).to_bytes(8, "little")


def host_strings(root: Path, extra: tuple[str, ...] = ()) -> set[str]:
    """Strings of the generating host that must not appear in an artifact."""

    user = getpass.getuser()
    found = {
        str(Path.home()),
        str(Path.cwd()),
        tempfile.gettempdir(),
        str(root.resolve()),
        str(root.resolve().parent),
        f"/{user}/",
        f"{user}@",
        "pyamplicol-loop-v2-",  # the v2 generation's staging directories
    }
    name = socket.gethostname()
    if len(name) >= 4:
        found.add(name)
    found.update(extra)
    return {s for s in found if s and s != "/"}


def _members(path: Path) -> list[tuple[str, int, bytes]]:
    from pyamplicol.generation.evaluator_container import PacbinReader

    out = []
    with PacbinReader.open(path) as reader:
        for member in reader.index.members:
            data = bytes(reader.read_member(member.logical_path, length=member.length))
            out.append((member.logical_path, int(member.kind), data))
    return out


def _text(data: bytes) -> bytes:
    """A member's bytes, zlib streams decompressed."""

    if data[:1] == b"\x78":
        try:
            return zlib.decompress(data)
        except zlib.error:
            pass
    return data


def scan(root: Path, extra: tuple[str, ...] = ()) -> dict[str, Any]:
    root = Path(root)
    forbidden = sorted(host_strings(root, extra))
    findings: list[dict[str, str]] = []
    kinds: Counter[int] = Counter()
    symjit = {"members": 0, "storage_v4": 0}
    files = 0

    def check(where: str, data: bytes) -> None:
        for item in forbidden:
            if item.encode() in data:
                findings.append({"where": where, "kind": "host-string", "text": item})
        body = data
        if body.startswith(PORTABLE_SHEBANG):
            # "#!/usr/bin/env NAME" is the portable interpreter lookup.
            body = body.split(b"\n", 1)[-1]
        match = ABSOLUTE.search(body)
        if match is not None:
            data = body
            start = max(0, match.start() - 40)
            excerpt = data[start : match.end() + 40].decode("latin-1")
            findings.append({"where": where, "kind": "absolute-path", "text": excerpt})

    for path in sorted(p for p in root.rglob("*") if p.is_file() or p.is_symlink()):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            findings.append(
                {"where": relative, "kind": "symlink", "text": os.readlink(path)}
            )
            continue
        files += 1
        data = path.read_bytes()
        if data.startswith(NATIVE_BINARY_MAGIC):
            # Host machine code (ELF, Mach-O, PE): not CPU portable. Scripts
            # of the API bundle (executable, text) are fine.
            findings.append({"where": relative, "kind": "native-binary", "text": ""})
        check(relative, data)
        if path.suffix == ".pacbin":
            for name, kind, member in _members(path):
                kinds[kind] += 1
                where = f"{relative}!{name}"
                if kind == NATIVE_LIBRARY_KIND:
                    findings.append(
                        {"where": where, "kind": "native-library", "text": ""}
                    )
                if kind == SYMJIT_APPLICATION_KIND:
                    symjit["members"] += 1
                    if SYMJIT_APPLICATION_MAGIC in member[:64]:
                        symjit["storage_v4"] += 1
                check(where, _text(member))
    target: dict[str, Any] = {}
    manifest = root / "artifact.json"
    if manifest.is_file():
        producer = json.loads(manifest.read_text()).get("producer", {})
        target = dict(producer.get("target") or {})
        if target.get("triple") != PORTABLE_TARGET or target.get("cpu_features"):
            findings.append(
                {"where": "artifact.json", "kind": "target", "text": json.dumps(target)}
            )
    return {
        "schema": SCHEMA,
        "kind": "scan",
        "files": files,
        "member_kinds": {str(k): v for k, v in sorted(kinds.items())},
        "symjit_applications": symjit,
        "target": target,
        "forbidden_strings": len(forbidden),
        "findings": findings,
        "clean": not findings,
    }


# ---------------------------------------------------------------- points


def points(legs: int, count: int, seed: int = 20261024, energy: float = 1000.0):
    """Massless 2 -> legs-2 points (RAMBO-like flat sampling, binary64), with
    every pair invariant above 1% of s, and one mu_R^2 per point. A host that
    has the source checkout may pass exact points instead (any JSON of the
    same shape)."""

    import math
    import random

    rng = random.Random(seed)
    out = []
    n = legs - 2
    while len(out) < count:
        q = []
        for _ in range(n):
            c = 2 * rng.random() - 1
            phi = 2 * math.pi * rng.random()
            e = -math.log(rng.random() * rng.random())
            s = math.sqrt(1 - c * c)
            q.append([e, e * s * math.cos(phi), e * s * math.sin(phi), e * c])
        total = [sum(p[i] for p in q) for i in range(4)]
        mass = math.sqrt(total[0] ** 2 - sum(t * t for t in total[1:]))
        b = [-t / mass for t in total[1:]]
        gamma = total[0] / mass
        a = 1 / (1 + gamma)
        x = energy / mass
        final = []
        for p in q:
            bq = sum(bi * pi for bi, pi in zip(b, p[1:], strict=True))
            e = x * (gamma * p[0] + bq)
            final.append(
                [e]
                + [
                    x * (pi + bi * p[0] + a * bq * bi)
                    for pi, bi in zip(p[1:], b, strict=True)
                ]
            )
        half = energy / 2
        momenta = [[half, 0.0, 0.0, half], [half, 0.0, 0.0, -half], *final]
        s = energy * energy

        def dot(u, v):
            return u[0] * v[0] - u[1] * v[1] - u[2] * v[2] - u[3] * v[3]

        if all(
            2 * abs(dot(momenta[i], momenta[j])) > 0.01 * s
            for i in range(legs)
            for j in range(i + 1, legs)
        ):
            out.append(momenta)
    return {
        "schema": SCHEMA,
        "kind": "points",
        "legs": legs,
        "points": [[[c.hex() for c in p] for p in point] for point in out],
        "mu_r2": [(energy * energy).hex() for _ in out],
    }


def _read_points(path: Path) -> tuple[list[list[list[float]]], list[float]]:
    data = json.loads(Path(path).read_text())
    momenta = [
        [[float.fromhex(c) for c in p] for p in point] for point in data["points"]
    ]
    scales = [float.fromhex(s) for s in data["mu_r2"]]
    return momenta, scales


# ---------------------------------------------------------------- evaluate


def _bits(value: Any) -> Any:
    """Every binary64 value as a hexadecimal literal (exact), recursively."""

    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value.hex()
    if isinstance(value, complex):
        return [value.real.hex(), value.imag.hex()]
    if hasattr(value, "finite") and hasattr(value, "double_pole"):
        return [_bits(value.finite), _bits(value.single_pole), _bits(value.double_pole)]
    if hasattr(value, "tolist"):
        return _bits(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_bits(v) for v in value]
    return str(value)


def evaluate(
    artifact: Path,
    process: str,
    points_file: Path,
    *,
    parameters: dict[str, str] | None = None,
    poles: str | None = None,
    arithmetic: str | None = None,
    precision: int | None = None,
) -> dict[str, Any]:
    """Load and evaluate through the public Python API under the sentinel.
    Import pyAmpliCol only inside (the sentinel sees module-level work).
    ``arithmetic``/``precision`` select a loop process's carrier (binary64
    by default; ``double-double`` or ``arbitrary``)."""

    from decimal import Decimal

    momenta, scales = _read_points(points_file)
    with GenerationSentinel() as sentinel:
        start = time.perf_counter()
        from pyamplicol import Runtime

        imported = time.perf_counter()
        options = {"model_parameters": parameters} if parameters else {}
        runtime = Runtime.load(artifact, process=process, **options)
        loaded = time.perf_counter()
        info = runtime.inspect() if hasattr(runtime, "inspect") else {}
        # Virtual (plan v4) processes report contribution_kind.
        loop = isinstance(info, dict) and info.get("contribution_kind") == "virtual"
        if loop:
            exact = [[[str(Decimal(c)) for c in p] for p in point] for point in momenta]
            mu = [str(Decimal(s)) for s in scales]
            kwargs: dict[str, Any] = {"poles": poles} if poles else {}
            if arithmetic:
                kwargs["arithmetic"] = arithmetic
            if precision:
                kwargs["precision"] = precision
            value = runtime.evaluate(exact, mu_r2=mu, **kwargs)
            results = {"born": _bits(value.born_values), "virtual": _bits(value.values)}
        else:
            value = runtime.evaluate(momenta)
            results = {"value": _bits(value)}
        done = time.perf_counter()
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "kind": "evaluate",
        "path": "python",
        "process": process,
        "points": len(momenta),
        "arithmetic": arithmetic or "double",
        "precision": precision,
        "results": results,
        "generation_calls": dict(sorted(sentinel.calls.items())),
        "seconds": {
            "import": imported - start,
            "load": loaded - imported,
            "evaluate": done - loaded,
        },
        "host": host(),
    }
    if loop:
        record["point_inputs"] = info.get("point_inputs")
        record["plan_sha256"] = info.get("plan_sha256")
    record["artifact_id"] = getattr(runtime, "artifact_id", None)
    return record


# ---------------------------------------------------------------- SDK

# C client of the loop SDK ABI (rusticol_virtual.h): reads "points legs",
# then per point 4*legs momentum components and mu_R^2 as C99 hexadecimal
# literals; prints per point its status, the Born per Born order and the
# virtual per sector and epsilon power (finite, 1/eps, 1/eps^2) as "%a".
CLIENT_SOURCE = r"""
/* SPDX-License-Identifier: 0BSD */
#include "rusticol_virtual.h"
#include <stdio.h>
#include <stdlib.h>

static void check(int status) {
  if (status != RUSTICOL_VIRTUAL_OK) {
    fprintf(stderr, "%s\n", rusticol_virtual_last_error_message());
    exit(1);
  }
}
static double next_value(FILE *input) {
  char text[128];
  if (fscanf(input, "%127s", text) != 1) {
    fprintf(stderr, "short input\n");
    exit(2);
  }
  return strtod(text, NULL);
}
int main(int argc, char **argv) {
  if (argc != 4 && argc != 5) {
    fprintf(stderr, "usage: %s artifact process points [poles]\n", argv[0]);
    return 2;
  }
  FILE *input = fopen(argv[3], "r");
  if (!input) { perror(argv[3]); return 2; }
  size_t points = 0, legs = 0;
  if (fscanf(input, "%zu %zu", &points, &legs) != 2 || !points || !legs) return 2;
  double *momenta = malloc(points * legs * 4 * sizeof(double));
  double *scales = malloc(points * sizeof(double));
  if (!momenta || !scales) return 2;
  for (size_t p = 0; p < points; p++) {
    for (size_t i = 0; i < legs * 4; i++) momenta[p * legs * 4 + i] = next_value(input);
    scales[p] = next_value(input);
  }
  fclose(input);
  rusticol_virtual_handle *runtime = NULL;
  check(rusticol_virtual_open(argv[1], argv[2], NULL, "model", &runtime));
  if (argc == 5) check(rusticol_virtual_set_poles(runtime, argv[4]));
  rusticol_virtual_shape_info shape;
  check(rusticol_virtual_shape(runtime, &shape));
  if (shape.external_count != legs || shape.epsilon_count != 3) {
    fprintf(stderr, "unexpected shape\n");
    return 1;
  }
  rusticol_virtual_batch_request request = {
      momenta, points * legs * 4, points, NULL, scales, points, 0, NULL, NULL};
  rusticol_virtual_result *result = NULL;
  check(rusticol_virtual_evaluate(runtime, &request, &result));
  rusticol_virtual_result_shape_info info;
  check(rusticol_virtual_result_shape(result, &info));
  uint32_t *statuses = malloc(points * sizeof(uint32_t));
  double *born = malloc((info.born_values + 1) * sizeof(double));
  double *values = malloc((info.virtual_values + 1) * sizeof(double));
  if (!statuses || !born || !values) return 2;
  check(rusticol_virtual_result_copy_statuses(result, statuses, points));
  check(rusticol_virtual_result_copy_born(result, born, info.born_values));
  check(rusticol_virtual_result_copy_virtual(result, values, info.virtual_values));
  size_t row = 0;
  for (size_t p = 0; p < points; p++) {
    printf("point %zu status %u\n", p, statuses[p]);
    if (statuses[p] == RUSTICOL_VIRTUAL_FAILED) continue;
    printf("born");
    for (size_t o = 0; o < shape.born_orders; o++)
      printf(" %a", born[row * shape.born_orders + o]);
    printf("\nvirtual");
    for (size_t s = 0; s < shape.virtual_orders * 3; s++)
      printf(" %a", values[row * shape.virtual_orders * 3 + s]);
    printf("\n");
    row++;
  }
  check(rusticol_virtual_result_free(result));
  check(rusticol_virtual_free(runtime));
  free(momenta); free(scales); free(statuses); free(born); free(values);
  return 0;
}
"""

# Symbolica's restricted-mode messages (stdout): the SDK touched Symbolica's
# licence check, so an unlicensed second instance aborts.
RESTRICTED_MARKERS = ("restricted Symbolica", "Symbolica thread")

# A plain Symbolica process (no pyAmpliCol, hence no package key) that makes
# one symbolic call, which takes this user's restricted thread permit, and
# keeps it until its stdin closes.
HOLDER_SOURCE = (
    "import sys, symbolica\n"
    "symbolica.S('pyamplicol_portability_permit')\n"
    "print('held', flush=True)\n"
    "sys.stdin.read()\n"
)
CONTROL_SOURCE = "import symbolica\nsymbolica.S('pyamplicol_portability_control')\n"


def _without_licence(env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if env is None else env)
    env.pop("SYMBOLICA_LICENSE", None)
    return env


def symbolica_lock_dir(env: dict[str, str] | None = None) -> str:
    """The directory of Symbolica's restricted-thread permit in ``env``, by
    Symbolica's own rule (``license.rs``, ``lock_directory``):
    ``SYMBOLICA_LOCK_DIR``, else ``$XDG_RUNTIME_DIR/symbolica/locks``, else
    the cache directory's ``symbolica/locks``. ``RestrictedPermit``, its
    control and every instance it covers get it as ``SYMBOLICA_LOCK_DIR``,
    so a fresh ``HOME`` or ``XDG_CACHE_HOME`` (``_fresh``) or a missing
    ``XDG_RUNTIME_DIR`` (cron, batch jobs, containers) cannot move an
    instance to another permit."""

    env = dict(os.environ if env is None else env)
    explicit = env.get("SYMBOLICA_LOCK_DIR")
    if explicit:
        return explicit
    runtime = env.get("XDG_RUNTIME_DIR")
    if runtime and os.path.isabs(runtime):
        return os.path.join(runtime, "symbolica", "locks")
    cache = env.get("XDG_CACHE_HOME")
    if not (cache and os.path.isabs(cache)):
        cache = os.path.join(env.get("HOME") or os.path.expanduser("~"), ".cache")
    return os.path.join(cache, "symbolica", "locks")


class RestrictedPermit:
    """Holds this user's restricted Symbolica thread permit inside the
    ``with`` block (``HOLDER_SOURCE``, without ``SYMBOLICA_LICENSE``).

    Without a licence, Symbolica admits one restricted thread per user (a lock
    file in its per-user lock directory); any other unlicensed Symbolica state
    access of the user aborts at once while the permit is held. A process that
    completes inside the block therefore touched no Symbolica state, which is
    deterministic, unlike two racing instances. ``record["verified"]`` is the
    positive control: a second plain Symbolica process aborted with the
    restricted-mode message (false with a valid licence found elsewhere, for
    example a registered key, in which case the check does not discriminate).

    The permit lives in Symbolica's lock directory (``lock_dir``, by default
    ``symbolica_lock_dir()`` of this process's environment). The default is
    the user's real, machine-wide permit: keep the block short, since any
    other unlicensed Symbolica process of the same user aborts meanwhile. A
    private directory (``SYMBOLICA_LOCK_DIR`` or ``lock_dir``, as the tests
    use) discriminates as well and leaves the user's permit alone.
    """

    def __init__(self, python: str = sys.executable, lock_dir: str | None = None):
        self.python = python
        self.process: subprocess.Popen[str] | None = None
        self.record: dict[str, Any] = {}
        # The permit's directory, explicit for the holder, the control and
        # every covered instance (``environment``).
        self.lock_dir = lock_dir or symbolica_lock_dir()

    def environment(self, env: dict[str, str]) -> dict[str, str]:
        """``env`` bound to this permit's lock directory."""

        return {**env, "SYMBOLICA_LOCK_DIR": self.lock_dir}

    def __enter__(self) -> RestrictedPermit:
        env = self.environment(_without_licence())
        self.process = subprocess.Popen(
            [self.python, "-c", HOLDER_SOURCE],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            preexec_fn=_no_core,
        )
        assert self.process.stdout is not None
        seen = []
        for line in self.process.stdout:
            seen.append(line)
            if line.strip() == "held":
                break
        else:
            self.process.wait()
            raise RuntimeError(
                "the permit holder did not start: " + "".join(seen)[-2000:]
            )
        control = subprocess.run(
            [self.python, "-c", CONTROL_SOURCE],
            capture_output=True,
            text=True,
            env=env,
            preexec_fn=_no_core,
        )
        restricted = any(
            m in control.stdout + control.stderr for m in RESTRICTED_MARKERS
        )
        self.record = {
            "control_returncode": control.returncode,
            "verified": control.returncode != 0 and restricted,
            "lock_dir_explicit": True,
        }
        return self

    def __exit__(self, *exc: object) -> None:
        assert self.process is not None
        try:
            self.process.communicate(input="", timeout=60)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.communicate()
        self.record["holder_returncode"] = self.process.returncode


def sdk_build(
    out: Path,
    *,
    include: Path | None = None,
    lib_dir: Path | None = None,
    rusticol_config: str | None = None,
    compiler: str = "cc",
) -> Path:
    """Compile ``CLIENT_SOURCE``: against a wheel (``rusticol-config``) or a
    header directory and a library directory with ``librusticol_loop_sdk``."""

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    source = out.with_suffix(".c")
    source.write_text(CLIENT_SOURCE)
    if rusticol_config:
        cflags = subprocess.run(
            [rusticol_config, "--cflags"], capture_output=True, text=True, check=True
        ).stdout.split()
        libs = subprocess.run(
            [rusticol_config, "--libs"], capture_output=True, text=True, check=True
        ).stdout.split()
    else:
        if include is None or lib_dir is None:
            raise SystemExit(
                "sdk-build needs --rusticol-config or --include and --lib-dir"
            )
        cflags = [f"-I{include}"]
        libs = [f"-L{lib_dir}", "-lrusticol_loop_sdk", f"-Wl,-rpath,{lib_dir}"]
    subprocess.run(
        [compiler, "-std=c11", "-O2", *cflags, str(source), *libs, "-o", str(out)],
        check=True,
    )
    return out


def _client_input(points_file: Path, destination: Path) -> int:
    momenta, scales = _read_points(points_file)
    legs = len(momenta[0])
    lines = [f"{len(momenta)} {legs}"]
    for point, scale in zip(momenta, scales, strict=True):
        flat = [c for p in point for c in p]
        lines.append(" ".join(c.hex() for c in flat) + " " + scale.hex())
    destination.write_text("\n".join(lines) + "\n")
    return legs


def _hex(text: str) -> str:
    return float.fromhex(text).hex()


def sdk(
    artifact: Path,
    process: str,
    points_file: Path,
    client: Path,
    *,
    licence: bool = True,
    instances: int = 1,
    workdir: Path | None = None,
    hold_permit: bool = False,
    poles: str | None = None,
) -> dict[str, Any]:
    """Evaluate through the C client; ``instances`` concurrent copies (the
    record holds the first's values and every instance's exit status), with
    ``hold_permit`` while ``RestrictedPermit`` holds the user's permit;
    ``poles`` is the pole selection (``rusticol_virtual_set_poles``; absent
    poles are NaN)."""

    workdir = Path(workdir or tempfile.mkdtemp(prefix="portability-sdk-"))
    workdir.mkdir(parents=True, exist_ok=True)
    feed = workdir / "points.txt"
    _client_input(points_file, feed)
    record = _parse_client("")
    outcome = _run_instances(
        [str(client), str(artifact), process, str(feed), *([poles] if poles else [])],
        workdir,
        licence=licence,
        instances=instances,
        hold_permit=hold_permit,
    )
    if outcome["exit_status"][0] == 0:
        record = _parse_client(outcome.pop("stdout"))
    outcome.pop("stdout", None)
    record.update(process=process, **outcome)
    return record


def _run_instances(
    command: list[str],
    cwd: Path,
    *,
    licence: bool,
    instances: int,
    hold_permit: bool,
) -> dict[str, Any]:
    """``instances`` concurrent runs of ``command``; the first's stdout, every
    exit status, the restricted-mode banner and the permit record."""

    env = dict(os.environ) if licence else _without_licence()
    permit = RestrictedPermit() if hold_permit else None
    if permit is not None:
        env = permit.environment(env)
        permit.__enter__()
    try:
        start = time.perf_counter()
        runs = [
            subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                cwd=cwd,
                preexec_fn=_no_core,
            )
            for _ in range(instances)
        ]
        outputs = [run.communicate() for run in runs]
        seconds = time.perf_counter() - start
    finally:
        if permit is not None:
            permit.__exit__()
    return {
        "stdout": outputs[0][0],
        "exit_status": [run.returncode for run in runs],
        "licence": licence,
        "symbolica_restricted": any(
            m in out for out, _ in outputs for m in RESTRICTED_MARKERS
        ),
        "permit": None if permit is None else permit.record,
        "stderr": [err[-2000:] for _, err in outputs],
        "seconds": {"run": seconds},
    }


def _parse_client(stdout: str) -> dict[str, Any]:
    """The C client's output as an ``evaluate`` record."""

    born: list[Any] = []
    virtual: list[Any] = []
    point_status: list[int] = []
    for line in stdout.splitlines():
        if line.startswith("point "):
            point_status.append(int(line.split()[3]))
        elif line.startswith("born"):
            born.append([_hex(x) for x in line.split()[1:]])
        elif line.startswith("virtual"):
            flat = [_hex(x) for x in line.split()[1:]]
            virtual.append([flat[k : k + 3] for k in range(0, len(flat), 3)])
    return {
        "schema": SCHEMA,
        "kind": "evaluate",
        "path": "sdk",
        "points": len(born),
        "results": {"born": born, "virtual": virtual},
        "point_status": point_status,
        "host": host(),
    }


# ---------------------------------------------------------------- tree C API

CHECKOUT = Path(__file__).resolve().parents[1]
# ``cargo rustc -p rusticol-capi --release --crate-type staticlib -- --print
# native-static-libs`` on x86_64 Linux (glibc).
CAPI_NATIVE_LIBS = "-ldl -lgcc_s -lutil -lrt -lpthread -lm -ldl -lc"


def capi_build(
    artifact: Path,
    out: Path,
    *,
    archive: Path | None = None,
    include: Path | None = None,
    libs: str = CAPI_NATIVE_LIBS,
    rusticol_config: str | None = None,
    compiler: str = "cc",
) -> Path:
    """Compile the artifact's own C example (``API/c/check_standalone.c`` of
    its API bundle, the tree engine's C distribution path) against a wheel
    (``rusticol-config``) or a ``rusticol-capi`` static archive, its header
    directory (default: this checkout's) and its native libraries."""

    source = Path(artifact) / "API" / "c" / "check_standalone.c"
    if not source.is_file():
        raise SystemExit(f"{source} missing: generate the artifact with its API bundle")
    if rusticol_config:
        cflags = subprocess.run(
            [rusticol_config, "--cflags"], capture_output=True, text=True, check=True
        ).stdout.split()
        link = subprocess.run(
            [rusticol_config, "--libs"], capture_output=True, text=True, check=True
        ).stdout.split()
    else:
        if archive is None:
            raise SystemExit("capi-build needs --rusticol-config or --archive")
        header = include or CHECKOUT / "rust" / "crates" / "rusticol-capi" / "include"
        cflags = [f"-I{header}"]
        link = [str(archive), *libs.split()]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [compiler, "-std=c11", "-O2", *cflags, str(source), "-o", str(out), *link],
        check=True,
    )
    return out


def _api_results(stdout: str) -> dict[str, Any]:
    """The ``--json`` output of an API-bundle example: every resolved value
    and both totals as hexadecimal literals (the C example prints ``%.17g``,
    which round-trips binary64 exactly)."""

    data = json.loads(stdout)
    return {
        "shape": data["shape"],
        "values": _bits([float(v) for v in data["values"]]),
        "resolved_sum": _bits([float(v) for v in data["resolved_sum"]]),
        "compatibility_total": _bits([float(v) for v in data["compatibility_total"]]),
    }


def capi(
    artifact: Path,
    client: Path,
    *,
    licence: bool = True,
    instances: int = 1,
    hold_permit: bool = False,
) -> dict[str, Any]:
    """Run the compiled example from the artifact directory (it reads
    ``artifact.json`` and the stored validation momenta) with ``--json``;
    ``instances`` concurrent copies, optionally while the permit is held."""

    outcome = _run_instances(
        [str(Path(client).resolve()), "--json"],
        Path(artifact),
        licence=licence,
        instances=instances,
        hold_permit=hold_permit,
    )
    stdout = outcome.pop("stdout")
    results = _api_results(stdout) if outcome["exit_status"][0] == 0 else {}
    return {
        "schema": SCHEMA,
        "kind": "evaluate",
        "path": "capi",
        "results": results,
        **outcome,
        "host": host(),
    }


def api_python(artifact: Path, env: dict[str, str]) -> dict[str, Any]:
    """The API bundle's Python example (``API/python/check_standalone.py
    --json``, the public ``Runtime``) in a fresh interpreter: the reference of
    the C example at the same stored validation momenta."""

    script = Path(artifact) / "API" / "python" / "check_standalone.py"
    run = subprocess.run(
        [sys.executable, str(script), "--json"],
        capture_output=True,
        text=True,
        env=env,
        cwd=artifact,
    )
    if run.returncode != 0:
        raise RuntimeError(f"{script} failed ({run.returncode}): {run.stderr[-4000:]}")
    return {
        "schema": SCHEMA,
        "kind": "evaluate",
        "path": "python-api-bundle",
        "results": _api_results(run.stdout),
        "host": host(),
    }


def tree_capi_check(
    artifact: Path,
    client: Path,
    scratch: Path,
    *,
    licence: bool = False,
    instances: int = 2,
    hold_permit: bool = True,
) -> dict[str, Any]:
    """The tree C API at the original and at a relocated copy: bitwise equal
    to each other and to the Python API, and (default) without
    ``SYMBOLICA_LICENSE`` in ``instances`` concurrent processes while
    ``RestrictedPermit`` holds the user's permit."""

    artifact = Path(artifact).resolve()
    scratch = Path(scratch).resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    verdict: dict[str, Any] = {"schema": SCHEMA, "kind": "tree-capi"}
    verdict["scan"] = scan(artifact)
    moved = scratch / "relocated" / f"copy-{os.getpid()}" / artifact.name
    if moved.exists():
        raise SystemExit(f"{moved} exists")
    shutil.copytree(artifact, moved, symlinks=True)
    env, _ = _fresh(scratch, "python", licence)
    python = api_python(artifact, env)
    runs = {
        "original": capi(artifact, client, licence=licence),
        "relocated": capi(
            moved,
            client,
            licence=licence,
            instances=instances,
            hold_permit=hold_permit,
        ),
    }
    for label, record in runs.items():
        (scratch / f"capi-{label}.json").write_text(json.dumps(record))
    verdict["capi"] = {
        label: {
            k: r[k]
            for k in ("exit_status", "symbolica_restricted", "permit", "seconds")
        }
        | {"stderr": [e for e in r["stderr"] if e]}
        for label, r in runs.items()
    }
    ran = all(r["exit_status"] == [0] * len(r["exit_status"]) for r in runs.values())
    verdict["capi_ran"] = ran
    if ran:
        verdict["capi_relocated_bitwise"] = compare(runs["original"], runs["relocated"])
        verdict["capi_equals_python"] = compare(python, runs["original"])
    permit = runs["relocated"]["permit"]
    verdict["capi_without_licence"] = (
        None
        if licence
        else ran
        and not any(r["symbolica_restricted"] for r in runs.values())
        and (permit is None or permit["verified"])
    )
    verdict["host"] = host()
    return verdict


# ---------------------------------------------------------------- compare


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key in sorted(value):
            out.update(_flatten(value[key], f"{prefix}.{key}"))
        return out
    if isinstance(value, list):
        out = {}
        for k, item in enumerate(value):
            out.update(_flatten(item, f"{prefix}[{k}]"))
        return out
    return {prefix: value}


def compare(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Bitwise equality of the binary64 results; mismatches with their
    relative difference."""

    left, right = _flatten(a["results"]), _flatten(b["results"])
    mismatches = []
    for key in sorted(set(left) | set(right)):
        x, y = left.get(key), right.get(key)
        if x == y:
            continue
        entry: dict[str, Any] = {"at": key, "a": x, "b": y}
        try:
            fx, fy = float.fromhex(x), float.fromhex(y)
            entry["relative"] = abs(fx - fy) / max(abs(fx), abs(fy), 1e-300)
        except (TypeError, ValueError):
            pass
        mismatches.append(entry)
    classes = (
        a.get("host", {}).get("symjit_class"),
        b.get("host", {}).get("symjit_class"),
    )
    return {
        "schema": SCHEMA,
        "kind": "compare",
        "values": len(left),
        "bitwise": not mismatches and len(left) > 0,
        "mismatches": mismatches[:20],
        "mismatch_count": len(mismatches),
        "max_relative": max((m.get("relative", 0.0) for m in mismatches), default=0.0),
        "symjit_classes": list(classes),
    }


# ---------------------------------------------------------------- check


def _fresh(
    scratch: Path, label: str, licence: bool, permit: RestrictedPermit | None = None
) -> tuple[dict[str, str], Path]:
    """Another working, home and temporary directory and user name (bound to
    ``permit``'s lock directory, if any)."""

    base = scratch / f"env-{label}"
    for name in ("cwd", "home", "tmp"):
        (base / name).mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(
        HOME=str(base / "home"),
        TMPDIR=str(base / "tmp"),
        USER=f"portability-{label}",
        LOGNAME=f"portability-{label}",
        XDG_CACHE_HOME=str(base / "home" / ".cache"),
    )
    if not licence:
        env.pop("SYMBOLICA_LICENSE", None)
    if permit is not None:
        env = permit.environment(env)
    return env, base / "cwd"


def _run_evaluate(
    artifact: Path, process: str, points_file: Path, out: Path, env, cwd, extra
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "evaluate",
        str(artifact),
        "--process",
        process,
        "--points",
        str(points_file),
        "--out",
        str(out),
        *extra,
    ]
    run = subprocess.run(command, capture_output=True, text=True, env=env, cwd=cwd)
    if run.returncode not in (0, 3):
        raise RuntimeError(f"evaluate failed ({run.returncode}): {run.stderr[-4000:]}")
    return json.loads(out.read_text())


def _run_evaluate_concurrently(
    artifact: Path,
    process: str,
    points_file: Path,
    scratch: Path,
    licence: bool,
    instances: int,
    extra: list[str],
    hold_permit: bool,
) -> dict[str, Any]:
    """``instances`` concurrent ``evaluate`` processes, each in its own fresh
    environment, optionally while ``RestrictedPermit`` holds the user's
    permit (then every one completing touched no Symbolica state that needs
    a licence or the permit)."""

    def run(permit: RestrictedPermit | None) -> dict[str, Any]:
        started = time.perf_counter()
        processes = []
        for k in range(instances):
            env, cwd = _fresh(scratch, f"concurrent-{k}", licence, permit)
            out = scratch / f"concurrent-{k}.json"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "evaluate",
                str(artifact),
                "--process",
                process,
                "--points",
                str(points_file),
                "--out",
                str(out),
                *extra,
            ]
            processes.append(
                subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env,
                    cwd=cwd,
                    preexec_fn=_no_core,
                )
            )
        outputs = [p.communicate() for p in processes]
        return {
            "exit_status": [p.returncode for p in processes],
            "symbolica_restricted": any(
                m in out + err for out, err in outputs for m in RESTRICTED_MARKERS
            ),
            "stderr": [err[-2000:] for _out, err in outputs],
            "seconds": time.perf_counter() - started,
        }

    if hold_permit:
        with RestrictedPermit() as permit:
            record = run(permit)
        record["permit"] = permit.record
    else:
        record = run(None)
        record["permit"] = None
    return record


def check(
    artifact: Path,
    process: str,
    scratch: Path,
    *,
    points_file: Path | None = None,
    legs: int | None = None,
    count: int = 2,
    client: Path | None = None,
    licence: bool = True,
    instances: int = 2,
    parameters: dict[str, str] | None = None,
    generation_seconds: float | None = None,
    hold_permit: bool = False,
    arithmetic: str | None = None,
    precision: int | None = None,
) -> dict[str, Any]:
    """Scan, relocate, evaluate both copies in fresh environments, compare;
    with ``hold_permit`` also ``instances`` concurrent Python evaluations of
    the relocated copy while ``RestrictedPermit`` holds the user's permit;
    the SDK too with ``client`` (its concurrent run, with ``hold_permit``,
    while the permit is held). ``arithmetic``/``precision``: the loop
    carrier of the Python evaluations."""

    artifact = Path(artifact).resolve()
    scratch = Path(scratch).resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    verdict: dict[str, Any] = {"schema": SCHEMA, "kind": "check", "process": process}
    verdict["scan"] = scan(artifact)
    if points_file is None:
        if legs is None:
            manifest = json.loads((artifact / "artifact.json").read_text())
            entry = {p["id"]: p for p in manifest["processes"]}[process]
            legs = len(entry["external_pdgs"])
        points_file = scratch / "points.json"
        points_file.write_text(json.dumps(points(legs, count)))
    moved = scratch / "relocated" / f"copy-{os.getpid()}" / artifact.name
    if moved.exists():
        raise SystemExit(f"{moved} exists")
    shutil.copytree(artifact, moved, symlinks=True)
    extra = []
    for name, value in (parameters or {}).items():
        extra += ["--parameter", f"{name}={value}"]
    if arithmetic:
        extra += ["--arithmetic", arithmetic]
    if precision:
        extra += ["--precision", str(precision)]
    records = {}
    for label, root in (("original", artifact), ("relocated", moved)):
        env, cwd = _fresh(scratch, label, licence)
        records[label] = _run_evaluate(
            root, process, points_file, scratch / f"{label}.json", env, cwd, extra
        )
    verdict["python"] = {
        label: {
            "seconds": r["seconds"],
            "generation_calls": sum(r["generation_calls"].values()),
            "generation_top": dict(Counter(r["generation_calls"]).most_common(8)),
            "point_inputs": r.get("point_inputs"),
        }
        for label, r in records.items()
    }
    verdict["relocated_bitwise"] = compare(records["original"], records["relocated"])
    verdict["no_generation"] = not any(r["generation_calls"] for r in records.values())
    if hold_permit:
        concurrent = _run_evaluate_concurrently(
            moved, process, points_file, scratch, licence, instances, extra, True
        )
        verdict["python_concurrent"] = concurrent
        verdict["python_without_licence"] = (
            None
            if licence
            else concurrent["exit_status"] == [0] * instances
            and not concurrent["symbolica_restricted"]
            and bool(concurrent["permit"]["verified"])
        )
    if generation_seconds is not None:
        verdict["generation_seconds"] = generation_seconds
        verdict["load_over_generation"] = (
            records["relocated"]["seconds"]["load"] / generation_seconds
        )
    if client is not None:
        # One instance per location for the values, then ``instances``
        # concurrent ones at the relocated copy: a second unlicensed instance
        # aborts if the SDK takes Symbolica's restricted (one thread per
        # user) permit; with ``hold_permit`` even the first one does.
        runs = {}
        for label, root, count_, held in (
            ("original", artifact, 1, False),
            ("relocated", moved, 1, False),
            ("concurrent", moved, instances, hold_permit),
        ):
            runs[label] = sdk(
                root,
                process,
                points_file,
                client,
                licence=licence,
                instances=count_,
                workdir=scratch / f"sdk-{label}",
                hold_permit=held,
            )
            (scratch / f"sdk-{label}.json").write_text(json.dumps(runs[label]))
        verdict["sdk"] = {
            label: {
                "exit_status": r["exit_status"],
                "symbolica_restricted": r["symbolica_restricted"],
                "permit": r["permit"],
                "seconds": r["seconds"],
                "stderr": [e for e in r["stderr"] if e],
            }
            for label, r in runs.items()
        }
        ran = all(
            runs[label]["exit_status"] == [0] for label in ("original", "relocated")
        )
        verdict["sdk_ran"] = ran
        if ran:
            verdict["sdk_relocated_bitwise"] = compare(
                runs["original"], runs["relocated"]
            )
            verdict["sdk_equals_python"] = compare(
                records["original"], runs["original"]
            )
        concurrent = runs["concurrent"]["exit_status"] == [0] * instances
        verdict["sdk_concurrent"] = concurrent
        verdict["sdk_without_licence"] = (
            None
            if licence
            else ran
            and concurrent
            and not any(r["symbolica_restricted"] for r in runs.values())
        )
    verdict["host"] = host()
    return verdict


# ---------------------------------------------------------------- CPU models

# QEMU user-mode CPU models of the x86-64 feature levels SymJIT distinguishes:
# AVX with FMA3 (the build host's class here), AVX without FMA3, SSE4.2 only.
CPU_MODELS = ("Haswell-v4", "SandyBridge-v2", "Nehalem-v2")


def _no_core() -> None:
    import resource

    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def cpu_matrix(
    artifact: Path,
    process: str,
    points_file: Path,
    qemu: str,
    scratch: Path,
    *,
    models: tuple[str, ...] = CPU_MODELS,
    client: Path | None = None,
) -> dict[str, Any]:
    """``evaluate`` (and, with ``client``, the SDK) natively and under each
    QEMU user-mode CPU model, each compared bitwise with the native run. A
    model the artifact cannot run on must fail cleanly (an error naming the
    reason), never with SIGILL."""

    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    tool = str(Path(__file__).resolve())

    # QEMU itself runs without the caller's LD_LIBRARY_PATH (its own host
    # libraries may need another C library than the paths there provide);
    # the emulated program gets the caller's value (``-E``).
    host_env = dict(os.environ)
    guest_library_path = host_env.pop("LD_LIBRARY_PATH", None)
    guest = (
        []
        if guest_library_path is None
        else ["-E", f"LD_LIBRARY_PATH={guest_library_path}"]
    )

    def run(prefix: list[str], label: str, sdk_run: bool) -> dict[str, Any]:
        out = scratch / f"{'sdk' if sdk_run else 'python'}-{label}.json"
        if sdk_run:
            feed = scratch / "points.txt"
            _client_input(points_file, feed)
            command = [*prefix, str(client), str(artifact), process, str(feed)]
        else:
            command = [
                *prefix,
                sys.executable,
                tool,
                "evaluate",
                str(artifact),
                "--process",
                process,
                "--points",
                str(points_file),
                "--out",
                str(out),
            ]
        start = time.perf_counter()
        done = subprocess.run(
            command,
            capture_output=True,
            text=True,
            preexec_fn=_no_core,
            env=host_env if prefix else None,
        )
        entry: dict[str, Any] = {
            "returncode": done.returncode,
            "seconds": time.perf_counter() - start,
            "sigill": done.returncode in (-4, 132),
        }
        if done.returncode != 0:
            entry["stderr"] = done.stderr[-1500:]
            return entry
        entry["record"] = (
            _parse_client(done.stdout) if sdk_run else json.loads(out.read_text())
        )
        return entry

    verdict: dict[str, Any] = {"schema": SCHEMA, "kind": "cpu-matrix", "models": {}}
    for sdk_run in (False, True) if client is not None else (False,):
        path = "sdk" if sdk_run else "python"
        native = run([], "native", sdk_run)
        verdict["models"].setdefault("native", {})[path] = {
            k: v for k, v in native.items() if k != "record"
        }
        for model in models:
            entry = run([qemu, *guest, "-cpu", model], model, sdk_run)
            if "record" in native and "record" in entry:
                entry["compare"] = compare(native["record"], entry["record"])
            entry.pop("record", None)
            verdict["models"].setdefault(model, {})[path] = entry
    return verdict


# ---------------------------------------------------------------- CLI


def _parameters(items: list[str] | None) -> dict[str, str]:
    out = {}
    for item in items or ():
        name, _, value = item.partition("=")
        out[name] = value
    return out


def _write(record: dict[str, Any], out: str | None) -> None:
    text = json.dumps(record, indent=1, sort_keys=True)
    if out:
        Path(out).write_text(text + "\n")
    else:
        print(text)


HOLD_HELP = (
    "run the concurrent instances while a plain Symbolica process holds this "
    "user's restricted thread permit (any other unlicensed Symbolica process "
    "of the user aborts meanwhile)"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("host", help="host CPU class and versions")
    p = sub.add_parser("scan", help="portability scan of an artifact")
    p.add_argument("artifact", type=Path)
    p.add_argument("--forbid", action="append", default=[], help="extra forbidden text")
    p.add_argument("--out")
    p = sub.add_parser("points", help="binary64 massless phase-space points")
    p.add_argument("--legs", type=int, required=True)
    p.add_argument("--count", type=int, default=2)
    p.add_argument("--seed", type=int, default=20261024)
    p.add_argument("--out")
    p = sub.add_parser("evaluate", help="evaluate through the Python API")
    p.add_argument("artifact", type=Path)
    p.add_argument("--process", required=True)
    p.add_argument("--points", type=Path, required=True)
    p.add_argument("--parameter", action="append", help="NAME=VALUE")
    p.add_argument("--poles")
    p.add_argument("--arithmetic", help="double, double-double or arbitrary")
    p.add_argument("--precision", type=int, help="decimal digits")
    p.add_argument("--out")
    p.add_argument(
        "--allow-generation",
        action="store_true",
        help="exit 0 even if generation code ran (default: exit 3)",
    )
    p = sub.add_parser("sdk-build", help="compile the C SDK client")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rusticol-config")
    p.add_argument("--include", type=Path)
    p.add_argument("--lib-dir", type=Path)
    p.add_argument("--compiler", default=os.environ.get("CC", "cc"))
    p = sub.add_parser("sdk", help="evaluate through the C SDK client")
    p.add_argument("artifact", type=Path)
    p.add_argument("--process", required=True)
    p.add_argument("--points", type=Path, required=True)
    p.add_argument("--client", type=Path, required=True)
    p.add_argument("--no-licence", action="store_true")
    p.add_argument("--instances", type=int, default=1)
    p.add_argument("--hold-permit", action="store_true", help=HOLD_HELP)
    p.add_argument("--poles", choices=("all", "none", "uv", "ir"))
    p.add_argument("--out")
    p = sub.add_parser("compare", help="bitwise comparison of two records")
    p.add_argument("a", type=Path)
    p.add_argument("b", type=Path)
    p.add_argument("--out")
    p = sub.add_parser("cpu-matrix", help="evaluate under QEMU user-mode CPU models")
    p.add_argument("artifact", type=Path)
    p.add_argument("--process", required=True)
    p.add_argument("--points", type=Path, required=True)
    p.add_argument("--qemu", required=True, help="qemu-x86_64 executable")
    p.add_argument("--scratch", type=Path, required=True)
    p.add_argument("--models", default=",".join(CPU_MODELS))
    p.add_argument("--client", type=Path)
    p.add_argument("--out")
    p = sub.add_parser("check", help="the whole acceptance on this host")
    p.add_argument("artifact", type=Path)
    p.add_argument("--process", required=True)
    p.add_argument("--scratch", type=Path, required=True)
    p.add_argument("--points", type=Path)
    p.add_argument("--count", type=int, default=2)
    p.add_argument("--client", type=Path)
    p.add_argument("--no-licence", action="store_true")
    p.add_argument("--instances", type=int, default=2)
    p.add_argument("--parameter", action="append", help="NAME=VALUE")
    p.add_argument("--generation-seconds", type=float)
    p.add_argument("--hold-permit", action="store_true", help=HOLD_HELP)
    p.add_argument("--arithmetic", help="double, double-double or arbitrary")
    p.add_argument("--precision", type=int, help="decimal digits")
    p.add_argument("--out")
    p = sub.add_parser("capi-build", help="compile an artifact's C example")
    p.add_argument("artifact", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--archive", type=Path, help="librusticol_capi.a")
    p.add_argument("--include", type=Path)
    p.add_argument("--libs", default=CAPI_NATIVE_LIBS)
    p.add_argument("--rusticol-config")
    p.add_argument("--compiler", default=os.environ.get("CC", "cc"))
    p = sub.add_parser("tree-capi", help="the tree C API acceptance on this host")
    p.add_argument("artifact", type=Path)
    p.add_argument("--client", type=Path, required=True)
    p.add_argument("--scratch", type=Path, required=True)
    p.add_argument("--licence", action="store_true", help="keep SYMBOLICA_LICENSE")
    p.add_argument("--instances", type=int, default=2)
    p.add_argument("--no-permit", action="store_true", help="do not hold the permit")
    p.add_argument("--out")
    args = parser.parse_args(argv)

    if args.command == "host":
        _write(host(), None)
        return 0
    if args.command == "scan":
        record = scan(args.artifact, tuple(args.forbid))
        _write(record, args.out)
        return 0 if record["clean"] else 1
    if args.command == "points":
        _write(points(args.legs, args.count, args.seed), args.out)
        return 0
    if args.command == "evaluate":
        record = evaluate(
            args.artifact,
            args.process,
            args.points,
            parameters=_parameters(args.parameter),
            poles=args.poles,
            arithmetic=args.arithmetic,
            precision=args.precision,
        )
        _write(record, args.out)
        return 3 if record["generation_calls"] and not args.allow_generation else 0
    if args.command == "sdk-build":
        print(
            sdk_build(
                args.out,
                include=args.include,
                lib_dir=args.lib_dir,
                rusticol_config=args.rusticol_config,
                compiler=args.compiler,
            )
        )
        return 0
    if args.command == "sdk":
        record = sdk(
            args.artifact,
            args.process,
            args.points,
            args.client,
            licence=not args.no_licence,
            instances=args.instances,
            hold_permit=args.hold_permit,
            poles=args.poles,
        )
        _write(record, args.out)
        return 0 if all(s == 0 for s in record["exit_status"]) else 1
    if args.command == "compare":
        record = compare(json.loads(args.a.read_text()), json.loads(args.b.read_text()))
        _write(record, args.out)
        return 0 if record["bitwise"] else 1
    if args.command == "cpu-matrix":
        record = cpu_matrix(
            args.artifact,
            args.process,
            args.points,
            args.qemu,
            args.scratch,
            models=tuple(m for m in args.models.split(",") if m),
            client=args.client,
        )
        _write(record, args.out)
        return 0
    if args.command == "check":
        record = check(
            args.artifact,
            args.process,
            args.scratch,
            points_file=args.points,
            count=args.count,
            client=args.client,
            licence=not args.no_licence,
            instances=args.instances,
            parameters=_parameters(args.parameter),
            generation_seconds=args.generation_seconds,
            hold_permit=args.hold_permit,
            arithmetic=args.arithmetic,
            precision=args.precision,
        )
        _write(record, args.out)
        return 0
    if args.command == "capi-build":
        print(
            capi_build(
                args.artifact,
                args.out,
                archive=args.archive,
                include=args.include,
                libs=args.libs,
                rusticol_config=args.rusticol_config,
                compiler=args.compiler,
            )
        )
        return 0
    if args.command == "tree-capi":
        record = tree_capi_check(
            args.artifact,
            args.client,
            args.scratch,
            licence=args.licence,
            instances=args.instances,
            hold_permit=not args.no_permit,
        )
        _write(record, args.out)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
