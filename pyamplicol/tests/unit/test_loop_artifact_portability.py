# SPDX-License-Identifier: 0BSD
"""Portability of prepared artifacts (requirements addendum G, HANDOFF §0.2).

An artifact prepared on one host (SymJIT -O2, the default) must load and
evaluate on another machine, user and path. The checks run through
``tools/loop_artifact_portability.py``, whose ``check`` is also the opt-in
cross-host acceptance (its docstring), on:

* the tool itself: the scan finds host paths, user and host names in files
  and in zlib PACBIN members, native binaries and a non-portable target, and
  passes portable scripts; the generation sentinel counts generation calls
  but not import-time set-up; the SymJIT code-path class of a CPU;
* a tree artifact (recurrence, SymJIT -O2 plane kernels, API bundle): no host
  string anywhere (the configuration records keep file names), every SymJIT
  member storage-v4 IR, target ``portable-64le`` without CPU features, a
  relocated copy in fresh interpreters with other working, home and temporary
  directories and user names gives bitwise-identical binary64 values with no
  generation call;
* plan v4 loop artifacts, gg -> gg (loop_physics) and the six-gluon control
  gg -> gggg (loop_nightly): the same, plus no load-time regeneration once the
  point inputs are stored natively (``point_inputs = "native-v1"``, WP3.1;
  until then this part is an expected failure with the reason);
* the loop SDK (C ABI through ``rusticol_virtual_open``): bitwise equal to the
  Python API at both locations (an expected failure until the SDK evaluates
  plan v4, generation-scale-2), and usable without a Symbolica licence by
  concurrent processes (the tree engine's rule for f64 evaluation,
  ``docs/user/symbolica-and-licensing.md``): an expected failure, the open
  defect ``NATIVE_LOOP_NEEDS_LICENCE``;
* unlicensed concurrent processes: two pyAmpliCol processes without
  ``SYMBOLICA_LICENSE`` load a tree and a loop artifact at the same time (the
  Python package key does not reach the Rust copy of Symbolica, whose
  restricted mode admits one thread per user); the loop case is the same
  open defect.

The seven-gluon artifact and other CPU feature levels (QEMU user-mode models)
are opt-in runs of the same tool (``check``, ``cpu-matrix``), recorded in
DESIGN §12.1.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import zlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "loop_artifact_portability.py"
_spec = importlib.util.spec_from_file_location("loop_artifact_portability", TOOL)
portability = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(portability)

# The point inputs are stored natively from WP3.1 (generation-scale) on.
NATIVE_POINT_INPUTS = (
    importlib.util.find_spec("pyamplicol.loops.v2.native_process") is not None
)
REGENERATES = (
    "WP3.1: the slice regenerates the trees, Born and counterterm inputs at "
    "load (point_inputs = 'regenerated-at-load')"
)
SDK_WITHOUT_V4 = (
    "the loop SDK refuses plan v4 until generation-scale-2 (V2Runtime and the "
    "rusticol_virtual_open dispatch)"
)
# Open defect (DESIGN §12.1, decision pending): every loop f64 path decodes
# Symbolica exact evaluators natively (legacy coefficients, the slice's Python
# point inputs through CoefficientEvaluator, the native-v1 tree program, the
# SDK's V2Runtime), which without
# SYMBOLICA_LICENSE takes Symbolica's restricted one-thread-per-user permit:
# a second unlicensed process aborts. Fix: f64 SymJIT kernels (WP3.1 PLAN
# §4.2, as the tree engine) or pyAmpliCol's library key in the native code.
NATIVE_LOOP_NEEDS_LICENCE = (
    "loop evaluation decodes Symbolica exact evaluators natively: without "
    "SYMBOLICA_LICENSE a second concurrent process aborts (DESIGN §12.1)"
)
RESTRICTED = "restricted Symbolica"

SCHEME = """
[loop.model]
loop_coupling_orders = ["QCD"]
uv_complete = true
model_r2_complete = true
[loop.model.scheme]
dimensional_regularization = "HV"
external_states = "physical-four-dimensional"
gamma5 = "source-UFO-four-dimensional-prescription"
renormalization = "source-UFO mass-dependent decoupling"
mass_scheme = "real-on-shell"
finite_restoration = "source-UFO-local-rules"
model_convention = "loop_sm lhv=1 supplied QCD rules"
"""


def _unlicensed() -> dict[str, str]:
    env = dict(os.environ)
    env.pop("SYMBOLICA_LICENSE", None)
    return env


# ------------------------------------------------------------------ the tool


def test_scan_reports_host_strings_absolute_paths_and_native_code(tmp_path):
    from io import BytesIO

    from pyamplicol.generation.evaluator_container import (
        PacbinMemberKind,
        PacbinMemberSource,
        write_pacbin,
    )

    def artifact(root: Path, *, leak: bool) -> Path:
        (root / "config").mkdir(parents=True)
        (root / "API").mkdir()
        target = {"triple": "portable-64le", "cpu_features": []}
        if leak:
            target = {"triple": "x86_64-unknown-linux-gnu", "cpu_features": ["avx2"]}
        (root / "artifact.json").write_text(
            json.dumps({"producer": {"target": target}})
        )
        output = str(root) if leak else "artifact"
        (root / "config/effective.toml").write_text(f'output = "{output}"\n')
        script = "#!/usr/bin/env python3\nprint('relative/path')\n"
        if leak:
            script += "open('/opt/elsewhere/data')\n"
        (root / "API/check.py").write_text(script)
        member = b'{"model": "loop-sm.json"}'
        if leak:
            member = b'{"model": "/home/someone/models/loop-sm.json"}'
        write_pacbin(
            root / "processes.pacbin",
            (
                PacbinMemberSource(
                    "plan.json.zlib",
                    PacbinMemberKind.LOOP_RUNTIME_PLAN,
                    BytesIO(zlib.compress(member)),
                ),
            ),
        )
        if leak:
            (root / "lib.so").write_bytes(b"\x7fELF\x02\x01\x01" + bytes(16))
        return root

    clean = portability.scan(artifact(tmp_path / "clean", leak=False))
    assert clean["clean"], clean["findings"]
    assert clean["member_kinds"] == {"11": 1}

    leaked = portability.scan(artifact(tmp_path / "leaked", leak=True))
    kinds = {(f["kind"], f["where"]) for f in leaked["findings"]}
    assert ("host-string", "config/effective.toml") in kinds
    assert ("absolute-path", "API/check.py") in kinds
    assert ("absolute-path", "processes.pacbin!plan.json.zlib") in kinds
    assert ("native-binary", "lib.so") in kinds
    assert ("target", "artifact.json") in kinds
    assert not leaked["clean"]


def test_sentinel_counts_generation_calls_not_import_set_up():
    from pyamplicol.loops.v2 import schedules

    with portability.GenerationSentinel() as sentinel:
        # A module body building a policy at import time is not generation.
        namespace = dict(vars(schedules))
        exec(compile("policy = cheapest(int)", "<module>", "exec"), namespace)
        assert not sentinel.calls
        schedules.cheapest(int)
    assert sentinel.calls == {"loops/v2/schedules.py:cheapest": 1}


def test_cpu_class_follows_symjit_code_generator_selection():
    cls = portability.symjit_class
    assert cls("x86_64", {"sse2", "avx", "fma", "avx2"}) == "x86-64-avx-fma"
    assert cls("x86_64", {"sse2", "avx"}) == "x86-64-avx-without-fma"
    assert cls("x86_64", {"sse2", "sse4_2"}) == "x86-64-sse"
    assert cls("aarch64", set()) == "aarch64"
    assert portability.symjit_class() == cls(None, portability._cpu_flags())


# ------------------------------------------------------------------ tree


@pytest.fixture(scope="module")
def tree_artifact(tmp_path_factory):
    """g g > g g g with the defaults: recurrence, SymJIT -O2 plane kernels,
    API bundle."""

    root = tmp_path_factory.mktemp("tree")
    output = root / "artifact"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pyamplicol",
            "generate",
            "g g > g g g",
            str(output),
            "--no-symbolica-suggestion",
            "--progress",
            "off",
        ],
        check=True,
        capture_output=True,
        cwd=root,
    )
    return output


def test_tree_jit_o2_artifact_relocates_bitwise(tree_artifact, tmp_path):
    verdict = portability.check(
        tree_artifact, "g_g_to_g_g_g", tmp_path, count=2, licence=False
    )
    scan = verdict["scan"]
    assert scan["clean"], scan["findings"]
    assert scan["target"] == {"triple": "portable-64le", "cpu_features": []}
    symjit = scan["symjit_applications"]
    assert symjit["members"] > 0 and symjit["storage_v4"] == symjit["members"]
    assert verdict["relocated_bitwise"]["bitwise"], verdict["relocated_bitwise"]
    assert verdict["no_generation"], verdict["python"]


def test_unlicensed_processes_load_tree_artifacts_concurrently(tree_artifact, tmp_path):
    """Without SYMBOLICA_LICENSE the first Symbolica state access of a process
    takes the user's one restricted-thread permit; f64 loading and evaluation
    must not access it, so a second concurrent process does not abort."""

    _concurrent_loads(tree_artifact, "g_g_to_g_g_g", tmp_path, 5)


def _concurrent_loads(artifact: Path, process: str, scratch: Path, legs: int) -> None:
    """Two unlicensed processes each load the artifact and evaluate a point
    (native programs may be built at the first evaluation); the first one
    stays alive (blocked on its stdin) until the second has finished."""

    points = scratch / "points.json"
    points.write_text(json.dumps(portability.points(legs, 1)))
    hold = scratch / "hold.py"
    hold.write_text(
        "import importlib.util, sys, time\n"
        f"spec = importlib.util.spec_from_file_location('tool', {str(TOOL)!r})\n"
        "tool = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(tool)\n"
        "tool.evaluate(sys.argv[1], sys.argv[2], sys.argv[3])\n"
        "print('loaded', flush=True)\n"
        "sys.stdin.read() if sys.argv[4] == 'hold' else None\n"
    )
    first = subprocess.Popen(
        [sys.executable, str(hold), str(artifact), process, str(points), "hold"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=_unlicensed(),
        cwd=scratch,
    )
    seen = []
    for line in first.stdout:
        seen.append(line)
        if line.strip() == "loaded":
            break
    else:
        first.wait()
        pytest.fail("the first process did not load: " + "".join(seen)[-2000:])
    second = subprocess.run(
        [sys.executable, str(hold), str(artifact), process, str(points), "go"],
        capture_output=True,
        text=True,
        env=_unlicensed(),
        cwd=scratch,
    )
    rest = "".join(seen) + first.communicate(input="", timeout=120)[0]
    assert second.returncode == 0, second.stdout[-2000:] + second.stderr[-2000:]
    assert first.returncode == 0, rest[-2000:]
    assert RESTRICTED not in second.stdout + rest


# ------------------------------------------------------------------ loops


def _loop_card(model: Path, output: Path, name: str, expression: str) -> str:
    return f"""
action = "generate"
[model]
source = {json.dumps(str(model))}
[process]
entries = [{{expression={json.dumps(expression)}, name={json.dumps(name)}}}]
loop_coupling_orders = ["QCD"]
coupling_order_policy = "minimal"
[color]
accuracy = "full"
contraction = "direct"
[loop]
r2_strategy = "model"
engine = "v2"
[loop.preparation]
fixed_parameters = {{MB = "0"}}
{SCHEME}
[generation]
output = {json.dumps(str(output))}
emit_api_bundle = false
[generation.validation]
post_build_validation = false
samples = 1
"""


LOOP_PROCESSES = {
    "gg": ("g g > g g", pytest.mark.loop_physics),
    "gg4g": ("g g > g g g g", pytest.mark.loop_nightly),
}


@pytest.fixture(scope="module")
def loop_artifact(loop_sm_compiled_model, loop_native, tmp_path_factory):
    """A plan v4 artifact per process, generated through the public API, with
    its generation time; ``check`` at a relocated copy (licence unset)."""

    import time

    from pyamplicol.api import Generator, ProcessRequest, ProcessSet
    from pyamplicol.config import resolve_config

    cache: dict[str, tuple[Path, float, dict]] = {}

    def get(name: str):
        if name not in cache:
            expression = LOOP_PROCESSES[name][0]
            root = tmp_path_factory.mktemp(f"loop-{name}")
            output = root / "artifact"
            card = root / "card.toml"
            card.write_text(
                _loop_card(loop_sm_compiled_model, output, name, expression)
            )
            start = time.perf_counter()
            Generator(resolve_config(card)).generate(
                ProcessSet((ProcessRequest.parse(expression, name=name),)), output
            )
            generation = time.perf_counter() - start
            verdict = portability.check(
                output,
                name,
                root / "check",
                count=1 if name == "gg4g" else 2,
                licence=False,
                generation_seconds=generation,
            )
            cache[name] = (output, generation, verdict)
        return cache[name]

    return get


def _loop_param(name: str):
    return pytest.param(name, marks=LOOP_PROCESSES[name][1], id=name)


def _six_gluons_need_native_inputs(name: str) -> None:
    if name == "gg4g" and not NATIVE_POINT_INPUTS:
        # The Python per-point path costs ~4 min per point at six gluons.
        pytest.xfail(f"{REGENERATES}; not run at six gluons")


@pytest.mark.parametrize("name", [_loop_param("gg"), _loop_param("gg4g")])
def test_loop_v4_artifact_relocates_bitwise(loop_artifact, name):
    _six_gluons_need_native_inputs(name)
    _, _, verdict = loop_artifact(name)
    scan = verdict["scan"]
    assert scan["clean"], scan["findings"]
    assert scan["target"] == {"triple": "portable-64le", "cpu_features": []}
    # No compiled code at all: the plan (kind 11) and, natively stored point
    # inputs, Symbolica exact evaluator states (kind 2), recompiled or
    # interpreted at load.
    assert set(scan["member_kinds"]) <= {"2", "11"}, scan["member_kinds"]
    assert verdict["relocated_bitwise"]["bitwise"], verdict["relocated_bitwise"]


@pytest.mark.parametrize("name", [_loop_param("gg"), _loop_param("gg4g")])
def test_loop_v4_artifact_loads_without_regeneration(request, loop_artifact, name):
    _six_gluons_need_native_inputs(name)
    if not NATIVE_POINT_INPUTS:
        request.applymarker(pytest.mark.xfail(reason=REGENERATES, strict=True))
    _, generation, verdict = loop_artifact(name)
    python = verdict["python"]
    assert {r["point_inputs"] for r in python.values()} == {"native-v1"}
    assert verdict["no_generation"], python
    # Loading is a small fraction of generating (measured: 4 % at gg -> gggg,
    # 2 % at gg -> ggggg).
    assert python["relocated"]["seconds"]["load"] < 0.25 * generation, python


@pytest.fixture(scope="session")
def loop_sdk_client(tmp_path_factory):
    """The C client of the tool against the staged build's loop SDK."""

    info = ROOT / ".artifacts" / "source-runtime" / "_build_info.json"
    if not info.is_file():
        pytest.fail(f"{info} missing: stage a native build first")
    build = Path(json.loads(info.read_text())["native_build"]["directory"])
    if not (build / "librusticol_loop_sdk.so").is_file():
        pytest.fail(f"{build} has no librusticol_loop_sdk.so")
    compiler = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc")
    if compiler is None:
        pytest.skip("no C compiler for the SDK client; set CC")
    return portability.sdk_build(
        tmp_path_factory.mktemp("sdk") / "client",
        include=ROOT / "rust" / "loop-reduction" / "sdk" / "include",
        lib_dir=build,
        compiler=compiler,
    )


def _sdk_evaluates_v4(loop_native) -> bool:
    return hasattr(loop_native, "evaluate_v2_artifact")


@pytest.mark.loop_physics
def test_loop_v4_sdk_equals_python_bitwise_at_both_locations(
    request, loop_artifact, loop_sdk_client, loop_native, tmp_path
):
    if not _sdk_evaluates_v4(loop_native):
        request.applymarker(pytest.mark.xfail(reason=SDK_WITHOUT_V4, strict=True))
    artifact, _, _ = loop_artifact("gg")
    verdict = portability.check(
        artifact, "gg", tmp_path, count=2, client=loop_sdk_client, instances=1
    )
    assert verdict["sdk_ran"], verdict["sdk"]
    assert verdict["sdk_relocated_bitwise"]["bitwise"], verdict["sdk_relocated_bitwise"]
    assert verdict["sdk_equals_python"]["bitwise"], verdict["sdk_equals_python"]


@pytest.mark.loop_physics
def test_loop_v4_sdk_needs_no_symbolica_licence(
    request, loop_artifact, loop_sdk_client, loop_native, tmp_path
):
    """The f64 SDK follows the tree engine's rule: no Symbolica licence, so
    concurrent unlicensed SDK processes (a Monte Carlo farm) all run. A
    Symbolica state access at open takes the user's single restricted-thread
    permit and aborts every other unlicensed instance."""

    reason = (
        NATIVE_LOOP_NEEDS_LICENCE if _sdk_evaluates_v4(loop_native) else SDK_WITHOUT_V4
    )
    request.applymarker(pytest.mark.xfail(reason=reason, strict=True))
    artifact, _, _ = loop_artifact("gg")
    points = tmp_path / "points.json"
    points.write_text(json.dumps(portability.points(4, 1)))
    record = portability.sdk(
        artifact,
        "gg",
        points,
        loop_sdk_client,
        licence=False,
        instances=2,
        workdir=tmp_path / "sdk",
    )
    assert record["exit_status"] == [0, 0], record["stderr"]
    assert not record["symbolica_restricted"]


@pytest.mark.loop_physics
def test_unlicensed_processes_load_loop_artifacts_concurrently(
    request, loop_artifact, tmp_path
):
    request.applymarker(
        pytest.mark.xfail(reason=NATIVE_LOOP_NEEDS_LICENCE, strict=True)
    )
    artifact, _, _ = loop_artifact("gg")
    _concurrent_loads(artifact, "gg", tmp_path, 4)
