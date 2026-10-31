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
  generation call; its C example through the tree C API (opt-in input
  ``PYAMPLICOL_CAPI_ARCHIVE``) is bitwise equal at both locations and to the
  Python API, and runs without a licence while another process holds the
  user's restricted Symbolica permit;
* plan v4 loop artifacts, gg -> gg (loop_physics), the six-gluon control
  gg -> gggg and the seven-gluon gg -> ggggg (loop_nightly): the same, plus
  no load-time regeneration (``point_inputs = "native-v2"``: the native
  programs and the binary64 SymJIT plane application of every saved
  evaluator, storage-v4 IR recompiled at load);
* the loop SDK (C ABI through ``rusticol_virtual_open``, which dispatches a
  plan v4 artifact to the native ``V2Runtime``): bitwise equal to the Python
  API at both locations and usable without a licence by concurrent processes;
* unlicensed concurrent processes (Symbolica's restricted mode admits one
  thread per user, so any state access of the Rust copy of Symbolica aborts
  while another process holds the permit): two pyAmpliCol processes without
  ``SYMBOLICA_LICENSE`` load and evaluate a tree artifact at the same time;
  two binary64 loop evaluations while the permit is held and the native
  library key is disabled (binary64 touches no Symbolica state, DESIGN §3.7);
  a DoubleFloat and an arbitrary-precision loop evaluation while the permit
  is held (the native library key unlocks the exact states), and, as the
  control, a DoubleFloat one with the key disabled aborts.

Every unlicensed process of this module uses a private Symbolica lock
directory (``private_symbolica_permit``): "the user's permit" above is that
directory's, which discriminates the same way and leaves the user's real
permit to their other processes.

Other CPU feature levels (QEMU user-mode models) are an opt-in run of the
same tool (``cpu-matrix``), recorded in DESIGN §12.1.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import zlib
from fractions import Fraction
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "loop_artifact_portability.py"
_spec = importlib.util.spec_from_file_location("loop_artifact_portability", TOOL)
portability = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(portability)

RESTRICTED = "restricted Symbolica"
# Disables pyAmpliCol's Symbolica library key in the native code
# (``pyamplicol::DISABLE_VARIABLE``, crate ``rust/crates/pyamplicol``).
NO_NATIVE_KEY = {"PYAMPLICOL_NATIVE_LIBRARY_KEY": "off"}

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


@pytest.fixture(scope="module", autouse=True)
def private_symbolica_permit(tmp_path_factory):
    """A private Symbolica lock directory (0700) for every unlicensed process
    of this module: the permit holder, its control, the instances it covers
    and the concurrent loads. Restricted mode admits one thread per lock
    directory, so the checks discriminate as they would with the user's real
    permit, which stays free for the user's other processes and for the other
    unlicensed test files run in parallel (the tool's ``--hold-permit`` keeps
    the real one)."""

    lock = tmp_path_factory.mktemp("symbolica-locks")
    lock.chmod(0o700)
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("SYMBOLICA_LOCK_DIR", str(lock))
        yield lock


def _unlicensed(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env.pop("SYMBOLICA_LICENSE", None)
    env.update(extra or {})
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


def test_sentinel_entries_name_existing_generation_code():
    """Every watched (file, qualified name) and file exists, so a renamed
    entry point cannot silently drop out of the sentinel."""

    import importlib

    for suffix, qualname in sorted(portability.GENERATION_CODE):
        module = importlib.import_module(suffix.removesuffix(".py").replace("/", "."))
        target = module
        for part in qualname.split("."):
            target = getattr(target, part)
        assert target.__code__.co_filename.replace(os.sep, "/").endswith(suffix)
    for suffix in portability.GENERATION_FILES:
        assert (ROOT / "src" / suffix).is_file(), suffix


def test_permit_lock_directory_follows_symbolica_and_is_explicit(
    tmp_path, private_symbolica_permit
):
    lock = portability.symbolica_lock_dir
    home = {"HOME": "/h"}
    assert lock({**home, "SYMBOLICA_LOCK_DIR": "/l"}) == "/l"
    assert lock({**home, "XDG_RUNTIME_DIR": "/r"}) == "/r/symbolica/locks"
    assert lock({**home, "XDG_CACHE_HOME": "/c"}) == "/c/symbolica/locks"
    assert lock(home) == "/h/.cache/symbolica/locks"
    assert lock({**home, "XDG_RUNTIME_DIR": "relative"}) == "/h/.cache/symbolica/locks"
    # The holder, its control and every covered instance share one lock
    # directory whatever their HOME and XDG_CACHE_HOME (``_fresh``): in this
    # module the private one, never the user's.
    permit = portability.RestrictedPermit()
    assert permit.lock_dir == str(private_symbolica_permit)
    assert permit.lock_dir != lock(
        {k: v for k, v in os.environ.items() if k != "SYMBOLICA_LOCK_DIR"}
    )
    assert private_symbolica_permit.stat().st_mode & 0o777 == 0o700
    env, _cwd = portability._fresh(tmp_path, "x", False, permit)
    assert env["SYMBOLICA_LOCK_DIR"] == permit.lock_dir
    assert env["HOME"] != os.environ.get("HOME")
    assert portability.RestrictedPermit(lock_dir="/p").lock_dir == "/p"


def test_native_library_key_equals_the_python_package_key():
    """pyAmpliCol's library key exists twice: registered with the Python
    ``symbolica`` module (``pyamplicol/__init__.py``) and in the native crate
    (``rust/crates/pyamplicol``). A renewal must change both; they are
    compared by digest, so a failure prints neither key."""

    import ast
    import hashlib
    import re

    tree = ast.parse((ROOT / "src" / "pyamplicol" / "__init__.py").read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_set_library_key"
    ]
    assert len(calls) == 1 and isinstance(calls[0].args[0], ast.Constant)
    python_key = calls[0].args[0].value
    source = (ROOT / "rust" / "crates" / "pyamplicol" / "src" / "lib.rs").read_text()
    match = re.search(r"const KEY: \[&str; \d+\] = \[(.*?)\];", source, re.S)
    assert match is not None
    native_key = "".join(re.findall(r'"([^"]*)"', match.group(1)))
    same = (
        hashlib.sha256(python_key.encode()).digest()
        == hashlib.sha256(native_key.encode()).digest()
    )
    assert python_key and same, "the native and the Python library keys differ"


def test_native_library_key_uses_symbolicas_documented_registration():
    """The key is registered with ``set_library_key!`` in the crate named after
    the package it is issued for, and its ``LibraryUnlock`` stays private to
    that crate (Symbolica's documented use)."""

    crate = ROOT / "rust" / "crates" / "pyamplicol"
    assert 'name = "pyamplicol"' in (crate / "Cargo.toml").read_text()
    source = (crate / "src" / "lib.rs").read_text()
    assert "symbolica::set_library_key!(" in source
    assert "pub(crate) static UNLOCK" in source
    for path in (ROOT / "rust").rglob("*.rs"):
        text = path.read_text(errors="replace")
        assert "for_crate(" not in text, path
        assert "pub mod licence" not in text, path


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


def _leaves(value) -> list[float]:
    """The numbers of an ``evaluate`` record's results: binary64 hexadecimal
    literals and extended-precision decimal strings."""

    if isinstance(value, (list, tuple)):
        return [x for item in value for x in _leaves(item)]
    if isinstance(value, dict):
        return [x for key in sorted(value) for x in _leaves(value[key])]
    if value is None or isinstance(value, bool):
        return []
    text = str(value)
    return [float.fromhex(text) if "0x" in text.lower() else float(text)]


# An exactly massless 2 -> 2 point in binary64 (100^2 + 150^2 + 300^2 =
# 350^2) and mu_R^2: every carrier sees the same on-shell rationals (the
# tool's random binary64 points are on shell only to binary64 rounding, which
# an extended carrier resolves as small external masses).
EXACT_2TO2 = {
    "points": [
        [
            [x.hex() for x in p]
            for p in (
                (350.0, 0.0, 0.0, 350.0),
                (350.0, 0.0, 0.0, -350.0),
                (350.0, 100.0, 150.0, 300.0),
                (350.0, -100.0, -150.0, -300.0),
            )
        ]
    ],
    "mu_r2": [(8315.25).hex()],
}


def _held_evaluations(
    artifact: Path,
    process: str,
    scratch: Path,
    legs: int,
    carriers: list[tuple[str, int | None]],
    extra_env: dict[str, str] | None = None,
    point: dict | None = None,
    parameters: dict[str, str] | None = None,
    poles: str | None = None,
) -> tuple[dict, list[subprocess.CompletedProcess]]:
    """One unlicensed ``evaluate`` process per (arithmetic, digits), all at
    once, while ``RestrictedPermit`` holds the user's permit: a process that
    completes touched no Symbolica state needing the permit (the permit
    record's positive control says the hold discriminates). ``point``: the
    points file's content (default: one of the tool's points); ``parameters``:
    model parameter values every process evaluates at; ``poles``: the pole
    selection (default: the API's)."""

    points = scratch / "points.json"
    points.write_text(json.dumps(point or portability.points(legs, 1)))
    with portability.RestrictedPermit() as permit:
        running = []
        for k, (arithmetic, digits) in enumerate(carriers):
            command = [
                sys.executable,
                str(TOOL),
                "evaluate",
                str(artifact),
                "--process",
                process,
                "--points",
                str(points),
                "--arithmetic",
                arithmetic,
                "--out",
                str(scratch / f"{k}-{arithmetic}.json"),
            ]
            if digits is not None:
                command += ["--precision", str(digits)]
            for name, value in (parameters or {}).items():
                command += ["--parameter", f"{name}={value}"]
            if poles is not None:
                command += ["--poles", poles]
            running.append(
                subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=permit.environment(_unlicensed(extra_env)),
                    cwd=scratch,
                    preexec_fn=portability._no_core,
                )
            )
        done = []
        for process_ in running:
            out, err = process_.communicate(timeout=1800)
            done.append(
                subprocess.CompletedProcess(
                    process_.args, process_.returncode, out, err
                )
            )
    return permit.record, done


@pytest.fixture(scope="module")
def tree_capi_link():
    """How to link the tree C API (``rusticol-capi``): a static archive
    (``PYAMPLICOL_CAPI_ARCHIVE``, from ``cargo rustc --release -p rusticol-capi
    --crate-type staticlib``) or an installed wheel
    (``PYAMPLICOL_VIRTUAL_BUNDLE_INSTALL``, its ``bin/rusticol-config``).
    Neither is a product of the staged native build (which builds the Python
    extension and the loop SDK), so the input skips with its name."""

    if archive := os.environ.get("PYAMPLICOL_CAPI_ARCHIVE"):
        if not Path(archive).is_file():
            pytest.fail(f"PYAMPLICOL_CAPI_ARCHIVE={archive} does not exist")
        return {"archive": Path(archive)}
    if installed := os.environ.get("PYAMPLICOL_VIRTUAL_BUNDLE_INSTALL"):
        config = Path(installed) / "bin" / "rusticol-config"
        if not config.is_file():
            pytest.fail(f"{config} does not exist")
        return {"rusticol_config": str(config)}
    pytest.skip("tree C API unavailable; set PYAMPLICOL_CAPI_ARCHIVE")


def test_tree_c_api_relocates_bitwise_without_a_licence(
    tree_artifact, tree_capi_link, tmp_path
):
    """The tree engine's C distribution path, the artifact's own C example
    (``API/c/check_standalone.c``): at a relocated copy, without
    ``SYMBOLICA_LICENSE``, two concurrent processes run while another process
    holds the user's restricted Symbolica permit (so neither touched Symbolica
    state, although the archive links Symbolica through the loop SDK), and
    every resolved value equals the original location and the bundle's Python
    example bitwise."""

    compiler = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc")
    if compiler is None:
        pytest.skip("no C compiler for the C example; set CC")
    client = portability.capi_build(
        tree_artifact,
        tmp_path / "bin" / "check_standalone",
        compiler=compiler,
        **tree_capi_link,
    )
    verdict = portability.tree_capi_check(tree_artifact, client, tmp_path / "check")
    assert verdict["scan"]["clean"], verdict["scan"]["findings"]
    assert verdict["capi_ran"], verdict["capi"]
    assert verdict["capi"]["relocated"]["permit"]["verified"], verdict["capi"]
    assert verdict["capi_without_licence"], verdict["capi"]
    assert verdict["capi_relocated_bitwise"]["bitwise"], verdict[
        "capi_relocated_bitwise"
    ]
    assert verdict["capi_equals_python"]["bitwise"], verdict["capi_equals_python"]


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
    # The seven-gluon artifact (about 45 min and 30 GB to generate).
    "gg5g": ("g g > g g g g g", pytest.mark.loop_nightly),
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
                count=2 if name == "gg" else 1,
                licence=False,
                generation_seconds=generation,
                hold_permit=name != "gg",
            )
            # The records stay with the artifact (a retained --basetemp is
            # the seven-gluon evidence of DESIGN §12.1).
            (root / "verdict.json").write_text(
                json.dumps({"generation_seconds": generation, **verdict}, indent=1)
            )
            cache[name] = (output, generation, verdict)
        return cache[name]

    return get


def _loop_param(name: str):
    return pytest.param(name, marks=LOOP_PROCESSES[name][1], id=name)


LOOP_NAMES = [_loop_param("gg"), _loop_param("gg4g"), _loop_param("gg5g")]


@pytest.mark.parametrize("name", LOOP_NAMES)
def test_loop_v4_artifact_relocates_bitwise(loop_artifact, name):
    _, _, verdict = loop_artifact(name)
    scan = verdict["scan"]
    assert scan["clean"], scan["findings"]
    assert scan["target"] == {"triple": "portable-64le", "cpu_features": []}
    # No native code: the plan (kind 11), the saved exact evaluator states
    # (kind 2) and their binary64 SymJIT plane applications (kind 1), all
    # storage-v4 IR that the loader recompiles for its CPU.
    assert set(scan["member_kinds"]) == {"1", "2", "11"}, scan["member_kinds"]
    assert scan["member_kinds"]["1"] == scan["member_kinds"]["2"]
    symjit = scan["symjit_applications"]
    assert symjit["members"] > 0 and symjit["storage_v4"] == symjit["members"]
    assert verdict["relocated_bitwise"]["bitwise"], verdict["relocated_bitwise"]
    if name != "gg":
        # Two concurrent unlicensed evaluations of the copy while another
        # process holds the user's restricted permit.
        assert verdict["python_without_licence"], verdict["python_concurrent"]


@pytest.mark.parametrize("name", LOOP_NAMES)
def test_loop_v4_artifact_loads_without_regeneration(loop_artifact, name):
    _, generation, verdict = loop_artifact(name)
    python = verdict["python"]
    assert {r["point_inputs"] for r in python.values()} == {"native-v2"}
    assert verdict["no_generation"], python
    # Loading is a small fraction of generating (measured with native-v2:
    # 3 % at gg -> gggg, 1.4 % at gg -> ggggg).
    assert python["relocated"]["seconds"]["load"] < 0.25 * generation, python


# Generates g g > g g in a fresh interpreter from the UFO source with the model
# cache ``argv[2]`` and writes the SHA-256 of its plan and native members.
_GENERATE_DIGESTS = """
import hashlib, json, sys
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, sys.argv[4])  # the checkout: tools/loop_reference
from tools.loop_reference.model_compatibility import prepare_loop_sm_scalar_declarations
from pyamplicol.loops.native import load_loop_native
from pyamplicol.loops.v2.artifact import generate_process
from pyamplicol.loops.v2.assemble import SliceRequest
from pyamplicol.models.loading import compile_model_source

ufo, cache, out = sys.argv[1:4]
model = compile_model_source(ufo, cache_dir=Path(cache))
model = replace(
    model,
    raw_loop_model=prepare_loop_sm_scalar_declarations(model.raw_loop_model).derived,
    _serialized_path=None,
    _prepared_bundle=None,
)
request = SliceRequest(
    "g g > g g", {"QCD": 2, "QED": 0}, {"QCD": 4, "QED": 0},
    fixed_parameters={"MB": "0"}, scale_parameter="MU_R",
)
generated = generate_process(model, request, load_loop_native())
plan = generated.plan_json
plan = plan.encode() if isinstance(plan, str) else plan
digest = lambda data: hashlib.sha256(data).hexdigest()
members = generated.native.members()
# The parameter planes' values at one real row of their inputs (hex).
from symbolica import Evaluator
from pyamplicol import _rusticol
values = {}
for prefix in ("", "scaleless-"):
    if prefix + "parameters.symjit" not in members:
        continue
    state = bytes(members[prefix + "parameters.bin"])
    shape = Evaluator.load(state).get_instructions()
    row = [0.75 + 0.125 * k for k in range(shape.input_count)]
    flat = _rusticol._evaluate_symjit_plane_application_v2(
        bytes(members[prefix + "parameters.symjit"]), shape.input_count,
        shape.output_count, [x for v in row for x in (v, 0.0)], 1,
    )
    values[prefix + "parameters.symjit"] = [float(x).hex() for x in flat]
Path(out).write_text(json.dumps(
    {"plan": digest(plan), "members": {k: digest(v) for k, v in members.items()},
     "values": values}
))
"""


@pytest.mark.loop_physics
def test_native_members_do_not_depend_on_the_model_cache(
    loop_sm_ufo, loop_sm_compiled_model, loop_native, tmp_path
):
    """g g > g g generated in two fresh interpreters from the UFO source, one
    with an empty model cache and one with the session's warm cache: the plan
    and every native member are byte-identical, except the parameter
    programs' plane applications (the process's and its scaleless
    companion's), whose SymJIT call tags are random
    (``SYMJIT_RANDOM_CALL_TAG``, test_symjit_plane_helper_functions.py; the
    companion's calls are the cut-argument canonicalisation's, MS-MASTERS);
    their values at one real row are bitwise the same. The
    compiled and the cached model list their helper functions in different
    orders; the parameter program names them in sorted order, so its saved
    state is the same."""

    warm = Path(loop_sm_compiled_model).parent / "model-cache"
    assert warm.is_dir(), warm
    script = tmp_path / "generate.py"
    script.write_text(_GENERATE_DIGESTS)
    caches = {"cold": tmp_path / "cold-cache", "warm": warm}
    running = {
        name: subprocess.Popen(
            [
                sys.executable,
                str(script),
                str(loop_sm_ufo),
                str(cache),
                f"{name}.json",
                str(ROOT),
            ],
            cwd=tmp_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for name, cache in caches.items()
    }
    for process in running.values():
        output, _ = process.communicate(timeout=1800)
        assert process.returncode == 0, output[-2000:]
    cold, warm_run = (
        json.loads((tmp_path / f"{name}.json").read_text()) for name in caches
    )
    assert cold["plan"] == warm_run["plan"]
    assert cold["members"].keys() == warm_run["members"].keys()
    differing = sorted(
        name
        for name, digest in cold["members"].items()
        if warm_run["members"][name] != digest
    )
    planes = {"parameters.symjit", "scaleless-parameters.symjit"}
    assert set(differing) <= planes, differing
    # Their values are bitwise the same.
    assert cold["values"].keys() == planes, cold["values"].keys()
    assert cold["values"] == warm_run["values"]


@pytest.mark.loop_physics
def test_loop_artifact_with_retired_or_unknown_point_inputs_is_refused(
    loop_artifact, tmp_path
):
    """A copy whose execution declaration names the retired point inputs
    ``native-v1`` (no SymJIT plane applications) or an unknown kind is refused
    at load with a typed error, its manifest digest kept consistent so the
    refusal is the reader's, not the integrity check's."""

    import hashlib

    from pyamplicol import ArtifactError, Runtime

    artifact, _, _ = loop_artifact("gg")
    expected = {
        "native-v1": "native-v1 stored no SymJIT plane applications",
        "native-v9": "unsupported plan v4 execution declaration",
    }
    for kind, message in expected.items():
        copy = tmp_path / kind
        shutil.copytree(artifact, copy)
        execution = copy / "processes" / "gg" / "execution.json"
        old = execution.read_bytes()
        new = old.replace(b'"native-v2"', f'"{kind}"'.encode())
        assert new != old and len(new) == len(old)
        execution.write_bytes(new)
        manifest = copy / "artifact.json"
        text = manifest.read_bytes()
        digests = (hashlib.sha256(old).hexdigest(), hashlib.sha256(new).hexdigest())
        # The payload record and the process record name the digest.
        assert text.count(digests[0].encode()) >= 1
        manifest.write_bytes(text.replace(digests[0].encode(), digests[1].encode()))
        with pytest.raises(ArtifactError, match=message):
            Runtime.load(copy, process="gg")


@pytest.mark.loop_physics
def test_loop_plane_applications_equal_their_exact_states(loop_artifact):
    """Every stored binary64 plane application (the interaction kernels and
    the parameter program) through the production loader equals Symbolica's
    binary64 evaluation of its exact state: at the parameter program's
    default inputs (the branches its helper functions take there) and at
    random inputs. The parameter program calls helper functions (inlined at
    export) and reads registered constants (pi), which Symbolica's Python
    instruction export writes as 0."""

    import numpy as np
    import symbolica

    from pyamplicol import _rusticol
    from pyamplicol._internal.physics.symbols import symbols
    from pyamplicol.loops.scalar_frontend import _producer_predicate
    from pyamplicol.loops.v2.artifact import read_native_process, read_plan

    for name in ("truth", "equal", "less", "less_equal", "real_part", "imaginary_part"):
        symbols.qualified(
            f"pyamplicol_scalar_v1::{name}", eval={"complex": _producer_predicate(name)}
        )
    artifact, _, _ = loop_artifact("gg")
    (pacbin,) = artifact.glob("processes/*/loop-runtime.pacbin")
    trees = read_native_process(pacbin, read_plan(pacbin)).trees
    assert len(trees.planes) == len(trees.kernels) + 1

    def check(state: bytes, plane: bytes, rows) -> None:
        evaluator = symbolica.Evaluator.load(state)
        exported = evaluator.get_instructions()
        inputs, outputs = exported.input_count, exported.output_count
        rows = np.ascontiguousarray(rows, dtype=np.complex128)
        expected = np.asarray(evaluator.evaluate_complex(rows), dtype=np.complex128)
        expected = expected.reshape(len(rows), outputs)
        flat = _rusticol._evaluate_symjit_plane_application_v2(
            plane, inputs, outputs, rows.view(np.float64).ravel().tolist(), len(rows)
        )
        got = np.asarray(flat).view(np.complex128).reshape(len(rows), outputs)
        same = (
            (got == expected)
            | (np.isnan(got) & np.isnan(expected))
            | (np.abs(got - expected) <= 1e-12 * np.abs(expected))
        )
        assert same.all(), np.argwhere(~same)[:8]

    rng = np.random.default_rng(20261024)
    parameters = trees.program["parameters"]
    exported = symbolica.Evaluator.load(trees.parameters).get_instructions()
    assert exported.sub_evaluators
    assert any(repr(value) == "0" for value in exported.constants)
    defaults = [float(Fraction(value)) for value in parameters["defaults"]]
    if parameters["scale"] is not None:
        defaults[parameters["scale"]] = 91.188
    check(trees.parameters, trees.planes[-1], [defaults])
    check(trees.parameters, trees.planes[-1], rng.uniform(0.5, 1.5, (3, len(defaults))))
    for state, plane in zip(trees.kernels, trees.planes, strict=False):
        width = symbolica.Evaluator.load(state).get_instructions().input_count
        rows = rng.uniform(0.5, 1.5, (3, width)) + 1j * rng.uniform(-1, 1, (3, width))
        check(state, plane, rows)


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


@pytest.mark.loop_physics
def test_loop_v4_sdk_equals_python_bitwise_at_both_locations(
    loop_artifact, loop_sdk_client, tmp_path
):
    artifact, _, _ = loop_artifact("gg")
    verdict = portability.check(
        artifact, "gg", tmp_path, count=2, client=loop_sdk_client, instances=1
    )
    assert verdict["sdk_ran"], verdict["sdk"]
    assert verdict["sdk_relocated_bitwise"]["bitwise"], verdict["sdk_relocated_bitwise"]
    assert verdict["sdk_equals_python"]["bitwise"], verdict["sdk_equals_python"]


@pytest.mark.loop_physics
def test_loop_v4_sdk_selects_poles_like_python(
    loop_artifact, loop_sdk_client, tmp_path
):
    """``rusticol_virtual_set_poles(handle, poles)`` (REQUIREMENTS_ADDENDUM I:
    the pole option reaches the SDKs): every selection ("all", "none" and
    the split "uv" and "ir", which run the stored companion of the
    certified-scaleless families) is bitwise the Python API's, the finite
    part is the same in all four, and "none" leaves the poles absent (NaN;
    None in Python)."""

    artifact, _, _ = loop_artifact("gg")
    points = tmp_path / "points.json"
    points.write_text(json.dumps(portability.points(4, 2)))
    records = {
        poles: portability.sdk(
            artifact,
            "gg",
            points,
            loop_sdk_client,
            workdir=tmp_path / f"sdk-{poles}",
            poles=poles,
        )
        for poles in ("all", "none", "uv", "ir")
    }
    assert all(r["exit_status"] == [0] for r in records.values()), records
    finite = None
    for poles, record in records.items():
        python = portability.evaluate(artifact, "gg", points, poles=poles)
        assert record["results"]["born"] == python["results"]["born"], poles
        sdk = record["results"]["virtual"]
        here = [[laurent[0] for laurent in point] for point in sdk]
        assert finite is None or here == finite, poles
        finite = here
        if poles == "none":
            assert here == [
                [laurent[0] for laurent in point]
                for point in python["results"]["virtual"]
            ]
            assert all(
                laurent[1:] == ["nan", "nan"] for point in sdk for laurent in point
            )
            assert all(
                laurent[1:] == [None, None]
                for point in python["results"]["virtual"]
                for laurent in point
            )
        else:
            assert sdk == python["results"]["virtual"], poles
    # The split selections differ from "all" in their poles.
    assert records["uv"]["results"]["virtual"] != records["all"]["results"]["virtual"]
    assert records["ir"]["results"]["virtual"] != records["uv"]["results"]["virtual"]


SDK = ROOT / "rust" / "loop-reduction" / "sdk"


@pytest.mark.loop_fast
def test_loop_sdk_cpp_pole_wrapper_compiles(tmp_path):
    """The C++ wrapper's ``set_poles`` takes every pole selection (syntax
    check against the C header; no SDK library needed)."""

    compiler = os.environ.get("CXX") or shutil.which("c++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("environment: no C++ compiler (CXX, c++ or g++)")
    source = tmp_path / "poles.cpp"
    source.write_text(
        '#include "rusticol_virtual.hpp"\n'
        "void select(rusticol::VirtualRuntime &runtime) {\n"
        '  for (const char *poles : {"all", "none", "uv", "ir"})\n'
        "    runtime.set_poles(poles);\n"
        "}\n"
    )
    run = subprocess.run(
        [compiler, "-std=c++17", "-fsyntax-only", f"-I{SDK / 'include'}", str(source)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr


@pytest.mark.loop_fast
def test_loop_sdk_fortran_pole_wrapper_compiles(tmp_path):
    """The Fortran module's ``set_poles`` takes every pole selection (the
    module compiled, a caller syntax-checked against it)."""

    compiler = os.environ.get("FC") or shutil.which("gfortran")
    if compiler is None:
        pytest.skip("environment: no Fortran compiler (FC or gfortran)")
    module = subprocess.run(
        [
            compiler,
            "-std=f2018",
            "-c",
            str(SDK / "fortran" / "rusticol_virtual.f90"),
            "-J",
            str(tmp_path),
            "-o",
            str(tmp_path / "rusticol_virtual.o"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert module.returncode == 0, module.stderr
    source = tmp_path / "poles.f90"
    source.write_text(
        "subroutine select(runtime)\n"
        "  use rusticol_virtual\n"
        "  type(virtual_runtime), intent(inout) :: runtime\n"
        '  call runtime%set_poles("all")\n'
        '  call runtime%set_poles("none")\n'
        '  call runtime%set_poles("uv")\n'
        '  call runtime%set_poles("ir")\n'
        "end subroutine\n"
    )
    run = subprocess.run(
        [compiler, "-std=f2018", "-fsyntax-only", f"-I{tmp_path}", str(source)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr


@pytest.mark.loop_physics
def test_loop_v4_sdk_needs_no_symbolica_licence(
    loop_artifact, loop_sdk_client, tmp_path
):
    """The f64 SDK follows the tree engine's rule: no Symbolica licence, so
    concurrent unlicensed SDK processes (a Monte Carlo farm) all run. A
    Symbolica state access takes the user's single restricted-thread permit,
    so the instances run while another process holds it: any access aborts."""

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
        hold_permit=True,
    )
    assert record["permit"]["verified"], record["permit"]
    assert record["exit_status"] == [0, 0], record["stderr"]
    assert not record["symbolica_restricted"]


@pytest.mark.loop_physics
def test_binary64_loop_evaluation_touches_no_symbolica_state(loop_artifact, tmp_path):
    """Binary64 runs the stored SymJIT plane applications (DESIGN §3.7): two
    concurrent unlicensed evaluations complete while another process holds
    the user's restricted permit, with pyAmpliCol's native library key
    disabled, so neither touched the Rust copy of Symbolica."""

    artifact, _, _ = loop_artifact("gg")
    permit, runs = _held_evaluations(
        artifact, "gg", tmp_path, 4, [("double", None)] * 2, NO_NATIVE_KEY
    )
    assert permit["verified"], permit
    for run in runs:
        assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]
        assert RESTRICTED not in run.stdout + run.stderr


@pytest.mark.loop_physics
def test_binary64_parameter_changes_need_no_key(loop_artifact, tmp_path):
    """Binary64 evaluates the stored parameter program's plane at any model
    parameters with no Symbolica (DESIGN §12.1 (e)): the translation
    canonicalises a cut argument's zero imaginary part
    (``symjit_plane::canonical_cut_row``), so there is
    nothing to check at run time. g g > g g on loop_sm completes unlicensed
    with the native key disabled while the permit is held at a new alpha_s,
    a new top mass and with the Z, W or Higgs width set to 0 (each of which
    needed the key under the former run-time side check, MS-MASTERS
    rounds 1-4)."""

    artifact, _, _ = loop_artifact("gg")
    changes = (
        {"aS": "0.12"},
        {"MT": "172"},
        {"WZ": "0"},
        {"WW": "0"},
        {"WH": "0"},
    )
    for index, parameters in enumerate(changes):
        scratch = tmp_path / f"change-{index}"
        scratch.mkdir()
        permit, (run,) = _held_evaluations(
            artifact,
            "gg",
            scratch,
            4,
            [("double", None)],
            NO_NATIVE_KEY,
            parameters=parameters,
        )
        assert permit["verified"], permit
        output = run.stdout[-2000:] + run.stderr[-2000:]
        assert run.returncode == 0, (parameters, output)
        assert RESTRICTED not in output, (parameters, output)


def _ew_selftest_point(expected: dict, indices) -> dict:
    """The points file of the shipped EW self-test's MadLoop points
    ``indices`` (``expected.json``), each at its own scale."""

    return {
        "schema": portability.SCHEMA,
        "kind": "points",
        "legs": 4,
        "points": [
            [[float(c).hex() for c in p] for p in expected["points"][k]]
            for k in indices
        ],
        "mu_r2": [float(expected["mu_r2"][k]).hex() for k in indices],
    }


@pytest.mark.loop_physics
def test_binary64_ew_evaluations_need_no_key(tmp_path):
    """The shipped EW self-test artifact (e- e+ > e- e+ [QED] on
    loop_qcd_qed_sm; its counterterms read log(-MZ^2/mu_R^2) and MW's square
    root of a difference, branch cuts of the parameters) evaluates MadLoop's
    second and third points at their own scales (mu_R^2 = 4.9e5 and 1.96e6)
    in binary64 with the native key disabled while the permit is held, for
    every pole selection (the split selections' regulated pass included),
    bitwise equal to the keyed run and equal to double-double (the exact
    states) by the binary64 rule of observables; so are a changed MZ, MW
    and LogSwitch = 0 (each of which needed the key under the former
    run-time side check). The per-output check of the stored parameter plane
    at such values: ``test_ew_parameter_plane_equals_its_state_off_the_``
    ``generation_values``."""

    from pyamplicol._internal import selftest_virtual_ew as selftest

    root = selftest.fixture_root()
    artifact = tmp_path / "artifact"
    shutil.copytree(root / "artifact", artifact)
    expected = json.loads((root / "expected.json").read_text(encoding="utf-8"))
    point = _ew_selftest_point(expected, (1, 2))

    def run(name, carrier, env, poles, parameters=None):
        scratch = tmp_path / name
        scratch.mkdir()
        permit, (process,) = _held_evaluations(
            artifact,
            selftest.NAME,
            scratch,
            4,
            [(carrier, None)],
            env,
            point=point,
            poles=poles,
            parameters=parameters,
        )
        assert permit["verified"], permit
        output = process.stdout[-2000:] + process.stderr[-2000:]
        assert process.returncode == 0, (name, output)
        assert RESTRICTED not in output, (name, output)
        return json.loads((scratch / f"0-{carrier}.json").read_text())["results"]

    def agree(binary64, exact):
        binary64, exact = _leaves(binary64), _leaves(exact)
        assert len(binary64) == len(exact) > 0
        scale = max(abs(v) for v in exact)
        # The binary64 rule of observables (1e-12, HANDOFF §0.5): a cut side
        # taken differently moves them at O(1e-5) and more.
        pairs = zip(binary64, exact, strict=True)
        return all(abs(a - b) <= 1e-12 * scale for a, b in pairs)

    for poles in ("all", "none", "uv", "ir"):
        off = run(f"{poles}-off", "double", NO_NATIVE_KEY, poles)
        assert off == run(f"{poles}-keyed", "double", None, poles), poles
        if poles == "all":
            assert agree(off, run("all-dd", "double-double", None, "all"))
    for index, parameters in enumerate(
        ({"MZ": "91.19"}, {"MW": "80.5"}, {"LogSwitch": "0"})
    ):
        off = run(f"change-{index}", "double", NO_NATIVE_KEY, "all", parameters)
        dd = run(f"change-{index}-dd", "double-double", None, "all", parameters)
        assert agree(off, dd), parameters


@pytest.mark.loop_fast
def test_ew_parameter_plane_equals_its_state_off_the_generation_values():
    """The shipped EW self-test artifact's stored binary64 parameter plane
    equals Symbolica's binary64 evaluation of its exact state output by
    output (1e-12 of each output's size, the plane check's criterion) at
    parameter values the generation never saw: new scales, alpha_s and
    alpha, the epsIR = 0 of the split pole selections' regulated pass,
    LogSwitch = 0, new masses, and MW = 100 > MZ, which puts MW's square
    root of a difference on its cut (with SymJIT's own signed zeros the
    plane took the other side there, output 2; MS-MASTERS round 5 makes the
    translation take exact arithmetic's side). This is the run-time
    behaviour: binary64 runs the stored plane unchecked."""

    import numpy as np
    import symbolica

    from pyamplicol import _rusticol
    from pyamplicol._internal import selftest_virtual_ew as selftest
    from pyamplicol._internal.physics.symbols import symbols
    from pyamplicol.loops.scalar_frontend import _producer_predicate
    from pyamplicol.loops.v2.artifact import read_native_process, read_plan

    for name in ("truth", "equal", "less", "less_equal", "real_part", "imaginary_part"):
        symbols.qualified(
            f"pyamplicol_scalar_v1::{name}", eval={"complex": _producer_predicate(name)}
        )
    (pacbin,) = (selftest.fixture_root() / "artifact").glob(
        "processes/*/loop-runtime.pacbin"
    )
    trees = read_native_process(pacbin, read_plan(pacbin)).trees
    record = trees.program["parameters"]
    names = list(record["names"])
    base = {
        name: float(Fraction(value))
        for name, value in zip(names, record["defaults"], strict=True)
    }
    rows = [[base[name] for name in names]]
    for change in (
        {"MU_R": 10.0},
        {"MU_R": 700.0},
        {"MU_R": 1400.0},
        {"MU_R": 1e5},
        {"aS": 0.05},
        {"aS": 0.2},
        {"aEWM1": 132.5},
        {"epsIR": 0.0},
        {"LogSwitch": 0.0},
        {"MT": 172.5},
        {"MZ": 91.19},
        {"MW": 100.0},
    ):
        assert set(change) <= set(names), change
        rows.append([{**base, **change}[name] for name in names])
    evaluator = symbolica.Evaluator.load(trees.parameters)
    outputs = evaluator.get_instructions().output_count
    rows = np.ascontiguousarray(rows, dtype=np.complex128)
    expected = np.asarray(evaluator.evaluate_complex(rows), dtype=np.complex128)
    expected = expected.reshape(len(rows), outputs)
    flat = _rusticol._evaluate_symjit_plane_application_v2(
        trees.planes[-1],
        len(names),
        outputs,
        rows.view(np.float64).ravel().tolist(),
        len(rows),
    )
    got = np.asarray(flat).view(np.complex128).reshape(len(rows), outputs)
    finite = np.isfinite(expected)
    scale = np.where(finite, np.abs(expected), 0).max(axis=0)
    same = (
        (got == expected)
        | (np.isnan(got) & np.isnan(expected))
        | (np.abs(got - expected) <= 1e-12 * scale)
    )
    assert same.all(), [(int(r), int(o)) for r, o in np.argwhere(~same)[:8]]
    # MW = 100 reaches a cut: an output real at the defaults is complex there.
    assert ((expected[0].imag == 0) & (expected[-1].imag != 0)).any()


@pytest.mark.loop_physics
def test_unlicensed_extended_precision_loop_evaluations_run_concurrently(
    loop_artifact, tmp_path
):
    """DoubleFloat and arbitrary precision evaluate the exact states in the
    Rust copy of Symbolica, which pyAmpliCol's library key unlocks: a
    DoubleFloat and a 32-digit evaluation complete concurrently with a
    binary64 one, without a licence, while another process holds the user's
    restricted permit, and agree with binary64; with the key disabled a
    DoubleFloat evaluation aborts (the control)."""

    artifact, _, _ = loop_artifact("gg")
    carriers = [("double", None), ("double-double", None), ("arbitrary", 32)]
    permit, runs = _held_evaluations(
        artifact, "gg", tmp_path, 4, carriers, point=EXACT_2TO2
    )
    assert permit["verified"], permit
    for run in runs:
        assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]
        assert RESTRICTED not in run.stdout + run.stderr
    values = [
        _leaves(json.loads((tmp_path / f"{k}-{a}.json").read_text())["results"])
        for k, (a, _digits) in enumerate(carriers)
    ]
    binary64 = values[0]
    scale = max(abs(v) for v in binary64)
    for extended in values[1:]:
        assert len(extended) == len(binary64)
        # The binary64 rule of observables (1e-12, HANDOFF §0.5).
        assert all(
            abs(a - b) <= 1e-12 * scale for a, b in zip(extended, binary64, strict=True)
        )
    control = tmp_path / "control"
    control.mkdir()
    _, (refused,) = _held_evaluations(
        artifact, "gg", control, 4, [("double-double", None)], NO_NATIVE_KEY
    )
    assert refused.returncode != 0
    assert RESTRICTED in refused.stdout + refused.stderr
