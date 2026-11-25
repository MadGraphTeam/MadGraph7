# SPDX-License-Identifier: 0BSD
"""Runtime options (USER 2026-11-19): one runtime-options TOML for every
interface, every option also settable on its own, and the per-runtime
thread bound (``evaluation.threads``).

The Python configuration reads the TOML itself (``config.runtime_options``);
the native SDKs read it with ``rusticol_loop_reduction::runtime_options``
(``rusticol_virtual_set_options``). These tests hold the two equal: the same
keys, the same accepted values, the same refusal texts, the same runtime
(bitwise values) from the same TOML; and the thread bound: values
independent of the count, per-runtime counts, ``RAYON_NUM_THREADS`` as the
fallback."""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.unit.test_loop_v2_sdk_extras import _error, _sdk
from tests.unit.test_loop_v2_sdk_induced import _evaluate, _python_flat, _same
from tests.unit.test_loop_v2_sdk_parity import POINT_173, ggtt  # noqa: F401

# The loop tier (the native loop extension, a generated g g > t t~ artifact).
pytestmark = pytest.mark.loop_physics

STATUS_INVALID_ARGUMENT = 1
POINTS = [POINT_173, POINT_173]
SCALES = ["1000000", "1000000"]

# Every runtime option at a value other than its default.
EVERY_KEY = """
[model]
source = "ignored: not [evaluation]"
[evaluation]
artifact = "ignored.pyamplicol"
momenta = "ignored.json"
mu_r2 = "1"
reduction = "ninjars"
reduction_target = 1e-7
reduction_unchecked = true
lane_rescue = false
arithmetic = "double-double"
poles = "none"
on_failure = "return"
kinematics_improvement = "projection"
kinematics_improvement_binary64 = true
threads = 2
[evaluation.stability_checks]
rotation_count = 1
reduction_direction = true
rescaled = true
"""

# Refusals: (TOML, the text both readers state).
REFUSALS = [
    (
        "[evaluation]\nredution = 'auto'",
        "unknown configuration field 'evaluation.redution'; did you mean "
        "'evaluation.reduction'?",
    ),
    (
        "[evaluation]\nthread = 2",
        "unknown configuration field 'evaluation.thread'; did you mean "
        "'evaluation.threads'?",
    ),
    (
        "[evaluation.stability_checks]\nrotatons = []",
        "unknown configuration field 'evaluation.stability_checks.rotatons'; "
        "did you mean 'evaluation.stability_checks.rotations'?",
    ),
    (
        "[evaluation]\npoles = 'x'",
        "evaluation.poles must be one of 'all', 'none', 'uv', 'ir'; got 'x'",
    ),
    ("[evaluation]\npoles = 1", "evaluation.poles must be a string"),
    (
        "[evaluation]\nreduction = 'x'",
        "evaluation.reduction must be one of 'auto', 'ninjars', 'tirust'; got 'x'",
    ),
    (
        "[evaluation]\narithmetic = 'quad'",
        "evaluation.arithmetic must be one of 'double', 'double-double', "
        "'arbitrary'; got 'quad'",
    ),
    (
        "[evaluation]\non_failure = 'x'",
        "evaluation.on_failure must be one of 'best-effort', 'return', 'raise'; "
        "got 'x'",
    ),
    ("[evaluation]\nprecision = 0", "evaluation.precision must be a positive integer"),
    ("[evaluation]\nprecision = 1.5", "evaluation.precision must be an integer"),
    (
        "[evaluation]\nprecision = 4294967296",
        "evaluation.precision must be at most 4294967295; got 4294967296",
    ),
    (
        "[evaluation.stability_checks]\nrotations = ['x:a:b']",
        "evaluation.stability_checks: Invalid literal for Fraction: 'a'",
    ),
    ("[evaluation]\nthreads = 0", "evaluation.threads must be a positive integer"),
    ("[evaluation]\nthreads = 'x'", "evaluation.threads must be an integer"),
    (
        "[evaluation]\nreduction_target = 'x'",
        "evaluation.reduction_target must be a number",
    ),
    ("[evaluation]\nlane_rescue = 1", "evaluation.lane_rescue must be true or false"),
    (
        "[evaluation]\nreduction_unchecked = 'yes'",
        "evaluation.reduction_unchecked must be true or false",
    ),
    (
        "[evaluation]\nkinematics_improvement_binary64 = 0",
        "evaluation.kinematics_improvement_binary64 must be true or false",
    ),
    (
        "[evaluation]\nkinematics_improvement = 3",
        "evaluation.kinematics_improvement must be a string",
    ),
    (
        "[evaluation]\nstability_checks = 1",
        "configuration section 'evaluation.stability_checks' must be a table",
    ),
    (
        "[evaluation.stability_checks]\nrotations = 'x'",
        "evaluation.stability_checks.rotations must be a list of strings",
    ),
    (
        "[evaluation.stability_checks]\nrotations = ['']",
        "evaluation.stability_checks.rotations must contain non-empty strings",
    ),
    (
        "[evaluation.stability_checks]\nrotation_count = -1",
        "evaluation.stability_checks.rotation_count must be an integer >= 0",
    ),
    (
        "[evaluation.stability_checks]\nrotations = ['x:3/5:4/5']\nrotation_count = 1",
        "evaluation.stability_checks takes rotations or rotation_count",
    ),
    (
        "[evaluation.stability_checks]\nrescaled = true",
        "evaluation.stability_checks: rescaled is a variant of the "
        "reduction-direction check",
    ),
    (
        "[evaluation.stability_checks]\nrotations = ['x:1:0']",
        "evaluation.stability_checks: the identity is no rotation check",
    ),
]


def _native():
    from pyamplicol.loops.native import load_loop_native

    return load_loop_native()


def _native_options(text: str) -> dict:
    return json.loads(_native().runtime_options(text))


def _python_options(text: str) -> dict:
    """The Python configuration's reading of ``text`` in the native reader's
    record form (absent keys None)."""

    from pyamplicol.config.runtime_options import RUNTIME_OPTIONS, read_runtime_options

    found = read_runtime_options(text)
    record = {name: found.get(name) for name in RUNTIME_OPTIONS}
    checks = found.get("stability_checks")
    if "stability_checks" in found:
        record["stability_checks"] = (
            None
            if checks is None
            else (checks.rotations, checks.reduction_direction, checks.rescaled)
        )
    return record


def test_both_readers_know_the_same_keys():
    """The native reader's keys are the run card's ``[evaluation]`` keys, and
    its runtime options are ``config.runtime_options.RUNTIME_OPTIONS``."""

    from pyamplicol.config import FIELD_REGISTRY
    from pyamplicol.config.runtime_options import RUNTIME_OPTIONS

    keys = json.loads(_native().runtime_options("", keys=True))
    sections = {
        "evaluation.stability",
        "evaluation.stability_checks",
        "evaluation.phase_space",
    }
    evaluation = {
        path
        for path in FIELD_REGISTRY
        if path.startswith("evaluation.") and path not in sections
    }
    assert set(keys["runtime"]) | set(keys["other"]) == evaluation
    runtime = {path.split(".")[1] for path in keys["runtime"]}
    assert runtime == set(RUNTIME_OPTIONS)


@pytest.mark.parametrize(
    "text",
    [
        EVERY_KEY,
        "",
        "[evaluation]",
        "[evaluation]\n",
        "[evaluation]\nprecision = 30",
        "[evaluation]\narithmetic = 'arbitrary'\nprecision = 40\npoles = 'uv'",
        "[evaluation]\nreduction_target = 1\nkinematics_improvement = 'off'",
        "[evaluation.stability_checks]\nrotations = ['x:3/5:4/5', 'z:0:1']",
        "[evaluation.stability_checks]\n",
    ],
)
def test_both_readers_read_the_same_values(text):
    native = _native_options(text)
    python = _python_options(text)
    for name, value in python.items():
        if name == "stability_checks":
            checks = native["stability_checks"]
            if checks is None:
                assert value is None or "stability_checks" not in text
            continue
        assert native[name] == value, (name, native[name], value)
    if "rotation_count = 1" in text:
        assert native["stability_checks"]["rotation_count"] == 1


@pytest.mark.parametrize(("text", "message"), REFUSALS)
def test_both_readers_refuse_with_the_same_text(text, message):
    from pyamplicol.config import ConfigurationError
    from pyamplicol.config.runtime_options import read_runtime_options

    with pytest.raises(ValueError) as native:
        _native().runtime_options(text)
    with pytest.raises(ConfigurationError) as python:
        read_runtime_options(text)
    assert str(native.value) == message
    assert str(python.value) == message


# ------------------------------------------------------------------ the SDK


def _sdk_types(sdk):
    sdk.rusticol_virtual_set_options.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    sdk.rusticol_virtual_set_threads.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    sdk.rusticol_virtual_open_with_options.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    sdk.rusticol_virtual_options_json.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    return sdk


def _open(sdk, artifact, options: str | None = None):
    handle = ctypes.c_void_p()
    status = sdk.rusticol_virtual_open_with_options(
        str(artifact).encode(),
        b"ggtt",
        None,
        None,
        None if options is None else options.encode(),
        ctypes.byref(handle),
    )
    assert status == 0, _error(sdk)
    return handle


def _state(sdk, handle) -> dict:
    needed = ctypes.c_size_t()
    assert sdk.rusticol_virtual_options_json(handle, None, 0, ctypes.byref(needed)) == 0
    buffer = ctypes.create_string_buffer(needed.value)
    assert (
        sdk.rusticol_virtual_options_json(
            handle, buffer, needed.value, ctypes.byref(needed)
        )
        == 0
    )
    return json.loads(buffer.value.decode())


def _floats(points):
    return [[[float(c) for c in leg] for leg in point] for point in points]


def test_the_sdk_applies_every_key_of_a_toml_and_reads_it_back(ggtt):  # noqa: F811
    sdk = _sdk_types(_sdk())
    handle = _open(sdk, ggtt)
    try:
        default = _state(sdk, handle)
        assert default["threads"] is None and default["lane_rescue"] is True
        assert sdk.rusticol_virtual_set_options(handle, EVERY_KEY.encode()) == 0, (
            _error(sdk)
        )
        state = _state(sdk, handle)
        assert state == {
            "arithmetic": "double-double",
            "precision": None,
            "poles": "none",
            "on_failure": "return",
            "reduction": "ninjars",
            "reduction_target": 1e-7,
            "reduction_unchecked": True,
            "lane_rescue": False,
            "kinematics_improvement": "projection",
            "kinematics_improvement_binary64": True,
            "stability_checks": {
                "rotations": state["stability_checks"]["rotations"],
                "reduction_direction": True,
                "rescaled": True,
                "cross_backend": False,
            },
            "threads": 2,
            "threads_requested": None,
        }
        assert len(state["stability_checks"]["rotations"]) == 1
        # Atomic: a TOML with one bad key changes nothing.
        bad = b"[evaluation]\nreduction = 'auto'\nthreads = 4\npoles = 'x'\n"
        assert sdk.rusticol_virtual_set_options(handle, bad) == STATUS_INVALID_ARGUMENT
        assert _error(sdk) == (
            "evaluation.poles must be one of 'all', 'none', 'uv', 'ir'; got 'x'"
        )
        assert _state(sdk, handle) == state
        # A setter after the TOML overrides its key; the others stay.
        assert sdk.rusticol_virtual_set_reduction(handle, b"auto") == 0
        assert sdk.rusticol_virtual_set_threads(handle, 1) == 0
        after = _state(sdk, handle)
        assert after == {**state, "reduction": "auto", "threads": 1}
        # A TOML key absent keeps the setting.
        assert (
            sdk.rusticol_virtual_set_options(handle, b"[evaluation]\nprecision = 30\n")
            == 0
        )
        assert _state(sdk, handle) == {
            **after,
            "arithmetic": "arbitrary",
            "precision": 30,
        }
        assert sdk.rusticol_virtual_set_threads(handle, 0) == 0
        assert _state(sdk, handle)["threads"] is None
    finally:
        sdk.rusticol_virtual_free(handle)


def test_the_sdk_sets_every_key_on_its_own(ggtt):  # noqa: F811
    """Each runtime option by its own setter (``precision`` through
    ``rusticol_virtual_set_arithmetic``, the stability-check fields through
    ``rusticol_virtual_set_stability_checks``), read back."""

    sdk = _sdk_types(_sdk())
    sdk.rusticol_virtual_set_reduction_target.argtypes = [
        ctypes.c_void_p,
        ctypes.c_double,
    ]
    sdk.rusticol_virtual_set_stability_checks.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_char_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
    ]
    handle = _open(sdk, ggtt)
    try:
        calls = [
            (sdk.rusticol_virtual_set_reduction, (b"tirust",), {"reduction": "tirust"}),
            (
                sdk.rusticol_virtual_set_reduction_target,
                (1e-6,),
                {"reduction_target": 1e-6},
            ),
            (
                sdk.rusticol_virtual_set_reduction_unchecked,
                (1,),
                {"reduction_unchecked": True},
            ),
            (sdk.rusticol_virtual_set_lane_rescue, (0,), {"lane_rescue": False}),
            (
                sdk.rusticol_virtual_set_arithmetic,
                (b"arbitrary", ctypes.c_uint32(40)),
                {"arithmetic": "arbitrary", "precision": 40},
            ),
            (sdk.rusticol_virtual_set_poles, (b"uv",), {"poles": "uv"}),
            (sdk.rusticol_virtual_set_on_failure, (b"raise",), {"on_failure": "raise"}),
            (
                sdk.rusticol_virtual_set_kinematics_improvement,
                (b"rescaling",),
                {"kinematics_improvement": "rescaling"},
            ),
            (
                sdk.rusticol_virtual_set_kinematics_improvement_binary64,
                (1,),
                {"kinematics_improvement_binary64": True},
            ),
            # A count every host has: above the usable cores the read-back is
            # the reduced count (``threads_requested`` holds the request).
            (sdk.rusticol_virtual_set_threads, (2,), {"threads": 2}),
        ]
        expected = _state(sdk, handle)
        for setter, arguments, changed in calls:
            assert setter(handle, *arguments) == 0, _error(sdk)
            expected.update(changed)
            assert _state(sdk, handle) == expected, setter
        assert (
            sdk.rusticol_virtual_set_stability_checks(
                handle, 0, b'["x:3/5:4/5"]', 1, 1, 0
            )
            == 0
        )
        checks = _state(sdk, handle)["stability_checks"]
        # The exact rotation about x with cos 3/5, sin 4/5.
        (matrix,) = checks["rotations"]
        assert sorted(c for row in matrix for c in row) == sorted(
            ["1", "0", "0", "0", "3/5", "-4/5", "0", "4/5", "3/5"]
        ), matrix
        assert checks["reduction_direction"] and checks["rescaled"]
    finally:
        sdk.rusticol_virtual_free(handle)


def test_the_same_toml_gives_the_same_values_in_python_and_the_sdk(ggtt):  # noqa: F811
    """Bitwise: ``Runtime.load(options=TOML)`` and ``rusticol_virtual_
    set_options`` of the same TOML (a binary64 NinjaRS call with a
    rotation check and two threads); ``inspect`` reads the options back."""

    from pyamplicol import Runtime

    text = (
        "[evaluation]\nreduction = 'ninjars'\npoles = 'none'\non_failure = 'return'\n"
        "threads = 2\n[evaluation.stability_checks]\nrotation_count = 1\n"
    )
    runtime = Runtime.load(ggtt, process="ggtt", options=text)
    python = runtime.evaluate(POINTS, mu_r2=SCALES)
    assert python.stability is not None and all(e is not None for e in python.stability)
    record = runtime.inspect()["runtime_options"]
    assert record["threads"] == 2 and record["poles"] == "none"
    assert record["reduction"] == "ninjars" and record["on_failure"] == "return"
    sdk = _sdk_types(_sdk())
    handle = _open(sdk, ggtt, text)
    try:
        found = _evaluate(sdk, handle, _floats(POINTS), [float(s) for s in SCALES])
    finally:
        sdk.rusticol_virtual_free(handle)
    assert _same(found["virtual"], _python_flat(python))


def test_python_takes_every_key_as_a_keyword_and_reads_it_back(ggtt):  # noqa: F811
    from pyamplicol import Runtime
    from pyamplicol.api.virtual_results import VirtualStabilityChecks

    runtime = Runtime.load(
        ggtt,
        process="ggtt",
        reduction="tirust",
        reduction_target=1e-6,
        reduction_unchecked=True,
        lane_rescue=False,
        arithmetic="arbitrary",
        precision=40,
        kinematics_improvement="rescaling",
        kinematics_improvement_binary64=True,
        threads=2,
    )
    assert runtime.inspect()["runtime_options"] == {
        "arithmetic": "arbitrary",
        "precision": 40,
        "reduction": "tirust",
        "reduction_target": 1e-6,
        "reduction_unchecked": True,
        "lane_rescue": False,
        "kinematics_improvement": "rescaling",
        "kinematics_improvement_binary64": True,
        "threads": 2,
        "threads_requested": None,
        "poles": "all",
        "on_failure": "best-effort",
        "stability_checks": None,
    }
    # The call options are keywords of every call; a keyword overrides the
    # runtime options' value, which overrides the default.
    checks = VirtualStabilityChecks(1)
    loaded = Runtime.load(
        ggtt,
        process="ggtt",
        options="[evaluation]\nthreads = 1\nlane_rescue = false\npoles = 'none'\n",
        lane_rescue=True,
    )
    record = loaded.inspect()["runtime_options"]
    assert record["threads"] == 1 and record["lane_rescue"] is True
    assert record["poles"] == "none"
    plain = loaded.evaluate(POINTS, mu_r2=SCALES)
    assert all(value[1] is None for point in plain.values for value in point)
    both = loaded.evaluate(POINTS, mu_r2=SCALES, poles="all", stability_checks=checks)
    assert all(value[1] is not None for point in both.values for value in point)
    assert both.stability is not None


def test_values_do_not_depend_on_the_thread_count(ggtt):  # noqa: F811
    from pyamplicol import Runtime

    results = {}
    for threads in (None, 1, 2, 8):
        runtime = Runtime.load(ggtt, process="ggtt", threads=threads)
        for arithmetic in ("double", "double-double"):
            evaluation = runtime.evaluate(
                POINTS, mu_r2=SCALES, arithmetic=arithmetic, on_failure="return"
            )
            results[threads, arithmetic] = [
                None
                if point is None
                else [tuple(str(c) for c in laurent) for laurent in point]
                for point in evaluation.values
            ]
    for (threads, arithmetic), values in results.items():
        assert values == results[None, arithmetic], (threads, arithmetic)


# The thread-count children count every thread of their process: numpy's
# BLAS pool (one thread per core, up to 64, started at import) is kept out.
_ONE_BLAS_THREAD = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}

THREADS_SCRIPT = r"""
import json, sys
from pyamplicol import Runtime

def threads():
    with open("/proc/self/status") as status:
        return int(next(l for l in status if l.startswith("Threads:")).split()[1])

artifact, count = sys.argv[1], sys.argv[2]
point = [["180.91015625", "0", "0", "180.91015625"],
         ["180.91015625", "0", "0", "-180.91015625"],
         ["180.91015625", "0", "31.74609375", "42.328125"],
         ["180.91015625", "0", "-31.74609375", "-42.328125"]]
before = threads()
kwargs = {} if count == "none" else {"threads": int(count)}
runtimes = [Runtime.load(artifact, process="ggtt", **kwargs)]
if len(sys.argv) > 3:
    runtimes.append(Runtime.load(artifact, process="ggtt", threads=int(sys.argv[3])))
for runtime in runtimes:
    runtime.evaluate([point] * 32, mu_r2=["1000000"] * 32)
    # An extended carrier too: the kinematics improvement and the generic
    # reduction engines.
    runtime.evaluate([point] * 2, mu_r2=["1000000"] * 2, arithmetic="double-double")
counts = [r.inspect()["runtime_options"]["threads"] for r in runtimes]
print(before, threads(), json.dumps(counts))
"""


def _thread_run(tmp_path, env_threads: str, *counts: str):
    script = tmp_path / "threads.py"
    script.write_text(THREADS_SCRIPT)
    run = subprocess.run(
        [sys.executable, str(script), *counts],
        capture_output=True,
        text=True,
        timeout=900,
        env={**os.environ, "RAYON_NUM_THREADS": env_threads, **_ONE_BLAS_THREAD},
        check=False,
    )
    assert run.returncode == 0, run.stderr[-4000:]
    before, after, recorded = run.stdout.strip().splitlines()[-1].split(" ", 2)
    return int(before), int(after), json.loads(recorded)


@pytest.mark.skipif(not Path("/proc/self/status").is_file(), reason="needs procfs")
def test_the_thread_bound_and_its_fallback(ggtt, tmp_path):  # noqa: F811
    """In fresh processes with ``RAYON_NUM_THREADS=16``: a runtime loaded
    with ``threads=1`` keeps its load and evaluation on its own pools (no
    pool of 16 workers starts: rayon's global pool never stops its
    workers); unset, the evaluation runs on pools of ``RAYON_NUM_THREADS``
    (the fallback, at least 16 workers alive); two runtimes keep their own
    counts."""

    before, after, recorded = _thread_run(tmp_path, "16", str(ggtt), "1")
    assert recorded == [1]
    assert after - before < 16, (before, after)
    before, after, recorded = _thread_run(tmp_path, "16", str(ggtt), "none")
    assert recorded == [None]
    assert after - before >= 16, (before, after)
    before, after, recorded = _thread_run(tmp_path, "16", str(ggtt), "1", "2")
    assert recorded == [1, 2]
    assert after - before < 16, (before, after)


def test_the_example_runtime_options_file_reads_alike():
    """``examples/data/runtime_options.toml`` (the documented example) is read
    identically by both readers."""

    text = (
        Path(__file__).resolve().parents[2] / "examples/data/runtime_options.toml"
    ).read_text(encoding="utf-8")
    native = _native_options(text)
    python = _python_options(text)
    assert native["threads"] == python["threads"] == 4
    for name in (
        "reduction",
        "reduction_target",
        "lane_rescue",
        "arithmetic",
        "poles",
        "on_failure",
    ):
        assert native[name] == python[name], name
    assert native["stability_checks"]["rotation_count"] == 1
    assert python["stability_checks"] is not None


# ------------------------------------------------------------ thread clamp

HUGE = 9223372036854775807


def test_a_huge_thread_count_is_clamped_to_the_usable_cores(ggtt):  # noqa: F811
    """``threads`` above the cores the process may use is clamped, not
    refused (the native ``effective_threads``): through the C entry point and
    through Python the call returns at once with the effective count, and the
    read-backs keep the requested one."""

    from pyamplicol import Runtime

    available = _native().effective_threads(HUGE)
    assert 1 <= available <= len(os.sched_getaffinity(0))
    sdk = _sdk_types(_sdk())
    handle = _open(sdk, ggtt, f"[evaluation]\nthreads = {HUGE}\n")
    try:
        state = _state(sdk, handle)
        assert (state["threads"], state["threads_requested"]) == (available, HUGE)
        assert sdk.rusticol_virtual_set_threads(handle, 2**32 - 1) == 0, _error(sdk)
        state = _state(sdk, handle)
        assert (state["threads"], state["threads_requested"]) == (available, 2**32 - 1)
        found = _evaluate(sdk, handle, _floats(POINTS), [float(s) for s in SCALES])
        assert found["statuses"]
        assert sdk.rusticol_virtual_set_threads(handle, 1) == 0
        state = _state(sdk, handle)
        assert (state["threads"], state["threads_requested"]) == (1, None)
    finally:
        sdk.rusticol_virtual_free(handle)
    runtime = Runtime.load(ggtt, process="ggtt", threads=10**30)
    record = runtime.inspect()["runtime_options"]
    assert (record["threads"], record["threads_requested"]) == (available, 10**30)
    runtime.evaluate(POINTS, mu_r2=SCALES)


AFFINITY_SCRIPT = r"""
import ctypes, json, os, sys
from pyamplicol import Runtime
from tests.unit.test_loop_v2_sdk_extras import _sdk
from tests.unit.test_loop_v2_sdk_induced import _evaluate

def threads():
    with open("/proc/self/status") as status:
        return int(next(l for l in status if l.startswith("Threads:")).split()[1])

artifact, mode = sys.argv[1], sys.argv[2]
count, limit = int(sys.argv[3]), int(sys.argv[4])
point = [[180.91015625, 0, 0, 180.91015625], [180.91015625, 0, 0, -180.91015625],
         [180.91015625, 0, 31.74609375, 42.328125],
         [180.91015625, 0, -31.74609375, -42.328125]]
before = threads()
if mode == "python":
    runtime = Runtime.load(artifact, process="ggtt", threads=count)
    runtime.evaluate([[[str(c) for c in p] for p in point]] * 8, mu_r2=["1000000"] * 8)
    record = runtime.inspect()["runtime_options"]
    effective = record["threads"]
else:
    sdk = _sdk()
    sdk.rusticol_virtual_open_with_options.argtypes = [ctypes.c_char_p] * 5 + [
        ctypes.POINTER(ctypes.c_void_p)]
    handle = ctypes.c_void_p()
    toml = f"[evaluation]\nthreads = {count}\n".encode()
    assert sdk.rusticol_virtual_open_with_options(
        artifact.encode(), b"ggtt", None, None, toml, ctypes.byref(handle)) == 0
    _evaluate(sdk, handle, [point] * 8, [1e6] * 8)
    effective = None
# Threads alive once the call's transient ones have exited: a pool, of the
# bound or of RAYON_NUM_THREADS, never exits. The transient ones are the
# threads of a decode pool the load replaced (docs "Threads") and
# Symbolica's licence checks, each of which waits for its server's answer
# (milliseconds on a near host, longer on a CI runner). The count is taken
# once it has been steady for 0.3 s, and, while it is above `limit`, not
# before a minute has passed.
import time
start = time.monotonic()
after, since = threads(), start
while True:
    steady = time.monotonic() - since >= 0.3
    waited = time.monotonic() - start
    if steady and (after - before <= limit or waited >= 60.0):
        break
    time.sleep(0.05)
    now = threads()
    if now != after:
        after, since = now, time.monotonic()
print(json.dumps([before, after, effective, len(os.sched_getaffinity(0))]))
"""


def _affinity_run(tmp_path, ggtt, mode, count, limit, cores=None):  # noqa: F811
    """The child's thread counts before and after one call; `limit` is the
    most threads the caller accepts the call to leave alive."""

    script = tmp_path / "affinity.py"
    script.write_text(AFFINITY_SCRIPT)
    command = [sys.executable, str(script), str(ggtt), mode, str(count), str(limit)]
    if cores is not None:
        command = ["taskset", "-c", cores, *command]
    root = Path(__file__).resolve().parents[2]
    # The script imports the test helpers (``tests.unit``): the checkout on
    # the child's path (CI's PYTHONPATH holds src/ only).
    path = os.pathsep.join(filter(None, (str(root), os.environ.get("PYTHONPATH"))))
    run = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=900,
        env={
            **os.environ,
            "RAYON_NUM_THREADS": "16",
            "PYTHONPATH": path,
            **_ONE_BLAS_THREAD,
        },
        cwd=root,
        check=False,
    )
    assert run.returncode == 0, run.stderr[-4000:]
    return json.loads(run.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(not Path("/proc/self/status").is_file(), reason="needs procfs")
@pytest.mark.skipif(
    len(os.sched_getaffinity(0)) < 2 or shutil.which("taskset") is None,
    reason="needs two cores and taskset",
)
def test_threads_above_the_affinity_mask_give_its_cores(ggtt, tmp_path):  # noqa: F811
    """Under ``taskset -c 0-1`` a runtime asking for 64 threads (and a
    handle asking for 2^63-1) runs on 2: the read-back says 2 and the
    threads alive after a binary64 call are at most the bound's pool
    (with ``RAYON_NUM_THREADS=16``, unbounded pools would add 16)."""

    first = sorted(os.sched_getaffinity(0))[:2]
    cores = f"{first[0]},{first[1]}"
    before, after, effective, mask = _affinity_run(
        tmp_path, ggtt, "python", 64, 4, cores
    )
    assert (effective, mask) == (2, 2)
    assert after - before <= 2 * 2, (before, after)
    before, after, _effective, mask = _affinity_run(
        tmp_path, ggtt, "sdk", HUGE, 4, cores
    )
    assert mask == 2
    assert after - before <= 2 * 2, (before, after)


@pytest.mark.skipif(not Path("/proc/self/status").is_file(), reason="needs procfs")
def test_the_sdk_thread_bound_counted(ggtt, tmp_path):  # noqa: F811
    """With ``RAYON_NUM_THREADS=16``, an SDK handle opened with ``threads =
    1`` or ``2`` evaluates on its own pool of that size (the threads alive
    after the call: at most the pool, no pool of 16)."""

    for count in (1, 2):
        before, after, _effective, _mask = _affinity_run(
            tmp_path, ggtt, "sdk", count, count + 1
        )
        assert 1 <= after - before <= count + 1, (count, before, after)


def test_evaluate_threads_reaches_the_runtime(ggtt, monkeypatch, capsys):  # noqa: F811
    """``pyamplicol evaluate ARTIFACT --threads 2``: the CLI loads the
    runtime with ``threads=2`` (its read-back says 2)."""

    from pyamplicol.api import services
    from pyamplicol.cli.main import main

    loaded = []
    original = services.Runtime.load.__func__

    def load(cls, *args, **kwargs):
        runtime = original(cls, *args, **kwargs)
        loaded.append(
            (kwargs.get("threads"), runtime.inspect()["runtime_options"]["threads"])
        )
        return runtime

    monkeypatch.setattr(services.Runtime, "load", classmethod(load))
    code = main(
        [
            "evaluate",
            str(ggtt),
            "--process",
            "ggtt",
            "--threads",
            "2",
            "--mu-r2",
            "1000000",
            "--poles",
            "none",
            "--json",
        ]
    )
    capsys.readouterr()
    assert code == 0
    assert loaded == [(2, 2)]


def test_a_tree_runtime_refuses_virtual_options_at_load():
    """As the tree C API's ``rusticol_runtime_set_options``: a virtual call
    option named in the TOML is refused, at load, by name; the kinematics
    keys are a tree runtime's."""

    import pyamplicol
    from pyamplicol import Runtime
    from pyamplicol.api.errors import CompatibilityError

    tree = Path(pyamplicol.__file__).parent / "assets/selftest/portable-64le/artifact"
    with pytest.raises(
        CompatibilityError, match="poles are options of virtual runtimes"
    ):
        Runtime.load(tree, options="[evaluation]\npoles = 'all'\n")
    runtime = Runtime.load(
        tree, options="[evaluation]\nkinematics_improvement_binary64 = false\n"
    )
    assert runtime.contribution_kind == "tree"


def test_a_one_line_options_string_is_an_existing_file_or_toml(tmp_path, monkeypatch):
    """A one-line string is a path when it names an existing file (with or
    without "=" in its name), else a TOML text: ``[evaluation]`` is an empty
    table in both readers (no option set); a string that is neither (a
    mistyped path, a bare word) is refused naming both readings. The native
    readers take TOML texts only (a file through ``set_options_file``): they
    read the same file's text alike and refuse the bare word as TOML."""

    from pyamplicol.config import ConfigurationError
    from pyamplicol.config.runtime_options import read_runtime_options

    monkeypatch.chdir(tmp_path)
    both = "not an existing file and not valid TOML"
    for text in ("[evaluation]", "[evaluation]\n"):
        assert read_runtime_options(text) == {}
        assert set(_native_options(text).values()) == {None}
    for text in (str(tmp_path / "no" / "opts.toml"), "opts", "x=y.toml"):
        with pytest.raises(ConfigurationError, match=both):
            read_runtime_options(text)
    with pytest.raises(ValueError, match="invalid TOML"):
        _native().runtime_options("opts")
    toml = "[evaluation]\nthreads = 3\n"
    for name in ("opts", "a=b.toml"):
        (tmp_path / name).write_text(toml)
        assert read_runtime_options(name) == {"threads": 3}
        assert read_runtime_options(str(tmp_path / name)) == {"threads": 3}
    assert _native_options(toml)["threads"] == 3
    assert read_runtime_options("[evaluation]\nthreads = 2") == {"threads": 2}
    assert read_runtime_options("threads = 2") == {}


def test_an_integer_beyond_64_bits_is_invalid_toml_in_both_readers():
    """TOML integers are signed 64-bit: the native reader (the toml crate)
    refuses a larger one as TOML, and so does Python's reader (tomllib reads
    it); within the range, a count above the usable cores is clamped."""

    from pyamplicol.config import ConfigurationError
    from pyamplicol.config.runtime_options import read_runtime_options

    text = "[evaluation]\nthreads = 1000000000000000000000\n"
    with pytest.raises(ValueError, match="invalid TOML"):
        _native().runtime_options(text)
    with pytest.raises(
        ConfigurationError, match=r"invalid TOML: the integer .* at evaluation\.threads"
    ):
        read_runtime_options(text)
    assert read_runtime_options(f"[evaluation]\nthreads = {2**63 - 1}\n") == {
        "threads": 2**63 - 1
    }


def test_inspect_gives_the_stability_checks_as_plain_data(ggtt):  # noqa: F811
    from pyamplicol import Runtime

    runtime = Runtime.load(
        ggtt,
        process="ggtt",
        options="[evaluation.stability_checks]\nrotations = ['x:3/5:4/5']\n"
        "reduction_direction = true\n",
    )
    record = runtime.inspect()["runtime_options"]["stability_checks"]
    assert json.loads(json.dumps(record)) == record
    assert record["reduction_direction"] is True and record["rescaled"] is False
    (matrix,) = record["rotations"]
    assert sorted(c for row in matrix for c in row) == sorted(
        ["1", "0", "0", "0", "3/5", "-4/5", "0", "4/5", "3/5"]
    )
