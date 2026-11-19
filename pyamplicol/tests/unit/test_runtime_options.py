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
import subprocess
import sys
from pathlib import Path

import pytest

from tests.unit.test_loop_v2_sdk_extras import _error, _sdk
from tests.unit.test_loop_v2_sdk_induced import _evaluate, _python_flat, _same
from tests.unit.test_loop_v2_sdk_parity import POINT_173, ggtt  # noqa: F401

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
    ("[evaluation]\nprecision = 0", "evaluation.precision must be an integer positive"),
    ("[evaluation]\nprecision = 1.5", "evaluation.precision must be an integer"),
    ("[evaluation]\nthreads = 0", "evaluation.threads must be an integer positive"),
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
        assert sdk.rusticol_virtual_set_threads(handle, 3) == 0
        after = _state(sdk, handle)
        assert after == {**state, "reduction": "auto", "threads": 3}
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
            (sdk.rusticol_virtual_set_threads, (5,), {"threads": 5}),
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
        threads=3,
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
        "threads": 3,
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


THREADS_SCRIPT = r"""
import sys, time
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
print(before, threads(), [r.inspect()["runtime_options"]["threads"] for r in runtimes])
"""


def _thread_run(tmp_path, env_threads: str, *counts: str):
    script = tmp_path / "threads.py"
    script.write_text(THREADS_SCRIPT)
    run = subprocess.run(
        [sys.executable, str(script), *counts],
        capture_output=True,
        text=True,
        timeout=900,
        env={**os.environ, "RAYON_NUM_THREADS": env_threads},
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
