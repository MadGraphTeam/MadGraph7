# SPDX-License-Identifier: 0BSD
"""The plan v4 SDK extras equal the Python API in every SDK language (M-FINAL
MF-SDK).

Two processes generated with their observable columns (g g > t t~, with WT and
MB fixed to zero, and u u~ > d d~ g; full colour, declared colour histories
and spin classes) are evaluated through the Python API (``Runtime.evaluate``,
``evaluate_correlated_many`` with one request per stored history and the
same spin vectors in each, ``evaluate_resolved``) and through the SDK's C
ABI, its C++ and Fortran wrappers and the pyAmpliCol UMAMI virtual extension
(``rust/loop-reduction/sdk/examples/extras_v4*``; the safe Rust ``Runtime``
in the opt-in nightly ``test_the_rust_sdk_extras_follow_the_python_api``):

* binary64 (two points per batch): the ordinary virtual, the colour-correlated
  virtuals of every history (in all, "uv" and "none" pole selections), the
  spin-correlated ones (a declared spin class, literal complex vectors) and
  the helicity-resolved ones (in all and "ir") are bitwise the Python API's;
* double-double and arbitrary precision (32 digits; one point): the ordinary,
  spin-correlated and resolved virtuals equal the Python API's ``Decimal``
  values exactly (the SDK's decimal literals; its doubles are their nearest
  binary64 values).

The stability rescue is the Python API's alone (user ruling Q9) and has no SDK
entry.
"""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import struct
import subprocess
from decimal import Decimal
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SDK = ROOT / "rust" / "loop-reduction" / "sdk"
UMAMI_HEADERS = ROOT / "src" / "pyamplicol" / "assets" / "api_templates" / "umami"

pytestmark = pytest.mark.loop_physics

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

# Points exact in binary64: g g > t t~ on shell at MT = 173 (E^2 - |p|^2 =
# 173^2 exactly), u u~ > d d~ g massless (integer Pythagorean momenta).
E1, P1, E2, X2, Z2 = (
    180.91015625,
    (31.74609375, 42.328125),
    332.275390625,
    201.8203125,
    199.365234375,
)
SPIN = [(0.5, 0.25), (0.75, -0.125), (0.375, 1.0), (-0.625, 0.5)]
CASES = {
    "ggtt": {
        "expression": "g g > t t~",
        "histories": [("T12", 1, 2), ("T13", 1, 3), ("T34", 3, 4)],
        "spin_classes": [[1], [1, 2]],
        "points": [
            [[E1, 0, 0, E1], [E1, 0, 0, -E1], [E1, 0, *P1], [E1, 0, -P1[0], -P1[1]]],
            [[E2, 0, 0, E2], [E2, 0, 0, -E2], [E2, X2, 0, Z2], [E2, -X2, 0, -Z2]],
        ],
        "spin": {1: SPIN, 2: [(-0.25, 0.75), (0.125, 0.5), (1.0, -0.375), (0.5, 0.25)]},
    },
    "uuddg": {
        "expression": "u u~ > d d~ g",
        "histories": [("T12", 1, 2), ("T35", 3, 5)],
        "spin_classes": [[5]],
        "points": [
            [
                [425, 0, 0, 425],
                [425, 0, 0, -425],
                [350, -300, -150, -100],
                [225, 150, -75, 150],
                [275, 150, 225, -50],
            ],
            [
                [425, 0, 0, 425],
                [425, 0, 0, -425],
                [350, -300, -150, 100],
                [275, 150, 225, 50],
                [225, 150, -75, -150],
            ],
        ],
        "spin": {5: SPIN},
    },
}
SCALES = [1.0e6, 4.9e5]
# The calls of examples/extras_v4*: (name, arithmetic, digits, poles, kind,
# points); kind 0 ordinary, 1 correlated, 2 spin-correlated, 3 resolved.
CALLS = [
    ("ordinary", "double", None, "all", 0, 2),
    ("correlated", "double", None, "all", 1, 2),
    ("correlated-spin", "double", None, "all", 2, 2),
    ("correlated-uv", "double", None, "uv", 1, 2),
    ("correlated-none", "double", None, "none", 1, 2),
    ("resolved", "double", None, "all", 3, 2),
    ("resolved-ir", "double", None, "ir", 3, 2),
    ("ordinary-dd", "double-double", None, "all", 0, 1),
    ("correlated-spin-dd", "double-double", None, "all", 2, 1),
    ("resolved-dd", "double-double", None, "all", 3, 1),
    ("ordinary-ap", "arbitrary", 32, "all", 0, 1),
    ("correlated-spin-ap", "arbitrary", 32, "all", 2, 1),
    ("resolved-ap", "arbitrary", 32, "all", 3, 1),
]


def _card(model: Path, output: Path, expression: str, name: str) -> str:
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
fixed_parameters = {{MB = "0", WT = "0"}}
{SCHEME}
[generation]
output = {json.dumps(str(output))}
emit_api_bundle = false
[generation.validation]
post_build_validation = false
samples = 1
"""


@pytest.fixture(scope="module")
def artifacts(loop_sm_compiled_model, loop_native, tmp_path_factory):
    """The two processes' plan v4 artifacts with their observable columns."""

    from pyamplicol.api import Generator, ProcessRequest, ProcessSet
    from pyamplicol.config import resolve_config
    from pyamplicol.correlators import CorrelatorConfig

    out = {}
    for name, case in CASES.items():
        root = tmp_path_factory.mktemp(name)
        card = root / "card.toml"
        card.write_text(
            _card(loop_sm_compiled_model, root / "artifact", case["expression"], name)
        )

        def emit(leg: int) -> list[dict]:
            return [{"kind": "emit-gluon", "emitter": leg, "emitted": -1}]

        correlators = CorrelatorConfig.from_json_dict(
            {
                "color_correlations": [
                    {"id": key, "bra": emit(a), "ket": emit(b)}
                    for key, a, b in case["histories"]
                ],
                "spin_correlations": case["spin_classes"],
            }
        )
        Generator(resolve_config(card)).generate(
            ProcessSet((ProcessRequest.parse(case["expression"], name=name),)),
            root / "artifact",
            correlators=correlators,
        )
        out[name] = root / "artifact"
    return out


def _laurent(value) -> list:
    return [value.finite, value.single_pole, value.double_pole]


def _flat(values, complex_layout: bool = False) -> list:
    """Python values in the SDK layout order: complex values (correlated
    outputs) as (re, im), an absent pole as None (twice in a complex
    layout)."""

    out: list = []
    for value in values:
        if value is None:
            out += [None, None] if complex_layout else [None]
        elif complex_layout:
            out += [value.real, value.imag]
        else:
            out.append(value)
    return out


@pytest.fixture(scope="module")
def references(artifacts):
    """Per process and call the Python API's Born and virtual values in the
    SDK layouts (``_flat``)."""

    from pyamplicol import Runtime
    from pyamplicol.correlators import CorrelatedRequest

    out = {}
    for name, case in CASES.items():
        runtime = Runtime.load(artifacts[name], process=name)
        histories = [item.id for item in runtime.available_color_correlations()]
        assert histories == ["born", *(h[0] for h in case["histories"])]
        points = [
            [[str(Decimal(c)) for c in p] for p in point] for point in case["points"]
        ]
        scales = [str(Decimal(s)) for s in SCALES]
        spin = {
            leg: [(str(Decimal(re)), str(Decimal(im))) for re, im in vector]
            for leg, vector in case["spin"].items()
        }
        calls = {}
        for label, arithmetic, digits, poles, kind, count in CALLS:
            options = {"mu_r2": scales[:count], "poles": poles}
            if arithmetic != "double":
                options["arithmetic"] = arithmetic
            if digits is not None:
                options["precision"] = digits
            batch = points[:count]
            if kind == 0:
                result = runtime.evaluate(batch, **options)
                born = [v for b in result.born_values for v in b]
                virtual = [x for v in result.values for lc in v for x in _laurent(lc)]
            elif kind == 3:
                result = runtime.evaluate_resolved(batch, **options)
                born = [
                    x
                    for b in result.born_values
                    for order in b
                    for row in order
                    for x in row
                ]
                virtual = [
                    x
                    for v in result.values
                    for sector in v
                    for row in sector
                    for lc in row
                    for x in _laurent(lc)
                ]
            else:
                vectors = spin if kind == 2 else {}
                found = runtime.evaluate_correlated_many(
                    batch,
                    {h: CorrelatedRequest(h, vectors) for h in histories},
                    **options,
                )
                born, virtual = [], []
                for p in range(count):
                    for h in histories:
                        born += list(found[h].born_values[p])
                        virtual += [
                            x for lc in found[h].values[p] for x in _laurent(lc)
                        ]
            complex_layout = kind in (1, 2)
            calls[label] = {
                "born": _flat(born, complex_layout),
                "virtual": _flat(virtual, complex_layout),
                "complex": complex_layout,
            }
        out[name] = calls
    return out


def _staged_sdk() -> Path:
    info = ROOT / ".artifacts" / "source-runtime" / "_build_info.json"
    if not info.is_file():
        pytest.fail(f"{info} missing: stage a native build first")
    build = Path(json.loads(info.read_text())["native_build"]["directory"])
    if not (build / "librusticol_loop_sdk.so").is_file():
        pytest.fail(f"{build} has no librusticol_loop_sdk.so")
    return build


def _compiler(variable: str, name: str) -> str:
    found = os.environ.get(variable) or shutil.which(name)
    if found is None:
        pytest.fail(f"no {name} on PATH for the SDK clients; set {variable}")
    return found


@pytest.fixture(scope="module")
def clients(tmp_path_factory):
    """The extras examples compiled against the staged build's loop SDK: C,
    C++, Fortran and UMAMI (the extension compiled with its client)."""

    build = _staged_sdk()
    root = tmp_path_factory.mktemp("extras-clients")
    examples = SDK / "examples"
    link = [f"-L{build}", "-lrusticol_loop_sdk", f"-Wl,-rpath,{build}", "-lm"]
    include = [f"-I{SDK / 'include'}"]
    cc, cxx, fc = (
        _compiler("CC", "cc"),
        _compiler("CXX", "c++"),
        _compiler("FC", "gfortran"),
    )
    out = {
        "c": root / "extras-c",
        "cpp": root / "extras-cpp",
        "fortran": root / "extras-fortran",
        "umami": root / "extras-umami",
    }
    commands = [
        [
            cc,
            "-std=c11",
            "-O2",
            *include,
            str(examples / "extras_v4.c"),
            *link,
            "-o",
            str(out["c"]),
        ],
        [
            cxx,
            "-std=c++17",
            "-O2",
            *include,
            str(examples / "extras_v4.cpp"),
            *link,
            "-o",
            str(out["cpp"]),
        ],
        [
            fc,
            "-std=f2018",
            "-O2",
            f"-J{root}",
            str(SDK / "fortran" / "rusticol_virtual.f90"),
            str(examples / "extras_v4.f90"),
            *link,
            "-o",
            str(out["fortran"]),
        ],
        [
            cc,
            "-std=c11",
            "-O2",
            *include,
            f"-I{SDK / 'umami'}",
            f"-I{UMAMI_HEADERS}",
            str(SDK / "umami" / "pyamplicol_umami_virtual_v1.c"),
            str(examples / "extras_v4_umami.c"),
            *link,
            "-o",
            str(out["umami"]),
        ],
    ]
    for command in commands:
        subprocess.run(command, check=True, cwd=root)
    return out


def _bits(value: float) -> int:
    return struct.unpack("<q", struct.pack("<d", value))[0]


def _feed(case: dict, path: Path, *, bits: bool) -> Path:
    """The examples' input: points, scales and spin vectors (C99 hexadecimal,
    or for the Fortran client IEEE bit patterns)."""

    def text(value: float) -> str:
        return str(_bits(float(value))) if bits else float(value).hex()

    lines = [f"{len(case['points'])} {len(case['points'][0])}"]
    for point, scale in zip(case["points"], SCALES, strict=True):
        lines.append(" ".join([text(c) for p in point for c in p] + [text(scale)]))
    lines.append(str(len(case["spin"])))
    for leg in sorted(case["spin"]):
        lines.append(str(leg))
        lines += [f"{text(re)} {text(im)}" for re, im in case["spin"][leg]]
    path.write_text("\n".join(lines) + "\n")
    return path


def _double(token: str) -> float:
    if token.startswith("b:"):
        return struct.unpack(">d", bytes.fromhex(token[2:]))[0]
    return float.fromhex(token)


def _parse(stdout: str) -> dict:
    found: dict = {}
    current: dict = {}
    for line in stdout.splitlines():
        kind, _, rest = line.strip().partition(" ")
        if kind == "call":
            current = found.setdefault(rest, {})
        elif kind in ("born", "virtual"):
            current[kind] = [_double(token) for token in rest.split()]
        elif kind == "decimal":
            current["decimal"] = json.loads(rest)
        elif kind == "refused":
            # A pole selection the process does not serve (loop-induced).
            current["refused"] = rest
        else:
            raise AssertionError(line)
    return found


def _decimals(values: list, complex_layout: bool) -> list:
    out: list = []
    for value in values:
        if value is None:
            out += [None, None] if complex_layout else [None]
        else:
            out += value if isinstance(value, list) else [value]
    return out


# Double-double and arbitrary-precision values of two native libraries agree to the
# double-double target (ruling R5: 1e-24 relative, measured and recorded), not
# bitwise: their summation order can differ with the host's CPU and thread count
# (Tests 42500959743 on a CI runner: 6e-28 relative on loop-induced rows, while a
# local run was bitwise). Binary64 stays bitwise.
EXTENDED_TOLERANCE = Decimal("1e-24")


def _agree(want: list, got: list) -> bool:
    """True when every entry of ``got`` is within the double-double target of
    ``want``, relative to the largest magnitude in ``want`` (a vanishing pole of a
    loop-induced row is judged against the point's finite part)."""
    scale = max((abs(Decimal(a)) for a in want if a is not None), default=Decimal(0))
    return all(
        (a is None) == (b is None)
        and (a is None or abs(Decimal(a) - Decimal(b)) <= EXTENDED_TOLERANCE * scale)
        for a, b in zip(want, got, strict=True)
    )


def _compare(name: str, expected: dict, found: dict) -> None:
    assert sorted(found) == sorted(expected), name
    for call, values in expected.items():
        got = found[call]
        assert ("decimal" in got) and (got["decimal"] is None) == (
            call.endswith(("-dd", "-ap")) is False
        ), (name, call)
        decimal = got["decimal"]
        for axis in ("born", "virtual"):
            want = values[axis]
            doubles = got[axis]
            assert len(doubles) == len(want), (name, call, axis)
            if decimal is None:
                # Binary64: bitwise (absent poles are NaNs).
                for k, (a, b) in enumerate(zip(want, doubles, strict=True)):
                    if a is None:
                        assert b != b, (name, call, axis, k)
                    else:
                        assert _bits(a) == _bits(b), (name, call, axis, k, a, b)
            else:
                literals = _decimals(decimal[axis], values["complex"])
                assert len(literals) == len(want), (name, call, axis)
                # The value to the double-double target, and the call's double is the
                # nearest double of its own literal.
                assert _agree(want, literals), (name, call, axis, want, literals)
                for k, b in enumerate(literals):
                    if b is not None:
                        assert float(Decimal(b)) == doubles[k], (name, call, axis, k)


def _run(
    client: Path, artifact: Path, name: str, feed: Path, extra: tuple = ()
) -> dict:
    run = subprocess.run(
        [str(client), str(artifact), name, str(feed), *extra],
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]
    return _parse(run.stdout)


@pytest.mark.parametrize("language", ["c", "cpp", "fortran", "umami"])
def test_sdk_extras_equal_the_python_api(
    language, artifacts, references, clients, tmp_path
):
    for name, case in CASES.items():
        feed = _feed(case, tmp_path / f"{name}.txt", bits=language == "fortran")
        found = _run(clients[language], artifacts[name], name, feed)
        _compare(f"{language}:{name}", references[name], found)


def test_sdk_capabilities_list_the_extras(artifacts):
    """The stored observables are announced: resolved and correlated outputs,
    the histories (``born`` first), the spin classes, the helicity rows and
    the three arithmetics; the correlated shape counts every history."""

    build = _staged_sdk()
    sdk = ctypes.CDLL(str(build / "librusticol_loop_sdk.so"))
    for name, case in CASES.items():
        handle = ctypes.c_void_p()
        status = sdk.rusticol_virtual_open(
            str(artifacts[name]).encode(),
            name.encode(),
            None,
            None,
            ctypes.byref(handle),
        )
        assert status == 0
        try:
            needed = ctypes.c_size_t()
            assert (
                sdk.rusticol_virtual_capabilities_json(
                    handle, None, ctypes.c_size_t(0), ctypes.byref(needed)
                )
                == 0
            )
            buffer = ctypes.create_string_buffer(needed.value)
            assert (
                sdk.rusticol_virtual_capabilities_json(
                    handle, buffer, needed, ctypes.byref(needed)
                )
                == 0
            )
            capabilities = json.loads(buffer.value)
            assert capabilities["resolved"] and capabilities["correlations"]
            assert capabilities["arithmetic"] == [
                "double",
                "double-double",
                "arbitrary",
            ]
            assert capabilities["correlation_histories"] == [
                "born",
                *(h[0] for h in case["histories"]),
            ]
            assert capabilities["spin_correlation_classes"] == case["spin_classes"]
            assert len(capabilities["helicity_rows"]) == len(
                capabilities["helicity_ids"]
            )
            # A refused arithmetic is an invalid argument.
            assert sdk.rusticol_virtual_set_arithmetic(handle, b"arbitrary", 0) == 1
            assert sdk.rusticol_virtual_set_arithmetic(handle, b"double-double", 5) == 1
            assert sdk.rusticol_virtual_set_arithmetic(handle, b"quad", 0) == 1
        finally:
            sdk.rusticol_virtual_free(handle)


@pytest.mark.loop_nightly
def test_the_rust_sdk_extras_follow_the_python_api(artifacts, references, tmp_path):
    """The safe Rust ``Runtime`` (``sdk/tests/v4_extras.rs``) on both
    processes, compared as the other languages. It builds the SDK's test in
    release into ``PYAMPLICOL_SDK_CARGO_TARGET_DIR`` (a private target
    directory; seconds on one that holds a native build of this tree made
    with ``tools/developer/build_loop_native.py``); it is not run without
    one."""

    from tools.developer.build_loop_native import _CODEGEN_DEFAULTS

    target = os.environ.get("PYAMPLICOL_SDK_CARGO_TARGET_DIR")
    if not target:
        pytest.skip(
            "opt-in: set PYAMPLICOL_SDK_CARGO_TARGET_DIR to a private cargo "
            "target directory to build and run the Rust SDK's control"
        )
    shared = os.environ.get("CARGO_TARGET_DIR")
    if shared and Path(shared).resolve() == Path(target).resolve():
        pytest.fail("PYAMPLICOL_SDK_CARGO_TARGET_DIR is the shared CARGO_TARGET_DIR")
    for name, case in CASES.items():
        feed = tmp_path / f"{name}.json"
        feed.write_text(
            json.dumps(
                {
                    "artifact": str(artifacts[name]),
                    "process": name,
                    "points": [
                        [_bits(float(c)) & (2**64 - 1) for p in point for c in p]
                        + [_bits(scale) & (2**64 - 1)]
                        for point, scale in zip(case["points"], SCALES, strict=True)
                    ],
                    "spin_legs": sorted(case["spin"]),
                    "spin": [
                        [
                            _bits(float(x)) & (2**64 - 1)
                            for pair in case["spin"][leg]
                            for x in pair
                        ]
                        for leg in sorted(case["spin"])
                    ],
                }
            )
        )
        output = tmp_path / f"{name}.out"
        env = {k: v for k, v in os.environ.items() if k != "CARGO_TARGET_DIR"}
        env = {
            **_CODEGEN_DEFAULTS,
            **env,
            "PYAMPLICOL_V4_EXTRAS_FEED": str(feed),
            "PYAMPLICOL_V4_EXTRAS_OUTPUT": str(output),
        }
        run = subprocess.run(
            [
                "cargo",
                "test",
                "--locked",
                "--release",
                "--target-dir",
                target,
                "-p",
                "rusticol-loop-sdk",
                "--test",
                "v4_extras",
                "--",
                "--ignored",
            ],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=7200,
            check=False,
        )
        assert run.returncode == 0, run.stdout[-4000:] + run.stderr[-4000:]
        _compare(f"rust:{name}", references[name], _parse(output.read_text()))


# ------------------------------------------------- C ABI through ctypes
#
# The SDK-only paths the language clients above do not take (mf-small3):
# the double-double and arbitrary-precision virtuals in the "uv" and "ir"
# pole selections (the companion of the certified-scaleless families, whose
# lanes the Python API reads at the working digits like every other lane),
# and the leading-colour resolved rows (per Born helicity row and Born basis
# element). The C ABI is the Rust SDK's (every wrapper calls it).


class _Request(ctypes.Structure):
    _fields_ = [
        ("momenta", ctypes.POINTER(ctypes.c_double)),
        ("momentum_count", ctypes.c_size_t),
        ("point_count", ctypes.c_size_t),
        ("point_ids", ctypes.POINTER(ctypes.c_uint64)),
        ("mu_r2", ctypes.POINTER(ctypes.c_double)),
        ("scale_count", ctypes.c_size_t),
        ("resolved", ctypes.c_uint32),
        ("r2_strategy", ctypes.c_char_p),
    ]


class _Axes(ctypes.Structure):
    _fields_ = [
        *(
            (name, ctypes.c_size_t)
            for name in (
                "external_count",
                "born_orders",
                "virtual_orders",
                "helicities",
                "born_color_components",
                "virtual_color_components",
                "epsilon_count",
            )
        ),
        ("resolved_available", ctypes.c_uint32),
    ]


class _ResultShape(ctypes.Structure):
    _fields_ = [
        ("requested_points", ctypes.c_size_t),
        ("successful_points", ctypes.c_size_t),
        ("born_values", ctypes.c_size_t),
        ("virtual_values", ctypes.c_size_t),
        ("resolved", ctypes.c_uint32),
        ("axes", _Axes),
    ]


class _Sources(ctypes.Structure):
    _fields_ = [
        ("legs", ctypes.POINTER(ctypes.c_size_t)),
        ("leg_count", ctypes.c_size_t),
        ("values", ctypes.POINTER(ctypes.c_double)),
        ("value_count", ctypes.c_size_t),
    ]


def _sdk():
    sdk = ctypes.CDLL(str(_staged_sdk() / "librusticol_loop_sdk.so"))
    sdk.rusticol_virtual_last_error_message.restype = ctypes.c_char_p
    return sdk


def _error(sdk) -> str:
    message = sdk.rusticol_virtual_last_error_message()
    return "" if message is None else message.decode()


def _request(points, scales, resolved: bool):
    flat = [float(c) for point in points for p in point for c in p]
    momenta = (ctypes.c_double * len(flat))(*flat)
    mu = (ctypes.c_double * len(scales))(*scales)
    request = _Request(momenta, len(flat), len(points), None, mu, len(scales))
    request.resolved = int(resolved)
    # The arrays stay alive with the request.
    return request, (momenta, mu)


def _open(sdk, artifact: Path, name: str):
    handle = ctypes.c_void_p()
    status = sdk.rusticol_virtual_open(
        str(artifact).encode(), name.encode(), None, None, ctypes.byref(handle)
    )
    assert status == 0, _error(sdk)
    return handle


def _sdk_call(sdk, artifact, name, points, poles, arithmetic, digits, resolved):
    """One ordinary or resolved C ABI evaluation: its double copies and its
    decimal record (None in binary64)."""

    handle = _open(sdk, artifact, name)
    try:
        assert sdk.rusticol_virtual_set_poles(handle, poles.encode()) == 0
        status = sdk.rusticol_virtual_set_arithmetic(
            handle, arithmetic.encode(), ctypes.c_uint32(digits or 0)
        )
        assert status == 0, _error(sdk)
        request, _keep = _request(points, SCALES[: len(points)], resolved)
        result = ctypes.c_void_p()
        status = sdk.rusticol_virtual_evaluate(
            handle, ctypes.byref(request), ctypes.byref(result)
        )
        assert status == 0, _error(sdk)
        try:
            shape = _ResultShape()
            assert sdk.rusticol_virtual_result_shape(result, ctypes.byref(shape)) == 0
            assert shape.successful_points == len(points)
            born = (ctypes.c_double * shape.born_values)()
            loop = (ctypes.c_double * shape.virtual_values)()
            assert sdk.rusticol_virtual_result_copy_born(result, born, len(born)) == 0
            assert (
                sdk.rusticol_virtual_result_copy_virtual(result, loop, len(loop)) == 0
            )
            needed = ctypes.c_size_t()
            get = sdk.rusticol_virtual_result_decimal_json
            assert get(result, None, ctypes.c_size_t(0), ctypes.byref(needed)) == 0
            buffer = ctypes.create_string_buffer(needed.value)
            assert get(result, buffer, needed, ctypes.byref(needed)) == 0
            decimal = json.loads(buffer.value)
        finally:
            sdk.rusticol_virtual_result_free(result)
    finally:
        sdk.rusticol_virtual_free(handle)
    return {"born": list(born), "virtual": list(loop), "decimal": decimal}


def _python_call(runtime, points, poles, arithmetic, digits, resolved):
    """The Python API's values of one call in the SDK layout (``references``)."""

    batch = [[[str(Decimal(c)) for c in p] for p in point] for point in points]
    options = {"mu_r2": [str(Decimal(s)) for s in SCALES[: len(points)]]}
    options["poles"] = poles
    if arithmetic != "double":
        options["arithmetic"] = arithmetic
    if digits is not None:
        options["precision"] = digits
    if resolved:
        result = runtime.evaluate_resolved(batch, **options)
        born = [x for b in result.born_values for o in b for row in o for x in row]
        virtual = [
            x
            for v in result.values
            for sector in v
            for row in sector
            for lc in row
            for x in _laurent(lc)
        ]
    else:
        result = runtime.evaluate(batch, **options)
        born = [v for b in result.born_values for v in b]
        virtual = [x for v in result.values for lc in v for x in _laurent(lc)]
    return {"born": _flat(born), "virtual": _flat(virtual), "complex": False}


def _has_companion(artifact: Path, name: str) -> bool:
    from pyamplicol.loops.native import load_loop_native
    from pyamplicol.loops.v2.artifact import admit

    found = admit(load_loop_native(), artifact, name)
    return found["scaleless_plan_sha256"] is not None


# (call, arithmetic, digits, poles, resolved, points): the ordinary calls go
# through the stored plan (``plan_export``), the resolved ones through the
# stored observable columns (``native_observables``).
SPLIT_CALLS = [
    ("ordinary-uv-dd", "double-double", None, "uv", False, 1),
    ("ordinary-ir-ap", "arbitrary", 32, "ir", False, 1),
    ("resolved-ir-dd", "double-double", None, "ir", True, 1),
    ("resolved-uv-ap", "arbitrary", 32, "uv", True, 1),
]


def test_sdk_split_poles_in_extended_precision_equal_the_python_api(artifacts):
    """Double-double and arbitrary(32) in the "uv" and "ir" selections: the
    SDK's decimals equal the Python API's ``Decimal`` values exactly (their
    companion lanes included: read outside the working-digit scope they were
    rounded at Python's default 28 digits, about 1e-28 off)."""

    from pyamplicol import Runtime

    sdk = _sdk()
    for name, case in CASES.items():
        assert _has_companion(artifacts[name], name), name
        runtime = Runtime.load(artifacts[name], process=name)
        expected, found = {}, {}
        for call, arithmetic, digits, poles, resolved, count in SPLIT_CALLS:
            points = case["points"][:count]
            expected[call] = _python_call(
                runtime, points, poles, arithmetic, digits, resolved
            )
            found[call] = _sdk_call(
                sdk, artifacts[name], name, points, poles, arithmetic, digits, resolved
            )
        _compare(f"split:{name}", expected, found)


LEADING_POINTS = [
    [[500, 0, 0, 500], [500, 0, 0, -500], [500, 300, 0, 400], [500, -300, 0, -400]],
    [[500, 0, 0, 500], [500, 0, 0, -500], [500, 0, 400, 300], [500, 0, -400, -300]],
]


@pytest.fixture(scope="module")
def leading(loop_sm_compiled_model, loop_native, tmp_path_factory):
    """u u~ > g g at leading colour (two Born basis elements)."""

    from pyamplicol.api import Generator, ProcessRequest, ProcessSet
    from pyamplicol.config import resolve_config

    root = tmp_path_factory.mktemp("leading")
    card = root / "card.toml"
    text = _card(loop_sm_compiled_model, root / "artifact", "u u~ > g g", "uugg")
    assert text.count('accuracy = "full"') == 1
    card.write_text(text.replace('accuracy = "full"', 'accuracy = "lc"'))
    Generator(resolve_config(card)).generate(
        ProcessSet((ProcessRequest.parse("u u~ > g g", name="uugg"),)),
        root / "artifact",
    )
    return root / "artifact"


def test_sdk_leading_colour_resolved_rows_equal_the_python_api(leading):
    """The SDK's resolved rows at leading colour, per Born helicity row and
    Born basis element (``basis-<b>``): bitwise the Python API's in binary64
    (two points, every pole), and its decimals in double-double ("ir")."""

    from pyamplicol import Runtime

    runtime = Runtime.load(leading, process="uugg")
    probe = runtime.evaluate_resolved(
        [[[str(c) for c in p] for p in LEADING_POINTS[0]]], mu_r2=str(SCALES[0])
    )
    assert len(probe.born_color_ids) > 1
    assert all(i.startswith("basis-") for i in probe.born_color_ids)
    sdk = _sdk()
    calls = [
        ("lc-resolved", "double", None, "all", 2),
        ("lc-resolved-ir-dd", "double-double", None, "ir", 1),
    ]
    expected, found = {}, {}
    for call, arithmetic, digits, poles, count in calls:
        points = LEADING_POINTS[:count]
        expected[call] = _python_call(runtime, points, poles, arithmetic, digits, True)
        found[call] = _sdk_call(
            sdk, leading, "uugg", points, poles, arithmetic, digits, True
        )
    _compare("lc:uugg", expected, found)


def test_sdk_leading_colour_selection_is_the_python_apis(leading):
    """A Born-interfered leading-colour process (m-lit lit-sdk-b): the Python
    API serves its helicity and colour-flow selection; the SDK's selection
    call refuses it typed (RUSTICOL_VIRTUAL_RUNTIME_ERROR, "served by the
    Python API"), and a colour resolution in the Python API's words."""

    from pyamplicol import Runtime
    from pyamplicol.api.errors import CompatibilityError
    from tests.unit.test_loop_v2_sdk_selection import STATUS_RUNTIME_ERROR, _call

    runtime = Runtime.load(leading, process="uugg")
    point = [[float(c) for c in p] for p in LEADING_POINTS[0]]
    exact = [[[str(Decimal(c)) for c in p] for p in point]]
    scale = float(SCALES[0])
    served = runtime.evaluate(exact, mu_r2=str(Decimal(scale)), helicities=["h0"])
    assert served.values[0] is not None
    with pytest.raises(CompatibilityError) as resolution:
        runtime.evaluate_resolved(
            exact, mu_r2=str(Decimal(scale)), color_resolution="amplitude-basis"
        )
    sdk = _sdk()
    handle = ctypes.c_void_p()
    status = sdk.rusticol_virtual_open(
        str(leading).encode(), b"uugg", None, None, ctypes.byref(handle)
    )
    assert status == 0, _error(sdk)
    try:
        for selectors in ({"ids": ["h0"]}, {"colours": ["1"]}, {"by_point": [0]}):
            found = _call(sdk, handle, [point], [scale], **selectors)
            assert found["status"] == STATUS_RUNTIME_ERROR, (selectors, found)
            assert "served by the Python API" in found["error"], found["error"]
        found = _call(
            sdk, handle, [point], [scale], resolved=True, resolution="amplitude-basis"
        )
        assert found["status"] == STATUS_RUNTIME_ERROR
        assert str(resolution.value) in found["error"]
    finally:
        sdk.rusticol_virtual_free(handle)


def test_sdk_spin_sources_of_another_length_are_refused(artifacts):
    """A spin-source buffer that is neither one row nor one row per point is
    an invalid argument with a message (status 1), never a read past it; one
    broadcast row is accepted (the safe Rust API's per-point decimal sources
    have the matching unit test, ``correlated::tests``)."""

    sdk = _sdk()
    case = CASES["uuddg"]
    legs = sorted(case["spin"])
    handle = _open(sdk, artifacts["uuddg"], "uuddg")
    try:
        request, _keep = _request(case["points"], SCALES, False)
        row = [x for leg in legs for pair in case["spin"][leg] for x in pair]
        for values, status in ((row + [0.0, 0.0], 1), (row, 0)):
            array = (ctypes.c_double * len(values))(*values)
            sources = _Sources(
                (ctypes.c_size_t * len(legs))(*legs),
                len(legs),
                array,
                len(values) // 2,
            )
            result = ctypes.c_void_p()
            found = sdk.rusticol_virtual_evaluate_correlated_with_sources(
                handle,
                ctypes.byref(request),
                ctypes.byref(sources),
                ctypes.byref(result),
            )
            assert found == status, _error(sdk)
            if status:
                assert not result.value
                assert "one row or one row per physical point" in _error(sdk)
            else:
                sdk.rusticol_virtual_correlated_result_free(result)
    finally:
        sdk.rusticol_virtual_free(handle)


def test_umami_extension_version_records_the_options_layout():
    """``PYAMPLICOL_UMAMI_VIRTUAL_VERSION`` names the options layout: 2 since
    the virtual C ABI version 2 dropped ``verification_json`` (the file and
    symbol names keep their ``_v1`` spelling); a layout change must bump it."""

    import re

    header = (SDK / "umami" / "pyamplicol_umami_virtual_v1.h").read_text()
    (version,) = re.findall(r"#define PYAMPLICOL_UMAMI_VIRTUAL_VERSION (\d+)", header)
    (body,) = re.findall(
        r"typedef struct \{(.*?)\} pyamplicol_umami_virtual_v1_options;", header, re.S
    )
    members = re.findall(r"(\w+);", re.sub(r"/\*.*?\*/", "", body, flags=re.S))
    layouts = {"1": ["resolved", "r2_strategy", "verification_json"]}
    layouts["2"] = ["resolved", "r2_strategy"]
    assert members == layouts[version] and version == "2"


WRAPPER = ROOT / "src" / "pyamplicol" / "_sdk" / "rust" / "rusticol.rs"
WRAPPER_CLIENT = """
#[path = "{wrapper}"]
#[allow(dead_code)]
mod rusticol;
use rusticol::virtual_amplitudes::Runtime;

fn main() -> Result<(), Box<dyn std::error::Error>> {{
    let args: Vec<String> = std::env::args().collect();
    let momenta: Vec<f64> = args[3].split(',').map(|v| v.parse().unwrap()).collect();
    let scale: f64 = args[4].parse()?;
    let mut runtime = Runtime::open(&args[1], Some(args[2].as_str()), None, None)?;
    runtime.set_poles("ir")?;
    assert!(runtime.set_arithmetic("arbitrary", 0).is_err());
    runtime.set_arithmetic("double-double", 0)?;
    let points = momenta.len() / (4 * runtime.shape().external_count);
    let result = runtime.evaluate(&momenta, points, None, &[scale], false, None)?;
    let bits = |values: &[f64]| -> String {{
        values.iter().map(|v| v.to_bits().to_string()).collect::<Vec<_>>().join(" ")
    }};
    println!("born {{}}", bits(&result.born));
    println!("virtual {{}}", bits(&result.virtual_values));
    println!("decimal {{}}", result.decimal_json);
    Ok(())
}}
"""


def test_packaged_rust_wrapper_compiles_and_serves_the_extras(artifacts, tmp_path):
    """The packaged single-file Rust wrapper (``_sdk/rust/rusticol.rs``)
    compiles and links against the staged loop SDK, and its
    ``set_arithmetic`` / ``set_poles`` / ``decimal_json`` serve a
    double-double "ir" virtual equal to the C ABI's (doubles bitwise, the
    decimal record equal); a refused arithmetic is an error."""

    build = _staged_sdk()
    source = tmp_path / "wrapper_client.rs"
    source.write_text(WRAPPER_CLIENT.format(wrapper=WRAPPER))
    binary = tmp_path / "wrapper-client"
    subprocess.run(
        [
            _compiler("RUSTC", "rustc"),
            "--edition=2021",
            "-D",
            "warnings",
            str(source),
            "-o",
            str(binary),
            "-L",
            f"native={build}",
            "-l",
            "dylib=rusticol_loop_sdk",
            "-C",
            f"link-arg=-Wl,-rpath,{build}",
        ],
        check=True,
        cwd=tmp_path,
    )
    case = CASES["uuddg"]
    point = case["points"][0]
    run = subprocess.run(
        [
            str(binary),
            str(artifacts["uuddg"]),
            "uuddg",
            ",".join(repr(float(c)) for p in point for c in p),
            repr(SCALES[0]),
        ],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]
    lines = dict(line.split(" ", 1) for line in run.stdout.splitlines())
    expected = _sdk_call(
        _sdk(), artifacts["uuddg"], "uuddg", [point], "ir", "double-double", None, False
    )
    for axis in ("born", "virtual"):
        found = [int(bits) for bits in lines[axis].split()]
        assert found == [_bits(v) & (2**64 - 1) for v in expected[axis]], axis
    assert json.loads(lines["decimal"]) == expected["decimal"]
