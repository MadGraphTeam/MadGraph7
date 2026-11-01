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
                for k, (a, b) in enumerate(zip(want, literals, strict=True)):
                    if a is None:
                        assert b is None, (name, call, axis, k)
                        continue
                    # The exact value, and its nearest double.
                    assert Decimal(a) == Decimal(b), (name, call, axis, k, a, b)
                    assert float(Decimal(a)) == doubles[k], (name, call, axis, k)


def _run(client: Path, artifact: Path, name: str, feed: Path) -> dict:
    run = subprocess.run(
        [str(client), str(artifact), name, str(feed)],
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

    import ctypes

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
