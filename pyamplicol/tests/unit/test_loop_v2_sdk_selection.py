# SPDX-License-Identifier: 0BSD
"""Loop-induced selections in every native SDK language (m-lit lit-sdk-b; the
USER's F1 ruling: the colour-resolved output and the single-helicity
selection at tree-level parity; PLAN §2.a.10, §3.4 LIT-OBS; RULED Q8, Q36,
Q37, Q39).

On the two LI-O1 rows (g g > h g g and u u~ > h g g, full colour, several
amplitude-basis elements; ``loop_v2_induced_observables_rows``):

* ``rusticol_virtual_evaluate_selected`` serves the amplitude-basis anchors
  (``color_resolution="amplitude-basis"``), the global helicity selection
  (one ID, a subset with the tree engine's ID form and a duplicate), the
  resolved selection in both colour resolutions and the per-point helicity
  selection, each binary64 BITWISE the Python API's
  (``evaluate_resolved(color_resolution=...)``, ``evaluate(helicities=...)``,
  ``evaluate_resolved(helicities=...)``, per point
  ``evaluate(helicities=[that point's ID])``) with identical statuses,
  failure reasons and reduction record, on "auto", "ninjars" and "tirust"
  (where the library builds it; otherwise refused at the setter, as the
  Python API's ``ReductionUnavailable``);
* the sums hold (the anchors of a row sum to the row's value, a complete
  selection to the total, within rounding; Q39), the result's axes name the
  Python API's IDs;
* double-double and arbitrary precision equal the Python ``Decimal`` values;
* colour flows, an unknown colour resolution or helicity ID and the other
  invalid selectors are refused before any evaluation, in the Python API's
  words where it has them, writing no result;
* the compiled clients (C, C++, Fortran, UMAMI and the packaged Rust wrapper)
  return the C ABI's values;
* the Python API's binary64 outputs (the native anchor form and row mask the
  SDK runs) are bitwise its ``Decimal`` oracle (``NATIVE_BINARY64 = False``),
  so the comparisons above are not self-referential (R7).
"""

from __future__ import annotations

import ctypes
import json
import math
import subprocess
from decimal import Decimal
from pathlib import Path

import pytest

from tests.unit.loop_v2_induced_observables_rows import ROWS, generate
from tests.unit.test_loop_v2_induced import _fixture
from tests.unit.test_loop_v2_sdk_extras import (
    SDK,
    UMAMI_HEADERS,
    WRAPPER,
    _bits,
    _compiler,
    _double,
    _error,
    _Request,
    _ResultShape,
    _sdk,
    _staged_sdk,
)

pytestmark = pytest.mark.loop_physics

STATUS_INVALID_ARGUMENT = 1
STATUS_RUNTIME_ERROR = 3
SUCCESS_STATUS = 4  # RUSTICOL_VIRTUAL_ACCURACY_NOT_ESTABLISHED
FAILED_STATUS = 3
BACKENDS = ("auto", "ninjars", "tirust")
# The row whose extended-precision runs take minutes (as
# test_loop_v2_sdk_induced.SLOW_EXTENDED): its clients run binary64 only.
SLOW_EXTENDED = "li_gg_hgg"


class _Selection(ctypes.Structure):
    _fields_ = [
        ("helicity_ids", ctypes.POINTER(ctypes.c_char_p)),
        ("helicity_count", ctypes.c_size_t),
        ("color_ids", ctypes.POINTER(ctypes.c_char_p)),
        ("color_count", ctypes.c_size_t),
        ("helicity_by_point", ctypes.POINTER(ctypes.c_uint32)),
        ("helicity_by_point_count", ctypes.c_size_t),
        ("color_flow_by_point", ctypes.POINTER(ctypes.c_uint32)),
        ("color_flow_by_point_count", ctypes.c_size_t),
        ("color_resolution", ctypes.c_char_p),
    ]


@pytest.fixture(scope="module")
def artifacts(loop_sm_ufo, loop_native, tmp_path_factory):
    return {
        name: generate(tmp_path_factory.mktemp(name), name, expression)
        for name, expression in ROWS.items()
    }


def _inputs(name: str, count: int = 2):
    """The fixture's first ``count`` points rounded to binary64 (the C ABI's
    inputs) and their mu_R^2, and the exact decimal texts of those doubles
    (the Python API's inputs: each double's exact value)."""

    fixture = _fixture(name)
    points = [
        [[float(Decimal(c)) for c in momentum] for momentum in point]
        for point in fixture["points"][:count]
    ]
    scales = [float(Decimal(mu)) ** 2 for mu in fixture["mu_r"][:count]]
    exact = [[[str(Decimal(c)) for c in p] for p in point] for point in points]
    return points, scales, exact, [str(Decimal(s)) for s in scales]


def _json(function, handle) -> dict:
    needed = ctypes.c_size_t()
    assert function(handle, None, ctypes.c_size_t(0), ctypes.byref(needed)) == 0
    buffer = ctypes.create_string_buffer(needed.value)
    assert function(handle, buffer, needed, ctypes.byref(needed)) == 0
    return json.loads(buffer.value)


def _open(sdk, artifact: Path):
    handle = ctypes.c_void_p()
    status = sdk.rusticol_virtual_open(
        str(artifact).encode(), b"p", None, None, ctypes.byref(handle)
    )
    assert status == 0, _error(sdk)
    return handle


def _tree_id(row) -> str:
    """The tree engine's ID of a capabilities helicity row (1-based legs)."""

    return "h:" + ",".join(f"{int(h):+d}" for _leg, h in sorted(row))


def _selectors(capabilities: dict, points: int):
    """One ID, a subset (the runtime's ID form, the tree engine's and a
    duplicate) and per point a distinct row."""

    rows = capabilities["helicity_rows"]
    count = len(rows)
    one = f"h{1 % count}"
    subset = [f"h{2 % count}", _tree_id(rows[5 % count]), f"h{2 % count}"]
    by_point = [(3 * p + 1) % count for p in range(points)]
    return one, subset, by_point


def _call(sdk, handle, points, scales, *, resolved=False, ids=(), by_point=(),
          colours=(), colour_by_point=(), resolution=None) -> dict:
    """One ``rusticol_virtual_evaluate_selected`` call: its status, message
    and, on success, shape, values, statuses, records and axes."""

    flat = [c for point in points for p in point for c in p]
    momenta = (ctypes.c_double * max(len(flat), 1))(*flat)
    mu = (ctypes.c_double * max(len(scales), 1))(*scales)
    request = _Request(momenta, len(flat), len(points), None, mu, len(scales))
    request.resolved = int(resolved)
    encoded = [i.encode() for i in ids]
    names = (ctypes.c_char_p * max(len(encoded), 1))(*encoded)
    flows = [c.encode() for c in colours]
    flow_names = (ctypes.c_char_p * max(len(flows), 1))(*flows)
    rows = (ctypes.c_uint32 * max(len(by_point), 1))(*by_point)
    flow_rows = (ctypes.c_uint32 * max(len(colour_by_point), 1))(*colour_by_point)
    selection = _Selection(
        names, len(encoded), flow_names, len(flows), rows, len(by_point),
        flow_rows, len(colour_by_point),
        None if resolution is None else resolution.encode(),
    )
    result = ctypes.c_void_p(1)
    status = sdk.rusticol_virtual_evaluate_selected(
        handle, ctypes.byref(request), ctypes.byref(selection), ctypes.byref(result)
    )
    if status != 0:
        # Nothing written: the output is reset to NULL.
        assert not result.value
        return {"status": status, "error": _error(sdk)}
    try:
        shape = _ResultShape()
        assert sdk.rusticol_virtual_result_shape(result, ctypes.byref(shape)) == 0
        loop = (ctypes.c_double * max(shape.virtual_values, 1))()
        statuses = (ctypes.c_uint32 * max(len(points), 1))()
        assert sdk.rusticol_virtual_result_copy_virtual(
            result, loop, shape.virtual_values
        ) == 0
        assert sdk.rusticol_virtual_result_copy_statuses(
            result, statuses, len(points)
        ) == 0
        return {
            "status": 0,
            "shape": shape,
            "virtual": list(loop)[: shape.virtual_values],
            "statuses": list(statuses)[: len(points)],
            "decimal": _json(sdk.rusticol_virtual_result_decimal_json, result),
            "axes": _json(sdk.rusticol_virtual_result_axes_json, result),
            "diagnostics": _json(sdk.rusticol_virtual_result_diagnostics_json, result),
        }
    finally:
        sdk.rusticol_virtual_result_free(result)


def _laurent(value) -> list:
    return [
        math.nan if x is None else x
        for x in (value.finite, value.single_pole, value.double_pole)
    ]


def _python_values(result, resolved: bool) -> list:
    """The Python API's values in the SDK layouts (successful points)."""

    out: list = []
    for point in result.values:
        if point is None:
            continue
        for sector in point:
            if resolved:
                for row in sector:
                    for value in row:
                        out += _laurent(value)
            else:
                out += _laurent(sector)
    return out


def _bitwise(found: list, expected: list) -> bool:
    return len(found) == len(expected) and all(
        (math.isnan(a) and math.isnan(float(b))) or _bits(a) == _bits(float(b))
        for a, b in zip(found, expected, strict=True)
    )


def _compare(label, found: dict, expected, resolved: bool) -> None:
    """Values bitwise, statuses, failure reasons and the reduction record."""

    assert found["status"] == 0, (label, found.get("error"))
    served = [value is not None for value in expected.values]
    assert found["statuses"] == [
        SUCCESS_STATUS if ok else FAILED_STATUS for ok in served
    ], label
    assert _bitwise(found["virtual"], _python_values(expected, resolved)), label
    failures = found["diagnostics"]["failures"]
    for index, ok in enumerate(served):
        if not ok:
            assert "; ".join(failures[str(index)]) == (
                expected.diagnostics[index].detail
            ), label
    record = expected.diagnostics[0].reduction
    assert found["diagnostics"]["reduction"] == {
        "requested": record.requested,
        "effective": record.effective,
        "reason": record.reason,
    }, label
    if resolved:
        assert found["axes"]["helicity_ids"] == list(expected.helicity_ids), label
        assert found["axes"]["color_ids"] == list(expected.virtual_color_ids), label
        assert found["shape"].axes.helicities == len(expected.helicity_ids), label
        assert found["shape"].axes.virtual_color_components == len(
            expected.virtual_color_ids
        ), label
    else:
        assert found["axes"] == {"helicity_ids": [], "color_ids": []}, label


def _per_point(runtime, exact, scales, by_point, **options):
    """The Python API's per-point selection: each point's own
    ``evaluate(helicities=[its ID])``, as one result's lists."""

    from types import SimpleNamespace

    values, diagnostics = [], []
    for p, row in enumerate(by_point):
        found = runtime.evaluate(
            [exact[p]], mu_r2=[scales[p]], helicities=[f"h{row}"],
            on_failure="return", **options,
        )
        values += list(found.values)
        diagnostics += list(found.diagnostics)
    return SimpleNamespace(values=values, diagnostics=diagnostics)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", sorted(ROWS))
def test_c_abi_selections_equal_the_python_api(artifacts, loop_native, name, backend):
    from pyamplicol import Runtime
    from pyamplicol.api.errors import ReductionUnavailable

    artifact = artifacts[name]
    runtime = Runtime.load(artifact, process="p")
    points, scales, exact, exact_scales = _inputs(name)
    sdk = _sdk()
    handle = _open(sdk, artifact)
    try:
        capabilities = _json(sdk.rusticol_virtual_capabilities_json, handle)
        assert capabilities["helicity_selection"] is True
        anchors = capabilities["color_resolutions"]["amplitude-basis"]
        assert capabilities["color_resolutions"]["contracted"] == ["full"]
        assert len(anchors) >= 2, anchors
        status = sdk.rusticol_virtual_set_reduction(handle, backend.encode())
        options = {"reduction": backend, "on_failure": "return"}
        if status != 0:
            # A backend this library does not build: refused at the setter,
            # as the Python API refuses the call.
            assert backend == "tirust" and status == STATUS_RUNTIME_ERROR
            with pytest.raises(ReductionUnavailable):
                runtime.evaluate(exact, mu_r2=exact_scales, helicities=["h0"], **options)
            return
        one, subset, by_point = _selectors(capabilities, len(points))
        calls = {
            "anchors": (
                {"resolved": True, "resolution": "amplitude-basis"},
                runtime.evaluate_resolved(
                    exact, mu_r2=exact_scales, color_resolution="amplitude-basis",
                    **options,
                ),
                True,
            ),
            "one": (
                {"ids": [one]},
                runtime.evaluate(exact, mu_r2=exact_scales, helicities=[one], **options),
                False,
            ),
            "subset": (
                {"ids": subset},
                runtime.evaluate(exact, mu_r2=exact_scales, helicities=subset, **options),
                False,
            ),
            "resolved-subset": (
                {"resolved": True, "ids": subset, "resolution": "contracted"},
                runtime.evaluate_resolved(
                    exact, mu_r2=exact_scales, helicities=subset, **options
                ),
                True,
            ),
            "anchors-subset": (
                {"resolved": True, "ids": subset, "resolution": "amplitude-basis"},
                runtime.evaluate_resolved(
                    exact, mu_r2=exact_scales, helicities=subset,
                    color_resolution="amplitude-basis", **options,
                ),
                True,
            ),
            "by-point": (
                {"by_point": by_point},
                _per_point(runtime, exact, exact_scales, by_point, reduction=backend),
                False,
            ),
        }
        found = {}
        for label, (selectors, expected, resolved) in calls.items():
            found[label] = _call(sdk, handle, points, scales, **selectors)
            if label == "by-point":
                # The per-point reference is one call per point: values,
                # statuses and reasons (its record is the same call's).
                assert found[label]["status"] == 0, found[label].get("error")
                served = [v is not None for v in expected.values]
                assert found[label]["statuses"] == [
                    SUCCESS_STATUS if ok else FAILED_STATUS for ok in served
                ]
                assert _bitwise(
                    found[label]["virtual"], _python_values(expected, False)
                ), label
                assert found[label]["diagnostics"]["selection"] == {
                    "helicity_ids": [[f"h{row}"] for row in by_point]
                }
                continue
            _compare(f"{name}:{backend}:{label}", found[label], expected, resolved)
        # The sums (Q39; the anchors' split): within rounding, where served.
        total = runtime.evaluate(exact, mu_r2=exact_scales, **options)
        contracted = runtime.evaluate_resolved(exact, mu_r2=exact_scales, **options)
        split = calls["anchors"][1]
        everything = _call(
            sdk, handle, points, scales,
            ids=[f"h{k}" for k in range(len(capabilities["helicity_rows"]))],
        )
        for p, point in enumerate(total.values):
            if point is None or split.values[p] is None:
                continue
            scale = max(abs(x.finite) for x in point) or 1.0
            for s, sector in enumerate(split.values[p]):
                for h, row in enumerate(sector):
                    row_value = contracted.values[p][s][h][0].finite
                    assert abs(sum(v.finite for v in row) - row_value) <= 1e-12 * scale
            if everything["statuses"][p] == SUCCESS_STATUS:
                width = len(everything["virtual"]) // sum(
                    s == SUCCESS_STATUS for s in everything["statuses"]
                )
                row = sum(s == SUCCESS_STATUS for s in everything["statuses"][:p])
                selected = everything["virtual"][row * width:(row + 1) * width]
                for k, value in enumerate(point):
                    assert abs(selected[3 * k] - value.finite) <= 1e-12 * scale
    finally:
        sdk.rusticol_virtual_free(handle)


def test_c_abi_selection_refusals(artifacts):
    """Every invalid selection is refused before any evaluation (no result
    written), in the Python API's words where it has them."""

    from pyamplicol import Runtime
    from pyamplicol.api.errors import CompatibilityError, EvaluationError

    name = "li_uux_hgg"
    runtime = Runtime.load(artifacts[name], process="p")
    points, scales, exact, exact_scales = _inputs(name)
    sdk = _sdk()
    handle = _open(sdk, artifacts[name])
    try:
        with pytest.raises(CompatibilityError) as flows:
            runtime.evaluate(exact, mu_r2=exact_scales, color_flows=["1"])
        with pytest.raises(CompatibilityError) as resolution:
            runtime.evaluate_resolved(exact, mu_r2=exact_scales, color_resolution="lc")
        with pytest.raises(EvaluationError) as unknown:
            runtime.evaluate(exact, mu_r2=exact_scales, helicities=["h:bogus"])
        cases = [
            ({"colours": ["1"]}, STATUS_RUNTIME_ERROR, str(flows.value)),
            ({"colour_by_point": [0, 0]}, STATUS_RUNTIME_ERROR, str(flows.value)),
            ({"resolved": True, "colours": ["1"]}, STATUS_RUNTIME_ERROR,
             str(flows.value)),
            ({"resolved": True, "resolution": "lc"}, STATUS_RUNTIME_ERROR,
             str(resolution.value)),
            ({"ids": ["h:bogus"]}, STATUS_INVALID_ARGUMENT, str(unknown.value)),
            ({"by_point": [0, 10**6]}, STATUS_INVALID_ARGUMENT, "out of range"),
            ({"by_point": [0]}, STATUS_INVALID_ARGUMENT, "one selector per point"),
            ({"ids": ["h0"], "by_point": [0, 1]}, STATUS_INVALID_ARGUMENT,
             "mutually exclusive"),
            ({"resolved": True, "by_point": [0, 1]}, STATUS_INVALID_ARGUMENT,
             "global helicity ids"),
            ({"resolution": "amplitude-basis"}, STATUS_INVALID_ARGUMENT,
             "resolved evaluation"),
        ]
        for selectors, status, message in cases:
            found = _call(sdk, handle, points, scales, **selectors)
            assert found["status"] == status, (selectors, found)
            assert message in found["error"], (selectors, found["error"])
        # A NULL selection is the ordinary evaluation.
        flat = [c for point in points for p in point for c in p]
        momenta = (ctypes.c_double * len(flat))(*flat)
        mu = (ctypes.c_double * len(scales))(*scales)
        request = _Request(momenta, len(flat), len(points), None, mu, len(scales))
        result = ctypes.c_void_p()
        assert sdk.rusticol_virtual_evaluate_selected(
            handle, ctypes.byref(request), None, ctypes.byref(result)
        ) == 0
        sdk.rusticol_virtual_result_free(result)
    finally:
        sdk.rusticol_virtual_free(handle)


@pytest.mark.parametrize("arithmetic,digits", [("double-double", 0), ("arbitrary", 32)])
def test_c_abi_extended_selections_equal_the_python_decimals(
    artifacts, arithmetic, digits
):
    """Double-double and arbitrary precision: the anchors and a helicity
    selection equal the Python API's ``Decimal`` values (one point of the
    row whose extended runs are fast), their doubles the nearest."""

    from pyamplicol import Runtime

    name = "li_uux_hgg"
    runtime = Runtime.load(artifacts[name], process="p")
    points, scales, exact, exact_scales = _inputs(name, 1)
    sdk = _sdk()
    handle = _open(sdk, artifacts[name])
    try:
        status = sdk.rusticol_virtual_set_arithmetic(
            handle, arithmetic.encode(), ctypes.c_uint32(digits)
        )
        assert status == 0, _error(sdk)
        options = {"arithmetic": arithmetic}
        if digits:
            options["precision"] = digits
        anchors = _call(sdk, handle, points, scales, resolved=True,
                        resolution="amplitude-basis")
        expected = runtime.evaluate_resolved(
            exact, mu_r2=exact_scales, color_resolution="amplitude-basis", **options
        )
        selected = _call(sdk, handle, points, scales, ids=["h1", "h3"])
        expected_selected = runtime.evaluate(
            exact, mu_r2=exact_scales, helicities=["h1", "h3"], **options
        )
        for found, reference, resolved in (
            (anchors, expected, True),
            (selected, expected_selected, False),
        ):
            assert found["status"] == 0, found.get("error")
            literals = found["decimal"]["virtual"]
            values: list = []
            for point in reference.values:
                for sector in point:
                    cells = [c for row in sector for c in row] if resolved else [sector]
                    for cell in cells:
                        values += [cell.finite, cell.single_pole, cell.double_pole]
            assert len(literals) == len(values)
            for literal, value in zip(literals, values, strict=True):
                if value is None:
                    assert literal is None
                    continue
                assert Decimal(literal) == Decimal(value), (literal, value)
            assert found["virtual"] == [
                math.nan if x is None else float(Decimal(x)) for x in literals
            ] or all(
                (a != a and x is None) or a == float(Decimal(x))
                for a, x in zip(found["virtual"], literals, strict=True)
            )
    finally:
        sdk.rusticol_virtual_free(handle)


def test_python_binary64_selections_run_the_native_implementation(artifacts):
    """R7 / [D3]: the Python API's binary64 anchors and helicity selections
    (the native anchor form, ``induced_anchor_form``, and the executor's row
    liveness, the code the SDK runs) are bitwise its ``Decimal`` oracle
    (``NATIVE_BINARY64 = False``), so the SDK comparisons above are not
    self-referential."""

    from pyamplicol import Runtime
    from pyamplicol.loops.v2 import induced_observables as io

    def values(runtime, exact, scales) -> list:
        out: list = []
        anchors = runtime.evaluate_resolved(
            exact, mu_r2=scales, color_resolution="amplitude-basis"
        )
        out += _python_values(anchors, True)
        selected = runtime.evaluate(exact, mu_r2=scales, helicities=["h1", "h2"])
        out += _python_values(selected, False)
        subset = runtime.evaluate_resolved(
            exact, mu_r2=scales, helicities=["h0", "h3"],
            color_resolution="amplitude-basis",
        )
        out += _python_values(subset, True)
        return out

    for name in sorted(ROWS):
        _points, _scales, exact, scales = _inputs(name)
        assert io.NATIVE_BINARY64
        found = values(Runtime.load(artifacts[name], process="p"), exact, scales)
        io.NATIVE_BINARY64 = False
        try:
            oracle = values(Runtime.load(artifacts[name], process="p"), exact, scales)
        finally:
            io.NATIVE_BINARY64 = True
        assert len(found) == len(oracle) > 0
        assert _bitwise(found, oracle), name


# ------------------------------------------------------------ clients

CLIENT_CALLS = [
    "anchors", "one", "subset", "resolved-subset", "anchors-subset", "by-point",
    "colour", "anchors-dd", "by-point-dd", "anchors-ap", "subset-ap",
]
UMAMI_CALLS = ["anchors", "by-point", "anchors-dd", "by-point-dd", "anchors-ap"]


@pytest.fixture(scope="module")
def selection_clients(tmp_path_factory):
    """The selection examples compiled against the staged build's loop SDK:
    C, C++, Fortran and UMAMI (the extension compiled with its client)."""

    build = _staged_sdk()
    root = tmp_path_factory.mktemp("selection-clients")
    examples = SDK / "examples"
    link = [f"-L{build}", "-lrusticol_loop_sdk", f"-Wl,-rpath,{build}", "-lm"]
    include = [f"-I{SDK / 'include'}"]
    cc, cxx, fc = (
        _compiler("CC", "cc"),
        _compiler("CXX", "c++"),
        _compiler("FC", "gfortran"),
    )
    out = {language: root / f"selection-{language}" for language in
           ("c", "cpp", "fortran", "umami")}
    commands = [
        [cc, "-std=c11", "-O2", *include, str(examples / "selection_v4.c"), *link,
         "-o", str(out["c"])],
        [cxx, "-std=c++17", "-O2", *include, str(examples / "selection_v4.cpp"),
         *link, "-o", str(out["cpp"])],
        [fc, "-std=f2018", "-O2", f"-J{root}", str(SDK / "fortran" / "rusticol_virtual.f90"),
         str(examples / "selection_v4.f90"), *link, "-o", str(out["fortran"])],
        [cc, "-std=c11", "-O2", *include, f"-I{SDK / 'umami'}", f"-I{UMAMI_HEADERS}",
         str(SDK / "umami" / "pyamplicol_umami_virtual_v1.c"),
         str(examples / "selection_v4_umami.c"), *link, "-o", str(out["umami"])],
    ]
    for command in commands:
        subprocess.run(command, check=True, cwd=root)
    return out


def _feed(path: Path, points, scales, ids, by_point, *, bits: bool) -> Path:
    def text(value: float) -> str:
        return str(_bits(float(value))) if bits else float(value).hex()

    lines = [f"{len(points)} {len(points[0])}"]
    for point, scale in zip(points, scales, strict=True):
        lines.append(" ".join([text(c) for p in point for c in p] + [text(scale)]))
    quoted = [f'"{i}"' if bits else i for i in ids]
    lines.append(" ".join([str(len(ids)), *quoted]))
    lines.append(" ".join(str(row) for row in by_point))
    path.write_text("\n".join(lines) + "\n")
    return path


def _parse(stdout: str) -> dict:
    found: dict = {}
    current: dict = {}
    for line in stdout.splitlines():
        kind, _, rest = line.strip().partition(" ")
        if kind == "call":
            current = found.setdefault(rest, {})
        elif kind == "virtual":
            current["virtual"] = [_double(token) for token in rest.split()]
        elif kind == "statuses":
            current["statuses"] = [int(token) for token in rest.split()]
        elif kind in ("decimal", "axes"):
            current[kind] = json.loads(rest)
        elif kind == "refused":
            current["refused"] = rest
        else:
            raise AssertionError(line)
    return found


@pytest.fixture(scope="module")
def client_cases(artifacts):
    """Per row the client inputs and the C ABI's answer to every client
    call (the reference the clients transport: itself the Python API's,
    above)."""

    sdk = _sdk()
    out = {}
    for name in sorted(ROWS):
        points, scales, _exact, _scales = _inputs(name)
        handle = _open(sdk, artifacts[name])
        try:
            capabilities = _json(sdk.rusticol_virtual_capabilities_json, handle)
            one, subset, by_point = _selectors(capabilities, len(points))
            ids = [one, *subset[1:]]
            calls = {
                "anchors": ("double", 0, 2, {"resolved": True, "resolution": "amplitude-basis"}),
                "one": ("double", 0, 2, {"ids": ids[:1]}),
                "subset": ("double", 0, 2, {"ids": ids}),
                "resolved-subset": ("double", 0, 2,
                                    {"resolved": True, "ids": ids, "resolution": "contracted"}),
                "anchors-subset": ("double", 0, 2, {"resolved": True, "ids": ids,
                                                    "resolution": "amplitude-basis"}),
                "by-point": ("double", 0, 2, {"by_point": by_point}),
                "colour": ("double", 0, 2, {"colours": ["1"]}),
                "anchors-dd": ("double-double", 0, 1,
                               {"resolved": True, "resolution": "amplitude-basis"}),
                "by-point-dd": ("double-double", 0, 1, {"by_point": by_point[:1]}),
                "anchors-ap": ("arbitrary", 32, 1,
                               {"resolved": True, "resolution": "amplitude-basis"}),
                "subset-ap": ("arbitrary", 32, 1, {"ids": ids}),
            }
            reference = {}
            for label, (arithmetic, digits, count, selectors) in calls.items():
                if name == SLOW_EXTENDED and arithmetic != "double":
                    continue
                assert sdk.rusticol_virtual_set_arithmetic(
                    handle, arithmetic.encode(), ctypes.c_uint32(digits)
                ) == 0
                reference[label] = _call(
                    sdk, handle, points[:count], scales[:count], **selectors
                )
            out[name] = {
                "points": points, "scales": scales, "ids": ids, "by_point": by_point,
                "anchors": len(capabilities["color_resolutions"]["amplitude-basis"]),
                "reference": reference,
            }
        finally:
            sdk.rusticol_virtual_free(handle)
    return out


def _check_client(label: str, found: dict, reference: dict, calls) -> None:
    expected = [call for call in calls if call in reference]
    assert sorted(found) == sorted(expected), (label, sorted(found))
    for call in expected:
        want, got = reference[call], found[call]
        if want["status"] != 0:
            assert want["error"] in got.get("refused", ""), (label, call, got)
            continue
        assert "refused" not in got, (label, call, got)
        assert got["statuses"] == want["statuses"], (label, call)
        assert _bitwise(got["virtual"], want["virtual"]), (label, call)
        assert got["decimal"] == want["decimal"], (label, call)
        assert got["axes"] == want["axes"], (label, call)


@pytest.mark.parametrize("language", ["c", "cpp", "fortran", "umami"])
def test_selection_clients_return_the_c_abi_values(
    language, artifacts, selection_clients, client_cases, tmp_path
):
    for name, case in client_cases.items():
        feed = _feed(
            tmp_path / f"{name}.txt", case["points"], case["scales"], case["ids"],
            case["by_point"], bits=language == "fortran",
        )
        command = [str(selection_clients[language]), str(artifacts[name]), "p", str(feed)]
        if language == "umami":
            command.append(str(case["anchors"]))
        if name == SLOW_EXTENDED:
            command.append("binary64")
        run = subprocess.run(command, capture_output=True, text=True, timeout=3600,
                             check=False)
        assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]
        calls = UMAMI_CALLS if language == "umami" else CLIENT_CALLS
        _check_client(f"{language}:{name}", _parse(run.stdout), case["reference"], calls)


WRAPPER_CLIENT = """
#[path = "{wrapper}"]
#[allow(dead_code)]
mod rusticol;
use rusticol::virtual_amplitudes::{{Runtime, Selection}};

fn main() -> Result<(), Box<dyn std::error::Error>> {{
    let args: Vec<String> = std::env::args().collect();
    let momenta: Vec<f64> = args[3].split(',').map(|v| f64::from_bits(v.parse().unwrap())).collect();
    let scales: Vec<f64> = args[4].split(',').map(|v| f64::from_bits(v.parse().unwrap())).collect();
    let ids: Vec<String> = args[5].split(' ').map(str::to_owned).collect();
    let rows: Vec<u32> = args[6].split(',').map(|v| v.parse().unwrap()).collect();
    let mut runtime = Runtime::open(&args[1], Some(args[2].as_str()), None, None)?;
    let points = scales.len();
    let calls: Vec<(&str, bool, Selection)> = vec![
        ("anchors", true, Selection {{ color_resolution: Some("amplitude-basis".into()), ..Default::default() }}),
        ("one", false, Selection {{ helicity_ids: ids[..1].to_vec(), ..Default::default() }}),
        ("subset", false, Selection {{ helicity_ids: ids.clone(), ..Default::default() }}),
        ("anchors-subset", true, Selection {{ helicity_ids: ids.clone(), color_resolution: Some("amplitude-basis".into()), ..Default::default() }}),
        ("by-point", false, Selection {{ helicity_by_point: rows.clone(), ..Default::default() }}),
        ("colour", false, Selection {{ color_ids: vec!["1".into()], ..Default::default() }}),
    ];
    for (name, resolved, selection) in calls {{
        println!("call {{name}}");
        match runtime.evaluate_selected(&momenta, points, None, &scales, resolved, None, &selection) {{
            Ok(result) => {{
                let statuses: Vec<String> = result.statuses.iter().map(|s| match s {{
                    rusticol::virtual_amplitudes::PointStatus::Failed => "3".to_owned(),
                    _ => "4".to_owned(),
                }}).collect();
                println!("statuses {{}}", statuses.join(" "));
                let bits: Vec<String> = result.virtual_values.iter().map(|v| format!("b:{{:016x}}", v.to_bits())).collect();
                println!("virtual {{}}", bits.join(" "));
                println!("decimal {{}}", result.decimal_json);
                println!("axes {{}}", result.axes_json);
            }}
            Err(error) => println!("refused {{error}}"),
        }}
    }}
    Ok(())
}}
"""


def test_packaged_rust_wrapper_serves_the_selections(artifacts, client_cases, tmp_path):
    """The packaged single-file Rust wrapper (``_sdk/rust/rusticol.rs``)
    compiles against the staged loop SDK and its ``evaluate_selected``
    returns the C ABI's binary64 values, statuses and axes (a refusal as
    an error)."""

    build = _staged_sdk()
    source = tmp_path / "selection_client.rs"
    source.write_text(WRAPPER_CLIENT.format(wrapper=WRAPPER))
    binary = tmp_path / "selection-wrapper"
    subprocess.run(
        [
            _compiler("RUSTC", "rustc"), "--edition=2021", "-D", "warnings",
            str(source), "-o", str(binary), "-L", f"native={build}",
            "-l", "dylib=rusticol_loop_sdk", "-C", f"link-arg=-Wl,-rpath,{build}",
        ],
        check=True,
        cwd=tmp_path,
    )
    mask = 2**64 - 1
    for name, case in client_cases.items():
        run = subprocess.run(
            [
                str(binary), str(artifacts[name]), "p",
                ",".join(str(_bits(c) & mask) for point in case["points"]
                         for p in point for c in p),
                ",".join(str(_bits(s) & mask) for s in case["scales"]),
                " ".join(case["ids"]),
                ",".join(str(row) for row in case["by_point"]),
            ],
            capture_output=True, text=True, timeout=1800, check=False,
        )
        assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]
        calls = ["anchors", "one", "subset", "anchors-subset", "by-point", "colour"]
        _check_client(f"rust:{name}", _parse(run.stdout), case["reference"], calls)


@pytest.mark.loop_nightly
def test_the_rust_sdk_selections_follow_the_c_abi(artifacts, client_cases, tmp_path):
    """The safe Rust ``Runtime::evaluate_selected``
    (``sdk/tests/v4_selection.rs``) on both rows, compared as the other
    clients (opt-in as ``test_loop_v2_sdk_induced.
    test_the_rust_sdk_observables_follow_the_python_api``:
    ``PYAMPLICOL_SDK_CARGO_TARGET_DIR``, a private cargo target directory)."""

    import os

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
    root = Path(__file__).resolve().parents[2]
    mask = 2**64 - 1
    for name, case in client_cases.items():
        feed = tmp_path / f"{name}.json"
        feed.write_text(
            json.dumps(
                {
                    "artifact": str(artifacts[name]),
                    "process": "p",
                    "points": [
                        [_bits(float(c)) & mask for p in point for c in p]
                        + [_bits(scale) & mask]
                        for point, scale in zip(
                            case["points"], case["scales"], strict=True
                        )
                    ],
                    "ids": case["ids"],
                    "by_point": case["by_point"],
                }
            )
        )
        output = tmp_path / f"{name}.out"
        env = {k: v for k, v in os.environ.items() if k != "CARGO_TARGET_DIR"}
        env = {
            **_CODEGEN_DEFAULTS,
            **env,
            "PYAMPLICOL_V4_SELECTION_FEED": str(feed),
            "PYAMPLICOL_V4_SELECTION_OUTPUT": str(output),
        }
        run = subprocess.run(
            [
                "cargo", "test", "--locked", "--release", "--target-dir", target,
                "-p", "rusticol-loop-sdk", "--test", "v4_selection", "--", "--ignored",
            ],
            cwd=root, env=env, capture_output=True, text=True, timeout=7200,
            check=False,
        )
        assert run.returncode == 0, run.stdout[-4000:] + run.stderr[-4000:]
        calls = [
            "anchors", "one", "subset", "resolved-subset", "anchors-subset",
            "by-point", "colour",
        ]
        _check_client(f"rust-sdk:{name}", _parse(output.read_text()), case["reference"], calls)
