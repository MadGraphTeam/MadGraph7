# SPDX-License-Identifier: 0BSD
"""The complex-mass scheme of a UFO (``loops.model_cms.complex_mass_scheme``):
a port of MG5_aMC's ``Model.change_mass_to_complex_scheme`` (m-sm PLAN §3.2
MS-CMS), checked against MG5_aMC's own transformed model (the first oracle,
item 2) and on its records."""

from __future__ import annotations

import cmath
import io
import json
import re
import subprocess
import sys
import tokenize
from decimal import Decimal, localcontext
from pathlib import Path

import pytest

from pyamplicol.loops.model_cms import (
    ComplexMassSchemeUnsupported,
    _rename_expression,
    complex_mass_scheme,
)
from pyamplicol.models.loop_import import RawLoopModel, load_raw_loop_model

pytestmark = pytest.mark.loop_fast

REPOSITORY = Path(__file__).resolve().parents[2]
TOLERANCE = 1e-12


def test_cms_identifier_rewrite_preserves_numeric_lexemes_and_other_tokens():
    source = "MW+MW2+cmath.MW+'MW'+1.0000000000000000000000000001e-200 - MW **2"
    result = _rename_expression(source, {"MW": "CMASS_MW"})

    def tokens(text):
        return [
            (x.type, x.string)
            for x in tokenize.generate_tokens(io.StringIO(text).readline)
        ]

    original = tokens(source)
    expected = list(original)
    expected[0] = (tokenize.NAME, "CMASS_MW")
    expected[-5] = (tokenize.NAME, "CMASS_MW")
    assert tokens(result) == expected
    # Every other character, spacing included, is kept.
    assert result == "CMASS_" + source[: -len("MW **2")] + "CMASS_MW **2"


@pytest.fixture(scope="module")
def gmu_raw(loop_gmu_ufo):
    return load_raw_loop_model(str(loop_gmu_ufo))


@pytest.fixture(scope="module")
def cms(gmu_raw):
    original = gmu_raw.to_dict()
    result = complex_mass_scheme(gmu_raw)
    assert result.original is gmu_raw and gmu_raw.to_dict() == original
    return result


def test_transform_follows_mg5_on_the_gmu_records(cms):
    """MG5's steps on the G_mu UFO at its default card: CMSParam = 1; each
    width met first (source order) keeps its mass and width real, a nonzero
    one gives its external mass a pole mass; the Yukawa inputs become their
    particles' (pole) masses; every other internal parameter is complex and
    reads the pole masses; every particle of such a mass (Goldstone bosons
    and ghosts included) propagates with it and width ZERO."""

    raw, original = cms.derived, cms.original
    assert dict(cms.masses) == {
        "MZ": "CMASS_MZ",
        "MW": "CMASS_MW",
        "MT": "CMASS_MT",
        "MH": "CMASS_MH",
    }
    assert cms.real == ("ZERO", "ZERO", "WZ", "MZ", "WW", "MW", "WT", "MT", "WH", "MH")
    assert cms.cms_parameter == "CMSParam" and cms.internal_widths == ()
    assert raw.exact_value(
        "parameters", raw.record("parameters", "CMSParam")["id"]
    ) == {
        "kind": "expression",
        "expression": "1.0",
    }
    assert cms.yukawas["ymt"] == "CMASS_MT" and cms.yukawas["ymb"] == "MB"
    ymt = raw.record("parameters", "ymt")["fields"]
    assert ymt["nature"] == "internal" and "lhablock" not in ymt
    for mass, alias in cms.masses.items():
        record = raw.record("parameters", alias)
        width = cms.widths[mass]
        assert raw.exact_value("parameters", record["id"])["expression"] == (
            f"cmath.sqrt({mass}**2 - complex(0,1) * {mass} * {width})"
        )
        for name in (mass, width):
            fields = raw.record("parameters", name)["fields"]
            assert fields["nature"] == "external" and fields["type"] == "real"
    sw2 = raw.record("parameters", "sw2")["fields"]
    assert sw2["type"] == "complex"
    assert sw2["value"]["expression"] == "1 - CMASS_MW**2/CMASS_MZ**2"
    for record in raw.collections["particles"]:
        fields = record["fields"]
        mass = original.collections["particles"][record["id"]]["fields"]["mass"]
        if mass["name"] in cms.masses:
            assert fields["mass"]["name"] == cms.masses[mass["name"]]
            assert fields["width"]["name"] == "ZERO"
        else:
            assert fields["mass"] == mass
    assert cms.propagators[9000003] == "CMASS_MW"  # a ghost of the W mass
    top = repr(raw.record("particles", "t")["fields"]["counterterm"])
    assert "CMASS_MT" in top and not re.search(r"(?<!\w)MT(?!\w)", top)
    for collection in ("vertices", "counterterm_vertices", "lorentz_structures"):
        assert raw.collections[collection] == original.collections[collection]
    # [U3]: the bound reglog is the UFO-standard principal one.
    bound = raw.metadata["function_bindings"]["reglog"]
    assert raw.collections["functions"][bound]["fields"]["expr"] == (
        "(0.0 if z==0.0 else cmath.log(z))"
    )
    assert raw.metadata["complex_mass_scheme"]["masses"] == dict(cms.masses)
    copied = RawLoopModel.from_dict(json.loads(json.dumps(raw.to_dict())))
    assert copied.to_dict() == raw.to_dict()
    with pytest.raises(ValueError, match="already applied"):
        complex_mass_scheme(raw)


def test_zero_widths_at_the_card_keep_their_masses_real(gmu_raw):
    """A width zero at the card (here fixed by the request) gives no pole
    mass, as MG5 skips a zero width; a request fixing a Yukawa input is
    refused, since the scheme derives them from the masses."""

    scheme = complex_mass_scheme(
        gmu_raw, fixed_parameters={"WZ": "0", "WW": "0", "WH": ("0", "0")}
    )
    assert dict(scheme.masses) == {"MT": "CMASS_MT"}
    assert scheme.derived.record("particles", "Z")["fields"]["mass"]["name"] == "MZ"
    none = complex_mass_scheme(
        gmu_raw, fixed_parameters=dict.fromkeys(("WZ", "WW", "WH", "WT"), "0")
    )
    assert not none.masses and not none.propagators
    # No pole mass: MG5 ties the Yukawas and returns, parameters stay real.
    assert none.derived.record("parameters", "sw2")["fields"]["type"] == "real"
    assert none.yukawas["ymt"] == "MT"
    with pytest.raises(ComplexMassSchemeUnsupported, match="cms-yukawa-input") as error:
        complex_mass_scheme(gmu_raw, fixed_parameters={"ymt": "172"})
    assert error.value.key == "cms-yukawa-input"


def test_an_internal_mass_with_a_width_is_refused(loop_sm_ufo):
    """loop_sm's W mass is internal: with its card width MG5 would switch the
    model's EW input scheme by a PDG-24 rule (not ported, m-sm PLAN §1.7), so
    the transform refuses it typed; with the W, Z and H widths zero only the
    top takes a pole mass (no CMSParam on loop_sm)."""

    raw = load_raw_loop_model(str(loop_sm_ufo))
    with pytest.raises(
        ComplexMassSchemeUnsupported, match="cms-internal-mass"
    ) as error:
        complex_mass_scheme(raw)
    assert error.value.key == "cms-internal-mass" and "'MW'" in str(error.value)
    scheme = complex_mass_scheme(
        raw, fixed_parameters=dict.fromkeys(("WZ", "WW", "WH"), "0")
    )
    assert dict(scheme.masses) == {"MT": "CMASS_MT"}
    assert scheme.cms_parameter is None and scheme.yukawas["ymt"] == "CMASS_MT"


@pytest.mark.loop_physics
def test_an_internal_width_takes_its_real_part(gmu_raw):
    """An internal width takes ``re()`` (MG5: ``width.expr = 're(...)'``),
    decided at its value on the card, and is left real."""

    payload = gmu_raw.to_dict()
    record = payload["collections"]["parameters"][
        gmu_raw.record("parameters", "WZ")["id"]
    ]
    record["fields"].update(nature="internal", lhablock=None, lhacode=None)
    record["fields"]["value"] = {"kind": "expression", "expression": "2*MZ/73"}
    scheme = complex_mass_scheme(RawLoopModel.from_dict(payload))
    assert scheme.internal_widths == ("WZ",) and "MZ" in scheme.masses
    fields = scheme.derived.record("parameters", "WZ")["fields"]
    assert fields["value"]["expression"] == "re(2*MZ/73)" and fields["type"] == "real"
    record["fields"]["value"] = {"kind": "expression", "expression": "0*MZ"}
    scheme = complex_mass_scheme(RawLoopModel.from_dict(payload))
    assert "MZ" not in scheme.masses and scheme.internal_widths == ()


# ------------------------------------------- the first oracle: MG5's own model


def _loop_sm_top_width_card(loop_sm_ufo, directory: Path) -> Path:
    """loop_sm's default card with the W, Z and H widths zero (Gamma_t from
    the card): the CM10 configuration (m-sm PLAN §4.1 code Lc)."""

    text = (Path(loop_sm_ufo) / "restrict_default.dat").read_text()
    for pdg in (23, 24, 25):
        text, count = re.subn(
            rf"(?m)^DECAY\s+{pdg}\s+\S+", f"DECAY  {pdg} 0.000000e+00", text
        )
        assert count == 1
    path = directory / "restrict_cms_top_width.dat"
    path.write_text(text)
    return path


ORACLE = {
    # case: (UFO fixture, restriction, fixed widths)
    "gmu-default": ("loop_gmu_ufo", "default", {}),
    "gmu-a0": ("loop_gmu_ufo", "a0", {}),
    "loop_sm-top-width": (
        "loop_sm_ufo",
        "default",
        dict.fromkeys(("WZ", "WW", "WH"), "0"),
    ),
}


def _mg5_values(madgraph_source, ufo: Path, restriction, fixed, directory: Path):
    """MG5_aMC's complex-mass model at the card of an ``ORACLE`` case
    (``tools/loop_reference/mg5_cms_model.py``, run with this Python)."""

    card = (
        _loop_sm_top_width_card(ufo, directory)
        if fixed
        else ufo / f"restrict_{restriction}.dat"
    )
    output = directory / "mg5.json"
    subprocess.run(
        [
            sys.executable,
            str(REPOSITORY / "tools/loop_reference/mg5_cms_model.py"),
            f"--mg5={madgraph_source}",
            f"--ufo={ufo}",
            f"--card={card}",
            f"--output={output}",
        ],
        check=True,
        cwd=directory,
        capture_output=True,
        timeout=600,
    )
    return json.loads(output.read_text())


def _mg5_outputs(raw, mg5):
    """Our (name, epsilon power) of each MG5 name: parameters (``mdl_``
    names, MG5's unprefixed ones), the pole masses (``CMASS_mdl_M``), the CT
    parameters' Laurent pieces (``_FIN_``, ``_1EPS_``, ``_2EPS_``), the
    couplings' (``_1eps``, ``_2eps``) and the particles' field counterterms
    (MG5's ``UVWfct_<particle>_<loop index>`` couplings, as anonymous
    definitions)."""

    parameters = {r["name"]: r for r in raw.collections["parameters"]}
    counterterms = {r["name"] for r in raw.collections["counterterm_parameters"]}
    couplings = {r["name"] for r in raw.collections["couplings"]}
    anonymous = {}
    for particle in raw.collections["particles"]:
        entries = (particle["fields"].get("counterterm") or {}).get("entries", ())
        for key, laurent in entries:
            name = f"UVWfct_{particle['fields']['name']}_{key['items'][-1]}"
            assert name not in anonymous
            anonymous[name] = {
                "kind": "laurent",
                "terms": tuple((int(p), e) for p, e in laurent["entries"]),
            }
    outputs = {}
    for name in mg5["parameters"]:
        if name.startswith("CMASS_mdl_"):
            outputs[name] = ("CMASS_" + name[len("CMASS_mdl_") :], 0)
            continue
        piece = re.fullmatch(r"mdl_(\w+)_(FIN|1EPS|2EPS)_", name)
        if piece and piece[1] in counterterms:
            outputs[name] = (piece[1], {"FIN": 0, "1EPS": -1, "2EPS": -2}[piece[2]])
            continue
        bare = name.removeprefix("mdl_")
        if bare in parameters and parameters[bare]["fields"]["nature"] == "internal":
            outputs[name] = (bare, 0)
    for name in mg5["couplings"]:
        piece = re.fullmatch(r"(\w+?)(?:_(\d)eps)?", name)
        if piece[1] in couplings or piece[1] in anonymous:
            outputs[name] = (piece[1], -int(piece[2] or 0))
    return outputs, anonymous


@pytest.mark.loop_physics
@pytest.mark.parametrize("case", sorted(ORACLE))
def test_parameters_and_couplings_equal_mg5_complex_mass_model(
    case, request, madgraph_source, loop_native, tmp_path
):
    """At the card point the transformed model's parameters, pole masses, CT
    parameters, couplings and field counterterms (every Laurent piece MG5
    writes) equal MG5_aMC 3.7.0's own complex-mass model
    (``tools/loop_reference/mg5_cms_model.py``: its ``import model`` with
    ``set complex_mass_scheme True``, evaluated by its Python model reader)
    to 1e-12, binary64 on both sides; every MG5 coupling is matched."""

    from pyamplicol.loops.model_restrictions import prepare_loop_restrictions
    from pyamplicol.loops.scalar_bindings import (
        NativeScalarBindings,
        prepare_scalar_bindings,
    )

    fixture, restriction, fixed = ORACLE[case]
    ufo = Path(request.getfixturevalue(fixture))
    mg5 = _mg5_values(madgraph_source, ufo, restriction, fixed, tmp_path)
    scheme = complex_mass_scheme(
        load_raw_loop_model(str(ufo), restriction=restriction),
        fixed_parameters=fixed,
    )
    assert sorted(mg5["cms_masses"]) == sorted(
        "CMASS_mdl_" + mass for mass in scheme.masses
    )
    raw = prepare_loop_restrictions(scheme.derived, simplify=False).derived
    outputs, anonymous = _mg5_outputs(raw, mg5)
    assert set(mg5["couplings"]) <= outputs.keys()
    plan = prepare_scalar_bindings(
        raw,
        tuple(outputs.values()),
        anonymous_definitions=anonymous,
        fixed_parameters=fixed,
    )
    values = NativeScalarBindings(plan, arithmetic="double").values.to_decimal()
    found = dict(zip(plan.outputs, values, strict=True))
    worst, complex_values = (0.0, ""), 0
    for name, key in outputs.items():
        reference = complex(
            *map(float, mg5["parameters"].get(name) or mg5["couplings"][name])
        )
        value = complex(*map(float, found[key]))
        scale = abs(reference)
        error = abs(value - reference) / scale if scale else abs(value)
        worst = max(worst, (error, name))
        complex_values += abs(reference.imag) > 1e-9 * scale
    assert worst[0] <= TOLERANCE, worst
    # Discriminating: the pole masses make many values complex.
    assert complex_values > (300 if case.startswith("gmu") else 20)


@pytest.mark.loop_physics
def test_pole_mass_bindings_through_the_structural_model(cms, loop_native):
    """The derived model lowers through the covariant structure (the loop
    side's propagators take the pole masses) and binds natively: CMASS_MW**2
    = MW**2 - i MW WW in every carrier, the zero-width update reuses the plan
    and gives real values, and the Yukawa input is no runtime parameter."""

    pytest.importorskip("symbolica")
    from symbolica import S

    from pyamplicol.loops.covariant_model import build_covariant_structure
    from pyamplicol.loops.model_restrictions import prepare_loop_restrictions
    from pyamplicol.loops.scalar_bindings import NativeScalarBindings

    raw = prepare_loop_restrictions(cms.derived).derived
    structure = build_covariant_structure(raw, compatibility="gmu_particle_anti")
    for pdg, alias in (
        (23, "CMASS_MZ"),
        (24, "CMASS_MW"),
        (9000003, "CMASS_MW"),
        (6, "CMASS_MT"),
    ):
        assert structure.native.particle_by_pdg(pdg).mass == S(f"UFO::{alias}")
    outputs = ("CMASS_MW", "sw", "ymt", "CMASS_MT", "CMSParam", "cWcft_UV_EW_R")
    plan = structure.scalar_bindings(tuple((name, 0) for name in outputs))
    assert "ymt" not in plan.external_names
    for arithmetic, precision, tolerance in (
        ("double", None, "3e-12"),
        ("double-double", None, "2e-27"),
        ("arbitrary", 80, "1e-70"),
    ):
        runtime = NativeScalarBindings(
            plan, arithmetic=arithmetic, precision=precision, parameters={}
        )
        with localcontext() as context:
            context.prec = 110
            values = runtime.values.to_decimal()
            real, imaginary = map(Decimal, values[0])
            mass, width = Decimal("80.419"), Decimal("2.09291")
            assert abs(real * real - imaginary * imaginary - mass * mass) < Decimal(
                tolerance
            )
            assert abs(2 * real * imaginary + mass * width) < Decimal(tolerance)
            assert values[2] == values[3] and tuple(map(Decimal, values[4])) == (1, 0)
    zero = runtime.update({"WW": "0", "WZ": "0", "WT": "0"})
    assert all(Decimal(pair[1]) == 0 for pair in zero.values.to_decimal()[:4])
    sw = complex(*map(float, values[1]))
    assert abs(sw.imag) > 1e-4 and abs(cmath.phase(sw)) < 0.1


@pytest.mark.loop_physics
def test_tree_model_view_equals_mg5_complex_mass_model(
    loop_gmu_ufo, madgraph_source, tmp_path
):
    """The compiled tree model under the scheme (``complex_mass_model``: the
    loader's restricted payload with the same decisions, compiled again)
    holds MG5's values: every internal parameter and coupling of the
    payload (the loader's own evaluation) equals MG5's complex-mass model to
    1e-12; the pole-mass particles propagate with width ZERO and the former
    Yukawa input is no external parameter; the raw sidecar is the scheme's."""

    from pyamplicol.loops.model_cms import complex_mass_model
    from pyamplicol.models.loading import compile_model_source

    compiled = compile_model_source(
        loop_gmu_ufo,
        cache_dir=tmp_path / "model-cache",
        require_supported=False,
        gauge="feynman",
    )
    model, scheme = complex_mass_model(compiled, cache_dir=tmp_path / "model-cache")
    assert model.raw_loop_model is scheme.derived and model.ir.gauge == "feynman"
    assert "ymt" not in model.parameter_defaults and "MT" in model.parameter_defaults
    particles = {record.name: record for record in model.ir.particles}
    for name, mass in (("t", "CMASS_MT"), ("Z", "CMASS_MZ"), ("G0", "CMASS_MZ")):
        assert (str(particles[name].mass), str(particles[name].width)) == (mass, "ZERO")
    mg5 = _mg5_values(madgraph_source, Path(loop_gmu_ufo), "default", {}, tmp_path)
    compared = 0
    for kind, records in (("parameters", "parameters"), ("couplings", "couplings")):
        for record in model.model[records]:
            if kind == "parameters" and record["nature"] == "external":
                continue
            name = record["name"]
            key = (
                name.replace("CMASS_", "CMASS_mdl_")
                if name.startswith("CMASS_")
                else name
                if kind == "couplings" or name in mg5[kind]
                else f"mdl_{name}"
            )
            reference = complex(*map(float, mg5[kind][key]))
            value = complex(*record["value"])
            scale = abs(reference)
            error = abs(value - reference) / scale if scale else abs(value)
            assert error <= TOLERANCE, (name, value, reference)
            compared += 1
    assert compared > 100
