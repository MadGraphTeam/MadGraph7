# SPDX-License-Identifier: 0BSD
"""Two-point counterterms beyond loop_sm: mixing, chiral and chirality-flipping
terms as clone lines (design §6.1, ``models.ct_line_insertions``).

loop_sm has only diagonal two-point counterterms, with a multiple of the
slashed momentum on its massless quarks. The test model is loop_sm with
synthetic rules that have the other forms:

* ``V_PYAMPLICOL_TEST_AZ``: photon-Z mixing (``R2_GG_1..3``), a mixing line
  between self-conjugate fields with different propagators. The test model
  couples the Z to quarks with the vector structure ``FFV1`` only (its
  ordinary and counterterm vertices keep their first coupling): the QCD
  loops of the slice then hold pure vector couplings (the axial ones need
  the electroweak sign oracles, WP3.2);
* ``V_PYAMPLICOL_TEST_UU``: chiral (``p-slash P_L``, ``p-slash P_R`` with
  different couplings) and chirality-flipping (``P_L``) terms on the
  massless u quark, whose lines the lowered model then propagates as Dirac
  lines;
* ``V_PYAMPLICOL_TEST_DS`` and ``V_PYAMPLICOL_TEST_SD``: d-s mixing with a
  multiple of the slashed momentum, a mixing line between Weyl fields;
* ``V_PYAMPLICOL_TEST_TT``: a lone chiral projector (``P_L``) on the massive
  top with a complex coupling, the form of the electroweak chirality-
  flipping mass counterterms (``l_vevexMass10``): its clone line's
  model-supplied propagator holds a projector between the top's own
  propagators. In g g > t t~ g its P_L - P_R difference is a gamma^5 on the
  internal top line, whose interference with the Born is the parity-odd
  epsilon(p1, p2, p3, p4) part, non-zero at a 2 -> 3 point with a complex
  coupling: the O6 id ``gg_ttxg`` fails if the line takes the other
  chirality (WP3.2 EWI-4b: projectors in model-supplied propagators).

Oracles: the slice with its counterterms from the recursion equals the slice
with every rule closed at sites (O6), and, without synthetic rules, a lowered
model that propagates the u quark as a Dirac field gives the same closure as
the Weyl one and the frozen legacy components (O1).
"""

from __future__ import annotations

import shutil

import pytest

from tools.loop_reference import legacy_capture

from .test_loop_v2_ct_closure import (  # noqa: F401 (fixtures)
    SLICE_CARRIERS,
    SLICE_PARAMETERS,
    SLICE_SCALE_SQUARED,
    SLICE_TOLERANCE,
    _assert_closure_equals_sites_and_legacy,
    _certify_furry_zeros,
    _route_differences,
    _slice_point,
    closure_ct_model,
    ct_slice,
)

pytestmark = pytest.mark.loop_physics

SYNTHETIC_LORENTZ = """

PYAMPLICOL_TEST_QQL = Lorentz(name = 'PYAMPLICOL_TEST_QQL',
                              spins = [ 2, 2 ],
                              structure = 'P(-1,1)*Gamma(-1,2,-2)*ProjM(-2,1)')

PYAMPLICOL_TEST_QQR = Lorentz(name = 'PYAMPLICOL_TEST_QQR',
                              spins = [ 2, 2 ],
                              structure = 'P(-1,1)*Gamma(-1,2,-2)*ProjP(-2,1)')

PYAMPLICOL_TEST_QQM = Lorentz(name = 'PYAMPLICOL_TEST_QQM',
                              spins = [ 2, 2 ],
                              structure = 'ProjM(2,1)')
"""

SYNTHETIC_COUPLINGS = """

PYAMPLICOL_TEST_AZ1 = Coupling(name = 'PYAMPLICOL_TEST_AZ1',
                               value = 'complex(0,1)*G**2/(7*cmath.pi**2)',
                               order = {'QCD':2})

PYAMPLICOL_TEST_AZ2 = Coupling(name = 'PYAMPLICOL_TEST_AZ2',
                               value = '-complex(0,1)*G**2/(11*cmath.pi**2)',
                               order = {'QCD':2})

PYAMPLICOL_TEST_AZ3 = Coupling(name = 'PYAMPLICOL_TEST_AZ3',
                               value = 'complex(0,1)*G**2*MZ**2/(13*cmath.pi**2)',
                               order = {'QCD':2})

PYAMPLICOL_TEST_UUL = Coupling(name = 'PYAMPLICOL_TEST_UUL',
                               value = 'complex(0,1)*G**2/(17*cmath.pi**2)',
                               order = {'QCD':2})

PYAMPLICOL_TEST_UUR = Coupling(name = 'PYAMPLICOL_TEST_UUR',
                               value = '-complex(0,1)*G**2/(19*cmath.pi**2)',
                               order = {'QCD':2})

PYAMPLICOL_TEST_UUM = Coupling(name = 'PYAMPLICOL_TEST_UUM',
                               value = 'complex(0,1)*G**2*MZ/(23*cmath.pi**2)',
                               order = {'QCD':2})

PYAMPLICOL_TEST_DS = Coupling(name = 'PYAMPLICOL_TEST_DS',
                              value = 'complex(0,1)*G**2/(29*cmath.pi**2)',
                              order = {'QCD':2})

PYAMPLICOL_TEST_TTM = Coupling(name = 'PYAMPLICOL_TEST_TTM',
                               value = '(1+2*complex(0,1))*G**2*MT/(31*cmath.pi**2)',
                               order = {'QCD':2})
"""

SYNTHETIC_VERTICES = """

V_PYAMPLICOL_TEST_AZ = CTVertex(name = 'V_PYAMPLICOL_TEST_AZ',
               particles = [ P.A, P.Z ],
               color = [ '1' ],
               lorentz = [ L.R2_GG_1, L.R2_GG_2, L.R2_GG_3 ],
               loop_particles = [ [[P.G]] ],
               couplings = {(0,0,0):C.PYAMPLICOL_TEST_AZ1,
                            (0,1,0):C.PYAMPLICOL_TEST_AZ2,
                            (0,2,0):C.PYAMPLICOL_TEST_AZ3},
               type = 'UVtree')

V_PYAMPLICOL_TEST_UU = CTVertex(name = 'V_PYAMPLICOL_TEST_UU',
               particles = [ P.u__tilde__, P.u ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.PYAMPLICOL_TEST_QQL, L.PYAMPLICOL_TEST_QQR,
                           L.PYAMPLICOL_TEST_QQM ],
               loop_particles = [[[P.u,P.G]]],
               couplings = {(0,0,0):C.PYAMPLICOL_TEST_UUL,
                            (0,1,0):C.PYAMPLICOL_TEST_UUR,
                            (0,2,0):C.PYAMPLICOL_TEST_UUM},
               type = 'R2')

V_PYAMPLICOL_TEST_DS = CTVertex(name = 'V_PYAMPLICOL_TEST_DS',
               particles = [ P.d__tilde__, P.s ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.R2_QQ_1 ],
               loop_particles = [ [[P.G]] ],
               couplings = {(0,0,0):C.PYAMPLICOL_TEST_DS},
               type = 'UVtree')

V_PYAMPLICOL_TEST_SD = CTVertex(name = 'V_PYAMPLICOL_TEST_SD',
               particles = [ P.s__tilde__, P.d ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.R2_QQ_1 ],
               loop_particles = [ [[P.G]] ],
               couplings = {(0,0,0):C.PYAMPLICOL_TEST_DS},
               type = 'UVtree')

V_PYAMPLICOL_TEST_TT = CTVertex(name = 'V_PYAMPLICOL_TEST_TT',
               particles = [ P.t__tilde__, P.t ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.PYAMPLICOL_TEST_QQM ],
               loop_particles = [[[P.t,P.G]]],
               couplings = {(0,0,0):C.PYAMPLICOL_TEST_TTM},
               type = 'R2')

# The Z couples to quarks through the vector structure only.
for _vertex in [*all_vertices, *all_CTvertices]:
    _pdgs = [abs(_particle.pdg_code) for _particle in _vertex.particles]
    if 23 in _pdgs and any(_pdg <= 6 for _pdg in _pdgs):
        _vertex.lorentz = [L.FFV1]
        _vertex.couplings = {
            _key: _coupling
            for _key, _coupling in _vertex.couplings.items()
            if _key[1] == 0
        }
"""


def synthetic_two_point_ufo(loop_sm_ufo, root):
    """loop_sm with the synthetic two-point rules, written below ``root``."""

    source = root / "loop_sm_two_point"
    shutil.copytree(
        loop_sm_ufo, source, ignore=shutil.ignore_patterns("__pycache__", "*.pkl")
    )
    for name, text in (
        ("lorentz.py", SYNTHETIC_LORENTZ),
        ("CT_couplings.py", SYNTHETIC_COUPLINGS),
        ("CT_vertices.py", SYNTHETIC_VERTICES),
    ):
        with (source / name).open("a", encoding="utf-8") as stream:
            stream.write(text)
    return source


@pytest.fixture(scope="module")
def two_point_model(loop_sm_ufo, tmp_path_factory):
    """The synthetic model compiled once, with loop_sm's scalar-arity view."""

    from dataclasses import replace

    from pyamplicol.models.loading import compile_model_source
    from tools.loop_reference.model_compatibility import (
        prepare_loop_sm_scalar_declarations,
    )

    root = tmp_path_factory.mktemp("two-point-model")
    model = compile_model_source(
        synthetic_two_point_ufo(loop_sm_ufo, root), cache_dir=root / "cache"
    )
    model = replace(
        model,
        raw_loop_model=prepare_loop_sm_scalar_declarations(
            model.raw_loop_model
        ).derived,
        _serialized_path=None,
        _prepared_bundle=None,
    )
    return model.write(root / "loop-sm-two-point.json")


SYNTHETIC_PREFIX = "V_PYAMPLICOL_TEST_"


def _synthetic(rules) -> dict[str, str]:
    """Prepared rule name -> synthetic UFO vertex name."""

    out = {}
    for name, rule in rules.items():
        vertices = {entry.vertex_name for entry in rule.entries}
        if any(vertex.startswith(SYNTHETIC_PREFIX) for vertex in vertices):
            (out[name],) = vertices
    return out


@pytest.fixture(scope="module")
def two_point_lowering(two_point_model):
    """The synthetic model, its prepared rules and its closure lowering."""

    from pyamplicol.loops.compiler import prepare_loop_request_model
    from pyamplicol.loops.v2.ct_recursion import lower_model_counterterms
    from pyamplicol.models.loading import load_compiled_model

    compiled = load_compiled_model(two_point_model)
    prepared = prepare_loop_request_model(
        compiled, fixed_parameters=legacy_capture.FIXED_PARAMETERS
    )
    model = lower_model_counterterms(
        compiled,
        prepared.counterterms.rules,
        fixed_parameters=prepared.structure.fixed_parameters,
        multi_trace=True,
    )
    return compiled, prepared, model


def test_two_point_rules_become_clone_lines(two_point_lowering):
    """Mixing, chiral and chirality-flipping two-point rules are lowered: the
    photon-Z line is an oriented clone pair (C read as the Z, C~ as the
    photon) with the propagators of both fields; the u quark's chiral and
    flipping terms make u a Dirac field of the lowered model, with a
    model-supplied line; the d-s line between Weyl fields keeps the
    massless propagator and carries kappa on the copies of C~."""

    from pyamplicol.loops.v2.ct_recursion import rule_key
    from pyamplicol.models.contracts import MODEL_SUPPLIED_PROPAGATOR_SOURCE
    from pyamplicol.models.ct_line_insertions import _engine_denominator

    _compiled, prepared, model = two_point_lowering
    synthetic = _synthetic(prepared.counterterms.rules)
    assert set(synthetic.values()) == {
        f"{SYNTHETIC_PREFIX}{name}" for name in ("AZ", "UU", "DS", "SD", "TT")
    }
    keys = {rule_key(rule): name for name, rule in prepared.counterterms.rules.items()}
    assert not {keys.get(key) for key in model.residual} & set(synthetic)
    lowered = model.lowered
    payload = lowered.compiled.model
    particles = {item["name"]: item for item in payload["particles"]}
    propagators = {item["particle"]: item for item in payload["propagators"]}

    def line(fields):
        return {
            clone
            for structure, clone in lowered.insertions.items()
            if structure.particles == fields
        }

    (az,) = line(("A", "Z"))
    anti = particles[az]["antiname"]
    assert anti != az and particles[anti]["antiname"] == az
    assert (lowered.line_fields[az], lowered.line_fields[anti]) == ("Z", "A")
    for clone, (entering, leaving) in ((az, ("A", "Z")), (anti, ("Z", "A"))):
        record = propagators[clone]
        assert record["pyamplicol_source"] == MODEL_SUPPLIED_PROPAGATOR_SOURCE
        assert record["denominator"] == "*".join(
            f"({_engine_denominator(propagators[name], particles[name])})"
            for name in (entering, leaving)
        )
        assert particles[clone]["mass"] == "ZERO"
    for rule in payload["vertex_rules"]:
        name = str(rule["name"])
        for clone, field in ((az, "Z"), (anti, "A")):
            if name.startswith(f"{clone}__"):
                original = next(
                    item
                    for item in payload["vertex_rules"]
                    if item["name"] == name.split("__", 1)[1]
                )
                assert field in original["particles"]
                assert clone in rule["particles"]

    assert lowered.dirac_fields == {"u", "u~"}
    for field in ("u", "u~"):
        assert propagators[field]["pyamplicol_source"] == (
            MODEL_SUPPLIED_PROPAGATOR_SOURCE
        )
    (uu,) = line(("u~", "u"))
    assert propagators[uu]["pyamplicol_source"] == MODEL_SUPPLIED_PROPAGATOR_SOURCE
    assert uu not in lowered.line_factors and f"{uu}~" not in lowered.line_factors

    (tt,) = {
        clone
        for structure, clone in lowered.insertions.items()
        if structure.lorentz == "PYAMPLICOL_TEST_QQM"
        and structure.particles == ("t~", "t")
    }
    assert propagators[tt]["pyamplicol_source"] == MODEL_SUPPLIED_PROPAGATOR_SOURCE
    assert "ProjM" in propagators[tt]["numerator"]

    (sd,) = line(("s~", "d"))
    assert (lowered.line_fields[sd], lowered.line_fields[f"{sd}~"]) == ("s", "d~")
    assert propagators[sd]["pyamplicol_source"] != MODEL_SUPPLIED_PROPAGATOR_SOURCE
    assert set(lowered.line_factors) >= {f"{sd}~"} and sd not in lowered.line_factors


# ------------------------------------------------------------ slice-level O6

# ``name -> (process, Born QCD, QED, leg masses, synthetic vertices admitted)``.
# u u~ -> e- e+ (a): the photon-Z line between the quark and lepton lines,
# the u quark's sources and lines as Dirac spinors; with the photon, the
# chiral and flipping u terms on its internal line. g g -> u u~ g: the u line
# terms with a gluon at leg 1 (the line can meet the root). A gluon with the
# photon-Z line adds the g g Z R2, whose Levi-Civita cell cancels between the
# quark doublets (``test_loop_v2_ct_cancellation``). With leg 1 the u~
# (u~ u -> e- e+ a) the internal u line is the other clone of its line. The
# chiral u terms need both orders of the legs: a clone line's propagator
# reads Pi's spinor chain from its receiving end (``ct_line_insertions.
# _reversed_chain``). g g -> t t~ g: the lone P_L on the internal top lines,
# both clones (module docstring).
TWO_POINT_O6 = {
    "uux_emep": ("u u~ > e- e+", 0, 2, (0,) * 4, {"AZ"}),
    "uux_emepa": ("u u~ > e- e+ a", 0, 3, (0,) * 5, {"AZ", "UU"}),
    "uxu_emepa": ("u~ u > e- e+ a", 0, 3, (0,) * 5, {"AZ", "UU"}),
    "gg_uuxg": ("g g > u u~ g", 3, 0, (0,) * 5, {"UU"}),
    "gg_ttxg": ("g g > t t~ g", 3, 0, (0, 0, 173, 173, 0), {"TT"}),
}


@pytest.fixture(scope="module")
def two_point_routes(two_point_model, loop_native):
    """``(name, counterterms) -> LoopSlice`` of the synthetic model."""

    from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest
    from pyamplicol.models.loading import load_compiled_model

    compiled = load_compiled_model(two_point_model)
    built = {}

    def build(name: str, counterterms: str):
        if (name, counterterms) not in built:
            process, qcd, qed, _masses, _admitted = TWO_POINT_O6[name]
            built[name, counterterms] = LoopSlice.generate(
                compiled,
                SliceRequest(
                    process,
                    {"QCD": qcd, "QED": qed},
                    {"QCD": qcd + 2, "QED": qed},
                    fixed_parameters=legacy_capture.FIXED_PARAMETERS,
                    counterterms=counterterms,
                ),
                loop_native,
            )
        return built[name, counterterms]

    return build


@pytest.mark.parametrize("carrier", SLICE_CARRIERS)
@pytest.mark.parametrize("name", tuple(TWO_POINT_O6))
def test_two_point_counterterms_from_recursion_equal_sites(
    two_point_routes, monkeypatch, name, carrier
):
    """O6 on the slice of the synthetic model: every component with the
    counterterms from the recursion equals the request with every rule at
    sites; the synthetic rules are lowered and admitted, and they change the
    counterterms (the sites without them differ)."""

    from dataclasses import replace

    from pyamplicol.loops.v2.assemble import Precision

    _certify_furry_zeros(monkeypatch)
    recursion = two_point_routes(name, "recursion")
    program = recursion.counterterm_program
    synthetic = _synthetic(recursion.prepared.counterterms.rules)
    assert not set(synthetic) & set(program.recursion.residual)
    admitted = {
        synthetic[rule].removeprefix(SYNTHETIC_PREFIX)
        for rule, alternatives in program.recursion.admission.items()
        if rule in synthetic and alternatives
    }
    assert admitted >= TWO_POINT_O6[name][4]
    masses = TWO_POINT_O6[name][3]
    worst = _route_differences(
        two_point_routes, name, carrier, SLICE_PARAMETERS, masses=masses
    )
    assert all(value <= SLICE_TOLERANCE[carrier] for value in worst.values()), worst
    if carrier != "double":
        return
    sites = two_point_routes(name, "sites")
    # The site terms are built on first use (``LoopSlice.site_terms``): a
    # copy with a fresh cache holding the terms without the synthetic rules.
    plain = replace(
        sites,
        _cache={
            "site-terms": {
                site: terms
                for site, terms in sites.site_terms.items()
                if site.rule_name not in synthetic
            }
        },
    )
    point = _slice_point(masses)
    mine, without = (
        route.evaluate(
            point,
            SLICE_SCALE_SQUARED,
            parameters=SLICE_PARAMETERS,
            precision=Precision.double(),
        )
        for route in (sites, plain)
    )
    changed = {
        component
        for component in ("uv_local", "r2")
        for sector, laurent in without.components[component].items()
        if any(
            abs(float(a) - float(b)) > 1e-6 * abs(float(b))
            for a, b in zip(
                mine.components[component][sector].coefficients,
                laurent.coefficients,
                strict=True,
            )
        )
    }
    expected = {"AZ": "uv_local", "UU": "r2", "TT": "r2"}
    assert changed >= {expected[vertex] for vertex in TWO_POINT_O6[name][4]}


# ------------------------------------------------ Dirac lines of Weyl fields


@pytest.mark.parametrize("name", ("dy", "uux_ggg", "uux_uuxg"))
def test_dirac_quark_lines_give_the_weyl_closure(
    ct_slice,  # noqa: F811
    closure_ct_model,  # noqa: F811
    name,
):
    """A lowered model that propagates the massless u quark as a Dirac field
    (sources, currents and clone lines; what a chiral or chirality-flipping
    two-point term requires) closes to the same CT amplitudes: O6 against the
    sites and O1 against the frozen legacy components, in binary64."""

    from pyamplicol.models.contracts import MODEL_SUPPLIED_PROPAGATOR_SOURCE

    ct = ct_slice(name)
    model = closure_ct_model(ct.vetoed, dirac=("u",))
    lowered = model.lowered
    assert lowered.dirac_fields == {"u", "u~"}
    sources = {
        item["particle"]: item.get("pyamplicol_source")
        for item in lowered.compiled.model["propagators"]
    }
    assert sources["u"] == sources["u~"] == MODEL_SUPPLIED_PROPAGATOR_SOURCE
    assert sources["d"] != MODEL_SUPPLIED_PROPAGATOR_SOURCE
    # No Weyl factor on the u line: its clones are model-supplied lines.
    u_line = lowered.insertions[
        next(s for s in lowered.insertions if s.particles == ("u~", "u"))
    ]
    assert u_line not in lowered.line_factors
    assert f"{u_line}~" not in lowered.line_factors
    assert sources[u_line] == MODEL_SUPPLIED_PROPAGATOR_SOURCE
    _assert_closure_equals_sites_and_legacy(ct, model, "double")
