# SPDX-License-Identifier: 0BSD
"""Exact source-plan symmetries; numeric defaults never prove a relation."""

import json
from dataclasses import replace
from fractions import Fraction
from types import ModuleType
from types import SimpleNamespace as Record

import pytest

from pyamplicol.loops.flavor_mappings import RetainedFlavorPlan, prove_flavor_relabeling

pytestmark = pytest.mark.loop_physics

SWAP = {1: 3, 3: 1, -1: -3, -3: -1}


def _plan(*, change=None, fixed=None, yukawa=False):
    pytest.importorskip("symbolica")
    from pyamplicol.loops.counterterms import prepare_counterterm_model
    from pyamplicol.loops.covariant_model import build_covariant_structure
    from pyamplicol.loops.model import build_loop_model
    from pyamplicol.models.loop_import import capture_raw_loop_model

    module = ModuleType("flavor_symmetry_fixture")
    module.gauge = [1]
    parameters = {}
    for name, nature, value in (
        ("ZERO", "internal", "0"),
        ("g", "external", "1"),
        ("MD", "external", "0"),
        ("MS", "external", "0"),
        ("yd", "external", "0"),
        ("ys", "external", "0"),
        ("MU_R", "external", "1000"),
        ("CD", "internal", "3"),
        ("CS", "internal", "3"),
    ):
        parameters[name] = Record(name=name, nature=nature, type="real", value=value)
    module.all_parameters = list(parameters.values())
    module.all_orders = [
        Record(name="QCD", expansion_order=99, hierarchy=1, perturbative_expansion=1),
        Record(name="QED", expansion_order=99, hierarchy=2, perturbative_expansion=1),
    ]
    particles = {}
    for pdg, name, antiname, spin, color, charge in (
        (1, "d", "d~", 2, 3, Fraction(-1, 3)),
        (-1, "d~", "d", 2, -3, Fraction(1, 3)),
        (3, "s", "s~", 2, 3, Fraction(-1, 3)),
        (-3, "s~", "s", 2, -3, Fraction(1, 3)),
        (21, "g", "g", 3, 8, 0),
        (25, "h", "h", 1, 1, 0),
    ):
        particles[name] = Record(
            name=name,
            antiname=antiname,
            pdg_code=pdg,
            spin=spin,
            color=color,
            charge=charge,
            mass=parameters["ZERO"],
            width=parameters["ZERO"],
            texname=name,
            antitexname=antiname,
        )
    module.all_particles = list(particles.values())
    for name, pdg in (("d", 1), ("s", 3)):
        particles[name].loop_particles = [[[pdg, 21], [pdg, 21]]]
        particles[name].counterterm = {(1, 0, 0): {-1: "g**2", 0: "g**2/2"}}
    ffv = Record(name="FFV", spins=[2, 2, 3], structure="Gamma(3,2,1)")
    ffs = Record(name="FFS", spins=[2, 2, 1], structure="Identity(2,1)")
    module.all_lorentz = [ffv, ffs]
    couplings = {}
    for name, value, order in (
        ("G", "complex(0,1)*g", {"QCD": 1}),
        ("YD", "yd", {"QED": 1}),
        ("YS", "ys", {"QED": 1}),
        ("RD", "complex(0,1)*g**3", {"QCD": 3}),
        ("RS", "complex(0,1)*g**3", {"QCD": 3}),
        ("UD", {-1: "g**3", 0: "g**3/2", 2: "g**3/7"}, {"QCD": 3}),
        ("US", {-1: "g**3", 0: "g**3/2", 2: "g**3/7"}, {"QCD": 3}),
    ):
        couplings[name] = Record(name=name, value=value, order=order)
    module.all_couplings = list(couplings.values())
    module.all_vertices = []
    module.all_CTvertices = []
    for name, suffix in (("d", "D"), ("s", "S")):
        legs = [particles[name + "~"], particles[name], particles["g"]]
        module.all_vertices.append(
            Record(
                name="V" + suffix,
                particles=legs,
                color=["T(3,2,1)"],
                lorentz=[ffv],
                couplings={(0, 0): couplings["G"]},
            )
        )
        module.all_vertices.append(
            Record(
                name="Y" + suffix,
                particles=[*legs[:2], particles["h"]],
                color=["Identity(1,2)"],
                lorentz=[ffs],
                couplings={(0, 0): couplings["Y" + suffix]},
            )
        )
        for kind, prefix in (("R2", "R"), ("UVtree", "U")):
            module.all_CTvertices.append(
                Record(
                    name=prefix + suffix,
                    type=kind,
                    particles=legs,
                    color=["T(3,2,1)"],
                    lorentz=[ffv],
                    couplings={(0, 0, 0): couplings[prefix + suffix]},
                    loop_particles=[[[particles[name]], [particles[name]]]]
                    if kind == "R2"
                    else [],
                )
            )
    if change == "masses":
        for name, mass in (("d", "MD"), ("s", "MS")):
            particles[name].mass = particles[name + "~"].mass = parameters[mass]
    elif change == "charge":
        particles["s"].charge, particles["s~"].charge = Fraction(2, 3), Fraction(-2, 3)
    elif change == "uv-log":
        couplings["UD"].value[0] = "g**3*cmath.log(MD+MU_R)"
        couplings["US"].value[0] = "g**3*cmath.log(MS+MU_R)"
    elif change == "positive-epsilon":
        couplings["US"].value[2] = "g**3/8"
    elif change == "loop-multiplicity":
        module.all_CTvertices[2].loop_particles[0].pop()
    elif change == "field":
        particles["s"].counterterm[(1, 0, 0)][0] = "g**2/3"
    elif change == "formal-uv":
        for suffix in ("D", "S"):
            couplings["U" + suffix].value[2] = "g**3*C" + suffix
    elif change in ("formal-field", "formal-diagnostic", "formal-field-zero"):
        if change != "formal-field-zero":
            for name, suffix in (("d", "D"), ("s", "S")):
                particles[name].counterterm[(1, 0, 0)][0] = "g**2*C" + suffix
        if change == "formal-diagnostic":
            parameters["CS"].value = "3 + 0*(1 if yd < 0 else 2)"
        elif change == "formal-field-zero":
            for name, suffix in (("d", "D"), ("s", "S")):
                particles[name].counterterm[(1, 0, 0)][2] = "g**2*(C" + suffix + "-3)"
    raw = capture_raw_loop_model(module)
    structure = build_covariant_structure(raw, fixed_parameters=fixed)
    source = build_loop_model(raw, scalar_program=structure.scalars)
    prepared = prepare_counterterm_model(source, structure.native)
    names = ("VD", "VS", "YD", "YS") if yukawa else ("VD", "VS")
    return RetainedFlavorPlan(structure, prepared, names, ("QCD",))


def _formal_plan(plan, *, unequal=False):
    from symbolica import S

    from pyamplicol.loops.source_color import SourceColorContinuation
    from pyamplicol.loops.wavefunction import prepare_external_field_rules
    from pyamplicol.models.loop_declaration import LoopSchemeDeclaration

    source = plan.counterterms.source
    nc = S("flavour_proof_test::Nc")
    color = SourceColorContinuation(
        source,
        nc,
        {"CD": nc, "CS": nc + (nc - 3 if unequal else 0)},
        frozenset(entry.key for entry in source.kernels),
        frozenset(
            (field.particle_id, field.correction, field.loop_group_index)
            for field in prepare_external_field_rules(source)
        ),
        LoopSchemeDeclaration(
            "HV", "4d", "none", "explicit", "real", "none", "synthetic"
        ),
    )
    return replace(plan, source_color=color)


@pytest.mark.parametrize(
    "change,match",
    [
        ("formal-uv", "Laurent"),
        ("formal-field", "external-field"),
        ("formal-field-zero", "external-field"),
    ],
)
def test_formal_symmetry_does_not_follow_from_equal_nc3_values(change, match):
    plan = _plan(change=change)
    assert prove_flavor_relabeling(plan, SWAP).species
    assert prove_flavor_relabeling(_formal_plan(plan), SWAP).species
    with pytest.raises(ValueError, match=match):
        prove_flavor_relabeling(_formal_plan(plan, unequal=True), SWAP)


def test_formal_field_proof_preserves_active_source_diagnostics():
    plan = _formal_plan(_plan(change="formal-diagnostic"))
    with pytest.raises(ValueError, match="external-field"):
        prove_flavor_relabeling(plan, SWAP)


def test_formal_proof_requires_owned_complete_declaration_and_scheme():
    from symbolica import E

    plan = _formal_plan(_plan())
    other = _formal_plan(_plan())
    with pytest.raises(ValueError, match="retained source owner"):
        replace(plan, source_color=other.source_color)
    for field in ("kernel_keys", "field_keys"):
        incomplete = replace(plan.source_color, **{field: frozenset()})
        with pytest.raises(NotImplementedError, match="no declared"):
            prove_flavor_relabeling(replace(plan, source_color=incomplete), SWAP)
    wrong_scheme = replace(plan.source_color, scheme_parameter_values={"CD": E("4")})
    with pytest.raises(ValueError, match="must remain fixed"):
        prove_flavor_relabeling(replace(plan, source_color=wrong_scheme), SWAP)
    guarded = _formal_plan(_plan(change="formal-diagnostic"))
    guarded_scheme = replace(
        guarded.source_color, scheme_parameter_values={"CS": E("3")}
    )
    with pytest.raises(ValueError, match="inactive source diagnostics"):
        prove_flavor_relabeling(replace(guarded, source_color=guarded_scheme), SWAP)


def test_exact_retained_plan_preserves_full_laurent_groups_and_source_owners():
    plan = _plan()
    original = plan.structure.raw.to_dict()
    proof = prove_flavor_relabeling(plan, SWAP)
    assert proof.plan is plan
    assert proof.species == tuple(sorted(SWAP.items()))
    assert (
        proof.ordinary_rule_count,
        proof.counterterm_rule_count,
        proof.external_field_rule_count,
    ) == (2, 4, 2)
    assert not proof.external_transport_validated and not proof.direct_r2_validated
    assert plan.structure.raw.to_dict() == original


@pytest.mark.parametrize(
    "change,match",
    [
        ("masses", "mass, width"),
        ("charge", "charge"),
        ("uv-log", "UV/R2"),
        ("positive-epsilon", "Laurent"),
        ("loop-multiplicity", "loop groups"),
        ("field", "external-field"),
    ],
)
def test_source_changes_cannot_be_hidden_by_equal_defaults(change, match):
    with pytest.raises(ValueError, match=match):
        prove_flavor_relabeling(_plan(change=change), SWAP)


def test_yukawas_are_independent_until_fixed_by_existing_scalar_plan():
    with pytest.raises(ValueError, match="ordinary tensors or couplings"):
        prove_flavor_relabeling(_plan(yukawa=True), SWAP)
    # Keep a nonzero Yukawa so the native structure retains both local rules.
    proof = prove_flavor_relabeling(
        _plan(yukawa=True, fixed={"yd": "0.5", "ys": "0.5"}), SWAP
    )
    assert proof.ordinary_rule_count == 4


def test_only_explicit_generation_constants_tie_mass_aliases():
    plan = _plan(change="masses", fixed={"MD": "0", "MS": "0"})
    assert prove_flavor_relabeling(plan, SWAP).species
    assert plan.structure.fixed_parameters == {"MD": ("0", "0"), "MS": ("0", "0")}


def test_permutation_source_ownership_and_unproved_domains_reject():
    plan = _plan()
    for mapping in ({1: 3}, {1: 3, 3: 1}, {1: 100}):
        with pytest.raises(ValueError):
            prove_flavor_relabeling(plan, mapping)
    with pytest.raises(NotImplementedError, match="charge conjugation"):
        prove_flavor_relabeling(plan, SWAP, charge_conjugated=True)
    with pytest.raises(NotImplementedError, match="model R2"):
        prove_flavor_relabeling(plan, SWAP, r2_strategy="direct")
    with pytest.raises(ValueError, match="source/scalar owner"):
        replace(plan, structure=_plan().structure)


@pytest.mark.parametrize(
    "kind",
    [
        "ordinary",
        "counterterm",
        "prepared-ordinary",
        "propagator",
        "coupling",
        "coupling-order",
    ],
)
def test_actual_prepared_native_overrides_cannot_hide_behind_equal_source(kind):
    from symbolica.community import hepkit as hep

    from pyamplicol.loops.counterterms import prepare_counterterm_model

    plan = _plan()
    native = (
        plan.counterterms.native
        if kind in ("counterterm", "prepared-ordinary", "coupling-order")
        else plan.structure.native
    )
    payload = json.loads(native.to_json())
    if kind in ("ordinary", "counterterm", "prepared-ordinary"):
        rule_name = (
            next(
                rule.name
                for rule in plan.counterterms.rules.values()
                if 3 in rule.particle_pdgs
            )
            if kind == "counterterm"
            else "VS"
        )
        rule = next(
            rule for rule in payload["vertex_rules"] if rule["name"] == rule_name
        )
        old = next(
            item
            for item in payload["lorentz_structures"]
            if item["name"] == rule["lorentz_structures"][0]
        )
        # Independent local override, retaining the original source catalogue.
        new = dict(
            old, name="ASYMMETRIC_LOCAL", structure="2*(" + old["structure"] + ")"
        )
        payload["lorentz_structures"].append(new)
        rule["lorentz_structures"][0] = new["name"]
    elif kind == "coupling-order":
        selected = next(
            rule for rule in plan.counterterms.rules.values() if 3 in rule.particle_pdgs
        )
        alias = selected.couplings[0][0]
        coupling = next(item for item in payload["couplings"] if item["name"] == alias)
        coupling["orders"] = [
            [name, power + (name == "QCD")] for name, power in coupling["orders"]
        ]
    elif kind == "propagator":
        prop = next(item for item in payload["propagators"] if item["particle"] == "s")
        prop["numerator"] = "2*(" + prop["numerator"] + ")"
    else:
        coupling = next(item for item in payload["couplings"] if item["name"] == "G")
        coupling["expression"] = "2*(" + coupling["expression"] + ")"
    changed = hep.Model.from_json(json.dumps(payload))
    if kind in ("counterterm", "prepared-ordinary", "coupling-order"):
        prepared = replace(plan.counterterms, native=changed)
        changed_plan = replace(plan, counterterms=prepared)
    else:
        structure = replace(plan.structure, native=changed)
        prepared = prepare_counterterm_model(plan.counterterms.source, changed)
        changed_plan = replace(plan, structure=structure, counterterms=prepared)
    with pytest.raises(ValueError, match=r"native|UV/R2"):
        prove_flavor_relabeling(changed_plan, SWAP)
