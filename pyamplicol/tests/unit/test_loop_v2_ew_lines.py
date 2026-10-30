# SPDX-License-Identifier: 0BSD
"""QCD corrections with colour-singlet fermion lines and W vertices
(WP3.2 EWI-1, EWI-1b, EWI-2).

Lepton legs (EWI-1). The kernel catalogue takes the loop species the
families route (``assemble._family_loop_species``; B2: a lepton at leg 1 has
no counterterm rule at the root to declare them), and the root closure signs
a colour-singlet fermion line that closes at the root with the tree engine's
merge sign (``conventions.colour_singlet_sign`` in ``ct_sites.SiteClosure``;
B4 at the root):

* e+ e- > u u~ with the photon and the Z is the massless quark form factor;
* e- u > e- u (lepton root) equals u e- > u e- (quark root) and legacy;
* e+ e- > t t~ h, the top Yukawa on loop top lines, equals MadLoop (nightly).

Processes with several colour-singlet fermion lines (EWI-5):
``test_loop_v2_ew_leptons``.

W vertices (EWI-2). A vertex whose particles are not closed under
conjugation has no leg for the tree particles of its currents; the tree
engine reads them through the model's conjugate rule, so the root closure
(B6) and loop segments (B5) take the conjugate rule's couplings and Lorentz
structures (``ct_sites.ConjugateVertices``, ``conventions.TreeReading``):

* u d~ > e+ ve (W at the root and on the loop quark line) is the form factor;
* W and Z production, single top and t-channel W equal stored legacy points
  (``tests/fixtures/loop/v2/ew_lines_legacy.json``);
* u u~ > d d~ with the W exchange of its electroweak Born, QCD corrections,
  equals MadLoop per squared order (loop_sm-full, Wolfenstein CKM);
* with a complex CKM (loop_sm ``zeromass_ckm``) the W on a loop quark line
  gives Catani's poles and the Born closed at an open W vertex is the tree
  runtime's amplitude up to a real sign; reading the vertex's own couplings
  (complex-conjugated, chiral projectors exchanged) misses Catani by 59 % and
  gives the closure the phase V_ub^*/V_ub;
* the counterterm-site oracle (O6) equals the production recursion with W
  vertices at the sites, read through their conjugate rules too.

Z and photon on quark lines (EWI-1b): u u~ > z g equals legacy; u u~ > z z
(nightly); the LC and NLC of u u~ > e- e+ g g are the formal N_c
projections of the full-colour cells (nightly).
"""

from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "loop"
LEGACY = json.loads((FIXTURES / "v2" / "ew_lines_legacy.json").read_text())
C_F = 4 / 3
ALPHA_S = 0.118
ZERO_WIDTHS = {"MB": "0", "WT": "0", "WZ": "0", "WW": "0", "WH": "0"}
PARAMETERS = {"aS": "0.118", "MT": "173"}
TOLERANCE = 1e-12


def _laurent(sectors) -> list[float]:
    total = [0.0, 0.0, 0.0]
    for laurent in sectors.values():
        total = [a + float(b) for a, b in zip(total, laurent.coefficients, strict=True)]
    return total


def _slice(model, native, process, born, loop, fixed=ZERO_WIDTHS, **options):
    from pyamplicol.models.loading import load_compiled_model

    compiled = model if not isinstance(model, Path) else load_compiled_model(model)
    return LoopSlice.generate(
        compiled,
        SliceRequest(process, born, loop, fixed_parameters=dict(fixed), **options),
        native,
    )


def _massless_points() -> tuple[list, list]:
    fixture = json.loads((FIXTURES / "legacy" / "dy.json").read_text())
    return fixture["points"], fixture["scales_squared"]


def _assert_form_factor(slice_, points, scales) -> None:
    """V/B = (alpha_s/2pi) C_F (-2/eps^2 - (3 + 2L)/eps - 8 + pi^2 - 3L - L^2),
    L = log(mu^2/s): one massless quark line through the loop."""

    a = ALPHA_S / (2 * math.pi) * C_F
    for point, scale in zip(points, scales, strict=True):
        result = slice_.evaluate(point, scale, parameters=PARAMETERS)
        s = (float(Fraction(point[0][0])) + float(Fraction(point[1][0]))) ** 2
        log = math.log(float(scale) / s)
        expected = [a * (math.pi**2 - 8 - 3 * log - log**2), a * (-3 - 2 * log), -2 * a]
        born = sum(float(x) for x in result.born.values())
        ratio = [x / born for x in _laurent(result.components["virtual"])]
        for mine, theirs in zip(ratio, expected, strict=True):
            assert abs(mine - theirs) <= TOLERANCE * max(map(abs, expected)), (
                ratio,
                expected,
            )


def _assert_catani(result) -> None:
    """The renormalised virtual's poles equal Catani's operator."""

    assert result.ir_poles
    virtual = result.components["virtual"]
    for sector, (single, double) in result.ir_poles.items():
        coefficients = [float(x) for x in virtual[sector].coefficients]
        scale = max(abs(float(single)), abs(float(double)))
        assert abs(coefficients[1] - float(single)) <= TOLERANCE * scale, sector
        assert abs(coefficients[2] - float(double)) <= TOLERANCE * scale, sector


def _assert_equal_results(mine, theirs) -> None:
    """Born per order (relative) and every virtual sector's coefficients
    (relative to the sector's largest) of two evaluations."""

    assert set(mine.born) == set(theirs.born)
    for order, value in theirs.born.items():
        assert abs(float(mine.born[order]) - float(value)) <= TOLERANCE * abs(
            float(value)
        )
    virtual, reference = mine.components["virtual"], theirs.components["virtual"]
    assert set(virtual) == set(reference)
    for sector, laurent in reference.items():
        values = [float(x) for x in laurent.coefficients]
        found = [float(x) for x in virtual[sector].coefficients]
        scale = max(map(abs, values))
        assert all(
            abs(a - b) <= TOLERANCE * scale for a, b in zip(found, values, strict=True)
        ), (
            sector,
            found,
            values,
        )


# ----------------------------------------------------------- conjugate rules


def _native(particles, rules, couplings):
    return {
        "particles": [
            {"name": n, "antiname": a, "spin": s, "pdg_code": p, "mass": "ZERO"}
            for n, a, s, p in particles
        ],
        "vertex_rules": [
            {
                "name": name,
                "particles": list(fields),
                "color_structures": list(colours),
                "lorentz_structures": ["L"],
                "couplings": [list(row)],
            }
            for name, fields, colours, row in rules
        ],
        "couplings": [{"name": name, "orders": [list(o)]} for name, o in couplings],
    }


PARTICLES = [
    ("u", "u~", 2, 2),
    ("u~", "u", 2, -2),
    ("d", "d~", 2, 1),
    ("d~", "d", 2, -1),
    ("g", "g", 3, 21),
    ("W+", "W-", 3, 24),
    ("W-", "W+", 3, -24),
    ("gh", "gh~", -1, 82),
    ("gh~", "gh", -1, -82),
]


@pytest.mark.loop_fast
def test_conjugate_rules_pair_open_vertices_at_their_orders():
    """A rule closed under conjugation reads itself; [d~ u W-] reads
    [u~ d W+] at the same orders and on the legs of its currents' tree
    particles (incoming field f -> tree particle anti(f)); a missing,
    ambiguous or multi-colour conjugate is refused; a rule with a ghost leg
    reads itself on its own legs, closed under conjugation or not (WP3.2
    EWI-7: the ghost Lagrangian pairs no vertex with a Hermitian
    conjugate)."""

    from pyamplicol.loops.v2.conventions import (
        FermionSignError,
        TreeReading,
        conjugate_vertex_rules,
    )

    qed, qcd = ("GW", ("QED", 1)), ("GS", ("QCD", 1))
    data = _native(
        PARTICLES,
        [
            ("V_g", ("u~", "u", "g"), ("T(3,2,1)",), ("GS",)),
            ("V_dxuW", ("d~", "u", "W-"), ("Identity(1,2)",), ("GW",)),
            ("V_uxdW", ("u~", "d", "W+"), ("Identity(1,2)",), ("GW",)),
            # A conjugate at other orders is no conjugate.
            ("V_dxuWg", ("d~", "u", "W-", "g"), ("T(4,2,1)",), ("GS",)),
            ("V_uxdWg", ("u~", "d", "W+", "g"), ("T(4,2,1)",), ("GW",)),
            ("V_gh", ("gh~", "gh", "W+"), ("1",), ("GW",)),
            ("V_ghx", ("gh", "gh~", "W-"), ("1",), ("GW",)),
        ],
        [qed, qcd],
    )
    pairs = conjugate_vertex_rules(data)
    assert pairs["V_g"] == "V_g"
    assert pairs["V_dxuW"] == "V_uxdW" and pairs["V_uxdW"] == "V_dxuW"
    assert pairs["V_dxuWg"] is None and pairs["V_uxdWg"] is None
    assert pairs["V_gh"] == "V_gh" and pairs["V_ghx"] == "V_ghx"
    reading = TreeReading(data)
    assert reading.vertex("V_gh", (2, 1, 3)) == ("V_gh", (2, 1, 3), None)
    # Loop in d~ (leg 1), loop out u~ (its anti on leg 2), the W- block (leg 3).
    assert reading.vertex("V_dxuW", (1, 2, 3)) == ("V_uxdW", (2, 1, 3), {0: 0})
    assert reading.vertex("V_g", (1, 2, 3)) == ("V_g", (2, 1, 3), None)
    with pytest.raises(FermionSignError, match="no unique conjugate rule"):
        reading.conjugate("V_dxuWg")


@pytest.mark.loop_fast
def test_conjugate_vertices_need_one_conjugate_with_one_colour_structure():
    from pyamplicol.loops.v2.ct_sites import ConjugateVertices

    def entries(name, fields, colours=(0,), order=(("QED", 1),), kind="tree"):
        return [
            SimpleNamespace(
                collection="vertices",
                kind=kind,
                vertex_name=name,
                particle_names=fields,
                order=order,
                color_index=colour,
            )
            for colour in colours
        ]

    data = _native(PARTICLES, [], [])
    open_ = entries("V_dxuW", ("d~", "u", "W-"))
    conjugate = entries("V_uxdW", ("u~", "d", "W+"))
    registry = ConjugateVertices([("V_dxuW", open_), ("V_uxdW", conjugate)], data)
    assert registry.named(open_) == ("V_uxdW", tuple(conjugate))
    assert registry.of(conjugate) == tuple(open_)
    # Another kind (a counterterm rule of the same fields) is not a conjugate.
    with pytest.raises(NotImplementedError, match="0 conjugate rules"):
        registry.of(entries("V_R2", ("d~", "u", "W-"), kind="R2"))
    twice = ConjugateVertices(
        [("a", conjugate), ("b", entries("V_b", ("u~", "d", "W+")))], data
    )
    with pytest.raises(NotImplementedError, match="2 conjugate rules"):
        twice.of(open_)
    several = ConjugateVertices(
        [("c", entries("V_c", ("u~", "d", "W+"), colours=(0, 1)))], data
    )
    with pytest.raises(NotImplementedError, match="several colour structures"):
        several.of(open_)

    # A counterterm vertex split by loop content (one rule per loop group,
    # MG5_aMC's R2 [W- A G+] per quark pair): each rule's conjugate is the
    # conjugate vertex's rule of the same loop content (|PDG| per group).
    def split(name, fields, groups):
        out = []
        for index, group in enumerate(groups):
            (entry,) = entries(name, fields, kind="R2")
            entry.loop_particles = (group,)
            entry.loop_group_index = index
            out.append((f"{name}_{index}", [entry]))
        return out

    plus = split("V_R2p", ("d~", "u", "W-"), ((24, 23), (2, 1), (4, 3)))
    minus = split("V_R2m", ("u~", "d", "W+"), ((-24, 23), (2, 1), (4, 3)))
    pairs = ConjugateVertices([*plus, *minus], data)
    for (_, rule), (name, conjugate) in zip(plus, minus, strict=True):
        assert pairs.named(rule) == (name, tuple(conjugate))
    unpaired = ConjugateVertices([*plus, *split("V_R2m", ("u~", "d", "W+"),
                                                ((6, 5), (6, 3)))], data)  # fmt: skip
    with pytest.raises(NotImplementedError, match="0 conjugate rules"):
        unpaired.named(plus[1][1])


@pytest.mark.loop_fast
def test_born_orders_the_request_leaves_open_are_bounded_by_the_vertices():
    """An order the request does not bound takes its largest tree power
    from the model's vertices (``assemble._tree_order_bounds``): n - 2 for
    renormalisable vertices, 2 (n - 2) with an effective g g H vertex of
    QCD = 2; a two-point vertex carrying the order is refused. A counterterm
    rule admitted for a loop particle no family routes is refused too."""

    from pyamplicol.loops.v2.assemble import (
        _check_declared_loop_species,
        _tree_order_bounds,
    )

    qcd, qed, heft = ("GS", ("QCD", 1)), ("GW", ("QED", 1)), ("GH", ("QCD", 2))
    rules = [
        ("V_g", ("u~", "u", "g"), ("T(3,2,1)",), ("GS",)),
        ("V_W", ("d~", "u", "W-"), ("Identity(1,2)",), ("GW",)),
    ]
    data = _native(PARTICLES, rules, [qcd, qed, heft])
    assert _tree_order_bounds(data, 5, ["QCD", "QED"]) == {"QCD": 3, "QED": 3}
    assert _tree_order_bounds(data, 5, ["QED"]) == {"QED": 3}
    effective = _native(
        PARTICLES,
        [*rules, ("V_ggH", ("g", "g", "W+"), ("1",), ("GH",))],
        [qcd, qed, heft],
    )
    assert _tree_order_bounds(effective, 5, ["QCD"]) == {"QCD": 6}
    mixing = _native(
        PARTICLES, [*rules, ("V_mix", ("W+", "W-"), ("1",), ("GW",))], [qcd, qed, heft]
    )
    assert _tree_order_bounds(mixing, 5, ["QCD"]) == {"QCD": 3}
    with pytest.raises(NotImplementedError, match="tree-level power is unbounded"):
        _tree_order_bounds(mixing, 5, ["QED"])
    site = SimpleNamespace(
        rule=SimpleNamespace(alternatives={0: (6,), 1: (1, 2)}, rule_name="R2_x"),
        rule_name="R2_x",
        alternatives=(0, 1),
    )
    _check_declared_loop_species([site], [1, 2, 6, 21])
    with pytest.raises(NotImplementedError, match=r"loop particles \[6\]"):
        _check_declared_loop_species([site], [1, 2, 21])


# --------------------------------------------------------------- lepton legs


@pytest.mark.loop_physics
def test_lepton_initiated_drell_yan_is_the_massless_form_factor(
    loop_sm_compiled_model, loop_native
):
    """e+ e- > u u~ with the photon and the Z: leg 1 is a lepton, so no
    counterterm rule at the root declares the QCD loop species (B2), and the
    lepton line closes at the root vertex (B4)."""

    slice_ = _slice(
        loop_sm_compiled_model,
        loop_native,
        "e+ e- > u u~",
        {"QCD": 0, "QED": 2},
        {"QCD": 2, "QED": 2},
    )
    _assert_form_factor(slice_, *_massless_points())


@pytest.mark.loop_physics
def test_lepton_and_quark_roots_agree(loop_sm_compiled_model, loop_native):
    """e- u > e- u, rooted at the lepton, equals u e- > u e-, rooted at the
    quark, at the same momenta, and both equal legacy."""

    from pyamplicol.models.loading import load_compiled_model

    compiled = load_compiled_model(loop_sm_compiled_model)
    entry = LEGACY["processes"]["emu_emu"]
    point = entry["point"]
    born, loop = entry["born_orders"], entry["loop_orders"]
    lepton = _slice(compiled, loop_native, "e- u > e- u", born, loop)
    quark = _slice(compiled, loop_native, "u e- > u e-", born, loop)
    swapped = [point[1], point[0], point[3], point[2]]
    first = lepton.evaluate(point, LEGACY["mu_r2"], parameters=PARAMETERS)
    second = quark.evaluate(swapped, LEGACY["mu_r2"], parameters=PARAMETERS)
    _assert_equal_results(first, second)
    _assert_legacy(first, entry)


# ------------------------------------------------------------------ W lines


@pytest.mark.loop_physics
def test_w_drell_yan_is_the_massless_form_factor(loop_sm_compiled_model, loop_native):
    """u d~ > e+ ve: the W vertex at the root and on the loop's quark line,
    both read through the conjugate rule [u~ d W+]."""

    slice_ = _slice(
        loop_sm_compiled_model,
        loop_native,
        "u d~ > e+ ve",
        {"QCD": 0, "QED": 2},
        {"QCD": 2, "QED": 2},
    )
    assert _open_vertex_terms(slice_)
    _assert_form_factor(slice_, *_massless_points())


def _open_vertex_terms(slice_) -> list:
    """Born root terms at a vertex not closed under conjugation (a W
    vertex), whose entries are the conjugate rule's
    (``ct_sites.SiteClosure.read_entries``)."""

    from pyamplicol.loops.v2.conventions import closed_under_conjugation

    data = json.loads(slice_.prepared.structure.native.to_json())
    anti = {p["name"]: p["antiname"] for p in data["particles"]}
    return [
        term
        for terms in slice_.born_terms.values()
        for term in terms
        if not closed_under_conjugation(term.entry.particle_names, anti)
    ]


def _assert_legacy(result, entry) -> None:
    (born,) = result.born.values()
    assert abs(float(born) - entry["born"]) <= TOLERANCE * abs(entry["born"])
    virtual = {str(k): v for k, v in result.components["virtual"].items()}
    assert set(virtual) == set(entry["virtual"])
    for sector, reference in entry["virtual"].items():
        values = [float(x) for x in virtual[sector].coefficients]
        scale = max(map(abs, reference))
        assert all(
            abs(a - b) <= TOLERANCE * scale
            for a, b in zip(values, reference, strict=True)
        ), (sector, values, reference)


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("gu_wpd", marks=pytest.mark.loop_physics),
        pytest.param("uux_zg", marks=pytest.mark.loop_physics),
        *(
            pytest.param(name, marks=pytest.mark.loop_nightly)
            for name in (
                "udx_wpg",
                "uux_wpwm",
                "bu_dt",
                # 5F tW: the chiral W vertex joins the massless b and the
                # massive top (the Dirac-Weyl case, measurement M4); b b~ >
                # w+ w- has that vertex twice, in the top t-channel exchange
                # (ymb = 0 with MB = 0: the default ymb is not unitary).
                "gb_twm",
                "bbx_wpwm",
                "uux_zz",
                "udx_epve",
                "epem_uux",
                "uux_epem",
            )
        ),
    ],
)
def test_electroweak_exchange_equals_legacy(loop_sm_compiled_model, loop_native, name):
    """QCD corrections with W/Z/photon bosons, W vertices on loop quark lines
    and lepton legs against stored public-legacy points (zero widths)."""

    entry = LEGACY["processes"][name]
    slice_ = _slice(
        loop_sm_compiled_model,
        loop_native,
        entry["process"],
        entry["born_orders"],
        entry["loop_orders"],
        entry.get("fixed_parameters", LEGACY["fixed_parameters"]),
    )
    result = slice_.evaluate(entry["point"], LEGACY["mu_r2"], parameters=PARAMETERS)
    _assert_legacy(result, entry)


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("udx_epve", marks=pytest.mark.loop_physics),
        *(
            pytest.param(name, marks=pytest.mark.loop_nightly)
            for name in ("gu_wpd", "bu_dt")
        ),
    ],
)
def test_counterterm_sites_equal_the_recursion_at_w_vertices(
    loop_sm_compiled_model, loop_native, name
):
    """O6 with W vertices at the counterterm sites (the R2 of [d~ u W-] is
    read through [u~ d W+], whose coupling the site scalar plan computes):
    the site oracle equals the production recursion, which equals legacy."""

    from pyamplicol.models.loading import load_compiled_model

    compiled = load_compiled_model(loop_sm_compiled_model)
    entry = LEGACY["processes"][name]
    results = [
        _slice(
            compiled,
            loop_native,
            entry["process"],
            entry["born_orders"],
            entry["loop_orders"],
            LEGACY["fixed_parameters"],
            counterterms=form,
        ).evaluate(entry["point"], LEGACY["mu_r2"], parameters=PARAMETERS)
        for form in ("recursion", "sites")
    ]
    _assert_legacy(results[0], entry)
    assert any(float(x) for x in _laurent(results[1].components["r2"]))
    _assert_equal_results(results[1], results[0])
    assert results[1].components.keys() == results[0].components.keys()
    for component, sectors in results[0].components.items():
        for sector, laurent in sectors.items():
            virtual = results[0].components["virtual"][sector].coefficients
            scale = max(abs(float(x)) for x in virtual)
            found = results[1].components[component][sector].coefficients
            assert all(
                abs(float(a) - float(b)) <= TOLERANCE * scale
                for a, b in zip(found, laurent.coefficients, strict=True)
            ), (component, sector)


@pytest.fixture(scope="module")
def restricted_loop_sm(loop_sm_ufo, tmp_path_factory):
    """loop_sm under one of its own restriction cards (scalar-arity view)."""

    from dataclasses import replace

    from pyamplicol.models.loading import compile_model_source, load_compiled_model
    from tools.loop_reference.model_compatibility import (
        prepare_loop_sm_scalar_declarations,
    )

    built = {}

    def compile_(restriction: str):
        if restriction not in built:
            root = tmp_path_factory.mktemp(f"loop-sm-{restriction}")
            model = compile_model_source(
                loop_sm_ufo, restriction=restriction, cache_dir=root / "cache"
            )
            model = replace(
                model,
                raw_loop_model=prepare_loop_sm_scalar_declarations(
                    model.raw_loop_model
                ).derived,
                _serialized_path=None,
                _prepared_bundle=None,
            )
            built[restriction] = load_compiled_model(model.write(root / "model.json"))
        return built[restriction]

    return compile_


@pytest.mark.loop_physics
def test_mixed_order_born_with_w_exchange_equals_madloop(
    restricted_loop_sm, loop_native
):
    """u u~ > d d~ with QCD and electroweak Born amplitudes: the W exchange
    closes at the root through [u~ d W+] (B6), and its QCD corrections put the
    W on loop quark lines (B5); every squared order equals MadLoop
    (``tools/loop_reference/uudd_mixed_sm_qcd.json``: loop_sm-full with the
    Wolfenstein CKM, Feynman gauge)."""

    reference = json.loads(
        (ROOT / "tools" / "loop_reference" / "uudd_mixed_sm_qcd.json").read_text()
    )
    fixed = {k: v for k, v in reference["fixed_parameters"].items() if k != "WTau"}
    slice_ = _slice(
        restricted_loop_sm("full"),
        loop_native,
        "u u~ > d d~",
        {"QCD": 2, "QED": 2},
        ({"QCD": 4, "QED": 0}, {"QCD": 2, "QED": 2}),
        fixed,
    )
    parameters = {k: v for k, v in reference["parameters"].items() if k != "MU_R"}

    def key(orders):
        return tuple(sorted((k, int(v)) for k, v in orders.items() if int(v)))

    born_key = {key(v): k for k, v in reference["born_squared_orders"].items()}
    loop_key = {key(v): k for k, v in reference["virtual_squared_orders"].items()}
    for point, row in zip(reference["points"], reference["rows"], strict=True):
        point = [[str(x) for x in leg] for leg in point]
        result = slice_.evaluate(point, reference["mu_r2"], parameters=parameters)
        assert set(result.born) == set(born_key)
        for order, value in result.born.items():
            theirs = row["born"][born_key[order]]
            assert abs(float(value) - theirs) <= TOLERANCE * abs(theirs), order
        assert set(result.components["virtual"]) == set(loop_key)
        for sector, laurent in result.components["virtual"].items():
            full = row["full"][loop_key[sector]]
            theirs = [full["finite"], full["single_pole"], full["double_pole"]]
            scale = max(map(abs, theirs))
            for mine, value in zip(laurent.coefficients, theirs, strict=True):
                assert abs(float(mine) - value) <= TOLERANCE * scale, sector


MW = "80.41900244575616010551713717311353321699"  # loop_sm's G_mu W mass


def _two_to_two(m3: str, m4: str) -> list[list[str]]:
    """A 2 -> 2 point at sqrt(s) = 1000, on shell to 40 digits."""

    from decimal import Decimal, localcontext

    with localcontext() as context:
        context.prec = 50
        a, b, root = Decimal(m3), Decimal(m4), Decimal(1000)
        s = root * root
        p = ((s - (a + b) ** 2) * (s - (a - b) ** 2)).sqrt() / (2 * root)
        e3, e4 = (p * p + a * a).sqrt(), (p * p + b * b).sqrt()
        v = [
            p * Decimal(4) / 5 * Decimal(5) / 13,
            p * Decimal(4) / 5 * Decimal(12) / 13,
        ]
        v.append(p * Decimal(3) / 5)
        half = root / 2
        legs = [
            [half, 0, 0, half],
            [half, 0, 0, -half],
            [e3, *v],
            [e4, *(-x for x in v)],
        ]
        return [[str(+Decimal(x)) for x in leg] for leg in legs]


def _complex(value) -> complex:
    return complex(float(value.real), float(value.imag))


def _born_closure_ratio(slice_, point, scale, parameters):
    """(ratio, residual): the Born closed at leg 1 against the tree runtime's
    amplitude roots on every element of the tree engine's colour basis, per
    helicity and order (``test_loop_v2_slice.
    test_born_closure_is_the_tree_runtime_born``), in 30 digits, fitted in
    binary64 by one complex ratio; the residual is relative to the largest
    runtime amplitude."""

    from pyamplicol.color.connections import contract_connected_tensor_nc_terms
    from pyamplicol.config import PoleSelection
    from pyamplicol.loops.v2 import assemble
    from pyamplicol.loops.v2.assemble import Precision
    from pyamplicol.loops.v2.colour import born_basis

    point = [[Fraction(x) for x in leg] for leg in point]
    precision = Precision.arbitrary(30)
    numbers = assemble._Numbers(precision)
    basis = born_basis(slice_.blocks)
    pairs = []
    with numbers.scope():
        closure = slice_._point(
            point, Fraction(scale), parameters, precision, PoleSelection.ALL
        )
        roots = assemble.TreeRows(slice_.blocks.born_dag, slice_.compiled).evaluate(
            [[assemble._to_decimal(x) for x in leg] for leg in point],
            closure.parameters,
            numbers,
            amplitudes=True,
        )

        def overlap(bra, ket):
            cells = contract_connected_tensor_nc_terms(bra, ket)
            return sum(
                (
                    numbers.colour(cell.scaled(Fraction(3) ** power), 0)
                    for power, cell in cells.items()
                ),
                numbers.zero,
            )

        runtime: dict = {}
        for root in slice_.blocks.born_roots:
            (value,) = roots[(root.root_id, "amplitude-root")]
            key = (
                tuple((label - 1, h) for label, h in root.helicities),
                root.coupling_orders,
                0
                if root.color_sector_id is None
                else basis.sector_ids.index(root.color_sector_id),
            )
            runtime[key] = runtime.get(key, numbers.zero) + value
        for row, vectors in closure.born_vectors.items():
            for order, vector in vectors.items():
                for bra in basis.tensors:
                    mine = sum(
                        (
                            value
                            * numbers.real(Fraction(3) ** power)
                            * overlap(bra, assemble._single(tensor, basis.legs))
                            for (tensor, power), value in vector.items()
                        ),
                        numbers.zero,
                    )
                    theirs = sum(
                        (
                            amplitude * overlap(bra, basis.tensors[c])
                            for (r, o, c), amplitude in runtime.items()
                            if (r, o) == (row, order)
                        ),
                        numbers.zero,
                    )
                    pairs.append((_complex(mine), _complex(theirs)))
    norm = sum(abs(theirs) ** 2 for _mine, theirs in pairs)
    ratio = sum(mine * theirs.conjugate() for mine, theirs in pairs) / norm
    largest = max(abs(theirs) for _mine, theirs in pairs)
    residual = max(abs(mine - ratio * theirs) for mine, theirs in pairs) / largest
    return ratio, residual


@pytest.mark.loop_physics
@pytest.mark.parametrize("name", ["loop_line", "root"])
def test_complex_ckm_vertices_are_read_through_their_conjugate_rule(
    restricted_loop_sm, loop_native, name
):
    """loop_sm with the Wolfenstein CKM and a massless b (its ``zeromass_ckm``
    card): V_ub is complex. g u > w+ b puts the W on the loop quark line:
    the virtual's poles are Catani's. u b~ > e+ ve closes its Born at the
    open W vertex: the closure is the tree runtime's amplitude up to a real
    sign (the closure's colour-flow and the runtime's Wick conventions)."""

    compiled = restricted_loop_sm("zeromass_ckm")
    fixed = {k: v for k, v in ZERO_WIDTHS.items() if k != "MB"}
    if name == "loop_line":
        slice_ = _slice(
            compiled,
            loop_native,
            "g u > w+ b",
            {"QCD": 1, "QED": 1},
            {"QCD": 3, "QED": 1},
            fixed,
        )
        _assert_catani(
            slice_.evaluate(_two_to_two(MW, "0"), "1000000", parameters=PARAMETERS)
        )
        return
    slice_ = _slice(
        compiled,
        loop_native,
        "u b~ > e+ ve",
        {"QCD": 0, "QED": 2},
        {"QCD": 2, "QED": 2},
        fixed,
    )
    assert _open_vertex_terms(slice_)
    ratio, residual = _born_closure_ratio(
        slice_, _two_to_two("0", "0"), "1000000", PARAMETERS
    )
    assert abs(abs(ratio) - 1) <= TOLERANCE and abs(ratio.imag) <= TOLERANCE, ratio
    assert residual <= TOLERANCE, residual
    _assert_form_factor(slice_, [_two_to_two("0", "0")], ["1000000"])


@pytest.mark.parametrize(
    "name",
    [
        *(
            pytest.param(name, marks=pytest.mark.loop_physics)
            for name in ("udx_epve", "emu_emu", "udx_wpg", "gu_wpd")
        ),
        *(
            pytest.param(name, marks=pytest.mark.loop_nightly)
            for name in ("gb_twm", "uux_wpwm", "bbx_wpwm")
        ),
    ],
)
def test_native_inputs_equal_the_python_inputs(
    loop_sm_compiled_model, loop_native, name
):
    """The native per-point inputs (``NativeExecutor(inputs="native")``: tree
    program, Born and counterterm root closures) equal the Python inputs on
    the conjugate rule's root terms (u d~ > e+ ve), on a lepton line closing
    at the root (e- u > e- u), on external W bosons (loop_sm's MW is a
    derived parameter: the parameter program forms it per point, and the W
    legs are bound on its shell, ``test_derived_w_masses_are_bound_on_shell``)
    and on a Weyl
    line's factor kappa at a counterterm root copy (g u > w+ d: a scaled
    coupling slot, ``native_trees._line_factor``), in binary64."""

    from pyamplicol.loops.v2.plan_export import NativeExecutor

    entry = LEGACY["processes"][name]
    slice_ = _slice(
        loop_sm_compiled_model,
        loop_native,
        entry["process"],
        entry["born_orders"],
        entry["loop_orders"],
        LEGACY["fixed_parameters"],
    )
    native = NativeExecutor(slice_, forward=False, inputs="native")
    python = NativeExecutor(slice_, forward=False, exported=native.exported)
    points = [entry["point"]]
    if name in ("udx_epve", "emu_emu"):
        # A second, massless point (all four legs massless).
        points.append(_massless_points()[0][0])
    scales = [LEGACY["mu_r2"]] * len(points)
    mine = native.evaluate(points, scales, parameters=PARAMETERS)
    theirs = python.evaluate(points, scales, parameters=PARAMETERS)
    for a, b in zip(mine, theirs, strict=True):
        assert a.born.keys() == b.born.keys()
        for order, value in b.born.items():
            assert abs(float(a.born[order]) - float(value)) <= TOLERANCE * abs(
                float(value)
            )
        assert a.components.keys() == b.components.keys()
        for component, sectors in b.components.items():
            for sector, laurent in sectors.items():
                virtual = b.components["virtual"][sector].coefficients
                scale = max(abs(float(x)) for x in virtual)
                for x, y in zip(
                    a.components[component][sector].coefficients,
                    laurent.coefficients,
                    strict=True,
                ):
                    assert abs(float(x) - float(y)) <= TOLERANCE * scale, component


@pytest.mark.loop_physics
@pytest.mark.parametrize("name", ["udx_wpg", "gu_wpd"])
def test_derived_w_masses_are_bound_on_shell(loop_sm_compiled_model, loop_native, name):
    """loop_sm's MW is a derived parameter. Both lanes bind the W legs on its
    shell (DESIGN §11.1; genericity audit M2) in every carrier: the Python
    lane's leg invariant is exactly the square of the MW its wavefunctions
    take (``TreeRows._model_values`` at the working digits, the slice's one
    mass evaluator ``assemble._model_masses``), the native lane's is exactly
    the square of its parameter program's MW carrier value (a binary64 value
    in binary64), and the two agree to the carrier's rounding; every leg of
    an external mass is bound to the same exact square in both lanes."""

    import math

    from pyamplicol.loops.v2.assemble import Precision, _leg_masses
    from pyamplicol.loops.v2.plan_export import NativeExecutor

    entry = LEGACY["processes"][name]
    slice_ = _slice(
        loop_sm_compiled_model,
        loop_native,
        entry["process"],
        entry["born_orders"],
        entry["loop_orders"],
        LEGACY["fixed_parameters"],
    )
    native = NativeExecutor(slice_, forward=False, inputs="native")
    assert native.exported.trees.program["parameters"]["derived_masses"]
    legs = [leg.mass.mass for leg in slice_.families.legs]
    (w,) = [k for k, mass in enumerate(legs) if mass == "MW"]
    parameters = {**LEGACY["fixed_parameters"], **PARAMETERS}
    rows = slice_.tree_rows
    (index,) = [
        k
        for k, record in enumerate(rows._records)
        if record.get("runtime_name") == "MW" and record["complex_component"] != "imag"
    ]
    carriers = [
        (Precision.double(), 5e-16),
        (Precision.double_double(), 1e-30),
        (Precision.arbitrary(32), 1e-30),
    ]
    for precision, rounding in carriers:
        digits = precision.working_digits
        python = _leg_masses(slice_, parameters, digits)
        wavefunction = Fraction(rows._model_values(parameters, digits)[index])
        assert python[w] == wavefunction**2
        _plan, bound = native.bound_masses(
            [entry["point"]], [LEGACY["mu_r2"]], PARAMETERS, precision
        )
        for k, mass in enumerate(legs):
            if k != w:
                assert bound[k] == python[k], (precision, k, mass)
        square = bound[w]
        root = Fraction(math.isqrt(square.numerator), math.isqrt(square.denominator))
        assert root * root == square
        if precision.arithmetic == "double":
            assert Fraction(float(root)) == root
        assert abs(square - python[w]) <= Fraction(rounding) * python[w]


@pytest.mark.loop_physics
def test_electroweak_line_stages_leave_quark_rooted_qcd_requests_unchanged(
    loop_sm_compiled_model, loop_native, monkeypatch
):
    """On a quark-rooted QCD request (u u~ > d d~ g) the families' loop
    species give the catalogue of the root rules' declared species, and the
    Born closure's block requests add no current: the families, schedules,
    seeds, the exported plan and the virtual are those of the generation
    without either (byte identity of WP3.2 EWI-1)."""

    from pyamplicol.loops.v2 import assemble, snapshot
    from pyamplicol.models.loading import load_compiled_model
    from tools.loop_reference import exact_points

    compiled = load_compiled_model(loop_sm_compiled_model)
    (point,) = exact_points.generate([0] * 5, count=1)
    point = [[exact_points.decimal_string(x) for x in leg] for leg in point]
    born, loop = {"QCD": 3, "QED": 0}, {"QCD": 5, "QED": 0}

    def digests():
        slice_ = _slice(compiled, loop_native, "u u~ > d d~ g", born, loop, {"MB": "0"})
        return (
            snapshot.stage_digests(slice_),
            snapshot.virtual_digest(slice_, [point], ["1000000"], PARAMETERS),
        )

    production = digests()
    with monkeypatch.context() as patch:
        patch.setattr(assemble, "_family_loop_species", lambda *args: None)
        patch.setattr(assemble, "_check_declared_loop_species", lambda *args: None)
        patch.setattr(assemble, "_born_block_requests", lambda *args: frozenset())
        declared = digests()
    assert production == declared


# ---------------------------------------------------------------- nightly


@pytest.mark.loop_nightly
def test_lepton_initiated_top_pair_higgs_equals_madloop(
    loop_sm_compiled_model, loop_native, loop_reference
):
    """e+ e- > t t~ h (QCD): the lepton at leg 1 and the top Yukawa on loop
    top lines (the scalar coupling of the tree reading's sign rule) against
    MadLoop's virtual at the fixture's three points."""

    fixture = loop_reference("top-pair-higgs-model")
    slice_ = _slice(
        loop_sm_compiled_model,
        loop_native,
        "e+ e- > t t~ h",
        {"QCD": 0, "QED": 3},
        {"QCD": 2, "QED": 3},
        fixture["fixed_parameters"],
    )
    parameters = {k: v for k, v in fixture["parameters"].items() if k != "MU_R"}
    for index, row in enumerate(fixture["rows"]["ee"]):
        result = slice_.evaluate(
            fixture["points"][index],
            str(int(fixture["mu_r"][index]) ** 2),
            parameters=parameters,
        )
        born = sum(float(x) for x in result.born.values())
        assert abs(born - row["born"]) <= TOLERANCE * row["born"]
        full = row["components"]["full"]
        reference = [full["finite"], full["single_pole"], full["double_pole"]]
        floor = row["born"] * ALPHA_S / (2 * math.pi)
        for mine, theirs in zip(
            _laurent(result.components["virtual"]), reference, strict=True
        ):
            assert abs(mine - theirs) <= TOLERANCE * max(abs(theirs), floor)
        _assert_catani(result)


@pytest.mark.loop_nightly
def test_lepton_initiated_three_partons_poles_and_sites(
    loop_sm_compiled_model, loop_native
):
    """e- e+ > u u~ g: Catani's poles, and the counterterm-site oracle (O6)
    equals the production recursion."""

    from pyamplicol.models.loading import load_compiled_model
    from tools.loop_reference import exact_points

    compiled = load_compiled_model(loop_sm_compiled_model)
    born, loop = {"QCD": 1, "QED": 2}, {"QCD": 3, "QED": 2}
    recursion = _slice(compiled, loop_native, "e- e+ > u u~ g", born, loop)
    sites = _slice(
        compiled, loop_native, "e- e+ > u u~ g", born, loop, counterterms="sites"
    )
    (point,) = exact_points.generate([0] * 5, count=1)
    point = [[exact_points.decimal_string(x) for x in leg] for leg in point]
    production = recursion.evaluate(point, "1000000", parameters=PARAMETERS)
    _assert_catani(production)
    _assert_equal_results(
        sites.evaluate(point, "1000000", parameters=PARAMETERS), production
    )


@pytest.mark.loop_nightly
@pytest.mark.parametrize("accuracy", ["lc", "nlc"])
def test_dilepton_two_gluon_colour_projections_are_formal_nc_projections(
    loop_sm_compiled_model, loop_native, monkeypatch, accuracy
):
    """u u~ > e- e+ g g (photon and Z on quark lines, closed quark loops): the
    LC (NLC) slice equals the N_c^P (N_c^P, N_c^(P-1), N_c^(P-2)) part of the
    exact colour cells of every skeleton (an unpruned LC slice evaluated one
    N_c power at a time), P the sector's leading degree, above which every
    part vanishes; the parts over every power sum to the full-colour result
    (design §7.4-§7.6, WP2.4)."""

    from pyamplicol.loops.v2 import assemble, colour
    from pyamplicol.models.loading import load_compiled_model
    from pyamplicol.models.loop_color_declaration import (
        resolve_loop_color_declaration,
    )

    fixture = json.loads((FIXTURES / "dilepton-two-gluon.json").read_text())
    compiled = load_compiled_model(loop_sm_compiled_model)
    # The shipped loop_sm declaration (MQ-NC): the closed quark loops of
    # g g Z / g g A reach this request, which the example QCD subset
    # (``legacy_capture.COLOUR_DECLARATION``) does not cover.
    declaration = resolve_loop_color_declaration(compiled.raw_loop_model)

    def build(accuracy, prune=True):
        request = SliceRequest(
            "u u~ > e- e+ g g",
            {"QCD": 2, "QED": 2},
            {"QCD": 4, "QED": 2},
            fixed_parameters=fixture["fixed_parameters"],
            accuracy=accuracy,
            colour_declaration=declaration,
            colour_contraction="direct",
        )
        return LoopSlice.generate(compiled, request, loop_native, prune=prune)

    projected, unpruned, full = build(accuracy), build("lc", prune=False), build("full")
    parameters = {k: v for k, v in fixture["parameters"].items() if k != "MU_R"}
    point, scale = fixture["points"][0], str(int(fixture["mu_r"][0]) ** 2)
    degrees = unpruned.degrees()

    def at(power):
        def kept(exponent, _degree, _accuracy):
            return exponent == power

        with monkeypatch.context() as patch:
            patch.setattr(colour, "kept", kept)
            patch.setattr(assemble, "kept", kept)
            return unpruned.evaluate(point, scale, parameters=parameters)

    top = max(
        value
        for kind in degrees.values()
        for value in kind.values()
        if value is not None
    )
    parts = {power: at(power) for power in range(top - 8, top + 1)}
    depth = {"lc": 0, "nlc": 2}[accuracy]
    mine = projected.evaluate(point, scale, parameters=parameters)
    reference = full.evaluate(point, scale, parameters=parameters)
    for order, value in mine.born.items():
        degree = degrees["born"][order]
        assert all(
            float(parts[p].born.get(order, 0)) == 0 for p in parts if p > degree
        ), order
        expected = sum(
            float(parts[p].born.get(order, 0))
            for p in range(degree - depth, degree + 1)
        )
        assert abs(float(value) - expected) <= TOLERANCE * abs(expected), order
        total = sum(float(part.born.get(order, 0)) for part in parts.values())
        assert abs(total - float(reference.born[order])) <= TOLERANCE * abs(total)
    for component, sectors in reference.components.items():
        for sector, laurent in sectors.items():
            values = [float(x) for x in laurent.coefficients]
            scale_ = max(map(abs, values)) or 1.0
            totals = [0.0, 0.0, 0.0]
            for part in parts.values():
                cell = part.components[component].get(sector)
                if cell is not None:
                    totals = [
                        t + float(c)
                        for t, c in zip(totals, cell.coefficients, strict=True)
                    ]
            assert all(
                abs(a - b) <= TOLERANCE * scale_
                for a, b in zip(totals, values, strict=True)
            )
            degree = degrees["virtual"][sector]
            for p in (p for p in parts if p > degree):
                cell = parts[p].components[component].get(sector)
                assert cell is None or not any(map(float, cell.coefficients)), (
                    component,
                    sector,
                    p,
                )
            expected = [0.0, 0.0, 0.0]
            for p in range(degree - depth, degree + 1):
                cell = parts[p].components[component].get(sector)
                if cell is not None:
                    expected = [
                        t + float(c)
                        for t, c in zip(expected, cell.coefficients, strict=True)
                    ]
            found = [float(x) for x in mine.components[component][sector].coefficients]
            assert all(
                abs(a - b) <= TOLERANCE * scale_
                for a, b in zip(found, expected, strict=True)
            ), (
                component,
                sector,
                found,
                expected,
            )
