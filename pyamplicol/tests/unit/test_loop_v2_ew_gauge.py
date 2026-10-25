# SPDX-License-Identifier: 0BSD
"""'t Hooft-Feynman-gauge trees (WP3.2 EWI-4a).

A compiled model takes its massive vectors in unitary gauge (the default:
the Goldstones are absorbed and leave the tree model) or in 't Hooft-Feynman
gauge (``models.loading.with_model_gauge``: the vectors propagate with
-i g / (p^2 - M^2) and the Goldstones are explicit fields), admissible only
for a UFO that declares gauge 1. Ghosts never enter a tree.

The checks can fail (review major 1 of wp32/PLAN.md): they use trees where a
massive vector joins massive or scalar currents, so that the unitary
k k / M^2 term and the Goldstone diagrams are both non-zero, and amplitude
values, so that a Goldstone vertex sign cannot square away:

* Feynman = unitary per helicity and colour sector, to 1e-13 of the largest
  amplitude: m+ m- > t t~ (massive-muon G_mu UFO), t t~ > z h,
  t t~ > t t~ at QED order, t t~ > w+ w-, b b~ > t t~ at QED order (the
  tree of M5's QCD-corrected premise) and b t~ > w- h (a G- s-channel) with
  a massive b;
* the Slavnov-Taylor identities k.M(V) = eta M_V M(G) of an outgoing Z and
  W-, with a non-zero Goldstone side: g g > t t~ z, both Z of b b~ > z z and
  g b > t w- with a massive b. eta is fixed by the UFO's conventions (-i for
  the Z, derived from the Z and G0 couplings of the top in the docstring of
  ``test_slavnov_taylor_identities``), so a flipped Goldstone coupling fails;
* the negative control: a model with the G0 t t~ coupling's sign flipped
  fails both checks.

Widths are zero (v2 is real on-shell; a width breaks the Ward identities of
the internal propagators). Points are binary64 on-shell sequential decays,
amplitudes are evaluated with 32 digits.
"""

from __future__ import annotations

import math
import random
from contextlib import contextmanager
from dataclasses import replace
from decimal import Decimal
from unittest import mock

import pytest

pytestmark = pytest.mark.loop_physics

TOLERANCE = 1e-13
ZERO_WIDTHS = {"WT": "0", "WZ": "0", "WW": "0", "WH": "0"}
GOLDSTONES = ("G0", "G+", "G-")


# ------------------------------------------------------------------ models


@pytest.fixture(scope="module")
def gmu(loop_gmu_ufo):
    from pyamplicol.models.loading import compile_model_source

    return compile_model_source(loop_gmu_ufo, require_supported=False)


@pytest.fixture(scope="module")
def gmu_massive_b(loop_gmu_ufo):
    from pyamplicol.models.loading import compile_model_source

    return compile_model_source(
        loop_gmu_ufo, restriction="with_b_mass", require_supported=False
    )


@pytest.fixture(scope="module")
def gmu_massive_muon(loop_gmu_ufo):
    from pyamplicol.models.loading import compile_model_source

    path = loop_gmu_ufo.parent / "loop_qcd_qed_sm_Gmu_massive_muon_py3"
    if not path.is_dir():
        pytest.skip(f"the massive-muon G_mu UFO is unavailable: {path}")
    return compile_model_source(path, require_supported=False)


def _feynman(compiled):
    from pyamplicol.models.loading import with_model_gauge

    return with_model_gauge(compiled, "feynman")


def _mass(compiled, name: str) -> float:
    if name == "ZERO":
        return 0.0
    return float(compiled.parameter_defaults[name][0])


# ------------------------------------------------------------------ points


def _two_body(mass, m1, m2, rng):
    momentum = math.sqrt((mass**2 - (m1 + m2) ** 2) * (mass**2 - (m1 - m2) ** 2)) / (
        2 * mass
    )
    theta, phi = math.acos(rng.uniform(-0.9, 0.9)), rng.uniform(0.0, 2 * math.pi)
    axis = (
        math.sin(theta) * math.cos(phi),
        math.sin(theta) * math.sin(phi),
        math.cos(theta),
    )
    first = [math.hypot(momentum, m1), *(momentum * x for x in axis)]
    second = [math.hypot(momentum, m2), *(-momentum * x for x in axis)]
    return first, second


def _boost(p, velocity):
    v2 = sum(x * x for x in velocity)
    if v2 == 0:
        return p
    gamma = 1 / math.sqrt(1 - v2)
    vp = sum(v * x for v, x in zip(velocity, p[1:], strict=True))
    return [
        gamma * (p[0] + vp),
        *(
            x + ((gamma - 1) * vp / v2 + gamma * p[0]) * v
            for x, v in zip(p[1:], velocity, strict=True)
        ),
    ]


def _point(incoming, outgoing, sqrt_s=1200.0, seed=7):
    """A binary64 on-shell 2 -> n point: beams along z, sequential decays."""

    rng = random.Random(seed)
    m1, m2 = incoming
    momentum = math.sqrt(
        (sqrt_s**2 - (m1 + m2) ** 2) * (sqrt_s**2 - (m1 - m2) ** 2)
    ) / (2 * sqrt_s)
    legs = [
        [math.hypot(momentum, m1), 0.0, 0.0, momentum],
        [math.hypot(momentum, m2), 0.0, 0.0, -momentum],
    ]
    total, rest = [sqrt_s, 0.0, 0.0, 0.0], list(outgoing)
    while len(rest) > 1:
        mass = math.sqrt(total[0] ** 2 - sum(x * x for x in total[1:]))
        recoil = (
            rest[1]
            if len(rest) == 2
            else sum(rest[1:]) + (mass - sum(rest)) * rng.uniform(0.2, 0.8)
        )
        first, second = _two_body(mass, rest[0], recoil, rng)
        velocity = [x / total[0] for x in total[1:]]
        legs.append(_boost(first, velocity))
        total, rest = _boost(second, velocity), rest[1:]
    legs.append(total)
    return legs


def _process_point(compiled, process, sqrt_s=1200.0, seed=7):
    names = process.replace(">", " ").split()
    masses = {p.name.lower(): p.mass for p in compiled.ir.particles}
    values = [_mass(compiled, masses[name.lower()]) for name in names]
    return _point(values[:2], values[2:], sqrt_s, seed)


# -------------------------------------------------------------- amplitudes


@contextmanager
def _goldstone_legs():
    """Admit Goldstones as external legs: the amplitude with a Goldstone in
    place of a vector is the other side of a Slavnov-Taylor identity, not a
    physical process."""

    from pyamplicol.processes.model import ModelParticleCatalog

    def admitted(catalog):
        return tuple(
            particle
            for particle in catalog.particles
            if particle.spin > 0
            and particle.ghost_number == 0
            and particle.propagating
            and particle.auxiliary_kind is None
        )

    with mock.patch.object(
        ModelParticleCatalog, "external_particles", property(admitted)
    ):
        yield


def _dag(compiled, process, orders, ignored=()):
    from pyamplicol.generation.dag_compiler import compile_generic_dag
    from pyamplicol.models.external import CompiledUFOModel
    from pyamplicol.processes.model import build_model_process_ir

    model = CompiledUFOModel(compiled)
    with _goldstone_legs():
        process_ir = build_model_process_ir(process, compiled.ir, color_accuracy="full")
    ignored_ids = tuple(
        record.pdg_code for record in compiled.ir.particles if record.name in ignored
    )
    return model, compile_generic_dag(
        process_ir,
        model=model,
        max_coupling_orders=orders,
        ignored_particle_ids=ignored_ids,
    )


def _amplitudes(compiled, process, orders, point, *, ward_leg=None, ignored=()):
    """Tree amplitudes per (helicities, colour sector) at a point
    (``assemble.TreeRows``, the v2 tree replay), evaluated with 32 digits:
    the cancellations of a Slavnov-Taylor identity cost binary64 up to
    3e-13 (b b~ > z z), the points' own rounding 1e-14. ``ward_leg``
    replaces that vector leg's polarisation by its momentum."""

    from pyamplicol.loops.v2.assemble import Precision, TreeRows, _Numbers
    from pyamplicol.loops.v2.blocks import _born_roots

    model, dag = _dag(compiled, process, orders, ignored)
    ghosts = {
        record.pdg_code for record in compiled.ir.particles if record.ghost_number
    }
    assert not any(current.index.particle_id in ghosts for current in dag.currents), (
        "a ghost entered a tree"
    )
    rows = TreeRows(dag, compiled, jit=False)
    exact = [[Decimal(repr(x)) for x in leg] for leg in point]
    sources = None
    if ward_leg is not None:
        sources = {ward_leg: [(value, Decimal(0)) for value in exact[ward_leg - 1]]}
    values = rows.evaluate(
        exact,
        ZERO_WIDTHS,
        _Numbers(Precision.arbitrary(32)),
        amplitudes=True,
        sources=sources,
    )
    amplitudes: dict = {}
    for root in _born_roots(dag, model):
        key = (root.helicities, root.color_sector_id)
        (value,) = values[(root.root_id, "amplitude-root")]
        value = complex(float(value.real), float(value.imag))
        amplitudes[key] = amplitudes.get(key, 0) + value
    return amplitudes


def _difference(mine, theirs) -> float:
    """The largest difference of two amplitude sets per (helicities, colour
    sector), relative to their largest amplitude."""

    keys = set(mine) | set(theirs)
    largest = max(abs(value) for value in (*mine.values(), *theirs.values()))
    return max(abs(mine.get(k, 0) - theirs.get(k, 0)) for k in keys) / largest


# ---------------------------------------------- Feynman = unitary on trees

# (model, process, orders, whether Goldstone diagrams contribute). In
# t t~ > w+ w- they do not (no G0 W+ W- vertex) and the k k / M^2 term of the
# s-channel Z vanishes on the on-shell W pair: it checks the g^{mu nu} part
# of the massive Feynman-gauge propagator, its mass included.
FEYNMAN_UNITARY = (
    ("massive_muon", "m+ m- > t t~", {"QCD": 0, "QED": 2}, True),
    ("default", "t t~ > z h", {"QCD": 0, "QED": 2}, True),
    ("default", "t t~ > t t~", {"QCD": 0, "QED": 2}, True),
    ("massive_b", "t t~ > w+ w-", {"QCD": 0, "QED": 2}, False),
    ("massive_b", "b b~ > t t~", {"QCD": 0, "QED": 2}, True),
    ("massive_b", "b t~ > w- h", {"QCD": 0, "QED": 2}, True),
)


@pytest.fixture(scope="module")
def models(request):
    return {
        "default": request.getfixturevalue("gmu"),
        "massive_b": request.getfixturevalue("gmu_massive_b"),
        "massive_muon": request.getfixturevalue("gmu_massive_muon"),
    }


@pytest.mark.parametrize(("model", "process", "orders", "goldstones"), FEYNMAN_UNITARY)
def test_feynman_gauge_trees_equal_unitary_gauge(
    models, model, process, orders, goldstones
):
    """Every amplitude of the 't Hooft-Feynman-gauge tree model equals the
    unitary-gauge one, while the Goldstone diagrams alone are not negligible
    (the check would see a wrong Goldstone vertex or propagator)."""

    unitary = models[model]
    feynman = _feynman(unitary)
    point = _process_point(unitary, process)
    reference = _amplitudes(unitary, process, orders, point)
    found = _amplitudes(feynman, process, orders, point)
    assert _difference(found, reference) <= TOLERANCE
    without_goldstones = _amplitudes(
        feynman, process, orders, point, ignored=GOLDSTONES
    )
    assert (_difference(without_goldstones, reference) > 1e-3) == goldstones


# ------------------------------------------ Slavnov-Taylor identities

# k.M(V) / (M_V M(G)) of an outgoing Z or W- in the UFO conventions of
# loop_qcd_qed_sm (``test_slavnov_taylor_identities``).
ETA = -1j

SLAVNOV_TAYLOR = (
    ("default", "g g > t t~ z", "g g > t t~ g0", 5, "MZ", {"QCD": 2, "QED": 1}),
    ("massive_b", "b b~ > z z", "b b~ > g0 z", 3, "MZ", {"QCD": 0, "QED": 2}),
    ("massive_b", "b b~ > z z", "b b~ > z g0", 4, "MZ", {"QCD": 0, "QED": 2}),
    ("massive_b", "g b > t w-", "g b > t g-", 4, "MW", {"QCD": 1, "QED": 1}),
)


def _slavnov_taylor(compiled, vector, goldstone, leg, mass, orders, eta):
    """(residual, Goldstone side): max |k.M(V) - eta M_V M(G)| relative to
    the largest |k.M(V)|, per helicity of the other legs and colour sector,
    and the Goldstone side's largest amplitude relative to it."""

    point = _process_point(compiled, vector)
    ward = _amplitudes(compiled, vector, orders, point, ward_leg=leg)
    side = _amplitudes(compiled, goldstone, orders, point)
    scale = _mass(compiled, mass)
    largest = max(abs(value) for value in ward.values())
    residual = 0.0
    for (helicities, colour), value in ward.items():
        key = (tuple((i, 0 if i == leg else h) for i, h in helicities), colour)
        expected = eta * scale * side.get(key, 0)
        residual = max(residual, abs(value - expected) / largest)
    return residual, max(abs(v) for v in side.values()) * scale / largest


@pytest.mark.parametrize(
    ("model", "vector", "goldstone", "leg", "mass", "orders"), SLAVNOV_TAYLOR
)
def test_slavnov_taylor_identities(models, model, vector, goldstone, leg, mass, orders):
    """k.M(V) = eta M_V M(G) for an outgoing Z or W-, eta = -i.

    Z: the top's coupling gamma^mu (g_V + g_A gamma^5), g_A = -i e / (4 s_W
    c_W) (GC_69, GC_76), gives k.J = -2 m_t g_A (ubar gamma^5 v) between
    on-shell tops (the vector current is conserved; the propagator terms
    cancel between the insertions), and the G0 coupling GC_117 (P_L - P_R) =
    -(m_t / v) gamma^5 gives M(G0) = -(m_t / v) (ubar gamma^5 v); with
    M_Z = e v / (2 s_W c_W), eta = 2 g_A v / M_Z = -i. The Z Z H and Z G0 H
    couplings (GC_99, GC_78) give the same eta.
    W-: the b -> t W- coupling i e / (sqrt(2) s_W) gamma^mu P_L (GC_67) gives
    k.J = g_W ubar_t (m_b P_R - m_t P_L) u_b, and the G- coupling
    (y_t P_L - y_b P_R) (GC_39, GC_48) is sqrt(2) / v times
    (m_t P_L - m_b P_R); with M_W = e v / (2 s_W), eta = -g_W v / (sqrt(2)
    M_W) = -i.
    """

    compiled = _feynman(models[model])
    residual, goldstone_side = _slavnov_taylor(
        compiled, vector, goldstone, leg, mass, orders, ETA
    )
    assert goldstone_side > 1e-3
    assert residual <= TOLERANCE


# -------------------------------------------------------- negative control


def _flipped_goldstone_coupling(compiled, vertex_particles):
    """A test-only model whose Goldstone vertex ``vertex_particles`` has the
    opposite coupling sign."""

    terms = tuple(
        replace(term, coupling_expression=f"-({term.coupling_expression})")
        if sorted(term.particles) == sorted(vertex_particles)
        else term
        for term in compiled.ir.vertex_terms
    )
    assert terms != compiled.ir.vertex_terms
    return replace(
        compiled,
        ir=replace(compiled.ir, vertex_terms=terms),
        _serialized_path=None,
        _prepared_bundle=None,
    )


def test_a_flipped_goldstone_coupling_fails_both_checks(gmu):
    flipped = _flipped_goldstone_coupling(_feynman(gmu), ("t~", "t", "G0"))
    orders = {"QCD": 0, "QED": 2}
    point = _process_point(gmu, "t t~ > z h")
    unitary = _amplitudes(gmu, "t t~ > z h", orders, point)
    assert _difference(_amplitudes(flipped, "t t~ > z h", orders, point), unitary) > 0.1
    residual, _side = _slavnov_taylor(
        flipped,
        "g g > t t~ z",
        "g g > t t~ g0",
        5,
        "MZ",
        {"QCD": 2, "QED": 1},
        ETA,
    )
    assert residual > 0.1


# ---------------------------------------------------------- the gauge option


def test_the_gauge_is_a_compile_option_recorded_off_the_default(gmu):
    """Feynman gauge changes the Goldstone dispositions and the recorded
    gauge (IR and compile options), nothing else; unitary gauge records
    neither, and a round trip keeps the gauge."""

    from pyamplicol.models.contracts import CompiledModelIR
    from pyamplicol.models.loading import ModelCompileOptions, with_model_gauge

    assert "gauge" not in ModelCompileOptions().canonical_payload()
    with pytest.raises(ValueError, match="model gauge"):
        ModelCompileOptions(gauge="landau")
    feynman = with_model_gauge(gmu, "feynman")
    assert gmu.ir.gauge == "unitary" and "gauge" not in gmu.ir.to_dict()
    assert "gauge" not in gmu.source["options"]
    assert (
        feynman.source["options"]
        == ModelCompileOptions(gauge="feynman").canonical_payload()
    )
    unitary_ir, feynman_ir = gmu.ir.to_dict(), feynman.ir.to_dict()
    assert {
        key
        for key in unitary_ir.keys() | feynman_ir.keys()
        if unitary_ir.get(key) != feynman_ir.get(key)
    } == {"gauge", "goldstone_partners"}
    assert {
        (record.goldstone, record.vector, record.policy)
        for record in feynman.ir.goldstone_partners
    } == {("G0", "Z", "explicit"), ("G+", "W+", "explicit"), ("G-", "W-", "explicit")}
    assert CompiledModelIR.from_dict(feynman_ir) == feynman.ir
    back = with_model_gauge(feynman, "unitary")
    assert back.ir == gmu.ir and back.source == gmu.source
    with pytest.raises(ValueError, match="model gauge"):
        with_model_gauge(gmu, "landau")


def test_compiling_in_feynman_gauge_applies_it_to_the_cached_compile(
    loop_gmu_ufo, gmu, tmp_path
):
    """compile_model_source(gauge=...) is the gauge applied to the
    gauge-independent compile, which is the only cache entry."""

    from pyamplicol.models.loading import compile_model_source, with_model_gauge

    cache = tmp_path / "models"
    feynman = compile_model_source(
        loop_gmu_ufo, require_supported=False, cache_dir=cache, gauge="feynman"
    )
    assert feynman.ir == with_model_gauge(gmu, "feynman").ir
    assert len(list(cache.glob("*.json"))) == 1
    assert (
        compile_model_source(loop_gmu_ufo, require_supported=False, cache_dir=cache).ir
        == gmu.ir
    )


def test_feynman_gauge_needs_a_model_declaring_gauge_1(gmu):
    """Admissible only for a source that declares gauge 1 (the G_mu UFO
    declares [1]); a model declaring only unitary gauge, or nothing (a JSON
    model keeps no raw source), is refused."""

    from pyamplicol.models.loading import declared_model_gauges, with_model_gauge

    assert declared_model_gauges(gmu) == (1,)
    raw = gmu.raw_loop_model
    unitary_only = replace(
        gmu, raw_loop_model=replace(raw, metadata={**raw.metadata, "gauge": (0,)})
    )
    with pytest.raises(ValueError, match=r"declares \[0\]"):
        with_model_gauge(unitary_only, "feynman")
    with pytest.raises(ValueError, match="declares none"):
        with_model_gauge(replace(gmu, raw_loop_model=None), "feynman")


def test_feynman_tree_model_keeps_goldstones_and_massive_feynman_propagators(gmu):
    """Unitary gauge absorbs G0, G+ and G- (the UFO's G- lost its Goldstone
    flag: ``compiler_gauge.close_goldstone_flags``), Feynman gauge keeps
    their vertices and propagates the Z and the W with -i g / (p^2 - M^2);
    Goldstones are never external states."""

    from pyamplicol.models.external import CompiledUFOModel
    from pyamplicol.processes.model import build_model_process_ir

    unitary, feynman = CompiledUFOModel(gmu), CompiledUFOModel(_feynman(gmu))
    pdg = {record.name: record.pdg_code for record in gmu.ir.particles}
    assert unitary.inactive_goldstone_names == frozenset(GOLDSTONES)
    assert feynman.inactive_goldstone_names == frozenset()
    for name in GOLDSTONES:
        assert not any(pdg[name] in v.particles for v in unitary.vertices)
        assert any(pdg[name] in v.particles for v in feynman.vertices)
        with pytest.raises(ValueError, match="not an external state"):
            build_model_process_ir(f"t t~ > z {name}", gmu.ir)
    for name in ("Z", "W+", "W-"):
        propagator = feynman._propagator_ir(pdg[name])
        assert (propagator.gauge, propagator.goldstone_policy) == (
            "feynman",
            "explicit",
        )
        assert propagator.denominator == "momentum_squared-mass_squared+i*mass*width"
        assert unitary._propagator_ir(pdg[name]).gauge == "unitary"


def test_the_lowered_model_cache_key_holds_the_gauge_off_the_default(
    gmu, tmp_path, monkeypatch
):
    import hashlib
    import json

    from pyamplicol.models import compiler_entry, loading

    monkeypatch.setattr(
        compiler_entry, "compile_ufo_model_ir", lambda *args, **kwargs: gmu.ir
    )
    payload = {"name": "probe"}
    keys = {
        gauge: loading._counterterm_ir(
            payload,
            tree_kernel_variant="transverse",
            u1_projected_vertices=frozenset(),
            cache_dir=tmp_path,
            gauge=gauge,
        )[1][1]
        for gauge in ("unitary", "feynman")
    }
    assert keys["unitary"] != keys["feynman"]
    assert (
        keys["unitary"]
        == hashlib.sha256(
            json.dumps(
                [payload, "transverse", [], loading._COUNTERTERM_IR_FINGERPRINT[0]],
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    )
