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
  amplitude: t b~ > t b~ at QED order (the G+ s-channel and the G0
  t-channel make about as much as the whole amplitude; it replaced
  m+ m- > t t~ on the massive-muon UFO, whose Goldstone share is 2e-3:
  WP3.2 EWI-4b), t t~ > z h, t t~ > t t~ at QED order, t t~ > w+ w-,
  b b~ > t t~ at QED order (the tree of M5's QCD-corrected premise) and
  b t~ > w- h (a G- s-channel), with a massive b where it enters;
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

from tools.loop_reference.ufo_source_oracle import coupling_ratio

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
def gmu_massive_b(loop_gmu_ufodb_unpatched_ufo):
    """UFOmodeldb's unpatched G_mu model with a massive b: the model of
    ``goldstone-mixing.json``'s MadLoop components (the issue-1 patch adds the
    b-mass term of the charge counterterm, which moves every massive-b EW
    virtual; ``test_loop_v2_ufodb_models`` holds the patched one against
    MadLoop and Recola)."""

    from pyamplicol.models.loading import compile_model_source

    return compile_model_source(
        loop_gmu_ufodb_unpatched_ufo, restriction="with_b_mass", require_supported=False
    )


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
    ("massive_b", "t b~ > t b~", {"QCD": 0, "QED": 2}, True),
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

    # Compared within one cache entry: a fresh compile's colour expressions
    # can differ in argument order from ``gmu`` (another process's compile)
    # when trees were evaluated earlier in this process (pre-existing
    # history dependence of the model compile, slice-status 2026-10-25).
    cache = tmp_path / "models"
    feynman = compile_model_source(
        loop_gmu_ufo, require_supported=False, cache_dir=cache, gauge="feynman"
    )
    assert len(list(cache.glob("*.json"))) == 1
    unitary = compile_model_source(
        loop_gmu_ufo, require_supported=False, cache_dir=cache
    )
    assert len(list(cache.glob("*.json"))) == 1
    assert unitary.ir.gauge == "unitary" and unitary.source == gmu.source
    assert feynman.ir == with_model_gauge(unitary, "feynman").ir
    assert with_model_gauge(feynman, "unitary").ir == unitary.ir


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


# ===================================================== loops and counterterms
# WP3.2 EWI-4b: electroweak requests take 't Hooft-Feynman gauge throughout.


def _colour_singlet_bosons(compiled) -> tuple[int, ...]:
    """The colour-singlet propagating scalars and vectors of ``compiled`` and
    its colour-singlet ghosts (PDG codes): the loop lines the gauge changes,
    chosen by spin, colour and ghost number."""

    return tuple(
        sorted(
            record.pdg_code
            for record in compiled.ir.particles
            if abs(int(record.color)) == 1
            and record.propagating
            and (int(record.spin) in (1, 3) or record.ghost_number)
        )
    )


def test_feynman_gauge_loop_lines_keep_rank_at_most_n(gmu):
    """Every loop segment (vertex times the next line's numerator) and every
    propagator numerator of the colour-singlet bosons and ghosts of the G_mu
    model has q-degree <= 1 in 't Hooft-Feynman gauge (massive vectors -i g,
    Goldstones and ghosts i), so a loop of N propagators has rank <= N
    (DESIGN §6.8). In unitary gauge the massive vector line (q q / M^2:
    rank N + 2) is refused."""

    from pyamplicol.loops.v2.catalogue import (
        LoopKernelCatalogueError,
        build_loop_kernel_catalogue,
    )

    species = _colour_singlet_bosons(gmu)
    catalogue = build_loop_kernel_catalogue(
        _feynman(gmu), loop_species=species, counterterm_entries=()
    )
    kinds = {line.propagator_kind for line in catalogue.propagator_numerators}
    assert {"vector", "scalar", "ufo-index-free"} <= kinds
    assert catalogue.segments
    assert max(segment.degree for segment in catalogue.segments) <= 1
    assert all(line.degree == 0 for line in catalogue.propagator_numerators)
    with pytest.raises(LoopKernelCatalogueError, match="unitary-gauge propagator"):
        build_loop_kernel_catalogue(gmu, loop_species=species, counterterm_entries=())


def test_electroweak_requests_need_a_model_declaring_feynman_gauge(gmu):
    """The gauge scope: an electroweak request takes the model in
    't Hooft-Feynman gauge (``ew_policy.feynman_gauge_model``) and is refused
    with a typed error on a model that cannot be taken in it (its source
    declares no gauge 1); the loop model preparation refuses every loop
    request on such a model already (``covariant_model``: the UV and R2
    rules are written for gauge 1), so the dry run reports it."""

    from pyamplicol.loops.v2 import ew_policy
    from pyamplicol.loops.v2.assemble import SliceRequest, prepare_request_model

    electroweak = SliceRequest(
        "u u~ > d d~",
        {"QCD": 2, "QED": 0},
        {"QCD": 2, "QED": 2},
        corrections=("QED",),
        fixed_parameters=ZERO_WIDTHS,
        compatibility="gmu_particle_anti",
        scale_parameter="MU_R",
    )
    feynman = ew_policy.feynman_gauge_model(gmu, electroweak)
    assert feynman.ir.gauge == "feynman" and gmu.ir.gauge == "unitary"
    assert ew_policy.feynman_gauge_model(feynman, electroweak) is feynman
    raw = gmu.raw_loop_model
    unitary_only = replace(
        gmu, raw_loop_model=replace(raw, metadata={**raw.metadata, "gauge": (0,)})
    )
    with pytest.raises(ew_policy.ElectroweakGaugeError, match=r"only the gauges \[0\]"):
        ew_policy.require_feynman_gauge(unitary_only, electroweak)
    with pytest.raises(ValueError, match="does not declare Feynman gauge"):
        prepare_request_model(unitary_only, electroweak)


def _mixing_terms(compiled, vector):
    """The CT structures and couplings of the G_mu UFO's two-point mixing of
    ``vector`` with its Goldstone (UVmass, QED^2, one epsilon slot)."""

    from pyamplicol.models.loading import CountertermStructure

    goldstone = {"Z": "G0", "W-": "G+", "A": "G0"}[vector]
    out = {}
    for index, lorentz in enumerate(("l_GpWmMass4", "l_GpWmMass5")):
        structure = CountertermStructure(
            (goldstone, vector), "1", lorentz, (("QED", 2),), "uv_local", 0
        )
        out[structure] = f"(2+{index}*𝑖)*UFO::{{}}::MZ"  # noqa: RUF001
    return out


def test_goldstone_vector_mixing_is_a_vertex_insertion(gmu):
    """A Goldstone-vector two-point term (Pi^mu = kappa p^mu) is lowered as a
    vertex insertion (``models.ct_line_insertions``): both clones are vector
    lines with the propagator kappa N_S N_V / (D_S D_V), the copies of the
    Goldstone's vertices carry the leg's momentum on a promoted vector leg,
    the vector's copies are plain; P(2,1) (the Goldstone's momentum) is
    -P(k) and P(2,2) is +P(k), P(k) the tree engine's momentum of the copy's
    leg (minus the momentum flowing in), as the tree engine reads every
    vertex (WP3.2 EWI-7b). In unitary gauge the absorbed Goldstone's
    terms are refused (inert in every request, REQUIREMENTS_ADDENDUM J)."""

    from symbolica import E

    from pyamplicol.models.ct_line_insertions import (
        _engine_denominator,
        _momentum_ratio,
        lower_line_insertions,
    )

    feynman = _feynman(gmu)
    payload = feynman.model
    particles = {item["name"]: item for item in payload["particles"]}
    propagators = {item["name"]: item for item in payload["propagators"]}
    lorentz = {item["name"]: item for item in payload["lorentz_structures"]}
    terms = _mixing_terms(feynman, "Z")
    ratios = {s.lorentz: _momentum_ratio(s, lorentz, particles) for s in terms}
    assert ratios == {"l_GpWmMass4": E("-1"), "l_GpWmMass5": E("1")}

    def lower(gauge, absorbed=frozenset()):
        return lower_line_insertions(
            payload,
            terms,
            prefix="PYAMPLICOL_CT",
            insertion_order="PYAMPLICOL_CT_INSERTION",
            slot_order=lambda structure: structure.slot_order,
            gauge=gauge,
            absorbed=absorbed,
        )

    lines = lower("feynman")
    assert not lines.refused and len(set(lines.clones.values())) == 1
    clones = {item["name"]: item for item in lines.particles}
    assert {int(item["spin"]) for item in clones.values()} == {3}
    assert not any(item["goldstoneboson"] for item in clones.values())
    assert sorted(lines.reads_as.values()) == ["G0", "Z"]
    (scale,) = lines.parameters
    assert E(scale["expression"]) == E("(2+𝑖)*UFO::{}::MZ-2*UFO::{}::MZ")  # noqa: RUF001
    for record in lines.propagators:
        assert record["pyamplicol_source"] == "model-supplied"
        assert "Metric" in record["numerator"] and scale["name"] in record["numerator"]
        expected = (
            _engine_denominator(
                propagators[particles[name]["propagator"]], particles[name]
            )
            for name in ("G0", "Z")
        )
        assert record["denominator"] == "*".join(f"({part})" for part in expected)
    promoted = {item["name"]: item for item in lines.lorentz_structures}
    assert promoted
    for rule in lines.vertex_rules:
        original = rule["name"].split("__", 1)[1]
        source = next(r for r in payload["vertex_rules"] if r["name"] == original)
        if lines.reads_as[next(p for p in rule["particles"] if p in clones)] == "G0":
            assert "G0" in source["particles"]
            for name in rule["lorentz_structures"]:
                record = promoted[name]
                leg = rule["particles"].index(
                    next(p for p in rule["particles"] if p in clones)
                )
                assert record["spins"][leg] == 3
                assert (
                    f"P(UFO::{{}}::idx(1,{leg + 1}),UFO::{{}}::idx(1,{leg + 1}))"
                    in (record["structure"])
                )
        else:
            assert "Z" in source["particles"]
            assert rule["lorentz_structures"] == source["lorentz_structures"]
    refused = lower("unitary", frozenset({"G0", "G+", "G-"}))
    assert set(refused.refused) == set(terms) and not refused.clones
    assert all("absorbs" in reason for reason in refused.refused.values())


def test_native_dag_builds_the_goldstone_model_blocks(gmu):
    """A4's native loop-request DAG equals the tree engine's on the
    't Hooft-Feynman-gauge model, whose blocks carry Goldstone currents
    (WP3.2 EWI-4b; the oracle of ``test_loop_v2_native_dag``)."""

    from pyamplicol.models.external import CompiledUFOModel

    from .test_loop_v2_native_dag import _assert_equal

    feynman = _feynman(gmu)
    model = CompiledUFOModel(feynman)
    python, _native = _assert_equal(feynman, model, "g g > t t~", {"QCD": 2, "QED": 2})
    goldstones = {
        record.pdg_code for record in feynman.ir.particles if record.goldstoneboson
    }
    assert any(current.index.particle_id in goldstones for current in python.currents)


# ---------------------------------------- nightly: electroweak loops and M5

# The fixture metric of tools/loop_v2_generate.py validate: |a - b| over
# max(|a|, |b|, |Born| alpha / 2 pi) for the Laurent coefficients.
EW_TOLERANCE = 1e-12


@pytest.fixture(scope="module")
def gg_ttx_ew(loop_gmu_ufo, loop_native, loop_reference):
    """g g > t t~ with QCD and QED corrections on the G_mu UFO (the request
    of ``required/gg_ttx_ew.json``, ``tools/loop_v2_generate``), generated
    once: 't Hooft-Feynman gauge throughout."""

    import argparse

    from pyamplicol.loops.v2.assemble import LoopSlice
    from pyamplicol.models.loading import compile_model_source
    from tools import loop_v2_generate

    reference = loop_reference("required/gg_ttx_ew")
    compiled = compile_model_source(loop_gmu_ufo, require_supported=False)
    request, _ignored = loop_v2_generate._fixture_request(
        reference, compiled, argparse.Namespace()
    )
    return reference, compiled, LoopSlice.generate(compiled, request, loop_native)


@pytest.mark.loop_nightly
def test_gg_ttx_electroweak_equals_madloop_at_point_0(gg_ttx_ew):
    """The full NLO EW virtual of g g > t t~ (QCD and QED corrections,
    G_mu, real masses, widths zero) equals MadLoop on the same UFO at the
    fixture's point 0 to 1e-12 per Laurent coefficient (the fixture's
    metric) through the native executor; the request took 't Hooft-Feynman
    gauge and every family has rank <= N."""

    from fractions import Fraction

    from pyamplicol.loops.v2.plan_export import NativeExecutor, export_plan
    from tools import loop_v2_generate

    reference, _compiled, slice_ = gg_ttx_ew
    assert slice_.compiled.ir.gauge == "feynman"
    assert all(
        family.rank <= len(key) for key, family in slice_.families.families.items()
    )
    executor = NativeExecutor(
        slice_, forward=False, exported=export_plan(slice_, forward=False)
    )
    (result,) = executor.evaluate(
        [reference["points"][0]],
        [str(Fraction(reference["mu_r"][0]) ** 2)],
        parameters=reference["parameters"],
    )
    mine = [float(x) for x in result.totals[:4]]
    madloop = reference["rows"][0]["providers"]["madloop"]
    alpha = float(reference["derived_parameters"]["alpha"])
    floor = abs(mine[0]) * alpha / (2 * math.pi)
    keys = ("born", "finite", "single_pole", "double_pole")
    for key, value in zip(keys, mine, strict=True):
        error = loop_v2_generate._relative(
            value, float(madloop[key]), 0.0 if key == "born" else floor
        )
        assert error <= EW_TOLERANCE, (key, value, madloop[key], error)


@pytest.mark.loop_nightly
def test_gg_ttx_electroweak_native_counterterm_program(gg_ttx_ew):
    """A4's native counterterm program on an electroweak control (the
    't Hooft-Feynman-gauge model's Goldstone vertices and counterterms, the
    lowered model's clone lines) equals the tree engine's, block by block
    (the block-set oracle of ``test_loop_v2_native_dag``)."""

    from pyamplicol.loops.v2 import ct_closure, dag_input

    _reference, _compiled, slice_ = gg_ttx_ew
    program = slice_.counterterm_program.blocks
    with dag_input.native_flags(dag=False):
        python = ct_closure.CountertermBlocks.build(
            program.recursion, program.blocks.blocks_dag.process
        )
    assert program.blocks.blocks_dag.decode() == python.blocks.blocks_dag
    assert program.blocks.blocks == python.blocks.blocks
    assert dict(program.clones) == dict(python.clones) and program.clones
    goldstones = {
        record.pdg_code
        for record in slice_.compiled.ir.particles
        if record.goldstoneboson
    }
    assert any(block.key.particle_id in goldstones for block in program.blocks.blocks)


@pytest.mark.loop_nightly
def test_gg_ttx_electroweak_validates_with_native_inputs(
    gg_ttx_ew, tmp_path, monkeypatch
):
    """``tools/loop_v2_generate.py validate --inputs native`` on g g > t t~
    EW: the native point inputs (WP3.2 EWI-N; ``point_inputs`` "native-v2",
    no listed refusal) at the fixture's point 0 within the fixture's
    tolerance (WP3.2 M8)."""

    import json

    from pyamplicol.loops.v2.assemble import LoopSlice
    from tools import loop_v2_generate

    _reference, compiled, slice_ = gg_ttx_ew
    monkeypatch.setattr(loop_v2_generate, "_fixture_model", lambda *a: compiled)
    monkeypatch.setattr(LoopSlice, "generate", lambda *a, **k: slice_)
    out = tmp_path / "record.json"
    arguments = ["validate", "gg_ttx_ew", "--inputs", "native", "--points", "0"]
    status = loop_v2_generate.main([*arguments, "--json", str(out)])
    record = json.loads(out.read_text())
    refusal = record.get("native_refusal")
    assert status == 0 and refusal is None, refusal
    assert record["point_inputs"] == "native-v2", record
    assert record["passed"] and record["points"][0]["max_error"] <= EW_TOLERANCE


@pytest.mark.loop_nightly
def test_qcd_corrections_agree_in_both_gauges(gmu_massive_b, loop_native):
    """M5, the premise of the gauge scope (WP3.2 EWI-4b): QCD corrections
    keep their unitary-gauge trees because both gauges are exact for them.
    b b~ > t t~ QED=2 [virt=QCD] with a massive b has an internal Z between
    massive currents: the 't Hooft-Feynman request inserts the QCD
    counterterms of the explicit G0 b b~ and G0 t t~ vertices, which the
    unitary one cannot hold (its tree model absorbs the Goldstones, DESIGN
    §6.2); Born and virtual agree (32 digits: binary64 holds the virtual to
    2e-9 only at this massive-b point, the reduction's conditioning that the
    precision-ap package recorded; M5 2026-10-26: 1.4e-16 at 32 digits)."""

    from pyamplicol.loops.v2.assemble import LoopSlice, Precision, SliceRequest

    request = SliceRequest(
        "b b~ > t t~",
        {"QCD": 0, "QED": 2},
        {"QCD": 2, "QED": 2},
        fixed_parameters=ZERO_WIDTHS,
        compatibility="gmu_particle_anti",
        scale_parameter="MU_R",
    )
    masses = [_mass(gmu_massive_b, name) for name in ("MB", "MB", "MT", "MT")]
    energy, cosine, sine = 500.0, 0.6, 0.8
    beam = math.sqrt(energy**2 - masses[0] ** 2)
    top = math.sqrt(energy**2 - masses[2] ** 2)
    point = [
        [repr(energy), "0", "0", repr(beam)],
        [repr(energy), "0", "0", repr(-beam)],
        [repr(energy), repr(top * sine), "0", repr(top * cosine)],
        [repr(energy), repr(-top * sine), "0", repr(-top * cosine)],
    ]
    goldstones = {
        record.pdg_code
        for record in gmu_massive_b.ir.particles
        if record.goldstoneboson
    }
    results = {}
    for gauge in ("unitary", "feynman"):
        compiled = gmu_massive_b if gauge == "unitary" else _feynman(gmu_massive_b)
        slice_ = LoopSlice.generate(compiled, request, loop_native)
        assert slice_.compiled.ir.gauge == gauge
        program = slice_.counterterm_program.blocks.blocks
        species = {block.key.particle_id for block in program.blocks}
        results[gauge] = (
            slice_.evaluate(
                point,
                "250000",
                parameters={"aS": "0.118"},
                precision=Precision.arbitrary(32),
            ),
            species & goldstones,
        )
    (unitary, absorbed), (feynman, explicit) = results["unitary"], results["feynman"]
    # The counterterm program carries Goldstone currents in 't Hooft-Feynman
    # gauge only: there its G0 counterterms are inserted.
    assert not absorbed and explicit
    for sector, born in unitary.born.items():
        assert float(feynman.born[sector]) == pytest.approx(float(born), rel=1e-13)
    for sector, laurent in unitary.components["virtual"].items():
        scale = max(abs(float(x)) for x in laurent.coefficients)
        for mine, theirs in zip(
            feynman.components["virtual"][sector].coefficients,
            laurent.coefficients,
            strict=True,
        ):
            assert abs(float(mine) - float(theirs)) <= 1e-14 * scale


@pytest.mark.loop_nightly
def test_born_closure_reads_the_z_h_goldstone_root_in_both_gauges(gmu, loop_native):
    """z h > t t~ QED=2 [virt=QCD]: the slice closes its Born and its
    counterterms at leg 1, the Z, where the 't Hooft-Feynman gauge adds the
    all-self-conjugate, momentum-odd Z-H-G0 vertex (the G0 s-channel). The
    root closure reads it as the tree engine does
    (``ct_sites.SiteClosure._reading_sign``, -1 on it; WP3.2 EWI-7b): Born
    and virtual agree between the gauges (one point, 32 digits). With the
    closure's own reading (sign 1) the Feynman-gauge Born was 56.37 against
    2.86 in unitary gauge."""

    from pyamplicol.loops.v2.assemble import LoopSlice, Precision, SliceRequest

    request = SliceRequest(
        "z h > t t~",
        {"QCD": 0, "QED": 2},
        {"QCD": 2, "QED": 2},
        fixed_parameters={"MB": "0", "ymb": "0", **ZERO_WIDTHS},
        compatibility="gmu_particle_anti",
        scale_parameter="MU_R",
    )
    point = [
        [repr(x) for x in leg] for leg in _process_point(gmu, "z h > t t~", 1000.0)
    ]
    results = {}
    for gauge in ("unitary", "feynman"):
        compiled = gmu if gauge == "unitary" else _feynman(gmu)
        slice_ = LoopSlice.generate(compiled, request, loop_native)
        assert slice_.compiled.ir.gauge == gauge
        results[gauge] = slice_.evaluate(
            point,
            "250000",
            parameters={"aS": "0.118"},
            precision=Precision.arbitrary(32),
        )
    unitary, feynman = results["unitary"], results["feynman"]
    for sector, born in unitary.born.items():
        assert float(feynman.born[sector]) == pytest.approx(float(born), rel=1e-13)
    for sector, laurent in unitary.components["virtual"].items():
        scale = max(abs(float(x)) for x in laurent.coefficients)
        for mine, theirs in zip(
            feynman.components["virtual"][sector].coefficients,
            laurent.coefficients,
            strict=True,
        ):
            assert abs(float(mine) - float(theirs)) <= 1e-14 * scale


def _born_and_virtual(compiled, process, point, loop_native):
    """Born and virtual of ``process`` (QED=2 [virt=QCD], G_mu, widths zero)
    at one point, 32 digits."""

    from pyamplicol.loops.v2.assemble import LoopSlice, Precision, SliceRequest

    request = SliceRequest(
        process,
        {"QCD": 0, "QED": 2},
        {"QCD": 2, "QED": 2},
        fixed_parameters=ZERO_WIDTHS,
        compatibility="gmu_particle_anti",
        scale_parameter="MU_R",
    )
    slice_ = LoopSlice.generate(compiled, request, loop_native)
    assert slice_.compiled.ir.gauge == compiled.ir.gauge
    return slice_.evaluate(
        point, "250000", parameters={"aS": "0.118"}, precision=Precision.arbitrary(32)
    )


def _assert_same_born_and_virtual(found, expected):
    for sector, born in expected.born.items():
        assert float(found.born[sector]) == pytest.approx(float(born), rel=1e-13)
    for sector, laurent in expected.components["virtual"].items():
        scale = max(abs(float(x)) for x in laurent.coefficients)
        for mine, theirs in zip(
            found.components["virtual"][sector].coefficients,
            laurent.coefficients,
            strict=True,
        ):
            assert abs(float(mine) - float(theirs)) <= 1e-14 * scale


@pytest.mark.loop_nightly
def test_root_closure_reads_the_w_goldstone_higgs_root_in_both_gauges_and_leg_orders(
    gmu_massive_b, loop_native
):
    """w+ h > t b~ QED=2 [virt=QCD] with a massive b: in 't Hooft-Feynman
    gauge its Born and its counterterms close at leg 1, the W+, on the
    W G H vertex, which is not closed under conjugation (read through its
    conjugate rule) and odd in the momenta. The root closure reads it as
    the tree engine does (``ct_sites.SiteClosure._reading_sign``, -1 on it,
    with each kernel leg taking the momentum of the block whose row it
    takes; genericity audit 2 G2-M7): Born and virtual equal the
    unitary-gauge ones, and the legs in the other order (h w+ > t b~, the H
    at leg 1 closing on the same vertex, at the same physical point) give the
    same values (one point, 32 digits). With the earlier reading sign 1 the
    Feynman-gauge Born is wrong."""

    unitary, feynman = gmu_massive_b, _feynman(gmu_massive_b)
    point = [
        [repr(x) for x in leg]
        for leg in _process_point(gmu_massive_b, "w+ h > t b~", 1000.0)
    ]
    swapped = [point[1], point[0], *point[2:]]
    reference = _born_and_virtual(unitary, "w+ h > t b~", point, loop_native)
    _assert_same_born_and_virtual(
        _born_and_virtual(feynman, "w+ h > t b~", point, loop_native), reference
    )
    _assert_same_born_and_virtual(
        _born_and_virtual(feynman, "h w+ > t b~", swapped, loop_native), reference
    )


@pytest.mark.loop_nightly
def test_born_closure_reads_the_w_w_vector_root_as_the_time_reversed_process(
    gmu, loop_native
):
    """w+ w- > u u~ closes its Born at leg 1, the W+, on the W+ W- Z and
    W+ W- A vertices, closed under conjugation with the W+ and W- rows on
    each other's UFO legs: each kernel leg takes the momentum of the block
    whose row it takes and the term the reading sign -1 (genericity audit 2
    G2-M7). Its summed squared Born equals the time-reversed u u~ > w+ w-'s
    (closed at a quark root) at the same momenta in both gauges (32
    digits); the earlier reading gave it 177 times that."""

    import math as _math

    from pyamplicol.loops.v2.assemble import LoopSlice, Precision, SliceRequest

    mass = _mass(gmu, "MW")
    energy, cosine, sine = 250.0, 0.6, 0.8
    momentum = _math.sqrt(energy**2 - mass**2)
    w_plus, w_minus = [energy, 0.0, 0.0, momentum], [energy, 0.0, 0.0, -momentum]
    quark = [energy, energy * sine, 0.0, energy * cosine]
    antiquark = [energy, -energy * sine, 0.0, -energy * cosine]
    fixed = {"MB": "0", "ymb": "0", **ZERO_WIDTHS}
    for compiled in (gmu, _feynman(gmu)):
        summed = {}
        # Initial spin x colour averages: 3 x 3 for the W pair, 2 x 2 x 3 x 3
        # for the quark pair.
        for process, legs, average in (
            ("w+ w- > u u~", (w_plus, w_minus, quark, antiquark), 9),
            ("u u~ > w+ w-", (quark, antiquark, w_plus, w_minus), 36),
        ):
            request = SliceRequest(
                process,
                {"QCD": 0, "QED": 2},
                {"QCD": 2, "QED": 2},
                fixed_parameters=fixed,
                compatibility="gmu_particle_anti",
                scale_parameter="MU_R",
            )
            slice_ = LoopSlice.generate(compiled, request, loop_native)
            result = slice_.evaluate(
                [[repr(x) for x in leg] for leg in legs],
                "250000",
                parameters={"aS": "0.118"},
                precision=Precision.arbitrary(32),
            )
            summed[process] = average * sum(float(x) for x in result.born.values())
        assert summed["w+ w- > u u~"] == pytest.approx(
            summed["u u~ > w+ w-"], rel=1e-13
        )


# g t > t t t~ (binary64, on shell to rounding): a top line joined to a top
# pair by a Z, a photon or a G0, with the gluon at leg 1.
GT_TTTX_POINT = [
    ["740.0236666666667", "0", "0", "740.0236666666667"],
    ["759.9763333333334", "0", "0", "-740.0236666666667"],
    [
        "542.1014268447362",
        "453.946802621722",
        "-234.14080644548642",
        "55.27513124065375",
    ],
    [
        "679.7890319233611",
        "-642.7918716317638",
        "125.73449804624319",
        "-56.51171284339019",
    ],
    [
        "278.1095412319028",
        "188.84506901004178",
        "108.40630839924326",
        "1.2365816027364396",
    ],
]


@pytest.mark.loop_nightly
def test_goldstone_vector_mixings_equal_the_site_oracle(
    gmu_massive_b, loop_native, monkeypatch
):
    """O6 for the Goldstone-vector mixings (WP3.2 EWI-4b): g t > t t t~ with
    QED corrections, whose trees reach the G0-Z, G0-A and G+-W- mixings
    (admitted and lowered as vertex insertions), equals the request with
    every rule closed at sites (the site oracle's two-block sites read the
    mixings with the catalogue's ct-site kernels). The site oracle's
    coverage rule has no electroweak certificate of the g t t~ witnesses
    with electroweak loop species, whose divergences the field and mass
    counterterms cover: certified for the comparison only. The mixings make
    0.3 % of the counterterms' finite part here; the opposite momentum sign
    differs from the oracle by twice that.

    The b is massive: the oracle admits a rule's loop species only where a
    divergent witness has exactly those species, while the UFO names the
    W for its W and G+ loops (``R2_Htt_b`` of [t~ t H], loop [W+, b], is
    proportional to MT (1 + MB^2/MW^2), non-zero at MB = 0). With a
    massless b the only divergent b-W loops of [t~ t H] and [t~ t G0] also
    carry a G+ (witness species (b, W, G+)), so the oracle drops those two
    R2 rules (about half the R2 here); with a massive b the H and G0
    couple to the b and a (b, W) witness exists. Production admits by
    coupling orders and particle_veto only (REQUIREMENTS_ADDENDUM J) and
    reads no witness."""

    from pyamplicol.loops.v2 import ct_sites
    from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest

    coverage = ct_sites._coverage

    def certified(witnesses, rules, names, corrections, partners, zeros):
        electroweak = tuple(
            ct_sites.CertifiedZero(signature, species, "electroweak (test)")
            for signature, group in witnesses.items()
            for species in group
        )
        return coverage(
            witnesses, rules, names, corrections, partners, (*zeros, *electroweak)
        )

    monkeypatch.setattr(ct_sites, "_coverage", certified)
    fixed = {"MTA": "0", "ymtau": "0", **ZERO_WIDTHS}
    results = {}
    for counterterms in ("recursion", "sites"):
        request = SliceRequest(
            "g t > t t t~",
            {"QCD": 1, "QED": 2},
            {"QCD": 1, "QED": 4},
            corrections=("QED",),
            fixed_parameters=fixed,
            compatibility="gmu_particle_anti",
            scale_parameter="MU_R",
            counterterms=counterterms,
        )
        slice_ = LoopSlice.generate(gmu_massive_b, request, loop_native)
        if counterterms == "recursion":
            lowered = slice_.counterterm_program.recursion.model.lowered
            spins = {p.name: int(p.spin) for p in lowered.compiled.ir.particles}
            mixings = {
                structure.particles
                for structure in lowered.insertions
                if {spins[name] for name in structure.particles} == {1, 3}
            }
            assert len(mixings) == 4, mixings
        results[counterterms] = slice_.evaluate(
            GT_TTTX_POINT, "1000000", parameters={"aS": "0.118"}
        )
    mine, oracle = results["recursion"], results["sites"]

    def totals(result, sector):
        uv = [
            a + b
            for a, b in zip(
                result.components["uv_local"][sector].coefficients,
                result.components["uv_external_fields"][sector].coefficients,
                strict=True,
            )
        ]
        return {
            "r2": result.components["r2"][sector].coefficients,
            "uv": uv,
            "virtual": result.components["virtual"][sector].coefficients,
        }

    for sector in oracle.components["virtual"]:
        expected, found = totals(oracle, sector), totals(mine, sector)
        for name, values in expected.items():
            scale = max(abs(complex(x)) for x in values)
            for a, b in zip(found[name], values, strict=True):
                assert abs(complex(a) - complex(b)) <= 1e-12 * scale, (name, a, b)


def _uv_component_equals_madloop(compiled, name, loop_reference, loop_native):
    """The UV counterterm component of the request of ``name`` in
    ``goldstone-mixing.json`` against MadLoop's Born x CT interference
    without its R2 amplitudes, per point, finite part and single pole within
    1e-12 of MadLoop's (binary64). Returns the slice and the (point, mu_R^2,
    fixture row) triples."""

    from fractions import Fraction

    from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest

    reference = loop_reference("goldstone-mixing")["processes"][name]
    known = set(compiled.parameter_defaults)
    request = SliceRequest(
        reference["process"].split(" QCD=")[0],
        reference["born_orders"],
        reference["loop_orders"],
        corrections=("QED",),
        fixed_parameters={
            key: value
            for key, value in reference["fixed_parameters"].items()
            if key in known
        },
        compatibility="gmu_particle_anti",
        scale_parameter="MU_R",
    )
    slice_ = LoopSlice.generate(compiled, request, loop_native)
    evaluated = []
    for point, mu, row in zip(
        reference["points"], reference["mu_r"], reference["rows"], strict=True
    ):
        result = slice_.evaluate(
            point, str(Fraction(mu) ** 2), parameters=reference["parameters"]
        )
        evaluated.append((point, str(Fraction(mu) ** 2), row))
        assert sum(float(x) for x in result.born.values()) == pytest.approx(
            row["born"], rel=1e-13
        )
        (sector,) = result.components["uv_local"]
        uv = [
            float(a) + float(b)
            for a, b in zip(
                result.components["uv_local"][sector].coefficients,
                result.components["uv_external_fields"][sector].coefficients,
                strict=True,
            )
        ]
        mixing = row.get("mixing") or {
            key: row["neutral"][key] + row["charged"][key] for key in row["ct"]
        }
        for index, key in enumerate(("finite", "single_pole")):
            expected = row["ct"][key] - row["r2"][key]
            # The opposite sign of the mixings would miss by twice theirs.
            assert abs(mixing[key]) > 1e-9 * abs(expected)
            assert abs(uv[index] - expected) <= 1e-12 * abs(expected), (key, uv)
        assert uv[2] == row["ct"]["double_pole"] == 0
    return slice_, evaluated


@pytest.mark.loop_nightly
def test_goldstone_vector_mixings_equal_madloop(
    gmu, loop_gmu_ufo, loop_reference, loop_native
):
    """The Goldstone-vector mixings' sign, decided by MadLoop (WP3.2 EWI-7b):
    g t > t t t~ with QED corrections (massless b), whose trees join the
    top line and the top pair through a Z, a photon or a G0, reaches the
    G0-Z and G0-A mixing counterterms (lowered as clone lines,
    ``ct_line_insertions._momentum_ratio``). Its UV counterterm component
    equals MadLoop's Born x CT interference without the R2 amplitudes
    (``goldstone-mixing.json``, ``tools/loop_reference/madloop_ct_selection.py``)
    at three points, finite part and single pole within 1e-12 (measured
    4e-14, MadLoop's binary64); the mixings make 0.3, 3 and 0.45 % of it,
    and the earlier reading (the scalar leg of the mixing taking the momentum
    flowing in) missed by twice that.

    The whole virtual is MadLoop's too (WP3.2 EWI-5): two massive top lines
    joined through the loop by Z, photon, H, G0, W and G+- (W-b and G+--b
    boxes and pentagons on both lines). The bare part equals MadLoop's
    virtual minus its counterterms (finite part and single pole), the single
    pole equals MadLoop's and the QED Catani operator's, and R2 and the
    finite part equal MadLoop's plus the R2 coupling MadLoop omits
    (``omitted_r2``: R2_G0tt_b of V_ttG02, loop label [W+, b], never matched
    in 't Hooft-Feynman gauge, where every b loop of t~ t G0 holds a G+-;
    as in ``required/emep_ttxh_ew``). The omitted term is (R2_G0tt_b /
    R2_G0tt) times MadLoop's R2_G0tt component, the ratio from the UFO's
    couplings at the card (0.869 at lhv = 1). Evaluated with 32 digits
    (binary64 loses 3e-11 of the finite part at point 2, MadLoop's own
    estimate there 1.5e-12; the binary64 conditioning is WP3.4's stability
    work): within MadLoop's accuracy of its virtual (its binary64: 1e-12,
    or MadLoop's own estimate ``virtual_accuracy`` where larger, the O1 rule;
    measured 4.6e-14), the poles within 1e-12 of Catani's (measured
    1.2e-16). Before the loop segments read the Goldstone vertices as the
    tree engine does (``catalogue._CatalogueBuilder.reading_sign``, EWI-5),
    the bare part missed MadLoop's by O(1) (single pole -1.07e-7 against
    -3.78e-8 at point 0, the virtual's -5.6e-8 against Catani's 1.35e-8).
    About 20 minutes."""

    from pyamplicol.loops.v2.assemble import Precision

    reference = loop_reference("goldstone-mixing")["processes"]["gt_tttx_ew"]
    slice_, evaluated = _uv_component_equals_madloop(
        gmu, "gt_tttx_ew", loop_reference, loop_native
    )
    omitted = reference["omitted_r2"]
    assert omitted["madloop_amplitudes"][omitted["coupling"]] == 0
    assert omitted["madloop_amplitudes"][omitted["proportional_to"]] > 0
    # restrict_default.dat's diagonal CKM (its Wolfenstein inputs are zero):
    # the patched UFO's CKM33 depends on them, the unpatched one's did not.
    wolfenstein = {name: "0" for name in ("lamWS", "AWS", "rhoWS", "etaWS")}
    card = {**wolfenstein, **reference["parameters"], **reference["fixed_parameters"]}
    ratio = coupling_ratio(
        loop_gmu_ufo, omitted["coupling"], omitted["proportional_to"], card
    )
    assert 0.5 < ratio < 1.0, ratio
    keys = ("finite", "single_pole", "double_pole")
    worst = {"madloop": 0.0, "catani": 0.0}
    for point, scale_squared, row in evaluated:
        result = slice_.evaluate(
            point,
            scale_squared,
            parameters=reference["parameters"],
            precision=Precision.arbitrary(32),
        )
        (sector,) = result.components["virtual"]

        def laurent(component, sector=sector, result=result):
            return [float(x) for x in result.components[component][sector].coefficients]

        madloop = [row["virtual"][key] for key in keys]
        scale = max(abs(x) for x in madloop)
        bound = max(1e-12, row["virtual_accuracy"])
        omitted_r2 = ratio * row["r2_g0tt"]["finite"]
        assert abs(omitted_r2) > 1e-3 * scale
        expected = {
            "virtual": [madloop[0] + omitted_r2, madloop[1], 0.0],
            "bare": [madloop[i] - row["ct"][key] for i, key in enumerate(keys[:2])]
            + [0.0],
            "r2": [row["r2"]["finite"] + omitted_r2, 0.0, 0.0],
        }
        for component, values in expected.items():
            for found, want in zip(laurent(component), values, strict=True):
                error = abs(found - want) / scale
                worst["madloop"] = max(worst["madloop"], error)
                assert error <= bound, (component, found, want)
        single, double = (float(x) for x in result.ir_poles[sector])
        virtual = laurent("virtual")
        for found, want in zip(virtual[1:], (single, double), strict=True):
            error = abs(found - want) / scale
            worst["catani"] = max(worst["catani"], error)
            assert error <= 1e-12, (virtual, single, double)
    print({key: f"{error:.1e}" for key, error in worst.items()})


@pytest.mark.loop_nightly
def test_charged_goldstone_vector_mixings_equal_madloop(
    gmu_massive_b, loop_reference, loop_native
):
    """The charged Goldstone-vector mixings' sign against MadLoop: g b > t t~ b
    with QED corrections and a massive b reaches the G-W+ mixing (with a
    t-channel W between the b-t lines), which makes 150 % of its UV
    counterterm component at point 0 (``goldstone-mixing.json``). The clone
    lines read the charged mixings as the neutral ones
    (``ct_line_insertions._momentum_ratio``). Its loops hold the W ghost
    vertices [ghA ghWm~ W-], read on their own legs since WP3.2 EWI-7
    (``conventions.conjugate_vertex_rules``; refused before). Evaluated by
    the native executor in double-double at the three points: Born, the UV
    counterterm component (the opposite mixing sign would miss it by three
    times its size), R2 and every pole of the whole virtual equal MadLoop's
    to 1e-12 and the poles Catani's; the finite parts to 1e-8 (MadLoop's
    double-precision counterterm parameters, below). Binary64 loses 2e-5 of
    the single pole to cancellations at point 0 (mb^2/s = 2e-5), which its
    stability estimate flags (one rotation and the rescaled reduction check:
    3e-4 to 1e-3 at the three points; the O1 rule takes such points to
    double-double). About 6 minutes."""

    from fractions import Fraction

    from pyamplicol.loops.v2.assemble import LoopSlice, Precision, SliceRequest
    from pyamplicol.loops.v2.plan_export import NativeExecutor, export_plan

    reference = loop_reference("goldstone-mixing")["processes"]["gb_ttxb_ew_mb"]
    known = set(gmu_massive_b.parameter_defaults)
    request = SliceRequest(
        reference["process"].split(" QCD=")[0],
        reference["born_orders"],
        reference["loop_orders"],
        corrections=("QED",),
        fixed_parameters={
            key: value
            for key, value in reference["fixed_parameters"].items()
            if key in known
        },
        compatibility="gmu_particle_anti",
        scale_parameter="MU_R",
    )
    slice_ = LoopSlice.generate(gmu_massive_b, request, loop_native)
    executor = NativeExecutor(
        slice_, forward=False, exported=export_plan(slice_, forward=False)
    )
    keys = ("finite", "single_pole", "double_pole")
    worst = {"uv": 0.0, "r2": 0.0, "virtual": 0.0, "catani": 0.0}
    finite = {"uv": 0.0, "virtual": 0.0}
    for point, mu, row in zip(
        reference["points"], reference["mu_r"], reference["rows"], strict=True
    ):
        (result,) = executor.evaluate(
            [point],
            [str(Fraction(mu) ** 2)],
            parameters=reference["parameters"],
            precision=Precision.double_double(),
        )
        assert float(sum(result.born.values())) == pytest.approx(row["born"], rel=1e-13)
        (sector,) = result.components["virtual"]

        def laurent(*names, result=result, sector=sector):
            return [
                sum(
                    float(result.components[name][sector].coefficients[i])
                    for name in names
                )
                for i in range(3)
            ]

        mixing = row.get("mixing") or {
            key: row["neutral"][key] + row["charged"][key] for key in row["ct"]
        }
        uv = [row["ct"][k] - row["r2"][k] for k in keys]
        for name, mine, want in (
            ("uv", laurent("uv_local", "uv_external_fields"), uv),
            ("r2", laurent("r2"), [row["r2"][k] for k in keys]),
            ("virtual", laurent("virtual"), [row["virtual"][k] for k in keys]),
        ):
            scale = max(abs(x) for x in want)
            for index, (found, expected) in enumerate(zip(mine, want, strict=True)):
                error = abs(found - expected) / scale
                if index == 0 and name != "r2":
                    # Finite parts: see below (user ruling 2026-10-29 ~10:20).
                    finite[name] = max(finite[name], error)
                    assert error <= 1e-8, (name, mine, want)
                    continue
                worst[name] = max(worst[name], error)
                assert error <= 1e-12, (name, mine, want)
        # The opposite sign of the mixings would miss by twice theirs.
        assert abs(mixing["finite"]) > 1e-9 * abs(row["ct"]["finite"])
        single, double = (float(x.real) for x in map(complex, result.ir_poles[sector]))
        virtual = laurent("virtual")
        scale = max(abs(single), abs(double))
        for found, want in zip(virtual[1:], (single, double), strict=True):
            error = abs(found - want) / scale
            worst["catani"] = max(worst["catani"], error)
            assert error <= 1e-12, (virtual, single, double)
    print({key: f"{error:.1e}" for key, error in worst.items()}, finite)
    # The finite parts of the UV component and of the virtual differ from
    # MadLoop's by at most 1.3e-10 and 1.1e-9 of their largest coefficients
    # (measured 2026-10-28), the same in binary64, double-double and 40
    # digits on this side, while R2 and every pole agree to 1e-13. At this
    # card the plan's counterterm parameters equal the UFO's own 40-digit
    # evaluation to 4e-26, while the UFO's binary64 evaluation of the massive
    # b's field counterterms (bWcft_UV_EW_R, dMB_bWcft_UV_EW_R) is off by
    # 4.5e-11, consistent with MadLoop's double-precision counterterm
    # parameters. Below the user's 1e-5 attention threshold (2026-10-29
    # ~10:20), so the finite parts are held to 1e-8.
