# SPDX-License-Identifier: 0BSD
"""What Catani's operators predict, decided structurally (WP3.2 ew-poles
batch 2; ``loops.v2.poles``, ``loops.v2.ew_policy``, ``assemble._ir_poles``).

* The strong and the electromagnetic corrections are the coupling orders the
  gluon's and the photon's vertices carry (``ew_policy.correction_orders``),
  whatever the orders are called (genericity audit m1).
* beta_0 (= gamma_g) runs over the request's eligible massless coloured
  species with their T(R) and spin weights, and each leg's gamma by its
  spin and representation; no prediction otherwise (genericity audit m3).
* A sector is predicted only where the requested corrections complete its
  loops (``poles.predicted_sectors``): the mixed QCD x EW sectors of a
  QCD-only request are not (slice-status DECISIONS 2026-10-25 ~13:30, routed
  from ew-lines: their prediction was 2 % off the virtual's single pole);
  nor is a sector whose Born trees join the charged legs in different ways
  through lines no requested correction circulates (``poles.
  separated_exchange``, mq-ewjoin: MQ-VALIDATE finding F1, u d > u d h at
  QCD = 0, whose single pole equals MadLoop's but misses Catani's by 0.1).
  Every unpredicted sector has a typed reason (``poles.sector_predictions``,
  ``LoopSlice.unpredicted_sectors``).
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

from pyamplicol.loops.v2.ew_policy import CorrectionOrders, correction_orders
from pyamplicol.loops.v2.families import add_orders
from pyamplicol.loops.v2.poles import (
    SEPARATED_EXCHANGE,
    UNCOMPUTED_BORN_PAIR,
    UNCORRECTED_EXCHANGE,
    predicted_sectors,
    sector_predictions,
    separated_exchange,
    strong_beta0,
    strong_gamma,
)

ROOT = Path(__file__).resolve().parents[2]
TOLERANCE = 1e-12
QCD2, QED2 = (("QCD", 2),), (("QED", 2),)
MIXED = (("QCD", 1), ("QED", 1))


# ----------------------------------------------- m1: orders by their bosons


def _particle(name, anti, pdg, spin, colour, charge=0.0, mass="ZERO", **extra):
    return SimpleNamespace(
        name=name,
        antiname=anti,
        pdg_code=pdg,
        spin=spin,
        color=colour,
        charge=charge,
        mass=mass,
        ghost_number=extra.get("ghost", 0),
        goldstoneboson=extra.get("goldstone", False),
        auxiliary_kind=extra.get("auxiliary"),
    )


def _vertex(particles, orders):
    return SimpleNamespace(particles=tuple(particles), coupling_orders=tuple(orders))


def _toy_model(strong: str, abelian: str):
    """A toy model with a gluon, a photon, a Z, an up quark and an electron
    whose orders are named ``strong`` and ``abelian`` (plus a Yukawa-like
    vertex of another order that neither boson carries)."""

    particles = [
        _particle("g", "g", 21, 3, 8),
        _particle("a", "a", 22, 3, 1),
        _particle("z", "z", 23, 3, 1, mass="MZ"),
        _particle("u", "u~", 2, 2, 3, 2 / 3),
        _particle("u~", "u", -2, 2, -3, -2 / 3),
        _particle("e-", "e+", 11, 2, 1, -1.0),
        _particle("e+", "e-", -11, 2, 1, 1.0),
        _particle("h", "h", 25, 1, 1, mass="MH"),
        _particle("gh", "gh~", 82, 1, 8, ghost=1),
    ]
    vertices = [
        _vertex(("g", "g", "g"), ((strong, 1),)),
        _vertex(("u~", "u", "g"), ((strong, 1),)),
        _vertex(("u~", "u", "a"), ((abelian, 1),)),
        _vertex(("e+", "e-", "a"), ((abelian, 1),)),
        _vertex(("u~", "u", "z"), ((abelian, 1),)),
        _vertex(("u~", "u", "h"), (("YUK", 1),)),
        _vertex(("g", "g", "g", "g"), ((strong, 2),)),
    ]
    return SimpleNamespace(
        ir=SimpleNamespace(particles=particles, parameters=[], vertex_terms=vertices)
    )


@pytest.mark.loop_fast
@pytest.mark.parametrize(("strong", "abelian"), [("QCD", "QED"), ("QS", "Qe")])
def test_correction_orders_follow_the_gauge_bosons(strong, abelian):
    """The strong order is the one the gluon's minimal gauge vertices carry,
    the abelian one the photon's charged-pair vertices': renamed orders are
    found alike, the Yukawa order and the Z's vertex play no part."""

    found = correction_orders(_toy_model(strong, abelian), {})
    assert found == CorrectionOrders(
        strong, abelian, frozenset({"g"}), frozenset({"a"})
    )


@pytest.mark.loop_fast
def test_correction_orders_are_none_without_a_single_common_order():
    """A gluon whose gauge vertices all carry the same two orders, or a
    model without a photon, identifies no such order (no operator, fail
    closed); the order every gauge vertex carries is found whatever else
    some of them carry."""

    model = _toy_model("QCD", "QED")
    both = (("QCD", 1), ("NP", 1))
    model.ir.vertex_terms[0] = _vertex(("g", "g", "g"), both)
    model.ir.vertex_terms[1] = _vertex(("u~", "u", "g"), both)
    assert correction_orders(model, {}).strong is None
    model.ir.vertex_terms[0] = _vertex(("g", "g", "g"), (("NP", 1),))
    assert correction_orders(model, {}).strong == "NP"
    model = _toy_model("QCD", "QED")
    model.ir.particles = [p for p in model.ir.particles if p.name != "a"]
    model.ir.vertex_terms = [v for v in model.ir.vertex_terms if "a" not in v.particles]
    assert correction_orders(model, {}).abelian is None


@pytest.mark.loop_physics
def test_shipped_models_take_qcd_and_qed_from_their_vertices(loop_sm_compiled_model):
    from pyamplicol.models.loading import load_compiled_model

    found = correction_orders(load_compiled_model(loop_sm_compiled_model), {})
    assert (found.strong, found.abelian) == ("QCD", "QED")
    assert found.gluons == frozenset({"G"}) and found.photons == frozenset({"A"})


# -------------------------------------------- m3: beta_0 and gamma by species

DIRAC_TRIPLET, MAJORANA_OCTET = (2, 3, False), (2, 8, True)
COMPLEX_TRIPLET, REAL_OCTET = (1, 3, False), (1, 8, True)


@pytest.mark.loop_fast
def test_beta0_weights_each_species_by_its_index_and_spin():
    """11/6 C_A - 2/3 T_R n_f for n_f Dirac triplets (gamma_g of QCD); a
    vetoed b is a four-flavour beta_0; a gluino (Majorana octet) takes
    1/3 C_A, a complex scalar triplet 1/6 T_F, a real scalar octet 1/12 C_A;
    a coloured vector or an uncovered representation gives no beta_0."""

    assert strong_beta0([DIRAC_TRIPLET] * 5) == Fraction(11, 2) - Fraction(5, 3)
    assert strong_beta0([DIRAC_TRIPLET] * 4) == Fraction(11, 2) - Fraction(4, 3)
    assert strong_beta0([]) == Fraction(11, 2)
    assert strong_beta0([MAJORANA_OCTET]) == Fraction(11, 2) - 1
    assert strong_beta0([COMPLEX_TRIPLET]) == Fraction(11, 2) - Fraction(1, 12)
    assert strong_beta0([REAL_OCTET]) == Fraction(11, 2) - Fraction(1, 4)
    assert strong_beta0([(3, 8, True)]) is None
    assert strong_beta0([(2, 6, False)]) is None


@pytest.mark.loop_fast
def test_gamma_by_spin_and_representation():
    beta0 = strong_beta0([DIRAC_TRIPLET] * 5)
    assert strong_gamma(3, 8, beta0, gauge=True) == beta0
    assert strong_gamma(2, 3, beta0, gauge=False) == 2  # 3/2 C_F
    assert strong_gamma(2, -3, beta0, gauge=False) == 2
    assert strong_gamma(2, 8, beta0, gauge=False) == Fraction(9, 2)  # 3/2 C_A
    assert strong_gamma(1, 3, beta0, gauge=False) == Fraction(8, 3)  # 2 C_F
    assert strong_gamma(3, 8, beta0, gauge=False) is None  # not the gluon
    assert strong_gamma(2, 6, beta0, gauge=False) is None
    assert strong_gamma(3, 8, None, gauge=True) is None


# ------------------------------------------------- sectors the operators cover


def _sectors(born_orders, loop_orders):
    out: dict = {}
    for a in born_orders:
        for b in loop_orders:
            out.setdefault(add_orders(a, b), set()).add((a, b))
    return tuple(sorted((sector, frozenset(pairs)) for sector, pairs in out.items()))


def _orders(**powers):
    return add_orders(tuple(powers.items()))


@pytest.mark.loop_fast
def test_qcd_corrections_to_mixed_borns_predict_the_pure_qcd_sector_only():
    """u u~ > d d~ with gluon and photon Borns, QCD corrections: the loop of
    order g^2 e^2 is also the QED correction of the gluon Born, whose
    photon exchanges a QCD-only request leaves out, so only QCD^6 is
    predicted; with QCD and QED corrections every sector is."""

    born = [_orders(QCD=2), _orders(QED=2)]
    loops = [_orders(QCD=4), _orders(QCD=2, QED=2)]
    sectors = _sectors(born, loops)

    def unknown(order):
        return None

    assert predicted_sectors(sectors, born, ["QCD"], unknown) == {_orders(QCD=6)}
    loops.append(_orders(QED=4))
    sectors = _sectors(born, loops)
    assert predicted_sectors(sectors, born, ["QCD", "QED"], unknown) == {
        sector for sector, _pairs in sectors
    }
    # Without the loop order e^4 the QED operator's (g^2, e^2) interference
    # in QCD^2 QED^4 has no loop to pair with (the QED correction of the
    # photon Born, interfered with the gluon Born): that sector drops.
    sectors = _sectors(born, loops[:2])
    assert predicted_sectors(sectors, born, ["QCD", "QED"], unknown) == {
        _orders(QCD=6),
        _orders(QCD=4, QED=2),
    }


@pytest.mark.loop_fast
def test_absent_trees_keep_their_sectors_predicted():
    """QCD corrections to Drell-Yan (Born e^2): the loop g^2 e^2 would also
    be a QED correction of a g^2 tree, which u u~ > e+ e- has none of, so
    its sector is predicted; when that tree's existence is unknown it is
    not. A requested correction whose lower tree exists outside the Born
    program (bounds) gives no prediction either."""

    born = [_orders(QED=2)]
    sectors = _sectors(born, [_orders(QCD=2, QED=2)])
    absent = {_orders(QCD=2): False}
    assert predicted_sectors(sectors, born, ["QCD"], absent.get) == {
        _orders(QCD=2, QED=4)
    }
    assert predicted_sectors(sectors, born, ["QCD"], lambda order: None) == set()
    # Joint request on a gluon Born whose photon-exchange tree the Born
    # bounds leave out (QED = 0): no prediction for the mixed sector.
    born = [_orders(QCD=2)]
    sectors = _sectors(born, [_orders(QCD=4), _orders(QCD=2, QED=2)])
    exists = {_orders(QED=2): True}
    assert predicted_sectors(sectors, born, ["QCD", "QED"], exists.get) == {
        _orders(QCD=6)
    }
    # g g > g g: no tree of order e^2, both sectors are predicted.
    absent = {_orders(QED=2): False}
    assert predicted_sectors(sectors, born, ["QCD", "QED"], absent.get) == {
        _orders(QCD=6),
        _orders(QCD=4, QED=2),
    }


@pytest.mark.loop_fast
def test_separated_exchanges_need_one_partition_or_one_abelian_component():
    """``poles.separated_exchange``: a non-abelian operator matches the
    computed loops when every Born tree joins the charged legs into the
    same components (the cross-component correlators vanish), an abelian
    one only with a single component per tree."""

    z = frozenset({frozenset({1, 3}), frozenset({2, 4})})
    w = frozenset({frozenset({1, 4}), frozenset({2, 3})})
    joined = frozenset({frozenset({1, 2, 3, 4})})
    assert not separated_exchange([z, z], abelian=False)
    assert not separated_exchange([joined], abelian=False)
    assert separated_exchange([z, w], abelian=False)
    assert not separated_exchange([joined], abelian=True)
    assert separated_exchange([z], abelian=True)


@pytest.mark.loop_fast
def test_every_unpredicted_sector_has_a_typed_reason():
    """``poles.sector_predictions``: None for a predicted sector, else why:
    an uncorrected exchange, an uncomputed Born pair or a separated
    exchange (asked only of the dressed Born amplitude orders)."""

    born = [_orders(QED=3)]
    sectors = _sectors(born, [_orders(QCD=2, QED=3)])
    sector = _orders(QCD=2, QED=6)
    asked = []

    def separated(name, orders):
        asked.append((name, orders))
        return True

    absent = {_orders(QCD=2, QED=1): False}
    assert sector_predictions(sectors, born, ["QCD"], absent.get) == {sector: None}
    assert sector_predictions(sectors, born, ["QCD"], absent.get, separated) == {
        sector: SEPARATED_EXCHANGE
    }
    assert asked == [("QCD", frozenset(born))]
    assert sector_predictions(sectors, born, ["QCD"], lambda order: True) == {
        sector: UNCORRECTED_EXCHANGE
    }
    born = [_orders(QCD=2), _orders(QED=2)]
    sectors = _sectors(born, [_orders(QCD=4), _orders(QCD=2, QED=2)])
    found = sector_predictions(sectors, born, ["QCD", "QED"], lambda order: None)
    assert found[_orders(QCD=2, QED=4)] == UNCOMPUTED_BORN_PAIR


@pytest.mark.loop_physics
def test_quark_lines_joined_by_two_weak_exchanges_have_no_prediction(
    loop_sm_compiled_model, loop_native
):
    """u d > u d h at QCD = 0 (VBF, m-qcd row D19): its Born trees exchange a
    Z (u -> u, d -> d) or a W (u -> d, d -> u) between the quark lines, and
    a QCD request leaves out the gluon exchanges between the lines (their
    loops carry the Z or W), whose interference Catani's operator sums: its
    sector is not predicted (separated-exchange). With one exchange only
    (u c > u c h: the Z; u b > d t: the W) every tree joins the legs alike
    and the cross-line correlators vanish: predicted. Structural, from the
    Born DAG (``assemble._born_partitions``); generation only."""

    from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest
    from pyamplicol.models.loading import load_compiled_model

    compiled = load_compiled_model(loop_sm_compiled_model)
    fixed = {"WT": "0", "WZ": "0", "WW": "0", "WH": "0"}
    expected = {
        "u d > u d h": ((0, 3), SEPARATED_EXCHANGE),
        "u c > u c h": ((0, 3), None),
        "u b > d t": ((0, 2), None),
    }
    for process, ((qcd, qed), reason) in expected.items():
        request = SliceRequest(
            process,
            {"QCD": qcd, "QED": qed},
            {"QCD": qcd + 2, "QED": qed},
            fixed_parameters=fixed,
        )
        slice_ = LoopSlice.generate(compiled, request, loop_native)
        sector = _orders(QCD=2 * qcd + 2, QED=2 * qed)
        assert [found for found, _pairs in slice_.sectors] == [sector], process
        unpredicted = {sector: reason} if reason else {}
        assert slice_.unpredicted_sectors() == unpredicted, process


@pytest.fixture(scope="module")
def mixed_uudd(loop_sm_compiled_model, loop_native):
    """u u~ > d d~ with QCD corrections to its gluon and photon Borns on
    loop_sm (Z, W and H vetoed; ``mixed-uudd-qcd-control``)."""

    from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest
    from pyamplicol.models.loading import load_compiled_model

    request = SliceRequest(
        "u u~ > d d~",
        {"QCD": 2, "QED": 2},
        ({"QCD": 4, "QED": 0}, {"QCD": 2, "QED": 2}),
        fixed_parameters={"MB": "0"},
        particle_veto=(23, 24, 25, 250, 251),
    )
    compiled = load_compiled_model(loop_sm_compiled_model)
    return LoopSlice.generate(compiled, request, loop_native)


@pytest.mark.loop_physics
def test_mixed_sectors_of_a_qcd_request_have_no_prediction(mixed_uudd, monkeypatch):
    """The pure QCD sector's poles equal Catani's; the mixed sectors have
    none. The operator's sum (what ``_ir_poles`` returned before) is twice
    the virtual's single pole in QCD^4 QED^2 (the gluon Born's QED poles
    the left-out gluon-photon boxes carry); in QCD^2 QED^4 the boxes' poles
    multiply the colour-forbidden gluon-photon Born interference and the sum
    happens to hold, which the structural rule does not claim."""

    from pyamplicol.loops.v2 import assemble

    point = [
        ["500", "0", "0", "500"],
        ["500", "0", "0", "-500"],
        ["500", "240", "320", "300"],
        ["500", "-240", "-320", "-300"],
    ]
    parameters = {"aS": "0.118", "MT": "173"}
    result = mixed_uudd.evaluate(point, "1000000", parameters=parameters)
    qcd = _orders(QCD=6)
    assert set(result.ir_poles) == {qcd}
    virtual = result.components["virtual"]
    single, double = (float(x) for x in result.ir_poles[qcd])
    scale = max(abs(single), abs(double))
    assert abs(float(virtual[qcd].single_pole) - single) <= TOLERANCE * scale
    assert abs(float(virtual[qcd].double_pole) - double) <= TOLERANCE * scale
    mixed = [sector for sector in virtual if sector != qcd]
    assert mixed
    monkeypatch.setattr(
        assemble._PointEvaluation,
        "_predicted_sectors",
        lambda self: frozenset(sector for sector, _pairs in self.slice.sectors),
    )
    unfiltered = mixed_uudd.evaluate(point, "1000000", parameters=parameters)
    sector = _orders(QCD=4, QED=2)
    predicted = float(unfiltered.ir_poles[sector][0])
    poles = float(virtual[sector].single_pole)
    assert abs(predicted - 2 * poles) <= 1e-9 * abs(poles), (predicted, poles)


@pytest.mark.loop_physics
def test_an_absent_lower_tree_is_decided_by_its_tree_program(
    loop_sm_compiled_model, loop_native
):
    """e+ e- > u u~ a with QCD corrections (Born e^3, loop g^2 e^3): a QED
    exchange of the loop would leave a tree of order g^2 e, for which every
    leg has a vertex of the request (e+ e- a, u u~ g, u u~ a), but no such
    tree exists (the lepton pair needs e^2): the tree program of that order
    decides, so the sector is predicted and its poles are the virtual's."""

    import json

    from pyamplicol.loops.v2 import assemble
    from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest
    from pyamplicol.models.loading import load_compiled_model

    compiled = load_compiled_model(loop_sm_compiled_model)
    request = SliceRequest(
        "e+ e- > u u~ a",
        {"QCD": 0, "QED": 3},
        {"QCD": 2, "QED": 3},
        fixed_parameters={"MB": "0", "WZ": "0", "WW": "0", "WT": "0", "WH": "0"},
    )
    slice_ = LoopSlice.generate(compiled, request, loop_native)
    sector = _orders(QCD=2, QED=6)
    assert assemble._predicted_sectors(slice_) == {sector}
    fixture = ROOT / "tests/fixtures/loop/legacy/uux_ddxg.json"
    point = json.loads(fixture.read_text())["points"][0]
    result = slice_.evaluate(point, "1000000", parameters={"aS": "0.118"})
    virtual = result.components["virtual"][sector]
    single, double = (float(x) for x in result.ir_poles[sector])
    scale = max(abs(single), abs(double))
    assert scale > 0
    assert abs(float(virtual.single_pole) - single) <= TOLERANCE * scale
    assert abs(float(virtual.double_pole) - double) <= TOLERANCE * scale


@pytest.mark.loop_physics
def test_vetoed_light_quark_leaves_beta0(loop_sm_compiled_model, loop_native):
    """g g > g g with the (massless) b vetoed: its loops and counterterms
    drop the b, so gamma_g takes four flavours (genericity audit m3; the
    model has five massless triplets) and the prediction equals the
    virtual's poles; five flavours would miss them by 2/3 T_R (alpha_s/2pi)
    per gluon leg times the Born."""

    from pyamplicol.loops.v2 import assemble
    from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest
    from pyamplicol.models.loading import load_compiled_model

    compiled = load_compiled_model(loop_sm_compiled_model)
    request = SliceRequest(
        "g g > g g",
        {"QCD": 2, "QED": 0},
        {"QCD": 4, "QED": 0},
        fixed_parameters={"MB": "0"},
        particle_veto=(5,),
    )
    slice_ = LoopSlice.generate(compiled, request, loop_native)
    point = [
        ["500", "0", "0", "500"],
        ["500", "0", "0", "-500"],
        ["500", "240", "320", "300"],
        ["500", "-240", "-320", "-300"],
    ]
    result = slice_.evaluate(point, "1000000", parameters={"aS": "0.118"})
    values = assemble._kernel_parameters(slice_.compiled, {"MB": 0})
    beta0 = assemble._strong_beta0(slice_, "QCD", values)
    assert beta0 == Fraction(11, 2) - Fraction(4, 3)
    ((sector, (single, double)),) = result.ir_poles.items()
    virtual = result.components["virtual"][sector]
    scale = max(abs(float(single)), abs(float(double)))
    assert abs(float(virtual.single_pole) - float(single)) <= TOLERANCE * scale
    assert abs(float(virtual.double_pole) - float(double)) <= TOLERANCE * scale
    # The five-flavour gamma_g would shift the single pole by 4 legs x
    # (1/3) x (alpha_s/2pi) x Born (sum_k T_j.T_k = -C_A per gluon leg).
    born = float(sum(result.born.values()))
    shift = 4 * (1 / 3) * 0.118 / (2 * 3.141592653589793) * born
    assert shift > 1e3 * TOLERANCE * scale


@pytest.mark.loop_nightly
def test_joint_request_predicts_its_cross_order_sector(
    loop_qcd_qed_sm_ufo, loop_native, monkeypatch
):
    """u u~ > d d~ with gluon and photon/Z Borns, QCD and QED corrections
    (full SM loops on loop_qcd_qed_sm, 't Hooft-Feynman gauge, widths zero),
    loop orders g^4 and g^2 e^2: QCD^6 and the mixed QCD^4 QED^2 are
    predicted and equal the virtual's poles. The mixed sector takes both
    operators with their cross-order correlators (the QCD operator on the
    gluon x photon/Z Born interference, colour-forbidden as a Born but not
    under T_i.T_j: 99.9 % of the single pole; the QED operator on the gluon
    Born squared: 0.12 %), each far above the tolerance (the agreement is
    3e-15 at this point). QCD^2 QED^4 has no prediction: the
    request has no e^4 loop for the QED operator's gluon x photon/Z
    interference to pair with. Nightly: generation about 2 min."""

    from pyamplicol.loops.v2 import assemble
    from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest
    from pyamplicol.models.loading import compile_model_source

    compiled = compile_model_source(loop_qcd_qed_sm_ufo, require_supported=False)
    request = SliceRequest(
        "u u~ > d d~",
        {"QCD": 2, "QED": 2},
        ({"QCD": 4, "QED": 0}, {"QCD": 2, "QED": 2}),
        corrections=("QCD", "QED"),
        fixed_parameters={"WT": "0", "WW": "0", "WZ": "0", "WH": "0"},
        compatibility="gmu_particle_anti",
        scale_parameter="MU_R",
    )
    slice_ = LoopSlice.generate(compiled, request, loop_native)
    shares = {}
    for name in ("_qcd_poles", "_qed_poles"):
        operator = getattr(assemble._PointEvaluation, name)

        def spy(self, operator=operator, name=name):
            shares[name] = found = operator(self)
            return found

        monkeypatch.setattr(assemble._PointEvaluation, name, spy)
    point = [
        ["500", "0", "0", "500"],
        ["500", "0", "0", "-500"],
        ["500", "240", "320", "300"],
        ["500", "-240", "-320", "-300"],
    ]
    result = slice_.evaluate(point, "1000000", parameters={"aS": "0.118", "MT": "173"})
    qcd, mixed = _orders(QCD=6), _orders(QCD=4, QED=2)
    assert set(result.ir_poles) == {qcd, mixed}
    assert _orders(QCD=2, QED=4) in result.components["virtual"]
    for sector in (qcd, mixed):
        virtual = result.components["virtual"][sector]
        single, double = (float(x) for x in result.ir_poles[sector])
        scale = max(abs(single), abs(double))
        assert abs(float(virtual.single_pole) - single) <= TOLERANCE * scale, sector
        assert abs(float(virtual.double_pole) - double) <= TOLERANCE * scale, sector
    single = abs(float(result.ir_poles[mixed][0]))
    for name in ("_qcd_poles", "_qed_poles"):
        assert abs(float(shares[name][mixed][0])) > 1e-4 * single, name
