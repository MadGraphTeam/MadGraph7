# SPDX-License-Identifier: 0BSD
"""Fail-closed guards of electroweak v2 requests (WP3.2 EWI-0b;
``loops.v2.ew_policy``), the user's decisions [U8] and [U9] of 2026-10-25.

* The width policy: an electroweak request whose Born or loop-side trees
  propagate a particle with a width that is not zero, coloured or not, is
  refused (``ElectroweakWidthError``) with the widths and the fix (fixed to
  zero, or the complex-mass scheme). On the G_mu UFO's default card
  g g > t t~ with QCD and QED corrections is refused for H, Z and t before
  its tree kernels (a width takes the full kernels) and its counterterm
  lowering.
* External photons: an electroweak request with an on-shell external photon
  takes the alpha(0) convention (WP3.2 EWI-6b, ``ew_policy.ExternalPhotons``;
  its values against MadLoop in ``test_loop_v2_external_photons.py``); it is
  refused (``ExternalPhotonError``) before any tree is built where the
  convention cannot be applied: a model without the conversion counterterm
  or whose loops couple with the alpha(0) parameter (loop_qcd_qed_sm), a
  veto of an electromagnetically perturbative species, 1/alpha(0) fixed to
  zero, or no electromagnetic correction.
* QCD-only requests are not electroweak, and no guard applies to them (their
  plans are unchanged); the current scan skips sources, auxiliary and
  amputated currents and asks a DAG that scans itself natively.

The joint request that does generate today (g g > g g with QCD and QED
corrections on loop_qcd_qed_sm, no width in its trees) is checked with its
Catani poles in ``test_loop_v2_qed_catani.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from pyamplicol.loops.v2 import assemble, ew_policy
from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest
from pyamplicol.loops.v2.ew_policy import ElectroweakWidthError, ExternalPhotonError

GMU = {"compatibility": "gmu_particle_anti", "scale_parameter": "MU_R"}
LIGHT_B = {"MB": "0", "ymb": "0"}


@pytest.fixture(scope="module")
def gmu(loop_gmu_ufo):
    from pyamplicol.models.loading import compile_model_source

    return compile_model_source(loop_gmu_ufo, require_supported=False)


def _stop(stage: str):
    def stop(*_args, **_kwargs):
        raise AssertionError(f"the guard did not refuse before {stage}")

    return stop


@pytest.mark.loop_physics
def test_default_card_widths_are_refused_before_the_kernels(
    gmu, loop_native, monkeypatch
):
    """g g > t t~ [QCD QED] at the G_mu UFO's default widths: the Born and
    the loop-side trees propagate the top (coloured) and the Z and the Higgs
    (colourless, which the coloured-width rule alone lets through)."""

    monkeypatch.setattr(assemble, "with_ward_kernels", _stop("the tree kernels"))
    request = SliceRequest(
        "g g > t t~",
        {"QCD": 2, "QED": 0},
        ({"QCD": 4, "QED": 0}, {"QCD": 2, "QED": 2}),
        corrections=("QCD", "QED"),
        fixed_parameters=LIGHT_B,
        **GMU,
    )
    with pytest.raises(ElectroweakWidthError) as refused:
        LoopSlice.generate(gmu, request, loop_native)
    message = str(refused.value)
    assert (
        "'g g > t t~': an electroweak request (corrections ['QCD', 'QED'])" in message
    )
    assert "H (WH), Z (WZ), t (WT), t~ (WT)." in message
    assert 'fixed_parameters (WH = "0", WT = "0", WZ = "0")' in message
    assert "complex-mass scheme" in message


PHOTON_JET = ("u u~ > a g", {"QCD": 1, "QED": 1})
PHOTON_JET_LOOPS = ({"QCD": 3, "QED": 1}, {"QCD": 1, "QED": 3})
ZERO_WIDTHS = {**LIGHT_B, "WT": "0", "WW": "0", "WZ": "0", "WH": "0"}


@pytest.fixture(scope="module")
def qcd_qed(loop_qcd_qed_sm_ufo):
    from pyamplicol.models.loading import compile_model_source

    return compile_model_source(loop_qcd_qed_sm_ufo, require_supported=False)


@pytest.fixture(scope="module")
def gmu_a0(loop_gmu_ufo, tmp_path_factory):
    """The G_mu UFO under its restriction a0 (1/alpha(0) = aEW0M1 an
    adjustable input; MG5_aMC's loop_qcd_qed_sm_Gmu-a0)."""

    from pyamplicol.models.loading import compile_model_source

    return compile_model_source(
        loop_gmu_ufo,
        restriction="a0",
        require_supported=False,
        cache_dir=tmp_path_factory.mktemp("gmu-a0"),
    )


@pytest.mark.loop_physics
@pytest.mark.parametrize(
    ("model", "options", "reason"),
    [
        ("qcd_qed", {}, "declares no conversion counterterm eCoup_DGmua0_UV_EW"),
        ("gmu", {}, r"no adjustable 1/alpha\(0\) at SMINPUTS \[1\] \(aEW0M1, aEW0"),
        # The Z's ghost goes with it (``ew_policy.gauge_closed_veto``).
        ("gmu_a0", {"particle_veto": (23,)}, r"removes the species \[23, 9000002\]"),
        (
            "gmu_a0",
            {"fixed_parameters": {**ZERO_WIDTHS, "aEW0M1": "0"}},
            r"aEW0M1 \(1/alpha\(0\)\) is fixed to zero",
        ),
    ],
    ids=["no-conversion", "default-card", "veto", "alpha0-zero"],
)
def test_external_photons_without_the_convention_are_refused_before_the_trees(
    request, loop_native, monkeypatch, model, options, reason
):
    """u u~ > a g [QCD QED] (widths fixed to zero): an on-shell photon takes
    alpha(0) (EWI-6b); where the convention cannot be applied the request is
    refused, naming it, before any tree is built. A veto of a loop species
    on the G_mu UFO, whose electroweak counterterms declare no loop content,
    is refused first, in the model preparation (``refusals``
    "veto-unlabelled-counterterms", WP3.2 EWI-5); the convention's own
    refusal of the veto is checked with that guard disabled."""

    from pyamplicol.loops.v2.refusals import LoopRequestUnsupported

    monkeypatch.setattr(assemble, "build_tree_blocks", _stop("the trees"))
    process, born = PHOTON_JET
    options = {"fixed_parameters": ZERO_WIDTHS, **options}
    request_ = SliceRequest(
        process, born, PHOTON_JET_LOOPS, corrections=("QCD", "QED"), **options, **GMU
    )
    if "particle_veto" in options:
        with pytest.raises(LoopRequestUnsupported) as vetoed:
            LoopSlice.generate(request.getfixturevalue(model), request_, loop_native)
        assert vetoed.value.key == "veto-unlabelled-counterterms"
        assert "removes the species [23, 9000002]" in str(vetoed.value)
        monkeypatch.setattr(
            assemble, "require_labelled_counterterms_for_veto", lambda *args: None
        )
    with pytest.raises(ExternalPhotonError, match=reason) as refused:
        LoopSlice.generate(request.getfixturevalue(model), request_, loop_native)
    message = str(refused.value)
    assert refused.value.key == "external-photon-scheme"
    assert "'u u~ > a g': tagged on-shell external photons (legs [3]" in message
    assert "alpha(0)" in message
    # The remedy for the model at hand (m-sm/PLAN.md §1.6): untagged
    # photons, or a restriction that sets 1/alpha(0) where one fixes it.
    assert 'leave them untagged (loop.external_photons = "none"' in message
    if model == "gmu":
        assert 'model.restriction = "a0"' in message


@pytest.mark.loop_physics
@pytest.mark.parametrize("model", ["qcd_qed", "gmu"])
def test_untagged_photons_need_no_alpha0_input(
    request, loop_native, monkeypatch, model
):
    """MS-PHOTON (m-sm/PLAN.md Q1 (a)): with its photon untagged
    (``external_photons = "none"``) u u~ > a g [QCD QED] needs neither the
    conversion counterterm nor 1/alpha(0): it reaches its trees on
    loop_qcd_qed_sm (alpha(M_Z), no alpha(0) input, as in MG5_aMC) and on the
    G_mu UFO's default card, where the default ("final") tag is refused."""

    monkeypatch.setattr(assemble, "build_tree_blocks", _stop("the trees"))
    compiled = request.getfixturevalue(model)
    process, born = PHOTON_JET
    options = {**GMU, "fixed_parameters": ZERO_WIDTHS}
    untagged = SliceRequest(
        process,
        born,
        PHOTON_JET_LOOPS,
        corrections=("QCD", "QED"),
        external_photons="none",
        **options,
    )
    with pytest.raises(AssertionError, match="before the trees"):
        LoopSlice.generate(compiled, untagged, loop_native)
    tagged = SliceRequest(
        process, born, PHOTON_JET_LOOPS, corrections=("QCD", "QED"), **options
    )
    with pytest.raises(ExternalPhotonError) as refused:
        LoopSlice.generate(compiled, tagged, loop_native)
    assert refused.value.key == "external-photon-scheme"


@pytest.mark.loop_physics
def test_tagged_legs_must_be_on_shell_photons(gmu_a0, loop_native, monkeypatch):
    """A leg list naming a leg that is no on-shell external photon is refused
    (``external-photon-legs``) before the trees, for an electroweak and a
    QCD-only request alike; a list of photon legs passes (the QCD-only
    request converts nothing: no electromagnetic correction, as MG5_aMC)."""

    monkeypatch.setattr(assemble, "build_tree_blocks", _stop("the trees"))
    process, born = PHOTON_JET
    for corrections, loops in (
        (("QCD", "QED"), PHOTON_JET_LOOPS),
        (("QCD",), PHOTON_JET_LOOPS[:1]),
    ):
        for legs, outcome in (((4,), ExternalPhotonError), ((3,), AssertionError)):
            request_ = SliceRequest(
                process,
                born,
                loops,
                corrections=corrections,
                fixed_parameters=ZERO_WIDTHS,
                external_photons=legs,
                **GMU,
            )
            with pytest.raises(outcome) as refused:
                LoopSlice.generate(gmu_a0, request_, loop_native)
            if outcome is ExternalPhotonError:
                assert refused.value.key == "external-photon-legs"
                assert "4 (G, final)" in str(refused.value)


@pytest.mark.loop_physics
def test_alpha0_parameters_can_be_named(qcd_qed, gmu_a0, gmu, loop_native, monkeypatch):
    """Genericity audit 2 G2-m5: the tagged photons' conversion counterterm
    and 1/alpha(0) are MG5_aMC's names by default and can be named
    (``SliceRequest.alpha0_conversion`` / ``alpha0_inverse``, ``[loop]
    alpha0_conversion_parameter`` / ``alpha0_inverse_parameter``): naming
    the default ones changes nothing; a conversion the model lacks or a
    1/alpha(0) that is no adjustable parameter is refused naming it; a named
    1/alpha(0) whose inverse is the loops' alpha (loop_qcd_qed_sm's aEWM1,
    declared alpha(0) here) declares loops that couple with alpha(0), so the
    tagged photon takes neither a rescaling nor a conversion and needs no
    conversion counterterm; one the loops' alpha only depends on (G_mu's MZ)
    is refused."""

    monkeypatch.setattr(assemble, "build_tree_blocks", _stop("the trees"))
    from pyamplicol.loops.v2.assemble import prepare_request_model
    from pyamplicol.processes.model import build_model_process_ir

    process, born = PHOTON_JET

    def scheme(compiled, **options):
        request_ = SliceRequest(
            process,
            born,
            PHOTON_JET_LOOPS,
            corrections=("QCD", "QED"),
            fixed_parameters=ZERO_WIDTHS,
            **options,
            **GMU,
        )
        prepared = prepare_request_model(compiled, request_)
        species = prepared.prepared.structure.perturbative_species_orders
        ir = build_model_process_ir(process, compiled.ir, color_accuracy="full")
        return ew_policy.external_photon_scheme(
            compiled, request_, ir, ZERO_WIDTHS, species
        )

    default = scheme(gmu_a0)
    assert default == ew_policy.ExternalPhotons(
        (3,), "aEW0M1", "eCoup_DGmua0_UV_EW", "QED"
    )
    named = scheme(
        gmu_a0, alpha0_conversion="eCoup_DGmua0_UV_EW", alpha0_inverse="aEW0M1"
    )
    assert named == default
    for options, reason in (
        (
            {"alpha0_conversion": "dZe_conv"},
            "declares no conversion counterterm dZe_conv",
        ),
        ({"alpha0_inverse": "nope"}, "names 'nope', which is no adjustable"),
    ):
        with pytest.raises(ExternalPhotonError, match=reason) as refused:
            scheme(gmu_a0, **options)
        assert refused.value.key == "external-photon-scheme"
    with pytest.raises(ExternalPhotonError, match="leaves MZ free"):
        scheme(gmu, alpha0_inverse="MZ")
    loops_alpha0 = scheme(qcd_qed, alpha0_inverse="aEWM1")
    assert loops_alpha0 == ew_policy.ExternalPhotons((3,), "aEWM1", None, "QED")
    request_ = SliceRequest(
        process,
        born,
        PHOTON_JET_LOOPS,
        corrections=("QCD", "QED"),
        fixed_parameters=ZERO_WIDTHS,
        alpha0_inverse="aEWM1",
        **GMU,
    )
    with pytest.raises(AssertionError, match="before the trees"):
        LoopSlice.generate(qcd_qed, request_, loop_native)


@pytest.mark.loop_physics
def test_stored_external_photon_requests_reach_their_trees(
    gmu_a0, loop_native, monkeypatch
):
    """u u~ > a g [QCD QED] on the G_mu UFO under restriction a0: a stored
    (plan v4) process's native point inputs take the alpha(0) convention
    (WP3.2 EWI-N: the photons' sources scaled by sqrt(alpha(0)/alpha), the
    conversion a field insertion), so its generation is no longer refused
    before the trees (the former listed key ``external-photon-alpha0``):
    it reaches them, as the slice of the Python inputs does. The native
    values are checked in ``test_loop_v2_ew_native.py``."""

    from pyamplicol.loops.v2.artifact import generate_process
    from pyamplicol.loops.v2.native_trees import NATIVE_EXPORT_REFUSALS

    assert "external-photon-alpha0" not in {r.key for r in NATIVE_EXPORT_REFUSALS}
    monkeypatch.setattr(assemble, "build_tree_blocks", _stop("the trees"))
    process, born = PHOTON_JET
    request = SliceRequest(
        process,
        born,
        PHOTON_JET_LOOPS,
        corrections=("QCD", "QED"),
        fixed_parameters=ZERO_WIDTHS,
        **GMU,
    )
    with pytest.raises(AssertionError, match="before the trees"):
        generate_process(gmu_a0, request, loop_native)
    with pytest.raises(AssertionError, match="before the trees"):
        LoopSlice.generate(gmu_a0, request, loop_native)


@pytest.mark.loop_physics
def test_external_photons_take_the_alpha0_convention(gmu_a0):
    """The G_mu UFO under its restriction a0 declares the convention:
    1/alpha(0) at SMINPUTS 1 (aEW0M1, in no coupling) and the conversion
    counterterm; a QCD-only request keeps the loops' coupling for its photon
    (it is no electroweak request)."""

    gmu = gmu_a0
    from pyamplicol.loops.v2.assemble import prepare_request_model
    from pyamplicol.processes.model import build_model_process_ir

    process, born = PHOTON_JET
    request = SliceRequest(
        process,
        born,
        PHOTON_JET_LOOPS,
        corrections=("QCD", "QED"),
        fixed_parameters=ZERO_WIDTHS,
        **GMU,
    )
    prepared = prepare_request_model(gmu, request)
    species = prepared.prepared.structure.perturbative_species_orders
    ir = build_model_process_ir(process, gmu.ir, color_accuracy="full")
    scheme = ew_policy.external_photon_scheme(gmu, request, ir, ZERO_WIDTHS, species)
    assert scheme == ew_policy.ExternalPhotons(
        (3,), "aEW0M1", "eCoup_DGmua0_UV_EW", "QED"
    )
    qcd = SliceRequest(process, born, {"QCD": 3, "QED": 1}, fixed_parameters=LIGHT_B)
    assert not ew_policy.electroweak(gmu, qcd, species)


# ------------------------------------------------------------------ units


def _record(name, pdg, *, spin=3, color=1, charge=0.0, mass="ZERO", width="ZERO"):
    return SimpleNamespace(
        name=name,
        antiname=name,
        auxiliary_kind=None,
        pdg_code=pdg,
        spin=spin,
        color=color,
        charge=charge,
        mass=mass,
        width=width,
        ghost_number=0,
        goldstoneboson=False,
    )


MODEL = SimpleNamespace(
    ir=SimpleNamespace(
        particles=(
            _record("a", 22),
            _record("g", 21, color=8),
            _record("Z", 23, mass="MZ", width="WZ"),
            _record("W+", 24, charge=1.0, mass="MW", width="WW"),
            _record("t", 6, spin=2, color=3, charge=2 / 3, mass="MT", width="WT"),
            _record("e-", 11, spin=2, charge=-1.0),
            _record("dark", 9000, mass="MDARK"),
        ),
        parameters=(
            SimpleNamespace(name="MZ", nature="external", value="91.188"),
            SimpleNamespace(name="WZ", nature="external", value="2.4952"),
            SimpleNamespace(name="WW", nature="external", value="2.085"),
            SimpleNamespace(name="WT", nature="external", value="1.5"),
            SimpleNamespace(name="MDARK", nature="external", value="1"),
        ),
        vertex_terms=(
            SimpleNamespace(particles=("g", "g", "g"), coupling_orders=(("QCD", 1),)),
        ),
    )
)


def _current(particle, mask, auxiliary=None):
    return SimpleNamespace(
        index=SimpleNamespace(
            particle_id=particle, momentum_mask=mask, auxiliary_kind=auxiliary
        )
    )


@pytest.mark.loop_fast
def test_the_width_scan_and_the_photon():
    """Sources, auxiliary currents and currents of all legs but one
    (amputated) do not count; widths that are ZERO or fixed to zero do not
    count; coloured and colourless widths do; a DAG with its own scan
    (A4's ``NativeDag``) is asked for it. The photon is the massless neutral
    colourless vector (a massive neutral vector is not, unless its mass is
    fixed to zero)."""

    five = SimpleNamespace(legs=(1, 2, 3, 4, 5))
    dag = SimpleNamespace(
        process=five,
        sources=(0,),
        currents=(
            _current(23, 0b00100),  # source
            _current(24, 0b00110, auxiliary="contact"),
            _current(6, 0b11110),  # amputated: all legs but leg 1
            _current(23, 0b01100),
            _current(21, 0b00011),
        ),
    )
    assert ew_policy.propagated_species(dag) == ((6, True), (21, False), (23, False))
    assert ew_policy.propagating_widths((dag,), MODEL, {}) == {"Z": "WZ"}
    assert ew_policy.propagating_widths((dag,), MODEL, {"WZ": "0"}) == {}
    native = SimpleNamespace(propagated_species=lambda: ((6, False), (24, True)))
    assert ew_policy.propagated_species(native) == ((6, False), (24, True))
    assert ew_policy.propagating_widths((native, None), MODEL, {"WZ": "0"}) == {
        "t": "WT"
    }
    request = SimpleNamespace(process="u u~ > e- e+", corrections=("QED",))
    with pytest.raises(ElectroweakWidthError, match=r"Z \(WZ\)\. .*WZ = \"0\""):
        ew_policy.require_zero_widths(MODEL, request, (dag,), {})
    ew_policy.require_zero_widths(MODEL, request, (dag,), {"WZ": "0.0"})
    parameters = {record.name: record for record in MODEL.ir.parameters}
    photon, gluon, z, w, top, electron, dark = MODEL.ir.particles
    assert ew_policy.is_photon(photon, parameters, {})
    assert not any(
        ew_policy.is_photon(record, parameters, {})
        for record in (gluon, z, w, top, electron, dark)
    )
    assert ew_policy.is_photon(dark, parameters, {"MDARK": "0"})


@pytest.mark.loop_fast
def test_only_electroweak_requests_are_guarded():
    """A request is electroweak with a correction other than the strong one
    (the gluon's vertices' order), or with a non-vetoed colour-singlet
    species perturbative in a requested correction; a QCD-only request whose
    perturbative species are coloured is not (no guard applies, its plan is
    unchanged)."""

    def request(*corrections, veto=()):
        return SimpleNamespace(
            corrections=corrections, particle_veto=veto, fixed_parameters={}
        )

    qcd, joint = request("QCD"), request("QCD", "QED")
    species = {21: ("QCD",), 6: ("QCD",), 22: ("QED",), 23: ("QED",), 11: ("QED",)}
    assert not ew_policy.electroweak(MODEL, qcd, species)
    assert ew_policy.electroweak(MODEL, joint, species)
    assert ew_policy.electroweak(MODEL, request("QED"), {})
    singlet = {**species, 9000: ("QCD",)}
    assert ew_policy.electroweak(MODEL, qcd, singlet)
    vetoed = request("QCD", veto=(9000,))
    assert not ew_policy.electroweak(MODEL, vetoed, singlet)
    # The strong correction is the gluon's order, whatever its name
    # (genericity audit m1): renamed, the decisions are the same.
    renamed = SimpleNamespace(
        ir=SimpleNamespace(
            particles=MODEL.ir.particles,
            parameters=MODEL.ir.parameters,
            vertex_terms=(
                SimpleNamespace(
                    particles=("g", "g", "g"), coupling_orders=(("QS", 1),)
                ),
            ),
        )
    )
    renamed_species = {21: ("QS",), 6: ("QS",), 22: ("Qe",), 23: ("Qe",)}
    assert not ew_policy.electroweak(renamed, request("QS"), renamed_species)
    assert ew_policy.electroweak(renamed, request("QS", "Qe"), renamed_species)
    assert ew_policy.electroweak(renamed, request("QCD"), {})
