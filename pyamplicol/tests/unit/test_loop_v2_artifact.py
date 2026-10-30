# SPDX-License-Identifier: 0BSD
"""Plan v4 artifacts and the public API through the v2 engine (WP2.6).

One-loop requests generate virtual processes through the shared-recursion
engine v2 (the only one-loop engine) and publish them as plan v4 processes
(``rusticol.loop-runtime.v4``).
Checks, on the public surface only (TOML card, ``Generator``, ``Runtime``, CLI):

* the inclusive virtual of the legacy O1 controls through the public path
  equals the frozen 200-bit legacy values per squared coupling order and
  Laurent coefficient: gg -> gg, u u~ -> d d~ g and u u~ -> t t~ (an open
  massive quark line through the loop) at full colour in binary64,
  double-double and (nightly) arbitrary precision, u u~ -> d d~ and
  u u~ -> u u~ (identical-flavour signs) likewise, gg -> gg and u u~ -> t t~
  at leading
  colour and u u~ -> d d~ g at next-to-leading colour in binary64 and
  double-double, and u u~ -> u u~ g and u u~ -> g g g at full colour
  (nightly); Born values are held to 1e-12 in binary64;
* colour correlations (dipoles, Casimirs, ordered two-emission histories and
  their swaps), complex spin sources and helicity-resolved outputs through
  the public generator and runtime equal the frozen legacy observables (O1)
  at full colour (g g > g g; d d~ > g g with s s~ > g g served as an exact
  flavour member) and at leading colour, in binary64 (virtuals held to
  1e-12 unless the value's own stability estimate E exceeds it, then to
  E <= 1e-9; Borns to 1e-12) and in double-double (1e-24); the NLC, LC/NLC
  member and five-leg records nightly, the full-colour ones also in
  arbitrary precision (60 digits, 1e-40);
* at non-default model parameters (aS = 0.13, MT = 150) g g > g g (inclusive
  and the T12 colour correlation) equals the removed legacy engine's frozen
  values at the same parameters, whether they are passed at load or set
  afterwards;
* the pole selection (``PoleSelection``, TOML/CLI/API/results/artifact):
  ``poles="none"`` returns the finite parts with absent poles, also for
  correlated and resolved outputs, from the same artifact that serves
  ``"all"``; booleans and unserved selections are refused;
* stability checks (rotations by count or exact specification, the
  reduction-direction check) return one estimate per point that bounds the
  true error, and no estimate without checks;
* a fresh interpreter loads and evaluates the stored native point inputs
  (``point_inputs = "native-v2"``, WP3.1) with generation forbidden, with
  bitwise-identical binary64 values also after unrelated Symbolica symbols
  were created (no call regenerates: ``test_loop_v2_stored_process.py``;
  a process whose native inputs cannot be exported is refused at
  generation);
* the native artifact loader (``load_loop_v2_artifact``) admits each plan v4
  process, cross-checks families, buckets, legs, helicity rows and sectors
  against the declarations, and refuses tampered declarations and digests;
* ``pyamplicol inspect`` summarizes v2 loop artifacts (engine, sectors,
  families, buckets, colour degrees, plan digest); an artifact of the removed
  legacy engine is refused at load and inspection with the reason;
* the artifact holds no absolute path, user or host name (relocatable,
  requirements addendum G) and loads from a copy at another path with
  bitwise-identical binary64 values;
* the loop orders follow the Born amplitude orders, one per Born order and
  correction: mixed Born orders (u u~ > d d~ with QCD and photon Born
  amplitudes) equal MadLoop's split-order control per squared order, and in
  the SM fail closed on the chiral Z loop vertex; on MG5_aMC's
  loop_qcd_qed_sm (scale parameter at LOOP 666, compatibility profile) the
  dry run reports the model preparation's refusal, QCD+QED corrections to
  u u~ > d d~ stop in counterterm admission and g g > g g with QCD and QED
  corrections equals the frozen legacy values (nightly);
* a coloured particle with a width in a tree current (g g > t t~ at the
  default top width) is refused with the fix; with the width fixed to zero
  g g > t t~ equals the frozen legacy values at the model's default
  parameters (60 digits nightly); evaluations refuse unknown parameters, the
  scale parameter and values other than a generation-fixed one;
* requests v2 does not cover yet fail with the reason, and a configuration
  naming the removed ``loop.engine`` is refused; the slice request
  records every ``SliceRequest`` field; a generation run's staged
  validation reuses its processes; O1 loads read the stored observable
  columns (nothing regenerates);
* a six-gluon plan v4 artifact stays below 20 MB (nightly).
"""

from __future__ import annotations

import io
import json
import math
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import pytest

from pyamplicol.api import (
    LaurentCoefficients,
    VirtualStabilityChecks,
    VirtualStabilityEstimate,
)
from pyamplicol.api.errors import (
    CompatibilityError,
    EvaluationError,
    GenerationError,
)
from pyamplicol.api.virtual_results import POLE_COMPONENTS, POLE_ORIGINS, parse_rotation
from pyamplicol.config import PoleSelection
from pyamplicol.loops.v2.stability import Rotation

ROOT = Path(__file__).resolve().parents[2]
LEGACY = ROOT / "tests" / "fixtures" / "loop" / "legacy"
DOUBLE_TOLERANCE = Decimal("1e-12")
DOUBLE_DOUBLE_TOLERANCE = Decimal("1e-24")
# Binary64 O1 rule (user, 2026-10-23): a value whose own stability estimate E
# exceeds 1e-12 is held to E, which must not exceed 1e-9 (and the value must
# then agree in double-double, which every comparison here runs anyway).
ESTIMATE_CAP = 1e-9
ARBITRARY_DIGITS = 60
ARBITRARY_TOLERANCE = Decimal("1e-40")
OBSERVABLES = LEGACY / "observables"
BINARY64_CHECKS = VirtualStabilityChecks(("z:3/5:4/5",), True)
GG_PARAMETERS = {"aS": "0.118", "MT": "173"}
# Certified-scaleless families of the companion plans (poles "uv" and "ir").
SCALELESS_FAMILIES = {"gg": 4, "uux_ddxg": 1}
POINT = [[500, 0, 0, 500], [500, 0, 0, -500], [500, 300, 0, 400], [500, -300, 0, -400]]

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


def _card(
    model, output, entries, *, accuracy="full", declared=True, extra="", process=""
):
    processes = ", ".join(
        f"{{expression={json.dumps(expression)}, name={json.dumps(name)}}}"
        for name, expression in entries
    )
    return f"""
action = "generate"
[model]
source = {json.dumps(str(model))}
[process]
entries = [{processes}]
loop_coupling_orders = ["QCD"]
coupling_order_policy = "minimal"
{process}
[color]
accuracy = "{accuracy}"
contraction = "direct"
[loop]
r2_strategy = "model"
[loop.preparation]
fixed_parameters = {{MB = "0"}}
{SCHEME if declared else ""}
[generation]
output = {json.dumps(str(output))}
emit_api_bundle = false
[generation.validation]
post_build_validation = true
samples = 1
{extra}
"""


def _generate(card_text, tmp: Path, entries, correlators=None):
    from pyamplicol.api import Generator, ProcessRequest, ProcessSet
    from pyamplicol.config import resolve_config

    card = tmp / "card.toml"
    card.write_text(card_text)
    processes = ProcessSet(
        tuple(
            ProcessRequest.parse(expression, name=name) for name, expression in entries
        )
    )
    output = tmp / "artifact"
    Generator(resolve_config(card)).generate(processes, output, correlators=correlators)
    return output


def _fixture(name: str) -> dict:
    return json.loads((LEGACY / f"{name}.json").read_text())


def _parameters(fixture: dict) -> dict[str, str]:
    return {k: v for k, v in fixture["parameters"].items() if k != "MU_R"}


def _assert_legacy(result, results, tolerance, virtual_tolerance=None):
    """Born and the complete virtual of every point against legacy O1
    (``virtual_tolerance(index)``, if given, holds the virtual of a point)."""

    for index in range(len(result.values)):
        born = Decimal(results["born"][index][0][0])
        ((mine,),) = (result.born_values[index],)
        assert abs(Decimal(mine) - born) <= tolerance * abs(born), (index, mine)
        reference = [Decimal(x[0]) for x in results["virtual"][index][0]]
        ((laurent,),) = (result.values[index],)
        scale = max(map(abs, reference))
        held = tolerance if virtual_tolerance is None else virtual_tolerance(index)
        for k, (a, b) in enumerate(zip(laurent, reference, strict=True)):
            assert abs(Decimal(a) - b) <= held * scale, (index, k, a, b)


def _assert_inclusive_o1(
    artifact, name: str, process: str, carriers=("double", "double-double")
) -> None:
    """The public inclusive virtual of one full-colour legacy control in the
    given carriers: binary64 (1e-12, no checks requested), double-double
    (1e-24), 60-digit arbitrary precision (1e-40)."""

    from pyamplicol import Runtime

    fixture = _fixture(name)
    results = fixture["results"]["arbitrary-200"]
    runtime = Runtime.load(
        artifact, process=process, model_parameters=_parameters(fixture)
    )
    points, scales = fixture["points"], fixture["scales_squared"]
    if "double" in carriers:
        double = runtime.evaluate(points, mu_r2=scales)
        assert double.shape == (len(points), 1, 3) and double.stability is None
        assert all(item.precision_bits == 53 for item in double.diagnostics)
        _assert_legacy(double, results, DOUBLE_TOLERANCE)
        # One squared coupling order per partial result; the sector names its
        # pairs.
        (sector,) = double.sectors
        assert double.for_order(sector.order).values == double.values
    if "double-double" in carriers:
        value = runtime.evaluate(points, mu_r2=scales, arithmetic="double-double")
        assert {item.precision_bits for item in value.diagnostics} == {106}
        assert isinstance(value.values[0][0].finite, Decimal)
        _assert_legacy(value, results, DOUBLE_DOUBLE_TOLERANCE)
    if "arbitrary" in carriers:
        value = runtime.evaluate(
            points, mu_r2=scales, arithmetic="arbitrary", precision=ARBITRARY_DIGITS
        )
        assert isinstance(value.values[0][0].finite, Decimal)
        _assert_legacy(value, results, ARBITRARY_TOLERANCE)


# ------------------------------------------------------------------ fast checks


@pytest.mark.loop_fast
def test_stability_checks_take_counts_exact_rotations_and_text():
    none = VirtualStabilityChecks()
    assert not none.enabled and none.rotations == ()
    counted = VirtualStabilityChecks(3)
    assert len(counted.rotations) == 3 and counted.enabled
    assert len({rotation.matrix for rotation in counted.rotations}) == 3
    assert all(not rotation.is_identity for rotation in counted.rotations)
    # Default rotations are deterministic.
    assert VirtualStabilityChecks(3).rotations == counted.rotations
    exact = VirtualStabilityChecks(("z:3/5:4/5", "1,1,0:0.7"), True)
    assert exact.rotations[0] == Rotation.about("z", "3/5", "4/5")
    assert exact.rotations[1] == parse_rotation("1,1,0:0.7")
    assert parse_rotation("x:-4/5:3/5").matrix[1][1] == Fraction(-4, 5)
    assert exact.to_checks().reduction_direction
    assert VirtualStabilityChecks(reduction_direction=True).enabled
    for bad in ("z:1/2:1/2", "w:1:0", "1,0:0.3", "z:1:0"):
        with pytest.raises(ValueError):
            VirtualStabilityChecks((bad,))
    with pytest.raises(ValueError):
        VirtualStabilityChecks(-1)
    with pytest.raises(TypeError):
        VirtualStabilityChecks(True)


@pytest.mark.loop_fast
def test_finite_only_results_have_absent_poles_and_estimates_are_typed():
    from pyamplicol.api.virtual_results import (
        VirtualEvaluation,
        VirtualLaneDiagnostic,
        VirtualPoleParts,
    )
    from pyamplicol.config import R2Strategy
    from pyamplicol.loops.orders import BornLoopPair, VirtualOrderSector
    from pyamplicol.loops.strategy import R2Selection

    sector = VirtualOrderSector(
        (("QCD", 6),), (BornLoopPair((("QCD", 2),), (("QCD", 4),)),)
    )
    lane = VirtualLaneDiagnostic("success", "double", 53, False)
    r2 = R2Selection(R2Strategy.MODEL, R2Strategy.MODEL, "test")
    estimate = VirtualStabilityEstimate(
        1e-13, 2e-13, None, None, (("rotation-0", 1e-14),)
    )
    result = VirtualEvaluation(
        ((1.0,),),
        (((-2.0, None, None),),),
        ((("QCD", 4),),),
        (sector,),
        (lane,),
        r2,
        poles="none",
        stability=(estimate,),
    )
    assert result.poles is PoleSelection.NONE
    assert result.values[0][0] == LaurentCoefficients(-2.0, None, None)
    assert result.finite == ((-2.0,),)
    with pytest.raises(ValueError, match="no poles"):
        result.single_pole  # noqa: B018
    assert result.sum_orders() == (LaurentCoefficients(-2.0, None, None),)
    assert result.for_order((("QCD", 6),)).stability == (estimate,)
    with pytest.raises(ValueError, match="finite-only"):
        VirtualEvaluation(
            ((1.0,),),
            (((-2.0, 1.0, None),),),
            ((("QCD", 4),),),
            (sector,),
            (lane,),
            r2,
            poles=PoleSelection.NONE,
        )
    # The selection is a PoleSelection, never a boolean.
    for bad, error in ((False, TypeError), ("both", ValueError)):
        with pytest.raises(error, match=r"pole selection|PoleSelection"):
            VirtualEvaluation(
                ((1.0,),),
                (((-2.0, 1.0, 3.0),),),
                ((("QCD", 4),),),
                (sector,),
                (lane,),
                r2,
                poles=bad,
            )
    # Pole parts exist exactly under "uv" and "ir", with every component.
    parts = VirtualPoleParts(
        *({name: ((-2.0, 0.5, 0.0),) for name in POLE_COMPONENTS} for _ in "ui")
    )
    split = VirtualEvaluation(
        ((1.0,),),
        (((-2.0, 0.5, 0.0),),),
        ((("QCD", 4),),),
        (sector,),
        (lane,),
        r2,
        poles="uv",
        pole_parts=(parts,),
    )
    assert split.poles is PoleSelection.UV
    assert split.pole_part("ir", "bare") == ((LaurentCoefficients(-2.0, 0.5, 0.0),),)
    assert split.for_order((("QCD", 6),)).pole_parts == split.pole_parts
    with pytest.raises(ValueError, match="pole component"):
        split.pole_part("uv", "counterterms")
    for selection, given, message in (
        ("ir", None, "carries its pole parts"),
        ("all", (parts,), "exist exactly under"),
    ):
        with pytest.raises(ValueError, match=message):
            VirtualEvaluation(
                ((1.0,),),
                (((-2.0, 1.0, 3.0),),),
                ((("QCD", 4),),),
                (sector,),
                (lane,),
                r2,
                poles=selection,
                pole_parts=given,
            )
    with pytest.raises(ValueError, match="pole parts name the components"):
        VirtualEvaluation(
            ((1.0,),),
            (((-2.0, 0.5, 0.0),),),
            ((("QCD", 4),),),
            (sector,),
            (lane,),
            r2,
            poles="uv",
            pole_parts=(VirtualPoleParts({"virtual": ((0.0, 0.0, 0.0),)}, {}),),
        )
    with pytest.raises(ValueError, match="pole estimates"):
        VirtualEvaluation(
            ((1.0,),),
            (((-2.0, 1.0, 3.0),),),
            ((("QCD", 4),),),
            (sector,),
            (lane,),
            r2,
            stability=(estimate,),
        )
    with pytest.raises(ValueError, match="estimate"):
        VirtualEvaluation(
            ((1.0,),),
            (((-2.0, 1.0, 3.0),),),
            ((("QCD", 4),),),
            (sector,),
            (lane,),
            r2,
            stability=(None,),
        )
    with pytest.raises(ValueError):
        VirtualStabilityEstimate(float("nan"), 0.0, None, None, (("x", 0.0),))
    with pytest.raises(ValueError):
        VirtualStabilityEstimate(0.0, 0.0, 1.0, None, (("x", 0.0),))


@pytest.mark.loop_fast
def test_resolved_pole_parts_follow_the_resolved_axes():
    """Resolved pole parts per sector, helicity and colour component: summed
    by ``total()``, kept by ``for_order``, rendered per entry."""

    from dataclasses import replace

    import prettytable

    from pyamplicol.api.virtual_results import (
        ResolvedVirtualEvaluation,
        VirtualLaneDiagnostic,
        VirtualPoleParts,
    )
    from pyamplicol.config import R2Strategy
    from pyamplicol.loops.orders import BornLoopPair, VirtualOrderSector
    from pyamplicol.loops.strategy import R2Selection
    from pyamplicol.reporting.virtual import virtual_evaluation_summary

    sector = VirtualOrderSector(
        (("QCD", 6),), (BornLoopPair((("QCD", 2),), (("QCD", 4),)),)
    )
    entries = (((-2.0, 0.25, -1.0),), ((-1.0, 0.75, -3.0),))
    parts = VirtualPoleParts(
        *({name: (entries,) for name in POLE_COMPONENTS} for _ in "ui")
    )
    result = ResolvedVirtualEvaluation(
        ((((1.0,), (2.0,)),),),
        ((entries,),),
        ((("QCD", 4),),),
        (sector,),
        (VirtualLaneDiagnostic("success", "double", 53, False),),
        R2Selection(R2Strategy.MODEL, R2Strategy.MODEL, "test"),
        ("h0", "h1"),
        ("full",),
        ("full",),
        "full",
        poles="ir",
        pole_parts=(parts,),
    )
    total = result.total()
    assert total.poles is PoleSelection.IR
    assert total.pole_part("uv", "bare") == ((LaurentCoefficients(-3.0, 1.0, -4.0),),)
    assert result.pole_part("ir")[0] == ((entries,),)[0]
    assert result.for_order((("QCD", 6),)).pole_parts == result.pole_parts
    text = virtual_evaluation_summary(result, prettytable=prettytable)
    assert (
        "infrared poles of the renormalised virtual" in text
        and "Pole parts by origin" in text
    )
    assert text.count("uv_external_fields") == 4  # 2 helicities x 2 origins
    with pytest.raises(ValueError, match="carries its pole parts"):
        replace(result, pole_parts=None)


@pytest.mark.loop_fast
def test_configuration_refuses_the_engine_and_selects_the_runtime_options(tmp_path):
    from pyamplicol.config import ConfigurationError, resolve_config

    card = tmp_path / "card.toml"
    for value in ("v2", "legacy"):
        card.write_text(f'action = "generate"\n[loop]\nengine = "{value}"\n')
        with pytest.raises(
            ConfigurationError,
            match=r"'loop\.engine' was removed: the legacy one-loop engine was "
            r"removed; v2 .* is the only one-loop engine",
        ):
            resolve_config(card)
    with pytest.raises(ConfigurationError, match=r"'loop\.engine' was removed"):
        resolve_config(None, action="generate", overrides=("loop.engine=legacy",))
    card.write_text(
        'action = "generate"\n[evaluation]\npoles = "none"\n'
        '[evaluation.stability_checks]\nrotations = ["z:3/5:4/5"]\n'
        "reduction_direction = true\n"
    )
    config = resolve_config(card).effective
    assert not hasattr(config.loop, "engine")
    assert config.evaluation.poles is PoleSelection.NONE
    checks = config.evaluation.stability_checks.to_checks()
    assert checks.rotations == (Rotation.about("z", "3/5", "4/5"),)
    assert checks.reduction_direction
    default = resolve_config(None, action="generate").effective
    assert default.evaluation.poles is PoleSelection.ALL
    assert default.evaluation.stability_checks.to_checks() is None
    for value in ("false", '"both"'):
        card.write_text(f'action = "generate"\n[evaluation]\npoles = {value}\n')
        with pytest.raises(ConfigurationError, match=r"evaluation\.poles"):
            resolve_config(card)
    for value in ("uv", "ir"):
        card.write_text(f'action = "generate"\n[evaluation]\npoles = "{value}"\n')
        assert resolve_config(card).effective.evaluation.poles is PoleSelection(value)
    card.write_text(
        'action = "generate"\n[evaluation.stability_checks]\n'
        'rotations = ["z:3/5:4/5"]\nrotation_count = 2\n'
    )
    with pytest.raises(ConfigurationError):
        resolve_config(card)


@pytest.mark.loop_fast
def test_slice_request_records_round_trip():
    """Every ``SliceRequest`` field is recorded and restored (a field added
    by another package needs no edit of the record); a record with an
    unknown or a missing field is refused."""

    from dataclasses import fields

    from pyamplicol.api.errors import ArtifactError
    from pyamplicol.loops.v2.artifact import request_from_record, request_record
    from pyamplicol.loops.v2.assemble import SliceRequest
    from pyamplicol.loops.v2.selection import Selection
    from pyamplicol.models.loop_color_declaration import LoopColorDeclaration

    declaration = LoopColorDeclaration.from_dict(
        json.loads(
            (ROOT / "examples/loop/source_colour/loop_sm_qcd_subset.json").read_text()
        )
    )
    request = SliceRequest(
        "g g > g g",
        {"QCD": 2, "QED": 0},
        ({"QCD": 4, "QED": 0}, {"QCD": 2, "QED": 2}),
        corrections=("QCD", "QED"),
        fixed_parameters={"MB": "0"},
        particle_veto=(23,),
        allow_anomalous_veto=True,
        accuracy="lc",
        colour_declaration=declaration,
        scale_parameter="MU_R",
        alpha_s_parameter="aS",
        compress_colour=False,
        colour_contraction="direct",
        compatibility="gmu_particle_anti",
        counterterms="sites",
        selection=Selection(((1, -1),), ((1, 2, 3, 4),)),
        external_photons=(3,),
        alpha0_conversion="dZe_conversion",
        alpha0_inverse="ainv0",
    )
    default = SliceRequest("g g > g g", {"QCD": 2}, {"QCD": 4})
    for item in fields(SliceRequest):
        if item.name not in ("process", "born_orders"):
            assert getattr(request, item.name) != getattr(default, item.name), item
    record = request_record(request)
    assert set(record) == {item.name for item in fields(SliceRequest)} | {"correlators"}
    assert json.loads(json.dumps(record)) == record
    assert request_from_record(record) == request
    single = request_record(default)
    assert single["loop_orders"] == {"QCD": 4}
    # The selection and the photon options are recorded only when set
    # (requests without them keep their records' bytes).
    assert not {
        "selection",
        "external_photons",
        "alpha0_conversion",
        "alpha0_inverse",
    } & set(single)
    assert request_from_record(single) == default
    with pytest.raises(ArtifactError, match=r"unknown.*future_option"):
        request_from_record({**record, "future_option": 1})
    with pytest.raises(ArtifactError, match=r"missing.*compress_colour"):
        request_from_record(
            {key: value for key, value in record.items() if key != "compress_colour"}
        )


@pytest.mark.loop_fast
def test_particle_veto_is_resolved_to_pdg_codes_and_refuses_the_site_oracle():
    """``generation.particle_veto`` names resolve to |PDG| codes (a name
    vetoes its antiparticle); the request keeps them sorted and unsigned;
    plan v4 refuses the site-closure counterterm oracle."""

    from dataclasses import replace
    from types import SimpleNamespace

    from pyamplicol.api.errors import ArtifactError
    from pyamplicol.generation.loop_service import particle_veto_codes
    from pyamplicol.loops.v2.artifact import generate_process
    from pyamplicol.loops.v2.assemble import SliceRequest

    particles = [
        {"name": "t", "antiname": "t~", "pdg_code": 6},
        {"name": "t~", "antiname": "t", "pdg_code": -6},
        {"name": "Z", "antiname": "Z", "pdg_code": 23},
    ]
    model = SimpleNamespace(model={"particles": particles})
    assert particle_veto_codes(model, ("t~", "Z", "t")) == (6, 23)
    with pytest.raises(GenerationError, match=r"does not have: \['b'\]"):
        particle_veto_codes(model, ("b",))
    request = SliceRequest(
        "g g > g g", {"QCD": 2}, {"QCD": 4}, particle_veto=(-6, 23, 6)
    )
    assert request.particle_veto == (6, 23)
    with pytest.raises(ArtifactError, match="tree recursion"):
        generate_process(None, replace(request, counterterms="sites"), None)


@pytest.mark.loop_physics
def test_native_export_refusals_fail_the_generation(
    loop_sm_compiled_model, loop_native, monkeypatch
):
    """A process whose native point inputs or observable columns cannot be
    exported is refused at generation (genericity audit M1; a stored plan v4
    process never regenerates at load, so there is no plan-only downgrade):
    a listed feature (``native_trees.NATIVE_EXPORT_REFUSALS``) as a
    ``GenerationError`` naming it, any other exception of the export as
    itself; an unlisted feature key is refused."""

    from pyamplicol.api.errors import GenerationError
    from pyamplicol.correlators import CorrelatorConfig
    from pyamplicol.loops.v2 import plan_export
    from pyamplicol.loops.v2.artifact import generate_process
    from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest
    from pyamplicol.loops.v2.native_observables import StoredObservables
    from pyamplicol.loops.v2.native_trees import NativeExportUnsupported
    from pyamplicol.models.loading import load_compiled_model

    with pytest.raises(ValueError, match="unknown native export refusal"):
        NativeExportUnsupported("plan-only", "a downgrade")
    compiled = load_compiled_model(loop_sm_compiled_model)
    request = SliceRequest(
        "d d~ > g g",
        {"QCD": 2, "QED": 0},
        {"QCD": 4, "QED": 0},
        fixed_parameters={"MB": "0"},
    )
    slice_ = LoopSlice.generate(compiled, request, loop_native)
    monkeypatch.setattr(LoopSlice, "generate", lambda *args, **kwargs: slice_)

    def refuse(error):
        def export(*args, **kwargs):
            raise error

        return export

    with monkeypatch.context() as patch:
        patch.setattr(plan_export, "export_plan", refuse(NotImplementedError("plain")))
        with pytest.raises(NotImplementedError, match=r"^plain$") as raised:
            generate_process(compiled, request, loop_native)
        assert not isinstance(raised.value, GenerationError)
        listed = NativeExportUnsupported("whole-rule-root", "a pole whole-rule root")
        patch.setattr(plan_export, "export_plan", refuse(listed))
        with pytest.raises(
            GenerationError, match=r"native point inputs.*whole-rule-root"
        ):
            generate_process(compiled, request, loop_native)
    # Declared observables whose columns cannot be stored: refused, never
    # stored without columns.
    columns = NativeExportUnsupported("observable-columns", "no native components")
    monkeypatch.setattr(StoredObservables, "export", refuse(columns))
    record = ROOT / "tests/fixtures/loop/legacy/observables/ddgg_ssgg.json"
    correlators = CorrelatorConfig.from_json_dict(
        {
            "color_correlations": json.loads(record.read_text())["histories"][:1],
            "spin_correlations": [],
        }
    )
    with pytest.raises(
        GenerationError, match=r"observable columns.*observable-columns"
    ):
        generate_process(compiled, request, loop_native, correlators)


@pytest.mark.loop_fast
def test_loop_orders_follow_the_born_amplitude_orders():
    """One exact loop order per Born amplitude order and correction (not the
    Born coupling-order limits raised by two, which for mixed or loose limits
    is the loop amplitude of no Born order); the scale parameter by its
    UFO/SLHA code (LOOP, 1), else the only one of block LOOP."""

    from types import SimpleNamespace

    from pyamplicol.generation.loop_service import _v2_scale_parameter, v2_loop_orders

    qcd, qed = (("QCD", 2),), (("QED", 2),)
    limits = {"QCD": 2, "QED": 0}
    assert v2_loop_orders((qcd,), ("QCD",), limits) == {"QCD": 4, "QED": 0}
    assert v2_loop_orders((qcd,), ("QCD", "QED"), limits) == (
        {"QCD": 4, "QED": 0},
        {"QCD": 2, "QED": 2},
    )
    mixed = {"QCD": 2, "QED": 2}
    assert v2_loop_orders((qcd, qed), ("QCD",), mixed) == (
        {"QCD": 4, "QED": 0},
        {"QCD": 2, "QED": 2},
    )
    assert v2_loop_orders((qcd, qed), ("QCD", "QED"), mixed) == (
        {"QCD": 4, "QED": 0},
        {"QCD": 2, "QED": 2},
        {"QCD": 0, "QED": 4},
    )
    assert v2_loop_orders((qcd,), ("QCD",), {"QCD": 4, "QED": 2}) == {
        "QCD": 4,
        "QED": 0,
    }
    for born, corrections in (((), ("QCD",)), ((qcd,), ())):
        with pytest.raises(GenerationError, match="at least one Born"):
            v2_loop_orders(born, corrections)

    def model(*records):
        parameters = tuple(
            SimpleNamespace(name=name, nature="external", lhablock=block, lhacode=code)
            for name, block, code in records
        )
        return SimpleNamespace(ir=SimpleNamespace(parameters=parameters))

    loop_sm = model(("MU_R", "LOOP", (1,)), ("aS", "SMINPUTS", (3,)))
    assert _v2_scale_parameter(loop_sm, {}) == "MU_R"
    assert _v2_scale_parameter(model(("MU_R", "LOOP", (666,))), {}) == "MU_R"
    for candidates, fixed in (
        (model(("A", "LOOP", (2,)), ("B", "LOOP", (3,))), {}),
        (loop_sm, {"MU_R": "91.188"}),
    ):
        with pytest.raises(GenerationError, match="scale parameter is not identified"):
            _v2_scale_parameter(candidates, fixed)


@contextmanager
def _capturing(found: dict) -> Iterator[None]:
    """Keep every process ``artifact.generate_process`` returns while the
    block runs (by plan digest), for tests that inspect the generated slices
    (nothing else keeps them)."""

    from pyamplicol.loops.v2 import artifact

    original = artifact.generate_process

    def generate(*args, **kwargs):
        generated = original(*args, **kwargs)
        found[generated.plan_sha256] = generated
        return generated

    artifact.generate_process = generate
    try:
        yield
    finally:
        artifact.generate_process = original


# The processes the ``full_colour`` fixture generated, by plan digest.
_FULL_COLOUR: dict = {}


# ------------------------------------------------------------------ public path


@pytest.fixture(scope="module")
def full_colour(loop_sm_compiled_model, loop_native, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("v2-artifact")
    entries = (
        ("gg", "g g > g g"),
        ("uux_ddxg", "u u~ > d d~ g"),
        ("tt", "u u~ > t t~"),
    )
    with _capturing(_FULL_COLOUR):
        return _generate(
            _card(loop_sm_compiled_model, tmp / "artifact", entries), tmp, entries
        )


@pytest.fixture(scope="module")
def quark_controls(loop_sm_compiled_model, loop_native, tmp_path_factory):
    """u u~ > d d~ and u u~ > u u~ (identical-flavour signs) through v2."""

    tmp = tmp_path_factory.mktemp("v2-quarks")
    entries = (("uudd", "u u~ > d d~"), ("uux_uux", "u u~ > u u~"))
    return _generate(
        _card(loop_sm_compiled_model, tmp / "artifact", entries), tmp, entries
    )


@pytest.mark.loop_physics
def test_public_generation_publishes_plan_v4(full_colour):
    from pyamplicol import Runtime
    from pyamplicol.artifacts import load_manifest

    manifest = load_manifest(full_colour, verify_payloads=True)
    assert list(manifest.runtime["required_runtime_capabilities"]) == [
        "rusticol.loop-runtime.v4"
    ]
    assert {record["id"] for record in manifest.processes} == {
        "gg",
        "uux_ddxg",
        "tt",
    }
    runtime = Runtime.load(full_colour, process="uux_ddxg")
    assert runtime.contribution_kind == "virtual"
    info = runtime.inspect()
    assert info["engine"] == "v2" and info["schedule"] == "backward"
    assert info["families"] == 42
    assert sum(count for _n, _r, count in info["buckets"]) == 42
    assert info["poles"] == ["all", "none", "uv", "ir"]
    assert not info["rescue_available"]
    execution = json.loads(
        (full_colour / "processes/uux_ddxg/execution.json").read_text()
    )
    assert info["scaleless_plan_sha256"] == execution["scaleless_plan_sha256"]
    # The public path takes its counterterms from the tree recursion (WP2.2):
    # no counterterm site closure, no veto unless requested (ADDENDUM J).
    assert info["request"]["counterterms"] == "recursion"
    assert info["request"]["particle_veto"] == []
    generated = [
        item.slice for plan, item in _FULL_COLOUR.items() if plan == info["plan_sha256"]
    ]
    assert generated
    for slice_ in generated:
        assert slice_.counterterm_program is not None
        assert slice_.sites.sites == () and not slice_.site_terms
        assert slice_.counterterm_program.closure.terms
    # The generation record names the request's inert CT rules (ADDENDUM J).
    (record,) = (
        r
        for r in manifest.extensions["generation"]["concrete_processes"]
        if r["id"] == "uux_ddxg"
    )
    assert record["filters"]["counterterms_inert"] == dict(
        sorted(generated[0].counterterm_program.recursion.inert.items())
    )
    # No native stage fell back (RUST-MIGRATION rule 2: the record would
    # name it).
    assert "native_fallbacks" not in record["filters"]
    physics = runtime.physics
    assert [p.pdg_id for p in physics.external_particles] == [2, -2, 1, -1, 21]


@pytest.mark.loop_physics
def test_public_particle_veto_equals_njet_five_flavours(
    loop_sm_compiled_model, loop_native, tmp_path
):
    """``[generation] particle_veto = ["t"]`` through the public path: stored
    in the request record (|PDG| codes) and the generation filters (names),
    and g g > g g is massless five-flavour QCD, equal to NJet's Nf=5 virtual
    at two of its points (binary64, 1e-12)."""

    from pyamplicol import Runtime
    from pyamplicol.artifacts import load_manifest

    entries = (("gg", "g g > g g"),)
    card = _card(loop_sm_compiled_model, tmp_path / "artifact", entries).replace(
        "emit_api_bundle = false", 'emit_api_bundle = false\nparticle_veto = ["t"]'
    )
    output = _generate(card, tmp_path, entries)
    manifest = load_manifest(output, verify_payloads=True)
    (record,) = manifest.extensions["generation"]["concrete_processes"]
    assert record["filters"]["particle_veto"] == ["t"]
    runtime = Runtime.load(output, process="gg", model_parameters={"aS": "0.118"})
    assert runtime.inspect()["request"]["particle_veto"] == [6]
    controls = json.loads(
        (ROOT / "tests/fixtures/loop/required/njet_nf5_controls.json").read_text()
    )["processes"]["gg_gg"]
    points = controls["points"][:2]
    scales = [str(int(mu) ** 2) for mu in controls["mu_r"][:2]]
    result = runtime.evaluate(points, mu_r2=scales)
    for index, row in enumerate(controls["rows"][:2]):
        ((born,),) = (result.born_values[index],)
        assert abs(float(born) - float(row["born"])) <= 1e-12 * abs(float(born))
        ((laurent,),) = (result.values[index],)
        floor = abs(float(born)) * 0.118 / (2 * math.pi)
        for mine, key in zip(
            laurent, ("finite", "single_pole", "double_pole"), strict=True
        ):
            reference = float(row[key])
            assert abs(float(mine) - reference) <= 1e-12 * max(abs(reference), floor), (
                index,
                key,
            )


@pytest.mark.loop_physics
def test_public_anomalous_veto_is_refused_unless_overridden_and_recorded(
    loop_sm_compiled_model, loop_native, tmp_path
):
    """[U2] through the public path: ``particle_veto = ["t"]`` keeps the
    axial g g Z R2 of u u~ > e- e+ g from cancelling between the up- and
    down-type quark loops (two flavours against three) while it cancels
    with every flavour, so generation is refused by default
    (``AnomalousVetoError``, naming the counterterm and the override); with
    ``allow_anomalous_veto = true`` it generates, and the artifact records
    the override in its request record and in the generation filters next
    to ``particle_veto``, with the lowered rules and their cells."""

    from pyamplicol import Runtime
    from pyamplicol.artifacts import load_manifest
    from pyamplicol.loops.v2.ct_recursion import AnomalousVetoError

    entries = (("uux_emepg", "u u~ > e- e+ g"),)
    card = _card(loop_sm_compiled_model, tmp_path / "artifact", entries).replace(
        "emit_api_bundle = false", 'emit_api_bundle = false\nparticle_veto = ["t"]'
    )
    refused = tmp_path / "refused"
    refused.mkdir()
    with pytest.raises(
        GenerationError, match=r"\[G G Z\].*R2_GGZ.*allow_anomalous_veto"
    ) as caught:
        _generate(card, refused, entries)
    assert isinstance(caught.value.__cause__, AnomalousVetoError)
    allowed = card.replace(
        'particle_veto = ["t"]', 'particle_veto = ["t"]\nallow_anomalous_veto = true'
    )
    output = _generate(allowed, tmp_path, entries)
    manifest = load_manifest(output, verify_payloads=True)
    (record,) = manifest.extensions["generation"]["concrete_processes"]
    filters = record["filters"]
    assert filters["particle_veto"] == ["t"]
    assert filters["allow_anomalous_veto"] is True
    lowered = filters["counterterms_anomalous_veto"]
    assert lowered and all(cells for cells in lowered.values())
    runtime = Runtime.load(
        output, process="uux_emepg", model_parameters={"aS": "0.118"}
    )
    request = runtime.inspect()["request"]
    assert request["particle_veto"] == [6] and request["allow_anomalous_veto"] is True


@pytest.mark.loop_physics
@pytest.mark.parametrize(
    "name,process", [("gg_gg", "gg"), ("uux_ddxg", "uux_ddxg"), ("tt", "tt")]
)
def test_public_evaluation_equals_legacy_o1(full_colour, name, process):
    """u u~ > t t~ has open top lines through the loop, which the tree
    reading evaluates with the flipped mass natively too."""

    _assert_inclusive_o1(full_colour, name, process)


@pytest.mark.loop_physics
@pytest.mark.parametrize("name", ["uudd", "uux_uux"])
def test_public_quark_controls_equal_legacy_o1(quark_controls, name):
    _assert_inclusive_o1(quark_controls, name, name)


@pytest.mark.loop_nightly
@pytest.mark.parametrize(
    "artifact,name,process",
    [
        ("full_colour", "gg_gg", "gg"),
        ("full_colour", "uux_ddxg", "uux_ddxg"),
        ("full_colour", "tt", "tt"),
        ("quark_controls", "uudd", "uudd"),
        ("quark_controls", "uux_uux", "uux_uux"),
    ],
)
def test_public_controls_equal_legacy_o1_at_sixty_digits(
    request, artifact, name, process
):
    """The default-tier controls in 60-digit arbitrary precision (1e-40)."""

    _assert_inclusive_o1(
        request.getfixturevalue(artifact), name, process, carriers=("arbitrary",)
    )


@pytest.mark.loop_nightly
def test_public_five_leg_controls_equal_legacy_o1(
    loop_sm_compiled_model, loop_native, tmp_path
):
    """u u~ > u u~ g and u u~ > g g g at full colour: binary64 with one
    rotation and the reduction check (virtuals held to max(1e-12, E), E <=
    1e-9; Borns to 1e-12; the first u u~ > g g g point is ill-conditioned in
    binary64) and double-double (1e-24)."""

    from pyamplicol import Runtime

    entries = (("uux_uuxg", "u u~ > u u~ g"), ("uux_ggg", "u u~ > g g g"))
    output = _generate(
        _card(loop_sm_compiled_model, tmp_path / "artifact", entries),
        tmp_path,
        entries,
    )
    for name, _expression in entries:
        fixture = _fixture(name)
        results = fixture["results"]["arbitrary-200"]
        runtime = Runtime.load(
            output, process=name, model_parameters=_parameters(fixture)
        )
        points, scales = fixture["points"], fixture["scales_squared"]
        double = runtime.evaluate(
            points, mu_r2=scales, stability_checks=BINARY64_CHECKS
        )
        _assert_legacy(
            double,
            results,
            DOUBLE_TOLERANCE,
            lambda index, value=double: _tolerance(value, index, "double"),
        )
        double_double = runtime.evaluate(
            points, mu_r2=scales, arithmetic="double-double"
        )
        _assert_legacy(double_double, results, DOUBLE_DOUBLE_TOLERANCE)


@pytest.mark.loop_physics
def test_finite_only_and_stability_estimates(full_colour):
    from pyamplicol import Runtime

    fixture = _fixture("uux_ddxg")
    results = fixture["results"]["arbitrary-200"]
    runtime = Runtime.load(
        full_colour, process="uux_ddxg", model_parameters=_parameters(fixture)
    )
    points, scales = fixture["points"], fixture["scales_squared"]
    full = runtime.evaluate(points, mu_r2=scales)
    assert full.poles is PoleSelection.ALL
    # ``clear`` rebuilds the native executors from the held plan JSON.
    runtime.clear()
    assert runtime.evaluate(points, mu_r2=scales, poles="all").values == full.values
    finite = runtime.evaluate(points, mu_r2=scales, poles="none")
    assert finite.poles is PoleSelection.NONE and finite.stability is None
    for a, b in zip(finite.values, full.values, strict=True):
        assert a[0] == LaurentCoefficients(b[0].finite, None, None)
    checked = runtime.evaluate(
        points,
        mu_r2=scales,
        stability_checks=VirtualStabilityChecks(2, reduction_direction=True),
    )
    assert checked.values == full.values
    for index, estimate in enumerate(checked.stability):
        assert dict(estimate.checks).keys() == {
            "rotation-0",
            "rotation-1",
            "reduction-direction",
        }
        reference = [Decimal(x[0]) for x in results["virtual"][index][0]]
        scale = max(map(abs, reference))
        error = max(
            abs(Decimal(a) - b)
            for a, b in zip(full.values[index][0], reference, strict=True)
        )
        assert float(error / scale) <= estimate.relative, (index, estimate)
        assert estimate.relative < 1e-9
    reduction_only = runtime.evaluate(
        points[:1],
        mu_r2=scales[:1],
        poles=PoleSelection.NONE,
        stability_checks=VirtualStabilityChecks(reduction_direction=True),
    )
    (estimate,) = reduction_only.stability
    assert estimate.single_pole is None and dict(estimate.checks).keys() == {
        "reduction-direction"
    }
    # Unknown selections and booleans are refused.
    with pytest.raises(ValueError, match="unknown pole selection 'both'"):
        runtime.evaluate(points[:1], mu_r2=scales[:1], poles="both")
    with pytest.raises(TypeError, match="PoleSelection"):
        runtime.evaluate(points[:1], mu_r2=scales[:1], poles=False)
    with pytest.raises(ValueError, match="poles 'uv' or 'ir'"):
        full.pole_part("uv")
    # The UV and IR selections (design §12.4, user decision 2026-10-23) from
    # the same artifact: the finite part bitwise that of "all"; "uv" gives
    # the bare loop's UV poles, which the counterterms' UV poles cancel, and
    # "ir" the renormalised virtual's IR poles, those of "all"; both origins'
    # parts per component. ``clear`` rebuilds the companion executor too
    # (same bits).
    split = {}
    for selection in (PoleSelection.UV, PoleSelection.IR):
        split[selection] = runtime.evaluate(points, mu_r2=scales, poles=selection)
        assert split[selection].poles is selection
        assert split[selection].pole_parts is not None
    runtime.clear()
    again = runtime.evaluate(points, mu_r2=scales, poles="uv")
    assert again.values == split[PoleSelection.UV].values
    assert again.pole_parts == split[PoleSelection.UV].pole_parts
    uv, ir = split[PoleSelection.UV], split[PoleSelection.IR]
    assert uv.pole_parts == ir.pole_parts
    for index, point in enumerate(full.values):
        (whole,) = point
        scale = max(abs(x) for x in whole)
        assert uv.values[index][0].finite == ir.values[index][0].finite == whole.finite
        assert ir.values[index] == ir.pole_part("ir")[index]
        (u,) = uv.values[index]
        (i,) = ir.values[index]
        (bare_uv,) = uv.pole_part("uv", "bare")[index]
        assert u == (whole.finite, bare_uv.single_pole, bare_uv.double_pole)
        assert u.double_pole == 0
        assert abs(i.single_pole - whole.single_pole) <= 1e-12 * scale
        assert abs(i.double_pole - whole.double_pole) <= 1e-12 * scale
        parts = uv.pole_parts[index]
        for component in POLE_COMPONENTS:
            (a,) = parts.uv[component]
            (b,) = parts.ir[component]
            assert a.finite == b.finite and a.double_pole == 0
        counterterms = parts.uv["uv"][0].single_pole + parts.uv["r2"][0].single_pole
        assert abs(u.single_pole) > 1e-3 * scale
        assert abs(u.single_pole + counterterms) <= 1e-12 * scale
        assert abs(parts.uv["virtual"][0].single_pole) <= 1e-12 * scale
    # Stability estimates exist in the split selections too.
    (estimate,) = runtime.evaluate(
        points[:1],
        mu_r2=scales[:1],
        poles="ir",
        stability_checks=VirtualStabilityChecks(reduction_direction=True),
    ).stability
    assert estimate.single_pole is not None and estimate.relative < 1e-9


@pytest.mark.loop_physics
def test_correlated_and_resolved_calls_split_the_poles(full_colour):
    """``poles="uv"``/``"ir"`` on correlated and resolved calls (design
    §12.4): finite parts bitwise those of ``"all"``; the resolved sum and the
    Born-metric correlated output carry the inclusive evaluation's pole parts
    for every component; ``for_order`` keeps them."""

    from pyamplicol import Runtime

    runtime = Runtime.load(full_colour, process="gg", model_parameters=GG_PARAMETERS)
    options = {"mu_r2": "1000000"}
    inclusive = runtime.evaluate([POINT], poles="ir", **options)
    born = runtime.evaluate_correlated([POINT], poles="uv", **options)
    resolved = runtime.evaluate_resolved([POINT], poles="ir", **options)
    full = runtime.evaluate_resolved([POINT], **options)
    assert resolved.poles is PoleSelection.IR and born.poles is PoleSelection.UV
    assert resolved.finite == full.finite
    assert resolved.pole_part("ir") == resolved.values
    total = resolved.total()
    scale = max(abs(x) for x in inclusive.values[0][0])
    for origin in ("uv", "ir"):
        for component in POLE_COMPONENTS:
            expected = inclusive.pole_part(origin, component)[0][0]
            for other in (total, born):
                got = other.pole_part(origin, component)[0][0]
                for x, y in zip(got, expected, strict=True):
                    assert abs(complex(x) - y) <= 1e-12 * scale, (origin, component)
    order = resolved.sectors[0].order
    assert resolved.for_order(order).pole_parts == resolved.pole_parts
    assert born.for_order(order).pole_parts == born.pole_parts


@pytest.mark.loop_physics
def test_v2_runtime_refuses_what_it_does_not_provide(full_colour):
    """Undeclared colour histories and spin classes, helicity selectors, direct
    R2 and a rescue policy are refused; the Born metric ("born") and resolved
    output need no declaration."""

    from pyamplicol import Runtime
    from pyamplicol.api.virtual_results import (
        VirtualArithmeticSpec,
        VirtualStabilityPolicy,
    )

    runtime = Runtime.load(full_colour, process="gg", model_parameters=GG_PARAMETERS)
    point = POINT
    assert [item.id for item in runtime.available_color_correlations()] == ["born"]
    born = runtime.evaluate_correlated([point], mu_r2="1000000")
    inclusive = runtime.evaluate([point], mu_r2="1000000")
    assert born.correlated and born.shape == inclusive.shape
    for a, b in zip(born.values[0][0], inclusive.values[0][0], strict=True):
        assert abs(complex(a) - b) <= 1e-12 * abs(b)
    resolved = runtime.evaluate_resolved([point], mu_r2="1000000")
    assert resolved.shape == (1, 1, 16, 1, 3) and resolved.stability is None
    total = resolved.total()
    for a, b in zip(total.values[0][0], inclusive.values[0][0], strict=True):
        assert abs(a - b) <= 1e-12 * abs(b)
    with pytest.raises(EvaluationError, match="spin-correlation class was not"):
        runtime.set_spin_correlation_vectors({3: [1, 0, 0, 0]})
    for call in (
        lambda: runtime.evaluate_correlated(
            [point], mu_r2="1000000", color_correlation="T12"
        ),
        lambda: runtime.evaluate([point], mu_r2="1000000", helicities=["h0"]),
        lambda: runtime.evaluate([point], mu_r2="1000000", r2_strategy="direct"),
        lambda: runtime.evaluate(
            [point],
            mu_r2="1000000",
            stability=VirtualStabilityPolicy(
                (VirtualArithmeticSpec("arbitrary", 40),), "1e-8", "1e-30"
            ),
        ),
    ):
        with pytest.raises(CompatibilityError):
            call()


# Unrelated Symbolica symbols created before a load: with 130 or 141 of them
# expand() reordered kernel terms (a refused load) and the Born components'
# summation (other binary64 bits) before the terms were put in a canonical
# order (``reference_executor.ExactKernel``).
FILLER_SYMBOLS = (0, 141)


@pytest.mark.loop_physics
def test_fresh_interpreter_loads_native_inputs_without_generation(
    full_colour, monkeypatch
):
    """A new interpreter loads the artifact's native point inputs and
    evaluates them with generation forbidden (no slice is regenerated),
    giving the same plan and bitwise-identical binary64 Born and virtual
    values whatever Symbolica symbols it created before. The correlation
    catalogue reads the stored observable columns: a load in this
    interpreter whose generator would give other plan bytes (another
    build's) never calls it (nothing regenerates,
    ``test_loop_v2_stored_process.py``)."""

    from pyamplicol import Runtime

    here = Runtime.load(full_colour, process="gg", model_parameters=GG_PARAMETERS)
    assert here.inspect()["point_inputs"] == "native-v2"
    value = here.evaluate([POINT], mu_r2="1000000")
    expected = [
        here.inspect()["plan_sha256"],
        [list(point) for point in value.born_values],
        [[list(laurent) for laurent in point] for point in value.values],
    ]
    script = "\n".join(
        (
            "import json, sys",
            "from symbolica import S",
            "fillers = [S(f'filler::f{k}') for k in range(int(sys.argv[2]))]",
            "from pyamplicol import Runtime",
            "from pyamplicol.loops.v2 import artifact, assemble",
            "def forbidden(*args, **kwargs):",
            "    raise AssertionError('a native load must not generate')",
            "artifact.generate_process = forbidden",
            "assemble.LoopSlice.generate = forbidden",
            "r = Runtime.load(sys.argv[1], process='gg', model_parameters=PARAMETERS)",
            "v = r.evaluate([POINT], mu_r2='1000000')",
            "print(json.dumps([r.inspect()['plan_sha256'], v.born_values, v.values]))",
        )
    )
    script = script.replace("PARAMETERS", repr(GG_PARAMETERS))
    script = script.replace("POINT", repr(POINT))
    for fillers in FILLER_SYMBOLS:
        process = subprocess.run(
            [sys.executable, "-c", script, str(full_colour), str(fillers)],
            capture_output=True,
            text=True,
            check=True,
        )
        fresh = json.loads(process.stdout.strip().splitlines()[-1])
        assert fresh == expected, fillers

    from pyamplicol.loops.v2 import artifact

    def drifted(*args, **kwargs):
        raise AssertionError("a load must not generate")

    monkeypatch.setattr(artifact, "generate_process", drifted)
    loaded = Runtime.load(full_colour, process="gg")
    # The stored columns serve the catalogue without the generator.
    assert [item.id for item in loaded.available_color_correlations()] == ["born"]
    resolved = loaded.evaluate_resolved([POINT], mu_r2="1000000")
    assert resolved.values


def _rewrite_payloads(source: Path, destination: Path, edits) -> Path:
    """Copy an artifact and replace JSON payloads under a re-signed manifest."""

    import shutil

    from pyamplicol.artifacts import ArtifactBuilder, load_manifest

    shutil.copytree(source, destination)
    manifest = load_manifest(destination)
    records = {record.path: record for record in manifest.payloads}
    with ArtifactBuilder(destination, mode="append") as builder:
        for path, edit in edits.items():
            value = edit(json.loads((destination / path).read_text()))
            record = records[path]
            builder.discard_payloads(path)
            builder.add_json(
                path,
                value,
                role=record.role,
                process_id=record.process_id,
                compact=True,
            )
        builder.finalize(
            kind=manifest.kind,
            producer=manifest.producer,
            model=manifest.model,
            configuration=manifest.configuration,
            processes=manifest.processes,
            runtime=manifest.runtime,
            dependencies=manifest.dependencies,
            default_process_id=manifest.default_process_id,
            extensions=manifest.extensions,
        )
    return destination


@pytest.mark.loop_physics
def test_native_loader_admits_plan_v4_and_refuses_tampering(
    full_colour, loop_native, tmp_path
):
    from pyamplicol import Runtime
    from pyamplicol.api.errors import ArtifactError
    from pyamplicol.loops.v2.artifact import admit

    for process, families in (("gg", 24), ("uux_ddxg", 42)):
        admitted = admit(loop_native, full_colour, process)
        physics = json.loads(
            (full_colour / f"processes/{process}/physics.json").read_text()
        )
        execution = json.loads(
            (full_colour / f"processes/{process}/execution.json").read_text()
        )
        assert admitted["runtime_capability"] == "rusticol.loop-runtime.v4"
        assert admitted["plan_sha256"] == execution["plan_sha256"]
        # The companion plan of the certified-scaleless families (poles uv/ir).
        assert admitted["scaleless_plan_sha256"] == execution["scaleless_plan_sha256"]
        assert admitted["scaleless_families"] == SCALELESS_FAMILIES[process]
        assert admitted["families"] == families == physics["metadata"]["families"]
        assert admitted["buckets"] == physics["metadata"]["buckets"]
        assert admitted["metadata"] == physics["metadata"]
        assert admitted["request"] == execution["request"]
        assert admitted["legs"] == len(physics["state_legs"])
        assert admitted["rows"] == len(physics["helicities"])
        assert admitted["columns"] == len(physics["metadata"]["sectors"])
    info = Runtime.load(full_colour, process="gg").inspect()
    assert info["native_admission"]["runtime_capability"] == "rusticol.loop-runtime.v4"
    assert info["plan_bytes"] > 0

    def shuffle_buckets(physics):
        buckets = physics["metadata"]["buckets"]
        buckets[0][2] += 1
        buckets[-1][2] -= 1
        return physics

    def foreign_digest(execution):
        execution["plan_sha256"] = "0" * 64
        return execution

    def native_inputs(execution):
        execution["point_inputs"] = "native"
        return execution

    def regenerated(execution):
        # The retired declaration of a process stored without native inputs
        # (its loads regenerated it; REQUIREMENTS_ADDENDUM G).
        execution["point_inputs"] = "regenerated-at-load"
        return execution

    def unresolved(physics):
        physics["metadata"]["resolved"] = False
        return physics

    def foreign_spin_leg(physics):
        physics["metadata"]["spin_correlations"] = [[5]]
        return physics

    def unserved_poles(physics):
        physics["metadata"]["poles"] = ["all", "none", "uv"]
        return physics

    def undeclared_companion(execution):
        execution["scaleless_plan_sha256"] = None
        return execution

    def foreign_companion(execution):
        execution["scaleless_plan_sha256"] = "0" * 64
        return execution

    def absent_companion_key(execution):
        # Required, also when null: an absent key is not read as null.
        del execution["scaleless_plan_sha256"]
        return execution

    def before_split(physics):
        physics["metadata"]["poles"] = ["all", "none"]
        return physics

    for name, edits, reason, loaded in (
        (
            "buckets",
            {"processes/gg/physics.json": shuffle_buckets},
            "buckets",
            "buckets",
        ),
        (
            "digest",
            {"processes/gg/execution.json": foreign_digest},
            "differs from its declared digest",
            "differs from its declared digest",
        ),
        (
            "inputs",
            {"processes/gg/execution.json": native_inputs},
            "unsupported plan v4 execution declaration",
            "unsupported plan v4 execution declaration",
        ),
        (
            "regenerated",
            {"processes/gg/execution.json": regenerated},
            "no longer supported .*regenerate the artifact",
            "no longer supported .*regenerate the artifact",
        ),
        (
            "poles",
            {"processes/gg/physics.json": unserved_poles},
            "unsupported pole selections",
            "unsupported pole selections",
        ),
        (
            "resolved",
            {"processes/gg/physics.json": unresolved},
            "serve resolved output",
            "serve resolved output",
        ),
        (
            "companion",
            {"processes/gg/execution.json": undeclared_companion},
            "stored exactly when declared",
            "stored exactly when declared",
        ),
        (
            "companion-digest",
            {"processes/gg/execution.json": foreign_companion},
            "companion plan v4 differs from its declared digest",
            "companion plan v4 differs from its declared digest",
        ),
        (
            "companion-key",
            {"processes/gg/execution.json": absent_companion_key},
            "missing field `scaleless_plan_sha256`",
            "missing field `scaleless_plan_sha256`",
        ),
        (
            # The declaration before the UV/IR split: refused (no backward
            # compatibility of internal artifact formats, AGENTS.md).
            "before-split",
            {"processes/gg/physics.json": before_split},
            "unsupported pole selections .*regenerate",
            "unsupported pole selections .*regenerate",
        ),
        (
            "spin",
            {"processes/gg/physics.json": foreign_spin_leg},
            "disagree with the declarations",
            "disagree with the declarations",
        ),
    ):
        tampered = _rewrite_payloads(full_colour, tmp_path / name, edits)
        with pytest.raises(ArtifactError, match=reason):
            admit(loop_native, tampered, "gg")
        with pytest.raises(ArtifactError, match=loaded):
            Runtime.load(tampered, process="gg")


@pytest.mark.loop_physics
def test_inspect_summarizes_v2_loop_artifacts(full_colour):
    from pyamplicol.artifacts import inspect_artifact
    from pyamplicol.cli.main import main

    inspection = inspect_artifact(full_colour)
    by_id = {process.id: process for process in inspection.processes}
    gg = by_id["gg"]
    assert gg.execution_mode == "loop_recurrence"
    assert gg.physical_helicities == 16 and gg.physical_color_components == 1
    loop = gg.loop
    assert loop is not None and loop.engine == "v2"
    assert loop.runtime_capability == "rusticol.loop-runtime.v4"
    assert loop.born_orders == ("QCD^4",)
    assert loop.sectors == (("QCD^6", (("QCD^2", "QCD^4"),)),)
    assert loop.families == 24 and sum(b[2] for b in loop.buckets) == 24
    assert loop.color_degrees is None and loop.schedule == "backward"
    assert loop.poles == ("all", "none", "uv", "ir")
    assert loop.stability_checks == ("rotation", "reduction-direction")
    assert loop.arithmetic == ("double", "double-double", "arbitrary")
    assert loop.resolved and not loop.correlations
    assert loop.color_correlations == () and loop.spin_correlations == ()
    assert loop.point_inputs == "native-v2"
    execution = json.loads((full_colour / "processes/gg/execution.json").read_text())
    assert loop.plan_sha256 == execution["plan_sha256"]
    assert by_id["uux_ddxg"].loop.families == 42

    stdout = io.StringIO()
    from contextlib import redirect_stdout

    with redirect_stdout(stdout):
        code = main(["inspect", str(full_colour), "--color", "never"])
    assert code == 0
    text = stdout.getvalue()
    assert "families / buckets" in text and "24 families in 3 (N, R) buckets" in text
    assert "rusticol.loop-runtime.v4" in text and "plan v4" in text


@pytest.mark.loop_fast
def test_legacy_loop_artifacts_are_refused_with_the_reason(saved_loop_fixture):
    """An artifact of the removed legacy engine (``rusticol.loop-runtime.v3``)
    is refused, typed, by the manifest reader, ``Runtime.load`` and
    ``inspect``: v2 is the only one-loop engine."""

    from pyamplicol import Runtime
    from pyamplicol.api.errors import CompatibilityError
    from pyamplicol.artifacts import inspect_artifact, load_manifest

    legacy = saved_loop_fixture("sdk-mixed-source")
    reason = (
        r"legacy-engine virtual processes \(rusticol\.loop-runtime\.v3\): the "
        r"legacy one-loop engine was removed; v2 .* is the only one-loop "
        r"engine; regenerate the artifact"
    )
    for load in (
        lambda: load_manifest(legacy, verify_payloads=False),
        lambda: Runtime.load(legacy),
        lambda: inspect_artifact(legacy),
    ):
        with pytest.raises(CompatibilityError, match=reason):
            load()
    # The artifact writer's own capability check (behind the manifest reader).
    from pyamplicol.generation.artifact_writer import _required_runtime_capabilities

    with pytest.raises(CompatibilityError, match=reason):
        _required_runtime_capabilities(
            {"required_runtime_capabilities": ["rusticol.loop-runtime.v3"]}
        )


@pytest.mark.loop_physics
def test_cli_generates_and_evaluates_through_v2(
    loop_sm_compiled_model, loop_native, tmp_path
):
    from pyamplicol.cli import run_cli

    entries = (("gg", "g g > g g"),)
    card = tmp_path / "card.toml"
    output = tmp_path / "artifact"
    card.write_text(_card(loop_sm_compiled_model, output, entries))
    # The removed engine option is refused with the reason.
    for value in ("v2", "legacy"):
        stderr = io.StringIO()
        code = run_cli(
            ("generate", "--card", str(card), "--loop-engine", value, "--dry-run"),
            stdout=io.StringIO(),
            stderr=stderr,
        )
        assert code != 0 and "'loop.engine' was removed" in stderr.getvalue()
    stdout, stderr = io.StringIO(), io.StringIO()
    code = run_cli(
        (
            "generate",
            "--card",
            str(card),
            "--dry-run",
            "--json",
            "--progress",
            "off",
        ),
        stdout=stdout,
        stderr=stderr,
    )
    assert code == 0, stderr.getvalue()
    plan = json.loads(stdout.getvalue())
    assert plan["unsupported_features"] == []
    coverage = plan["estimated_coverage"]
    assert coverage["engine"] == "v2"
    assert coverage["runtime_capability"] == "rusticol.loop-runtime.v4"
    assert coverage["processes"][0]["loop_orders"] == {"QCD": 4, "QED": 0}
    assert coverage["processes"][0]["scale_parameter"] == "MU_R"
    assert coverage["processes"][0]["loop_admission_checked"] is False
    assert coverage["processes"][0]["native_export_checked"] is False
    from pyamplicol.loops.v2.native_trees import NATIVE_EXPORT_REFUSALS

    assert [item.split(":")[0] for item in coverage["native_export_refusals"]] == [
        refusal.key for refusal in NATIVE_EXPORT_REFUSALS
    ]
    assert not output.exists()
    code = run_cli(
        ("generate", "--card", str(card), "--progress", "off"),
        stdout=io.StringIO(),
        stderr=stderr,
    )
    assert code == 0, stderr.getvalue()
    point = POINT
    momenta = tmp_path / "momenta.json"
    momenta.write_text(json.dumps([point, point]))
    stdout = io.StringIO()
    code = run_cli(
        (
            "evaluate",
            str(output),
            "--process",
            "gg",
            "--momenta",
            str(momenta),
            "--mu-r2",
            "1000000",
            "--json",
            "--progress",
            "off",
            "--poles",
            "none",
            "--stability-rotation",
            "z:3/5:4/5",
            "--stability-reduction-direction",
        ),
        stdout=stdout,
        stderr=stderr,
    )
    assert code == 0, stderr.getvalue()
    payload = json.loads(stdout.getvalue())
    assert payload["poles"] == "none" and payload["shape"] == [2, 1, 3]
    assert payload["values"][0][0][1:] == [None, None]
    assert [name for name, _ in payload["stability"][0]["checks"]] == [
        "rotation-0",
        "reduction-direction",
    ]
    stdout = io.StringIO()
    code = run_cli(
        (
            "evaluate",
            str(output),
            "--process",
            "gg",
            "--momenta",
            str(momenta),
            "--mu-r2",
            "1000000",
            "--json",
            "--progress",
            "off",
        ),
        stdout=stdout,
        stderr=stderr,
    )
    assert code == 0, stderr.getvalue()
    payload = json.loads(stdout.getvalue())
    assert payload["poles"] == "all" and payload["stability"] is None
    assert payload["pole_parts"] is None
    full = payload["values"]
    stdout = io.StringIO()
    code = run_cli(
        (
            "evaluate",
            str(output),
            "--process",
            "gg",
            "--momenta",
            str(momenta),
            "--mu-r2",
            "1000000",
            "--json",
            "--progress",
            "off",
            "--poles",
            "ir",
        ),
        stdout=stdout,
        stderr=stderr,
    )
    assert code == 0, stderr.getvalue()
    payload = json.loads(stdout.getvalue())
    assert payload["poles"] == "ir" and payload["shape"] == [2, 1, 3]
    assert [point[0][0] for point in payload["values"]] == [p[0][0] for p in full]
    (parts, _) = payload["pole_parts"]
    assert parts["ir"]["virtual"] == payload["values"][0]
    assert set(parts["uv"]) == set(POLE_COMPONENTS) and parts["uv"]["bare"][0][2] == 0
    stdout = io.StringIO()
    code = run_cli(
        (
            "evaluate",
            str(output),
            "--process",
            "gg",
            "--momenta",
            str(momenta),
            "--mu-r2",
            "1000000",
            "--resolved",
            "--poles",
            "none",
            "--stability-reduction-direction",
            "--color",
            "never",
            "--progress",
            "off",
        ),
        stdout=stdout,
        stderr=stderr,
    )
    assert code == 0, stderr.getvalue()
    text = stdout.getvalue()
    assert "Finite-only evaluation" in text and "Stability estimates" in text
    assert "reduction-direction" in text and "absent" in text
    stdout = io.StringIO()
    code = run_cli(
        (
            "evaluate",
            str(output),
            "--process",
            "gg",
            "--momenta",
            str(momenta),
            "--mu-r2",
            "1000000",
            "--poles",
            "uv",
            "--color",
            "never",
            "--progress",
            "off",
        ),
        stdout=stdout,
        stderr=stderr,
    )
    assert code == 0, stderr.getvalue()
    text = stdout.getvalue()
    assert "ultraviolet poles of the bare loop amplitude" in text
    assert "Pole parts by origin" in text and "uv_external_fields" in text


@pytest.mark.loop_physics
@pytest.mark.parametrize(
    "name,expression,accuracy",
    [
        ("gg_gg", "g g > g g", "lc"),
        ("tt", "u u~ > t t~", "lc"),
        ("uux_ddxg", "u u~ > d d~ g", "nlc"),
        ("uux_uuxg", "u u~ > u u~ g", "nlc"),
    ],
)
def test_projected_colour_through_v2_equals_legacy_o1(
    loop_sm_colour_model, loop_native, tmp_path, name, expression, accuracy
):
    """LC and NLC projections in binary64 (1e-12) and double-double (1e-24),
    identical-flavour lines included (u u~ > u u~ g: blocks of one colour
    word and two N_c grades, ``generation.dag_grades``)."""

    from pyamplicol import Runtime

    entries = ((name, expression),)
    output = _generate(
        _card(
            loop_sm_colour_model,
            tmp_path / "artifact",
            entries,
            accuracy=accuracy,
            declared=False,
        ),
        tmp_path,
        entries,
    )
    fixture = _fixture(name)
    reference = fixture["colour_projections"][accuracy]
    results = reference["results"]["arbitrary-200"]
    runtime = Runtime.load(output, process=name, model_parameters=_parameters(fixture))
    info = runtime.inspect()
    assert info["color_accuracy"] == accuracy
    assert list(info["color_degrees"].values()) == [reference["virtual_degree"]]
    points, scales = fixture["points"], fixture["scales_squared"]
    result = runtime.evaluate(points, mu_r2=scales)
    assert result.color_accuracy == accuracy
    _assert_legacy(result, results, DOUBLE_TOLERANCE)
    double_double = runtime.evaluate(points, mu_r2=scales, arithmetic="double-double")
    _assert_legacy(double_double, results, DOUBLE_DOUBLE_TOLERANCE)


# ------------------------------------------------------------------ observables


def _observables_record(name: str) -> dict:
    return json.loads((OBSERVABLES / f"{name}.json").read_text())


def _parts(value) -> tuple[Decimal, Decimal]:
    """(real, imaginary) of a public value or a legacy ``[re, im]`` pair."""

    if isinstance(value, list | tuple) and len(value) == 2:
        return Decimal(value[0]), Decimal(value[1])
    if isinstance(value, complex):
        return Decimal(repr(value.real)), Decimal(repr(value.imag))
    if isinstance(value, Decimal | float | int):
        return Decimal(repr(value) if isinstance(value, float) else value), Decimal(0)
    return Decimal(value.real), Decimal(value.imag)  # CorrelatedValue


def _deviation(actual, expected) -> Decimal:
    from decimal import localcontext

    with localcontext(prec=80):
        (a, b), (c, d) = _parts(actual), _parts(expected)
        return ((a - c) ** 2 + (b - d) ** 2).sqrt()


def _magnitude(value) -> Decimal:
    return _deviation(value, (0, 0))


CARRIER_TOLERANCES = {
    "double": DOUBLE_TOLERANCE,
    "double-double": DOUBLE_DOUBLE_TOLERANCE,
    "arbitrary": ARBITRARY_TOLERANCE,
}


def _estimated(estimate: float) -> Decimal:
    """A binary64 virtual's tolerance under the user's rule: 1e-12, or its
    own estimate E when E exceeds it, which must not exceed 1e-9."""

    assert estimate <= ESTIMATE_CAP, estimate
    return max(DOUBLE_TOLERANCE, Decimal(repr(estimate)))


def _tolerance(result, index: int, carrier: str) -> Decimal:
    """The O1 tolerance of one evaluated point's virtual (module docstring)."""

    if carrier != "double":
        return CARRIER_TOLERANCES[carrier]
    assert result.stability is not None
    return _estimated(result.stability[index].relative)


def _assert_record(result, expected, index: int, carrier: str, label) -> None:
    """Born per order (held to the carrier's tolerance: the estimate is the
    virtual's) and virtual per sector against one legacy record."""

    borns = expected["born"][index]
    scale = max(_magnitude(x) for x in borns)
    tolerance = CARRIER_TOLERANCES[carrier]
    for actual, value in zip(result.born_values[index], borns, strict=True):
        assert _deviation(actual, value) <= tolerance * scale, (label, index)
    sectors = expected["virtual"][index]
    tolerance = _tolerance(result, index, carrier)
    for laurent, values in zip(result.values[index], sectors, strict=True):
        scale = max(_magnitude(x) for x in values)
        for actual, value in zip(laurent, values, strict=True):
            assert _deviation(actual, value) <= tolerance * scale, (label, index)


def _correlators(record: dict):
    from pyamplicol.correlators import CorrelatorConfig

    return CorrelatorConfig.from_json_dict(
        {
            "color_correlations": record["histories"],
            "spin_correlations": record["spin_classes"],
        }
    )


@pytest.fixture(scope="module")
def observables_artifact(
    loop_sm_compiled_model, loop_sm_colour_model, loop_native, tmp_path_factory
):
    """The public v2 artifact of one legacy observables record (generated with
    its colour histories and spin classes; members by exact flavour
    grouping), built once per module."""

    built: dict[str, Path] = {}

    def build(name: str) -> Path:
        if name not in built:
            record = _observables_record(name)
            accuracy = record.get("accuracy", "full")
            entries = tuple(
                (key, expression) for key, expression in record["processes"]
            )
            tmp = tmp_path_factory.mktemp(name)
            card = _card(
                loop_sm_colour_model if accuracy != "full" else loop_sm_compiled_model,
                tmp / "artifact",
                entries,
                accuracy=accuracy,
                declared=accuracy == "full",
                process="flavor_blind = true" if len(entries) > 1 else "",
            )
            built[name] = _generate(card, tmp, entries, _correlators(record))
        return built[name]

    return build


def _spin_vectors(record: dict, legs: str) -> dict[int, tuple]:
    """The literal (real, imaginary) source vectors of one legacy spin class."""

    return {
        int(leg): tuple(
            (Decimal(re), Decimal(im)) for re, im in record["spin_vectors"][k]
        )
        for k, leg in enumerate(legs.split(","))
    }


def _assert_resolved(result, runtime, resolved: dict, carrier: str) -> None:
    """``_compare_resolved`` with sums at 80 digits (Decimal's default
    context would round 60-digit values to 28)."""

    from decimal import localcontext

    with localcontext(prec=80):
        _compare_resolved(result, runtime, resolved, carrier)


def _compare_resolved(result, runtime, resolved: dict, carrier: str) -> None:
    """Per helicity row, the Born and the virtual summed over the Born colour
    components against the legacy resolved record (whose colour anchors are
    diagram tensors: the colour axis is compared through its sum). A row is
    held relative to its own scale, a structural zero (a vanishing Born row,
    whose entries are rounding noise with large relative estimates) relative
    to the point's largest at the carrier's tolerance; in binary64 the Born
    rows at 1e-12 and a row's virtual at its entries' largest estimate under
    the user's rule (``_estimated``)."""

    rows = [tuple(h.values) for h in runtime.physics.helicities]
    legacy = [tuple(row) for row in resolved["helicities"]]
    floor = ARBITRARY_TOLERANCE
    for index in range(len(result.values)):
        born_scale = max(
            _magnitude(x) for order in resolved["born"][index] for x in order
        )
        virtual_scale = max(
            _magnitude(x)
            for sector in resolved["virtual"][index]
            for row in sector
            for x in row
        )
        for h, row in enumerate(rows):
            target = legacy.index(row)
            tolerance = CARRIER_TOLERANCES[carrier]
            born = [order[target] for order in resolved["born"][index]]
            scale = max(map(_magnitude, born))
            structural_zero = scale <= floor * born_scale
            scale = born_scale if structural_zero else scale
            for order, value in enumerate(born):
                mine = sum(
                    (_parts(x)[0] for x in result.born_values[index][order][h]),
                    Decimal(0),
                )
                assert _deviation(mine, value) <= tolerance * scale, (index, row)
            if carrier == "double" and not structural_zero:
                tolerance = _estimated(
                    max(e.relative for e in result.stability[index][h])
                )
            for sector, values in enumerate(resolved["virtual"][index]):
                targets = values[target]
                scale = max(map(_magnitude, targets))
                scale = virtual_scale if scale <= floor * virtual_scale else scale
                for k, value in enumerate(targets):
                    mine = sum(
                        (
                            _parts(laurent[k])[0]
                            for laurent in result.values[index][sector][h]
                        ),
                        Decimal(0),
                    )
                    assert _deviation(mine, value) <= tolerance * scale, (
                        index,
                        row,
                        k,
                    )
        # Every legacy row v2 does not generate is a structural zero.
        for row in set(legacy) - set(rows):
            target = legacy.index(row)
            assert all(
                _magnitude(order[target]) <= floor * born_scale
                for order in resolved["born"][index]
            )


BINARY64_AND_DD = ("double", "double-double")
ALL_CARRIERS = (*BINARY64_AND_DD, "arbitrary")


def _case_id(value) -> str:
    return value if isinstance(value, str) else "+".join(value)


@pytest.mark.loop_physics
@pytest.mark.parametrize(
    "name,carriers",
    (
        ("gg_gg", BINARY64_AND_DD),
        ("ddgg_ssgg", BINARY64_AND_DD),
        ("gg_gg_lc", BINARY64_AND_DD),
    ),
    ids=_case_id,
)
def test_public_correlated_and_resolved_calls_equal_legacy_o1(
    observables_artifact, name, carriers
):
    """O1 through the public surface: every legacy directed colour history,
    every joint complex spin source and (where recorded) the helicity-resolved
    outputs, for the representative and its exact flavour member (served by
    the representative: one artifact process), in binary64 with one rotation
    and the reduction-direction check and in double-double (the full-colour
    records in 60-digit arbitrary precision nightly)."""

    _assert_public_o1(observables_artifact, name, carriers)


NON_DEFAULT_PARAMETERS = {"aS": "0.13", "MT": "150"}


@pytest.mark.loop_physics
def test_public_v2_runtime_at_non_default_model_parameters(
    observables_artifact, loop_sm_compiled_model, loop_native, tmp_path
):
    """Every other public-path test runs at loop_sm's defaults (aS = 0.118,
    MT = 173, also the frozen fixtures' values), where a runtime that ignored
    its parameters would pass. At aS = 0.13 and MT = 150, g g > g g at full
    colour through the v2 artifact of its observables record equals the
    removed legacy engine's frozen values at the same parameters
    (double-double, ``legacy/gg_gg_non_default_parameters.json``), at the
    record's first point: the inclusive virtual and the T12 colour
    correlation, in binary64 (the user's rule) and double-double (1e-24).
    Parameters passed at load and through ``set_model_parameters`` give
    bitwise-identical binary64 values; the resolved rows (double-double) sum
    to the inclusive value; the Born scales as aS^2 against the defaults."""

    from decimal import localcontext

    from pyamplicol import Runtime
    from pyamplicol.correlators import CorrelatedRequest

    record = _observables_record("gg_gg")
    frozen = _fixture("gg_gg_non_default_parameters")
    assert frozen["model_parameters"] == NON_DEFAULT_PARAMETERS
    points, scales = record["points"][:1], record["scales_squared"][:1]
    assert (frozen["points"], frozen["scales_squared"]) == (points, scales)
    requests = {"T12": CorrelatedRequest("T12", {})}
    expected = frozen["results"]
    output = observables_artifact("gg_gg")
    runtime = Runtime.load(
        output, process="gg", model_parameters=NON_DEFAULT_PARAMETERS
    )
    later = Runtime.load(output, process="gg")
    later.set_model_parameters(NON_DEFAULT_PARAMETERS)
    for carrier in BINARY64_AND_DD:
        options = CARRIER_OPTIONS[carrier]
        results = {
            "inclusive": runtime.evaluate(points, mu_r2=scales, **options),
            **runtime.evaluate_correlated_many(
                points, requests, mu_r2=scales, **options
            ),
        }
        for label, result in results.items():
            target = {
                "born": expected[label]["born"],
                "virtual": expected[label]["virtual"],
            }
            _assert_record(result, target, 0, carrier, label)
        if carrier == "double":
            again = {
                "inclusive": later.evaluate(points, mu_r2=scales, **options),
                **later.evaluate_correlated_many(
                    points, requests, mu_r2=scales, **options
                ),
            }
            for label, result in results.items():
                assert again[label].born_values == result.born_values, label
                assert again[label].values == result.values, label
    inclusive = runtime.evaluate(points, mu_r2=scales, arithmetic="double-double")
    resolved = runtime.evaluate_resolved(
        points, mu_r2=scales, arithmetic="double-double"
    )
    default = Runtime.load(output, process="gg", model_parameters=GG_PARAMETERS)
    born = default.evaluate(points, mu_r2=scales, arithmetic="double-double")
    with localcontext(prec=80):
        ((value,),) = inclusive.born_values
        ((initial,),) = born.born_values
        ratio = (Decimal("0.13") / Decimal("0.118")) ** 2
        assert abs(value / initial - ratio) <= DOUBLE_DOUBLE_TOLERANCE * ratio
        rows = sum(
            (_parts(x)[0] for row in resolved.born_values[0][0] for x in row),
            Decimal(0),
        )
        assert abs(rows - value) <= DOUBLE_DOUBLE_TOLERANCE * value
        ((laurent,),) = inclusive.values
        scale = max(abs(Decimal(c)) for c in laurent)
        for k, coefficient in enumerate(laurent):
            summed = sum(
                (_parts(entry[k])[0] for row in resolved.values[0][0] for entry in row),
                Decimal(0),
            )
            assert abs(summed - Decimal(coefficient)) <= DOUBLE_DOUBLE_TOLERANCE * scale


@pytest.mark.loop_nightly
@pytest.mark.parametrize(
    "name,carriers",
    (
        ("gg_gg", ("arbitrary",)),
        ("ddgg_ssgg", ("arbitrary",)),
        ("gg_gg_nlc", BINARY64_AND_DD),
        ("ddgg_ssgg_lc", BINARY64_AND_DD),
        ("ddgg_ssgg_nlc", BINARY64_AND_DD),
        ("uux_ggg", ALL_CARRIERS),
        ("uux_ggg_spin", ALL_CARRIERS),
        ("uux_ggg_lc", BINARY64_AND_DD),
        ("uux_ggg_nlc", BINARY64_AND_DD),
    ),
    ids=_case_id,
)
def test_public_correlated_and_resolved_calls_equal_legacy_o1_nightly(
    observables_artifact, name, carriers
):
    """The same O1 on every other frozen legacy observables record: NLC, LC
    and NLC members, and the five-leg u u~ > g g g (directed and spin) at
    full, leading and next-to-leading colour, whose first point is
    binary64-ill-conditioned (virtuals held to their own estimate there, and
    to 1e-24 in double-double); the full-colour records also in 60 digits."""

    _assert_public_o1(observables_artifact, name, carriers)


CARRIER_OPTIONS = {
    "double": {"stability_checks": BINARY64_CHECKS},
    "double-double": {"arithmetic": "double-double"},
    "arbitrary": {"arithmetic": "arbitrary", "precision": ARBITRARY_DIGITS},
}


def _assert_public_o1(observables_artifact, name: str, carriers) -> None:
    from pyamplicol import Runtime
    from pyamplicol.artifacts import load_manifest
    from pyamplicol.correlators import CorrelatedRequest

    record = _observables_record(name)
    output = observables_artifact(name)
    manifest = load_manifest(output)
    representative = record["processes"][0][0]
    assert [process["id"] for process in manifest.processes] == [representative]
    points, scales = record["points"], record["scales_squared"]
    # Without the generation objects of this interpreter: the loads read the
    # stored observable columns and the members listed at generation.
    for process, _expression in record["processes"]:
        expected = record["results"]["arbitrary-200"][process]
        runtime = Runtime.load(
            output, process=process, model_parameters=record["parameters"]
        )
        assert runtime.physics.process_id == process
        assert runtime.representative_process_key == representative
        histories = tuple(expected.get("directed", {}))
        assert [item.id for item in runtime.available_color_correlations()] == [
            "born",
            *(history["id"] for history in record["histories"]),
        ]
        requests = {
            history: CorrelatedRequest(history, {}) for history in histories
        } | {
            f"spin:{legs}": CorrelatedRequest("born", _spin_vectors(record, legs))
            for legs in expected.get("spin", {})
        }
        for carrier in carriers:
            options = CARRIER_OPTIONS[carrier]
            values = runtime.evaluate_correlated_many(
                points, requests, mu_r2=scales, **options
            )
            for label, result in values.items():
                assert result.correlated and result.poles is PoleSelection.ALL
                assert (result.stability is not None) == (carrier == "double")
                target = (
                    expected["spin"][label.removeprefix("spin:")]
                    if label.startswith("spin:")
                    else expected["directed"][label]
                )
                for index in range(len(points)):
                    _assert_record(result, target, index, carrier, (process, label))
            if "resolved" in expected:
                resolved = runtime.evaluate_resolved(points, mu_r2=scales, **options)
                assert (resolved.stability is not None) == (carrier == "double")
                _assert_resolved(resolved, runtime, expected["resolved"], carrier)


@pytest.mark.loop_physics
def test_finite_only_correlated_and_resolved_from_the_same_artifact(
    observables_artifact,
):
    """``poles="none"`` on correlated and resolved calls of the artifact that
    computed them with poles: equal finite parts, absent poles, estimates per
    point (correlated) and per resolved entry only when checks ran."""

    from pyamplicol import Runtime
    from pyamplicol.correlators import CorrelatedRequest

    record = _observables_record("gg_gg")
    runtime = Runtime.load(
        observables_artifact("gg_gg"), model_parameters=record["parameters"]
    )
    points, scales = record["points"], record["scales_squared"]
    requests = {"T12": CorrelatedRequest("T12", {})}
    full = runtime.evaluate_correlated_many(points, requests, mu_r2=scales)["T12"]
    finite = runtime.evaluate_correlated_many(
        points, requests, mu_r2=scales, poles="none"
    )["T12"]
    assert finite.poles is PoleSelection.NONE and finite.stability is None
    for a, b in zip(finite.values, full.values, strict=True):
        assert a[0] == LaurentCoefficients(b[0].finite, None, None)
    checked = runtime.evaluate_correlated(
        points,
        color_correlation="T12",
        mu_r2=scales,
        poles="none",
        stability_checks=VirtualStabilityChecks(reduction_direction=True),
    )
    assert [dict(e.checks).keys() for e in checked.stability] == [
        {"reduction-direction"}
    ] * len(points)
    assert all(e.single_pole is None for e in checked.stability)
    resolved = runtime.evaluate_resolved(points, mu_r2=scales)
    finite = runtime.evaluate_resolved(
        points, mu_r2=scales, poles="none", stability_checks=VirtualStabilityChecks(1)
    )
    assert finite.poles is PoleSelection.NONE and resolved.stability is None
    with pytest.raises(ValueError, match="finite-only"):
        finite.single_pole  # noqa: B018
    for a, b in zip(finite.values, resolved.values, strict=True):
        for sector_a, sector_b in zip(a, b, strict=True):
            for row_a, row_b in zip(sector_a, sector_b, strict=True):
                for x, y in zip(row_a, row_b, strict=True):
                    assert x == LaurentCoefficients(y.finite, None, None)
    (point_estimates, *_rest) = finite.stability
    assert len(point_estimates) == len(finite.helicity_ids)
    assert all(
        dict(estimate.checks).keys() == {"rotation-0"}
        for row in point_estimates
        for estimate in row
    )
    total = finite.total()
    assert total.poles is PoleSelection.NONE and total.stability is None

    from pyamplicol.artifacts import inspect_artifact

    (process,) = inspect_artifact(observables_artifact("gg_gg")).processes
    loop = process.loop
    assert loop.correlations and loop.resolved
    assert loop.color_correlations == ("casimir", "T12", "T34")
    assert loop.spin_correlations == ((3,), (3, 4))


# ------------------------------------------------------------------ portability


def _strings(path: Path) -> str:
    """The text of one artifact file; a PACBIN's plan members (the process
    plan and the companion of certified-scaleless families) decompressed."""

    from pyamplicol.loops.v2.artifact import SCALELESS_PLAN_MEMBER, read_plan

    if path.name == "loop-runtime.pacbin":
        plans = (read_plan(path), read_plan(path, SCALELESS_PLAN_MEMBER))
        return "".join(p or "" for p in plans) + path.read_bytes().decode("latin-1")
    return path.read_bytes().decode("latin-1")


@pytest.mark.loop_physics
def test_artifacts_hold_no_host_paths_and_load_relocated(
    full_colour, observables_artifact, tmp_path
):
    """Requirements addendum G within WP2.6: no absolute path of the
    generating host (working directories, the model source, the output, the
    home and temporary directories, any path under a common root), no user
    or host name in any file of a plan v4 artifact (the plan member
    decompressed); a copy under another path loads without the original and
    gives bitwise-identical binary64 values."""

    import getpass
    import re
    import shutil
    import socket
    import tempfile

    from pyamplicol import Runtime

    forbidden = {
        str(Path.home()),
        str(ROOT),
        str(Path.cwd()),
        str(full_colour.parent),
        tempfile.gettempdir(),
        "pyamplicol-loop-v2-",  # the generation's staging directories
        f"/{getpass.getuser()}/",
        f"{getpass.getuser()}@",
        socket.gethostname(),
    }
    # Any absolute POSIX path under a common root, or a Windows drive path.
    absolute = re.compile(
        r"(?<![\w.])/(?:tmp|var|home|shared|nix|usr|opt|root|Users|private|mnt"
        r"|scratch|srv|data|run)/|\b[A-Za-z]:[\\/](?:Users|Windows|tmp)\b"
    )
    roots = (full_colour, observables_artifact("ddgg_ssgg"))
    for root in roots:
        forbidden.add(str(root.resolve().parent))
    for root in roots:
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            text = _strings(path)
            for item in forbidden:
                assert item not in text, (path.relative_to(root), item)
            found = absolute.search(text)
            assert found is None, (path.relative_to(root), found and found.group())
    moved = tmp_path / "elsewhere" / "copy"
    shutil.copytree(full_colour, moved)
    points = [
        POINT,
        [[500, 0, 0, 500], [500, 0, 0, -500], [500, 0, 300, 400], [500, 0, -300, -400]],
    ]
    script = "\n".join(
        (
            "import json, sys",
            "from pyamplicol import Runtime",
            "r = Runtime.load(sys.argv[1], process='gg', model_parameters=PARAMETERS)",
            "v = r.evaluate(POINTS, mu_r2='1000000')",
            "s = [r.evaluate(POINTS, mu_r2='1000000', poles=p) for p in ('uv', 'ir')]",
            "s = [[e.values, [e.pole_part(o, c) for o in ORIGINS for c in PARTS]]"
            " for e in s]",
            "d = r.inspect()['plan_sha256']",
            "print(json.dumps([d, v.born_values, v.values, s]))",
        )
    )
    script = script.replace("ORIGINS", repr(POLE_ORIGINS))
    script = script.replace("PARTS", repr(POLE_COMPONENTS))
    script = script.replace("PARAMETERS", repr(GG_PARAMETERS))
    script = script.replace("POINTS", repr(points))
    # A fresh interpreter in another working directory evaluates the copy's
    # stored native point inputs.
    process = subprocess.run(
        [sys.executable, "-c", script, str(moved)],
        capture_output=True,
        text=True,
        check=True,
        cwd=tmp_path,
    )
    digest, born, values, split = json.loads(process.stdout.strip().splitlines()[-1])
    here = Runtime.load(full_colour, process="gg", model_parameters=GG_PARAMETERS)
    original = here.evaluate(points, mu_r2="1000000")
    assert digest == here.inspect()["plan_sha256"]
    assert born == [list(point) for point in original.born_values]
    assert values == [[list(laurent) for laurent in point] for point in original.values]
    # The "uv" and "ir" selections run the stored companion plan: bitwise too.
    for selection, (got, parts) in zip(("uv", "ir"), split, strict=True):
        local = here.evaluate(points, mu_r2="1000000", poles=selection)
        expected = [
            local.values,
            [local.pole_part(o, c) for o in POLE_ORIGINS for c in POLE_COMPONENTS],
        ]
        assert [got, parts] == json.loads(json.dumps(expected)), selection


@pytest.mark.loop_physics
def test_unsupported_requests_name_the_reason(
    loop_sm_compiled_model, loop_native, tmp_path
):
    """A correction the source model does not expand perturbatively (QED in
    loop_sm; several declared corrections take one loop order each) is
    refused with its reason before any generation. The dry run of a
    default-configuration request plans v2 and enumerates no loop diagram.
    (The forced seed FFT is served: ``test_loop_v2_seed_fft``.)"""

    from pyamplicol.api import Generator, ProcessRequest, ProcessSet
    from pyamplicol.config import resolve_config

    entries = (("gg", "g g > g g"),)
    card = tmp_path / "card.toml"
    processes = ProcessSet((ProcessRequest.parse("g g > g g", name="gg"),))
    for edit, reason in (
        (
            ('loop_coupling_orders = ["QCD"]', 'loop_coupling_orders = ["QCD", "QED"]'),
            r"require declared NLO model support.*\['QED'\]",
        ),
    ):
        card.write_text(
            _card(loop_sm_compiled_model, tmp_path / "a", entries).replace(*edit)
        )
        generator = Generator(resolve_config(card))
        with pytest.raises(GenerationError, match=reason):
            generator.generate(processes, tmp_path / "a")
        assert not (tmp_path / "a").exists()
        assert generator.plan(processes).unsupported_features
    card.write_text(_card(loop_sm_compiled_model, tmp_path / "b", entries))
    plan = Generator(resolve_config(card)).plan(processes)
    assert plan.unsupported_features == ()
    assert plan.estimated_coverage["engine"] == "v2"
    assert plan.estimated_coverage["processes"][0]["engine"] == "v2"
    assert plan.estimated_coverage["processes"][0]["loop_graph_count"] == 0


# ------------------------------------------------------- orders, widths, models

MIXED_LIMITS = """coupling_order_policy = "explicit"
[process.max_coupling_orders]
QCD = 2
QED = 2"""
# Z, W, Higgs and Goldstone bosons: photon exchange only, as the slice tests
# request it (the public process syntax has no exclusion).
PHOTON_ONLY = (23, 24, -24, 25, 250, 251, -251)
COMPATIBILITY = 'fixed_parameters = {MB = "0"}\ncompatibility = "gmu_particle_anti"'


def _canonical(orders: dict) -> tuple:
    return tuple(sorted((name, power) for name, power in orders.items() if power))


@pytest.mark.loop_physics
def test_mixed_born_orders_take_one_loop_order_per_born_order(
    loop_sm_compiled_model, loop_native, loop_reference, tmp_path, monkeypatch
):
    """u u~ > d d~ with QCD and electroweak Born amplitudes (explicit limits
    QCD 2, QED 2): the dry run derives one loop order per Born amplitude
    order (QCD^4 and QCD^2 QED^2, not the limits raised by two) and does not
    claim loop admission; in the SM it generates, the W exchange of the
    electroweak Born closing at leg 1 through the conjugate rule of its
    vertex (WP3.2 EWI-2; its values against MadLoop with the Wolfenstein CKM:
    ``test_loop_v2_ew_lines``). With photon exchange only
    (Z, W and scalars excluded through the slice request) the public
    generator and runtime, regenerating at load, equal MadLoop's split-order
    control in every squared order (MadLoop binary64): each Born order within
    1e-12 of its own value (the colour-forbidden QCD^2 QED^2 interference, a
    structural zero, within 1e-12 of the point's largest Born) and each
    sector's Laurent coefficients within 1e-12 of the sector's largest
    coefficient."""

    from dataclasses import replace

    from pyamplicol import Runtime
    from pyamplicol.api import Generator, ProcessRequest, ProcessSet
    from pyamplicol.config import resolve_config
    from pyamplicol.generation import loop_service

    entries = (("uudd", "u u~ > d d~"),)
    processes = ProcessSet((ProcessRequest.parse("u u~ > d d~", name="uudd"),))
    card = tmp_path / "card.toml"
    card.write_text(
        _card(loop_sm_compiled_model, tmp_path / "a", entries).replace(
            'coupling_order_policy = "minimal"', MIXED_LIMITS
        )
    )
    generator = Generator(resolve_config(card))
    plan = generator.plan(processes)
    assert plan.unsupported_features == ()
    (coverage,) = plan.estimated_coverage["processes"]
    assert dict(coverage["born_orders"]) == {"QCD": 2, "QED": 2}
    assert [dict(orders) for orders in coverage["loop_orders"]] == [
        {"QCD": 4, "QED": 0},
        {"QCD": 2, "QED": 2},
    ]
    assert coverage["loop_admission_checked"] is False
    generator.generate(processes, tmp_path / "a")

    request = loop_service._v2_request
    monkeypatch.setattr(
        loop_service,
        "_v2_request",
        lambda *args: replace(request(*args), particle_veto=PHOTON_ONLY),
    )
    generator.generate(processes, tmp_path / "b")
    reference = loop_reference("mixed-uudd-qcd-control")
    points = json.loads((ROOT / "tools/loop_reference/uudd_qcd.json").read_text())
    born_key = {_canonical(v): k for k, v in reference["born_orders"].items()}
    loop_key = {_canonical(v): k for k, v in reference["loop_orders"].items()}
    runtime = Runtime.load(
        tmp_path / "b", process="uudd", model_parameters=GG_PARAMETERS
    )
    momenta = [[[str(x) for x in leg] for leg in point] for point in points["points"]]
    result = runtime.evaluate(momenta, mu_r2="1000000")
    assert set(result.born_orders) == set(born_key)
    assert [sector.order for sector in result.sectors] == sorted(loop_key)
    assert len(result.values) == len(reference["rows"]) == 3
    structural = (("QCD", 2), ("QED", 2))
    worst = 0.0
    for index, row in enumerate(reference["rows"]):
        borns = dict(zip(result.born_orders, result.born_values[index], strict=True))
        largest = max(abs(value) for value in row["born"].values())
        for order, key in born_key.items():
            expected = row["born"][key]
            scale = largest if order == structural else abs(expected)
            assert abs(borns[order] - expected) <= 1e-12 * scale, (index, order)
        for sector, laurent in zip(result.sectors, result.values[index], strict=True):
            expected = row["full"][loop_key[sector.order]]
            wanted = [expected[k] for k in ("finite", "single_pole", "double_pole")]
            scale = max(map(abs, wanted))
            for actual, target in zip(laurent, wanted, strict=True):
                assert abs(actual - target) <= 1e-12 * scale, (index, sector.order)
                worst = max(worst, abs(actual - target) / scale)
    print(f"mixed Born orders: largest sector deviation {worst:.2e}")


GGTT = (("ggtt", "g g > t t~"),)
WT_ZERO = ('fixed_parameters = {MB = "0"}', 'fixed_parameters = {MB = "0", WT = "0"}')


def _ggtt_generator(model: Path, tmp_path: Path, output: str, edit=None):
    from pyamplicol.api import Generator
    from pyamplicol.config import resolve_config

    text = _card(model, tmp_path / output, GGTT)
    card = tmp_path / f"{output}.toml"
    card.write_text(text.replace(*edit) if edit else text)
    return Generator(resolve_config(card))


def _ggtt_runtime(model: Path, tmp_path: Path):
    """g g > t t~ through v2 with WT fixed to zero, the provider reference's
    exact on-shell points and scales, and the removed legacy engine's frozen
    values there at the model defaults (its propagators carried no width)."""

    from pyamplicol import Runtime
    from pyamplicol.api import ProcessRequest, ProcessSet

    processes = ProcessSet((ProcessRequest.parse("g g > t t~", name="ggtt"),))
    _ggtt_generator(model, tmp_path, "v2", WT_ZERO).generate(processes, tmp_path / "v2")
    required = json.loads(
        (ROOT / "tests/fixtures/loop/required/gg_ttx.json").read_text()
    )
    scales = [str(Decimal(mu) ** 2) for mu in required["mu_r"]]
    frozen = _fixture("gg_ttx_model_defaults")
    assert frozen["model_parameters"] == GG_PARAMETERS
    assert (frozen["points"], frozen["scales_squared"]) == (required["points"], scales)
    runtime = Runtime.load(
        tmp_path / "v2", process="ggtt", model_parameters=GG_PARAMETERS
    )
    return runtime, required["points"], scales, frozen["results"]


@pytest.mark.loop_physics
def test_coloured_widths_in_tree_currents_are_fixed_to_zero(
    loop_sm_compiled_model, loop_native, tmp_path
):
    """g g > t t~ propagates the top in t-channel tree currents. At loop_sm's
    default top width the v2 request is refused, by the dry run and before
    any loop generation, naming the fix (v2 is real on-shell: a width there
    would make the Born, its colour correlations and the virtual
    inconsistent). With WT fixed to zero ([loop.preparation]) the public v2
    virtual equals the removed legacy engine's frozen values (its propagators
    carried no width) at the model's default parameters (WT = 1.4915
    included) at the three exact on-shell points and scales of the provider
    reference: binary64 1e-12 and double-double 1e-24, against legacy in
    double-double (60 digits nightly). An evaluation refuses a WT other than the fixed
    zero, the scale parameter and unknown names."""

    import re

    from pyamplicol import Runtime
    from pyamplicol.api import ProcessRequest, ProcessSet

    processes = ProcessSet((ProcessRequest.parse("g g > t t~", name="ggtt"),))
    reason = (
        r"coloured-width: 'ggtt' \([^)]*\): tree currents propagate coloured "
        r"particles whose width is not fixed to zero: t \(WT\); fix it in "
        r'\[loop\.preparation\] fixed_parameters \(WT = "0"\).*'
        r'model\.restriction = "no_widths".*real on-shell'
    )
    refused = _ggtt_generator(loop_sm_compiled_model, tmp_path, "a")
    (blocker,) = refused.plan(processes).unsupported_features
    assert re.search(reason, blocker), blocker
    with pytest.raises(GenerationError, match=reason):
        refused.generate(processes, tmp_path / "a")
    assert not (tmp_path / "a").exists()
    runtime, points, scales, frozen = _ggtt_runtime(loop_sm_compiled_model, tmp_path)
    results = frozen["double-double"]
    _assert_legacy(runtime.evaluate(points, mu_r2=scales), results, DOUBLE_TOLERANCE)
    _assert_legacy(
        runtime.evaluate(points, mu_r2=scales, arithmetic="double-double"),
        results,
        DOUBLE_DOUBLE_TOLERANCE,
    )
    runtime.set_model_parameters({"WT": "0"})
    for parameters, refusal in (
        ({"WT": "1.4915"}, r"'WT' was fixed to 0 at generation"),
        ({"MU_R": "100"}, r"'MU_R' is the renormalisation scale.*mu_r2"),
        ({"WX": "1"}, r"unknown external parameter 'WX'"),
    ):
        with pytest.raises(ValueError, match=refusal):
            runtime.set_model_parameters(parameters)
        with pytest.raises(ValueError, match=refusal):
            Runtime.load(tmp_path / "v2", process="ggtt", model_parameters=parameters)


@pytest.mark.loop_nightly
def test_coloured_width_fixed_to_zero_equals_legacy_at_sixty_digits(
    loop_sm_compiled_model, loop_native, tmp_path
):
    """g g > t t~ with WT fixed to zero against the frozen legacy values at
    the model defaults, both at 60 digits (1e-40)."""

    runtime, points, scales, frozen = _ggtt_runtime(loop_sm_compiled_model, tmp_path)
    options = {"arithmetic": "arbitrary", "precision": ARBITRARY_DIGITS}
    results = frozen["arbitrary-60"]
    _assert_legacy(
        runtime.evaluate(points, mu_r2=scales, **options),
        results,
        ARBITRARY_TOLERANCE,
    )


def _qed_model_generator(
    model: Path, tmp_path: Path, name: str, entries, *, profiled, fixed='MB = "0"'
):
    from pyamplicol.api import Generator
    from pyamplicol.config import resolve_config

    text = _card(model, tmp_path / name, entries).replace(
        'loop_coupling_orders = ["QCD"]', 'loop_coupling_orders = ["QCD", "QED"]'
    )
    card = tmp_path / f"{name}.toml"
    card.write_text(
        text.replace(
            'fixed_parameters = {MB = "0"}',
            COMPATIBILITY.replace('{MB = "0"}', f"{{{fixed}}}"),
        )
        if profiled
        else text
    )
    return Generator(resolve_config(card))


@pytest.mark.loop_physics
def test_qed_declaring_model_names_its_blockers(
    loop_qcd_qed_sm_ufo, loop_native, tmp_path
):
    """MG5_aMC's loop_qcd_qed_sm with QCD and QED corrections through the
    public v2 path (its MU_R, at LHA code LOOP 666, is resolved at the public
    boundary). Without the compatibility profile the structural source
    repair restores its G- Goldstone flag (MS-MODEL, rule
    ``antiparticle-goldstone-flag``): the dry run admits the request. With
    the profile (still accepted), u u~ > d d~ at the card's W and Z widths
    is refused by the
    electroweak width policy (WP3.2 EWI-0b, the user's decision [U8]: the
    widths fixed to zero, or the complex-mass scheme), before its tree
    kernels and the counterterm lowering. With those widths fixed to zero
    its electroweak counterterms are lowered (their CT scalars with the
    complex-mass scheme's helper recms and the CMS reglog are
    parameter-program references, WP3.2 EWI-3) and the request generates in
    't Hooft-Feynman gauge (WP3.2 EWI-4b: W, Z, Goldstones and ghosts on its
    loop lines), recorded in its generation filters; its Goldstone-vector
    mixings are lowered as vertex insertions, so none is inert (in unitary
    gauge the tree model absorbed the Goldstones and the G0-photon mixing
    was inert, REQUIREMENTS_ADDENDUM J)."""

    from pyamplicol.api import ProcessRequest, ProcessSet
    from pyamplicol.artifacts import load_manifest
    from pyamplicol.loops.v2.ew_policy import ElectroweakWidthError
    from pyamplicol.models.loading import compile_model_source

    entries = (("uudd", "u u~ > d d~"),)
    processes = ProcessSet((ProcessRequest.parse("u u~ > d d~", name="uudd"),))
    # The dry run reads compiled models from the model cache only: populate it
    # (keyed by the model compiler's sources, so a compiler change empties it).
    compile_model_source(loop_qcd_qed_sm_ufo, require_supported=False)
    plain = _qed_model_generator(
        loop_qcd_qed_sm_ufo, tmp_path, "a", entries, profiled=False
    )
    assert plain.plan(processes).unsupported_features == ()
    profiled = _qed_model_generator(
        loop_qcd_qed_sm_ufo, tmp_path, "b", entries, profiled=True
    )
    with pytest.raises(
        ElectroweakWidthError,
        match=r"'u u~ > d d~': an electroweak request .* with a width that is not "
        r"fixed to zero: W\+ \(WW\), W- \(WW\), Z \(WZ\)\..* fixed_parameters "
        r'\(WW = "0", WZ = "0"\); .*complex-mass scheme',
    ):
        profiled.generate(processes, tmp_path / "b")
    assert not (tmp_path / "b").exists()
    zero_widths = _qed_model_generator(
        loop_qcd_qed_sm_ufo,
        tmp_path,
        "c",
        entries,
        profiled=True,
        fixed='MB = "0", WW = "0", WZ = "0"',
    )
    zero_widths.generate(processes, tmp_path / "c")
    (record,) = load_manifest(tmp_path / "c").extensions["generation"][
        "concrete_processes"
    ]
    assert record["filters"]["gauge"] == "feynman"
    inert = record["filters"]["counterterms_inert"]
    assert not any("between spins" in reason for reason in inert.values()), inert


@pytest.mark.loop_nightly
def test_qed_declaring_model_joint_corrections_equal_legacy(
    loop_qcd_qed_sm_ufo, loop_native, tmp_path
):
    """g g > g g with QCD and QED corrections on loop_qcd_qed_sm
    (compatibility profile ``gmu_particle_anti``) through the public v2 path
    equals the removed legacy engine's frozen values: Born and the QCD^6
    virtual in binary64 (1e-12) and at 60 digits (1e-40, legacy at 60
    digits). v2 also lists the
    QCD^4 QED^2 sector, a structural zero (no one-loop g g > g g amplitude of
    order QCD^2 QED^2), which legacy omits. The request takes 't Hooft-Feynman
    gauge (WP3.2 EWI-4b, recorded in the generation filters). The electroweak
    CT rules are lowered (their CT scalars with the complex-mass scheme's
    helpers are parameter-program references, WP3.2 EWI-3; the
    Goldstone-vector mixings as vertex insertions, EWI-4b), so no rule is
    inert for a UFO helper or a mixing between spins (REQUIREMENTS_ADDENDUM
    J). The referenced CT scalars are native parameter-program outputs, so
    the v2 artifact stores native point inputs."""

    from pyamplicol import Runtime
    from pyamplicol.api import ProcessRequest, ProcessSet
    from pyamplicol.artifacts import load_manifest

    entries = (("gg", "g g > g g"),)
    processes = ProcessSet((ProcessRequest.parse("g g > g g", name="gg"),))
    _qed_model_generator(
        loop_qcd_qed_sm_ufo, tmp_path, "v2", entries, profiled=True
    ).generate(processes, tmp_path / "v2")
    runtimes = {
        "v2": Runtime.load(
            tmp_path / "v2", process="gg", model_parameters=GG_PARAMETERS
        )
    }
    (record,) = load_manifest(tmp_path / "v2").extensions["generation"][
        "concrete_processes"
    ]
    assert record["filters"]["gauge"] == "feynman"
    inert = record["filters"]["counterterms_inert"]
    assert not any(
        "UFO helper" in reason or "between spins" in reason for reason in inert.values()
    ), inert
    # The referenced CT scalars are native parameter-program outputs: no
    # load-time regeneration (REQUIREMENTS_ADDENDUM G).
    assert runtimes["v2"].inspect()["point_inputs"] == "native-v2"
    frozen = _fixture("gg_gg_qcd_qed_sm")
    assert frozen["model_parameters"] == GG_PARAMETERS
    assert (frozen["points"], frozen["scales_squared"]) == ([POINT], ["1000000"])
    results = frozen["results"]["arbitrary-60"]
    for options, tolerance in (
        ({}, DOUBLE_TOLERANCE),
        (
            {"arithmetic": "arbitrary", "precision": ARBITRARY_DIGITS},
            ARBITRARY_TOLERANCE,
        ),
    ):
        value = runtimes["v2"].evaluate([POINT], mu_r2="1000000", **options)
        assert [sector.order for sector in value.sectors] == [
            (("QCD", 4), ("QED", 2)),
            (("QCD", 6),),
        ]
        assert all(c == 0 for c in value.values[0][0])
        qcd = value.for_order((("QCD", 6),))
        _assert_legacy(qcd, results, tolerance)


# ------------------------------------------------------------------ size


@pytest.mark.loop_nightly
def test_six_gluon_plan_v4_artifact_is_below_twenty_megabytes(
    loop_sm_compiled_model, loop_native, tmp_path
):
    """gg -> gggg (1100 families, 45 MB of plan JSON without the forward
    oracle, and the companion plan of 6 certified-scaleless families)
    publishes < 20 MB (measured: 11 MB with the native point inputs and the
    counterterm program of the recursion; 8.8 MB on the slice without native
    point inputs, 5.3 MB of it the compiled model and 3.5 MB the plan PACBIN
    with the companion)."""

    from pyamplicol.loops.v2.artifact import admit, plan_digest, read_plan

    entries = (("gg4g", "g g > g g g g"),)
    card = _card(loop_sm_compiled_model, tmp_path / "artifact", entries).replace(
        "post_build_validation = true", "post_build_validation = false"
    )
    output = _generate(card, tmp_path, entries)
    size = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    print(f"gg > gggg plan v4 artifact: {size} bytes")
    assert size < 20 * 10**6, size
    execution = json.loads((output / "processes/gg4g/execution.json").read_text())
    plan = read_plan(output / "processes/gg4g/loop-runtime.pacbin")
    assert plan_digest(plan) == execution["plan_sha256"]
    assert len(json.loads(plan)["families"]) == 1100
    admitted = admit(loop_native, output, "gg4g")
    assert admitted["families"] == 1100
    assert admitted["plan_sha256"] == execution["plan_sha256"]
    # The companion plan of the certified-scaleless families (poles uv, ir).
    assert admitted["scaleless_families"] == 6
    assert admitted["scaleless_plan_sha256"] == execution["scaleless_plan_sha256"]
