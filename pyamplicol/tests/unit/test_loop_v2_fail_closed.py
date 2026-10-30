# SPDX-License-Identifier: 0BSD
"""Fail-closed generation of loop v2 (package fail-closed; second genericity
audit, ``workspace/reports/engine-v2/genericity-audit-2/REPORT.md``).

An unsupported structure is refused with a typed error naming a listed rule,
never served as a silently wrong value, and the rule is structural:

* G2-M1 (``refusals.LOOP_ADMISSION_REFUSALS`` key
  ``colour-accuracy-nc-exact``): LC and NLC over tree blocks that are not exact
  in N_c (blocks mixing U(1)-subtraction grades: identical-flavour quark lines
  compiled without the grade split of ``generation.dag_grades``, which every
  LC and NLC request runs; ``_without_grade_split``) are refused at
  generation, through ``LoopSlice.generate``, the public Python API and the
  CLI, before any family or export, and every export refuses them again, so
  no stored plan, native runtime or SDK serves a grade-0 projection.
  Identical flavours with N_c-exact blocks (u u~ > u u~, and u u~ > u u~ g with
  the split) and full colour are not refused: the criterion is N_c-exactness,
  not the flavours.
* G2-M4 interim (``native_trees.NATIVE_EXPORT_REFUSALS`` key
  ``plane-mismatch``): the SymJIT plane check's disagreement is a listed
  refusal naming the kernel or the parameter-program output.
* G2-m4 (the former ``external-photon-alpha0``): resolved by WP3.2 EWI-N,
  the native point inputs take the alpha(0) convention of on-shell external
  photons (``test_loop_v2_ew_native.py``, ``test_loop_v2_ew_policy.py``).
* G2-m1 (``fermion-line-representation``): a toy leptoquark UFO (loop_sm with
  one triplet scalar and its u e vertex, ``loop_v2_ufo_variants``) is refused
  before any tree is built, naming the vertex, whichever DAG lane is on; with
  the admission scan bypassed the native DAG refuses with the listed key
  (``rusticol_core::loop_v2::dag::FERMION_LINE_REPRESENTATION``).

The dry run lists both tables. The stored processes of the controls the
guards do not refuse keep the bytes of the slice before the package.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from pathlib import Path

import pytest

from pyamplicol.loops.v2.refusals import LoopRequestUnsupported

ROOT = Path(__file__).resolve().parents[2]
FIXED = {"MB": "0"}
# Identical-flavour quark lines (blocks mixing U(1)-subtraction grades
# without the grade split) and a relabelled copy (another flavour, permuted
# legs).
MIXED = {
    "uux_uuxg": ("u u~ > u u~ g", {"QCD": 3, "QED": 0}),
    "gg_uuxuux": ("g g > u u~ u u~", {"QCD": 4, "QED": 0}),
    "ddx_gddx": ("d d~ > g d d~", {"QCD": 3, "QED": 0}),
}
NC_EXACT_REFUSAL = "colour-accuracy-nc-exact"
LINE_REFUSAL = "fermion-line-representation"


def _declaration():
    from pyamplicol.models.loop_color_declaration import LoopColorDeclaration
    from tools.loop_reference import legacy_capture

    return LoopColorDeclaration.from_dict(
        json.loads(legacy_capture.COLOUR_DECLARATION.read_text())
    )


def _request(process, born, accuracy, **changes):
    from pyamplicol.loops.v2.assemble import SliceRequest

    loops = {**born, "QCD": born["QCD"] + 2}
    colour = (
        {"colour_declaration": _declaration(), "colour_contraction": "direct"}
        if accuracy != "full"
        else {}
    )
    return SliceRequest(
        process,
        born,
        loops,
        fixed_parameters=FIXED,
        accuracy=accuracy,
        **colour,
        **changes,
    )


def _without_grade_split(monkeypatch) -> None:
    """Compile the slice's loop-request DAGs (loop blocks and the CT
    program, either DAG lane) without the grade split of
    ``generation.dag_grades``: identical-flavour blocks then mix
    U(1)-subtraction grades, as before package identical-flavour-lc."""

    from pyamplicol.loops.v2 import assemble, ct_closure

    for module, name in (
        (assemble, "build_tree_blocks"),
        (ct_closure, "counterterm_program_blocks"),
    ):
        original = getattr(module, name)

        def unsplit(*args, original=original, **kwargs):
            return original(*args, **{**kwargs, "u1_grades": False})

        monkeypatch.setattr(module, name, unsplit)


def _stop(stage: str):
    def stop(*_args, **_kwargs):
        raise AssertionError(f"the guard did not refuse before {stage}")

    return stop


@pytest.fixture(scope="module")
def loop_sm(loop_sm_compiled_model):
    from pyamplicol.models.loading import load_compiled_model

    return load_compiled_model(loop_sm_compiled_model)


# ------------------------------------------------------------------ tables


@pytest.mark.loop_fast
def test_refusals_are_listed_typed_and_name_their_rule():
    from pyamplicol.api.errors import GenerationError
    from pyamplicol.loops.v2.native_trees import (
        NATIVE_EXPORT_REFUSALS,
        NativeExportUnsupported,
    )
    from pyamplicol.loops.v2.refusals import LOOP_ADMISSION_REFUSALS

    assert [refusal.key for refusal in LOOP_ADMISSION_REFUSALS] == [
        NC_EXACT_REFUSAL,
        LINE_REFUSAL,
        # WP3.2 EWI-5's interim veto guard (``test_loop_v2_ew_leptons.py``).
        "veto-unlabelled-counterterms",
        # fail-closed-2's restriction guard (``test_loop_v2_restricted_zero.py``).
        "loop-vertex-outside-tree-model",
        # MQ-MODEL's QCD guards (``test_loop_v2_stock_loop_sm.py``).
        "loop-induced",
        "coloured-width",
        # M-SM MS-CMS's mass-scheme policy (``test_loop_model_cms.py``).
        "electroweak-width",
        "mass-scheme",
        "complex-mass-scheme",
        "cms-internal-mass",
        "cms-yukawa-input",
        "cms-external-width",
        "complex-fixed-parameter",
        "lorentz-slot-arity",
        "majorana-fermion",
        "multi-fermion-contact",
        # MQ-FFT's forced seed FFT guard (``test_loop_v2_seed_fft.py``).
        "forced-seed-fft-unserved",
        # MQ-SELECT's correlated-selection, subleading- and interfering-flow guards
        # (``test_loop_v2_selection.py``).
        "selected-correlators",
        "subleading-colour-flows",
        "interfering-colour-flows",
        # MQ-NC's declared-N_c guard (``test_loop_v2_nc_declaration.py``).
        "colour-declaration-coverage",
        # MS-PHOTON's external-photon guards (``test_loop_v2_external_photons.py``,
        # ``test_loop_v2_ew_policy.py``).
        "external-photon-scheme",
        "external-photon-legs",
        # m-sm MS-MODEL: the typed electroweak gauge policy, the antiparticle
        # metadata and the [U2] veto (``test_loop_v2_refusal_keys.py``).
        "feynman-gauge",
        "antiparticle-metadata",
        "anomalous-veto",
    ]
    with pytest.raises(ValueError, match="unknown loop admission refusal"):
        LoopRequestUnsupported("lc-downgrade", "a projection")
    error = LoopRequestUnsupported(NC_EXACT_REFUSAL, "'u u~ > u u~ g': detail")
    assert isinstance(error, GenerationError) and error.key == NC_EXACT_REFUSAL
    message = str(error)
    assert message.startswith(
        f"{NC_EXACT_REFUSAL}: 'u u~ > u u~ g': detail. Rule: LC and NLC project"
    )
    assert 'Instead: full colour (color.accuracy = "full")' in message
    keys = {refusal.key for refusal in NATIVE_EXPORT_REFUSALS}
    assert "plane-mismatch" in keys
    # On-shell external photons take native inputs (WP3.2 EWI-N).
    assert "external-photon-alpha0" not in keys
    assert NativeExportUnsupported("plane-mismatch", "x").feature == "plane-mismatch"
    with pytest.raises(ValueError, match="unknown native export refusal"):
        NativeExportUnsupported("external-photon-alpha0", "x")


@pytest.mark.loop_fast
def test_plane_mismatch_is_a_listed_refusal_naming_the_output(loop_native):
    """A program reading a branch cut of its inputs without a real point to
    certify its real-axis evaluation at (``test_loop_v2_ew_native.py``) is
    refused as ``plane-mismatch``, naming the output: a tree program's
    kernel by its class and index, a parameter-program output by its key, a
    derived coupling by its UFO coupling and vertex, a counterterm
    coefficient with its epsilon power and N_c grade. With a real point it
    compiles, on the square root's cut (x = 1.118 > y = 0.736: a negated
    real, whose zero imaginary part the translation makes +0, exact
    arithmetic's side; M-SM MS-MASTERS) and off it."""

    from types import SimpleNamespace

    from symbolica import E, Expression, S

    from pyamplicol.loops.v2 import native_trees

    state = bytes(
        Expression.evaluator_multiple(
            [E("x + y"), E("sqrt(1 - x^2/y^2)")], [S("x"), S("y")], jit_compile=False
        ).save()
    )

    on_the_cut = {0: [[1.118, 0.736]]}

    def refused(states, where=None) -> str:
        with pytest.raises(native_trees.NativeExportUnsupported) as error:
            native_trees._plane_applications(states, where, None)
        assert error.value.feature == "plane-mismatch"
        return str(error.value)

    term = SimpleNamespace(
        id=42, coupling="GC_100", vertex="V_77", particles=("W+", "a")
    )
    model = SimpleNamespace(ir=SimpleNamespace(vertex_terms=(term,)))
    empty = {"kernels": [], "propagators": []}
    trees = {"interactions": {"kernels": [[]], "propagators": [[]]}}
    keys = (("model", "MWx"), ("model", "derived_coupling_42"))
    outputs = native_trees._PlaneOutputs(
        {"interactions": empty}, keys, (model, None)
    ).where
    message = refused((state,), outputs)
    # The output once, then the native's statement of the refusal.
    assert message.startswith(
        "plane-mismatch: parameter-program output 1 (model coupling GC_100 of "
        "vertex V_77 (W+ a)): loop v2 trees: output 1 reads a branch-cut"
    ), message
    assert "keep this program on the exact evaluator" not in message
    counterterm = (("model", "MWx"), ("counterterm", "UV_GT", 0, 1))
    outputs = native_trees._PlaneOutputs(
        {"interactions": empty}, counterterm, (model, None)
    ).where
    assert "(counterterm UV_GT, epsilon power 0, N_c grade 1)" in refused(
        (state,), outputs
    )
    outputs = native_trees._PlaneOutputs(
        {**trees, "counterterm_trees": trees}, keys, (model, None)
    ).where
    assert outputs(1, 1) == "output 1 of propagator kernel 0 of the tree program"
    assert outputs(2, 3) == "output 3 of vertex kernel 0 of the counterterm program"
    assert outputs(4, 0) == "parameter-program output 0 (model MWx)"
    assert "output 1 of saved evaluator 0" in refused((state,))
    # With a real point, on the cut and off it, it compiles.
    off = bytes(
        Expression.evaluator_multiple(
            [E("sqrt(1 + x^2/y^2)")], [S("x"), S("y")], jit_compile=False
        ).save()
    )
    assert native_trees._plane_applications((off,), outputs, on_the_cut)
    assert native_trees._plane_applications((state,), outputs, on_the_cut)
    assert native_trees._plane_applications((state,), outputs, {0: [[0.736, 1.118]]})


# ---------------------------------------------------- G2-M1: N_c exactness


@pytest.mark.loop_physics
@pytest.mark.parametrize(
    ("name", "accuracy"),
    [
        ("uux_uuxg", "lc"),
        ("uux_uuxg", "nlc"),
        ("gg_uuxuux", "lc"),
        ("gg_uuxuux", "nlc"),
        ("ddx_gddx", "lc"),
    ],
)
def test_projections_over_blocks_not_exact_in_nc_are_refused_before_the_families(
    loop_sm, loop_native, monkeypatch, name, accuracy
):
    """LC and NLC of identical-flavour quark lines compiled without the
    grade split: the typed refusal names the rule and full colour, raised
    right after the trees (before their kernels, the families and any
    export)."""

    from pyamplicol.loops.v2 import assemble
    from pyamplicol.loops.v2.assemble import LoopSlice

    process, born = MIXED[name]
    _without_grade_split(monkeypatch)
    monkeypatch.setattr(assemble, "with_ward_kernels", _stop("the tree kernels"))
    with pytest.raises(LoopRequestUnsupported) as refused:
        LoopSlice.generate(loop_sm, _request(process, born, accuracy), loop_native)
    assert refused.value.key == NC_EXACT_REFUSAL
    message = str(refused.value)
    assert f"'{process}': {accuracy.upper()} needs N_c-exact tree currents" in message
    assert re.search(
        r"mix U\(1\)-subtraction grades \(\d+ current\(s\): legs [\d ]+ as PDG -?\d+",
        message,
    ), message
    assert 'full colour (color.accuracy = "full")' in message


@pytest.mark.loop_physics
def test_identical_flavours_with_nc_exact_blocks_and_full_colour_generate(
    loop_sm, loop_native
):
    """The guard is keyed on N_c exactness, not on flavours: u u~ > u u~ has
    identical flavours and N_c-exact blocks (LC generates), the grade split
    makes the blocks of u u~ > u u~ g N_c-exact (LC generates, graded), and
    full colour of u u~ > u u~ g keeps every power at N_c = 3 (generates)."""

    from pyamplicol.loops.v2.assemble import LoopSlice

    lc = LoopSlice.generate(
        loop_sm, _request("u u~ > u u~", {"QCD": 2, "QED": 0}, "lc"), loop_native
    )
    assert lc.colour.formal_nc and lc.table.formal_nc
    split = LoopSlice.generate(loop_sm, _request(*MIXED["uux_uuxg"], "lc"), loop_native)
    assert split.blocks.graded and split.colour.formal_nc and split.table.formal_nc
    full = LoopSlice.generate(
        loop_sm, _request(*MIXED["uux_uuxg"], "full"), loop_native
    )
    assert not full.colour.formal_nc


@pytest.mark.loop_physics
def test_exports_refuse_projections_over_blocks_not_exact_in_nc(
    loop_sm, loop_native, monkeypatch
):
    """Defence in depth: an LC slice of u u~ > u u~ g built without the
    grade split and with the generation guard bypassed is refused by every
    export layer in turn
    (``export_plan``, the seed records, the native trees' counterterm
    program) and by the native seed records of column sets, so no stored
    plan carries its grade-0 projection."""

    from pyamplicol.loops.v2 import assemble, native_cells, plan_export, refusals
    from pyamplicol.loops.v2.assemble import LoopSlice

    with monkeypatch.context() as patch:
        _without_grade_split(patch)
        patch.setattr(assemble, "require_nc_exact", lambda *args: None)
        slice_ = LoopSlice.generate(
            loop_sm, _request(*MIXED["uux_uuxg"], "lc"), loop_native
        )
    assert not slice_.colour.formal_nc

    def refused() -> str:
        with pytest.raises(LoopRequestUnsupported) as error:
            plan_export.export_plan(
                slice_, forward=False, trees=True, born_squared=True
            )
        assert error.value.key == NC_EXACT_REFUSAL
        return str(error.value)

    assert "'u u~ > u u~ g': LC needs N_c-exact" in refused()
    original = refusals.require_nc_exact
    calls = []

    def off_at_first_call(blocks, accuracy, process):
        calls.append(blocks)
        if len(calls) > 1:
            return original(blocks, accuracy, process)

    # ``export_plan``'s own check off: its seed records refuse.
    monkeypatch.setattr(refusals, "require_nc_exact", off_at_first_call)
    assert "the seed records of an LC request" in refused()
    # The seed-table check off too: the native trees refuse over their
    # counterterm program's blocks (the second call; the slice's own blocks
    # are ``export_plan``'s first).
    calls.clear()
    monkeypatch.setattr(plan_export, "require_formal_nc_tables", lambda *a: None)
    assert "'u u~ > u u~ g': LC needs N_c-exact" in refused()
    assert len(calls) == 2
    # The native seed records of a column set check their tables.
    with pytest.raises(LoopRequestUnsupported, match="native seed records"):
        native_cells.project_records(slice_, plan_export.sector_columns(slice_), (), {})


# The bytes of the controls the guards do not refuse, measured on the slice
# before package fail-closed (work/v2-slice 6da37352, native digest a1e0606a;
# equal at 59d47cb3 and 11542fd5 except g g > u u~ u u~'s native trees, whose
# counterterm program carries the root-closure reading signs of 6da37352,
# ew-epsilon's G2-M7): per control the SHA-256 of the stored plan JSON, of
# the companion plan JSON, and of every native member but the parameter
# programs' plane applications (``_identity``). A package that changes these
# plans on purpose re-measures them on its base and states why.
# Re-pinned by sdk-parity (merged with fb54246c): the native digests only,
# for three record fields it adds (genericity audit 2 M2/m3): the tree
# programs' ``parameters.scale_outputs`` and the native record's
# ``externals`` and ``preparation_fixed_parameters``; with those removed every
# native digest equals the former pin, and the plan and companion digests
# are unchanged.
# Re-pinned by mq-select: the LC native digests only, for the leading-colour
# flow catalogue the stored observables record now holds (``color_flows``,
# ``native_observables._flows_record``; runtime colour-flow selection); with
# it removed both LC native digests equal the former pins, the plan and
# companion digests are unchanged and NLC and full colour keep every byte.
# Re-pinned again by mq-select (fix round 2): the catalogue's flow records
# drop ``computed`` and ``representative`` (read by nothing); with the two
# fields put back both LC native digests equal the round-1 pins.
# Re-pinned by mq-fft (M-QCD MQ-FFT, design §7.3): g g > u u~ u u~'s native
# digest only, for the seed Fourier record its native process record now
# carries (S_2 on the two gluons, certified); with that record removed the
# native digest equals the former pin (``FOURIER_FREE``), and its plan and
# companion digests are unchanged (the plan v4 bytes do not depend on the
# seed contraction).
CONTROLS = {
    "uux_ddxg_lc": ("u u~ > d d~ g", {"QCD": 3, "QED": 0}, "lc"),
    "uux_ddxg_nlc": ("u u~ > d d~ g", {"QCD": 3, "QED": 0}, "nlc"),
    "uux_ddxssx_lc": ("u u~ > d d~ s s~", {"QCD": 4, "QED": 0}, "lc"),
    "uux_ddxssx_nlc": ("u u~ > d d~ s s~", {"QCD": 4, "QED": 0}, "nlc"),
    "uux_uuxg_full": (*MIXED["uux_uuxg"], "full"),
    "gg_uuxuux_full": (*MIXED["gg_uuxuux"], "full"),
}
CONTROL_BYTES = {
    "uux_ddxg_lc": (
        "f8595356dbb40148749b9828679d4a368518cb0922e3f680fe8313af978ace95",
        "8c32bb3bc825e3c3fed06ca962091ad64fe3b87ffc636747a0d0482b9ac83fb2",
        "3771d997dc0a75d410e4ec6a1f30765317e68c0ee2df09e114eccc2f7f083f65",
    ),
    "uux_ddxg_nlc": (
        "4cd00549a72bfa194778472289d5af0c31c7dfc471c27d54bd116152eb62e2ad",
        "f27586cffefed6fd94ceaef0155a11590ccba22e423f7014efe0f49457ddb249",
        "99a27d7b629adf25e46deb4a3a9936bfcd6f0794f88ea3f797c05717841728fb",
    ),
    "uux_ddxssx_lc": (
        "eadb28c424c1d45046399f8b6dd8832d805c0eb1ffb59906b86ead2927c3d09d",
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "9cc0b75266b77e6bd39c237496afe8637282573278a064432bc1eeb99dcad1dc",
    ),
    "uux_ddxssx_nlc": (
        "698711e3b9670e135f20719c82041c020bbd846f404ee963eb74b92e186432ab",
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "a6b26a742d6c3aa1b125e94b3ad5c92e31219f28ffaa1c38881a581621fc9ece",
    ),
    "uux_uuxg_full": (
        "a80ff312a2155ddbbc9f02f352b90335427eb9e19b560702e21b5431450cf28e",
        "f3f96c6e2a657899cef204358546e9e6e2d865093ab7a2cda27706af315c6ad5",
        "3decb1412ce99184479564e7ecbe1fc7583dbd757ba66287a54c5df551e75ffe",
    ),
    "gg_uuxuux_full": (
        "52aebd812ec8a579a8268de84370f850b69b3a05f83a8cb54fb109eb74620769",
        "50b2f50da81a8619ebb3b2b5dd5b078c6c7aa1fe79727531eaebe2a703793d23",
        "f814e4f90bbe6ac8ed84b8f4f0f1616d6c45d1b014ef52968934bc88a9117f83",
    ),
}
# The native digests of the controls without their seed Fourier record (the
# pins before mq-fft).
FOURIER_FREE = {
    "gg_uuxuux_full": (
        "4121edeec2a19785a76b8cb34a9e053fc5a132301b3e917a19d787d371056c8a"
    ),
}


def _identity(generated, *, fourier: bool = True) -> tuple[str, str, str]:
    """``CONTROL_BYTES``' digests of a stored process. The parameter
    programs' plane applications (``*parameters.symjit``) are left out: their
    SymJIT call tags are random (``SYMJIT_RANDOM_CALL_TAG``, strict xfail
    ``test_symjit_plane_helper_functions``); their exact evaluators
    (``*parameters.bin``) are in. ``fourier=False`` drops the seed Fourier
    records from the native process record (``FOURIER_FREE``)."""

    import dataclasses

    sha = lambda data: hashlib.sha256(data).hexdigest()  # noqa: E731
    native = generated.native
    if not fourier:
        record = {
            key: value for key, value in native.record.items() if key != "fourier"
        }
        if record.get("scaleless") is not None:
            record["scaleless"] = {
                key: value
                for key, value in record["scaleless"].items()
                if key != "fourier"
            }
        native = dataclasses.replace(native, record=record)
    members = native.members()
    native = hashlib.sha256()
    for name in sorted(members):
        if not name.endswith("parameters.symjit"):
            native.update(
                name.encode() + b"\0" + hashlib.sha256(members[name]).digest()
            )
    return (
        sha(generated.plan_json.encode()),
        sha((generated.scaleless_plan_json or "").encode()),
        native.hexdigest(),
    )


@pytest.mark.loop_physics
@pytest.mark.parametrize(
    "name",
    [
        "uux_ddxg_lc",
        pytest.param("uux_ddxg_nlc", marks=pytest.mark.loop_nightly),
        pytest.param("uux_ddxssx_lc", marks=pytest.mark.loop_nightly),
        pytest.param("uux_ddxssx_nlc", marks=pytest.mark.loop_nightly),
        "uux_uuxg_full",
        pytest.param("gg_uuxuux_full", marks=pytest.mark.loop_nightly),
    ],
)
def test_plans_the_guards_do_not_refuse_keep_their_bytes(loop_sm, loop_native, name):
    """Byte identity of the stored processes the package does not refuse:
    LC/NLC of distinct-flavour quark lines (u u~ > d d~ g, u u~ > d d~ s s~)
    and full colour of identical-flavour ones (u u~ > u u~ g,
    g g > u u~ u u~) give the plan, companion plan and native member bytes
    of the slice before the package (``CONTROL_BYTES``)."""

    from pyamplicol.loops.v2.artifact import generate_process

    process, born, accuracy = CONTROLS[name]
    generated = generate_process(
        loop_sm, _request(process, born, accuracy), loop_native
    )
    assert _identity(generated) == CONTROL_BYTES[name]
    if name in FOURIER_FREE:
        assert "fourier" in generated.native.record
        assert _identity(generated, fourier=False)[2] == FOURIER_FREE[name]
    else:
        assert "fourier" not in generated.native.record


# ---------------------------------------- G2-M1 through the API and the CLI

_SCHEME_FREE_CARD = """
action = "generate"
[model]
source = {model}
[process]
entries = [{entries}]
loop_coupling_orders = ["QCD"]
coupling_order_policy = "minimal"
[color]
accuracy = "{accuracy}"
contraction = "direct"
[loop]
r2_strategy = "model"
[loop.preparation]
fixed_parameters = {{MB = "0"}}
[generation]
output = {output}
emit_api_bundle = false
[generation.validation]
post_build_validation = false
"""


def _card(model, output, name, expression, accuracy) -> str:
    entry = f"{{expression={json.dumps(expression)}, name={json.dumps(name)}}}"
    return _SCHEME_FREE_CARD.format(
        model=json.dumps(str(model)),
        entries=entry,
        accuracy=accuracy,
        output=json.dumps(str(output)),
    )


@pytest.mark.loop_physics
@pytest.mark.parametrize("name", ["uux_uuxg", "gg_uuxuux"])
def test_public_generation_and_cli_refuse_lc_and_nlc_over_blocks_not_exact_in_nc(
    loop_sm_colour_model, loop_native, monkeypatch, tmp_path, name
):
    """Identical-flavour quark lines compiled without the grade split,
    through ``Generator.generate`` (the typed error itself) and ``pyamplicol
    generate`` (exit code 2, the rule and full colour named), at LC and NLC,
    with post-build validation off: nothing is written. The dry run lists
    the refusal."""

    from pyamplicol.api import Generator, ProcessRequest, ProcessSet
    from pyamplicol.cli import run_cli
    from pyamplicol.config import resolve_config
    from pyamplicol.loops.v2.refusals import LOOP_ADMISSION_REFUSALS

    _without_grade_split(monkeypatch)
    expression = MIXED[name][0]
    processes = ProcessSet((ProcessRequest.parse(expression, name=name),))
    for accuracy in ("lc", "nlc"):
        output = tmp_path / f"api-{accuracy}"
        card = tmp_path / f"api-{accuracy}.toml"
        card.write_text(_card(loop_sm_colour_model, output, name, expression, accuracy))
        with pytest.raises(LoopRequestUnsupported) as refused:
            Generator(resolve_config(card)).generate(processes, output)
        assert refused.value.key == NC_EXACT_REFUSAL
        # The public route names the process in the model's spelling.
        assert f"{accuracy.upper()} needs N_c-exact tree currents" in str(refused.value)
        assert not output.exists()
        output = tmp_path / f"cli-{accuracy}"
        card = tmp_path / f"cli-{accuracy}.toml"
        card.write_text(_card(loop_sm_colour_model, output, name, expression, accuracy))
        stderr = io.StringIO()
        code = run_cli(
            ("generate", "--card", str(card), "--progress", "off"),
            stdout=io.StringIO(),
            stderr=stderr,
        )
        assert code == 2
        text = stderr.getvalue()
        assert f"{NC_EXACT_REFUSAL}: '" in text
        assert f"{accuracy.upper()} needs N_c-exact tree currents" in text
        assert 'full colour (color.accuracy = "full")' in text
        assert not output.exists()
    stdout = io.StringIO()
    code = run_cli(
        ("generate", "--card", str(card), "--dry-run", "--json", "--progress", "off"),
        stdout=stdout,
        stderr=io.StringIO(),
    )
    assert code == 0
    coverage = json.loads(stdout.getvalue())["estimated_coverage"]
    assert [item.split(":")[0] for item in coverage["loop_admission_refusals"]] == [
        refusal.key for refusal in LOOP_ADMISSION_REFUSALS
    ]


# ------------------------------------- G2-m1: fermion-line representation


@pytest.fixture(scope="module")
def leptoquark(loop_sm_ufo, tmp_path_factory):
    """loop_sm with one scalar leptoquark and its u e vertices
    (``loop_v2_ufo_variants.leptoquark_ufo``), with loop_sm's checked
    scalar-arity view (as ``loop_sm_compiled_model``)."""

    from dataclasses import replace

    from pyamplicol.models.loading import compile_model_source, load_compiled_model
    from tools.loop_reference.model_compatibility import (
        prepare_loop_sm_scalar_declarations,
    )

    from .loop_v2_ufo_variants import leptoquark_ufo

    root = tmp_path_factory.mktemp("leptoquark")
    source = leptoquark_ufo(loop_sm_ufo, root / "loop_sm_lq")
    model = compile_model_source(
        source, cache_dir=root / "cache", require_supported=False
    )
    model = replace(
        model,
        raw_loop_model=prepare_loop_sm_scalar_declarations(
            model.raw_loop_model
        ).derived,
        _serialized_path=None,
        _prepared_bundle=None,
    )
    return load_compiled_model(model.write(root / "loop-sm-lq.json"))


EW_BORN = {"QCD": 0, "QED": 2}


@pytest.mark.loop_physics
@pytest.mark.parametrize("native_dag", [True, False], ids=["native-dag", "python-dag"])
def test_a_fermion_line_changing_representation_is_refused_before_the_trees(
    leptoquark, loop_native, monkeypatch, native_dag
):
    """u e- > u e- [QCD] (leptoquark exchange): refused before any tree is
    built, naming the vertex and the rule, in both DAG lanes. A request that
    cannot use the vertex (QCD Born orders bound QED to 0; a process none of
    whose trees or loops reaches it) is not refused by it."""

    from pyamplicol.loops.v2 import assemble, dag_input
    from pyamplicol.loops.v2.assemble import LoopSlice
    from pyamplicol.loops.v2.refusals import representation_changing_vertices
    from pyamplicol.processes.model import build_model_process_ir

    monkeypatch.setattr(dag_input, "NATIVE_DAG", native_dag)
    monkeypatch.setattr(assemble, "build_tree_blocks", _stop("the trees"))
    request = _request("u e- > u e-", EW_BORN, "full")
    with pytest.raises(LoopRequestUnsupported) as refused:
        LoopSlice.generate(leptoquark, request, loop_native)
    assert refused.value.key == LINE_REFUSAL
    message = str(refused.value)
    assert message.startswith(
        f"{LINE_REFUSAL}: 'u e- > u e-': the vertex V_LQ (u~ e- lq), "
        "V_LQc (e+ u lq~) joins"
    )
    assert "joins a coloured and a colour-singlet fermion" in message
    assert "particle_veto" in message
    # Vetoing the leptoquark removes the vertex from every tree and loop.
    with pytest.raises(AssertionError, match="before the trees"):
        LoopSlice.generate(
            leptoquark,
            _request("u e- > u e-", EW_BORN, "full", particle_veto=(9000005,)),
            loop_native,
        )

    def found(process, born):
        ir = build_model_process_ir(process, leptoquark.ir, color_accuracy="full")
        loops = {**born, "QCD": born["QCD"] + 2}
        return representation_changing_vertices(
            leptoquark,
            ir,
            order_limits=(born, loops),
            loop_species=(21, 1, -1, 2, -2, 9000005, -9000005),
        )

    assert found("u u~ > e+ e-", EW_BORN)  # t-channel leptoquark
    # Bounded to QED^0 the vertex is not admitted.
    ir = build_model_process_ir("u e- > u e-", leptoquark.ir, color_accuracy="full")
    assert (
        representation_changing_vertices(
            leptoquark, ir, order_limits=({"QCD": 2, "QED": 0},)
        )
        == ()
    )
    # m+ m- reaches no electron current from its legs: no tree or loop of
    # it can use the vertex.
    assert found("u u~ > m+ m-", EW_BORN) == ()


@pytest.mark.loop_physics
def test_the_native_dag_refuses_the_same_line_with_the_listed_key(
    leptoquark, loop_native, monkeypatch
):
    """With the admission scan bypassed, the native DAG builder refuses the
    representation-changing merge as ``DagError::Unsupported`` with the listed
    key (``FERMION_LINE_REPRESENTATION``), which ``dag_input`` raises as the
    typed refusal naming the vertex's particles, not an untyped error."""

    from pyamplicol.loops.v2 import assemble, dag_input
    from pyamplicol.loops.v2.assemble import LoopSlice

    monkeypatch.setattr(dag_input, "NATIVE_DAG", True)
    monkeypatch.setattr(
        assemble, "require_colour_preserving_fermion_lines", lambda *a, **k: None
    )
    with pytest.raises(LoopRequestUnsupported) as refused:
        LoopSlice.generate(
            leptoquark, _request("u e- > u e-", EW_BORN, "full"), loop_native
        )
    assert refused.value.key == LINE_REFUSAL
    assert "a merge of a coloured and a colour-singlet fermion current" in str(
        refused.value
    )
    assert "particles [" in str(refused.value)


@pytest.mark.loop_physics
def test_standard_model_requests_use_no_representation_changing_vertex(loop_sm):
    """loop_sm has quark and lepton lines but no vertex joining them."""

    from pyamplicol.loops.v2.refusals import representation_changing_vertices
    from pyamplicol.processes.model import build_model_process_ir

    for process, born in (
        ("u u~ > e+ e- g", {"QCD": 1, "QED": 2}),
        ("u d~ > e+ ve g", {"QCD": 1, "QED": 2}),
        ("e+ e- > u u~ g", {"QCD": 1, "QED": 2}),
    ):
        ir = build_model_process_ir(process, loop_sm.ir, color_accuracy="full")
        assert (
            representation_changing_vertices(
                loop_sm, ir, order_limits=(born, {"QCD": 3, "QED": 2})
            )
            == ()
        )
