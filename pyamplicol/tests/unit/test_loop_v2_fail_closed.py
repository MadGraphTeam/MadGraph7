# SPDX-License-Identifier: 0BSD
"""Fail-closed generation of loop v2 (package fail-closed; second genericity
audit, ``workspace/reports/engine-v2/genericity-audit-2/REPORT.md``).

An unsupported structure is refused with a typed error naming a listed rule,
never served as a silently wrong value, and the rule is structural:

* G2-M1 (``refusals.LOOP_ADMISSION_REFUSALS`` key
  ``colour-accuracy-nc-exact``): LC and NLC over tree blocks that are not exact
  in N_c (identical-flavour quark lines, whose blocks mix U(1)-subtraction
  grades) are refused at generation, through ``LoopSlice.generate``, the public
  Python API and the CLI, before any family or export, and every export
  refuses them again, so no stored plan, native runtime or SDK serves a
  grade-0 projection. Identical flavours with N_c-exact blocks (u u~ > u u~)
  and full colour are not refused: the criterion is N_c-exactness, not the
  flavours.
* G2-M4 interim (``native_trees.NATIVE_EXPORT_REFUSALS`` key
  ``plane-mismatch``): the SymJIT plane check's disagreement is a listed
  refusal naming the kernel or the parameter-program output.
* G2-m4 (``external-photon-alpha0``): ``test_loop_v2_ew_policy.py`` (the
  alpha(0) model); the export's own refusal is checked here.
* G2-m1 (``fermion-line-representation``): a toy leptoquark UFO (loop_sm with
  one triplet scalar and its u e vertex, ``loop_v2_ufo_variants``) is refused
  before any tree is built, naming the vertex, whichever DAG lane is on; with
  the admission scan bypassed the native DAG refuses with the listed key
  (``rusticol_core::loop_v2::dag::FERMION_LINE_REPRESENTATION``).

The dry run lists both tables. Plans of every request the guards do not
refuse are unchanged (the guards on and off give the same bytes).
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest

from pyamplicol.loops.v2.refusals import LoopRequestUnsupported

ROOT = Path(__file__).resolve().parents[2]
FIXED = {"MB": "0"}
# Identical-flavour quark lines (blocks mixing U(1)-subtraction grades) and
# a relabelled copy (another flavour, permuted legs).
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
    assert {"plane-mismatch", "external-photon-alpha0"} <= {
        refusal.key for refusal in NATIVE_EXPORT_REFUSALS
    }
    for key in ("plane-mismatch", "external-photon-alpha0"):
        assert NativeExportUnsupported(key, "x").feature == key


@pytest.mark.loop_fast
def test_export_refuses_external_photons_with_the_listed_key():
    """The export's own refusal (the generation refuses a stored process
    earlier, ``test_loop_v2_ew_policy.py``): a listed feature, no bare
    NotImplementedError."""

    from types import SimpleNamespace

    from pyamplicol.loops.v2.native_trees import NativeExportUnsupported
    from pyamplicol.loops.v2.plan_export import export_plan

    slice_ = SimpleNamespace(
        blocks=SimpleNamespace(blocks=()),
        request=SimpleNamespace(accuracy="full", process="u u~ > a g"),
        external_photons=object(),
    )
    with pytest.raises(NativeExportUnsupported) as refused:
        export_plan(slice_, forward=False, trees=True)
    assert refused.value.feature == "external-photon-alpha0"


@pytest.mark.loop_fast
def test_plane_mismatch_is_a_listed_refusal_naming_the_output(loop_native):
    """A square-root argument on its branch cut at the plane check's real
    probes (x = 1.118 > y = 0.736 in row 0: SymJIT and Symbolica give the
    negated real opposite signed zeros) is refused as ``plane-mismatch``,
    naming the parameter-program output by its key, or the kernel."""

    from symbolica import E, Expression, S

    from pyamplicol.loops.v2 import native_trees

    state = bytes(
        Expression.evaluator_multiple(
            [E("x + y"), E("sqrt(1 - x^2/y^2)")], [S("x"), S("y")], jit_compile=False
        ).save()
    )
    with pytest.raises(native_trees.NativeExportUnsupported) as refused:
        native_trees._plane_applications(
            (state,), (("model", "MWx"), ("coupling", "GC_W", 0))
        )
    assert refused.value.feature == "plane-mismatch"
    message = str(refused.value)
    assert "parameter-program output 1 (coupling GC_W 0) at probe row 0" in message
    assert "imaginary part" in message
    assert "keep this program on the exact evaluator" not in message
    with pytest.raises(native_trees.NativeExportUnsupported) as refused:
        native_trees._plane_applications((state,))
    assert "output 1 of tree-program kernel 0" in str(refused.value)
    # A program off the cut compiles as before.
    off = bytes(
        Expression.evaluator_multiple(
            [E("sqrt(1 + x^2/y^2)")], [S("x"), S("y")], jit_compile=False
        ).save()
    )
    assert native_trees._plane_applications((off,), (("model", "a"),))


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
    """LC and NLC of identical-flavour quark lines: the typed refusal names
    the rule and full colour, raised right after the trees (before their
    kernels, the families and any export)."""

    from pyamplicol.loops.v2 import assemble
    from pyamplicol.loops.v2.assemble import LoopSlice

    process, born = MIXED[name]
    monkeypatch.setattr(assemble, "with_ward_kernels", _stop("the tree kernels"))
    with pytest.raises(LoopRequestUnsupported) as refused:
        LoopSlice.generate(loop_sm, _request(process, born, accuracy), loop_native)
    assert refused.value.key == NC_EXACT_REFUSAL
    message = str(refused.value)
    assert f"'{process}': {accuracy.upper()} needs N_c-exact tree currents" in message
    assert "mix U(1)-subtraction grades (words (" in message
    assert 'full colour (color.accuracy = "full")' in message


@pytest.mark.loop_physics
def test_identical_flavours_with_nc_exact_blocks_and_full_colour_generate(
    loop_sm, loop_native
):
    """The guard is keyed on N_c exactness, not on flavours: u u~ > u u~ has
    identical flavours and N_c-exact blocks (LC generates), and full colour of
    u u~ > u u~ g keeps every power at N_c = 3 (generates)."""

    from pyamplicol.loops.v2.assemble import LoopSlice

    lc = LoopSlice.generate(
        loop_sm, _request("u u~ > u u~", {"QCD": 2, "QED": 0}, "lc"), loop_native
    )
    assert lc.colour.formal_nc and lc.table.formal_nc
    full = LoopSlice.generate(
        loop_sm, _request(*MIXED["uux_uuxg"], "full"), loop_native
    )
    assert not full.colour.formal_nc


@pytest.mark.loop_physics
def test_exports_refuse_projections_over_blocks_not_exact_in_nc(
    loop_sm, loop_native, monkeypatch
):
    """Defence in depth: an LC slice of u u~ > u u~ g built with the
    generation guard bypassed is refused by every export layer in turn
    (``export_plan``, the seed records, the native trees) and by the native
    seed records of column sets, so no stored plan carries its grade-0
    projection."""

    from pyamplicol.loops.v2 import assemble, native_cells, plan_export, refusals
    from pyamplicol.loops.v2.assemble import LoopSlice

    with monkeypatch.context() as patch:
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
    # The seed-table check off too: the native trees refuse (their call is
    # the second).
    calls.clear()
    monkeypatch.setattr(plan_export, "require_formal_nc_tables", lambda *a: None)
    assert "'u u~ > u u~ g': LC needs N_c-exact" in refused()
    assert len(calls) == 2
    # The native seed records of a column set check their tables.
    with pytest.raises(LoopRequestUnsupported, match="native seed records"):
        native_cells.project_records(slice_, plan_export.sector_columns(slice_), (), {})


def _plan_digests(process) -> tuple[str, ...]:
    return tuple(
        hashlib.sha256(text.encode()).hexdigest()
        for text in (process.plan_json, process.scaleless_plan_json or "")
    )


@pytest.mark.loop_physics
def test_guards_leave_the_plans_they_do_not_refuse_unchanged(
    loop_sm, loop_native, monkeypatch
):
    """u u~ > d d~ at NLC (distinct flavours, N_c-exact blocks): the stored
    plan and companion plan bytes are the same with every guard of the
    package on and off (no guard has a side effect on what it admits). The
    package's comparison against the slice before it (LC/NLC of u u~ > d d~ g
    and u u~ > d d~ s s~, full colour of the identical-flavour controls) is
    in its report."""

    from pyamplicol.loops.v2 import assemble, plan_export, refusals
    from pyamplicol.loops.v2.artifact import generate_process

    request = _request("u u~ > d d~", {"QCD": 2, "QED": 0}, "nlc")
    guarded = _plan_digests(generate_process(loop_sm, request, loop_native))
    off = lambda *args, **kwargs: None  # noqa: E731
    for name in ("require_nc_exact", "require_colour_preserving_fermion_lines"):
        monkeypatch.setattr(assemble, name, off)
    for name in ("require_nc_exact", "require_formal_nc_tables"):
        monkeypatch.setattr(refusals, name, off)
    monkeypatch.setattr(plan_export, "require_formal_nc_tables", off)
    assert _plan_digests(generate_process(loop_sm, request, loop_native)) == guarded


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
engine = "v2"
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
def test_public_generation_and_cli_refuse_lc_and_nlc_of_identical_flavours(
    loop_sm_colour_model, loop_native, tmp_path, name
):
    """Through ``Generator.generate`` (the typed error itself) and
    ``pyamplicol generate`` (exit code 2, the rule and full colour named), at
    LC and NLC, with post-build validation off: nothing is written. The dry
    run lists the refusal."""

    from pyamplicol.api import Generator, ProcessRequest, ProcessSet
    from pyamplicol.cli import run_cli
    from pyamplicol.config import resolve_config
    from pyamplicol.loops.v2.refusals import LOOP_ADMISSION_REFUSALS

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
        assert f"{accuracy.upper()} needs N_c-exact tree currents" in str(
            refused.value
        )
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
