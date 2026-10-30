# SPDX-License-Identifier: 0BSD
"""Every refusal a one-loop request reaches has a listed key, visible in the
public dry run (m-sm PLAN MS-MODEL item 4).

* Configuration refusals (``refusals.LOOP_CONFIGURATION_REFUSALS``): run-card
  settings and loop-model declarations v2 does not take; the dry run shows
  each as its listed text (``key: message. Rule: ... Instead: ...``) and
  generation raises ``LoopRequestUnsupported`` with the key.
* Admission refusals that need only the model and the Born DAG
  (``external-photon-scheme``, ``feynman-gauge``, ``loop-induced``,
  ``colour-declaration-coverage``, ...) fire in the dry run too.
* Internal refusals (``refusals.LOOP_INTERNAL_REFUSALS``) are engine
  conditions met during generation; their error classes carry the key, which
  the generation boundary keeps (``refusals.keyed_error``), and the dry run
  lists them. Native export refusals surface typed (``NativeExportRefused``).
* Multiparticle requests (m-sm PLAN Q2 (a)): a member without a Born is
  skipped with a typed ``loop-induced`` record (dry run ``skipped_members``,
  artifact extension ``loop_skipped_members``); a request with no member
  left is refused.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

ZERO_WIDTHS = {"WT": "0", "WZ": "0", "WW": "0", "WH": "0"}

LOOP_SM_DECLARATION = """
[loop.model]
loop_coupling_orders = {orders}
uv_complete = {uv}
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
    output: Path,
    expression: str,
    *,
    model: str = "loop_sm",
    orders=("QCD",),
    restriction: str | None = None,
    fixed: dict[str, str] | None = None,
    process: str = "",
    colour: str = "",
    loop: str = "",
    tables: str = "",
    generation: str = "",
) -> str:
    fixed = ZERO_WIDTHS if fixed is None else fixed
    parameters = ", ".join(f"{name} = {json.dumps(v)}" for name, v in fixed.items())
    restricted = (
        "" if restriction is None else f"restriction = {json.dumps(restriction)}"
    )
    return f"""
action = "generate"
[model]
source = {json.dumps(model)}
{restricted}
[process]
entries = [{{expression = {json.dumps(expression)}, name = "p"}}]
loop_coupling_orders = {json.dumps(list(orders))}
coupling_order_policy = "minimal"
{process}
[color]
{colour}
[loop]
r2_strategy = "model"
{loop}
[loop.preparation]
fixed_parameters = {{{parameters}}}
{tables}
[generation]
output = {json.dumps(str(output))}
emit_api_bundle = false
{generation}
[generation.validation]
post_build_validation = false
"""


def _generator(text: str, path: Path):
    from pyamplicol.api import Generator
    from pyamplicol.config import resolve_config

    path.write_text(text)
    return Generator(resolve_config(path))


def _processes(expression: str, aliases=()):
    from pyamplicol.api import ProcessRequest, ProcessSet

    return ProcessSet((ProcessRequest.parse(expression, name="p"),), aliases)


def _compile(source: str, restriction: str | None = None):
    from pyamplicol.models.loading import compile_model_source

    return compile_model_source(source, restriction=restriction or "default")


def _keys(plan, listing: str) -> list[str]:
    return [item.split(":")[0] for item in plan.estimated_coverage[listing]]


def _single_blocker(plan, key: str) -> str:
    (blocker,) = plan.unsupported_features
    assert blocker.startswith(f"{key}: "), blocker
    return blocker


# ------------------------------------------------------- configuration keys


def _aliases():
    from pyamplicol.api.requests import ProcessAlias

    return (ProcessAlias("q", "p", (1, 0, 2, 3)),)


# key -> card settings (table text by section) and process-set aliases.
CONFIGURATION = {
    "fft-basis-trace": {"colour": 'contraction = "auto"\nfft_basis = "trace"'},
    "colour-selectors": {"process": "max_color_sectors = 1"},
    "process-aliases": {"aliases": True},
    "process-group-analysis": {"process": 'analyze_process_groups = "numerical"'},
    "direct-r2": {"loop": 'r2_strategy = "direct"'},
    "relation-discovery": {
        "generation": '[generation.relation_discovery]\nmode = "diagnostic"'
    },
    "evaluator-backend": {"tables": '[evaluator]\nbackend = "cpp"'},
    "evaluator-tuning": {"tables": "[evaluator.jit]\noptimization_level = 1"},
    "covariant-multiplets": {"tables": "covariant_multiplets = {25 = [250]}"},
}


@pytest.mark.loop_fast
@pytest.mark.parametrize("key", sorted(CONFIGURATION))
def test_configuration_refusals_fire_in_the_dry_run_and_at_generation(tmp_path, key):
    """Each configuration refusal of ``_feature_blockers`` / ``_v2_blockers``
    is its listed key in the dry run (decided before any model is loaded)
    and the typed ``LoopRequestUnsupported`` of generation."""

    from pyamplicol.api.errors import GenerationError
    from pyamplicol.loops.v2.refusals import (
        LoopRequestUnsupported,
        listed_configuration_keys,
    )

    settings = dict(CONFIGURATION[key])
    aliases = _aliases() if settings.pop("aliases", False) else ()
    text = _card(tmp_path / "out", "u u~ > e- e+", **settings)
    if "r2_strategy" in settings.get("loop", ""):
        # r2_strategy is set once, in [loop].
        text = text.replace('r2_strategy = "model"\n', "", 1)
    generator = _generator(text, tmp_path / "card.toml")
    plan = generator.plan(_processes("u u~ > e- e+", aliases))
    features = [f for f in plan.unsupported_features if f.startswith(f"{key}: ")]
    assert features, plan.unsupported_features
    assert key in listed_configuration_keys()
    assert _keys(plan, "loop_configuration_refusals") == list(
        listed_configuration_keys()
    )
    with pytest.raises(GenerationError) as refused:
        generator.generate(_processes("u u~ > e- e+", aliases), tmp_path / "out")
    assert isinstance(refused.value, LoopRequestUnsupported)
    assert refused.value.key == plan.unsupported_features[0].split(":")[0]
    assert not (tmp_path / "out").exists()


@pytest.mark.loop_physics
@pytest.mark.parametrize(
    "key,expression,settings",
    [
        ("loop-model-corrections", "u u~ > e- e+", {"orders": ("QED",)}),
        (
            "loop-model-completion",
            "u u~ > e- e+",
            {"loop": LOOP_SM_DECLARATION.format(orders='["QCD"]', uv="false")},
        ),
        (
            "undeclared-corrections",
            "u u~ > e- e+",
            {
                "orders": ("QED",),
                "loop": LOOP_SM_DECLARATION.format(orders='["QCD", "QED"]', uv="true"),
            },
        ),
        ("scale-parameter", "u u~ > e- e+", {"fixed": {**ZERO_WIDTHS, "MU_R": "91"}}),
        (
            "veto-external-particle",
            "u u~ > e- e+",
            {"generation": 'particle_veto = ["e-"]'},
        ),
        # Every member of p p > h h is loop-induced (m-sm Q2 (a)).
        ("loop-induced", "p p > h h", {"restriction": "no_b_mass"}),
    ],
)
def test_loop_sm_request_refusals_fire_in_the_dry_run(
    tmp_path, key, expression, settings
):
    """Model-level configuration refusals on the shipped loop_sm (no
    [loop.model] unless stated): the dry run's single feature is the listed
    key."""

    _compile("loop_sm", settings.get("restriction"))
    plan = _generator(
        _card(tmp_path / "out", expression, **settings), tmp_path / "card.toml"
    ).plan(_processes(expression))
    _single_blocker(plan, key)


def _modified(compiled, *, raw=None, metadata=None, tree_only=False):
    """The public handle of ``compiled`` with its raw NLO source changed
    (``raw``, ``metadata``) or removed (``tree_only``)."""

    from pyamplicol.api.models import _compiled_model_from_payload

    raw_model = compiled.raw_loop_model
    if raw is not None:
        raw_model = raw(raw_model)
    if metadata is not None:
        raw_model = replace(raw_model, metadata={**raw_model.metadata, **metadata})
    return _compiled_model_from_payload(
        replace(
            compiled,
            raw_loop_model=None if tree_only else raw_model,
            _serialized_path=None,
            _prepared_bundle=None,
        )
    )


def _retexed(raw):
    """loop_sm's declarations with one parameter's TeX name changed: another
    declarations digest, so no shipped profile applies."""

    collections = dict(raw.collections)
    first, *rest = collections["parameters"]
    first = {**first, "fields": {**first["fields"], "texname": "zero"}}
    collections["parameters"] = (first, *rest)
    return replace(raw, collections=collections)


def _unpaired(raw):
    """loop_sm with the W-'s width changed (its antiparticle W+ keeps WW):
    a particle-antiparticle metadata mismatch."""

    collections = dict(raw.collections)
    particles = list(collections["particles"])
    (z_width,) = [r["fields"]["width"] for r in particles if r["name"] == "Z"]
    for index, record in enumerate(particles):
        if record["name"] == "W-":
            fields = {**record["fields"], "width": z_width}
            particles[index] = {**record, "fields": fields}
    collections["particles"] = tuple(particles)
    return replace(raw, collections=collections)


@pytest.mark.loop_physics
@pytest.mark.parametrize(
    "key,change",
    [
        ("loop-model-declaration", {"raw": _retexed}),
        ("feynman-gauge", {"metadata": {"gauge": [0]}}),
        ("tree-only-model", {"tree_only": True}),
        ("antiparticle-metadata", {"raw": _unpaired}),
    ],
)
def test_model_refusals_fire_in_the_dry_run(tmp_path, key, change):
    """A modified copy of loop_sm passed to the dry run: another
    declarations digest (no shipped profile: ``loop-model-declaration``), a
    source without gauge 1 (``feynman-gauge``, model preparation), a
    compiled model without its NLO sidecar (``tree-only-model``)."""

    from pyamplicol.loops.shipped_profiles import shipped_loop_profile

    compiled = _compile("loop_sm")
    model = _modified(compiled, **change)
    if key == "loop-model-declaration":
        assert shipped_loop_profile(_retexed(compiled.raw_loop_model)) is None
    # Another declarations digest needs the declaration in the card.
    loop = (
        LOOP_SM_DECLARATION.format(orders='["QCD"]', uv="true")
        if key == "antiparticle-metadata"
        else ""
    )
    plan = _generator(
        _card(tmp_path / "out", "u u~ > e- e+", loop=loop), tmp_path / "card.toml"
    ).plan(_processes("u u~ > e- e+"), model=model)
    _single_blocker(plan, key)


@pytest.mark.loop_physics
@pytest.mark.parametrize(
    "key,model,expression,orders,settings",
    [
        # loop_qcd_qed_sm has no alpha(0) input (MG5_aMC refuses tagged
        # photons there too): a final-state photon is tagged by default; the
        # remedy names G_mu with restriction a0 (m-sm MS-PHOTON).
        ("external-photon-scheme", "loop_qcd_qed_sm", "u u~ > a g", ("QCD", "QED"), {}),
        # A tag list naming a leg that is no on-shell photon (leg 4: the
        # gluon; m-sm MS-PHOTON's photon_tags, which the dry run calls).
        (
            "external-photon-legs",
            "loop_qcd_qed_sm",
            "u u~ > a g",
            ("QCD", "QED"),
            {"loop": "external_photons = [4]"},
        ),
        # The shipped N_c data are scheme-gated (MQ-NC): none covers the
        # QCD+QED models.
        (
            "colour-declaration-coverage",
            "loop_qcd_qed_sm",
            "u u~ > d d~",
            ("QCD", "QED"),
            {"colour": 'accuracy = "lc"'},
        ),
        # The card's Z width in the Born currents (the width policy, called
        # by the dry run on the Born DAG; m-sm MS-CMS owns the policy).
        (
            "electroweak-width",
            "loop_qcd_qed_sm_Gmu",
            "e- e+ > m- m+",
            ("QED",),
            {"fixed": {}},
        ),
    ],
)
def test_electroweak_refusals_fire_in_the_dry_run(
    tmp_path, key, model, expression, orders, settings
):
    from pyamplicol.loops.v2.ew_policy import ElectroweakWidthError  # noqa: F401

    _compile(model)
    plan = _generator(
        _card(tmp_path / "out", expression, model=model, orders=orders, **settings),
        tmp_path / "card.toml",
    ).plan(_processes(expression))
    blocker = _single_blocker(plan, key)
    if key == "electroweak-width":
        assert "whose tree currents propagate particles with a width" in blocker
        assert "WZ" in blocker
    if key == "external-photon-scheme":
        assert "restriction a0" in blocker


@pytest.mark.loop_physics
def test_multiparticle_members_without_a_born_are_skipped_in_the_dry_run(tmp_path):
    """p p > e+ e- on loop_sm (QCD): g g > e+ e- has no Born amplitude; it is
    skipped with its typed record, the quark members are planned."""

    _compile("loop_sm", "no_b_mass")
    plan = _generator(
        _card(tmp_path / "out", "p p > e+ e-", restriction="no_b_mass"),
        tmp_path / "card.toml",
    ).plan(_processes("p p > e+ e-"))
    assert plan.unsupported_features == ()
    (skipped,) = plan.estimated_coverage["skipped_members"]
    assert skipped["refusal"] == "loop-induced"
    assert skipped["request"] == "p"
    assert skipped["process"].casefold() == "g g > e+ e-"
    assert skipped["reason"].startswith("loop-induced: ")
    planned = [record["process"] for record in plan.estimated_coverage["processes"]]
    assert len(planned) == 5 and all(" > e+ e-" in p for p in planned), planned
    assert skipped["process"] not in planned


# ---------------------------------------------------------- internal keys


@pytest.mark.loop_fast
def test_internal_and_engine_errors_keep_their_keys():
    """The engine's error classes carry their listed key
    (``refusal_key``); the generation boundary (``keyed_error``) types them,
    and a native export refusal keeps its feature key."""

    from pyamplicol.api.errors import GenerationError
    from pyamplicol.loops.v2.artifact import _not_storable
    from pyamplicol.loops.v2.catalogue import LoopKernelCatalogueError
    from pyamplicol.loops.v2.conventions import FermionSignError, ReadingSignError
    from pyamplicol.loops.v2.ct_recursion import (
        AnomalousVetoError,
        CountertermCouplingError,
        UnloweredCountertermError,
    )
    from pyamplicol.loops.v2.ct_sites import CountertermCompletenessError
    from pyamplicol.loops.v2.native_trees import (
        NATIVE_EXPORT_REFUSALS,
        NativeExportUnsupported,
    )
    from pyamplicol.loops.v2.refusals import (
        LoopEngineRefusal,
        LoopRequestUnsupported,
        NativeExportRefused,
        keyed_error,
        listed_internal_keys,
        listed_refusal_keys,
    )

    internal = {
        FermionSignError: "fermion-sign",
        ReadingSignError: "reading-sign",
        UnloweredCountertermError: "unlowered-counterterm",
        CountertermCouplingError: "counterterm-coupling",
        LoopKernelCatalogueError: "loop-kernel-catalogue",
        CountertermCompletenessError: "counterterm-completeness",
    }
    assert sorted(internal.values()) == sorted(listed_internal_keys())
    for cls, key in internal.items():
        typed = keyed_error(cls("the engine's message"))
        assert isinstance(typed, LoopEngineRefusal) and isinstance(
            typed, GenerationError
        )
        assert typed.key == key
        assert str(typed).startswith(f"{key}: the engine's message")
    typed = keyed_error(AnomalousVetoError("a doublet"))
    assert isinstance(typed, LoopRequestUnsupported)
    assert typed.key == "anomalous-veto" and "anomalous-veto" in listed_refusal_keys()
    plain = ValueError("unrelated")
    assert keyed_error(plain) is plain
    feature = NATIVE_EXPORT_REFUSALS[0].key
    refused = _not_storable(
        type("R", (), {"process": "u u~ > d d~"})(),
        "native point inputs",
        NativeExportUnsupported(feature, "a listed feature"),
    )
    assert isinstance(refused, NativeExportRefused)
    assert refused.key == feature and str(refused).startswith(f"{feature}: ")


@pytest.mark.loop_fast
def test_typed_electroweak_policy_classes_carry_their_keys():
    from pyamplicol.loops.v2.ew_policy import (
        ElectroweakGaugeError,
        ExternalPhotonError,
    )
    from pyamplicol.loops.v2.refusals import LoopRequestUnsupported

    for error, key in (
        # m-sm MS-PHOTON's class takes its key at each raise site.
        (
            ExternalPhotonError("external-photon-scheme", "why"),
            "external-photon-scheme",
        ),
        (ExternalPhotonError("external-photon-legs", "why"), "external-photon-legs"),
        (ElectroweakGaugeError("why"), "feynman-gauge"),
    ):
        assert isinstance(error, LoopRequestUnsupported) and error.key == key
        assert str(error).startswith(f"{key}: why. Rule: ")


@pytest.mark.loop_physics
def test_multiparticle_artifact_records_its_skipped_members(loop_native, tmp_path):
    """The public generation of p p > e+ e- (loop_sm, QCD) stores the members
    with a Born, records g g > e+ e- in the artifact's
    ``loop_skipped_members`` extension, and a stored member evaluates as its
    own single-process generation (bitwise in binary64)."""

    from pyamplicol.api import Runtime
    from pyamplicol.artifacts import load_manifest
    from pyamplicol.generation.loop_service import SKIPPED_MEMBERS_EXTENSION

    _compile("loop_sm", "no_b_mass")
    output = tmp_path / "multi"
    result = _generator(
        _card(output, "p p > e+ e-", restriction="no_b_mass"),
        tmp_path / "multi.toml",
    ).generate(_processes("p p > e+ e-"), output)
    stored = [request.expression for request in result.processes.requests]
    assert len(stored) == 5 and not any(e.casefold().startswith("g g") for e in stored)
    manifest = load_manifest(output)
    record = manifest.extensions[SKIPPED_MEMBERS_EXTENSION]
    (skipped,) = record["members"]
    assert skipped["refusal"] == "loop-induced"
    assert skipped["process"].casefold() == "g g > e+ e-"
    member = next(
        request
        for request in result.processes.requests
        if request.expression.casefold().startswith("u u~")
    )
    single = tmp_path / "single"
    _generator(
        _card(single, member.expression, restriction="no_b_mass"),
        tmp_path / "single.toml",
    ).generate(_processes(member.expression), single)
    point = [
        [500.0, 0.0, 0.0, 500.0],
        [500.0, 0.0, 0.0, -500.0],
        [500.0, 300.0, 200.0, 346.41016151377545],
        [500.0, -300.0, -200.0, -346.41016151377545],
    ]
    values = [
        Runtime.load(path, process=name).evaluate([point], mu_r2=[8315.0])
        for path, name in ((output, member.name), (single, "p"))
    ]
    assert values[0].born_values == values[1].born_values
    assert values[0].values == values[1].values


# ------------------------------------------------- plan export: trace steps


@pytest.mark.loop_fast
def test_trace_steps_stay_within_the_family_rank():
    """A trace step's output rank is its largest source storage plus its
    largest content degree, which the plan loader requires within the family
    rank. Two rings of a family that peak at different positions (a bubble
    with degree 1 at either propagator: rank 1) split into steps by source
    storage; a step within the rank stays one step (plans the loader admits
    are exported as before). u u~ > z z [virt=QED] on loop_qcd_qed_sm_Gmu
    met it (its plan was refused at load: "a trace step above the family
    rank")."""

    from types import SimpleNamespace

    from pyamplicol.loops.v2.plan_export import _trace_groups

    degrees = {"a": 0, "b": 1, "c": 1}

    def shape(content):
        return SimpleNamespace(degree=degrees[content])

    def edge(content, storage):
        return content, SimpleNamespace(storage=storage)

    bubble = [edge("a", 1), edge("b", 0), edge("c", 0), edge("a", 1)]
    groups = _trace_groups(bubble, 1, shape)
    assert [[c for c, _ in g] for g in groups] == [["b", "c"], ["a", "a"]]
    assert all(
        max(s.storage for _, s in g) + max(degrees[c] for c, _ in g) <= 1
        for g in groups
    )
    assert _trace_groups(bubble, 2, shape) == [bubble]


@pytest.mark.loop_fast
def test_family_rank_bounds_every_step():
    """A family is written with the largest of its trie rank and its steps'
    output ranks (``plan_export.family_rank``; MS-VALIDATE's F1): a trie node
    joining a degree-0 and a degree-1 ring (h h > h h [virt=QED]: its
    storage-1 source and its degree-1 segment make a rank-2 step in a
    rank-1 trie) raises the rank to what the loader requires, while a family
    whose steps stay within the trie rank keeps it (its plan is written as
    before). The rows that met it load and pass on the nightly tier (SC04,
    SC06, SC07, TV03, V301, VH01-VH03, VV02 in
    ``tests/unit/test_loop_v2_msm_nightly_*.py``)."""

    from pyamplicol.loops.v2.plan_export import family_rank

    within = [{"rank_in": 0, "degree": 1}, {"rank_in": 1, "degree": 0}]
    assert family_rank(1, within) == 1
    joined = [*within, {"rank_in": 1, "degree": 1}]
    assert family_rank(1, joined) == 2
    assert all(s["rank_in"] + s["degree"] <= family_rank(1, joined) for s in joined)
    assert family_rank(3, []) == 3


@pytest.mark.loop_nightly
def test_split_trace_steps_equal_the_reference_executor(loop_native):
    """u u~ > z z [virt=QED] on the shipped loop_qcd_qed_sm_Gmu (widths
    zero): its exported plan splits the trace steps of its N = 2 families
    whose rings peak at different positions; the native schedule-B family
    numerators equal the Python reference executor's (which walks the tries
    itself) in binary64 and at 40 digits."""

    from fractions import Fraction

    from pyamplicol.loops.v2.assemble import LoopSlice, Precision, SliceRequest
    from pyamplicol.loops.v2.plan_export import NativeExecutor, compare_numerators

    compiled = _compile("loop_qcd_qed_sm_Gmu")
    request = SliceRequest(
        "u u~ > z z",
        {"QCD": 0, "QED": 2},
        {"QCD": 0, "QED": 4},
        corrections=("QED",),
        fixed_parameters=ZERO_WIDTHS,
        scale_parameter="MU_R",
    )
    slice_ = LoopSlice.generate(compiled, request, loop_native)
    executor = NativeExecutor(slice_, forward=False)
    plan = json.loads(executor._plan_json)
    ranks = [
        (
            family["rank"],
            max(s["rank_in"] + s["degree"] for s in family["steps"] if s["trace"] >= 0),
        )
        for family in plan["families"]
    ]
    assert all(out <= rank for rank, out in ranks), ranks
    energy = 1426.271508 * 2
    point = [
        ["1426.271508", "0", "0", "1426.271508"],
        ["1426.271508", "0", "0", "-1426.271508"],
        ["1426.271508", "0", "-1138.6827936", "854.0120952"],
        ["1426.271508", "0", "1138.6827936", "-854.0120952"],
    ]
    scale = str(Fraction(1000) ** 2)
    for precision, tolerance in (
        (Precision.double(), 1e-13),
        (Precision.arbitrary(40), 1e-35),
    ):
        reference, _ = slice_.family_numerators(point, scale, precision=precision)
        (native,) = executor.family_numerators([point], [scale], precision=precision)
        assert (
            compare_numerators(native, reference.numerators, energy=energy) <= tolerance
        )
