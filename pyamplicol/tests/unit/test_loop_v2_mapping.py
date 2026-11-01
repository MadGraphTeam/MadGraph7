# SPDX-License-Identifier: 0BSD
"""Processes a stored plan v4 process serves through a leg mapping (M-FINAL
MF-MAP, ``loops.v2.mapping``).

One public artifact (loop_sm, QCD, full colour, ``process.flavor_blind``)
stores two representatives, ``d d~ > g g`` and ``d g > d g``, and serves:

* two explicit permutation aliases (``ProcessAlias``: ``d~ d > g g`` and
  ``d g > g d``), published as the manifest's alias records;
* two process-group members proved at generation (``observables.
  admit_member``), each a species substitution composed with a side
  permutation (``s~ s > g g``, ``g s > s g``), and one pure side permutation
  (``g d > g d``), listed in ``process-groups.json``;
* a side-permuted selector expression (``g d > d g``).

A second one stores ``u u~ > d d~ g`` and serves an alias and a member whose
permutations are three-cycles of the final legs (not involutions, so a
permutation applied in the wrong direction fails): ``u u~ > d~ g d`` and
``u u~ > g s s~`` (d -> s proved, composed with the cycle).

Each evaluates bitwise equal to its stored representative at the permuted
momenta: the inclusive virtual, the declared colour histories read on its
own labels and relabelled onto the representative's (correlation composed
with the mapping), a spin source on a relabelled leg, and the resolved
helicity rows. A declared history or spin class whose image the
representative does not store is not served (typed refusal). One alias and
one member also equal their own direct generation (tolerance), and the
representative's plan is byte-identical to its generation alone (mappings do
not touch a stored plan). A process-group relation that is not exact
(numerical analysis) is refused at load; numerical analysis itself stays
refused at generation (``test_loop_v2_refusal_keys``).
"""

from __future__ import annotations

import json
import shutil
from decimal import Decimal
from pathlib import Path

import pytest

pytestmark = pytest.mark.loop_physics

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

PARAMETERS = {"aS": "0.118", "MT": "173"}
POINTS = [
    [
        ["500", "0", "0", "500"],
        ["500", "0", "0", "-500"],
        ["500", "300", "0", "400"],
        ["500", "-300", "0", "-400"],
    ],
    [
        ["650", "0", "0", "650"],
        ["650", "0", "0", "-650"],
        ["650", "600", "0", "250"],
        ["650", "-600", "0", "-250"],
    ],
]
SCALES = ["1000000", "1000000"]
POINTS_5 = [
    [
        ["663.0", "0.0", "0.0", "663.0"],
        ["663.0", "0.0", "0.0", "-663.0"],
        ["331.5", "-12.24", "315.18", "-102.0"],
        ["442.0", "-112.24", "127.68", "408.0"],
        ["552.5", "124.48", "-442.86", "-306.0"],
    ]
]
SPIN = {
    4: tuple(
        (Decimal(re), Decimal(im))
        for re, im in (("0.4", "1"), ("0.1", "-0.3"), ("0.8", "0.2"), ("-0.6", "0.5"))
    )
}

REQUESTS = (
    ("ddx", "d d~ > g g"),
    ("sxs", "s~ s > g g"),
    ("dg", "d g > d g"),
    ("gs", "g s > s g"),
    ("gd", "g d > g d"),
)
# (alias, target, representative leg -> alias leg)
ALIASES = (("ddx_swap", "ddx", (1, 0, 3, 2)), ("dg_out", "dg", (0, 1, 3, 2)))
# Directed histories T_i (bra) . T_j (ket), i in {1, 2}, j in {3, 4}: closed
# under the permutations used; "T12" is not closed under exchanging the
# initial legs.
HISTORIES = [
    {
        "id": f"T{i}{j}",
        "bra": [{"kind": "emit-gluon", "emitter": i, "emitted": -1}],
        "ket": [{"kind": "emit-gluon", "emitter": j, "emitted": -1}],
    }
    for i, j in ((1, 3), (1, 4), (2, 3), (2, 4), (1, 2))
]
SPIN_CLASSES = [[4]]
# The three-cycle artifact: T1j closed under the cycles of the final legs.
REQUESTS_5 = (("udg", "u u~ > d d~ g"), ("ssg", "u u~ > g s s~"))
ALIASES_5 = (("udg_cycle", "udg", (0, 1, 4, 2, 3)),)
HISTORIES_5 = [
    {
        "id": f"T1{j}",
        "bra": [{"kind": "emit-gluon", "emitter": 1, "emitted": -1}],
        "ket": [{"kind": "emit-gluon", "emitter": j, "emitted": -1}],
    }
    for j in (3, 4, 5)
]
# mapped process -> (artifact, representative, representative leg -> public
# leg, served histories, spin class (4,) served)
SERVED = ("T13", "T14", "T23", "T24")
MAPPED = {
    "ddx_swap": ("mapped", "ddx", (1, 0, 3, 2), SERVED, False),
    "dg_out": ("mapped", "dg", (0, 1, 3, 2), (*SERVED, "T12"), False),
    "sxs": ("mapped", "ddx", (1, 0, 2, 3), SERVED, True),
    "gs": ("mapped", "dg", (1, 0, 2, 3), SERVED, True),
    "gd": ("mapped", "dg", (1, 0, 3, 2), SERVED, False),
    "udg_cycle": ("cycle", "udg", (0, 1, 4, 2, 3), ("T13", "T14", "T15"), False),
    "ssg": ("cycle", "udg", (0, 1, 3, 4, 2), ("T13", "T14", "T15"), False),
}


def _card(model: Path, output: Path, entries, process: str = "") -> str:
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
accuracy = "full"
contraction = "direct"
[loop]
r2_strategy = "model"
[loop.preparation]
fixed_parameters = {{MB = "0"}}
{SCHEME}
[generation]
output = {json.dumps(str(output))}
emit_api_bundle = false
[generation.validation]
post_build_validation = true
samples = 1
"""


def _generate(
    model: Path,
    root: Path,
    entries,
    aliases=(),
    process: str = "",
    histories=HISTORIES,
    spin=SPIN_CLASSES,
):
    from pyamplicol.api import Generator, ProcessAlias, ProcessRequest, ProcessSet
    from pyamplicol.config import resolve_config
    from pyamplicol.correlators import CorrelatorConfig

    card = root / "card.toml"
    card.write_text(_card(model, root / "artifact", entries, process))
    Generator(resolve_config(card)).generate(
        ProcessSet(
            tuple(ProcessRequest.parse(e, name=n) for n, e in entries),
            tuple(ProcessAlias(*alias) for alias in aliases),
        ),
        root / "artifact",
        correlators=CorrelatorConfig.from_json_dict(
            {"color_correlations": histories, "spin_correlations": spin}
        ),
    )
    return root / "artifact"


@pytest.fixture(scope="module")
def mapped(loop_sm_compiled_model, loop_native, tmp_path_factory):
    """The mapping artifact (two representatives serving the aliases, the
    members and side permutations; generation validates every served
    process against the Python reference executor)."""

    del loop_native
    return _generate(
        loop_sm_compiled_model,
        tmp_path_factory.mktemp("mapped"),
        REQUESTS,
        ALIASES,
        "flavor_blind = true",
    )


@pytest.fixture(scope="module")
def cycle(loop_sm_compiled_model, loop_native, tmp_path_factory):
    """``u u~ > d d~ g`` serving an alias and a member whose permutations
    are three-cycles of the final legs."""

    del loop_native
    return _generate(
        loop_sm_compiled_model,
        tmp_path_factory.mktemp("cycle"),
        REQUESTS_5,
        ALIASES_5,
        "flavor_blind = true",
        HISTORIES_5,
        [],
    )


def _load(artifact: Path, process: str):
    from pyamplicol import Runtime

    return Runtime.load(artifact, process=process, model_parameters=PARAMETERS)


def _to_representative(points, permutation):
    """Public points in the representative's leg order:
    q[r] = p[permutation[r]]."""

    return [[point[p] for p in permutation] for point in points]


def _bits(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [_bits(v) for v in value]
    if isinstance(value, complex):
        return [value.real.hex(), value.imag.hex()]
    if isinstance(value, float):
        return value.hex()
    if hasattr(value, "finite"):
        return [_bits(value.finite), _bits(value.single_pole), _bits(value.double_pole)]
    return str(value)


def _values(result):
    return _bits([result.born_values, result.values])


def test_artifact_publishes_aliases_and_permuted_members(mapped):
    """The representatives alone are stored; the aliases are manifest alias
    records and the members ``process-groups.json`` members of basis
    ``exact-loop-flavor`` with their representative-to-member permutation."""

    from pyamplicol.artifacts import inspect_artifact, load_manifest
    from pyamplicol.artifacts.process_groups import read_process_groups

    manifest = load_manifest(mapped)
    records = {record["id"]: record for record in manifest.processes}
    assert sorted(records) == ["ddx", "dg"]
    aliases = {
        alias["id"]: (record["id"], tuple(alias["external_permutation"]))
        for record in records.values()
        for alias in record["aliases"]
    }
    assert aliases == {name: (target, perm) for name, target, perm in ALIASES}
    inspected = {
        alias.id: alias.representative_id
        for process in inspect_artifact(mapped).processes
        for alias in process.aliases
    }
    assert inspected == {name: target for name, target, _perm in ALIASES}
    groups = read_process_groups(mapped)
    members = {
        member.process_id: (representative, member.permutation, str(member.basis))
        for representative, group in groups.groups.items()
        for member in group.members
        if not member.generated
    }
    assert members == {
        name: (MAPPED[name][1], MAPPED[name][2], "exact-loop-flavor")
        for name in ("sxs", "gs", "gd")
    }


@pytest.mark.parametrize("name", sorted(MAPPED))
def test_mapped_process_is_its_representative_at_permuted_momenta(request, name):
    """Inclusive, colour-correlated (every served history, relabelled onto
    the representative's), spin-correlated and resolved outputs of a mapped
    process equal its representative's at the permuted momenta, bitwise."""

    from pyamplicol.correlators import CorrelatedRequest

    artifact, representative, permutation, served, spin = MAPPED[name]
    mapped = request.getfixturevalue(artifact)
    points = POINTS if artifact == "mapped" else POINTS_5
    scales = SCALES[: len(points)]
    runtime = _load(mapped, name)
    stored = _load(mapped, representative)
    assert runtime.physics.process_id == name
    assert runtime.representative_process_key == representative
    assert runtime.external_permutation == permutation
    q = _to_representative(points, permutation)
    assert _values(runtime.evaluate(points, mu_r2=scales)) == _values(
        stored.evaluate(q, mu_r2=scales)
    )
    available = [item.id for item in runtime.available_color_correlations()]
    assert available[0] == "born" and sorted(available[1:]) == sorted(served)
    declared = HISTORIES if artifact == "mapped" else HISTORIES_5
    histories = {item["id"]: item for item in declared}
    for history in served:
        # The history read on the mapped process's labels, relabelled onto
        # the representative's (public leg p is representative leg
        # permutation.index(p)).
        bra = permutation.index(histories[history]["bra"][0]["emitter"] - 1) + 1
        ket = permutation.index(histories[history]["ket"][0]["emitter"] - 1) + 1
        image = f"T{bra}{ket}"
        left = runtime.evaluate_correlated(
            points, color_correlation=history, mu_r2=scales
        )
        right = stored.evaluate_correlated(q, color_correlation=image, mu_r2=scales)
        assert _values(left) == _values(right), (history, image)
    if spin:
        image = permutation.index(3) + 1
        left = runtime.evaluate_correlated_many(
            points, {"s": CorrelatedRequest("born", SPIN)}, mu_r2=scales
        )["s"]
        right = stored.evaluate_correlated_many(
            q, {"s": CorrelatedRequest("born", {image: SPIN[4]})}, mu_r2=scales
        )["s"]
        assert _values(left) == _values(right)
    resolved = runtime.evaluate_resolved(points[:1], mu_r2=scales[:1])
    reference = stored.evaluate_resolved(q[:1], mu_r2=scales[:1])
    rows = [tuple(h.values) for h in runtime.physics.helicities]
    stored_rows = [tuple(h.values) for h in stored.physics.helicities]
    assert len(rows) == len(stored_rows)
    for h, row in enumerate(rows):
        # The member's helicity on leg permutation[r] is the
        # representative's on leg r.
        target = stored_rows.index(tuple(row[p] for p in permutation))
        assert _bits(resolved.values[0][0][h]) == _bits(reference.values[0][0][target])
        assert _bits(resolved.born_values[0][0][h]) == _bits(
            reference.born_values[0][0][target]
        )


def test_unserved_correlations_are_refused(mapped):
    """A declared history (or spin class) whose image the representative
    does not store is not served: ``T12`` read on ``s~ s > g g`` is the
    representative's T21; the class (4,) of ``d~ d > g g`` is its leg 3."""

    from pyamplicol.api.errors import CompatibilityError, EvaluationError
    from pyamplicol.correlators import CorrelatedRequest

    runtime = _load(mapped, "sxs")
    with pytest.raises(CompatibilityError, match="T12"):
        runtime.evaluate_correlated(POINTS, color_correlation="T12", mu_r2=SCALES)
    alias = _load(mapped, "ddx_swap")
    with pytest.raises(EvaluationError, match="not declared"):
        alias.evaluate_correlated_many(
            POINTS, {"s": CorrelatedRequest("born", SPIN)}, mu_r2=SCALES
        )


def test_selectors_reach_the_mapped_processes(mapped):
    """Alias and member expressions select them; an expression equal to a
    stored one up to a side permutation selects the stored process with
    that permutation (and keeps its id); a crossing is not a mapping."""

    from pyamplicol.api.errors import PyAmpliColError

    assert _load(mapped, "D~ D > G G").physics.process_id == "ddx_swap"
    assert _load(mapped, "s~ s > g g").physics.process_id == "sxs"
    permuted = _load(mapped, "g d > d g")
    assert permuted.physics.process_id == "dg"
    assert permuted.external_permutation == (1, 0, 2, 3)
    assert permuted.physics.process == "G d > d G"
    stored = _load(mapped, "dg")
    q = _to_representative(POINTS, (1, 0, 2, 3))
    assert _values(permuted.evaluate(POINTS, mu_r2=SCALES)) == _values(
        stored.evaluate(q, mu_r2=SCALES)
    )
    with pytest.raises(PyAmpliColError):
        _load(mapped, "d g > d d~")


def test_non_exact_group_relations_are_refused(mapped, tmp_path):
    """A process-group member of any basis but ``exact-loop-flavor`` (a
    relation found by numerical analysis) is refused at load, typed."""

    from dataclasses import replace

    from pyamplicol.api.errors import EvaluationError
    from pyamplicol.artifacts.process_groups import (
        read_process_groups,
        write_process_groups,
    )
    from pyamplicol.generation.process_grouping import (
        GroupingBasis,
        ProcessGroupMap,
    )

    copy = tmp_path / "artifact"
    shutil.copytree(mapped, copy)
    groups = read_process_groups(copy)
    edited = ProcessGroupMap()
    for group in groups.groups.values():
        edited.add_group(
            [
                replace(member, basis=GroupingBasis.NUMERICAL)
                if member.process_id == "gs"
                else member
                for member in group.members
            ]
        )
    write_process_groups(copy, edited)
    with pytest.raises(EvaluationError, match="numerical"):
        _load(copy, "gs")
    assert _load(copy, "sxs").representative_process_key == "ddx"


@pytest.fixture(scope="module")
def direct(loop_sm_compiled_model, loop_native, tmp_path_factory):
    """``d d~ > g g`` (unmapped), ``d~ d > g g`` and ``s~ s > g g`` each
    generated as its own process."""

    del loop_native
    return _generate(
        loop_sm_compiled_model,
        tmp_path_factory.mktemp("direct"),
        (("ddx", "d d~ > g g"), ("dxd", "d~ d > g g"), ("sxs", "s~ s > g g")),
    )


def test_mappings_equal_direct_generation(mapped, direct):
    """The alias ``d~ d > g g`` and the member ``s~ s > g g`` equal their own
    generations (inclusive and a composed history, binary64 at 1e-12 of
    each sector's largest coefficient), and the representative's stored plan
    and native record are byte-identical to its generation alone."""

    for name, own in (("ddx_swap", "dxd"), ("sxs", "sxs")):
        for history in ("born", "T13", "T24"):
            left = _load(mapped, name).evaluate_correlated(
                POINTS, color_correlation=history, mu_r2=SCALES
            )
            right = _load(direct, own).evaluate_correlated(
                POINTS, color_correlation=history, mu_r2=SCALES
            )
            for a, b in zip(left.values, right.values, strict=True):
                for x, y in zip(a, b, strict=True):
                    size = max(abs(complex(v)) for v in y)
                    for u, v in zip(x, y, strict=True):
                        assert abs(complex(u) - complex(v)) <= 1e-12 * size, (
                            name,
                            history,
                        )
    left, right = _load(mapped, "ddx").inspect(), _load(direct, "ddx").inspect()
    assert left["plan_sha256"] == right["plan_sha256"]
    assert left["scaleless_plan_sha256"] == right["scaleless_plan_sha256"]
    member = "processes/ddx/loop-runtime.pacbin"
    assert (mapped / member).read_bytes() == (direct / member).read_bytes()


def test_double_double_through_the_mapping(mapped):
    """A composed spin and colour correlation in double-double: the member
    ``g s > s g`` equals ``d g > d g`` at the permuted momenta (member leg 4
    is representative leg 4; T13 is the representative's T23)."""

    from pyamplicol.correlators import CorrelatedRequest

    left = _load(mapped, "gs").evaluate_correlated_many(
        POINTS[:1],
        {"s": CorrelatedRequest("T13", SPIN)},
        mu_r2=SCALES[:1],
        arithmetic="double-double",
    )["s"]
    right = _load(mapped, "dg").evaluate_correlated_many(
        _to_representative(POINTS[:1], (1, 0, 2, 3)),
        {"s": CorrelatedRequest("T23", SPIN)},
        mu_r2=SCALES[:1],
        arithmetic="double-double",
    )["s"]
    assert _values(left) == _values(right)
