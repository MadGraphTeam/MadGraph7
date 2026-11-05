# SPDX-License-Identifier: 0BSD
"""The built-in SM against the UFO sm over its 2 -> 2 processes, in every lane.

``tools/developer/builtin_sm_ufo_sweep.py`` enumerates every 2 -> 2 process
of the built-in particle table the parser admits (charge, colour, baryon and
lepton-family number; 1993) and keeps one per class under crossing, CP and
the relabellings that map the built-in vertex table onto itself, then adds
every 1 -> 2 decay, a 2 -> 3 / 1 -> 3 sample and electroweak exchange between
quark lines at explicit coupling orders. The committed reference
(``tests/fixtures/tree/builtin-sm-ufo-sm.json``, MG5 sm with its no_masses
restriction, the built-in masses, widths and couplings) gives the UFO value
at two points, or null where the UFO sm has no tree amplitude. Each lane of
the built-in SM must reproduce every value to 1e-10 and refuse every null
with a ``GenerationError``. The nightly variant runs each of the 901
relabelling representatives of the 2 -> 2 processes in every lane against
the live UFO sm.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

from pyamplicol import Generator, ModelSource, Runtime
from pyamplicol.api.errors import GenerationError
from pyamplicol.config import EvaluatorConfig
from tools.developer.builtin_sm_ufo_sweep import (
    LANES,
    evaluate,
    nightly_processes,
    on_shell_points,
    reference_processes,
    run_config,
    split_entry,
)

REFERENCE = Path(__file__).resolve().parents[1] / "fixtures/tree/builtin-sm-ufo-sm.json"
NIGHTLY_SHARDS = 16


def _reference() -> dict[str, list[float] | None]:
    return json.loads(REFERENCE.read_text(encoding="utf-8"))["processes"]


_QUARKS = frozenset({"d", "u", "s", "c", "b", "t"})
_RECURRENCE_REFUSAL = (
    "no exact closure for physical LC sector|local-color view requires a direct"
)


def _recurrence_refuses(entry: str) -> bool:
    """Four quarks without a gluon exchange: the recurrence lane refuses them.

    Four quarks of four flavours (lines that can only exchange a W) or a
    request at QCD = 0. The recurrence builder asks for a closure in every
    quark-antiquark pairing of the colour plan, which the first leave empty,
    and needs a colour reducer the second lack; it refuses both with a
    ``GenerationError`` for UFO models too. The other lanes serve them.
    """

    process, orders = split_entry(entry)
    quarks = [name.rstrip("~") for name in process.replace(">", " ").split()]
    quarks = [name for name in quarks if name in _QUARKS]
    return len(quarks) == 4 and (
        len(set(quarks)) == 4 or (orders is not None and orders.get("QCD") == 0)
    )


def _lane_values(
    entries: list[str], execution_mode: str, root: Path
) -> dict[str, list[float]]:
    """The built-in values of ``entries`` (all served), one artifact per set
    of coupling orders; in the recurrence lane, the entries it refuses
    (``_recurrence_refuses``) must raise its typed refusal instead."""

    groups: dict[str, list[str]] = defaultdict(list)
    for index, entry in enumerate(entries):
        if execution_mode == "recurrence" and _recurrence_refuses(entry):
            process, orders = split_entry(entry)
            with pytest.raises(GenerationError, match=_RECURRENCE_REFUSAL):
                Generator(run_config(execution_mode, orders)).generate(
                    [process],
                    root / f"refused-{index}",
                    model=ModelSource.built_in_sm(),
                )
            continue
        process, orders = split_entry(entry)
        groups[json.dumps(orders, sort_keys=True)].append(entry)
    values = {}
    for index, (orders, members) in enumerate(sorted(groups.items())):
        artifact = root / f"artifact-{index}"
        Generator(run_config(execution_mode, json.loads(orders))).generate(
            [split_entry(entry)[0] for entry in members],
            artifact,
            model=ModelSource.built_in_sm(),
        )
        for entry in members:
            process = split_entry(entry)[0]
            runtime = Runtime.load(artifact, process=process)
            values[entry] = [
                value.real for value in runtime.evaluate(on_shell_points(process))
            ]
    return values


def _mismatches(
    expected: dict[str, list[float]], values: dict[str, list[float]]
) -> list[str]:
    return [
        f"{entry}: built-in {values[entry]} UFO {reference}"
        for entry, reference in expected.items()
        if entry in values and values[entry] != pytest.approx(reference, rel=1e-10)
    ]


def test_the_reference_holds_the_generated_processes() -> None:
    assert list(_reference()) == list(reference_processes())


@pytest.mark.parametrize("execution_mode", LANES)
def test_builtin_sm_matches_the_ufo_sm(execution_mode: str, tmp_path: Path) -> None:
    served = {entry: value for entry, value in _reference().items() if value}
    values = _lane_values(list(served), execution_mode, tmp_path)
    assert not _mismatches(served, values)


def test_builtin_sm_refuses_what_the_ufo_sm_has_no_tree_amplitude_for(
    tmp_path: Path,
) -> None:
    """Refusals come from process selection and the Born-order sweep, which
    every lane shares; the compiled lane reaches them without a prepared
    bundle."""

    served = []
    for index, (entry, value) in enumerate(_reference().items()):
        if value is not None:
            continue
        process, orders = split_entry(entry)
        try:
            Generator(run_config("compiled", orders)).generate(
                [process],
                tmp_path / f"artifact-{index}",
                model=ModelSource.built_in_sm(),
            )
        except GenerationError:
            continue
        served.append(entry)
    assert not served


@pytest.fixture(scope="module")
def live_ufo_sm(
    madgraph_source: Path, tmp_path_factory: pytest.TempPathFactory
) -> ModelSource:
    root = tmp_path_factory.mktemp("ufo-sm-no-masses")
    bundle = root / "sm-no_masses.pyamplicol-model"
    ModelSource.from_path(
        madgraph_source / "models/sm", restriction="no_masses"
    ).compile(
        cache_dir=root / "cache", prepared_output=bundle, evaluator=EvaluatorConfig()
    )
    return ModelSource.from_path(bundle)


@pytest.mark.loop_nightly
@pytest.mark.parametrize("shard", range(NIGHTLY_SHARDS))
def test_builtin_sm_matches_the_live_ufo_sm_nightly(
    live_ufo_sm: ModelSource, shard: int, tmp_path: Path
) -> None:
    """One shard of ``nightly_processes``; run the shards as separate
    processes."""

    entries = nightly_processes()[shard::NIGHTLY_SHARDS]
    reference = {
        entry: evaluate(entry, live_ufo_sm, "compiled", tmp_path / f"ufo-{index}")
        for index, entry in enumerate(entries)
    }
    served = {entry: value for entry, value in reference.items() if value}
    failures = []
    for execution_mode in LANES:
        values = _lane_values(list(served), execution_mode, tmp_path / execution_mode)
        failures += [f"{execution_mode} {line}" for line in _mismatches(served, values)]
    for index, entry in enumerate(entry for entry in entries if entry not in served):
        if evaluate(
            entry, ModelSource.built_in_sm(), "compiled", tmp_path / f"b-{index}"
        ):
            failures.append(
                f"compiled {entry}: served, the UFO sm has no tree amplitude"
            )
    assert not failures
