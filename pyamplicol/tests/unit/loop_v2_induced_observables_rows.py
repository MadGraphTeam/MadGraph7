# SPDX-License-Identifier: 0BSD
"""Loop-induced artifacts with declared observables (m-lit LIT-OBS): the
LI-O1 fixture rows generated through the public ``Generator`` with every
dipole T_i.T_j and every gluon's spin class declared. Shared by
``test_loop_v2_induced_observables.py`` and the both-backends check
``tools/loop_v2_observables_backends.py``.
"""

from __future__ import annotations

from pathlib import Path

from tests.unit.test_loop_v2_induced import _card, _fixture

# The LI-O1 rows (PLAN §4): fixture name -> process.
ROWS = {"li_gg_hgg": "g g > h g g", "li_uux_hgg": "u u~ > h g g"}


def dipoles(legs):
    return [
        {
            "id": f"T{i}T{j}",
            "bra": [{"kind": "emit-gluon", "emitter": i, "emitted": -1}],
            "ket": [{"kind": "emit-gluon", "emitter": j, "emitted": -1}],
        }
        for i in legs
        for j in legs
        if i != j
    ]


def coloured(pdgs):
    return [k for k, pdg in enumerate(pdgs, 1) if abs(pdg) in (1, 2, 3, 4, 5, 6, 21)]


def gluons(pdgs):
    return [k for k, pdg in enumerate(pdgs, 1) if pdg == 21]


def generate(root: Path, name: str, expression: str, aliases=()):
    """The loop-induced artifact of fixture row ``name`` through the public
    Generator, every dipole T_i.T_j and every gluon's spin class declared
    (``aliases``: ``ProcessAlias`` arguments on process ``p``)."""

    from pyamplicol.api import Generator, ProcessAlias, ProcessRequest, ProcessSet
    from pyamplicol.config import resolve_config
    from pyamplicol.correlators import CorrelatorConfig

    fixture = _fixture(name)
    pdgs = fixture["pdgs"]
    root.mkdir(parents=True, exist_ok=True)
    card = root / "card.toml"
    card.write_text(_card(root / "artifact", (("p", expression),), fixture))
    correlators = CorrelatorConfig.from_json_dict(
        {
            "color_correlations": dipoles(coloured(pdgs)),
            "spin_correlations": [[leg] for leg in gluons(pdgs)],
        }
    )
    Generator(resolve_config(card)).generate(
        ProcessSet(
            (ProcessRequest.parse(expression, name="p"),),
            tuple(ProcessAlias(*alias) for alias in aliases),
        ),
        root / "artifact",
        correlators=correlators,
    )
    return root / "artifact"


def fold_record(artifact: Path, name: str = "p") -> dict:
    """Process ``name``'s fold record in ``artifact`` (M-LIT lit-fold, F1:
    folding is on by default, so the artifacts of ``generate`` are folded
    wherever the exact test certifies a group; their consumer tests run on
    folded plans, USER 2026-11-08 ~22:39, DECISIONS 22:39)."""

    import json

    manifest = json.loads((artifact / "artifact.json").read_text())
    (filters,) = (
        record["filters"]
        for record in manifest["extensions"]["generation"]["concrete_processes"]
        if record["id"] == name
    )
    return filters["loop_induced_fold"]
