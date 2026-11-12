# SPDX-License-Identifier: 0BSD
"""What the CLI's tables mean (``--fields-description-help``).

By default every command prints its tables and short status lines only; with
``--fields-description-help`` it adds, after its tables, a short heading per
table and one line per column ("column - what it is"), in plain words.
"""

from __future__ import annotations

import textwrap
from collections.abc import Sequence
from typing import Any

# One description block: a heading and its (column, text) lines; a column of
# None is a free sentence under the heading.
Block = tuple[str, Sequence[tuple[str | None, str]]]

_WIDTH = 100

_MADLOOP = (
    "The measure follows MadLoop (arXiv:1103.0621, sect. 3.4) and "
    "MadGraph5_aMC@NLO (arXiv:1405.0301, section 'Checks, stability, and "
    "recovery of numerically-unstable integral reductions')."
)

_COLOURS = (
    "Colours: green is good, yellow is acceptable but close to the limit or "
    "served by a fallback, red is a problem, grey is not applicable."
)

_DEFAULT_MARK = "* marks the default reduction backend of this kind of process."


def _render(blocks: Sequence[Block]) -> str:
    out = ["Field descriptions", "=================="]
    for heading, lines in blocks:
        out += ["", heading]
        width = max((len(name) for name, _ in lines if name), default=0)
        for name, text in lines:
            if name is None:
                out += textwrap.wrap(
                    text, _WIDTH, initial_indent="  ", subsequent_indent="  "
                )
                continue
            first = f"  {name.ljust(width)} - "
            out += textwrap.wrap(
                text,
                _WIDTH,
                initial_indent=first,
                subsequent_indent=" " * len(first),
            )
    return "\n".join(out)


def _profile(report: Any) -> list[Block]:
    blocks: list[Block] = [
        (
            "pyAmpliCol loop profile",
            [
                ("process", "the process profiled, its identifier and expression"),
                (
                    "contribution",
                    "virtual: Born-loop interference; loop-induced: |A_loop|^2",
                ),
                (
                    "runtime setup",
                    "what the timed calls compute: binary64, finite part only, "
                    "the backends' own acceptance checks; points per call and "
                    "the number of timed calls",
                ),
                ("backends", "the reduction backends compared. " + _DEFAULT_MARK),
            ],
        )
    ]
    if report.runtime is not None:
        blocks += [
            (
                "Runtime per reduction backend",
                [
                    ("backend", "the reduction backend. " + _DEFAULT_MARK),
                    ("time / point", "wall time per phase-space point, median call"),
                    ("min - max call", "the fastest and the slowest call, per point"),
                    (
                        "vs fastest",
                        "time relative to the fastest backend; 'all failed' when "
                        "no point got a value",
                    ),
                    (
                        "best-effort",
                        "shown when some points were refused by the backend's own "
                        "checks and returned with their best available value "
                        "(the default policy); those points are timed",
                    ),
                    (
                        "lanes",
                        "shown when points were served by more than one backend: "
                        "the default backend hands a point it refuses to the "
                        "other one",
                    ),
                    ("failed", "shown when points got no value at all"),
                ],
            ),
            (
                "Time per point and stage",
                [
                    ("stage", "a part of the evaluation"),
                    ("trees", "the tree currents feeding the loops"),
                    ("dressing", "attaching the loop propagators and vertices"),
                    ("scatter", "collecting the loop numerators"),
                    ("reduction", "the tensor-integral reduction (the backend)"),
                    ("rest", "everything else in the call"),
                    ("share", "the stage's fraction of the call"),
                    ("A / B", "the time ratio of the two backends per stage"),
                ],
            ),
        ]
    if report.stability is not None:
        setup = report.stability_setup or {}
        blocks += [
            (
                "Stable digits of binary64 vs double-double",
                [
                    (
                        None,
                        "Each flat (RAMBO) phase-space point is evaluated in "
                        "binary64 and again in double-double on the same backend. "
                        "stable digits = -log10 of the relative difference between "
                        "the binary64 and the double-double finite part, capped to "
                        "binary64's 15.95 digits. " + _MADLOOP,
                    ),
                    (
                        "percentile",
                        "the points sorted from worst to best: the row 1 % is the "
                        "point below which 1 % of the points lie",
                    ),
                    ("backend", "the reduction backend. " + _DEFAULT_MARK),
                    (
                        "true digits",
                        "the stable digits of the point at that percentile; "
                        "'- (N >= n)' when n points are needed to fill it",
                    ),
                    ("class", "the number of points between this and the previous row"),
                    (
                        "|est-true|",
                        "how far the binary64 estimate of the digits (one default "
                        "rotation of the point) is from the true digits: minimum, "
                        "median and maximum over the class",
                    ),
                    (
                        "est-true med",
                        "the median of estimate minus truth: positive means the "
                        "estimate is optimistic (red), negative pessimistic (blue)",
                    ),
                ],
            ),
            (
                "Default checks against the truth",
                [
                    (
                        None,
                        f"A point is accurate when its true relative error is "
                        f"below the target {setup.get('target', '1e-05')}.",
                    ),
                    ("measured", "points with a double-double reference"),
                    (
                        "best-eff./none",
                        "points refused by the backend's checks and returned with "
                        "their best available value / points without any value",
                    ),
                    ("flagged", "points the default checks call unstable"),
                    ("flagged inacc.", "of those, the truly inaccurate ones"),
                    ("missed", "truly inaccurate points the checks did not flag"),
                    (
                        "time",
                        "the backend's binary64 and double-double calls, summed over "
                        "the workers",
                    ),
                ],
            ),
            (
                "Best-effort values against the truth",
                [
                    ("points", "the points returned with their best available value"),
                    ("true min/med/max", "their true stable digits"),
                    ("unc. median", "the digits of the uncertainty they carry"),
                    ("|unc-true| med", "how far that uncertainty is from the truth"),
                    (
                        "unc. covers",
                        "uncertainties at least as large as the true error",
                    ),
                ],
            ),
        ]
    blocks.append(
        (
            "Failure causes (under a table, after ---)",
            [
                (
                    None,
                    "One line per backend and cause, most frequent first: the "
                    "number of points, their share of the backend's points, the "
                    "backend, and the cause. 'best-effort': refused, returned "
                    "with the best available value; 'no value': no result; 'no "
                    "reference': the double-double re-run failed.",
                ),
                (None, _COLOURS),
            ],
        )
    )
    return blocks


def _virtual(value: Any) -> list[Block]:
    from pyamplicol.api.virtual_results import VirtualOrderEvaluation

    result = value.evaluation if isinstance(value, VirtualOrderEvaluation) else value
    loop_induced = getattr(result, "contribution", "virtual") == "loop-induced"
    lines: list[tuple[str | None, str]] = [
        (
            None,
            "Coefficients of the Laurent series in epsilon with the normalisation "
            "C_epsilon = (4 pi)^epsilon / Gamma(1 - epsilon); colour accuracy "
            f"{str(result.color_accuracy).upper()}; R2 "
            f"{result.r2.effective.value} ({result.r2.reason}).",
        ),
        ("point", "the phase-space point, in input order"),
        ("virtual squared order", "the coupling powers of this sector"),
        ("finite [0]", "the finite part"),
        ("single pole [-1]", "the 1/epsilon coefficient"),
        ("double pole [-2]", "the 1/epsilon^2 coefficient"),
        (None, "Signed values carry an explicit + so that the digits align."),
    ]
    blocks: list[Block] = [
        (
            "|A_loop|^2 (loop-induced)" if loop_induced else "Born-loop interference",
            lines,
        )
    ]
    if not loop_induced:
        blocks.insert(
            0,
            (
                "Born contributions",
                [
                    ("Born squared order", "the coupling powers of the Born"),
                    ("Born", "the Born matrix element squared"),
                ],
            ),
        )
    blocks += [
        (
            "Contributing order pairs",
            [("Born x loop orders", "the amplitude orders that make each sector")],
        ),
        (
            "Evaluation diagnostics",
            [
                (
                    "status",
                    "success, rescued, best-effort (refused but returned with "
                    "its best value) or failed",
                ),
                ("arithmetic", "the precision that produced the value"),
                ("bits", "the mantissa bits of that precision"),
                ("reduction", "the reduction backend that served the point"),
                ("accuracy established", "whether a requested target was proven"),
                ("detail", "the reason of a refusal or fallback"),
            ],
        ),
    ]
    if result.stability is not None:
        blocks.append(
            (
                "Stability estimates",
                [
                    (
                        None,
                        "Relative error estimates from the requested checks: "
                        "green within the target, red above.",
                    ),
                    ("checks", "each check and its own estimate"),
                ],
            )
        )
    return blocks


def _check(report: Any) -> list[Block]:
    from pyamplicol.loops.v2 import checks
    from pyamplicol.loops.v2.induced import ESTIMATE_FACTOR

    return [
        (
            "pyAmpliCol check",
            [
                ("source", "the run card or artifact whose request was checked"),
                (
                    "points",
                    "seeded phase-space points per process, lifted to exact "
                    "momentum conservation; mu_R^2 = s",
                ),
                (
                    "carriers",
                    "the precisions tried in order: binary64, then double-double "
                    "where binary64 cannot decide, then arbitrary precision",
                ),
                ("seconds", "the wall time of the whole check"),
                (None, f"Regeneration: {report.regeneration}."),
            ],
        ),
        (
            "One table per process",
            [
                (
                    "check",
                    "poles: the poles against Catani's operator (Born-interfered) "
                    "or their absence (loop-induced); ward: gauge invariance of "
                    "each external gluon or photon; goldstone: the Goldstone "
                    "equivalence of external massive vectors",
                ),
                ("subject", "the sector, coefficient or leg tested"),
                ("carrier", "the precision that decided the row"),
                ("ratio", "the deviation over its allowed bound; at most 1 passes"),
                ("dev / scale", "the deviation over the row's scale"),
                ("E / scale", "the error estimate over the same scale"),
                (
                    None,
                    f"Bounds: poles |pole - Catani| <= {checks.ESTIMATES} E + "
                    f"{checks.CATANI_FLOOR:g} |Catani| in binary64 and "
                    f"{checks.CATANI_DD_FLOOR:g} relative in double-double; probes "
                    f"within {ESTIMATE_FACTOR} times their own estimate.",
                ),
                ("verdict", "pass, fail, undecided, not applicable or not run"),
                (
                    "fallback / reason",
                    "why a carrier did not decide, or why no verdict",
                ),
                (None, _COLOURS),
            ],
        ),
    ]


def _generation() -> list[Block]:
    return [
        (
            "Generation summary",
            [
                ("output", "the artifact directory written"),
                ("processes", "the process requests generated"),
                ("existing-output policy", "what happens to an existing output"),
                ("schema version", "the artifact format version"),
                ("files", "the files written and their total size"),
            ],
        ),
        (
            "Generated Processes",
            [
                ("name", "the identifier to select the process later"),
                ("process", "the process as generated"),
            ],
        ),
    ]


def field_descriptions(value: Any) -> str:
    """The plain-words description of the tables ``render_summary`` prints
    for ``value``."""

    from pyamplicol.api.results import GenerationResult
    from pyamplicol.api.virtual_results import (
        ResolvedVirtualEvaluation,
        VirtualEvaluation,
        VirtualOrderEvaluation,
    )
    from pyamplicol.loop_profile import LoopProfileReport
    from pyamplicol.loops.v2.checks import CheckReport

    blocks: list[Block]
    if isinstance(value, LoopProfileReport):
        blocks = _profile(value)
    elif isinstance(
        value, (VirtualEvaluation, ResolvedVirtualEvaluation, VirtualOrderEvaluation)
    ):
        blocks = _virtual(value)
    elif isinstance(value, CheckReport):
        blocks = _check(value)
    elif isinstance(value, GenerationResult):
        blocks = _generation()
    else:
        blocks = [
            (
                "Tables",
                [
                    (
                        None,
                        "Each row names a field and its value; numbers of both "
                        "signs carry an explicit + so that the digits align.",
                    )
                ],
            )
        ]
    return _render(blocks)


__all__ = ["field_descriptions"]
