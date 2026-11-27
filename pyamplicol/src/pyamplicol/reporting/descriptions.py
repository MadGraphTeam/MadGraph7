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


# How ``resources.peak_memory`` read a peak (``PeakMemory.method``).
_PEAK_METHODS = {
    "VmHWM after clear_refs": (
        "the kernel's high-water mark VmHWM, reset when the calls start "
        "(/proc/self/clear_refs)"
    ),
    "VmHWM": (
        "the kernel's lifetime high-water mark VmHWM (not reset: the reset was "
        "refused, or pyAmpliCol ran as a library inside another program)"
    ),
}


def _peak_ram(report: Any) -> str:
    """The ``peak RAM`` column's description, with how this run measured
    it."""

    threads = max(
        (
            int(getattr(row.profile, "threads", None) or 1)
            for row in report.runtime
            if row.profile is not None
        ),
        default=1,
    )
    methods = dict.fromkeys(
        row.profile.peak_memory.method
        for row in report.runtime
        if row.profile is not None and row.profile.peak_memory is not None
    )
    measured = "; ".join(
        _PEAK_METHODS.get(
            method,
            "getrusage's lifetime peak ru_maxrss (no resettable mark on this platform)"
            if method.startswith("ru_maxrss")
            else method,
        )
        for method in methods
    )
    return (
        "the memory to expect for each Monte Carlo worker evaluating this "
        "matrix element on this backend: the peak resident set size of a "
        "process that loads the artifact and evaluates on this backend alone "
        "(each backend is profiled in its own such process), at the runtime "
        "setup's points per call, over its binary64 warm-up and timed calls, "
        "before the reference timing (whose own peak is in --json, "
        "reference), on one core (every thread pool at one thread): the "
        "memory of one Monte Carlo worker"
        + (f"; measured from {measured}" if measured else "")
        + "; '<= x': the platform only knows the process's lifetime peak x, "
        "which these calls did not raise; '-': not measurable here"
        + (
            f"; with --runtime-threads {threads}: that process's on {threads} "
            "threads (its per-thread work space included), before the "
            "one-thread timing"
            if threads > 1
            else ""
        )
    )


def _profile(report: Any) -> list[Block]:
    from .loop_profile import _truth

    label, name, _ = _truth(report)
    # --runtime-threads N > 1: the timed calls ran on N threads.
    threads = max(
        (
            int(getattr(row.profile, "threads", None) or 1)
            for row in report.runtime or ()
            if getattr(row, "profile", None) is not None
        ),
        default=1,
    )
    many = (
        f"; with --runtime-threads {threads}: on {threads} threads, the pools "
        f"of each backend's process at {threads} threads working inside each "
        "call, the calls still one at a time (not the cost of one core: a "
        "Monte Carlo run giving each worker one core should use the default), "
        "then a short one-thread timing of the same calls for the gain"
        if threads > 1
        else ""
    )
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
                    "one fully optimistic pass per point (the backend's "
                    "unchecked reduction alone, no stability checks, no "
                    "rescue, no best-effort re-evaluation; with --checked the "
                    "backends' own acceptance checks, a refused point served "
                    "best-effort, the default backend as auto); the "
                    "distinct points per call (a new batch every call); the "
                    "timed calls (their count and size from the rule of the "
                    "counts row), after the untimed warm-up; the reference "
                    "points the multiplier comes from; on one core: each "
                    "backend is timed in its "
                    "own process whose thread pools all run one thread (what "
                    "one Monte Carlo worker pays; --n_core is the stability "
                    "run's worker count only)" + many,
                ),
                (
                    "counts",
                    "how the counts were set and what the whole profile took: "
                    "from --completion-time (the plan aims at that wall time; "
                    "the options it overrules; 'cannot be met' with what the "
                    "minimums take when they alone exceed it), as given "
                    "(counts on the command line or in the card), or from the "
                    "default time budgets (10 s for the runtime timing, one "
                    "minute per --n_core worker for the stability run)",
                ),
                (
                    "ground truth",
                    "the reference precision of both tables (--ground-truth): "
                    "the stability run's true values and the reference time "
                    "multipliers; 'none (estimates only)': no reference pass, "
                    "the stability tables show the estimates alone",
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
                    (
                        "time / point",
                        "the time per phase-space point on one core of the "
                        "runtime setup's pass: the median timed call's wall "
                        f"time over its points; '({label} xN)', muted: the "
                        f"same optimistic pass in {name} (the ground truth) on "
                        "the same backend and core - the unchecked reduction, "
                        "no stability checks, no rescue, no best-effort "
                        "re-evaluation - takes N times this time per point: N "
                        "is its median call's time per point over this one, "
                        "after one untimed point, its calls of one SIMD chunk "
                        "(16 points; fewer when its budget affords fewer, at "
                        "least one) on the first points of the binary64 "
                        "calls' batches, at least one call, more while they "
                        "fit its budget (a quarter of the backend's runtime "
                        f"share); '({label} -)': no reference call ran (its "
                        "points, calls and failures are in --json, reference)"
                        + (
                            f"; on {threads} threads the time per point is the "
                            f"call's wall time with {threads} threads working "
                            "over its points, the multiplier the reference "
                            f"pass's on {threads} threads too; '(xS vs 1 "
                            "thread)', muted: the same calls on one thread take "
                            "S times as long (a short timing of its own after "
                            f"them, a quarter of the binary64 budget, at least "
                            "one call; --json one_thread_seconds_per_point)"
                            if threads > 1
                            else ""
                        ),
                    ),
                    (
                        "min - max call",
                        "the fastest and the slowest timed call, per point, on "
                        "one core",
                    ),
                    (
                        "vs fastest",
                        "time relative to the fastest backend; 'all failed' when "
                        "no point got a value",
                    ),
                    ("peak RAM", _peak_ram(report)),
                    (
                        "best-effort",
                        "with --checked, shown when some points were refused by "
                        "the backend's own checks and returned with their best "
                        "available value; those points are timed, their "
                        "unchecked re-evaluation and the checks of its "
                        "uncertainty included",
                    ),
                    (
                        "lanes",
                        "with --checked, shown when points were served by more "
                        "than one backend: the default backend (as auto) hands "
                        "a point it refuses to the other one",
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
                    (
                        "rest",
                        "everything else in the call: its Python part and the "
                        "native work outside the stages above",
                    ),
                    (
                        "share",
                        "the stage's fraction of the timed calls' total time, "
                        "read from the native stage reports of the same calls; "
                        "the stages add up to the time per point",
                    ),
                    ("A / B", "the time ratio of the two backends per stage"),
                ],
            ),
        ]
    if report.stability is not None and any(
        getattr(row, "estimate_only", False) for row in report.stability
    ):
        setup = report.stability_setup or {}
        blocks += [
            (
                "Estimated digits of binary64 (no ground truth)",
                [
                    (
                        None,
                        "--ground-truth none: each flat point is evaluated in "
                        "binary64 only, with the default rotation test; no "
                        "reference, so no true digits. " + _MADLOOP,
                    ),
                    (
                        "percentile",
                        "the points sorted by their estimated digits from worst "
                        "to best: the row 1 % is the point below which 1 % lie",
                    ),
                    ("backend", "the reduction backend. " + _DEFAULT_MARK),
                    (
                        "est. digits",
                        "the estimated stable digits of the point at that "
                        "percentile; '- (N >= n)' when n points are needed",
                    ),
                    ("class", "the number of points between this and the previous row"),
                ],
            ),
            (
                "Default checks, estimates only",
                [
                    ("evaluated", "the points evaluated in binary64"),
                    (
                        "best-eff./none",
                        "points refused by the backend's checks and returned with "
                        "their best available value / points without any value",
                    ),
                    (
                        "flagged",
                        "points the default checks call unstable (estimate above "
                        f"the target {setup.get('target', '1e-05')}, a "
                        "best-effort value, or none)",
                    ),
                    (
                        "time",
                        "the wall time of the backend's binary64 calls with their "
                        "stability checks, summed over the workers (each on one "
                        "core, one thread per pool)",
                    ),
                ],
            ),
        ]
    elif report.stability is not None:
        setup = report.stability_setup or {}
        blocks += [
            (
                f"Stable digits of binary64 vs {name}",
                [
                    (
                        None,
                        "Each flat (RAMBO on diet) phase-space point is evaluated in "
                        f"binary64 and again in {name} (the ground truth) on the "
                        "same backend. stable digits = -log10 of the relative "
                        f"difference between the binary64 and the {name} finite "
                        "part, capped to binary64's 15.95 digits. " + _MADLOOP,
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
                    ("measured", f"points with a {name} reference"),
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
                        "the wall time of the backend's binary64 calls with "
                        "their stability checks (the default rotation test "
                        "evaluates each point twice), summed over the workers "
                        "(each on one core, one thread per pool), "
                        "after each worker's untimed start-up (loading, first "
                        f"calls); '({label} xN)': its {name} reference calls "
                        "(one evaluation per point) took N times as long in "
                        "total, so N is not the cost ratio of one evaluation "
                        f"(the runtime table's '({label} xN)'); under the time "
                        "when "
                        "the table would pass the line width (the points and "
                        "time per point of each precision are in --json, "
                        "precision_timing)",
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
                    f"reference': the {name} re-run failed; '... in {name}': "
                    "the same in the runtime's reference timing (counted over "
                    "its points).",
                ),
                (None, _COLOURS),
            ],
        )
    )
    return blocks


def _virtual(value: Any) -> list[Block]:
    from pyamplicol.api.virtual_results import VirtualOrderEvaluation

    result = value.evaluation if isinstance(value, VirtualOrderEvaluation) else value
    from pyamplicol.reporting.virtual import virtual_titles

    loop_induced = getattr(result, "contribution", "virtual") == "loop-induced"
    born_title, virtual_title = virtual_titles(result)
    poles = str(getattr(result.poles, "value", result.poles))
    colour = {
        "lc": "leading colour",
        "nlc": "next-to-leading colour",
    }.get(str(result.color_accuracy), "full colour")
    single, double = {
        "uv": (
            "the 1/epsilon coefficient of the UV poles of the bare loop "
            "amplitude, before renormalisation",
            "zero: the bare loop has no UV double pole",
        ),
        "ir": (
            "the 1/epsilon coefficient of the IR poles of the renormalised virtual",
            "the 1/epsilon^2 coefficient, infrared",
        ),
        "none": ("not computed", "not computed"),
    }.get(
        poles,
        (
            "the 1/epsilon coefficient of the renormalised virtual",
            "the 1/epsilon^2 coefficient of the renormalised virtual",
        ),
    )
    lines: list[tuple[str | None, str]] = [
        (
            None,
            "Coefficients of the Laurent series in epsilon with the normalisation "
            "C_epsilon = (4 pi)^epsilon / Gamma(1 - epsilon); "
            f"{colour}; R2 {result.r2.effective.value}: {result.r2.reason}.",
        ),
        ("point", "the phase-space point, in input order"),
        ("virtual squared order", "the coupling powers of this sector"),
        (
            "finite [0]",
            "the finite part of the renormalised virtual, the same for every "
            "pole selection",
        ),
        ("single pole [-1]", single),
        ("double pole [-2]", double),
        (None, "Signed values carry an explicit + so that the digits align."),
    ]
    if getattr(result, "correlated", False):
        lines.append(
            (
                None,
                "Directed colour correlations keep the complex Born and "
                "Laurent coefficients.",
            )
        )
    blocks: list[Block] = [(virtual_title, lines)]
    if not loop_induced:
        blocks.insert(
            0,
            (
                born_title,
                [
                    ("Born squared order", "the coupling powers of the Born"),
                    ("Born", "the Born matrix element squared"),
                    *(
                        [
                            (
                                "Born anchor",
                                "a leading-colour component of the Born; the "
                                "anchors are signed and may overlap, so they "
                                "are not sampling probabilities",
                            )
                        ]
                        if "anchors" in born_title
                        else []
                    ),
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
    if getattr(result, "pole_parts", None) is not None:
        blocks.append(
            (
                "Pole parts by origin",
                [
                    (
                        None,
                        "Each pole split by its origin: P/eps = P_UV/eps_UV + "
                        "P_IR/eps_IR.",
                    ),
                    ("origin", "uv or ir"),
                    (
                        "component",
                        "bare: the loop itself; r2: the rational terms; "
                        "uv_local: the coupling and mass counterterms; "
                        "uv_external_fields: the external-field factors; "
                        "uv: the two counterterm parts together",
                    ),
                    ("single pole [-1]", "the 1/epsilon coefficient of that part"),
                    ("double pole [-2]", "the 1/epsilon^2 coefficient of that part"),
                ],
            )
        )
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


def _statistics_method(statistics: Any) -> str:
    """How a generation's peak RAM was measured, in words."""

    method = str(getattr(statistics, "peak_rss_method", "") or "")
    if not method or method == "unavailable":
        return ""
    process, _, descendants = method.partition("; ")
    words = {
        "VmHWM after clear_refs": (
            "the kernel's high-water mark VmHWM, reset when the generation "
            "starts (/proc/self/clear_refs)"
        ),
        "VmHWM": (
            "the kernel's lifetime high-water mark VmHWM (not reset: "
            "pyAmpliCol ran as a library inside another program, the reset "
            "was refused, or the generation ran inside another measured "
            "region)"
        ),
    }.get(
        process,
        "getrusage's lifetime peak ru_maxrss (no resettable mark on this platform)"
        if process.startswith("ru_maxrss")
        else process,
    )
    if descendants.startswith("descendants every "):
        interval, _, source = descendants.removeprefix("descendants every ").partition(
            " from "
        )
        sampler, _, rusage = source.partition(" and ")
        descendants = f"the started processes summed every {interval} ({sampler})" + (
            ", and their largest peak from getrusage RUSAGE_CHILDREN" if rusage else ""
        )
    return f"; measured from {words}" + (f"; {descendants}" if descendants else "")


def _generation_statistics(statistics: Any) -> list[tuple[str | None, str]]:
    """The descriptions of the ``generation time`` and ``peak RAM`` rows
    (``summary._generation_statistics_rows``)."""

    return [
        (
            "generation time",
            "the wall-clock time of the generation: the whole generate call, "
            "from reading the processes to the last validation (the start-up "
            "of the command before it is not included); in brackets each "
            "generation phase (setup: reading the request and loading the "
            "generation engine; then the phases of the progress display) and "
            "the unattributed rest (between and around the phases), which add "
            "up to it; '-': not recorded (an artifact written before "
            "pyAmpliCol recorded it)",
        ),
        (
            "peak RAM",
            "the most memory the generation needed at once: the peak resident "
            "memory of the generating process, its threads (native thread "
            "pools included), and every process it started (compilers)"
            + _statistics_method(statistics)
            + "; '<= x': only the process's lifetime peak x is known, which "
            "the generation did not raise; '>= x': the generation started "
            "processes, so the peak of all of them together is at least x "
            "(in brackets the generating process's own peak when the peak is "
            "not it and, when above it, the largest one of a started process: "
            "the system credits a started process with the memory of the "
            "process that starts it); "
            "'-': not recorded",
        ),
    ]


def _generation(statistics: Any = None) -> list[Block]:
    return [
        (
            "Generation summary",
            [
                ("output", "the artifact directory written"),
                ("processes", "the process requests generated"),
                ("existing-output policy", "what happens to an existing output"),
                ("schema version", "the artifact format version"),
                ("files", "the files written and their total size"),
                *_generation_statistics(statistics),
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


# The columns of a table without a description of its own.
_GENERIC_COLUMNS: list[tuple[str | None, str]] = [
    ("field", "the name of the quantity"),
    (
        "value",
        "its value; signed numbers carry an explicit + so that the digits align",
    ),
]


def _inspection(statistics: Any) -> list[Block]:
    return [
        (
            "Artifact",
            [
                ("path", "the artifact directory"),
                ("artifact type", "a process artifact or a process set"),
                ("artifact ID", "the digest of the payloads the runtime loads"),
                ("created", "when the artifact was written (UTC)"),
                *_generation_statistics(statistics),
                ("producer", "the pyAmpliCol version that wrote it"),
                ("target", "the platform its payloads are built for"),
                ("model", "the model, its restriction and its source"),
                ("default process", "the process evaluated when none is named"),
                ("contents", "the processes and aliases it holds"),
                ("runtime", "the runtime engine and its version"),
                ("capabilities", "the runtime features it requires"),
                ("payloads", "its files and their total size"),
                ("integrity", "whether every payload matches its digest"),
            ],
        ),
        ("Other tables", _GENERIC_COLUMNS),
    ]


def _tree_evaluation(resolved: bool) -> list[Block]:
    columns: list[tuple[str | None, str]] = [
        ("point", "the phase-space point, in input order"),
    ]
    if resolved:
        columns += [
            ("helicity", "the helicities of the external legs, in process order"),
            ("color", "the colour flow or colour component"),
        ]
    columns.append(
        (
            "matrix element",
            "the squared tree matrix element, summed over what the row does "
            "not resolve",
        )
    )
    if resolved:
        columns.append(
            (
                None,
                "(LC) or (NLC) in the title: leading or next-to-leading "
                "colour, not full colour.",
            )
        )
    return [("Resolved Evaluation" if resolved else "Summed Evaluation", columns)]


def _benchmark() -> list[Block]:
    return [
        (
            "Benchmark summary",
            [
                ("metric", "what was measured or how"),
                (
                    "value",
                    "its value; times per point are means with their standard error",
                ),
                (
                    None,
                    "peak RAM: the peak resident set size of this process over "
                    "the profiled region (warm-ups, calibration and timed "
                    "blocks); '<= x': the platform only knows the process's "
                    "lifetime peak x, which the region did not raise.",
                ),
            ],
        ),
        (
            "Timing breakdown",
            [
                ("component", "a part of one evaluation"),
                ("mean +/- standard error", "its time per point"),
                ("relative standard error", "the standard error over the mean"),
            ],
        ),
        (
            "Native work counters",
            [
                ("area", "the part of the native runtime"),
                ("counter", "what is counted"),
                ("mean", "the mean count"),
                ("normalization", "per point or per runtime call"),
            ],
        ),
    ]


def _diagnostics() -> list[Block]:
    return [
        (
            "Summary",
            [
                ("package version", "the installed pyAmpliCol version"),
                ("checks", "how many checks ran"),
                ("interrupted", "yes when Ctrl-C stopped the run early"),
            ],
        ),
        (
            "Checks",
            [
                ("name", "the check"),
                (
                    "status",
                    "pass, fail, or warning when the check could not run or "
                    "was not run",
                ),
                (
                    "detail",
                    "what was found; the physics checks compare with MadLoop "
                    "and give the largest relative difference",
                ),
            ],
        ),
    ]


def _examples() -> list[Block]:
    return [
        (
            "Packaged Examples",
            [
                ("name", "run it with: pyamplicol examples run NAME"),
                ("action", "what the card does: generate, evaluate, profile ..."),
                ("description", "what the example shows"),
            ],
        )
    ]


def field_descriptions(value: Any) -> str:
    """The plain-words description of the tables ``render_summary`` prints
    for ``value``."""

    from decimal import Decimal

    from pyamplicol.api.results import (
        BenchmarkResult,
        GenerationResult,
        ResolvedEvaluation,
    )
    from pyamplicol.api.virtual_results import (
        ResolvedVirtualEvaluation,
        VirtualEvaluation,
        VirtualOrderEvaluation,
    )
    from pyamplicol.artifacts.inspection import ArtifactInspection
    from pyamplicol.diagnostics import DiagnosticReport
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
        blocks = _generation(value.statistics)
    elif isinstance(value, ArtifactInspection):
        blocks = _inspection(value.generation_statistics)
    elif isinstance(value, ResolvedEvaluation):
        blocks = _tree_evaluation(True)
    elif (
        isinstance(value, tuple)
        and value
        and all(
            isinstance(item, (int, float, complex, Decimal))
            and not isinstance(item, bool)
            for item in value
        )
    ):
        blocks = _tree_evaluation(False)
    elif isinstance(value, BenchmarkResult):
        blocks = _benchmark()
    elif isinstance(value, DiagnosticReport):
        blocks = _diagnostics()
    elif (
        isinstance(value, (list, tuple))
        and value
        and all(type(item).__name__ == "ExampleEntry" for item in value)
    ):
        blocks = _examples()
    else:
        blocks = [("Tables", _GENERIC_COLUMNS)]
    return _render(blocks)


__all__ = ["field_descriptions"]
