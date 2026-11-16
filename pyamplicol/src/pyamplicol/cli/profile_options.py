# SPDX-License-Identifier: 0BSD
"""``pyamplicol profile`` options of loop artifacts: what the profile
measures (``--runtime``, ``--stability``; neither: the runtime only), the
number of flat points of the stability statistics and the seed and energy of
the flat points (``evaluation.phase_space``, ``cli.flat_points``). The
reduction backends compared are both by default; ``--loop-reduction``
selects one (``auto``: the process type's default)."""

from __future__ import annotations

import argparse


def add_profile_mode_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--runtime",
        dest="benchmark.runtime",
        action="store_true",
        default=argparse.SUPPRESS,
        help=(
            "loop artifacts: time per point and per stage of each reduction "
            "backend (the default when neither --runtime nor --stability is given)"
        ),
    )
    parser.add_argument(
        "--stability",
        dest="benchmark.stability",
        action="store_true",
        default=argparse.SUPPRESS,
        help=(
            "loop artifacts: stable-digit statistics of each reduction backend "
            "over flat phase-space points (binary64 against a re-run in the "
            "--ground-truth precision) and the accuracy of the default rotation "
            "estimate"
        ),
    )
    parser.add_argument(
        "--stability-points",
        dest="benchmark.stability_points",
        type=int,
        default=argparse.SUPPRESS,
        metavar="COUNT",
        help=(
            "loop artifacts: number of flat points of --stability (default: "
            "what one minute per --n_core worker affords, at least 5)"
        ),
    )
    parser.add_argument(
        "--n_core",
        "--n-core",
        dest="benchmark.n_core",
        type=int,
        default=argparse.SUPPRESS,
        metavar="N",
        help=(
            "loop artifacts: worker processes of --stability only (default "
            "10, at most the usable cores; 1 runs in-process; the result does "
            "not depend on it, each worker runs one thread); the runtime "
            "timing runs on one core unless --runtime-threads is given"
        ),
    )
    parser.add_argument(
        "--runtime-threads",
        dest="benchmark.runtime_threads",
        type=int,
        default=argparse.SUPPRESS,
        metavar="N",
        help=(
            "loop artifacts: native threads of each runtime timing call "
            "(benchmark.runtime_threads; default 1: the time per point of one "
            "core, what a Monte Carlo worker pays). N threads work inside each "
            "call, the calls stay one at a time; the binary64 calls and the "
            "reference pass, with a short one-thread timing for the gain. At "
            "most the usable cores. The --stability workers stay at one "
            "thread each"
        ),
    )
    parser.add_argument(
        "--ground-truth",
        dest="benchmark.ground_truth",
        default=argparse.SUPPRESS,
        metavar="{double-double,arbitrary[:DIGITS],none}",
        help=(
            "loop artifacts: the reference of both tables (benchmark."
            "ground_truth): the stability run's true values and the "
            "reference-precision time multipliers; double-double (dd, the "
            "default), arbitrary precision (arb) at DIGITS decimal digits "
            "(default: the built-in rescue's, 57), or none: no reference "
            "pass, the stability run reports its estimates only"
        ),
    )
    parser.add_argument(
        "--completion-time",
        dest="benchmark.completion_time",
        default=argparse.SUPPRESS,
        metavar="DURATION",
        help=(
            "loop artifacts: the wall time the whole profile aims at "
            "(benchmark.completion_time; seconds, or 90s, 10m, 1h30m); it "
            "overrules every count (--target-runtime, --minimum-samples, "
            "--batch-size, --warmup-runs, --stability-points), never below "
            "one timed call per backend and five stability points. Without "
            "it: the counts given, else 10 s for the runtime timing and one "
            "minute per --n_core worker for the stability run"
        ),
    )
    parser.add_argument(
        "--samples-output",
        dest="benchmark.samples_output",
        default=argparse.SUPPRESS,
        metavar="PATH",
        help=(
            "loop artifacts: write every stability sample's record (inputs, "
            "values, estimates, reference, true error, task timings; "
            "columnar JSON) to PATH (benchmark.samples_output); --json then "
            "names the file instead of holding the records"
        ),
    )
    parser.add_argument(
        "--seed",
        dest="evaluation.phase_space.seed",
        type=int,
        default=argparse.SUPPRESS,
        metavar="SEED",
        help=(
            "seed of the first flat phase-space point (evaluation.phase_space."
            "seed): the timed points without --momenta and the --stability points"
        ),
    )
    parser.add_argument(
        "--sqrt-s",
        dest="evaluation.phase_space.sqrt_s",
        type=float,
        default=argparse.SUPPRESS,
        metavar="ENERGY",
        help=(
            "centre-of-mass energy of the flat phase-space points "
            "(evaluation.phase_space.sqrt_s; default: the artifact's "
            "validation-point energy)"
        ),
    )


__all__ = ["add_profile_mode_options"]
