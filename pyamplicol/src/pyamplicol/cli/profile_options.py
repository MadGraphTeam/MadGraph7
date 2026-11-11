# SPDX-License-Identifier: 0BSD
"""``pyamplicol profile`` options of loop artifacts: what the profile
measures (``--runtime``, ``--stability``; neither: the runtime only) and the
flat points of the stability statistics. The reduction backends compared are
both by default; ``--loop-reduction`` selects one (``auto``: the
process type's default)."""

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
            "over flat phase-space points (binary64 against a double-double "
            "re-run) and the accuracy of the default rotation estimate"
        ),
    )
    parser.add_argument(
        "--stability-points",
        dest="benchmark.stability_points",
        type=int,
        default=argparse.SUPPRESS,
        metavar="COUNT",
        help="loop artifacts: number of flat points of --stability",
    )
    parser.add_argument(
        "--stability-seed",
        dest="benchmark.stability_seed",
        type=int,
        default=argparse.SUPPRESS,
        metavar="SEED",
        help=(
            "loop artifacts: seed of the first flat point of --stability "
            "(default: generation.validation.seed)"
        ),
    )
    parser.add_argument(
        "--sqrt-s",
        dest="benchmark.sqrt_s",
        type=float,
        default=argparse.SUPPRESS,
        metavar="ENERGY",
        help=(
            "loop artifacts: centre-of-mass energy of the flat points of "
            "--stability (default: the artifact's generation energy)"
        ),
    )


__all__ = ["add_profile_mode_options"]
