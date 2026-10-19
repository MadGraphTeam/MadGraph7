#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""One native bulk virtual evaluation through the installed Python API."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pyamplicol import Runtime
from pyamplicol.api.errors import EvaluationError
from pyamplicol.api.virtual_results import VirtualArithmeticSpec, VirtualStabilityPolicy
from pyamplicol.cli.main import write_result


def exact_json(path):
    return json.loads(Path(path).read_text(), parse_float=str, parse_int=str)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("process")
    parser.add_argument("mu_r2")
    parser.add_argument("--resolved", action="store_true")
    parser.add_argument(
        "--arithmetic", choices=("double", "double-double", "arbitrary")
    )
    parser.add_argument("--precision", type=int)
    parser.add_argument("--format", choices=("json", "human"), default="json")
    parser.add_argument("--parameters", help="exact model-parameter JSON file")
    parser.add_argument("--stability", help="explicit verification/rescue policy JSON")
    args = parser.parse_args()
    lines = Path("API/validation_points.dat").read_text().splitlines()
    if not lines or lines[0] != "RUSTICOL_VALIDATION_POINTS_V1":
        parser.error("invalid validation-point file")
    points = []
    for line in lines[1:]:
        fields = line.split()
        if not fields or fields[0] != args.process:
            continue
        n = int(fields[1])
        if len(fields) != 2 + 4 * n:
            parser.error("invalid validation-point dimensions")
        points.append([fields[2 + 4 * i : 6 + 4 * i] for i in range(n)])
    if not points:
        parser.error("no validation points for the selected process")
    policy = None
    if args.stability:
        raw = json.loads(Path(args.stability).read_text(), parse_float=str)
        policy = VirtualStabilityPolicy(
            tuple(VirtualArithmeticSpec(**item) for item in raw.pop("higher")), **raw
        )
    runtime = Runtime.load(
        ".",
        process=args.process,
        arithmetic=args.arithmetic,
        precision=args.precision,
        model_parameters=exact_json(args.parameters) if args.parameters else None,
    )
    operation = runtime.evaluate_resolved if args.resolved else runtime.evaluate
    try:
        result = operation(points, mu_r2=args.mu_r2, stability=policy)
    except EvaluationError as error:
        # Failed calls have no numerical result; retain original lane identities
        # and verification attempts instead of printing placeholder coefficients.
        write_result(
            {
                "error": str(error),
                "failures": error.failures,
                "verification": error.verification,
            },
            format="json",
            stream=sys.stderr,
        )
        return 1
    # The public formatter retains every order/pole/colour axis and diagnostics.
    write_result(result, format=args.format, stream=sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
