# SPDX-License-Identifier: 0BSD
"""Evaluate totals and resolved components through the public Runtime facade."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from pyamplicol import Runtime


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("momenta", type=Path)
    parser.add_argument("--process")
    parser.add_argument("--parameters", type=Path)
    parser.add_argument("--helicity", action="append")
    parser.add_argument("--color-flow", action="append")
    return parser


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_complex(value: complex) -> dict[str, float]:
    return {"real": value.real, "imag": value.imag}


def main() -> int:
    args = _parser().parse_args()
    parameters = _read_json(args.parameters) if args.parameters else None
    momenta = _read_json(args.momenta)
    runtime = Runtime.load(
        args.artifact,
        process=args.process,
        model_parameters=parameters,
    )
    selectors = {
        "helicities": args.helicity,
        "color_flows": args.color_flow,
    }
    totals = runtime.evaluate(momenta, **selectors)
    resolved = runtime.evaluate_resolved(momenta, **selectors)
    resolved_totals = resolved.total()
    if len(totals) != len(resolved_totals) or any(
        not math.isclose(total.real, check.real, rel_tol=1e-12, abs_tol=1e-15)
        or not math.isclose(total.imag, check.imag, rel_tol=1e-12, abs_tol=1e-15)
        for total, check in zip(totals, resolved_totals, strict=True)
    ):
        raise RuntimeError("resolved components do not reproduce summed values")

    print(
        json.dumps(
            {
                "color_accuracy": resolved.color_accuracy,
                "helicity_ids": resolved.helicity_ids,
                "color_flow_ids": resolved.color_flow_ids,
                "totals": [_json_complex(value) for value in totals],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
