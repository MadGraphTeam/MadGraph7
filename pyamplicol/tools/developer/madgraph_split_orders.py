#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Squared coupling-order references from MadGraph standalone ``split orders``.

MadGraph exports per-squared-order Born matrix elements
(``SMATRIX_SPLITORDERS``) when a process carries squared-order constraints.
This tool generates one standalone output per process, links a small driver
that reads fixed momenta, and records ``ANS(0:NSQSO)`` with the squared-order
labels of the generated ``SQSPLITORDERS`` table. MadGraph's inverse label
getter reads a misspelled COMMON in v3.7, so labels come from the table.

Example::

    python tools/developer/madgraph_split_orders.py \\
        --madgraph "$WORKSPACE/reference/sources/MG5_aMC_v3_7_0" \\
        --points points.json --work work --output references.json \\
        --process "u u~ > d d~" --process "g g > t t~"

``points.json`` maps each process to a list of points, each a list of
``[E, px, py, pz]`` momenta in MadGraph's leg order. T-channel widths are kept
(``zerowidth_tchannel False``) so propagators match a fixed-width recursion.

``--colour lc`` records pyAmpliCol's leading colour instead: the diagonal of
MadGraph's colour matrix, off-diagonal entries set to zero. That is the
colour-flow leading colour only where MadGraph's colour basis is the colour-flow
basis, products of Kronecker deltas ``T(i,j)``, so it is refused for processes
with external gluons.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

_SPLIT_ORDER_NAMES = ("QCD", "QED")
_DATA_RE = re.compile(r"DATA \(SQSPLITORDERS\(\s*(\d+),I\),I=\s*1,\s*\d+\) /([^/]*)/")
# One packed colour-matrix row and the basis element it belongs to.
_CF_RE = re.compile(
    r"(DATA \(CF\(I\),I=\s*\d+,\s*\d+\) /)([^/]*)(/\n)((?:C\s+-?\d.*\n)+)"
)
_DELTA_RE = re.compile(r"T\(\d+,\d+\)")
_DRIVER = """      PROGRAM PYAMPLICOL_SPLIT_ORDERS
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INTEGER NSQSO, NPOINTS, I, J, K
      REAL*8 P(0:3,NEXTERNAL), ANS(0:999)
      CALL SETPARA('../../Cards/param_card.dat')
      CALL GET_NSQSO_BORN(NSQSO)
      OPEN(10, FILE='pyamplicol_momenta.dat', STATUS='OLD')
      READ(10,*) NPOINTS
      DO K=1,NPOINTS
        DO I=1,NEXTERNAL
          READ(10,*) (P(J,I),J=0,3)
        ENDDO
        CALL SMATRIX_SPLITORDERS(P,ANS)
        WRITE(*,'(A,1000ES26.17E3)') 'PYAMPLICOL_MG_ORDERS ',
     $   (ANS(J),J=0,NSQSO)
      ENDDO
      END
"""


def _run(args: list[str], cwd: Path) -> str:
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(
            f"command failed ({result.returncode}): {' '.join(args)}\n"
            f"{result.stdout[-4000:]}\n{result.stderr[-4000:]}"
        )
    return result.stdout


def _generate(madgraph: Path, model: str, process: str, output: Path) -> Path:
    constraints = " ".join(f"{name}^2<=99" for name in _SPLIT_ORDER_NAMES)
    card = output.with_suffix(".mg5")
    card.write_text(
        "\n".join(
            (
                "set automatic_html_opening False",
                "set auto_update 0",
                "set zerowidth_tchannel False",
                f"import model {model}",
                f"generate {process} {constraints}",
                f"output standalone {output} -f",
                "",
            )
        )
    )
    _run(["python3", str(madgraph / "bin" / "mg5_aMC"), str(card)], output.parent)
    (subprocess_dir,) = sorted((output / "SubProcesses").glob("P*_*"))
    return subprocess_dir


def _squared_orders(matrix: str) -> list[dict[str, int]]:
    rows = {
        int(index): [int(value) for value in values.split(",")]
        for index, values in _DATA_RE.findall(matrix)
    }
    if not rows or sorted(rows) != list(range(1, len(rows) + 1)):
        raise SystemExit("MadGraph split-order table is missing or not dense")
    return [
        dict(zip(_SPLIT_ORDER_NAMES, rows[index], strict=True))
        for index in range(1, len(rows) + 1)
    ]


def _leading_colour(matrix: str) -> str:
    """Keep the diagonal of the packed colour matrix (one DATA row per basis
    element: diagonal entry first, then the doubled off-diagonal entries)."""

    rows = _CF_RE.findall(matrix)
    if not rows:
        raise SystemExit("MadGraph colour matrix not found")
    for _head, _values, _end, basis in rows:
        for line in basis.splitlines():
            factors = line[1:].split()[1:]
            if not factors or not all(_DELTA_RE.fullmatch(f) for f in factors):
                raise SystemExit(
                    "leading colour needs a basis of Kronecker deltas, got "
                    + line.strip()
                )

    def diagonal(match: re.Match[str]) -> str:
        values = match.group(2).split(",")
        kept = [values[0]] + [re.sub(r"-?\d+", "0", value) for value in values[1:]]
        return match.group(1) + ",".join(kept) + match.group(3) + match.group(4)

    return _CF_RE.sub(diagonal, matrix)


def _evaluate(
    subprocess_dir: Path, points: list[list[list[float]]]
) -> list[list[float]]:
    (subprocess_dir / "pyamplicol_split_orders.f").write_text(_DRIVER)
    momenta = [str(len(points))]
    momenta.extend(
        " ".join(repr(float(value)) for value in momentum)
        for point in points
        for momentum in point
    )
    (subprocess_dir / "pyamplicol_momenta.dat").write_text("\n".join(momenta) + "\n")
    _run(["make", "matrix.o"], subprocess_dir)
    _run(
        [
            "gfortran",
            "-O2",
            "-o",
            "pyamplicol_split_orders",
            "pyamplicol_split_orders.f",
            "matrix.o",
            "-L../../lib",
            "-ldhelas",
            "-lmodel",
        ],
        subprocess_dir,
    )
    stdout = _run(["./pyamplicol_split_orders"], subprocess_dir)
    rows = [
        [float(value) for value in line.split()[1:]]
        for line in stdout.splitlines()
        if line.startswith("PYAMPLICOL_MG_ORDERS")
    ]
    if len(rows) != len(points):
        raise SystemExit("MadGraph driver did not evaluate every point")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--madgraph", type=Path, required=True)
    parser.add_argument("--model", default="sm")
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--process", action="append", required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--colour", choices=("full", "lc"), default="full")
    options = parser.parse_args(argv)
    points = json.loads(options.points.read_text())
    options.work.mkdir(parents=True, exist_ok=True)
    references = []
    for index, process in enumerate(options.process):
        subprocess_dir = _generate(
            options.madgraph.resolve(),
            options.model,
            process,
            options.work.resolve() / f"process-{index}",
        )
        matrix = subprocess_dir / "matrix.f"
        orders = _squared_orders(matrix.read_text())
        if options.colour == "lc":
            matrix.write_text(_leading_colour(matrix.read_text()))
        values = _evaluate(subprocess_dir, points[process])
        if any(len(row) != len(orders) + 1 for row in values):
            raise SystemExit("MadGraph split-order output has the wrong width")
        references.append(
            {
                "process": process,
                "squared_orders": orders,
                "points": points[process],
                "total": [row[0] for row in values],
                "partials": [row[1:] for row in values],
            }
        )
    options.output.write_text(
        json.dumps(
            {
                "kind": "pyamplicol-madgraph-split-orders-v1",
                "madgraph": options.madgraph.name,
                "model": options.model,
                "colour": options.colour,
                "zerowidth_tchannel": False,
                "processes": references,
            },
            indent=1,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
