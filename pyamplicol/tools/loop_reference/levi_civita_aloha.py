# SPDX-License-Identifier: 0BSD
"""ALOHA reference values of the Levi-Civita counterterm structures.

Compiles the unchanged ALOHA amplitude routines of ``R2_GGZ`` and
``R2_GGGVa`` (MG5_aMC ``loop_sm``, loop mode: explicit complex momenta) and
evaluates them on the exact off-shell inputs ``INPUTS``, printing the values
``tests/unit/test_loop_v2_levi_civita.py`` freezes. The routines return
``-i COUP`` times the Lorentz structure; the phase and coupling are removed,
as in ``ct_tensor_oracle``.

    python tools/loop_reference/levi_civita_aloha.py \\
        --mg5 "$MG5" --build-dir "$BUILD"
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

STRUCTURES = ("R2_GGZ", "R2_GGGVa")

# Per structure: (momenta into the vertex, complex polarisations) per sample.
INPUTS = {
    "R2_GGZ": (
        (
            (
                ("7/3", "-2", "5/3", "1/2"),
                ("-4/3", "3", "-1/6", "2"),
                ("-1", "-1", "-3/2", "-5/2"),
            ),
            (
                (("1/7", "2/11"), ("-3/7", "1/11"), ("5/7", "0"), ("2/7", "-4/11")),
                (("-2/7", "3/11"), ("1/7", "-5/11"), ("3/7", "2/11"), ("-6/7", "1/11")),
                (("4/7", "-1/11"), ("-1/7", "6/11"), ("2/7", "-3/11"), ("5/7", "2/11")),
            ),
        ),
        (
            (
                ("13/2", "1", "-7/3", "4"),
                ("-5/2", "-3", "1/3", "-1"),
                ("-4", "2", "2", "-3"),
            ),
            (
                (("3/5", "-1/3"), ("1/5", "2/3"), ("-4/5", "1/3"), ("2/5", "0")),
                (("-1/5", "1/3"), ("2/5", "-1/3"), ("1/5", "1"), ("-3/5", "2/3")),
                (("1", "1/3"), ("-2/5", "-2/3"), ("3/5", "1/3"), ("1/5", "-1")),
            ),
        ),
    ),
    "R2_GGGVa": (
        (
            (
                ("2", "1", "-1/2", "3"),
                ("-1", "2/3", "4", "-2"),
                ("5/2", "-3", "1", "1/3"),
                ("-7/2", "4/3", "-9/2", "-4/3"),
            ),
            (
                (("1/3", "-2/5"), ("2/3", "1/5"), ("-1/3", "3/5"), ("1", "0")),
                (("-2/3", "1/5"), ("1/3", "-1/5"), ("1", "2/5"), ("-1/3", "4/5")),
                (("0", "3/5"), ("-1", "1/5"), ("2/3", "-2/5"), ("1/3", "1/5")),
                (("4/3", "-1/5"), ("1/3", "2/5"), ("-2/3", "0"), ("1", "-3/5")),
            ),
        ),
        (
            (
                ("-3", "1/2", "2", "-1"),
                ("4", "-5/2", "1", "3/2"),
                ("1/2", "3", "-4", "2"),
                ("-3/2", "-1", "1", "-5/2"),
            ),
            (
                (("2/9", "1/4"), ("-5/9", "3/4"), ("1/9", "-1/4"), ("4/9", "1/2")),
                (("1/9", "-3/4"), ("2/9", "1/4"), ("-7/9", "0"), ("1/3", "1/4")),
                (("-4/9", "1/2"), ("1/9", "-1/4"), ("5/9", "3/4"), ("-2/9", "1/4")),
                (("1/3", "0"), ("-1/9", "1/2"), ("2/9", "-3/4"), ("7/9", "1/4")),
            ),
        ),
    ),
}


def aloha_values(mg5: Path, build_dir: Path) -> dict[str, list[list[float]]]:
    import numpy as np

    sys.dont_write_bytecode = True
    sys.path.insert(0, str(mg5.resolve()))
    build_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(build_dir)  # ALOHA's parser writes its tables to the working directory
    import aloha
    from aloha import create_aloha

    model = create_aloha.AbstractALOHAModel(str(mg5.resolve() / "models" / "loop_sm"))
    model.look_for_symmetries()
    aloha.loop_mode = True
    for name in STRUCTURES:
        routine = create_aloha.AbstractRoutineBuilder(
            getattr(model.model.lorentz, name)
        )
        model.compute_aloha(routine, routines=[(0, [])])
    model.write(str(build_dir), "Fortran")
    library_path = build_dir.resolve() / "aloha_levi_civita.so"
    subprocess.run(
        [
            "gfortran",
            "-O2",
            "-fPIC",
            "-shared",
            *(str(build_dir / f"{name}_0.f") for name in STRUCTURES),
            "-o",
            str(library_path),
        ],
        check=True,
    )
    library = ctypes.CDLL(str(library_path.resolve()))
    out: dict[str, list[list[float]]] = {}
    for name in STRUCTURES:
        values = []
        for momenta, polarizations in INPUTS[name]:
            legs = len(momenta)
            function = getattr(library, name.lower() + "_0_")
            function.argtypes = [ctypes.c_void_p] * (legs + 2)
            function.restype = None
            waves = [
                np.asarray(
                    [*(complex(Fraction(x)) for x in p), *(_complex(x) for x in e)],
                    dtype=np.complex128,
                )
                for p, e in zip(momenta, polarizations, strict=True)
            ]
            coupling = np.asarray([complex(3 / 7, 2 / 5)], dtype=np.complex128)
            result = np.empty(1, dtype=np.complex128)
            function(
                *(a.ctypes.data_as(ctypes.c_void_p) for a in (*waves, coupling, result))
            )
            value = result[0] / (-1j * coupling[0])
            values.append([value.real, value.imag])
        out[name] = values
    return out


def _complex(pair: tuple[str, str]) -> complex:
    return complex(float(Fraction(pair[0])), float(Fraction(pair[1])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mg5", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(aloha_values(args.mg5, args.build_dir), indent=1))
