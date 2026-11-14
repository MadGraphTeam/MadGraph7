# SPDX-License-Identifier: 0BSD
"""Fixture of ``tests/correlated_precision.rs``: artifacts and Python references.

Usage, from the repository root with the development environment::

    PYTHONPATH=src python \
        rust/crates/rusticol-core/tests/fixtures/correlated_precision.py DIR

writes into DIR

* ``builtin/``: the built-in SM processes ``u d > u d g``, ``u u~ > g g g`` and
  ``g g > g g g`` at full colour (compiled), declaring every dipole
  ``T_k . T_l`` (k < l, id ``Tkl``) and every gluon leg as a spin class;
* ``ufo/``: ``u d > u d``, ``g g > g g`` and ``g g > g g g`` of the UFO
  ``models/json/sm`` (restriction ``no_b_mass``, QCD only, full colour,
  compiled), whose couplings are derived model parameters;
* ``heft/``: ``g g > h`` of the built-in SM-HEFT (compiled), whose coupling
  G_H is a derived model parameter;
* ``ufo_w/``: ``u d~ > w+ g``, ``u d~ > w+ h`` and ``u d~ > w+ h g`` of the
  same UFO model (compiled), whose W mass is a derived model parameter (G_mu
  scheme);
* ``reference.json``: the exact integer points and, per built-in process and
  point, the Python exact correlated path (``Runtime.evaluate_correlated_many``,
  arbitrary precision) at ``REFERENCE_DIGITS`` digits, divided by the Python
  executor's exact normalization, for the Born, every dipole, two transverse
  spin vectors of every gluon leg and one complex spin vector combined with a
  dipole.
"""

from __future__ import annotations

import json
import sys
from decimal import Decimal, localcontext
from pathlib import Path

import pyamplicol
from pyamplicol import (
    ColorCorrelator,
    CorrelatedRequest,
    CorrelatorConfig,
    Generator,
    ModelSource,
    ProcessSet,
    Runtime,
)
from pyamplicol.config import (
    ColorConfig,
    EvaluatorConfig,
    GenerationConfig,
    ProcessConfig,
    RunConfig,
)

REFERENCE_DIGITS = 100

# Exact massless points: beams (E, 0, 0, +-E), outgoing integer momenta.
POINTS_2TO3 = (
    (650, ((390, -360, -120, -90), (380, -120, 200, 300), (530, 480, -80, -210))),
    (540, ((390, -360, -120, -90), (250, 120, -160, -150), (440, 240, 280, 240))),
    (490, ((390, -360, -120, -90), (340, 160, 240, 180), (250, 200, -120, -90))),
)
POINTS_2TO2 = (
    (700, ((700, 200, 300, 600), (700, -200, -300, -600))),
    (450, ((450, 50, 200, 400), (450, -50, -200, -400))),
    (2200, ((2200, 1200, 400, -1800), (2200, -1200, -400, 1800))),
)

BUILTIN = {
    "ud_udg": ("u d > u d g", (5,)),
    "uux_ggg": ("u u~ > g g g", (3, 4, 5)),
    "gg_ggg": ("g g > g g g", (1, 2, 3, 4, 5)),
}
UFO = {
    "ud_ud": "u d > u d",
    "gg_gg": "g g > g g",
    "gg_ggg": "g g > g g g",
}


def momenta(energy, outgoing):
    return ((energy, 0, 0, energy), (energy, 0, 0, -energy), *outgoing)


def cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def transverse_vectors(momentum):
    """Two orthogonal integer 3-vectors orthogonal to the momentum, as
    four-vectors with zero energy (not normalized)."""
    direction = momentum[1:]
    axis = (0, 1, 0) if direction[0] == direction[1] == 0 else (0, 0, 1)
    first = cross(direction, axis)
    second = cross(direction, first)
    return ((0, *first), (0, *second))


def requests(legs, spin_legs, point):
    """Request id -> (colour id, {leg: four (real, imaginary) strings})."""
    result = {"born": ("born", {})}
    for left in range(1, legs + 1):
        for right in range(left + 1, legs + 1):
            result[f"T{left}{right}"] = (f"T{left}{right}", {})
    for leg in spin_legs:
        for index, vector in enumerate(transverse_vectors(point[leg - 1])):
            result[f"spin{leg}_{index}"] = (
                "born",
                {leg: tuple((str(component), "0") for component in vector)},
            )
    # A complex (circular) vector with a dipole: e_1 + i e_2.
    leg = spin_legs[0]
    first, second = transverse_vectors(point[leg - 1])
    result[f"spin{leg}_complex_T12"] = (
        "T12",
        {leg: tuple((str(a), str(b)) for a, b in zip(first, second, strict=True))},
    )
    return result


def generate(expressions, output, model, correlators=None, max_coupling_orders=None):
    config = RunConfig(
        action="generate",
        color=ColorConfig(accuracy="full"),
        evaluator=EvaluatorConfig(backend="jit", execution_mode="compiled"),
        generation=GenerationConfig(workers=1, emit_api_bundle=False),
        process=(
            ProcessConfig()
            if max_coupling_orders is None
            else ProcessConfig(
                coupling_order_policy="explicit",
                max_coupling_orders=max_coupling_orders,
            )
        ),
    )
    names = tuple(expressions)
    Generator(config).generate(
        ProcessSet.from_expressions(tuple(expressions.values()), names=names),
        output,
        model=model,
        correlators=correlators,
    )


def main(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=False)
    for name, (expression, spin_legs) in BUILTIN.items():
        legs = len(expression.split()) - 1
        declarations = CorrelatorConfig(
            color_correlations=tuple(
                ColorCorrelator.dipole(f"T{left}{right}", left, right)
                for left in range(1, legs + 1)
                for right in range(left + 1, legs + 1)
            ),
            spin_correlations=tuple((leg,) for leg in spin_legs),
        )
        generate(
            {name: expression},
            root / "builtin" / name,
            ModelSource.built_in_sm(),
            declarations,
        )
    ufo_root = Path(pyamplicol.__file__).parent / "assets/models/json/sm"
    generate(
        UFO,
        root / "ufo",
        ModelSource.from_path(ufo_root / "sm.json", restriction="no_b_mass"),
        max_coupling_orders={"QED": 0},
    )
    generate(
        {"gg_h": "g g > h"},
        root / "heft",
        ModelSource.built_in_sm_heft(),
        max_coupling_orders={"QCD": 0, "QED": 0, "HIG": 1},
    )
    generate(
        {
            "udx_wpg": "u d~ > w+ g",
            "udx_wph": "u d~ > w+ h",
            "udx_wphg": "u d~ > w+ h g",
        },
        root / "ufo_w",
        ModelSource.from_path(ufo_root / "sm.json", restriction="no_b_mass"),
        max_coupling_orders={"QCD": 1, "QED": 2},
    )
    reference = {
        "digits": REFERENCE_DIGITS,
        "points_2to3": [momenta(e, outgoing) for e, outgoing in POINTS_2TO3],
        "points_2to2": [momenta(e, outgoing) for e, outgoing in POINTS_2TO2],
        "builtin": {},
    }
    for name, (expression, spin_legs) in BUILTIN.items():
        runtime = Runtime.load(root / "builtin" / name, process=name)
        executor = runtime._correlator_evaluator()
        entries = []
        for point in reference["points_2to3"]:
            selected = requests(len(point), spin_legs, point)
            values = runtime.evaluate_correlated_many(
                (point,),
                {
                    key: CorrelatedRequest(colour, vectors)
                    for key, (colour, vectors) in selected.items()
                },
                precision=REFERENCE_DIGITS,
            )
            normalization = _python_normalization(executor, point)
            with localcontext() as context:
                context.prec = REFERENCE_DIGITS + 10
                entries.append(
                    {
                        "point": point,
                        "requests": [
                            {
                                "id": key,
                                "color_correlation": colour,
                                "spin_vectors": {
                                    str(leg): [list(component) for component in vector]
                                    for leg, vector in vectors.items()
                                },
                                "raw": [
                                    str(Decimal(values[key][0].real) / normalization),
                                    str(Decimal(values[key][0].imag) / normalization),
                                ],
                            }
                            for key, (colour, vectors) in selected.items()
                        ],
                    }
                )
        reference["builtin"][name] = {
            "expression": expression,
            "spin_legs": list(spin_legs),
            "entries": entries,
        }
        print(name, "done", flush=True)
    (root / "reference.json").write_text(json.dumps(reference, indent=1) + "\n")


def _python_normalization(executor, point):
    """The exact normalization the Python executor multiplies every value by."""
    exact = executor._executor if hasattr(executor, "_executor") else executor
    batch = exact.coherent_amplitudes((point,), precision=REFERENCE_DIGITS)
    return Decimal(batch.normalization_factor)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
