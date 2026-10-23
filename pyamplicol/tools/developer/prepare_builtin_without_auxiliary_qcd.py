#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Prepare a diagnostic built-in model without auxiliary four-gluon kernels.

This intentionally incomplete model is useful only for locating recurrence
parity failures. It removes the trivalent auxiliary-tensor decomposition of the
four-gluon interaction while retaining the rest of the built-in model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from recurrence_z6g_benchmark import _generation_config

from pyamplicol import ModelSource
from pyamplicol.models.builtin.definitions import BuiltinSMDefinitionMixin


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    output = arguments.output.expanduser().resolve()

    original = BuiltinSMDefinitionMixin._build_vertices

    def without_auxiliary_vertices(
        model: BuiltinSMDefinitionMixin,
    ) -> list[object]:
        return [
            vertex
            for vertex in original(model)
            if int(vertex.kind) not in {1, 2, 3}
        ]

    BuiltinSMDefinitionMixin._build_vertices = without_auxiliary_vertices
    try:
        ModelSource.built_in_sm().compile(
            use_cache=False,
            prepared_output=output,
            evaluator=_generation_config(
                "recurrence",
                validation_samples=1,
            ).evaluator,
        )
    finally:
        BuiltinSMDefinitionMixin._build_vertices = original
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
