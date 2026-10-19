# SPDX-License-Identifier: 0BSD
"""Prepare the supplied MadLoop 3.7.0 source with explicit QCD colour model data.

This example input declares only the source entries used by the validated
Drell-Yan and four-gluon QCD controls. It is not a complete SM continuation.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from pyamplicol.loops.model_compatibility import prepare_loop_sm_scalar_declarations
from pyamplicol.models.loading import compile_model_source
from pyamplicol.models.loop_color_declaration import (
    LoopColorDeclaration,
    with_loop_color_declaration,
)
from pyamplicol.models.loop_declaration import (
    LoopModelDeclaration,
    with_loop_model_declaration,
)


def prepare(source: Path, output: Path) -> Path:
    model = compile_model_source(source, cache_dir=output.parent / ".model-cache")
    if model.raw_loop_model is None:
        raise ValueError("the source model has no lossless NLO source snapshot")
    # Explicit known source repair; this verifies the actual scalar declaration.
    raw = prepare_loop_sm_scalar_declarations(model.raw_loop_model).derived
    model = replace(
        model,
        raw_loop_model=raw,
        _serialized_path=None,
        _prepared_bundle=None,
    )
    data = LoopColorDeclaration.from_dict(
        json.loads(Path(__file__).with_name("loop_sm_qcd_subset.json").read_text())
    )
    model = with_loop_model_declaration(
        model, LoopModelDeclaration(data.scheme, ("QCD",), True, True)
    )
    model = with_loop_color_declaration(model, data)
    return model.write(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", type=Path, help="supplied MadLoop 3.7.0 models/loop_sm"
    )
    parser.add_argument("output", type=Path, help="compiled model JSON destination")
    args = parser.parse_args()
    print(prepare(args.source, args.output))
