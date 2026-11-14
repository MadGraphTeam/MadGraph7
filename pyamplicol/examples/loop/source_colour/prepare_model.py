# SPDX-License-Identifier: 0BSD
"""Prepare the supplied MadLoop 3.7.0 source with explicit QCD colour model data.

This example input declares only the source entries used by the validated
Drell-Yan, four-gluon and outgoing W+ plus gluon QCD controls. It is not a
complete SM continuation.
"""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from pyamplicol.models.loading import compile_model_source
from pyamplicol.models.loop_color_declaration import (
    LoopColorDeclaration,
    with_loop_color_declaration,
)
from pyamplicol.models.loop_declaration import (
    LoopModelDeclaration,
    with_loop_model_declaration,
)


def _scalar_arity_view():
    """The checkout's source-compatibility view (``tools/loop_reference``,
    test support: neither the wheel nor the sdist ships it)."""

    root = Path(__file__).resolve().parents[3]
    if not (root / "tools/loop_reference/model_compatibility.py").is_file():
        raise SystemExit(
            "prepare_model.py runs from a pyAmpliCol source checkout: it uses "
            "the checkout's tools/loop_reference/model_compatibility.py"
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from tools.loop_reference.model_compatibility import (
        prepare_loop_sm_scalar_declarations,
    )

    return prepare_loop_sm_scalar_declarations


def prepare(source: Path, output: Path) -> Path:
    prepare_loop_sm_scalar_declarations = _scalar_arity_view()
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
