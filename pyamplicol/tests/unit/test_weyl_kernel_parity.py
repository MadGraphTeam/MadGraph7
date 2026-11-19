# SPDX-License-Identifier: 0BSD
"""The exact Weyl kernels agree with the native binary64 ones at both energy signs.

The recurrence-export golden vectors pin the native ``weyl`` source bits for
every (orientation, helicity, chirality) at positive and negative energies.
Both implementations derive helicity = chirality * sgn(E), so the exact Decimal
spinors, rounded to binary64, must reproduce them.
"""

from __future__ import annotations

import json
import struct
from decimal import Decimal
from pathlib import Path

import pytest

from pyamplicol.runtime.symbolica_exact import _antiquark_weyl, _quark_weyl

_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "rust/crates/rusticol-core/tests/fixtures/recurrence_export_golden.json"
)


def _floats(text: str) -> list[float]:
    return [
        struct.unpack(">d", bytes.fromhex(text[index : index + 16]))[0]
        for index in range(0, len(text), 16)
    ]


def _cases() -> list[dict[str, object]]:
    if not _FIXTURE.exists():
        return []
    golden = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    return [
        {**case, "forms": golden["momentum_forms"]}
        for case in golden["cases"]
        if case["name"].startswith("source weyl")
    ]


@pytest.mark.parametrize("case", _cases(), ids=lambda case: str(case["name"]))
def test_exact_weyl_sources_match_the_native_golden_bits(
    case: dict[str, object],
) -> None:
    (source,) = case["sources"]  # type: ignore[misc]
    form = [_floats(component) for component in case["forms"][source["momentum_form"]]]  # type: ignore[index]
    kernel = _antiquark_weyl if source["orientation"] == "antiparticle" else _quark_weyl
    factor = complex(source["factor"]["re"], source["factor"]["im"])
    current = int(source["current"])
    expected = [_floats(case["expected"]["currents"][str(current + c)]) for c in (0, 1)]  # type: ignore[index]
    for point in range(len(form[0])):
        momentum = [Decimal(form[component][point]) for component in range(4)]
        spinor = kernel(momentum, int(source["helicity"]), int(source["chirality"]))
        for component, (re, im) in enumerate(spinor):
            value = complex(float(re), float(im)) * factor
            native = complex(*expected[component][2 * point : 2 * point + 2])
            scale = 1.0 + abs(native)
            assert abs(value - native) <= 2.0e-15 * scale, (point, component)
            assert (value == 0) == (native == 0)
