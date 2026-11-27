# SPDX-License-Identifier: 0BSD
"""q q~ > W+ W- g: a loop vertex attaches every tree of a field current
(mq-ewjoin; m-qcd rows D28 and D29).

The tree engine splits one field current into DAG currents whose keys
differ only by the flavour flow of their trees: the u current of the legs
{u, W+, W-} of u u~ > w+ w- g is a u -> u line through gamma*/Z* and a
u -> d -> u line through two W's. A loop vertex attaching that current
needs their sum (``assemble._BlockRows``; the native tree program's
``block_row_terms``). Keeping one of them broke the virtual: the double pole
was 0.45 of Catani's and the gluon's Ward identity failed by 0.13 of the
physical scale in double-double; the s- and t-channel trees alone (vetoes)
were right, and so were the 2 -> 2 process (no loop attaches such a
current) and b b~ > w+ w- g on no_b_mass (its t-channel current through the
massive top is a Dirac one, the s-channel one a Weyl one: different keys).

Default tier: u u~ > w+ w- g on loop_sm (the conftest view), one exact
on-shell point in double-double: Catani's poles (Python-input executor) and
the Ward identity of the gluon (native inputs, ``ward.ward_check``). The
MadLoop rows D28 (4F) and D29 (b b~, 5F*) run in the M-QCD campaign.
"""

from __future__ import annotations

from decimal import Decimal, localcontext

import pytest

from pyamplicol.loops.v2 import ward
from pyamplicol.loops.v2.assemble import LoopSlice, Precision, SliceRequest
from pyamplicol.loops.v2.plan_export import NativeExecutor, export_plan

pytestmark = pytest.mark.loop_physics

PARAMETERS = {"aS": "0.118", "MT": "173"}
FIXED = {"WT": "0", "WZ": "0", "WW": "0", "WH": "0"}
MU_R2 = "1000000"
# MW of loop_sm's defaults (derived from MZ, G_F and alpha at tree level).
MW = "80.41900244575616010551713717311353321699"


def _point(mass: str) -> list[list[str]]:
    """sqrt(s) = 1000, the gluon with E = 230 along (2, 3, 6)/7, the W pair
    back to back along (1, -2, 2)/3 in its rest frame, boosted: on shell to
    60 digits."""

    with localcontext() as context:
        context.prec = 60
        root, energy, m = Decimal(1000), Decimal(230), Decimal(mass)
        gluon = [energy, *(energy * Decimal(x) / 7 for x in (2, 3, 6))]
        pair = [root - energy, *(-x for x in gluon[1:])]
        m2 = pair[0] ** 2 - sum(x * x for x in pair[1:])
        q = (m2 / 4 - m * m).sqrt()
        axis = [Decimal(1) / 3, Decimal(-2) / 3, Decimal(2) / 3]
        beta = [x / pair[0] for x in pair[1:]]
        b2 = sum(x * x for x in beta)
        gamma = 1 / (1 - b2).sqrt()

        def boost(k):
            bk = sum(x * y for x, y in zip(beta, k[1:], strict=True))
            factor = (gamma - 1) * bk / b2 + gamma * k[0]
            return [gamma * (k[0] + bk), *(k[i + 1] + factor * beta[i] for i in range(3))]

        e = (q * q + m * m).sqrt()
        legs = [
            [root / 2, 0, 0, root / 2],
            [root / 2, 0, 0, -root / 2],
            boost([e, *(q * x for x in axis)]),
            boost([e, *(-q * x for x in axis)]),
            gluon,
        ]
        return [[format(+Decimal(x), "f") for x in leg] for leg in legs]


def _relative(a, b) -> float:
    a, b = Decimal(str(a)), Decimal(str(b))
    scale = max(abs(a), abs(b))
    return 0.0 if scale == 0 else float(abs(a - b) / scale)


@pytest.fixture(scope="module")
def compiled(loop_sm_compiled_model):
    from pyamplicol.models.loading import load_compiled_model

    return load_compiled_model(loop_sm_compiled_model)


def test_split_field_currents_add_in_the_loops(compiled, loop_native):
    """u u~ > w+ w- g (Born g e^2, loop g^3 e^2), double-double at one point:
    the u current of legs {1, 3, 4} is two tree blocks of one name and
    chirality (flavour flows u and u d u), and the virtual's poles equal
    Catani's within 1e-24 (measured 6e-32); the Ward identity of the gluon
    (epsilon_5 -> c p_5 / E_5, c = 1, i) holds to 1e-25 of the physical
    scale (measured 3e-31; before the fix the double pole was 0.45 of
    Catani's and the Ward residual 0.13 of the scale)."""

    request = SliceRequest(
        "u u~ > w+ w- g",
        {"QCD": 1, "QED": 2},
        {"QCD": 3, "QED": 2},
        fixed_parameters=FIXED,
    )
    slice_ = LoopSlice.generate(compiled, request, loop_native)
    split = [
        block.key
        for block in slice_.blocks.blocks
        if block.key.external_mask == 0b1101 and block.key.chirality == 1
    ]
    assert sorted(key.flavour_flow for key in split) == [(-2,), (-2, -1, -2)]
    assert len({(key.word, key.particle_id, key.coupling_orders) for key in split}) == 1
    point = _point(MW)
    executor = NativeExecutor(
        slice_, forward=False, exported=export_plan(slice_, forward=False)
    )
    (result,) = executor.evaluate(
        [point], [MU_R2], parameters=PARAMETERS, precision=Precision.double_double()
    )
    assert result.ir_poles
    virtual = result.components["virtual"]
    for sector, (single, double) in result.ir_poles.items():
        coefficients = virtual[sector].coefficients
        assert _relative(coefficients[2], double) <= 1e-24, (sector, "double")
        assert _relative(coefficients[1], single) <= 1e-24, (sector, "single")
    ward_slice = ward.ward_slice(compiled, request, loop_native)
    check = ward.ward_check(
        NativeExecutor(ward_slice, inputs="native", forward=False),
        point,
        MU_R2,
        "double-double",
        parameters=PARAMETERS,
    )
    assert [row["leg"] for row in check["rows"]] == [5, 5]
    assert check["passed"] is True, check["rows"]
