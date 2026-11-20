# SPDX-License-Identifier: 0BSD
"""The Levi-Civita sign test (WP3.2 EWI-7b, EW-20; PLAN [U6](d)).

``u u~ > e- e+ g g [virt=QCD]`` with ``particle_veto = [d, s, b]``: the closed
quark loops keep the up-type flavours only, so neither the axial parts of the
bare loops with a Z (the 4D numerator's gamma5 traces) nor the axial Z-gluon
R2 (``R2_GGZ``, ``R2_GGGVa``: the UFO ``Epsilon``, lowered as tree kernels,
``models.tensors.LEVI_CIVITA``) cancel between the members of a doublet, as
they do without the veto (``test_loop_v2_ct_cancellation``). Bare and R2 are
compared separately against MadLoop with ``/ d s b``
(``tests/fixtures/loop/epsilon-sign.json``, ``tools/loop_reference/
madloop_top_pair_model.py --set epsilon-sign``): their relative sign is the
point, and no identity of the full sum is required (MASTER_PLAN §4.9).

The counterterms come from the recursion (production: the uncancelled
Levi-Civita cells lowered into the tree model) and, as the interim pre-check
and oracle (O6), from the site closure (the catalogue's Levi-Civita kernels).

The same request also checks the sign without a provider: the up-type quark
triangle breaks the gluon Ward identity of the bare loops (the anomaly), and
the Levi-Civita R2 restores it; with the opposite sign the virtual's Ward
residual would be twice the R2 part. The test caught the root closure's
reading of these vertices (``ct_sites.SiteClosure._reading_sign``): the site
route had the opposite sign of the tree engine, MadLoop and the Ward
identity.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "loop" / "epsilon-sign.json"
)
# |PDG| codes of d, s and b (MadLoop's ``/ d s b``): an anomalous veto (it
# breaks the quark doublets), which the requests take with the override
# ``allow_anomalous_veto`` ([U2]; refused by default).
VETO = (1, 3, 5)
TOLERANCE = 1e-12
# Points evaluated at 32 digits rather than binary64 (the Python reference
# path has no DoubleFloat carrier): point 0's bare double and single poles
# carry 3.7e-12 and 1.1e-12 of binary64 error of pyAmpliCol's own (its exact
# Catani double pole -(alpha_s/2 pi)(2 C_F + 2 C_A) Born is met to 3.0e-12 in
# binary64 and 4e-14 at 32 digits; MadLoop's to 8.3e-13), above the 1e-12
# every coefficient is held to.
HIGH_PRECISION_POINTS = frozenset({0})


def _laurent(result, component: str) -> list[float]:
    total = [0.0, 0.0, 0.0]
    for laurent in result.components[component].values():
        total = [a + float(b) for a, b in zip(total, laurent.coefficients, strict=True)]
    return total


@pytest.mark.loop_nightly
@pytest.mark.parametrize("counterterms", ("recursion", "sites"))
def test_epsilon_sign_bare_and_r2_equal_madloop(
    loop_sm_compiled_model, loop_native, counterterms
):
    """Bare and R2, each per Laurent coefficient, against MadLoop ``/ d s b``
    within 1e-12 of the coefficient (binary64; point 0 at 32 digits,
    ``HIGH_PRECISION_POINTS``). The vetoed request's R2 differs from the
    unvetoed one (``dilepton-two-gluon.json``, same points and scales): the
    d, s, b loops are gone and the axial cells no longer cancel."""

    from pyamplicol.loops.v2.assemble import LoopSlice, Precision, SliceRequest
    from pyamplicol.models.loading import load_compiled_model

    fixture = json.loads(FIXTURE.read_text())
    unvetoed = json.loads((FIXTURE.parent / "dilepton-two-gluon.json").read_text())
    assert fixture["points"] == unvetoed["points"]
    assert fixture["mu_r"] == unvetoed["mu_r"]
    slice_ = LoopSlice.generate(
        load_compiled_model(loop_sm_compiled_model),
        SliceRequest(
            "u u~ > e- e+ g g",
            {"QCD": 2, "QED": 2},
            {"QCD": 4, "QED": 2},
            fixed_parameters=fixture["fixed_parameters"],
            particle_veto=VETO,
            allow_anomalous_veto=True,
            counterterms=counterterms,
        ),
        loop_native,
    )
    parameters = {k: v for k, v in fixture["parameters"].items() if k != "MU_R"}
    for index, (point, mu, row) in enumerate(
        zip(fixture["points"], fixture["mu_r"], fixture["rows"]["uu"], strict=True)
    ):
        result = slice_.evaluate(
            point,
            str(int(mu) ** 2),
            parameters=parameters,
            precision=(
                Precision.arbitrary(32)
                if index in HIGH_PRECISION_POINTS
                else Precision.double()
            ),
        )
        born = sum(float(x) for x in result.born.values())
        assert abs(born - row["born"]) <= TOLERANCE * abs(row["born"])
        for component in ("bare", "r2"):
            reference = row["components"][component]
            values = _laurent(result, component)
            for value, key in zip(
                values, ("finite", "single_pole", "double_pole"), strict=True
            ):
                bound = TOLERANCE * abs(reference[key])
                assert abs(value - reference[key]) <= bound, (
                    index,
                    component,
                    key,
                    value,
                    reference[key],
                )
        # The veto matters: the R2 without the d, s and b loops differs.
        other = unvetoed["rows"][index]["components"]["r2"]["finite"]
        assert not math.isclose(row["components"]["r2"]["finite"], other, rel_tol=1e-6)


# ``name -> (process, Born orders, Ward legs)``: the Levi-Civita R2 inside
# the counterterm program's trees (u u~ > e- e+ g g) and at its root, the
# gluon of leg 1 (g u > u e- e+: the root closure's reading sign,
# ``ct_sites.SiteClosure._reading_sign``; without it this identity fails by
# twice the R2 part, 86 and 53 times the physical scale).
WARD = {
    "uux_emepgg": ("u u~ > e- e+ g g", {"QCD": 2, "QED": 2}, (5, 6)),
    "gu_uemep": ("g u > u e- e+", {"QCD": 1, "QED": 2}, (1,)),
}


@pytest.mark.loop_physics
@pytest.mark.parametrize("name", tuple(WARD))
def test_epsilon_sign_restores_the_gluon_ward_identity(
    loop_sm_compiled_model, loop_native, name
):
    """O10 on each external gluon (epsilon_j -> c p_j on the loop side, c = 1
    and c = i; the recursion's counterterms; one exact on-shell point, as
    the identity needs exact momentum conservation, in DoubleFloat: the
    bare part's binary64 pole residuals reach 2e-11 of the physical scale):
    the virtual vanishes to 1e-25 of the larger of the physical scale and
    the R2 part (the O10 bound of ``test_loop_v2_observables`` in
    DoubleFloat) while the R2 part does not: it is above 1e-2 of the
    physical scale for every leg and factor and cancels the bare part's
    anomaly."""

    from pyamplicol.loops.v2.assemble import Precision, SliceRequest
    from pyamplicol.loops.v2.observables import (
        INCLUSIVE,
        NativeObservables,
        ObservableSlice,
    )
    from pyamplicol.models.loading import load_compiled_model

    from .test_loop_v2_ct_closure import _slice_point
    from .test_loop_v2_observables import _ward_scale

    process, born, legs = WARD[name]
    fixture = json.loads(FIXTURE.read_text())
    observables = ObservableSlice.generate(
        load_compiled_model(loop_sm_compiled_model),
        SliceRequest(
            process,
            born,
            {"QCD": born["QCD"] + 2, "QED": born["QED"]},
            fixed_parameters=fixture["fixed_parameters"],
            particle_veto=VETO,
            allow_anomalous_veto=True,
        ),
        loop_native,
        ward=True,
    )
    parameters = {k: v for k, v in fixture["parameters"].items() if k != "MU_R"}
    point = _slice_point((0,) * len(process.replace(">", "").split()))
    scale = str(int(fixture["mu_r"][0]) ** 2)
    physical = _ward_scale(
        observables.slice.evaluate(
            point, scale, parameters=parameters, precision=Precision.double()
        )
    )
    native = NativeObservables.correlated(observables, (INCLUSIVE,), lanes=4)
    for leg in legs:
        for factor in ((1, 0), (0, 1)):
            (result,) = native.evaluate(
                [point],
                [scale],
                ward=(leg, factor),
                parameters=parameters,
                precision=Precision.double_double(),
            )
            run = result[INCLUSIVE]
            r2 = [
                max(
                    abs(complex(laurent.coefficients[k]))
                    for laurent in run.components["r2"].values()
                )
                for k in range(3)
            ]
            size = [max(a, b) for a, b in zip(physical, r2, strict=True)]
            for sector, laurent in run.components["virtual"].items():
                for k, value in enumerate(laurent.coefficients):
                    bound = 1e-25 * size[k]
                    assert abs(complex(value)) <= bound, (leg, factor, sector, k)
            assert r2[0] > 1e-2 * physical[0], (leg, factor, r2, physical)
