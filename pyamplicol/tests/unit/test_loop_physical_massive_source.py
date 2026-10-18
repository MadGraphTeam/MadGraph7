# SPDX-License-Identifier: 0BSD
"""Massive-muon Gmu source components, including isolated Yukawa loop currents."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def source_model():
    symbolica = pytest.importorskip("symbolica")
    pytest.importorskip("_loop_reduction")
    path = os.environ.get("PYAMPLICOL_EW_UFO_REFERENCE")
    if not symbolica.is_licensed() or not path:
        pytest.skip("licensed Symbolica and full Gmu UFO reference required")
    from pyamplicol.loops.counterterms import prepare_counterterm_model
    from pyamplicol.loops.covariant_model import build_covariant_structure
    from pyamplicol.loops.model import build_loop_model
    from pyamplicol.loops.model_electroweak import (
        massive_neutral_lepton_counterterm_validations,
    )
    from pyamplicol.models.loop_import import load_raw_loop_model

    zeros = (
        "aEW0M1",
        "MD",
        "MU",
        "MS",
        "MC",
        "MB",
        "Me",
        "MTA",
        "lamWS",
        "AWS",
        "rhoWS",
        "etaWS",
        "ymdo",
        "ymup",
        "yms",
        "ymc",
        "ymb",
        "yme",
        "ymtau",
    )
    ones = ("epsUV", "epsIR", "LogSwitch", "GfRedefinitionChoice")
    structure = build_covariant_structure(
        load_raw_loop_model(path, restriction="full"),
        compatibility="gmu_particle_anti",
        fixed_parameters={**dict.fromkeys(zeros, "0"), **dict.fromkeys(ones, "1")},
        covariant_multiplets={
            23: [23, 250, 9000002, -9000002],
            24: [24, 251, 9000003, -9000004],
            -24: [-24, -251, 9000004, -9000003],
        },
    )
    source = build_loop_model(structure.raw, scalar_program=structure.scalars)
    prepared = prepare_counterterm_model(
        source,
        structure.native,
        primitive_validations=massive_neutral_lepton_counterterm_validations(source),
        vertex_spinor_frame="ufo",
    )
    return structure, prepared


@pytest.fixture(scope="module")
def reference():
    path = (
        Path(__file__).resolve().parents[2]
        / "tools/loop_reference/massive_muon_ew.json"
    )
    data = json.loads(path.read_text())
    # The reference uses exactly on-shell terminating decimals, not rounded
    # high-energy momenta whose tiny shell residual would mask muon-mass tests.
    mass = Fraction(data["mass"])
    for point in data["points"]:
        for index, leg in enumerate(point):
            energy, *spatial = map(Fraction, leg)
            assert energy**2 - sum(value**2 for value in spatial) == (
                0 if index < 2 else mass**2
            )
    return data


def _ffs_only(bare, model):
    from pyamplicol.loops.physical_process import NativeBareProcessProgram

    rules = {
        rule.name
        for rule in model.vertex_rules
        if any(
            tuple(model.lorentz_structure(name).spins) == (2, 2, 1)
            for name in rule.lorentz_structures
        )
    }
    loops = tuple(
        graph
        for graph in bare.process.loops
        if any(vertex.interaction in rules for vertex in graph.vertices)
    )
    return NativeBareProcessProgram(
        replace(bare.process, loops=loops),
        state_kinds=bare.state_kinds,
        initial_average=bare.initial_average,
        model_state=bare.model_state,
        model_output_indices=dict(
            zip(bare.model_names, bare.model_output_indices, strict=True)
        ),
        model_scale_view=bare.model_scale_view,
        vertex_spinor_frame=bare.vertex_spinor_frame,
        backend=bare.backend,
        **bare.options,
        reduction_engine=bare.reduction_engine,
        coefficient_engine=bare.coefficient_engine,
    )


@pytest.mark.parametrize(
    "arithmetic,precision,tolerance,bit",
    [
        ("double", None, "2e-11", 0),
        ("double-double", None, "2e-25", 80),
        ("arbitrary", 80, "2e-62", 180),
    ],
)
def test_massive_muon_source_components_and_isolated_ffs(
    source_model, reference, arithmetic, precision, tolerance, bit
):
    from symbolica import E, Expression

    from pyamplicol.loops.model_expressions import parse_exact_ufo
    from pyamplicol.loops.physical_source import source_model_program

    structure, prepared = source_model
    program = source_model_program(
        structure,
        prepared,
        ("e-", "e+"),
        ("m-", "m+"),
        selected_corrections=["QED"],
        state_kinds=("u", "v_bar", "u_bar", "v"),
        initial_average=E("4"),
        born_order_bounds={"QCD": 0, "QED": 2},
        loop_order_bounds={"QCD": 0, "QED": 4},
        parameters=reference["parameters"],
        arithmetic=arithmetic,
        precision=precision,
        reduction_engine="fast" if arithmetic == "double" else "generic",
        coefficient_engine="jit" if arithmetic == "double" else "interpreted",
    )
    bare = program.source_bare.bare
    assert len(bare.loop_graphs) == 145 and len(program.matched.insertions) == 64
    assert all(
        item.rule.has_validated_tensor_basis for item in program.matched.insertions
    )
    assert not program.virtual.field_currents
    points = np.asarray(reference["points"], dtype=str)
    scales = reference["scales_squared"]
    result = program.evaluate_decimal(points, scales)
    np.testing.assert_allclose(
        result.born[:, 0].astype(float), reference["born"], rtol=2e-11
    )
    for name, target in reference["components"].items():
        np.testing.assert_allclose(
            getattr(result, name)[:, :, 0].astype(float), target, rtol=2e-9, atol=3e-16
        )
    assert np.all(result.uv_external_fields.astype(float) == 0)

    # The reference filters only terminal accumulation of the 47 identified
    # MadLoop records with an FFS interaction. Full interference alone would
    # conceal this small contribution among much larger gauge loops.
    filtered = _ffs_only(bare, prepared.native)
    ffs = filtered.evaluate_decimal(points, scales)
    np.testing.assert_allclose(
        ffs.virtual[:, :, 0].astype(float), reference["ffs_bare"], rtol=2e-8, atol=1e-19
    )

    # Only the two massless electron legs carry double IR poles; stable massive
    # muons do not. This independent exact identity tests the actual carrier.
    alpha = parse_exact_ufo(
        "2**(1/2)*0.0000116639*80.419**2*(1-80.419**2/91.188**2)/cmath.pi"
    )
    ratio = (
        Expression.evaluator_multiple(
            [-2 * alpha / (2 * E("pi"))],
            [],
            n_cores=1,
            jit_compile=False,
        )
        .evaluate_complex_with_prec([], 100)[0]
        .to_decimal_tuple()[0]
    )
    with localcontext() as context:
        context.prec = 110
        for born, virtual in zip(result.born, result.virtual, strict=True):
            assert abs(Decimal(virtual[2, 0]) / Decimal(born[0]) - ratio) < Decimal(
                tolerance
            )
        factor = Decimal(2) if not bit else 1 + Decimal(2) ** -bit
        program.update_parameters({"Gf": str(Decimal("0.0000116639") * factor)})
        rebound = program.evaluate_decimal(points[:1], scales[:1])
        assert Decimal(rebound.born[0, 0]) != Decimal(result.born[0, 0])
        assert abs(
            Decimal(rebound.born[0, 0]) / Decimal(result.born[0, 0]) - factor**2
        ) < Decimal(tolerance)
        for name in reference["components"]:
            assert (
                abs(
                    Decimal(getattr(rebound, name)[0, 0, 0])
                    / Decimal(getattr(result, name)[0, 0, 0])
                    - factor**3
                )
                < Decimal(tolerance) * 100
            )
    empty = program.evaluate_decimal(points[:0], [])
    assert empty.born.shape == (0, 2) and empty.virtual.shape == (0, 3, 2)
