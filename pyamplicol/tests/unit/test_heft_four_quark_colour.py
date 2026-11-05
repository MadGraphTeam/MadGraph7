# SPDX-License-Identifier: 0BSD
"""The HEFT H g g vertex between two quark lines.

Colour-flow gluons are U(N) currents; between fermion lines the U(1) part is
subtracted by an auxiliary that only fermion vertices absorb. The identity of
the H g g vertex couples to that part, so its gluon legs must be projected onto
SU(N) also at tree level (``models.compiler_color_flow.u1_singlet_identity_vertices``).
"""

from __future__ import annotations

from decimal import Decimal, localcontext
from pathlib import Path

from pyamplicol import Generator, ModelSource, ProcessSet, Runtime
from pyamplicol._internal.sm_heft import compile_sm_heft_source
from pyamplicol.config import (
    ColorConfig,
    EvaluatorConfig,
    GenerationConfig,
    ProcessConfig,
    RunConfig,
)
from pyamplicol.models.compiler_entry import compile_ufo_model_ir
from pyamplicol.models.contracts import (
    U1_PROJECTION_SUFFIX,
    U1_SUBTRACTION_PREFIX,
)

_PI = Decimal("3.14159265358979323846264338327950288419716939937510582097494")
_METRIC = (1, -1, -1, -1)

Vector = tuple[Decimal, Decimal, Decimal, Decimal]


def _dot(p: Vector, q: Vector) -> Decimal:
    return sum((_METRIC[i] * p[i] * q[i] for i in range(4)), Decimal(0))


def _quark_line(p: Vector, q: Vector) -> list[list[Decimal]]:
    """Tr(p/ gamma_mu q/ gamma_nu) = 4 (p_mu q_nu + q_mu p_nu - g_mu_nu p.q)."""

    lower_p = [_METRIC[i] * p[i] for i in range(4)]
    lower_q = [_METRIC[i] * q[i] for i in range(4)]
    pq = _dot(p, q)
    return [
        [
            4 * (lower_p[m] * lower_q[n] + lower_q[m] * lower_p[n])
            - (4 * _METRIC[m] * pq if m == n else 0)
            for n in range(4)
        ]
        for m in range(4)
    ]


def _effective_vertex(k1: Vector, k2: Vector) -> list[list[Decimal]]:
    """V^{mu nu} = k1.k2 g^{mu nu} - k2^mu k1^nu of L = -(C/4) H G^a G^a, both
    gluon momenta incoming (mu belongs to k1's gluon, nu to k2's)."""

    k12 = _dot(k1, k2)
    return [
        [(_METRIC[m] * k12 if m == n else 0) - k2[m] * k1[n] for n in range(4)]
        for m in range(4)
    ]


def test_four_quark_heft_trees_project_the_internal_gluons_onto_su_n(
    tmp_path: Path,
) -> None:
    """d(p1) d~(p2) > H u(p3) u~(p4) through H g* g*, summed over colours and
    helicities: |M|^2 = g^4 C^2 Tr(T^a T^b) Tr(T^a T^b)
    L1_{mu mu'} V^{mu nu} V^{mu' nu'} L2_{nu nu'} / (k1^2 k2^2)^2 with
    k1 = p1 + p2, k2 = -(p3 + p4) and Tr(T^a T^b) Tr(T^a T^b) = (N^2 - 1)/4; an
    unprojected U(N) gluon would give N^2/4, 9/8 times more. With
    |M(g g > H)|^2 = 4 C^2 m_H^4 and the averages 1/36 and 1/256, the ratio
    cancels C."""

    output = tmp_path / "heft_four_quark"
    config = RunConfig(
        action="generate",
        color=ColorConfig(accuracy="full"),
        evaluator=EvaluatorConfig(backend="jit", execution_mode="compiled"),
        generation=GenerationConfig(workers=1, emit_api_bundle=False),
        process=ProcessConfig(
            coupling_order_policy="explicit",
            max_coupling_orders={"QCD": 2, "QED": 0, "HIG": 1},
        ),
    )
    Generator(config).generate(
        ProcessSet.from_expressions(
            ("g g > h", "d d~ > h u u~"), names=("gg_h", "ddx_huux")
        ),
        output,
        model=ModelSource.built_in_sm_heft(),
    )
    alpha_s = Decimal("0.125")

    def evaluate(name: str, point: tuple[Vector, ...]) -> Decimal:
        runtime = Runtime.load(output, process=name, model_parameters={"aS": 0.125})
        return Decimal(str(runtime.evaluate((point,), precision=40)[0]))

    half_mass = Decimal("62.5")
    zero = Decimal(0)
    gg_h = evaluate(
        "gg_h",
        (
            (half_mass, zero, zero, half_mass),
            (half_mass, zero, zero, -half_mass),
            (2 * half_mass, zero, zero, zero),
        ),
    )
    p1: Vector = (Decimal(500), zero, zero, Decimal(500))
    p2: Vector = (Decimal(500), zero, zero, Decimal(-500))
    # sqrt(s) = 1000 GeV, m_H = 125 GeV: u (375 GeV) and u~ (187.5 GeV) at right
    # angles leave the Higgs boson exactly on its mass shell.
    for p3, p4 in (
        (
            (Decimal(375), Decimal(225), zero, Decimal(300)),
            (Decimal("187.5"), zero, Decimal("187.5"), zero),
        ),
        (
            (Decimal(375), zero, zero, Decimal(375)),
            (Decimal("187.5"), Decimal(150), Decimal("112.5"), zero),
        ),
    ):
        p_h: Vector = tuple(p1[i] + p2[i] - p3[i] - p4[i] for i in range(4))  # type: ignore[assignment]
        assert _dot(p_h, p_h) == (2 * half_mass) ** 2
        observed = evaluate("ddx_huux", (p1, p2, p_h, p3, p4)) / gg_h
        with localcontext() as context:
            context.prec = 50
            k1: Vector = tuple(p1[i] + p2[i] for i in range(4))  # type: ignore[assignment]
            k2: Vector = tuple(-(p3[i] + p4[i]) for i in range(4))  # type: ignore[assignment]
            first, second = _quark_line(p1, p2), _quark_line(p3, p4)
            vertex = _effective_vertex(k1, k2)
            contraction = sum(
                (
                    first[a][b] * vertex[a][c] * vertex[b][d] * second[c][d]
                    for a in range(4)
                    for b in range(4)
                    for c in range(4)
                    for d in range(4)
                ),
                Decimal(0),
            )
            kinematics = contraction / (_dot(k1, k1) * _dot(k2, k2)) ** 2
            g_squared = 4 * _PI * alpha_s
            colour = Decimal(2)  # (N^2 - 1)/4
            expected = (
                Decimal(256)
                / 36
                * g_squared**2
                * colour
                * kinematics
                / (4 * (2 * half_mass) ** 4)
            )
        assert abs(observed / expected - 1) <= Decimal("1e-30")


def test_projected_vertices_get_no_singlet_identity_copies() -> None:
    """The H g g identity gets the auxiliary copies of
    ``synthesize_u1_singlet_identities`` (two stages ``aux H -> aux``, one
    closing ``aux aux -> H``) unless its gluon legs are projected by
    ``synthesize_u1_projections`` (``u1_projected_vertices``, e.g. the R2
    G G H counterterm of loop_sm in the counterterm closure): never both."""

    payload = dict(compile_sm_heft_source().model)
    keep = {"g", "H", "u", "u~"}
    payload["vertex_rules"] = [
        rule for rule in payload["vertex_rules"] if set(rule["particles"]) <= keep
    ]
    assert any(rule["name"] == "V_HEFT_HGG" for rule in payload["vertex_rules"])

    def singlet_copies(projected: frozenset[str]) -> list[tuple[str, ...]]:
        ir = compile_ufo_model_ir(payload, u1_projected_vertices=projected)
        return sorted(
            kernel.particles
            for kernel in ir.oriented_kernels
            if kernel.vertex == f"V_HEFT_HGG{U1_PROJECTION_SUFFIX}"
            and all(
                name == "H" or name.startswith(U1_SUBTRACTION_PREFIX)
                for name in kernel.particles
            )
        )

    auxiliary = f"{U1_SUBTRACTION_PREFIX}g"
    assert singlet_copies(frozenset()) == [
        ("H", auxiliary, auxiliary),
        (auxiliary, "H", auxiliary),
        (auxiliary, auxiliary, "H"),
    ]
    assert singlet_copies(frozenset({"V_HEFT_HGG"})) == []
