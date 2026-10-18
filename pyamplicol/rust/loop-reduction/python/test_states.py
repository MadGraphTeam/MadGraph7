# SPDX-License-Identifier: 0BSD
"""Precision and convention checks for dependency-owned external wavefunctions."""

from decimal import Decimal, localcontext
from itertools import product

import pytest
from _loop_reduction import (
    CoefficientEvaluator,
    NativeBatch,
    massless_spinors_batch,
    on_shell_wavefunctions_batch,
    wavefunctions_batch,
    wavefunctions_partial_batch,
)
from symbolica import E, Expression, S
from symbolica.community import tensor

MODES = [
    ("double", None, "2e-12"),
    ("double-double", None, "2e-28"),
    ("arbitrary", 80, "2e-75"),
]


def real_batch(rows, mode, precision, ids=None, columns=4):
    return NativeBatch.from_decimal(
        [[str(value), "0"] for row in rows for value in row],
        len(rows),
        columns,
        mode,
        precision,
        ids,
    )


def evaluator(expressions, parameters, mode, precision):
    return CoefficientEvaluator(
        Expression.evaluator_multiple(
            expressions, parameters, n_cores=1, jit_compile=False
        ).save(),
        mode,
        precision,
    )


def assert_zero(batch, tolerance):
    assert all(
        abs(Decimal(component)) < Decimal(tolerance)
        for value in batch.to_decimal()
        for component in value
    )


@pytest.mark.parametrize("mode,precision,tolerance", MODES)
def test_dependency_massless_spinors_match_calibrated_core(mode, precision, tolerance):
    rows, helicities, kinds = [], [], []
    for point, kind, helicity in product(
        [(13, 3, 4, 12), (13, 3, 4, -12), (5, 0, 0, 5), (5, 0, 0, -5)],
        ["u", "u_bar", "v", "v_bar"],
        [-1, 1],
    ):
        rows.append(point)
        helicities.append(helicity)
        kinds.append(kind)
    for count in (32, 1, 0, 5):
        momenta = real_batch(
            rows[:count], mode, precision, [99 - i for i in range(count)]
        )
        actual = wavefunctions_batch(momenta, helicities[:count], kinds[:count])
        expected = massless_spinors_batch(momenta, helicities[:count], kinds[:count])
        assert actual.row_ids == expected.row_ids == momenta.row_ids
        with localcontext() as context:
            context.prec = 100
            for a, b in zip(actual.to_decimal(), expected.to_decimal(), strict=True):
                assert all(
                    abs(Decimal(x) - Decimal(y)) < Decimal(tolerance)
                    for x, y in zip(a, b, strict=True)
                )


@pytest.mark.parametrize("mode,precision,tolerance", MODES)
def test_massive_spinor_sums_use_dependency_gamma_matrices(mode, precision, tolerance):
    points = [(5, 1, 2, 2), (5, 0, 0, 3), (5, 0, 0, -3), (4, 0, 0, 0)]
    specifications = list(product(["u", "u_bar", "v", "v_bar"], [-1, 1]))
    momenta = real_batch([p for p in points for _ in specifications], mode, precision)
    states = wavefunctions_batch(
        momenta,
        [helicity for _ in points for _, helicity in specifications],
        [kind for _ in points for kind, _ in specifications],
    ).reshape_rows(4, 32, [17, 4, 90, 2])
    gamma = tensor.TensorExpression.dirac_gamma(4).to_tensor().to_dense()
    state_symbols = [S(f"state_sum_{i}") for i in range(32)]
    momentum_symbols = list(S("state_E", "state_px", "state_py", "state_pz"))
    mass = S("state_mass")
    differences = []
    for kind_offset, sign in [(0, 1), (16, -1)]:
        for i, j in product(range(4), repeat=2):
            spin_sum = sum(
                (
                    state_symbols[kind_offset + 4 * h + i]
                    * state_symbols[kind_offset + 8 + 4 * h + j]
                    for h in range(2)
                ),
                E("0"),
            )
            slash = sum(
                (
                    gamma[i, j, mu] * momentum_symbols[mu] * (1 if mu == 0 else -1)
                    for mu in range(4)
                ),
                E("0"),
            )
            differences.append(spin_sum - slash - (sign * mass if i == j else 0))
    checker = evaluator(
        differences, state_symbols + momentum_symbols + [mass], mode, precision
    )
    inputs = NativeBatch.concat_columns(
        [
            states,
            real_batch(points, mode, precision, states.row_ids),
            real_batch([[4]] * 4, mode, precision, states.row_ids, columns=1),
        ]
    )
    assert_zero(checker.evaluate_batch(inputs), tolerance)


@pytest.mark.parametrize("mode,precision,tolerance", MODES)
def test_massive_vector_transversality_and_completeness(mode, precision, tolerance):
    point = [5, 1, 2, 2]
    kinds = ["epsilon"] * 3 + ["epsilon_bar"] * 3
    states = wavefunctions_batch(
        real_batch([point] * 6, mode, precision), [-1, 0, 1] * 2, kinds
    ).reshape_rows(1, 24, [8])
    symbols = [S(f"vector_sum_{i}") for i in range(24)]
    metric = tensor.Representation.mink(4).g("mu", "nu").to_tensor().to_dense()
    differences = []
    for h in range(3):
        differences.append(
            sum(
                (symbols[4 * h + mu] * point[mu] * metric[mu, mu] for mu in range(4)),
                E("0"),
            )
        )
    for mu, nu in product(range(4), repeat=2):
        differences.append(
            sum(
                (symbols[4 * h + mu] * symbols[12 + 4 * h + nu] for h in range(3)),
                E("0"),
            )
            + metric[mu, nu]
            - E(str(point[mu] * point[nu])) / 16
        )
    assert_zero(
        evaluator(differences, symbols, mode, precision).evaluate_batch(states),
        tolerance,
    )


@pytest.mark.parametrize("mode,precision,tolerance", MODES)
def test_explicit_on_shell_factory_retains_original_mass_precision(
    mode, precision, tolerance
):
    with localcontext() as context:
        context.prec = 150
        power = {"double": 20, "double-double": 80, "arbitrary": 180}[mode]
        tiny = Decimal(2) ** -power
        masses = [Decimal(4), Decimal(4) + tiny]
        momenta, states = on_shell_wavefunctions_batch(
            real_batch([[0, 0, 0]] * 2, mode, precision, [91, 3], columns=3),
            real_batch(
                [[mass] for mass in masses], mode, precision, [91, 3], columns=1
            ),
            [1, 1],
            ["u", "u"],
        )
        assert momenta.row_ids == states.row_ids == [91, 3]
        energy = [Decimal(pair[0]) for pair in momenta.take_columns([0]).to_decimal()]
        assert abs((energy[1] - energy[0]) / tiny - 1) < Decimal("1e-7")
        assert (
            wavefunctions_batch(momenta, [1, 1], ["u", "u"]).to_decimal()
            == states.to_decimal()
        )
        symbols = list(
            S("state_norm_0", "state_norm_1", "state_norm_2", "state_norm_3")
        )
        norm = evaluator(
            [sum((x * x.conj() for x in symbols), E("0"))], symbols, mode, precision
        )
        norms = [Decimal(pair[0]) for pair in norm.evaluate_batch(states).to_decimal()]
        for actual, mass in zip(norms, masses, strict=True):
            assert abs(actual - 2 * mass) < Decimal(tolerance)
        assert abs((norms[1] - norms[0]) / (2 * tiny) - 1) < Decimal("1e-7")


def test_external_state_domains_shapes_scalars_and_preserved_four_momenta():
    batch = real_batch([[6, 0, 0, 3]], "double", None, [77])
    original = batch.to_decimal()
    wavefunctions_batch(batch, [1], ["u"])
    assert batch.to_decimal() == original
    scalar = wavefunctions_batch(batch, [0], ["scalar"])
    assert scalar.columns == 1 and scalar.row_ids == [77]
    assert Decimal(scalar.to_decimal()[0][0]) == 1
    empty = wavefunctions_batch(real_batch([], "double", None), [], [], components=1)
    assert empty.columns == 1 and empty.rows == 0
    rest = real_batch([[4, 0, 0, 0]], "double", None, [81])
    with pytest.raises(ValueError, match=r"lane 81.*longitudinal"):
        wavefunctions_batch(rest, [0], ["epsilon"])
    for helicity, kind in [(0, "u"), (1, "scalar"), (2, "epsilon"), (1, "unknown")]:
        with pytest.raises(ValueError):
            wavefunctions_batch(batch, [helicity], [kind])
    with pytest.raises(ValueError, match="component count"):
        wavefunctions_batch(batch, [1], ["u"], components=1)
    for mass in ("-1", "NaN", "inf"):
        with pytest.raises(ValueError, match="nonnegative real mass"):
            on_shell_wavefunctions_batch(
                real_batch([[1, 2, 3]], "double", None, [77], columns=3),
                real_batch([[mass]], "double", None, [77], columns=1),
                [1],
                ["u"],
            )
    with pytest.raises(ValueError, match="identities"):
        on_shell_wavefunctions_batch(
            real_batch([[1, 2, 3]], "double", None, [7], columns=3),
            real_batch([[4]], "double", None, [8], columns=1),
            [1],
            ["u"],
        )


@pytest.mark.parametrize("mode,precision,tolerance", MODES)
def test_partial_states_omit_failures_and_preserve_original_lanes(
    mode, precision, tolerance
):
    rows = [(5, 0, 0, 3), (4, 0, 0, 0), (13, 3, 4, 12), ("inf", 0, 0, 0)]
    helicities, kinds, ids = (
        [0, 0, 1, 1],
        ["epsilon", "epsilon", "u", "u"],
        [91, 3, 47, 8],
    )
    for count in (4, 1, 3, 0, 4):
        momenta = real_batch(rows[:count], mode, precision, ids[:count])
        values, statuses = wavefunctions_partial_batch(
            momenta, helicities[:count], kinds[:count]
        )
        good = [i for i in (0, 2) if i < count]
        assert values.row_ids == [ids[i] for i in good]
        assert values.precision_bits == momenta.precision_bits
        assert values.nonfinite_rows() == []
        assert [row for row, _ in statuses] == ids[:count]
        assert [status == "success" for _, status in statuses] == [
            i in good for i in range(count)
        ]
        expected = wavefunctions_batch(
            momenta.take_rows(good),
            [helicities[i] for i in good],
            [kinds[i] for i in good],
        )
        assert values.to_decimal() == expected.to_decimal()
        if count > 1:
            assert "longitudinal" in statuses[1][1]
            with pytest.raises(ValueError, match=r"lane 3.*longitudinal"):
                wavefunctions_batch(momenta, helicities[:count], kinds[:count])
    failed, statuses = wavefunctions_partial_batch(
        real_batch([rows[1]], mode, precision, [3]), [0], ["epsilon"]
    )
    assert failed.rows == 0 and failed.columns == 4 and failed.row_ids == []
    assert len(statuses) == 1 and statuses[0][0] == 3


def test_partial_states_retry_original_input_in_wider_exponent_domain():
    rows = [(5, 0, 0, 3), ("5e200", 0, 0, "3e200"), ("5e400", 0, 0, "3e400")]
    for mode in ("double", "double-double"):
        values, statuses = wavefunctions_partial_batch(
            real_batch(rows, mode, None, [7, 19, 28]), [0] * 3, ["epsilon"] * 3
        )
        assert values.row_ids == [7]
        assert statuses[0] == (7, "success")
        # The middle input is finite in both fixed carriers; the dependency's
        # wavefunction constructor detects nonfinite intermediate components.
        assert statuses[1][0] == 19 and statuses[1][1] != "success"
        assert statuses[2][0] == 28 and statuses[2][1] != "success"
    # Reparse the original decimal inputs; promoting an overflowed buffer cannot
    # recover these finite momenta or their exact longitudinal polarization.
    values, statuses = wavefunctions_partial_batch(
        real_batch(rows, "arbitrary", 80, [7, 19, 28]), [0] * 3, ["epsilon"] * 3
    )
    assert statuses == [(7, "success"), (19, "success"), (28, "success")]
    assert values.row_ids == [7, 19, 28] and values.nonfinite_rows() == []
    with localcontext() as context:
        context.prec = 90
        for row in range(3):
            for pair, expected in zip(
                values.to_decimal()[row * 4 : (row + 1) * 4],
                ("0.75", "0", "0", "1.25"),
                strict=True,
            ):
                assert abs(Decimal(pair[0]) - Decimal(expected)) < Decimal("1e-75")
                assert Decimal(pair[1]) == 0


def test_partial_states_metadata_errors_are_not_rescue_candidates():
    batch = real_batch([[5, 0, 0, 3]], "double", None, [17])
    for helicity, kind in [(0, "u"), (1, "scalar"), (2, "epsilon"), (1, "unknown")]:
        with pytest.raises(ValueError):
            wavefunctions_partial_batch(batch, [helicity], [kind])
    with pytest.raises(ValueError, match="metadata dimensions"):
        wavefunctions_partial_batch(batch, [], [])
    with pytest.raises(ValueError, match="component count"):
        wavefunctions_partial_batch(batch, [1], ["u"], components=1)
    empty, statuses = wavefunctions_partial_batch(
        real_batch([], "double", None), [], [], components=1
    )
    assert empty.rows == 0 and empty.columns == 1 and statuses == []
