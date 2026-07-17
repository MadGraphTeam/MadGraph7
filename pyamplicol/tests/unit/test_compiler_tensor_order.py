# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import pytest

from pyamplicol.models.compiler_kernels import _ordered_dense_tensor_components
from pyamplicol.models.compiler_tensor_ordering import (
    identity_ordering_for_materialized_axes,
)


class _Argument:
    def __init__(self, label: str) -> None:
        self.label = label

    def to_canonical_string(self) -> str:
        return f"spenso::mink(4,{self.label})"


class _Structure:
    def __init__(self) -> None:
        self.coordinates = ((0, 0), (0, 1), (1, 0), (1, 1))

    def set_name(self, _name: str) -> None:
        pass

    def to_expression(self) -> tuple[_Argument, ...]:
        return (_Argument("ufo_l_1_4"), _Argument("ufo_l_1_3"))

    def __getitem__(self, index: int) -> tuple[int, int]:
        return self.coordinates[index]


class _Tensor:
    def __init__(self) -> None:
        # Storage follows the actual (leg 4, leg 3) structure.
        self.values = (0, 10, 1, 11)
        self._structure = _Structure()

    def to_dense(self) -> None:
        pass

    def structure(self) -> _Structure:
        return self._structure

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> int:
        return self.values[index]


def test_dense_tensor_components_follow_physical_ufo_axis_order() -> None:
    ordered = _ordered_dense_tensor_components(
        _Tensor(),
        ("ufo_l_1_3", "ufo_l_1_4"),
    )

    assert ordered.values == (0, 1, 10, 11)
    assert tuple(axis.extent for axis in ordered.ordering.axes) == (2, 2)
    assert ordered.ordering.component_basis == (0, 1, 2, 3)


class _RectangularStructure(_Structure):
    def __init__(self) -> None:
        self.coordinates = ((2, 1), (0, 0), (1, 0), (2, 0), (0, 1), (1, 1))


class _RectangularTensor(_Tensor):
    def __init__(self) -> None:
        self.values = (12, 0, 1, 2, 10, 11)
        self._structure = _RectangularStructure()


def test_dense_tensor_components_validate_non_square_cartesian_grid() -> None:
    ordered = _ordered_dense_tensor_components(
        _RectangularTensor(),
        ("ufo_l_1_3", "ufo_l_1_4"),
    )

    assert ordered.values == (0, 1, 2, 10, 11, 12)
    assert tuple(axis.extent for axis in ordered.ordering.axes) == (2, 3)
    assert ordered.ordering.canonical_size == 6


class _IncompleteStructure(_Structure):
    def __init__(self) -> None:
        self.coordinates = ((0, 0), (0, 1), (1, 0))


class _IncompleteTensor(_Tensor):
    def __init__(self) -> None:
        self.values = (0, 1, 10)
        self._structure = _IncompleteStructure()


def test_dense_tensor_components_reject_incomplete_cartesian_grid() -> None:
    with pytest.raises(ValueError, match="complete Cartesian component grid"):
        _ordered_dense_tensor_components(
            _IncompleteTensor(),
            ("ufo_l_1_3", "ufo_l_1_4"),
        )


class _Spin2Structure:
    def __init__(self) -> None:
        self.coordinates = tuple(
            reversed(tuple((right, left) for right in range(4) for left in range(4)))
        )

    def set_name(self, _name: str) -> None:
        pass

    def to_expression(self) -> tuple[_Argument, ...]:
        return (_Argument("ufo_l_2_3"), _Argument("ufo_l_1_3"))

    def __getitem__(self, index: int) -> tuple[int, int]:
        return self.coordinates[index]


class _Spin2Tensor(_Tensor):
    def __init__(self) -> None:
        self._structure = _Spin2Structure()
        self.values = tuple(
            left * 4 + right for right, left in self._structure.coordinates
        )


def test_spin2_axis_transpose_and_storage_permutation_canonicalize() -> None:
    labels = ("ufo_l_1_3", "ufo_l_2_3")
    ordered = _ordered_dense_tensor_components(_Spin2Tensor(), labels)

    assert ordered.values == tuple(range(16))
    assert ordered.ordering.basis == "lorentz-rank-2"
    assert ordered.ordering.ordering_id == identity_ordering_for_materialized_axes(
        labels,
        (4, 4),
    ).ordering_id
