# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from pyamplicol.models.compiler_kernels import _ordered_dense_tensor_components


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
    assert _ordered_dense_tensor_components(
        _Tensor(),
        ("ufo_l_1_3", "ufo_l_1_4"),
    ) == (0, 1, 10, 11)
