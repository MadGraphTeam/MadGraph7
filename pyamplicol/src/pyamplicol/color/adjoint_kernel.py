# SPDX-License-Identifier: 0BSD
"""Exact DDM Gram kernels without constructing the quadratic trace matrix.

With Tr(Ta Tb) = delta_ab, a half-ladder is the nested-commutator trace
Tr([...[T1,T2],...,T(n-1)] Tn).  Both U(1) contributions vanish under the
commutators, so the inexpensive U(Nc) trace cycle kernel is sufficient.
Applying the left/right commutator finite differences takes 2*(n-2) sweeps
over S_(n-1); the result retained on the two-anchor subgroup has (n-2)!
entries.  All arithmetic is integer, including NLC's Nc-power truncation.

There is no gluon cap.  The dense kernel uses int64 while its exact bound
fits and Python integers otherwise.  Beyond ``DENSE_ADJOINT_GLUONS`` an
overlap is evaluated on its own from the commutator expansions of both
ladders (``ladder_overlap``), so no (n-2)!-entry table is built.
"""

from __future__ import annotations

import math
from functools import lru_cache
from itertools import chain, permutations

import numpy as np

# The largest ladder whose dense kernel is built (two anchors plus the
# symmetric-group FFT degree of ten), and the size up to which the automatic
# tree basis choice prefers DDM.  A resource choice, not a cap.
DENSE_ADJOINT_GLUONS = 12
_INT64_MAX = int(np.iinfo(np.int64).max)


def _check_arguments(
    total_gluons: object, accuracy: str, nc: object, full_col_acc: object
) -> None:
    if type(total_gluons) is not int or total_gluons < 2:
        raise ValueError("adjoint colour kernel requires at least 2 external adjoints")
    if accuracy not in {"lc", "nlc", "full"}:
        raise ValueError(f"unknown adjoint colour accuracy {accuracy!r}")
    if type(nc) is not int or nc < 2:
        raise ValueError("adjoint colour kernel requires integer Nc >= 2")
    if type(full_col_acc) is not int or full_col_acc < 0:
        raise ValueError("adjoint full-colour accuracy must be nonnegative")


def _minimum_power(total_gluons: int, accuracy: str, full_col_acc: int) -> int:
    """The lowest U(Nc) trace-pair power kept by an accuracy."""

    depth = 2 * full_col_acc if accuracy == "full" else 2 if accuracy == "nlc" else 0
    return max(total_gluons - depth, 0)


def _permutation_ranks(rows: np.ndarray) -> np.ndarray:
    """Rank a batch of image permutations in lexicographic order."""

    degree = rows.shape[1]
    ranks = np.zeros(len(rows), dtype=np.int64)
    for position in range(degree - 1):
        digit = rows[:, position].astype(np.int64)
        for preceding in range(position):
            digit -= rows[:, preceding] < rows[:, position]
        ranks += digit * math.factorial(degree - position - 1)
    return ranks


def _trace_cycle_counts(rows: np.ndarray) -> np.ndarray:
    """Cycles of c^-1 h c h^-1, with h fixing the final external label."""

    count, degree = rows.shape
    total = degree + 1
    # One visited bit per label; S_63 cannot be enumerated anyway.
    mask = np.uint32 if total <= 32 else np.uint64
    indices = np.arange(count)
    commutator = np.empty((count, total), dtype=np.uint8)
    for position in range(degree - 1):
        commutator[indices, rows[:, position]] = (
            rows[:, position + 1].astype(np.int16) - 1
        ) % total
    commutator[indices, rows[:, -1]] = total - 2
    commutator[:, -1] = (rows[:, 0].astype(np.int16) - 1) % total
    visited = np.zeros(count, dtype=mask)
    cycles = np.zeros(count, dtype=np.uint8)
    bits = np.left_shift(mask(1), np.arange(total, dtype=mask))
    for start in range(total):
        active = (visited & bits[start]) == 0
        cycles += active
        position = np.full(count, start, dtype=np.uint8)
        while np.any(active):
            visited |= np.where(active, bits[position], mask(0))
            position = commutator[indices, position]
            active &= (visited & bits[position]) == 0
    return cycles


@lru_cache(maxsize=4)
def adjoint_color_kernel(
    total_gluons: int,
    *,
    accuracy: str = "full",
    nc: int = 3,
    full_col_acc: int = 20,
) -> np.ndarray:
    """Return an immutable exact relative DDM kernel in lexicographic order.

    ``lc`` and ``nlc`` retain powers >= n and >= n-2 respectively.  Since the
    finite differences have Nc-independent integer coefficients, truncating
    the seed commutes with the basis transformation.  The LC helper exists
    for factor diagnostics; adjoint generation itself requires NLC/full.
    The dtype is int64 when the finite differences provably fit, otherwise
    object (exact Python integers).  The table has (n-2)! entries and the
    sweeps visit S_(n-1): memory is the only bound (``ladder_overlap``
    evaluates single entries without it).
    """

    _check_arguments(total_gluons, accuracy, nc, full_col_acc)
    if total_gluons == 2:
        # The degenerate ladder is Tr(Ta Tb)=delta_ab. There is no
        # commutator to remove U(1), so retain SU(Nc)'s exact dimension.
        leading_only = accuracy == "lc" or (accuracy == "full" and full_col_acc == 0)
        kernel = np.asarray([nc * nc - (0 if leading_only else 1)], dtype=np.int64)
        kernel.flags.writeable = False
        return kernel
    degree = total_gluons - 1
    # Each difference at most doubles the largest absolute input.  Check once
    # before allocating; numpy's int64 arithmetic must never silently wrap.
    fits = nc**total_gluons * 4 ** (total_gluons - 2) <= _INT64_MAX
    order = math.factorial(degree)
    rows = np.fromiter(
        chain.from_iterable(permutations(range(degree))),
        dtype=np.uint8,
        count=order * degree,
    ).reshape(order, degree)
    cycles = _trace_cycle_counts(rows)
    powers = np.asarray(
        [nc**power for power in range(total_gluons + 1)],
        dtype=np.int64 if fits else object,
    )
    current = powers[cycles]
    minimum_power = _minimum_power(total_gluons, accuracy, full_col_acc)
    if minimum_power:
        current[cycles < minimum_power] = 0
    for generator in range(degree, 1, -1):
        # s_i^-1 acts on images: 0->1->...->i-1->0.
        relabel = np.arange(degree, dtype=np.uint8)
        relabel[:generator] = (relabel[:generator] + 1) % generator
        relative = relabel[rows]
        ranks = _permutation_ranks(relative)
        current = current - current[ranks]
    for generator in range(degree, 1, -1):
        # Right multiplication by s_i moves position i-1 to the beginning.
        relative = rows.copy()
        relative[:, 0] = rows[:, generator - 1]
        relative[:, 1:generator] = rows[:, : generator - 1]
        ranks = _permutation_ranks(relative)
        current = current - current[ranks]
    kernel = current[: math.factorial(degree - 1)].copy()
    kernel.flags.writeable = False
    return kernel


def _ladder_traces(word: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Signs and words of the traces of Tr([...[T_w1, T_w2], ...] T_wn).

    Each middle letter goes right (+) or left (-) of the growing commutator,
    so the 2^(n-2) traces are (reversed(left letters), w1, right letters, wn).
    """

    middle = word[1:-1]
    count = 1 << len(middle)
    signs = np.empty(count, dtype=np.int64)
    words = np.empty((count, len(word)), dtype=np.int64)
    for choice in range(count):
        left = [a for i, a in enumerate(middle) if choice >> i & 1]
        right = [a for i, a in enumerate(middle) if not choice >> i & 1]
        signs[choice] = -1 if len(left) % 2 else 1
        words[choice] = (*reversed(left), word[0], *right, word[-1])
    return signs, words


def _cyclic_neighbours(words: np.ndarray, step: int) -> np.ndarray:
    """neighbour[r, a]: the label ``step`` places after label a in trace r."""

    out = np.empty_like(words)
    out[np.arange(len(words))[:, None], words] = np.roll(words, -step, axis=1)
    return out


def _permutation_cycles(images: np.ndarray) -> np.ndarray:
    """Cycle counts of a batch of permutations of range(n) (rows of images)."""

    count, n = images.shape
    rows = np.arange(count)
    visited = np.zeros((count, n), dtype=bool)
    cycles = np.zeros(count, dtype=np.int64)
    for start in range(n):
        active = ~visited[:, start]
        cycles += active
        position = np.full(count, start)
        while np.any(active):
            visited[rows[active], position[active]] = True
            position = np.where(active, images[rows, position], position)
            active &= ~visited[rows, position]
    return cycles


def ladder_overlap(
    left: tuple[int, ...],
    right: tuple[int, ...],
    *,
    accuracy: str,
    nc: int = 3,
    full_col_acc: int = 20,
    batch: int = 1 << 20,
) -> int:
    """One exact DDM overlap <H(left)|H(right)> for any number of gluons.

    Both half-ladders expand into 2^(n-2) signed traces.  With
    Tr(T^a T^b) = delta_ab a U(Nc) trace pair <Tr(alpha)|Tr(beta)> is
    Nc^c, c the number of cycles of pred_alpha o succ_beta on the labels;
    the U(1) pieces cancel under the commutators.  Powers below the
    accuracy's minimum are dropped term by term, as in
    ``adjoint_color_kernel``, and the signs are summed per power in exact
    integers.  Cost: 4^(n-2) trace pairs of n labels, in batches of at most
    ``batch`` pairs; no (n-2)! table.
    """

    n = len(left)
    _check_arguments(n, accuracy, nc, full_col_acc)
    position = {label: i for i, label in enumerate(left)}
    if len(position) != n or sorted(position.get(a, -1) for a in right) != list(
        range(n)
    ):
        raise ValueError("DDM colour words must be permutations of one label set")
    if n == 2:
        return int(
            adjoint_color_kernel(
                2, accuracy=accuracy, nc=nc, full_col_acc=full_col_acc
            )[0]
        )
    left_signs, left_words = _ladder_traces(tuple(range(n)))
    right_signs, right_words = _ladder_traces(tuple(position[a] for a in right))
    predecessor = _cyclic_neighbours(left_words, -1)
    successor = _cyclic_neighbours(right_words, 1)
    totals = np.zeros(n + 1, dtype=np.int64)
    step = max(1, batch // len(right_words))
    for start in range(0, len(left_words), step):
        block = predecessor[start : start + step]
        # (pred_alpha o succ_beta)[a] = pred_alpha[succ_beta[a]] for each pair.
        images = np.take_along_axis(
            np.broadcast_to(block[:, None, :], (len(block), *successor.shape)),
            np.broadcast_to(successor[None], (len(block), *successor.shape)),
            axis=2,
        ).reshape(-1, n)
        signs = np.outer(left_signs[start : start + step], right_signs).reshape(-1)
        np.add.at(totals, _permutation_cycles(images), signs)
    minimum = _minimum_power(n, accuracy, full_col_acc)
    return sum(
        int(total) * nc**power
        for power, total in enumerate(totals)
        if power >= minimum and total
    )


def adjoint_color_factor(
    left: tuple[int, ...],
    right: tuple[int, ...],
    *,
    accuracy: str,
    nc: int = 3,
    full_col_acc: int = 20,
) -> int:
    """Evaluate one DDM overlap from its middle-label relative permutation.

    Ladders above ``DENSE_ADJOINT_GLUONS`` are evaluated on their own
    (``ladder_overlap``) instead of from the dense kernel.
    """

    if (
        len(left) < 2
        or len(left) != len(right)
        or left[0] != right[0]
        or left[-1] != right[-1]
        or len(set(left)) != len(left)
        or set(left) != set(right)
    ):
        raise ValueError("DDM colour words must have matching fixed endpoints")
    if len(left) > DENSE_ADJOINT_GLUONS:
        return ladder_overlap(
            left, right, accuracy=accuracy, nc=nc, full_col_acc=full_col_acc
        )
    ranks_by_label = {label: index for index, label in enumerate(left[1:-1])}
    relative = [ranks_by_label[label] for label in right[1:-1]]
    rank = 0
    remaining = list(range(len(relative)))
    for position, value in enumerate(relative):
        offset = remaining.index(value)
        rank += offset * math.factorial(len(relative) - position - 1)
        remaining.pop(offset)
    return int(
        adjoint_color_kernel(
            len(left), accuracy=accuracy, nc=nc, full_col_acc=full_col_acc
        )[rank]
    )
