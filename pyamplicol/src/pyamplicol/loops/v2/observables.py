# SPDX-License-Identifier: 0BSD
"""Observables of the v2 engine (design §7.7, §8.4, §8.5, §9; WP2.5).

Every observable is a set of *seed columns* on the one generated slice: the
recursion, the families and the reductions are those of the inclusive virtual
(``assemble``); only the seeds, the Born bra and the local contractions
change. An output is described by a ``Column``: an optional directed colour
history, an optional set of Born helicity rows and an optional set of Born
colour-basis elements on the Born (bra) side.

* **Directed colour correlations** (§7.7). For a history (bra, ket) of
  ordered colour connections, V = B^dagger K_BL L + L^dagger K_LB B with
  K_BL[b, c] = <bra(b)|ket(kappa_c)> and K_LB = (K_BL^swapped)^dagger. The
  first term is seeded with the directed seed table, the second with the
  table of the swapped history and conjugated after reduction (one column if
  the history is symmetric). Counterterm and field components use the same
  directed overlaps; the Born is B^dagger K_BB B. Results are complex.
* **Spin sources** (§8.4). Literal complex four-vectors replace the
  wavefunction of their legs in the Born and in every block (the tree runtime's
  source slots, crossing phase applied once, nothing else); each source leg
  keeps one helicity row. Spin sources need the full tree kernels (blocks with
  a non-transverse source, ``blocks.KERNEL_VARIANT_BY_BLOCK_CLASS``), so a slice
  declaring spin classes is generated on ``compile_tree_kernel_variant(...,
  "full")``.
* **Asymmetric Ward mode** (§8.4, oracle O10). The Born bra keeps physical
  polarisations; blocks, counterterm currents and the field insertions' Born
  amplitudes (the loop side) read epsilon_j -> p_j for one vector leg j, with
  full kernels. The renormalised interference vanishes when UV and R2 are
  complete and the amplitude is gauge invariant.
* **Resolved outputs** (§8.5). Per Born helicity row (one-hot seed columns)
  and per Born colour component, a component being one element of the tree
  engine's Born colour basis (decision 3 of §1.6: the anchor is a Born basis
  element; kind ``born-basis-anchored-interference``). The Born of a
  component is Re(B_b^dagger G B), every other component 2 Re after the loop
  integration, so ``ResolvedObservables.total()`` equals the inclusive
  evaluation exactly.
* **Flavour members and aliases** (§9). An identity-port member (a species
  substitution of the representative's external PDGs) is admitted by the
  exact source proof ``loops.flavor_mappings.prove_flavor_relabeling`` and
  evaluated by the representative on the same momenta. A permutation alias
  (legs permuted within the initial and within the final state, possibly
  composed with an admitted species substitution) relabels momenta, colour
  histories, spin legs and resolved helicity rows onto the representative.
  Fermion permutation signs multiply bra and ket alike and cancel in every
  bilinear output.

Mixed coupling-order sectors are kept apart: every column is split per
squared-order sector, so each output carries its per-coupling-order partials
(REQUIREMENTS_ADDENDUM A).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from decimal import Decimal
from itertools import permutations, product
from typing import Any, Literal

from pyamplicol.color.connections import (
    ColorConnection,
    ConnectedColorTensor,
    TensorTerm,
    apply_color_connection,
    contract_connected_tensor_nc_terms,
)
from pyamplicol.color.correlator_matrices import ColorCorrelator
from pyamplicol.correlators import CorrelatorConfig

from . import reference_executor as rx
from .assemble import (
    COMPONENTS,
    FIELD_COMPONENT,
    LAURENT,
    Laurent,
    LoopSlice,
    PointResult,
    Precision,
    SliceRequest,
    _fraction,
    _interfere,
    _Numbers,
    _single,
    _to_decimal,
)
from .colour import SeedTable
from .conventions import LOOP_FACTOR_DENOMINATOR
from .ct_sites import LAURENT_POWERS
from .families import Orders, add_orders

Resolution = Literal["helicity", "colour", "helicity-colour"]
ComplexPair = tuple[Decimal, Decimal]
# Kind of the resolved colour components (design §1.6, decision 3).
BORN_BASIS_COMPONENTS = "born-basis-anchored-interference"
INCLUSIVE = "born"


# --------------------------------------------------------------------- columns


@dataclass(frozen=True)
class Column:
    """One observable output: a directed history (``None``: the Born colour
    metric), the Born helicity rows it keeps (``None``: all) and the Born
    colour-basis elements of its bra (``None``: all)."""

    key: Hashable
    history: ColorCorrelator | None = None
    rows: frozenset[rx.Row] | None = None
    basis: frozenset[int] | None = None

    @property
    def directed(self) -> bool:
        return self.history is not None and bool(self.history.bra)


def connect(
    tensor: ConnectedColorTensor, connection: ColorConnection
) -> ConnectedColorTensor:
    """``connection`` applied to every term of ``tensor`` (N_c powers and the
    1/sqrt(2) normalisation of each emission kept)."""

    terms: list[TensorTerm] = []
    for term in tensor.terms:
        image = apply_color_connection(
            term.tensor, connection, coefficient=term.coefficient
        )
        terms.extend(
            replace(item, nc_power=item.nc_power + term.nc_power)
            for item in image.terms
        )
    return ConnectedColorTensor(
        connection.output_legs,
        tuple(terms),
        tensor.normalization_power + connection.order,
    )


class DirectedColour:
    """Exact directed overlaps and seed tables of one colour history.

    ``overlap(left, right)`` is <bra(left)|ket(right)> of two literal Born-
    space tensors as exact Laurent coefficients in N_c; ``table`` is the seed
    table with cells K[c, b] = <bra(b)|ket(kappa_c)>; ``swapped`` is the same
    for the history with bra and ket exchanged (the L^dagger K_LB B term).
    """

    def __init__(self, table: SeedTable, history: ColorCorrelator):
        legs = table.basis.legs
        self.history = history
        self._bra = ColorConnection(legs, history.bra)
        self._ket = ColorConnection(legs, history.ket)
        if self._bra.output_legs != self._ket.output_legs:
            raise ValueError(
                f"colour history {history.id!r} has incompatible bra/ket outputs"
            )
        self._legs = legs
        self._images: dict[tuple[bool, Any], ConnectedColorTensor] = {}
        self._overlaps: dict[tuple[Any, Any, bool], dict] = {}
        self.symmetric = history.bra == history.ket
        self.table = self._table(table, swapped=False)
        self.swapped = self.table if self.symmetric else self._table(table, True)

    def _image(self, tensor: ConnectedColorTensor, bra: bool) -> ConnectedColorTensor:
        key = (bra, tensor)
        found = self._images.get(key)
        if found is None:
            found = self._images[key] = connect(tensor, self._bra if bra else self._ket)
        return found

    def _table(self, table: SeedTable, swapped: bool) -> SeedTable:
        bras = [self._image(b, not swapped) for b in table.basis.tensors]
        cells = []
        for c in range(len(table.classes)):
            ket = self._image(table.tensor(c), swapped)
            cells.append(
                tuple(contract_connected_tensor_nc_terms(bra, ket) for bra in bras)
            )
        return replace(table, cells=tuple(cells))

    def overlap(self, left: Any, right: Any, swapped: bool = False) -> dict:
        """<bra(left)|ket(right)> (bra and ket exchanged when ``swapped``) of
        two literal colour tensors, exact in N_c."""

        key = (left, right, swapped)
        found = self._overlaps.get(key)
        if found is None:
            bra = self._image(_single(left, self._legs), not swapped)
            ket = self._image(_single(right, self._legs), swapped)
            found = self._overlaps[key] = {
                power: value
                for power, value in contract_connected_tensor_nc_terms(bra, ket).items()
                if value
            }
        return found


# --------------------------------------------------------------------- seeds


@dataclass(frozen=True)
class _SeedColumn:
    """One numerator column: a seed table, one sector and the Born it reads."""

    table: SeedTable
    pairs: frozenset[tuple[Orders, Orders]]
    degree: int | None
    rows: frozenset[rx.Row] | None
    basis: frozenset[int] | None


class ColumnSeeds:
    """Seeds of observable columns (an ``rx.SeedSource``).

    Column k of U0[c](h) sums K_k[c, b] conj(A0_b(h)) over the (Born order,
    loop order) pairs of its sector, the Born basis elements it keeps and,
    for a resolved column, only on its own helicity rows (one-hot seeds).
    Contracted directly in the carrier.
    """

    algorithm = "direct"

    def __init__(
        self,
        columns: Sequence[_SeedColumn],
        class_of: Mapping[Any, Any],
        born: Mapping[rx.Row, Mapping[tuple[int, Orders], Mapping[int, Any]]],
        numbers: _Numbers,
        accuracy: str,
    ):
        self.columns = len(columns)
        self.class_of = class_of
        self._numbers = numbers
        self._admitted = {
            (c, order)
            for column in columns
            for c, order, _b, _a in column.table.admitted(column.pairs)
        }

        def cast(cell):
            return numbers.pair(numbers.real(cell.real), numbers.real(cell.imag))

        self._cast = cast
        self._rows: dict[rx.Row, list[dict]] = {}
        for row, amplitudes in born.items():
            values = []
            for column in columns:
                if column.rows is not None and row not in column.rows:
                    values.append({})
                    continue
                keep = column.basis
                values.append(
                    column.table.seeds(
                        column.pairs,
                        amplitudes,
                        accuracy,
                        cast=cast,
                        conjugate=lambda value: value.conjugate(),
                        degree=column.degree,
                        skip=None
                        if keep is None
                        else lambda c, b, keep=keep: b not in keep,
                    )
                )
            self._rows[row] = values

    def seed(self, skeleton, contents):
        reading = rx._reading(skeleton, contents)
        combination = self.class_of.get(reading)
        if combination is None:
            raise KeyError("reading without a seed class")
        order = skeleton.orders
        for c, _x in combination:
            if (c, order) not in self._admitted:
                raise KeyError(f"no column admits class {c} at {order}")
        weights = [(c, self._cast(x)) for c, x in combination]
        zero = self._numbers.zero

        def total(seeds):
            value = zero
            for c, weight in weights:
                found = seeds.get((c, order))
                if found is not None:
                    value = value + weight * found
            return value

        return (combination, order), {
            row: tuple(total(seeds) for seeds in columns)
            for row, columns in self._rows.items()
        }


# --------------------------------------------------------------------- results


@dataclass(frozen=True)
class ResolvedObservables:
    """Per (helicity row, Born colour component) one ``PointResult``.

    ``helicities`` lists the rows (``None`` when colour-only), ``components``
    the Born basis elements (``None`` when helicity-only). Every entry keeps
    the inclusive initial average; ``total()`` sums them to the inclusive
    result.
    """

    entries: Mapping[tuple[rx.Row | None, int | None], PointResult]
    helicities: tuple[rx.Row, ...] | None
    components: tuple[int, ...] | None
    component_kind: str = BORN_BASIS_COMPONENTS

    def total(self) -> PointResult:
        return _sum_results(list(self.entries.values()))


def _sum_results(results: Sequence[PointResult]) -> PointResult:
    first = results[0]

    def add(values):
        total = values[0]
        for value in values[1:]:
            total = total + value
        return total

    born = {
        order: add([r.born[order] for r in results if order in r.born])
        for order in first.born
    }
    components = {
        name: {
            sector: add([r.components[name][sector] for r in results])
            for sector in first.components[name]
        }
        for name in first.components
    }
    totals = tuple(
        add([r.totals[k] for r in results]) for k in range(len(first.totals))
    )
    families: dict = defaultdict(list)
    for result in results:
        for lane, value in result.families.items():
            families[lane].append(value)
    return PointResult(
        born,
        components,
        totals,
        {lane: add(values) for lane, values in families.items()},
        None,
        colour_contraction=first.colour_contraction,
    )


# --------------------------------------------------------------------- slice


@dataclass
class ObservableSlice:
    """A generated slice and its declared observables.

    ``correlators`` declares directed colour histories and spin classes as the
    public ``CorrelatorConfig`` does; ``ward`` prepares the asymmetric Ward
    mode. Spin classes and the Ward mode generate the slice on the full tree
    kernels.
    """

    slice: LoopSlice
    histories: tuple[ColorCorrelator, ...] = ()
    spin_classes: tuple[tuple[int, ...], ...] = ()
    ward: bool = False
    _directed: dict[Hashable, DirectedColour] = field(default_factory=dict, repr=False)

    @classmethod
    def generate(
        cls,
        compiled: Any,
        request: SliceRequest,
        native: Any,
        *,
        correlators: CorrelatorConfig | None = None,
        ward: bool = False,
    ) -> ObservableSlice:
        correlators = correlators or CorrelatorConfig()
        full = bool(correlators.spin_correlations) or ward
        if full:
            from .catalogue import compile_tree_kernel_variant

            compiled = compile_tree_kernel_variant(compiled, "full")
        slice_ = LoopSlice.generate(compiled, request, native)
        return cls.of(slice_, correlators=correlators, ward=ward)

    @classmethod
    def of(
        cls,
        slice_: LoopSlice,
        *,
        correlators: CorrelatorConfig | None = None,
        ward: bool = False,
    ) -> ObservableSlice:
        """The observables of an already generated slice."""

        from pyamplicol.color.correlator_matrices import process_color_legs

        correlators = correlators or CorrelatorConfig()
        correlators = correlators._resolve_color(
            process_color_legs(slice_.blocks.process)
        )
        if (correlators.spin_correlations or ward) and (
            slice_.blocks.kernel_variant != "full"
        ):
            raise ValueError(
                "spin sources and the Ward mode need the full tree kernels "
                "(ObservableSlice.generate compiles them)"
            )
        out = cls(
            slice_,
            tuple(correlators.color_correlations),
            tuple(correlators.spin_correlations),
            ward,
        )
        for history in out.histories:
            out.directed(history)
        return out

    # -- declarations ---------------------------------------------------------

    def directed(self, history: ColorCorrelator) -> DirectedColour:
        """The directed colour data of ``history`` (built once)."""

        key = (history.bra, history.ket)
        found = self._directed.get(key)
        if found is None:
            if self.slice.request.accuracy != "full" and history.bra:
                raise NotImplementedError(
                    "directed colour correlations at LC/NLC need the complete-"
                    "family degree convention (design §7.7); use full colour"
                )
            found = self._directed[key] = DirectedColour(self.slice.table, history)
        return found

    def history(self, identifier: str) -> ColorCorrelator:
        if identifier == INCLUSIVE:
            return ColorCorrelator(INCLUSIVE)
        for history in self.histories:
            if history.id == identifier:
                return history
        raise KeyError(f"undeclared colour correlation {identifier!r}")

    def _spin_legs(self, vectors: Mapping[int, Any]) -> tuple[int, ...]:
        legs = tuple(sorted(vectors))
        if legs and legs not in self.spin_classes:
            raise ValueError(f"spin legs {legs} are not a declared spin class")
        return legs

    @property
    def rows(self) -> tuple[rx.Row, ...]:
        from .plan_export import born_rows

        return born_rows(self.slice)

    @property
    def components(self) -> tuple[int, ...]:
        """The Born colour components: the Born basis elements."""

        return tuple(range(len(self.slice.basis)))

    # -- evaluation -----------------------------------------------------------

    def evaluate(
        self,
        point: Sequence[Sequence[Any]],
        scale_squared: Any,
        *,
        correlations: Sequence[str] = (INCLUSIVE,),
        spin: Mapping[int, Any] | None = None,
        ward: int | None = None,
        parameters: Mapping[str, str] | None = None,
        precision: Precision | None = None,
        poles: bool = True,
    ) -> dict[str, PointResult]:
        """Correlated outputs at one point, one ``PointResult`` per history id.

        ``spin`` maps source legs (a declared spin class) to literal complex
        four-vectors: four real decimals or (real, imaginary) pairs.
        ``ward`` is the vector leg of the asymmetric Ward mode. Directed
        outputs are complex; the others real.
        """

        columns = [
            Column(identifier, self.history(identifier)) for identifier in correlations
        ]
        results = self._run(
            columns, point, scale_squared, parameters, precision, poles, spin, ward
        )
        return {column.key: results[column.key] for column in columns}

    def evaluate_resolved(
        self,
        point: Sequence[Sequence[Any]],
        scale_squared: Any,
        *,
        resolution: Resolution = "helicity-colour",
        spin: Mapping[int, Any] | None = None,
        parameters: Mapping[str, str] | None = None,
        precision: Precision | None = None,
        poles: bool = True,
    ) -> ResolvedObservables:
        """Helicity- and/or colour-resolved outputs (design §8.5)."""

        if resolution not in ("helicity", "colour", "helicity-colour"):
            raise ValueError(f"unknown resolution {resolution!r}")
        rows = self._rows_of(spin) if "helicity" in resolution else (None,)
        components = self.components if "colour" in resolution else (None,)
        columns = [
            Column(
                (row, b),
                None,
                None if row is None else frozenset({row}),
                None if b is None else frozenset({b}),
            )
            for row in rows
            for b in components
        ]
        results = self._run(
            columns, point, scale_squared, parameters, precision, poles, spin, None
        )
        return ResolvedObservables(
            {column.key: results[column.key] for column in columns},
            None if rows == (None,) else tuple(rows),
            None if components == (None,) else tuple(components),
        )

    def _rows_of(self, spin) -> tuple[rx.Row, ...]:
        fixed = self._fixed(spin or {})
        return tuple(
            row
            for row in self.rows
            if all(dict(row)[leg - 1] == h for leg, h in fixed.items())
        )

    def _fixed(self, spin: Mapping[int, Any]) -> dict[int, int]:
        """The one helicity row each source leg keeps (its lowest helicity)."""

        out = {}
        for leg in spin:
            values = {h for row in self.rows for i, h in row if i == leg - 1}
            if not values:
                raise ValueError(f"leg {leg} has no helicity rows")
            out[leg] = min(values)
        return out

    def _run(
        self, columns, point, scale_squared, parameters, precision, poles, spin, ward
    ) -> dict[Hashable, PointResult]:
        slice_ = self.slice
        precision = precision or Precision.double()
        spin = dict(spin or {})
        self._spin_legs(spin)
        if ward is not None and not self.ward:
            raise ValueError("the Ward mode was not declared at generation")
        exact_point = tuple(tuple(_fraction(x) for x in momentum) for momentum in point)
        scale = _fraction(scale_squared)
        digits = precision.working_digits
        with _Numbers(precision).scope():
            sources = {leg: _vector(value, digits) for leg, value in spin.items()}
            loop_sources = None
            if ward is not None:
                if ward in spin:
                    raise ValueError("the Ward leg cannot also be a spin source")
                loop_sources = {
                    ward: tuple(
                        (_to_decimal(x), Decimal(0)) for x in exact_point[ward - 1]
                    )
                }
            evaluation = slice_._point(
                exact_point,
                scale,
                dict(parameters or {}),
                precision,
                poles,
                sources=sources or None,
                loop_sources=loop_sources,
                fixed=self._fixed(spin),
            )
            return _ObservableEvaluation(self, evaluation, columns).results


def _vector(value: Any, digits: int) -> tuple[ComplexPair, ...]:
    """A literal complex four-vector as decimal (real, imaginary) pairs."""

    components = list(value)
    if len(components) != 4:
        raise ValueError("a spin source is a complex four-vector")
    out = []
    for component in components:
        if isinstance(component, complex):
            re, im = component.real, component.imag
        elif isinstance(component, (tuple, list)):
            re, im = component
        else:
            re, im = component, 0
        out.append((_to_decimal(_fraction(re)), _to_decimal(_fraction(im))))
    return tuple(out)


# --------------------------------------------------------------------- one point


class _ObservableEvaluation:
    """The columns of one point: seeds, dressing, reduction and assembly."""

    def __init__(self, observables: ObservableSlice, evaluation, columns):
        self.observables = observables
        self.evaluation = evaluation
        self.slice = observables.slice
        self.numbers = evaluation.numbers
        self.columns = list(columns)
        self._seed_columns: list[_SeedColumn] = []
        # (column key, sector) -> (direct numerator column, swapped column)
        self._lanes: dict[tuple[Hashable, Orders], tuple[int, int]] = {}
        self._plan()
        execution = evaluation.dress(
            seeds=ColumnSeeds(
                self._seed_columns,
                self.slice.class_of,
                evaluation.amplitudes,
                self.numbers,
                self.slice.request.accuracy,
            )
        )
        evaluation._reduce(
            execution.numerators, False, columns=list(range(len(self._seed_columns)))
        )
        self.results = {column.key: self._result(column) for column in self.columns}

    # -- columns ---------------------------------------------------------------

    def _plan(self) -> None:
        slice_ = self.slice
        accuracy = slice_.request.accuracy
        degrees = slice_.degrees()["virtual"] if accuracy != "full" else {}
        for column in self.columns:
            tables = (slice_.table, slice_.table)
            if column.directed:
                directed = self.observables.directed(column.history)
                tables = (directed.table, directed.swapped)
            for sector, pairs in slice_.sectors:
                indices = []
                for k, table in enumerate(tables):
                    if k and tables[1] is tables[0]:
                        indices.append(indices[0])
                        continue
                    indices.append(len(self._seed_columns))
                    self._seed_columns.append(
                        _SeedColumn(
                            table,
                            pairs,
                            degrees.get(sector),
                            column.rows,
                            column.basis,
                        )
                    )
                self._lanes[column.key, sector] = tuple(indices)

    # -- assembly --------------------------------------------------------------

    def _result(self, column: Column) -> PointResult:
        slice_, numbers = self.slice, self.numbers
        evaluation = self.evaluation
        width = 3 if evaluation.poles else 1
        born, local = self._local(column)
        average = slice_.normalisation.average
        factor = numbers.pair(0, 1) * (
            1 / (LOOP_FACTOR_DENOMINATOR * numbers.pi * numbers.pi * average)
        )
        omega: dict[int, list[Any]] = {}
        families = {}
        index = {}
        for key, (direct, swapped) in self._lanes.items():
            if key[0] == column.key:
                index[direct] = (key[1], "direct")
                index.setdefault(swapped, (key[1], "swapped"))
        for (family, loop, k), value in evaluation.families.items():
            if k not in index:
                continue
            terms = omega.setdefault(k, [numbers.zero] * width)
            for i, coefficient in enumerate(value.coefficients):
                terms[i] = terms[i] + coefficient
            sector, part = index[k]
            if part == "direct":
                families[family, loop, sector] = value
        real = not column.directed
        zero = numbers.zero
        components: dict[str, dict[Orders, Laurent]] = {name: {} for name in COMPONENTS}
        for sector, _pairs in slice_.sectors:
            direct, swapped = self._lanes[column.key, sector]
            a = omega.get(direct, [zero] * width)
            b = omega.get(swapped, [zero] * width)
            bare = [
                factor * x + (factor * y).conjugate() for x, y in zip(a, b, strict=True)
            ]
            values = {"bare": bare}
            for name in ("r2", "uv_local", FIELD_COMPONENT):
                values[name] = [
                    local.get((name, sector, power), zero)
                    for power in LAURENT_POWERS[:width]
                ]
            values["uv"] = [
                x + y
                for x, y in zip(
                    values["uv_local"], values[FIELD_COMPONENT], strict=True
                )
            ]
            values["virtual"] = [
                x + y + z
                for x, y, z in zip(
                    values["bare"], values["uv"], values["r2"], strict=True
                )
            ]
            for name in COMPONENTS:
                components[name][sector] = Laurent(
                    *[value.real if real else value for value in values[name]]
                )
        inverse = 1 / numbers.real(average)
        born = {
            order: value.real * inverse if real else value * inverse
            for order, value in born.items()
        }
        empty = numbers.real(0) if real else zero
        totals = [_add(born.values(), empty)]
        for name in COMPONENTS:
            for k in range(width):
                totals.append(
                    _add(
                        (
                            laurent.coefficients[k]
                            for laurent in components[name].values()
                        ),
                        empty,
                    )
                )
        return PointResult(
            born,
            components,
            tuple(totals),
            families,
            None,
            colour_contraction="direct",
        )

    def _local(self, column: Column):
        """Born, counterterm and field interferences of one column, complex
        and before the initial average of the virtual pieces is applied to the
        Born only (the local pieces are divided here)."""

        slice_, numbers = self.slice, self.numbers
        evaluation = self.evaluation
        accuracy = slice_.request.accuracy
        degrees = slice_.degrees() if accuracy != "full" else None
        index = {tensor: b for b, tensor in enumerate(slice_.basis.tensors)}
        legs = slice_.basis.legs
        directed = (
            self.observables.directed(column.history) if column.directed else None
        )
        if directed is None:
            direct = swapped = slice_.overlaps
        else:

            def direct(left, right):
                return directed.overlap(left, right)

            def swapped(left, right):
                return directed.overlap(left, right, swapped=True)

        def interfere(bra, ket, degree, overlap, grade=0):
            return _interfere(bra, ket, degree, accuracy, overlap, numbers, grade)

        def degree_of(kind, sector):
            return None if degrees is None else degrees[kind][sector]

        def masked(vector):
            if column.basis is None:
                return vector
            return {
                key: value
                for key, value in vector.items()
                if index[_single(key[0], legs)] in column.basis
            }

        sectors = {pair: sector for sector, pairs in slice_.sectors for pair in pairs}
        powers = LAURENT_POWERS if evaluation.poles else (0,)
        born: dict[Orders, Any] = defaultdict(lambda: numbers.zero)
        local: dict[tuple[str, Orders, int], Any] = defaultdict(lambda: numbers.zero)
        average = slice_.normalisation.average
        for row, vectors in evaluation.born_vectors.items():
            if column.rows is not None and row not in column.rows:
                continue
            bras = {order: masked(vector) for order, vector in vectors.items()}
            for left, right in product(vectors, repeat=2):
                squared = add_orders(left, right)
                born[squared] += interfere(
                    bras[left], vectors[right], degree_of("born", squared), direct
                )
            for (component, power, loop_order), vector in evaluation.site_vectors[
                row
            ].items():
                for order, bra in bras.items():
                    sector = sectors.get((order, loop_order))
                    if sector is None:
                        continue
                    degree = degree_of("virtual", sector)
                    value = interfere(bra, vector, degree, direct)
                    other = (
                        value
                        if directed is None
                        else interfere(bra, vector, degree, swapped)
                    )
                    local[component, sector, power] += value + other.conjugate()
            loop_vectors = evaluation.loop_vectors[row]
            for insertion in slice_.fields:
                name = slice_.field_names[insertion.identity]
                ket = loop_vectors.get(insertion.born_order)
                if ket is None:
                    continue
                for order, bra in bras.items():
                    sector = sectors.get((order, insertion.orders))
                    if sector is None:
                        continue
                    degree = degree_of("virtual", sector)
                    for power in powers:
                        for grade, coupling in evaluation.scalars.get(
                            (name, power), {}
                        ).items():
                            value = coupling * interfere(
                                bra, ket, degree, direct, grade
                            )
                            other = (
                                value
                                if directed is None
                                else coupling
                                * interfere(bra, ket, degree, swapped, grade)
                            )
                            local[FIELD_COMPONENT, sector, power] += (
                                value + other.conjugate()
                            )
        inverse = 1 / numbers.real(average)
        return dict(born), {key: value * inverse for key, value in local.items()}


def _add(values: Iterable[Any], zero: Any) -> Any:
    total = zero
    for value in values:
        total = total + value
    return total


# --------------------------------------------------------------------- flavour


@dataclass(frozen=True)
class FlavourMember:
    """A process evaluated by a representative slice (design §9).

    ``legs[i]`` is the representative leg (0-based) of member leg i, the
    identity for an identity-port member and a permutation within the initial
    and within the final legs for an alias. ``species`` is the signed-PDG
    substitution proved on the representative's retained source plan
    (empty when the member differs only by the leg permutation).
    """

    process: str
    representative: str
    legs: tuple[int, ...]
    species: Mapping[int, int]
    symmetry: Any = field(default=None, repr=False)

    @property
    def alias(self) -> bool:
        return self.legs != tuple(range(len(self.legs)))

    def point(self, point: Sequence[Sequence[Any]]) -> list:
        """The representative's momenta of a member point."""

        out: list = [None] * len(self.legs)
        for member, representative in enumerate(self.legs):
            out[representative] = point[member]
        return out

    def label(self, leg: int) -> int:
        """The representative's 1-based label of member leg ``leg``."""

        return self.legs[leg - 1] + 1

    def history(self, history: ColorCorrelator) -> ColorCorrelator:
        """A member colour history on the representative's labels."""

        from pyamplicol.color.connections import EmitGluon, SplitGluon

        def relabel(label: int) -> int:
            return self.label(label) if label > 0 else label

        def emission(step):
            if isinstance(step, EmitGluon):
                return EmitGluon(
                    relabel(step.emitter_label), relabel(step.emitted_label)
                )
            return SplitGluon(
                relabel(step.parent_label),
                relabel(step.quark_label),
                relabel(step.antiquark_label),
            )

        return ColorCorrelator(
            history.id,
            tuple(map(emission, history.bra)),
            tuple(map(emission, history.ket)),
        )

    def row(self, row: rx.Row) -> rx.Row:
        """The member helicity row of a representative row."""

        inverse = {
            representative: member for member, representative in enumerate(self.legs)
        }
        return tuple(sorted((inverse[leg], h) for leg, h in row))


def admit_member(slice_: LoopSlice, process: str) -> FlavourMember:
    """Admit ``process`` as an identity-port member or a permutation alias of
    the slice's process, or raise ``ValueError``.

    The species substitution, if any, is proved exactly on the slice's
    retained source plan (``prove_flavor_relabeling``); a leg permutation
    must keep the initial and the final states and map every leg to a leg of
    the same (substituted) particle.
    """

    from pyamplicol.loops.flavor_mappings import (
        RetainedFlavorPlan,
        prove_flavor_relabeling,
    )
    from pyamplicol.loops.flavor_transport import identity_species_candidate
    from pyamplicol.processes.model import build_model_process_ir

    representative = slice_.blocks.process
    member = build_model_process_ir(process, slice_.compiled.ir, color_accuracy="full")
    rep_legs, mem_legs = representative.legs, member.legs
    if [leg.side for leg in rep_legs] != [leg.side for leg in mem_legs]:
        raise ValueError("a flavour member keeps the initial and final leg counts")
    sides = {
        side: [i for i, leg in enumerate(rep_legs) if leg.side == side]
        for side in ("initial", "final")
    }
    prepared = slice_.prepared
    plan = None
    candidates = []
    for initial in permutations(sides["initial"]):
        for final in permutations(sides["final"]):
            order = (*initial, *final)
            species = identity_species_candidate(
                tuple(rep_legs[k].pdg for k in order),
                tuple(leg.pdg for leg in mem_legs),
            )
            if species is None:
                continue
            candidates.append((len(species), order, species))
    # Prefer pure permutations, then the identity port.
    candidates.sort(key=lambda item: (item[0], item[1]))
    errors = []
    for _size, order, species in candidates:
        symmetry = None
        if species:
            if plan is None:
                plan = RetainedFlavorPlan(
                    prepared.structure,
                    prepared.counterterms,
                    tuple(rule.name for rule in prepared.structure.native.vertex_rules),
                    tuple(slice_.request.corrections),
                    slice_.continuation,
                )
            try:
                symmetry = prove_flavor_relabeling(plan, species)
            except (ValueError, NotImplementedError) as error:
                errors.append(str(error))
                continue
        return FlavourMember(
            process,
            representative.process,
            tuple(order),
            dict(species),
            symmetry,
        )
    raise ValueError(
        f"{process!r} is neither a proved flavour member nor an alias of "
        f"{representative.process!r}" + (f": {errors[0]}" if errors else "")
    )


class MemberObservables:
    """Observables of an admitted member through its representative."""

    def __init__(self, observables: ObservableSlice, member: FlavourMember):
        self.observables = observables
        self.member = member

    def evaluate(
        self, point, scale_squared, *, correlations=(INCLUSIVE,), spin=None, **options
    ) -> dict[str, PointResult]:
        member = self.member
        observables = self.observables
        histories = []
        for identifier in correlations:
            if identifier == INCLUSIVE:
                continue
            histories.append(member.history(observables.history(identifier)))
        relabelled = replace(observables, histories=tuple(histories))
        relabelled._directed = observables._directed
        return relabelled.evaluate(
            member.point(point),
            scale_squared,
            correlations=correlations,
            spin=None
            if spin is None
            else {member.label(leg): value for leg, value in spin.items()},
            **options,
        )

    def evaluate_resolved(self, point, scale_squared, **options) -> ResolvedObservables:
        member = self.member
        resolved = self.observables.evaluate_resolved(
            member.point(point), scale_squared, **options
        )
        return ResolvedObservables(
            {
                (None if row is None else member.row(row), b): value
                for (row, b), value in resolved.entries.items()
            },
            None
            if resolved.helicities is None
            else tuple(member.row(row) for row in resolved.helicities),
            resolved.components,
        )


__all__ = [
    "BORN_BASIS_COMPONENTS",
    "INCLUSIVE",
    "LAURENT",
    "Column",
    "ColumnSeeds",
    "DirectedColour",
    "FlavourMember",
    "MemberObservables",
    "ObservableSlice",
    "ResolvedObservables",
    "admit_member",
    "connect",
]
