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
  directed overlaps; the Born is B^dagger K_BB B. Results are complex. At LC
  and NLC every directed output is projected at the leading degree of its
  history's complete family (all diagonal catalogue histories of its order
  and output legs, every pair, both directions; ``colour_family``), and the
  slice's colour pruning keeps the skeletons live in any of those tables.
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

``ObservableSlice`` evaluates with the Python reference executor;
``NativeObservables`` runs a fixed column set in the native v2 executor
(binary64, DoubleFloat, MPFR), one batch per call with per-point spin
sources, for the representative or an admitted member. Both attach rotation
and reduction-direction stability estimates on request
(REQUIREMENTS_ADDENDUM C).
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from decimal import Decimal, localcontext
from fractions import Fraction
from functools import cached_property
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
    LOCAL_COMPONENTS,
    Laurent,
    LoopSlice,
    PointResult,
    Precision,
    SliceRequest,
    _fraction,
    _interfere,
    _max,
    _Numbers,
    _single,
    _to_decimal,
)
from .colour import SeedTable
from .conventions import LOOP_FACTOR_DENOMINATOR
from .ct_sites import LAURENT_POWERS
from .families import Orders, add_orders
from .plan_export import (
    DEFAULT_LANES,
    FailedPoints,
    NativeExecutor,
    SeedColumn,
    export_plan,
)
from .stability import (
    ROTATION,
    Budget,
    StabilityChecks,
    StabilityEstimate,
)

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
        self._seeds = table
        self.symmetric = history.bra == history.ket
        self.table = self._table(table, swapped=False)

    @cached_property
    def swapped(self) -> SeedTable:
        """The seed table of the history with bra and ket exchanged."""

        return self.table if self.symmetric else self._table(self._seeds, True)

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


def colour_family(
    legs: Sequence[Any], history: ColorCorrelator
) -> tuple[ColorConnection, ...]:
    """The complete family of a directed colour history: every diagonal
    catalogue history of its order and output legs (the LC/NLC degree
    authority of design §7.7, legacy ``sun-graded-complete-history-family``).
    The identity history is its own family."""

    from pyamplicol.color.connection_catalogue import (
        build_color_correlator_catalogue,
        canonical_color_connections,
    )

    legs = tuple(legs)
    bra, ket = canonical_color_connections(
        (ColorConnection(legs, history.bra), ColorConnection(legs, history.ket))
    )
    if bra.order == 0:
        return (bra,)
    family = []
    for item in build_color_correlator_catalogue(
        legs, through_order=bra.order, include_physical_splittings=True
    ):
        if item.order != bra.order or item.bra != item.ket:
            continue
        connection = ColorConnection(legs, item.bra)
        if connection.output_legs == bra.output_legs:
            family.append(connection)
    if bra not in family or ket not in family:
        raise ValueError(
            f"colour history {history.id!r} is absent from its complete family"
        )
    return tuple(family)


# --------------------------------------------------------------------- seeds


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
        columns: Sequence[SeedColumn],
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
    result (at the carrier's ``digits``; binary64 when ``None``).
    """

    entries: Mapping[tuple[rx.Row | None, int | None], PointResult]
    helicities: tuple[rx.Row, ...] | None
    components: tuple[int, ...] | None
    component_kind: str = BORN_BASIS_COMPONENTS
    digits: int | None = None

    def total(self) -> PointResult:
        if self.digits is None:
            return _sum_results(list(self.entries.values()))
        with localcontext(prec=self.digits):
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
    # Complete-family LC/NLC degrees per colour family (design §7.7) and
    # the family of each history (bra, ket).
    _families: dict[tuple, dict] = field(default_factory=dict, repr=False)
    _family_of: dict[tuple, tuple] = field(default_factory=dict, repr=False)

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
        # LC/NLC colour histories prune with their own seed tables (``of``).
        prune = not correlators.color_correlations
        slice_ = LoopSlice.generate(compiled, request, native, prune=prune)
        return cls.of(slice_, correlators=correlators, ward=ward)

    @classmethod
    def of(
        cls,
        slice_: LoopSlice,
        *,
        correlators: CorrelatorConfig | None = None,
        ward: bool = False,
    ) -> ObservableSlice:
        """The observables of an already generated slice.

        At LC/NLC a slice generated without colour pruning is pruned here,
        keeping every skeleton live in the inclusive seed table or in the
        seed table of any history pair of a declared directed history's
        complete family (at that family's degrees). A slice already pruned
        for the inclusive virtual alone cannot serve directed histories.
        """

        from pyamplicol.color.correlator_matrices import process_color_legs

        if slice_.request.colour_contraction == "fft":
            raise ValueError(
                "observable columns are contracted directly (design §7.3: no "
                "correlated or resolved seed FFT); generate the slice with "
                "colour contraction 'auto' or 'direct'"
            )
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
        directed = [history for history in out.histories if history.bra]
        if slice_.request.accuracy != "full":
            if slice_.colour_pruned and directed:
                raise ValueError(
                    "an LC/NLC slice pruned for the inclusive virtual cannot "
                    "serve directed colour histories; generate it with "
                    "ObservableSlice.generate"
                )
            if not slice_.colour_pruned:
                criteria: list = []
                for history in directed:
                    family = out.family(history)
                    if family not in out._families:
                        tables: list[SeedTable] = []
                        degrees = out._families[family] = out._family_degrees(
                            family, tables
                        )
                        criteria += [(table, degrees["virtual"]) for table in tables]
                slice_._prune_colour(criteria)
                slice_._plan_contraction()
        for history in out.histories:
            out.directed(history)
        return out

    # -- declarations ---------------------------------------------------------

    def directed(self, history: ColorCorrelator) -> DirectedColour:
        """The directed colour data of ``history`` (built once)."""

        key = (history.bra, history.ket)
        found = self._directed.get(key)
        if found is None:
            found = self._directed[key] = DirectedColour(self.slice.table, history)
        return found

    def degrees(self, history: ColorCorrelator | None) -> dict[str, dict]:
        """Leading N_c degrees P of an output's colour (LC/NLC, design §7.7):
        the slice's for the Born colour metric; for a directed history the
        largest over every history pair (alpha, beta) of its complete family
        (``colour_family``), per squared-order sector, of the Born, seed,
        counterterm and field cells, both directions included."""

        if history is None or not history.bra:
            return self.slice.degrees()
        family = self.family(history)
        found = self._families.get(family)
        if found is None:
            if self.slice.colour_pruned:
                raise ValueError(
                    f"colour history {history.id!r} was not declared when the "
                    "LC/NLC slice was pruned"
                )
            found = self._families[family] = self._family_degrees(family)
        return found

    def family(self, history: ColorCorrelator) -> tuple[ColorConnection, ...]:
        """The complete colour family of ``history`` (built once)."""

        key = (history.bra, history.ket)
        found = self._family_of.get(key)
        if found is None:
            found = self._family_of[key] = colour_family(self.slice.basis.legs, history)
        return found

    def _family_degrees(
        self,
        family: Sequence[ColorConnection],
        tables: list[SeedTable] | None = None,
    ) -> dict[str, dict]:
        """The complete-family degrees; ``tables`` collects every pair's
        seed table (the pruning criteria)."""

        slice_ = self.slice
        out: dict[str, dict] = {"born": {}, "virtual": {}}
        for alpha in family:
            for beta in family:
                pair = DirectedColour(
                    slice_.table,
                    ColorCorrelator("family", alpha.emissions, beta.emissions),
                )
                degrees = slice_.degrees(overlaps=pair.overlap, table=pair.table)
                for kind, values in degrees.items():
                    for sector, value in values.items():
                        out[kind][sector] = _max(out[kind].get(sector), value)
                if tables is not None:
                    tables.append(pair.table)
        return out

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
        stability: StabilityChecks | None = None,
    ) -> dict[str, PointResult]:
        """Correlated outputs at one point, one ``PointResult`` per history id.

        ``spin`` maps source legs (a declared spin class) to literal complex
        four-vectors: four real decimals or (real, imaginary) pairs.
        ``ward`` is the vector leg of the asymmetric Ward mode. Directed
        outputs are complex; the others real. ``stability`` runs rotation
        and reduction-direction checks per output (``stability``; spin
        vectors rotate with the momenta); without checks no estimate exists.
        """

        columns = [
            Column(identifier, self.history(identifier)) for identifier in correlations
        ]
        results = self._run(
            columns,
            point,
            scale_squared,
            parameters,
            precision,
            poles,
            spin,
            ward,
            stability,
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
        stability: StabilityChecks | None = None,
    ) -> ResolvedObservables:
        """Helicity- and/or colour-resolved outputs (design §8.5)."""

        columns = self.resolved_columns(resolution, spin)
        results = self._run(
            columns,
            point,
            scale_squared,
            parameters,
            precision,
            poles,
            spin,
            None,
            stability,
        )
        return self.resolved_result(columns, results, precision)

    def project_loop(
        self,
        point: Sequence[Sequence[Any]],
        scale_squared: Any,
        tensors: Sequence[ConnectedColorTensor],
        *,
        rows: Iterable[rx.Row] | None = None,
        parameters: Mapping[str, str] | None = None,
        precision: Precision | None = None,
        poles: bool = True,
    ) -> dict[tuple[rx.Row, int, Orders], Laurent]:
        """Colour projections <T_k|M(h)> of the bare loop amplitude.

        Resolved seed columns on arbitrary literal colour structures
        ``tensors`` (e.g. a published colour decomposition, oracle O14): the
        seed of column (h, k) is K'[c, k] = <T_k|kappa_c> on Born helicity row
        h with a unit Born, so its numerator is the loop amplitude's colour
        projection, i/(16 pi^2) sum_c K'[c, k] L_c(h), at amplitude level
        (complex, no initial average) per loop-amplitude order. Full colour.
        """

        from .colour import BornBasis

        (precision or Precision.double()).require_reference()
        slice_ = self.slice
        if slice_.request.accuracy != "full":
            raise ValueError("loop colour projections are full colour")
        table = slice_.table
        legs = table.basis.legs
        tensors = tuple(tensors)
        if any(tensor.legs != legs for tensor in tensors):
            raise ValueError("projection tensors need the process's colour legs")
        unit: Orders = ()
        basis = BornBasis(
            legs,
            tuple(range(len(tensors))),
            tensors,
            tuple(frozenset({unit}) for _ in tensors),
        )
        cells = tuple(
            tuple(
                contract_connected_tensor_nc_terms(tensor, table.tensor(c))
                for tensor in tensors
            )
            for c in range(len(table.classes))
        )
        projection = replace(table, basis=basis, cells=cells)
        loops = sorted({order for orders in table.orders for order in orders})
        rows = tuple(self.rows if rows is None else rows)
        evaluation = self.point(point, scale_squared, parameters, precision, poles)
        numbers = evaluation.numbers
        columns, labels = [], []
        for row in rows:
            for k in range(len(tensors)):
                for loop in loops:
                    columns.append(
                        SeedColumn(
                            projection,
                            frozenset({(unit, loop)}),
                            None,
                            frozenset({row}),
                            frozenset({k}),
                        )
                    )
                    labels.append((row, k, loop))
        with numbers.scope():
            one = {0: numbers.pair(1)}
            born = {row: {(k, unit): one for k in range(len(tensors))} for row in rows}
            seeds = ColumnSeeds(columns, slice_.class_of, born, numbers, "full")
            execution = evaluation.dress(seeds=seeds)
            evaluation._reduce(
                execution.numerators, None, columns=list(range(len(columns)))
            )
            factor = numbers.pair(0, 1) * (
                1 / (LOOP_FACTOR_DENOMINATOR * numbers.pi * numbers.pi)
            )
            width = 3 if poles else 1
            out = {label: [numbers.zero] * width for label in labels}
            for (_family, _loop, k), value in evaluation.families.items():
                terms = out[labels[k]]
                for i, coefficient in enumerate(value.coefficients):
                    terms[i] = terms[i] + factor * coefficient
            return {label: Laurent(*terms) for label, terms in out.items()}

    def resolved_columns(
        self, resolution: Resolution, spin: Mapping[int, Any] | None = None
    ) -> list[Column]:
        """One column per (helicity row, Born basis element) of a resolution;
        a spin source leg keeps its source row only."""

        if resolution not in ("helicity", "colour", "helicity-colour"):
            raise ValueError(f"unknown resolution {resolution!r}")
        rows = self._rows_of(spin) if "helicity" in resolution else (None,)
        components = self.components if "colour" in resolution else (None,)
        return [
            Column(
                (row, b),
                None,
                None if row is None else frozenset({row}),
                None if b is None else frozenset({b}),
            )
            for row in rows
            for b in components
        ]

    @staticmethod
    def resolved_result(
        columns: Sequence[Column],
        results: Mapping[Hashable, PointResult],
        precision: Precision | None = None,
    ) -> ResolvedObservables:
        rows = tuple(dict.fromkeys(column.key[0] for column in columns))
        components = tuple(dict.fromkeys(column.key[1] for column in columns))
        precision = precision or Precision.double()
        return ResolvedObservables(
            {column.key: results[column.key] for column in columns},
            None if rows == (None,) else rows,
            None if components == (None,) else components,
            digits=None
            if precision.arithmetic == "double"
            else precision.working_digits,
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
        self,
        columns,
        point,
        scale_squared,
        parameters,
        precision,
        poles,
        spin,
        ward,
        stability=None,
    ) -> dict[Hashable, PointResult]:
        plan = ColumnPlan(self, columns)
        stability = stability or StabilityChecks()
        precision = precision or Precision.double()
        precision.require_reference()
        reference = self._evaluated(
            plan,
            point,
            scale_squared,
            parameters,
            precision,
            poles,
            spin,
            ward,
            stability.frame,
        )
        results = plan.results(reference)
        if not stability.enabled:
            return results
        rotated = []
        for rotation in stability.rotations:
            evaluation = self._evaluated(
                plan,
                _rotated_point(rotation, point),
                scale_squared,
                parameters,
                precision,
                poles,
                _rotated_spin(rotation, spin),
                ward,
                None,
            )
            rotated.append((evaluation, plan.results(evaluation)))
        return plan.checked(
            results, reference, rotated, precision.unit_roundoff, stability.reduction
        )

    def _evaluated(
        self,
        plan,
        point,
        scale_squared,
        parameters,
        precision,
        poles,
        spin,
        ward,
        retest,
    ):
        """One point's evaluation with the plan's lanes reduced (and
        re-reduced in the re-test frame ``retest``, ``stability.FRAMES``)."""

        evaluation = self.point(
            point, scale_squared, parameters, precision, poles, spin, ward
        )
        with evaluation.numbers.scope():
            execution = evaluation.dress(seeds=plan.seeds(evaluation))
            evaluation._reduce(
                execution.numerators, retest, columns=list(range(plan.width))
            )
        return evaluation

    def point(
        self,
        point,
        scale_squared,
        parameters=None,
        precision=None,
        poles=True,
        spin=None,
        ward=None,
    ):
        """The slice's point evaluation (trees, Born, counterterm and field
        vectors per helicity row) with the spin sources or the Ward vector."""

        precision = precision or Precision.double()
        spin = dict(spin or {})
        self._spin_legs(spin)
        if ward is not None and not self.ward:
            raise ValueError("the Ward mode was not declared at generation")
        exact_point = tuple(tuple(_fraction(x) for x in momentum) for momentum in point)
        with _Numbers(precision).scope():
            sources = {leg: _vector(value) for leg, value in spin.items()}
            loop_sources = None
            if ward is not None:
                if ward in spin:
                    raise ValueError("the Ward leg cannot also be a spin source")
                loop_sources = {
                    ward: tuple(
                        (_to_decimal(x), Decimal(0)) for x in exact_point[ward - 1]
                    )
                }
            return self.slice._point(
                exact_point,
                _fraction(scale_squared),
                dict(parameters or {}),
                precision,
                poles,
                sources=sources or None,
                loop_sources=loop_sources,
                fixed=self._fixed(spin),
            )


def _components(value: Any) -> list[tuple[Fraction, Fraction]]:
    """A literal complex four-vector as exact (real, imaginary) pairs."""

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
        out.append((_fraction(re), _fraction(im)))
    return out


def _rotated(rotation: Any, value: Any) -> list[tuple[Fraction, Fraction]]:
    """A spin vector rotated with the momenta (real and imaginary parts)."""

    parts = _components(value)
    real = rotation.apply([[re for re, _im in parts]])[0]
    imag = rotation.apply([[im for _re, im in parts]])[0]
    return list(zip(real, imag, strict=True))


def _rotated_point(rotation: Any, point: Sequence[Sequence[Any]]) -> Any:
    return rotation.apply(tuple(tuple(_fraction(x) for x in p) for p in point))


def _rotated_spin(rotation: Any, spin: Mapping[int, Any] | None) -> Any:
    """Spin sources rotated with the momenta (``None`` stays ``None``)."""

    if spin is None:
        return None
    return {leg: _rotated(rotation, value) for leg, value in spin.items()}


def _vector(value: Any) -> tuple[ComplexPair, ...]:
    """A literal complex four-vector as decimal (real, imaginary) pairs."""

    return tuple((_to_decimal(re), _to_decimal(im)) for re, im in _components(value))


# --------------------------------------------------------------------- one point


class ColumnPlan:
    """The numerator columns of a set of outputs (point independent).

    Per output and squared-order sector one seed column, two for a directed
    history that is not symmetric (the direct and the swapped table);
    ``lanes[key, sector]`` gives (direct, swapped) column indices, equal for
    a single column. ``seeds`` builds the Python seed source of a point and
    ``results`` assembles the outputs from the point's reduced lanes
    ``evaluation.families[(family, loop order, column index)]`` (from the
    Python executor or the native one).
    """

    def __init__(self, observables: ObservableSlice, columns: Iterable[Column]):
        self.observables = observables
        self.slice = slice_ = observables.slice
        self.columns = list(columns)
        self.seed_columns: list[SeedColumn] = []
        self.lanes: dict[tuple[Hashable, Orders], tuple[int, int]] = {}
        accuracy = slice_.request.accuracy
        for column in self.columns:
            degrees = (
                observables.degrees(column.history)["virtual"]
                if accuracy != "full"
                else {}
            )
            tables = (slice_.table, slice_.table)
            if column.directed:
                directed = observables.directed(column.history)
                tables = (directed.table, directed.swapped)
            for sector, pairs in slice_.sectors:
                indices: list[int] = []
                for k, table in enumerate(tables):
                    if k and tables[1] is tables[0]:
                        indices.append(indices[0])
                        continue
                    indices.append(len(self.seed_columns))
                    self.seed_columns.append(
                        SeedColumn(
                            table,
                            pairs,
                            degrees.get(sector),
                            column.rows,
                            column.basis,
                        )
                    )
                self.lanes[column.key, sector] = (indices[0], indices[1])

    @property
    def width(self) -> int:
        return len(self.seed_columns)

    def seeds(self, evaluation) -> ColumnSeeds:
        return ColumnSeeds(
            self.seed_columns,
            self.slice.class_of,
            evaluation.amplitudes,
            evaluation.numbers,
            self.slice.request.accuracy,
        )

    def results(self, evaluation) -> dict[Hashable, PointResult]:
        with evaluation.numbers.scope():
            assembly = _Assembly(self, evaluation)
            return {column.key: assembly.result(column) for column in self.columns}

    def _lanes_of(self, key: Hashable) -> dict[int, tuple[Orders, int]]:
        """Numerator column -> (sector, multiplicity in the output's bare):
        2 for a single column (2 Re), 1 each for a direct/swapped pair."""

        out: dict[int, tuple[Orders, int]] = {}
        for (column, sector), (direct, swapped) in self.lanes.items():
            if column == key:
                if direct == swapped:
                    out[direct] = (sector, 2)
                else:
                    out[direct] = (sector, 1)
                    out[swapped] = (sector, 1)
        return out

    def _budget(self, key, lanes, width: int, local=()) -> dict[Orders, Budget]:
        """Per sector sum_lanes m |delta Omega| / (16 pi^2 average) plus the
        absolute local components (``assemble._PointEvaluation._lane_budget``
        for observable columns)."""

        factor = 1 / (
            LOOP_FACTOR_DENOMINATOR * math.pi**2 * self.slice.normalisation.average
        )
        mine = self._lanes_of(key)
        budget = {sector: [0.0] * width for sector, _pairs in self.slice.sectors}
        for (_family, _loop, column), errors in lanes.items():
            found = mine.get(column)
            if found is None:
                continue
            sector, multiplicity = found
            for k, error in enumerate(errors):
                budget[sector][k] += multiplicity * factor * abs(complex(error))
        for components in local:
            for sector, laurent in components.items():
                for k, value in enumerate(laurent.coefficients):
                    budget[sector][k] += abs(complex(value))
        return {sector: Budget(tuple(values)) for sector, values in budget.items()}

    def checked(
        self,
        results: Mapping[Hashable, PointResult],
        reference,
        rotated: Sequence[tuple[Any, Mapping[Hashable, PointResult]]],
        roundoff: float,
        reduction: str | None,
    ) -> dict[Hashable, PointResult]:
        """``results`` of the evaluation ``reference`` with their stability
        estimates from the rotated (evaluation, results) runs and, for the
        reduction check ``reduction`` (``StabilityChecks.reduction``), the
        reference's re-test lanes."""

        return {
            key: replace(
                result,
                stability=self.estimate(
                    key, reference, result, rotated, roundoff, reduction
                ),
            )
            for key, result in results.items()
        }

    def estimate(
        self,
        key,
        reference,
        result: PointResult,
        rotated,
        roundoff: float,
        reduction: str | None,
    ) -> StabilityEstimate:
        """The stability estimate of output ``key`` (``stability``): rotation
        budgets from the rotated evaluations' lanes and local components, the
        budget of the reduction check ``reduction`` (reversed or rescaled
        frame) from the re-test lanes."""

        width = 3 if reference.poles else 1
        local = [result.components[name] for name in LOCAL_COMPONENTS]
        scale = self._budget(
            key,
            {lane: value.coefficients for lane, value in reference.families.items()},
            width,
            local,
        )
        budgets = {}
        zero = Laurent(*[0] * width)
        for index, (evaluation, results) in enumerate(rotated):
            other = results[key]
            lanes = {
                lane: tuple(
                    complex(a) - complex(b)
                    for a, b in zip(
                        evaluation.families.get(lane, zero).coefficients,
                        reference.families.get(lane, zero).coefficients,
                        strict=True,
                    )
                )
                for lane in reference.families.keys() | evaluation.families.keys()
            }
            changes = [
                {
                    sector: Budget(
                        tuple(
                            abs(complex(a) - complex(b))
                            for a, b in zip(
                                laurent.coefficients,
                                other.components[name][sector].coefficients,
                                strict=True,
                            )
                        )
                    )
                    for sector, laurent in result.components[name].items()
                }
                for name in LOCAL_COMPONENTS
            ]
            budgets[ROTATION.format(index)] = self._budget(key, lanes, width, changes)
        deviations = None
        if reduction is not None:
            if reference.retest_families is None:
                raise ValueError(f"the {reduction} check did not run")
            deviations = {
                lane: tuple(
                    abs(complex(a) - complex(b))
                    for a, b in zip(
                        value.coefficients,
                        reference.retest_families[lane].coefficients,
                        strict=True,
                    )
                )
                for lane, value in reference.families.items()
            }
            budgets[reduction] = self._budget(key, deviations, width)
        magnitudes = {
            sector: Laurent(*[abs(complex(x)) for x in laurent.coefficients])
            for sector, laurent in result.components["virtual"].items()
        }
        return StabilityEstimate.from_checks(
            magnitudes,
            budgets,
            scale=scale,
            roundoff=roundoff,
            families=deviations,
        )


class _Assembly:
    """The outputs of one point from its reduced lanes and local vectors."""

    def __init__(self, plan: ColumnPlan, evaluation):
        self.plan = plan
        self.observables = plan.observables
        self.slice = plan.slice
        self.evaluation = evaluation
        self.numbers = evaluation.numbers
        self._lanes = plan.lanes

    # -- assembly --------------------------------------------------------------

    def result(self, column: Column) -> PointResult:
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
        degrees = (
            self.observables.degrees(column.history) if accuracy != "full" else None
        )
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


# --------------------------------------------------------------------- native


class NativeObservables:
    """A fixed set of observable columns run by the native v2 executor.

    The plan is exported with the columns' seed weights (directed tables,
    Born basis masks) and, for helicity-resolved columns, their helicity rows
    (``column_rows``); spin sources and the Ward vector change only the batch
    inputs (Born components and block rows). Assembly is ``ColumnPlan``'s.
    """

    def __init__(
        self,
        observables: ObservableSlice,
        columns: Iterable[Column],
        *,
        member: FlavourMember | None = None,
        lanes: int = DEFAULT_LANES,
        threads: int | None = None,
    ):
        self.observables = observables
        self.member = member
        self.plan = ColumnPlan(observables, columns)
        exported = export_plan(
            observables.slice, forward=False, columns=self.plan.seed_columns
        )
        self.executor = NativeExecutor(
            observables.slice, exported=exported, lanes=lanes, threads=threads
        )

    @classmethod
    def correlated(
        cls,
        observables: ObservableSlice,
        correlations: Sequence[str],
        *,
        member: FlavourMember | None = None,
        **options,
    ) -> NativeObservables:
        """Colour-history outputs (of an admitted ``member``: its histories
        on the representative's labels)."""

        columns = (
            [Column(name, observables.history(name)) for name in correlations]
            if member is None
            else member.columns(observables, correlations)
        )
        return cls(observables, columns, member=member, **options)

    @classmethod
    def resolved(
        cls,
        observables: ObservableSlice,
        resolution: Resolution = "helicity-colour",
        *,
        spin_legs: Iterable[int] = (),
        member: FlavourMember | None = None,
        **options,
    ) -> NativeObservables:
        """Resolved outputs; ``spin_legs`` are the source legs the evaluations
        will carry (each keeps one helicity row)."""

        sources = dict.fromkeys(spin_legs)
        if member is not None:
            sources = member.spin(sources)
        return cls(
            observables,
            observables.resolved_columns(resolution, sources),
            member=member,
            **options,
        )

    def evaluate(
        self,
        points: Sequence[Sequence[Sequence[Any]]],
        scales: Sequence[Any],
        *,
        spin: Mapping[int, Any] | Sequence[Mapping[int, Any] | None] | None = None,
        ward: int | None = None,
        parameters: Mapping[str, str] | None = None,
        precision: Precision | None = None,
        poles: bool = True,
        stability: StabilityChecks | None = None,
    ) -> list[dict[Hashable, PointResult]]:
        """The outputs of every point, dressed and reduced in one native batch.

        ``spin`` is one source assignment for every point or one per point
        (``None`` entries: ordinary helicities). ``precision`` selects the
        native carrier: binary64, double-double (``Precision.double_double``:
        trees formed at its working digits, dressing and reduction in
        DoubleFloat) or MPFR. ``stability`` re-evaluates the rotated points
        (spin vectors rotate with the momenta) and runs the native re-test in
        the reduction check's frame (``stability.StabilityChecks``); without
        checks no estimate exists (``stability=None``). A failed point fails
        alone, also in a rotation check's evaluation: ``FailedPoints`` carries
        the complete outputs of the others (``None`` for a failed point)."""

        stability = stability or StabilityChecks()
        spins = (
            list(spin)
            if spin is not None and not isinstance(spin, Mapping)
            else [spin] * len(points)
        )
        if len(spins) != len(points):
            raise ValueError("one spin source assignment per point")
        member = self.member
        if member is not None:
            points = [member.point(point) for point in points]
            spins = [member.spin(value) for value in spins]
            ward = None if ward is None else member.label(ward)
        batch = _Batch(
            tuple(scales), parameters, precision or Precision.double(), poles, ward
        )
        references, failures = self._evaluations(batch, points, spins, stability.frame)
        runs = []
        for k, rotation in enumerate(stability.rotations):
            evaluations, failed = self._evaluations(
                batch,
                [_rotated_point(rotation, point) for point in points],
                [_rotated_spin(rotation, value) for value in spins],
                None,
            )
            runs.append(evaluations)
            for index, reasons in failed.items():
                failures.setdefault(index, []).extend(
                    f"{ROTATION.format(k)}: {reason}" for reason in reasons
                )
        out: list[dict[Hashable, PointResult] | None] = []
        for index, reference in enumerate(references):
            if index in failures:
                out.append(None)
                continue
            result = self.plan.results(reference)
            if stability.enabled:
                result = self.plan.checked(
                    result,
                    reference,
                    [(run[index], self.plan.results(run[index])) for run in runs],
                    batch.precision.unit_roundoff,
                    stability.reduction,
                )
            if member is not None:
                result = {_member_key(member, key): v for key, v in result.items()}
            out.append(result)
        if failures:
            raise FailedPoints(out, failures)
        return out

    def _evaluations(
        self, batch: _Batch, points, spins, retest: str | None
    ) -> tuple[list, dict[int, list[str]]]:
        """Point evaluations with the plan's lanes reduced natively (and
        re-reduced in the frame ``retest``); also the failures per point."""

        evaluations = [
            self.observables.point(
                point,
                scale,
                batch.parameters,
                batch.precision,
                batch.poles,
                spin,
                batch.ward,
            )
            for point, scale, spin in zip(points, batch.scales, spins, strict=True)
        ]
        failures = self.executor.reduce(
            evaluations, batch.precision, batch.poles, retest
        )
        return evaluations, failures


@dataclass(frozen=True)
class _Batch:
    """The point-independent inputs of one native observable batch."""

    scales: tuple[Any, ...]
    parameters: Mapping[str, str] | None
    precision: Precision
    poles: bool
    ward: int | None


def _member_key(member: FlavourMember, key: Hashable) -> Hashable:
    """A member's output key: resolved (row, component) keys carry member
    helicity rows; history ids are unchanged."""

    if isinstance(key, tuple):
        row, component = key
        return (None if row is None else member.row(row), component)
    return key


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

    def spin(self, spin: Mapping[int, Any] | None) -> dict[int, Any] | None:
        """Member spin sources on the representative's labels."""

        if spin is None:
            return None
        return {self.label(leg): value for leg, value in spin.items()}

    def columns(
        self, observables: ObservableSlice, correlations: Iterable[str]
    ) -> list[Column]:
        """The output columns of member colour histories (declared by id on
        the representative, read on the member's labels)."""

        return [
            Column(name, self.history(observables.history(name)))
            for name in correlations
        ]

    def resolved(self, resolved: ResolvedObservables) -> ResolvedObservables:
        """Representative resolved outputs keyed by member helicity rows."""

        return replace(
            resolved,
            entries={
                (None if row is None else self.row(row), b): value
                for (row, b), value in resolved.entries.items()
            },
            helicities=None
            if resolved.helicities is None
            else tuple(self.row(row) for row in resolved.helicities),
        )


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
    """Observables of an admitted member through its representative: member
    momenta, spin legs, the Ward leg and colour histories are relabelled onto
    the representative, resolved helicity rows back onto the member."""

    def __init__(self, observables: ObservableSlice, member: FlavourMember):
        self.observables = observables
        self.member = member

    def evaluate(
        self,
        point,
        scale_squared,
        *,
        correlations=(INCLUSIVE,),
        spin=None,
        ward=None,
        **options,
    ) -> dict[str, PointResult]:
        member = self.member
        observables = self.observables
        columns = member.columns(observables, correlations)
        results = observables._run(
            columns,
            member.point(point),
            scale_squared,
            options.get("parameters"),
            options.get("precision"),
            options.get("poles", True),
            member.spin(spin),
            None if ward is None else member.label(ward),
            options.get("stability"),
        )
        return {column.key: results[column.key] for column in columns}

    def evaluate_resolved(
        self, point, scale_squared, *, spin=None, **options
    ) -> ResolvedObservables:
        member = self.member
        resolved = self.observables.evaluate_resolved(
            member.point(point), scale_squared, spin=member.spin(spin), **options
        )
        return member.resolved(resolved)


__all__ = [
    "BORN_BASIS_COMPONENTS",
    "INCLUSIVE",
    "LAURENT",
    "Column",
    "ColumnPlan",
    "ColumnSeeds",
    "DirectedColour",
    "FlavourMember",
    "MemberObservables",
    "NativeObservables",
    "ObservableSlice",
    "ResolvedObservables",
    "admit_member",
    "connect",
]
