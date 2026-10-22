# SPDX-License-Identifier: 0BSD
"""Cold-path fixed-Nf color projection for bounded physical controls.

The Born observable has its own leading degree. Bare, UV, R2 and external-field
virtual rows share one leading degree across all Laurent sectors, before model
inputs or helicities are evaluated. Distinct currents remain independent rows
unless the caller has an exact identity permitting their combination.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pyamplicol.config.models import ColorAccuracy

from .._internal.physics.symbols import symbols
from .color_bilinears import SUNColorSpace
from .color_projection import prepare_color_projection


def _color_order_domains(bare):
    return (
        tuple(bare.born_term_orders),
        tuple(
            graph.order
            for graph, terms in zip(bare.loop_graphs, bare.loop_finals, strict=True)
            for _ in terms
        ),
        tuple(tuple(sector.pairs) for sector in bare.order_layout.born),
        tuple(
            tuple((pair.born, pair.loop) for pair in sector.pairs)
            for sector in bare.order_layout.virtual
        ),
    )


@dataclass(frozen=True, slots=True)
class FormalColorMetrics:
    """Exact projected rectangles owned by one physical metric revision.

    This record is made at the existing common-degree projection boundary.
    It retains the pre-substitution expressions; it never infers them from
    execution weights at Nc=3. External normalization is applied only by bound().
    """

    nc: Any
    accuracy: str
    revision: int
    initial_average: Any
    born_orders: tuple
    loop_orders: tuple
    born_pairs: tuple
    virtual_pairs: tuple
    born_degrees: tuple[int | None, ...]
    virtual_degrees: tuple[int | None, ...]
    born: tuple[tuple[Any, ...], ...]
    loop: tuple[tuple[Any, ...], ...]
    born_tensors: tuple[Any, ...]
    loop_tensors: tuple[Any, ...]

    def bound(self):
        return tuple(
            tuple(
                tuple(value.replace(self.nc, 3) / self.initial_average for value in row)
                for row in matrix
            )
            for matrix in (self.born, self.loop)
        )

    def require_columns(self, bare):
        """Check actual column/order identities without repeating tensor algebra."""
        domains = (
            self.born_orders,
            self.loop_orders,
            self.born_pairs,
            self.virtual_pairs,
        )
        if domains != _color_order_domains(bare):
            raise ValueError("formal colour columns or order sectors differ from owner")
        for retained, actual in (
            (self.born_tensors, bare.born_color_tensors),
            (self.loop_tensors, bare.loop_color_tensors),
        ):
            if len(retained) != len(actual) or any(
                left is not right for left, right in zip(retained, actual, strict=True)
            ):
                raise ValueError("formal colour tensor columns differ from owner")
        if self.initial_average != bare.initial_average:
            raise ValueError("formal colour normalization differs from owner")

    def _require_bound_weights(self, born, loop):
        for formal, numerical in zip(self.bound(), (born, loop), strict=True):
            if len(formal) != len(numerical) or any(
                len(left) != len(right)
                or any(
                    (a - b).together() != 0 for a, b in zip(left, right, strict=True)
                )
                for left, right in zip(formal, numerical, strict=True)
            ):
                raise ValueError(
                    "formal colour projection differs from execution weights"
                )

    def require_binding(self, bare, born, loop):
        """Check actual columns and their exact numerical binding."""
        self.require_columns(bare)
        self._require_bound_weights(born, loop)

    def require_owner_layout(self, bare):
        """Validate retained metadata/columns without rebuilding bound weights."""
        if (
            getattr(bare, "formal_color_metrics", None) is not self
            or self.revision != getattr(bare, "_color_metrics_revision", 0)
            or self.accuracy != getattr(bare, "color_accuracy", "full")
            or self.born_degrees != tuple(getattr(bare, "born_color_degrees", ()))
            or self.virtual_degrees != tuple(getattr(bare, "virtual_color_degrees", ()))
        ):
            raise ValueError("formal colour metrics have a different projection owner")
        self.require_columns(bare)

    def require_owner(self, bare):
        self.require_owner_layout(bare)
        self._require_bound_weights(bare.born_metric, bare.loop_metric)


@dataclass(frozen=True, slots=True)
class PhysicalColorProjection:
    accuracy: str
    born_degree: int | None
    virtual_degree: int | None
    additional_rows: tuple[Any, ...]
    born_degrees: tuple[int | None, ...] = ()
    virtual_degrees: tuple[int | None, ...] = ()
    formal_metrics: FormalColorMetrics | None = None


def color_projection_metadata(bare):
    """Describe the already compiled fixed-Nf, Nc=3 projection, if present."""
    accuracy = ColorAccuracy(getattr(bare, "color_accuracy", "full")).value
    if not getattr(bare, "_formal_color_projected", False):
        if accuracy != "full":
            raise ValueError("projected colour accuracy has no formal projection")
        return None
    born = tuple(bare.born_color_degrees)
    virtual = tuple(bare.virtual_color_degrees)
    if (
        len(born) != len(bare.order_layout.born)
        or len(virtual) != len(bare.order_layout.virtual)
        or any(
            value is not None and type(value) is not int for value in (*born, *virtual)
        )
    ):
        raise ValueError("colour projection degrees differ from order catalogue")
    return {"accuracy": accuracy, "born_degrees": born, "virtual_degrees": virtual}


def restore_color_projection(bare, metadata):
    """Restore descriptive state only; projector bytecode owns the numerics."""
    bare._formal_color_projected = metadata is not None
    bare.color_accuracy = "full" if metadata is None else metadata["accuracy"]
    bare.born_color_degrees = (
        () if metadata is None else tuple(metadata["born_degrees"])
    )
    bare.virtual_color_degrees = (
        () if metadata is None else tuple(metadata["virtual_degrees"])
    )
    bare.born_color_degree = (
        bare.born_color_degrees[0] if len(bare.born_color_degrees) == 1 else None
    )
    bare.virtual_color_degree = (
        bare.virtual_color_degrees[0] if len(bare.virtual_color_degrees) == 1 else None
    )
    return (
        None
        if metadata is None
        else PhysicalColorProjection(
            bare.color_accuracy,
            bare.born_color_degree,
            bare.virtual_color_degree,
            (),
            bare.born_color_degrees,
            bare.virtual_color_degrees,
            getattr(bare, "formal_color_metrics", None),
        )
    )


def expand_color_conditionals(expression, nc):
    """Collect native Laurent coefficients through Nc-independent ``if``.

    Symbolica owns bottom-up branch mapping and polynomial collection. The
    conditional is linear in its selected branch, so each coefficient retains
    the *same runtime condition*. No mass or kinematic condition is evaluated.
    Nc-dependent conditions and opaque functional Nc dependence are rejected.
    """
    from symbolica import E

    condition, yes, no = symbols.qualified(
        "loop_color_if::condition_", "loop_color_if::yes_", "loop_color_if::no_"
    )
    conditional = symbols.qualified("symbolica::if")

    def coefficients(branch):
        result = {}
        for power, value in branch.together().expand().coefficient_list(nc):
            if nc in value.get_all_symbols():
                raise NotImplementedError(
                    "conditional branch has unsupported functional "
                    "or non-Laurent Nc dependence"
                )
            polynomial = power.to_rational_polynomial(vars=[nc])
            degree = polynomial.numerator().degree(
                nc
            ) - polynomial.denominator().degree(nc)
            if power != nc**degree:
                raise NotImplementedError(
                    "conditional branch requires integer Laurent Nc powers"
                )
            result[power] = value
        return result

    def transform(match):
        test = match[condition]
        if nc in test.get_all_symbols():
            raise NotImplementedError(
                "Nc-dependent conditions require a separate projection prescription"
            )
        left, right = coefficients(match[yes]), coefficients(match[no])
        return sum(
            (
                power
                * conditional(test, left.get(power, E("0")), right.get(power, E("0")))
                for power in left.keys() | right.keys()
            ),
            E("0"),
        )

    # Native replacement callbacks log Python exceptions rather than propagating
    # them. Record rejected branches and raise outside the dependency callback.
    failures = []

    def rewrite(match):
        try:
            return transform(match)
        except Exception as error:
            failures.append(error)
            return conditional(match[condition], match[yes], match[no])

    result = expression.replace(
        conditional(condition, yes, no), rewrite, nested=True, rhs_cache_size=0
    )
    if failures:
        raise NotImplementedError(str(failures[0])) from failures[0]
    return result


def split_color_grades(expressions, nc):
    """Separate exact scalar aliases into Nc powers without merging currents.

    Each returned ``(degree, coefficients)`` retains the original alias order,
    including zero entries. The coefficients contain no Nc. A caller evaluates
    a separate current for each grade and projects ``Nc**degree * overlap``;
    projecting the two factors independently would lose the common structural
    degree. This scalar split does not prove which Lorentz terms an alias feeds.
    """
    from symbolica import E

    expressions = tuple(expressions)
    normalized = tuple(
        expand_color_conditionals(expression, nc).together().expand()
        for expression in expressions
    )
    rows = prepare_color_projection(nc, [[value] for value in normalized]).rows
    grades = sorted({degree for row in rows for degree, _ in row})
    coefficients = tuple(dict(row) for row in rows)
    result = tuple(
        (degree, tuple(row.get(degree, E("0")) for row in coefficients))
        for degree in grades
    )
    for column, expected in enumerate(normalized):
        reconstructed = sum(
            (nc**degree * values[column] for degree, values in result), E("0")
        )
        if (reconstructed - expected).together().expand() != 0:
            raise ValueError("scalar color-grade reconstruction is incomplete")
    return result


def project_physical_color(
    bare, nc, additional_rows, accuracy, *, additional_pairs=None
):
    """Project exact unaveraged rows, then bind Nc=3 and physical averages.

    Additional rows include *both* their model scalar color invariants and
    tensor overlaps, e.g. ``Nc**grade * Born_CT_overlap`` after scalar grading.
    They are returned without an average for the caller's native contraction.
    Each metric cell belongs to the squared-order sector of its exact amplitude
    pair. Additional rows supply those same ordered pairs explicitly for a
    multi-order process. Bare, UV, R2 and field rows share one leading degree per
    virtual sector; Born sectors have their own independent degrees.
    This seam does not infer a model's SU(Nc) continuation or alias incidence.
    """
    from symbolica import E

    accuracy = ColorAccuracy(accuracy)
    additional_rows = tuple(tuple(row) for row in additional_rows)
    if additional_pairs is None:
        if additional_rows and len(bare.order_layout.virtual) != 1:
            raise ValueError(
                "multi-order additional colour rows require exact order pairs"
            )
        additional_owners = (0,) * len(additional_rows)
    else:
        additional_pairs = tuple(additional_pairs)
        if len(additional_pairs) != len(additional_rows):
            raise ValueError("additional colour rows and order pairs differ")
    space = SUNColorSpace(nc)
    born = space.rectangular(bare.born_color_tensors, bare.born_color_tensors)
    loop = space.rectangular(bare.born_color_tensors, bare.loop_color_tensors)
    born_pairs = {
        pair: index
        for index, sector in enumerate(bare.order_layout.born)
        for pair in sector.pairs
    }
    virtual_pairs = {
        (pair.born, pair.loop): index
        for index, sector in enumerate(bare.order_layout.virtual)
        for pair in sector.pairs
    }
    loop_orders = tuple(
        graph.order
        for graph, finals in zip(bare.loop_graphs, bare.loop_finals, strict=True)
        for _ in finals
    )
    if len(loop_orders) != len(bare.loop_color_tensors):
        raise ValueError("loop colour terms and amplitude orders differ")
    born_owners = tuple(
        born_pairs[left, right]
        for left in bare.born_term_orders
        for right in bare.born_term_orders
    )
    loop_owners = tuple(
        virtual_pairs.get((left, right))
        for left in bare.born_term_orders
        for right in loop_orders
    )
    if additional_pairs is not None:
        additional_owners = tuple(virtual_pairs.get(pair) for pair in additional_pairs)

    def project(rows, owners, count):
        projected = [E("0")] * len(rows)
        degrees = []
        for sector in range(count):
            indices = [index for index, owner in enumerate(owners) if owner == sector]
            plan = prepare_color_projection(
                nc,
                [
                    [expand_color_conditionals(sum(rows[index], E("0")), nc)]
                    for index in indices
                ],
            )
            degrees.append(plan.leading_degree)
            for index, value in zip(indices, plan.project(accuracy), strict=True):
                projected[index] = value
        return tuple(projected), tuple(degrees)

    projected_born, born_degrees = project(
        [[value] for row in born for value in row],
        born_owners,
        len(bare.order_layout.born),
    )
    loop_rows = [[value] for row in loop for value in row]
    rows = [*loop_rows, *additional_rows]
    projected_virtual, virtual_degrees = project(
        rows, (*loop_owners, *additional_owners), len(bare.order_layout.virtual)
    )
    width = len(bare.born_colors)
    loops = len(bare.loop_color_tensors)

    formal = FormalColorMetrics(
        nc,
        accuracy.value,
        getattr(bare, "_color_metrics_revision", 0) + 1,
        bare.initial_average,
        *_color_order_domains(bare),
        born_degrees,
        virtual_degrees,
        tuple(tuple(projected_born[i * width : (i + 1) * width]) for i in range(width)),
        tuple(
            tuple(projected_virtual[i * loops : (i + 1) * loops]) for i in range(width)
        ),
        tuple(bare.born_color_tensors),
        tuple(bare.loop_color_tensors),
    )
    bare.set_color_metrics(*formal.bound())
    # Only this actual symbolic projection can install formal data. The public
    # numerical replacement boundary never accepts a record whose authority
    # could otherwise be inferred merely from equality after Nc=3 binding.
    bare.formal_color_metrics = formal
    bare.color_accuracy = accuracy.value
    bare.born_color_degrees = born_degrees
    bare.virtual_color_degrees = virtual_degrees
    bare.born_color_degree = born_degrees[0] if len(born_degrees) == 1 else None
    bare.virtual_color_degree = (
        virtual_degrees[0] if len(virtual_degrees) == 1 else None
    )
    bare._formal_color_projected = True
    return PhysicalColorProjection(
        accuracy.value,
        bare.born_color_degree,
        bare.virtual_color_degree,
        tuple(value.replace(nc, 3) for value in projected_virtual[len(loop_rows) :]),
        born_degrees,
        virtual_degrees,
        formal,
    )
