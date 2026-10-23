# SPDX-License-Identifier: 0BSD
"""Assembly of the v2 vertical slice: from a process to the complete virtual.

``LoopSlice.generate`` builds every generation object of one loop request once
(design §4): tree blocks and the Born (WP1.2), families and B tries (WP1.3),
generic colour seeds (WP1.4), counterterm sites and external fields (WP1.7)
and the kernel catalogue (WP1.1). ``LoopSlice.evaluate`` then runs, per phase-
space point and in binary64 or arbitrary precision (design §5.1):

1. the tree runtime's block currents (``TreeRows``) and the Born colour vector
   per helicity row, closed at the vertex of leg 1 (``conventions``);
2. seeds U0 = sum_b K[c, b] conj(A0_b) per squared-order sector (one seed
   column per sector, so every family numerator splits by coupling order);
3. schedule B of the Python reference executor (WP1.5) with the leaf weights
   and the tree reading of ``conventions``;
4. NinjaRS reductions of every (family, loop order, sector) lane through the
   v2 adapter (WP1.6), with the reduction-direction check on request;
5. the counterterm sites and field factors contracted with the Born;
6. the six components [virtual, bare, uv, r2, uv_local, uv_external_fields]
   per sector and Laurent coefficient, with virtual = bare + uv_local + r2 +
   uv_external_fields and uv = uv_local + uv_external_fields, and the 19 totals
   over sectors (Born, then the six components times three coefficients).

Runtime options (REQUIREMENTS_ADDENDUM B and C):

* ``poles=False`` evaluates the same generated slice in finite-only mode: the
  reducer returns no pole columns, counterterm and field couplings are bound
  only at eps^0, and the IR-pole check is skipped. Pole coefficients are then
  absent (``None``), not zero.
* ``stability`` requests estimates (``stability.StabilityChecks``): rotation
  checks re-evaluate the point with rotated momenta, the reduction-direction
  check re-reduces every lane in the reversed frame; each check's error budget
  sums the deviations of every reduction lane (and, for rotations, of the
  local components), and ``stability.StabilityEstimate`` combines them with a
  calibrated safety factor and rounding floor, for the Laurent vector and per
  coefficient. With no check requested the result carries no estimate.

Colour (design §7). Colour vectors keep every literal tensor's N_c power,
and with a source-owned continuation (``SliceRequest.colour_declaration``)
the counterterm and field couplings are split into exact N_c grades
(``loops.source_color``); whole-rule continuations (the general-N_c
four-gluon R2) replace their fixed-SU(3) rule by checked terms
(``WholeRuleTerm``). Full colour evaluates every power at N_c = 3; leading and
next-to-leading colour keep, per squared-order sector, the powers P (LC) or
P, P-1 and P-2 (NLC) of the leading degree P of the Born square and of the
virtual (seed, counterterm and field cells, see ``LoopSlice.degrees``) before
substituting N_c = 3, and drop every skeleton whose seed cells lie below
them before any numerics. LC and NLC need the continuation. Seeds use the
certified colour compressions by default (``SliceRequest.compress_colour``).

The IR-pole check (design §10.2 item 3, O9) predicts the 1/eps^2 and 1/eps
coefficients of the renormalised virtual from Catani's operator, built from
the same Born colour vectors and exact colour-correlation matrices (QCD
corrections at full colour, massless coloured legs), with alpha_s and the
light flavours taken from the bound model parameters.

This is the Python reference path of the slice (gate S1); the native executor
(WP2.1) replaces steps 1-5 without changing the conventions.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import product
from typing import Any, Literal

from pyamplicol.color.connections import (
    ColorConnection,
    ConnectedColorTensor,
    EmitGluon,
    ExactColorCoefficient,
    TensorTerm,
    apply_color_connection,
    contract_connected_tensor_nc_terms,
    evaluate_color_nc_terms,
)
from pyamplicol.color.contraction_types import NC

from . import generator_input
from . import reference_executor as rx
from .blocks import TreeBlocks, build_tree_blocks, requested_loop_blocks
from .catalogue import build_loop_kernel_catalogue
from .colour import (
    SYMMETRIC_GROUP_FFT,
    BornBasis,
    ColourAccuracy,
    ContractionPolicy,
    SeedColour,
    SeedCombination,
    SeedFourier,
    SeedTable,
    _adjoint_rename,
    _colour_key,
    _relabelled_colour,
    build_seed_table,
    extend_seed_table,
    kept,
    seed_fourier,
)
from .conventions import (
    FERMION_SPIN,
    LOOP_FACTOR_DENOMINATOR,
    LeafWeights,
    Normalisation,
    TreeReading,
)
from .ct_sites import (
    LAURENT_POWERS,
    CountertermSites,
    SiteClosure,
    SiteTerm,
    generate_counterterm_sites,
)
from .ct_sites import (
    Carrier as SiteCarrier,
)
from .families import (
    ZERO,
    FamilyKey,
    FamilySet,
    LoopOrderSelection,
    ModelBlockTable,
    ModelLoopCatalogue,
    Orders,
    _set_partitions,
    add_orders,
    generate_families,
    mass_name,
    shifts,
)
from .fields import FIELD_COMPONENT, FieldInsertion, external_field_insertions
from .generator_input import NativeGeneration
from .poly_affine import MonomialLayout, leg_momentum
from .schedules import (
    STRUCTURED,
    LoopSchedules,
    SkeletonView,
    build_schedules,
    cheapest,
)
from .stability import (
    ROTATION,
    Budget,
    StabilityChecks,
    StabilityEstimate,
    differences,
)

COMPONENTS = ("virtual", "bare", "uv", "r2", "uv_local", FIELD_COMPONENT)
# Components summed from local insertions (no reduction lane).
LOCAL_COMPONENTS = ("r2", "uv_local", FIELD_COMPONENT)
LAURENT = ("finite", "single_pole", "double_pole")
GUARD_DIGITS = 20
# Interpreted Symbolica replays derive their precision from the inputs' digits.
_SYMBOLICA_PRECISION_FLOOR = 40
# Decimal digits of a DoubleFloat (106 bits): the inputs of a double-double
# evaluation are formed with these plus the guard digits.
DOUBLE_DOUBLE_DIGITS = 32


# --------------------------------------------------------------------- precision


@dataclass(frozen=True, slots=True)
class Precision:
    """Arithmetic of an evaluation: binary64, double-double or ``digits``
    decimal digits.

    Double-double (``DoubleFloat``) is a carrier of the native executor
    (``plan_export.NativeExecutor``): its per-point inputs (trees, Born,
    counterterm scalars) are formed in Decimal arithmetic at
    ``working_digits``, dressing and reduction run in DoubleFloat. The Python
    reference path has no double-double arithmetic and refuses it.
    """

    arithmetic: Literal["double", "double-double", "arbitrary"] = "double"
    digits: int | None = None

    def __post_init__(self) -> None:
        if self.arithmetic in ("double", "double-double"):
            if self.digits is not None:
                raise ValueError(f"{self.arithmetic} takes no digits")
        elif self.arithmetic == "arbitrary":
            if not isinstance(self.digits, int) or self.digits < 1:
                raise ValueError("arbitrary precision needs a positive digit count")
        else:
            raise ValueError(f"unknown arithmetic {self.arithmetic!r}")

    @classmethod
    def double(cls) -> Precision:
        return cls("double")

    @classmethod
    def double_double(cls) -> Precision:
        return cls("double-double")

    @classmethod
    def arbitrary(cls, digits: int) -> Precision:
        return cls("arbitrary", digits)

    @property
    def unit_roundoff(self) -> float:
        """The unit roundoff: 2^-53 in binary64, 2^-104 in double-double (a
        DoubleFloat operation rounds to about 2^-104 relative, not to its
        106-bit significand), half a unit of the last of ``digits`` decimal
        digits otherwise."""
        if self.arithmetic == "double":
            return 2.0**-53
        if self.arithmetic == "double-double":
            return 2.0**-104
        return 0.5 * 10.0 ** (1 - self.digits)

    @property
    def working_digits(self) -> int:
        """Decimal digits of the Python arithmetic and of the tree replay."""
        if self.arithmetic == "double":
            return _SYMBOLICA_PRECISION_FLOOR
        digits = DOUBLE_DOUBLE_DIGITS if self.digits is None else self.digits
        return max(digits + GUARD_DIGITS, _SYMBOLICA_PRECISION_FLOOR)

    def require_reference(self) -> None:
        """Refuse a carrier the Python reference path has no arithmetic for."""
        if self.arithmetic == "double-double":
            raise ValueError(
                "double-double evaluations run in the native executor "
                "(plan_export.NativeExecutor)"
            )


class _Numbers:
    """Carrier numbers of one precision: Python complex or Decimal ``rx.Complex``."""

    def __init__(self, precision: Precision):
        self.precision = precision
        self.double = precision.arithmetic == "double"
        self.carrier = (
            rx.Carrier.f64() if self.double else rx.Carrier.ap(precision.working_digits)
        )
        self.zero = 0j if self.double else rx.Complex(Decimal(0), Decimal(0))
        with self.scope():
            self.sqrt2 = math.sqrt(2.0) if self.double else Decimal(2).sqrt()
            self.pi = math.pi if self.double else _decimal_pi()

    def scope(self):
        return localcontext(prec=self.precision.working_digits)

    def real(self, value: Any) -> Any:
        if self.double:
            return float(value)
        if isinstance(value, Fraction):
            return Decimal(value.numerator) / Decimal(value.denominator)
        return +Decimal(value)

    def pair(self, real: Any, imag: Any = 0) -> Any:
        if self.double:
            return complex(float(real), float(imag))
        return rx.Complex(self.real(real), self.real(imag))

    def colour(self, value: ExactColorCoefficient, power: int) -> Any:
        """``value * 2^(-power/2)`` (``ct_sites.Carrier.number``)."""
        number = self.pair(self.real(value.real), self.real(value.imag))
        scale = (
            self.sqrt2 ** (-power)
            if self.double
            else _decimal_power(self.sqrt2, -power)
        )
        return number * scale

    def strings(self, value: Any) -> list[str]:
        if self.double:
            return [repr(float(value.real)), repr(float(value.imag))]
        return [str(value.real), str(value.imag)]


def _decimal_pi() -> Decimal:
    """pi at the active Decimal precision (Machin's formula)."""

    def arctan_inverse(x: int) -> Decimal:
        power = total = Decimal(1) / x
        x2, n, sign = x * x, 1, 1
        while True:
            power /= x2
            n += 2
            sign = -sign
            term = power / n
            if term == 0:
                return total
            total += sign * term

    with localcontext() as context:
        context.prec += 5
        value = 4 * (4 * arctan_inverse(5) - arctan_inverse(239))
    return +value


def _decimal_power(base: Decimal, exponent: int) -> Decimal:
    value = Decimal(1)
    for _ in range(abs(exponent)):
        value = value * base if exponent > 0 else value / base
    return value


# --------------------------------------------------------------------- tree rows


class TreeRows:
    """Currents of the tree runtime at one point (design §3.2 exports).

    The stage program of a tree DAG (the blocks DAG; the Born DAG for its
    amplitude roots) is replayed by Symbolica evaluators, in binary64 or at a
    given number of decimal digits. Rows are the tree runtime's values: the
    source wavefunction of a single leg, otherwise the current including its
    own propagator, keyed by (current id, variant); ``amplitudes=True`` adds
    the amplitude-root outputs, keyed by (root id, "amplitude-root").

    The stage replay of ``generation/numerical_current_warmup.py`` is not
    reused: it replays at decimal precision only and binds the model's
    binary64 defaults, while the slice also needs binary64 replays and derived
    parameters of overridden inputs at the carrier's precision.
    """

    def __init__(self, dag: Any, compiled: Any, *, blocks: bool = True):
        from pyamplicol.generation.contracts import StageCompilationInput
        from pyamplicol.generation.runtime_schema import (
            build_runtime_expression_schema,
        )
        from pyamplicol.generation.stage_planning import (
            build_generic_stage_compiler_blueprint,
        )
        from pyamplicol.models.external import CompiledUFOModel

        self.compiled = compiled
        model = CompiledUFOModel(compiled)
        with _exact_u1_subtraction_weights():
            # Every block enters a loop vertex with its own propagator, also
            # where no tree interaction consumes it (reachability-pruned DAGs).
            runtime_schema = build_runtime_expression_schema(
                dag,
                model,
                process_id="loop-v2-blocks",
                propagated_currents=(current.id for current in dag.currents if blocks),
            )
            blueprint = build_generic_stage_compiler_blueprint(
                StageCompilationInput(dag, model, runtime_schema)
            )
        if not blueprint.expression_ready:
            raise ValueError(
                f"tree blocks are not expression ready: {blueprint.blockers}"
            )
        self.schema = runtime_schema.to_mapping()
        self.stages = blueprint.stages
        self._amplitude_stage = blueprint.amplitude_stage
        self._roots = {
            int(root["output_index"]): int(root["root_id"])
            for root in self.schema["amplitude_stage"]["roots"]
        }
        # Built on the first replay: native per-point trees (``native_trees``)
        # read the stage expressions only.
        self._evaluators: tuple[Any, ...] | None = None
        self._amplitude_evaluator = None
        layout = self.schema["parameter_layout"]
        self._value_count = int(layout["value_component_count"])
        self._offset = self._value_count + int(layout["momentum_parameter_count"])
        self._size = int(layout["parameter_count_if_flattened"])
        self._records = sorted(
            self.schema["model_parameters"], key=lambda r: r["parameter_index"]
        )
        self._slots = defaultdict(list)
        for slot in self.schema["value_storage"]["value_slots"]:
            self._slots[int(slot["component_start"])].append(slot)
        self._definitions = {
            record.name: record.resolved_expression
            for record in compiled.ir.parameters
            if record.nature == "internal"
        }
        self._definitions.update(
            {
                f"derived_coupling_{term.id}": term.coupling_expression
                for term in compiled.ir.vertex_terms
            }
        )

    def _model_values(self, parameters: Mapping[str, str], digits: int) -> list:
        from symbolica import E

        from pyamplicol._internal.physics.symbols import symbols
        from pyamplicol.runtime.symbolica_exact import _upcast_complex_inputs

        external = {
            record["name"]: Decimal(
                parameters.get(record["name"], repr(record["default"]))
            )
            for record in self._records
            if record["kind"] == "external_parameter"
        }
        model_symbols = symbols.model(self.compiled.name)
        substitutions = dict(
            zip(
                (
                    E(model_symbols.symbol(name).to_canonical_string())
                    for name in external
                ),
                _upcast_complex_inputs(
                    [(value, Decimal(0)) for value in external.values()], digits
                ),
                strict=True,
            )
        )
        derived, values = {}, []
        for record in self._records:
            if record["kind"] == "external_parameter":
                values.append(external[record["name"]])
                continue
            name = record["runtime_name"]
            if name not in derived:
                value = E(self._definitions[name]).evaluate(
                    substitutions, decimal_digit_precision=digits
                )
                derived[name] = tuple(Decimal(x) for x in value.to_decimal_tuple())
            values.append(derived[name][record["complex_component"] == "imag"])
        return values

    def evaluate(
        self,
        point: Sequence[Sequence[Decimal]],
        parameters: Mapping[str, str],
        numbers: _Numbers,
        *,
        amplitudes: bool = False,
    ) -> dict[tuple[int, str], tuple[Any, ...]]:
        from pyamplicol.runtime.symbolica_exact import _fill_momenta, _fill_sources

        digits = numbers.precision.working_digits
        values = self._model_values(parameters, digits)
        zero = (Decimal(0), Decimal(0))
        state = [zero] * max(self._size, self._offset + len(values))
        with localcontext() as context:
            context.prec = digits + 10
            _fill_sources(state, point, self.schema, tuple(values))
            _fill_momenta(state, point, self.schema)
        for index, value in enumerate(values):
            state[self._offset + index] = (value, Decimal(0))
        rows: dict[tuple[int, str], tuple[Any, ...]] = {}
        convert = numbers.pair
        for start in range(self._value_count):
            for slot in self._slots.get(start, ()):
                if slot["variant"] == "source":
                    stop = int(slot["component_stop"])
                    rows[(int(slot["current_id"]), "source")] = tuple(
                        convert(*z) for z in state[start:stop]
                    )
        if self._evaluators is None:
            self._evaluators = tuple(_stage_evaluator(stage) for stage in self.stages)
        for stage, evaluator in zip(self.stages, self._evaluators, strict=True):
            outputs, converted = self._replay(stage, evaluator, state, numbers)
            for slot in stage.output_slots:
                state[slot.component_start : slot.component_stop] = outputs[
                    slot.output_start : slot.output_stop
                ]
                for record in self._slots.get(slot.component_start, ()):
                    if int(record["current_id"]) == slot.current_id:
                        rows[(slot.current_id, str(record["variant"]))] = tuple(
                            converted[slot.output_start : slot.output_stop]
                        )
        if amplitudes:
            stage = self._amplitude_stage
            if self._amplitude_evaluator is None:
                self._amplitude_evaluator = _stage_evaluator(stage)
            _outputs, converted = self._replay(
                stage, self._amplitude_evaluator, state, numbers
            )
            for slot in stage.output_slots:
                root = self._roots[slot.component_start]
                rows[(root, "amplitude-root")] = tuple(
                    converted[slot.output_start : slot.output_stop]
                )
        return rows

    @staticmethod
    def _replay(stage, evaluator, state, numbers: _Numbers):
        """One stage on the global state: (Decimal outputs, carrier values)."""

        from pyamplicol.runtime.symbolica_exact import _upcast_complex_inputs

        zero = (Decimal(0), Decimal(0))
        inputs = [zero] * stage.parameter_count
        for component in stage.input_components:
            inputs[component.parameter_index] = state[component.global_component]
        if numbers.double:
            raw = evaluator.evaluate_complex(
                [[complex(float(re), float(im)) for re, im in inputs]]
            )[0]
            return [(Decimal(z.real), Decimal(z.imag)) for z in raw], [
                complex(z) for z in raw
            ]
        digits = numbers.precision.working_digits
        raw = evaluator.evaluate_complex_with_prec(
            _upcast_complex_inputs(tuple(inputs), digits), digits
        )
        outputs = [tuple(Decimal(x) for x in value.to_decimal_tuple()) for value in raw]
        return outputs, [numbers.pair(*z) for z in outputs]


def _stage_evaluator(stage: Any) -> Any:
    from symbolica import Expression, FunctionDefinition

    return Expression.evaluator_multiple(
        list(stage.output_expressions),
        list(stage.parameter_symbols),
        functions=[
            FunctionDefinition(function, arguments, body, inlining="always")
            for function, arguments, body in stage.symbolica_functions
        ],
    )


def _exact_u1_subtraction_weights():
    """Build stage expressions with the exact U(1)-subtraction weight 1/N_c.

    The tree engine stores that colour weight as binary64 (``GenericDAG``
    interactions), which caps multi-line block rows at ~1e-17 in any carrier;
    graded U(1) currents replace it in WP2.4 (design §3.2 item 4). Until then
    the slice builds its stage expressions with the exact rational (binary64
    values are unchanged: 1/N_c rounds to the same double). Tree programs
    carry the U(1) sign in the weight (``generation/fermion_ordering.py``),
    so -1/N_c is mapped as well.
    """

    from symbolica import E

    from pyamplicol.generation.stage_expressions import exact_weights

    return exact_weights(
        {complex(1 / NC): E(f"1/{NC}"), complex(-1 / NC): E(f"-1/{NC}")}
    )


# --------------------------------------------------------------------- seeds

# Loop-aware reachability of the tree blocks (WP3.1 I1a); False builds every
# block current the tree model allows (the families and every result are
# unchanged, ``tests/unit/test_loop_v2_generate.py``).
LOOP_BLOCK_REACHABILITY = True


def _field_level_families(
    data, process, catalogue, selection, request, native: Any
) -> FamilySet:
    """The families of the field-level block table (design §3.2 item 1);
    ``native`` is the loop core, or None for the Python recursion."""

    names = {int(entry["pdg_code"]): entry["name"] for entry in data["particles"]}
    incoming = [names[leg.pdg] for leg in process.legs if leg.is_initial]
    outgoing = [names[leg.pdg] for leg in process.legs if leg.is_final]
    table = ModelBlockTable(
        data,
        incoming,
        outgoing,
        cap=selection.cap,
        ignored_particle_ids=request.ignored_particle_ids,
    )
    if native is not None:
        return NativeGeneration(table, catalogue, selection, native).families()
    return generate_families(table, catalogue, selection)


# Schedule-B frame policy of the slice: the cheapest frame by undressed rows
# (``NativeGeneration.schedules`` runs the same policy natively).
_FRAME_MODEL = STRUCTURED
_FRAME_POLICY = cheapest(lambda mask: 2 ** mask.bit_count(), _FRAME_MODEL)


class SectorSeeds:
    """Seeds with one column per squared-order sector (an ``rx.SeedSource``).

    Column s of U0[c](h) sums K[c, b] conj(A0_b(h)) over the (Born order,
    loop order) pairs of sector s, so each family numerator splits by
    squared coupling order (REQUIREMENTS_ADDENDUM A). A skeleton's seed is
    its combination of table classes (``colour.build_seed_table``). With
    ``fourier`` (binary64 only) the certified orbit pairs are contracted by
    the native symmetric-group FFT (``colour.SeedFourier``, design §7.3);
    ``algorithm`` reports the contraction used.
    """

    def __init__(
        self,
        table: SeedTable,
        class_of: Mapping[Any, SeedCombination],
        born: Mapping[rx.Row, Mapping[tuple[int, Orders], Mapping[int, Any]]],
        sectors: Sequence[frozenset[tuple[Orders, Orders]]],
        numbers: _Numbers,
        accuracy: ColourAccuracy,
        degrees: Sequence[int | None] | None = None,
        *,
        fourier: SeedFourier | None = None,
        native: Any = None,
    ):
        self.columns = len(sectors)
        self.class_of = class_of
        self._numbers = numbers
        self._admitted = {
            (c, order)
            for pairs in sectors
            for c, order, _b, _a in table.admitted(pairs)
        }
        self._rows: dict[rx.Row, list[dict]] = {}

        def cast(cell: ExactColorCoefficient) -> Any:
            return numbers.pair(numbers.real(cell.real), numbers.real(cell.imag))

        self._cast = cast
        degrees = degrees or [None] * len(sectors)
        if fourier is not None and numbers.double:
            self.algorithm = SYMMETRIC_GROUP_FFT
            rows = list(born)
            columns = [
                fourier.seeds(
                    table,
                    pairs,
                    [born[row] for row in rows],
                    accuracy,
                    native,
                    degree=degree,
                )
                for pairs, degree in zip(sectors, degrees, strict=True)
            ]
            for r, row in enumerate(rows):
                self._rows[row] = [column[r] for column in columns]
            return
        self.algorithm = "direct"
        for row, amplitudes in born.items():
            self._rows[row] = [
                table.seeds(
                    pairs,
                    amplitudes,
                    accuracy,
                    cast=cast,
                    conjugate=lambda value: value.conjugate(),
                    degree=degree,
                )
                for pairs, degree in zip(sectors, degrees, strict=True)
            ]

    def seed(self, skeleton, contents):
        reading = rx._reading(skeleton, contents)
        combination = self.class_of.get(reading)
        if combination is None:
            raise KeyError("reading without a seed class")
        order = skeleton.orders
        for c, _x in combination:
            if (c, order) not in self._admitted:
                raise KeyError(f"no sector admits class {c} at {order}")
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


# --------------------------------------------------------------------- whole rules


@dataclass(frozen=True)
class WholeRuleTerm:
    """One term of a whole-rule colour continuation, closed on one block choice.

    Whole-rule continuations (``loops.qcd_r2_color``, design §6.1) replace a
    fixed-SU(3) counterterm group by checked general-N_c terms: a colour
    tensor, a Lorentz structure that is a sum of metric products and an exact
    coefficient. ``blocks[i]`` attaches to UFO leg i + 1; ``pairings`` are
    (rational, ((a, b), ...)) with the product of Minkowski contractions of
    the block rows of legs a and b; ``grades`` are the coefficient's exact
    N_c grades (degree, expression of model ``parameters``). Only the eps^0
    coefficient exists.
    """

    blocks: tuple[Any, ...]
    sign: int
    colour: ConnectedColorTensor
    pairings: tuple[tuple[Fraction, tuple[tuple[int, int], ...]], ...]
    grades: tuple[tuple[int, Any], ...]
    parameters: tuple[str, ...]

    def kinematic(self, rows: Sequence[Sequence[Any]], numbers: _Numbers) -> Any:
        total = numbers.zero
        for coefficient, pairs in self.pairings:
            value = numbers.pair(numbers.real(coefficient))
            for a, b in pairs:
                x, y = rows[a], rows[b]
                value = value * (x[0] * y[0] - x[1] * y[1] - x[2] * y[2] - x[3] * y[3])
            total = total + value
        return self.sign * total

    def coefficients(self, values: Mapping[str, Any], numbers: _Numbers) -> dict:
        """The grade coefficients at the bound model ``values`` (name -> number)."""

        from symbolica import Replacement

        from pyamplicol._internal.physics.symbols import symbols

        digits = numbers.precision.working_digits
        out = {}
        for degree, expression in self.grades:
            bound = expression.replace_multiple(
                [
                    Replacement(
                        symbols.qualified(f"UFO::{name}"),
                        _number_expression(values[name]),
                    )
                    for name in self.parameters
                ]
            )
            value = bound.evaluate({}, digits + 10)
            out[degree] = numbers.pair(
                Decimal(value.real.to_decimal()), Decimal(value.imag.to_decimal())
            )
        return out


def _number_expression(value: Any) -> Any:
    from symbolica import E, Expression

    return E(str(Decimal(value.real))) + E(str(Decimal(value.imag))) * Expression.I


def _whole_rule_terms(closure: SiteClosure, site) -> tuple[WholeRuleTerm, ...]:
    """The replacement terms of a whole-rule site, closed on every block choice."""

    from symbolica.community import hepkit

    from pyamplicol.loops.physical_color import split_color_grades

    from .colour import relabel, spenso_colour_network

    continuation = site.rule.whole_rule_continuation
    out = []
    for term in continuation.terms:
        kernel = term.kernel
        legs = {
            hepkit.Symbols.half_edge()(half, 1).to_canonical_string(): index + 1
            for index, half in enumerate(kernel.leg_half_edges)
        }

        def rename(label, legs=legs):
            return ("leg", legs[label]) if label in legs else ("dummy", label)

        network = relabel(spenso_colour_network(kernel.color.to_expression()), rename)
        pairings = _metric_pairings(kernel.lorentz.to_expression(), legs)
        grades = tuple(
            (degree, expression)
            for degree, (expression,) in split_color_grades(
                (term.coefficient,), continuation.nc
            )
        )
        parameters = tuple(
            sorted(
                {
                    symbol.get_name().rsplit("::", 1)[-1]
                    for _degree, expression in grades
                    for symbol in expression.get_all_symbols(False)
                    if symbol.get_name().startswith("UFO::")
                }
            )
        )
        for choice, sign, colour in closure.close(network, site.blocks):
            out.append(
                WholeRuleTerm(choice, sign, colour, pairings, grades, parameters)
            )
    return tuple(out)


def _metric_pairings(expression: Any, legs: Mapping[str, int]):
    """A sum of products of Minkowski metrics as (rational, leg-index pairs)."""

    from symbolica import AtomType

    from .colour import _exact_number, _head

    expression = expression.expand()
    terms = list(expression) if expression.get_type() == AtomType.Add else [expression]
    out = []
    for term in terms:
        factors = list(term) if term.get_type() == AtomType.Mul else [term]
        coefficient, pairs = Fraction(1), []
        for factor in factors:
            if factor.get_type() == AtomType.Num:
                number = _exact_number(factor)
                if number.imag:
                    raise NotImplementedError("complex metric-product coefficient")
                coefficient *= number.real
                continue
            if _head(factor) != "g":
                raise NotImplementedError(f"Lorentz factor {factor} is not a metric")
            ends = []
            for slot in factor:
                if _head(slot) != "mink":
                    raise NotImplementedError(f"metric slot {slot} is not Minkowski")
                _dimension, port = list(slot)
                ends.append(legs[port.to_canonical_string()] - 1)
            pairs.append(tuple(ends))
        out.append((coefficient, tuple(pairs)))
    return tuple(out)


# --------------------------------------------------------------------- results


@dataclass(frozen=True, slots=True)
class Laurent:
    """Laurent coefficients; poles are ``None`` in finite-only evaluations."""

    finite: Any
    single_pole: Any = None
    double_pole: Any = None

    @property
    def coefficients(self) -> tuple[Any, ...]:
        if self.single_pole is None:
            return (self.finite,)
        return (self.finite, self.single_pole, self.double_pole)

    def __add__(self, other: Laurent) -> Laurent:
        values = [
            a + b for a, b in zip(self.coefficients, other.coefficients, strict=True)
        ]
        return Laurent(*values)


@dataclass(frozen=True)
class PointResult:
    """One point: Born and the six components per squared-order sector.

    ``born[order]`` is the Born of a squared Born order, ``components[name]
    [sector]`` a ``Laurent`` per virtual sector, ``totals`` the 19 sums over
    sectors (Born, then each component's Laurent coefficients; 7 entries in
    finite-only mode). ``families`` keeps each reduction lane's raw integrals
    ``(family, loop order, sector) -> Laurent``. ``ir_poles`` is Catani's
    prediction of the virtual's (single, double) poles per sector, ``None``
    in finite-only mode, at leading colour and for non-QCD corrections;
    ``stability`` is ``None`` unless checks were asked.
    ``colour_contraction`` is the seed contraction used ("direct" or
    "symmetric-group-fft").
    """

    born: Mapping[Orders, Any]
    components: Mapping[str, Mapping[Orders, Laurent]]
    totals: tuple[Any, ...]
    families: Mapping[tuple[FamilyKey, Orders, Orders], Laurent]
    ir_poles: Mapping[Orders, tuple[Any, Any]] | None
    stability: StabilityEstimate | None = None
    colour_contraction: str = "direct"

    def virtual(self, sector: Orders) -> Laurent:
        return self.components["virtual"][sector]


# --------------------------------------------------------------------- slice


@dataclass(frozen=True)
class SliceRequest:
    """A loop request of the slice: a process, the bounds of its Born coupling
    orders and the exact coupling orders of its loop amplitude."""

    process: str
    born_orders: Mapping[str, int]
    loop_orders: Mapping[str, int]
    corrections: tuple[str, ...] = ("QCD",)
    fixed_parameters: Mapping[str, str] = field(default_factory=dict)
    ignored_particle_ids: tuple[int, ...] = ()
    accuracy: ColourAccuracy = "full"
    # A source-owned formal-N_c continuation (``LoopColorDeclaration``); LC
    # needs it for the counterterm and field colour grades (design §6.7).
    colour_declaration: Any = None
    # External model parameters of the renormalisation scale and of alpha_s
    # (IR-pole check). By default the UFO/SLHA codes identify them: (LOOP, 1)
    # and (SMINPUTS, 3), as in MG5_aMC loop models.
    scale_parameter: str | None = None
    alpha_s_parameter: str | None = None
    # Certified colour compressions (design §7.2): adjoint rings and closed
    # fundamental loops seeded by primitives; False keeps every generic seed.
    compress_colour: bool = True
    # Seed contraction (design §7.3): "auto" uses the native symmetric-group
    # FFT on certified orbits in binary64 at NLC and full colour, "fft" also
    # refuses a request without such orbits, "direct" never uses it. Other
    # carriers and LC always contract directly.
    colour_contraction: ContractionPolicy = "auto"


@dataclass
class LoopSlice:
    """Generation objects of one process and their per-point evaluation."""

    request: SliceRequest
    compiled: Any
    prepared: Any
    data: Mapping[str, Any]
    blocks: TreeBlocks
    families: FamilySet
    plan: LoopSchedules
    colour: SeedColour
    catalogue: Any
    closure: SiteClosure
    sites: CountertermSites
    fields: tuple[FieldInsertion, ...]
    field_names: Mapping[tuple[int, str, int], str]
    born_terms: Mapping[Orders, tuple[SiteTerm, ...]]
    continuation: Any
    basis: BornBasis
    table: SeedTable
    class_of: Mapping[Any, SeedCombination]
    sectors: tuple[tuple[Orders, frozenset[tuple[Orders, Orders]]], ...]
    normalisation: Normalisation
    layout: MonomialLayout
    native: Any
    reading: TreeReading
    weights: LeafWeights
    scale_parameter: str
    alpha_s_parameter: str | None
    loop_catalogue: Any = None
    fourier: SeedFourier | None = None
    # The native generator stages of this process (WP3.1), if they ran.
    generation: NativeGeneration | None = None
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def site_terms(self) -> Mapping[Any, tuple[SiteTerm | WholeRuleTerm, ...]]:
        """The terms of every counterterm site (the slice's counterterm form,
        design §6.6), built on first use: generation and the plan need none."""

        found = self._cache.get("site-terms")
        if found is None:
            found = self._cache["site-terms"] = {
                site: (
                    self.closure.terms(site)
                    if site.rule.whole_rule_continuation is None
                    else _whole_rule_terms(self.closure, site)
                )
                for site in self.sites.sites
            }
        return found

    @property
    def tree_rows(self) -> TreeRows:
        """The blocks DAG's tree runtime, built on first use."""

        found = self._cache.get("tree-rows")
        if found is None:
            found = self._cache["tree-rows"] = TreeRows(
                self.blocks.blocks_dag, self.compiled
            )
        return found

    # -- generation ------------------------------------------------------------

    @classmethod
    def generate(cls, compiled: Any, request: SliceRequest, native: Any) -> LoopSlice:
        """Build every generation object of ``request`` (no diagram is listed).

        ``compiled`` is the compiled loop model and ``native`` the staged loop
        core (``loops.native.load_loop_native``).
        """

        from pyamplicol.loops.compiler import prepare_loop_request_model
        from pyamplicol.loops.wavefunction import prepare_external_field_rules
        from pyamplicol.models.external import CompiledUFOModel
        from pyamplicol.processes.model import build_model_process_ir

        if request.accuracy not in ("full", "lc", "nlc"):
            raise ValueError(f"unknown colour accuracy {request.accuracy!r}")
        declaration = request.colour_declaration
        fixed: dict[str, Any] = dict(request.fixed_parameters)
        if declaration is not None:
            from pyamplicol.loops.source_color_declaration import (
                source_color_fixed_parameters,
            )

            fixed = source_color_fixed_parameters(
                compiled.raw_loop_model, declaration, fixed_parameters=fixed
            )
        elif request.accuracy != "full":
            raise ValueError(
                f"{request.accuracy.upper()} needs a source-owned formal-N_c "
                "continuation of the counterterms (design §6.7)"
            )
        scale_parameter = _external_parameter(
            compiled, request.scale_parameter, _LHA_SCALE, fixed
        )
        alpha_s_parameter = (
            _external_parameter(
                compiled, request.alpha_s_parameter, _LHA_ALPHA_S, fixed
            )
            if set(request.corrections) == {"QCD"}
            else None
        )
        prepared = prepare_loop_request_model(
            compiled,
            born_order_bounds=dict(request.born_orders),
            loop_order_bounds=dict(request.loop_orders),
            fixed_parameters=fixed,
        )
        continuation = None
        if declaration is not None:
            from pyamplicol.loops.source_color_declaration import (
                prepare_declared_source_color,
            )

            prepared, continuation = prepare_declared_source_color(
                prepared, declaration, scheme=declaration.scheme
            )
        structure = prepared.structure
        data = json.loads(structure.native.to_json())
        loop = add_orders(tuple(request.loop_orders.items()))
        selection = LoopOrderSelection(
            frozenset(request.corrections), frozenset({loop})
        )
        process = build_model_process_ir(
            request.process, compiled.ir, color_accuracy="full"
        )
        loop_catalogue = ModelLoopCatalogue(
            data, species_orders=structure.perturbative_species_orders
        )
        native_families = "families" in generator_input.NATIVE_STAGES

        def tree_program(reachable):
            blocks = build_tree_blocks(
                process,
                CompiledUFOModel(compiled),
                born_coupling_orders=dict(request.born_orders),
                block_coupling_orders=dict(request.loop_orders),
                ignored_particle_ids=request.ignored_particle_ids,
                reachability=(
                    None
                    if reachable is None
                    else requested_loop_blocks(reachable.block_requests(), data)
                ),
            )
            table = blocks.block_table(data)
            if native_families:
                generation = NativeGeneration(table, loop_catalogue, selection, native)
                return blocks, table, generation, generation.families()
            return (
                blocks,
                table,
                None,
                generate_families(table, loop_catalogue, selection),
            )

        reachable = None
        if LOOP_BLOCK_REACHABILITY:
            # Loop-aware reachability (design §3.2 item 1): the tree program
            # builds only the block currents the admitted positions of the
            # field-level families consume.
            reachable = _field_level_families(
                data,
                process,
                loop_catalogue,
                selection,
                request,
                native if native_families else None,
            )
        blocks, table, generation, families = tree_program(reachable)
        if reachable is not None and (
            families.families != reachable.families
            or families.scaleless != reachable.scaleless
        ):
            # The field-level table does not predict this process's tree
            # blocks (e.g. electroweak currents the tree compiler does not
            # build split closed-loop flavour classes, ``families.
            # _closed_loop_classes``): every block current is built.
            blocks, table, generation, families = tree_program(None)
        colour = SeedColour(blocks, data, compiled.ir.vertex_terms)
        if generation is not None and "schedules" in generator_input.NATIVE_STAGES:
            plan = generation.schedules(colour, model=_FRAME_MODEL)
        else:
            plan = build_schedules(
                families,
                loop_catalogue,
                colour,
                policy=_FRAME_POLICY,
            )
        sites = generate_counterterm_sites(
            families,
            table,
            prepared.counterterms.rules,
            data,
            selection,
            forbidden=request.ignored_particle_ids,
            whole_rules=continuation is not None,
        )
        insertions = external_field_insertions(
            table.legs,
            prepare_external_field_rules(prepared.counterterms.source),
            data,
            selection,
            blocks.born_orders,
            forbidden=request.ignored_particle_ids,
        )
        ordinary = _ordinary_entries(prepared, data)
        entries = [e for site in sites.sites for e in site.rule.entries]
        entries += [e for rows in ordinary for e in rows]
        catalogue = build_loop_kernel_catalogue(
            compiled, counterterm_entries=list({id(e): e for e in entries}.values())
        )
        closure = SiteClosure(blocks, data, catalogue)
        readings = rx.reading_skeletons(plan)
        weights = LeafWeights(data, closure.fermion_ends)
        for skeleton in readings:
            weights.fermion_sign(skeleton)  # refuses fermion lines it does not cover
        born_terms = _born_terms(blocks, table, closure, ordinary)
        basis = _born_tensors(closure, born_terms)
        seed_table, class_of = build_seed_table(
            colour, readings, basis, compress=request.compress_colour
        )
        born_orders = blocks.born_orders
        sectors: dict[Orders, set[tuple[Orders, Orders]]] = defaultdict(set)
        for a in born_orders:
            for b in selection.loop_orders:
                sectors[add_orders(a, b)].add((a, b))
        incoming = [leg.pdg for leg in process.legs if leg.is_initial]
        outgoing = [leg.pdg for leg in process.legs if leg.is_final]
        from .fields import field_definitions

        names = {
            identity: name
            for identity, (name, _source) in field_definitions(insertions).items()
        }
        slice_ = cls(
            request=request,
            compiled=compiled,
            prepared=prepared,
            data=data,
            blocks=blocks,
            families=families,
            plan=plan,
            colour=colour,
            catalogue=catalogue,
            closure=closure,
            sites=sites,
            fields=insertions,
            field_names=names,
            born_terms=born_terms,
            continuation=continuation,
            basis=basis,
            table=seed_table,
            class_of=class_of,
            sectors=tuple(
                (sector, frozenset(pairs)) for sector, pairs in sorted(sectors.items())
            ),
            normalisation=Normalisation.of(structure.native, incoming, outgoing),
            layout=MonomialLayout.native(
                native,
                # The executor bounds a node's rank by its inputs' rank plus
                # the largest degree of its edges.
                max(
                    node.rank_in + max(segment.q_degree for segment, _ in node.edges)
                    for schedule in plan.schedules.values()
                    for node in schedule.trie.nodes
                ),
            ),
            native=native,
            reading=TreeReading(data),
            weights=weights,
            scale_parameter=scale_parameter,
            alpha_s_parameter=alpha_s_parameter,
            loop_catalogue=loop_catalogue,
            generation=generation,
        )
        if request.accuracy != "full":
            slice_._prune_colour()
        slice_._plan_contraction()
        return slice_

    def _plan_contraction(self) -> None:
        """Certify the seed FFT's orbits per ``colour_contraction``."""

        policy = self.request.colour_contraction
        if policy not in ("auto", "direct", "fft"):
            raise ValueError(f"unknown colour contraction {policy!r}")
        if policy == "direct":
            return
        if self.request.accuracy == "lc":
            if policy == "fft":
                raise ValueError("leading-colour seeds are contracted directly")
            return
        self.fourier = seed_fourier(self.table)
        if self.fourier is None and policy == "fft":
            raise ValueError(
                "the seed table has no certified symmetric-group orbit pair"
            )

    def _prune_colour(self) -> None:
        """LC and NLC: drop, before any numerics, every skeleton whose seed
        cells all lie below the N_c powers the projection keeps (design §7.5,
        §7.6). Structural: exact seed cells, Born grades and the sector
        degrees P; the projection itself is unchanged.
        """

        degrees = self.degrees()["virtual"]
        accuracy = self.request.accuracy
        grades = self._born_grades()
        pairs = [
            (degrees[sector], a, loop)
            for sector, sector_pairs in self.sectors
            for a, loop in sector_pairs
        ]
        live: dict[tuple[int, Orders], bool] = {}

        def class_live(c: int, order: Orders) -> bool:
            found = live.get((c, order))
            if found is None:
                found = live[c, order] = any(
                    value and kept(power + grade, degree, accuracy)
                    for degree, a, loop in pairs
                    if loop == order
                    for b, born_orders in enumerate(self.basis.orders)
                    if a in born_orders
                    for power, value in self.table.cells[c][b].items()
                    for grade in grades.get((b, a), ())
                )
            return found

        def keep(skeleton) -> bool:
            return any(
                class_live(c, skeleton.orders) for c, _x in self.class_of[skeleton]
            )

        # Each surviving family keeps its frame, so the pruned schedules dress
        # the remaining skeletons exactly as before (dropped ones contributed
        # exact zeros). A reading that is new nonetheless gets its own seed:
        # it differs from its skeleton's by the orientation signs of the
        # vertex colours. The degrees P stay those of every class.
        frames = {
            key: schedule.trie.frame for key, schedule in self.plan.schedules.items()
        }

        def same_frame(family, candidates):
            return next(
                trie for frame, trie in candidates if frame == frames[family.key]
            )

        if self.generation is not None and "schedules" in generator_input.NATIVE_STAGES:
            self.plan = self.generation.schedules(self.colour, frames=frames, keep=keep)
        else:
            self.plan = build_schedules(
                self.families,
                self.loop_catalogue,
                self.colour,
                policy=same_frame,
                view=SkeletonView(self.loop_catalogue, self.colour, keep=keep),
            )
        self.table, self.class_of = extend_seed_table(
            self.table,
            self.class_of,
            self.colour,
            rx.reading_skeletons(self.plan),
            compress=self.request.compress_colour,
        )

    # -- evaluation ------------------------------------------------------------

    def evaluate(
        self,
        point: Sequence[Sequence[Any]],
        scale_squared: Any,
        *,
        parameters: Mapping[str, str] | None = None,
        precision: Precision | None = None,
        poles: bool = True,
        stability: StabilityChecks | None = None,
    ) -> PointResult:
        """The complete virtual at one exact point (decimal strings or rationals).

        ``point`` lists the physical momenta (E, px, py, pz) of the legs in
        process order, incoming momenta as given; ``scale_squared`` is mu_R^2.
        ``parameters`` overrides external model parameters (decimal strings).
        """

        precision = precision or Precision.double()
        precision.require_reference()
        stability = stability or StabilityChecks()
        exact_point = tuple(tuple(_fraction(x) for x in momentum) for momentum in point)
        scale = _fraction(scale_squared)
        reference = self._evaluate(
            exact_point,
            scale,
            dict(parameters or {}),
            precision,
            poles,
            stability.frame,
        )
        if not stability.enabled:
            return reference.result
        budgets = {}
        for index, rotation in enumerate(stability.rotations):
            rotated = self._evaluate(
                rotation.apply(exact_point),
                scale,
                dict(parameters or {}),
                precision,
                poles,
                None,
            )
            budgets[ROTATION.format(index)] = reference.rotation_budget(rotated)
        if stability.reduction:
            budgets[stability.reduction] = reference.retest_budget
        estimate = StabilityEstimate.from_checks(
            reference.result.components["virtual"],
            budgets,
            scale=reference.absolute,
            roundoff=precision.unit_roundoff,
            families=reference.family_deviations,
        )
        return replace(reference.result, stability=estimate)

    def _evaluate(self, point, scale, parameters, precision, poles, retest):
        with _Numbers(precision).scope():
            return self._point(point, scale, parameters, precision, poles).finish(
                retest
            )

    def _point(self, point, scale, parameters, precision, poles) -> _PointEvaluation:
        values = self._point_parameters(parameters, scale, precision)
        return _PointEvaluation(self, point, scale, values, _Numbers(precision), poles)

    def _point_parameters(
        self, parameters: Mapping[str, str], scale: Fraction, precision: Precision
    ) -> dict[str, str]:
        """A point's parameter values: the caller's, the request's fixed
        parameters and the renormalisation scale (the root of ``scale``)."""

        values = dict(parameters)
        values.update(self.request.fixed_parameters)
        digits = precision.working_digits
        values[self.scale_parameter] = _decimal_string(
            _sqrt_fraction(scale, digits), digits
        )
        return values

    def family_numerators(
        self,
        point: Sequence[Sequence[Any]],
        scale_squared: Any,
        *,
        parameters: Mapping[str, str] | None = None,
        precision: Precision | None = None,
        schedule: Literal["B", "F"] = "B",
        exact: bool = False,
    ) -> tuple[rx.Execution, tuple[tuple[Fraction, ...], ...]]:
        """The Born-weighted family numerators at one point (oracles O2, O4).

        Returns the executor's run (one column per squared-order sector, in
        the families' B frames) and the all-incoming exact momenta of its
        frame shifts.
        """

        precision = precision or Precision.double()
        precision.require_reference()
        exact_point = tuple(tuple(_fraction(x) for x in momentum) for momentum in point)
        with _Numbers(precision).scope():
            evaluation = self._point(
                exact_point,
                _fraction(scale_squared),
                dict(parameters or {}),
                precision,
                True,
            )
            execution = evaluation.dress(schedule, exact=exact)
        return execution, tuple(tuple(p) for p in evaluation.incoming)

    # -- scalar bindings -------------------------------------------------------

    def _scalar_plan(self, powers: tuple[int, ...]):
        """The scalar plan of ``scalars`` and its (name, power) -> ((grade,
        column), ...) layout; only structurally nonzero grades are kept."""

        from pyamplicol.loops.scalar_bindings import prepare_scalar_bindings

        from .fields import field_definitions

        cached = self._cache.get(("plan", powers))
        if cached is None:
            structure = self.prepared.structure
            definitions = field_definitions(self.fields)
            outputs = {
                (entry.coupling.name, power)
                for site in self.sites.sites
                if site.rule.whole_rule_continuation is None
                for entry in site.rule.entries
                for power in powers
            }
            outputs.update(
                (name, power) for name, _ in definitions.values() for power in powers
            )
            outputs.update(
                (term.entry.coupling.name, 0)
                for terms in self.born_terms.values()
                for term in terms
            )
            # Only whole-rule sites (LC/NLC continuations) have whole-rule
            # terms; the other site terms are not needed here.
            outputs.update(
                (name, 0)
                for site in self.sites.sites
                if site.rule.whole_rule_continuation is not None
                for term in self.site_terms[site]
                if isinstance(term, WholeRuleTerm)
                for name in term.parameters
            )
            options = {
                "program": structure.scalars,
                "fixed_parameters": structure.fixed_parameters,
                "anonymous_definitions": {
                    name: source for name, source in definitions.values()
                },
            }
            if self.continuation is None:
                plan = prepare_scalar_bindings(
                    structure.raw, sorted(outputs), **options
                )
                layout = {output: ((0, i),) for i, output in enumerate(plan.outputs)}
            else:
                graded = self.continuation.scalars(sorted(outputs), **options)
                plan = graded.coefficients
                layout = defaultdict(list)
                for degree, columns in zip(graded.grades, graded.columns, strict=True):
                    for output, column in zip(
                        graded.formal.outputs, columns, strict=True
                    ):
                        if graded.coefficients.expressions[column].value != 0:
                            layout[output].append((degree, column))
                layout = {output: tuple(rows) for output, rows in layout.items()}
            cached = self._cache[("plan", powers)] = (plan, layout)
        return cached

    def scalars(
        self, parameters: Mapping[str, str], numbers: _Numbers, poles: bool
    ) -> dict[tuple[str, int], dict[int, Any]]:
        """Born, counterterm and field couplings per (name, eps power) and N_c
        grade (``{degree: value}``; only degree 0 without a continuation).

        With a continuation the source scalars are split into exact N_c
        grades (``SourceColorContinuation.scalars``); the whole-rule
        coefficients' model parameters are bound as well.
        """

        from pyamplicol.loops.scalar_bindings import NativeScalarBindings

        plan, layout = self._scalar_plan(LAURENT_POWERS if poles else (0,))
        arithmetic = "double" if numbers.double else "arbitrary"
        state = NativeScalarBindings(
            plan,
            arithmetic=arithmetic,
            precision=None if numbers.double else numbers.precision.working_digits,
            backend=self.native,
            parameters={
                name: value
                for name, value in parameters.items()
                if name in plan.external_names
            },
        )
        values = [
            numbers.pair(Decimal(re), Decimal(im))
            for re, im in state.values.to_decimal()
        ]
        return {
            output: {degree: values[column] for degree, column in rows}
            for output, rows in layout.items()
        }

    # -- colour degrees -------------------------------------------------------

    def overlaps(self, left: Any, right: Any) -> Mapping[int, ExactColorCoefficient]:
        """<left|right> of two literal colour tensors, exact in N_c."""

        cache = self._cache.setdefault("overlaps", {})
        found = cache.get((left, right))
        if found is None:
            legs = self.basis.legs
            cells = contract_connected_tensor_nc_terms(
                _single(left, legs), _single(right, legs)
            )
            found = cache[(left, right)] = {
                power: value for power, value in cells.items() if value
            }
        return found

    def _born_grades(self) -> dict[tuple[int, Orders], set[int]]:
        """N_c grades of the Born amplitude per (basis element, Born order):
        each literal tensor's N_c power plus its coupling grades."""

        found = self._cache.get("born-grades")
        if found is not None:
            return found
        _plan, layout = self._scalar_plan(LAURENT_POWERS)
        index = {tensor: b for b, tensor in enumerate(self.basis.tensors)}
        out: dict[tuple[int, Orders], set[int]] = defaultdict(set)
        for order, terms in self.born_terms.items():
            for term in terms:
                couplings = {
                    g for g, _c in layout.get((term.entry.coupling.name, 0), ())
                }
                for item in term.colour.terms:
                    b = index[_single(item.tensor, self.basis.legs)]
                    out[b, order].update(item.nc_power + g for g in couplings)
        self._cache["born-grades"] = out
        return out

    def degrees(self) -> dict[str, dict[Orders, int | None]]:
        """Leading N_c degrees P per squared-order sector (design §7.4).

        ``"born"`` per squared Born order; ``"virtual"`` per virtual sector, the
        largest degree of a nonzero exact cell among the seed, counterterm and
        field cells (coupling grades included). Structural: no numerics.
        """

        found = self._cache.get("degrees")
        if found is not None:
            return found
        _plan, layout = self._scalar_plan(LAURENT_POWERS)

        def grades(name: str, powers=LAURENT_POWERS) -> set[int]:
            return {g for power in powers for g, _c in layout.get((name, power), ())}

        def keys(terms, grade_of) -> set:
            return {
                (item.tensor, item.nc_power + grade)
                for term in terms
                for item in term.colour.terms
                for grade in grade_of(term)
            }

        def degree(bra: set, ket: set, offset: int = 0) -> int | None:
            powers = [
                power + p + q + offset
                for left, p in bra
                for right, q in ket
                for power in self.overlaps(left, right)
            ]
            return max(powers, default=None)

        born = {
            order: keys(terms, lambda t: grades(t.entry.coupling.name, (0,)))
            for order, terms in self.born_terms.items()
        }
        counterterms: dict[Orders, set] = defaultdict(set)
        for site, terms in self.site_terms.items():
            counterterms[site.orders] |= keys(
                terms,
                lambda t: (
                    {d for d, _e in t.grades}
                    if isinstance(t, WholeRuleTerm)
                    else grades(t.entry.coupling.name)
                ),
            )
        out = {"born": {}, "virtual": {}}
        for a, b in product(born, repeat=2):
            squared = add_orders(a, b)
            value = degree(born[a], born[b])
            out["born"][squared] = _max(out["born"].get(squared), value)
        born_grades = self._born_grades()
        for sector, pairs in self.sectors:
            value = self.table.degree(pairs, born_grades)
            for a, b in pairs:
                value = _max(value, degree(born[a], counterterms.get(b, set())))
            for insertion in self.fields:
                if (
                    a := next((a for a, b in pairs if b == insertion.orders), None)
                ) is None or insertion.born_order not in born:
                    continue
                for grade in grades(self.field_names[insertion.identity]):
                    value = _max(
                        value, degree(born[a], born[insertion.born_order], grade)
                    )
            out["virtual"][sector] = value
        self._cache["degrees"] = out
        return out

    # -- colour correlations (IR-pole check) ------------------------------------

    def dipoles(self) -> dict[tuple[int, int], tuple[tuple[Fraction, ...], ...]]:
        """<t|T_i.T_j|t'> at N_c = 3 over the Born tensors, for coloured i != j.

        The external adjoint labels are summed, so a cell is unchanged when one
        permutation of them relabels both tensors and the legs i, j it moves:
        row t is relabelled by the permutation that numbers its adjoint labels
        in reading order (``colour._adjoint_rename``) and each distinct
        relabelled cell is contracted once (as ``BornBasis.gram``).
        """

        found = self._cache.get("dipoles")
        if found is not None:
            return found
        legs = self.basis.legs
        coloured = [leg.label for leg in legs if leg.representation != 1]
        adjoint = sorted(leg.label for leg in legs if leg.representation == 8)
        items = [
            (tensor, term) for tensor in self.basis.tensors for term in tensor.terms
        ]
        images: dict[tuple[int, Hashable], Any] = {}

        def image(label: int, key: Hashable, term: Any, rename) -> Any:
            found = images.get((label, key))
            if found is None:
                connection = ColorConnection(legs, (EmitGluon(label, -1),))
                found = images[label, key] = apply_color_connection(
                    _relabelled_colour(term.tensor, rename),
                    connection,
                    coefficient=term.coefficient,
                )
            return found

        cells: dict[tuple, Fraction] = {}
        out: dict[tuple[int, int], list[list[Fraction]]] = {
            (i, j): [] for i in coloured for j in coloured if i != j
        }
        for tensor, term in items:
            rename = _adjoint_rename(tensor, adjoint)
            keys = [_term_key(other, rename) for _tensor, other in items]
            left = _term_key(term, rename)
            for i in coloured:
                for j in coloured:
                    if i == j:
                        continue
                    a, b = rename.get(i, i), rename.get(j, j)
                    row = []
                    for (_other_tensor, other), key in zip(items, keys, strict=True):
                        pair = (a, b, left, key)
                        cell = cells.get(pair)
                        if cell is None:
                            cell = cells[pair] = _real_colour(
                                evaluate_color_nc_terms(
                                    contract_connected_tensor_nc_terms(
                                        image(a, left, term, rename),
                                        image(b, key, other, rename),
                                    )
                                )
                            )
                        row.append(cell)
                    out[(i, j)].append(row)
        out = {pair: tuple(map(tuple, rows)) for pair, rows in out.items()}
        gram = tuple(
            tuple(_real_colour(evaluate_color_nc_terms(cell)) for cell in row)
            for row in self.basis.gram
        )
        # Colour conservation: sum_{j != i} T_i.T_j = -C_i on every Born vector.
        for i in coloured:
            casimir = _casimir(
                next(leg for leg in legs if leg.label == i).representation
            )
            for row, cells in enumerate(gram):
                for column, cell in enumerate(cells):
                    total = sum(out[(i, j)][row][column] for j in coloured if j != i)
                    if total != -casimir * cell:
                        raise ValueError(
                            f"colour correlators of leg {i} violate colour conservation"
                        )
        self._cache["dipoles"] = out
        return out


# --------------------------------------------------------------------- one point


class _PointEvaluation:
    """All per-point work of ``LoopSlice.evaluate`` in one carrier."""

    def __init__(self, slice_, point, scale, parameters, numbers, poles):
        self.slice = slice_
        self.numbers = numbers
        self.poles = poles
        process = slice_.blocks.process
        labels = sorted(leg.label for leg in process.legs)
        initial = {leg.label for leg in process.legs if leg.is_initial}
        # All-incoming momenta (families, executor) and decimal rows (trees).
        self.incoming = [
            [x if label in initial else -x for x in point[label - 1]]
            for label in labels
        ]
        digits = numbers.precision.working_digits
        with localcontext(prec=digits + 10):
            decimal_point = [
                tuple(_to_decimal(x) for x in momentum) for momentum in point
            ]
        rows = slice_.tree_rows.evaluate(decimal_point, parameters, numbers)
        scalars = slice_.scalars(parameters, numbers, poles)
        self.parameters = parameters
        self.scale = scale
        self._born_and_counterterms(rows, scalars, labels)
        self.rows = rows

    def finish(self, retest: str | None) -> _PointEvaluation:
        """Dress (schedule B), reduce and assemble the complete virtual."""
        self._reduce(self.dress().numerators, retest)
        self._assemble()
        return self

    # -- Born, counterterms and fields -------------------------------------------

    def _site_carrier(self) -> SiteCarrier:
        numbers = self.numbers
        constants = self.constants
        kernels: dict[int, rx.ExactKernel] = {}

        def kernel(spec, values):
            found = kernels.get(spec.kernel_id)
            if found is None:
                found = kernels[spec.kernel_id] = rx.ExactKernel(spec)
            return found.evaluate(values, constants)

        return SiteCarrier(numbers.colour, kernel, numbers.zero)

    def _born_and_counterterms(self, rows, scalars, labels) -> None:
        """Born colour vectors per helicity row, their square and the
        counterterm and field components, projected per sector (design §7.4).

        Colour vectors keep each literal tensor's N_c power (plus the coupling
        grade), so leading colour keeps exactly the sector's degree P before
        N_c = 3 is substituted; full colour keeps every power.
        """

        slice_, numbers = self.slice, self.numbers
        self.constants = rx.KernelConstants(
            _kernel_parameters(slice_.compiled, self.parameters), numbers.carrier
        )
        carrier = self._site_carrier()
        closure = slice_.closure
        momenta = {}
        for mask in range(1, 1 << len(labels)):
            total = leg_momentum(mask, self.incoming)
            momenta[mask] = [numbers.pair(-x) for x in total]  # tree momentum
        helicities = defaultdict(set)
        for block in slice_.blocks.blocks:
            if block.is_source:
                for row in block.rows:
                    ((leg, value),) = row.helicities
                    helicities[leg].add(value)

        def block_rows(term, helicity):
            values = []
            for block in term.blocks:
                want = tuple(sorted((leg, helicity[leg]) for leg in block.key.labels))
                row = next((r for r in block.rows if r.helicities == want), None)
                if row is None:
                    return None
                variant = "source" if block.is_source else "propagated"
                values.append(rows[(row.current_id, variant)])
            return values

        def kinematic(term, helicity) -> Any:
            values = block_rows(term, helicity)
            if values is None:
                return None
            if isinstance(term, WholeRuleTerm):
                return term.kinematic(values, numbers)
            vectors = [momenta[block.key.external_mask] for block in term.blocks]
            return term.sign * closure.value(term, values, vectors, carrier)

        degrees = slice_.degrees() if slice_.request.accuracy != "full" else None
        overlap = slice_.overlaps

        accuracy = slice_.request.accuracy

        weights: dict = {}
        # Per helicity: each bra's projection onto the ket keys, z[r] =
        # sum_l conj(bra_l) <l|r>; every interference with that bra is then
        # sum_r z[r] ket_r (one B x K product per bra instead of one per
        # interference).
        projections: dict = {}
        one = numbers.pair(1)

        def interfere(bra, ket, degree, grade=0):
            projection = projections.setdefault((id(bra), degree, grade), {})
            total = numbers.zero
            for key, b in ket.items():
                z = projection.get(key)
                if z is None:
                    z = projection[key] = _interfere(
                        bra,
                        {key: one},
                        degree,
                        accuracy,
                        overlap,
                        numbers,
                        grade,
                        weights,
                    )
                total = total + z * b
            return total

        def degree_of(kind, sector):
            return None if degrees is None else degrees[kind][sector]

        index = {tensor: b for b, tensor in enumerate(slice_.basis.tensors)}
        powers = LAURENT_POWERS if self.poles else (0,)
        whole = {}
        for terms in slice_.site_terms.values():
            for term in terms:
                if isinstance(term, WholeRuleTerm) and term not in whole:
                    whole[term] = term.coefficients(
                        {name: scalars[name, 0][0] for name in term.parameters},
                        numbers,
                    )
        born = defaultdict(lambda: numbers.zero)
        counterterms = defaultdict(lambda: numbers.zero)
        self.amplitudes: dict[rx.Row, dict[tuple[int, Orders], Any]] = {}
        self.born_vectors: dict[rx.Row, dict[Orders, dict]] = {}
        sectors = {pair: sector for sector, pairs in slice_.sectors for pair in pairs}
        for choice in product(*(sorted(helicities[label]) for label in labels)):
            helicity = dict(zip(labels, choice, strict=True))
            vectors: dict[Orders, dict] = {}
            projections.clear()
            for order, terms in slice_.born_terms.items():
                vector: dict = {}
                for term in terms:
                    value = kinematic(term, helicity)
                    if value is None:
                        continue
                    for grade, coupling in scalars[term.entry.coupling.name, 0].items():
                        _add_graded(
                            vector, term.colour, coupling * value, grade, numbers
                        )
                vectors[order] = vector
            row = tuple((label - 1, helicity[label]) for label in labels)
            self.born_vectors[row] = vectors
            self.amplitudes[row] = self._amplitudes(vectors, index)
            for left, right in product(vectors, repeat=2):
                squared = add_orders(left, right)
                born[squared] += interfere(
                    vectors[left], vectors[right], degree_of("born", squared)
                )
            loops: dict[tuple[str, int, Orders], dict] = defaultdict(dict)
            for site, terms in slice_.site_terms.items():
                for term in terms:
                    value = kinematic(term, helicity)
                    if value is None:
                        continue
                    value = len(site.alternatives) * value
                    for power in powers:
                        if isinstance(term, WholeRuleTerm):
                            grades = whole[term] if power == 0 else {}
                        else:
                            grades = scalars.get((term.entry.coupling.name, power), {})
                        for grade, coupling in grades.items():
                            _add_graded(
                                loops[site.component, power, site.orders],
                                term.colour,
                                coupling * value,
                                grade,
                                numbers,
                            )
            for (component, power, loop_order), vector in loops.items():
                for order, amplitude in vectors.items():
                    sector = sectors.get((order, loop_order))
                    if sector is None:
                        continue
                    value = interfere(amplitude, vector, degree_of("virtual", sector))
                    counterterms[component, sector, power] += value + value.conjugate()
            for insertion in slice_.fields:
                name = slice_.field_names[insertion.identity]
                ket = vectors.get(insertion.born_order)
                if ket is None:
                    continue
                for order, amplitude in vectors.items():
                    sector = sectors.get((order, insertion.orders))
                    if sector is None:
                        continue
                    for power in powers:
                        for grade, coupling in scalars.get((name, power), {}).items():
                            degree = degree_of("virtual", sector)
                            # The field coupling's N_c grade joins the
                            # overlap's powers (projection and N_c = 3).
                            value = coupling * interfere(amplitude, ket, degree, grade)
                            counterterms[FIELD_COMPONENT, sector, power] += (
                                value + value.conjugate()
                            )
        average = slice_.normalisation.average
        self.born = {order: value.real / average for order, value in born.items()}
        self.counterterms = {
            key: value.real / average for key, value in counterterms.items()
        }

    def _amplitudes(self, vectors, index) -> dict[tuple[int, Orders], dict[int, Any]]:
        """A0 per (Born tensor, Born order) split into N_c grades (seeds).

        A Born term's literal tensor keeps the N_c power of its colour
        reduction (e.g. the -1/N_c of a U(1) piece) and of its coupling, so
        the seeds project K[c, b] N_c^g exactly (``SeedTable.seeds``).
        """

        out: dict[tuple[int, Orders], dict[int, Any]] = {}
        legs = self.slice.basis.legs
        for order, vector in vectors.items():
            for (tensor, power), value in vector.items():
                grades = out.setdefault((index[_single(tensor, legs)], order), {})
                grades[power] = value + grades[power] if power in grades else value
        return out

    # -- loop numerators -----------------------------------------------------------

    def dress(
        self, schedule: Literal["B", "F"] = "B", *, exact: bool = False
    ) -> rx.Execution:
        """Family numerators of one schedule (F in the B tries' frames).

        ``exact`` dresses the carrier's tree rows and seeds, read as exact
        rationals, in exact arithmetic (kernel constants to the carrier's
        digits): schedules then agree exactly (O4).
        """

        slice_, numbers = self.slice, self.numbers
        rows = self.rows
        carrier = (
            rx.Carrier.exact(numbers.precision.working_digits)
            if exact
            else numbers.carrier
        )
        constants = (
            rx.KernelConstants(
                _kernel_parameters(slice_.compiled, self.parameters), carrier
            )
            if exact
            else self.constants
        )
        degrees = (
            slice_.degrees()["virtual"] if slice_.request.accuracy != "full" else {}
        )
        seeds = SectorSeeds(
            slice_.table,
            slice_.class_of,
            self.amplitudes,
            [pairs for _sector, pairs in slice_.sectors],
            numbers,
            slice_.request.accuracy,
            [degrees.get(sector) for sector, _pairs in slice_.sectors],
            fourier=None if exact else slice_.fourier,
            native=slice_.native,
        )
        self.colour_contraction = seeds.algorithm
        kinematics = rx.CatalogueKinematics(
            slice_.catalogue,
            slice_.compiled.ir.vertex_terms,
            slice_.data,
            _BlockRows(slice_, rows),
            self.incoming,
            constants,
            reading=slice_.reading,
        )
        executor = rx.ReferenceExecutor(
            kinematics,
            seeds,
            momenta=self.incoming,
            layout=slice_.layout,
            carrier=carrier,
            weight=slice_.weights,
        )
        with carrier.scope():
            if schedule == "B":
                return executor.run_b(slice_.plan)
            if schedule == "F":
                return executor.run_f(
                    rx.plan_skeletons(slice_.plan), rx.trie_targets(slice_.plan)
                )
        raise ValueError(f"unknown schedule {schedule!r}")

    # -- reduction -----------------------------------------------------------------

    def _reduce(self, numerators, retest: str | None) -> None:
        """Reduce every lane; ``retest`` re-reduces them in that frame
        (``stability.FRAMES``: "reversed" or "rescaled")."""
        slice_, numbers = self.slice, self.numbers
        native = slice_.native
        arithmetic = "double" if numbers.double else "arbitrary"
        digits = None if numbers.double else numbers.precision.digits
        lanes, groups = [], []
        masses = _mass_values(
            slice_.compiled,
            self.parameters,
            {mass for key in numerators for _mask, mass in key[0]},
        )

        def batch(values, lane: int):
            return native.NativeBatch.from_decimal(
                values, 1, len(values), arithmetic, digits, [lane]
            )

        def real(values):
            working = numbers.precision.working_digits
            return [[_decimal_string(x, working), "0"] for x in values]

        for (family, loop_order), numerator in sorted(numerators.items()):
            sequence = numerator.sequence
            n = len(sequence)
            vectors = [leg_momentum(mask, self.incoming) for mask in shifts(sequence)]
            matrix = [
                _minkowski([a - b for a, b in zip(vectors[i], vectors[j], strict=True)])
                for i in range(n)
                for j in range(n)
            ]
            for column, coefficients in enumerate(numerator.columns):
                if not any(coefficients):
                    continue
                lane = len(lanes)
                lanes.append((family, loop_order, slice_.sectors[column][0]))
                groups.append(
                    (
                        n,
                        batch(real(x for v in vectors for x in v), lane),
                        batch(real(masses[mass] for _mask, mass in sequence), lane),
                        batch(real([self.scale]), lane),
                        batch(real(matrix), lane),
                        batch([numbers.strings(c) for c in coefficients], lane),
                    )
                )
        reducer = native.V2Reducer(
            arithmetic, digits, "fast" if numbers.double else "generic"
        )
        direct, check = reducer.reduce(
            groups,
            poles=self.poles,
            check=retest is not None,
            rescale=retest == "rescaled",
        )
        self.families = {}
        self.retest_families = {} if check is not None else None
        for lane, batch in zip(lanes, direct, strict=True):
            self.families[lane] = self._laurent(batch, lane)
        if check is not None:
            for lane, batch in zip(lanes, check, strict=True):
                self.retest_families[lane] = self._laurent(batch, lane)

    def _laurent(self, batch, lane) -> Laurent:
        ((_row, status),) = batch.statuses
        if status != "success":
            raise ArithmeticError(f"reduction lane {lane} failed: {status}")
        values = [
            self.numbers.pair(Decimal(re), Decimal(im))
            for re, im in batch.values.to_decimal()
        ]
        return Laurent(*values)

    # -- assembly --------------------------------------------------------------------

    def _bare(self, families) -> dict[Orders, Laurent]:
        numbers = self.numbers
        average = self.slice.normalisation.average
        factor = numbers.pair(0, 1) * (
            1 / (LOOP_FACTOR_DENOMINATOR * numbers.pi * numbers.pi * average)
        )
        out: dict[Orders, list[Any]] = {}
        for (_family, _loop, sector), value in families.items():
            terms = out.setdefault(sector, [numbers.zero] * len(value.coefficients))
            for k, coefficient in enumerate(value.coefficients):
                terms[k] = terms[k] + coefficient
        return {
            sector: Laurent(*[2 * (factor * value).real for value in terms])
            for sector, terms in out.items()
        }

    def _assemble(self) -> None:
        slice_, numbers = self.slice, self.numbers
        width = 3 if self.poles else 1
        zero = numbers.real(0)
        bare = self._bare(self.families)
        components: dict[str, dict[Orders, Laurent]] = {name: {} for name in COMPONENTS}
        for sector, _pairs in slice_.sectors:
            values = {
                "bare": bare.get(sector, Laurent(*[zero] * width)),
            }
            for name in ("r2", "uv_local", FIELD_COMPONENT):
                values[name] = Laurent(
                    *[
                        self.counterterms.get((name, sector, power), zero)
                        for power in LAURENT_POWERS[:width]
                    ]
                )
            values["uv"] = values["uv_local"] + values[FIELD_COMPONENT]
            values["virtual"] = values["bare"] + values["uv"] + values["r2"]
            for name in COMPONENTS:
                components[name][sector] = values[name]
        totals = [sum(self.born.values(), zero)]
        for name in COMPONENTS:
            for k in range(width):
                totals.append(
                    sum(
                        (
                            laurent.coefficients[k]
                            for laurent in components[name].values()
                        ),
                        zero,
                    )
                )
        # Catani's operator is the full-colour pole prediction of QCD
        # corrections with massless coloured legs.
        checked = (
            slice_.request.accuracy == "full"
            and slice_.alpha_s_parameter
            and self._massless_coloured_legs()
        )
        ir_poles = self._ir_poles() if self.poles and checked else None
        self.result = PointResult(
            dict(self.born),
            components,
            tuple(totals),
            dict(self.families),
            ir_poles,
            colour_contraction=self.colour_contraction,
        )
        self.absolute = self._lane_budget(
            {lane: value.coefficients for lane, value in self.families.items()},
            [components[name] for name in LOCAL_COMPONENTS],
        )
        self.retest_budget = None
        self.family_deviations = None
        if self.retest_families is not None:
            self.family_deviations = {
                lane: tuple(
                    abs(a - b)
                    for a, b in zip(
                        self.families[lane].coefficients,
                        self.retest_families[lane].coefficients,
                        strict=True,
                    )
                )
                for lane in self.families
            }
            self.retest_budget = self._lane_budget(self.family_deviations)

    # -- stability budgets (``stability``) -------------------------------------------

    def _lane_budget(self, lanes, local=()) -> dict[Orders, Budget]:
        """Per sector, sum_lanes |delta bare| <= 2 |delta Omega| / (16 pi^2
        average) plus the absolute local components."""

        factor = 2 / (
            LOOP_FACTOR_DENOMINATOR * math.pi**2 * self.slice.normalisation.average
        )
        width = 3 if self.poles else 1
        budget = {sector: [0.0] * width for sector, _pairs in self.slice.sectors}
        for (_family, _loop, sector), errors in lanes.items():
            for k, error in enumerate(errors):
                budget[sector][k] += factor * float(abs(error))
        for components in local:
            for sector, laurent in components.items():
                for k, value in enumerate(laurent.coefficients):
                    budget[sector][k] += abs(float(value))
        return {sector: Budget(tuple(values)) for sector, values in budget.items()}

    def rotation_budget(self, rotated: _PointEvaluation) -> dict[Orders, Budget]:
        """The error budget of one rotation check: every reduction lane's
        integral and every local component is rotation invariant."""

        zero = Laurent(*[self.numbers.zero] * (3 if self.poles else 1))
        lanes = {
            lane: tuple(
                a - b
                for a, b in zip(
                    rotated.families.get(lane, zero).coefficients,
                    self.families.get(lane, zero).coefficients,
                    strict=True,
                )
            )
            for lane in self.families.keys() | rotated.families.keys()
        }
        local = [
            differences(self.result.components[name], rotated.result.components[name])
            for name in LOCAL_COMPONENTS
        ]
        return self._lane_budget(lanes, local)

    # -- IR poles ----------------------------------------------------------------------

    def _massless_coloured_legs(self) -> bool:
        """Whether every coloured leg is massless (Catani's massless dipoles)."""

        return not any(
            any(self.incoming[leg.label - 1])
            and _minkowski(self.incoming[leg.label - 1])
            for leg in self.slice.basis.legs
            if leg.representation != 1
        )

    def _ir_poles(self) -> dict[Orders, tuple[Any, Any]]:
        """Catani's single and double poles of the renormalised virtual.

        V_poles = (alpha_s / 2 pi) sum_i sum_{j != i} (1/C_i) (C_i/eps^2 +
        gamma_i/eps) (1 + eps log(mu^2 / |2 p_i.p_j|)) <T_i.T_j>, massless
        coloured legs, QCD corrections, n_f light flavours in gamma_g.
        """

        slice_, numbers = self.slice, self.numbers
        legs = {leg.label: leg for leg in slice_.basis.legs}
        coloured = sorted(
            label for label, leg in legs.items() if leg.representation != 1
        )
        dipoles = slice_.dipoles()
        three = Fraction(NC)
        physical = []
        for vectors in self.born_vectors.values():
            physical.append({})
            for order, vector in vectors.items():
                folded = physical[-1].setdefault(order, {})
                for (tensor, power), value in vector.items():
                    scaled = value * numbers.real(three**power)
                    folded[tensor] = folded.get(tensor, numbers.zero) + scaled
        values = _kernel_parameters(slice_.compiled, self.parameters)
        light = _light_flavours(slice_.data, values)
        alpha = numbers.real(values[slice_.alpha_s_parameter])
        prefactor = alpha / (2 * numbers.pi)
        mu2 = numbers.real(self.scale)
        correction = add_orders(tuple((name, 2) for name in slice_.request.corrections))
        out = {}
        for sector, _pairs in slice_.sectors:
            single = double = numbers.real(0)
            for i in coloured:
                representation = legs[i].representation
                casimir = _casimir(representation)
                gamma = _gamma(representation, light)
                for j in coloured:
                    if j == i:
                        continue
                    correlated = numbers.real(0)
                    for vectors in physical:
                        for a, left in vectors.items():
                            for b, right in vectors.items():
                                if add_orders(a, b, correction) != sector:
                                    continue
                                correlated += self._sandwich(
                                    left, right, dipoles[(i, j)]
                                )
                    correlated = correlated / slice_.normalisation.average
                    invariant = 2 * _minkowski_dot(
                        self.incoming[i - 1], self.incoming[j - 1]
                    )
                    log = _log(mu2 / numbers.real(abs(invariant)), numbers)
                    double += correlated
                    single += (
                        numbers.real(gamma) / numbers.real(casimir) + log
                    ) * correlated
            out[sector] = (prefactor * single, prefactor * double)
        return out

    def _sandwich(self, left, right, matrix) -> Any:
        slice_ = self.slice
        index = {tensor: b for b, tensor in enumerate(slice_.basis.tensors)}
        total = self.numbers.zero
        for t, a in left.items():
            row = matrix[index[_single(t, slice_.basis.legs)]]
            for u, b in right.items():
                cell = row[index[_single(u, slice_.basis.legs)]]
                if cell:
                    total = total + a.conjugate() * b * self.numbers.real(cell)
        return total.real


# --------------------------------------------------------------------- helpers


class _NativePoint(_PointEvaluation):
    """A point whose block rows and Born components the native executor
    forms (``plan_export.NativeExecutor(inputs="native")``, WP3.1).

    Only parameter-dependent values are prepared here (kernel constants); the
    result holds the Born (when the executor forms |A0|^2) and the ``bare``
    component per sector. Counterterms and external fields join with the
    native counterterm program (WP3.1 I5).
    """

    def __init__(self, slice_, point, scale, parameters, numbers):
        """``parameters`` are the point's complete values
        (``LoopSlice._point_parameters``)."""

        self.slice = slice_
        self.numbers = numbers
        self.poles = True
        process = slice_.blocks.process
        labels = sorted(leg.label for leg in process.legs)
        initial = {leg.label for leg in process.legs if leg.is_initial}
        self.incoming = [
            [x if label in initial else -x for x in point[label - 1]]
            for label in labels
        ]
        self.scale = scale
        self.parameters = parameters
        self.constants = rx.KernelConstants(
            _kernel_parameters(slice_.compiled, parameters), numbers.carrier
        )
        self.born: dict[Orders, Any] = {}
        self.counterterms: dict = {}
        self.colour_contraction = "direct"
        self.retest_families = None

    def _assemble(self) -> None:
        slice_, numbers = self.slice, self.numbers
        width = 3 if self.poles else 1
        zero = numbers.real(0)
        bare = self._bare(self.families)
        components = {
            "bare": {
                sector: bare.get(sector, Laurent(*[zero] * width))
                for sector, _pairs in slice_.sectors
            }
        }
        totals = [sum(self.born.values(), zero)]
        for k in range(width):
            totals.append(
                sum(
                    (
                        laurent.coefficients[k]
                        for laurent in components["bare"].values()
                    ),
                    zero,
                )
            )
        self.result = PointResult(
            dict(self.born),
            components,
            tuple(totals),
            dict(self.families),
            None,
            colour_contraction=self.colour_contraction,
        )
        self.absolute = self._lane_budget(
            {lane: value.coefficients for lane, value in self.families.items()}
        )
        self.retest_budget = None
        self.family_deviations = None
        if self.retest_families is not None:
            self.family_deviations = {
                lane: tuple(
                    abs(a - b)
                    for a, b in zip(
                        self.families[lane].coefficients,
                        self.retest_families[lane].coefficients,
                        strict=True,
                    )
                )
                for lane in self.families
            }
            self.retest_budget = self._lane_budget(self.family_deviations)

    def rotation_budget(self, rotated: _PointEvaluation) -> dict[Orders, Budget]:
        zero = Laurent(*[self.numbers.zero] * (3 if self.poles else 1))
        lanes = {
            lane: tuple(
                a - b
                for a, b in zip(
                    rotated.families.get(lane, zero).coefficients,
                    self.families.get(lane, zero).coefficients,
                    strict=True,
                )
            )
            for lane in self.families.keys() | rotated.families.keys()
        }
        return self._lane_budget(lanes)


def _max(a: int | None, b: int | None) -> int | None:
    return b if a is None else a if b is None else max(a, b)


def _add_graded(vector: dict, tensor, value, grade: int, numbers: _Numbers) -> None:
    """``vector += value * tensor``, keyed by (literal tensor, N_c power)."""

    for term in tensor.terms:
        key = (term.tensor, term.nc_power + grade)
        weight = numbers.colour(term.coefficient, tensor.normalization_power)
        vector[key] = vector.get(key, numbers.zero) + weight * value


def _interfere(
    bra: dict,
    ket: dict,
    degree: int | None,
    accuracy: ColourAccuracy,
    overlaps,
    numbers,
    grade: int = 0,
    weights: dict | None = None,
) -> Any:
    """N_c^grade sum conj(bra) ket <t|t'> at N_c = 3, keeping the N_c powers
    that ``accuracy`` keeps below the leading degree ``degree``
    (``colour.kept``). ``weights`` caches the carrier colour weights of one
    point (they do not depend on the helicities)."""

    weights = {} if weights is None else weights
    total = numbers.zero
    for (left, p), a in bra.items():
        conjugate = a.conjugate()
        for (right, q), b in ket.items():
            for power, cell in overlaps(left, right).items():
                exponent = power + p + q + grade
                if not kept(exponent, degree, accuracy):
                    continue
                key = (left, right, power, exponent)
                weight = weights.get(key)
                if weight is None:
                    weight = weights[key] = numbers.colour(
                        cell.scaled(Fraction(NC) ** exponent), 0
                    )
                total = total + conjugate * b * weight
    return total


class _BlockRows:
    """Tree rows of the slice as ``rx.BlockRows``, as Feynman-rule currents."""

    def __init__(self, slice_: LoopSlice, rows):
        from .blocks import _NativeFields

        fields = _NativeFields.of(slice_.data)
        index = slice_._cache.get("block-rows")
        if index is None:
            index = defaultdict(lambda: defaultdict(dict))
            for block in slice_.blocks.blocks:
                key = block.key
                if key.auxiliary_kind is not None:
                    continue
                name = fields.incoming(key.particle_id)["name"]
                bucket = index[(key.external_mask, name, key.coupling_orders, key.word)]
                for row in block.rows:
                    helicities = tuple((label - 1, h) for label, h in row.helicities)
                    bucket[key.chirality][helicities] = (block, row)
            slice_._cache["block-rows"] = index
        self._index = index
        self._rows = rows

    def rows(self, block, word):
        found = self._index[(block.mask, block.field, block.orders, word)]
        return {chirality: tuple(rows) for chirality, rows in found.items()}

    def value(self, block, word, chirality, row, dimension):
        tree, entry = self._index[(block.mask, block.field, block.orders, word)][
            chirality
        ][row]
        variant = "source" if tree.is_source else "propagated"
        sign = tree.wick_sign
        return tuple(sign * value for value in self._rows[(entry.current_id, variant)])


def _ordinary_entries(prepared, data) -> tuple[tuple[Any, ...], ...]:
    """Ordinary (non-ghost) tree vertex entries by vertex, for the Born closure."""

    spins = {entry["name"]: entry["spin"] for entry in data["particles"]}
    out = defaultdict(list)
    for entry in prepared.model.kernels:
        if entry.kind == "tree" and all(
            spins.get(name, -1) != -1 for name in entry.particle_names
        ):
            out[entry.vertex_id].append(entry)
    return tuple(tuple(rows) for _vertex, rows in sorted(out.items()))


def _born_terms(blocks, table, closure: SiteClosure, ordinary) -> dict:
    """Born terms rooted at the vertex holding leg 1, by amplitude order."""

    full = (1 << len(table.legs)) - 1
    born = set(blocks.born_orders)
    out = defaultdict(list)
    for entries in ordinary:
        fields = entries[0].particle_names
        order = add_orders(tuple(entries[0].order))
        for parts in _set_partitions(full, len(fields), len(fields)):
            if 1 not in parts:
                continue
            for choice in product(*(table.currents(part) for part in parts)):
                if sorted(b.field for b in choice) != sorted(fields):
                    continue
                total = add_orders(order, *(b.orders for b in choice))
                if total not in born:
                    continue
                free, attachments = sorted(choice), []
                for name in fields:
                    block = next(b for b in free if b.field == name)
                    free.remove(block)
                    attachments.append(block)
                out[total].extend(closure.vertex_terms(entries, attachments))
    return {order: tuple(terms) for order, terms in out.items()}


def _born_tensors(closure: SiteClosure, born_terms) -> BornBasis:
    """The literal colour tensors of the Born closure, as a seed basis."""

    one = ExactColorCoefficient(1)
    tensors: dict[Any, set[Orders]] = {}
    for order, terms in born_terms.items():
        for term in terms:
            for item in term.colour.terms:
                tensors.setdefault(item.tensor, set()).add(order)
    ordered = list(tensors)
    return BornBasis(
        closure.legs,
        tuple(range(len(ordered))),
        tuple(
            ConnectedColorTensor(closure.legs, (TensorTerm(one, tensor),))
            for tensor in ordered
        ),
        tuple(frozenset(tensors[tensor]) for tensor in ordered),
    )


def _single(tensor, legs) -> ConnectedColorTensor:
    return ConnectedColorTensor(legs, (TensorTerm(ExactColorCoefficient(1), tensor),))


def _kernel_parameters(compiled, parameters: Mapping[str, str]) -> dict[str, Fraction]:
    values = {
        name: Fraction(str(value[0]))
        for name, value in compiled.parameter_defaults.items()
    }
    values.update({name: Fraction(value) for name, value in parameters.items()})
    return values


def _mass_values(
    compiled, parameters: Mapping[str, str], masses: Iterable[Any]
) -> dict[Any, Fraction]:
    """Squared real loop masses per mass id (widths are not part of the slice)."""

    values = _kernel_parameters(compiled, parameters)

    return {
        mass: Fraction(0) if mass.massless else values[mass.mass] ** 2
        for mass in masses
    }


def _light_flavours(data, values: Mapping[str, Fraction]) -> int:
    """Quark flavours (colour triplets) whose bound mass is zero."""

    return sum(
        1
        for p in data["particles"]
        if int(p["spin"]) == FERMION_SPIN
        and int(p["color"]) == 3
        and (mass_name(p["mass"]) == ZERO or values.get(p["mass"]) == 0)
    )


# UFO/SLHA codes of the renormalisation scale and of alpha_s (SliceRequest).
_LHA_SCALE = ("LOOP", (1,))
_LHA_ALPHA_S = ("SMINPUTS", (3,))


def _external_parameter(
    compiled, name: str | None, code: tuple[str, tuple[int, ...]], fixed
) -> str:
    """An adjustable external model parameter, given or found by its code."""

    externals = {
        record.name: record
        for record in compiled.ir.parameters
        if record.nature == "external"
    }
    if name is None:
        block, numbers = code
        found = [
            record.name
            for record in externals.values()
            if (record.lhablock or "").upper() == block
            and tuple(record.lhacode) == numbers
        ]
        if len(found) != 1:
            raise ValueError(
                f"no unique external parameter with code {block} {numbers}; "
                "name it in the SliceRequest"
            )
        (name,) = found
    if name not in externals or name in fixed:
        raise ValueError(f"{name!r} is not an adjustable external model parameter")
    return name


def _casimir(representation: int) -> Fraction:
    if representation == 8:
        return Fraction(NC)
    if abs(representation) == 3:
        return Fraction(NC * NC - 1, 2 * NC)
    raise NotImplementedError(f"colour representation {representation}")


def _gamma(representation: int, light: int) -> Fraction:
    if representation == 8:
        return Fraction(11, 6) * NC - Fraction(2, 3) * Fraction(1, 2) * light
    return Fraction(3, 2) * _casimir(representation)


def _term_key(term: Any, rename: Mapping[int, int]) -> Hashable:
    """A key of one Born tensor term with its adjoint labels renamed."""

    return term.coefficient, term.nc_power, _colour_key(term.tensor, rename)


def _real_colour(value: ExactColorCoefficient) -> Fraction:
    if value.imag:
        raise ValueError("a colour correlation is not real")
    return value.real


def _minkowski(vector) -> Fraction:
    e, x, y, z = vector
    return e * e - x * x - y * y - z * z


def _minkowski_dot(a, b) -> Fraction:
    return a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]


def _log(value, numbers: _Numbers):
    if numbers.double:
        return math.log(value)
    return value.ln()


def _fraction(value: Any) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    return Fraction(str(value))


def _to_decimal(value: Fraction) -> Decimal:
    return Decimal(value.numerator) / Decimal(value.denominator)


def _decimal_string(value: Any, digits: int) -> str:
    """Exact decimal digits of a terminating rational, else ``digits``
    significant ones (the carrier's working digits)."""

    value = _fraction(value)
    denominator = value.denominator
    for prime in (2, 5):
        while denominator % prime == 0:
            denominator //= prime
    if denominator == 1:
        with localcontext(
            prec=len(str(value.numerator)) + len(str(value.denominator)) + 5
        ):
            return str(Decimal(value.numerator) / Decimal(value.denominator))
    with localcontext(prec=digits):
        return str(Decimal(value.numerator) / Decimal(value.denominator))


def _sqrt_fraction(value: Fraction, digits: int) -> Fraction:
    """sqrt of an exact rational: exact when it is a square, else to ``digits``."""

    numerator, denominator = value.numerator, value.denominator
    a, b = math.isqrt(numerator), math.isqrt(denominator)
    if a * a == numerator and b * b == denominator:
        return Fraction(a, b)
    with localcontext(prec=digits + 10):
        return Fraction((Decimal(numerator) / Decimal(denominator)).sqrt())


__all__ = [
    "COMPONENTS",
    "LAURENT",
    "Laurent",
    "LoopSlice",
    "PointResult",
    "Precision",
    "SectorSeeds",
    "SliceRequest",
    "TreeRows",
    "WholeRuleTerm",
]
