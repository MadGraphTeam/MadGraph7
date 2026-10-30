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

* ``poles="none"`` evaluates the same generated slice in finite-only mode:
  the reducer returns no pole columns, counterterm and field couplings are
  bound only at eps^0, and the IR-pole check is skipped. Pole coefficients
  are then absent (``None``), not zero. ``poles="uv"`` returns the bare
  loop's UV poles as the virtual's (the counterterms' UV poles in their
  components) and ``poles="ir"`` the renormalised virtual's IR poles, both
  with every component's UV and IR parts (``poles``, design §12.4); the
  finite part is the same in all four modes.
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

import copy
import functools
import json
import math
from collections import defaultdict
from collections.abc import Callable, Container, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import permutations, product
from typing import Any, Literal, NamedTuple

from pyamplicol.color.connections import (
    ColorConnection,
    ConnectedColorTensor,
    EmitGluon,
    ExactColorCoefficient,
    TensorTerm,
    apply_color_connection,
    evaluate_color_nc_terms,
)
from pyamplicol.color.contraction_types import NC
from pyamplicol.config import PoleSelection

from . import ew_policy, generator_input
from . import reference_executor as rx
from .blocks import (
    TreeBlocks,
    build_tree_blocks,
    requested_loop_blocks,
    vetoed_particle_ids,
    with_ward_kernels,
)
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
    contract_colour,
    extend_seed_table,
    kept,
    seed_fourier,
)
from .conventions import (
    LOOP_FACTOR_DENOMINATOR,
    LeafWeights,
    Normalisation,
    TreeReading,
)
from .ct_sites import (
    LAURENT_POWERS,
    ConjugateVertices,
    CountertermSites,
    SiteClosure,
    SiteTerm,
    generate_counterterm_sites,
    vanishes_identically,
    vanishing_loop_vertices,
)
from .ct_sites import (
    Carrier as SiteCarrier,
)
from .families import (
    ZERO,
    BlockCurrent,
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
    onshell_leg,
    orders_within,
    shifts,
)
from .fields import FIELD_COMPONENT, FieldInsertion, external_field_insertions
from .generator_input import NativeGeneration
from .poles import (
    ORIGINS,
    UNTAGGED_PHOTON_CONVENTION,
    PoleRegulators,
    catani_poles,
    colour_casimir,
    coupling_value,
    dressed_born,
    exact_charge,
    has_poles,
    photon_gamma,
    pole_mode,
    pole_regulators,
    qed_catani_poles,
    qed_coupling,
    qed_gamma,
    reducer_poles,
    regulated_parameters,
    sector_predictions,
    sector_sum,
    select_lane,
    select_sector,
    separated_exchange,
    split_sector,
    splits,
    strong_beta0,
    strong_gamma,
)
from .poly_affine import MonomialLayout, leg_momentum
from .refusals import (
    require_colour_preserving_fermion_lines,
    require_declared_colour_fields,
    require_declared_colour_rules,
    require_labelled_counterterms_for_veto,
    require_nc_exact,
)
from .schedules import (
    STRUCTURED,
    LoopSchedules,
    SkeletonView,
    build_schedules,
    cheapest,
)
from .skeletons import Skeleton
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


def _canonical_runtime_schema(raw: Mapping[str, Any]) -> dict[str, Any]:
    """``RuntimeExpressionSchema.from_mapping(raw).to_mapping()`` without
    keeping the contract's canonical JSON and payload cache: the same keys,
    order, values and checks."""

    from pyamplicol.generation.contracts import _REQUIRED_RUNTIME_SCHEMA_KEYS

    try:
        canonical = json.dumps(
            raw, allow_nan=False, separators=(",", ":"), sort_keys=True
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "runtime expression schema must contain canonical JSON values"
        ) from exc
    payload = json.loads(canonical)
    del canonical
    if not isinstance(payload, dict):
        raise ValueError("runtime expression schema root must be an object")
    missing = sorted(_REQUIRED_RUNTIME_SCHEMA_KEYS - payload.keys())
    if missing:
        raise ValueError("runtime expression schema is missing: " + ", ".join(missing))
    return {str(key): value for key, value in payload.items()}


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

    def __init__(
        self,
        dag: Any,
        compiled: Any,
        *,
        jit: bool = True,
        propagated_currents: Iterable[int] | None = None,
    ):
        from pyamplicol.generation.runtime_schema import build_runtime_schema
        from pyamplicol.models.external import CompiledUFOModel

        self.compiled = compiled
        self.dag = dag
        self.model = CompiledUFOModel(compiled)
        with _exact_u1_subtraction_weights():
            # Every block enters a loop vertex with its own propagator, also
            # where no tree interaction consumes it (reachability-pruned DAGs):
            # by default every current is propagated.
            raw = build_runtime_schema(
                dag,
                self.model,
                process_id="loop-v2-blocks",
                propagated_currents=(
                    [current.id for current in dag.currents]
                    if propagated_currents is None
                    else propagated_currents
                ),
            )
        # The runtime schema's canonical mapping (``RuntimeExpressionSchema.
        # to_mapping`` of ``build_runtime_expression_schema``); the contract
        # object, whose canonical JSON and cached payload would double it
        # (0.24 GB for g g > g g g g's counterterm program), is built from it
        # on the first replay (``runtime_schema``): native per-point trees
        # read the mapping only.
        self.schema = _canonical_runtime_schema(raw)
        del raw
        self._runtime_schema: Any = None
        # The stage expressions are built on first use (``stages``): native
        # per-point trees compile their own, stage by stage (``native_trees``).
        self._blueprint = None
        self._roots = {
            int(root["output_index"]): int(root["root_id"])
            for root in self.schema["amplitude_stage"]["roots"]
        }
        # Built on the first replay: native per-point trees (``native_trees``)
        # read the stage expressions only.
        self._jit = jit
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

    @property
    def runtime_schema(self) -> Any:
        """The runtime schema contract (``RuntimeExpressionSchema``) of the
        DAG, built on first use from ``schema``: its canonical JSON is that of
        ``build_runtime_expression_schema`` (canonical JSON round-trips)."""

        if self._runtime_schema is None:
            from pyamplicol.generation.contracts import RuntimeExpressionSchema

            self._runtime_schema = RuntimeExpressionSchema.from_mapping(self.schema)
        return self._runtime_schema

    def _built(self) -> Any:
        from pyamplicol.generation.contracts import StageCompilationInput
        from pyamplicol.generation.stage_planning import (
            build_generic_stage_compiler_blueprint,
        )

        if self._blueprint is None:
            with _exact_u1_subtraction_weights():
                blueprint = build_generic_stage_compiler_blueprint(
                    StageCompilationInput(self.dag, self.model, self.runtime_schema)
                )
            if not blueprint.expression_ready:
                raise ValueError(
                    f"tree blocks are not expression ready: {blueprint.blockers}"
                )
            self._blueprint = blueprint
        return self._blueprint

    @property
    def stages(self) -> tuple[Any, ...]:
        return self._built().stages

    @property
    def _amplitude_stage(self) -> Any:
        return self._built().amplitude_stage

    def _model_values(self, parameters: Mapping[str, str], digits: int) -> list:
        external = {
            record["name"]: Decimal(
                parameters.get(record["name"], repr(record["default"]))
            )
            for record in self._records
            if record["kind"] == "external_parameter"
        }
        substitutions = _substitutions(self.compiled, external, digits)
        derived, values = {}, []
        for record in self._records:
            if record["kind"] == "external_parameter":
                values.append(external[record["name"]])
                continue
            name = record["runtime_name"]
            if name not in derived:
                derived[name] = _definition_value(
                    self._definitions[name], substitutions, digits
                )
            values.append(derived[name][record["complex_component"] == "imag"])
        return values

    def evaluate(
        self,
        point: Sequence[Sequence[Decimal]],
        parameters: Mapping[str, str],
        numbers: _Numbers,
        *,
        amplitudes: bool = False,
        sources: Mapping[int, Sequence[tuple[Decimal, Decimal]]] | None = None,
    ) -> dict[tuple[int, str], tuple[Any, ...]]:
        """The rows at one point; ``sources`` replaces the wavefunction of
        every helicity source of a vector leg (1-based label) by a literal
        complex four-vector (spin sources, Ward vectors; design §8.4), with
        the source's crossing phase and nothing else applied."""

        from pyamplicol.runtime.correlated_exact import _fill_correlated_sources
        from pyamplicol.runtime.symbolica_exact import _fill_momenta, _fill_sources

        digits = numbers.precision.working_digits
        values = self._model_values(parameters, digits)
        zero = (Decimal(0), Decimal(0))
        state = [zero] * max(self._size, self._offset + len(values))
        with localcontext() as context:
            context.prec = digits + 10
            if sources:
                _fill_correlated_sources(
                    state, point, self.schema, tuple(values), dict(sources)
                )
            else:
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
            self._evaluators = tuple(
                _stage_evaluator(stage, jit=self._jit) for stage in self.stages
            )
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
                self._amplitude_evaluator = _stage_evaluator(stage, jit=self._jit)
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


def _stage_evaluator(stage: Any, *, jit: bool = True) -> Any:
    """The Symbolica evaluator of one tree stage, independent of the process
    history (``_canonical_stage``); ``jit=False`` keeps it interpreted (one
    point at a time costs the same, without the JIT compile)."""

    from symbolica import Expression, FunctionDefinition

    outputs, parameters, functions = _canonical_stage(stage)
    return Expression.evaluator_multiple(
        outputs,
        parameters,
        functions=[
            FunctionDefinition(function, arguments, body, inlining="always")
            for function, arguments, body in functions
        ],
        jit_compile=jit,
    )


# Canonical evaluator symbols (``_canonical_symbols``), allocated in blocks of
# 2^16 whose first id is a multiple of 2^16.
_CANONICAL_SYMBOLS: list[Any] = []
_CANONICAL_BLOCK = 1 << 16


def _canonical_symbols(count: int) -> list[Any]:
    """The first ``count`` canonical symbols ``pyamplicol_v2_canonical::c<k>``,
    ordered the same way by Symbolica in every process.

    Symbolica numbers symbols in creation order and sorts the terms of a sum
    by the bytes of their packed symbol ids (``AtomView::cmp_terms``; ids are
    packed little-endian, in 1, 2 or 4 bytes by size). The term order of a
    normalised expression, and with it the evaluator built from it, therefore
    depends on the absolute ids of its symbols, i.e. on every symbol created
    earlier in the interpreter (the model compile, earlier generations). The
    canonical symbols are created in whole blocks of 2^16 starting at an id
    that is a multiple of 2^16 (unrelated padding symbols fill the gap; the
    next id is the number of symbols defined so far): every canonical id is
    then packed in four bytes as (k mod 256, k div 256 mod 256, block), so
    symbol k sorts by k's own bytes in every process history. (Two blocks
    order by their block numbers' low byte, which wraps only beyond 2^24
    symbols.) Creating a block costs about 0.2 s. Each new block is checked
    (``_check_canonical_block``): a misaligned block, or a Symbolica whose
    term order no longer follows those bytes, raises instead of silently
    making generation depend on the process history again."""

    from symbolica import Expression

    from pyamplicol._internal.physics.symbols import symbols

    while len(_CANONICAL_SYMBOLS) < count:
        start = len(_CANONICAL_SYMBOLS)
        padding = -len(Expression.get_all_symbol_names()) % _CANONICAL_BLOCK
        for k in range(padding):
            symbols.qualified(f"pyamplicol_v2_canonical::pad{start}_{k}")
        block = [
            symbols.qualified(f"pyamplicol_v2_canonical::c{start + k}")
            for k in range(_CANONICAL_BLOCK)
        ]
        _check_canonical_block(block)
        _CANONICAL_SYMBOLS.extend(block)
    return _CANONICAL_SYMBOLS[:count]


# Block offsets whose relative order changes under every misalignment of the
# block by 1 to 2^16 - 1 ids (both byte boundaries and the block's end).
_ALIGNMENT_PROBE = (0, 1, 255, 256, 257, 511, 65280, 65535)


def _check_canonical_block(block: list[Any]) -> None:
    """Raise unless a sum of the probe symbols of a fresh canonical block
    normalises in the order of their offsets' bytes (low byte first): the
    order of an aligned block in every process history."""

    from symbolica import Expression

    probe = sum((block[k] for k in _ALIGNMENT_PROBE), Expression.num(0))
    expected = sorted(_ALIGNMENT_PROBE, key=lambda k: (k % 256, k // 256))
    if [str(term) for term in probe] != [str(block[k]) for k in expected]:
        raise RuntimeError(
            "canonical Symbolica symbols are not ordered by their block offset "
            f"(got {probe}): the symbol ids are misaligned or Symbolica's term "
            "order changed, so tree stages would depend on the process history"
        )


# A stage with function definitions is renamed on Symbolica's portable text
# (``_renamed_text``: one native pass per expression, linear in its size and
# in parallel; the renamed text, on short canonical names, parses faster
# than the original), not by ``replace_multiple`` with a rule per symbol and
# repeated call passes (46 s of the 2->3 electroweak parameter program, M-SM
# MS-GEN). Both give the same expressions; the ``replace_multiple`` passes
# stay the oracle (``text_canonical_stage(False)``).
TEXT_CANONICAL_STAGE = True


def text_canonical_stage(enabled: bool) -> Any:
    """``TEXT_CANONICAL_STAGE`` set to ``enabled`` for a ``with`` block and
    restored after it: ``text_canonical_stage(False)`` is the oracle lane."""

    from contextlib import contextmanager

    @contextmanager
    def switch():
        global TEXT_CANONICAL_STAGE
        before = TEXT_CANONICAL_STAGE
        TEXT_CANONICAL_STAGE = bool(enabled)
        try:
            yield
        finally:
            TEXT_CANONICAL_STAGE = before

    return switch()


def _renamed_text(
    expressions: Sequence[Any],
    variables: Mapping[Any, Any],
    heads: Mapping[Any, Any],
) -> list[Any]:
    """``expressions`` with every symbol of ``variables`` and every call of a
    function of ``heads`` renamed (old -> new symbol): their portable text
    (``repr``: every namespace but the default one) renamed natively in
    parallel (``loop_v2::generate::rename``: a token is a maximal run of
    name characters, ``:``, ``{`` and ``}``, a head when a ``(`` follows it
    and a variable otherwise) and parsed back, a chunk at a time."""

    from symbolica import E

    native = _loop_native()
    variable_names = [(repr(old), repr(new)) for old, new in variables.items()]
    head_names = [(repr(old), repr(new)) for old, new in heads.items()]
    renamed: list[Any] = []
    for start in range(0, len(expressions), _RENAME_CHUNK):
        texts = native.loop_rename_symbols(
            [repr(e) for e in expressions[start : start + _RENAME_CHUNK]],
            variable_names,
            head_names,
        )
        renamed.extend(E(text) for text in texts)
    return renamed


# Expressions per native rename call (the parameter program's outputs reach
# megabytes each on the electroweak UFOs).
_RENAME_CHUNK = 256


@functools.cache
def _loop_native() -> Any:
    """The loop extension, verified once per process: ``load_loop_native``
    checks the native build identity on every call (0.2 s in a source
    checkout), and a generation's export renames a few hundred stages."""

    from ..native import load_loop_native

    return load_loop_native()


def _canonical_stage(stage: Any) -> tuple[list[Any], list[Any], list[Any]]:
    """A stage's outputs, parameters and function definitions on canonical
    symbols (``_canonical_symbols``).

    The stage's own symbols (model parameters, momentum and value slots, the
    model functions and their arguments) got their ids when the process
    history first needed them, so the same stage generated after other
    generations in one interpreter normalised its sums in another order and
    its evaluator rounded differently in binary64 (DESIGN WP2.5). Renamed
    positionally (parameters in input order, then the function heads without
    symbol attributes and the function arguments in definition order), the
    stage normalises and compiles identically in every process. The renaming
    is exact: values do not change beyond rounding."""

    from symbolica import E, Replacement

    from pyamplicol._internal.physics.symbols import symbols

    parameters = list(stage.parameter_symbols)
    definitions = list(stage.symbolica_functions)
    if not definitions and parameters:
        # Positional stage inputs (``ParamBuilder.add_parameter_list``) map to
        # the canonical symbols of the same index: swap the namespace in the
        # id-independent canonical string and parse it back, linear in the
        # stage's size (a replacement per input is quadratic: 100 s for the
        # 5763 inputs of a g g > g g g stage).
        canonical = _canonical_symbols(len(parameters))
        source = parameters[0].to_canonical_string()[:-1]
        target = canonical[0].to_canonical_string()[:-1]
        if all(
            symbol.to_canonical_string() == f"{source}{index}"
            for index, symbol in enumerate(parameters)
        ):
            return (
                [
                    E(expression.to_canonical_string().replace(source, target))
                    for expression in stage.output_expressions
                ],
                canonical,
                [],
            )
    heads = [
        function
        for function, _arguments, _body in definitions
        if not function.get_attributes()
    ]
    arguments = list(
        dict.fromkeys(
            argument for _function, items, _body in definitions for argument in items
        )
    )
    old = [*parameters, *heads, *arguments]
    renamed = dict(zip(old, _canonical_symbols(len(old)), strict=True))
    if TEXT_CANONICAL_STAGE:
        head_set = set(heads)
        variable_map = {k: v for k, v in renamed.items() if k not in head_set}
        head_map = {k: v for k, v in renamed.items() if k in head_set}
        bodies = _renamed_text(
            [body for _head, _items, body in definitions], variable_map, head_map
        )
        return (
            _renamed_text(stage.output_expressions, variable_map, head_map),
            [renamed[parameter] for parameter in parameters],
            [
                (
                    renamed.get(head, head),
                    tuple(renamed[argument] for argument in items),
                    body,
                )
                for (head, items, _body), body in zip(definitions, bodies, strict=True)
            ],
        )
    tail = symbols.qualified("pyamplicol_v2_canonical::args___")
    # Two passes: a single ``replace_multiple`` does not descend into a call
    # its head rule replaced, so the arguments of a model-function call kept
    # their old symbols (the evaluator then refused them).
    symbols = [
        Replacement(symbol, renamed[symbol]) for symbol in (*parameters, *arguments)
    ]
    calls = [Replacement(head(tail), renamed[head](tail)) for head in heads]

    def rename(expression: Any) -> Any:
        expression = expression.replace_multiple(symbols)
        # Nor into the arguments of a call another rule replaced: repeat
        # until nested calls (``value_3(0, 0, value_4(0))``) are renamed too.
        while calls:
            renamed_calls = expression.replace_multiple(calls)
            if renamed_calls == expression:
                break
            expression = renamed_calls
        return expression

    return (
        [rename(expression) for expression in stage.output_expressions],
        [renamed[parameter] for parameter in parameters],
        [
            (
                renamed.get(head, head),
                tuple(renamed[argument] for argument in items),
                rename(body),
            )
            for head, items, body in definitions
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
# The certified-scaleless companion compresses its seeds with the slice's
# certificates (``LoopSlice._companion_seeds``); False builds its own table
# (its generic seeds: the oracle). The extended table has no native build
# stage: a native LC/NLC prune of the companion needs one (next run).
SHARED_COMPANION_CERTIFICATES = True


def _field_level_table(data, process, selection, request, cap=None) -> ModelBlockTable:
    """The field-level block table of a loop request (design §3.2 item 1),
    within the loop orders' ``cap`` unless another is given."""

    names = {int(entry["pdg_code"]): entry["name"] for entry in data["particles"]}
    incoming = [names[leg.pdg] for leg in process.legs if leg.is_initial]
    outgoing = [names[leg.pdg] for leg in process.legs if leg.is_final]
    return ModelBlockTable(
        data,
        incoming,
        outgoing,
        cap=selection.cap if cap is None else cap,
        ignored_particle_ids=vetoed_particle_ids(request.particle_veto),
    )


def _block_order_bounds(request) -> dict[str, int]:
    """Coupling-order bounds of the request's tree blocks: they serve the
    loops and the Born closure at leg 1, so an order the loops bound takes
    the Born's bound where that is larger (u u~ > d d~ with QCD loops of the
    QCD Born only still closes its photon Born on photon currents)."""

    born = request.born_orders
    return {
        name: max(int(power), int(born.get(name, power)))
        for name, power in request.loop_order_bounds.items()
    }


def _born_table(data, process, selection, request, table) -> ModelBlockTable:
    """The field-level table of the Born closure's currents: ``table`` (the
    loop orders' cap) when that covers the Born orders, else one within the
    Born's own (a Born order beyond the loop cap, e.g. the electroweak Born
    of a request for QCD loops only, interferes with the QCD loops). An
    order the request does not bound is bounded by its largest power in a
    tree of the n legs (``_tree_order_bounds``)."""

    names = sorted(
        {str(name) for entry in data["couplings"] for name, _power in entry["orders"]}
    )
    tree = _tree_order_bounds(
        data, len(process.legs), [n for n in names if n not in request.born_orders]
    )
    born = {
        name: int(request.born_orders.get(name, tree.get(name, 0)))
        for name in names
        if request.born_orders.get(name, tree.get(name, 0))
    }
    loop = dict(selection.cap)
    if all(power <= loop.get(name, 0) for name, power in born.items()):
        return table
    cap = {name: max(power, born.get(name, 0)) for name, power in loop.items()}
    cap.update((name, power) for name, power in born.items() if name not in loop)
    return _field_level_table(
        data, process, selection, request, cap=tuple(sorted(cap.items()))
    )


def _tree_order_bounds(data, legs: int, names: Sequence[str]) -> dict[str, int]:
    """The largest power of each coupling order of ``names`` in a tree with
    ``legs`` external legs, from the model's vertices: a tree's vertices have
    sum(arity - 2) = legs - 2, so an order's power is at most legs - 2 times
    its largest power per arity - 2 over the vertices (legs - 2 for
    renormalisable vertices; more for effective ones, e.g. a g g H vertex of
    QCD = 2). A two-point vertex carrying one of the orders leaves its power
    unbounded and is refused."""

    wanted = set(names)
    orders = {entry["name"]: entry["orders"] for entry in data["couplings"]}
    ratio: dict[str, Fraction] = {}
    for rule in data["vertex_rules"]:
        arity = len(rule["particles"])
        for row in rule["couplings"]:
            for coupling in row:
                for name, power in orders[coupling] if coupling else ():
                    if name not in wanted or int(power) <= 0:
                        continue
                    if arity <= 2:
                        raise NotImplementedError(
                            f"vertex {rule['name']} has {arity} legs and coupling "
                            f"order {name}={power}: its tree-level power is unbounded"
                        )
                    ratio[name] = max(
                        ratio.get(name, Fraction(0)), Fraction(int(power), arity - 2)
                    )
    return {name: int((legs - 2) * value) for name, value in ratio.items()}


def _field_level_families(
    data, process, catalogue, selection, request, native: Any, table=None
) -> FamilySet:
    """The families of the field-level block table (design §3.2 item 1);
    ``native`` is the loop core, or None for the Python recursion."""

    if table is None:
        table = _field_level_table(data, process, selection, request)
    if native is not None:
        return NativeGeneration(table, catalogue, selection, native).families()
    return generate_families(table, catalogue, selection)


def _born_block_requests(
    table: ModelBlockTable, data, born_orders: Mapping[str, int]
) -> frozenset[BlockCurrent]:
    """Block currents the Born closure at leg 1 attaches (``_born_terms``), at
    field level: loop-aware reachability builds only the currents loop
    positions consume, and the Born root of a process whose leg 1 is a lepton
    (e+ e- > u u~: the quark pair's photon current) or of a mixed-order Born
    (the W exchange of u u~ > d d~) attaches currents no loop position does.
    A vertex of the model attaches leg 1 alone and one current of the
    vertex's fields per part of the other legs, within ``born_orders`` (an
    order it does not name is unbounded, as for the tree program).
    """

    full = (1 << len(table.legs)) - 1
    bounds = {str(name): int(power) for name, power in born_orders.items()}

    def within(orders: Orders) -> bool:
        return all(power <= bounds.get(name, power) for name, power in orders)

    spins = {entry["name"]: int(entry["spin"]) for entry in data["particles"]}
    coupling_orders = {
        entry["name"]: tuple(sorted((name, power) for name, power in entry["orders"]))
        for entry in data["couplings"]
    }
    by_arity: dict[int, list[tuple[tuple[str, ...], set[Orders]]]] = defaultdict(list)
    for rule in data["vertex_rules"]:
        fields = tuple(rule["particles"])
        if any(spins[name] == -1 for name in fields):
            continue
        orders = {
            add_orders(coupling_orders[name])
            for row in rule["couplings"]
            for name in row
            if name is not None
        }
        by_arity[len(fields)].append((fields, orders))
    by_field: dict[int, dict[str, list[BlockCurrent]]] = {}

    def currents(mask: int) -> dict[str, list[BlockCurrent]]:
        found = by_field.get(mask)
        if found is None:
            found = by_field[mask] = defaultdict(list)
            for block in table.currents(mask):
                found[block.field].append(block)
        return found

    out: set[BlockCurrent] = set()
    for arity, rules in sorted(by_arity.items()):
        for rest in _set_partitions(full & ~1, arity - 1, arity - 1):
            pools = [currents(part) for part in (1, *rest)]
            for fields, orders in rules:
                for arrangement in set(permutations(fields)):
                    options = [
                        pool.get(name, ())
                        for pool, name in zip(pools, arrangement, strict=True)
                    ]
                    if not all(options):
                        continue
                    for choice in product(*options):
                        if any(
                            within(
                                add_orders(order, *(block.orders for block in choice))
                            )
                            for order in orders
                        ):
                            out.update(choice)
    return frozenset(out)


# Schedule-B frame policy of the slice: the cheapest frame by undressed rows,
# a segment priced by its legs' helicity rows (``generator_input.leg_rows``;
# ``NativeGeneration.schedules(leg_rows=...)`` runs the same policy natively).
# No count per leg is assumed: the rows are the request's.
_FRAME_MODEL = STRUCTURED


def _frame_policy(rows: Sequence[int]) -> Any:
    return cheapest(generator_input.frame_rows(rows), _FRAME_MODEL)


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


def _site_coefficient(compiled: Any, fixed: Mapping[str, Any], continuation: Any):
    """``(rule, entry, power) -> exact Laurent coefficient`` of a CT entry in
    the payload's symbols (``ct_recursion.CoefficientLowering``, built on
    first use)."""

    lowering: list[Any] = []

    def coefficient(_rule: Any, entry: Any, power: int) -> Any:
        if not lowering:
            from .ct_recursion import CoefficientLowering, _payload_identities

            lowering.append(
                CoefficientLowering(
                    compiled,
                    fixed,
                    None if continuation is None else _payload_identities(continuation),
                )
            )
        return lowering[0](entry, power)

    return coefficient


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
    prediction of the virtual's (single, double) poles per sector (the QCD
    and QED operators, massless and massive legs) for the sectors whose
    loops the requested corrections complete (``poles.predicted_sectors``;
    the others are absent), ``None`` in finite-only and ``"uv"`` mode, at
    leading colour and where the operators make no prediction (a correction
    without one, or no predicted sector); ``stability`` is ``None``
    unless checks were asked. ``colour_contraction`` is the seed contraction
    used ("direct" or "symmetric-group-fft").

    ``poles`` is the evaluation's pole selection (``PoleSelection``). In
    ``"uv"`` and ``"ir"`` the components, totals and families carry the
    selected pole parts (``poles.select_sector``: in ``"uv"`` the virtual's
    poles are the bare loop's UV poles, the unrenormalised amplitude's), and
    ``pole_origins[origin][component][sector]`` gives both origins' parts of
    every renormalised component (``Laurent`` with the combined finite part;
    ``pole_origins["uv"]["virtual"]`` is the renormalised virtual's UV
    part), ``None`` otherwise.

    ``loop_born`` exists with loop-side sources (the asymmetric Ward mode,
    ``ward``): per squared Born order Re(A0^dagger G A0'), the physical Born
    bra against the Born amplitude read on the loop side (where a gluon's
    polarisation is c p_j it vanishes with the tree Ward identity); ``born``
    and ``totals[0]`` stay the physical Born. ``None`` otherwise.
    """

    born: Mapping[Orders, Any]
    components: Mapping[str, Mapping[Orders, Laurent]]
    totals: tuple[Any, ...]
    families: Mapping[tuple[FamilyKey, Orders, Orders], Laurent]
    ir_poles: Mapping[Orders, tuple[Any, Any]] | None
    stability: StabilityEstimate | None = None
    colour_contraction: str = "direct"
    poles: PoleSelection = PoleSelection.ALL
    pole_origins: Mapping[str, Mapping[str, Mapping[Orders, Laurent]]] | None = None
    loop_born: Mapping[Orders, Any] | None = None

    def virtual(self, sector: Orders) -> Laurent:
        return self.components["virtual"][sector]


# --------------------------------------------------------------------- slice


@dataclass(frozen=True)
class SliceRequest:
    """A loop request of the slice: a process, the bounds of its Born coupling
    orders and the exact coupling orders of its loop amplitude, or a sequence
    of several (mixed squared-order sectors, e.g. QCD corrections to QCD and
    electroweak Born currents)."""

    process: str
    born_orders: Mapping[str, int]
    loop_orders: Mapping[str, int] | Sequence[Mapping[str, int]]
    # The correction orders. None (the default): the model's strong
    # correction, identified by the gluon's vertices (``ew_policy.
    # request_corrections``; refused for a model without one), resolved by
    # ``resolve_request`` when the slice is generated. The public routes name
    # them (``[process] loop_coupling_orders``).
    corrections: tuple[str, ...] | None = None
    fixed_parameters: Mapping[str, str] = field(default_factory=dict)
    # Vetoed particles (|PDG| codes, ``generation.particle_veto``): removed
    # from every tree vertex of every generation step (Born, blocks, loop
    # vertices and propagators, the counterterm program), and every
    # counterterm term whose loop content holds one is dropped.
    particle_veto: tuple[int, ...] = ()
    # [U2]: a Levi-Civita counterterm cell the veto keeps from cancelling
    # that cancels without it (an anomalous veto, e.g. one breaking a quark
    # doublet) refuses the request (``ct_recursion.AnomalousVetoError``)
    # unless this override lowers it (``ct_recursion``, "Anomalous vetoes";
    # the artifact records it).
    allow_anomalous_veto: bool = False
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
    # FFT on certified orbits in binary64 at NLC and full colour (other
    # carriers, exact dressing and LC contract directly), "direct" never uses
    # it. "fft" requires it: it refuses LC and a table without such orbits at
    # generation, and an evaluation in any other carrier when it dresses.
    colour_contraction: ContractionPolicy = "auto"
    # An explicit structural source-compatibility profile of the model
    # preparation (``loop.preparation.compatibility``; e.g. "gmu_particle_anti"
    # restores the Goldstone flag MG5_aMC's loop_qcd_qed_sm loses on G-).
    compatibility: str | None = None
    # Counterterms (design §6.1, §6.6): "recursion" takes them from the tree
    # recursion over the CT-lowered model (``ct_closure``), "sites" from the
    # site closure of every rule, the permanent oracle (O6).
    counterterms: Literal["recursion", "sites"] = "recursion"
    # A leading-colour selection (MQ-SELECT, ``selection.Selection``): the
    # Born helicities of some legs and the colour flows kept, to which the
    # generation prunes its recursion; None (the default) keeps every
    # helicity and flow (the request record then has no entry).
    selection: Any = None
    # The on-shell external photons that couple with alpha(0) (MG5_aMC's
    # ``!a!`` tags; m-sm/PLAN.md Q1 (a), ``ew_policy.photon_tags``): "final"
    # (the default: final-state photons, isolated photons; initial ones are
    # PDF photons and keep the loops' coupling), "all", "none", or the
    # 1-based labels of the tagged legs (initial legs allowed). Only a request
    # that takes the electromagnetic correction converts its tagged photons;
    # the request record names it only when it is not the default.
    external_photons: str | tuple[int, ...] = "final"
    # The model's names of the tagged photons' conversion counterterm
    # (delta Z_e of the loops' scheme - delta Z_e^alpha(0)) and of its
    # adjustable 1/alpha(0) (genericity audit 2 G2-m5; ``[loop]
    # alpha0_conversion_parameter``, ``alpha0_inverse_parameter``): MG5_aMC's
    # ``eCoup_DGmua0_UV_EW`` and the unique adjustable SLHA SMINPUTS 1
    # parameter by default. A named 1/alpha(0) whose inverse is the loops'
    # alpha declares loops that couple with alpha(0): tagged photons then
    # take neither a rescaling nor a conversion. Recorded only when set.
    alpha0_conversion: str = ew_policy.ALPHA0_CONVERSION
    alpha0_inverse: str | None = None

    def __post_init__(self) -> None:
        if self.selection is not None and self.accuracy != "lc":
            raise ValueError(
                "helicity and colour-flow selection is a leading-colour option "
                "(accuracy='lc')"
            )
        object.__setattr__(
            self,
            "external_photons",
            ew_policy.photon_tag_option(self.external_photons),
        )
        if not isinstance(self.alpha0_conversion, str) or not self.alpha0_conversion:
            raise ValueError("alpha0_conversion names a counterterm parameter")
        if self.alpha0_inverse is not None and (
            not isinstance(self.alpha0_inverse, str) or not self.alpha0_inverse
        ):
            raise ValueError("alpha0_inverse names a model parameter or is None")
        veto = tuple(sorted({abs(int(pdg)) for pdg in self.particle_veto}))
        if 0 in veto:
            raise ValueError("particle_veto takes particle PDG codes")
        object.__setattr__(self, "particle_veto", veto)
        if not isinstance(self.allow_anomalous_veto, bool):
            raise ValueError("allow_anomalous_veto must be a boolean")

    @property
    def exact_loop_orders(self) -> tuple[Mapping[str, int], ...]:
        """The exact coupling orders of every admitted loop amplitude."""

        if isinstance(self.loop_orders, Mapping):
            return (self.loop_orders,)
        return tuple(self.loop_orders)

    @property
    def loop_order_bounds(self) -> dict[str, int]:
        """Per coupling the largest power of any loop amplitude."""

        bounds: dict[str, int] = {}
        for orders in self.exact_loop_orders:
            for name, power in orders.items():
                bounds[name] = max(bounds.get(name, 0), power)
        return bounds


def resolve_request(compiled: Any, request: SliceRequest) -> SliceRequest:
    """``request`` with its correction orders resolved
    (``ew_policy.request_corrections``) and, for LC and NLC, its colour
    declaration: the model's own (``resolve_loop_color_declaration``: the
    source's declaration, else the shipped one of its counterterm vertices,
    ``models.loop_color_declarations``) when the request names none; its
    ``particle_veto`` with the ghosts of the vector bosons it removes
    (``ew_policy.gauge_closed_veto``). Unchanged when it names both and its
    veto is so closed."""

    if request.corrections is None:
        request = replace(
            request, corrections=ew_policy.request_corrections(compiled, request)
        )
    if request.particle_veto:
        # A vetoed vector boson takes its Faddeev-Popov ghosts with it.
        veto = ew_policy.gauge_closed_veto(compiled, request.particle_veto)
        if veto != request.particle_veto:
            request = replace(request, particle_veto=veto)
    raw = getattr(compiled, "raw_loop_model", None)
    if (
        request.accuracy != "full"
        and request.colour_declaration is None
        and raw is not None
    ):
        from pyamplicol.models.loop_color_declaration import (
            resolve_loop_color_declaration,
        )

        declaration = resolve_loop_color_declaration(raw)
        if declaration is not None:
            request = replace(request, colour_declaration=declaration)
    return request


def check_model_parameters(
    parameters: Mapping[str, str],
    externals: Container[str] | None,
    scale_parameter: str | None,
    fixed: Mapping[str, Any],
    prepared: Container[str] = (),
) -> dict[str, str]:
    """The caller's real model parameter values, validated as a whole before
    any is applied (the Python API's and the SDK's one rule, genericity audit
    2 M2; ``rusticol_loop_reduction::v2_runtime::checked_parameters`` is its
    native twin), by name (so a mapping with several refused entries is
    refused with the same message in every client): an unknown name (not an
    external parameter of the model, ``externals``; None only for a stored
    process of an older build, whose record lacks its model's names), the
    renormalisation scale's parameter (each point's ``mu_r2`` sets it) and a
    value other than the one a parameter was fixed to at generation
    (``fixed``, ``LoopSlice.fixed_parameters``: ``[loop.preparation]
    fixed_parameters`` and those the model preparation fixes, ``prepared``,
    which the refusal names as such) are refused with a ``ValueError``. A
    known parameter the process does not read is accepted (it has no
    effect). An empty name is the Python API reader's ``TypeError``; every
    value is read first, as the Python API reads it (``runtime.virtual.
    _number``: a finite decimal number of ASCII text)."""

    from pyamplicol.runtime.virtual import _number

    if any(not isinstance(name, str) or not name for name in parameters):
        # The Python API's reader's refusal (``runtime.virtual._parameters``).
        raise TypeError("model parameter names must be nonempty strings")
    names = sorted(parameters)
    for name in names:
        _number(parameters[name], f"parameter {name!r} real part")
    for name in names:
        value = parameters[name]
        if externals is not None and name not in externals:
            raise ValueError(f"unknown external parameter {name!r}")
        if name == scale_parameter:
            raise ValueError(
                f"{name!r} is the renormalisation scale: pass it per point as mu_r2"
            )
        if name in fixed and Decimal(str(value)) != Decimal(str(fixed[name])):
            raise ValueError(fixed_refusal(name, fixed[name], name in prepared))
    return dict(parameters)


def fixed_refusal(name: str, value: Any, prepared: bool) -> str:
    """The refusal of a value other than the one ``name`` was fixed to at
    generation, naming who fixed it: the request (``[loop.preparation]
    fixed_parameters``) or the model preparation (``prepared``: a scheme
    constant, e.g. a colour declaration's, which no card entry sets);
    ``v2_runtime::fixed_refusal`` is its native twin."""

    if prepared:
        return (
            f"parameter {name!r} was fixed to {value} at generation by the model "
            "preparation (a scheme constant, e.g. of the colour declaration); "
            "it is not a run-time parameter"
        )
    return (
        f"parameter {name!r} was fixed to {value} at generation "
        "([loop.preparation] fixed_parameters); regenerate the artifact to "
        "change it"
    )


class RequestModel(NamedTuple):
    """The model preparation of one request (``prepare_request_model``)."""

    prepared: Any
    # The declared source-colour continuation, or None without a declaration.
    continuation: Any
    # The request's fixed parameters with the source-colour ones.
    fixed_parameters: dict[str, Any]
    scale_parameter: str
    alpha_s_parameter: str | None


def prepare_request_model(compiled: Any, request: SliceRequest) -> RequestModel:
    """The model preparation of ``LoopSlice.generate``, before any tree,
    family or loop generation: the source-colour fixed parameters (LC and
    NLC need a colour declaration), the scale and alpha_s parameters,
    ``prepare_loop_request_model``, with a colour declaration
    ``prepare_declared_source_color``, the gauge refusal of an electroweak
    request (``ew_policy.require_feynman_gauge``) and the listed refusal of a
    veto of loop species on unlabelled counterterms
    (``refusals.require_labelled_counterterms_for_veto``). The public dry run
    (``Generator.plan``) runs it alone to report its refusals."""

    from pyamplicol.loops.compiler import prepare_loop_request_model

    request = resolve_request(compiled, request)
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
    # alpha_s of the QCD operator of the IR-pole prediction (Catani, WP3.2
    # EWI-6), one rule for every correction set (genericity audit m2): the
    # named parameter (``loop.alpha_s_parameter``), which must be an
    # adjustable external parameter, else the unique adjustable SLHA
    # SMINPUTS 3 one, else none: the operator then makes no prediction (the
    # virtual itself does not read it).
    alpha_s_parameter = None
    # The strong correction is identified by the gluon's vertices, not by
    # its order's name (``ew_policy.correction_orders``; genericity m1).
    strong = ew_policy.correction_orders(compiled, fixed).strong
    if strong is not None and strong in request.corrections:
        try:
            alpha_s_parameter = _external_parameter(
                compiled, request.alpha_s_parameter, _LHA_ALPHA_S, fixed
            )
        except ValueError:
            if request.alpha_s_parameter is not None:
                raise
    prepared = prepare_loop_request_model(
        compiled,
        born_order_bounds=dict(request.born_orders),
        loop_order_bounds=request.loop_order_bounds,
        fixed_parameters=fixed,
        compatibility=request.compatibility,
    )
    continuation = None
    if declaration is not None:
        from pyamplicol.loops.source_color_declaration import (
            prepare_declared_source_color,
        )

        prepared, continuation = prepare_declared_source_color(
            prepared, declaration, scheme=declaration.scheme
        )
    if ew_policy.electroweak(
        compiled, request, prepared.structure.perturbative_species_orders
    ):
        # The gauge scope (WP3.2 EWI-4b), refused here so that the dry run
        # reports it (a source without gauge 1 is refused by the loop
        # preparation already, a bundle prepared in unitary gauge here);
        # ``LoopSlice.generate`` switches the gauge.
        ew_policy.require_feynman_gauge(compiled, request)
    # A veto reaches the counterterms through their loop-content labels
    # only: refused where the model's counterterm terms of a requested
    # correction are not all labelled (``refusals``, listed; WP3.2 EWI-5
    # interim, EWI-7's [U2] guard), here so that the dry run reports it.
    require_labelled_counterterms_for_veto(
        prepared.model.kernels,
        request.process,
        request.particle_veto,
        request.corrections,
        prepared.structure.perturbative_species_orders,
    )
    return RequestModel(
        prepared, continuation, fixed, scale_parameter, alpha_s_parameter
    )


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
    # The CT amplitudes from the tree recursion (``ct_closure``); the sites
    # are then empty (site closure is the "sites" test oracle only).
    counterterm_program: Any = None
    # Whether LC/NLC pruning dropped skeletons (``_prune_colour``).
    colour_pruned: bool = False
    # The native generator stages of this process (WP3.1), if they ran.
    generation: NativeGeneration | None = None
    # Site -> cells left out of its closure because they cancel between the
    # rules closed on the same blocks (``SiteClosure.cancelled_cells``).
    cancelled_cells: Mapping[Any, Any] = field(default_factory=dict)
    # The certified-scaleless families as a slice of their own (same blocks,
    # Born, counterterms and degrees): dressed only by the "uv" and "ir" pole
    # selections, whose bare UV and IR parts they carry (design §12.4).
    scaleless: LoopSlice | None = None
    # The alpha(0) convention of the tagged on-shell external photons of an
    # electroweak request (WP3.2 EWI-6b, per leg since MS-PHOTON,
    # ``ew_policy.ExternalPhotons``; None without a tagged photon): the
    # point's (alpha(0) / alpha)^n scales every output
    # (``_PointEvaluation.average``), the conversion is a field insertion.
    external_photons: ew_policy.ExternalPhotons | None = None
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def fixed_parameters(self) -> dict[str, str]:
        """The external parameters fixed at generation, with their exact
        values: the request's (``SliceRequest.fixed_parameters``, as given)
        and the model's external parameters its preparation fixes besides
        (a colour declaration's external scheme constants,
        ``source_color_fixed_parameters``), from the set the scalar plans and
        the counterterm program were built with (``prepared.structure.
        fixed_parameters``; the names an applied source restriction removed
        from the model are not parameters a caller can set). Every runtime
        refuses a caller's other value of any of them and evaluates at these
        (``check_model_parameters``; the native record's
        ``fixed_parameters``; genericity audit 2, M2)."""

        found = self._cache.get("fixed-parameters")
        if found is None:
            externals = self._external_parameters()
            found = dict(self.request.fixed_parameters)
            prepared = self.prepared.structure.fixed_parameters or {}
            for name, value in prepared.items():
                if name in found or name not in externals:
                    continue
                real, imaginary = (value, "0") if isinstance(value, str) else value
                if Decimal(imaginary) != 0:
                    raise ValueError(
                        f"parameter {name!r} is fixed to a complex value; v2 "
                        "loop runtimes take real external parameters"
                    )
                found[name] = real
            self._cache["fixed-parameters"] = found
        return dict(found)

    @property
    def preparation_fixed_parameters(self) -> frozenset[str]:
        """The names in ``fixed_parameters`` the model preparation fixed (a
        colour declaration's scheme constants), not the request."""

        return frozenset(self.fixed_parameters) - frozenset(
            self.request.fixed_parameters
        )

    @property
    def pole_regulators(self) -> PoleRegulators | None:
        """The model's adjustable pole regulators (``poles.pole_regulators``:
        epsUV and epsIR of MadLoop's electroweak UFOs), which split the
        counterterms' poles by origin in the "uv" and "ir" selections; None
        for a model without them (loop_sm) or with one of them fixed."""

        key = "pole-regulators"
        if key not in self._cache:
            self._cache[key] = pole_regulators(
                self.compiled.ir.parameters,
                self.fixed_parameters,
                getattr(self.compiled, "raw_loop_model", None),
            )
        return self._cache[key]

    def _external_parameters(self) -> frozenset[str]:
        """The names of the model's external parameters (those a caller
        sets)."""

        found = self._cache.get("external-parameters")
        if found is None:
            found = self._cache["external-parameters"] = frozenset(
                record.name
                for record in self.compiled.ir.parameters
                if record.nature == "external"
            )
        return found

    @property
    def site_terms(self) -> Mapping[Any, tuple[SiteTerm | WholeRuleTerm, ...]]:
        """The terms of every counterterm site (the slice's counterterm form,
        design §6.6), built on first use: generation and the plan need none."""

        found = self._cache.get("site-terms")
        if found is None:
            found = self._cache["site-terms"] = {
                site: self._terms_of(site) for site in self.sites.sites
            }
        return found

    def _terms_of(self, site: Any) -> tuple[SiteTerm | WholeRuleTerm, ...]:
        found = self._cache.get("site-terms")
        if found is not None:
            # A slice whose terms were set (a copy without some sites, in
            # tests) has none for the sites left out.
            return found.get(site, ())
        cache = self._cache.setdefault("terms-of", {})
        terms = cache.get(site)
        if terms is None:
            if site.rule.whole_rule_continuation is None:
                terms = self.closure.terms(site, self.cancelled_cells.get(site, ()))
                _plans, layout = self._scalar_plan(LAURENT_POWERS)
                # A coupling the scalar plan does not compute would make its
                # term vanish without an error (the Python and native
                # evaluations read the layout).
                missing = sorted(
                    {
                        term.entry.coupling.name
                        for term in terms
                        for power in LAURENT_POWERS
                        if (term.entry.coupling.name, power) not in layout
                    }
                )
                if missing:
                    raise ValueError(
                        f"counterterm site {site.rule_name} binds couplings "
                        f"{missing} that the slice's scalar plan does not compute"
                    )
            else:
                terms = _whole_rule_terms(self.closure, site)
            cache[site] = terms
        return terms

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
    def generate(
        cls,
        compiled: Any,
        request: SliceRequest,
        native: Any,
        *,
        prune: bool = True,
        native_inputs: bool = False,
    ) -> LoopSlice:
        """Build every generation object of ``request`` (no diagram is listed).

        ``compiled`` is the compiled loop model and ``native`` the staged loop
        core (``loops.native.load_loop_native``). ``prune=False`` defers the
        LC/NLC colour pruning to the caller (``_prune_colour`` with the seed
        tables of further observables). ``native_inputs`` tells that the
        caller exports the slice's native point inputs (a stored plan v4
        process, ``artifact.generate_process``): a listed feature their export
        does not cover (``native_trees.NATIVE_EXPORT_REFUSALS``) is then refused
        (``NativeExportUnsupported``) as soon as it is known, before the
        trees are built.

        Listed loop-admission refusals (``refusals.LOOP_ADMISSION_REFUSALS``,
        typed ``LoopRequestUnsupported``) are raised before the structures they
        concern are used: a veto of a loop species on a model whose
        counterterms of the correction are not all labelled with their loop
        content in the model preparation (``prepare_request_model``, which
        the dry run runs too), a vertex joining fermions of different sign
        ownership before any tree, LC/NLC over tree blocks that are not exact
        in N_c right after the trees.
        """

        from pyamplicol.loops.wavefunction import prepare_external_field_rules
        from pyamplicol.models.external import CompiledUFOModel
        from pyamplicol.processes.model import build_model_process_ir

        if request.accuracy not in ("full", "lc", "nlc"):
            raise ValueError(f"unknown colour accuracy {request.accuracy!r}")
        if request.counterterms not in ("recursion", "sites"):
            raise ValueError(f"unknown counterterm form {request.counterterms!r}")
        request = resolve_request(compiled, request)
        prepared, continuation, fixed, scale_parameter, alpha_s_parameter = (
            prepare_request_model(compiled, request)
        )
        declaration = request.colour_declaration
        structure = prepared.structure
        data = json.loads(structure.native.to_json())
        selection = LoopOrderSelection(
            frozenset(request.corrections),
            frozenset(add_orders(tuple(o.items())) for o in request.exact_loop_orders),
        )
        process = build_model_process_ir(
            request.process, compiled.ir, color_accuracy="full"
        )
        # Under the complex-mass scheme external states are stable (EW-25,
        # ``model_cms``): an external particle of a complex pole mass is
        # refused (``cms-external-width``).
        from pyamplicol.loops.model_cms import require_stable_external_states

        require_stable_external_states(
            compiled,
            (leg.pdg for leg in process.legs if leg.pdg is not None),
            repr(request.process),
        )
        # Fail-closed guards of electroweak requests (WP3.2 EWI-0b), before
        # any tree, family or counterterm generation (``ew_policy``).
        electroweak = ew_policy.electroweak(
            compiled, request, structure.perturbative_species_orders
        )
        photons = None
        if electroweak:
            # Tagged on-shell external photons couple with alpha(0) (WP3.2
            # EWI-6b, [U9]; per leg, MS-PHOTON: ``request.external_photons``);
            # a model that cannot take that convention is refused.
            photons = ew_policy.external_photon_scheme(
                compiled, request, process, fixed, structure.perturbative_species_orders
            )
            # One gauge throughout (DESIGN §6.8, WP3.2 EWI-4b): Born, blocks,
            # loop lines and counterterms of a request with electroweak loop
            # species take 't Hooft-Feynman gauge with explicit Goldstones
            # and ghosts; QCD-only requests keep the model's gauge.
            compiled = ew_policy.feynman_gauge_model(compiled, request)
        else:
            # A tag list naming a leg that is no on-shell photon is refused
            # for every request; without the electromagnetic correction the
            # tags convert nothing (as in MG5_aMC).
            ew_policy.photon_tags(compiled, request, process, fixed)
        # A fermion line that changes colour representation (a vertex joining
        # a coloured and a colour-singlet fermion that a tree or a loop of the
        # request can use) has no sign in the blocks' conventions: refused
        # before any tree (``refusals``, listed).
        require_colour_preserving_fermion_lines(
            compiled,
            process,
            order_limits=(dict(request.born_orders), _block_order_bounds(request)),
            ignored_particle_ids=vetoed_particle_ids(request.particle_veto),
            loop_species=_correction_species(structure, request),
        )
        # A vertex whose couplings vanish identically at the model's
        # restriction (a W vertex of a CKM element a diagonal-CKM restriction
        # sets to zero) contributes nothing and the restricted tree model
        # holds no colour or Lorentz term of it: it is no loop vertex
        # (fail-closed-2; decided from the coupling values,
        # ``ct_sites.restricted_to_zero``, the counterterm root closure's
        # rule too).
        loop_catalogue = ModelLoopCatalogue(
            data,
            species_orders=structure.perturbative_species_orders,
            veto=request.particle_veto,
            vanishing=vanishing_loop_vertices(
                compiled,
                prepared.model.kernels,
                (rule["name"] for rule in data["vertex_rules"]),
            ),
        )
        native_families = "families" in generator_input.NATIVE_STAGES
        model = compiled

        def tree_program(reachable):
            blocks = build_tree_blocks(
                process,
                CompiledUFOModel(model),
                born_coupling_orders=dict(request.born_orders),
                block_coupling_orders=_block_order_bounds(request),
                ignored_particle_ids=vetoed_particle_ids(request.particle_veto),
                reachability=(
                    None
                    if reachable is None
                    else requested_loop_blocks(
                        reachable.block_requests() | born_requests, data
                    )
                ),
                # LC and NLC project formal N_c: every block of one grade.
                u1_grades=request.accuracy != "full",
            )
            if electroweak:
                # Widths in the Born or the loop-side trees, before their
                # kernels (a width takes the full ones, 38 s at g g > t t~
                # on the G_mu model) and the counterterm lowering
                # (``ew_policy.require_zero_widths``).
                ew_policy.require_zero_widths(
                    model, request, (blocks.born_dag, blocks.blocks_dag), fixed
                )
            # Trees that propagate a coloured particle with a width (or a
            # model-supplied propagator) take the full tree kernels: the
            # transverse Yang-Mills ones are then not the Feynman rules
            # (design §3.2 item 2, §6.1). Everything below uses that model.
            # LC and NLC need N_c-exact tree blocks: refused before any family
            # or export (``refusals``, listed; every block set the slice uses
            # passes here).
            require_nc_exact(blocks, request.accuracy, request.process)
            blocks, kernels = with_ward_kernels(blocks, model, fixed)
            if request.selection is not None and request.selection.helicities:
                # A helicity selection keeps the selected rows of every block
                # (``selection.restrict_rows``): Born rows, frames, dressing
                # and the tree program follow.
                from .selection import restrict_rows

                blocks = restrict_rows(blocks, request.selection.helicities)
            table = blocks.block_table(data)
            if native_families:
                generation = NativeGeneration(table, loop_catalogue, selection, native)
                return blocks, kernels, table, generation, generation.families()
            return (
                blocks,
                kernels,
                table,
                None,
                generate_families(table, loop_catalogue, selection),
            )

        reachable = None
        born_requests: frozenset[BlockCurrent] = frozenset()
        if LOOP_BLOCK_REACHABILITY and request.counterterms == "recursion":
            # Loop-aware reachability (design §3.2 item 1): the tree program
            # builds only the block currents the admitted positions of the
            # field-level families consume. Site closure attaches counterterm
            # vertices to the tree currents of every leg partition, which
            # loop positions need not consume (a two-point counterterm on an
            # electroweak line), so the site route builds every current.
            field_table = _field_level_table(data, process, selection, request)
            reachable = _field_level_families(
                data,
                process,
                loop_catalogue,
                selection,
                request,
                native if native_families else None,
                table=field_table,
            )
            born_requests = _born_block_requests(
                _born_table(data, process, selection, request, field_table),
                data,
                request.born_orders,
            )
        blocks, compiled, table, generation, families = tree_program(reachable)
        if reachable is not None and (
            families.families != reachable.families
            or families.scaleless != reachable.scaleless
        ):
            # The field-level table does not predict this process's tree
            # blocks (e.g. electroweak currents the tree compiler does not
            # build split closed-loop flavour classes, ``families.
            # _closed_loop_classes``): every block current is built.
            blocks, compiled, table, generation, families = tree_program(None)

        def schedules():
            colour = SeedColour(blocks, data, compiled.ir.vertex_terms)
            rows = generator_input.leg_rows(blocks)
            if generation is not None and "schedules" in generator_input.NATIVE_STAGES:
                return colour, generation.schedules(
                    colour, model=_FRAME_MODEL, leg_rows=rows
                )
            return colour, build_schedules(
                families, loop_catalogue, colour, policy=_frame_policy(rows)
            )

        colour, plan = schedules()
        program = None
        if request.counterterms == "recursion":
            # Production form of the counterterms (design §6.1): the tree
            # recursion over the CT-lowered model with exactly one insertion,
            # closed at leg 1; graded by N_c with a colour declaration (design
            # §6.7). No site closure and no loop matching. The block rows
            # (``tree_rows``) are built on first use, after the CT-lowered
            # model: native trees canonicalise their stages
            # (``_canonical_stage``), and the Python replay is an oracle.
            from .ct_closure import CountertermProgram

            program = CountertermProgram.build(
                compiled,
                prepared,
                request.process,
                data,
                selection,
                blocks.born_orders,
                veto=request.particle_veto,
                allow_anomalous_veto=request.allow_anomalous_veto,
                fixed_parameters=fixed,
                declaration=declaration,
                continuation=continuation,
                # What determines ``prepared`` besides the arguments above.
                scope=(
                    tuple(sorted(request.born_orders.items())),
                    tuple(sorted(request.loop_order_bounds.items())),
                    request.compatibility,
                ),
                u1_grades=request.accuracy != "full",
            )
            # Its root terms close on its blocks with their N_c grades too.
            require_nc_exact(program.blocks.blocks, request.accuracy, request.process)
        sites = (
            # The O6 oracle: every rule closed at sites, witness admission.
            generate_counterterm_sites(
                families,
                table,
                prepared.counterterms.rules,
                data,
                selection,
                forbidden=request.particle_veto,
                whole_rules=continuation is not None,
            )
            if program is None
            else CountertermSites((), {}, (), 0)
        )
        field_rules = prepare_external_field_rules(prepared.counterterms.source)
        source_field_rules = field_rules
        alpha0_rules: tuple[Any, ...] = ()
        if photons is not None and photons.conversion is not None:
            # alpha(0)'s field-like term on each tagged photon (EWI-6b).
            alpha0_rules = _alpha0_field_rules(
                prepared.counterterms.source, photons, process
            )
            field_rules = (*field_rules, *alpha0_rules)
        insertions = external_field_insertions(
            table.legs,
            field_rules,
            data,
            selection,
            blocks.born_orders,
            forbidden=request.particle_veto,
        )
        if alpha0_rules:
            # The conversion is per tagged leg, not per species: an untagged
            # photon of the same species takes none (MS-PHOTON).
            ours = {id(rule) for rule in alpha0_rules}
            tagged = set(photons.legs)
            insertions = tuple(
                insertion
                for insertion in insertions
                if id(insertion.rule) not in ours or insertion.leg + 1 in tagged
            )
        # LC and NLC: every source field factor an external leg takes and
        # every counterterm rule a site closes needs declared N_c data (the
        # recursion's rules are checked in ``counterterm_program_blocks``;
        # ``refusals``, listed; MQ-NC). The alpha(0) photon terms are the
        # engine's own, not source factors. Full colour (with or without a
        # declaration) reads the source couplings at N_c = 3: no check.
        projected = continuation if request.accuracy != "full" else None
        require_declared_colour_fields(
            projected,
            (i for i in insertions if i.rule in source_field_rules),
            request.process,
        )
        if program is None:
            closed = {id(site.rule) for site in sites.sites}
            require_declared_colour_rules(
                projected,
                {
                    name: rule
                    for name, rule in prepared.counterterms.rules.items()
                    if id(rule) in closed
                },
                None,
                request.process,
            )
        ordinary = _ordinary_entries(prepared, data)
        root_sites = (*sites.sites, *(() if program is None else program.root_sites))
        # The root closure reads a vertex not closed under conjugation (a W
        # coupling) through its conjugate rule (WP3.2 EWI-2).
        conjugates = ConjugateVertices(
            (
                *((rows[0].vertex_name, rows) for rows in ordinary),
                *(
                    (name, rule.entries)
                    for name, rule in prepared.counterterms.rules.items()
                ),
            ),
            data,
        )
        entries = [e for site in root_sites for e in site.rule.entries]
        entries += _conjugate_site_entries(root_sites, conjugates, program)
        entries += [e for rows in ordinary for e in rows]
        # Root vertices read on a Goldstone-vector mixing line's Goldstone
        # side (WP3.2 EWI-7): their promoted entries.
        from .ct_closure import promoted_root_entries

        entries += promoted_root_entries(
            None if program is None else program.blocks,
            ordinary,
            lambda rows: rows if conjugates.closed(rows) else conjugates.of(rows),
            data,
        )
        _check_declared_loop_species(
            root_sites, _eligible_loop_species(loop_catalogue, selection, data)
        )
        catalogue = build_loop_kernel_catalogue(
            compiled,
            loop_species=_family_loop_species(families, data),
            counterterm_entries=list({id(e): e for e in entries}.values()),
        )
        closure = SiteClosure(blocks, data, catalogue, conjugates)
        # Oracle sites: cells without a site kernel that cancel between the
        # rules closed on the same blocks (the Levi-Civita R2 of a Z with
        # gluons over complete quark doublets) are left out; any other such
        # cell fails below.
        cancelled = closure.cancelled_cells(
            sites.sites,
            _site_coefficient(compiled, structure.fixed_parameters, continuation),
            lambda expression: vanishes_identically(compiled, expression),
        )
        readings = rx.reading_skeletons(plan)
        weights = LeafWeights(data, closure.fermion_ends)
        reading = TreeReading(data)
        rules = set()
        for skeleton in readings:
            weights.fermion_sign(skeleton)  # refuses fermion lines it does not cover
            rules.update(
                term.vertex.rule
                for content in skeleton.contents
                for term in content.terms
            )
        for rule in sorted(rules):
            reading.conjugate(rule)  # refuses vertices it cannot read
        if program is not None:
            # The program's root terms and the Born close the same vertices
            # on blocks of the same words: one colour cache serves both.
            program.bind(ordinary, data, catalogue, colours=closure)
        born_terms = _born_terms(blocks, table, closure, ordinary)
        if missing := set(blocks.born_orders) - set(born_terms):
            # The Born program has roots there, so their virtual sectors would
            # be silently zero. Seen for a mixed-order Born whose electroweak
            # block currents no admitted loop position consumes (loop-aware
            # reachability does not build them); without reachability its
            # closure refuses the W and Z vertices.
            raise NotImplementedError(
                f"the Born root closure at leg 1 has no terms at the Born "
                f"orders {sorted(missing)} of the tree program"
            )
        inclusive_born = born_terms
        if request.selection is not None and request.selection.flows is not None:
            # A leading-colour selection keeps its flows' Born tensors only
            # (``selection.restrict_born_terms``): the basis, the seed table,
            # the colour pruning and the Born and counterterm bras follow.
            from .selection import restrict_born_terms

            born_terms = restrict_born_terms(
                born_terms, closure.legs, request.selection.flows
            )
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
            reading=reading,
            weights=weights,
            scale_parameter=scale_parameter,
            alpha_s_parameter=alpha_s_parameter,
            loop_catalogue=loop_catalogue,
            generation=generation,
            counterterm_program=program,
            cancelled_cells=cancelled,
            external_photons=photons,
        )
        if born_terms is not inclusive_born:
            slice_._refuse_inexact_flows(inclusive_born)
        if request.accuracy != "full" and prune:
            slice_._prune_colour()
        slice_._plan_contraction()
        slice_.scaleless = slice_._scaleless_companion()
        return slice_

    def _scaleless_companion(self, prune: bool | None = None) -> LoopSlice | None:
        """The certified-scaleless families (``FamilySet.scaleless_families``)
        as a slice sharing this one's blocks, Born, counterterms, kernels and
        colour degrees, with its own schedules and seed table (contracted
        directly), or None without such families. At LC/NLC it is pruned for
        the inclusive virtual unless ``prune`` is False (the observables'
        companion, ``ObservableSlice.scaleless``, prunes it for its declared
        histories).

        Their integrals vanish (scaleless), so the combined Laurent never needs
        them; their UV residues, e.g. B0(0;0,0) = 1/eps_UV - 1/eps_IR, are part
        of the bare loop's UV and IR poles (``poles``).
        """

        families = self.families
        # A massless tadpole (N = 1) has neither a UV nor an IR pole: after
        # the shift q -> q - V its integrand has no term of degree -4.
        kept = {
            key: family
            for key, family in families.scaleless_families.items()
            if len(key) >= 2
        }
        if not kept:
            return None
        twin_families = replace(
            families,
            families=kept,
            scaleless=(),
            rejected=(),
            scaleless_families={},
        )
        generation = None
        if self.generation is not None and "schedules" in generator_input.NATIVE_STAGES:
            # The native stages schedule the companion's subset of their
            # certified-scaleless families (and prune it at LC/NLC below).
            generation = self.generation.scaleless_companion(twin_families)
            plan = generation.schedules(
                self.colour,
                model=_FRAME_MODEL,
                leg_rows=generator_input.leg_rows(self.blocks),
            )
        else:
            plan = build_schedules(
                twin_families,
                self.loop_catalogue,
                self.colour,
                policy=_frame_policy(generator_input.leg_rows(self.blocks)),
            )
        readings = rx.reading_skeletons(plan)
        for skeleton in readings:
            self.weights.fermion_sign(skeleton)
        table, class_of = self._companion_seeds(readings)
        twin = replace(
            self,
            families=twin_families,
            plan=plan,
            table=table,
            class_of=class_of,
            fourier=None,
            colour_pruned=False,
            scaleless=None,
            generation=generation,
        )
        if self.request.accuracy != "full" and prune is not False:
            twin._prune_colour()
        return twin

    def _companion_seeds(
        self, readings: Sequence[Skeleton]
    ) -> tuple[SeedTable, dict[Skeleton, SeedCombination]]:
        """The seed table of the certified-scaleless companion's readings.

        The companion shares this slice's colour (blocks, vertices, Born
        basis), so the compressions this slice's table certified
        (``SeedTable.certificates``: a certificate holds for every skeleton
        of its kind in the process, by its theorem, ``COMPRESSION_THEOREMS``)
        compress the companion's readings too (``extend_seed_table`` on an
        empty table with those certificates): the companion's bubbles with a
        quartic partner are adjoint rings. On its own (``build_seed_table``)
        the companion has no single-legged instance to certify them from
        five legs on (one of its two blocks spans at least two legs), so
        every reading keeps its generic seed, an exact colour-network
        reduction each (g g > g g g g: 8,640 readings, 1,080 classes against
        60; with native-seeds' in-crate reductions 1.44 s against 0.32 s,
        companion plan 10.1 MB against 8.2 MB; before them 60 s, and about
        30 min of the g g > g g g g g generation). With
        ``SHARED_COMPANION_CERTIFICATES`` off it builds its own table (the
        oracle).

        The extended table records no native build stage, so a native LC/NLC
        prune (native-seeds' ``prune_switch.patch``, next run) must route the
        companion's prune through the Python callback or record a build
        source on the extend path (design §12.4, lean companion).
        """

        compress = self.request.compress_colour
        if not SHARED_COMPANION_CERTIFICATES or not compress:
            return build_seed_table(
                self.colour, readings, self.basis, compress=compress
            )
        empty = SeedTable(
            self.basis,
            (),
            (),
            (),
            self.colour.formal_nc,
            certificates=dict(self.table.certificates),
        )
        return extend_seed_table(empty, {}, self.colour, readings, compress=True)

    def _plan_contraction(self) -> None:
        """Certify the seed FFT's orbits per ``colour_contraction``."""

        policy = self.request.colour_contraction
        if policy not in ("auto", "direct", "fft"):
            raise ValueError(f"unknown colour contraction {policy!r}")
        if policy == "direct":
            return
        from .refusals import LoopRequestUnsupported

        if self.request.accuracy == "lc":
            if policy == "fft":
                raise LoopRequestUnsupported(
                    "forced-seed-fft-unserved",
                    f"{self.request.process!r}: leading-colour seeds are "
                    "contracted directly",
                )
            return
        self.fourier = seed_fourier(self.table)
        if self.fourier is None and policy == "fft":
            raise LoopRequestUnsupported(
                "forced-seed-fft-unserved",
                f"{self.request.process!r}: the seed table has no certified "
                "symmetric-group orbit pair",
            )

    def _prune_colour(
        self, observables: Iterable[tuple[SeedTable, Mapping[Orders, Any]]] = ()
    ) -> None:
        """LC and NLC: drop, before any numerics, every skeleton whose seed
        cells all lie below the N_c powers the projection keeps (design §7.5,
        §7.6). Structural: exact seed cells, Born grades and the sector
        degrees P; the projection itself is unchanged. ``observables`` adds
        the seed tables of further outputs (directed colour histories) with
        their virtual sector degrees: a skeleton live in any table is kept.
        """

        accuracy = self.request.accuracy
        selection = self.request.selection
        # A colour-flow selection's Born basis is its flows' tensors
        # (``selection.restrict_born_terms``): pruned ring-aware below.
        ring_aware = selection is not None and selection.flows is not None
        grades = self._born_grades()
        criteria = [(self.table, self.degrees()["virtual"]), *observables]
        live: dict[tuple[int, Orders], bool] = {}

        def class_live(c: int, order: Orders) -> bool:
            found = live.get((c, order))
            if found is None:
                found = live[c, order] = any(
                    value and kept(power + grade, degrees[sector], accuracy)
                    for table, degrees in criteria
                    for sector, sector_pairs in self.sectors
                    for a, loop in sector_pairs
                    if loop == order
                    for b, born_orders in enumerate(self.basis.orders)
                    if a in born_orders
                    for power, value in table.cells[c][b].items()
                    for grade in grades.get((b, a), ())
                )
            return found

        def own(skeleton) -> bool:
            return any(
                class_live(c, skeleton.orders) for c, _x in self.class_of[skeleton]
            )

        keep = own
        if ring_aware:
            # The selected flows' basis elements only: a skeleton is kept
            # when its ring (``SkeletonView`` readings, rotations and the
            # reflection) is live in the plan's reading of that ring, whose
            # seed it dresses, or in its own.
            view = SkeletonView(self.loop_catalogue, self.colour)
            rings: dict[Any, bool] = {}
            for reading in rx.reading_skeletons(self.plan):
                ring = view.view(reading).canonical
                rings[ring] = rings.get(ring, False) or own(reading)

            def keep(skeleton) -> bool:
                return rings.get(view.view(skeleton).canonical, False) or own(skeleton)

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
        self.colour_pruned = True

    # -- evaluation ------------------------------------------------------------

    def evaluate(
        self,
        point: Sequence[Sequence[Any]],
        scale_squared: Any,
        *,
        parameters: Mapping[str, str] | None = None,
        precision: Precision | None = None,
        poles: PoleSelection | str = PoleSelection.ALL,
        stability: StabilityChecks | None = None,
    ) -> PointResult:
        """The complete virtual at one exact point (decimal strings or rationals).

        ``point`` lists the physical momenta (E, px, py, pz) of the legs in
        process order, incoming momenta as given; ``scale_squared`` is mu_R^2.
        ``parameters`` overrides external model parameters (decimal strings).
        ``poles`` is a ``PoleSelection`` or its value: ``"all"`` (default),
        ``"none"`` (finite only), ``"uv"`` or ``"ir"`` (``poles``, design
        §12.4).
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

    def _point(
        self, point, scale, parameters, precision, poles, **sources
    ) -> _PointEvaluation:
        values = self._point_parameters(parameters, scale, precision)
        return _PointEvaluation(
            self, point, scale, values, _Numbers(precision), poles, **sources
        )

    def _point_parameters(
        self, parameters: Mapping[str, str], scale: Fraction, precision: Precision
    ) -> dict[str, str]:
        """A point's parameter values: the caller's (``check_model_parameters``:
        unknown names, the scale parameter and a fixed parameter at another
        value are refused, as by the Python API's and the SDK's runtimes), the
        fixed parameters (``fixed_parameters``: the request's and those the
        model preparation fixes) and the renormalisation scale (the root of
        ``scale``)."""

        fixed = self.fixed_parameters
        values = check_model_parameters(
            parameters,
            self._external_parameters(),
            self.scale_parameter,
            fixed,
            self.preparation_fixed_parameters,
        )
        values.update(fixed)
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
                PoleSelection.ALL,
            )
            execution = evaluation.dress(schedule, exact=exact)
        return execution, tuple(tuple(p) for p in evaluation.incoming)

    # -- scalar bindings -------------------------------------------------------

    def _scalar_plan(self, powers: tuple[int, ...]):
        """The scalar plans of ``scalars`` and their (name, power) -> ((grade,
        column), ...) layout over the concatenated columns; only structurally
        nonzero grades are kept, and every output has an entry (none for a
        structurally zero one), so a coupling the layout lacks is no output.

        The Born couplings have a program of their own: the common
        subexpressions of a scalar program, and so the binary64 rounding of
        its outputs, depend on all of them, and the Born (which also weights
        the bare loop) must not depend on which counterterm or field couplings
        a request binds.
        """

        from .fields import field_definitions

        cached = self._cache.get(("plan", powers))
        if cached is None:
            definitions = field_definitions(self.fields)
            born = {
                (term.entry.coupling.name, 0)
                for terms in self.born_terms.values()
                for term in terms
            }
            # The entries the site terms close: a vertex not closed under
            # conjugation is read through its conjugate rule, whose couplings
            # its terms bind (``SiteClosure.read_entries``).
            outputs = {
                (entry.coupling.name, power)
                for site in self.sites.sites
                if site.rule.whole_rule_continuation is None
                for entry in self.closure.read_entries(site.rule.entries)
                for power in powers
            }
            outputs.update(
                (name, power) for name, _ in definitions.values() for power in powers
            )
            if self.counterterm_program is not None:
                for term in self.counterterm_program.closure.terms:
                    if term.coupling is None:
                        outputs.update((name, 0) for name in term.term.parameters)
                    elif term.coupling[1] in powers:
                        outputs.add(term.coupling)
            # Only whole-rule sites (LC/NLC continuations) have whole-rule
            # terms; the other site terms are not needed here. A term's
            # parameters do not depend on the leg labels, so one site per
            # relabelling orbit of identical legs is closed.
            orbits = _AdjointOrbits(self.basis.legs, self.blocks.process, ())
            closed = set()
            for site in self.sites.sites:
                identity = orbits.site_key(site)
                if site.rule.whole_rule_continuation is None or identity in closed:
                    continue
                closed.add(identity)
                outputs.update(
                    (name, 0)
                    for term in self._terms_of(site)
                    if isinstance(term, WholeRuleTerm)
                    for name in term.parameters
                )
            plans: list[Any] = []
            layout: dict[tuple[str, int], tuple[tuple[int, int], ...]] = {}
            offset = 0
            for part in (born, outputs - born):
                if not part:
                    continue
                plan, local = self._scalar_part(sorted(part), definitions)
                plans.append(plan)
                for output, rows in local.items():
                    layout[output] = tuple((g, c + offset) for g, c in rows)
                offset += len(plan.expressions)
            cached = self._cache[("plan", powers)] = (tuple(plans), layout)

        return cached

    def _scalar_part(self, outputs, definitions):
        """One scalar program over ``outputs`` and its local layout."""

        from pyamplicol.loops.scalar_bindings import prepare_scalar_bindings

        structure = self.prepared.structure
        options = {
            "program": structure.scalars,
            "fixed_parameters": structure.fixed_parameters,
            "anonymous_definitions": {
                name: source for name, source in definitions.values()
            },
        }
        if self.continuation is None:
            plan = prepare_scalar_bindings(structure.raw, outputs, **options)
            return plan, {output: ((0, i),) for i, output in enumerate(plan.outputs)}
        graded = self.continuation.scalars(outputs, **options)
        layout = defaultdict(list)
        for degree, columns in zip(graded.grades, graded.columns, strict=True):
            for output, column in zip(graded.formal.outputs, columns, strict=True):
                if graded.coefficients.expressions[column].value != 0:
                    layout[output].append((degree, column))
        return graded.coefficients, {
            output: tuple(layout.get(output, ())) for output in outputs
        }

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

        # One plan for both modes: the scalar program's common subexpressions,
        # and so the rounding of the finite couplings, do not depend on
        # whether the poles are evaluated (finite-only parts stay bitwise
        # equal to the full ones).
        plans, layout = self._scalar_plan(LAURENT_POWERS)
        powers = LAURENT_POWERS if poles else (0,)
        # binary64 couplings are the correctly rounded values (32 digits,
        # rounded once): they do not depend on the scalar program's
        # evaluation order.
        digits = (
            _COUPLING_DIGITS if numbers.double else numbers.precision.working_digits
        )
        values = []
        for plan in plans:
            state = NativeScalarBindings(
                plan,
                arithmetic="arbitrary",
                precision=digits,
                backend=self.native,
                parameters={
                    name: value
                    for name, value in parameters.items()
                    if name in plan.external_names
                },
            )
            values.extend(
                numbers.pair(Decimal(re), Decimal(im))
                for re, im in state.values.to_decimal()
            )
        return {
            output: {degree: values[column] for degree, column in rows}
            for output, rows in layout.items()
            if output[1] in powers
        }

    # -- colour degrees -------------------------------------------------------

    def overlaps(self, left: Any, right: Any) -> Mapping[int, ExactColorCoefficient]:
        """<left|right> of two literal colour tensors, exact in N_c."""

        cache = self._cache.setdefault("overlaps", {})
        found = cache.get((left, right))
        if found is None:
            legs = self.basis.legs
            cells = contract_colour(_single(left, legs), _single(right, legs))
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

    def _flow_degrees(
        self, born_terms: Mapping[Orders, tuple[Any, ...]]
    ) -> dict[tuple[Any, Any, Orders], int]:
        """The leading N_c degree of the squared Born's term between two
        colour flows ``x`` and ``y`` (``selection._item_flow`` keys) per
        squared Born order ``s``, keyed ``(x, y, s)``, for the Born terms
        ``born_terms`` (e.g. the unrestricted ones of a colour-flow
        selection): the slice's grading of the Born (``_born_grades``: an
        item's N_c power plus its coupling's grades, so the N_c grade split
        of identical-flavour lines is included) and the exact overlaps of the
        flows' tensors. Structural, as ``degrees``. Within one Born order the
        cross term of two distinct flows closes fewer index loops than a
        flow's own term, so it is below the mean of their own degrees, hence
        below the leading degree, and only a flow's own term (``x == y``) is
        graded there; between two Born orders (QCD and electroweak Born
        amplitudes) a cross term can be leading (u u~ > d d~: the QCD flow
        (2,4,3,1) with the photon's (2,1,3,4) at QCD^2 QED^2), so every pair
        of flows is graded there."""

        from .fields import field_definitions
        from .selection import _item_flow

        _plan, layout = self._scalar_plan(LAURENT_POWERS)
        missing = {
            (term.entry.coupling.name, 0)
            for terms in born_terms.values()
            for term in terms
        } - set(layout)
        if missing:
            # Couplings only the unselected flows' Born terms bind.
            _part, extra = self._scalar_part(
                sorted(missing), field_definitions(self.fields)
            )
            layout = {**layout, **extra}
        keys: dict[Orders, dict[Any, set[tuple[Any, int]]]] = defaultdict(
            lambda: defaultdict(set)
        )
        for order, terms in born_terms.items():
            for term in terms:
                grades = {g for g, _c in layout.get((term.entry.coupling.name, 0), ())}
                for item in term.colour.terms:
                    keys[order][_item_flow(item)].update(
                        (item.tensor, item.nc_power + g) for g in grades
                    )
        out: dict[tuple[Any, Any, Orders], int] = {}
        for a, b in product(keys, repeat=2):
            squared = add_orders(a, b)
            pairs = ((x, x) for x in keys[a]) if a == b else product(keys[a], keys[b])
            for x, y in pairs:
                for left, p in keys[a][x]:
                    for right, q in keys[b][y]:
                        for power, value in self.overlaps(left, right).items():
                            if value:
                                key = (x, y, squared)
                                degree = power + p + q
                                out[key] = max(out.get(key, degree), degree)
        return out

    def _refuse_inexact_flows(
        self, inclusive: Mapping[Orders, tuple[Any, ...]]
    ) -> None:
        """A colour-flow selection (``selection.restrict_born_terms``) is the
        evaluation-time selection of its flows only if, at every squared Born
        order of the unrestricted Born ``inclusive`` (``_flow_degrees``):

        * no selected flow's term with an unselected flow reaches the leading
          degree there (the restricted Born drops it; possible between two
          Born orders only): refused, ``interfering-colour-flows``;
        * the selected flows' terms reach the leading degree there, or none
          is nonzero (an exact zero there): else they would set a lower
          degree and return a subleading value, so a selection below the
          process's leading colour wherever it reaches it is refused,
          ``subleading-colour-flows``. Next to a leading flow a subleading
          one contributes exact zeros (the projection keeps the process's
          degree P, as an evaluation-time selection of it is exactly
          zero)."""

        from .colour import outer_flow
        from .refusals import LoopRequestUnsupported
        from .selection import _item_flow, leading_colour_plan, plan_flows

        degrees = self._flow_degrees(inclusive)
        kept = {
            _item_flow(item)
            for terms in self.born_terms.values()
            for term in terms
            for item in term.colour.terms
        }
        top: dict[Orders, int] = {}
        reached: dict[Orders, int] = {}
        for (x, y, squared), degree in degrees.items():
            top[squared] = max(top.get(squared, degree), degree)
            if x in kept and y in kept:
                reached[squared] = max(reached.get(squared, degree), degree)
        selected = [list(w) for w in self.request.selection.flows]
        below = [s for s, degree in sorted(reached.items()) if degree < top[s]]
        if below or not reached:
            orders = below or sorted(top)
            raise LoopRequestUnsupported(
                "subleading-colour-flows",
                f"{self.request.process!r}: the selected colour flows "
                f"{sorted(selected)} are below the process's leading colour at "
                f"the squared Born orders {[dict(s) for s in orders]}, and no "
                "selected flow reaches it there",
            )
        crossing = {
            (squared, y if x in kept else x)
            for (x, y, squared), degree in degrees.items()
            if (x in kept) != (y in kept) and degree >= top[squared]
        }
        if crossing:
            # The unselected flows by their tree words (a piece of no planar
            # flow by its strings and traces).
            representations = {
                leg.label: leg.representation for leg in self.closure.legs
            }
            plan = leading_colour_plan(self.compiled, self.request.process)
            words = {
                outer_flow(flow.word, representations): list(flow.word)
                for flow in plan_flows(plan, ())
            }
            named = sorted((s, str(words.get(f, f))) for s, f in crossing)
            raise LoopRequestUnsupported(
                "interfering-colour-flows",
                f"{self.request.process!r}: the selected colour flows "
                f"{sorted(selected)} interfere at the leading N_c power with the "
                "unselected flows "
                + ", ".join(f"{flow} at {dict(s)}" for s, flow in named),
            )

    def degrees(
        self,
        overlaps: Callable[[Any, Any], Mapping[int, Any]] | None = None,
        table: SeedTable | None = None,
    ) -> dict[str, dict[Orders, int | None]]:
        """Leading N_c degrees P per squared-order sector (design §7.4).

        ``"born"`` per squared Born order; ``"virtual"`` per virtual sector, the
        largest degree of a nonzero exact cell among the seed, counterterm and
        field cells (coupling grades included). Structural: no numerics.
        ``overlaps`` and ``table`` replace the Born colour metric and the seed
        table, e.g. by those of a directed colour history (observables).
        """

        inclusive = overlaps is None and table is None
        found = self._cache.get("degrees") if inclusive else None
        if found is not None:
            return found
        from . import native_cells  # WP3.1 C3 switch: the "degrees" stage

        if inclusive and (found := native_cells.slice_degrees(self)) is not None:
            self._cache["degrees"] = found
            return found
        overlaps = overlaps or self.overlaps
        table = table or self.table
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

        born = {
            order: frozenset(keys(terms, lambda t: grades(t.entry.coupling.name, (0,))))
            for order, terms in self.born_terms.items()
        }
        # The relabelling orbits apply to the inclusive metric only: a
        # directed colour history (observables) is not invariant under the
        # relabellings of identical adjoint legs.
        orbits = (
            _AdjointOrbits(self.basis.legs, self.blocks.process, born.values())
            if inclusive
            else None
        )
        leading: dict[tuple[frozenset, Hashable], int | None] = {}

        def overlap_degree(bra: frozenset, right: Any) -> int | None:
            return max(
                (
                    power + p
                    for left, p in bra
                    for power, value in overlaps(left, right).items()
                    if value
                ),
                default=None,
            )

        def degree(bra: frozenset, ket: set, offset: int = 0) -> int | None:
            # max over the bra of a ket tensor's overlap degrees is constant
            # on the ket tensor's orbit (the bra is invariant): one
            # representative per orbit is contracted.
            out = None
            for right, q in ket:
                if orbits is None:
                    found = overlap_degree(bra, right)
                else:
                    rename = orbits.rename(right)
                    key = (bra, _colour_key(right, rename))
                    if key not in leading:
                        image = _relabelled_colour(right, rename)
                        leading[key] = overlap_degree(bra, image)
                    found = leading[key]
                if found is not None:
                    out = _max(out, found + q + offset)
            return out

        # Sites in one orbit of the relabellings have relabelled terms, so
        # their keys reach the same degrees: one site per orbit is closed.
        counterterms: dict[Orders, set] = defaultdict(set)
        closed: set[Hashable] = set()
        for site in self.sites.sites:
            if orbits is not None:
                identity = orbits.site_key(site)
                if identity in closed:
                    continue
                closed.add(identity)
            counterterms[site.orders] |= keys(
                self._terms_of(site),
                lambda t: (
                    {d for d, _e in t.grades}
                    if isinstance(t, WholeRuleTerm)
                    else grades(t.entry.coupling.name)
                ),
            )
        if self.counterterm_program is not None:
            for item in self.counterterm_program.closure.terms:
                term = item.term
                if isinstance(term, WholeRuleTerm):
                    found = {d + item.grade for d, _e in term.grades}
                else:
                    found = {
                        g + item.grade
                        for g, _c in layout.get(item.coupling, ())
                        if item.coupling[1] in LAURENT_POWERS
                    }
                counterterms[item.orders] |= {
                    (part.tensor, part.nc_power + grade)
                    for part in term.colour.terms
                    for grade in found
                }
        out = {"born": {}, "virtual": {}}
        for a, b in product(born, repeat=2):
            squared = add_orders(a, b)
            value = degree(born[a], born[b])
            out["born"][squared] = _max(out["born"].get(squared), value)
        born_grades = self._born_grades()
        for sector, pairs in self.sectors:
            value = table.degree(pairs, born_grades)
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
        if inclusive:
            self._cache["degrees"] = out
        return out

    # -- colour correlations (IR-pole check) ------------------------------------

    def unpredicted_sectors(self) -> dict[Orders, str]:
        """The squared-order sectors without a Catani prediction and why
        (``poles.UNPREDICTED_REASONS``, ``poles.sector_predictions``):
        ``PointResult.ir_poles`` leaves them out, and they are checked
        against a provider per split order only."""

        return {
            sector: reason
            for sector, reason in _sector_predictions(self).items()
            if reason is not None
        }

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
        from . import dipoles  # WP3.1 C4 switch: the "dipoles" stage

        if (found := dipoles.slice_operators(self)) is not None:
            self._cache["dipoles"] = found
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
                                    contract_colour(
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


# Decimal digits of the binary64 couplings before their single rounding.
_COUPLING_DIGITS = 32
# A missing cache entry (``_PointEvaluation._closed_counterterms``).
_UNSET = object()


class _PointEvaluation:
    """All per-point work of ``LoopSlice.evaluate`` in one carrier."""

    def __init__(
        self,
        slice_,
        point,
        scale,
        parameters,
        numbers,
        poles,
        *,
        sources=None,
        loop_sources=None,
        fixed=None,
    ):
        """``sources`` replaces the wavefunctions of vector legs (1-based
        label -> literal complex four-vector) in the Born and the blocks,
        ``loop_sources`` in the blocks and counterterm currents only (the
        loop side, e.g. the asymmetric Ward mode), and ``fixed`` keeps one
        helicity row per listed leg (label -> helicity): the row that carries
        a source (``observables``, design §8.4)."""

        self.slice = slice_
        self.numbers = numbers
        # ``mode`` selects the pole parts (``poles``); ``poles`` says whether
        # any pole column is computed (every mode but finite-only).
        self.mode = pole_mode(poles)
        self.poles = has_poles(self.mode)
        # The reducer's split columns per lane in "uv" and "ir" mode.
        self.split_lanes: dict | None = None
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
        rows = slice_.tree_rows.evaluate(
            decimal_point, parameters, numbers, sources=sources
        )
        loop_rows = rows
        if loop_sources:
            loop_rows = slice_.tree_rows.evaluate(
                decimal_point,
                parameters,
                numbers,
                sources={**(sources or {}), **loop_sources},
            )
        scalars = slice_.scalars(parameters, numbers, self.poles)
        program = slice_.counterterm_program
        # CT amplitudes are loop side: they take the loop sources too. Their
        # pole slots are off only in finite-only mode (``self.poles``: every
        # selection but "none"; design §12.4 "Counterterm slots").
        self.counterterm_rows = (
            None
            if program is None
            else program.point(
                decimal_point,
                parameters,
                numbers,
                self.poles,
                sources={**(sources or {}), **(loop_sources or {})} or None,
            )
        )
        self.parameters = parameters
        self.scale = scale
        self.scalars = scalars
        self.fixed = dict(fixed or {})
        self._born_and_counterterms(rows, scalars, labels, loop_rows)
        self.rows = loop_rows
        # What a twin at other parameters needs (``counterterm_twin``).
        self._twin_inputs = (point, sources, loop_sources, fixed)
        self.counterterms_uv: dict | None = None

    def counterterm_twin(self) -> _PointEvaluation | None:
        """The point at the IR regulator 0 (``poles.regulated_parameters``)
        with pole slots, formed once: its counterterm and field components
        are this point's counterterms' UV parts (module docstring of
        ``poles``, "Models with pole regulators"); None for a model without
        pole regulators."""

        regulators = self.slice.pole_regulators
        if regulators is None:
            return None
        twin = self.__dict__.get("_counterterm_twin")
        if twin is None:
            point, sources, loop_sources, fixed = self._twin_inputs
            with self.numbers.scope():
                twin = self._counterterm_twin = _PointEvaluation(
                    self.slice,
                    point,
                    self.scale,
                    regulated_parameters(self.parameters, regulators),
                    self.numbers,
                    PoleSelection.ALL,
                    sources=sources,
                    loop_sources=loop_sources,
                    fixed=fixed,
                )
        return twin

    def _counterterms_uv(self) -> dict | None:
        """The UV parts of the counterterm and field components per
        (component, sector, power) (``counterterm_twin``), or None (no pole
        regulators: ``poles.split_sector``'s remainder definition)."""

        if self.counterterms_uv is None:
            twin = self.counterterm_twin()
            if twin is not None:
                self.counterterms_uv = twin.counterterms
        return self.counterterms_uv

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

    def _born_and_counterterms(self, rows, scalars, labels, loop_rows=None) -> None:
        """Born colour vectors per helicity row, their square and the
        counterterm and field components, projected per sector (design §7.4).

        Colour vectors keep each literal tensor's N_c power (plus the coupling
        grade), so leading colour keeps exactly the sector's degree P before
        N_c = 3 is substituted; full colour keeps every power.

        The Born (the bra) reads ``rows``; counterterm currents and the field
        insertions' Born amplitudes (the loop side, the ket) read
        ``loop_rows``. Per helicity row the amplitude-level vectors stay
        available to the observables (``born_vectors``, ``loop_vectors``,
        ``site_vectors``).
        """

        loop_rows = rows if loop_rows is None else loop_rows

        slice_, numbers = self.slice, self.numbers
        self.constants = rx.KernelConstants(
            _constant_parameters(
                slice_.compiled, self.parameters, numbers.precision.working_digits
            ),
            numbers.carrier,
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

        def block_rows(term, helicity, rows):
            values = []
            for block in term.blocks:
                want = tuple(sorted((leg, helicity[leg]) for leg in block.key.labels))
                row = next((r for r in block.rows if r.helicities == want), None)
                if row is None:
                    return None
                variant = "source" if block.is_source else "propagated"
                values.append(rows[(row.current_id, variant)])
            return values

        def kinematic(term, helicity, rows=rows) -> Any:
            values = block_rows(term, helicity, rows)
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
        # The physical bra against the loop side's Born (loop-side sources).
        loop_born = defaultdict(lambda: numbers.zero)
        counterterms = defaultdict(lambda: numbers.zero)
        self.amplitudes: dict[rx.Row, dict[tuple[int, Orders], Any]] = {}
        self.born_vectors: dict[rx.Row, dict[Orders, dict]] = {}
        self.loop_vectors: dict[rx.Row, dict[Orders, dict]] = {}
        self.site_vectors: dict[rx.Row, dict[tuple[str, int, Orders], dict]] = {}
        sectors = {pair: sector for sector, pairs in slice_.sectors for pair in pairs}

        def born_vectors(helicity, rows) -> dict[Orders, dict]:
            vectors: dict[Orders, dict] = {}
            projections.clear()
            for order, terms in slice_.born_terms.items():
                vector: dict = {}
                for term in terms:
                    value = kinematic(term, helicity, rows)
                    if value is None:
                        continue
                    for grade, coupling in scalars[term.entry.coupling.name, 0].items():
                        _add_graded(
                            vector, term.colour, coupling * value, grade, numbers
                        )
                vectors[order] = vector
            return vectors

        choices = [
            sorted(helicities[label])
            if label not in self.fixed
            else [self.fixed[label]]
            for label in labels
        ]
        for choice in product(*choices):
            helicity = dict(zip(labels, choice, strict=True))
            vectors = born_vectors(helicity, rows)
            row = tuple((label - 1, helicity[label]) for label in labels)
            self.born_vectors[row] = vectors
            self.loop_vectors[row] = (
                vectors if loop_rows is rows else born_vectors(helicity, loop_rows)
            )
            self.amplitudes[row] = self._amplitudes(vectors, index)
            for left, right in product(vectors, repeat=2):
                squared = add_orders(left, right)
                born[squared] += interfere(
                    vectors[left], vectors[right], degree_of("born", squared)
                )
                if loop_rows is not rows:
                    ket = self.loop_vectors[row][right]
                    loop_born[squared] += interfere(
                        vectors[left], ket, degree_of("born", squared)
                    )
            loops: dict[tuple[str, int, Orders], dict] = defaultdict(dict)
            for site, terms in slice_.site_terms.items():
                for term in terms:
                    value = kinematic(term, helicity, loop_rows)
                    if value is None:
                        continue
                    value = len(site.alternatives) * value
                    for power in powers:
                        if isinstance(term, WholeRuleTerm):
                            grades = whole[term] if power == 0 else {}
                        else:
                            grades = scalars[term.entry.coupling.name, power]
                        for grade, coupling in grades.items():
                            _add_graded(
                                loops[site.component, power, site.orders],
                                term.colour,
                                coupling * value,
                                grade,
                                numbers,
                            )
            if self.counterterm_rows is not None:
                self._closed_counterterms(loops, helicity, momenta, scalars, carrier)
            self.site_vectors[row] = dict(loops)
            for (component, power, loop_order), vector in loops.items():
                for order, amplitude in vectors.items():
                    sector = sectors.get((order, loop_order))
                    if sector is None:
                        continue
                    value = interfere(amplitude, vector, degree_of("virtual", sector))
                    counterterms[component, sector, power] += value + value.conjugate()
            for insertion in slice_.fields:
                name = slice_.field_names[insertion.identity]
                ket = self.loop_vectors[row].get(insertion.born_order)
                if ket is None:
                    continue
                for order, amplitude in vectors.items():
                    sector = sectors.get((order, insertion.orders))
                    if sector is None:
                        continue
                    for power in powers:
                        for grade, coupling in scalars[name, power].items():
                            degree = degree_of("virtual", sector)
                            # The field coupling's N_c grade joins the
                            # overlap's powers (projection and N_c = 3).
                            value = coupling * interfere(amplitude, ket, degree, grade)
                            counterterms[FIELD_COMPONENT, sector, power] += (
                                value + value.conjugate()
                            )
        average = self.average
        self.born = {order: value.real / average for order, value in born.items()}
        self.loop_born = (
            None
            if loop_rows is rows
            else {order: value.real / average for order, value in loop_born.items()}
        )
        self.counterterms = {
            key: value.real / average for key, value in counterterms.items()
        }

    def _closed_counterterms(self, loops, helicity, momenta, scalars, carrier) -> None:
        """CT amplitudes of the tree recursion at one helicity row, into
        ``loops`` (component, power, loop orders) -> colour vector
        (``ct_closure``: root terms on the CT program's blocks)."""

        program = self.slice.counterterm_program
        rows, scales = self.counterterm_rows
        numbers = self.numbers
        powers = LAURENT_POWERS if self.poles else (0,)
        closure = program.closure.closure
        whole = self.__dict__.setdefault("_closure_whole_rules", {})
        # Each block's rows by helicities (once per slice), and each block's
        # value at this helicity row (once per row; None: no such row).
        tables = self.slice._cache.setdefault("counterterm_block_rows", {})
        at_row: dict[int, Any] = {}
        for item in program.closure.terms:
            if item.power not in powers:
                continue
            values = []
            for block in item.term.blocks:
                value = at_row.get(block.id, _UNSET)
                if value is _UNSET:
                    table = tables.get(block.id)
                    if table is None:
                        table = tables[block.id] = {}
                        for r in block.rows:
                            table.setdefault(r.helicities, r)
                    row = table.get(
                        tuple(sorted((leg, helicity[leg]) for leg in block.key.labels))
                    )
                    value = at_row[block.id] = (
                        None
                        if row is None
                        else rows[
                            (
                                row.current_id,
                                "source" if block.is_source else "propagated",
                            )
                        ]
                    )
                if value is None:
                    break
                values.append(value)
            else:
                if isinstance(item.term, WholeRuleTerm):
                    if item.term not in whole:
                        whole[item.term] = item.term.coefficients(
                            {
                                name: scalars[name, 0][0]
                                for name in item.term.parameters
                            },
                            numbers,
                        )
                    value = item.multiplicity * item.term.kinematic(values, numbers)
                    for grade, coupling in whole[item.term].items():
                        _add_graded(
                            loops[item.component, item.power, item.orders],
                            item.term.colour,
                            coupling * value,
                            grade + item.grade,
                            numbers,
                        )
                    continue
                vectors = [
                    momenta[block.key.external_mask] for block in item.term.blocks
                ]
                value = item.term.sign * closure.value(
                    item.term, values, vectors, carrier
                )
                value = item.multiplicity * value
                if item.scale is not None:
                    value = scales[item.scale] * value
                for grade, coupling in scalars[item.coupling].items():
                    _add_graded(
                        loops[item.component, item.power, item.orders],
                        item.term.colour,
                        coupling * value,
                        grade + item.grade,
                        numbers,
                    )

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
        self,
        schedule: Literal["B", "F"] = "B",
        *,
        exact: bool = False,
        seeds: rx.SeedSource | None = None,
    ) -> rx.Execution:
        """Family numerators of one schedule (F in the B tries' frames).

        ``exact`` dresses the carrier's tree rows and seeds, read as exact
        rationals, in exact arithmetic (kernel constants to the carrier's
        digits): schedules then agree exactly (O4). ``seeds`` replaces the
        sector seeds by other observable columns (``observables``).
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
                _constant_parameters(
                    slice_.compiled, self.parameters, numbers.precision.working_digits
                ),
                carrier,
            )
            if exact
            else self.constants
        )
        degrees = (
            slice_.degrees()["virtual"] if slice_.request.accuracy != "full" else {}
        )
        own = seeds is None
        if own:
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
        self.colour_contraction = getattr(seeds, "algorithm", "direct")
        if (
            own
            and slice_.request.colour_contraction == "fft"
            and self.colour_contraction != SYMMETRIC_GROUP_FFT
        ):
            # An explicit "fft" requires the FFT: the exact dressing,
            # DoubleFloat and arbitrary precision contract the sector seeds
            # directly, so they refuse instead of falling back. Observable
            # columns (``seeds`` given) are contracted directly by design;
            # a forced FFT refuses them where they are declared or served
            # (``observables.FORCED_FFT_OBSERVABLES``), and the generation's
            # validation still runs them as its reference.
            raise ValueError(
                "colour contraction 'fft' contracts binary64 sector seeds only; "
                "exact, arbitrary-precision and observable seeds are contracted "
                "directly (request 'auto' or 'direct')"
            )
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

    def _reduce(
        self, numerators, retest: str | None, columns=None, scaleless=None
    ) -> None:
        """Reduce every nonzero (family, loop order, column) lane; lanes are
        labelled by ``columns[column]`` (by default the column's sector).
        ``retest`` re-reduces them in that frame (``stability.FRAMES``:
        "reversed" or "rescaled"). The "uv" and "ir" pole selections also
        dress and reduce the certified-scaleless families for their UV
        residues: the slice's companion (``LoopSlice.scaleless``) on the
        sector columns, or for observable ``columns`` ``scaleless`` =
        (companion slice or None, its seeds of an evaluation;
        ``observables.ColumnPlan``)."""
        slice_, numbers = self.slice, self.numbers
        split = splits(self.mode)
        lanes, groups = self._groups(numerators, 0, columns)
        twin_lanes: list = []
        companion, seeds = slice_.scaleless, None
        if split and columns is not None:
            if scaleless is None:
                raise ValueError(
                    "observable columns in the uv and ir pole selections need "
                    "their companion column plan"
                )
            companion, seeds = scaleless
        if split and companion is not None:
            twin = copy.copy(self)
            twin.slice = companion
            dressed = twin.dress(seeds=None if seeds is None else seeds(twin))
            twin_lanes, twin_groups = twin._groups(
                dressed.numerators, len(lanes), columns
            )
            groups = groups + twin_groups
        arithmetic = "double" if numbers.double else "arbitrary"
        digits = None if numbers.double else numbers.precision.digits
        reducer = slice_.native.V2Reducer(
            arithmetic, digits, "fast" if numbers.double else "generic"
        )
        direct, check = reducer.reduce(
            groups,
            poles=reducer_poles(self.mode),
            check=retest is not None,
            rescale=retest == "rescaled",
        )
        everything = lanes + twin_lanes

        def values(batches, chosen):
            return {
                lane: self._values(batch, lane)
                for lane, batch in zip(everything, batches, strict=True)
                if lane in chosen
            }

        main, extra = set(lanes), set(twin_lanes)
        self.set_lanes(
            values(direct, main),
            None if check is None else values(check, main),
            values(direct, extra) if extra else None,
            None if check is None or not extra else values(check, extra),
        )

    def _groups(self, numerators, offset: int, columns=None) -> tuple[list, list]:
        """Reduction lanes ``(family, loop order, column label)`` and their
        native groups, with batch row ids from ``offset``."""
        slice_, numbers = self.slice, self.numbers
        if columns is None:
            columns = [sector for sector, _pairs in slice_.sectors]
        native = slice_.native
        arithmetic = "double" if numbers.double else "arbitrary"
        digits = None if numbers.double else numbers.precision.digits
        lanes, groups = [], []
        working = numbers.precision.working_digits
        masses = _mass_values(
            slice_.compiled,
            self.parameters,
            {mass for key in numerators for _mask, mass in key[0]},
            working,
        )
        onshell = _leg_masses(slice_, self.parameters, working)
        full = (1 << len(onshell)) - 1

        def invariant(arc: int, a, b):
            # A single-leg arc is the leg's squared mass (on-shell external
            # invariants, design §11.1), exactly the value its propagators
            # take; any other arc is p(arc)^2 of the given momenta.
            leg = onshell_leg(arc, full)
            if leg is not None:
                return onshell[leg]
            return _minkowski([x - y for x, y in zip(a, b, strict=True)])

        def batch(values, lane: int):
            return native.NativeBatch.from_decimal(
                values, 1, len(values), arithmetic, digits, [lane]
            )

        def real(values):
            working = numbers.precision.working_digits
            return [[_decimal_string(x, working), "0"] for x in values]

        def complex_values(values):
            # A complex-mass scheme's squared pole masses are complex.
            working = numbers.precision.working_digits
            return [
                [_decimal_string(x.real, working), _decimal_string(x.imag, working)]
                if isinstance(x, rx.Complex)
                else [_decimal_string(x, working), "0"]
                for x in values
            ]

        for (family, loop_order), numerator in sorted(numerators.items()):
            sequence = numerator.sequence
            n = len(sequence)
            arcs = shifts(sequence)
            vectors = [leg_momentum(mask, self.incoming) for mask in arcs]
            matrix = [
                invariant(arcs[i] ^ arcs[j], vectors[i], vectors[j])
                for i in range(n)
                for j in range(n)
            ]
            for column, coefficients in enumerate(numerator.columns):
                if not any(coefficients):
                    continue
                lane = offset + len(lanes)
                lanes.append((family, loop_order, columns[column]))
                groups.append(
                    (
                        n,
                        batch(real(x for v in vectors for x in v), lane),
                        batch(
                            complex_values(masses[mass] for _mask, mass in sequence),
                            lane,
                        ),
                        batch(real([self.scale]), lane),
                        batch(real(matrix), lane),
                        batch([numbers.strings(c) for c in coefficients], lane),
                    )
                )
        return lanes, groups

    def _values(self, batch, lane) -> list[Any]:
        ((_row, status),) = batch.statuses
        if status != "success":
            raise ArithmeticError(f"reduction lane {lane} failed: {status}")
        return [
            self.numbers.pair(Decimal(re), Decimal(im))
            for re, im in batch.values.to_decimal()
        ]

    def set_lanes(
        self,
        direct: Mapping,
        check: Mapping | None,
        scaleless: Mapping | None = None,
        scaleless_check: Mapping | None = None,
    ) -> None:
        """Each lane's raw integrals in the evaluation's pole mode.

        In ``"uv"`` and ``"ir"`` mode the reducer returns ``[finite, 1/eps,
        1/eps^2, 1/eps_UV]`` per lane; ``families`` (and the re-test's) keep
        the selected parts and ``split_lanes`` the reducer's columns, from
        which ``_assemble`` takes the combined and UV parts of ``bare``.
        ``scaleless`` are the lanes of the certified-scaleless families
        (``LoopSlice.scaleless``): their combined integrals vanish exactly in
        dimensional regularisation, so only their UV residues are kept.
        """

        split = splits(self.mode)
        zero = self.numbers.zero

        def laurent(values) -> Laurent:
            if split:
                return Laurent(*select_lane(values, self.mode, zero))
            return Laurent(*values)

        def vanishing(lanes: Mapping) -> dict:
            return {lane: [zero, zero, zero, v[3]] for lane, v in lanes.items()}

        direct = dict(direct)
        check = None if check is None else dict(check)
        if split and scaleless:
            direct.update(vanishing(scaleless))
            if check is not None and scaleless_check is not None:
                check.update(vanishing(scaleless_check))
        self.split_lanes = direct if split else None
        self.families = {lane: laurent(values) for lane, values in direct.items()}
        self.retest_families = (
            None
            if check is None
            else {lane: laurent(values) for lane, values in check.items()}
        )

    # -- assembly --------------------------------------------------------------------

    def _bare(self, families) -> dict[Orders, Laurent]:
        numbers = self.numbers
        average = self.average
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
        split = self.split_lanes
        if split is None:
            bare = self._bare(self.families)
            origins = None
        else:
            # The combined Laurent of every lane, and its UV part (``poles``).
            bare = self._bare({lane: Laurent(*v[:3]) for lane, v in split.items()})
            bare_uv = self._bare(
                {lane: Laurent(v[0], v[3], numbers.zero) for lane, v in split.items()}
            )
            origins = {origin: {name: {} for name in COMPONENTS} for origin in ORIGINS}
            # A model with pole regulators: the counterterms' own UV parts.
            counterterms_uv = self._counterterms_uv()
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
            if origins is not None:
                ultraviolet = None
                if counterterms_uv is not None:
                    ultraviolet = {
                        name: Laurent(
                            *[
                                counterterms_uv.get((name, sector, power), zero)
                                for power in LAURENT_POWERS
                            ]
                        )
                        for name in ("r2", "uv_local", FIELD_COMPONENT)
                    }
                parts = split_sector(
                    values,
                    bare_uv.get(sector, Laurent(zero, zero, zero)),
                    FIELD_COMPONENT,
                    Laurent,
                    ultraviolet,
                )
                for origin, part in parts.items():
                    for name in COMPONENTS:
                        origins[origin][name][sector] = part[name]
                values = select_sector(parts, self.mode, Laurent)
            else:
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
        # Catani's operators are the full-colour pole prediction of QCD and
        # QED corrections (massless and stable massive coloured or charged
        # legs; ``_ir_poles``); in "uv" mode the renormalised virtual has no
        # IR pole to predict.
        checked = slice_.request.accuracy == "full" and self._stable_coloured_legs()
        ir_poles = (
            self._ir_poles() if self.poles and self.mode != "uv" and checked else None
        )
        self.result = PointResult(
            dict(self.born),
            components,
            tuple(totals),
            dict(self.families),
            ir_poles,
            colour_contraction=self.colour_contraction,
            poles=self.mode,
            pole_origins=origins,
            loop_born=self.loop_born,
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

        factor = 2 / (LOOP_FACTOR_DENOMINATOR * math.pi**2 * float(self.average))
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
        integral and every local component is rotation invariant.

        The differences are taken at the carrier's working digits: a
        Decimal negation outside that context rounds the subtrahend to the
        default 28 digits, which put a spurious 1e-28 relative floor under
        every double-double and arbitrary-precision rotation budget."""

        zero = Laurent(*[self.numbers.zero] * (3 if self.poles else 1))
        with self.numbers.scope():
            lanes = {
                lane: tuple(
                    a - b
                    for a, b in zip(
                        rotated.families.get(lane, zero).coefficients,
                        self.families.get(lane, zero).coefficients,
                        strict=True,
                    )
                )
                # Insertion order, not a set (the budget's summation order;
                # a set's follows PYTHONHASHSEED).
                for lane in dict.fromkeys([*self.families, *rotated.families])
            }
            local = [
                differences(
                    self.result.components[name], rotated.result.components[name]
                )
                for name in LOCAL_COMPONENTS
            ]
        return self._lane_budget(lanes, local)

    # -- normalisation -----------------------------------------------------------------

    @property
    def average(self) -> Any:
        """The divisor of every output of this point: the slice's spin and
        colour average and symmetry factor (``Normalisation.average``),
        divided by (alpha(0) / alpha)^n for n tagged on-shell external photons
        of an electroweak request (``LoopSlice.external_photons``, WP3.2
        EWI-6b; per leg, MS-PHOTON):
        the Born, the loop lanes, the counterterm and field components, the
        correlators of Catani's operator and the stability budgets all take
        each photon's coupling e(0) instead of the loops' e. Without such
        photons it is the slice's average itself (outputs unchanged)."""

        found = self.__dict__.get("_average")
        if found is None:
            found = self.__dict__["_average"] = self._scaled_average()
        return found

    def _scaled_average(self) -> Any:
        slice_ = self.slice
        average = slice_.normalisation.average
        photons = getattr(slice_, "external_photons", None)
        if photons is None or photons.conversion is None:
            # No tagged photon, or loops that couple with alpha(0) already.
            return average
        values = _kernel_parameters(slice_.compiled, self.parameters)
        inverse = values.get(photons.alpha0_inverse)
        if not inverse:
            raise ValueError(
                f"the on-shell external photons of {slice_.request.process!r} "
                f"couple with alpha(0) (WP3.2 EWI-6b): set {photons.alpha0_inverse} "
                "(1/alpha(0), e.g. 137.035999) among the parameters, not "
                f"{inverse}"
            )
        alpha = self._qed_alpha()
        if alpha is None:
            raise ValueError(
                "alpha of the photon couplings is not real and positive at this point"
            )
        numbers = self.numbers
        ratio = numbers.real(1 / inverse) / alpha
        return numbers.real(average) / ratio ** len(photons.legs)

    # -- IR poles ----------------------------------------------------------------------

    def _stable_coloured_legs(self) -> bool:
        """Whether every massive leg that a requested correction dresses has
        a zero width (coloured legs for QCD, charged legs for QED): Catani's
        operators describe stable (real-mass) emitters."""

        values = _kernel_parameters(self.slice.compiled, self.parameters)
        abelian = self._correction_orders().abelian
        charged = (
            _charged_fields(self.slice.compiled)
            if abelian is not None and abelian in self.slice.request.corrections
            else frozenset()
        )
        return all(
            leg.mass.width == ZERO or values.get(leg.mass.width, 0) == 0
            for leg in self.slice.families.legs
            if (abs(leg.colour) != 1 or leg.field in charged) and leg.mass.mass != ZERO
        )

    def _ir_poles(self) -> dict[Orders, tuple[Any, Any]] | None:
        """Catani's single and double poles of the renormalised virtual per
        sector, or None where the operators make no prediction.

        One operator per correction (WP3.2 EWI-6), each over the Born
        interferences of the amplitude orders a, b whose orders plus that
        correction's two powers are the sector (cross-order interferences
        included): QCD (``_qcd_poles``: alpha_s / 2 pi, colour-correlated
        <T_i.T_j>) and QED (``_qed_poles``: alpha / 2 pi, Q_i Q_j times the
        colour-summed Born). A sector's prediction is their sum
        (``poles.sector_sum``); a correction without an operator, or one of
        them making no prediction, gives none (never a partial sum).

        The strong and the electromagnetic corrections are recognised by the
        gluon's and the photon's vertices (``ew_policy.correction_orders``),
        not by their orders' names (genericity audit m1). Only the sectors
        whose loops the requested corrections complete are predicted
        (``poles.predicted_sectors``; ``_predicted_sectors``): the mixed
        QCD x EW sectors of a QCD-only request (MadLoop's loop selection
        leaves out the gluon-photon boxes) have no prediction and are absent
        from the mapping; None when no sector is predicted.
        """

        orders = self._correction_orders()
        operators = {}
        if orders.strong is not None:
            operators[orders.strong] = self._qcd_poles
        if orders.abelian is not None:
            operators.setdefault(orders.abelian, self._qed_poles)
        predictions = []
        for name in self.slice.request.corrections:
            operator = operators.get(name)
            prediction = None if operator is None else operator()
            if prediction is None:
                return None
            predictions.append(prediction)
        predicted = self._predicted_sectors()
        out = {
            sector: poles
            for sector, poles in sector_sum(predictions).items()
            if sector in predicted
        }
        return out or None

    def _correction_orders(self) -> ew_policy.CorrectionOrders:
        """The slice's strong and electromagnetic orders
        (``ew_policy.correction_orders``), found once per slice."""

        return _correction_orders(self.slice)

    def _predicted_sectors(self) -> frozenset[Orders]:
        """The sectors ``_ir_poles`` predicts (``poles.predicted_sectors``),
        found once per slice."""

        return _predicted_sectors(self.slice)

    def _qcd_poles(self) -> dict[Orders, tuple[Any, Any]] | None:
        """The QCD operator per sector (``poles.catani_poles``), or None
        where it makes no prediction.

        V_poles = (alpha_s / 2 pi) sum_i sum_{j != i} (1/C_i) (C_i/eps^2 +
        gamma_i/eps) (1 + eps log(mu^2 / |2 p_i.p_j|)) <T_i.T_j> for massless
        coloured legs, gamma_i by each leg's spin and representation
        (``poles.strong_gamma``; gamma_g = beta_0 over the request's
        eligible massless coloured species, ``_strong_beta0``); massive
        emitters take the massive form. The legs' masses are the model's
        runtime values (a symbolic mass bound to 0 is massless), not the p^2
        of the given (possibly rounded) momenta. No prediction for a leg or a
        loop species the operator does not cover (genericity audit m3).
        """

        slice_, numbers = self.slice, self.numbers
        strong = self._correction_orders().strong
        if slice_.alpha_s_parameter is None or strong is None:
            return None
        legs = {leg.label: leg for leg in slice_.basis.legs}
        coloured = sorted(
            label for label, leg in legs.items() if leg.representation != 1
        )
        values = _kernel_parameters(slice_.compiled, self.parameters)
        beta0 = _strong_beta0(slice_, strong, values)
        gammas = {}
        for leg in slice_.families.legs:
            label = leg.index + 1
            if label in coloured:
                gammas[label] = _strong_leg_gamma(slice_, leg, beta0)
        # Every leg's mass from the slice's one mass evaluator (an external
        # mass exactly, a derived one at the working digits; M2).
        mass_of = _model_masses(
            slice_.compiled, self.parameters, numbers.precision.working_digits
        )
        masses = {
            leg.index + 1: mass_of(leg.mass.mass) ** 2 for leg in slice_.families.legs
        }
        if any(
            colour_casimir(legs[j].representation) is None
            or (masses[j] == 0 and gammas[j] is None)
            for j in coloured
        ):
            return None
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
        alpha = numbers.real(values[slice_.alpha_s_parameter])
        prefactor = alpha / (2 * numbers.pi)
        mu2 = numbers.real(self.scale)
        correction = add_orders(((strong, 2),))
        out = {}
        for sector, _pairs in slice_.sectors:

            def correlated(i: int, j: int, sector=sector) -> Any:
                total = numbers.real(0)
                for vectors in physical:
                    for a, left in vectors.items():
                        for b, right in vectors.items():
                            if add_orders(a, b, correction) != sector:
                                continue
                            total += self._sandwich(left, right, dipoles[(i, j)])
                return total / self.average

            prediction = catani_poles(
                coloured,
                {label: legs[label].representation for label in coloured},
                {label: self.incoming[label - 1] for label in coloured},
                {label: masses[label] for label in coloured},
                correlated,
                casimir=_casimir,
                gamma=lambda j: gammas[j],
                mu2=mu2,
                log=lambda x: _log(x, numbers),
                real=numbers.real,
                sqrt=lambda x: math.sqrt(x) if numbers.double else x.sqrt(),
            )
            if prediction is None:
                return None
            single, double = prediction
            out[sector] = (prefactor * single, prefactor * double)
        return out

    def _qed_poles(self) -> dict[Orders, tuple[Any, Any]] | None:
        """The QED operator per sector (``poles.qed_catani_poles``), or None
        where it makes no prediction.

        The correlator is Q_i Q_j (the legs' all-incoming charges) times the
        colour-summed Born interferences that the sector's QED correction
        dresses (``poles.dressed_born``, cross-order ones included); the
        prefactor alpha / 2 pi takes alpha of the model's photon couplings at
        this point (``_qed_alpha``: alpha_Gmu in a G_mu model), the coupling
        of the loops. A leg is massless where the model's runtime mass is
        zero (``ZERO``, a compile-time zero or a value bound to 0); a derived
        mass is the slice's one mass evaluator's (``_model_masses``). Tagged
        on-shell external photons couple with alpha(0) (``LoopSlice.
        external_photons``, WP3.2 EWI-6b): the Born is the alpha(0) one
        (``average``) and they add no gamma_gamma. Untagged ones keep the
        loops' coupling (MS-PHOTON, per leg): each adds -gamma_gamma/eps
        (``poles.qed_catani_poles``, convention "loop-coupling";
        ``_photon_gamma``). No prediction for a massless charged vector.
        """

        slice_, numbers = self.slice, self.numbers
        abelian = self._correction_orders().abelian
        if abelian is None:
            return None
        compiled = slice_.compiled
        parameters = {record.name: record for record in compiled.ir.parameters}
        records = {record.name: record for record in compiled.ir.particles}
        fixed = slice_.request.fixed_parameters
        mass_of = _model_masses(
            compiled, self.parameters, numbers.precision.working_digits
        )
        _tagged, untagged = ew_policy.slice_photon_tags(slice_)
        charges: dict[int, Fraction] = {}
        masses: dict[int, Fraction] = {}
        spins: dict[int, int] = {}
        for leg in slice_.families.legs:
            record = records[leg.field]
            if ew_policy.is_photon(record, parameters, fixed):
                continue
            charge = exact_charge(record.charge)
            if charge == 0:
                continue
            label, name = leg.index + 1, leg.mass.mass
            if name == ZERO or ew_policy.is_zero(name, parameters, {}):
                masses[label] = Fraction(0)
            else:
                masses[label] = mass_of(name) ** 2
            charges[label] = charge
            spins[label] = int(record.spin)
        alpha = self._qed_alpha()
        if alpha is None:
            return None
        prefactor = alpha / (2 * numbers.pi)
        mu2 = numbers.real(self.scale)
        zero = numbers.real(0)
        gamma_photon = self._photon_gamma(abelian, mass_of) if untagged else None
        out = {}
        for sector, _pairs in slice_.sectors:
            prediction = qed_catani_poles(
                charges,
                {label: self.incoming[label - 1] for label in charges},
                masses,
                dressed_born(self.born, sector, abelian, zero),
                gamma=lambda j: qed_gamma(spins[j], charges[j]),
                mu2=mu2,
                log=lambda x: _log(x, numbers),
                real=numbers.real,
                sqrt=lambda x: math.sqrt(x) if numbers.double else x.sqrt(),
                photons=len(untagged),
                gamma_photon=gamma_photon,
                convention=UNTAGGED_PHOTON_CONVENTION,
            )
            if prediction is None:
                return None
            single, double = prediction
            out[sector] = (prefactor * single, prefactor * double)
        return out

    def _photon_gamma(self, abelian: str, mass_of: Callable[[str], Any]) -> Fraction:
        """gamma_gamma of an untagged on-shell external photon
        (``poles.photon_gamma``): -2/3 sum_f N_c^f Q_f^2 over the fermion
        species perturbative in the electromagnetic correction, not vetoed,
        that are massless at this point (a compile-time zero, a value bound
        to 0, or a runtime value 0), each species once."""

        slice_ = self.slice
        compiled = slice_.compiled
        parameters = {record.name: record for record in compiled.ir.parameters}
        veto = set(slice_.request.particle_veto)
        orders = slice_.prepared.structure.perturbative_species_orders
        perturbative = {
            abs(int(pdg)) for pdg, found in orders.items() if abelian in set(found)
        }
        light: dict[int, tuple[Fraction, int]] = {}
        for record in compiled.ir.particles:
            pdg = abs(int(record.pdg_code))
            if (
                pdg in light
                or pdg in veto
                or pdg not in perturbative
                or int(record.spin) != ew_policy.FERMION_SPIN
                or int(record.ghost_number) != 0
            ):
                continue
            charge = exact_charge(record.charge)
            if charge == 0:
                continue
            name = str(record.mass)
            if not (
                name == ZERO
                or ew_policy.is_zero(name, parameters, slice_.request.fixed_parameters)
                or mass_of(name) == 0
            ):
                continue
            light[pdg] = (charge, abs(int(record.color)))
        return photon_gamma(light.values())

    def _qed_alpha(self) -> Any:
        """alpha of the model's photon couplings at this point's parameters
        in the carrier (``poles.qed_coupling``, ``poles.coupling_value``), or
        None if the model has no charge-universal one."""

        slice_ = self.slice
        cache = slice_._cache
        if "qed-coupling" not in cache:
            compiled = slice_.compiled
            parameters = {record.name: record for record in compiled.ir.parameters}
            fixed = slice_.request.fixed_parameters
            cache["qed-coupling"] = qed_coupling(
                compiled, lambda r: ew_policy.is_photon(r, parameters, fixed)
            )
        coupling = cache["qed-coupling"]
        if coupling is None:
            return None
        return coupling_value(
            coupling,
            _kernel_parameters(slice_.compiled, self.parameters),
            self.numbers.carrier,
        )

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

    Nothing numeric is prepared here: the executor binds every
    parameter-dependent value natively. The result holds the Born (when the
    executor forms |A0|^2) and the ``bare`` component per sector, and, when
    the executor forms the counterterm and external-field components
    (``native_counterterms``), every component as ``_PointEvaluation``
    assembles it, without Catani's pole prediction (WP3.1 I5c). ``slice_``
    may be a native process's structural view (``native_process``).
    """

    def __init__(self, slice_, point, scale, parameters, numbers):
        """``parameters`` are the point's complete values
        (``LoopSlice._point_parameters``)."""

        self.slice = slice_
        self.numbers = numbers
        # The pole mode is set per batch (``plan_export.NativeExecutor``).
        self.mode = pole_mode(PoleSelection.ALL)
        self.poles = True
        self.split_lanes = None
        process = slice_.blocks.process
        labels = sorted(leg.label for leg in process.legs)
        initial = {leg.label for leg in process.legs if leg.is_initial}
        self.incoming = [
            [x if label in initial else -x for x in point[label - 1]]
            for label in labels
        ]
        self.scale = scale
        self.parameters = parameters
        self.born: dict[Orders, Any] = {}
        # With loop-side sources: the executor's loop-side Born interference.
        self.loop_born: dict[Orders, Any] | None = None
        self.counterterms: dict = {}
        # The counterterm and field components with the IR regulator at 0
        # (a model with pole regulators, split selections): formed by the
        # executor (``plan_export.NativeExecutor.reduce``).
        self.counterterms_uv: dict | None = None
        self.native_counterterms = False
        self.colour_contraction = "direct"
        self.retest_families = None

    def counterterm_twin(self) -> None:
        # A native point has no Python twin: the executor forms its
        # counterterms at the IR regulator 0 (``counterterms_uv``).
        if self.slice.pole_regulators is not None:
            raise ValueError(
                "a native point's counterterms at the IR regulator 0 are the "
                "executor's (counterterms_uv), not a Python twin's"
            )
        return None

    def _counterterms_uv(self) -> dict | None:
        if self.counterterms_uv is None and self.slice.pole_regulators is not None:
            raise ValueError(
                "the uv and ir pole selections of a model with pole regulators "
                "need the executor's counterterm components at the IR regulator 0"
            )
        return self.counterterms_uv

    def _scaled_average(self) -> Any:
        # The native trees take the alpha(0) of on-shell external photons in
        # the photons' sources (``native_trees._parameter_program``'s
        # ``photons``): every output the executor forms already carries
        # (alpha(0) / alpha)^n, so the divisor is the slice's average.
        return self.slice.normalisation.average

    def _stable_coloured_legs(self) -> bool:
        # Catani's prediction is not formed natively (WP3.1 I5c).
        return False

    def _ir_poles(self) -> None:
        return None

    def _assemble(self) -> None:
        if self.native_counterterms:
            super()._assemble()
            return
        if self.split_lanes is not None:
            raise NotImplementedError(
                "the uv and ir pole selections need the counterterm components"
            )
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
            loop_born=self.loop_born,
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
        if self.native_counterterms:
            # Every component: the local ones are rotation invariant too.
            return super().rotation_budget(rotated)
        zero = Laurent(*[self.numbers.zero] * (3 if self.poles else 1))
        with self.numbers.scope():
            lanes = {
                lane: tuple(
                    a - b
                    for a, b in zip(
                        rotated.families.get(lane, zero).coefficients,
                        self.families.get(lane, zero).coefficients,
                        strict=True,
                    )
                )
                # Insertion order (the budget's summation order), not a set.
                for lane in dict.fromkeys([*self.families, *rotated.families])
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
    """Tree rows of the slice as ``rx.BlockRows``, as Feynman-rule currents,
    keyed by the contents' block names (``PositionColour.block_names``).

    A row is the sum of the tree blocks of one name, chirality and helicity
    row: the tree engine splits a field current into several DAG currents
    whose keys differ only by the flavour flow of their trees (u u~ > w+ w- g:
    the u current of legs {1, 3, 4} through gamma*/Z* and through a t-channel
    d), and a loop vertex attaches their sum. ``_index`` holds per row the
    ``(block, row)`` entries in block order; ``value`` adds their Feynman-rule
    currents in that order (one entry: its current, as before)."""

    def __init__(self, slice_: LoopSlice, rows):
        from .blocks import _NativeFields
        from .colour import block_name

        fields = _NativeFields.of(slice_.data)
        index = slice_._cache.get("block-rows")
        if index is None:
            index = defaultdict(lambda: defaultdict(dict))
            trees = slice_.blocks
            for block in trees.blocks:
                key = block.key
                if key.auxiliary_kind is not None:
                    continue
                name = fields.incoming(key.particle_id)["name"]
                bucket = index[
                    (
                        key.external_mask,
                        name,
                        key.coupling_orders,
                        block_name(trees, block),
                    )
                ]
                for row in block.rows:
                    helicities = tuple((label - 1, h) for label, h in row.helicities)
                    entries = bucket[key.chirality].get(helicities, ())
                    bucket[key.chirality][helicities] = (*entries, (block, row))
            slice_._cache["block-rows"] = index
        self._index = index
        self._rows = rows

    def rows(self, block, word):
        found = self._index[(block.mask, block.field, block.orders, word)]
        return {chirality: tuple(rows) for chirality, rows in found.items()}

    def value(self, block, word, chirality, row, dimension):
        entries = self._index[(block.mask, block.field, block.orders, word)][chirality][
            row
        ]
        total = None
        for tree, entry in entries:
            variant = "source" if tree.is_source else "propagated"
            sign = tree.wick_sign
            current = tuple(
                sign * value for value in self._rows[(entry.current_id, variant)]
            )
            total = (
                current
                if total is None
                else tuple(a + b for a, b in zip(total, current, strict=True))
            )
        return total


def _family_loop_species(families: FamilySet, data) -> tuple[int, ...]:
    """PDG codes of the loop species the families route (WP3.2 EWI-1): the
    cut species and the loop lines of every admitted and certified-scaleless
    family's routings, with the members of merged closed-loop flavour
    classes. The request's loop content, not the loop particles its
    counterterm rules declare (an electroweak UV rule declares none, and a
    lepton at leg 1 has no counterterm rule at the root); species the
    families never route (a unitary-gauge W with joint corrections on a
    gluon process) get no kernels."""

    names: set[str] = set()
    for group in (families.families, families.scaleless_families):
        for family in group.values():
            for closure in family.closures:
                cut, positions = families.routing(closure)
                names.add(cut)
                for position in positions:
                    names.update((position.vertex.loop_in, position.vertex.loop_out))
    names.update(
        member
        for member, representative in families.merged.items()
        if representative in names
    )
    pdg = {entry["name"]: int(entry["pdg_code"]) for entry in data["particles"]}
    return tuple(sorted(pdg[name] for name in names))


def _correction_species(structure: Any, request: SliceRequest) -> tuple[int, ...]:
    """PDG codes (both charge states) of the species perturbative in a
    requested correction (``perturbative_species_orders``, keyed by |PDG|):
    the loop lines' species before the catalogue is built."""

    corrections = set(request.corrections)
    return tuple(
        sorted(
            sign * abs(int(pdg))
            for pdg, orders in structure.perturbative_species_orders.items()
            if corrections & set(orders)
            for sign in (1, -1)
        )
    )


def _eligible_loop_species(
    loop_catalogue: ModelLoopCatalogue, selection: LoopOrderSelection, data
) -> tuple[int, ...]:
    """PDG codes of the species perturbative in a requested correction (the
    family recursion's eligibility, ``families._Recursion``), vetoed
    particles excluded, ghosts included."""

    pdg = {entry["name"]: int(entry["pdg_code"]) for entry in data["particles"]}
    return tuple(
        sorted(
            pdg[name]
            for name in loop_catalogue.loop_species()
            if loop_catalogue.species(name).perturbative_orders & selection.corrections
        )
    )


def _conjugate_site_entries(sites, conjugates: ConjugateVertices, program):
    """Entries of the conjugate rules through which the root closure reads
    the counterterm ``sites`` whose vertices are not closed under conjugation
    (``ct_sites.ConjugateVertices``), for the kernel catalogue.

    The production program reads such a root site through its conjugate
    rule, so that rule must be lowered: a conjugate rule the program records
    as inert (its vertex enters no tree inside a current) is refused.
    """

    inert = {} if program is None else program.recursion.inert
    out: dict[int, Any] = {}
    for site in sites:
        if conjugates.closed(site.rule.entries):
            continue
        name, entries = conjugates.named(site.rule.entries)
        if name in inert:
            raise NotImplementedError(
                f"counterterm root site {site.rule_name} is read through its "
                f"conjugate rule {name}, which the program leaves inert "
                f"({inert[name]})"
            )
        out.update((id(entry), entry) for entry in entries)
    return list(out.values())


def _check_declared_loop_species(sites, loop_species: Sequence[int]) -> None:
    """The loop particles of every admitted counterterm alternative are
    eligible loop species of the request (``_eligible_loop_species``): a rule
    admitted for loops of species the request cannot route (vetoed or not
    perturbative in its corrections) would renormalise a loop the virtual
    does not contain, so the request is refused as the other admission
    checks refuse theirs (NotImplementedError)."""

    known = {abs(pdg) for pdg in loop_species}
    for site in sites:
        for index in site.alternatives:
            missing = set(site.rule.alternatives[index]) - known
            if missing:
                raise NotImplementedError(
                    f"counterterm rule {site.rule_name} is admitted for loop "
                    f"particles {sorted(missing)} that are no loop species of the "
                    f"request's families {sorted(known)}"
                )


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


def _substitutions(compiled, external: Mapping[str, Decimal], digits: int) -> dict:
    """The model's external parameter symbols bound to ``external`` (real) at
    ``digits`` decimal digits, for ``_definition_value``."""

    from symbolica import E

    from pyamplicol._internal.physics.symbols import symbols
    from pyamplicol.runtime.symbolica_exact import _upcast_complex_inputs

    model_symbols = symbols.model(compiled.name)
    return dict(
        zip(
            (E(model_symbols.symbol(name).to_canonical_string()) for name in external),
            _upcast_complex_inputs(
                [(value, Decimal(0)) for value in external.values()], digits
            ),
            strict=True,
        )
    )


def _definition_value(
    definition: str, substitutions: Mapping[Any, Any], digits: int
) -> tuple[Decimal, Decimal]:
    """A derived parameter's resolved definition (in the external parameters)
    at ``digits`` decimal digits: (real, imaginary)."""

    from symbolica import E

    value = E(definition).evaluate(substitutions, decimal_digit_precision=digits)
    return tuple(Decimal(x) for x in value.to_decimal_tuple())


def _model_masses(
    compiled,
    parameters: Mapping[str, str],
    digits: int,
    *,
    pole_masses: bool = False,
) -> Callable[[str], Any]:
    """The model's mass parameters at ``parameters``, by name, as exact
    values: an external parameter its exact value; a derived one (loop_sm's
    MW, an internal mass of any model) its resolved definition at ``digits``
    decimal digits (the value ``TreeRows._model_values`` gives the
    wavefunctions at the working digits), read exactly. The one mass
    evaluator of the slice: loop propagator masses (``_mass_values``), the
    legs' on-shell invariants (``_leg_masses``) and Catani's massive emitters
    (``_PointEvaluation._qcd_poles``, ``_qed_poles``) take the same values,
    so a derived mass's legs are on the shell its propagators and sources
    take (DESIGN §11.1). A derived mass must be real and positive, except,
    with ``pole_masses``, a complex-mass scheme's pole mass (an internal
    parameter of complex type, ``model_cms``: sqrt(M^2 - i M Gamma) with a
    positive real part), an exact ``rx.Complex``: loop propagators and the
    kernel constants take it; external states are stable
    (``model_cms.require_stable_external_states``), so legs and Catani's
    emitters never read one."""

    values = _kernel_parameters(compiled, parameters)
    definitions = {
        record.name: record.resolved_expression
        for record in compiled.ir.parameters
        if record.nature == "internal"
    }
    complex_typed = {
        record.name
        for record in compiled.ir.parameters
        if record.nature == "internal"
        and str(getattr(record, "parameter_type", "real")) == "complex"
    }
    found: dict[str, Any] = {}
    substitutions: list[dict] = []

    def mass(name: str) -> Any:
        if name == ZERO:
            return Fraction(0)
        if name in values:
            return values[name]
        if name not in found:
            if name not in definitions:
                raise KeyError(f"the mass {name!r} is no model parameter")
            if not substitutions:
                external = {
                    record.name: Decimal(
                        parameters.get(
                            record.name,
                            str(compiled.parameter_defaults[record.name][0]),
                        )
                    )
                    for record in compiled.ir.parameters
                    if record.nature == "external"
                    and record.name in compiled.parameter_defaults
                }
                substitutions.append(_substitutions(compiled, external, digits))
            real, imaginary = _definition_value(
                definitions[name], substitutions[0], digits
            )
            if real <= 0 or (imaginary != 0 and name not in complex_typed):
                raise ValueError(
                    f"the derived mass {name!r} is not real and positive ({real}, "
                    f"{imaginary})"
                )
            found[name] = (
                Fraction(real)
                if imaginary == 0
                else rx.Complex(Fraction(real), Fraction(imaginary))
            )
        value = found[name]
        if isinstance(value, rx.Complex) and not pole_masses:
            raise ValueError(
                f"the mass {name!r} is a complex pole mass ({value!r}): an "
                "external state or a Catani emitter of that mass is unstable "
                "(cms-external-width)"
            )
        return value

    return mass


def _squared(value: Any) -> Any:
    """A mass's square: a ``Fraction``, or an ``rx.Complex`` for a pole mass."""

    return value * value if isinstance(value, rx.Complex) else value**2


def _constant_parameters(
    compiled, parameters: Mapping[str, str], digits: int
) -> dict[str, Fraction]:
    """The model parameter values of the kernel constants
    (``rx.KernelConstants``): every external parameter exactly, and every
    particle's derived mass the slice's one mass evaluator gives
    (``_model_masses``; a loop propagator's kernel reads its mass). A derived
    mass that is not real and positive here is left out (a constant reading
    it is refused by ``KernelConstants``)."""

    values = _kernel_parameters(compiled, parameters)
    internal = {
        record.name for record in compiled.ir.parameters if record.nature == "internal"
    }
    derived = sorted({str(record.mass) for record in compiled.ir.particles} & internal)
    if derived:
        mass_of = _model_masses(compiled, parameters, digits, pole_masses=True)
        for name in derived:
            try:
                values[name] = mass_of(name)
            except ValueError:
                continue
    return values


def _mass_values(
    compiled, parameters: Mapping[str, str], masses: Iterable[Any], digits: int
) -> dict[Any, Any]:
    """Squared loop masses per mass id, a derived mass at ``digits`` decimal
    digits (``_model_masses``): real (widths are not part of the slice), or
    a complex-mass scheme's complex pole mass squared (an ``rx.Complex``,
    M^2 - i M Gamma to the working digits; ``model_cms``)."""

    mass_of = _model_masses(compiled, parameters, digits, pole_masses=True)
    return {
        mass: Fraction(0) if mass.massless else _squared(mass_of(mass.mass))
        for mass in masses
    }


def _leg_masses(
    slice_, parameters: Mapping[str, str], digits: int
) -> tuple[Fraction, ...]:
    """Each external leg's squared mass in leg order, the value of the
    invariants its mass shell fixes (``families.onshell_leg``; DESIGN §11.1):
    the value ``_mass_values`` gives its mass id (``_model_masses``: a
    derived mass at ``digits`` decimal digits), 0 for a massless leg (a mass
    parameter bound to 0 included)."""

    mass_of = _model_masses(slice_.compiled, parameters, digits)
    return tuple(mass_of(leg.mass.mass) ** 2 for leg in slice_.families.legs)


def _charged_fields(compiled) -> frozenset[str]:
    """Names of the model's electrically charged particles."""

    return frozenset(
        str(record.name) for record in compiled.ir.particles if float(record.charge)
    )


def _alpha0_field_rules(source, photons, process) -> tuple[Any, ...]:
    """The field-like term of the alpha(0) convention (WP3.2 EWI-6b) as an
    external-field rule of each tagged photon's species (the generation keeps
    its insertions on the tagged legs only, MS-PHOTON): coefficient
    -(delta Z_e^{G_mu} - delta Z_e^{alpha(0)}) (the model's conversion
    parameter ``photons.conversion``; the field component adds 2 Re of the
    coefficient times the Born per photon leg, MG5_aMC's
    -2 n_gamma DGmua0 Born), of the electromagnetic correction, with no loop
    species (no particle_veto applies: ``ew_policy.external_photon_scheme``
    refuses vetoes of electromagnetically perturbative species)."""

    from pyamplicol.loops.wavefunction import ExternalFieldRule, _laurent

    labels = set(photons.legs)
    pdgs = {abs(int(leg.pdg)) for leg in process.legs if int(leg.label) in labels}
    terms = ((0, f"-{photons.conversion}"),)
    value = {"kind": "mapping", "entries": terms}
    out = []
    for particle in source.raw.collections["particles"]:
        fields = particle["fields"]
        pdg = int(fields["pdg_code"])
        if pdg not in pdgs:
            continue
        groups = fields.get("loop_particles")
        coefficient, diagnostic = _laurent(value, source)
        out.append(
            ExternalFieldRule(
                particle["id"],
                pdg,
                fields["spin"],
                fields["color"],
                photons.correction,
                # After the model's own loop groups of the photon, if any.
                len(groups) if isinstance(groups, tuple) else 0,
                ((),),
                coefficient,
                diagnostic,
                source.scalar_program,
                terms,
            )
        )
    if {rule.pdg for rule in out} != pdgs:
        raise ValueError("an external photon species has no source particle record")
    return tuple(out)


def _correction_orders(slice_) -> ew_policy.CorrectionOrders:
    """``ew_policy.correction_orders`` of a slice's model and fixed
    parameters, kept in its cache."""

    cache = slice_._cache
    found = cache.get("correction-orders")
    if found is None:
        found = cache["correction-orders"] = ew_policy.correction_orders(
            slice_.compiled, slice_.request.fixed_parameters
        )
    return found


def _predicted_sectors(slice_) -> frozenset[Orders]:
    """The sectors whose IR poles the request's operators predict
    (``_sector_predictions``)."""

    return frozenset(
        sector
        for sector, reason in _sector_predictions(slice_).items()
        if reason is None
    )


def _sector_predictions(slice_) -> dict[Orders, str | None]:
    """``poles.sector_predictions`` of the slice (None: predicted, else the
    typed reason), kept in the slice's cache.

    A tree of an order the Born program does not hold is absent when the
    order lies within the request's Born bounds (the program holds every
    tree there), or when an external leg's species has no vertex of the
    request (vetoed particles excluded) whose coupling orders fit within it
    (every leg of a tree attaches to one vertex, and powers only add up).
    Otherwise one tree DAG of the process within that order decides
    (``artifact.born_amplitude_orders``, compiled as the slice's Born with
    the request's particle_veto, every coupling order of the request bounded
    by its power in the order): e+ e- > t t~ h has no tree of order
    g_s^2 e, which the vertex test alone cannot tell. It runs only for a
    loop order whose lower tree lies outside the Born program, once per
    slice, at the first IR-pole prediction. A correction's separated
    exchanges are ``_born_partitions``'."""

    cache = slice_._cache
    found = cache.get("sector-predictions")
    if found is not None:
        return found
    compiled, request = slice_.compiled, slice_.request
    bounds = add_orders(tuple(request.born_orders.items()))
    named = set(request.born_orders) | {
        name for orders in request.exact_loop_orders for name in orders
    }
    names = {str(record.name): record for record in compiled.ir.particles}
    veto = set(request.particle_veto)
    vertices: dict[str, list[Orders]] = defaultdict(list)
    for term in compiled.ir.vertex_terms:
        records = [names.get(name) for name in term.particles]
        if any(r is None or abs(int(r.pdg_code)) in veto for r in records):
            continue
        orders = add_orders(term.coupling_orders)
        for record in records:
            vertices[str(record.name)].append(orders)
            vertices[str(record.antiname)].append(orders)
    legs = [leg.field for leg in slice_.families.legs]

    def tree_exists(order: Orders) -> bool:
        if orders_within(order, bounds):
            return False
        if any(
            not any(orders_within(orders, order) for orders in vertices[field])
            for field in legs
        ):
            return False
        from .artifact import born_amplitude_orders

        powers = dict(order)
        found, _dag = born_amplitude_orders(
            compiled,
            request.process,
            {name: powers.get(name, 0) for name in named | set(powers)},
            request.particle_veto,
        )
        return order in found

    correction_orders = _correction_orders(slice_)

    def separated(name: str, orders: frozenset[Orders]) -> bool:
        if name == correction_orders.strong:
            abelian = False
            charged = frozenset(
                leg.label for leg in slice_.basis.legs if leg.representation != 1
            )
        elif name == correction_orders.abelian:
            abelian = True
            charged = frozenset(
                leg.index + 1
                for leg in slice_.families.legs
                if exact_charge(names[leg.field].charge) != 0
            )
        else:
            # A correction without an operator makes no prediction anyway.
            return False
        partitions = _born_partitions(slice_, charged)
        return separated_exchange(
            (p for order in orders for p in partitions.get(order, ())),
            abelian=abelian,
        )

    found = cache["sector-predictions"] = sector_predictions(
        slice_.sectors,
        slice_.blocks.born_orders,
        request.corrections,
        tree_exists,
        separated,
    )
    return found


def _born_partitions(slice_, charged: frozenset[int]) -> dict[Orders, frozenset]:
    """Per Born amplitude order, the partitions (``poles.Partition``) of its
    trees: the ``charged`` leg labels grouped by the lines of species that
    the request's corrections circulate (``perturbative_species_orders``,
    the loop selection's species condition), kept per slice and leg set.

    A recursion over the Born DAG, never over its trees: per current the
    set of its trees' (closed components, open component) states, the open
    one the charged legs its own line joins; a line of a species no
    requested correction circulates closes it. A root joins the open
    components of its currents (and of its closure vertex's leg). Without
    such a line between charged legs every tree has one component, and no
    state is formed."""

    cache = slice_._cache.setdefault("born-partitions", {})
    found = cache.get(charged)
    if found is not None:
        return found
    dag = slice_.blocks.born_dag
    if hasattr(dag, "materialize"):
        dag = dag.materialize()
    species = slice_.prepared.structure.perturbative_species_orders or {}
    corrections = frozenset(slice_.request.corrections)
    full = (1 << len(slice_.blocks.process.legs)) - 1
    charged_mask = sum(1 << (label - 1) for label in charged)
    orders = {root.root_id: root.coupling_orders for root in slice_.blocks.born_roots}

    def closes(node) -> bool:
        index = node.index
        return (
            not node.is_source
            and index.auxiliary_kind is None
            and not species.get(abs(int(index.particle_id)), frozenset()) & corrections
        )

    if not any(
        closes(node)
        and node.index.external_mask & charged_mask
        and (full ^ node.index.external_mask) & charged_mask
        for node in dag.currents
    ):
        one = frozenset({frozenset({charged})} if charged else {frozenset()})
        found = cache[charged] = {order: one for order in set(orders.values())}
        return found
    incoming: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for interaction in dag.interactions:
        incoming[interaction.result_id].append(
            (interaction.left_id, interaction.right_id)
        )
    states: dict[int, frozenset] = {}
    empty: frozenset = frozenset()

    def current(identifier: int) -> frozenset:
        got = states.get(identifier)
        if got is not None:
            return got
        node = dag.currents[identifier]
        if node.is_source:
            label = node.source_leg_label
            got = frozenset(
                {(empty, frozenset({label}) if label in charged else empty)}
            )
        else:
            joined = {
                (closed_l | closed_r, open_l | open_r)
                for left, right in incoming[identifier]
                for closed_l, open_l in current(left)
                for closed_r, open_r in current(right)
            }
            if closes(node):
                joined = {
                    (closed | frozenset({open_}) if open_ else closed, empty)
                    for closed, open_ in joined
                }
            got = frozenset(joined)
        states[identifier] = got
        return got

    out: dict[Orders, set] = defaultdict(set)
    for root in dag.amplitude_roots:
        left, right = dag.currents[root.left_id], dag.currents[root.right_id]
        sink = full ^ (left.index.external_mask | right.index.external_mask)
        vertex = frozenset(label for label in charged if sink & (1 << (label - 1)))
        for closed_l, open_l in current(root.left_id):
            for closed_r, open_r in current(root.right_id):
                component = open_l | open_r | vertex
                partition = closed_l | closed_r
                if component:
                    partition = partition | frozenset({component})
                out[orders[root.id]].add(partition)
    found = cache[charged] = {order: frozenset(p) for order, p in out.items()}
    return found


def _strong_beta0(slice_, strong: str, values: Mapping[str, Any]) -> Fraction | None:
    """beta_0 of the strong correction (``poles.strong_beta0``) over the
    request's eligible massless coloured species: perturbative in the strong
    order, not vetoed, neither ghosts nor the gluon, massless at this point
    (``ZERO`` or bound to 0); None where it is not covered (genericity
    audit m3: a vetoed b gives the four-flavour value, a light octet fermion
    or coloured scalar its own share)."""

    gluons = _correction_orders(slice_).gluons
    veto = set(slice_.request.particle_veto)
    # The prepared model's particles (masses restricted to ZERO where the
    # request's card makes them vanish), once per particle-antiparticle pair.
    records = {}
    for record in slice_.data["particles"]:
        records.setdefault(abs(int(record["pdg_code"])), record)
    species = []
    for pdg, orders in slice_.prepared.structure.perturbative_species_orders.items():
        pdg = abs(int(pdg))
        record = records.get(pdg)
        if strong not in set(orders) or pdg in veto or record is None:
            continue
        if (
            abs(int(record["color"])) == 1
            or int(record.get("ghost_number") or 0) != 0
            or record["name"] in gluons
        ):
            continue
        mass = record["mass"]
        if not (mass_name(mass) == ZERO or values.get(mass) == 0):
            continue
        real = record["name"] == record.get("antiname")
        species.append((int(record["spin"]), int(record["color"]), real))
    return strong_beta0(sorted(species))


def _strong_leg_gamma(slice_, leg, beta0: Fraction | None) -> Fraction | None:
    """gamma_j of a coloured external leg (``poles.strong_gamma``)."""

    record = next(r for r in slice_.data["particles"] if r["name"] == leg.field)
    return strong_gamma(
        int(record["spin"]),
        int(record["color"]),
        beta0,
        gauge=record["name"] in _correction_orders(slice_).gluons,
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
    found = colour_casimir(representation)
    if found is None:
        raise NotImplementedError(f"colour representation {representation}")
    return found


class _AdjointOrbits:
    """Relabellings of identical external adjoint legs (WP3.1; design §7.4).

    A class holds external adjoint legs of one particle (all-outgoing PDG
    code) whose transpositions leave every given Born key set {(literal
    tensor, N_c power)} invariant, checked exactly (no sets: every leg of
    the particle); the relabellings are the permutations within classes.
    The leading-degree maximum over an invariant key set is then constant
    on the orbit of the other tensor
    (``rename`` maps it to a representative), and a counterterm site's terms
    are those of any site of its orbit, relabelled (``site_key``: its blocks
    partition the legs, so two sites are related iff each block holds equally
    many legs of each class and the same other legs).
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> _AdjointOrbits:
        from . import native_cells  # WP3.1 C3 switch: the "orbits" stage

        return super().__new__(native_cells.orbits_class(cls))

    def __init__(
        self,
        legs: Sequence[Any],
        process: Any,
        born_keys: Iterable[Iterable[tuple]],
    ):
        pdg = {leg.label: leg.outgoing_pdg for leg in process.legs}
        sets = [tuple(keys) for keys in born_keys]
        adjoint = sorted(leg.label for leg in legs if leg.representation == 8)
        reference = [
            frozenset((_colour_key(t, {}), p) for t, p in keys) for keys in sets
        ]
        parent = {label: label for label in adjoint}

        def find(label: int) -> int:
            while parent[label] != label:
                label = parent[label]
            return label

        for i, a in enumerate(adjoint):
            for b in adjoint[i + 1 :]:
                if pdg[a] != pdg[b] or find(a) == find(b):
                    continue
                swap = {a: b, b: a}
                if all(
                    frozenset((_colour_key(t, swap), p) for t, p in keys) == ref
                    for keys, ref in zip(sets, reference, strict=True)
                ):
                    parent[find(b)] = find(a)
        classes: dict[int, list[int]] = defaultdict(list)
        for label in adjoint:
            classes[find(label)].append(label)
        self.classes = tuple(
            tuple(members) for members in classes.values() if len(members) > 1
        )
        self._class = {
            label: c for c, members in enumerate(self.classes) for label in members
        }

    def rename(self, tensor: Any) -> dict[int, int]:
        """A relabelling taking ``tensor`` to its orbit's representative: the
        members of each class in reading order of its canonical key are given
        the class's labels in increasing order."""

        if not self.classes:
            return {}
        strings, traces = _colour_key(tensor, {})
        read = [a for _f, labels, _e in strings for a in labels]
        read += [a for trace in traces for a in trace]
        out: dict[int, int] = {}
        for members in self.classes:
            order = [a for a in dict.fromkeys(read) if a in members]
            order += [a for a in members if a not in order]
            out.update(zip(order, members, strict=True))
        return out

    def site_key(self, site: Any) -> Hashable:
        def mask_key(mask: int) -> tuple:
            labels = [i + 1 for i in range(mask.bit_length()) if mask >> i & 1]
            counts = [0] * len(self.classes)
            fixed = []
            for label in labels:
                c = self._class.get(label)
                if c is None:
                    fixed.append(label)
                else:
                    counts[c] += 1
            return tuple(fixed), tuple(counts)

        return (
            site.rule_name,
            site.alternatives,
            tuple(
                (block.field, block.orders, mask_key(block.mask))
                for block in site.blocks
            ),
        )


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
