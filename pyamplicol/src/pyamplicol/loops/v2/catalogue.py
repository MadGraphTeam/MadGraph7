# SPDX-License-Identifier: 0BSD
"""Process-blind loop kernel catalogue of the v2 engine (DESIGN.md §3.7, WP1.1).

The catalogue compiles, once per model, every local kernel the open-loop
recursion needs: loop segments (split and fused), loop-propagator numerators,
loop closures and counterterm sites. It reads the same UFO Lorentz structures,
spenso networks and tree propagator rules as the tree engine, so loop lines,
blocks and the Born share one set of conventions. No family, skeleton or
Feynman diagram is consulted.

Conventions
-----------
* UFO momenta are incoming. A loop line entering a vertex at leg ``a`` carries
  ``q + P``, attached block ``j`` carries ``p_j`` and the line leaving at leg
  ``b`` is the current of ``anti(particle_b)`` with momentum
  ``q + P + sum_j p_j``. This is the tree engine's oriented-kernel convention
  with the open-loop current in place of the left current.
* A segment is the matrix ``X[beta][beta'](q)`` that maps component ``beta`` of
  the incoming open-loop current to component ``beta'`` of the outgoing one.
  It contains the vertex and the numerator of the outgoing loop propagator;
  denominators belong to NinjaRS. Every vertex of a model is available with
  every ordered (loop-in, loop-out) leg pair, i.e. both traversal orientations.
* Coefficients are raw monomial coefficients of ``q = (q0, q1, q2, q3)``
  labelled by exponent tuples, with no symmetry factors. The native side places
  them with ``ninjars::fast::tensor_layout`` (``rusticol_core::loop_v2::poly``).
* ``fused`` segments take the attached block currents as inputs; ``split``
  segments do not and carry one block-component index per attached leg, so
  ``X(J) = sum_c J_c X_c``.
* Segment couplings are the UFO coupling times the colour-flow normalisation
  ``2^(-p/2)`` that the tree kernels apply. Colour structures are recorded for
  the seed tables but are never part of a kernel. Counterterm sites are bare
  Lorentz contractions; their Laurent couplings and colour belong to the
  counterterm program.
* Loop lines are full current states (chirality 0). Attached blocks come in
  every current state the tree engine produces, including Weyl-projected
  massless fermions.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import cache
from itertools import permutations, product
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast

from ..._internal.physics.symbols import ModelSymbolRegistry, symbols
from ...models import compiler_symbolica as _sym
from ...models import tensors as _tensors
from ...models.compiler_entry import compile_ufo_model_ir
from ...models.compiler_kernels import (
    _input_tensor_expression,
    _ordered_dense_tensor_components,
    _spin_axis_labels,
)
from ...models.expressions import _imaginary_unit_for, _minkowski_square_expression
from ...models.external import CompiledUFOModel
from ...models.prepared_catalog import (
    PreparedContractKind,
    PreparedKernelCatalog,
    PreparedKernelInput,
    PreparedKernelSpec,
    TreeKernelVariant,
    build_prepared_kernel_catalog,
)
from ...models.prepared_catalog_builder import (
    _Candidate,
    _candidate_from_payload,
    _model_parameter_names,
    _particle_runtime_parameter_names,
    _register_candidate,
    _state_chiralities,
)
from ...models.prepared_catalog_helpers import (
    canonical_expressions,
    component_descriptors,
    formal_components,
    formal_model_parameter,
    model_parameter_descriptor,
    used_input_descriptors,
)
from ...models.tensors import normalize_lorentz_expression

if TYPE_CHECKING:
    from ...models.contracts import CompiledParticleRecord, CompiledVertexTerm
    from ...models.loading import CompiledModel
    from ..model import KernelBasisEntry

LOOP_KERNEL_CATALOGUE_ABI = "pyamplicol-loop-kernel-catalogue-v1"

LoopSegmentForm: TypeAlias = Literal["fused", "split"]
LOOP_SEGMENT_FORMS: tuple[LoopSegmentForm, ...] = ("fused", "split")


class LoopKernelCatalogueError(ValueError):
    """Raised when a model cannot provide a complete exact loop kernel."""


@dataclass(frozen=True, order=True, slots=True)
class LoopMonomialEntry:
    """One output coefficient of a loop kernel.

    ``loop_in`` and ``out`` index the incoming and outgoing open-loop current
    components, ``blocks`` the block components of a split segment (one per
    attached leg) and ``exponents`` the powers of ``(q0, q1, q2, q3)``.
    """

    loop_in: int
    out: int
    exponents: tuple[int, int, int, int]
    blocks: tuple[int, ...] = ()

    @property
    def degree(self) -> int:
        return sum(self.exponents)

    def label(self) -> str:
        blocks = "" if not self.blocks else ":b" + ".".join(map(str, self.blocks))
        powers = ".".join(map(str, self.exponents))
        return f"in{self.loop_in}{blocks}:out{self.out}:q{powers}"


@dataclass(frozen=True, order=True, slots=True)
class LoopSegmentKey:
    term_id: int
    loop_in_leg: int
    loop_out_leg: int
    block_chiralities: tuple[int, ...]
    form: LoopSegmentForm


@dataclass(frozen=True, slots=True)
class LoopSegmentBinding:
    """One vertex term traversed from ``loop_in_leg`` to ``loop_out_leg``."""

    key: LoopSegmentKey
    kernel_id: int
    vertex: str
    particles: tuple[str, ...]
    attached_legs: tuple[int, ...]
    loop_in_species: int
    loop_out_species: int
    block_species: tuple[int, ...]
    loop_in_dimension: int
    loop_out_dimension: int
    block_dimensions: tuple[int, ...]
    color_source: str
    color_expression: str
    color_normalization_power: int
    coupling_orders: tuple[tuple[str, int], ...]
    entries: tuple[LoopMonomialEntry, ...] = field(repr=False)

    @property
    def degree(self) -> int:
        """Structural q-degree Delta of vertex times propagator numerator."""

        return max(entry.degree for entry in self.entries)


@dataclass(frozen=True, slots=True)
class LoopPropagatorNumeratorBinding:
    """Numerator of one oriented loop line; NinjaRS owns the denominator."""

    species: int
    kernel_id: int
    dimension: int
    propagator_kind: str
    mass_parameter: str | None
    width_parameter: str | None
    entries: tuple[LoopMonomialEntry, ...] = field(repr=False)

    @property
    def degree(self) -> int:
        return max(entry.degree for entry in self.entries)


@dataclass(frozen=True, slots=True)
class LoopClosureBinding:
    """Trace over the cut index of a loop line of ``species``.

    The closure is a pure trace. The closed fermion- and ghost-loop -1 is part
    of the skeleton's Wick sign epsilon_s (DESIGN.md §3.3, sewing rule), so it
    is applied there exactly once and never here.
    """

    species: int
    kernel_id: int
    dimension: int


@dataclass(frozen=True, order=True, slots=True)
class CountertermSiteKey:
    collection: str
    vertex_id: int
    color_index: int
    lorentz_index: int
    loop_group_index: int
    block_chiralities: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CountertermSiteBinding:
    """One UFO counterterm basis entry applied to block currents at the root."""

    key: CountertermSiteKey
    kernel_id: int
    kind: str
    vertex: str
    particles: tuple[str, ...]
    block_species: tuple[int, ...]
    block_dimensions: tuple[int, ...]
    lorentz_name: str
    color_source: str
    loop_particles: tuple[tuple[int, ...], ...] | None
    coupling_orders: tuple[tuple[str, int], ...]


@dataclass(frozen=True, order=True, slots=True)
class UnsupportedCountertermSite:
    """A counterterm basis entry whose Lorentz structure cannot be lowered."""

    entry_key: tuple[str, int, int, int, int]
    vertex: str
    lorentz_name: str
    reason: str


@dataclass(frozen=True, slots=True)
class LoopKernelCatalogue:
    """Exact loop kernels plus typed bindings into them."""

    model_name: str
    loop_species: tuple[int, ...]
    kernels: tuple[PreparedKernelSpec, ...]
    segments: tuple[LoopSegmentBinding, ...]
    propagator_numerators: tuple[LoopPropagatorNumeratorBinding, ...]
    closures: tuple[LoopClosureBinding, ...]
    counterterm_sites: tuple[CountertermSiteBinding, ...]
    unsupported_counterterms: tuple[UnsupportedCountertermSite, ...] = ()

    def __post_init__(self) -> None:
        ids = tuple(kernel.kernel_id for kernel in self.kernels)
        if ids != tuple(range(len(self.kernels))):
            raise LoopKernelCatalogueError("loop kernel ids must be contiguous")
        signatures = tuple(kernel.canonical_signature for kernel in self.kernels)
        if signatures != tuple(sorted(set(signatures))):
            raise LoopKernelCatalogueError(
                "loop kernels must have unique sorted canonical signatures"
            )
        bindings: tuple[
            LoopSegmentBinding
            | LoopPropagatorNumeratorBinding
            | LoopClosureBinding
            | CountertermSiteBinding,
            ...,
        ] = (
            *self.segments,
            *self.propagator_numerators,
            *self.closures,
            *self.counterterm_sites,
        )
        if any(not 0 <= binding.kernel_id < len(ids) for binding in bindings):
            raise LoopKernelCatalogueError("loop binding references an unknown kernel")
        for context, keys in (
            ("segment", tuple(binding.key for binding in self.segments)),
            (
                "propagator numerator",
                tuple(binding.species for binding in self.propagator_numerators),
            ),
            (
                "closure",
                tuple(binding.species for binding in self.closures),
            ),
            (
                "counterterm site",
                tuple(binding.key for binding in self.counterterm_sites),
            ),
        ):
            if len(keys) != len(set(keys)):
                raise LoopKernelCatalogueError(f"loop {context} keys must be unique")
        polynomials: tuple[LoopSegmentBinding | LoopPropagatorNumeratorBinding, ...] = (
            *self.segments,
            *self.propagator_numerators,
        )
        for polynomial in polynomials:
            kernel = self.kernels[polynomial.kernel_id]
            if kernel.output_layout != tuple(
                entry.label() for entry in polynomial.entries
            ):
                raise LoopKernelCatalogueError(
                    "loop binding entries disagree with their kernel output layout"
                )

    @property
    def by_id(self) -> Mapping[int, PreparedKernelSpec]:
        return MappingProxyType({kernel.kernel_id: kernel for kernel in self.kernels})

    def segment(self, key: LoopSegmentKey) -> LoopSegmentBinding:
        for binding in self.segments:
            if binding.key == key:
                return binding
        raise KeyError(key)

    def propagator_numerator(self, species: int) -> LoopPropagatorNumeratorBinding:
        for binding in self.propagator_numerators:
            if binding.species == species:
                return binding
        raise KeyError(species)


def build_loop_kernel_catalogue(
    compiled: CompiledModel,
    *,
    loop_species: Iterable[int] | None = None,
    counterterm_entries: Iterable[KernelBasisEntry] | None = None,
) -> LoopKernelCatalogue:
    """Build every loop kernel of ``compiled`` for the given loop species.

    ``counterterm_entries`` default to the UV and R2 basis entries of the
    model's raw NLO sidecar. ``loop_species`` (PDG codes, closed under
    conjugation here; ghosts are not added) default to the particles the
    model declares as loop particles of those entries, closed under
    conjugation, plus the ghosts coupled to them. Loop requests pass the
    species their families route (``assemble._family_loop_species``):
    electroweak UV rules declare no loop particles, and a request whose leg 1
    is a lepton has no counterterm rule at its root.
    """

    _sym._ensure_symbolica()
    if counterterm_entries is None:
        counterterm_entries = _source_counterterm_entries(compiled)
    entries = tuple(counterterm_entries)
    model = CompiledUFOModel(compiled)
    builder = _CatalogueBuilder(compiled, model)
    species = (
        builder.declared_loop_species(entries)
        if loop_species is None
        else builder.conjugation_closure(int(pdg) for pdg in loop_species)
    )
    if not species:
        raise LoopKernelCatalogueError(
            f"model {compiled.name!r} declares no loop species"
        )
    for term in compiled.ir.vertex_terms:
        builder.add_vertex_segments(term, species)
    for pdg in species:
        builder.add_propagator_numerator(pdg)
        builder.add_closures(pdg)
    for entry in entries:
        builder.add_counterterm_sites(entry)
    return builder.finish(species)


def compile_tree_kernel_variant(
    compiled: CompiledModel,
    variant: TreeKernelVariant,
) -> CompiledModel:
    """Return ``compiled`` with its tree kernels in ``variant``.

    The variant is recorded in the compiled IR (and so in its serialized form
    and in every prepared catalog built from it). Switching variant recompiles
    the UFO payload; only the eligible massless Yang-Mills kernels change.
    Built-in sources lower only the transverse triple-gluon current, so any
    other variant is refused rather than silently returned transverse.
    """

    if compiled.ir.tree_kernel_variant == variant:
        return compiled
    if compiled.source.get("kind") == "built-in-sm":
        raise LoopKernelCatalogueError(
            f"built-in model {compiled.name!r} only provides the "
            f"{compiled.ir.tree_kernel_variant!r} tree kernel variant; compile "
            f"a UFO model for the {variant!r} variant"
        )
    return replace(
        compiled,
        ir=compile_ufo_model_ir(compiled.model, tree_kernel_variant=variant),
        _serialized_path=None,
        _prepared_bundle=None,
    )


def prepared_tree_kernel_catalog(
    compiled: CompiledModel,
    variant: TreeKernelVariant,
) -> PreparedKernelCatalog:
    """Prepared tree kernel catalog of ``compiled`` in ``variant``."""

    model = CompiledUFOModel(compile_tree_kernel_variant(compiled, variant))
    return build_prepared_kernel_catalog(model)


@dataclass(frozen=True, slots=True)
class LoopKernelEvaluator:
    """One loop kernel as a Symbolica evaluator.

    Binary64 evaluation is SymJIT-compiled. The same evaluator keeps the exact
    coefficients, so its saved state is the exact ``ExpressionEvaluator``
    consumed by DoubleFloat and arbitrary-precision runtimes.
    """

    kernel: PreparedKernelSpec
    evaluator: Any = field(repr=False)

    def evaluate_complex(self, rows: Any) -> Any:
        """Evaluate a batch of input rows (points x inputs) in binary64."""

        return self.evaluator.evaluate_complex(rows)

    def evaluate_with_precision(
        self,
        row: Sequence[Any],
        decimal_digits: int,
    ) -> list[Any]:
        """Evaluate one input row in arbitrary precision."""

        return list(
            self.evaluator.evaluate_complex_with_prec(list(row), decimal_digits)
        )

    def exact_state(self) -> bytes:
        return bytes(self.evaluator.save())


def compile_loop_kernel(kernel: PreparedKernelSpec) -> LoopKernelEvaluator:
    """Compile one loop kernel with the prepared-kernel evaluator settings."""

    from ...evaluators.symbolica_compile import _symbolica_evaluator_kwargs
    from ...evaluators.symbolica_settings import SymbolicaEvaluatorSettings

    _sym._ensure_symbolica()
    evaluator = _sym.Expression.evaluator_multiple(
        [_sym.E(expression) for expression in kernel.exact_expressions],
        [_sym.E(item.symbol) for item in kernel.inputs],
        **_symbolica_evaluator_kwargs(
            SymbolicaEvaluatorSettings(),
            verbose=False,
            jit_compile=True,
        ),
    )
    return LoopKernelEvaluator(kernel, evaluator)


def loop_momentum_symbols(model_name: str) -> tuple[Any, ...]:
    """The loop-momentum components ``q0..q3`` used during extraction."""

    return tuple(
        symbols.model_owned(model_name, f"loop_v2::q{index}") for index in range(4)
    )


def _source_counterterm_entries(compiled: CompiledModel) -> tuple[Any, ...]:
    if compiled.raw_loop_model is None:
        raise LoopKernelCatalogueError(
            f"model {compiled.name!r} has no raw NLO sidecar; pass "
            "counterterm_entries and loop_species explicitly"
        )
    from ..model import build_loop_model

    source = build_loop_model(compiled.raw_loop_model)
    return tuple(entry for entry in source.kernels if entry.kind != "tree")


@dataclass(slots=True)
class _PendingBinding:
    candidate: _Candidate
    build: Any


class _CatalogueBuilder:
    """Accumulates exact candidates and their bindings for one model."""

    def __init__(self, compiled: CompiledModel, model: CompiledUFOModel) -> None:
        self.compiled = compiled
        self.model = model
        self.model_symbols: ModelSymbolRegistry = symbols.model(compiled.name)
        self.particles: dict[str, CompiledParticleRecord] = {
            particle.name: particle for particle in compiled.ir.particles
        }
        self.by_pdg = {
            particle.pdg_code: particle for particle in compiled.ir.particles
        }
        self.q = loop_momentum_symbols(compiled.name)
        parameter_names = _model_parameter_names(model)
        self.parameter_index = {
            name: index for index, name in enumerate(parameter_names)
        }
        self.parameter_symbols = {
            name: formal_model_parameter(model, name, index)
            for name, index in self.parameter_index.items()
        }
        self.runtime_parameter_symbols = {
            symbols.runtime_model_parameter(compiled.name, name): formal
            for name, formal in self.parameter_symbols.items()
        }
        self.parameter_descriptors = tuple(
            model_parameter_descriptor(model, self.parameter_symbols[name], name, index)
            for name, index in self.parameter_index.items()
        )
        self.lorentz = {
            str(record["name"]): record
            for record in cast(
                "Iterable[Mapping[str, object]]",
                compiled.model.get("lorentz_structures", ()),
            )
        }
        self.candidates: dict[str, _Candidate] = {}
        self.proof_classes: dict[str, set[str]] = {}
        self.pending: list[_PendingBinding] = []
        self.unsupported_counterterms: list[UnsupportedCountertermSite] = []

    # -- species -----------------------------------------------------------

    def anti(self, pdg: int) -> int:
        record = self.by_pdg[pdg]
        return self.particles[record.antiname].pdg_code

    def is_ghost(self, pdg: int) -> bool:
        return self.by_pdg[pdg].ghost_number != 0

    def conjugation_closure(self, pdgs: Iterable[int]) -> tuple[int, ...]:
        species: set[int] = set()
        for pdg in pdgs:
            if pdg not in self.by_pdg:
                raise LoopKernelCatalogueError(f"loop species {pdg} is not a particle")
            species.update((pdg, self.anti(pdg)))
        return tuple(sorted(species))

    def declared_loop_species(
        self,
        entries: Sequence[KernelBasisEntry],
    ) -> tuple[int, ...]:
        raw_particles = (
            ()
            if self.compiled.raw_loop_model is None
            else self.compiled.raw_loop_model.collections["particles"]
        )
        declared = {
            self.particles[str(raw_particles[index]["name"])].pdg_code
            for entry in entries
            for alternative in entry.loop_particles or ()
            for index in alternative
        }
        species = set(self.conjugation_closure(declared))
        # Gauge loops come with their Faddeev-Popov ghosts, which a UFO loop
        # declaration leaves implicit (MadLoop adds them the same way).
        for term in self.compiled.ir.vertex_terms:
            pdgs = tuple(self.particles[name].pdg_code for name in term.particles)
            ghosts = {pdg for pdg in pdgs if self.is_ghost(pdg)}
            others = [pdg for pdg in pdgs if not self.is_ghost(pdg)]
            if ghosts and others and set(others) <= species:
                species.update(ghosts)
        return tuple(sorted(species))

    # -- segments ----------------------------------------------------------

    def add_vertex_segments(
        self,
        term: CompiledVertexTerm,
        loop_species: Sequence[int],
    ) -> None:
        records = tuple(self.particles[name] for name in term.particles)
        pdgs = tuple(record.pdg_code for record in records)
        if any(record.auxiliary_kind is not None for record in records):
            return
        legs = range(len(records))
        for loop_in, loop_out in permutations(legs, 2):
            if (
                pdgs[loop_in] not in loop_species
                or self.anti(pdgs[loop_out]) not in loop_species
            ):
                continue
            attached = tuple(leg for leg in legs if leg not in (loop_in, loop_out))
            if any(self.is_ghost(pdgs[leg]) for leg in attached):
                continue
            for chiralities in product(
                *(_state_chiralities(self.model, pdgs[leg]) for leg in attached)
            ):
                self._add_segment(term, loop_in, loop_out, attached, chiralities)

    def _add_segment(
        self,
        term: CompiledVertexTerm,
        loop_in: int,
        loop_out: int,
        attached: tuple[int, ...],
        chiralities: tuple[int, ...],
    ) -> None:
        records = tuple(self.particles[name] for name in term.particles)
        pdgs = tuple(record.pdg_code for record in records)
        in_species, out_species = pdgs[loop_in], self.anti(pdgs[loop_out])
        in_dimension = int(self.model.current_dimension(in_species, 0))
        out_dimension = int(self.model.current_dimension(out_species, 0))
        rows = formal_components(self.model, "loop-current", in_dimension)
        shift = formal_components(self.model, "loop-momentum", 4)
        block_dimensions = tuple(
            int(self.model.current_dimension(pdgs[leg], chirality))
            for leg, chirality in zip(attached, chiralities, strict=True)
        )
        block_values = formal_components(
            self.model, "block-current", sum(block_dimensions)
        )
        block_momentum_values = formal_components(
            self.model, "block-momentum", 4 * len(attached)
        )
        blocks: list[tuple[Any, ...]] = []
        currents: dict[int, tuple[Any, ...]] = {loop_in: rows}
        momenta: dict[int, tuple[Any, ...]] = {
            loop_in: tuple(self.q[index] + shift[index] for index in range(4))
        }
        offset = 0
        for position, (leg, chirality, dimension) in enumerate(
            zip(attached, chiralities, block_dimensions, strict=True)
        ):
            values = block_values[offset : offset + dimension]
            offset += dimension
            blocks.append(values)
            currents[leg] = self._embedded_current(pdgs[leg], chirality, values)
            momenta[leg] = block_momentum_values[4 * position : 4 * position + 4]
        momenta[loop_out] = tuple(
            -sum((momenta[leg][index] for leg in momenta), _sym.E("0"))
            for index in range(4)
        )
        vertex = _contract_lorentz(
            self.model_symbols,
            term.lorentz_expression,
            tuple(record.spin for record in records),
            currents,
            momenta,
            output_leg=loop_out,
            tag=f"loop_v2_t{term.id}",
        )
        coupling = self._coupling(term)
        outputs = self._numerator(
            out_species,
            tuple(coupling * component for component in vertex),
            tuple(-component for component in momenta[loop_out]),
        )
        if len(outputs) != out_dimension:
            raise LoopKernelCatalogueError(
                f"segment of {term.vertex} produced {len(outputs)} components for "
                f"species {out_species}, expected {out_dimension}"
            )
        available = (
            *component_descriptors(self.model, "loop-momentum", shift),
            *component_descriptors(self.model, "block-current", block_values),
            *component_descriptors(self.model, "block-momentum", block_momentum_values),
            *self.parameter_descriptors,
        )
        context = f"segment {term.vertex}/{term.id} {loop_in}->{loop_out}"
        metadata: dict[str, Any] = {
            "vertex": term.vertex,
            "particles": tuple(term.particles),
            "attached_legs": attached,
            "loop_in_species": in_species,
            "loop_out_species": out_species,
            "block_species": tuple(pdgs[leg] for leg in attached),
            "loop_in_dimension": in_dimension,
            "loop_out_dimension": out_dimension,
            "block_dimensions": block_dimensions,
            "color_source": term.color_source,
            "color_expression": term.color_expression,
            "color_normalization_power": int(term.lc_color_normalization_power),
            "coupling_orders": tuple(term.coupling_orders),
        }
        for form in LOOP_SEGMENT_FORMS:
            lowered = self._polynomial_candidate(
                "loop-segment",
                outputs,
                rows,
                tuple(blocks) if form == "split" else (),
                available,
                context=f"{form} {context}",
            )
            if lowered is None:
                if any(chiralities):
                    return  # a chirality selection rule, not a missing kernel
                raise LoopKernelCatalogueError(f"{context} lowered to zero")
            entries, candidate = lowered
            key = LoopSegmentKey(term.id, loop_in, loop_out, chiralities, form)
            self.pending.append(
                _PendingBinding(
                    candidate,
                    lambda kernel_id, key=key, entries=entries: LoopSegmentBinding(
                        key=key,
                        kernel_id=kernel_id,
                        entries=entries,
                        **metadata,
                    ),
                )
            )

    def _embedded_current(
        self,
        pdg: int,
        chirality: int,
        values: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        embedded = self.model._embed_weyl_current(pdg, chirality, values)
        return tuple(
            value if isinstance(value, _sym.Expression) else _sym.E(str(int(value)))
            for value in embedded
        )

    def _coupling(self, term: CompiledVertexTerm) -> Any:
        coupling = _sym.E(term.coupling_expression).replace_multiple(
            [
                _sym.Replacement(source, target)
                for source, target in self.runtime_parameter_symbols.items()
            ]
        )
        power = int(term.lc_color_normalization_power)
        return coupling * _sym.E(f"2^(-{power}/2)") if power else coupling

    # -- loop lines --------------------------------------------------------

    def _numerator(
        self,
        pdg: int,
        value: tuple[Any, ...],
        momentum: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        """Apply the numerator of the loop line ``pdg`` carrying ``momentum``."""

        names = _particle_runtime_parameter_names(self.model, pdg)
        model = self.model.with_runtime_parameters(
            {name: self.parameter_symbols[name] for name in names}
        )
        try:
            metadata = model._propagator_ir(pdg, 0)
        except ValueError:
            metadata = None
        if metadata is None:
            # Ghosts have no tree propagator rule; their UFO numerator carries
            # no index, so it multiplies the single current component.
            factor = self._index_free_ufo_numerator(pdg)
            return tuple((factor * component).expand() for component in value)
        if metadata.kind == "vector" and metadata.gauge != "feynman":
            raise LoopKernelCatalogueError(
                f"loop line {pdg} has a {metadata.gauge}-gauge propagator; v2 loop "
                "lines need Feynman gauge with explicit Goldstones and ghosts"
            )
        propagated = model.propagator_component_expression(
            pdg,
            value,
            momentum,
            chirality=0,
            propagator=metadata,
        )
        square = _minkowski_square_expression(momentum)
        if metadata.mass_class == "massless":
            denominator = square
        else:
            mass, width = model.mass(pdg), model.width(pdg)
            imaginary = _imaginary_unit_for((*momentum, mass, width))
            denominator = square - mass * mass + imaginary * mass * width
        numerator = tuple(
            (_sym.E("0") + component * denominator).expand() for component in propagated
        )
        q_names = {symbol.to_canonical_string() for symbol in self.q}
        for component in numerator:
            for _key, coefficient in component.coefficient_list(*self.q):
                if q_names & {
                    symbol.to_canonical_string()
                    for symbol in coefficient.get_all_symbols(False)
                }:
                    raise LoopKernelCatalogueError(
                        f"loop line {pdg} has no polynomial numerator over the "
                        "standard denominator"
                    )
        return numerator

    def _index_free_ufo_numerator(self, pdg: int) -> Any:
        record = self.model._propagator_record(pdg)
        if record is None or record.custom or self.by_pdg[pdg].spin not in (-1, 1):
            raise LoopKernelCatalogueError(
                f"loop line {pdg} has no supported propagator numerator"
            )
        numerator = self.model_symbols.expression(record.numerator)
        return numerator.replace_multiple(
            [
                _sym.Replacement(self.model_symbols.symbol(name), formal)
                for name, formal in self.parameter_symbols.items()
            ]
        )

    def add_propagator_numerator(self, pdg: int) -> None:
        dimension = int(self.model.current_dimension(pdg, 0))
        rows = formal_components(self.model, "loop-current", dimension)
        shift = formal_components(self.model, "loop-momentum", 4)
        outputs = self._numerator(
            pdg,
            rows,
            tuple(self.q[index] + shift[index] for index in range(4)),
        )
        lowered = self._polynomial_candidate(
            "loop-propagator-numerator",
            outputs,
            rows,
            (),
            (
                *component_descriptors(self.model, "loop-momentum", shift),
                *self.parameter_descriptors,
            ),
            context=f"loop propagator numerator {pdg}",
        )
        if lowered is None:
            raise LoopKernelCatalogueError(f"loop line {pdg} has a zero numerator")
        entries, candidate = lowered
        try:
            metadata = self.model._propagator_ir(pdg, 0)
        except ValueError:
            metadata = None
        kind = "ufo-index-free" if metadata is None else metadata.kind
        # The mass id follows the lowered numerator: a line the tree engine
        # treats as massless has no runtime mass, whatever the UFO declares.
        massive = metadata is None or metadata.mass_class == "massive"
        mass = self.model.runtime_mass_parameter_name(pdg) if massive else None
        width = self.model.runtime_width_parameter_name(pdg) if massive else None
        self.pending.append(
            _PendingBinding(
                candidate,
                lambda kernel_id: LoopPropagatorNumeratorBinding(
                    species=pdg,
                    kernel_id=kernel_id,
                    dimension=dimension,
                    propagator_kind=kind,
                    mass_parameter=mass,
                    width_parameter=width,
                    entries=entries,
                ),
            )
        )

    def add_closures(self, pdg: int) -> None:
        dimension = int(self.model.current_dimension(pdg, 0))
        values = formal_components(self.model, "loop-current", dimension * dimension)
        trace = sum(
            (values[index * (dimension + 1)] for index in range(dimension)),
            _sym.E("0"),
        )
        exact = canonical_expressions((trace,), context="loop closure")
        inputs = used_input_descriptors(
            (trace,),
            component_descriptors(self.model, "loop-current", values),
        )
        candidate = self._candidate("loop-closure", exact, inputs, ("scalar:c0",))
        self.pending.append(
            _PendingBinding(
                candidate,
                lambda kernel_id: LoopClosureBinding(
                    species=pdg,
                    kernel_id=kernel_id,
                    dimension=dimension,
                ),
            )
        )

    # -- counterterm sites -------------------------------------------------

    def add_counterterm_sites(self, entry: KernelBasisEntry) -> None:
        try:
            records = tuple(self.particles[name] for name in entry.particle_names)
        except KeyError as error:
            raise LoopKernelCatalogueError(
                f"counterterm {entry.vertex_name} refers to unknown particle {error}"
            ) from error
        pdgs = tuple(record.pdg_code for record in records)
        if any(self.is_ghost(pdg) for pdg in pdgs):
            raise LoopKernelCatalogueError(
                f"counterterm {entry.vertex_name} has a ghost leg"
            )
        lorentz = self.lorentz.get(entry.lorentz_name)
        if lorentz is None:
            raise LoopKernelCatalogueError(
                f"counterterm Lorentz structure {entry.lorentz_name!r} is not in "
                f"the compiled payload of {self.compiled.name!r}"
            )
        spins = tuple(record.spin for record in records)
        try:
            # R2 rules of neutral currents carry Epsilon (e.g. R2_GGZ).
            expression = normalize_lorentz_expression(
                _with_levi_civita(str(lorentz["structure"])),
                spins,
                model_symbols=self.model_symbols,
            ).expression
        except ValueError as error:
            # Recorded, never dropped: counterterm completeness refuses every
            # process that needs this entry.
            self.unsupported_counterterms.append(
                UnsupportedCountertermSite(
                    _counterterm_entry_key(entry),
                    entry.vertex_name,
                    entry.lorentz_name,
                    str(error),
                )
            )
            return
        for chiralities in product(
            *(_state_chiralities(self.model, pdg) for pdg in pdgs)
        ):
            self._add_counterterm_site(entry, pdgs, spins, expression, chiralities)

    def _add_counterterm_site(
        self,
        entry: KernelBasisEntry,
        pdgs: tuple[int, ...],
        spins: tuple[int, ...],
        lorentz_expression: str,
        chiralities: tuple[int, ...],
    ) -> None:
        dimensions = tuple(
            int(self.model.current_dimension(pdg, chirality))
            for pdg, chirality in zip(pdgs, chiralities, strict=True)
        )
        values = formal_components(self.model, "block-current", sum(dimensions))
        momentum_values = formal_components(self.model, "block-momentum", 4 * len(pdgs))
        currents: dict[int, tuple[Any, ...]] = {}
        momenta: dict[int, tuple[Any, ...]] = {}
        offset = 0
        for leg, (pdg, chirality, dimension) in enumerate(
            zip(pdgs, chiralities, dimensions, strict=True)
        ):
            currents[leg] = self._embedded_current(
                pdg, chirality, values[offset : offset + dimension]
            )
            offset += dimension
            momenta[leg] = momentum_values[4 * leg : 4 * leg + 4]
        (site,) = _contract_lorentz(
            self.model_symbols,
            lorentz_expression,
            spins,
            currents,
            momenta,
            output_leg=None,
            tag=f"loop_v2_ct{entry.vertex_id}",
        )
        context = f"counterterm site {entry.vertex_name}/{entry.lorentz_name}"
        exact = canonical_expressions((site,), context=context)
        if exact == ("0",):
            if any(chiralities):
                return  # a chirality selection rule, not a missing kernel
            raise LoopKernelCatalogueError(f"{context} lowered to zero")
        inputs = used_input_descriptors(
            (site,),
            (
                *component_descriptors(self.model, "block-current", values),
                *component_descriptors(self.model, "block-momentum", momentum_values),
            ),
        )
        candidate = self._candidate("ct-site", exact, inputs, ("scalar:c0",))
        key = CountertermSiteKey(*_counterterm_entry_key(entry), chiralities)
        self.pending.append(
            _PendingBinding(
                candidate,
                lambda kernel_id: CountertermSiteBinding(
                    key=key,
                    kernel_id=kernel_id,
                    kind=entry.kind,
                    vertex=entry.vertex_name,
                    particles=tuple(entry.particle_names),
                    block_species=pdgs,
                    block_dimensions=dimensions,
                    lorentz_name=entry.lorentz_name,
                    color_source=entry.color,
                    loop_particles=entry.loop_particles,
                    coupling_orders=tuple(entry.order),
                ),
            )
        )

    # -- candidates --------------------------------------------------------

    def _polynomial_candidate(
        self,
        contract_kind: PreparedContractKind,
        outputs: Sequence[Any],
        rows: Sequence[Any],
        blocks: Sequence[Sequence[Any]],
        available: Sequence[PreparedKernelInput],
        *,
        context: str,
    ) -> tuple[tuple[LoopMonomialEntry, ...], _Candidate] | None:
        coefficients = _polynomial_coefficients(outputs, rows, blocks, self.q)
        entries = tuple(sorted(coefficients))
        if not entries:
            return None
        values = tuple(coefficients[entry] for entry in entries)
        exact = canonical_expressions(values, context=context)
        inputs = used_input_descriptors(values, available)
        layout = tuple(entry.label() for entry in entries)
        return entries, self._candidate(contract_kind, exact, inputs, layout)

    def _candidate(
        self,
        contract_kind: PreparedContractKind,
        exact: tuple[str, ...],
        inputs: tuple[PreparedKernelInput, ...],
        layout: tuple[str, ...],
    ) -> _Candidate:
        candidate = _candidate_from_payload(
            contract_kind,
            {"abi": LOOP_KERNEL_CATALOGUE_ABI},
            exact,
            inputs,
            layout,
            proof_class=None,
        )
        _register_candidate(self.candidates, self.proof_classes, candidate)
        return candidate

    def finish(self, loop_species: tuple[int, ...]) -> LoopKernelCatalogue:
        signatures = tuple(sorted(self.candidates))
        kernel_ids = {signature: index for index, signature in enumerate(signatures)}
        kernels = tuple(
            self.candidates[signature].spec(
                kernel_ids[signature],
                tuple(sorted(self.proof_classes.get(signature, ()))),
            )
            for signature in signatures
        )
        built = [
            pending.build(kernel_ids[pending.candidate.canonical_signature])
            for pending in self.pending
        ]

        def select(kind: type) -> tuple[Any, ...]:
            return tuple(binding for binding in built if isinstance(binding, kind))

        return LoopKernelCatalogue(
            model_name=self.compiled.name,
            loop_species=loop_species,
            kernels=kernels,
            segments=tuple(
                sorted(select(LoopSegmentBinding), key=lambda item: item.key)
            ),
            propagator_numerators=tuple(
                sorted(
                    select(LoopPropagatorNumeratorBinding),
                    key=lambda item: item.species,
                )
            ),
            closures=tuple(
                sorted(
                    select(LoopClosureBinding),
                    key=lambda item: item.species,
                )
            ),
            counterterm_sites=tuple(
                sorted(select(CountertermSiteBinding), key=lambda item: item.key)
            ),
            unsupported_counterterms=tuple(sorted(set(self.unsupported_counterterms))),
        )


def _counterterm_entry_key(entry: KernelBasisEntry) -> tuple[str, int, int, int, int]:
    return (
        entry.collection,
        entry.vertex_id,
        entry.color_index,
        entry.lorentz_index,
        -1 if entry.loop_group_index is None else entry.loop_group_index,
    )


# The four-dimensional Levi-Civita tensor of UFO Lorentz structures.
UFO_LEVI_CIVITA = symbols.qualified_name("loop_v2_ufo_levi_civita")


def _permutation_sign(permutation: Sequence[int]) -> int:
    sign, order = 1, list(permutation)
    for index in range(len(order)):
        while order[index] != index:
            target = order[index]
            order[index], order[target] = order[target], order[index]
            sign = -sign
    return sign


@cache
def ufo_levi_civita_tensor() -> Any:
    """The dense ``Epsilon`` of UFO Lorentz structures (``UFO_LEVI_CIVITA``).

    ALOHA (``aloha_object.L_Epsilon``) represents Epsilon by its contravariant
    components with epsilon^{0123} = -1 (epsilon_{0123} = +1) and contracts
    every Lorentz index with the metric. Spenso contracts Minkowski slots the
    same way (vectors are contravariant components), so the components are
    ALOHA's: minus the sign of the index permutation. The ``ct-site`` kernels
    of R2_GGZ and R2_GGGVa equal compiled ALOHA routines to rounding
    (``tests/unit/test_loop_v2_levi_civita.py``).
    """

    _sym._ensure_symbolica()
    minkowski = _sym.Representation.mink(4)
    components = [
        _sym.E(str(-_permutation_sign(axes) if len(set(axes)) == 4 else 0))
        for axes in product(range(4), repeat=4)
    ]
    return _sym.Tensor.dense(
        _sym.TensorName(UFO_LEVI_CIVITA)(*[minkowski] * 4), components
    )


def _with_levi_civita(structure: str) -> str:
    """A wrapped UFO Lorentz structure with ``Epsilon`` as ``UFO_LEVI_CIVITA``.

    The tree model compiler refuses Epsilon (``models.tensors``); the loop
    catalogue admits it for counterterm sites only, with the slot labels the
    normaliser gives the other Lorentz indices.
    """

    if "Epsilon" not in structure:
        return structure
    _sym._ensure_symbolica()
    _tensors._ensure_symbolica()
    minkowski = _sym.Representation.mink(4)
    name = _sym.TensorName(UFO_LEVI_CIVITA)
    wildcards = _tensors._WILDCARDS

    def replace(match: Mapping[Any, Any]) -> Any:
        return name(
            *(
                minkowski(_tensors._ufo_lorentz_label(match[wildcards[label]]))
                for label in ("a_", "b_", "c_", "d_")
            )
        ).to_expression()

    return _tensors._replace(
        _sym.E(structure), "UFO::Epsilon(a_,b_,c_,d_)", replace
    ).to_canonical_string()


def _contract_lorentz(
    model_symbols: ModelSymbolRegistry,
    lorentz_expression: str,
    spins: Sequence[int],
    currents: Mapping[int, Sequence[Any]],
    momenta: Mapping[int, Sequence[Any]],
    *,
    output_leg: int | None,
    tag: str,
) -> tuple[Any, ...]:
    """Contract a normalized UFO Lorentz tensor with currents and momenta.

    This is the k-leg form of the tree engine's oriented-kernel network: every
    leg except ``output_leg`` receives a current, every leg receives its
    incoming momentum, and the open index of ``output_leg`` (if any) is
    returned in the tree engine's component order.
    """

    library = _sym.TensorLibrary.hep_lib_atom()
    if UFO_LEVI_CIVITA in lorentz_expression:
        library.register(ufo_levi_civita_tensor())
    minkowski = _sym.Representation.mink(4)
    momentum_names = {
        leg: _sym.TensorName(model_symbols.ufo_momentum_tensor_name(leg + 1))
        for leg in range(len(spins))
    }
    expression = _sym.E(lorentz_expression)
    for leg, components in sorted(currents.items()):
        expression *= _input_tensor_expression(
            library,
            kind=0,
            side=f"{tag}_leg{leg}",
            spin=spins[leg],
            leg=leg + 1,
            components=tuple(components),
            model_symbols=model_symbols,
        )
    for leg, momentum in momenta.items():
        library.register(
            _sym.Tensor.dense(momentum_names[leg](minkowski), tuple(momentum))
        )
    labels = (
        ()
        if output_leg is None
        else _spin_axis_labels(spins[output_leg], output_leg + 1)
    )
    network = _sym.TensorNetwork(
        _sym.as_tensor(expression) if labels else expression, library
    )
    network.execute(library=library)
    return tuple(
        _sym.E("0") + component
        for component in _ordered_dense_tensor_components(
            network.result_tensor(library), labels
        ).values
    )


def _polynomial_coefficients(
    outputs: Sequence[Any],
    rows: Sequence[Any],
    blocks: Sequence[Sequence[Any]],
    q: Sequence[Any],
) -> dict[LoopMonomialEntry, Any]:
    """Extract the coefficients of a map linear in ``rows`` and each block."""

    variables = (*rows, *(value for block in blocks for value in block), *q)
    names = tuple(variable.to_canonical_string() for variable in variables)
    block_slices: list[slice] = []
    start = len(rows)
    for block in blocks:
        block_slices.append(slice(start, start + len(block)))
        start += len(block)
    coefficients: dict[LoopMonomialEntry, Any] = {}
    for out, output in enumerate(outputs):
        expanded = (_sym.E("0") + output).expand()
        for key, coefficient in expanded.coefficient_list(*variables):
            if coefficient.to_canonical_string() == "0":
                continue
            degrees = _monomial_degrees(key, variables, names)
            row = _single_linear_index(degrees[: len(rows)])
            block_indices = tuple(
                _single_linear_index(degrees[item]) for item in block_slices
            )
            if row is None or None in block_indices:
                raise LoopKernelCatalogueError(
                    "loop kernel is not linear in the open-loop current and blocks"
                )
            entry = LoopMonomialEntry(
                row,
                out,
                tuple(degrees[-4:]),  # type: ignore[arg-type]
                block_indices,  # type: ignore[arg-type]
            )
            coefficients[entry] = coefficients.get(entry, _sym.E("0")) + coefficient
    return {
        entry: value
        for entry, value in coefficients.items()
        if value.expand().to_canonical_string() != "0"
    }


def _monomial_degrees(
    monomial: Any,
    variables: Sequence[Any],
    names: Sequence[str],
) -> tuple[int, ...]:
    polynomial = monomial.to_polynomial(vars=list(variables))
    present = {
        variable.to_canonical_string() for variable in polynomial.get_variables()
    }
    if not present <= set(names):
        raise LoopKernelCatalogueError(f"unexpected monomial {monomial}")
    return tuple(
        int(polynomial.degree(variable)) if name in present else 0
        for variable, name in zip(variables, names, strict=True)
    )


def _single_linear_index(degrees: Sequence[int]) -> int | None:
    if sorted(degrees)[-1:] != [1] or sum(degrees) != 1:
        return None
    return list(degrees).index(1)


__all__ = [
    "LOOP_KERNEL_CATALOGUE_ABI",
    "LOOP_SEGMENT_FORMS",
    "CountertermSiteBinding",
    "CountertermSiteKey",
    "LoopClosureBinding",
    "LoopKernelCatalogue",
    "LoopKernelCatalogueError",
    "LoopKernelEvaluator",
    "LoopMonomialEntry",
    "LoopPropagatorNumeratorBinding",
    "LoopSegmentBinding",
    "LoopSegmentForm",
    "LoopSegmentKey",
    "UnsupportedCountertermSite",
    "build_loop_kernel_catalogue",
    "compile_loop_kernel",
    "compile_tree_kernel_variant",
    "loop_momentum_symbols",
    "prepared_tree_kernel_catalog",
]
