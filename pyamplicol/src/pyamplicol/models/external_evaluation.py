# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from .._internal.physics.symbols import symbols
from ._physics_ir import PropagatorGauge, PropagatorIR, PropagatorKind
from .base import (
    PropagatorLoweringRule,
)
from .compiler_kernels import (
    DenseTensors,
    _as_expression,
    _execute_dense_network,
    _spin_axis_labels,
    _spin_representations,
    _spin_slots,
)
from .contracts import U1_SUBTRACTION_AUXILIARY_KIND
from .expressions import (
    _expr_antifermion_propagator_dirac,
    _expr_antifermion_propagator_weyl,
    _expr_fermion_propagator_dirac,
    _expr_fermion_propagator_weyl,
    _expr_minkowski_dot,
    _imaginary_unit_for,
    _minkowski_square_expression,
)
from .tensors import (
    normalize_lorentz_expression,
)

if TYPE_CHECKING:
    pass

from . import compiler_symbolica as _sym
from .external_helpers import _expr_spin2_propagator, _is_numeric, _replace_symbols


class ExternalModelEvaluationMixin:
    def vertex_component_expression(
        self,
        kind: int,
        left: Sequence[Any],
        right: Sequence[Any],
        *,
        result_particle_id: int,
        result_chirality: int,
        left_chirality: int = 0,
        right_chirality: int = 0,
        coupling: tuple[Any, Any] = (1.0, 0.0),
        left_momentum: Sequence[Any] | None = None,
        right_momentum: Sequence[Any] | None = None,
    ) -> tuple[Any, ...]:
        del coupling
        kernel = self._kernel(kind)
        expected_result = self._particle_records_by_name[kernel.particles[2]].pdg_code
        if int(result_particle_id) != expected_result:
            raise ValueError(
                f"kernel {kind} returns particle {expected_result}, "
                f"not {result_particle_id}"
            )
        left_momentum = tuple(left_momentum or (0.0, 0.0, 0.0, 0.0))
        right_momentum = tuple(right_momentum or (0.0, 0.0, 0.0, 0.0))
        if len(left_momentum) != 4 or len(right_momentum) != 4:
            raise ValueError("UFO kernels require four-component input momenta")
        runtime_parameter_values = {
            name: self._parameter_value(name) for name in kernel.runtime_parameters
        }
        if not all(
            _is_numeric(value)
            for value in (
                *left,
                *right,
                *left_momentum,
                *right_momentum,
                *runtime_parameter_values.values(),
            )
        ):
            return self._kernel_function_component_calls(
                kernel,
                left,
                right,
                left_chirality=left_chirality,
                right_chirality=right_chirality,
                result_chirality=result_chirality,
                left_momentum=left_momentum,
                right_momentum=right_momentum,
                runtime_parameter_values=runtime_parameter_values,
            )
        components = self._projected_kernel_components(
            kernel,
            left,
            right,
            left_chirality=left_chirality,
            right_chirality=right_chirality,
            result_chirality=result_chirality,
            left_momentum=left_momentum,
            right_momentum=right_momentum,
            runtime_parameter_values=runtime_parameter_values,
        )
        coupling_expression = self._resolved_kernel_coupling_expression(
            kernel,
            runtime_parameter_values,
            numeric=True,
        )
        return tuple(component * coupling_expression for component in components)

    def symbolica_function_definitions(
        self,
    ) -> Mapping[tuple[_sym.Expression, tuple[_sym.Expression, ...]], _sym.Expression]:
        """Return lazily materialized kernel functions used by stage expressions."""

        return self._symbolica_kernel_functions

    def propagator_lowering_rule(
        self,
        particle_id: int,
        chirality: int = 0,
    ) -> PropagatorLoweringRule:
        auxiliary_kind = self.auxiliary_kind(particle_id)
        if (
            auxiliary_kind is not None
            and auxiliary_kind != U1_SUBTRACTION_AUXILIARY_KIND
        ):
            return PropagatorLoweringRule(
                particle_id=particle_id,
                chirality=0,
                backend="identity",
                full_tensor_network_ready=True,
                applies_propagator=False,
                kernel="ufo_contact_auxiliary_no_propagator",
                kind="identity",
                mass_class="not-applicable",
                auxiliary_policy=auxiliary_kind,
                description="synthetic UFO contact current with no propagator",
            )
        if chirality != 0 and self.is_chiral_eligible(particle_id):
            return PropagatorLoweringRule(
                particle_id=particle_id,
                chirality=chirality,
                backend="spenso-ufo-weyl",
                full_tensor_network_ready=True,
                applies_propagator=True,
                kernel="ufo_weyl_fermion",
                kind="weyl-fermion",
                mass_class="massless",
                description="UFO massless fermion projected to a Weyl current",
            )
        spin = self.spin(particle_id)
        massive = (
            complex(
                self._parameter_default(
                    self._particle_records_by_pdg[int(particle_id)].mass
                )
            ).real
            != 0.0
        )
        custom = self._propagator_record(particle_id)
        if custom is not None and custom.custom:
            return PropagatorLoweringRule(
                particle_id=particle_id,
                chirality=0,
                backend="spenso-ufo-custom",
                full_tensor_network_ready=True,
                applies_propagator=True,
                kernel="ufo_custom_propagator",
                kind="custom",
                mass_class="massive" if massive else "massless",
                gauge="model-supplied",
                numerator=custom.numerator,
                denominator=custom.denominator,
                custom_source=custom.name,
                goldstone_policy="model-supplied",
                description=(
                    "model-supplied UFO propagator lowered without gauge conversion"
                ),
            )
        # A massive vector takes the compiled model's gauge: unitary (its
        # Goldstone absorbed) or 't Hooft-Feynman (its Goldstone explicit,
        # ``compiler_gauge``). A massless vector is Feynman-gauge in both.
        feynman_massive = massive and self.compiled.ir.gauge == "feynman"
        standard_contracts: dict[
            int,
            tuple[PropagatorKind, str, PropagatorGauge | None],
        ] = {
            1: ("scalar", "ufo_scalar_propagator", None),
            2: ("dirac-fermion", "ufo_dirac_propagator", None),
            3: (
                "vector",
                (
                    "ufo_massive_vector_propagator_feynman"
                    if feynman_massive
                    else "ufo_massive_vector_propagator"
                    if massive
                    else "ufo_massless_vector_propagator"
                ),
                "unitary" if massive and not feynman_massive else "feynman",
            ),
            5: (
                "spin2",
                (
                    "ufo_massive_spin2_fierz_pauli"
                    if massive
                    else "ufo_massless_spin2_de_donder"
                ),
                "fierz-pauli" if massive else "de-donder",
            ),
        }
        try:
            kind, kernel, gauge = standard_contracts[spin]
        except KeyError as exc:
            raise ValueError(
                f"UFO spin code {spin} has no supported propagator contract"
            ) from exc
        return PropagatorLoweringRule(
            particle_id=particle_id,
            chirality=0,
            backend="symbolica-ufo",
            full_tensor_network_ready=spin in {1, 2, 3, 5},
            applies_propagator=True,
            kernel=kernel,
            kind=kind,
            mass_class="massive" if massive else "massless",
            gauge=gauge,
            goldstone_policy=(
                "explicit"
                if spin == 3 and feynman_massive
                else "absorbed"
                if spin == 3 and massive
                else "not-applicable"
            ),
            description="UFO propagator with runtime mass and width",
        )

    def propagator_component_expression(
        self,
        particle_id: int,
        value: Sequence[Any],
        momentum: Sequence[Any],
        *,
        chirality: int = 0,
        propagator: PropagatorIR | None = None,
    ) -> tuple[Any, ...]:
        metadata = propagator or self._propagator_ir(particle_id, chirality)
        if metadata.identity.canonical_id != self._particle_identity_ir(
            particle_id
        ).canonical_id or metadata.chirality != int(chirality):
            raise ValueError("propagator metadata does not match the current")
        if not metadata.applies_propagator:
            return tuple(value)
        if metadata.kind == "weyl-fermion":
            components = tuple(value)
            current_momentum = tuple(momentum)
            return (
                _expr_antifermion_propagator_weyl(
                    components,
                    current_momentum,
                    chirality,
                )
                if metadata.identity.orientation == "antiparticle"
                else _expr_fermion_propagator_weyl(
                    components,
                    current_momentum,
                    chirality,
                )
            )
        if metadata.kind == "custom":
            custom = self._propagator_record(particle_id)
            if (
                custom is None
                or not custom.custom
                or custom.name != metadata.custom_source
            ):
                raise ValueError(
                    "custom propagator contract does not match its compiled source"
                )
            return self._evaluate_custom_propagator(
                particle_id,
                value,
                momentum,
            )
        if len(momentum) != 4:
            raise ValueError("UFO propagators require four-momentum components")
        if metadata.mass_class == "massless":
            mass = 0.0
            width = 0.0
        else:
            mass = self.mass(particle_id)
            width = self.width(particle_id)
        imaginary = _imaginary_unit_for((*value, *momentum, mass, width))
        if metadata.kind == "scalar":
            if len(value) != 1:
                raise ValueError("scalar propagator expects one current component")
            denominator = (
                _minkowski_square_expression(momentum)
                - mass * mass
                + imaginary * mass * width
            )
            return (imaginary * value[0] / denominator,)
        if metadata.kind == "dirac-fermion":
            components = tuple(value)
            current_momentum = tuple(momentum)
            return (
                _expr_antifermion_propagator_dirac(
                    components,
                    current_momentum,
                    mass,
                    width,
                )
                if metadata.identity.orientation == "antiparticle"
                else _expr_fermion_propagator_dirac(
                    components,
                    current_momentum,
                    mass,
                    width,
                )
            )
        if metadata.kind == "vector":
            if len(value) != 4:
                raise ValueError("vector propagator expects four current components")
            current = tuple(value)
            current_momentum = tuple(momentum)
            if metadata.gauge == "feynman":
                denominator = _minkowski_square_expression(current_momentum)
                if metadata.mass_class == "massive":
                    # 't Hooft-Feynman gauge: -i g / (p^2 - M^2 + i M W).
                    denominator = denominator - mass * mass + imaginary * mass * width
                return tuple(
                    -imaginary * component / denominator for component in current
                )
            if metadata.gauge != "unitary":
                raise ValueError(
                    f"unsupported vector propagator gauge {metadata.gauge!r}"
                )
            denominator = (
                _minkowski_square_expression(current_momentum)
                - mass * mass
                + imaginary * mass * width
            )
            longitudinal = _expr_minkowski_dot(current, current_momentum) / (
                mass * mass
            )
            return tuple(
                -imaginary
                * (current[index] - current_momentum[index] * longitudinal)
                / denominator
                for index in range(4)
            )
        if metadata.kind == "spin2":
            dimension = self._runtime_parameters.get("dim", 4.0)
            return _expr_spin2_propagator(
                tuple(value),
                tuple(momentum),
                mass,
                width,
                dimension=dimension,
                massive=metadata.mass_class == "massive",
            )
        raise NotImplementedError(
            f"generic propagator kind {metadata.kind!r} is not implemented"
        )

    def runtime_parameter_names_for_vertex(self, kind: int) -> tuple[str, ...]:
        return self._kernel(kind).runtime_parameters

    def runtime_derived_parameter_definitions(self) -> dict[str, str]:
        return dict(self._runtime_derived_definitions)

    def runtime_derived_parameter_definitions_for(
        self,
        names: Sequence[str],
    ) -> dict[str, str]:
        return {
            name: self._runtime_derived_definitions[name]
            for name in names
            if name in self._runtime_derived_definitions
        }

    def runtime_derived_parameter_domains(self) -> dict[str, str]:
        """Return phases that follow exactly from UFO parameter declarations."""

        return self.runtime_derived_parameter_domains_for(
            tuple(self._runtime_derived_definitions)
        )

    def runtime_derived_parameter_domains_for(
        self,
        names: Sequence[str],
    ) -> dict[str, str]:
        """Return provable phases for only the requested derived couplings."""

        parameter_symbols = {
            self._model_symbols.symbol(parameter.name): self._runtime_domain_symbol(
                parameter.name,
                self._runtime_parameter_domain(parameter.name),
            )
            for parameter in self._parameter_records.values()
        }
        domains: dict[str, str] = {}
        for name in names:
            cached = self._runtime_derived_domain_cache.get(name)
            if cached is not None:
                domains[name] = cached
                continue
            if name not in self._runtime_derived_definitions:
                continue
            parameter = self._parameter_records.get(name)
            if parameter is not None:
                domain = self._runtime_parameter_domain(name)
                self._runtime_derived_domain_cache[name] = domain
                domains[name] = domain
                continue
            expression = self._model_symbols.expression(
                self._runtime_derived_definitions[name]
            )
            for source, replacement in parameter_symbols.items():
                expression = expression.replace(source, replacement)
            if expression.is_real():
                domain = "real"
            elif (-1j * expression).is_real():
                domain = "imaginary"
            else:
                domain = "complex"
            self._runtime_derived_domain_cache[name] = domain
            domains[name] = domain
        return domains

    def _runtime_parameter_domain(
        self,
        name: str,
        visiting: frozenset[str] = frozenset(),
    ) -> str:
        cached = self._runtime_parameter_domain_cache.get(name)
        if cached is not None:
            return cached
        parameter = self._parameter_records[name]
        if parameter.parameter_type.lower() == "real":
            domain = "real"
        elif parameter.nature.lower() != "internal" or name in visiting:
            domain = "complex"
        else:
            expression = self._model_symbols.expression(parameter.resolved_expression)
            symbols = set(expression.get_all_symbols(False))
            nested_visiting = visiting | {name}
            for dependency in self._parameter_records.values():
                source = self._model_symbols.symbol(dependency.name)
                if source not in symbols:
                    continue
                replacement = self._runtime_domain_symbol(
                    dependency.name,
                    self._runtime_parameter_domain(
                        dependency.name,
                        nested_visiting,
                    ),
                )
                expression = expression.replace(source, replacement)
            if expression.is_real():
                domain = "real"
            elif (-1j * expression).is_real():
                domain = "imaginary"
            else:
                domain = "complex"
        self._runtime_parameter_domain_cache[name] = domain
        return domain

    def _runtime_domain_symbol(self, name: str, domain: str):
        domain_symbol = symbols.runtime_domain(
            self.name,
            name,
            domain,
        )
        return domain_symbol

    def auxiliary_kind(self, particle_id: int) -> str | None:
        return self._particle_records_by_pdg[int(particle_id)].auxiliary_kind

    def runtime_derived_parameter_defaults(self) -> dict[str, complex]:
        return self.runtime_derived_parameter_defaults_for(
            tuple(self._runtime_derived_definitions)
        )

    def runtime_derived_parameter_defaults_for(
        self,
        names: Sequence[str],
    ) -> dict[str, complex]:
        substitutions: dict[_sym.Expression, Any] | None = None
        defaults: dict[str, complex] = {}
        for name in names:
            cached = self._runtime_derived_default_cache.get(name)
            if cached is not None:
                defaults[name] = cached
                continue
            expression = self._runtime_derived_definitions.get(name)
            if expression is None:
                continue
            template = self._runtime_derived_expression_cache.get(name)
            if template is None:
                template = self._model_symbols.expression(expression)
                self._runtime_derived_expression_cache[name] = template
            if substitutions is None:
                substitutions = {
                    self._model_symbols.symbol(parameter_name): value
                    for parameter_name, value in self._runtime_parameters.items()
                }
            try:
                value = complex(template.evaluate(substitutions))
            except Exception:
                value = complex(_replace_symbols(template, substitutions))
            self._runtime_derived_default_cache[name] = value
            defaults[name] = value
        return defaults

    def runtime_parameter_names_for_particle(self, pdg: int) -> tuple[str, ...]:
        particle = self._particle_records_by_pdg[int(pdg)]
        custom = self._propagator_record(pdg)
        if custom is not None and custom.custom:
            return self._custom_propagator_expression(int(pdg))[2]
        return tuple(
            name
            for name in (particle.mass, particle.width)
            if name.upper() != "ZERO"
            and (
                name in self.compiled.parameter_defaults
                or name in self._runtime_derived_definitions
            )
        )

    def _custom_propagator_expression(
        self,
        particle_id: int,
    ) -> tuple[_sym.Expression, _sym.Expression, tuple[str, ...]]:
        """Numerator, scalar denominator and runtime parameters of a propagator.

        Numerator and denominator are normalised separately: the denominator
        is a Lorentz scalar, and a negative power of a momentum square is not
        a valid tensor network (e.g. the squared denominator of a line with a
        counterterm insertion). Derived model parameters (e.g. ``G``) stay
        symbols, listed with the external ones as the propagator's runtime
        parameters.
        """

        cached = self._custom_propagator_expressions.get(int(particle_id))
        if cached is not None:
            return cached
        record = self._propagator_record(particle_id)
        if record is None or not record.custom:
            raise ValueError(f"particle {particle_id} has no custom UFO propagator")
        spin = self.spin(particle_id)
        numerator, denominator = (
            self._resolved_parameters(
                self._model_symbols.expression(
                    normalize_lorentz_expression(
                        source,
                        (spin, spin),
                        model_symbols=self._model_symbols,
                        vertex=False,
                    ).expression
                )
            )
            for source in (record.numerator, record.denominator)
        )
        symbols = set(numerator.get_all_symbols(False)) | set(
            denominator.get_all_symbols(False)
        )
        names = tuple(
            sorted(
                parameter.name
                for parameter in self._parameter_records.values()
                if (
                    parameter.nature == "external"
                    or parameter.name in self._runtime_derived_definitions
                )
                and self._model_symbols.symbol(parameter.name) in symbols
            )
        )
        result = numerator, denominator, names
        self._custom_propagator_expressions[int(particle_id)] = result
        return result

    def _resolved_parameters(self, expression: _sym.Expression) -> _sym.Expression:
        """Internal constants resolved; derived parameters stay runtime inputs.

        Derived parameters are evaluated exactly at every precision, whereas a
        compiled propagator kernel rounds constants such as pi to binary64.
        """

        for parameter in self._parameter_records.values():
            if (
                parameter.nature == "external"
                or parameter.name in self._runtime_derived_definitions
            ):
                continue
            expression = expression.replace(
                self._model_symbols.symbol(parameter.name),
                self._model_symbols.expression(parameter.resolved_expression),
            )
        return expression

    def _custom_propagator_template(
        self,
        particle_id: int,
    ) -> tuple[_sym.Expression, ...]:
        cached = self._custom_propagator_templates.get(int(particle_id))
        if cached is not None:
            return cached
        numerator, denominator, _names = self._custom_propagator_expression(particle_id)
        template = custom_propagator_components(
            self._model_symbols,
            self.name,
            self.spin(particle_id),
            self.dimension(particle_id),
            numerator,
            denominator,
        )
        self._custom_propagator_templates[int(particle_id)] = template
        return template

    def _evaluate_custom_propagator(
        self,
        particle_id: int,
        value: Sequence[Any],
        momentum: Sequence[Any],
    ) -> tuple[Any, ...]:
        if len(value) != self.dimension(particle_id):
            raise ValueError(
                f"custom propagator for {particle_id} expects "
                f"{self.dimension(particle_id)} current components"
            )
        if len(momentum) != 4:
            raise ValueError("UFO custom propagators require four-momentum components")
        substitutions = {
            **{
                symbols.custom_propagator_component(
                    self.name,
                    "input",
                    index,
                ): component
                for index, component in enumerate(value)
            },
            **{
                symbols.custom_propagator_component(
                    self.name,
                    "momentum",
                    index,
                ): component
                for index, component in enumerate(momentum)
            },
            **{
                self._model_symbols.symbol(name): self._parameter_value(name)
                for name in self._custom_propagator_expression(particle_id)[2]
            },
        }
        return tuple(
            _replace_symbols(component, substitutions)
            for component in self._custom_propagator_template(particle_id)
        )

    def runtime_mass_parameter_name(self, pdg: int) -> str | None:
        name = self._particle_records_by_pdg[int(pdg)].mass
        return (
            name
            if name in self.compiled.parameter_defaults
            or name in self._runtime_derived_definitions
            else None
        )

    def runtime_width_parameter_name(self, pdg: int) -> str | None:
        name = self._particle_records_by_pdg[int(pdg)].width
        return (
            name
            if name in self.compiled.parameter_defaults
            or name in self._runtime_derived_definitions
            else None
        )

    def runtime_parameter_defaults(self) -> dict[str, tuple[float, float]]:
        return dict(self.compiled.parameter_defaults)


def _explicit_momentum_squares(
    expression: _sym.Expression,
    model_symbols: Any,
    square: _sym.Expression,
) -> _sym.Expression:
    """Self-contracted momentum powers ``p(mu)^(2k)`` as ``square^k``.

    Normalisation writes ``p.p`` as ``p(mu)^2`` with a contracted label, and
    canonical relabelling merges two such squares of one term into
    ``p(mu)^4``, which is no valid tensor network; both legs of a propagator
    carry the same momentum, whose square is the scalar ``square``.
    """

    odd: list[str] = []

    def power(match: Mapping[_sym.Expression, _sym.Expression]) -> _sym.Expression:
        exponent = int(str(match[_sym.E("n_")]))
        if exponent % 2:
            odd.append(str(exponent))
        return square ** (exponent // 2)

    for leg in (1, 2):
        name = model_symbols.ufo_momentum_tensor_name(leg)
        expression = expression.replace(_sym.E(f"{name}(a_)^n_"), power)
    if odd:
        raise ValueError(f"odd powers {odd} of a propagator momentum")
    return expression


def custom_propagator_components(
    model_symbols: Any,
    model_name: str,
    spin: int,
    dimension: int,
    numerator: _sym.Expression,
    denominator: _sym.Expression,
) -> tuple[_sym.Expression, ...]:
    """Components of a normalised propagator ``numerator / denominator``.

    Component ``i`` is the propagated current, linear in the input components
    ``custom_propagator::input::c{j}`` and a function of the momentum
    components ``custom_propagator::momentum::c{mu}`` (the momentum of both
    propagator legs). The denominator must be a Lorentz scalar.
    """

    components = tuple(
        symbols.custom_propagator_component(model_name, "input", index)
        for index in range(dimension)
    )
    momenta = tuple(
        symbols.custom_propagator_component(model_name, "momentum", index)
        for index in range(4)
    )
    # Both networks run on canonical symbols (``_execute_dense_network``).
    library = DenseTensors()
    representations = _spin_representations(spin)
    if representations:
        name = symbols.custom_propagator_tensor_name(model_name)
        library.register_dense(name, representations, components)
        numerator *= _sym.TensorName(name)(*_spin_slots(spin, 1)).to_expression()
    else:
        numerator *= components[0]
    minkowski = _sym.Representation.mink(4)
    square = momenta[0] ** 2 - momenta[1] ** 2 - momenta[2] ** 2 - momenta[3] ** 2
    numerator, denominator = (
        _explicit_momentum_squares(part, model_symbols, square)
        for part in (numerator, denominator)
    )
    for leg in (1, 2):
        library.register_dense(
            model_symbols.ufo_momentum_tensor_name(leg), (minkowski,), momenta
        )
    tensor_components = _execute_dense_network(
        numerator, library, _spin_axis_labels(spin, 2)
    ).values
    (denominator,) = _execute_dense_network(denominator, library, ()).values
    if len(tensor_components) != dimension:
        raise ValueError(
            f"custom propagator of {model_name!r} produced {len(tensor_components)} "
            f"components, expected {dimension}"
        )
    return tuple(
        _as_expression(component) / denominator for component in tensor_components
    )
