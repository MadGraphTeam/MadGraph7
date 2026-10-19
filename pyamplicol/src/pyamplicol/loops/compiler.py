# SPDX-License-Identifier: 0BSD
"""Internal generic source-model/request compiler, separate from public service.

Consumes the existing compiled-model raw sidecar or an explicit raw model. It
never selects a hardcoded process, uses binary64 parameter defaults, or creates
another artifact identity. The public service in generation.loop_service selects
the published feature scope and writes the portable complete runtime plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pyamplicol.models.loop_import import RawLoopModel

from .counterterms import prepare_counterterm_model
from .covariant_model import build_covariant_structure
from .model import build_loop_model
from .model_catalogue import certified_source_counterterm_bases
from .model_restrictions import prepare_loop_restrictions
from .physical_source import source_model_program
from .request_scope import prepare_loop_request_scope


@dataclass(frozen=True, slots=True)
class PreparedLoopRequestModel:
    compiled_model: Any = field(repr=False)
    original: RawLoopModel = field(repr=False)
    declarations: Any = field(repr=False)
    structure: Any = field(repr=False)
    model: Any = field(repr=False)
    counterterms: Any = field(repr=False)
    scope: Any

    def compile(self, incoming, outgoing, *, selected_corrections, **options):
        """Assemble a complete request within this prepared order scope.

        This internal method retains existing active tensor, Laurent, field and
        source-owner gates. Unsupported physical paths fail explicitly; a local
        basis coverage proof never substitutes for those downstream checks.
        """
        born = options.pop("born_order_bounds", self.scope.born_order_bounds)
        loop = options.pop("loop_order_bounds", self.scope.loop_order_bounds)
        self.scope.require_bounds(born, loop)
        if self.compiled_model is not None:
            if "source_contracts" in options:
                raise ValueError(
                    "compiled source contracts are owned by the compiled model"
                )
            from pyamplicol.models import CompiledUFOModel

            from .request import prepare_loop_external_request

            source_model = CompiledUFOModel(self.compiled_model)
            contracts = {}
            external = prepare_loop_external_request(
                self.structure.native, incoming, outgoing
            )
            particles = {p.pdg_code: p for p in self.compiled_model.ir.particles}
            for pdg in dict.fromkeys(external.external_pdgs):
                native = self.structure.native.particle_by_pdg(pdg)
                if native.spin != 3:
                    continue
                particle = particles[pdg]
                if (
                    native.spin != particle.spin
                    or native.color != source_model.color_rep(particle.pdg_code)
                    or native.antiparticle.pdg_code
                    != source_model.anti_particle(particle.pdg_code)
                ):
                    raise ValueError(
                        "loop and compiled source particle contracts differ"
                    )
                contracts[particle.pdg_code] = source_model._source_ir(
                    particle.pdg_code
                )
            options["source_contracts"] = contracts
        return source_model_program(
            self.structure,
            self.counterterms,
            incoming,
            outgoing,
            selected_corrections=selected_corrections,
            born_order_bounds=born,
            loop_order_bounds=loop,
            **options,
        )


def prepare_loop_request_model(
    model,
    *,
    born_order_bounds=None,
    loop_order_bounds=None,
    simplify=None,
    fixed_parameters=None,
    compatibility=None,
    covariant_multiplets=None,
):
    """Prepare source declarations once for model-driven process compilation.

    A CompiledModel contributes its captured raw UFO and existing simplify
    setting. Its normalized binary64 numerical payload is never consumed. A raw
    source uses exact restriction simplification by default. Exact fixed values
    and existing explicit structural compatibility profiles retain their prior
    semantics; neither is inferred from a model or process display name.
    """
    from pyamplicol.models.loading import CompiledModel

    if isinstance(model, CompiledModel):
        owner, raw = model, model.raw_loop_model
        if raw is None:
            raise NotImplementedError(
                "compiled model has no complete raw NLO sidecar; recompile an NLO UFO"
            )
        if simplify is None:
            simplify = model.source.get("options", {}).get("simplify", True)
    elif isinstance(model, RawLoopModel):
        owner, raw = None, model
        if simplify is None:
            simplify = True
    else:
        raise TypeError(
            "loop model preparation requires a CompiledModel or RawLoopModel"
        )
    declarations = prepare_loop_restrictions(raw, simplify=simplify)
    scope = prepare_loop_request_scope(
        declarations.derived,
        born_order_bounds=born_order_bounds,
        loop_order_bounds=loop_order_bounds,
    )
    structure = build_covariant_structure(
        declarations.derived,
        request_scope=scope,
        fixed_parameters=fixed_parameters,
        compatibility=compatibility,
        covariant_multiplets=covariant_multiplets,
    )
    ir = build_loop_model(structure.raw, scalar_program=structure.scalars)
    counterterms = prepare_counterterm_model(
        ir,
        structure.native,
        entries=tuple(
            entry
            for entry in ir.kernels
            if entry.kind != "tree" and entry.key in scope.admitted_keys
        ),
        primitive_validations=certified_source_counterterm_bases(ir),
        vertex_spinor_frame="ufo",
    )
    scope.require_local_counterterms(counterterms)
    return PreparedLoopRequestModel(
        owner, raw, declarations, structure, ir, counterterms, scope
    )
