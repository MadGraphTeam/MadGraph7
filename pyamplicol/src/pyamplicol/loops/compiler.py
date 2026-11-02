# SPDX-License-Identifier: 0BSD
"""Internal generic source-model preparation of one-loop requests.

Consumes the existing compiled-model raw sidecar or an explicit raw model. It
never selects a hardcoded process, uses binary64 parameter defaults, or creates
another artifact identity. The v2 engine (``loops.v2.assemble``) prepares its
request models here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pyamplicol.models.loop_import import RawLoopModel

from .counterterms import prepare_counterterm_model
from .covariant_model import build_covariant_structure, repair_source_declarations
from .model import build_loop_model
from .model_restrictions import prepare_loop_restrictions
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
    The v2 engine lowers every counterterm through the tree compiler.
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
    # Structural source repairs (constant-tensor slot arity) of the vertices
    # the request reaches, recorded in the derived source. A source without
    # such a defect in scope, or already repaired (the public route repairs
    # the compiled model it publishes), is prepared as it is, byte for byte.
    repaired = repair_source_declarations(
        raw, within={tuple(key[:2]) for key in scope.admitted_keys}
    )
    if repaired is not raw:
        raw = repaired
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
        vertex_spinor_frame="ufo",
    )
    scope.require_local_counterterms(counterterms)
    return PreparedLoopRequestModel(
        owner, raw, declarations, structure, ir, counterterms, scope
    )
