# SPDX-License-Identifier: 0BSD
"""Conservative, model-independent equality keys for UMAMI representatives.

These keys describe the executed recurrence circuit, not its particle names or
its values at sample points.  Local callable expressions are interned after
binding their inputs; mutable parameter names remain free symbols.  Different
arena layouts can miss an equality, which is preferable to guessing one.
This export-only analysis never modifies the numerical schedule.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from .._internal.physics.symbols import ModelSymbolRegistry

if TYPE_CHECKING:
    from ..models.contracts import CompiledModelIR
    from ..models.prepared import PreparedKernelPack, PreparedKernelRecord
    from .recurrence_schedule_sharing import RecurrenceProcessRemap

_TOKEN = re.compile(r"[A-Za-z_][A-Za-z_0-9]*(?:::[A-Za-z_][A-Za-z_0-9]*)*")
_NONE = (1 << 32) - 1


def _key(value: object) -> str:
    digest = hashlib.sha256()
    for chunk in json.JSONEncoder(sort_keys=True, separators=(",", ":"), allow_nan=False).iterencode(value):
        digest.update(chunk.encode("utf-8"))
    return digest.hexdigest()


class ParameterSemantics:
    """Resolve definitions, never substitute the defaults of external inputs."""

    def __init__(self, ir: CompiledModelIR | Mapping[str, object]):
        payload = ir if isinstance(ir, Mapping) else ir.to_dict()
        self._parameters = {str(p["name"]): p for p in _objects(payload["parameters"])}
        self._couplings = {str(c["name"]): c for c in _objects(payload["couplings"])}
        registry = ModelSymbolRegistry(str(payload["name"]))
        self._names = {
            spelling: name
            for name in (*self._parameters, *self._couplings)
            for spelling in (name, registry.qualified_name(name))
        }
        self._cache: dict[str, object] = {}
        self._active: set[str] = set()

    def parameter(self, name: str) -> object:
        if name in self._cache:
            return self._cache[name]
        if name in self._active:
            raise ValueError("cyclic symbolic parameter definition")
        self._active.add(name)
        try:
            parameter = self._parameters.get(name)
            coupling = self._couplings.get(name)
            if parameter is not None:
                if parameter["nature"] == "external":
                    value: object = ("external", name, parameter["parameter_type"])
                elif parameter.get("expression") is not None:
                    value = self.expression(str(parameter["expression"]))
                else:
                    value = ("constant", parameter["parameter_type"], parameter.get("value"))
            elif coupling is not None:
                value = self.expression(str(coupling["expression"]))
            else:
                # Runtime-owned parameters retain their actual names.  There
                # is no model-specific interpretation or default-value match.
                value = ("runtime-parameter", name)
            self._cache[name] = value
            return value
        finally:
            self._active.remove(name)

    def expression(self, expression: str, bindings: Mapping[str, object] | None = None) -> object:
        local = bindings or {}
        pieces: list[object] = []
        end = 0
        for match in _TOKEN.finditer(expression):
            pieces.append(expression[end:match.start()])
            token = match.group()
            if token in local:
                pieces.append(("input", local[token]))
            elif token in self._names:
                pieces.append(("parameter", self.parameter(self._names[token])))
            else:
                pieces.append(token)
            end = match.end()
        pieces.append(expression[end:])
        return ("expression", tuple(pieces))


def _kernel_key(kernel: PreparedKernelRecord, parameters: ParameterSemantics,
                parameter_names: Mapping[int, str], coupling: object,
                parent_permutation: tuple[int, int]) -> str:
    bindings: dict[str, object] = {}
    for contract in kernel.input_contracts:
        role = str(contract["role"])
        component = int(contract["component"])
        if role == "model-parameter":
            name = parameter_names[int(contract["model_parameter_index"])]
            value: object = (role, parameters.parameter(name), component)
        elif role in {"coupling-real", "coupling-imag"}:
            value = (role, coupling, component)
        else:
            value = (role, component)
        bindings[str(contract["symbol"])] = value
    return _key((kernel.contract_kind, kernel.output_layout, parent_permutation,
                 [parameters.expression(expr, bindings) for expr in kernel.exact_expressions]))


def recurrence_structural_keys(
    *, exact_sections: Mapping[str, object], runtime_metadata: Mapping[str, object],
    kernel_pack: PreparedKernelPack, ir: CompiledModelIR | Mapping[str, object],
    physics: Mapping[str, object], remap: RecurrenceProcessRemap,
    color_contraction_payload: bytes | None,
) -> dict[str, str]:
    """Prove equality only for fully comparable, unremapped exact circuits.

The same ordered instruction graph and fully bound local operations imply the
same amplitudes for every momentum/parameter value.  Colour contraction and
normalization are included, so equality is of the complete squared observable.
Unknown plans return no key; ordinary generation/evaluation stays available.
"""
    if exact_sections.get("abi") != "pyamplicol-recurrence-exact-sections-v1":
        return {}
    if (remap.source_slots != tuple(range(len(remap.source_slots)))
        or any(v != 1 for v in (*remap.source_momentum_signs, *remap.source_helicity_signs))
        or remap.state_template_changes or remap.source_template_changes
        or remap.direct_executor_changes or remap.parameter_slot_changes
        or any(
            remap.source_state_indices[start:end] != tuple(range(end - start))
            for start, end in zip(remap.source_state_offsets, remap.source_state_offsets[1:])
        )
        or remap.public_flow_ids != tuple(range(len(remap.public_flow_ids)))
        or remap.physical_sector_ids != tuple(range(len(remap.physical_sector_ids)))):
        return {}
    catalog = kernel_pack.recurrence_template_catalog
    direct_catalog = kernel_pack.recurrence_direct_template_catalog
    if catalog is None or direct_catalog is None:
        return {}
    # Reuse the exact executor's authoritative bindings, including intrinsics
    # whose algebraic kernel is distinct from their native callable payload.
    from ..runtime.recurrence_exact._plan import (
        _executor_couplings, _executor_exact_kernel_ids, _executor_parent_permutations,
    )
    try:
        exact_ids = _executor_exact_kernel_ids(kernel_pack)
        couplings = _executor_couplings(kernel_pack, exact_ids)
        parent_orders = _executor_parent_permutations(kernel_pack)
        parameter_names = {p.prepared_parameter_id: p.name for p in catalog.parameters
                           if p.prepared_parameter_id is not None}
        parameters = ParameterSemantics(ir)
        kernels = {k.kernel_id: k for k in kernel_pack.kernels}
        executor_keys: dict[int, str] = {}
        for row in _rows(exact_sections["executors"]):
            executor_id = int(row[0])
            kernel_id = exact_ids.get(executor_id)
            expression = (_kernel_key(kernels[kernel_id], parameters, parameter_names,
                          tuple(str(v) for v in couplings.get(executor_id, ())),
                          parent_orders.get(executor_id, (0, 1)))
                          if kernel_id is not None else ("intrinsic", row[7]))
            if kernel_id is None and row[7] is None:
                return {}
            executor_keys[executor_id] = _key((row[1:6], expression))
        source_keys: dict[int, str] = {}
        for item in _objects(runtime_metadata["source_templates"]):
            source_ir = _mapping(item["source_ir"])
            identity = _mapping(source_ir["identity"])
            mass_name = source_ir.get("mass_parameter")
            source_keys[int(item["source_template_id"])] = _key((
                item["dimension"], item["helicity"], item["chirality"], item["spin_state"],
                source_ir["wavefunction_family"], identity["orientation"], item["crossing"],
                source_ir.get("basis"),
                None if source_ir.get("width_parameter") is None else parameters.parameter(str(source_ir["width_parameter"])),
                None if mass_name is None else parameters.parameter(str(mass_name)),
            ))
        program = {k: v for k, v in exact_sections.items()
                   if k not in {"process_id", "semantic_digest", "runtime_layout_digest", "executors"}}
        # Current IDs and state-template IDs are diagnostic; the actual arena
        # ranges, stage clearing, bound source operations and kernels remain.
        program["currents"] = [[*r[:2], None, *r[3:]] for r in _rows(program["currents"])]
        program["row_groups"] = [[*r[:3], None if r[3] == _NONE else executor_keys[int(r[3])], *r[4:]]
                                 for r in _rows(program["row_groups"])]
        if program["strategy"] != "all-flow-union":
            program["sources"] = [[*r[:3], source_keys[int(r[3])], *r[4:]]
                                  for r in _rows(program["sources"])]
        program["source_dispatch_variants"] = [
            [*r[:6], source_keys[int(r[6])], None, None, r[9],
             None if r[10] == _NONE else executor_keys[int(r[10])], *r[11:]]
            for r in _rows(program["source_dispatch_variants"])
        ]
        # Keep all normalization dependencies, not just their default number.
        normalization = runtime_metadata["normalization"]
        external_layout = [(v["source_slot"], v["is_initial"])
                           for v in _objects(runtime_metadata["external_legs"])]
        parameter_projection = [(v["runtime_slot"], parameters.parameter(str(v["runtime_name"])), v["component"])
                                for v in _objects(runtime_metadata["parameter_projection"])]
        color_key = None if color_contraction_payload is None else hashlib.sha256(color_contraction_payload).hexdigest()
        circuit = _key((program, normalization, external_layout, parameter_projection,
                        physics["color_accuracy"], color_key,
                        [h["values"] for h in _objects(physics["helicities"])]))
        return {str(c["id"]): _key((circuit, c["index"], c.get("coefficient", 1.0)))
                for c in _objects(physics["color_components"])}
    except (KeyError, TypeError, ValueError, IndexError):
        # Unsupported/missing semantic data cannot authorize grouping.
        return {}


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("semantic record must be a mapping")
    return value


def _sequence(value: object) -> Sequence[object]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise TypeError("semantic records must be a sequence")
    return value


def _rows(value: object) -> tuple[Sequence[object], ...]:
    return tuple(_sequence(v) for v in _sequence(value))


def _objects(value: object) -> tuple[Mapping[str, object], ...]:
    return tuple(_mapping(v) for v in _sequence(value))
