# SPDX-License-Identifier: 0BSD
"""Export UMAMI lookup tables without changing numerical artifact contents.

Only generation certificates and exact structural keys authorize reuse.  In
particular, equal masses or parameter defaults never establish an amplitude
equivalence.  The runtime evaluates normalized physical contributions; optional
integration multiplicities live in the JSON, not in the C provider.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from .api_bundle import ApiBundlePayload

GROUPING_MODES = ("exact", "flavour_blind_observables", "none")


@dataclass(frozen=True)
class UmamiProcessInput:
    process_id: str
    expression: str
    color_accuracy: str
    external_pdgs: tuple[int, ...]
    physics: Mapping[str, Any]
    aliases: tuple[Mapping[str, Any], ...] = ()
    color_sectors: tuple[Mapping[str, Any], ...] = ()
    # Keys certify equality of the normalized, available-helicity-summed
    # squared contribution, in this process's external momentum order.
    structural_keys: Mapping[str, str] | None = None


def _records(value: Any) -> list[Mapping[str, Any]]:
    return [record for record in value or () if isinstance(record, Mapping)]


def _finite(value: Any, context: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"UMAMI {context} must be finite")
    return result


def _pair(value: Any) -> tuple[float, float]:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return _finite(value[0], "parameter"), _finite(value[1], "parameter")
    return _finite(value, "parameter"), 0.0


def _model_tables(model: Mapping[str, Any]) -> tuple[dict, dict, dict]:
    ir = model.get("ir", model.get("model", model))
    particles = {int(p["pdg_code"]): p for p in _records(ir.get("particles"))}
    parameters = {str(p["name"]): p for p in _records(ir.get("parameters"))}
    defaults = {
        str(k): _pair(v) for k, v in model.get("parameter_defaults", {}).items()
    }
    for name, parameter in parameters.items():
        if name not in defaults and parameter.get("value") is not None:
            defaults[name] = _pair(parameter["value"])
        if name not in defaults:
            with suppress(TypeError, ValueError):
                defaults[name] = _pair(parameter.get("resolved_expression"))
    return particles, parameters, defaults


def _mass(
    particle: Mapping, parameters: Mapping, defaults: Mapping
) -> tuple[str, float]:
    name = str(particle["mass"])
    if name in defaults:
        real, imaginary = defaults[name]
    else:
        try:
            real, imaginary = float(name), 0.0
        except ValueError as exc:
            # UFO's reserved ZERO is a literal, not an independent parameter.
            if name.upper() != "ZERO":
                raise ValueError(f"UMAMI cannot resolve default mass {name!r}") from exc
            real, imaginary = 0.0, 0.0
    if imaginary or real < 0.0 or not math.isfinite(real):
        raise ValueError(f"UMAMI external mass {name!r} must be nonnegative and real")
    declaration = parameters.get(name, {})
    semantic = (
        name
        if declaration.get("nature") == "external"
        else str(declaration.get("resolved_expression", name))
    )
    if not declaration and real == 0.0:
        semantic = "0"
    return semantic, real


def _alpha_s_name(model: Mapping, physics: Mapping, parameters: Mapping) -> str | None:
    public = {p["name"]: p for p in _records(physics.get("model_parameters"))}
    explicit = model.get("extensions", {}).get("alpha_s_parameter")
    candidates = (
        [str(explicit)]
        if explicit
        else [
            name
            for name, p in parameters.items()
            if str(p.get("lhablock", "")).upper() == "SMINPUTS"
            and list(p.get("lhacode", ())) == [3]
            and p.get("nature") == "external"
        ]
    )
    candidates = [name for name in candidates if public.get(name, {}).get("mutable")]
    return candidates[0] if len(candidates) == 1 else None


def _flow_id(word: Sequence[int]) -> str:
    return "flow:" + (",".join(str(label) for label in word) if word else "singlet")


def _word(sector: Mapping) -> tuple[int, ...]:
    return tuple(
        sector.get("word_labels") or next(iter(sector.get("color_words", ())), ())
    )


def _permutation(values: Sequence[int], count: int) -> tuple[int, ...]:
    result = tuple(int(i) for i in values)
    if sorted(result) != list(range(count)):
        raise ValueError("UMAMI external permutation is not a complete bijection")
    return result


def _color_flow(
    word: Sequence[int],
    external: Sequence[Mapping],
    particles: Mapping,
    sector: Mapping | None = None,
) -> list[list[int]]:
    """Assign LHE tags from all-outgoing fundamental chains/adjoint cycles."""
    labels = {int(p["label"]): i for i, p in enumerate(external)}
    roles = {}
    for label, slot in labels.items():
        p = external[slot]
        representation = int(particles[int(p["pdg"])]["color"])
        if p["role"] == "initial" and abs(representation) == 3:
            representation = -representation
        roles[label] = representation
    colored = [int(label) for label in word if roles[int(label)] != 1]
    if set(colored) != {label for label, rep in roles.items() if rep != 1}:
        raise ValueError("UMAMI LC word must contain every coloured external leg once")
    if len(colored) != len(set(colored)):
        raise ValueError("UMAMI LC word repeats an external leg")
    if any(rep not in (1, 3, -3, 8) for rep in roles.values()):
        raise ValueError(
            "UMAMI LHE tags require singlet, fundamental or adjoint colour"
        )
    result = [[0, 0] for _ in external]
    edge = 501
    if sector is not None and sector.get("kind") == "open-lines":
        lines = _records(sector.get("open_color_lines"))
        chains = [
            [
                int(line["fundamental_label"]),
                *(int(label) for label in line.get("adjoint_labels", ())),
                int(line["antifundamental_label"]),
            ]
            for line in lines
        ]
        flattened = [label for chain in chains for label in chain]
        if sorted(flattened) != sorted(colored):
            raise ValueError("UMAMI colour-sector lines do not cover the LC word")
        for chain in chains:
            if (
                roles[chain[0]] != 3
                or roles[chain[-1]] != -3
                or any(roles[label] != 8 for label in chain[1:-1])
            ):
                raise ValueError(
                    "UMAMI colour-sector endpoints disagree with the model"
                )
            result[labels[chain[0]]][0] = edge
            for label in chain[1:-1]:
                result[labels[label]] = [edge + 1, edge]
                edge += 1
            result[labels[chain[-1]]][1] = edge
            edge += 1
    elif colored and all(roles[label] == 8 for label in colored):
        for index, label in enumerate(colored):
            result[labels[label]] = [edge + index, edge + (index - 1) % len(colored)]
    else:
        current = None
        for label in colored:
            role = roles[label]
            if role == 3:
                if current is not None:
                    raise ValueError(
                        "UMAMI LC word has an unterminated fundamental chain"
                    )
                current = edge
                result[labels[label]][0] = current
            elif role == 8 and current is not None:
                result[labels[label]] = [current + 1, current]
                current += 1
            elif role == -3 and current is not None:
                result[labels[label]][1] = current
                edge = current + 1
                current = None
            else:
                raise ValueError(
                    "UMAMI LC word does not describe complete fundamental chains"
                )
        if current is not None:
            raise ValueError("UMAMI LC word has an unterminated fundamental chain")
    for index, p in enumerate(external):
        if p["role"] == "initial":
            result[index].reverse()
    return result


def _safe_replay(process: UmamiProcessInput) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Convert certified label maps; weights may fold reflection, not one flow."""
    physics = process.physics
    external = _records(physics["external_particles"])
    count = len(external)
    labels = {int(p["label"]): i for i, p in enumerate(external)}
    identity = tuple(range(count))
    colors = {str(c["id"]): c for c in _records(physics.get("color_components"))}
    result = {key: (key, identity) for key in colors}
    certificate = physics.get("extensions", {}).get("lc_topology_replay")
    if not certificate:
        # These public nontrivial mappings are certified trace reflections.
        for key, color in colors.items():
            representative = str(color.get("representative_id", key))
            if representative in colors and float(color.get("coefficient", 1.0)) == 1.0:
                result[key] = (representative, identity)
        return result
    sectors = {int(s["id"]): s for s in process.color_sectors}
    sector_flow_ids = {_flow_id(_word(s)) for s in sectors.values()}
    helicities = {tuple(h["values"]) for h in _records(physics.get("helicities"))}
    for partition in _records(certificate.get("partitions")):
        if (partition.get("proof") or {}).get("status") != "proven":
            continue
        representative = sectors.get(int(partition["representative_sector_id"]))
        if representative is None:
            continue
        rep_id = _flow_id(_word(representative))
        for sector_id, pairs in zip(
            partition["active_sector_ids"], partition["label_permutations"], strict=True
        ):
            sector = sectors.get(int(sector_id))
            if sector is None:
                continue
            mapping = dict(pairs)
            try:
                perm = _permutation(
                    [labels[int(mapping[int(p["label"])])] for p in external], count
                )
            except (KeyError, ValueError):
                continue
            # No beam/PDF symmetry is assumed. Restricted helicity coverage
            # must also be invariant under the proposed external relabelling.
            if any(
                perm[i] != i for i, p in enumerate(external) if p["role"] == "initial"
            ):
                continue
            if {tuple(h[i] for i in perm) for h in helicities} != helicities:
                continue
            word = _word(sector)
            target_ids = [_flow_id(word)]
            if sector.get("kind") == "single-trace" and len(word) > 2:
                reflected = _flow_id((word[0], *reversed(word[1:])))
                # Only an additional public flow absent from the sector plan
                # is a folded reflection. An explicit reversed sector has its
                # own proof/permutation and must not be overwritten here.
                if reflected not in sector_flow_ids:
                    target_ids.append(reflected)
            for target in target_ids:
                if target in colors and rep_id in colors:
                    result[target] = (rep_id, perm)
    return result


def _intern(table: list, value: Any) -> int:
    try:
        return table.index(value)
    except ValueError:
        table.append(value)
        return len(table) - 1


def build_umami_metadata(
    *,
    processes: Sequence[UmamiProcessInput],
    compiled_model: Mapping[str, Any],
    grouping: str = "exact",
    artifact_id: str | None = None,
) -> tuple[dict[str, Any], ...]:
    """Return separate, directly consumable metadata documents per provider."""
    if grouping not in GROUPING_MODES:
        raise ValueError(f"unknown UMAMI grouping mode {grouping!r}")
    particles, parameters, defaults = _model_tables(compiled_model)
    groups: list[tuple[tuple, list[UmamiProcessInput], list[tuple[str, float]]]] = []
    for process in processes:
        external = _records(process.physics.get("external_particles"))
        if not external or len(external) != len(process.external_pdgs):
            raise ValueError(
                f"UMAMI process {process.process_id!r} lacks resolved external metadata"
            )
        if tuple(int(p["pdg"]) for p in external) != process.external_pdgs:
            raise ValueError("UMAMI external PDGs disagree with resolved physics")
        incoming = sum(p["role"] == "initial" for p in external)
        if [p["role"] for p in external] != ["initial"] * incoming + ["final"] * (
            len(external) - incoming
        ):
            raise ValueError("UMAMI requires initial legs before final legs")
        # The public catalog has already evaluated internal/derived parameters
        # for the artifact's defaults. Reuse those values instead of adding a
        # second symbolic evaluator here (the parameter card can omit them).
        process_defaults = {
            **defaults,
            **{
                str(p["name"]): (
                    _finite(p["default_real"], "parameter"),
                    _finite(p.get("default_imaginary", 0.0), "parameter"),
                )
                for p in _records(process.physics.get("model_parameters"))
            },
        }
        masses = [
            _mass(particles[pdg], parameters, process_defaults)
            for pdg in process.external_pdgs
        ]
        helicity_domain = tuple(
            tuple(h["values"]) for h in _records(process.physics.get("helicities"))
        )
        signature = (
            len(external),
            incoming,
            process.color_accuracy,
            tuple(masses),
            helicity_domain,
        )
        # Alternative selections of the same external process are not additive.
        compatible = next(
            (
                g
                for g in groups
                if g[0] == signature
                and all(
                    p.external_pdgs[:incoming] != process.external_pdgs[:incoming]
                    or sorted(p.external_pdgs[incoming:])
                    != sorted(process.external_pdgs[incoming:])
                    for p in g[1]
                )
            ),
            None,
        )
        if compatible is None:
            groups.append((signature, [process], masses))
        else:
            compatible[1].append(process)
    documents = []
    for index, (_, members, masses) in enumerate(groups):
        provider_id = f"p{index}"
        documents.append(
            _build_provider(
                provider_id,
                members,
                masses,
                compiled_model,
                particles,
                parameters,
                grouping,
                len(groups) == 1,
            )
        )
        documents[-1]["provider"]["artifact_id"] = artifact_id
    return tuple(documents)


def _build_provider(
    provider_id, processes, masses, model, particles, parameters, grouping, single
):
    first_external = _records(processes[0].physics["external_particles"])
    count = len(first_external)
    incoming = sum(p["role"] == "initial" for p in first_external)
    identity = list(range(count))
    data: dict[str, Any] = {
        "kind": "pyamplicol-umami-metadata",
        "schema_version": 1,
        "provider": {
            "id": provider_id,
            "library": "libumami.so" if single else f"libumami_{provider_id}.so",
            "particle_count": count,
            "incoming_count": incoming,
            "masses": [m[1] for m in masses],
            "mass_parameters": [m[0] for m in masses],
            "color_accuracy": processes[0].color_accuracy,
        },
        "grouping": {
            "mode": grouping,
            "assumptions": [
                "final-state permutation/flavour-blind observables and cuts"
            ]
            if grouping == "flavour_blind_observables"
            else [],
        },
        "channels": [],
        "pdg_ids": [],
        "color_orders": [],
        "color_flows": [],
        "helicities": [],
        "runtime_processes": [],
        "aliases": [],
        "xml_header": (
            '<generator name="pyAmpliCol">UMAMI artifact provider</generator>'
        ),
    }
    structural_representatives: dict[str, dict] = {}
    orbit_entries: dict[tuple, dict] = {}
    channel_lookup = {}
    physical_count = 0
    for process_index, process in enumerate(processes):
        physics = process.physics
        external = _records(physics["external_particles"])
        labels = {int(p["label"]): i for i, p in enumerate(external)}
        alpha_s = _alpha_s_name(model, physics, parameters)
        helicities = _records(physics.get("helicities"))
        data["runtime_processes"].append(
            {
                "id": process.process_id,
                "expression": process.expression,
                "pdgs": list(process.external_pdgs),
                "color_accuracy": process.color_accuracy,
                "alpha_s_parameter": alpha_s,
                "parameters": [
                    {
                        **p,
                        "is_complex": (
                            parameters.get(p["name"], {}).get("parameter_type")
                            == "complex"
                            or p.get("kind") == "coupling"
                            or p.get("default_imaginary", 0.0) != 0.0
                        ),
                    }
                    for p in _records(physics.get("model_parameters"))
                ],
                "helicity_ids": [h["id"] for h in helicities],
                "coverage": dict(physics.get("coverage", {})),
            }
        )
        data["aliases"].extend(
            {"process_id": process.process_id, **dict(a)} for a in process.aliases
        )
        helicity_groups: dict[str, list[int]] = {}
        helicity_mappings = []
        for h in helicities:
            hindex = _intern(data["helicities"], list(h["values"]))
            key = str(
                h["id"] if grouping == "none" else h.get("representative_id", h["id"])
            )
            helicity_groups.setdefault(key, []).append(hindex)
            helicity_mappings.append(
                {
                    "helicity": hindex,
                    "representative_id": h.get("representative_id", h["id"]),
                    "coefficient": h.get("coefficient", 1.0),
                    "structural_zero": h.get("structural_zero", False),
                }
            )
        replay = _safe_replay(process) if grouping != "none" else {}
        sectors_by_word = {_word(sector): sector for sector in process.color_sectors}
        colors = _records(physics.get("color_components"))
        for color in colors:
            physical_count += 1
            color_id = str(color["id"])
            contracted = process.color_accuracy != "lc"
            word = tuple(int(v) for v in color.get("word", ()))
            order = [labels[label] for label in word]
            order += [i for i in identity if i not in order]
            order = list(_permutation(order, count))
            color_index = (
                None
                if contracted
                else _intern(
                    data["color_flows"],
                    _color_flow(word, external, particles, sectors_by_word.get(word)),
                )
            )
            rep_color, perm = replay.get(color_id, (color_id, tuple(identity)))
            runtime = {
                "process_index": process_index,
                "process_id": process.process_id,
                "color_id": None if contracted else rep_color,
                "color_index": 0 if color_index is None else color_index,
                "permutation": list(perm),
                "factor": 1.0,
            }
            semantic_key = (process.structural_keys or {}).get(color_id)
            if grouping != "none" and semantic_key:
                if semantic_key in structural_representatives:
                    runtime = dict(structural_representatives[semantic_key])
                else:
                    # The proof key already includes normalization and colour
                    # contraction; exact cross-flavour equality needs no cuts.
                    structural_representatives[semantic_key] = dict(runtime)
            member = {
                "process_id": process.process_id,
                "pdgs": list(process.external_pdgs),
                "color_id": None if contracted else color_id,
                "runtime": dict(runtime),
                "factor": 1.0,
            }
            orbit = (
                runtime["process_index"],
                runtime["color_id"],
                tuple(process.external_pdgs[:incoming]),
            )
            if grouping == "flavour_blind_observables" and orbit in orbit_entries:
                entry = orbit_entries[orbit]
                # Keep distinct incoming PDGs available to PDF weighting.
                pdg_index = _intern(data["pdg_ids"], list(process.external_pdgs))
                term = next(
                    (m for m in entry["matrix_elements"] if m["pdg_ids"] == pdg_index),
                    None,
                )
                if term is None:
                    entry["matrix_elements"].append(
                        {"pdg_ids": pdg_index, "factor": 1.0}
                    )
                else:
                    term["factor"] += 1.0
                entry["members"].append(member)
                continue
            pdg_index = _intern(data["pdg_ids"], list(process.external_pdgs))
            entry = {
                "matrix_elements": [{"pdg_ids": pdg_index, "factor": 1.0}],
                "color_order": _intern(data["color_orders"], order),
                "color_flows": color_index,
                "helicities": list(helicity_groups.values()),
                "helicity_mappings": helicity_mappings,
                "runtime": runtime,
                "members": [member],
                "multichannels": [],
            }
            # A single complete map per contribution. Maps are shared only if
            # their fixed external masses agree (already a provider invariant).
            channel_key = tuple(order)
            if channel_key not in channel_lookup:
                channel_lookup[channel_key] = len(data["channels"])
                data["channels"].append({"phasespace_order": order, "processes": []})
            channel_index = channel_lookup[channel_key]
            entry["multichannels"] = [channel_index]
            data["channels"][channel_index]["processes"].append(entry)
            orbit_entries[orbit] = entry
    alpha_names = {p["alpha_s_parameter"] for p in data["runtime_processes"]}
    data["provider"]["alpha_s_parameter"] = (
        next(iter(alpha_names)) if len(alpha_names) == 1 else None
    )
    data["provider"]["supports_alpha_s"] = all(name is not None for name in alpha_names)
    data["provider"]["helicity_count"] = max(
        (len(p["helicity_ids"]) for p in data["runtime_processes"]), default=0
    )
    data["provider"]["color_count"] = max(1, len(data["color_flows"]))
    data["grouping"]["physical_contributions"] = physical_count
    data["grouping"]["exported_contributions"] = sum(
        len(c["processes"]) for c in data["channels"]
    )
    return data


def _json_payload(path: str, value: Any) -> ApiBundlePayload:
    return ApiBundlePayload(
        path,
        (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
        "sdk-metadata",
        "application/json",
    )


def umami_bundle_payloads(
    *,
    processes: Sequence[UmamiProcessInput],
    compiled_model: Mapping[str, Any],
    grouping: str = "exact",
    artifact_id: str | None = None,
) -> tuple[ApiBundlePayload, ...]:
    """Create JSON, provider headers and Make variables from one representation."""
    from .umami_c import provider_header

    documents = build_umami_metadata(
        processes=processes,
        compiled_model=compiled_model,
        grouping=grouping,
        artifact_id=artifact_id,
    )
    payloads = []
    entries = []
    for data in documents:
        provider = data["provider"]
        identifier = provider["id"]
        relative = (
            "metadata.json"
            if len(documents) == 1
            else f"providers/{identifier}/metadata.json"
        )
        payloads.append(_json_payload(f"API/umami/{relative}", data))
        entries.append(
            {"id": identifier, "metadata": relative, "library": provider["library"]}
        )
        payloads.append(
            ApiBundlePayload(
                f"API/umami/provider_{identifier}.h",
                provider_header(data).encode(),
                "api-source",
                "text/x-chdr",
            )
        )
    if len(documents) != 1:
        payloads.append(
            _json_payload(
                "API/umami/metadata.json",
                {
                    "kind": "pyamplicol-umami-provider-index",
                    "schema_version": 1,
                    "providers": entries,
                },
            )
        )
    make = "UMAMI_PROVIDERS := " + " ".join(p["id"] for p in entries) + "\n"
    make += f"UMAMI_SINGLE_PROVIDER := {int(len(documents) == 1)}\n"
    payloads.append(
        ApiBundlePayload(
            "API/umami/providers.mk", make.encode(), "api-build-file", "text/x-makefile"
        )
    )
    return tuple(payloads)


__all__ = [
    "GROUPING_MODES",
    "UmamiProcessInput",
    "build_umami_metadata",
    "umami_bundle_payloads",
]
