# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from pyamplicol.artifacts.umami import (
    UmamiProcessInput,
    build_umami_metadata,
    umami_bundle_payloads,
)
from pyamplicol.artifacts.umami_otf import build_umami_on_the_fly_physics
from pyamplicol.processes.ir import (
    CanonicalProcessIR,
    ColorEndpointSummary,
    ProcessLegIR,
)


def model():
    # Deliberately non-SM particle/parameter names and PDGs.
    return {
        "ir": {
            "particles": [
                {"pdg_code": 990, "color": 8, "mass": "ZERO"},
                {"pdg_code": 991, "color": 3, "mass": "ZERO"},
                {"pdg_code": -991, "color": -3, "mass": "ZERO"},
                {"pdg_code": 992, "color": 3, "mass": "ZERO"},
                {"pdg_code": -992, "color": -3, "mass": "ZERO"},
                {"pdg_code": 995, "color": 1, "mass": "mass_s"},
            ],
            "parameters": [
                {
                    "name": "strong_input",
                    "nature": "external",
                    "parameter_type": "real",
                    "lhablock": "SMINPUTS",
                    "lhacode": [3],
                    "value": [0.12, 0],
                },
                {"name": "mass_s", "nature": "external", "value": [7, 0]},
            ],
        },
        "parameter_defaults": {"strong_input": [0.12, 0], "mass_s": [7, 0]},
    }


def process(
    identifier="p", pdgs=(990, 990, 990, 990), accuracy="lc", words=((1, 2, 3, 4),)
):
    helicities = [
        {
            "id": "h:all-plus",
            "values": [1] * len(pdgs),
            "representative_id": "h:all-plus",
            "coefficient": 1,
            "structural_zero": False,
        }
    ]
    colors = [
        {
            "id": "flow:" + ",".join(map(str, word)),
            "word": list(word),
            "representative_id": "flow:" + ",".join(map(str, word)),
            "coefficient": 1,
        }
        for word in words
    ]
    if accuracy != "lc":
        colors = [{"id": "color:contracted", "kind": "contracted-color"}]
    physics = {
        "external_particles": [
            {
                "pdg": pdg,
                "label": i + 1,
                "index": i,
                "role": "initial" if i < 2 else "final",
            }
            for i, pdg in enumerate(pdgs)
        ],
        "helicities": helicities,
        "color_components": colors,
        "coverage": {
            "helicities": "selected",
            "color": "complete" if accuracy == "lc" else "contracted",
        },
        "model_parameters": [
            {
                "name": "strong_input",
                "kind": "external",
                "default_real": 0.12,
                "default_imaginary": 0,
                "mutable": True,
            }
        ],
        "extensions": {},
    }
    return UmamiProcessInput(identifier, identifier, accuracy, tuple(pdgs), physics)


def export(*processes, grouping="exact", compiled_model=None):
    return build_umami_metadata(
        processes=processes, compiled_model=compiled_model or model(), grouping=grouping
    )


def entries(document):
    return [p for channel in document["channels"] for p in channel["processes"]]


def with_replay():
    p = process(words=((1, 2, 3, 4), (1, 2, 4, 3)))
    sectors = (
        {"id": 0, "kind": "single-trace", "word_labels": [1, 2, 3, 4]},
        {"id": 1, "kind": "single-trace", "word_labels": [1, 2, 4, 3]},
    )
    p.physics["extensions"]["lc_topology_replay"] = {
        "partitions": [
            {
                "representative_sector_id": 0,
                "active_sector_ids": [0, 1],
                "label_permutations": [
                    [[1, 1], [2, 2], [3, 3], [4, 4]],
                    [[1, 1], [2, 2], [3, 4], [4, 3]],
                ],
                "replay_weights": [
                    2,
                    2,
                ],  # Folded runtime weights are not physical-flow multiplicities.
                "proof": {"status": "proven"},
            }
        ]
    }
    return replace(p, color_sectors=sectors)


def test_generic_gluon_tables_have_physical_pdgs_and_complete_permutations():
    (data,) = export(process())
    assert data["pdg_ids"] == [[990, 990, 990, 990]]
    assert data["provider"]["masses"] == [0] * 4
    assert data["provider"]["alpha_s_parameter"] == "strong_input"
    assert data["provider"]["supports_alpha_s"]
    assert sorted(data["channels"][0]["phasespace_order"]) == list(range(4))
    assert data["color_flows"][0] == [[504, 501], [501, 502], [503, 502], [504, 503]]
    assert entries(data)[0]["matrix_elements"] == [{"pdg_ids": 0, "factor": 1}]


def test_quark_chain_and_singlet_derive_from_model_not_particle_names():
    (data,) = export(process(pdgs=(991, -991, 995, 990), words=((2, 4, 1),)))
    assert data["provider"]["masses"] == [0, 0, 7, 0]
    assert data["color_flows"] == [[[502, 0], [0, 501], [0, 0], [502, 501]]]
    assert data["color_orders"] == [[1, 3, 0, 2]]


@pytest.mark.parametrize("accuracy", ["nlc", "full"])
def test_contracted_color_has_no_fake_flows(accuracy):
    (data,) = export(process(accuracy=accuracy))
    assert data["color_flows"] == []
    assert entries(data)[0]["color_flows"] is None
    assert entries(data)[0]["runtime"]["color_id"] is None


def test_fixed_metadata_and_overlapping_selections_split_providers():
    docs = export(
        process("a"),
        process("b"),
        process("c", accuracy="full"),
        process("d", pdgs=(991, -991, 995, 990), words=((2, 4, 1),)),
    )
    assert len(docs) == 4
    payloads = {
        p.path: p
        for p in umami_bundle_payloads(
            processes=[process("a"), process("b")], compiled_model=model()
        )
    }
    index = json.loads(payloads["API/umami/metadata.json"].content)
    assert index["kind"] == "pyamplicol-umami-provider-index"
    assert len(index["providers"]) == 2
    assert b"UMAMI_SINGLE_PROVIDER := 0" in payloads["API/umami/providers.mk"].content


def test_permuted_duplicate_final_states_are_not_summed_in_one_provider():
    a = process("a", pdgs=(990, 990, 991, -991), words=((3, 1, 2, 4),))
    b = process("b", pdgs=(990, 990, -991, 991), words=((4, 1, 2, 3),))
    assert len(export(a, b)) == 2


def test_exact_replay_preserves_physical_entries_and_runtime_mappings():
    (data,) = export(with_replay())
    rows = entries(data)
    assert len(rows) == 2
    assert rows[1]["runtime"]["permutation"] == [0, 1, 3, 2]
    assert rows[1]["runtime"]["color_id"] == "flow:1,2,3,4"
    assert all(row["runtime"]["factor"] == 1 for row in rows)
    assert all(row["matrix_elements"][0]["factor"] == 1 for row in rows)


def test_aggressive_orbit_weight_is_in_json_only():
    (data,) = export(with_replay(), grouping="flavour_blind_observables")
    (row,) = entries(data)
    assert row["matrix_elements"] == [{"pdg_ids": 0, "factor": 2}]
    assert row["runtime"]["factor"] == 1
    assert len(row["members"]) == 2
    assert data["grouping"]["physical_contributions"] == 2
    assert data["grouping"]["exported_contributions"] == 1


def test_none_avoids_umami_representative_reuse():
    (data,) = export(with_replay(), grouping="none")
    rows = entries(data)
    assert len(rows) == 2
    assert rows[1]["runtime"]["color_id"] == "flow:1,2,4,3"
    assert rows[1]["runtime"]["permutation"] == [0, 1, 2, 3]


def test_incoming_swap_and_noninvariant_restricted_helicity_are_not_grouped():
    p = with_replay()
    p.physics["extensions"]["lc_topology_replay"]["partitions"][0][
        "label_permutations"
    ][1] = [[1, 2], [2, 1], [3, 3], [4, 4]]
    (data,) = export(p, grouping="flavour_blind_observables")
    assert len(entries(data)) == 2
    p = with_replay()
    p.physics["helicities"][0]["values"] = [1, 1, -1, 1]
    (data,) = export(p, grouping="flavour_blind_observables")
    assert len(entries(data)) == 2


def test_unproven_replay_is_not_used():
    p = with_replay()
    p.physics["extensions"]["lc_topology_replay"]["partitions"][0]["proof"] = None
    (data,) = export(p, grouping="flavour_blind_observables")
    assert len(entries(data)) == 2


def test_structural_crossflavour_equality_keeps_incoming_pdf_flavours():
    a = replace(
        process("a", pdgs=(991, -991, 990, 990), words=((2, 3, 4, 1),)),
        structural_keys={"flow:2,3,4,1": "certified-key"},
    )
    b = replace(
        process("b", pdgs=(992, -992, 990, 990), words=((2, 3, 4, 1),)),
        structural_keys={"flow:2,3,4,1": "certified-key"},
    )
    (data,) = export(a, b, grouping="flavour_blind_observables")
    rows = entries(data)
    assert len(rows) == 2
    assert rows[1]["runtime"]["process_id"] == "a"
    assert data["pdg_ids"] == [[991, -991, 990, 990], [992, -992, 990, 990]]


def test_equal_defaults_are_not_an_equivalence_and_aliases_not_multiplicity():
    a = process("a", pdgs=(991, -991, 990, 990), words=((2, 3, 4, 1),))
    b = process("b", pdgs=(992, -992, 990, 990), words=((2, 3, 4, 1),))
    a = replace(
        a,
        aliases=(
            {
                "id": "alias",
                "external_pdgs": list(a.external_pdgs),
                "external_permutation": [1, 0, 2, 3],
            },
        ),
    )
    (data,) = export(a, b, grouping="flavour_blind_observables")
    assert len(entries(data)) == 2
    assert entries(data)[1]["runtime"]["process_id"] == "b"
    assert len(data["aliases"]) == 1
    assert data["grouping"]["physical_contributions"] == 2


def test_alpha_s_not_guessed_from_parameter_name():
    m = model()
    m["ir"]["parameters"][0]["lhablock"] = "UNRELATED"
    (data,) = export(process(), compiled_model=m)
    assert data["provider"]["alpha_s_parameter"] is None
    assert not data["provider"]["supports_alpha_s"]


def test_payloads_have_sdk_roles_and_generated_c_matches_json():
    payloads = {
        p.path: p
        for p in umami_bundle_payloads(
            processes=[with_replay()], compiled_model=model()
        )
    }
    data = json.loads(payloads["API/umami/metadata.json"].content)
    header = payloads["API/umami/provider_p0.h"].content.decode()
    assert "#define UMAMI_HAS_ALPHA_S 1" in header
    assert "static const UmamiProvider umami_provider" in header
    assert "umami_c1_f0_permutation[] = {0, 1, 3, 2}" in header
    assert data["provider"]["library"] == "libumami.so"
    assert payloads["API/umami/metadata.json"].role == "sdk-metadata"


def test_unknown_grouping_rejected():
    with pytest.raises(ValueError, match="grouping"):
        export(process(), grouping="guess")


def test_mass_parameter_semantics_not_only_current_value_split_providers():
    m = model()
    m["ir"]["particles"].append({"pdg_code": 996, "color": 1, "mass": "other_mass"})
    m["ir"]["parameters"].append(
        {"name": "other_mass", "nature": "external", "value": [7, 0]}
    )
    a = process("a", pdgs=(991, -991, 995, 990), words=((2, 4, 1),))
    b = process("b", pdgs=(991, -991, 996, 990), words=((2, 4, 1),))
    assert len(export(a, b, compiled_model=m)) == 2


def test_helicity_equivalences_are_explicit_and_disabled_in_none():
    p = process()
    h = copy.deepcopy(p.physics["helicities"][0])
    h.update(id="h:all-minus", values=[-1] * 4)
    p.physics["helicities"].append(h)
    (exact,) = export(p)
    (plain,) = export(p, grouping="none")
    assert entries(exact)[0]["helicities"] == [[0, 1]]
    assert entries(plain)[0]["helicities"] == [[0], [1]]


@pytest.mark.parametrize("accuracy,flows", [("lc", 6), ("full", 1)])
def test_otf_sdk_axes_do_not_change_compact_public_metadata(accuracy, flows):
    p = process(accuracy=accuracy)
    compact = {
        k: v
        for k, v in p.physics.items()
        if k not in {"helicities", "color_components", "coverage", "extensions"}
    }
    process_ir = CanonicalProcessIR(
        process="x x > x x",
        key="x",
        color_accuracy=accuracy,
        legs=tuple(
            ProcessLegIR(
                label=i + 1,
                side="initial" if i < 2 else "final",
                particle="x",
                outgoing_particle="x",
                pdg=990,
                outgoing_pdg=990,
                statistics="bosonic",
                wavefunction_family="vector",
                color_role="adjoint",
                source_orientation="self-conjugate",
            )
            for i in range(4)
        ),
        color_endpoints=ColorEndpointSummary(0, 0, 0),
    )
    seed = SimpleNamespace(
        external_sources=tuple(
            SimpleNamespace(
                public_label=i + 1,
                source_states=(
                    SimpleNamespace(public_helicity=-1),
                    SimpleNamespace(public_helicity=1),
                ),
            )
            for i in reversed(range(4))
        )
    )
    resolved, sectors = build_umami_on_the_fly_physics(compact, process_ir, seed)
    assert "helicities" not in compact
    assert len(resolved["helicities"]) == 16
    assert resolved["helicities"][0]["id"] == "h:-1,-1,-1,-1"
    assert len(resolved["color_components"]) == flows
    (data,) = export(replace(p, physics=resolved, color_sectors=sectors))
    assert len(entries(data)) == flows
