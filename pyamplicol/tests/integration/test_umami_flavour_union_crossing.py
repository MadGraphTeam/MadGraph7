# SPDX-License-Identifier: 0BSD
"""UMAMI providers of leading-colour artifacts with crossed flavour unions.

A crossed member is evaluated through its template's runtime at sign-flipped
momenta (``momentum_signs``), with its helicity labels flipped by rho. Every
physical contribution, per exported helicity and LC flow, must equal the
provider of an artifact generated per flavour at 1e-12 on independent phase-
space points, with the same LHE colour tags; mixed shuffled batches and
helicity-sampled calls must give the same values.

``p p > j j`` always runs; ``p p > j j j`` and ``p p > j j j j`` only with
``PYAMPLICOL_RUN_FLAVOUR_UNION=pp3j,pp4j`` (either or both).
"""

from __future__ import annotations

import ctypes as ct
import importlib.util
import json
import os
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from pyamplicol import Generator, ProcessSet
from pyamplicol.artifacts.process_groups import read_process_groups
from pyamplicol.artifacts.umami_c import helicity_export, provider_header
from pyamplicol.artifacts.umami_reweight import (
    load_umami_metadata,
    write_reweight_provider,
)
from tests.integration.test_flavour_union_crossing import _member_points
from tests.integration.test_flavour_union_crossing_generation import _config
from tests.integration.test_umami_api import (
    _canonical_tags,
    _config_executable,
    _library,
    _unavailable,
)
from tests.integration.test_umami_flavor_switch import IN, OUT

_OPT_IN_ENV = "PYAMPLICOL_RUN_FLAVOUR_UNION"
_SETS = {"pp2j": "p p > j j", "pp3j": "p p > j j j", "pp4j": "p p > j j j j"}
_CHUNK = 4096


def _make(directory: Path, *arguments: str) -> None:
    completed = subprocess.run(
        [
            "make",
            "-C",
            str(directory),
            f"RUSTICOL_CONFIG={_config_executable()}",
            *arguments,
        ],
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.fixture(scope="module", params=sorted(_SETS))
def built(request, tmp_path_factory):
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        pytest.skip("the Rusticol extension has not been built")
    name = request.param
    if name != "pp2j" and name not in os.environ.get(_OPT_IN_ENV, "").split(","):
        pytest.skip(f"set {_OPT_IN_ENV}={name} to run it")
    if shutil.which("make") is None or shutil.which(os.environ.get("CC", "cc")) is None:
        _unavailable("UMAMI integration requires make and a C compiler")
    root = tmp_path_factory.mktemp(f"umami-crossing-{name}")
    jobs = {
        "crossing": _config("lc", unions=True, crossing=True, bundle=True),
        "flavour": _config("lc", unions=False, crossing=False, bundle=True),
    }
    result = {"name": name, "root": root, "artifacts": {}, "umami": {}}
    try:
        for kind, config in jobs.items():
            path = result["artifacts"][kind] = root / kind
            Generator(config).generate(
                ProcessSet.from_expressions((_SETS[name],)), path
            )
            build = root / f"build-{kind}"
            _make(path / "API/umami", f"BUILD_DIR={build}")
            result["umami"][kind] = [
                (data, _library(build / data["provider"]["library"]))
                for data in load_umami_metadata(path)
            ]
        yield result
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _records(data):
    for channel in data["channels"]:
        for entry in channel["processes"]:
            yield entry["runtime"]
            for member in entry["members"]:
                yield member["runtime"]


def _members(documents):
    """(process id, canonical physical tags) -> (provider, channel, flavour,
    entry, member) for every exported physical contribution."""

    result = {}
    for provider, (data, _) in enumerate(documents):
        for channel, description in enumerate(data["channels"]):
            for flavour, entry in enumerate(description["processes"]):
                for member in entry["members"]:
                    tags = _canonical_tags(data["color_flows"][member["color_flows"]])
                    result.setdefault(
                        (member["process_id"], tags),
                        (provider, channel, flavour, entry, member),
                    )
    return result


def _input_legs(entry, member):
    """Physical leg of each input leg of the entry (input = physical[legs])."""

    legs = [0] * len(entry["runtime"]["permutation"])
    for j, slot in enumerate(entry["runtime"]["permutation"]):
        legs[slot] = member["runtime"]["permutation"][j]
    return legs


class _Provider:
    def __init__(self, library, artifact):
        self.library = library
        self.handle = ct.c_void_p()
        assert (
            library.umami_initialize(ct.byref(self.handle), os.fsencode(artifact)) == 0
        )

    def close(self):
        assert self.library.umami_free(self.handle) == 0

    def call(self, momenta, channels, flavours, helicities=None, sample=False):
        """Totals, colour indices (and sampled helicities) of events given as
        input momenta [event, leg, component]."""

        outputs = []
        for start in range(0, len(momenta), _CHUNK):
            stop = start + _CHUNK
            status, values = self._call(
                momenta[start:stop],
                channels[start:stop],
                flavours[start:stop],
                None if helicities is None else helicities[start:stop],
                sample,
            )
            assert status == 0
            outputs.append(values)
        return [np.concatenate(column) for column in zip(*outputs, strict=True)]

    def _call(self, momenta, channels, flavours, helicities, sample):
        count = len(momenta)
        # UMAMI layout: [component][leg][event]
        flat = np.ascontiguousarray(np.transpose(momenta, (2, 1, 0)), dtype=np.float64)
        inputs = {
            IN["UMAMI_IN_MOMENTA"]: flat,
            IN["UMAMI_IN_FLAVOR_INDEX"]: np.asarray(flavours, dtype=np.intc),
            IN["UMAMI_IN_CHANNEL_INDEX"]: np.asarray(channels, dtype=np.uintc),
            IN["UMAMI_IN_RANDOM_HELICITY"]: np.full(count, 0.37),
        }
        if helicities is not None:
            inputs[IN["UMAMI_IN_HELICITY_INDEX"]] = np.asarray(
                helicities, dtype=np.intc
            )
        outputs = {
            OUT["UMAMI_OUT_MATRIX_ELEMENT"]: np.zeros(count),
            OUT["UMAMI_OUT_COLOR_INDEX"]: np.zeros(count, dtype=np.intc),
        }
        if sample:
            outputs[OUT["UMAMI_OUT_HELICITY_INDEX"]] = np.zeros(count, dtype=np.intc)
        status = self.library.umami_matrix_element(
            self.handle,
            count,
            count,
            0,
            len(inputs),
            (ct.c_int * len(inputs))(*inputs),
            (ct.c_void_p * len(inputs))(*(v.ctypes.data for v in inputs.values())),
            len(outputs),
            (ct.c_int * len(outputs))(*outputs),
            (ct.c_void_p * len(outputs))(*(v.ctypes.data for v in outputs.values())),
        )
        return status, list(outputs.values())


def _crossed_members(artifact):
    group_map = read_process_groups(artifact)
    assert group_map is not None
    return {
        member.process_id: member
        for group in group_map.groups.values()
        for member in group.members[1:]
        if member.crossed
    }


def test_crossed_metadata_has_momentum_signs_and_keeps_templates_together(built):
    crossed = _crossed_members(built["artifacts"]["crossing"])
    assert crossed
    documents = [data for data, _ in built["umami"]["crossing"]]
    references = [data for data, _ in built["umami"]["flavour"]]
    assert all(data["schema_version"] == 6 for data in documents + references)
    assert any("momentum_signs" in json.dumps(data) for data in documents)
    assert not any("momentum_signs" in json.dumps(data) for data in references)
    assert len(documents) == len(references)
    seen = set()
    for data in documents:
        sources = defaultdict(set)
        for channel in data["channels"]:
            for entry in channel["processes"]:
                for member in entry["members"]:
                    sources[member["runtime"]["process_id"]].add(member["process_id"])
        for record in _records(data):
            identifier = record.get("assignment_key")
            if identifier in crossed:
                assert record["momentum_signs"] == list(
                    crossed[identifier].crossing_momentum_signs
                )
                seen.add(identifier)
                # the template's own contributions are in the same provider
                assert record["process_id"] in sources[record["process_id"]]
            else:
                assert "momentum_signs" not in record
    assert seen == set(crossed)
    # a crossing artifact is a reweight reference, never a target (FC crossing)
    write_reweight_provider(
        built["artifacts"]["crossing"],
        built["artifacts"]["flavour"],
        built["root"] / "reweight",
        switch_lines=2,
    )
    with pytest.raises(ValueError, match="reweight targets with crossed"):
        write_reweight_provider(
            built["artifacts"]["flavour"],
            built["artifacts"]["crossing"],
            built["root"] / "reweight-target",
        )


def _helicity_index(data):
    return {tuple(vector): label for label, vector in enumerate(data["helicities"])}


def _physical_tags(tags, legs):
    physical = [None] * len(legs)
    for slot, leg in enumerate(legs):
        physical[leg] = tags[slot]
    return _canonical_tags(physical)


def test_crossed_members_equal_per_flavour_providers(built):
    ours_docs, theirs_docs = built["umami"]["crossing"], built["umami"]["flavour"]
    ours_members, theirs_members = _members(ours_docs), _members(theirs_docs)
    assert set(ours_members) == set(theirs_members)
    crossed = _crossed_members(built["artifacts"]["crossing"])
    keys = sorted(ours_members)
    assert {key[0] for key in keys} >= set(crossed)
    n = len(ours_members[keys[0]][4]["pdgs"])
    points = np.array(_member_points("p p > " + " ".join(["j"] * (n - 2))))
    helicity_index = [_helicity_index(data) for data, _ in theirs_docs]
    # Events (key, point, label) per side as columns; label -1 is the sum.
    columns = {side: defaultdict(list) for side in ("ours", "theirs")}
    legs = {"ours": [], "theirs": []}
    for k, key in enumerate(keys):
        provider, channel, flavour, entry, member = ours_members[key]
        other = theirs_members[key]
        data = ours_docs[provider][0]
        own, theirs = _input_legs(entry, member), _input_legs(other[3], other[4])
        legs["ours"].append(own)
        legs["theirs"].append(theirs)
        labels, other_labels = [-1], [-1]
        for label in helicity_export(data, entry["runtime"]):
            # the exported vector in physical legs, then the reference's label
            physical = [0] * n
            for slot, leg in enumerate(own):
                physical[leg] = data["helicities"][label][slot]
            labels.append(label)
            other_labels.append(
                helicity_index[other[0]][tuple(physical[leg] for leg in theirs)]
            )
        count = len(labels) * len(points)
        for side, where, values in (
            ("ours", (provider, channel, flavour), labels),
            ("theirs", other[:3], other_labels),
        ):
            column = columns[side]
            column["key"] += [k] * count
            column["point"] += [p for _ in values for p in range(len(points))]
            column["label"] += [v for v in values for _ in range(len(points))]
            for name, value in zip(
                ("provider", "channel", "flavour"), where, strict=True
            ):
                column[name] += [value] * count
    columns = {
        side: {name: np.asarray(values) for name, values in column.items()}
        for side, column in columns.items()
    }
    legs = {side: np.asarray(value) for side, value in legs.items()}
    handles = {
        side: [
            _Provider(library, built["artifacts"][kind])
            for _, library in built["umami"][kind]
        ]
        for side, kind in (("ours", "crossing"), ("theirs", "flavour"))
    }

    def evaluate(side, selection, sample=False):
        column = columns[side]
        provider = int(column["provider"][selection[0]])
        assert (column["provider"][selection] == provider).all()
        labels = column["label"][selection]
        summed = labels[0] < 0
        assert ((labels < 0) == summed).all()
        momenta = points[
            column["point"][selection][:, None], legs[side][column["key"][selection]]
        ]
        return handles[side][provider].call(
            momenta,
            column["channel"][selection],
            column["flavour"][selection],
            None if summed else labels,
            sample=sample,
        )

    try:
        values, colors = {}, {}
        for side, column in columns.items():
            values[side] = np.empty(len(column["key"]))
            colors[side] = np.empty(len(column["key"]), dtype=np.intc)
            groups = column["provider"] * 2 + (column["label"] < 0)
            for group in np.unique(groups):
                selection = np.flatnonzero(groups == group)
                value, color = evaluate(side, selection)
                values[side][selection], colors[side][selection] = value, color
        ours, theirs = values["ours"], values["theirs"]
        column = columns["theirs"]
        summed = column["label"] < 0
        # per (contribution, point): the scale of its helicity values
        scale = np.zeros((len(keys), len(points)))
        np.maximum.at(
            scale,
            (column["key"][~summed], column["point"][~summed]),
            np.abs(theirs[~summed]),
        )
        reference = np.where(
            summed, np.abs(theirs), scale[column["key"], column["point"]]
        )
        # 1e-12 relative down to 1e-5 of the point's largest helicity value; below,
        # roundoff of the two programs' cancellations: 1e-17 of that value
        tolerance = 1.0e-12 * np.maximum(np.abs(theirs), 1.0e-5 * reference)
        bad = np.flatnonzero(np.abs(ours - theirs) > tolerance)
        assert not len(bad), [
            (
                keys[column["key"][i]],
                column["point"][i],
                column["label"][i],
                ours[i],
                theirs[i],
                reference[i],
                float(abs(ours[i] - theirs[i]) / abs(theirs[i])),
            )
            for i in bad[:8]
        ]
        crossed_events = np.array([keys[k][0] in crossed for k in column["key"]])
        error = np.abs(ours - theirs)
        bands = []
        for low, high in ((1.0e-6, np.inf), (1.0e-9, 1.0e-6)):
            band = (np.abs(theirs) > low * reference) & (
                np.abs(theirs) <= high * reference
            )
            relative = np.max(error[band] / np.abs(theirs[band]), initial=0.0)
            bands.append(
                f"{int(band.sum())} in ({low:g}, {high:g}] of their scale, worst "
                f"relative {float(relative):.2e}"
            )
        small = (np.abs(theirs) <= 1.0e-9 * reference) & (reference > 0)
        bands.append(
            f"{int(small.sum())} below, worst "
            f"{float(np.max(error[small] / reference[small], initial=0.0)):.2e} of it"
        )
        print(
            f"{built['name']}: {len(keys)} contributions, {len(ours)} values "
            f"({int(crossed_events.sum())} crossed, {int((~summed).sum())} per "
            f"helicity); " + "; ".join(bands)
        )
        assert crossed_events[~summed].any()
        # LHE colour tags of the summed events, in physical legs
        for i in np.flatnonzero(summed):
            key = keys[column["key"][i]]
            provider, _, _, entry, member = ours_members[key]
            other = theirs_members[key]
            tags = ours_docs[provider][0]["color_flows"][colors["ours"][i]]
            other_tags = theirs_docs[other[0]][0]["color_flows"][colors["theirs"][i]]
            assert _physical_tags(tags, _input_legs(entry, member)) == _physical_tags(
                other_tags, _input_legs(other[3], other[4])
            ), key
        # every flavour and flow of a provider in one shuffled call, and sampled
        own = columns["ours"]
        for provider in np.unique(own["provider"]):
            selection = np.flatnonzero(
                (own["provider"] == provider) & (own["label"] < 0)
            )
            selection = selection[np.random.default_rng(5).permutation(len(selection))]
            shuffled = evaluate("ours", selection)[0]
            assert np.allclose(shuffled, ours[selection], rtol=1.0e-13, atol=0.0)
            sampled = evaluate("ours", selection, sample=True)
            assert np.allclose(sampled[0], ours[selection], rtol=1.0e-13, atol=0.0)
            for i, label in zip(selection, sampled[2], strict=True):
                entry = ours_members[keys[own["key"][i]]][3]
                exported = helicity_export(ours_docs[provider][0], entry["runtime"])
                assert label in exported
    finally:
        for providers in handles.values():
            for provider in providers:
                provider.close()


def test_a_provider_without_momentum_signs_is_refused(built):
    """A provider whose contributions lack the signs of crossed assignments
    fails at load against the artifact (the core's crossing differs)."""

    artifact = built["artifacts"]["crossing"]
    copy = built["root"] / "stripped"
    shutil.copytree(artifact / "API/umami", copy)
    stripped = []
    for data, _ in built["umami"]["crossing"]:
        data = json.loads(json.dumps(data))
        for record in _records(data):
            record.pop("momentum_signs", None)
        name = f"provider_{data['provider']['id']}.h"
        (copy / name).write_text(provider_header(data), encoding="utf-8")
        stripped.append(data)
    build = built["root"] / "build-stripped"
    _make(copy, f"BUILD_DIR={build}")
    refused = 0
    for data in stripped:
        library = _library(build / data["provider"]["library"])
        handle = ct.c_void_p()
        if library.umami_initialize(ct.byref(handle), os.fsencode(artifact)) != 0:
            refused += 1
            continue
        provider = _Provider.__new__(_Provider)
        provider.library, provider.handle = library, handle
        try:
            crossed = [
                (c, f)
                for c, channel in enumerate(data["channels"])
                for f, entry in enumerate(channel["processes"])
                if entry["runtime"].get("assignment_key") in _crossed_members(artifact)
            ]
            if not crossed:
                continue
            channel, flavour = crossed[0]
            n = data["provider"]["particle_count"]
            point = np.array(_member_points("p p > " + " ".join(["j"] * (n - 2)))[:1])
            status, _ = provider._call(point, [channel], [flavour], None, False)
            assert status != 0
            refused += 1
        finally:
            provider.close()
    assert refused > 0


def test_append_cannot_drop_the_crossed_records(built):
    """Recurrence artifacts (unions, crossing) refuse appends, so the OTF append
    path, which re-reads the previous UMAMI metadata, never rebuilds crossed
    records from a compact process (its writer path is a unit test)."""

    if built["name"] != "pp2j":
        pytest.skip("checked on p p > j j")
    copy = built["root"] / "appended"
    shutil.copytree(built["artifacts"]["crossing"], copy)
    before = (copy / "API/umami/metadata.json").read_bytes()
    with pytest.raises(Exception, match="regenerate the recurrence process set"):
        Generator(_config("lc", unions=True, crossing=True, bundle=True)).generate(
            ProcessSet.from_expressions(("g g > g g g",), names=("g5",)),
            copy,
            mode="append",
        )
    assert (copy / "API/umami/metadata.json").read_bytes() == before
