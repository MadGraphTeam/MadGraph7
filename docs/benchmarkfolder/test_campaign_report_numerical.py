"""Portable numerical report validation; synthetic records, no native jobs."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SPEC = importlib.util.spec_from_file_location("campaign_report_numerical", HERE / "campaign_report_numerical.py")
COLLECTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COLLECTOR)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = data if isinstance(data, bytes) else data.encode()
    path.write_bytes(raw)
    return sha(raw)


def dump(path, value):
    return write(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


class Evidence:
    def __init__(self, root, frozen=None):
        self.root, self.files, self.frozen = root, {}, frozen

    def path(self, name):
        path = self.root
        for part in Path(COLLECTOR._relative(name)).parts:
            path /= part
            if path.is_symlink():
                raise ValueError("Evidence symlink")
        return path

    def exists(self, name):
        if self.frozen is not None:
            return name in self.frozen or any(key.startswith(name.rstrip("/") + "/") for key in self.frozen)
        return self.path(name).exists()

    def add(self, name, expected=None):
        if self.frozen is not None and name not in self.frozen:
            raise ValueError("Unfrozen report input")
        raw = self.path(name).read_bytes()
        if expected is not None and sha(raw) != expected:
            raise ValueError("Evidence hash mismatch")
        self.files[name] = {"sha256": sha(raw), "bytes": len(raw)}
        return raw

    def json(self, name, expected=None):
        return json.loads(self.add(name, expected), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))

    def tree(self, name):
        names = (sorted(key for key in self.frozen if key.startswith(name + "/")) if self.frozen is not None
                 else [path.relative_to(self.root).as_posix() for path in sorted(self.path(name).rglob("*"))
                       if path.is_file() or path.is_symlink()])
        for key in names:
            self.add(key)


class Fixture:
    def __init__(self, root):
        self.root = root
        self.relative = "campaigns/final"
        self.campaign = root / self.relative
        self.config = json.loads((HERE / "cases.json").read_text())
        self.baseline = {"revision": COLLECTOR.BASELINE, "archive_sha256": "1" * 64,
                         "files": {"source.f": "2" * 64}}
        shared = {"cases.json": dump(self.campaign / "cases.json", self.config),
                  "inputs/baseline.json": dump(self.campaign / "inputs/baseline.json", self.baseline)}
        for case in self.config["cases"]:
            n = case["nfinal"] + 2
            raw = ("100\n" + "1 0 0 0\n" * (100 * n)).encode()
            name = "points/" + case["id"]
            shared[name + ".dat"] = write(self.campaign / (name + ".dat"), raw)
            metadata = {"inputs": {"count": 100, "incoming_masses": [0., 0.], "masses": [0.] * case["nfinal"]},
                        "points": [{"momenta": [[1., 0., 0., 0.]] * n}] * 100, "sha256": sha(raw)}
            shared[name + ".json"] = dump(self.campaign / (name + ".json"), metadata)
        source = b"retained exporter source\n"
        identity = {"revision": "candidate-commit", "files": {"source.py": sha(source)}, "dirty": False}
        identity["files_sha256"] = COLLECTOR._canonical(identity["files"])
        self.native = {"lib/libmadloop_tirust.so": "3" * 64, "lib/libtirust.a": "4" * 64,
                       "lib/tirust-config.json": "5" * 64}
        self.manifest = {"schema": "madloop-tirust-campaign-v1", "id": "final",
                         "baseline": {"revision": COLLECTOR.BASELINE, "record_sha256": shared["inputs/baseline.json"]},
                         "candidate": {"exporter_identity": identity, "source_snapshot": "inputs/numerical-sources",
                                       "numerical_sources": {"source.py": {"sha256": sha(source)}},
                                       "native": {"files": self.native}},
                         "shared_inputs": {"files": shared, "policy_sha256": COLLECTOR._canonical(self.config)},
                         "export_environment_binding_version": 1, "export_environment": {"PYTHONHASHSEED": "0"},
                         "administrative_worker_timeout_seconds": {case["id"]: 3600. for case in self.config["cases"]}}
        write(self.campaign / "inputs/numerical-sources/source.py", source)
        dump(self.campaign / ("inputs/source-identities/" + COLLECTOR._canonical(identity) + ".json"), identity)
        manifest_hash = dump(self.campaign / "manifest.json", self.manifest)
        self.binding = {"id": "final", "manifest_sha256": manifest_hash}
        self.protocols = {name: (ROOT / "tools" / name).read_bytes() for name in COLLECTOR.PROTOCOLS}
        self.compare = COLLECTOR._parsers(self.protocols, "fixture")[1]

    def folder(self, case=None):
        case = case or self.config["cases"][0]
        return self.campaign / ("experiments/numerical/full100-warmup-" + case["id"] + "-1")

    def complete_case(self, case=None, offset=0.):
        case = case or self.config["cases"][0]
        folder = self.folder(case)
        for name, raw in self.protocols.items():
            write(folder / name, raw)
        config_hash = dump(folder / "cases.json", self.config)
        dump(folder / "campaign-manifest.json", self.manifest)
        dump(folder / "specification.json", {"case": case, "campaign_binding": self.binding,
             "performance_eligible": False, "measurement_phase": "steady-warmup", "worker_timeout_seconds": 3600.,
             "configuration_sha256": config_hash, "protocols": {name: sha(raw) for name, raw in self.protocols.items()}})
        published, reference = [], None
        for source, backend in COLLECTOR.LANES:
            lane = folder / f"{source}-{backend}"
            card_hash = write(self.campaign / f"inputs/{source}/{case['id']}/param_card.dat", "physical card\n")
            command_hash = write(self.campaign / f"commands/{source}-{case['id']}.mg5", "generate " + case["process"] + "\n")
            identity = self.manifest["candidate"]["exporter_identity"] if source == "candidate" else {
                "revision": COLLECTOR.BASELINE, "archive_sha256": self.baseline["archive_sha256"],
                "files_sha256": COLLECTOR._canonical(self.baseline["files"]), "dirty": False}
            generation = {"campaign_binding": self.binding, "case": case, "source": source,
                          "source_identity": identity, "parameter_card_sha256": card_hash,
                          "command_sha256": command_hash, "export_environment_binding_version": 1,
                          "export_environment": {"PYTHONHASHSEED": "0"}}
            params = {"prefix": "TEST_", "nexternal": case["nfinal"] + 2, "nborn": 1}
            rendered = self.protocols["tirust_benchmark_worker.f"].decode()
            for key, token in (("prefix", "PREFIX"), ("nexternal", "NEXTERNAL"), ("nborn", "NBORN")):
                rendered = rendered.replace("@" + token + "@", str(params[key]))
            runtime = {"libraries": {"/external/libmadloop_tirust.so": self.native["lib/libmadloop_tirust.so"]},
                       "archives": {"/external/libtirust.a": self.native["lib/libtirust.a"]},
                       "metadata": {"/external/tirust-config.json": self.native["lib/tirust-config.json"]}}
            recipe = {"compiler": [{"sha256": "6" * 64, "version": "GNU Fortran test"}], "flags": ["-O2"]}
            flags = {"schema": "madloop-effective-compiler-flags-v1", "environment": {},
                     "scopes": {name: {"recipes": {"file.f": recipe}} for name in ("process", "MODEL", "DHELAS")}}
            compilation = {"campaign_binding": self.binding, "masses": [0.] * (case["nfinal"] + 2),
                           "numerical_build_from_sources": True, "full_numerical_rebuild_this_invocation": True,
                           "worker_render_parameters": params, "driver_sha256": sha(rendered.encode()),
                           "worker_template_sha256": sha(self.protocols["tirust_benchmark_worker.f"]),
                           "sources": {"SubProcesses/P0_test/check_sa.f": sha(rendered.encode()),
                                       "Source/make_opts": sha(b"-O2\n")}, "settings": {"Source/make_opts": "-O2\n"},
                           "compiler": "GNU Fortran test", "effective_compiler_flags": flags,
                           "executable_sha256": "7" * 64,
                           "link_inputs_before": {key: runtime[key] for key in ("archives", "metadata")},
                           "link_inputs_after": {key: runtime[key] for key in ("archives", "metadata")}}
            for name, value in (("generation.json", generation), ("compilation.json", compilation)):
                dump(self.campaign / f"inputs/{source}/{case['id']}" / name, value)
            settings = {name.lower(): value for name, value in self.config["madloop_parameters"].items()}
            if source == "candidate":
                settings.update({name.lower(): value for name, value in self.config["candidate_parameters"].items()})
            settings["mlreductionlib"] = str(backend)
            point = "points/" + case["id"]
            inputs = {"campaign_binding": self.binding, "generation": generation, "compilation": compilation,
                      "settings": settings, "param_card_sha256": card_hash, "runtime": runtime,
                      "points_sha256": self.manifest["shared_inputs"]["files"][point + ".dat"],
                      "points_metadata_sha256": self.manifest["shared_inputs"]["files"][point + ".json"],
                      "worker_timeout_seconds": 3600., "numerical_cards": {"Cards/MadLoopParams.dat": "8" * 64}}
            dump(lane / (source + "-inputs.json"), inputs)
            dump(lane / "inputs-after.json", {"runtime": runtime, "executable_sha256": compilation["executable_sha256"],
                 "points_sha256": inputs["points_sha256"], "numerical_cards": inputs["numerical_cards"]})
            code = 210 + backend
            delta = offset if backend == 9 else 0.
            raw = "# retained warmup\n"
            rows = []
            for point_id in range(1, 101):
                for order in (0, 1):
                    numbers = [1., 2. + delta, -3., -4., 1e-10]
                    raw += f"{point_id} {point_id} {order} {code} " + " ".join(map(repr, numbers)) + "\n"
                    rows.append({"point": point_id, "order": order, "status": code,
                                 "coefficients": numbers[:4], "accuracy": numbers[4],
                                 "actual_backend": backend, "precision": "DP"})
            write(lane / "raw/warmup.dat", raw)
            write(lane / "raw/dimensions.dat", "1 1 1\n")
            dump(lane / "raw/ready.json", {"points": 100, "ndim": 1, "nso": 1, "nborn": 1, "cold_seconds": .5})
            observation = {"cold_seconds": .5, "seconds": None, "seconds_per_point": None,
                "rows": rows, "status_counts": {str(code): 100}, "timed_status_counts": None,
                "backend_precision_counts": {f"{backend}:DP": 100}, "native_dp_only": True,
                "answer_dimensions": {"allocated": 1, "loop_orders": 1, "born_orders": 1},
                "measurement_phase": "steady-warmup", "performance_eligible": False,
                "warmup_first_call": 1, "warmup_last_call": 100,
                "warmup_call_point_order": [{"call": p, "point": p} for p in range(1, 101)],
                "raw_outputs": {"warmup.dat": raw, "dimensions.dat": "1 1 1\n"}}
            dump(lane / "raw/observation.json", observation)
            if source == "baseline":
                reference = observation
            result = {"source": source, "backend": backend, "status": "complete", "performance_eligible": False,
                      "observation": observation, "comparison": self.compare(reference["rows"], rows)}
            dump(lane / "result.json", result)
            published.append(result)
        dump(folder / "results.json", {"case": case, "performance_eligible": False, "results": published})

    def collect(self, evidence=None):
        return COLLECTOR.collect_numerical(evidence or Evidence(self.root), self.relative, self.manifest, self.config)


@pytest.fixture
def fixture(tmp_path):
    return Fixture(tmp_path)


def test_missing_runs_preserve_all_cases_and_lanes(fixture):
    result = fixture.collect()
    assert result["status"] == "pending"
    assert len(result["cases"]) == 11
    assert all(len(case["lanes"]) == 6 and case["status"] == "pending" for case in result["cases"])


def test_complete_case_reparsed_without_runtime_or_observability_claim(fixture):
    fixture.complete_case()
    result = fixture.collect()
    case = result["cases"][0]
    assert case["all_six_complete"] and case["regression_pass"] and case["ordinary_target_pass"]
    assert result["status"] == "pending" and result["performance_eligible"] is False
    assert result["fpe_flags"] == result["internal_retries"] == "unobserved"
    assert all(row["accepted_dp_count"] == 100 and row["accepted_qp_count"] == 0 for row in case["lanes"])
    assert all(not {"seconds", "seconds_per_point", "native_dp_only"} & row.keys() for row in case["lanes"])


def replace_observed_candidate_identity(fixture, identity, retain_snapshot=True):
    case = fixture.config["cases"][0]
    provenance = fixture.campaign / f"inputs/candidate/{case['id']}/generation.json"
    generation = json.loads(provenance.read_text())
    generation["source_identity"] = identity
    dump(provenance, generation)
    for backend in COLLECTOR.BACKENDS:
        path = fixture.folder() / f"candidate-{backend}/candidate-inputs.json"
        inputs = json.loads(path.read_text())
        inputs["generation"] = generation
        dump(path, inputs)
    if retain_snapshot:
        dump(fixture.campaign / ("inputs/source-identities/" + COLLECTOR._canonical(identity) + ".json"), identity)


def test_documentation_only_revision_preserves_observed_identity(fixture):
    fixture.complete_case()
    identity = {**fixture.manifest["candidate"]["exporter_identity"], "revision": "later-documentation-commit"}
    replace_observed_candidate_identity(fixture, identity)
    observed = fixture.collect()["cases"][0]
    assert observed["all_six_complete"]
    assert all(lane["identity"]["generator_revision"] == "later-documentation-commit"
               for lane in observed["lanes"] if lane["source"] == "candidate")
    assert fixture.manifest["candidate"]["exporter_identity"]["revision"] == "candidate-commit"


def test_changed_source_cannot_masquerade_as_documentation_only_revision(fixture):
    fixture.complete_case()
    identity = copy.deepcopy(fixture.manifest["candidate"]["exporter_identity"])
    identity.update(revision="later-source-commit", files={"source.py": "0" * 64})
    identity["files_sha256"] = COLLECTOR._canonical(identity["files"])
    replace_observed_candidate_identity(fixture, identity)
    observed = fixture.collect()["cases"][0]
    assert all(lane["status"] == "failed" and "exporter identity" in lane["error"]["message"]
               for lane in observed["lanes"] if lane["source"] == "candidate")


def test_documentation_only_revision_requires_its_retained_identity(fixture):
    fixture.complete_case()
    identity = {**fixture.manifest["candidate"]["exporter_identity"], "revision": "unretained-documentation-commit"}
    replace_observed_candidate_identity(fixture, identity, retain_snapshot=False)
    observed = fixture.collect()["cases"][0]
    assert all(lane["status"] == "failed" and lane["error"]["kind"] == "integrity"
               for lane in observed["lanes"] if lane["source"] == "candidate")


def test_duplicate_loader_basename_retains_both_identities():
    runtime = {"libraries": {"/nix/old/ld-linux-x86-64.so.2": "1" * 64,
                             "/nix/new/ld-linux-x86-64.so.2": "2" * 64},
               "archives": {}, "metadata": {}}
    assert COLLECTOR._runtime_names(runtime)["ld-linux-x86-64.so.2"] == ("1" * 64, "2" * 64)


def test_reused_baseline_command_is_bound_to_recorded_predecessor(fixture):
    fixture.complete_case()
    case = fixture.config["cases"][0]
    provenance = fixture.campaign / f"inputs/baseline/{case['id']}"
    generation = json.loads((provenance / "generation.json").read_text())
    old_generation = {**generation, "campaign_binding": None}
    old_provenance = fixture.root / f"inputs/baseline/{case['id']}"
    old_generation_hash = dump(old_provenance / "generation.json", old_generation)
    old_compilation_hash = write(old_provenance / "compilation.json", (provenance / "compilation.json").read_bytes())
    command_name = f"commands/baseline-{case['id']}.mg5"
    write(fixture.root / command_name, (fixture.campaign / command_name).read_bytes())
    generation["reused_from"] = {"generation_sha256": old_generation_hash,
                                  "compilation_sha256": old_compilation_hash}
    manifest = {**fixture.manifest, "predecessor": {"id": "legacy"}}
    evidence = Evidence(fixture.root)
    COLLECTOR._generation_command(evidence, fixture.relative, case, generation, manifest)
    assert command_name in evidence.files
    assert fixture.relative + "/" + command_name not in evidence.files
    old_generation["source_identity"]["revision"] = "changed"
    dump(old_provenance / "generation.json", old_generation)
    with pytest.raises(ValueError, match="hash"):
        COLLECTOR._generation_command(Evidence(fixture.root), fixture.relative, case, generation, manifest)


def test_accepted_qp_counts_are_reparsed_without_calling_them_native(fixture):
    fixture.complete_case()
    folder = fixture.folder()
    lane = folder / "candidate-9"
    observation = json.loads((lane / "raw/observation.json").read_text())
    raw_lines = (lane / "raw/warmup.dat").read_text().splitlines(keepends=True)
    for index, line in enumerate(raw_lines):
        if not line.startswith("#") and int(line.split()[1]) > 87:
            raw_lines[index] = line.replace(" 219 ", " 323 ", 1)
    raw = "".join(raw_lines)
    write(lane / "raw/warmup.dat", raw)
    for row in observation["rows"]:
        if row["point"] > 87:
            row.update(status=323, precision="QP")
    observation.update(status_counts={"219": 87, "323": 13},
                       backend_precision_counts={"9:DP": 87, "9:QP": 13}, native_dp_only=False)
    observation["raw_outputs"]["warmup.dat"] = raw
    dump(lane / "raw/observation.json", observation)
    result = json.loads((lane / "result.json").read_text())
    result["observation"] = observation
    dump(lane / "result.json", result)
    top = json.loads((folder / "results.json").read_text())
    top["results"][-1] = result
    dump(folder / "results.json", top)
    collected = fixture.collect()["cases"][0]["lanes"][-1]
    assert collected["status"] == "complete"
    assert (collected["accepted_dp_count"], collected["accepted_qp_count"]) == (87, 13)


def test_all_eleven_required_for_pass(fixture):
    for case in fixture.config["cases"]:
        fixture.complete_case(case)
    assert fixture.collect()["status"] == "passed"


@pytest.mark.parametrize("offset,regression,target", [(1e-6, True, False), (.01, False, False)])
def test_target_miss_and_regression_failure_remain_separate(fixture, offset, regression, target):
    fixture.complete_case(offset=offset)
    result = fixture.collect()
    case = result["cases"][0]
    assert case["regression_pass"] is regression and case["ordinary_target_pass"] is target
    assert len(case["lanes"][-1]["ordinary_target_misses"]) == 200
    assert result["status"] == ("pending" if regression else "failed")


def test_failure_marker_wins_over_complete_raw_data(fixture):
    fixture.complete_case()
    dump(fixture.folder() / "candidate-9/failure.json", {"source": "candidate", "backend": 9,
         "status": "failed", "performance_eligible": False, "error": "resource failure after evaluation"})
    row = fixture.collect()["cases"][0]["lanes"][-1]
    assert row["status"] == "failed" and row["points"] is None
    assert row["error"]["kind"] == "recorded_failure"


@pytest.mark.parametrize("filename,mutate", [
    ("candidate-inputs.json", lambda d: d["runtime"]["libraries"].update({"/external/libmadloop_tirust.so": "0" * 64})),
    ("candidate-inputs.json", lambda d: d["generation"]["source_identity"].update(revision="different")),
    ("candidate-inputs.json", lambda d: d.update(points_sha256="0" * 64)),
    ("candidate-inputs.json", lambda d: d["settings"].update(ctmoderun=4)),
    ("candidate-inputs.json", lambda d: d["settings"].update(mlreductionlib="8")),
    ("candidate-inputs.json", lambda d: d["compilation"].update(numerical_build_from_sources=False)),
    ("inputs-after.json", lambda d: d.update(executable_sha256="0" * 64)),
    ("inputs-after.json", lambda d: d["numerical_cards"].update({"Cards/MadLoopParams.dat": "0" * 64})),
    ("raw/ready.json", lambda d: d.update(points=99)),
    ("raw/observation.json", lambda d: d.update(cold_seconds=2.)),
])
def test_completed_lane_corruption_is_failed_integrity(fixture, filename, mutate):
    fixture.complete_case()
    path = fixture.folder() / "candidate-9" / filename
    value = json.loads(path.read_text())
    mutate(value)
    dump(path, value)
    row = fixture.collect()["cases"][0]["lanes"][-1]
    assert row["status"] == "failed", row
    assert row["error"]["kind"] == "integrity"


def test_raw_warmup_change_cannot_be_hidden_by_summary(fixture):
    fixture.complete_case()
    path = fixture.folder() / "candidate-9/raw/warmup.dat"
    path.write_text(path.read_text().replace("1 1 0 219", "1 2 0 219", 1))
    assert fixture.collect()["cases"][0]["lanes"][-1]["status"] == "failed"


def test_saved_protocol_hash_is_required_before_replay(fixture):
    fixture.complete_case()
    path = fixture.folder() / "tirust_benchmark_warmup.py"
    path.write_text(path.read_text() + "\nraise RuntimeError('must not execute')\n")
    result = fixture.collect()["cases"][0]
    assert result["status"] == "failed" and "hash" in result["error"].lower()


def test_top_level_saved_program_is_never_executed(fixture):
    fixture.protocols["tirust_benchmark.py"] += b"\nraise RuntimeError('top level must not execute')\n"
    fixture.complete_case()
    assert fixture.collect()["cases"][0]["all_six_complete"]


def test_frozen_pending_dossier_ignores_future_runs(fixture):
    frozen = {p.relative_to(fixture.root).as_posix() for p in fixture.root.rglob("*") if p.is_file()}
    fixture.complete_case()
    assert fixture.collect(Evidence(fixture.root, frozen))["cases"][0]["status"] == "pending"


@pytest.mark.parametrize("name", ["warmup.dat", "dimensions.dat", "ready.json"])
def test_frozen_dossier_cannot_omit_required_raw_inputs(fixture, name):
    fixture.complete_case()
    frozen = {p.relative_to(fixture.root).as_posix() for p in fixture.root.rglob("*") if p.is_file()}
    omitted = (fixture.folder() / "candidate-9/raw" / name).relative_to(fixture.root).as_posix()
    frozen.remove(omitted)
    result = fixture.collect(Evidence(fixture.root, frozen))["cases"][0]
    assert result["lanes"][-1]["status"] == "failed"
    assert "Unfrozen report input" in result["lanes"][-1]["error"]["message"]
    assert result["all_six_complete"] is False


def test_lane_result_waits_for_top_level_publication(fixture):
    fixture.complete_case()
    path = fixture.folder() / "results.json"
    result = json.loads(path.read_text())
    result["results"].pop()
    dump(path, result)
    assert fixture.collect()["cases"][0]["lanes"][-1]["status"] == "pending"


def test_cross_campaign_manifest_fails_closed(fixture):
    wrong = copy.deepcopy(fixture.manifest)
    wrong["id"] = "other"
    with pytest.raises(ValueError, match="manifest"):
        COLLECTOR.collect_numerical(Evidence(fixture.root), fixture.relative, wrong, fixture.config)


def test_symlink_raw_input_is_rejected(fixture):
    fixture.complete_case()
    path = fixture.folder() / "candidate-9/raw/warmup.dat"
    original = path.read_bytes()
    path.unlink()
    other = fixture.root / "outside-warmup.dat"
    other.write_bytes(original)
    path.symlink_to(other)
    with pytest.raises(ValueError, match="symlink"):
        fixture.collect()


def test_closed_case_allowlist_never_opens_active_run(tmp_path, monkeypatch):
    fixture = Fixture(tmp_path)
    fixture.complete_case()
    active_case = fixture.config['cases'][1]
    active = fixture.folder(active_case).relative_to(tmp_path).as_posix()
    fixture.folder(active_case).mkdir(parents=True)
    (fixture.folder(active_case) / 'results.json').write_text('actively changing invalid data')
    evidence = Evidence(tmp_path)
    original_path = evidence.path
    def guarded(name):
        assert not name.startswith(active), 'Active numerical run was opened'
        return original_path(name)
    monkeypatch.setattr(evidence, 'path', guarded)
    observed = COLLECTOR.collect_numerical(evidence, fixture.relative, fixture.manifest,
                                          fixture.config, {fixture.config['cases'][0]['id']})
    assert observed['cases'][0]['all_six_complete']
    assert observed['cases'][1]['status'] == 'pending'
    assert observed['cases'][1]['complete_lanes'] == 0
    assert all(not name.startswith(active) for name in evidence.files)
