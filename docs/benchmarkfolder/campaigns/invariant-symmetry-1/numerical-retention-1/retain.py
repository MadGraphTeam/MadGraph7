#!/usr/bin/env python3
"""Append-once retention of explicitly closed numerical cases; offline replay.

The default plan reads no campaign execution artifacts. Sealing consumes an
explicit closed-case confirmation; it never opens mutable supervisor status.
"""
import argparse
from contextlib import contextmanager
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tempfile
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
CAMPAIGN = "invariant-symmetry-1"
MANIFEST_SHA = "ce49d147c1117875cd7e9692223bec546c100b3a4f0c3f89eef9206c323223d0"
SUPERVISOR_INDEX_SHA = "fde535243092336b3988a902a88aaff8223d6d3dfb37cdfe68e7fc4cd152e243"
SESSION = 63790
SCHEMA = "madloop-tirust-portable-numerical-checkpoint-v1"
CONFIRMATION_SCHEMA = "madloop-tirust-closed-numerical-confirmation-v1"
DOCS = "docs/benchmarkfolder"
CASE_ORDER = ["qcd_n1_uux_z", "ew_n1_udx_wp", "li_gg_h", "qcd_n2_gg_ttx",
              "ew_n2_epem_mupmum", "mixed_uux_ga", "li_gg_hh", "qcd_n3_uux_zgg",
              "ew_n3_epem_ttxa", "qcd_n4_uux_zzgg", "ew_n4_epem_mupmumaa"]
PROTOCOLS = {"tirust_benchmark_numerical.py", "tirust_benchmark.py",
             "tirust_benchmark_worker.py", "tirust_benchmark_worker.f",
             "tirust_benchmark_warmup.py", "tirust_benchmark_compiler.py",
             "tirust_benchmark_campaign.py"}
PACKAGE_SOURCES = ("retain.py", "campaign_report_numerical.py", "numerical_supervisor.py",
                   "numerical-supervisor-source-index.json", "README.md", "test_retain.py", "source-index.json",
                   "synthetic_fixture.py", "fixture-cases.json",
                   *("protocol-" + name for name in sorted(PROTOCOLS)))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def parsed(raw):
    def invalid(value):
        raise ValueError("Nonfinite JSON value: " + value)
    return json.loads(raw.decode("utf-8"), parse_constant=invalid)


def safe(name):
    require(isinstance(name, str) and name and "\\" not in name, "Invalid relative evidence path")
    path = PurePosixPath(name)
    require(not path.is_absolute() and ".." not in path.parts and str(path) == name,
            "Unsafe evidence path")
    return name


def identity(raw):
    return {"sha256": sha(raw), "bytes": len(raw)}


def text_only(raw, name):
    require(not raw.startswith(b"\x7fELF") and b"\x00" not in raw,
            "Binary/native artifact outside retention scope: " + name)
    require(PurePosixPath(name).suffix not in (".a", ".o", ".so", ".dll", ".dylib", ".pyc"),
            "Native artifact outside retention scope: " + name)
    raw.decode("utf-8", errors="strict")
    return raw


def campaign_path():
    return DOCS + "/campaigns/" + CAMPAIGN


def supervisor_path():
    return campaign_path() + "/numerical-qualification-1"


def confirmation_cases(value):
    require(value.get("schema") == CONFIRMATION_SCHEMA
            and value.get("campaign_binding") == {"id": CAMPAIGN, "manifest_sha256": MANIFEST_SHA}
            and value.get("supervisor_session_id") == SESSION
            and value.get("supervisor_source_index_sha256") == SUPERVISOR_INDEX_SHA,
            "Confirmation belongs to another campaign or supervisor")
    closed = value["closed_cases"]
    ids = [row["case"] for row in closed]
    require(ids and len(ids) == len(set(ids)) and ids == [name for name in CASE_ORDER if name in ids],
            "Closed cases must be distinct and follow the registered order")
    for row in closed:
        require(row.get("status") in ("passed", "failed") and row.get("finished_utc")
                and type(row.get("returncode")) is int, "Only explicitly closed cases may be retained")
        if row["status"] == "passed":
            require(row["returncode"] == 0 and row.get("raw_replay_passed") is True,
                    "Passed case lacks its completed supervisor replay")
    terminal = value.get("terminal_confirmation")
    if terminal is not None:
        require(terminal.get("observed_terminal") is True and terminal.get("session_id") == SESSION
                and type(terminal.get("exit_code")) is int, "Invalid terminal confirmation")
    if ids == CASE_ORDER:
        require(terminal is not None, "Full-matrix retention requires observed supervisor termination")
    return ids


def load_collector(package=HERE):
    source = Path(package) / "campaign_report_numerical.py"
    spec = importlib.util.spec_from_file_location("frozen_numerical_checkpoint_collector", source)
    module = importlib.util.module_from_spec(spec)
    # This fixed local source is hash-bound by the source or package index.
    # It imports only the standard library and executes no programs.
    spec.loader.exec_module(module)
    return module


class Store:
    """Either a confined source snapshot or an entirely in-memory replay."""
    def __init__(self, members=None, root=None):
        self.members = {} if members is None else dict(members)
        self.root = Path(root).resolve() if root is not None else None
        self.allowed, self.trees, self.captured, self.walked = set(), set(), {}, {}

    def permit(self, *names):
        self.allowed.update(safe(name) for name in names)

    def permit_tree(self, name):
        self.trees.add(safe(name))

    def authorized(self, name):
        name = safe(name)
        require(name in self.allowed or any(name == tree or name.startswith(tree + "/") for tree in self.trees),
                "File outside explicit selected-case allowlist: " + name)
        return name

    def path(self, name):
        name = self.authorized(name)
        path = self.root
        for part in PurePosixPath(name).parts:
            path /= part
            require(not path.is_symlink(), "Symlink in retained source: " + name)
        return path

    def add(self, name, expected=None):
        name = safe(name)
        if name not in self.members:
            require(self.root is not None, "Missing frozen evidence: " + name)
            path = self.path(name)
            require(stat.S_ISREG(path.stat().st_mode), "Nonregular retained source")
            self.members[name] = text_only(path.read_bytes(), name)
            self.captured[name] = identity(self.members[name])
        raw = self.members[name]
        if expected is not None:
            require(sha(raw) == expected, "Evidence SHA-256 differs: " + name)
        return raw

    def json(self, name, expected=None):
        return parsed(self.add(name, expected))

    def exists(self, name):
        name = safe(name)
        if name in self.members or any(key.startswith(name + "/") for key in self.members):
            return True
        return self.path(name).exists() if self.root is not None else False

    def tree(self, name):
        name = safe(name)
        if self.root is not None:
            path = self.path(name)
            names = self._walk(path)
            self.walked[name] = names
        else:
            names = sorted(key for key in self.members if key == name or key.startswith(name + "/"))
        for item in names:
            self.add(item)
        return names

    def _walk(self, path):
        if not path.exists():
            return []
        paths = sorted(path.rglob("*")) if path.is_dir() else [path]
        names = []
        for item in paths:
            require(not item.is_symlink(), "Symlink in selected evidence tree")
            mode = item.lstat().st_mode
            if stat.S_ISDIR(mode):
                continue
            require(stat.S_ISREG(mode), "Nonregular selected evidence tree member")
            names.append(item.relative_to(self.root).as_posix())
        return names

    def unchanged(self):
        for name, value in self.captured.items():
            require(identity(self.path(name).read_bytes()) == value, "Closed evidence changed during capture: " + name)
        for name, names in self.walked.items():
            require(self._walk(self.path(name)) == names, "Selected evidence membership changed during capture")


class ReportEvidence:
    def __init__(self, store):
        self.store, self.root = store, Path("/portable-checkpoint") / DOCS

    def add(self, name, expected=None):
        return self.store.add(DOCS + "/" + safe(name), expected)

    def json(self, name, expected=None):
        return parsed(self.add(name, expected))

    def exists(self, name):
        return self.store.exists(DOCS + "/" + safe(name))

    def tree(self, name):
        return self.store.tree(DOCS + "/" + safe(name))


def configure_allowlist(store, manifest, ids):
    prefix, supervisor = campaign_path(), supervisor_path()
    store.permit(prefix + "/manifest.json", prefix + "/cases.json",
                 supervisor + "/specification.json", supervisor + "/configuration-initial.json",
                 supervisor + "/source-index.json")
    for name in manifest["shared_inputs"]["files"]:
        store.permit(prefix + "/" + safe(name))
    for name in manifest["candidate"]["numerical_sources"]:
        store.permit(prefix + "/" + safe(manifest["candidate"]["source_snapshot"]) + "/" + safe(name))
    for case_id in ids:
        store.permit(supervisor + "/" + case_id + ".log")
        store.permit_tree(prefix + "/experiments/numerical/full100-warmup-" + case_id + "-1")
        for source in ("baseline", "candidate"):
            store.permit_tree(prefix + "/inputs/" + source + "/" + case_id)
            store.permit(prefix + "/commands/" + source + "-" + case_id + ".mg5")
    # Only declared predecessor manifests and selected baseline generation
    # records/CLI commands may be consulted. Never inspect ancestor raw trees.
    ancestor = manifest
    seen = {prefix}
    while "predecessor" in ancestor:
        previous = ancestor["predecessor"]
        root = DOCS if previous["id"] == "legacy" else DOCS + "/" + safe(previous["report"])
        require(root not in seen, "Cyclic baseline ancestry")
        seen.add(root)
        for case_id in ids:
            for name in ("generation.json", "compilation.json"):
                store.permit(root + "/inputs/baseline/" + case_id + "/" + name)
            store.permit(root + "/commands/baseline-" + case_id + ".mg5")
        if previous["id"] == "legacy":
            break
        store.permit(root + "/manifest.json")
        ancestor = store.json(root + "/manifest.json", previous["manifest_sha256"])


def capture_physical_origins(store, manifest, ids):
    origins = manifest["physical_card_origins"]
    require(origins["scope"] == "physical-input-ancestry-only"
            and origins["origin"]["id"] == manifest["predecessor"]["id"]
            and origins["origin"]["report"] == manifest["predecessor"]["report"]
            and origins["origin"]["manifest_sha256"] == manifest["predecessor"]["manifest_sha256"],
            "Physical input ancestry differs from the declared predecessor")
    config = store.json(campaign_path() + "/cases.json")
    expected_binding = {key: origins["origin"][key] for key in ("id", "manifest_sha256")}
    for case_id in ids:
        card = origins["cards"][case_id]
        require(card["files"], "Missing declared physical-card evidence")
        for name, checksum in card["files"].items():
            require(checksum == card["param_card_sha256"], "Declared physical-card copies disagree")
            path = DOCS + "/" + safe(name)
            store.permit(path)
            store.add(path, checksum)
        observation_prefix = origins["origin"]["report"] + "/experiments/numerical/full100-warmup-" + case_id + "-1/"
        selected = {name: checksum for name, checksum in origins["observations"].items()
                    if name.startswith(observation_prefix)}
        require(selected, "No declared physical-input observation for selected case")
        sources = set()
        for name, checksum in selected.items():
            path = DOCS + "/" + safe(name)
            store.permit(path)
            observed = store.json(path, checksum)
            generation = observed["generation"]
            require(generation["case"] == next(case for case in config["cases"] if case["id"] == case_id)
                    and observed.get("campaign_binding") == generation.get("campaign_binding") == expected_binding
                    and observed["param_card_sha256"] == generation["parameter_card_sha256"] == card["param_card_sha256"]
                    and observed["points_sha256"] == manifest["shared_inputs"]["files"]["points/" + case_id + ".dat"]
                    and observed["points_metadata_sha256"] == manifest["shared_inputs"]["files"]["points/" + case_id + ".json"],
                    "Declared physical-input observation differs")
            sources.add(generation["source"])
        require(sources == {"baseline", "candidate"}, "Physical-card ancestry omits a source")


def capture_static(store, confirmation, package=HERE):
    ids = confirmation_cases(confirmation)
    store.permit("closed-case-confirmation.json")
    prefix, supervisor = campaign_path(), supervisor_path()
    store.permit(prefix + "/manifest.json")
    manifest = store.json(prefix + "/manifest.json", MANIFEST_SHA)
    configure_allowlist(store, manifest, ids)
    source_index = store.json(supervisor + "/source-index.json", SUPERVISOR_INDEX_SHA)
    require(store.add(supervisor + "/source-index.json") == (Path(package) / "numerical-supervisor-source-index.json").read_bytes(),
            "Supervisor source seal differs from the reviewed retention proposal")
    spec = store.json(supervisor + "/specification.json", confirmation["supervisor_specification_sha256"])
    require(spec["manifest_sha256"] == MANIFEST_SHA and spec["campaign"] == CAMPAIGN
            and spec["source_index_sha256"] == SUPERVISOR_INDEX_SHA
            and spec["supervisor_sha256"] == sha((Path(package) / "numerical_supervisor.py").read_bytes()),
            "Supervisor specification differs from its frozen source")
    require([item["case"] for item in spec["schedule"]] == CASE_ORDER, "Supervisor schedule differs")
    for item in spec["schedule"]:
        label = "full100-warmup-" + item["case"] + "-1"
        require(item["label"] == label and item["command"][1:] == ["-B", "tools/tirust_benchmark_numerical.py",
                "--campaign", CAMPAIGN, "--case", item["case"], "--label", label, "--warmup-only",
                "--backends", "1", "6", "7", "8", "9", "--cpu", str(spec["cpu"])],
                "Supervisor command differs from the six-lane untimed schedule")
    require(spec["protocols"] == {name: source_index["inputs"]["tools/" + name] for name in PROTOCOLS},
            "Supervisor protocol closure differs from its source seal")
    for name, checksum in {**source_index["files"], **source_index.get("validation_files", {})}.items():
        path = supervisor + "/" + safe(name)
        store.permit(path)
        store.add(path, checksum)
    store.add(supervisor + "/configuration-initial.json")
    preparation = prefix + "/preparation-1"
    for name, checksum in spec["preparation"].items():
        require(name in ("status.json", "specification.json", "source-index.json"), "Unexpected preparation binding")
        path = preparation + "/" + name
        store.permit(path)
        store.add(path, checksum)
    prepared = store.json(preparation + "/status.json")
    require(prepared.get("status") == "passed" and prepared.get("amplitude_evaluations") == 0,
            "Numerical admission lacks passed source preparation")
    prep_index = store.json(preparation + "/source-index.json")
    for name, checksum in prep_index["files"].items():
        path = preparation + "/" + safe(name)
        store.permit(path)
        store.add(path, checksum)
    # These are frozen text source/index records, not current native paths.
    # Store their declared hashes without reopening live tools or native files.
    capture_physical_origins(store, manifest, ids)
    for case_id in ids:
        generation = store.json(prefix + "/inputs/candidate/" + case_id + "/generation.json")
        collector = load_collector(package)
        name = prefix + "/inputs/source-identities/" + collector._canonical(generation["source_identity"]) + ".json"
        store.permit(name)
        require(store.json(name) == generation["source_identity"], "Retained candidate source identity differs")
    return manifest, spec


@contextmanager
def portable_tree(members):
    with tempfile.TemporaryDirectory(prefix="numerical-checkpoint-replay-") as directory:
        root = Path(directory)
        for name, raw in members.items():
            target = root / safe(name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        yield root


def replay_supervisor(store, confirmation, specification, package=HERE):
    collector = load_collector(package)
    ids = confirmation_cases(confirmation)
    protocol_raw = {}
    for name in PROTOCOLS:
        protocol_raw[name] = (Path(package) / ("protocol-" + name)).read_bytes()
        require(sha(protocol_raw[name]) == specification["protocols"][name], "Frozen protocol source differs")
    warmup, compare, _ = collector._parsers(protocol_raw, "frozen-checkpoint-protocol")
    bench_functions = collector._functions(protocol_raw["tirust_benchmark.py"],
        ("content_digest", "parse_chain", "validate_worker_modes"), {"hashlib": hashlib, "json": json}, "frozen-benchmark")
    bench = SimpleNamespace(**{name: bench_functions[name] for name in
                               ("content_digest", "parse_chain", "validate_worker_modes")})
    bench.compare_rows = compare
    bench.case_worker_timeout = lambda case: specification["worker_timeouts"][case["id"]]
    source = (Path(package) / "numerical_supervisor.py").read_bytes()
    require(sha(source) == specification["supervisor_sha256"], "Frozen supervisor source hash differs")
    with portable_tree(store.members) as root:
        (root / "tools").mkdir(exist_ok=True)
        for name, raw in protocol_raw.items():
            (root / "tools" / name).write_bytes(raw)
        warmup["Path"] = Path
        namespace = {"bench": bench, "parse_warmup": warmup["parse_warmup"], "Path": Path,
            "ROOT": root, "REPORT": root / campaign_path(), "CAMPAIGN": CAMPAIGN,
            "EXPECTED_MANIFEST": MANIFEST_SHA, "PROTOCOLS": sorted(PROTOCOLS),
            "EXPECTED_LANES": list(collector.LANES),
            "read": lambda path: parsed(Path(path).read_bytes()),
            "digest": lambda path: sha(Path(path).read_bytes())}
        functions = collector._functions(source, ("runtime_files", "replay_lane", "observed_result"),
                                           namespace, "frozen-numerical-supervisor")
        config = store.json(campaign_path() + "/cases.json")
        results = {}
        for state in confirmation["closed_cases"]:
            item = next(item for item in specification["schedule"] if item["case"] == state["case"])
            value = functions["observed_result"](item, state["returncode"], config, specification["compiled_inputs"])
            require(all(state.get(key) == expected for key, expected in value.items()),
                    "Closed supervisor entry differs from exact frozen replay: " + state["case"])
            results[state["case"]] = value
        return results


def validate(store, confirmation, package=HERE):
    manifest, spec = capture_static(store, confirmation, package)
    ids = confirmation_cases(confirmation)
    prefix, supervisor = campaign_path(), supervisor_path()
    collector = load_collector(package)
    config = store.json(prefix + "/cases.json")
    for state in confirmation["closed_cases"]:
        case_id = state["case"]
        case = next(case for case in config["cases"] if case["id"] == case_id)
        directory = prefix + "/experiments/numerical/full100-warmup-" + case_id + "-1"
        store.tree(directory)
        store.add(supervisor + "/" + case_id + ".log", state["console_sha256"])
        if state.get("results_path") is not None:
            require(state["results_path"] == directory[len(prefix) + 1:] + "/results.json", "Closed result path differs")
            store.add(prefix + "/" + state["results_path"], state["results_sha256"])
        for source in ("baseline", "candidate"):
            evidence = prefix + "/inputs/" + source + "/" + case_id
            store.tree(evidence)
            binding = spec["compiled_inputs"][source + "/" + case_id]
            generation = store.json(evidence + "/generation.json", binding["generation_sha256"])
            store.add(evidence + "/compilation.json", binding["compilation_sha256"])
            require(sha(store.add(evidence + "/param_card.dat")) == binding["param_card_sha256"]
                    == manifest["physical_card_origins"]["cards"][case_id]["param_card_sha256"],
                    "Admitted physical card differs from declared ancestry")
            # Capture ancestry even if a lane failed before publishing a result.
            # This collector routine follows only hash-bound predecessor records
            # and the original CLI file; no generated work tree is consulted.
            collector._generation_command(ReportEvidence(store), "campaigns/" + CAMPAIGN,
                                           case, generation, manifest)
            if generation.get("reused_from") is not None:
                proof = store.json(evidence + "/object-relink-proof.json")
                require(proof == generation["reused_from"]["object_relink_proof"]
                        and proof.get("passed") is True and proof.get("objects_unchanged") is True
                        and proof.get("original_executable_unchanged") is True
                        and proof["expected_executable_sha256"] == proof["relinked_executable_sha256"],
                        "Retained baseline relink proof differs or failed")
                store.add(evidence + "/tirust_benchmark_relink.py", proof["protocol_sha256"])
    numerical = collector.collect_numerical(ReportEvidence(store), "campaigns/" + CAMPAIGN,
                                            manifest, config, closed_cases=set(ids))
    for name in store.members:
        store.authorized(name)
    replayed = replay_supervisor(store, confirmation, spec, package)
    selected = [case for case in numerical["cases"] if case["id"] in ids]
    for case in selected:
        state = next(row for row in confirmation["closed_cases"] if row["case"] == case["id"])
        if state["status"] == "passed":
            require(case["all_six_complete"] and case["regression_pass"] is True,
                    "Passed supervisor entry failed independent six-lane replay: " + case["id"])
    flags = qualification_flags(confirmation)
    return {"schema": SCHEMA, "campaign_binding": confirmation["campaign_binding"],
        "case_ids": ids, "selected_case_count": len(ids), "selected_lane_slots": 6 * len(ids),
        **flags, "numerical": numerical, "supervisor_replay": replayed,
        "performance_acceptance_eligible": False, "overall_goal_qualified": False,
        "fpe_flags_observed": False, "internal_retry_counts_observed": False}


def qualification_flags(confirmation):
    ids = confirmation_cases(confirmation)
    passed = all(row["status"] == "passed" for row in confirmation["closed_cases"])
    terminal = confirmation.get("terminal_confirmation")
    return {"selected_status": "passed" if passed else "failed", "supervisor_terminal": terminal,
            "full_matrix_closed": ids == CASE_ORDER,
            "full_matrix_regression_passed": ids == CASE_ORDER and passed and terminal["exit_code"] == 0}


def load(package=HERE, expected_index_sha256=None):
    package = Path(package)
    raw = (package / "index.json").read_bytes()
    if expected_index_sha256 is not None:
        require(sha(raw) == expected_index_sha256, "Selected checkpoint index SHA differs")
    index = parsed(raw)
    require(index.get("schema") == SCHEMA and index.get("closed") is True
            and index.get("performance_acceptance_eligible") is False, "Unclosed numerical checkpoint")
    actual = {}
    for file in package.rglob("*"):
        require(not file.is_symlink(), "Symlink in checkpoint package")
        require(file.is_dir() or file.is_file(), "Nonregular checkpoint package member")
        if file.is_file() and file != package / "index.json":
            actual[safe(file.relative_to(package).as_posix())] = identity(file.read_bytes())
    require(actual == index["files"], "Outer checkpoint inventory differs")
    payload_name = safe(index["payload"])
    require(payload_name == "payload-" + actual[payload_name]["sha256"] + ".json.gz", "Payload is not content addressed")
    content = parsed(gzip.decompress((package / payload_name).read_bytes()))
    require(set(content) == {"schema", "content"} and content["schema"] == 1, "Unexpected payload schema")
    members = {}
    for name, value in index["retained_files"].items():
        raw = content["content"][value["sha256"]].encode("utf-8")
        require(identity(raw) == value, "Retained member identity differs")
        members[safe(name)] = text_only(raw, name)
    require(set(content["content"]) == {value["sha256"] for value in index["retained_files"].values()},
            "Unreferenced payload content")
    require(index["retained_file_count"] == len(members)
            and index["retained_bytes"] == sum(len(raw) for raw in members.values()),
            "Retained payload counts differ")
    verify_proposal(package)
    return index, Store(members)


def read_retained_json(name, package_root=None, expected_index_sha256=None):
    _, store = load(package_root or HERE, expected_index_sha256)
    return store.json(safe(name))


def replay_completed(package_root=None, expected_index_sha256=None):
    package = Path(package_root or HERE)
    index, store = load(package, expected_index_sha256)
    confirmation = store.json("closed-case-confirmation.json")
    summary = validate(store, confirmation, package)
    require(summary == parsed((package / "summary.json").read_bytes()), "Checkpoint summary differs from offline replay")
    require(summary["case_ids"] == index["case_ids"] and summary["full_matrix_closed"] == index["full_matrix_closed"],
            "Checkpoint index selection differs")
    return summary["supervisor_replay"]


def verify_proposal(package=HERE):
    package = Path(package)
    index = parsed((package / "source-index.json").read_bytes())
    require(set(index["files"]) == set(PACKAGE_SOURCES) - {"source-index.json"}, "Proposal source inventory differs")
    for name, value in index["files"].items():
        require(identity((package / name).read_bytes()) == value, "Proposal source changed: " + name)
    validation = index.get("validation_files", {})
    require(set(validation) <= {"tests.log", "dry-run.json"}, "Unexpected proposal validation file")
    for name, value in validation.items():
        require(identity(text_only((package / name).read_bytes(), name)) == value,
                "Proposal validation changed: " + name)
    return index


def seal(root, confirmation_path, destination, package=HERE):
    root, destination, package = Path(root).resolve(), Path(destination), Path(package)
    proposal = verify_proposal(package)
    require((root / "tools/tirust_benchmark.py").is_file(), "Run sealing from the repository root")
    require(not destination.exists(), "Checkpoint destination already exists; no overwrite or retry")
    raw_confirmation = Path(confirmation_path).read_bytes()
    confirmation = parsed(raw_confirmation)
    confirmation_cases(confirmation)
    # Exclusive creation makes even a failed seal append-only. No automatic
    # retry can replace a partial package or conceal its recorded failure.
    destination.mkdir(parents=True, exist_ok=False)
    try:
        store = Store({"closed-case-confirmation.json": text_only(raw_confirmation, "closed-case-confirmation.json")}, root)
        summary = validate(store, confirmation, package)
        store.unchanged()
        require(Path(confirmation_path).read_bytes() == raw_confirmation, "Closed-case confirmation changed during capture")
        for name in (*PACKAGE_SOURCES, *proposal.get("validation_files", {})):
            shutil.copyfile(package / name, destination / name)
        (destination / "summary.json").write_bytes(encoded(summary))
        content = {sha(raw): raw.decode("utf-8") for raw in store.members.values()}
        payload = gzip.compress(encoded({"schema": 1, "content": content}), compresslevel=9, mtime=0)
        payload_name = "payload-" + sha(payload) + ".json.gz"
        (destination / payload_name).write_bytes(payload)
        inventory = {name: identity(raw) for name, raw in sorted(store.members.items())}
        index = {"schema": SCHEMA, "closed": True, "campaign_binding": confirmation["campaign_binding"],
            "case_ids": summary["case_ids"], "full_matrix_closed": summary["full_matrix_closed"],
            "performance_acceptance_eligible": False, "overall_goal_qualified": False,
            "source_index_sha256": sha((package / "source-index.json").read_bytes()),
            "retained_files": inventory, "retained_file_count": len(inventory),
            "retained_bytes": sum(value["bytes"] for value in inventory.values()),
            "payload": payload_name, "input_allowlist": sorted(store.captured),
            "input_tree_allowlist": sorted(store.walked),
            "files": {file.name: identity(file.read_bytes()) for file in sorted(destination.iterdir())},
            "replay": {"path": "retain.py", "function": "replay_completed"}}
        (destination / "index.json").write_bytes(encoded(index))
        replay_completed(destination)
        return index
    except BaseException as error:
        (destination / "sealing-failure.json").write_bytes(encoded({"status": "failed", "error": repr(error),
            "performance_acceptance_eligible": False, "retry_authorized": False}))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")
    run = sub.add_parser("seal")
    run.add_argument("--confirmation", type=Path, required=True)
    run.add_argument("--destination", type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--package", type=Path, default=HERE)
    verify.add_argument("--index-sha256")
    args = parser.parse_args()
    if args.command == "seal":
        result = seal(Path.cwd(), args.confirmation, args.destination)
        print(json.dumps({"status": "sealed", "case_ids": result["case_ids"], "retained_files": result["retained_file_count"]}))
    elif args.command == "verify":
        result = replay_completed(args.package, args.index_sha256)
        print(json.dumps({"status": "portable-replay-passed", "case_ids": list(result), "native_calls": 0,
                          "performance_acceptance_eligible": False}))
    else:
        print(json.dumps({"status": "source-only-no-ingestion", "campaign": CAMPAIGN,
            "supervisor_session_id": SESSION, "case_order": CASE_ORDER,
            "requires": "explicit closed-case confirmation; full11 also requires observed supervisor termination",
            "active_case_reads": False, "native_calls": 0, "performance_acceptance_eligible": False}, indent=2))


if __name__ == "__main__":
    main()
