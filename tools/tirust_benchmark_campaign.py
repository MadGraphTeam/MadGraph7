"""Append-only benchmark campaigns with pinned inputs and failure ancestry.

The legacy report and frozen baseline are read-only here. A campaign owns its
generated outputs, cards, UFO import copy and report evidence. Registration does
not measure anything and does not select a result from an earlier campaign.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess


SCHEMA = "madloop-tirust-campaign-v1"
MODEL = "loop_qcd_qed_sm_Gmu"
NATIVE_NAMES = ("lib/libmadloop_tirust.so", "lib/libtirust.a", "lib/tirust-config.json")
HISTORICAL_STAGES = {"activate", "summarize", "validate-retained", "summary", "historical"}
QUALIFIED_LABELS = {"tirust-first", "backend-1", "backend-6", "backend-7", "backend-8"}


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", value):
        raise ValueError("Campaign ID must use letters, digits, underscores and hyphens")
    if value == "legacy":
        raise ValueError("legacy is reserved for the original campaign")
    return value


def _inside(root, path):
    root, path = Path(root).resolve(), Path(path)
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"Campaign path escapes its root: {path}")
    return path


def _paths(bench, campaign_id):
    campaign_id = _id(campaign_id)
    report = _inside(bench.LEGACY_REPORT, Path(bench.LEGACY_REPORT) / "campaigns" / campaign_id)
    work = _inside(bench.LEGACY_WORK, Path(bench.LEGACY_WORK) / "campaigns" / campaign_id)
    # No aliasing even to another campaign inside the same permitted root.
    for base, target in ((Path(bench.LEGACY_REPORT), report), (Path(bench.LEGACY_WORK), work)):
        relative = target.relative_to(base)
        cursor = base
        for part in relative.parts:
            cursor /= part
            if cursor.is_symlink():
                raise ValueError(f"Campaign directory may not be a symlink: {cursor}")
    return report, work


def _inventory(bench, directory):
    directory = Path(directory)
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Frozen campaign input may not be a symlink: {path}")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix not in (".pyc", ".pkl", ".log"):
            files[path.relative_to(directory).as_posix()] = bench.digest(path)
    return files


def _numerical_path(name):
    """Exclude exporter/protocol/git metadata from optimization eligibility."""
    path = Path(name)
    if name == "vendor/tirust/mtirust.f90":
        return True
    if name.startswith(("vendor/tirust/src/", "vendor/tirust/pole-kernels/src/")):
        return path.suffix == ".rs"
    return (name.startswith(("Template/loop_material/StandAlone/SubProcesses/",
                             "madgraph/iolibs/template_files/loop/",
                             "madgraph/iolibs/template_files/loop_optimized/"))
            and path.suffix.lower() in (".f", ".f90", ".inc"))


def _numerical_hash(name, data):
    # Ignore generated Fortran banners and comment-only edits. Preserve quoted
    # literals, line boundaries and all executable syntax. Rust is pinned as
    # exact production source bytes; installer/config/test files are excluded.
    if Path(name).suffix.lower() not in (".f", ".f90", ".inc"):
        return _hash(data)
    lines = []
    free_form = Path(name).suffix.lower() == ".f90"
    for line in data.decode().splitlines():
        if not free_form and line[:1] in ("C", "c", "*", "!"):
            continue
        quote, output, index = None, [], 0
        while index < len(line):
            character = line[index]
            if quote:
                output.append(character)
                if character == quote:
                    if index + 1 < len(line) and line[index + 1] == quote:
                        output.append(line[index + 1])
                        index += 1
                    else:
                        quote = None
            elif character in ("'", '"'):
                quote = character
                output.append(character)
            elif character == "!":
                break
            elif not character.isspace():
                output.append(character.lower())
            index += 1
        if output:
            lines.append("".join(output))
    return _hash("\n".join(lines).encode())


def _sources(bench, identity, historical=False):
    files = {}
    for name, checksum in sorted(identity.get("files", {}).items()):
        if checksum is None or not _numerical_path(name):
            continue
        path = Path(bench.ROOT) / name
        if path.is_file() and bench.digest(path) == checksum:
            data = path.read_bytes()
        elif historical:
            data = None
            for manifest_path in sorted((Path(bench.LEGACY_REPORT) / "campaigns").glob("*/manifest.json")):
                candidate = _read(manifest_path)["candidate"]
                if candidate["numerical_sources"].get(name, {}).get("sha256") != checksum:
                    continue
                snapshot = _inside(manifest_path.parent, manifest_path.parent / candidate["source_snapshot"] / name)
                if snapshot.is_file() and bench.digest(snapshot) == checksum:
                    data = snapshot.read_bytes()
                    break
            if data is None:
                data = subprocess.check_output(["git", "show", f"{identity['revision']}:{name}"], cwd=bench.ROOT)
            if _hash(data) != checksum:
                raise ValueError(f"Historical numerical source is unavailable: {name}")
        else:
            raise ValueError(f"Numerical source changed during registration: {name}")
        files[name] = {"sha256": checksum, "numerical_sha256": _numerical_hash(name, data)}
    if not files or not any(name.startswith("vendor/tirust/") for name in files):
        raise ValueError("Numerical source inventory lacks the TIRust bridge")
    return files


def _native(bench, prefix):
    prefix = Path(prefix).expanduser().resolve()
    _inside(bench.ROOT, prefix)
    files = {}
    for name in NATIVE_NAMES:
        path = _inside(prefix, prefix / name)
        if not path.is_file():
            raise ValueError(f"Incomplete native TIRust installation: {path}")
        files[name] = bench.digest(path)
    config = _read(prefix / "lib/tirust-config.json")
    for name, checksum in config.get("artifact_sha256", {}).items():
        relative = "lib/" + name
        if relative in files and files[relative] != checksum:
            raise ValueError(f"TIRust config disagrees with native artifact: {name}")
    # Preserve the full install/interface/ABI footprint, but only library bytes
    # count toward the numerical identity (metadata edits cannot reset a gate).
    for name in ("lib/tirust-link-before.txt", "lib/tirust-native-libs.txt",
                 "include/mtirust.f90", "include/tirust.h", "Cargo.toml", "Cargo.lock"):
        path = _inside(prefix, prefix / name)
        if path.is_file():
            files[name] = bench.digest(path)
    return {"prefix": str(prefix), "files": files}


def _native_from_inputs(inputs):
    runtime = inputs["runtime"]
    found = {}
    for section in ("libraries", "archives"):
        for name, checksum in runtime.get(section, {}).items():
            if Path(name).name in ("libmadloop_tirust.so", "libtirust.a"):
                key = "lib/" + Path(name).name
                if key in found and found[key] != checksum:
                    raise ValueError("Conflicting native TIRust artifacts in predecessor")
                found[key] = checksum
    if set(found) != {"lib/libmadloop_tirust.so", "lib/libtirust.a"}:
        raise ValueError("Predecessor lacks recorded native TIRust DSO/archive")
    return found


def _numerical_identity(bench, sources, native_files):
    return bench.content_digest({
        # Rust and the installed Fortran adapter execute through the native
        # artifacts. An edited but unbuilt bridge is not a new computation.
        # Repository MadLoop templates are compiled afresh in private outputs.
        "sources": {name: item["numerical_sha256"] for name, item in sources.items()
                    if not name.startswith("vendor/tirust/")},
        "native": {name: native_files[name] for name in NATIVE_NAMES[:2]}})


def _qualified_failures(bench, report):
    report = Path(report)
    failures = []
    for summary in sorted((report / "runs").glob("*/*/summary.json")):
        _inside(report, summary)
        value = _read(summary)
        if (value.get("label") not in QUALIFIED_LABELS or value.get("acceptance_eligible") is not True
                or value.get("passed") is not False):
            continue
        # Hash every retained raw observation and provenance file, including
        # non-selected attempts, not merely a mutable headline or selector.
        files = {path.relative_to(report).as_posix(): bench.digest(_inside(report, path))
                 for path in sorted(summary.parent.rglob("*"))
                 if path.is_file() and "__pycache__" not in path.parts
                 and path.suffix not in (".pyc", ".tmp")}
        failures.append({"case": summary.parent.parent.name, "label": value["label"],
                         "directory": summary.parent.relative_to(report).as_posix(), "files": files})
    return failures


def _predecessor(bench, predecessor):
    legacy = Path(bench.LEGACY_REPORT)
    report = legacy if predecessor == "legacy" else _paths(bench, predecessor)[0]
    summary = report / "summary.json"
    if not summary.is_file():
        raise ValueError("Predecessor must have a retained summary")
    manifest_path = report / "manifest.json"
    previous_manifest = _read(manifest_path) if predecessor != "legacy" else None
    failures = _qualified_failures(bench, report)
    if previous_manifest:
        numerical = previous_manifest["candidate"]["numerical_identity_sha256"]
    else:
        # The legacy campaign predates campaign manifests. Recover its actual
        # numerical inputs from a retained qualified failure, not current HEAD.
        if not failures:
            raise ValueError("Legacy predecessor has no retained qualified failure")
        inputs = _read(report / failures[0]["directory"] / "candidate-inputs.json")
        identity = inputs["generation"]["source_identity"]
        numerical = _numerical_identity(bench, _sources(bench, identity, historical=True),
                                        _native_from_inputs(inputs))
    discoveries = [{"path": path.relative_to(legacy).as_posix(), "sha256": bench.digest(path), "record": record}
                   for path, record in _read_discoveries(bench, report)] if previous_manifest else []
    if previous_manifest:
        ancestor_failures = _verify_predecessor(bench, previous_manifest["predecessor"])
        if any(item["record"] not in ancestor_failures for item in discoveries):
            raise ValueError("Inherited predecessor discovery or its failure evidence changed or disappeared")
    return {"id": predecessor, "report": report.relative_to(legacy).as_posix(),
            "manifest_sha256": bench.digest(manifest_path) if previous_manifest else None,
            "summary_sha256": bench.digest(summary), "qualified_failures": failures,
            "policy_sha256": bench.content_digest(_read(report / "cases.json")),
            "numerical_identity_sha256": numerical,
            "inherited_discoveries": discoveries,
            "ancestor": previous_manifest.get("predecessor") if previous_manifest else None}


def _verify_predecessor(bench, predecessor):
    legacy = Path(bench.LEGACY_REPORT)
    report = _inside(legacy, legacy / predecessor["report"])
    summary = (_inside(legacy, legacy / predecessor["summary_snapshot"])
               if predecessor.get("summary_snapshot") else report / "summary.json")
    if bench.digest(summary) != predecessor["summary_sha256"]:
        raise ValueError("Predecessor summary snapshot changed after campaign registration")
    if predecessor["manifest_sha256"] is not None:
        if bench.digest(report / "manifest.json") != predecessor["manifest_sha256"]:
            raise ValueError("Predecessor campaign manifest changed")
    current = _qualified_failures(bench, report)
    current_by_directory = {item["directory"]: item for item in current}
    for retained in predecessor["qualified_failures"]:
        if current_by_directory.get(retained["directory"]) != retained:
            raise ValueError("Predecessor qualified failures changed or were omitted")
    found = [{"predecessor_id": predecessor["id"], "report": predecessor["report"],
              "policy_sha256": predecessor["policy_sha256"], "failure": item} for item in current]
    ancestor = predecessor.get("ancestor")
    if ancestor:
        found += _verify_predecessor(bench, ancestor)
    for discovery in predecessor.get("inherited_discoveries", []):
        path = _inside(legacy, legacy / discovery["path"])
        if (not path.is_file() or bench.digest(path) != discovery["sha256"]
                or _read(path) != discovery["record"] or discovery["record"] not in found):
            raise ValueError("Inherited predecessor discovery or its failure evidence changed or disappeared")
    return found


def _registered_failures(predecessor):
    result = []
    while predecessor:
        result += [{"predecessor_id": predecessor["id"], "report": predecessor["report"],
                    "policy_sha256": predecessor["policy_sha256"], "failure": item}
                   for item in predecessor["qualified_failures"]]
        result += [item["record"] for item in predecessor.get("inherited_discoveries", [])]
        predecessor = predecessor.get("ancestor")
    return list({json.dumps(item, sort_keys=True): item for item in result}.values())


def _read_discoveries(bench, report):
    records = []
    for path in sorted((report / "inputs/predecessor/discoveries").glob("*.json")):
        _inside(report, path)
        record = _read(path)
        if path.stem != bench.content_digest(record):
            raise ValueError("Predecessor discovery record changed")
        records.append((path, record))
    return records


def _failure_numerical_identity(bench, record):
    origin = _inside(bench.LEGACY_REPORT, Path(bench.LEGACY_REPORT) / record["report"])
    inputs = _read(_inside(origin, origin / record["failure"]["directory"] / "candidate-inputs.json"))
    return _numerical_identity(bench, _sources(bench, inputs["generation"]["source_identity"], historical=True),
                               _native_from_inputs(inputs))


def _retain_discoveries(bench, report, manifest, found):
    """Later predecessor runs may finish; retain new failures without resealing."""
    directory = report / "inputs/predecessor/discoveries"
    registered = _registered_failures(manifest["predecessor"])
    known = {bench.content_digest(item) for item in registered}
    records = list(registered)
    for path, record in _read_discoveries(bench, report):
        records.append(record)
        known.add(path.stem)
    for record in records:
        origin = _inside(bench.LEGACY_REPORT, Path(bench.LEGACY_REPORT) / record["report"])
        for name, checksum in record["failure"]["files"].items():
            path = _inside(origin, origin / name)
            if not path.is_file() or bench.digest(path) != checksum:
                raise ValueError("Discovered predecessor failure evidence changed or disappeared")
        if record not in found:
            raise ValueError("Discovered predecessor qualified failure changed or disappeared")
    for record in found:
        identity = bench.content_digest(record)
        if identity in known:
            continue
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / (identity + ".json")
        try:
            with destination.open("x") as stream:
                json.dump(record, stream, sort_keys=True, indent=2)
                stream.write("\n")
        except FileExistsError:
            if _read(destination) != record:
                raise ValueError("Concurrent predecessor discovery disagrees")
    # Recheck even previously discovered records: writing a discovery must not
    # allow a later command to forget its disqualifying identity.
    for record in found:
        if (record["policy_sha256"] == manifest["shared_inputs"]["policy_sha256"]
                and _failure_numerical_identity(bench, record) == manifest["candidate"]["numerical_identity_sha256"]):
            raise ValueError("A qualified predecessor failure has this campaign's numerical identity/policy")


def _reject_renamed_failure(candidate, predecessor, policy):
    node = predecessor
    while node:
        if node["policy_sha256"] == policy and node["numerical_identity_sha256"] == candidate:
            raise ValueError("New campaign has unchanged numerical identity and policy; retained result remains authoritative")
        node = node.get("ancestor")


def _shared_files(bench, source_report):
    names = ["cases.json", "inputs/baseline.json", f"inputs/{MODEL}.json"]
    names += [path.relative_to(source_report).as_posix() for path in sorted((source_report / "points").glob("*"))
              if path.is_file() and path.suffix in (".dat", ".json")]
    config = _read(source_report / "cases.json")
    for case in config["cases"]:
        for suffix in (".dat", ".json"):
            if f"points/{case['id']}{suffix}" not in names:
                raise ValueError(f"Missing frozen points for campaign case {case['id']}")
    return {name: bench.digest(_inside(source_report, source_report / name)) for name in names}


def register_campaign(bench, campaign_id, tirust_prefix, predecessor="legacy", reason=""):
    """Register one namespace per numerical identity/policy, before any runs."""
    _paths(bench, campaign_id)
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Campaign registration requires an optimization reason")
    # Serialize identity checks and exclusive manifest publication across IDs;
    # two simultaneous registrations must not both claim the same experiment.
    directory = _inside(bench.LEGACY_WORK, Path(bench.LEGACY_WORK) / "campaigns")
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".registration.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _register_campaign(bench, campaign_id, tirust_prefix, predecessor, reason)


def _register_campaign(bench, campaign_id, tirust_prefix, predecessor, reason):
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Campaign registration requires an optimization reason")
    report, work = _paths(bench, campaign_id)
    if report.exists() or work.exists():
        raise ValueError("Campaign IDs are append-only; choose a new ID")
    previous = _predecessor(bench, predecessor)
    source_report = Path(bench.LEGACY_REPORT) / previous["report"]
    previous["summary_snapshot"] = f"campaigns/{campaign_id}/inputs/predecessor/summary.json"
    source_model = (Path(bench.LEGACY_WORK) / "models" / MODEL if predecessor == "legacy"
                    else _paths(bench, predecessor)[1] / "models" / MODEL)
    shared = _shared_files(bench, source_report)
    policy = bench.content_digest(_read(source_report / "cases.json"))
    model_pin = _read(source_report / f"inputs/{MODEL}.json")
    model_files = _inventory(bench, source_model)
    if model_files != model_pin["files"]:
        raise ValueError("Frozen source UFO differs from its retained pin")
    exporter = bench.source_identity(bench.ROOT, persist=False)
    sources = _sources(bench, exporter)
    native = _native(bench, tirust_prefix)
    numerical = _numerical_identity(bench, sources, native["files"])
    _reject_renamed_failure(numerical, previous, policy)
    for failure in _registered_failures(previous):
        if failure["policy_sha256"] == policy and _failure_numerical_identity(bench, failure) == numerical:
            raise ValueError("A qualified predecessor failure already has this numerical identity/policy")
    # Registering duplicates before either finishes would permit selecting a
    # later favorable timing. The numerical bytes and policy define one run.
    for existing in sorted((Path(bench.LEGACY_REPORT) / "campaigns").glob("*/manifest.json")):
        old = _read(existing)
        if (old["candidate"]["numerical_identity_sha256"] == numerical
                and old["shared_inputs"]["policy_sha256"] == policy):
            raise ValueError("An existing campaign already registered this numerical identity/policy")
    baseline_record = Path(bench.BASELINE_RECORD)
    record = _read(baseline_record)
    if record["revision"] != bench.BASELINE_REVISION:
        raise ValueError("Unexpected shared baseline revision")
    manifest = {"schema": SCHEMA, "id": campaign_id, "created_utc": bench.utc_now(), "reason": reason,
        "predecessor": previous,
        "baseline": {"revision": bench.BASELINE_REVISION, "record_sha256": bench.digest(baseline_record)},
        "shared_inputs": {"files": shared, "policy_sha256": policy, "model_tree": model_files},
        "candidate": {"exporter_identity": exporter, "numerical_sources": sources, "native": native,
                      "source_snapshot": "inputs/numerical-sources",
                      "numerical_identity_sha256": numerical}}
    report.parent.mkdir(parents=True, exist_ok=True)
    work.parent.mkdir(parents=True, exist_ok=True)
    report.mkdir()
    created_work = False
    try:
        work.mkdir()
        created_work = True
        for name in shared:
            destination = report / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_report / name, destination)
        snapshot = report / "inputs/predecessor/summary.json"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_report / "summary.json", snapshot)
        for name, identity in sources.items():
            snapshot = _inside(report, report / manifest["candidate"]["source_snapshot"] / name)
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(_inside(bench.ROOT, Path(bench.ROOT) / name), snapshot)
            if bench.digest(snapshot) != identity["sha256"]:
                raise ValueError("Numerical source changed while snapshotting campaign inputs")
        destination = work / "models" / MODEL
        shutil.copytree(source_model, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pkl", "*.log"))
        if _inventory(bench, destination) != model_files:
            raise ValueError("UFO changed while copying campaign inputs")
        for name, checksum in shared.items():
            if bench.digest(report / name) != checksum:
                raise ValueError("Frozen report input changed during registration")
        _verify_predecessor(bench, previous)
        # Exclusive publication seals preregistration. No command runs before
        # activate_campaign verifies this manifest and its copied input bytes.
        with (report / "manifest.json").open("x") as stream:
            json.dump(manifest, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
    except BaseException:
        shutil.rmtree(report)
        if created_work:
            shutil.rmtree(work)
        raise
    return manifest


def activate_campaign(bench, campaign_id):
    """Select private campaign paths; leave the single shared baseline intact."""
    report, work = _paths(bench, campaign_id)
    manifest_path = report / "manifest.json"
    manifest = _read(manifest_path)
    if manifest.get("schema") != SCHEMA or manifest.get("id") != campaign_id:
        raise ValueError("Campaign manifest identity/schema mismatch")
    bench.REPORT, bench.WORK, bench.MODEL_ROOT = report, work, work / "models"
    bench.TIRUST_PREFIX = Path(manifest["candidate"]["native"]["prefix"])
    bench.CAMPAIGN_ID, bench.CAMPAIGN_MANIFEST = campaign_id, manifest
    bench.CAMPAIGN_MANIFEST_SHA256 = bench.digest(manifest_path)
    verify_campaign(bench, stage="activate")
    return {"id": campaign_id, "report": report, "work": work,
            "tirust_prefix": bench.TIRUST_PREFIX, "manifest": manifest,
            "manifest_sha256": bench.CAMPAIGN_MANIFEST_SHA256}


def campaign_binding(bench):
    if not getattr(bench, "CAMPAIGN_ID", None):
        return None
    return {"id": bench.CAMPAIGN_ID, "manifest_sha256": bench.CAMPAIGN_MANIFEST_SHA256}


def campaign_summary_fields(bench):
    binding = campaign_binding(bench)
    if binding is None:
        return {}
    manifest = bench.CAMPAIGN_MANIFEST
    previous = manifest["predecessor"]
    failures = _registered_failures(previous)
    failures += [record for path, record in _read_discoveries(bench, Path(bench.REPORT))]
    return {"campaign_binding": binding,
            "campaign_numerical_identity_sha256": manifest["candidate"]["numerical_identity_sha256"],
            "campaign_predecessor": {"id": previous["id"], "manifest_sha256": previous["manifest_sha256"],
                "summary_sha256": previous["summary_sha256"],
                "qualified_failures": [{"predecessor_id": item["predecessor_id"],
                    "case": item["failure"]["case"], "label": item["failure"]["label"],
                    "directory": item["failure"]["directory"],
                    "evidence_sha256": bench.content_digest(item["failure"]["files"])}
                    for item in failures]}}


def verify_campaign(bench, stage="run", inputs=None):
    """Verify current preregistration, or retained evidence for historical use."""
    if not getattr(bench, "CAMPAIGN_ID", None):
        return {}
    report, work = _paths(bench, bench.CAMPAIGN_ID)
    manifest = bench.CAMPAIGN_MANIFEST
    if (Path(bench.REPORT) != report or Path(bench.WORK) != work
            or bench.digest(report / "manifest.json") != bench.CAMPAIGN_MANIFEST_SHA256
            or _read(report / "manifest.json") != manifest):
        raise ValueError("Campaign manifest or selected namespace changed")
    if Path(bench.BASELINE).resolve() != (Path(bench.LEGACY_WORK) / "baseline").resolve():
        raise ValueError("Campaign must use the one frozen baseline source")
    if bench.digest(bench.BASELINE_RECORD) != manifest["baseline"]["record_sha256"]:
        raise ValueError("Shared frozen baseline record changed")
    for name, checksum in manifest["shared_inputs"]["files"].items():
        if bench.digest(_inside(report, report / name)) != checksum:
            raise ValueError(f"Frozen campaign input changed: {name}")
    if _inventory(bench, Path(bench.MODEL_ROOT) / MODEL) != manifest["shared_inputs"]["model_tree"]:
        raise ValueError("Private campaign UFO changed")
    candidate = manifest["candidate"]
    for name, identity in candidate["numerical_sources"].items():
        snapshot = _inside(report, report / candidate["source_snapshot"] / name)
        if not snapshot.is_file() or bench.digest(snapshot) != identity["sha256"]:
            raise ValueError("Frozen campaign numerical source snapshot changed")
    discovered = _verify_predecessor(bench, manifest["predecessor"])
    _retain_discoveries(bench, report, manifest, discovered)
    if stage not in HISTORICAL_STAGES:
        # Defense for namespaces created by older harness versions, before
        # duplicate identities were prohibited at registration.
        for other_path in sorted((Path(bench.LEGACY_REPORT) / "campaigns").glob("*/manifest.json")):
            if other_path.parent == report:
                continue
            other = _read(other_path)
            if (other["candidate"]["numerical_identity_sha256"] == candidate["numerical_identity_sha256"]
                    and other["shared_inputs"]["policy_sha256"] == manifest["shared_inputs"]["policy_sha256"]
                    and _qualified_failures(bench, other_path.parent)):
                raise ValueError("A sibling campaign already has a qualified failure for this numerical identity/policy")
        current = bench.source_identity(bench.ROOT, persist=False)
        if current["files_sha256"] != candidate["exporter_identity"]["files_sha256"]:
            raise ValueError("Candidate exporter/source bytes changed after preregistration")
        if _sources(bench, current) != candidate["numerical_sources"]:
            raise ValueError("Numerical source inventory changed after preregistration")
        if _native(bench, bench.TIRUST_PREFIX) != candidate["native"]:
            raise ValueError("Native TIRust installation changed after preregistration")
    if inputs is not None:
        record = inputs
        generation = record.get("generation", record)
        binding = record.get("campaign_binding", generation.get("campaign_binding"))
        if binding != campaign_binding(bench):
            raise ValueError("Recorded inputs belong to a different campaign")
        if generation.get("source") == "candidate" or "source_identity" in generation:
            identity = generation.get("source_identity", {})
            if generation.get("source") != "baseline" and identity.get("files_sha256") != candidate["exporter_identity"]["files_sha256"]:
                raise ValueError("Recorded candidate source differs from campaign preregistration")
        if "runtime" in record and generation.get("source") != "baseline":
            runtime_files = {}
            native_basenames = {Path(name).name for name in NATIVE_NAMES}
            for section in ("libraries", "archives", "metadata"):
                for name, checksum in record["runtime"].get(section, {}).items():
                    basename = Path(name).name
                    if basename not in native_basenames:
                        continue
                    if basename in runtime_files and runtime_files[basename] != checksum:
                        raise ValueError("Conflicting recorded runtime library basenames")
                    runtime_files[basename] = checksum
            for name in NATIVE_NAMES:
                if runtime_files.get(Path(name).name) != candidate["native"]["files"][name]:
                    raise ValueError(f"Recorded TIRust runtime differs from campaign: {name}")
    return campaign_summary_fields(bench)
