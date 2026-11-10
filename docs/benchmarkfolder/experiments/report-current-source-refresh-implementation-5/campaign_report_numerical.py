"""Read-only numerical report ingestion for registered 100-point warmup runs.

Only retained files enter this collector. It neither imports the live campaign
harness nor opens paths recorded in machine metadata. Selected pure functions
from each hash-verified historical parser are replayed against in-memory raw
files; module top-level code, subprocesses and native libraries are not run.
"""
from __future__ import annotations

import ast
import hashlib
import io
import json
import math
from pathlib import Path
import re
from types import SimpleNamespace


SCHEMA = "madloop-tirust-campaign-numerical-report-v1"
BASELINE = "9228fc58e8f8e48dadc70a6fcfb6617845aaeb1f"
BACKENDS = (1, 6, 7, 8, 9)
LANES = (("baseline", 1), *(("candidate", b) for b in BACKENDS))
PROTOCOLS = {"tirust_benchmark_numerical.py", "tirust_benchmark.py",
             "tirust_benchmark_worker.py", "tirust_benchmark_worker.f",
             "tirust_benchmark_compiler.py", "tirust_benchmark_campaign.py",
             "tirust_benchmark_warmup.py"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _canonical(value):
    return _sha(json.dumps(value, sort_keys=True, separators=(",", ":"),
                           allow_nan=False).encode())


def _relative(name):
    _require(isinstance(name, str) and name and "\\" not in name,
             "Invalid report-relative path")
    path = Path(name)
    _require(not path.is_absolute() and ".." not in path.parts,
             "Report path escapes its retained directory")
    return path.as_posix()


def _join(base, name):
    return (Path(_relative(base)) / _relative(name)).as_posix()


def _exists(evidence, name):
    if hasattr(evidence, "exists"):
        return evidence.exists(_relative(name))
    path = Path(evidence.root)
    for part in Path(_relative(name)).parts:
        path /= part
        _require(not path.is_symlink(), "Evidence symlink: " + name)
    return path.exists()


def _functions(raw, names, namespace, filename):
    """Load named definitions only, never the retained program's top level."""
    parsed = ast.parse(raw.decode(), filename=filename)
    selected = [node for node in parsed.body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    _require(len(selected) == len(names) and {node.name for node in selected} == set(names),
             "Missing or duplicated retained parser definition: " + filename)
    for node in selected:
        _require(not node.decorator_list and not any(isinstance(child, (ast.Import, ast.ImportFrom))
                    for child in ast.walk(node)), "Retained parser is not a pure definition")
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(module, filename, "exec"), namespace)
    return namespace


def _literal(raw, name):
    values = [node.value for node in ast.parse(raw.decode()).body
              if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)]
    _require(len(values) == 1, "Missing retained parser constant: " + name)
    return ast.literal_eval(values[0])


class _MemoryPath:
    """The warmup parser sees only the two already-bound raw files."""
    def __init__(self, files, name=""):
        self.files, self.name = files, name

    def __truediv__(self, name):
        _require(not self.name and name in ("warmup.dat", "dimensions.dat"),
                 "Unexpected raw parser path")
        return _MemoryPath(self.files, name)

    def is_file(self):
        return self.name in self.files

    def open(self, *, encoding="utf-8", newline=None):
        return io.StringIO(self.files[self.name].decode(encoding), newline=newline)


def _parsers(protocols, directory):
    worker_raw = protocols["tirust_benchmark_worker.py"]
    worker = _functions(worker_raw, ("MeasurementError", "validate_status"),
                        {"BACKEND_CODES": _literal(worker_raw, "BACKEND_CODES")},
                        directory + "/tirust_benchmark_worker.py")
    worker = SimpleNamespace(**{key: value for key, value in worker.items() if key != "__builtins__"})
    warmup = _functions(protocols["tirust_benchmark_warmup.py"],
                        ("_call_groups", "parse_warmup"),
                        {"worker": worker, "MeasurementError": worker.MeasurementError, "math": math},
                        directory + "/tirust_benchmark_warmup.py")
    comparison = _functions(protocols["tirust_benchmark.py"], ("compare_rows",), {},
                            directory + "/tirust_benchmark.py")["compare_rows"]
    compiler_raw = protocols["tirust_benchmark_compiler.py"]
    compiler = _functions(compiler_raw, ("_signature", "validate_compiler_equality"),
                          {"SCHEMA": _literal(compiler_raw, "SCHEMA")},
                          directory + "/tirust_benchmark_compiler.py")["validate_compiler_equality"]
    return warmup, comparison, compiler


def _export_environment(generation, manifest):
    expected = {key: manifest[key] for key in ("export_environment_binding_version", "export_environment")}
    reused = generation.get("reused_from")
    if reused is not None:
        _require(generation["source"] == "baseline" and isinstance(reused, dict),
                 "Only baseline generation can inherit source provenance")
        _require(reused.get("reuse_environment_binding") == expected
                 and reused.get("source_generation_repeated") is False
                 and reused.get("source_export_environment_recorded") == ("export_environment" in generation)
                 and reused.get("source_export_environment") == generation.get("export_environment"),
                 "Inherited baseline export environment differs")
        if "export_environment" not in generation:
            return
    _require(all(generation.get(key) == value for key, value in expected.items()),
             "Generation export environment differs from the registered campaign")


def _runtime_names(runtime):
    result = {}
    for section in ("libraries", "archives", "metadata"):
        _require(isinstance(runtime.get(section), dict), "Missing runtime dependency inventory")
        for name, checksum in runtime[section].items():
            _require(re.fullmatch(r"[0-9a-f]{64}", checksum) is not None,
                     "Invalid runtime dependency hash")
            basename = Path(name).name
            # Nix may retain distinct loader paths with the same basename.
            # Keep all hashes; the original full paths remain bound in inputs.
            result.setdefault(basename, set()).add(checksum)
    return {name: tuple(sorted(values)) for name, values in result.items()}


def _generation_command(evidence, campaign, case, generation, manifest):
    """Resolve inherited baseline CLI evidence only through sealed ancestry."""
    visited = set()
    while generation.get("reused_from") is not None:
        _require(campaign not in visited and generation["source"] == "baseline",
                 "Invalid inherited generation ancestry")
        visited.add(campaign)
        reused = generation["reused_from"]
        previous = manifest["predecessor"]
        campaign = "." if previous["id"] == "legacy" else _join("campaigns", previous["id"])
        provenance = _join(campaign, f"inputs/baseline/{case['id']}")
        old_generation = evidence.json(_join(provenance, "generation.json"), reused["generation_sha256"])
        evidence.add(_join(provenance, "compilation.json"), reused["compilation_sha256"])
        _require(all(old_generation[key] == generation[key] for key in
                     ("case", "source", "source_identity", "command_sha256", "parameter_card_sha256")),
                 "Inherited baseline generation differs from its recorded origin")
        generation = old_generation
        if campaign != ".":
            manifest = evidence.json(_join(campaign, "manifest.json"), previous["manifest_sha256"])
        else:
            _require(generation.get("reused_from") is None, "Legacy baseline unexpectedly inherits a campaign")
    command = _join(campaign, f"commands/{generation['source']}-{case['id']}.mg5")
    evidence.add(command, generation["command_sha256"])


def _inputs(evidence, campaign, directory, source, backend, case, spec, config,
            manifest, binding, protocols, baseline_record):
    lane = _join(directory, f"{source}-{backend}")
    inputs = evidence.json(_join(lane, source + "-inputs.json"))
    generation, compilation = inputs["generation"], inputs["compilation"]
    _require(all(record.get("campaign_binding") == binding for record in (inputs, generation, compilation)),
             "Lane generation/build/input belongs to another campaign")
    _require(generation["case"] == case and generation["source"] == source,
             "Generation case/source differs from lane")
    _export_environment(generation, manifest)
    provenance = _join(campaign, f"inputs/{source}/{case['id']}")
    evidence.tree(provenance)
    _require(evidence.json(_join(provenance, "generation.json")) == generation
             and evidence.json(_join(provenance, "compilation.json")) == compilation,
             "Lane provenance differs from retained process input records")
    for scope, checksum in compilation.get("build_logs", {}).items():
        _require(scope in ("process", "MODEL", "DHELAS"), "Unknown numerical build log scope")
        evidence.add(_join(provenance, "compilation-" + scope + ".log"), checksum)
    reused = generation.get("reused_from")
    if reused is not None:
        proof = evidence.json(_join(provenance, "object-relink-proof.json"))
        _require(proof == reused["object_relink_proof"]
                 and proof.get("passed") is True and proof.get("objects_unchanged") is True
                 and proof.get("original_executable_unchanged") is True
                 and proof["expected_executable_sha256"] == proof["relinked_executable_sha256"],
                 "Inherited baseline object-relink proof differs or failed")
        evidence.add(_join(provenance, "tirust_benchmark_relink.py"), proof["protocol_sha256"])
    card_hash = _sha(evidence.add(_join(provenance, "param_card.dat")))
    _require(card_hash == inputs["param_card_sha256"] == generation["parameter_card_sha256"],
             "Physical card identity differs")
    _generation_command(evidence, campaign, case, generation, manifest)
    identity = generation["source_identity"]
    if source == "candidate":
        registered = manifest["candidate"]["exporter_identity"]
        # Documentation-only commits change the recorded HEAD without changing
        # the exporter. Bind every source byte and preserve the observed identity
        # separately; never rewrite a generation record to the registered HEAD.
        _require(identity["files"] == registered["files"]
                 and identity["files_sha256"] == registered["files_sha256"]
                 and _canonical(identity["files"]) == identity["files_sha256"],
                 "Candidate exporter identity differs from preregistration")
        snapshot = _join(campaign, "inputs/source-identities/" + _canonical(identity) + ".json")
        _require(evidence.json(snapshot) == identity, "Retained candidate source identity differs")
    else:
        _require(identity["revision"] == BASELINE == baseline_record["revision"]
                 and identity["files_sha256"] == _canonical(baseline_record["files"])
                 and identity["archive_sha256"] == baseline_record["archive_sha256"],
                 "Baseline source differs from its frozen record")
    point_name = "points/" + case["id"]
    shared = manifest["shared_inputs"]["files"]
    point_raw = evidence.add(_join(campaign, point_name + ".dat"), shared[point_name + ".dat"])
    metadata = evidence.json(_join(campaign, point_name + ".json"), shared[point_name + ".json"])
    _require(inputs["points_sha256"] == _sha(point_raw) == metadata["sha256"]
             and inputs["points_metadata_sha256"] == shared[point_name + ".json"]
             and metadata["inputs"]["count"] == 100
             and len(metadata["points"]) == 100
             and metadata["inputs"]["incoming_masses"] + metadata["inputs"]["masses"] == compilation["masses"],
             "Saved points, count, metadata or compiled masses differ")
    point_lines = point_raw.splitlines()
    _require(int(point_lines[0]) == 100 and len(point_lines) == 1 + 100 * (case["nfinal"] + 2),
             "Saved physical point stream is incomplete")
    settings = inputs["settings"]
    expected = dict(config["madloop_parameters"])
    if source == "candidate":
        expected.update(config["candidate_parameters"])
    _require(settings["mlreductionlib"] == str(backend)
             and settings["ctmoderun"] == -1 and settings["ctmodeinit"] == 1
             and all(settings.get(key.lower()) == value for key, value in expected.items()),
             "Lane numerical policy differs from campaign")
    cards = inputs["numerical_cards"]
    _require(isinstance(cards, dict) and "Cards/MadLoopParams.dat" in cards
             and len(set(cards.values())) == 1
             and all(re.fullmatch(r"[0-9a-f]{64}", value) for value in cards.values()),
             "Missing or inconsistent numerical-card hash inventory")
    _require(inputs["worker_timeout_seconds"] == spec["worker_timeout_seconds"],
             "Recorded worker deadline differs")
    _require(compilation.get("numerical_build_from_sources") is True,
             "Numerical compilation lacks source-build provenance")
    template = protocols["tirust_benchmark_worker.f"]
    rendered = template.decode()
    parameters = compilation["worker_render_parameters"]
    _require(parameters["nexternal"] == case["nfinal"] + 2, "Compiled process multiplicity differs")
    for key, token in (("prefix", "PREFIX"), ("nexternal", "NEXTERNAL"), ("nborn", "NBORN")):
        rendered = rendered.replace("@" + token + "@", str(parameters[key]))
    _require(compilation["worker_template_sha256"] == _sha(template)
             and compilation["driver_sha256"] == _sha(rendered.encode()),
             "Compiled worker differs from retained protocol")
    drivers = [value for name, value in compilation["sources"].items()
               if re.fullmatch(r"SubProcesses/P[^/]+/check_sa\.f", name)]
    _require(drivers == [compilation["driver_sha256"]], "Compiled driver source inventory differs")
    for name, text in compilation["settings"].items():
        _require(_sha(text.encode()) == compilation["sources"][name], "Compiled make settings differ")
    _require(compilation["link_inputs_before"] == compilation["link_inputs_after"],
             "Link-time dependencies changed")
    for section in ("archives", "metadata"):
        _require(inputs["runtime"][section] == compilation["link_inputs_after"][section],
                 "Runtime differs from linked dependency inputs")
    runtime = _runtime_names(inputs["runtime"])
    if source == "candidate":
        for name in ("lib/libmadloop_tirust.so", "lib/libtirust.a", "lib/tirust-config.json"):
            _require(runtime.get(Path(name).name) == (manifest["candidate"]["native"]["files"][name],),
                     "Recorded TIRust native identity differs from campaign")
    _require(re.fullmatch(r"[0-9a-f]{64}", compilation["executable_sha256"]) is not None,
             "Missing executable identity")
    after = evidence.json(_join(lane, "inputs-after.json"))
    _require(after["executable_sha256"] == compilation["executable_sha256"]
             and after["points_sha256"] == inputs["points_sha256"]
             and after["numerical_cards"] == cards
             and all(after["runtime"][key] == inputs["runtime"][key]
                     for key in ("libraries", "archives", "metadata")),
             "Post-evaluation input binding differs")
    return inputs, {
        "generator_revision": identity["revision"], "generator_files_sha256": identity["files_sha256"],
        "executable_sha256": compilation["executable_sha256"],
        "driver_sha256": compilation["driver_sha256"],
        "compiler_policy_sha256": _canonical(compilation["effective_compiler_flags"]),
        "points_sha256": inputs["points_sha256"], "physical_card_sha256": card_hash,
        "runtime_libraries_sha256": _canonical(inputs["runtime"]["libraries"])}


def _empty_lane(source, backend, path):
    return {"source": source, "backend": backend, "status": "pending", "points": None,
            "accepted_dp_count": None, "accepted_qp_count": None, "status_counts": {},
            "regression_pass": None, "ordinary_target_pass": None, "maximum_relative_error": None,
            "failures": [], "ordinary_target_misses": [], "identity": None, "error": None,
            "evidence_path": path}


def collect_numerical(evidence, campaign_dir, manifest, config, closed_cases=None):
    """Collect all eleven cases and six explicit lanes, preserving missing data.

    Integrity failures in a published lane are returned as failed report rows;
    a changed campaign seal raises ValueError. A failure marker always wins.
    Every declared case is returned even while other campaign lanes are active.
    """
    root = Path(evidence.root).resolve()
    candidate = Path(campaign_dir)
    campaign = _relative(candidate.relative_to(root).as_posix() if candidate.is_absolute()
                         else candidate.as_posix())
    retained_manifest = evidence.json(_join(campaign, "manifest.json"))
    _require(retained_manifest == manifest and manifest["schema"] == "madloop-tirust-campaign-v1",
             "Selected campaign manifest differs")
    binding = {"id": manifest["id"], "manifest_sha256": _sha(evidence.add(_join(campaign, "manifest.json")))}
    shared = manifest["shared_inputs"]["files"]
    _require(evidence.json(_join(campaign, "cases.json"), shared["cases.json"]) == config
             and _canonical(config) == manifest["shared_inputs"]["policy_sha256"]
             and config["baseline_revision"] == manifest["baseline"]["revision"] == BASELINE
             and config["points_per_case"] == 100 and config["relative_tolerance"] == 3e-6
             and config["ordinary_target_tolerance"] == 1e-8
             and tuple(config["backends"]) == BACKENDS
             and len(config["cases"]) == 11
             and len({case["id"] for case in config["cases"]}) == 11,
             "Selected campaign configuration differs from the required matrix")
    baseline_record = evidence.json(_join(campaign, "inputs/baseline.json"),
                                    manifest["baseline"]["record_sha256"])
    # Bind the frozen model/input closure and numerical exporter snapshot once.
    for name, checksum in shared.items():
        evidence.add(_join(campaign, name), checksum)
    for name, identity in manifest["candidate"]["numerical_sources"].items():
        evidence.add(_join(_join(campaign, manifest["candidate"]["source_snapshot"]), name), identity["sha256"])
    cases = []
    for case in config["cases"]:
        directory = _join(campaign, "experiments/numerical/full100-warmup-" + case["id"] + "-1")
        lanes = [_empty_lane(source, backend, _join(directory, f"{source}-{backend}")) for source, backend in LANES]
        # A deliberately frozen partial report must never open an active run,
        # even if that run publishes another complete lane during collection.
        if closed_cases is not None and case["id"] not in closed_cases:
            cases.append({**case, "directory": directory, "status": "pending",
                          "complete_lanes": 0, "all_six_complete": False,
                          "regression_pass": None, "ordinary_target_pass": None,
                          "error": None, "lanes": lanes})
            continue
        parsers = protocols = spec = top = None
        case_error = None
        if _exists(evidence, directory):
            evidence.tree(directory)
            try:
                spec = evidence.json(_join(directory, "specification.json"))
                _require(spec["campaign_binding"] == binding and spec["case"] == case
                         and spec["performance_eligible"] is False
                         and spec["measurement_phase"] == "steady-warmup"
                         and spec["worker_timeout_seconds"] == manifest["administrative_worker_timeout_seconds"][case["id"]],
                         "Numerical run scope, campaign, case or deadline differs")
                _require(evidence.json(_join(directory, "campaign-manifest.json")) == manifest
                         and _sha(evidence.add(_join(directory, "campaign-manifest.json"))) == binding["manifest_sha256"]
                         and evidence.json(_join(directory, "cases.json")) == config
                         and _sha(evidence.add(_join(directory, "cases.json"))) == spec["configuration_sha256"] == shared["cases.json"],
                         "Numerical run manifest/configuration differs")
                _require(PROTOCOLS.issubset(spec["protocols"]), "Numerical run omits a required retained protocol")
                protocols = {name: evidence.add(_join(directory, _relative(name)), checksum)
                             for name, checksum in spec["protocols"].items()}
                parsers = _parsers(protocols, directory)
                if _exists(evidence, _join(directory, "results.json")):
                    top = evidence.json(_join(directory, "results.json"))
                    _require(top["case"] == case and top["performance_eligible"] is False,
                             "Top-level numerical results have a different scope")
                    _require([(r["source"], r["backend"]) for r in top["results"]]
                             == list(LANES[:len(top["results"])]) and len(top["results"]) <= len(LANES),
                             "Numerical results omit, duplicate or reorder published lanes")
            except (OSError, ValueError, KeyError, TypeError, SyntaxError) as error:
                case_error = str(error)
        observations, validated_inputs = {}, {}
        for position, lane in enumerate(lanes):
            source, backend = lane["source"], lane["backend"]
            path = lane["evidence_path"]
            if not _exists(evidence, path):
                continue
            try:
                if _exists(evidence, _join(path, "failure.json")):
                    failure = evidence.json(_join(path, "failure.json"))
                    _require(failure["source"] == source and failure["backend"] == backend
                             and failure["status"] == "failed" and failure["performance_eligible"] is False,
                             "Failure marker identity differs")
                    lane.update(status="failed", error={"kind": "recorded_failure", "message": failure.get("error"),
                                                        "record": failure})
                    if case_error:
                        lane["error"]["validation_error"] = case_error
                    continue
                if case_error:
                    raise ValueError(case_error)
                if not _exists(evidence, _join(path, "result.json")) or top is None or position >= len(top["results"]):
                    continue
                result = evidence.json(_join(path, "result.json"))
                _require(result == top["results"][position] and result["source"] == source
                         and result["backend"] == backend and result["status"] == "complete"
                         and result["performance_eligible"] is False, "Published lane result identity differs")
                inputs, identity = _inputs(evidence, campaign, directory, source, backend, case, spec,
                                            config, manifest, binding, protocols, baseline_record)
                warmup, compare, compiler_check = parsers
                compiler_check(inputs["compilation"]["effective_compiler_flags"],
                               inputs["compilation"]["effective_compiler_flags"])
                raw_path = _join(path, "raw")
                raw = {name: evidence.add(_join(raw_path, name)) for name in ("warmup.dat", "dimensions.dat")}
                warmup["Path"] = lambda unused, files=raw: _MemoryPath(files)
                ready = evidence.json(_join(raw_path, "ready.json"))
                _require(ready["nborn"] == inputs["compilation"]["worker_render_parameters"]["nborn"],
                         "READY Born dimension differs from the compiled worker")
                observed = warmup["parse_warmup"]("retained-raw", [backend], 100, ready)
                _require(observed == evidence.json(_join(raw_path, "observation.json")) == result["observation"],
                         "Raw warmup reparse differs from stored observation")
                _require(observed["measurement_phase"] == "steady-warmup"
                         and observed["performance_eligible"] is False
                         and observed["seconds"] is None and observed["seconds_per_point"] is None
                         and observed["timed_status_counts"] is None,
                         "Warmup evidence cannot supply runtime measurements")
                _require(all((r["status"] // 100, r["status"] // 10 % 10) in ((2, 1), (3, 2))
                             for r in observed["rows"]), "Invalid steady precision/status tuple")
                if source in validated_inputs:
                    previous = validated_inputs[source]
                    _require(inputs["generation"] == previous["generation"]
                             and inputs["compilation"] == previous["compilation"],
                             "Backend lanes used different source/build records")
                if observations:
                    previous = next(iter(validated_inputs.values()))
                    _require(inputs["param_card_sha256"] == previous["param_card_sha256"]
                             and inputs["points_sha256"] == previous["points_sha256"]
                             and inputs["compilation"]["compiler"] == previous["compilation"]["compiler"],
                             "Lane physical inputs or compiler differ")
                    compiler_check(previous["compilation"]["effective_compiler_flags"],
                                   inputs["compilation"]["effective_compiler_flags"])
                    common = set(previous["settings"]) & set(inputs["settings"]) - {"mlreductionlib"}
                    _require(all(previous["settings"][key] == inputs["settings"][key] for key in common),
                             "Cross-backend numerical settings differ")
                    old_runtime, new_runtime = _runtime_names(previous["runtime"]), _runtime_names(inputs["runtime"])
                    # Model/HELAS archives are generated per source. IREGI is
                    # also present in generic link inventories, but backend 3
                    # is absent from every explicit chain in this matrix.
                    _require(all(old_runtime[key] == new_runtime[key] for key in old_runtime.keys() & new_runtime.keys()
                                 if key not in {"libmodel.a", "libdhelas.a", "libiregi.a"}),
                             "Shared numerical libraries differ")
                reference = observed if source == "baseline" else observations.get(("baseline", 1))
                comparison = compare(reference["rows"], observed["rows"], config["relative_tolerance"]) if reference else None
                _require(comparison == result["comparison"], "Saved cross-backend comparison differs from raw results")
                usage = observed["backend_precision_counts"]
                lane.update(status="complete", points=100, accepted_dp_count=usage.get(f"{backend}:DP", 0),
                            accepted_qp_count=usage.get(f"{backend}:QP", 0),
                            status_counts=observed["status_counts"], identity=identity)
                if comparison is not None:
                    misses = [row for row in comparison["residuals"]
                              if not ((row["symmetric_relative_error"] is not None
                                       and row["symmetric_relative_error"] < 1e-8)
                                      or (row["pole_to_finite"] is not None and row["pole_to_finite"] < 1e-8))]
                    lane.update(regression_pass=comparison["passed"], ordinary_target_pass=comparison["ordinary_target_passed"],
                                maximum_relative_error=comparison["maximum_relative_error"],
                                failures=comparison["failures"], ordinary_target_misses=misses)
                observations[(source, backend)] = observed
                validated_inputs[source] = inputs
            except Exception as error:
                lane.update(status="failed", error={"kind": "integrity", "type": type(error).__name__, "message": str(error)})
        complete = all(lane["status"] == "complete" for lane in lanes)
        cases.append({**case, "directory": directory,
                      "status": "failed" if any(lane["status"] == "failed" for lane in lanes) or case_error
                                else "complete" if complete else "pending",
                      "complete_lanes": sum(lane["status"] == "complete" for lane in lanes),
                      "all_six_complete": complete,
                      "regression_pass": all(lane["regression_pass"] is True for lane in lanes) if complete else None,
                      "ordinary_target_pass": all(lane["ordinary_target_pass"] is True for lane in lanes) if complete else None,
                      "error": case_error, "lanes": lanes})
    status = ("failed" if any(case["status"] == "failed" or case["regression_pass"] is False for case in cases)
              else "passed" if all(case["all_six_complete"] and case["regression_pass"] is True for case in cases)
              else "pending")
    return {"schema": SCHEMA, "status": status, "campaign_binding": binding, "performance_eligible": False,
            "scope": "Final steady warmup traversal only; accepted precision does not enumerate internal attempts",
            "fpe_flags": "unobserved", "internal_retries": "unobserved", "cases": cases}
