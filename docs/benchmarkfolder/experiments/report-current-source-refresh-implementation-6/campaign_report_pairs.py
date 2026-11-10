"""Read-only replay of retained paired protocol, without campaign activation."""
import ast
from contextlib import contextmanager
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics
import sys
import types


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def validate_identity(value, manifest, binding, baseline_record=None):
    """Check recorded identities only; never inspect today's private build."""
    generation = value["generation"]
    compiled = value["compilation"]
    if any(record.get("campaign_binding") != binding for record in (value, generation, compiled)):
        raise ValueError("Retained input has a different campaign binding")
    actual = generation["source_identity"]
    if generation["source"] == "candidate":
        registered = manifest["candidate"]["exporter_identity"]
        if (actual["files_sha256"] != registered["files_sha256"]
                or actual.get("files") != registered.get("files")):
            raise ValueError("Different candidate numerical/exporter source")
        runtime = value["runtime"]
        for name in ("libmadloop_tirust.so", "libtirust.a", "tirust-config.json"):
            expected = manifest["candidate"]["native"]["files"]["lib/" + name]
            matches = [checksum for group in ("libraries", "archives", "metadata")
                       for path, checksum in runtime.get(group, {}).items() if Path(path).name == name]
            if not matches or any(checksum != expected for checksum in matches):
                raise ValueError("Different registered native artifact: " + name)
    elif generation["source"] != "baseline" or actual["revision"] != manifest["baseline"]["revision"]:
        raise ValueError("Different baseline source")
    elif baseline_record is not None:
        files_hash = sha(json.dumps(baseline_record["files"], sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode())
        if (actual.get("files_sha256") != files_hash
                or actual.get("archive_sha256") != baseline_record["archive_sha256"]
                or actual["revision"] != baseline_record["revision"]):
            raise ValueError("Different frozen baseline source bytes")


def snapshot_paths(evidence):
    """Keep saved snapshots reproducible after new append-only runs appear."""
    class ReadOnlyPath(type(Path())):
        def _name(self):
            try:
                return self.relative_to(evidence.root).as_posix()
            except ValueError:
                raise ValueError("Retained parser attempted access outside report evidence")

        def exists(self):
            name = self._name()
            return evidence.exists(name)

        def is_file(self, **kwargs):
            name = self._name()
            return evidence.exists(name) and (evidence.frozen is None or name in evidence.frozen) and super().is_file()

        def is_dir(self, **kwargs):
            return evidence.exists(self._name()) and super().is_dir()

        def glob(self, pattern, **kwargs):
            self._name()
            for item in super().glob(pattern, **kwargs):
                name = item._name()
                if evidence.exists(name):
                    yield item

        def read_bytes(self):
            name = self._name()
            return evidence.add(name)

        def read_text(self, encoding=None, errors=None, **kwargs):
            return self.read_bytes().decode(encoding or "utf-8", errors or "strict")

        def open(self, mode="r", buffering=-1, encoding=None, errors=None, newline=None):
            if mode not in ("r", "rt", "rb"):
                raise ValueError("Retained report parser cannot write evidence")
            raw = self.read_bytes()
            return io.BytesIO(raw) if mode == "rb" else io.StringIO(
                raw.decode(encoding or "utf-8", errors or "strict"), newline=newline)
    return ReadOnlyPath


@contextmanager
def retained_protocol(evidence, directory, manifest, summary=None):
    """Load pure saved functions; replace all live guards/output with checks.

    Nothing invokes a compiler, executable, campaign activation, or the live
    model directory. The inherited protocol retains its parsing and statistics.
    """
    spec = evidence.json(directory + "/specification.json")
    protocols = spec.get("protocol_sources", spec.get("protocols", {}))
    prefix = directory + ("/protocol-sources/" if "protocol_sources" in spec else "/")
    sources = {name: evidence.add(prefix + name, checksum).decode()
               for name, checksum in protocols.items()}
    required = {"tirust_benchmark.py", "tirust_benchmark_worker.py", "tirust_benchmark_compiler.py", "tirust_benchmark_campaign.py"}
    if not required.issubset(sources):
        raise ValueError("Incomplete retained protocol closure")
    saved_modules = {name: sys.modules.get(name) for name in (
        "tirust_benchmark", "tirust_benchmark_worker", "tirust_benchmark_compiler", "tirust_benchmark_campaign")}
    saved_path = list(sys.path)
    binding = {"id": manifest["id"], "manifest_sha256": sha(evidence.add("campaigns/" + manifest["id"] + "/manifest.json"))}
    baseline_record = evidence.json("campaigns/" + manifest["id"] + "/inputs/baseline.json",
                                    manifest["baseline"]["record_sha256"])
    try:
        for name in ("tirust_benchmark_worker", "tirust_benchmark_compiler", "tirust_benchmark_campaign"):
            module = types.ModuleType(name)
            module.__file__ = str(evidence.path(prefix + name + ".py"))
            # Top-level execution is limited to definitions/imports/constants.
            tree = ast.parse(sources[name + ".py"])
            tree.body = [node for node in tree.body if isinstance(node,
                (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign))]
            exec(compile(tree, module.__file__, "exec"), module.__dict__)
            sys.modules[name] = module
        bench = types.ModuleType("tirust_benchmark")
        frozen_path = snapshot_paths(evidence)
        sys.modules["tirust_benchmark_worker"].Path = frozen_path
        namespace = bench.__dict__
        namespace.update(ast=ast, hashlib=hashlib, json=json, math=math, Path=frozen_path, re=re,
                         statistics=statistics, sys=sys, ROOT=evidence.root.parents[1],
                         REPORT=frozen_path(evidence.path("campaigns/" + manifest["id"])),
                         CAMPAIGN_ID=manifest["id"], CAMPAIGN_MANIFEST=manifest,
                         CAMPAIGN_MANIFEST_SHA256=binding["manifest_sha256"],
                         WORKER_TIMEOUT_BINDING_VERSION=1, BASELINE_REVISION=manifest["baseline"]["revision"])
        names = {"content_digest", "campaign_binding", "campaign_fields", "case_worker_timeout",
                 "validate_worker_timeout_binding", "parse_chain", "compare_rows", "render_worker",
                 "verify_worker_template", "validate_worker_modes", "same_compiler_policy",
                 "validate_retained_result", "aggregate_usage", "selected_run_directory", "summarize"}
        tree = ast.parse(sources["tirust_benchmark.py"])
        definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        if {node.name for node in definitions} != names:
            raise ValueError("Retained paired protocol is missing required functions")
        exec(compile(ast.Module(body=definitions, type_ignores=[]), prefix + "tirust_benchmark.py", "exec"), namespace)
        def bound_digest(path):
            relative = Path(path).resolve().relative_to(evidence.root).as_posix()
            return sha(evidence.add(relative))
        def guard(stage, inputs=None):
            if inputs is not None:
                validate_identity(inputs, manifest, binding, baseline_record)
        namespace.update(digest=bound_digest, campaign_guard=guard, utc_now=lambda: "retained-replay",
                         subprocess=types.SimpleNamespace(check_output=lambda *a, **k: "0"),
                         dump=lambda *a, **k: None)
        sys.modules["tirust_benchmark"] = bench
        if summary is not None:
            campaign = sys.modules["tirust_benchmark_campaign"]
            campaign.campaign_summary_fields = lambda _bench: {
                key: summary[key] for key in summary if key.startswith("campaign_")}
        yield bench, sources
    finally:
        sys.path[:] = saved_path
        for name, module in saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def collect_pairs(evidence, campaign_dir, manifest, config):
    summary_name = campaign_dir + "/summary.json"
    present = evidence.exists(summary_name)
    summary = evidence.json(summary_name) if present else None
    paths = evidence.tree(campaign_dir + "/runs") if evidence.exists(campaign_dir + "/runs") else []
    if not present:
        return {"status": "pending", "main_status": "pending", "regression_status": "pending",
                "cases": [], "observed_candidate_revisions": [], "summary_path": None,
                "scope": "Paired acceptance requires a separately summarized, completely retained timing campaign"}
    if summary.get("campaign_binding") != {"id": manifest["id"], "manifest_sha256": sha(evidence.add(campaign_dir + "/manifest.json"))}:
        raise ValueError("Paired summary belongs to another campaign")
    # A protocol is required to certify any measured campaign row. With no
    # retained pair, allow only the genuinely empty pending summary.
    specs = sorted(name for name in paths if name.endswith("/specification.json"))
    if not specs:
        if (summary["gate"].get("passed") or any(row.get("retained_attempts")
                or row["performance"].get("pairs", 0) for row in summary["cases"])):
            raise ValueError("Measured paired summary lacks its raw protocol")
        return {"status": "pending", "main_status": "pending", "regression_status": "pending",
                "cases": summary["cases"], "observed_candidate_revisions": [],
                "summary_path": summary_name, "scope": "No paired observations retained"}
    if evidence.exists(campaign_dir + "/qualification.json"):
        evidence.add(campaign_dir + "/qualification.json")
    # One saved parser may only replay other runs with the same complete parser
    # closure. A differing retained protocol requires a separately audited report
    # adapter, never silently applying the first run's interpretation.
    validate_protocol_uniformity(evidence, specs)
    protocol_directory = str(Path(specs[0]).parent)
    with retained_protocol(evidence, protocol_directory, manifest, summary) as (bench, _):
        recomputed = bench.summarize(config)
    for key in ("cases", "gate", "candidate_revision", "candidate_revisions", "candidate_source_hashes", "reduction_chains"):
        if recomputed[key] != summary[key]:
            raise ValueError("Paired summary differs from raw replay: " + key)
    states = acceptance_states(recomputed["cases"], recomputed["gate"])
    details = timing_details(evidence, campaign_dir, recomputed["cases"], config)
    return {**states,
            "cases": recomputed["cases"], "observed_candidate_revisions": recomputed["candidate_revisions"],
            "summary_path": summary_name, **details,
            "scope": "Actual full-amplitude paired times include all automatic rescue work. Only main processes have runtime acceptance gates; control speeds are descriptive, with the original generic raw statuses preserved."}


def acceptance_states(cases, gate):
    main = [case for case in cases if case["role"] == "main"]
    def state(rows):
        if any(row.get("passed") is False or row.get("status") in ("failed", "timing-ineligible") for row in rows):
            return "failed"
        if rows and all(row.get("complete") and row.get("passed") for row in rows):
            return "passed"
        return "pending"
    main_status = state([case["performance"] for case in main])
    regressions = state([row for case in main for row in case["existing_backend_regressions"]])
    authoritative = [attempt for case in main for attempt in case.get("retained_attempts", [])
                     if attempt.get("acceptance_eligible") is True and attempt.get("passed") is False]
    if any(attempt.get("label") == "tirust-first" for attempt in authoritative):
        main_status = "failed"
    if any(attempt.get("label") in {"backend-1", "backend-6", "backend-7", "backend-8"}
           for attempt in authoritative):
        regressions = "failed"
    return {"status": "failed" if "failed" in (main_status, regressions) or gate["failed_cases"] else
            "passed" if gate["passed"] else "pending", "main_status": main_status, "regression_status": regressions}


def validate_protocol_uniformity(evidence, specs):
    expected = None
    for name in specs:
        spec = evidence.json(name)
        sources = spec.get("protocol_sources")
        if not isinstance(sources, dict) or not sources:
            raise ValueError("Paired attempt has no retained parser inventory")
        if expected is not None and sources != expected:
            raise ValueError("Paired attempts use different retained parser sources")
        for source, checksum in sources.items():
            evidence.add(str(Path(name).parent / "protocol-sources" / source), checksum)
        expected = sources


def timing_details(evidence, campaign_dir, cases, config):
    """Describe validated durations and controls without rewriting raw verdicts."""
    durations, controls = [], []
    for case in cases:
        for attempt in case.get("retained_attempts", []):
            if attempt.get("status") == "invalid-evidence":
                continue
            path = campaign_dir + "/runs/" + case["id"] + "/" + attempt["directory"] + "/summary.json"
            result = evidence.json(path)
            durations.append({"case": case["id"], "role": case["role"],
                "label": attempt["label"], "directory": attempt["directory"],
                "selected": attempt["selected"],
                "acceptance_eligible": result["acceptance_eligible"],
                "minimum_batch_duration_met": result["minimum_batch_duration_met"],
                "minimum_seconds": {source: min(item["seconds"] for item in observations)
                    for source, observations in result["raw_observations"].items()},
                "raw_speed_passed": result["passed"],
                "speed_scope": "acceptance" if case["role"] == "main" else "descriptive"})
        if case["role"] != "main":
            checks = case["correctness"]
            numerical = ("failed" if any(row.get("passed") is False or row.get("status") == "failed" for row in checks)
                         else "passed" if checks and all(row.get("passed") is True for row in checks) else "pending")
            controls.append({"case": case["id"], "role": case["role"],
                "raw_summary_status": case["status"], "numerical_status": numerical,
                "speed_scope": "descriptive", "runtime_acceptance_applicable": False,
                "raw_authoritative_qualified_failures": case.get("authoritative_qualified_failures", [])})
    return {"batch_durations": durations, "controls": controls,
            "minimum_batch_seconds": config["minimum_batch_seconds"],
            "eligibility_floor_seconds": .95 * config["minimum_batch_seconds"],
            "calibration_target_seconds": 2. * config["minimum_batch_seconds"]}
