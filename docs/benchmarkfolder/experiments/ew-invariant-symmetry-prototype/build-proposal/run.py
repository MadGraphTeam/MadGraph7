#!/usr/bin/env python3
"""One private adapter compile/link proposal. Plan is read-only; no replay API."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import signal
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PROPOSAL = HERE.parent
ROOT = PROPOSAL.parents[3]
CAUSE = ROOT / "docs/benchmarkfolder/experiments/dp-rescue-cause-observer"
CASE, SUB = "ew_n3_epem_ttxa", "SubProcesses/P0_epem_ttxa"
ADAPTER = SUB + "/TIRUST_interface.f"
OBJECT = SUB + "/TIRUST_interface.o"
CANDIDATE_SHA = "a30b838fc2839034a2599d2bf3c04df35ec674847afeadaf111127142dfccbb3"
ORIGINAL_SHA = "e60570ec599263807165f2f289add2076a4c5b71b9e09fe0de83306e9ab2f7ff"
ANCHORS = {
    "build": (ROOT / ".tirust-build/dp-rescue-cause-observer/cause-2-qualified/build/build.json",
              "b3f71a6493ce315dfd45d9a9da2302e16ae80c155f5a147ec65cb8a20f1e71e3"),
    "numerical": (ROOT / ".tirust-build/dp-rescue-cause-observer/cause-2-qualified/controller-native/attempt1/execution.json",
                  "c308d6bdf48db24bc428e0c556618e4191fe07535720708f954b62331376f8b4"),
    "snapshot": (CAUSE / "preparation/source-1/snapshot.json",
                 "2aa5e87ead00c163d0f987dbef257654a8a22ec748f7d3f71be68aeaa64e7759"),
    "environment": (CAUSE / "controller/native-inputs.json",
                    "386cf528491a499a4ebbc9b6a9988e85430644dd1d06654c8846e91d0b868011"),
    "compiler_support": (CAUSE / "controller/build-plan.json",
                         "20137ff143ca65980df1e53d487294d2677a886c5de7eb0d573eb827994b9b7c"),
}
HELPERS = {
    "preparation": (ROOT / "docs/benchmarkfolder/experiments/qcd-n4-cache-occupancy/prepare.py",
                    "86c4172aec2203700ea2481f2180de96f4be4265e552eefc2e12ad9ccc5d75c7"),
    "build": (ROOT / "docs/benchmarkfolder/experiments/qcd-n4-cache-occupancy/build.py",
              "dbf4099342243a0d868b5be451f60123bfcb268fd0f1fe5f0bd012dd7c2acbaf"),
    "lifecycle": (CAUSE / "controller/run.py",
                  "6a0512b620f0f9723af8811bbbd128f3b1f0990aac64e3aaf1da60ec7e5cab10"),
}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def helpers():
    import hashlib
    import importlib.util
    loaded = {}
    for name, (path, expected) in HELPERS.items():
        need(hashlib.sha256(path.read_bytes()).hexdigest() == expected, "Reviewed helper changed: " + name)
        spec = importlib.util.spec_from_file_location("ew_adapter_" + name, path)
        loaded[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loaded[name])
    return loaded["preparation"], loaded["build"], loaded["lifecycle"]


def inputs(preparation, helper):
    need(Path.cwd().resolve() == ROOT, "Use the MadGraph repository cwd")
    anchor_pins = {name: preparation.pin(path, expected) for name, (path, expected) in ANCHORS.items()}
    data = {name: json.loads(Path(pin["path"]).read_text()) for name, pin in anchor_pins.items()}
    previous, numerical = data["build"], data["numerical"]
    need(previous["status"] == "complete" and previous["protected_inputs_unchanged"] is True,
         "Original build is not qualified")
    need(numerical["complete"] and numerical["numerical_parity_passed"] and numerical["runtime_provenance_validated"]
         and numerical["protected_before"] == numerical["protected_after"], "Previous numerical provenance failed")
    arm = previous["cases"][CASE]["arms"]["original"]
    saved = data["snapshot"]["cases"][CASE]
    need(numerical["arms"][CASE + "/original"]["executable"] == arm["executable"]
         and numerical["arms"][CASE + "/original"]["runtime"] == arm["runtime"], "Original executable/runtime join differs")
    need(arm["sources"][ADAPTER]["sha256"] == ORIGINAL_SHA, "Original adapter differs")
    candidate = preparation.pin(PROPOSAL / "TIRUST_interface.f", CANDIDATE_SHA)
    scope = saved["effective_compiler_flags"]["scopes"]["process"]
    compiles = [shlex.split(line) for line in scope["expanded_make_output"].splitlines()
                if " -c TIRUST_interface.f " in line]
    links = [row for row in previous["commands"] if row["cwd"] == str(Path(arm["executable"]["path"]).parent)
             and "-o" in row["argv"] and row["argv"][row["argv"].index("-o") + 1] == "check"]
    need(len(compiles) == len(links) == 1 and links[0]["returncode"] == 0 and links[0]["process_reaped"],
         "Ambiguous or unsuccessful recorded recipe")
    compiler = scope["recipes"]["TIRUST_interface.f"]["compiler"][0]
    need(compiles[0][0] == compiler["path"] == links[0]["argv"][0], "Compiler recipe differs")
    protected = {pin["path"]: pin for pin in anchor_pins.values()}
    for collection in (arm["sources"], arm["objects"], arm["extra_objects"], arm["extra_sources"], arm["link_inputs"],
                       saved["cards"], saved["initial_resource_inventory"], saved["points"], data["snapshot"]["installed_native"]):
        helper.records(collection, protected)
    for path, sha in arm["runtime"].items():
        protected[path] = preparation.pin(Path(path), sha)
    for name, value in data["compiler_support"]["compiler_support_files"].items():
        if str(Path(compiler["path"]).parent.parent) in name:
            helper.records(value, protected)
    for pin in (candidate, compiler, arm["executable"], arm["interpreter"], arm["link_map"]):
        protected[pin["path"]] = pin
    for path, expected in HELPERS.values():
        protected[str(path)] = preparation.pin(path, expected)
    plan = {
        "schema": "ew-invariant-adapter-build-proposal-v1", "case": CASE, "candidate": candidate,
        "original_adapter": arm["sources"][ADAPTER], "anchors": anchor_pins,
        "original_executable": arm["executable"], "original_driver": arm["objects"][SUB + "/check_sa.o"],
        "unchanged_observer_support": arm["extra_objects"], "compiler": compiler,
        "compile_argv": compiles[0], "qualified_link_argv": links[0]["argv"],
        "qualified_link_cwd": links[0]["cwd"], "runtime": arm["runtime"],
        "interpreter": arm["interpreter"], "runtime_environment": arm["runtime_environment"],
        "clone_source_count": len(arm["sources"]), "original_object_inventory_count": len(arm["objects"]),
        "changed_numerical_objects": [OBJECT], "compile_commands": 1, "link_commands": 1,
        "loader_list_commands": 1, "jobs": 1, "seconds_cap": 180, "default_cpu": 12, "allowed_cpus": [11, 12],
        "numerical_execution": False, "rust_build": False, "source_only": True,
        "driver_protocol": "Unchanged qualified raw-word full101 bootstrap+100 READY/STOP/BYE driver; host module plus noop shim reused",
        "environment": "Frozen cause environment, with TMPDIR redirected to the new private build; no Cargo/cache copy or invocation",
        "command_adaptations": ["Compile in materialized private source/include tree with unchanged argv",
            "Link uses absolute existing objects except the new adapter; changes only executable and map output paths",
            "Original archive search paths, libraries, runtime RPATH and support objects remain exact"],
    }
    return plan, arm, protected, data["environment"]["environment"]


def compile_private(args, plan, arm, protected, environment, preparation, helper, lifecycle):
    work = ROOT / ".tirust-build/ew-invariant-symmetry-prototype" / args.label / "build"
    need(not work.exists() and work.resolve() == work, "Private build attempt exists or path aliases")
    need(not any(os.environ.get(k) for k in ("LD_PRELOAD", "LD_AUDIT", "LD_DEBUG")), "Ambient loader interception")
    build = helper.Build(work, lifecycle, preparation)
    old_affinity, handlers, error, created = os.sched_getaffinity(0), {}, None, False
    record = {"schema": "ew-invariant-adapter-build-v1", "status": "started", "cpu": args.cpu, "jobs": 1,
              "seconds_cap": 180, "plan_sha256": helper.digest(HERE / "plan.json"),
              "builder_sha256": helper.digest(__file__), "candidate": plan["candidate"],
              "anchors": plan["anchors"], "protected_before": protected,
              "numerical_execution_performed": False, "numerical_parity_qualified": False,
              "loaded_numerical_runtime_verified": False, "performance_eligible": False}
    try:
        for signum in lifecycle.SIGNALS:
            handlers[signum] = signal.signal(signum, lifecycle.interrupted)
        os.sched_setaffinity(0, {args.cpu})
        work.mkdir(parents=True, exist_ok=False)
        created = True
        helper.write(work / "build.json", record)
        helper.verify(protected, preparation, build.deadline)
        clone = work / "generated"
        clones = {}
        for rel, source in arm["sources"].items():
            if rel != ADAPTER:
                target = preparation.relative(clone, rel)
                build.copy(source["path"], target)
                clones[str(target)] = {"sha256": source["sha256"]}
        for rel, source in arm["objects"].items():
            if rel.endswith(".mod"):
                target = preparation.relative(clone, rel)
                if target.exists():
                    need(helper.digest(target) == source["sha256"], "Conflicting compiler module")
                else:
                    build.copy(source["path"], target)
                clones[str(target)] = {"sha256": source["sha256"]}
        source = clone / ADAPTER
        build.copy(plan["candidate"]["path"], source)
        clones[str(source)] = {"sha256": CANDIDATE_SHA}
        sub = clone / SUB
        env = dict(environment)
        (work / "tmp").mkdir()
        env["TMPDIR"] = str(work / "tmp")
        build.command(plan["compile_argv"], sub, env, "compile-adapter")
        replacement = preparation.pin(clone / OBJECT)
        argv = list(plan["qualified_link_argv"])
        old_sub = Path(plan["qualified_link_cwd"])
        actual_objects = {}
        for i, token in enumerate(argv):
            if token.endswith(".o"):
                path = clone / OBJECT if token == "TIRUST_interface.o" else old_sub / token
                argv[i] = str(path)
                actual_objects[token] = preparation.pin(path)
        need(len(actual_objects) == 23 and "host_observer.o" in actual_objects and "observer_noop.o" in actual_objects,
             "Qualified link object list changed")
        executable, link_map = sub / "check", sub / "private-link.map"
        argv[argv.index("-o") + 1] = str(executable)
        maps = [i for i, token in enumerate(argv) if token.startswith("-Wl,-Map,")]
        need(len(maps) == 1, "Expected one actual link map output")
        argv[maps[0]] = "-Wl,-Map," + str(link_map)
        build.command(argv, sub, env, "link-candidate")
        old_adapter = arm["objects"][OBJECT]["resolved_path"]
        allowed = {path: value["sha256"] for path, value in arm["link_inputs"].items() if path != old_adapter}
        allowed[replacement["resolved_path"]] = replacement["sha256"]
        loads = {}
        for line in link_map.read_text().splitlines():
            if line.startswith("LOAD "):
                path = Path(line[5:].strip())
                path = (path if path.is_absolute() else sub / path).resolve(strict=True)
                need(str(path) in allowed, "Unbound linker LOAD input: " + str(path))
                loads[str(path)] = preparation.pin(path, allowed[str(path)])
        need(set(loads) == set(allowed), "Actual linker input closure differs beyond the adapter object")
        runtime_env = {"PATH": os.defpath, "LANG": "C", "LC_ALL": "C", "TZ": "UTC", **arm["runtime_environment"]}
        expected = {"interpreter": arm["interpreter"], "retained_runtime": {"expected_elf_closure": {
            path: {"resolved_path": path, "sha256": sha} for path, sha in arm["runtime"].items()}}}
        closure, interpreter = helper.runtime(build, executable, expected, runtime_env, "candidate")
        helper.verify(clones, preparation, build.deadline)
        helper.verify(protected, preparation, build.deadline)
        record.update(status="complete", protected_after=protected, sources=clones,
                      original_executable=arm["executable"], executable=preparation.pin(executable),
                      changed_object=replacement, link_objects=actual_objects, link_inputs=loads,
                      link_map=preparation.pin(link_map), runtime=closure, interpreter=interpreter,
                      launch_prefix=[interpreter["path"]], runtime_environment=runtime_env,
                      dependency_listing_verified=True, compile_environment=env,
                      qualification_pending="Compiled IEEE/input checks and separately authorized full101 numerical replay")
    except BaseException as exc:
        error = exc
        record.update(status="failed", error=repr(exc))
        raise
    finally:
        try:
            if error is not None:
                try:
                    helper.verify(protected, preparation)
                    record["protected_after"] = protected
                except BaseException as preservation_error:
                    record["protected_after_error"] = repr(preservation_error)
                    error.add_note("Protected-input verification error: " + repr(preservation_error))
            if created:
                record["commands"] = build.commands
                record["artifacts"] = {str(p.relative_to(work)): preparation.pin(p)
                    for p in sorted(work.rglob("*")) if p.is_file() and p.name != "build.json"}
                helper.write(work / "build.json", record)
        except BaseException as metadata_error:
            if error is None:
                raise
            error.add_note("Build metadata error: " + repr(metadata_error))
        finally:
            os.sched_setaffinity(0, old_affinity)
            for signum, handler in handlers.items():
                signal.signal(signum, handler)
    return preparation.pin(work / "build.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("plan", "build"))
    parser.add_argument("--label", default="attempt1")
    parser.add_argument("--cpu", type=int, choices=(11, 12), default=12)
    parser.add_argument("--self-sha256")
    parser.add_argument("--plan-sha256")
    args = parser.parse_args()
    need(re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,40}", args.label), "Unsafe build label")
    preparation, helper, lifecycle = helpers()
    plan, arm, protected, environment = inputs(preparation, helper)
    need(json.loads((HERE / "plan.json").read_text()) == plan, "Saved plan differs from exact pinned input recipes")
    for path in (HERE / "plan.json", Path(__file__).resolve()):
        protected[str(path)] = preparation.pin(path)
    helper.verify(protected, preparation)
    if args.phase == "plan":
        print(json.dumps({"source_only": True, "native_execution": False, "build_performed": False,
              "plan_sha256": helper.digest(HERE / "plan.json"), "builder_sha256": helper.digest(__file__),
              "protected_files": len(protected), "cpu": args.cpu, "jobs": 1, "seconds_cap": 180,
              "compile_commands": 1, "link_commands": 1, "loader_list_commands": 1}, sort_keys=True))
        return
    need(args.self_sha256 == helper.digest(__file__) and args.plan_sha256 == helper.digest(HERE / "plan.json"),
         "Reviewed builder and plan hashes are required; root instruction is separate")
    print(json.dumps({"build_manifest": compile_private(args, plan, arm, protected, environment,
                     preparation, helper, lifecycle), "numerical_execution": False}, sort_keys=True))


if __name__ == "__main__":
    main()
