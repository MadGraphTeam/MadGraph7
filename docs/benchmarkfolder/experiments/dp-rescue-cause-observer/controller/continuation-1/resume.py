#!/usr/bin/env python3
"""One bounded continuation of the unchanged cause-2 build; no numerical calls."""
import ast
import hashlib
import json
import os
from pathlib import Path
import runpy

HERE = Path(__file__).resolve().parent
CONTROLLER = HERE.parent
ROOT = CONTROLLER.parents[4]
BUILDER_SHA = "42a662c5549f539374fb8e8f11aeff709f921457dd77337e7168360174493d12"
PLAN_SHA = "20137ff143ca65980df1e53d487294d2677a886c5de7eb0d573eb827994b9b7c"
RUN_SHA = "6a0512b620f0f9723af8811bbbd128f3b1f0990aac64e3aaf1da60ec7e5cab10"
FAILED_SHA = "7749f7d9ddc11f718297b5cd45d3f17261c245fe22e018ea95956e26cccae35e"
WORK = ROOT / ".tirust-build/dp-rescue-cause-observer/cause-2/build"
REPORT = ROOT / ".tirust-build/dp-rescue-cause-observer/cause-2-continuation-1"


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    if Path.cwd().resolve() != ROOT or WORK.resolve() != WORK or REPORT.resolve() != REPORT:
        raise ValueError("Run only in the repository with unaliased private paths")
    for path, expected in ((CONTROLLER / "build.py", BUILDER_SHA),
                           (CONTROLLER / "build-plan.json", PLAN_SHA),
                           (CONTROLLER / "run.py", RUN_SHA),
                           (WORK / "build.json", FAILED_SHA)):
        if sha(path) != expected:
            raise ValueError("Frozen source or prior failure changed: " + str(path))
    base = runpy.run_path(str(CONTROLLER / "build.py"))
    snapshot, native, plan = base["load_inputs"](PLAN_SHA)
    failed = base["read"](WORK / "build.json")
    if (failed["status"] != "failed" or failed["protected_inputs_unchanged"] is not True
            or failed["commands"][-1]["timed_out"] is not True
            or failed["commands"][-1]["process_reaped"] is not True or failed["cases"]):
        raise ValueError("Unexpected prior attempt state")
    if any(os.environ.get(key) for key in ("LD_PRELOAD", "LD_AUDIT", "LD_DEBUG",
                                           "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER")):
        raise ValueError("Unsupported compiler/runtime interception")
    REPORT.mkdir(parents=True, exist_ok=False)

    class Continuation(base["Build"]):
        def command(self, argv, cwd, name, env=None):
            if name == "observed-cargo-build" and (
                    list(map(str, argv)) != failed["commands"][-1]["argv"]
                    or str(cwd) != failed["commands"][-1]["cwd"]):
                raise ValueError("Continuation changed the exact Cargo command")
            # Keep staged sources/target paths unchanged, but put every new log
            # and commands.json in the separate continuation directory.
            original_work = self.work
            self.work = REPORT
            try:
                return super().command(argv, cwd, name, env)
            finally:
                self.work = original_work

    build = Continuation(WORK, 1800, native)
    record = {"schema": "dp-rescue-observer-build-v1", "status": "started",
              "source_index_sha256": base["SOURCE_INDEX"], "build_plan_sha256": PLAN_SHA,
              "builder_sha256": BUILDER_SHA,
              "native_inputs_sha256": sha(CONTROLLER / "native-inputs.json"),
              "continuation_source": base["pin"](Path(__file__)),
              "prior_failed_manifest": base["pin"](WORK / "build.json"),
              "prior_failure_remains_failed": True, "reused_artifacts_checked": 0,
              "cpu": 11, "deadline_seconds": 1800, "cases": {}, "artifacts": {},
              "historical_environment_reproduced": False, "old_dso_byte_reproduction_claim": False,
              "numerical_execution_performed": False, "cause_attribution_qualified": False}
    inputs = base["gather_pins"](snapshot)
    inputs.update(base["native_pins"](native))
    inputs.update(base["gather_pins"](plan["toolchain_link_inputs"]))
    inputs.update(base["gather_pins"](plan["compiler_support_files"]))
    for path in (CONTROLLER / "build.py", CONTROLLER / "run.py", CONTROLLER / "build-plan.json",
                 CONTROLLER / "native-inputs.json", Path(__file__), WORK / "build.json",
                 WORK / "commands.json"):
        inputs[str(path)] = base["pin"](path)
    for event in failed["commands"]:
        inputs[event["log"]] = base["pin"](Path(event["log"]))
    # Source/helper inputs are immutable; Cargo target/build products and its
    # top-level .global-cache access database/lock files may change.
    for name, item in failed["artifacts"].items():
        if (Path(name).is_relative_to(WORK / "native")
                or Path(name).is_relative_to(WORK / "cargo-home/git")
                or Path(name).is_relative_to(WORK / "cargo-home/registry")):
            inputs[name] = item
    build.install_handlers()
    original_error = None
    try:
        os.sched_setaffinity(0, {11})
        record["protected_before"] = {name: base["pin"](Path(name)) for name in inputs}
        for item in inputs.values():
            build.checkpoint()
            base["check_pin"](item)
        # Verify the complete retained artifact inventory before cache reuse.
        # Its historical cache hashes remain in the untouched failed manifest.
        for item in failed["artifacts"].values():
            build.checkpoint()
            base["check_pin"](item)
            record["reused_artifacts_checked"] += 1
        bridge, upstream = WORK / "native/bridge", WORK / "native/tirust"
        cargo_home = WORK / "cargo-home"
        selection = base["locked_cache_paths"](native)
        private_packages = [{**item, "root": str(cargo_home / Path(item["root"]).relative_to(
                             native["environment"]["CARGO_HOME"]))}
                            for item in native["external_packages"]]
        for item in private_packages:
            base["verify_tree"](item)
        env = dict(build.env)
        env.update(CARGO_HOME=str(cargo_home), CARGO_TARGET_DIR=str(WORK / "cargo-target"),
                   CARGO_BUILD_BUILD_DIR=str(WORK / "cargo-build"), CARGO_BUILD_JOBS="1",
                   CARGO_NET_OFFLINE="true", RUSTC=build.tool("rustc"), CC=build.tool("cc"),
                   AR=build.tool("ar"), RUST_TEST_THREADS="1", RAYON_NUM_THREADS="1")
        for name, rel in (("XDG_CACHE_HOME", "xdg-cache"), ("XDG_CONFIG_HOME", "xdg-config"),
                          ("SYMBOLICA_LOCK_DIR", "symbolica-locks"), ("TMPDIR", "tmp"),
                          ("GMP_MPFR_SYS_CACHE", "gmp-cache")):
            if not (WORK / rel).is_dir():
                raise ValueError("Missing original private build directory")
            env[name] = str(WORK / rel)
        # Execute the exact frozen native_build tail, starting at map_path:
        # Cargo invocation, source/lock checks, local MPFR binding, and return.
        # No source/cache copy, patch, dependency preparation, or C helper rebuild.
        module = ast.parse((CONTROLLER / "build.py").read_text())
        fn = next(node for node in module.body if isinstance(node, ast.FunctionDef)
                  and node.name == "native_build")
        starts = [i for i, node in enumerate(fn.body) if isinstance(node, ast.Assign)
                  and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                  and node.targets[0].id == "map_path"]
        if len(starts) != 1:
            raise ValueError("Frozen native tail anchor differs")
        fn.name = "resume_native"
        names = "build snapshot native work bridge upstream env selection private_packages helper".split()
        fn.args = ast.arguments(posonlyargs=[], args=[ast.arg(arg=x) for x in names],
                                vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])
        fn.body = fn.body[starts[0]:]
        scope = dict(base)
        tail = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
        record["frozen_native_tail_ast_sha256"] = hashlib.sha256(ast.dump(tail).encode()).hexdigest()
        exec(compile(tail, str(CONTROLLER / "build.py"), "exec"), scope)
        observed, native_result = scope["resume_native"](
            build, snapshot, native, WORK, bridge, upstream, env, selection,
            private_packages, bridge / "observer_flags.o")
        record["native"] = native_result
        record["observed_dso"] = base["pin"](observed)
        record["original_dso"] = snapshot["installed_native"]["lib/libmadloop_tirust.so"]
        if record["original_dso"]["sha256"] != base["DSO_SHA"]:
            raise ValueError("Original installed DSO changed")
        record["cases"] = base["host_build"](build, snapshot, observed, plan)
        build.checkpoint()
        record["status"] = "complete"
    except BaseException as exc:
        original_error = exc
        record["status"], record["error"] = "failed", repr(exc)
        raise
    finally:
        try:
            record["protected_after"] = {name: base["pin"](Path(name)) for name in inputs}
            record["protected_inputs_unchanged"] = record.get("protected_before") == record["protected_after"]
            if not record["protected_inputs_unchanged"]:
                record["status"] = "failed"
            record["commands"] = build.events
            record["artifacts"] = {str(path): base["pin"](path)
                                   for folder in (WORK, REPORT) for path in sorted(folder.rglob("*"))
                                   if path.is_file() and path != REPORT / "build.json"}
            base["dump"](REPORT / "build.json", record)
        except BaseException as metadata_error:
            record["status"] = "failed"
            record["postprocessing_error"] = repr(metadata_error)
            try:
                base["dump"](REPORT / "build.json", record)
            except BaseException as second_error:
                metadata_error.add_note("Continuation failure record also failed: " + repr(second_error))
            if original_error is None:
                raise
            original_error.add_note("Continuation finalization failed: " + repr(metadata_error))
        finally:
            build.restore_handlers()
    if record["status"] != "complete":
        raise RuntimeError("Continuation did not qualify")
    print(json.dumps({"manifest": str(REPORT / "build.json"), "sha256": sha(REPORT / "build.json")}))


if __name__ == "__main__":
    main()
