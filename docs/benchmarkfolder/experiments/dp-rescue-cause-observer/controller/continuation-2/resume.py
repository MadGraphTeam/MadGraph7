#!/usr/bin/env python3
"""Finish only unchanged Fortran builds after the exact pinned-module guard fix."""
import ast
import hashlib
import json
import os
from pathlib import Path
import runpy

HERE = Path(__file__).resolve().parent
CONTROLLER = HERE.parent
ROOT = CONTROLLER.parents[4]
WORK = ROOT / ".tirust-build/dp-rescue-cause-observer/cause-2/build"
PRIOR = ROOT / ".tirust-build/dp-rescue-cause-observer/cause-2-continuation-1/build.json"
REPORT = ROOT / ".tirust-build/dp-rescue-cause-observer/cause-2-continuation-2"
PRIOR_SHA = "0ef8947065e4505834e9095cec86a584fdb4e42014165cceaab709f231b3efe2"
BUILDER_SHA = "42a662c5549f539374fb8e8f11aeff709f921457dd77337e7168360174493d12"
PLAN_SHA = "20137ff143ca65980df1e53d487294d2677a886c5de7eb0d573eb827994b9b7c"
RUN_SHA = "6a0512b620f0f9723af8811bbbd128f3b1f0990aac64e3aaf1da60ec7e5cab10"
HOST_SHA = "79224d3d00041dcf6e8faf19b156ce8f8c7fba647e39f209af4b8c829a153776"


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    if Path.cwd().resolve() != ROOT or WORK.resolve() != WORK or REPORT.resolve() != REPORT:
        raise ValueError("Unrecognized working directory")
    for path, expected in ((CONTROLLER / "build.py", BUILDER_SHA),
                           (CONTROLLER / "build-plan.json", PLAN_SHA),
                           (CONTROLLER / "run.py", RUN_SHA),
                           (HERE / "host-build-resumed.py", HOST_SHA), (PRIOR, PRIOR_SHA)):
        if sha(path) != expected:
            raise ValueError("Frozen controller or previous failure changed")
    base = runpy.run_path(str(CONTROLLER / "build.py"))
    snapshot, native, plan = base["load_inputs"](PLAN_SHA)
    prior = base["read"](PRIOR)
    if (prior["status"] != "failed" or prior["protected_inputs_unchanged"] is not True
            or prior["error"] != "ValueError('Private library copy is not a frozen archive')"
            or prior["native"]["same_local_mpfr_instance"] is not True
            or any(x["returncode"] != 0 or x["process_reaped"] is not True for x in prior["commands"])):
        raise ValueError("Unexpected checkpoint state")
    if any(os.environ.get(key) for key in ("LD_PRELOAD", "LD_AUDIT", "LD_DEBUG",
                                           "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER")):
        raise ValueError("Unsupported compiler/runtime interception")
    REPORT.mkdir(parents=True, exist_ok=False)

    class HostContinuation(base["Build"]):
        def command(self, argv, cwd, name, env=None):
            if "cargo" in Path(str(argv[0])).name:
                raise ValueError("Rust rebuild is outside this continuation")
            original_work = self.work
            self.work = REPORT
            try:
                return super().command(argv, cwd, name, env)
            finally:
                self.work = original_work

    build = HostContinuation(WORK, 180, native)
    record = {"schema": "dp-rescue-observer-build-v1", "status": "started",
              "source_index_sha256": base["SOURCE_INDEX"], "build_plan_sha256": PLAN_SHA,
              "builder_sha256": BUILDER_SHA,
              "native_inputs_sha256": sha(CONTROLLER / "native-inputs.json"),
              "continuation_source": base["pin"](Path(__file__)),
              "host_guard_source": base["pin"](HERE / "host-build-resumed.py"),
              "prior_failed_manifest": base["pin"](PRIOR), "prior_failure_remains_failed": True,
              "cpu": 11, "deadline_seconds": 180, "cases": {}, "artifacts": {},
              "native": prior["native"], "original_dso": prior["original_dso"],
              "observed_dso": prior["observed_dso"], "reused_artifacts_checked": 0,
              "historical_environment_reproduced": False, "old_dso_byte_reproduction_claim": False,
              "numerical_execution_performed": False, "cause_attribution_qualified": False}
    inputs = dict(prior["protected_after"])
    for path in (PRIOR, PRIOR.parent / "commands.json", Path(__file__), HERE / "host-build-resumed.py"):
        inputs[str(path)] = base["pin"](path)
    inputs[prior["observed_dso"]["path"]] = prior["observed_dso"]
    for event in prior["commands"]:
        inputs[event["log"]] = base["pin"](Path(event["log"]))
    completed_case = WORK / "cases/qcd_n3_uux_zgg/original"
    for name, value in prior["artifacts"].items():
        if Path(name).is_relative_to(completed_case):
            inputs[name] = value
    build.install_handlers()
    original_error = None
    try:
        os.sched_setaffinity(0, {11})
        record["protected_before"] = {name: base["pin"](Path(name)) for name in inputs}
        for item in inputs.values():
            build.checkpoint()
            base["check_pin"](item)
        for item in prior["artifacts"].values():
            build.checkpoint()
            base["check_pin"](item)
            record["reused_artifacts_checked"] += 1
        source = (HERE / "host-build-resumed.py").read_text()
        ast.parse(source)
        scope = dict(base)
        exec(compile(source, str(HERE / "host-build-resumed.py"), "exec"), scope)
        record["cases"] = scope["resume_host_build"](
            build, snapshot, Path(prior["observed_dso"]["path"]), plan)
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
            record["status"], record["postprocessing_error"] = "failed", repr(metadata_error)
            try:
                base["dump"](REPORT / "build.json", record)
            except BaseException as second_error:
                metadata_error.add_note("Failure record also failed: " + repr(second_error))
            if original_error is None:
                raise
            original_error.add_note("Finalization failed: " + repr(metadata_error))
        finally:
            build.restore_handlers()
    if record["status"] != "complete":
        raise RuntimeError("Host continuation did not qualify")
    print(json.dumps({"manifest": str(REPORT / "build.json"), "sha256": sha(REPORT / "build.json")}))


if __name__ == "__main__":
    main()
