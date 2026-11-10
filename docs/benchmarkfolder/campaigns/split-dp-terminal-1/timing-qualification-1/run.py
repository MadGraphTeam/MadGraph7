#!/usr/bin/env python3
"""Schedule the unchanged formal campaign only after separate root clearance."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import preflight as check


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value, exclusive=False):
    with Path(path).open("x" if exclusive else "w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def clearance(admission, check_launch_time=True):
    """Validate an external root attestation; this program never creates one."""
    path = HERE / "quiet-window-clearance.json"
    check.require(path.is_file() and not path.is_symlink(), "Root quiet-window clearance is pending")
    value = check.read(path)
    check.require(value.get("authorized_by") == "/root" and value.get("decision") == "authorized"
                  and value.get("manifest_sha256") == check.MANIFEST_SHA
                  and value.get("source_freeze_sha256") == check.sha(HERE / "source-freeze.json")
                  and value.get("admission_sha256") == check.sha(HERE / "admission.json")
                  and value.get("command") == admission["command"] and value.get("cpu") == check.CPU,
                  "Quiet-window clearance does not authorize these exact inputs/command")
    check.require(value.get("owned_native_and_compilation_jobs") == []
                  and value.get("all_owned_native_and_compilation_jobs_idle") is True
                  and value.get("host_conditions_acceptable") is True
                  and value.get("reserve_window_until_controller_and_owned_workers_exit") is True
                  and isinstance(value.get("notes"), str) and value["notes"].strip(),
                  "Root must confirm a reserved quiet window and record its basis")
    issued = datetime.fromisoformat(value["issued_utc"])
    deadline = datetime.fromisoformat(value["launch_not_after_utc"])
    current = datetime.now(timezone.utc)
    check.require(issued.tzinfo is not None and deadline.tzinfo is not None and issued <= deadline,
                  "Quiet-window authorization timestamps are invalid")
    if check_launch_time:
        check.require(issued <= current <= deadline, "Quiet-window launch authorization is not current")
    return {"path": path.name, "sha256": check.sha(path), "attestation": value}


def unstarted():
    check.verify_no_formal_runs()
    for name in ("specification.json", "status.json", "campaign.log"):
        check.require(not (HERE / name).exists(), "Timing attempt already started; never retry or overwrite")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--preflight", action="store_true", help="Verify completed numerical inputs; no workload")
    actions.add_argument("--execute", action="store_true", help="Requires root-authored quiet-window clearance")
    args = parser.parse_args()
    check.freeze_check()
    check.require(check.CPU in os.sched_getaffinity(0), "Reserved CPU9 is unavailable")
    if not args.preflight and not args.execute:
        print(json.dumps({"status": "source-only-no-execution", "command": check.command(),
                          "cpu": check.CPU, "quiet_window_clearance": "separate root-written file required",
                          "ordinary_numerical_prerequisite": "all 11 cases / 6 explicit lanes passed",
                          "overall_goal_qualified": False}, indent=2))
        return
    unstarted()
    if args.execute:
        # Fail before costly rehashing when approval has not been supplied.
        check.require((HERE / "quiet-window-clearance.json").is_file(), "Root quiet-window clearance is pending")
    else:
        check.require(not (HERE / "admission.json").exists(), "Admission already frozen; no replacement")
    admitted = check.verify_prerequisites()
    if args.preflight:
        write(HERE / "admission.json", admitted, exclusive=True)
        print(json.dumps({"status": "prerequisites-verified-no-execution",
                          "admission_sha256": check.sha(HERE / "admission.json"),
                          "quiet_window_clearance": "pending", "overall_goal_qualified": False}))
        return
    check.require(admitted == check.read(HERE / "admission.json"), "Inputs changed after admission")
    approval = clearance(admitted)
    # This is an operational root attestation, not an OS-wide exclusivity proof.
    # No lock is acquired here: the unchanged pair controller owns its flock.
    import tirust_benchmark as bench
    from process_cleanup import OwnedChild, SignalRequests
    specification = {"created_utc": now(), "campaign": check.CAMPAIGN,
        "manifest_sha256": check.MANIFEST_SHA, "command": admitted["command"], "cpu": check.CPU,
        "source_freeze_sha256": check.sha(HERE / "source-freeze.json"),
        "admission_sha256": check.sha(HERE / "admission.json"), "quiet_window": approval,
        "machine": bench.machine_identity(check.CPU), "host_before": bench.host_load_snapshot(),
        "supervisor_pid": os.getpid(), "registered_pairs": 7, "calibration_target_seconds": 2.0,
        "registered_duration_floor_seconds": 0.95,
        "statistics": "two-sided 95% Student-t on paired log ratios; upper endpoint <= 1.05",
        "ordinary_numerical_qualification_passed": True, "overall_goal_qualified": False,
        "stress_scope": "Independent unresolved stress/scientific items may remain open; ordinary timing cannot qualify the overall goal",
        "scheduling_only": True, "automatic_retries": False, "outer_numerical_lock": False}
    write(HERE / "specification.json", specification, exclusive=True)
    state = {"status": "running", "started_utc": now(), "supervisor_pid": os.getpid(),
             "overall_goal_qualified": False}
    write(HERE / "status.json", state)
    child = owned = None
    requests = SignalRequests()
    requests.__enter__()
    try:
        requests.check()
        # Recheck the seal and authorization immediately before Popen.
        check.freeze_check()
        check.require(clearance(admitted) == approval, "Root quiet-window clearance changed")
        unlaunched_runs = check.REPORT / "runs"
        check.require(not unlaunched_runs.exists() or not any(unlaunched_runs.iterdir()),
                      "Concurrent formal launch detected")
        with (HERE / "campaign.log").open("xb") as stream:
            child = subprocess.Popen(admitted["command"], cwd=check.ROOT, env=bench.environment(),
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            owned = OwnedChild(child)
            state["child_pid"] = child.pid
            write(HERE / "status.json", state)
            print(f"START formal campaign child_pid={child.pid}", flush=True)
            code = owned.wait(requests)
            requests.check()
            survivors = [row for row in owned.live() if row["pid"] != child.pid]
            check.require(not survivors, "Controller exited with observed owned descendants still live")
            state["observed_owned_processes"] = list(owned.records.values())
            state["cleanup_verified"] = True
            child = owned = None
        state.update(returncode=code, finished_utc=now(), host_after=bench.host_load_snapshot(),
                     console_sha256=check.sha(HERE / "campaign.log"))
        # Numerical cards deliberately vary by formal backend; the admission
        # checks compare retained per-lane cards, not those current selections.
        check.require(check.verify_prerequisites() == admitted, "Bound prerequisites changed during timing")
        check.require(clearance(admitted, check_launch_time=False)["sha256"] == approval["sha256"],
                      "Clearance changed during timing")
        requests.check()
        summary_path = check.REPORT / "summary.json"
        check.require(summary_path.is_file(), "Formal controller did not produce a summary")
        state["summary_sha256"] = check.sha(summary_path)
        state["main_gate"] = check.read(summary_path)["gate"]
        # Preserve controller success/failure exactly; detailed report ingestion
        # handles descriptive control speeds and independent scientific gates.
        state["status"] = "passed" if code == 0 else "failed"
        write(HERE / "status.json", state)
        if code:
            raise RuntimeError(f"Formal campaign exited {code}; failed/incomplete gates are retained, no automatic retry")
        print(json.dumps({"status": "controller-complete", "main_gate": state["main_gate"],
                          "overall_goal_qualified": False}), flush=True)
    except BaseException as error:
        if owned is not None:
            try:
                state["cleanup"] = owned.stop()
                state["cleanup_verified"] = not state["cleanup"]["survivors"]
            except BaseException as cleanup_error:
                state["cleanup_verified"] = False
                state["cleanup_error"] = repr(cleanup_error)
        elif child is not None and child.poll() is None:
            state["cleanup_verified"] = False
            state["cleanup_error"] = "Controller ownership was not established; positively-owned cleanup required before further admission"
        state.update(status="failed", finished_utc=now(), error=repr(error))
        try:
            write(HERE / "status.json", state)
        except BaseException:
            pass
        raise
    finally:
        requests.__exit__(None, None, None)


if __name__ == "__main__":
    main()
