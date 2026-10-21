"""Explicit-pair dispatch and failure retention, without physics subprocesses."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import subprocess
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
import ninjars_benchmark_pair_v3 as pair


class ExplicitPairCollectorTest(unittest.TestCase):
    def setUp(self):
        (ROOT / ".benchmark-work").mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / ".benchmark-work", prefix="pair-v3-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.report, self.work = self.root / "report", self.root / "work"
        (self.report / "points").mkdir(parents=True)
        (self.report / "points/case.dat").write_text("2\n")
        self.case = {"id": "case", "nfinal": 2}
        self.config = {"points_per_case": 2, "relative_tolerance": 1e-7, "scale_tolerance": 1e-10}
        self.protocol = {"pair_targets": [list(t) for t in pair.TARGETS], "protocol_identity_sha256": "payload"}
        pair.bench.dump(self.report / "protocol.json", self.protocol)
        self.args = SimpleNamespace(cpu=207, repetitions=7, minimum_seconds=1., timeout=3.)
        self.instances, self.commands, self.prepared = [], [], []
        self.fail_parse = None
        self.fail_run = None
        self.fail_init = None
        self.fail_close = None
        self.fail_prepare = None
        owner = self

        class FakeWorker:
            def __init__(self, command, directory, log, **kwargs):
                self.directory, self.log = Path(directory), log
                self.key = owner.directory_keys[self.directory]
                self.points, self.ndim, self.nso, self.nborn = 2, 1, 1, 1
                self.sequence, self.closed = 0, False
                if owner.fail_init == self.key:
                    raise RuntimeError("initialize failure")
                owner.instances.append(self)

            def run(self, loops):
                self.sequence += 1
                owner.commands.append((self.key, self.sequence, loops))
                for name, value in {"timing.dat": ".1 1\n", "values.dat": "RAW VALUES\n",
                                    "timed_status.dat": "218 2\n", "dimensions.dat": "1 1 1\n"}.items():
                    (self.directory / name).write_text(value)
                if owner.fail_run == (self.key, self.sequence):
                    raise RuntimeError("command failure")
                return .5

            def close(self):
                self.closed = True
                if owner.fail_close == self.key:
                    raise RuntimeError("cleanup failure")

        self.directory_keys = {}

        def prepare(source, case, backend, config, work, report, clean):
            key = source, backend
            self.prepared.append(key)
            if self.fail_prepare == key:
                raise RuntimeError("prepare failure")
            job = self.work / f"{source}-{backend}"
            directory = job / "outputs" / source / case["id"] / "SubProcesses/P0_case"
            directory.mkdir(parents=True)
            card = job / "outputs" / source / case["id"] / "Cards/param_card.dat"
            card.parent.mkdir(); card.write_text("same physics")
            self.directory_keys[directory] = key
            return job, directory, {"backend": backend, "source": source,
                "executable_sha256": str(key), "parameter_card_sha256": "same-card",
                "shared_numerical_settings": {"selected_backend_qp_integrand": True}}

        def parse(directory, backend, points, loops):
            worker = next(w for w in self.instances if w.directory == directory)
            if self.fail_parse == (worker.key, worker.sequence):
                raise ValueError("malformed physical rows")
            code = 216 if backend == 6 else 218
            return {"answer_dimensions": {"allocated": 1, "loop_orders": 1, "born_orders": 1},
                "seconds": 1., "seconds_per_point": .5, "cold_seconds": .1,
                "rows": [{"point": p, "order": o, "status": code,
                          "coefficients": [1., 2., 3., 4.], "accuracy": 1e-12}
                         for p in [1, 2] for o in [0, 1]],
                "status_counts": {str(code): 2}, "timed_status_counts": {str(code): 2 * loops}}

        for target, value in [("prepare", prepare), ("Worker", FakeWorker), ("parse_outputs", parse),
                              ("validate_param_card", mock.Mock()), ("environment", mock.Mock(return_value={}))]:
            patcher = mock.patch.object(pair.workers, target, value)
            patcher.start(); self.addCleanup(patcher.stop)
        for target, value in [("configure", mock.Mock())]:
            patcher = mock.patch.object(pair.bench, target, value)
            patcher.start(); self.addCleanup(patcher.stop)
        patcher = mock.patch.object(pair.supplement, "comparable_numerical_settings", lambda r: r["shared_numerical_settings"])
        patcher.start(); self.addCleanup(patcher.stop)
        self.dispatch = pair.Dispatch(self.report)

    def run_pair(self):
        pair.run_pair_case(self.case, self.config, self.args, self.work, self.report, "run", self.dispatch)

    def events(self):
        return [json.loads(line) for line in self.dispatch.path.read_text().splitlines()]

    def test_exact_targets_and_alternating_matched_repetitions(self):
        self.run_pair()
        self.assertEqual(self.prepared, list(pair.TARGETS))
        self.assertEqual(len(self.instances), 2)
        starts = [e for e in self.events() if e["phase"] == "measurement" and e["event"] == "start"]
        self.assertEqual([(e["source"], e["backend"]) for e in starts],
            [target for rep in range(7) for target in (pair.TARGETS if rep % 2 == 0 else reversed(pair.TARGETS))])
        self.assertEqual([e["repetition"] for e in starts], [rep for rep in range(7) for _ in range(2)])
        self.assertEqual(len(self.commands), 16)
        self.assertTrue(all(w.closed for w in self.instances))
        self.assertEqual([e["index"] for e in self.events()], list(range(len(self.events()))))
        for end in self.events():
            if end["phase"] in ["calibration", "measurement"] and end["event"] == "complete":
                raw = pair.supplement.checked(end["record"])
                data = json.loads(raw.read_text())
                start = self.events()[data["dispatch_index"]]
                self.assertEqual((start["source"], start["backend"], start["repetition"]),
                                 (end["source"], end["backend"], end["repetition"]))
                self.assertTrue(data["raw_capture"])

    def test_cartesian_candidate6_injection_rejected_before_prepare(self):
        self.protocol["pair_targets"].append(["candidate", 6])
        pair.bench.dump(self.report / "protocol.json", self.protocol)
        with self.assertRaisesRegex(ValueError, "Injected"):
            self.run_pair()
        self.assertFalse(self.prepared)

    def test_prepare_failure_never_initializes_workers(self):
        self.fail_prepare = pair.TARGETS[1]
        with self.assertRaisesRegex(RuntimeError, "prepare"):
            self.run_pair()
        self.assertFalse(self.instances)
        failure = json.loads((self.report / "rejected/case-failure.json").read_text())
        self.assertEqual(failure["stage"], "prepare")

    def test_second_initialize_failure_closes_first_worker(self):
        self.fail_init = pair.TARGETS[1]
        with self.assertRaisesRegex(RuntimeError, "initialize"):
            self.run_pair()
        self.assertEqual(len(self.instances), 1)
        self.assertTrue(self.instances[0].closed)
        self.assertEqual(json.loads((self.report / "rejected/case-failure.json").read_text())["raw_capture"], {})

    def test_calibration_parse_failure_retains_raw_before_acceptance(self):
        self.fail_parse = (pair.TARGETS[0], 1)
        with self.assertRaisesRegex(ValueError, "malformed"):
            self.run_pair()
        self.assertTrue(all(w.closed for w in self.instances))
        failure = json.loads((self.report / "rejected/case-failure.json").read_text())
        self.assertEqual(failure["stage"], "calibration")
        self.assertEqual(pair.supplement.checked(failure["raw_capture"]["values.dat"]).read_text(), "RAW VALUES\n")
        self.assertFalse((self.report / "data/case-baseline-6-calibration.json").exists())

    def test_command_failure_keeps_completed_pair_and_failed_raw(self):
        self.fail_run = (pair.TARGETS[1], 2)
        with self.assertRaisesRegex(RuntimeError, "command"):
            self.run_pair()
        self.assertTrue((self.report / "data/case-baseline-6-0.json").exists())
        self.assertFalse((self.report / "data/case-candidate-8-0.json").exists())
        failure = json.loads((self.report / "rejected/case-failure.json").read_text())
        self.assertEqual(failure["raw_outputs"]["values.dat"], "RAW VALUES\n")
        self.assertTrue(all(w.closed for w in self.instances))

    def test_validation_failure_preserves_unaccepted_raw_and_closes_both(self):
        with mock.patch.object(pair.bench, "validate_repetition", side_effect=RuntimeError("drift")):
            with self.assertRaisesRegex(RuntimeError, "drift"):
                self.run_pair()
        failure = json.loads((self.report / "rejected/case-failure.json").read_text())
        self.assertTrue(failure["raw_capture"])
        self.assertFalse((self.report / "data/case-baseline-6-0.json").exists())
        self.assertTrue(all(w.closed for w in self.instances))

    def test_cleanup_failure_is_failed_and_other_worker_still_closed(self):
        self.fail_close = pair.TARGETS[1]
        with self.assertRaisesRegex(RuntimeError, "cleanup"):
            self.run_pair()
        self.assertTrue(all(w.closed for w in self.instances))
        self.assertTrue((self.report / "rejected/case-cleanup-failure.json").exists())

    def test_cleanup_ledger_failure_cannot_leave_workers_open(self):
        original = self.dispatch.event
        def event(case, phase, state, key, *args, **kwargs):
            if phase == "cleanup" and state == "start":
                raise OSError("ledger unavailable")
            return original(case, phase, state, key, *args, **kwargs)
        with mock.patch.object(self.dispatch, "event", event):
            with self.assertRaisesRegex(RuntimeError, "ledger unavailable"):
                self.run_pair()
        self.assertTrue(all(w.closed for w in self.instances))

    def test_owned_prepare_timeout_reaps_group_and_restores_hooks(self):
        process = mock.Mock(pid=os.getpid())
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("make", 21600),
                                    subprocess.TimeoutExpired("make", 10), 0]
        original_run, original_subprocess = pair.bench.run, pair.workers.subprocess
        with mock.patch.object(pair.subprocess, "Popen", return_value=process), \
                mock.patch.object(pair.os, "killpg", side_effect=self.signal_group) as killpg:
            with self.assertRaises(subprocess.TimeoutExpired):
                with pair.owned_preparation(self.report):
                    pair.bench.run(["make", "check"], self.root, self.report / "logs/build.log")
        self.assertEqual(process.wait.call_count, 3)
        self.assertEqual([call for call in killpg.call_args_list if call.args[1] != 0],
            [mock.call(process.pid, pair.signal.SIGTERM), mock.call(process.pid, pair.signal.SIGKILL)])
        self.assertIs(pair.bench.run, original_run)
        self.assertIs(pair.workers.subprocess, original_subprocess)

    def test_fresh_representative_child_is_explicitly_owned_and_reaped_on_timeout(self):
        process = mock.Mock(pid=os.getpid())
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("check_fresh", 7200), 0]
        command = ["taskset", "-c", "190", "./check_fresh", "points.dat", "1"]
        with mock.patch.object(pair.subprocess, "Popen", return_value=process), \
                mock.patch.object(pair.os, "killpg", side_effect=self.signal_group) as killpg:
            with self.assertRaises(subprocess.TimeoutExpired):
                with pair.owned_preparation(self.report, fresh_cpu=190):
                    pair.workers.subprocess.run(command, cwd=self.root,
                        env={name: "1" for name in pair.THREADS}, stdout=mock.Mock(),
                        stderr=subprocess.STDOUT, timeout=7200., check=True)
        self.assertEqual(process.wait.call_count, 2)
        self.assertEqual([call for call in killpg.call_args_list if call.args[1] != 0],
                         [mock.call(process.pid, pair.signal.SIGTERM)])
        events = [json.loads(line) for line in (self.report / "process-ledger.jsonl").read_text().splitlines()]
        self.assertEqual(events[0]["command"], command)
        self.assertTrue(events[-1]["reaped"])

    def test_pair_default_cannot_launch_fresh_or_change_fresh_cpu_threads(self):
        command = ["taskset", "-c", "190", "./check_fresh", "points.dat", "1"]
        options = dict(cwd=self.root, env={name: "1" for name in pair.THREADS},
                       stdout=mock.Mock(), stderr=subprocess.STDOUT, timeout=7200., check=True)
        with mock.patch.object(pair.subprocess, "Popen") as popen:
            with pair.owned_preparation(self.report):
                with self.assertRaisesRegex(ValueError, "frozen prepare"):
                    pair.workers.subprocess.run(command, **options)
            with pair.owned_preparation(self.report, fresh_cpu=190):
                changed = {**options, "env": {**options["env"], "RAYON_NUM_THREADS": "2"}}
                with self.assertRaisesRegex(ValueError, "fresh-process override"):
                    pair.workers.subprocess.run(command, **changed)
                with self.assertRaisesRegex(ValueError, "frozen prepare"):
                    pair.workers.subprocess.run([*command[:2], "207", *command[3:]], **options)
        popen.assert_not_called()

    @staticmethod
    def signal_group(pid, sig):
        if sig == 0:
            raise ProcessLookupError("original group is gone")

    def test_interruption_before_popen_returns_still_reaps_registered_child(self):
        process = mock.Mock(pid=os.getpid())
        process.wait.return_value = 0
        process.poll.return_value = None
        def popen(*args, **kwargs):
            # Invoke the installed deferred handler; no OS signal is sent.
            pair.signal.getsignal(pair.signal.SIGTERM)(pair.signal.SIGTERM, None)
            return process
        with mock.patch.object(pair.subprocess, "Popen", side_effect=popen), \
                mock.patch.object(pair.os, "killpg", side_effect=self.signal_group) as killpg:
            with self.assertRaisesRegex(SystemExit, "interruption"):
                with pair.owned_preparation(self.report):
                    pair.workers.subprocess.Popen(["fake-worker"], cwd=self.root, start_new_session=True)
        self.assertEqual(process.wait.call_count, 1)
        self.assertEqual(killpg.call_count, 2)

    def test_ledger_failure_after_spawn_cannot_escape_owned_cleanup(self):
        process = mock.Mock(pid=os.getpid())
        process.wait.return_value = 0
        process.poll.return_value = None
        with mock.patch.object(pair.subprocess, "Popen", return_value=process), \
                mock.patch.object(pair, "process_event", side_effect=OSError("ledger failed")), \
                mock.patch.object(pair.os, "killpg", side_effect=self.signal_group) as killpg:
            with self.assertRaisesRegex(OSError, "ledger failed"):
                with pair.owned_preparation(self.report):
                    pair.workers.subprocess.Popen(["fake-worker"], cwd=self.root, start_new_session=True)
        self.assertEqual(process.wait.call_count, 1)
        self.assertEqual(killpg.call_count, 2)

    def test_near_immediate_exit_is_retained_and_reaped_without_proc_identity(self):
        process = mock.Mock(pid=999999999)
        process.wait.return_value = 0
        process.poll.return_value = 0
        with mock.patch.object(pair.subprocess, "Popen", return_value=process), \
                mock.patch.object(pair.os, "killpg", side_effect=self.signal_group):
            with pair.owned_preparation(self.report):
                pair.workers.subprocess.Popen(["fake-worker"], cwd=self.root, start_new_session=True)
        events = [json.loads(line) for line in (self.report / "process-ledger.jsonl").read_text().splitlines()]
        self.assertEqual(events[0]["exited_before_identity"], 0)
        self.assertIsNone(events[0]["start_ticks"])
        self.assertEqual(events[-1]["event"], "reaped")

    def test_unknown_initial_identity_refuses_replacement_and_cleans_other_child(self):
        unknown, known = mock.Mock(pid=999999998), mock.Mock(pid=os.getpid())
        for process in [unknown, known]:
            process.wait.return_value = 0
            process.poll.return_value = 0
        known.poll.return_value = None
        original_read, calls = Path.read_text, [0]
        def read(path, *args, **kwargs):
            if str(path) == f"/proc/{unknown.pid}/stat":
                calls[0] += 1
                if calls[0] == 1:
                    raise FileNotFoundError("exited before initial identity")
                return original_read(Path(f"/proc/{os.getpid()}/stat"), *args, **kwargs)
            return original_read(path, *args, **kwargs)
        with mock.patch.object(pair.subprocess, "Popen", side_effect=[unknown, known]), \
                mock.patch.object(Path, "read_text", read), \
                mock.patch.object(pair.os, "killpg", side_effect=self.signal_group) as killpg:
            with pair.owned_preparation(self.report):
                pair.workers.subprocess.Popen(["missing-original"], cwd=self.root, start_new_session=True)
                pair.workers.subprocess.Popen(["still-owned"], cwd=self.root, start_new_session=True)
        known.wait.assert_called_once()
        unknown.wait.assert_not_called()
        self.assertEqual(killpg.call_count, 2)
        self.assertTrue(all(call.args[0] != unknown.pid for call in killpg.call_args_list))

    def test_no_execution_without_explicit_gate(self):
        with mock.patch.object(pair, "preflight") as preflight:
            with self.assertRaisesRegex(ValueError, "execute-numerical"):
                pair.collect(self.root / "absent-spec", self.root / "absent-handoff")
        preflight.assert_not_called()

    def test_mismatched_handoff_rejected_before_preflight(self):
        spec = self.root / "spec.json"; spec.write_text("{}")
        handoff = self.root / "handoff.json"; handoff.write_text("{}")
        with mock.patch.object(pair, "validate_spec", return_value={}), mock.patch.object(pair, "preflight") as preflight:
            with self.assertRaisesRegex(ValueError, "handoff"):
                pair.collect(spec, handoff, execute_numerical=True)
        preflight.assert_not_called()

    def test_multiple_runtime_threads_rejected_before_preflight(self):
        spec = self.root / "spec.json"; spec.write_text("{}")
        handoff = self.root / "handoff.json"
        handoff.write_text(json.dumps({"root_authorized": True, "phase": pair.PAIR_SCHEMA,
            "cpu": 207, "specification_sha256": pair.workers.digest(spec)}))
        environment = {key: "1" for key in pair.THREADS}; environment["RAYON_NUM_THREADS"] = "2"
        with mock.patch.object(pair, "validate_spec", return_value={}), \
                mock.patch.object(pair.os, "sched_getaffinity", return_value={207}), \
                mock.patch.dict(pair.os.environ, environment), mock.patch.object(pair, "preflight") as preflight:
            with self.assertRaisesRegex(ValueError, "thread counts"):
                pair.collect(spec, handoff, execute_numerical=True)
        preflight.assert_not_called()

    def test_committed_production_delta_is_rejected_even_when_file_hash_matches(self):
        name = "madgraph/__init__.py"
        spec = {"production_sources": {name: pair.workers.digest(ROOT / name)}, "sources": {}}
        def git_run(command, **kwargs):
            if pair.NUMERICAL_REVISION in command:
                raise subprocess.CalledProcessError(1, command)
        with mock.patch.object(pair.subprocess, "check_output", return_value=(name+"\0").encode()), \
                mock.patch.object(pair.subprocess, "run", side_effect=git_run) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                pair.check_production(spec)
        self.assertTrue(any(pair.NUMERICAL_REVISION in call.args[0] for call in run.call_args_list))

    def test_spec_rejects_target_injection_and_insufficient_pairs(self):
        runtime = pair.policy.seal({"schema": "reducer-runtime-signature-v2", "libraries": {"libninjars.a": pair.ARCHIVE_SHA256}})
        spec = {"schema": pair.SCHEMA, "pair_targets": [list(t) for t in pair.TARGETS], "role_map": pair.ROLES,
            "cpu": 207, "repetitions": 7, "minimum_seconds": 1., "timeout": 86400.,
            "literal_baseline_revision": pair.bench.BASELINE_REVISION,
            "candidate_generator_identity_sha256": pair.GENERATOR_ID,
            "candidate_runtime_signature": runtime, "production_sources": {"source": "hash"},
            "sources": {name: {"sha256": pair.workers.digest(ROOT / "tools" / name)}
                        for name in pair.workers.SOURCE_NAMES}}
        spec["sources"]["collector"] = {"sha256": pair.workers.digest(Path(pair.__file__))}
        with mock.patch.object(pair, "RUNTIME_ID", runtime["identity_sha256"]):
            pair.validate_spec(pair.policy.seal(spec))
            for field, value in [("pair_targets", [["baseline", 6], ["candidate", 8], ["candidate", 6]]),
                                 ("repetitions", 6), ("cpu", 190), ("minimum_seconds", .5)]:
                changed = copy.deepcopy(spec); changed[field] = value
                with self.subTest(field=field), self.assertRaises(ValueError):
                    pair.validate_spec(pair.policy.seal(changed))

    def test_explicit_new_release_cannot_inherit_any_historical_identity(self):
        cohort = {"numerical_revision": "1" * 40,
                  "generation_revision": "2" * 40,
                  "generator_identity_sha256": "3" * 64,
                  "runtime_identity_sha256": "4" * 64,
                  "archive_sha256": "5" * 64}
        self.assertEqual(pair.candidate_cohort({"candidate_cohort": cohort}), cohort)
        for name in cohort:
            changed = dict(cohort); del changed[name]
            with self.subTest(missing=name), self.assertRaisesRegex(ValueError, "complete"):
                pair.candidate_cohort({"candidate_cohort": changed})
        for value in (None, {}, {**cohort, "numerical_revision": "HEAD"},
                      {**cohort, "archive_sha256": "5" * 40},
                      {**cohort, "unexpected": "ignored"}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                pair.candidate_cohort({"candidate_cohort": value})

    def test_new_cohort_rejects_old_runtime_even_when_other_spec_fields_match(self):
        archive = "5" * 64
        runtime = pair.policy.seal({"schema": "reducer-runtime-signature-v2",
                                   "libraries": {"libninjars.a": archive}})
        cohort = {"numerical_revision": "1" * 40, "generation_revision": "2" * 40,
                  "generator_identity_sha256": "3" * 64,
                  "runtime_identity_sha256": runtime["identity_sha256"],
                  "archive_sha256": archive}
        spec = {"schema": pair.SCHEMA, "pair_targets": [list(t) for t in pair.TARGETS],
                "role_map": pair.ROLES, "cpu": 207, "repetitions": 7,
                "minimum_seconds": 1., "timeout": 86400.,
                "literal_baseline_revision": pair.bench.BASELINE_REVISION,
                "candidate_cohort": cohort,
                "candidate_generator_identity_sha256": cohort["generator_identity_sha256"],
                "candidate_runtime_signature": runtime, "production_sources": {"source": "hash"},
                "sources": {name: {"sha256": pair.workers.digest(ROOT / "tools" / name)}
                            for name in pair.workers.SOURCE_NAMES}}
        spec["sources"]["collector"] = {"sha256": pair.workers.digest(Path(pair.__file__))}
        pair.validate_spec(pair.policy.seal(spec))
        for field, value in (("runtime_identity_sha256", pair.RUNTIME_ID),
                             ("archive_sha256", pair.ARCHIVE_SHA256),
                             ("generator_identity_sha256", pair.GENERATOR_ID)):
            changed = copy.deepcopy(spec); changed["candidate_cohort"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "cohort"):
                pair.validate_spec(pair.policy.seal(changed))

    def test_new_cohort_production_check_uses_its_committed_numerical_revision(self):
        name = "madgraph/__init__.py"
        cohort = {"numerical_revision": "1" * 40, "generation_revision": "2" * 40,
                  "generator_identity_sha256": "3" * 64,
                  "runtime_identity_sha256": "4" * 64, "archive_sha256": "5" * 64}
        spec = {"candidate_cohort": cohort,
                "production_sources": {name: pair.workers.digest(ROOT / name)}, "sources": {}}
        def git_run(command, **kwargs):
            if cohort["numerical_revision"] in command:
                raise subprocess.CalledProcessError(1, command)
        with mock.patch.object(pair.subprocess, "check_output", return_value=(name+"\0").encode()), \
                mock.patch.object(pair.subprocess, "run", side_effect=git_run) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                pair.check_production(spec)
        self.assertTrue(any(cohort["numerical_revision"] in call.args[0] for call in run.call_args_list))
        self.assertFalse(any(pair.NUMERICAL_REVISION in call.args[0] for call in run.call_args_list))

    def old_origin(self):
        output = self.root / "old-origin"
        directory = output / "SubProcesses/P0_case"
        directory.mkdir(parents=True)
        (output / "Source").mkdir()
        (output / "Source/model.f").write_text("unchanged original source\n")
        (directory / "check").write_text("untouched original executable bytes")
        (directory / "check_sa.f").write_text("original fresh driver\n")
        identity = {"revision": pair.bench.BASELINE_REVISION,
            "status": "recorded-before-generation", "verification": "provenance.json:baseline_source_integrity"}
        generation = {"source_identity": identity}
        compilation = {"seconds": 1., "compiler": "GNU Fortran (GCC)15.3.0", "settings": {},
            "executable_sha256": pair.workers.digest(directory / "check"),
            "driver_sha256": pair.workers.digest(directory / "check_sa.f")}
        pair.bench.dump(output / "generation.json", generation)
        pair.bench.dump(output / "compilation.json", compilation)
        return output, generation, compilation, pair.bench.generated_source_manifest(output)

    def test_authentic_old_literal_compilation_record_is_preserved_and_accepted(self):
        output, generation, compilation, observed = self.old_origin()
        observed.update(recording="observed after original export", observed_utc="historical timestamp",
                        scope="Historical observed generated sources; source payload remains exact")
        before = (output / "compilation.json").read_bytes()
        identity = pair.validate_origin_records("baseline", output, generation, compilation, observed)
        self.assertEqual(identity, generation["source_identity"])
        self.assertEqual(before, (output / "compilation.json").read_bytes())
        self.assertNotIn("source_identity", compilation)
        self.assertNotIn("generated_sources_sha256", compilation)

    def test_candidate_cannot_use_missing_modern_compilation_fields(self):
        output, generation, compilation, observed = self.old_origin()
        with self.assertRaisesRegex(ValueError, "candidate requires modern"):
            pair.validate_origin_records("candidate", output, generation, compilation, observed)

    def test_old_observed_manifest_drift_rejects_before_original_acceptance(self):
        output, generation, compilation, observed = self.old_origin()
        (output / "Source/model.f").write_text("different input\n")
        with self.assertRaisesRegex(ValueError, "source manifest"):
            pair.validate_origin_records("baseline", output, generation, compilation, observed)


if __name__ == "__main__":
    unittest.main()
