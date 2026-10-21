"""Private semantic/operational mocks; no real CLI, compiler or reducer."""
from contextlib import contextmanager
import ast
import copy
import importlib.util
import hashlib
import io
import json
from pathlib import Path
import pickle
import shutil
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
import ninjars_profile_cli_v3 as cli


def profile(label="Ninja", total="1.23", coefs="0.3", reduction="0.93"):
    block = " time / PSpoint ========== " + total + "ms\n|= Coefs. computation time... " + coefs + "ms (24%)\n|= Loop evaluation time...... " + reduction + "ms (76%)\n"
    return ("= Reduction tool tested...... " + label + "\n= Helicity sum" + block + "= One helicity" + block +
        "|= Number of loop groups..... 42\n")


class ProfileCLI(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=cli.ROOT, prefix=".test-cli-profile-")
        self.addCleanup(self.temp.cleanup); self.root = Path(self.temp.name)

    def put(self, name, text="bound input"):
        path = self.root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text)
        return cli.bound(path)

    def case(self):
        return {"model": "loop_sm", "process": "g g > t t~ [virt=QCD]", "seed": 930002,
            "physical_card": self.put("param_card.dat"), "ninja_archive": self.put("ninja/libninja.a"),
            "ninjars_archive": self.put("ninjars/libninjars.a")}

    def test_selected_archive_accepts_root_nested_and_equal_duplicates(self):
        for archive in ('libninja.a', 'libninjars.a'):
            for libraries in ({archive:'expected'}, {'reducer_lib/'+archive:'expected'},
                              {archive:'expected','reducer_lib/'+archive:'expected'}):
                with self.subTest(archive=archive,libraries=libraries):
                    self.assertTrue(cli.selected_archive_matches(libraries,archive,'expected'))

    def test_selected_archive_rejects_missing_exact_basename(self):
        for libraries in ({}, {'libninja.a.bak':'expected'}, {'other/libninjars.a':'expected'}):
            with self.subTest(libraries=libraries):
                self.assertFalse(cli.selected_archive_matches(libraries,'libninja.a','expected'))

    def test_selected_archive_rejects_wrong_root_or_nested_hash(self):
        for name in ('libninja.a','ninja_lib/libninja.a'):
            with self.subTest(name=name):
                self.assertFalse(cli.selected_archive_matches({name:'wrong'},'libninja.a','expected'))

    def test_selected_archive_rejects_any_conflicting_duplicate(self):
        for libraries in ({'libninja.a':'expected','ninja_lib/libninja.a':'wrong'},
                          {'libninja.a':'wrong','ninja_lib/libninja.a':'expected'}):
            with self.subTest(libraries=libraries):
                self.assertFalse(cli.selected_archive_matches(libraries,'libninja.a','expected'))

    def test_both_requested_backend_labels_and_units(self):
        for backend, label in cli.LABELS.items():
            result = cli.parse_profile(profile(label), backend)
            self.assertEqual(result["backend"], backend)
            self.assertEqual(result["single_helicity"]["reduction_estimate"]["milliseconds"], .93)
            self.assertEqual(result["reducible_loop_groups"], 42)

    def test_other_backends_are_out_of_scope(self):
        for backend in (1, 7):
            with self.assertRaisesRegex(ValueError, "Only Ninja"): cli.parse_profile(profile(), backend)

    def test_wrong_or_duplicate_backend_label_is_rejected(self):
        for text in (profile("NinjaRS"), profile() + "= Reduction tool tested...... Ninja\n"):
            with self.assertRaisesRegex(ValueError, "backend label"): cli.parse_profile(text, 6)

    def test_no_normal_gate_or_timed_status_invented(self):
        result = cli.parse_profile(profile(), 6)
        self.assertFalse(result["acceptance_eligible"])
        self.assertFalse(result["timed_status_counts_available"])
        self.assertIn("no direct reduction stopwatch", result["reduction_boundary"])
        self.assertNotIn("dp", result); self.assertNotIn("qp", result)

    def test_missing_repeated_or_wrong_units_rejected(self):
        for text in (profile().replace("ms", "s", 1), profile().replace("= One helicity", "= Other"), profile() + profile()):
            with self.assertRaises(ValueError): cli.parse_profile(text, 6)

    def test_nonfinite_and_nonpositive_totals_rejected(self):
        for total in ("1e999", "nan", "-1", "0"):
            with self.subTest(total=total), self.assertRaises(ValueError): cli.parse_profile(profile(total=total), 6)

    def test_partition_checked_at_actual_printed_rounding(self):
        cli.parse_profile(profile(total="0.201", coefs="0.2", reduction="0.00123"), 6)
        with self.assertRaisesRegex(ValueError, "partition"): cli.parse_profile(profile(total="1.23", coefs="0.3", reduction="0.7"), 6)

    def test_negative_estimate_kept_without_clamping(self):
        result = cli.parse_profile(profile(total="1", coefs="1.2", reduction="-0.2"), 6)
        self.assertEqual(result["single_helicity"]["reduction_estimate"]["milliseconds"], -.2)

    def test_missing_loopgroup_inventory_not_fabricated(self):
        self.assertIsNone(cli.parse_profile(profile().replace("groups..... 42", "groups..... NA"), 6)["reducible_loop_groups"])
        with self.assertRaises(ValueError): cli.parse_profile(profile().replace("groups..... 42", "groups..... -1"), 6)

    def test_actual_two_tool_cards_first100_then_cached0(self):
        case = self.case()
        for backend in (6, 8):
            for points in (0, 100):
                text = cli.command_card(case, backend, points)
                self.assertIn("check profile -reuse %d " % points, text)
                self.assertIn("--reduction=%d " % backend, text)
                self.assertEqual(text.count("check profile"), 1)
                self.assertNotIn("--CTModeRun=-1", text)
        for backend, points in ((1, 100), (7, 100), (6, 7)):
            with self.assertRaises(ValueError): cli.command_card(case, backend, points)

    def test_physical_card_drift_rejected_before_cli(self):
        case = self.case(); cli.policy.path(case["physical_card"]["path"]).write_text("changed")
        with self.assertRaisesRegex(ValueError, "Changed"): cli.command_card(case, 8, 100)

    def test_unknown_or_mismatched_scope_is_not_pairable(self):
        self.assertFalse(cli.same_stage_scope({}, {}))
        left = dict(physical_card_sha256="card", timing_energy_source="driver", helicity_index=2,
            physical_helicity=[-1, -1, -1, -1], loop_optimized_output=True, direct_dp=True,
            helicity_table_sha256="table", helicity_filter_sha256="filter", selected_helicity_weight=1,
            common_madloop_settings={"ctmoderun": 1}, compiler_settings_sha256="flags", actual_split_orders_target="default-all")
        self.assertTrue(cli.same_stage_scope(left, dict(left)))
        for key in left:
            right = dict(left); right[key] = None
            self.assertFalse(cli.same_stage_scope(left, right))
        self.assertFalse(cli.same_stage_scope(left, dict(left, helicity_index=3)))
        for key, value in (("actual_split_orders_target", -1), ("helicity_table_sha256", "different-table"),
                ("helicity_filter_sha256", "different-filter"), ("selected_helicity_weight", 2)):
            self.assertFalse(cli.same_stage_scope(left, dict(left, **{key: value})))

    def test_cached_stability_is_exactly_first100_not_new_sample(self):
        path = self.root / "saved.pkl"; path.write_bytes(pickle.dumps({"DP_stability": [{}] * 100, "QP_stability": [{}, {"Accuracy": 0.}] + [{}] * 98}))
        result = cli.cached_stability(path)
        self.assertEqual(result["dp_points"], 100); self.assertEqual(result["qp_points"], 1)
        self.assertIn("not matched-point CDF", result["scope"])

    def test_short_cache_and_executable_pickle_rejected(self):
        path = self.root / "saved.pkl"; path.write_bytes(pickle.dumps({"DP_stability": [{}] * 99}))
        with self.assertRaises(ValueError): cli.cached_stability(path)
        path.write_bytes(pickle.dumps(Path("untrusted-global")))
        with self.assertRaisesRegex(ValueError, "primitive"): cli.cached_stability(path)

    def test_no_launch_without_flag_or_exact_handoff(self):
        with mock.patch.object(cli.workers, "subprocess") as subprocess:
            with self.assertRaisesRegex(ValueError, "held"): cli.execute_one(self.root / "none", 0, self.root / "none")
            subprocess.Popen.assert_not_called()
        case = self.case()
        spec = self.root / "spec.json"; spec.write_text(json.dumps({"phase": cli.PHASE, "schedule": [{"case": "qcd-2", "backend": 8}], "cases": {"qcd-2": case}}))
        handoff = self.root / "handoff.json"; handoff.write_text("{}")
        with mock.patch.object(cli.workers, "subprocess") as subprocess:
            with self.assertRaisesRegex(ValueError, "handoff"): cli.execute_one(spec, 0, handoff, True)
            subprocess.Popen.assert_not_called()

    def test_source_drift_blocks_owned_cli_startup(self):
        source = self.put("source.py"); cli.policy.path(source["path"]).write_text("drift")
        spec = self.root / "spec.json"; spec.write_text(json.dumps({"phase": cli.PHASE, "ready": True,
            "schedule": [{"case": "qcd-2", "backend": 8}], "sources": {"source": source}}))
        handoff = self.root / "handoff.json"; handoff.write_text(json.dumps({"root_authorized": True, "phase": cli.PHASE, "cpu": 207, "specification_sha256": cli.workers.digest(spec)}))
        with mock.patch.object(cli.os, "sched_getaffinity", return_value={207}), mock.patch.object(cli.workers, "subprocess") as subprocess:
            with self.assertRaisesRegex(ValueError, "Changed"): cli.execute_one(spec, 0, handoff, True)
            subprocess.Popen.assert_not_called()

    def test_held_spec_is_not_launched_even_with_matching_handoff(self):
        spec = self.root / "spec.json"; spec.write_text(json.dumps({"phase": cli.PHASE, "ready": False,
            "schedule": [{"case": "qcd-2", "backend": 8}]}))
        handoff = self.root / "handoff.json"; handoff.write_text(json.dumps({"root_authorized": True,
            "phase": cli.PHASE, "cpu": 207, "specification_sha256": cli.workers.digest(spec)}))
        with mock.patch.object(cli.workers, "subprocess") as subprocess:
            with self.assertRaisesRegex(ValueError, "still held"): cli.execute_one(spec, 0, handoff, True)
            subprocess.Popen.assert_not_called()

    def test_mock_cli_failure_retains_raw_and_exits_owned_context(self):
        case = self.case(); exits = []
        report, work = self.root / "report", self.root / "work"
        source = self.put("source.py", "synthetic CLI source")
        producer = cli.bound(Path(cli.__file__))
        spec = self.root / "spec.json"; spec.write_text(json.dumps({"phase": cli.PHASE, "ready": True,
            "schedule": [{"case": "qcd-2", "backend": 8, "repetition": 0, "npoints": 100}],
            "sources": {"producer": producer, "madgraph/various/process_checks.py": source},
            "cases": {"qcd-2": case}, "report": cli.policy.relative(report),
            "work": cli.policy.relative(work), "timeout_seconds": 1}))
        handoff = self.root / "handoff.json"; handoff.write_text(json.dumps({"root_authorized": True,
            "phase": cli.PHASE, "cpu": 207, "specification_sha256": cli.workers.digest(spec)}))

        @contextmanager
        def owned(path):
            try: yield
            finally: exits.append(path)

        def fake(command, **kwargs):
            kwargs["stdout"].write("injected actual-CLI failure\n")
            (kwargs["cwd"] / "profile_optimized_failed.log").write_text("failed raw output")
            (kwargs["cwd"] / "MG5_debug").write_text("actual-work debug traceback\n")
            return mock.Mock(wait=mock.Mock(return_value=1))

        with mock.patch.object(cli.os, "sched_getaffinity", return_value={207}), \
                mock.patch.object(cli.ownership, "owned_runtime", owned), \
                mock.patch.object(cli.ownership, "profile_environment", return_value={k: "1" for k in cli.pair.THREADS}), \
                mock.patch.object(cli.workers.subprocess, "Popen", side_effect=fake) as spawn:
            with self.assertRaisesRegex(ValueError, "CLI failed"): cli.execute_one(spec, 0, handoff, True)
        self.assertEqual(spawn.call_count, 1); self.assertEqual(len(exits), 1)
        cell = report / "qcd-2/8/0"
        result = json.loads((cell / "result.json").read_text())
        self.assertFalse(result["complete"]); self.assertFalse(result["acceptance_eligible"])
        self.assertEqual((cell / "retained-work/profile_optimized_failed.log").read_text(), "failed raw output")
        self.assertIn("retained-work/profile_optimized_failed.log", result["files"])
        debug = cell / "retained-work/MG5_debug"
        self.assertEqual(debug.read_text(), "actual-work debug traceback\n")
        self.assertEqual(result["files"]["retained-work/MG5_debug"], cli.bound(debug))
        inputs = {}; failed = cli.failure_snapshot(cell / "result.json", inputs=inputs)
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["raw_files"]["retained-work/MG5_debug"], cli.bound(debug))
        self.assertIn(cli.policy.relative(debug), inputs)
        self.assertNotIn("timing", failed)
        debug.write_text("tampered retained debug")
        with self.assertRaisesRegex(ValueError, "Changed"):
            cli.failure_snapshot(cell / "result.json")

    def metadata(self, backend=6):
        def text_record(text):
            data = text.encode(); return {"text":text,"sha256":hashlib.sha256(data).hexdigest(),"bytes":len(data)}
        card = cli.MadLoopParam(io.StringIO((cli.ROOT / "Template/loop_material/StandAlone/Cards/MadLoopParams.dat").read_text()))
        card["MLReductionLib"] = str(backend); card["CTModeRun"] = 1; card["CTModeInit"] = 1
        stream = io.StringIO(); card.write(stream); card_text = stream.getvalue()
        parameters = dict(card)
        timing, stages = {"Booting_time": .1}, []
        export = self.root / "synthetic-export"
        for name, key, n, seconds in (("boot", None, 3, 0.), ("polarized_full", "run_polarized_total", 100, .00123),
                ("unpolarized_full", "run_unpolarized_total", 50, .00123),
                ("unpolarized_skip", "run_unpolarized_coefs", 50, .0003),
                ("polarized_skip", "run_polarized_coefs", 100, .0003)):
            wall = .1+n*seconds; value = (wall-.1)/n
            if key: timing[key] = value
            driver = ("      PARAMETER (NPSPOINTS = %d)\n      PARAMETER (READPS = .FALSE.)\n" % n +
                ("      CALL ML_SLOOPMATRIXHEL_THRES(P,2,MATELEM)\n" if name.startswith("polarized") else "      CALL ML_SLOOPMATRIX_THRES(P,MATELEM)\n")+
                "      SQRTS=max(1000d0,2.0d0*TOTMASS)\n")
            loop = "      PARAMETER (SKIPLOOPEVAL=.%s.)\n      PARAMETER (BOOTANDSTOP=.%s.)\n" % (
                "TRUE" if name.endswith("skip") else "FALSE", "TRUE" if name == "boot" else "FALSE")
            snapshot = export / "SavedRunInfos" / name; snapshot.mkdir(parents=True, exist_ok=True)
            for filename, source in (("check_sa.f",driver),("loop_matrix.f",loop)):(snapshot/filename).write_text(source)
            source_records = {filename:cli.bound(snapshot/filename) for filename in ("check_sa.f","loop_matrix.f")}
            for filename, record in source_records.items(): record["path"] = str((snapshot/filename).resolve())
            stages.append({"name": name, "actual_npoints": n, "read_ps": False,
                "helicity_one_based_index": 2 if name.startswith("polarized") else -1,
                "flags": {"SKIPLOOPEVAL": name.endswith("skip"), "BOOTANDSTOP": name == "boot"},
                "wall_seconds": wall, "boot_wall_seconds": .1, "derived_seconds_per_point": value,
                "effective_madloop_parameters": dict(parameters),
                "parameter_card": text_record("same physical card"), "subprocess_parameter_card": text_record("same physical card"),
                "effective_madloop_card": text_record(card_text), "madloop_card_aliases": [text_record(card_text)],
                "energy_assignment_lines": ["SQRTS=max(1000d0,2.0d0*TOTMASS)"],
                "source_snapshots":source_records, "check_sa":source_records["check_sa.f"], "loop_matrix":source_records["loop_matrix.f"],
                "runtime_files": {"Source/make_opts": {"sha256": "f"*64}}})
        return {"schema": "madloop-profile-timing-metadata-v1", "reduction_backends": [backend],
            "export_dir":str(export), "requested_split_orders":-1,
            "backend_labels": [cli.LABELS[backend]], "selected_timing_helicity": {
                "one_based_index": 2, "physical_tuple": [-1,-1,-1,-1], "filter_tokens": ["-10000","1"], "total_count": 2,
                "helicity_configurations":text_record("1 1 1 1\n-1 -1 -1 -1\n"), "filter":text_record("-10000 1\n")},
            "stages": stages, "timing_result": timing}

    def test_exact_raw_metadata_and_printed_units_crosscheck(self):
        result = cli.metadata_timing(self.metadata(), 6, cli.parse_profile(profile(), 6))
        self.assertTrue(result["scope_complete"]); self.assertEqual(result["adaptive_base_points"], 50)
        self.assertAlmostEqual(result["raw_timing"]["single_helicity"]["reduction_estimate_seconds_per_point"], .00093)

    def test_metadata_requires_actual_backend_and_direct_dp_card(self):
        for alter in (lambda m: m.update(reduction_backends=[8]),
                lambda m: m["stages"][1]["effective_madloop_parameters"].update(CTModeRun=-1),
                lambda m: m["stages"][1]["effective_madloop_parameters"].update(MLReductionLib="8")):
            metadata = self.metadata(); alter(metadata)
            with self.assertRaises(ValueError): cli.metadata_timing(metadata, 6, cli.parse_profile(profile(), 6))

    def test_metadata_rejects_missing_reordered_wrong_count_or_skip_stage(self):
        for alter in (lambda m: m["stages"].pop(), lambda m: m["stages"].reverse(),
                lambda m: m["stages"][1].update(actual_npoints=50),
                lambda m: m["stages"][2]["flags"].update(SKIPLOOPEVAL=True)):
            metadata = self.metadata(); alter(metadata)
            with self.assertRaises(ValueError): cli.metadata_timing(metadata, 6, cli.parse_profile(profile(), 6))

    def test_metadata_raw_derived_partition_cannot_be_replaced_by_printed_or_calibration(self):
        metadata = self.metadata(); metadata["stages"][1]["derived_seconds_per_point"] *= 1000
        with self.assertRaisesRegex(ValueError, "derived value"): cli.metadata_timing(metadata, 6, cli.parse_profile(profile(), 6))
        metadata = self.metadata(); metadata["timing_result"]["run_polarized_total"] = 1.23
        with self.assertRaises(ValueError): cli.metadata_timing(metadata, 6, cli.parse_profile(profile(), 6))

    def test_metadata_missing_energy_or_compiler_scope_stays_incomparable(self):
        for missing in ("energy_assignment_lines", "runtime_files"):
            metadata = self.metadata()
            for stage in metadata["stages"][1:]:
                if missing == "runtime_files": stage[missing] = {}
                else:
                    # An actually absent energy assignment remains unknown, not equal missing values.
                    stage[missing] = []
                    path = Path(stage["source_snapshots"]["check_sa.f"]["path"])
                    path = cli.policy.path(path); path.write_text(path.read_text().replace("      SQRTS=max(1000d0,2.0d0*TOTMASS)\n", ""))
                    record = cli.bound(path); record["path"] = str(path)
                    stage["source_snapshots"]["check_sa.f"] = record; stage["check_sa"] = dict(record)
            result = cli.metadata_timing(metadata, 6, cli.parse_profile(profile(), 6))
            self.assertFalse(result["scope_complete"])

    def test_metadata_rejects_physical_card_or_energy_drift_between_stages(self):
        for key, value in (("parameter_card", {"sha256":"x"*64,"text":"drift","bytes":5}), ("energy_assignment_lines", ["SQRTS=2000d0"])):
            metadata = self.metadata(); metadata["stages"][2][key] = value
            with self.assertRaises(ValueError): cli.metadata_timing(metadata, 6, cli.parse_profile(profile(), 6))

    def test_retained_driver_bytes_are_authority_for_stage_count(self):
        metadata = self.metadata(); path = Path(metadata["stages"][1]["source_snapshots"]["check_sa.f"]["path"])
        path.write_text(path.read_text().replace("NPSPOINTS = 100", "NPSPOINTS = 99"))
        with self.assertRaisesRegex(ValueError, "source snapshot"): cli.metadata_timing(metadata, 6, cli.parse_profile(profile(), 6))

    def test_embedded_table_card_filter_text_hashes_cannot_drift(self):
        for alter in (lambda m:m["selected_timing_helicity"]["filter"].update(text="1 1"),
                lambda m:m["selected_timing_helicity"].update(physical_tuple=[1,1,1,1]),
                lambda m:m["stages"][1]["effective_madloop_card"].update(text="different card")):
            metadata = self.metadata(); alter(metadata)
            with self.assertRaises(ValueError): cli.metadata_timing(metadata, 6, cli.parse_profile(profile(), 6))

    def test_captured_settings_must_equal_real_embedded_card_parser(self):
        metadata = self.metadata(); metadata["stages"][1]["effective_madloop_parameters"]["OSThres"] = 2e-8
        with self.assertRaisesRegex(ValueError, "card text"): cli.metadata_timing(metadata, 6, cli.parse_profile(profile(), 6))

    def test_copied_stage_sources_survive_later_cli_overwrite(self):
        import shutil
        metadata = self.metadata(); original = Path(metadata["export_dir"]) / "SavedRunInfos"
        retained = self.root / "retained-stage-inputs"; shutil.copytree(original,retained)
        (original / "polarized_full/check_sa.f").write_text("later CLI source")
        cli.metadata_timing(metadata, 6, cli.parse_profile(profile(), 6), retained)
        (retained / "polarized_full/check_sa.f").write_text("changed retained source")
        with self.assertRaisesRegex(ValueError, "source snapshot"): cli.metadata_timing(metadata, 6, cli.parse_profile(profile(), 6), retained)

    def test_actual_two_tool_scope_comparison_drops_only_selector(self):
        left = cli.metadata_timing(self.metadata(6), 6, cli.parse_profile(profile(), 6))
        right = cli.metadata_timing(self.metadata(8), 8, cli.parse_profile(profile("NinjaRS"), 8))
        self.assertTrue(cli.same_stage_scope(left["stage_scope"], right["stage_scope"]))
        right["stage_scope"]["physical_helicity"] = [1,1,1,1]
        self.assertFalse(cli.same_stage_scope(left["stage_scope"], right["stage_scope"]))

    def test_static_runtime_snapshot_reference_matches_perstage_predecessor(self):
        metadata = self.metadata(); printed = cli.parse_profile(profile(), 6)
        predecessor = cli.metadata_timing(metadata, 6, printed)
        metadata["runtime_files"] = metadata["stages"][1]["runtime_files"]
        for stage in metadata["stages"]:
            stage.pop("runtime_files"); stage["runtime_files_reference"] = "runtime_files"
        self.assertEqual(cli.metadata_timing(metadata, 6, printed), predecessor)
        metadata["stages"][1]["runtime_files_reference"] = "unbound-runtime"
        with self.assertRaisesRegex(ValueError, "runtime reference"): cli.metadata_timing(metadata, 6, printed)

    def test_actual_profile_wrapper_retimes_reuse0_and_copies_distinct_metadata(self):
        candidate = cli.ROOT / "madgraph/various/process_checks.py"
        node = next(n for n in ast.parse(candidate.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "check_profile")
        metadata_path = self.root / "profile_timing_metadata.json"
        events, saved = [], {"DP_stability": [{}] * 100}

        class Process(dict): pass
        generated = iter(({'processes': [{'has_born': True}]},
                          Process(has_born=True)))

        class Profiler:
            loop_optimized_output = True
            def __init__(self, **kwargs): pass
            def time_matrix_element(self, matrix, reusing, *args, **kwargs):
                events.append(("time", kwargs["options"]["npoints"]))
                value = float(sum(name == "time" for name, _ in events))
                metadata_path.write_text(json.dumps({"new_timing": value}))
                return {"run_polarized_total": value, "timing_metadata": {"path": str(metadata_path)}}
            def check_matrix_element_stability(self, matrix, **kwargs):
                events.append(("stability", kwargs["options"]["npoints"]))
                if kwargs["options"]["npoints"] == 0:
                    self_outer.assertTrue(kwargs["options"]["reuse"])
                return saved

        self_outer = self
        namespace = {"FakeInterface": lambda: None, "base_objects": SimpleNamespace(Process=Process),
            "LoopMatrixElementTimer": Profiler, "generate_loop_matrix_element": lambda *a, **k: ({}, next(generated)), "copy": copy}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(candidate), "exec"), namespace)
        for points in (100, 0):
            timing, stability = namespace["check_profile"]({"model": "SM"}, options={"reuse": True,"npoints": points}, MLOptions={"MLReductionLib": [8]})
            (self.root / ("metadata-%d.json" % points)).write_bytes(metadata_path.read_bytes())
            self.assertIs(stability, saved)
            self.assertEqual(len(stability["DP_stability"]), 100)
        self.assertEqual(events, [("time",100),("stability",100),("time",0),("stability",0)])
        first = json.loads((self.root / "metadata-100.json").read_text())
        second = json.loads((self.root / "metadata-0.json").read_text())
        self.assertEqual(first["new_timing"], 1.); self.assertEqual(second["new_timing"], 2.)
        self.assertEqual(first["profile_timing_result"]["run_polarized_total"], 1.)
        self.assertEqual(second["profile_timing_result"]["run_polarized_total"], 2.)

    def test_actual_cached_zero_branch_skips_stability_process_creation(self):
        candidate = cli.ROOT / "madgraph/various/process_checks.py"
        timer = next(n for n in ast.parse(candidate.read_text()).body if isinstance(n,ast.ClassDef) and n.name == "LoopMatrixElementTimer")
        method = next(n for n in timer.body if isinstance(n,ast.FunctionDef) and n.name == "check_matrix_element_stability")
        branches = [n for n in ast.walk(method) if isinstance(n,ast.If) and ast.unparse(n.test) == "nPoints == 0"]
        self.assertEqual(len(branches), 1)
        branch = branches[0]
        self.assertTrue(any(isinstance(n,ast.Continue) for n in ast.walk(branch)))
        popens = [n for n in ast.walk(method) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr == "Popen"]
        self.assertTrue(popens); self.assertTrue(all(n.lineno > branch.end_lineno for n in popens))

    def test_actual_source_split_order_target_default_all_or_minus_one_only(self):
        cli.metadata_timing(self.metadata(), 6, cli.parse_profile(profile(), 6))
        for target in (-1, 1):
            metadata = self.metadata()
            for stage in metadata["stages"]:
                path = Path(stage["source_snapshots"]["check_sa.f"]["path"])
                path.write_text(path.read_text() + "      CALL ML_SET_COUPLINGORDERS_TARGET(%d)\n" % target)
                record = cli.bound(path); record["path"] = str(path)
                stage["source_snapshots"]["check_sa.f"] = record; stage["check_sa"] = dict(record)
                stage["split_orders_target"] = target
            if target == -1: cli.metadata_timing(metadata, 6, cli.parse_profile(profile(), 6))
            else:
                with self.assertRaisesRegex(ValueError, "coupling-order target"): cli.metadata_timing(metadata, 6, cli.parse_profile(profile(), 6))
        metadata["stages"][0]["split_orders_target"] = -1
        with self.assertRaises(ValueError): cli.metadata_timing(metadata, 6, cli.parse_profile(profile(), 6))

    def mock_collected_result(self, mode="ok", declared=None):
        area = self.root / mode; area.mkdir()
        case = self.case(); source = self.put(mode + "/source.py", "frozen synthetic CLI source")
        spec = area / "spec.json"; handoff = area / "handoff.json"
        producer = cli.bound(Path(cli.__file__))
        scope = cli.declared_scope(declared or {})
        spec.write_text(json.dumps({"phase":cli.PHASE,"ready":True,"timeout_seconds":1,
            "work":cli.policy.relative(area/"work"),"report":cli.policy.relative(area/"report"),
            "sources":{"producer":producer,"madgraph/various/process_checks.py":source},
            "cases":{"qcd-2":case},"schedule":[{"case":"qcd-2","backend":8,"repetition":0,"npoints":scope["initial_stability_points"]}],
            **(declared or {})}))
        handoff.write_text(json.dumps({"root_authorized":True,"phase":cli.PHASE,"cpu":207,"specification_sha256":cli.workers.digest(spec)}))
        archive = case["ninjars_archive"]["sha256"]

        @contextmanager
        def owned(path): yield

        def fake(command, **kwargs):
            output = kwargs["cwd"] / "SAVEDTMP_CHECK_mock"
            metadata = self.metadata(8)
            old = Path(metadata["export_dir"]); shutil.move(old,output)
            metadata["export_dir"] = str(output)
            for stage in metadata["stages"]:
                for record in stage["source_snapshots"].values(): record["path"] = str(output/Path(record["path"]).relative_to(old))
                data = b"bound input"; physical = {"text":data.decode(),"sha256":hashlib.sha256(data).hexdigest(),"bytes":len(data)}
                stage["parameter_card"] = physical; stage["subprocess_parameter_card"] = dict(physical)
            runtime = {"Source/make_opts":{"sha256":"f"*64},"lib/libninjars.a":{"sha256":archive}}
            metadata["runtime_files"] = runtime
            for stage in metadata["stages"]: stage.pop("runtime_files"); stage["runtime_files_reference"] = "runtime_files"
            (output/"SavedRunInfos/profile_timing_metadata.json").write_text(json.dumps(metadata))
            directory = output/"SubProcesses/P0_mock"; (directory/"MadLoop5_resources").mkdir(parents=True)
            (directory/"MadLoop5_resources/MadLoopParams.dat").write_text(metadata["stages"][1]["effective_madloop_card"]["text"])
            for name in ("check","check_sa.f","loop_matrix.f"):(directory/name).write_text("not executed synthetic output")
            (output/"Cards").mkdir();(output/"Cards/param_card.dat").write_text("bound input")
            points = scope["initial_stability_points"]
            (directory/"SavedStabilityRun_NinjaRS_0.pkl").write_bytes(pickle.dumps({"DP_stability":[{}]*points,"QP_stability":[{}]*points}))
            (kwargs["cwd"]/"profile_optimized_mock.log").write_text(profile("NinjaRS"))
            kwargs["stdout"].write("Checking timing for process mock with 50 PS points.\n")
            if mode == "source": cli.policy.path(source["path"]).write_text("post-timing drift")
            return mock.Mock(wait=mock.Mock(return_value=0))

        with mock.patch.object(cli.os,"sched_getaffinity",return_value={207}), mock.patch.object(cli.ownership,"owned_runtime",owned), \
                mock.patch.object(cli.ownership,"profile_environment",return_value={k:"1" for k in cli.pair.THREADS}), \
                mock.patch.object(cli.workers.subprocess,"Popen",side_effect=fake), \
                mock.patch.object(cli.bench,"generated_source_manifest",return_value={"synthetic":True}), \
                mock.patch.object(cli.policy,"library_hashes",return_value={"libninjars.a":"0"*64 if mode == "archive" else archive}):
            if mode == "ok": cli.execute_one(spec,0,handoff,True)
            else:
                with self.assertRaises(ValueError): cli.execute_one(spec,0,handoff,True)
        result = area/"report/qcd-2/8/0/result.json"
        expected={"ninja_archive":case["ninja_archive"]["sha256"],"ninjars_archive":archive,"process_checks":source["sha256"]}
        return result,case,source,expected

    def test_mock_postguard_failures_cannot_retain_complete_true(self):
        for mode in ("archive","source"):
            path,_,_,_ = self.mock_collected_result(mode)
            result = json.loads(path.read_text())
            self.assertFalse(result["complete"]); self.assertIn("error",result)
            with self.assertRaisesRegex(ValueError,"Incomplete"): cli.report_snapshot(path)

    def test_readonly_authority_recomputes_raw_and_needs_no_original_build_products(self):
        path,case,source,expected = self.mock_collected_result()
        inputs = {}; observed = cli.report_snapshot(path,expected=expected,inputs=inputs)
        self.assertEqual(observed["item"]["backend"],8);self.assertFalse(observed["acceptance_eligible"])
        self.assertEqual(observed["stage_scope"]["actual_split_orders_target"],"default-all")
        self.assertTrue(inputs);self.assertEqual(observed["new_stability_points"],100)
        for record in (case["physical_card"],case["ninja_archive"],case["ninjars_archive"],source):cli.policy.path(record["path"]).unlink()
        self.assertEqual(cli.report_snapshot(path,expected=expected)["raw_timing"],observed["raw_timing"])
        portable = importlib.util.spec_from_file_location("retained_cli",path.parent/"profile_cli.py")
        copied = importlib.util.module_from_spec(portable);portable.loader.exec_module(copied)
        self.assertEqual(copied.ROOT,cli.ROOT)
        with self.assertRaisesRegex(ValueError,"pins"):cli.report_snapshot(path,expected={**expected,"ninjars_archive":"0"*64})

    def test_readonly_authority_rejects_tampered_raw_summary_or_missing_inventory(self):
        path,_,_,expected = self.mock_collected_result();original=json.loads(path.read_text())
        modified=json.loads(path.read_text());modified["raw_timing"]["single_helicity"]["seconds_per_point"] *= 2
        path.write_text(json.dumps(modified))
        with self.assertRaisesRegex(ValueError,"disagrees"):cli.report_snapshot(path,expected)
        modified=json.loads(json.dumps(original));del modified["files"]["profile.log"];path.write_text(json.dumps(modified))
        with self.assertRaisesRegex(ValueError,"Missing actual"):cli.report_snapshot(path,expected)
        path.write_text(json.dumps(original));(path.parent/"profile.log").write_text("different raw log")
        with self.assertRaisesRegex(ValueError,"Changed"):cli.report_snapshot(path,expected)

    def test_failure_authority_retains_only_attempt_error_and_custody_without_live_inputs(self):
        path,case,source,expected = self.mock_collected_result("archive")
        for record in (case["physical_card"],case["ninja_archive"],case["ninjars_archive"],source):
            cli.policy.path(record["path"]).unlink()
        inputs = {}; failed = cli.failure_snapshot(path,expected,inputs)
        self.assertEqual(failed["status"],"failed");self.assertIn("archive mismatch",failed["error"])
        self.assertEqual(failed["item"],{"case":"qcd-2","backend":8,"repetition":0,"npoints":100})
        self.assertEqual(failed["source"],cli.bound(path));self.assertIn("profile.log",failed["raw_files"])
        self.assertIn(cli.policy.relative(path),inputs)
        for key in ("timing","raw_timing","acceptance_eligible","stage_scope","median_seconds"):
            self.assertNotIn(key,failed)
        with self.assertRaisesRegex(ValueError,"Incomplete"):cli.report_snapshot(path,expected)

    def test_failure_authority_rejects_success_wrong_cell_pins_producer_and_raw_drift(self):
        success,_,_,expected = self.mock_collected_result()
        with self.assertRaisesRegex(ValueError,"failed CLI attempt"):cli.failure_snapshot(success,expected)
        path,_,_,expected = self.mock_collected_result("archive"); original = json.loads(path.read_text())
        with self.assertRaisesRegex(ValueError,"pins"):cli.failure_snapshot(path,{**expected,"process_checks":"0"*64})
        for key,value in (("item",{**original["item"],"repetition":1}), ("producer_role","other-producer"), ("error","")):
            changed = copy.deepcopy(original);changed[key]=value;path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):cli.failure_snapshot(path,expected)
        changed=copy.deepcopy(original);del changed["files"]["commands.mg5"];path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError,"Missing retained"):cli.failure_snapshot(path,expected)
        path.write_text(json.dumps(original));(path.parent/"profile.log").write_text("changed failure raw log")
        with self.assertRaisesRegex(ValueError,"Changed"):cli.failure_snapshot(path,expected)

    def test_declared_light_scope_and_historical_defaults_are_strict(self):
        self.assertEqual(cli.declared_scope({}), {"repetitions":7,"initial_stability_points":100})
        self.assertEqual(cli.declared_scope({"repetitions":3,"initial_stability_points":20}),
            {"repetitions":3,"initial_stability_points":20})
        for field, values in (("repetitions", (2, True, 3., "3", None)),
                ("initial_stability_points", (0, 19, 99, True, 20., "20", None))):
            for value in values:
                with self.subTest(field=field,value=value), self.assertRaises(ValueError):
                    cli.declared_scope({field:value})

    def test_declared_three_session_first20_then_reuse0_schedule(self):
        scope = cli.declared_scope({"repetitions":3,"initial_stability_points":20})
        for backend in (6,8):
            for repetition in range(3):
                cli.check_declared_item({"backend":backend,"repetition":repetition,
                    "npoints":20 if repetition == 0 else 0},scope)
        for item in ({"backend":8,"repetition":0,"npoints":100}, {"backend":8,"repetition":0,"npoints":0},
                {"backend":8,"repetition":1,"npoints":20}, {"backend":8,"repetition":3,"npoints":0},
                {"backend":8,"repetition":-1,"npoints":0}, {"backend":8,"repetition":True,"npoints":0},
                {"backend":8,"repetition":1,"npoints":False}, {"backend":7,"repetition":0,"npoints":20}):
            with self.subTest(item=item), self.assertRaisesRegex(ValueError,"schedule"):
                cli.check_declared_item(item,scope)

    def test_first20_command_uses_same_real_profile_command(self):
        case = self.case()
        for backend in (6,8):
            text = cli.command_card(case,backend,20)
            self.assertIn("check profile -reuse 20 ",text)
            self.assertIn("--reduction=%d " % backend,text)
            self.assertEqual(text.replace("-reuse 20 ","-reuse 100 "),cli.command_card(case,backend,100))

    def test_twenty_point_cache_requires_explicit_exact_dp_and_qp_lengths(self):
        path = self.root / "twenty.pkl"
        path.write_bytes(pickle.dumps({"DP_stability":[{}]*20,"QP_stability":[{}]*19+[{'Accuracy':0.}]}))
        self.assertEqual(cli.cached_stability(path,20)["dp_points"],20)
        self.assertEqual(cli.cached_stability(path,20)["qp_points"],1)
        with self.assertRaisesRegex(ValueError,"first100"): cli.cached_stability(path)
        for expected in (0,19,True,20.,None):
            with self.subTest(expected=expected), self.assertRaises(ValueError): cli.cached_stability(path,expected)
        for data in ({"DP_stability":[{}]*19,"QP_stability":[{}]*20},
                {"DP_stability":[{}]*20,"QP_stability":[{}]*19}, {"DP_stability":[{}]*20}):
            path.write_bytes(pickle.dumps(data))
            with self.assertRaisesRegex(ValueError,"first20"): cli.cached_stability(path,20)

    def test_light20_success_and_failure_readers_return_declared_scope(self):
        scope = {"repetitions":3,"initial_stability_points":20}
        path,_,_,expected = self.mock_collected_result(declared=scope)
        observed = cli.report_snapshot(path,expected)
        self.assertEqual(observed["declared_scope"],scope)
        self.assertEqual(observed["new_stability_points"],20)
        self.assertEqual(observed["stability"]["dp_points"],20)
        path,_,_,expected = self.mock_collected_result("archive",declared=scope)
        self.assertEqual(cli.failure_snapshot(path,expected)["declared_scope"],scope)

    def test_invalid_light_schedule_is_rejected_before_process_or_report_creation(self):
        case = self.case();spec = self.root/"light.json";handoff = self.root/"handoff.json"
        for item in ({"case":"qcd-2","backend":8,"repetition":0,"npoints":100},
                {"case":"qcd-2","backend":8,"repetition":3,"npoints":0}):
            spec.write_text(json.dumps({"phase":cli.PHASE,"ready":True,"sources":{},"cases":{"qcd-2":case},
                "schedule":[item],"repetitions":3,"initial_stability_points":20}))
            handoff.write_text(json.dumps({"root_authorized":True,"phase":cli.PHASE,"cpu":207,
                "specification_sha256":cli.workers.digest(spec)}))
            with mock.patch.object(cli.os,"sched_getaffinity",return_value={207}), \
                    mock.patch.object(cli.workers.subprocess,"Popen") as spawn:
                with self.assertRaisesRegex(ValueError,"schedule"):cli.execute_one(spec,0,handoff,True)
                spawn.assert_not_called()

    def test_light_readonly_authority_rejects_wrong_cache_count_even_with_refreshed_custody(self):
        path,_,_,expected = self.mock_collected_result(declared={"repetitions":3,"initial_stability_points":20})
        result = json.loads(path.read_text()); cache = path.parent/"SavedStabilityRun_NinjaRS_0.pkl"
        cache.write_bytes(pickle.dumps({"DP_stability":[{}]*100,"QP_stability":[{}]*100}))
        result["files"][cache.name] = cli.bound(cache);result["stability"]["cache"] = cli.bound(cache)
        path.write_text(json.dumps(result))
        with self.assertRaisesRegex(ValueError,"first20"):cli.report_snapshot(path,expected)


if __name__ == "__main__": unittest.main()
