"""Cohort labels must not hide retries, migrations, failures or altered inputs."""
import importlib.util
import copy
import io
import json
import os
from contextlib import redirect_stdout
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
import tirust_benchmark_cohorts as cohort


class TestCampaignCohorts(unittest.TestCase):
    def setUp(self):
        # Reuse the campaign fixture's actual source/native/policy registration,
        # without inheriting or rerunning its unrelated test suite.
        from tests.unit_tests.loop.test_tirust_benchmark_campaign import TestCampaign
        import tirust_benchmark_campaign as campaign
        self.campaign = campaign
        self.fixture = TestCampaign()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.bench = self.fixture.bench
        self.bench.campaign_binding = lambda: campaign.campaign_binding(self.bench)
        self.bench.campaign_guard = lambda stage, inputs=None: campaign.verify_campaign(self.bench, stage, inputs)
        patcher = mock.patch.object(cohort, "bench", self.bench)
        patcher.start()
        self.addCleanup(patcher.stop)

    def activate(self):
        self.fixture.register()
        self.campaign.activate_campaign(self.bench, "optimized")

    def evidence(self):
        self.activate()
        binding = self.bench.campaign_binding()
        report, _ = cohort.locations("cohort-test")
        report.mkdir(parents=True)
        for source, target in (("manifest.json", "campaign-manifest.json"), ("cases.json", "cases.json")):
            shutil.copy2(self.bench.REPORT / source, report / target)
        manifest = self.bench.CAMPAIGN_MANIFEST
        spec = {"campaign_binding": binding, "case": {"id": "example"},
                "config": json.loads((report / "cases.json").read_text())}
        generation = {"campaign_binding": binding, "source": "candidate",
                      **self.campaign.export_environment_binding(),
                      "source_identity": manifest["candidate"]["exporter_identity"]}
        shared = manifest["shared_inputs"]["files"]
        native = manifest["candidate"]["native"]["files"]
        inputs = {"campaign_binding": binding, "generation": generation,
                  "points_sha256": shared["points/example.dat"],
                  "point_metadata_sha256": shared["points/example.json"],
                  "runtime": {"libraries": {"/native/libmadloop_tirust.so": native["lib/libmadloop_tirust.so"]},
                              "archives": {"/native/libtirust.a": native["lib/libtirust.a"]},
                              "metadata": {"/native/tirust-config.json": native["lib/tirust-config.json"]}}}
        compiled = {"campaign_binding": binding, "original": {"campaign_binding": binding}}
        results = {"campaign_binding": binding}
        return report, spec, inputs, compiled, results

    def test_explicit_namespace_preserves_legacy_locations(self):
        original_work = self.bench.WORK
        self.assertEqual(cohort.locations("sample"),
                         (self.bench.LEGACY_REPORT / "experiments/cohorts/sample",
                          cohort.ROOT / ".tirust-build/cohorts/sample"))
        self.activate()
        self.assertEqual(cohort.locations("sample"),
                         (self.bench.REPORT / "experiments/cohorts/sample",
                          self.bench.WORK / "cohort-diagnostics/sample"))
        self.assertEqual(original_work, self.bench.LEGACY_WORK)
        with self.assertRaises(ValueError):
            cohort.locations("../legacy")

    def test_private_configuration_restores_namespace_on_failure(self):
        self.activate()
        original_work = self.bench.WORK
        private = original_work / "cohort-diagnostics/example"
        def fail(*args):
            self.assertEqual(self.bench.WORK, private)
            raise RuntimeError("card write failed")
        self.bench.configure = fail
        with self.assertRaisesRegex(RuntimeError, "card write failed"):
            cohort.configure_clone(private, "candidate", {}, [9], {})
        self.assertEqual(self.bench.WORK, original_work)
        self.bench.campaign_guard("cohort-before")

    def test_retained_source_and_native_bindings_are_enforced(self):
        args = self.evidence()
        cohort.validate_campaign_evidence(*args)
        for section, key in (("libraries", "/native/libmadloop_tirust.so"),
                             ("archives", "/native/libtirust.a"),
                             ("metadata", "/native/tirust-config.json")):
            bad = copy.deepcopy(args)
            bad[2]["runtime"][section][key] = "wrong"
            with self.subTest(section=section), self.assertRaisesRegex(ValueError, "Recorded TIRust runtime"):
                cohort.validate_campaign_evidence(*bad)
        bad = copy.deepcopy(args)
        bad[2]["generation"]["source_identity"]["files_sha256"] = "old candidate"
        with self.assertRaisesRegex(ValueError, "candidate source"):
            cohort.validate_campaign_evidence(*bad)
        # Reparse is historical: a later live library/source cannot invalidate
        # correctly bound retained observations from this campaign.
        self.fixture._install("a later library\n")
        cohort.validate_campaign_evidence(*args)

    def test_retained_manifest_policy_and_points_cannot_be_substituted(self):
        args = self.evidence()
        for key in ("points_sha256", "point_metadata_sha256"):
            bad = copy.deepcopy(args)
            bad[2][key] = "other points"
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "preregistration"):
                cohort.validate_campaign_evidence(*bad)
        bad = copy.deepcopy(args)
        bad[1]["config"]["points_per_case"] = 99
        with self.assertRaisesRegex(ValueError, "preregistration"):
            cohort.validate_campaign_evidence(*bad)
        (args[0] / "campaign-manifest.json").write_text("{}")
        with self.assertRaises((ValueError, KeyError)):
            cohort.validate_campaign_evidence(*args)

    def test_cross_campaign_generation_or_evidence_is_rejected_before_copy(self):
        args = self.evidence()
        for position in (1, 2, 3, 4):
            bad = copy.deepcopy(args)
            bad[position]["campaign_binding"] = {"id": "other", "manifest_sha256": "other"}
            with self.subTest(position=position), self.assertRaises(ValueError):
                cohort.validate_campaign_evidence(*bad)
        original = Path(self.fixture.temporary.name) / "wrong-output"
        original.mkdir()
        (original / "compilation.json").write_text(json.dumps({"campaign_binding": self.bench.campaign_binding()}))
        (original / "generation.json").write_text(json.dumps({"campaign_binding": None}))
        with mock.patch.object(cohort.shutil, "copytree") as copytree:
            with self.assertRaisesRegex(ValueError, "different campaign"):
                cohort.build_clone(original, original / "clone", original, {})
            copytree.assert_not_called()

    def test_live_campaign_drift_prevents_native_run(self):
        self.activate()
        self.fixture._install("changed after registration\n")
        with mock.patch.object(cohort, "build_clone") as build:
            with self.assertRaisesRegex(ValueError, "Native TIRust installation"):
                cohort.run(None)
            build.assert_not_called()

    def test_legacy_evidence_needs_no_campaign_manifest(self):
        cohort.validate_campaign_evidence(Path("unused"), {}, {"generation": {}}, {"original": {}}, {})


class TestCampaignCohortCLI(unittest.TestCase):
    def test_selection_is_explicit_and_applies_before_validation(self):
        import tirust_benchmark_campaign as campaign
        for selected in (None, "final-candidate"):
            args = ["cohorts", "validate", "--label", "sample"]
            if selected:
                args += ["--campaign", selected]
            calls = []
            with mock.patch.object(sys, "argv", args), \
                 mock.patch.object(campaign, "activate_campaign", side_effect=lambda *a: calls.append("activate")) as activate, \
                 mock.patch.object(cohort, "validate_report", side_effect=lambda *a: calls.append("validate") or {}) as validate, \
                 redirect_stdout(io.StringIO()):
                cohort.main()
            self.assertEqual(calls, ["activate", "validate"] if selected else ["validate"])
            if selected:
                activate.assert_called_once_with(cohort.bench, selected)
            else:
                activate.assert_not_called()
            validate.assert_called_once()


class TestCohortEvidence(unittest.TestCase):
    def test_strict_status_and_retry_categories(self):
        for code, retry, backend, expected in (
                (218, [0, 0, 0, 0], 8, "dp"),
                (218, [1, 1, 0, 0], 8, "compensated_dp"),
                (322, [1, 0, 1, 0], 8, "qp"),
                (323, [0, 0, 0, 0], 9, "qp"),
                (219, [0, 0, 0, 0], 9, "dp"),
                (229, [0, 0, 0, 0], 9, "failed"),
                (313, [0, 0, 0, 0], 9, "failed"),
                (218, [1, 0, 1, 0], 8, "failed"),
                (218, [1, 0, 0, 0], 8, "failed"),
                (219, [1, 1, 0, 0], 9, "failed"),
                (420, [0, 0, 0, 0], 9, "failed")):
            with self.subTest(code=code, retry=retry):
                self.assertEqual(cohort.category(code, retry, backend), expected)
        self.assertEqual(cohort.category(219, [0]*4, 9, False), "failed")
        self.assertEqual(cohort.category(323, [0]*4, 9, True, [False, True]), "failed")
        self.assertEqual(cohort.category(219, [0]*4, 9, True, [True, False]), "failed")

    def test_absolute_per_call_counters_are_not_deltas(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "calls.dat"
            path.write_text("1 1 218 1 1 0 0 1 0 0\n2 1 218 0 0 0 0 1 0 0\n3 1 218 1 1 0 0 1 0 0\n")
            calls = cohort.read_calls(path, 8, [1], 3)
            self.assertEqual([r["category"] for r in calls], ["compensated_dp", "dp", "compensated_dp"])
            with self.assertRaises(ValueError):
                cohort.read_calls(path, 8, [2], 3)

    def test_indices_reject_reordering_duplicates_substitution_and_bool(self):
        for indices in ([2, 1], [1, 1], [0], [101], [True]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                cohort.validate_indices(indices, 100)

    def test_generated_counter_ownership_is_required(self):
        with self.assertRaisesRegex(ValueError, "layout"):
            cohort.render(cohort.TEMPLATE.read_text(), {"prefix": "STUB_", "nexternal": 3, "nborn": 2}, "")
        params = {"prefix": "STUB_", "nexternal": 3, "nborn": 2}
        with self.assertRaisesRegex(ValueError, "FPE flag layout"):
            cohort.render(cohort.TEMPLATE.read_text(), params,
                          STUB.replace(" common /STUB_FPE_IN_REDUCTION/", "c common /STUB_FPE_IN_REDUCTION/"))

    def test_driver_build_plan_rejects_shared_numerical_rebuild(self):
        bad = subprocess.CompletedProcess([], 0, stdout="gfortran -c loop_matrix.f\ngfortran -o check check_sa.o\n", stderr="")
        with mock.patch.object(cohort.subprocess, "run", return_value=bad), self.assertRaisesRegex(ValueError, "numerical-object"):
            cohort.link_commands(Path("."), Path("."), {})
        bad.stdout = "make -C ../../Source MODEL\ngfortran -o check check_sa.o\n"
        with mock.patch.object(cohort.subprocess, "run", return_value=bad), self.assertRaisesRegex(ValueError, "Unexpected"):
            cohort.link_commands(Path("."), Path("."), {})

    def test_failure_raw_evidence_survives_parser_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / "cohort-calls.dat").write_text("1 1 420 0 0 0 0 0 0 1\n")
            result = cohort.read_observation(directory, 9, [1], 1, "dp")
            self.assertFalse(result["pure_cohort"])
            self.assertEqual(result["migrations"][0]["status"], 420)
            self.assertIsNotNone(result["numerical_error"])

    def test_missing_failure_flags_cannot_silently_downgrade_new_trace(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "calls.dat"
            path.write_text("1 1 219 0 0 0 0 1\n")
            with self.assertRaisesRegex(ValueError, "Malformed"):
                cohort.read_calls(path, 9, [1], 1)
            self.assertEqual(cohort.read_calls(path, 9, [1], 1, flags_required=False)[0]["category"], "dp")

    def test_fixed_qp_reported_failure_is_independent_of_return_code(self):
        filename = ROOT / "docs/benchmarkfolder/experiments/qcd-n3-adjudication/fpe-completeness/run.py"
        spec = importlib.util.spec_from_file_location("qcd3_fpe_test", filename)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "flags.dat"
            path.write_text("0 0 1 109 0 0\n1 1 1 120 0 0\n2 1 1 120 0 1\n")
            result = helper.summarize_flags(path, 1)
            self.assertFalse(result["all_observed_flags_false"])
            self.assertFalse(result["final_qp_flags_false"])
            self.assertEqual(result["flagged_calls"], [[2, 1, 1, 120, 0, 1]])
            path.write_text("0 0 1 109 0 0\n1 1 1 120 0 0\n")
            with self.assertRaisesRegex(ValueError, "Incomplete/reordered fixed-QP"):
                helper.summarize_flags(path, 1)


STUB = '''
subroutine SETPARA(card)
 character(*) card
 real*8 mu_r
 common /scale/ mu_r
 mu_r=2d0
end subroutine
subroutine UPDATE_AS_PARAM()
end subroutine
subroutine STUB_GET_ANSWER_DIMENSION(n)
 integer n
 n=2
end subroutine
subroutine STUB_GET_NSQSO_LOOP(n)
 integer n
 n=1
end subroutine
subroutine STUB_FORCE_STABILITY_CHECK(force)
 logical force
 if (.not.force) stop 91
end subroutine
subroutine STUB_SLOOPMATRIX_THRES(p,ans,threshold,acc,code)
 real*8 p(0:3,3),ans(0:3,0:2),threshold,acc(0:1)
 integer code,j,point,migrate
 integer NRS_RETRY_COUNT,NRS_RETRY_PASS,NRS_RETRY_FAIL,NRS_RETRY_HARD_SKIP
 common /STUB_NRS_RETRY_STATS/ NRS_RETRY_COUNT,NRS_RETRY_PASS,NRS_RETRY_FAIL,NRS_RETRY_HARD_SKIP
 logical FPE_IN_DP_REDUCTION,FPE_IN_QP_REDUCTION
 common /STUB_FPE_IN_REDUCTION/ FPE_IN_DP_REDUCTION,FPE_IN_QP_REDUCTION
 character*32 setting
 integer,save::calls=0
 calls=calls+1
 point=nint(p(0,3))
 if(threshold.ne.-1d0) stop 92
 NRS_RETRY_COUNT=0
 NRS_RETRY_PASS=0
 NRS_RETRY_FAIL=0
 NRS_RETRY_HARD_SKIP=0
 FPE_IN_DP_REDUCTION=.false.
 FPE_IN_QP_REDUCTION=.false.
 ans(:,0)=(/11d0,12d0,13d0,14d0/)
 ans(:,1)=(/21d0,22d0,23d0,24d0/)
 ans(0,2)=31d0
 acc=(/1d-10,2d-10/)
 code=218
 if(point.eq.2)then
   NRS_RETRY_COUNT=1
   NRS_RETRY_PASS=1
 endif
 if(point.eq.3) code=322
 call get_environment_variable('COHORT_STUB_MIGRATE',setting)
 if(len_trim(setting).gt.0.and.calls.gt.7.and.point.eq.1)code=322
 call get_environment_variable('COHORT_STUB_FPE',setting)
 if(len_trim(setting).gt.0.and.calls.gt.7.and.point.eq.1)FPE_IN_QP_REDUCTION=.true.
 if(calls.eq.1)code=108
end subroutine
subroutine STUB_EXIT_MADLOOP()
end subroutine
'''


@unittest.skipUnless(shutil.which("gfortran"), "GNU Fortran required")
class TestCompiledCohorts(unittest.TestCase):
    def test_full_warmup_then_exact_subsets_and_every_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            params = {"prefix": "STUB_", "nexternal": 3, "nborn": 2}
            (directory / "driver.f").write_text(cohort.render(cohort.TEMPLATE.read_text(), params, STUB))
            (directory / "coupl.inc").write_text("      REAL*8 MU_R\n      COMMON /SCALE/ MU_R\n")
            (directory / "pmass.inc").write_text("      PMASS=(/0D0,0D0,2D0/)\n")
            (directory / "stub.f90").write_text(STUB)
            subprocess.run([shutil.which("gfortran"), "-ffixed-line-length-none", "-fcheck=all", "-o", "check", "driver.f", "stub.f90"],
                           cwd=directory, check=True, capture_output=True)
            points = directory / "points.dat"
            points.write_text("3\n" + "".join(f"1 0 0 1\n1 0 0 -1\n{i} 0 0 0\n" for i in (1, 2, 3)))
            with cohort.Worker([directory / "check", points], directory, directory / "worker.log", timeout=5) as worker:
                frozen = cohort.classification(directory, 8, 3, "parent-sha")
                self.assertEqual(frozen["indices"], {"dp": [1], "compensated_dp": [2], "qp": [3], "failed": []})
                self.assertEqual(frozen["rows"][2]["accuracy"], 1e-10)  # Born-only column
                warm = (directory / "warmup.dat").read_text().splitlines()[1:]
                self.assertEqual([int(line.split()[1]) for line in warm[::3]], [1, 1, 2, 3])
                result = cohort.measure(worker, directory / "batch", 8, [1, 2], 3, "dp", frozen["rows"])
                self.assertEqual(result["timed_calls"], 6)
                self.assertEqual([r["point"] for r in result["migrations"]], [2, 2, 2])
                self.assertEqual(result["observation"]["timed_status_counts"], {"218": 6})
                self.assertFalse(result["pure_cohort"])
                self.assertEqual(len(result["observation"]["rows"]), 6)
                self.assertEqual(result["fpe_flag_counts"], {"fpe_dp": 0, "fpe_qp": 0})
                self.assertEqual(len(frozen["warmup_flags"]), 4)
            with self.assertRaises(FileExistsError):
                cohort.exclusive_json(directory / "batch/result.json", {})
            env = dict(os.environ, COHORT_STUB_MIGRATE="1")
            with cohort.Worker([directory / "check", points], directory, directory / "migration.log", timeout=5, env=env) as worker:
                frozen = cohort.classification(directory, 8, 3, "parent-sha")
                result = cohort.measure(worker, directory / "migrated", 8, [1], 2, "dp", frozen["rows"])
                self.assertEqual(frozen["indices"]["dp"], [1])
                self.assertFalse(result["pure_cohort"])
                self.assertEqual([r["category"] for r in result["migrations"]], ["qp", "qp"])
            env = dict(os.environ, COHORT_STUB_FPE="1")
            with cohort.Worker([directory / "check", points], directory, directory / "fpe.log", timeout=5, env=env) as worker:
                frozen = cohort.classification(directory, 8, 3, "parent-sha")
                result = cohort.measure(worker, directory / "fpe", 8, [1], 2, "dp", frozen["rows"])
                self.assertFalse(result["pure_cohort"])
                self.assertEqual(result["observation"]["timed_status_counts"], {"218": 2})
                self.assertEqual(result["fpe_flag_counts"], {"fpe_dp": 0, "fpe_qp": 2})
                self.assertEqual([r["category"] for r in result["migrations"]], ["failed", "failed"])


if __name__ == "__main__":
    unittest.main()
