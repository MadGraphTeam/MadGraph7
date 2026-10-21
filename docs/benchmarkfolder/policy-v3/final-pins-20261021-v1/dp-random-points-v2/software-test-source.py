"""Python phase-space/source controls; no compiler or numerical reductions."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("dp_random_points", ROOT / "tools/ninjars_dp_random_points.py")
points = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(points)


def case(nfinal=2, masses=None, seed=20261021, induced=False):
    definition = {"id": "synthetic", "model": "loop_sm", "process": "synthetic [sqrvirt=QCD]" if induced else "synthetic [virt=QCD]", "nfinal": nfinal}
    return {"definition": definition, "masses": masses or [0., 0., *([0.] * nfinal)],
            "seed": seed, "cuts": points.case_cuts(definition)}


class RandomPointConstruction(unittest.TestCase):
    def test_reproducible_independent_seed_streams(self):
        item = case(nfinal=4)
        first = points.stream(item, 12, 1000., 10000)
        self.assertEqual(first, points.stream(item, 12, 1000., 10000))
        item["seed"] += 1
        self.assertNotEqual(first[0], points.stream(item, 12, 1000., 10000)[0])

    def test_ambient_random_state_is_unchanged(self):
        before = random.getstate()
        points.stream(case(), 3, 1000., 100)
        self.assertEqual(before, random.getstate())

    def test_massive_rambo_points_pass_independent_physical_validator(self):
        item = case(nfinal=4, masses=[0., 0., 173., 173., 0., 0.])
        raw, accounting = points.stream(item, 20, 1000., 10000)
        from ninjars_supplement_v2 import point_metadata
        with tempfile.TemporaryDirectory(dir=ROOT, prefix=".test-random-points-") as directory:
            filename = Path(directory) / "points.dat"
            filename.write_bytes(raw)
            result = point_metadata(filename, item["masses"], 4)
        self.assertEqual(result["points"], 20)
        self.assertLess(result["max_relative_conservation_residual"], 1e-10)
        self.assertLess(result["max_relative_mass_shell_residual"], 1e-10)
        self.assertGreater(accounting["min_final_pt_gev"], 50.)
        self.assertGreater(accounting["min_pair_delta_r"], .5)
        self.assertEqual(accounting["proposals"], 20 + accounting["rejected"])
        self.assertEqual(sum(accounting["rejection_reasons"].values()), accounting["rejected"])

    def test_boosted_one_body_is_descriptive_with_one_invariant(self):
        item = case(nfinal=1, masses=[0., 0., 125.])
        raw, accounting = points.stream(item, 10, 1000., 10)
        values = [[float(value) for value in line.split()] for line in raw.decode().splitlines()[1:]]
        self.assertEqual(accounting["proposals"], 10)
        self.assertEqual(accounting["rejected"], 0)
        self.assertIsNone(accounting["min_pair_delta_r"])
        self.assertGreater(len({tuple(p) for p in values[2::3]}), 1)
        for p in values[2::3]:
            self.assertAlmostEqual(p[0] ** 2 - sum(x * x for x in p[1:]), 125. ** 2, places=8)

    def test_loop_reference_omits_pt_cut_only(self):
        regular, induced = case(), case(induced=True)
        self.assertEqual(regular["cuts"]["pt_min_gev"], 50.)
        self.assertIsNone(induced["cuts"]["pt_min_gev"])
        self.assertEqual(induced["cuts"]["delta_r_min"], .5)
        event = [[500., 0., 0., 500.], [500., 0., 0., -500.], [500., 1., 0., 499.], [500., -1., 0., -499.]]
        self.assertEqual(points.rejection(event, regular["cuts"]), "pt")
        self.assertIsNone(points.rejection(event, induced["cuts"]))

    def test_strict_cut_boundary_and_wrapped_azimuth(self):
        self.assertEqual(points.rejection([[0.] * 4] * 2 + [[50., 50., 0., 0.], [50., -50., 0., 0.]], case()["cuts"]), "pt")
        epsilon = .01
        event = [[0.] * 4] * 2 + [[100., -100., epsilon, 0.], [100., -100., -epsilon, 0.]]
        self.assertLess(points.cut_values(event)[1], .001)
        self.assertEqual(points.rejection(event, case()["cuts"]), "delta_r")

    def test_pseudorapidity_is_used_for_massive_particles(self):
        event = [[0.] * 4] * 2 + [[200., 60., 0., 80.], [200., 0., 60., -80.]]
        self.assertAlmostEqual(points.cut_values(event)[1], math.hypot(2 * math.asinh(80. / 60.), math.pi / 2))

    def test_invalid_dimensions_masses_energy_and_counts_fail(self):
        for item, count, energy, budget in [(case(masses=[0., 0., float("nan"), 0.]), 1, 1000., 10),
                (case(), 0, 1000., 10), (case(), True, 1000., 10), (case(), 1, float("nan"), 10),
                (case(), 1, 1000., 0), (case(masses=[0., 0., 600., 600.]), 1, 1000., 10),
                (case(nfinal=1, masses=[1., 1., 1.]), 1, 1000., 10)]:
            with self.subTest(item=item, count=count, energy=energy):
                with self.assertRaises(ValueError):
                    points.stream(item, count, energy, budget)

    def test_changed_cut_policy_and_proposal_exhaustion_fail(self):
        item = case()
        item["cuts"]["pt_min_gev"] = 0.
        with self.assertRaisesRegex(ValueError, "cuts differ"):
            points.stream(item, 1, 1000., 10)
        with mock.patch.object(points, "rejection", return_value="pt"):
            with self.assertRaisesRegex(ValueError, "budget exhausted"):
                points.stream(case(), 1, 1000., 2)

    def test_nonphysical_proposals_are_errors_not_rejections(self):
        with mock.patch.object(points, "proposal", return_value=[[1., 0., 0., 1.]] * 4):
            with self.assertRaisesRegex(ValueError, "Nonphysical"):
                points.stream(case(), 1, 1000., 2)


class RandomPointAuthority(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT, prefix=".test-random-authority-")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.release = {name: "synthetic-" + name for name in points.RELEASE_KEYS}
        cohort = self.directory / "cohort.json"
        self.cohort = {**self.release, "generation_root": str((self.directory / "generation").relative_to(ROOT))}
        points.dump(cohort, self.cohort)
        from ninjars_benchmark_policy_v2 import seal
        cases, core_cases, controls_cases, stress_cases, stress_proofs = {}, {}, {}, {}, {}
        for index, name in enumerate(points.CASE_ORDER):
            actual = name.replace("supplement-", "control-") if name.startswith("supplement-") else name
            kind = "controls-generation" if name.startswith("supplement-") else "stress-generation" if name.startswith("stress-") else "generation"
            status_root = self.directory / kind
            card = status_root / "inputs/candidate" / actual / "param_card.dat"
            card.parent.mkdir(parents=True)
            card.write_text("# synthetic metadata-only mass card\n")
            item = case(seed=20261021 + index, induced=name.startswith("supplement-"))
            item["definition"]["id"] = name
            definition = {**item["definition"], "id": actual}
            entry = {"status": "complete", "case": definition, "masses": item["masses"],
                     "physical_inputs": {"param_card": points.bound(card)}}
            if kind == "stress-generation":
                proof = status_root / (name + "-source.json")
                points.dump(proof, seal({"schema": "stress-generation-source-v1", "case": definition,
                    "physical_card": points.bound(card), "reducer_runtime": {"identity_sha256": self.release["runtime_identity_sha256"]}}))
                stress_proofs[name] = points.bound(proof)
                entry["source_proof"] = points.bound(proof)
                stress_cases[name] = entry
            elif kind == "controls-generation":
                controls_cases[actual] = entry
            else:
                core_cases[actual] = entry
            cases[name] = {**item, "actual_generation_case": actual, "physical_card": points.bound(card)}
        for kind, entries in [("generation", core_cases), ("controls-generation", controls_cases)]:
            points.dump(self.directory / kind / "status.json", {"complete": True,
                "git_revision": self.release["generation_revision"],
                "generator_tree_state_sha256": self.release["generator_identity_sha256"], "cases": entries})
        stress_root = self.directory / "stress-generation"
        status = stress_root / "preparation-status.json"
        points.dump(status, {"status": "complete", "cases": stress_cases})
        early = stress_root / "early-result.json"
        points.dump(early, seal({"schema": "stress-early-build-result-v1", "status": points.bound(status), "source_proofs": stress_proofs}))
        points.dump(stress_root / "final-result.json", seal({"schema": "final-pin-stress-generation-v1", "status": "complete",
            "early_build_result": points.bound(early), **{key: self.release[key] for key in
                ("generation_revision", "generator_identity_sha256", "runtime_identity_sha256", "archive_sha256")}}))
        for name, item in cases.items():
            actual = item["actual_generation_case"]
            if name.startswith("stress-"):
                item["generation"] = points.bound(stress_root / "final-result.json")
                item["origin_authority"] = {"kind": "stress-v3", "generation_case": actual, "generation": item["generation"]}
            elif name.startswith("supplement-"):
                item["generation"] = points.bound(self.directory / "controls-generation/status.json")
                item["origin_authority"] = {"kind": "controls-v3", "generation_case": actual, "generation": item["generation"],
                    "measurement": points.bound(ROOT / "docs/benchmarkfolder/policy-v3/paper-timing-software-proof-v1/specification-held-v2.json")}
            else:
                item["generation"] = points.bound(self.directory / "generation/status.json")
                item["origin_authority"] = None
        sources = {}
        for relative, role in points.SOURCE_ROLES.items():
            snapshot = self.directory / "sources" / relative
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes((ROOT / relative).read_bytes())
            sources[relative] = {"role": role, "snapshot": points.bound(snapshot)}
        self.config = {"schema": points.CONFIG_SCHEMA, "sample_kind": "independent-random-dp-measurement",
            "release": self.release, "candidate_cohort": points.bound(cohort), "points_per_case": 100,
            "seed_base": 20261021, "energy_gev": 1000., "max_proposals_per_case": 1000000,
            "boost_component_interval": [-.25, .25], "case_order": list(points.CASE_ORDER), "cases": cases,
            "sources": sources, "readiness_inputs": {}, "amplitude_evaluations": 0, "initialized_symbolica_workers": 0}
        points.validate_origins(self.config, self.cohort, self.config["readiness_inputs"])
        self.filename = self.directory / "configuration.json"
        points.dump(self.filename, self.config)

    def test_round_trip_replays_exact_points_and_collects_input_bindings(self):
        filename = points.generate(self.filename)
        inputs = {}
        manifest, config = points.validate_manifest(filename, self.release, inputs)
        self.assertEqual(len(manifest["cases"]), 12)
        self.assertEqual(config, self.config)
        self.assertGreaterEqual(len(inputs), 32)
        self.assertEqual(manifest["amplitude_evaluations"], 0)
        self.assertTrue(all(row["points_count"] == 100 for row in manifest["cases"].values()))

    def test_exact_release_identity_is_required(self):
        filename = points.generate(self.filename)
        for release in [{**self.release, "extra": "x"}, {**self.release, "archive_sha256": "wrong"}]:
            with self.assertRaises(ValueError):
                points.validate_manifest(filename, release)

    def test_seed_count_cut_source_and_alias_configuration_changes_fail(self):
        for mutate in [lambda config: config.update(points_per_case=99),
                       lambda config: config.update(energy_gev=999.),
                       lambda config: config["cases"]["qcd-1"].update(seed=1),
                       lambda config: config["cases"]["qcd-1"]["definition"].update(id="wrong"),
                       lambda config: config["sources"].pop("madgraph/various/rambo.py")]:
            config = copy.deepcopy(self.config)
            mutate(config)
            with self.assertRaises(ValueError):
                points.validate_configuration(config)

    def test_rebound_case_fields_cannot_replace_actual_generation_authorities(self):
        for name, mutate in [("qcd-2", lambda row: row["definition"].update(process="changed process")),
                ("ew-2", lambda row: row["definition"].update(model="different_model")),
                ("qcd-3", lambda row: row["definition"].update(nfinal=4)),
                ("stress-qcd-4-gg-ttgg", lambda row: row.update(masses=[0., 0., 10., 10.])),
                ("supplement-gg-h", lambda row: row.update(actual_generation_case="control-gg-hh")),
                ("supplement-gg-hh", lambda row: row["origin_authority"].update(generation_case="control-gg-h")),
                ("ew-4", lambda row: row.update(physical_card=self.config["cases"]["qcd-4"]["physical_card"]))]:
            config = copy.deepcopy(self.config)
            mutate(config["cases"][name])
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "retained CLI authority"):
                points.validate_configuration(config)

    def test_readiness_input_closure_cannot_be_omitted(self):
        config = copy.deepcopy(self.config)
        config["readiness_inputs"] = {}
        with self.assertRaisesRegex(ValueError, "closure is incomplete"):
            points.validate_configuration(config)

    def test_execution_import_closure_cannot_be_labelled_reference_only(self):
        for relative in ["tools/ninjars_benchmark.py", "tools/ninjars_benchmark_workers.py",
                         "tools/ninjars_benchmark_policy_v2.py", "tools/ninjars_benchmark_references_v2.py"]:
            config = copy.deepcopy(self.config)
            config["sources"][relative]["role"] = "proposal-reference"
            with self.subTest(relative=relative), self.assertRaisesRegex(ValueError, "source inventory differs"):
                points.validate_configuration(config)

    def test_rebound_accounting_cannot_replace_deterministic_observations(self):
        filename = points.generate(self.filename)
        manifest = json.loads(filename.read_text())
        row = manifest["cases"]["qcd-1"]
        metadata_path = points.path(row["metadata"]["path"])
        metadata = json.loads(metadata_path.read_text())
        row["proposals"] += 1
        metadata["proposals"] += 1
        points.dump(metadata_path, metadata)
        row["metadata"] = points.bound(metadata_path)
        points.dump(filename, manifest)
        with self.assertRaisesRegex(ValueError, "accounting differs"):
            points.validate_manifest(filename)

    def test_rebound_point_changes_cannot_replace_frozen_seed(self):
        filename = points.generate(self.filename)
        manifest = json.loads(filename.read_text())
        row = manifest["cases"]["qcd-1"]
        point_path = points.path(row["points"]["path"])
        point_path.write_text(point_path.read_text().replace("100\n", "99\n", 1))
        row["points"] = points.bound(point_path)
        points.dump(filename, manifest)
        with self.assertRaisesRegex(ValueError, "deterministic seed"):
            points.validate_manifest(filename)

    def test_existing_streams_and_outside_repository_paths_are_rejected(self):
        points.generate(self.filename)
        with self.assertRaisesRegex(ValueError, "already exist"):
            points.generate(self.filename)
        with self.assertRaisesRegex(ValueError, "inside the repository"):
            points.path("/tmp/unbound-random-points")


if __name__ == "__main__":
    unittest.main()
