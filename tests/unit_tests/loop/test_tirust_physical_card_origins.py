"""Physical inputs can be inherited explicitly; measured results cannot."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import tirust_benchmark_local as local
import tirust_benchmark_campaign as campaign


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n" if not isinstance(value, str) else value)


class PhysicalCardOrigins(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.base = self.root / "docs/benchmarkfolder"
        self.prior = self.base / "campaigns/prior"
        self.next = self.base / "campaigns/next"
        for name, value in (("ROOT", self.root), ("LEGACY_REPORT", self.base),
                            ("LEGACY_WORK", self.root / ".benchmark-work/tirust")):
            context = patch.object(local.bench, name, value)
            context.start()
            self.addCleanup(context.stop)
        self.config = {"cases": [{"id": "one", "model": "loop_sm", "process": "u u~ > z [virt=QCD]"},
                                  {"id": "two", "model": campaign.MODEL, "process": "e+ e- > mu+ mu- [virt=QED]"}],
                       "points_per_case": 100, "repetitions": 7}
        write(self.prior / "cases.json", self.config)
        for case in self.config["cases"]:
            write(self.prior / ("points/" + case["id"] + ".dat"), "100\nfrozen points\n")
            write(self.prior / ("points/" + case["id"] + ".json"), {"count": 100})
        write(self.prior / ("inputs/" + campaign.MODEL + ".json"), {"files": {"model.py": "frozen"}})
        self.shared = {"files": {p.relative_to(self.prior).as_posix(): local.bench.digest(p)
                                  for p in self.prior.rglob("*") if p.is_file()},
                       "policy_sha256": local.bench.content_digest(self.config),
                       "model_tree": {"model.py": "frozen"}}
        self.prior_manifest = {"schema": campaign.SCHEMA, "id": "prior", "shared_inputs": self.shared}
        write(self.prior / "manifest.json", self.prior_manifest)
        self.previous = {"id": "prior", "report": "campaigns/prior",
                         "manifest_sha256": local.bench.digest(self.prior / "manifest.json")}
        binding = {"id": "prior", "manifest_sha256": self.previous["manifest_sha256"]}
        for case in self.config["cases"]:
            for source in ("baseline", "candidate"):
                card = self.prior / "inputs" / source / case["id"] / "param_card.dat"
                write(card, "physical card " + case["id"] + "\n")
                record = {"campaign_binding": binding, "param_card_sha256": local.bench.digest(card),
                          "points_sha256": self.shared["files"]["points/" + case["id"] + ".dat"],
                          "points_metadata_sha256": self.shared["files"]["points/" + case["id"] + ".json"],
                          "generation": {"campaign_binding": binding, "source": source, "case": case,
                                         "parameter_card_sha256": local.bench.digest(card)}}
                write(self.prior / "experiments/numerical" / case["id"] / source / (source + "-inputs.json"), record)
        self.manifest = {"schema": campaign.SCHEMA, "id": "next", "predecessor": self.previous,
                         "shared_inputs": self.shared}
        self.manifest["physical_card_origins"] = self.collect()

    def collect(self):
        return campaign._physical_card_origins(local.bench, self.previous, self.shared)

    def verify(self):
        return campaign.verify_physical_card_origins(local.bench, self.manifest)

    def test_complete_explicit_ancestry_contains_only_input_evidence(self):
        value = self.verify()
        self.assertEqual(set(value["cards"]), {"one", "two"})
        self.assertEqual(len(value["observations"]), 4)
        self.assertEqual(value["scope"], "physical-input-ancestry-only")
        self.assertFalse(any("result" in name for name in value["observations"]))
        self.assertFalse((self.next / "experiments").exists())

    def test_old_manifest_without_declaration_is_unchanged(self):
        del self.manifest["physical_card_origins"]
        self.assertIsNone(self.verify())

    def test_mismatched_policy_model_or_points_rejected(self):
        for key in self.shared:
            with self.subTest(key=key):
                changed = copy.deepcopy(self.shared)
                changed[key] = "changed"
                with self.assertRaisesRegex(ValueError, "model, points, or policy"):
                    campaign._physical_card_origins(local.bench, self.previous, changed)

    def test_all_frozen_cases_required(self):
        paths = self.manifest["physical_card_origins"]["observations"]
        chosen = {name: digest for name, digest in paths.items() if "/one/" in name}
        with self.assertRaisesRegex(ValueError, "every frozen case"):
            campaign._physical_card_origins(local.bench, self.previous, self.shared, chosen)

    def test_declared_hash_and_card_bytes_are_rechecked(self):
        path = next(iter(self.manifest["physical_card_origins"]["observations"]))
        self.manifest["physical_card_origins"]["observations"][path] = "changed"
        with self.assertRaisesRegex(ValueError, "observation changed"):
            self.verify()
        self.manifest["physical_card_origins"] = self.collect()
        write(self.prior / "inputs/candidate/one/param_card.dat", "changed")
        with self.assertRaisesRegex(ValueError, "card bytes"):
            self.verify()

    def test_case_ancestry_point_and_generation_hash_rejected(self):
        path = next((self.prior / "experiments/numerical").glob("*/*/*-inputs.json"))
        original = json.loads(path.read_text())
        edits = [lambda v: v["generation"]["case"].update(model="wrong"),
                 lambda v: v["campaign_binding"].update(id="unrelated"),
                 lambda v: v.update(points_sha256="wrong"),
                 lambda v: v["generation"].update(parameter_card_sha256="wrong")]
        for edit in edits:
            with self.subTest(edit=edit):
                value = copy.deepcopy(original)
                edit(value)
                write(path, value)
                with self.assertRaises(ValueError):
                    self.collect()
                write(path, original)

    def test_declared_ancestry_cannot_switch_to_unrelated_campaign(self):
        self.manifest["physical_card_origins"]["origin"]["id"] = "unrelated"
        with self.assertRaisesRegex(ValueError, "origins changed"):
            self.verify()

    def test_undeclared_later_observation_is_not_imported(self):
        write(self.prior / "experiments/numerical/later/candidate/candidate-inputs.json", {"bad": "not declared"})
        self.assertEqual(len(self.verify()["observations"]), 4)

    def test_paths_reject_escape_symlink_and_nonobservation(self):
        paths = ("../outside", str(self.prior / "manifest.json"), "campaigns/prior/manifest.json")
        for name in paths:
            with self.subTest(name=name), self.assertRaises(ValueError):
                campaign._physical_card_origins(local.bench, self.previous, self.shared, {name: "bad"})
        path = next((self.prior / "experiments/numerical").glob("*/*/*-inputs.json"))
        saved = path.with_suffix(".saved")
        path.rename(saved)
        path.symlink_to(saved.name)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.verify()

    def test_local_import_has_no_implicit_predecessor_fallback(self):
        write(self.next / "manifest.json", self.manifest)
        # Ancestry/failure collection is independently exercised by existing
        # campaign tests. No measurements are fabricated for this new campaign.
        with patch.object(campaign, "_predecessor", return_value={}), \
             patch.object(campaign, "_verify_predecessor", return_value=[]):
            value = local.origin_inventory("next")
            self.assertEqual(set(value["physical_cards"]), {"one", "two"})
            self.assertEqual(len(value["files"]), 8)
            del self.manifest["physical_card_origins"]
            write(self.next / "manifest.json", self.manifest)
            self.assertEqual(local.origin_inventory("next")["physical_cards"], {})

    def test_own_observation_conflict_cannot_override_declared_card(self):
        write(self.next / "manifest.json", self.manifest)
        write(self.next / "experiments/numerical/own/candidate/candidate-inputs.json",
              {"generation": {"case": {"id": "one"}}, "param_card_sha256": "different"})
        with patch.object(campaign, "_predecessor", return_value={}), \
             patch.object(campaign, "_verify_predecessor", return_value=[]), \
             self.assertRaisesRegex(ValueError, "Observed and declared"):
            local.origin_inventory("next")

    def test_registration_seals_origins_without_copying_measurements(self):
        from tests.unit_tests.loop import test_tirust_benchmark_campaign as fixtures
        fixture = fixtures.TestCampaign("runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        prior = fixture.register("observed")
        report = fixture.bench.LEGACY_REPORT / "campaigns/observed"
        write(report / "summary.json", {"pending": True})
        card = report / "inputs/candidate/example/param_card.dat"
        write(card, "physical input\n")
        binding = {"id": "observed", "manifest_sha256": fixture.bench.digest(report / "manifest.json")}
        write(report / "experiments/numerical/ordinary/candidate/candidate-inputs.json",
              {"campaign_binding": binding, "param_card_sha256": fixture.bench.digest(card),
               "generation": {"case": {"id": "example"}, "source": "candidate", "campaign_binding": binding,
                              "parameter_card_sha256": fixture.bench.digest(card)},
               "points_sha256": prior["shared_inputs"]["files"]["points/example.dat"],
               "points_metadata_sha256": prior["shared_inputs"]["files"]["points/example.json"]})
        fixture._install("changed native bytes\n")
        manifest = fixture.register("successor", predecessor="observed", inherit_physical_cards=True)
        self.assertEqual(set(manifest["physical_card_origins"]["cards"]), {"example"})
        target = fixture.bench.LEGACY_REPORT / "campaigns/successor"
        self.assertFalse((target / "experiments").exists())
        self.assertFalse((target / "runs").exists())
        campaign.activate_campaign(fixture.bench, "successor")
        campaign.verify_campaign(fixture.bench)
        write(card, "mutated after registration\n")
        with self.assertRaisesRegex(ValueError, "card bytes"):
            campaign.verify_campaign(fixture.bench)


if __name__ == "__main__":
    unittest.main()
