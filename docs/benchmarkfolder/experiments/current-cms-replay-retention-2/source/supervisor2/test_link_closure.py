"""Temporary-fixture tests for external link graphs and private worker copies.

No inherited packet, dependency installation, compiler, or native child is used.
"""
import copy
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("cms_link_closure_tests", HERE / "link_closure.py")
L = importlib.util.module_from_spec(SPEC)
with patch.object(subprocess, "Popen", side_effect=AssertionError("Link helper must never launch commands")):
    SPEC.loader.exec_module(L)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def put(path, text="fixture\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def link(path, target):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(target)
    return path


class Fixture:
    def __init__(self, root, candidate=True):
        self.root = root
        self.master = root / "master"
        self.worker = root / "worker"
        self.ambient = root / "unbound-ambient"
        self.roots = {name: root / "external" / name for name in
                      ("collier_include", "collier", "ninja_include", "ninja_lib", "ninjars_lib", "tirust_lib")}
        for path in self.roots.values():
            path.mkdir(parents=True)
        collier = self.roots["collier"]
        put(collier / "include/collier.mod", "COLLIER module\n")
        put(collier / "libcollier.so.1", "COLLIER library\n")
        link(collier / "libcollier.so", collier / "libcollier.so.1")
        link(self.roots["collier_include"] / "collier.mod", collier / "include/collier.mod")
        ninja_include = self.roots["ninja_include"]
        put(ninja_include / "ninja/detail/implementation.hh", "Ninja nested header\n")
        link(ninja_include / "ninja/public.hh", "detail/implementation.hh")
        link(ninja_include / "api.hh", "ninja/public.hh")
        put(self.roots["ninja_lib"] / "libninja.a", "Ninja archive\n")
        put(self.roots["ninjars_lib"] / "libninjars.a", "NinjaRS archive\n")
        put(self.roots["ninjars_lib"] / "mninjars.mod", "NinjaRS module\n")
        put(self.roots["tirust_lib"] / "libtirust.a", "TIRust archive\n")
        put(self.roots["tirust_lib"] / "mtirust.mod", "TIRust module\n")
        self.readonly = [put(root / "private-mg/vendor/CutTools/libcts.a", "CutTools archive\n"),
                         put(root / "private-mg/vendor/IREGI/libiregi.a", "IREGI archive\n")]
        self.handles = {"collier_include": self.roots["collier_include"], "collier_lib": collier,
                        "ninja_include": ninja_include, "ninja_lib": self.roots["ninja_lib"],
                        "ninjars_include": self.roots["ninjars_lib"], "ninjars_lib": self.roots["ninjars_lib"]}
        if candidate:
            self.handles.update(tirust_include=self.roots["tirust_lib"], tirust_lib=self.roots["tirust_lib"])
        for name, target in self.handles.items():
            link(self.master / "lib" / name, target)
        link(self.master / "lib/libcts.a", self.readonly[0])
        link(self.master / "lib/libiregi.a", self.readonly[1])
        put(self.master / "Cards/MadLoopParams.dat", "numerical card\n")
        put(self.master / "Cards/param_card.dat", "physical card\n")
        resources = self.master / "SubProcesses/MadLoop5_resources"
        resources.mkdir(parents=True)
        link(resources / "MadLoopParams.dat", "../MadLoopParams.dat")
        link(resources / "param_card.dat", "../../Cards/param_card.dat")
        put(resources / "HelFilter.dat", "filter\n")
        link(self.master / "SubProcesses/MadLoopParams.dat", "../Cards/MadLoopParams.dat")
        put(self.master / "SubProcesses/MadLoopParams.inc", "parameter include\n")
        put(self.master / "SubProcesses/loop_matrix.f", "loop source\n")
        process = self.master / "SubProcesses/P0"
        put(process / "check_sa.f", "worker source\n")
        put(process / "check", "inert executable bytes; never run\n")
        link(process / "loop_matrix.f", "../loop_matrix.f")
        link(process / "MadLoopParams.inc", "../MadLoopParams.inc")
        link(process / "MadLoop5_resources", resources)
        link(process / "param_card.dat", self.master / "Cards/param_card.dat")
        self.audit = {"external_roots": {str(path): L.tree_snapshot(path) for path in self.roots.values()},
                      "readonly_files": {str(path): sha(path) for path in self.readonly}}
        self.expected = L.tree_snapshot(self.master)

    def prepare(self):
        return L.prepare_worker(self.master, self.worker, self.expected, self.audit)


class LinkTests(unittest.TestCase):
    def setUp(self):
        self.block = patch.object(subprocess, "Popen", side_effect=AssertionError("No subprocess permitted"))
        self.block.start()
        self.addCleanup(self.block.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.counter = 0

    def fixture(self, candidate=True):
        self.counter += 1
        return Fixture(self.root / str(self.counter), candidate)

    def test_snapshot_records_membership_hash_types_and_complete_alias_chain(self):
        fixture = self.fixture()
        snapshot = L.tree_snapshot(fixture.roots["ninja_include"])
        self.assertEqual(snapshot["root"], str(fixture.roots["ninja_include"]))
        self.assertEqual(snapshot["root_resolved"], str(fixture.roots["ninja_include"]))
        self.assertEqual(set(snapshot["entries"]), {"api.hh", "ninja", "ninja/detail", "ninja/public.hh",
                                                   "ninja/detail/implementation.hh"})
        self.assertEqual(snapshot["entry_count"], 5)
        file_record = snapshot["entries"]["ninja/detail/implementation.hh"]
        self.assertEqual(file_record["kind"], "regular_file")
        self.assertIsInstance(file_record["mode"], str)
        self.assertEqual(file_record["sha256"], sha(fixture.roots["ninja_include"] / "ninja/detail/implementation.hh"))
        alias = snapshot["entries"]["api.hh"]
        self.assertEqual(alias["kind"], "symlink")
        self.assertEqual(alias["readlink"], "ninja/public.hh")
        self.assertEqual(alias["resolved_type"], "file")
        self.assertEqual(alias["resolved"], str(fixture.roots["ninja_include"] / "ninja/detail/implementation.hh"))
        self.assertGreaterEqual(len(alias["symlink_chain"]), 2)
        L.verify_tree(fixture.roots["ninja_include"], snapshot)
        L.verify_snapshot(snapshot)

    def test_six_unique_external_envelopes_accept_baseline_six_and_candidate_eight_handles(self):
        for candidate, count in ((False, 6), (True, 8)):
            with self.subTest(candidate=candidate):
                fixture = self.fixture(candidate)
                self.assertEqual(len(fixture.audit["external_roots"]), 6)
                self.assertEqual(len(fixture.handles), count)
                L.verify_external_graph(fixture.audit)
                context = fixture.prepare()
                for name, target in fixture.handles.items():
                    handle = fixture.worker / "lib" / name
                    self.assertTrue(handle.is_symlink())
                    self.assertEqual(handle.resolve(strict=True), target)
                    self.assertEqual(os.readlink(handle), os.readlink(fixture.master / "lib" / name))
                L.assert_worker_inputs(fixture.worker, context)

    def test_full_audit_schema_binds_observations_historical_leaves_and_transitive_aliases(self):
        fixture = self.fixture()
        leaves = {}
        observations = {}
        for path in fixture.readonly:
            checksum = sha(path)
            leaves[str(path)] = {"observed_sha256": checksum, "all_recorded_bindings_match": True,
                                 "references": [{"sha256": checksum}], "size": path.stat().st_size}
            observations[str(path)] = L.tree_snapshot(path.parent)["entries"][path.name]
        alias = fixture.roots["ninja_include"] / "api.hh"
        alias_record = copy.deepcopy(fixture.audit["external_roots"][str(fixture.roots["ninja_include"])]["entries"]["api.hh"])
        alias_record["logical_path"] = str(alias)
        audit = {"schema": "cms-replay-link-closure-audit-v1",
                 "external_directory_envelopes": copy.deepcopy(fixture.audit["external_roots"]),
                 "external_directory_alias_target_envelopes": {},
                 "historical_leaf_bindings": leaves,
                 "logical_bound_leaf_observations": observations,
                 "transitive_symlink_nodes": {str(alias): alias_record}}
        L.verify_external_graph(audit)
        fixture.audit = audit
        context = fixture.prepare()
        L.assert_worker_inputs(fixture.worker, context)
        mutations = [lambda value: value["logical_bound_leaf_observations"][str(fixture.readonly[0])].update(sha256="0" * 64),
                     lambda value: value["historical_leaf_bindings"][str(fixture.readonly[0])]["references"][0].update(sha256="0" * 64),
                     lambda value: value["transitive_symlink_nodes"][str(alias)].update(readlink="different-alias.hh")]
        for mutate in mutations:
            changed = copy.deepcopy(audit)
            mutate(changed)
            with self.assertRaises(L.LinkClosureError):
                L.verify_external_graph(changed)

    def test_cross_envelope_header_and_absolute_collier_aliases_are_bound(self):
        fixture = self.fixture()
        L.verify_external_graph(fixture.audit)
        header = fixture.roots["collier_include"] / "collier.mod"
        library = fixture.roots["collier"] / "libcollier.so"
        self.assertEqual(header.resolve(strict=True), fixture.roots["collier"] / "include/collier.mod")
        self.assertTrue(Path(os.readlink(library)).is_absolute())
        context = fixture.prepare()
        L.assert_worker_inputs(fixture.worker, context)

    def test_cross_envelope_directory_alias_is_bound_without_broad_parent_access(self):
        fixture = self.fixture()
        link(fixture.roots["ninja_include"] / "collier_headers", fixture.roots["collier"] / "include")
        fixture.audit["external_roots"][str(fixture.roots["ninja_include"])] = L.tree_snapshot(fixture.roots["ninja_include"])
        L.verify_external_graph(fixture.audit)
        context = fixture.prepare()
        L.assert_worker_inputs(fixture.worker, context)

    def test_external_membership_bytes_target_and_type_changes_are_rejected(self):
        changes = [lambda f: put(f.roots["ninja_lib"] / "new.a", "new\n"),
                   lambda f: (f.roots["ninja_lib"] / "libninja.a").unlink(),
                   lambda f: (f.roots["ninja_lib"] / "libninja.a").write_text("changed\n"),
                   lambda f: ((f.roots["collier_include"] / "collier.mod").unlink(),
                              link(f.roots["collier_include"] / "collier.mod", f.roots["ninjars_lib"] / "mninjars.mod")),
                   lambda f: ((f.roots["ninja_lib"] / "libninja.a").unlink(),
                              (f.roots["ninja_lib"] / "libninja.a").mkdir())]
        for change in changes:
            fixture = self.fixture()
            change(fixture)
            with self.assertRaises(L.LinkClosureError):
                L.verify_external_graph(fixture.audit)

    def test_same_bytes_at_another_alias_target_cannot_inherit_identity(self):
        fixture = self.fixture()
        original = fixture.roots["collier"] / "include/collier.mod"
        second = put(fixture.roots["collier"] / "include/second.mod", original.read_text())
        fixture.audit["external_roots"][str(fixture.roots["collier"])] = L.tree_snapshot(fixture.roots["collier"])
        alias = fixture.roots["collier_include"] / "collier.mod"
        alias.unlink()
        alias.symlink_to(second)
        with self.assertRaises(L.LinkClosureError):
            L.verify_external_graph(fixture.audit)

    def test_unbound_ambient_target_is_rejected_even_when_graph_snapshot_matches(self):
        for directory in (False, True):
            fixture = self.fixture()
            ambient = put(fixture.ambient / "unknown.hh", "unbound\n")
            alias = fixture.roots["ninja_include"] / "ninja/detail/ambient"
            link(alias, ambient.parent if directory else ambient)
            fixture.audit["external_roots"][str(fixture.roots["ninja_include"])] = L.tree_snapshot(fixture.roots["ninja_include"])
            with self.assertRaises(L.LinkClosureError):
                L.verify_external_graph(fixture.audit)

    def test_dangling_cycles_and_special_files_never_enter_snapshot(self):
        for kind in ("dangling", "file-cycle", "directory-cycle", "fifo"):
            with self.subTest(kind=kind):
                root = self.root / kind
                root.mkdir()
                if kind == "dangling":
                    link(root / "missing", "not-present")
                elif kind == "file-cycle":
                    link(root / "a", "b")
                    link(root / "b", "a")
                elif kind == "directory-cycle":
                    link(root / "self", ".")
                else:
                    os.mkfifo(root / "pipe")
                with self.assertRaises(L.LinkClosureError):
                    L.tree_snapshot(root)

    def test_snapshot_detects_mode_and_nested_directory_membership_changes(self):
        fixture = self.fixture()
        root = fixture.roots["ninja_include"]
        original = L.tree_snapshot(root)
        put(root / "ninja/detail/new.hh", "new header\n")
        with self.assertRaises(L.LinkClosureError):
            L.verify_tree(root, original)
        (root / "ninja/detail/new.hh").unlink()
        header = root / "ninja/detail/implementation.hh"
        header.chmod(header.stat().st_mode ^ 0o100)
        with self.assertRaises(L.LinkClosureError):
            L.verify_snapshot(original)

    def test_readonly_leaf_hashes_remain_bound_after_worker_copy(self):
        fixture = self.fixture()
        context = fixture.prepare()
        fixture.readonly[0].write_text("changed donor archive\n")
        with self.assertRaises(L.LinkClosureError):
            L.assert_worker_inputs(fixture.worker, context)

    def test_copy_rebases_absolute_master_links_before_any_configuration(self):
        fixture = self.fixture()
        master_before = L.tree_snapshot(fixture.master)
        external_before = copy.deepcopy(fixture.audit)
        context = fixture.prepare()
        process = fixture.worker / "SubProcesses/P0"
        self.assertEqual((process / "MadLoop5_resources").resolve(strict=True),
                         fixture.worker / "SubProcesses/MadLoop5_resources")
        self.assertEqual((process / "param_card.dat").resolve(strict=True), fixture.worker / "Cards/param_card.dat")
        self.assertNotIn(str(fixture.master), os.readlink(process / "param_card.dat"))
        resource_card = fixture.worker / "SubProcesses/MadLoop5_resources/MadLoopParams.dat"
        self.assertEqual(L.assert_private_path(fixture.worker, resource_card), fixture.worker / "Cards/MadLoopParams.dat")
        resource_card.write_text("configured only inside worker\n")
        self.assertEqual((fixture.master / "Cards/MadLoopParams.dat").read_text(), "numerical card\n")
        L.verify_tree(fixture.master, master_before)
        L.verify_external_graph(external_before)
        L.assert_worker_inputs(fixture.worker, context, allow_file_changes=("Cards/MadLoopParams.dat",))

    def test_every_regular_copy_has_a_distinct_inode(self):
        fixture = self.fixture()
        fixture.prepare()
        count = 0
        for name, record in fixture.expected["entries"].items():
            if record["kind"] == "regular_file":
                count += 1
                master = fixture.master / name
                worker = fixture.worker / name
                self.assertFalse(worker.is_symlink())
                self.assertNotEqual((master.stat().st_dev, master.stat().st_ino),
                                    (worker.stat().st_dev, worker.stat().st_ino))
                self.assertEqual(sha(master), sha(worker))
        self.assertGreater(count, 0)

    def test_replacing_private_copy_with_same_bytes_master_hardlink_is_denied(self):
        for name in ("Cards/MadLoopParams.dat", "SubProcesses/P0/check_sa.f"):
            fixture = self.fixture()
            context = fixture.prepare()
            worker = fixture.worker / name
            worker.unlink()
            os.link(fixture.master / name, worker)
            with self.assertRaises(L.LinkClosureError):
                L.assert_worker_inputs(fixture.worker, context, allow_file_changes=("Cards/MadLoopParams.dat",))

    def test_new_allowed_regular_output_cannot_share_any_inherited_inode(self):
        for external in (False, True):
            fixture = self.fixture()
            context = fixture.prepare()
            inherited = fixture.readonly[0] if external else fixture.master / "Cards/param_card.dat"
            output = fixture.worker / "SubProcesses/P0/timing.dat"
            original = inherited.read_bytes()
            os.link(inherited, output)
            with self.assertRaises(L.LinkClosureError):
                L.assert_private_path(fixture.worker, output)
            with self.assertRaises(L.LinkClosureError):
                L.assert_worker_inputs(fixture.worker, context, allow_new_paths=("SubProcesses/P0/timing.dat",))
            self.assertEqual(inherited.read_bytes(), original)

    def test_private_card_resource_filter_fifo_and_fortran_paths_resolve_inside_worker(self):
        fixture = self.fixture()
        fixture.prepare()
        paths = ("Cards/MadLoopParams.dat", "Cards/param_card.dat", "SubProcesses/MadLoopParams.dat",
                 "SubProcesses/MadLoop5_resources/MadLoopParams.dat", "SubProcesses/MadLoop5_resources/HelFilter.dat",
                 "SubProcesses/P0/worker-protocol.fifo", "SubProcesses/P0/check_sa.f",
                 "SubProcesses/P0/loop_matrix.f", "SubProcesses/P0/MadLoopParams.inc")
        for name in paths:
            with self.subTest(name=name):
                target = L.assert_private_path(fixture.worker, fixture.worker / name)
                self.assertTrue(target.is_relative_to(fixture.worker))

    def test_private_write_intent_rejects_external_aliases_and_lexical_escapes(self):
        fixture = self.fixture()
        fixture.prepare()
        targets = (fixture.worker / "lib/ninjars_lib/new-card.dat", fixture.readonly[0],
                   fixture.worker / "../ambient-file", fixture.master / "Cards/param_card.dat")
        for path in targets:
            with self.subTest(path=path), self.assertRaises(L.LinkClosureError):
                L.assert_private_path(fixture.worker, path)

    def test_external_card_or_fortran_link_is_refused_without_writing_through(self):
        for name in ("Cards/MadLoopParams.dat", "SubProcesses/P0/check_sa.f", "SubProcesses/P0/MadLoopParams.inc"):
            fixture = self.fixture()
            source = fixture.master / name
            source.unlink()
            source.symlink_to(fixture.readonly[0])
            fixture.expected = L.tree_snapshot(fixture.master)
            before = fixture.readonly[0].read_bytes()
            with self.assertRaises(L.LinkClosureError):
                fixture.prepare()
            self.assertEqual(fixture.readonly[0].read_bytes(), before)

    def test_external_resource_directory_is_refused_before_any_configuration(self):
        fixture = self.fixture()
        alias = fixture.master / "SubProcesses/P0/MadLoop5_resources"
        alias.unlink()
        alias.symlink_to(fixture.roots["ninjars_lib"])
        fixture.expected = L.tree_snapshot(fixture.master)
        with self.assertRaises(L.LinkClosureError):
            fixture.prepare()
        L.verify_external_graph(fixture.audit)

    def test_existing_worker_or_changed_master_prevents_copy(self):
        fixture = self.fixture()
        fixture.worker.mkdir()
        sentinel = put(fixture.worker / "sentinel", "preserve\n")
        with self.assertRaises(L.LinkClosureError):
            fixture.prepare()
        self.assertEqual(sentinel.read_text(), "preserve\n")
        fixture = self.fixture()
        put(fixture.master / "unexpected-source.f", "new\n")
        with self.assertRaises(L.LinkClosureError):
            fixture.prepare()
        self.assertFalse(fixture.worker.exists())

    def test_destination_alias_cannot_create_files_in_an_external_envelope(self):
        fixture = self.fixture()
        parent = fixture.root / "worker-parent"
        parent.symlink_to(fixture.roots["ninjars_lib"], target_is_directory=True)
        fixture.worker = parent / "worker"
        with self.assertRaises(L.LinkClosureError):
            fixture.prepare()
        self.assertFalse((fixture.roots["ninjars_lib"] / "worker").exists())
        L.verify_external_graph(fixture.audit)

    def test_destination_ancestor_alias_cannot_write_into_master_or_ambient(self):
        for master_target in (False, True):
            fixture = self.fixture()
            target = fixture.master if master_target else fixture.ambient
            target.mkdir(exist_ok=True)
            parent = fixture.root / "worker-parent"
            parent.symlink_to(target, target_is_directory=True)
            fixture.worker = parent / "worker"
            with self.assertRaises(L.LinkClosureError):
                fixture.prepare()
            self.assertFalse((target / "worker").exists())
            L.verify_tree(fixture.master, fixture.expected)

    def test_worker_changes_require_exact_explicit_file_and_path_permissions(self):
        fixture = self.fixture()
        context = fixture.prepare()
        card = fixture.worker / "Cards/MadLoopParams.dat"
        card.write_text("new card\n")
        with self.assertRaises(L.LinkClosureError):
            L.assert_worker_inputs(fixture.worker, context)
        L.assert_worker_inputs(fixture.worker, context, allow_file_changes=("Cards/MadLoopParams.dat",))
        new = put(fixture.worker / "SubProcesses/P0/timing.dat", "retained output\n")
        with self.assertRaises(L.LinkClosureError):
            L.assert_worker_inputs(fixture.worker, context, allow_file_changes=("Cards/MadLoopParams.dat",))
        L.assert_worker_inputs(fixture.worker, context, allow_file_changes=("Cards/MadLoopParams.dat",),
                               allow_new_paths=("SubProcesses/P0/timing.dat",))
        self.assertTrue(new.is_file())

    def test_allowing_content_changes_does_not_allow_link_retarget_or_file_deletion(self):
        fixture = self.fixture()
        context = fixture.prepare()
        card = fixture.worker / "Cards/MadLoopParams.dat"
        card.unlink()
        card.symlink_to(fixture.master / "Cards/MadLoopParams.dat")
        with self.assertRaises(L.LinkClosureError):
            L.assert_worker_inputs(fixture.worker, context, allow_file_changes=("Cards/MadLoopParams.dat",))
        fixture = self.fixture()
        context = fixture.prepare()
        (fixture.worker / "Cards/MadLoopParams.dat").unlink()
        with self.assertRaises(L.LinkClosureError):
            L.assert_worker_inputs(fixture.worker, context, allow_file_changes=("Cards/MadLoopParams.dat",))

    def test_allowed_new_output_names_require_the_exact_declared_node_type(self):
        for kind in ("symlink", "directory"):
            fixture = self.fixture()
            context = fixture.prepare()
            output = fixture.worker / "SubProcesses/P0/timing.dat"
            if kind == "symlink":
                output.symlink_to(fixture.master / "Cards/param_card.dat")
            else:
                output.mkdir()
            with self.assertRaises(L.LinkClosureError):
                L.assert_worker_inputs(fixture.worker, context, allow_new_paths=("SubProcesses/P0/timing.dat",))
        fixture = self.fixture()
        context = fixture.prepare()
        directory = fixture.worker / "SubProcesses/P0/worker-raw"
        directory.mkdir()
        L.assert_worker_inputs(fixture.worker, context, allow_new_paths={"SubProcesses/P0/worker-raw": "directory"})
        put(directory / "startup.json", "retained fixture output\n")
        with self.assertRaises(L.LinkClosureError):
            L.assert_worker_inputs(fixture.worker, context, allow_new_paths={"SubProcesses/P0/worker-raw": "directory"})
        L.assert_worker_inputs(fixture.worker, context,
                               allow_new_paths={"SubProcesses/P0/worker-raw": "directory",
                                                "SubProcesses/P0/worker-raw/startup.json": "regular_file"})

    def test_worker_validation_rechecks_master_and_external_envelopes(self):
        for change in (lambda f: (f.master / "SubProcesses/P0/check_sa.f").write_text("master drift\n"),
                       lambda f: put(f.roots["ninja_lib"] / "injected.a", "new external\n")):
            fixture = self.fixture()
            context = fixture.prepare()
            change(fixture)
            with self.assertRaises(L.LinkClosureError):
                L.assert_worker_inputs(fixture.worker, context)


if __name__ == "__main__":
    unittest.main()
