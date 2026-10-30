################################################################################
#
# Copyright (c) 2026 The MadGraph7 Development team and Contributors
#
# This file is a part of the MadGraph7 project, an application which
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph7 license which should accompany this
# distribution.
#
################################################################################
"""Merging the gridpacks of different subprocess selections."""
from __future__ import absolute_import
import io
import json
import os
import tarfile
import tempfile
import unittest

from madgraph.iolibs.template_files.mg7 import merge_gridpacks as merge

RUN_CARD = """[run]
seed = %(seed)s
device = ["%(device)s"]

[phasespace]
merge_subprocesses = false
%(selection)s
[gridpack]
run_mode = "%(mode)s"
"""


def write_globals(path, globals_):
    with tarfile.open(path, "w") as archive:
        for name, content in globals_.items():
            info = tarfile.TarInfo(name + ".npy")
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))


def write_data(path, subprocess_ids, seed=1, device="cpu", fix_max_weight=True,
               source_hash="abc", shared=b"pdf", extra_channels=()):
    """The data/ of a run over the subprocesses ``subprocess_ids`` of a process
    with four."""
    os.makedirs(path)
    channels = {}
    for i in subprocess_ids:
        for k in range(2):
            channel = {"subprocess_index": i, "function": "f%d.%d" % (i, k)}
            if fix_max_weight:
                channel["max_weight"] = 1.5
            channels["%d.%d" % (i, k)] = channel
    for name in extra_channels:
        channels[name] = {"subprocess_index": 0}
    with open(os.path.join(path, "channels.json"), "w") as f:
        # pretty-printed, to make the text scanning earn its keep
        f.write(json.dumps(channels, indent=1))
    data = {
        "matrix_elements": ["lib/me%d.so" % i for i in range(4)],
        "matrix_element_indices": list(subprocess_ids),
        "custom_functions": [], "me_parameters": {}, "source_hash": source_hash,
    }
    with open(os.path.join(path, "data.json"), "w") as f:
        json.dump(data, f)
    selection = "subprocess_selection = %s\n" % list(subprocess_ids)
    card = RUN_CARD % dict(seed=seed, device=device, selection=selection,
                           mode="fix_max_weight" if fix_max_weight else "minimal")
    meta = {"beam1_energy": 6500.0, "headers": [
        {"name": "MG5ProcCard", "content": "generate p p > t t~ j",
         "escape_content": True},
        {"name": "MG7RunCard", "content": card, "escape_content": True}]}
    with open(os.path.join(path, "lhe_meta.json"), "w") as f:
        json.dump(meta, f)
    for name in ("lhe.json", "systematics.json"):
        with open(os.path.join(path, name), "w") as f:
            f.write('{"same": true}')
    own = {"subproc%d.channel0.grid" % i: b"vegas%d" % i for i in subprocess_ids}
    write_globals(os.path.join(path, "globals.tar"), dict(own, pdf=shared))


class TestMergeGridpacks(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def path(self, name):
        return os.path.join(self.tmp.name, name)

    def merged(self, *datas):
        paths = []
        for i, kwargs in enumerate(datas):
            paths.append(self.path("in%d" % i))
            write_data(paths[-1], **kwargs)
        merge.merge_gridpack_data(paths, self.path("out"))
        return self.path("out")

    def test_union_of_channels_and_matrix_elements(self):
        out = self.merged(dict(subprocess_ids=[1, 3], seed=1),
                          dict(subprocess_ids=[0, 2], seed=2, device="cuda"))
        with open(os.path.join(out, "channels.json")) as f:
            channels = json.load(f)
        self.assertEqual(sorted(channels), sorted(
            "%d.%d" % (i, k) for i in range(4) for k in range(2)))
        self.assertEqual(channels["3.1"]["function"], "f3.1")
        with open(os.path.join(out, "data.json")) as f:
            data = json.load(f)
        self.assertEqual(data["matrix_element_indices"], [0, 1, 2, 3])
        self.assertEqual(len(data["matrix_elements"]), 4)

    def test_channel_text_is_kept(self):
        """The channels are not written out again, but copied as they are."""
        out = self.merged(dict(subprocess_ids=[0]), dict(subprocess_ids=[1]))
        with open(os.path.join(out, "channels.json")) as f:
            text = f.read()
        self.assertIn('"1.0":{\n  "subprocess_index": 1,\n  "function": "f1.0"', text)

    def test_globals_are_united_without_duplicates(self):
        out = self.merged(dict(subprocess_ids=[0]), dict(subprocess_ids=[1]))
        with tarfile.open(os.path.join(out, "globals.tar")) as archive:
            names = archive.getnames()
            self.assertEqual(sorted(names), sorted([
                "pdf.npy", "subproc0.channel0.grid.npy", "subproc1.channel0.grid.npy"]))
            self.assertEqual(archive.extractfile("subproc1.channel0.grid.npy").read(),
                             b"vegas1")

    def test_run_card_in_lhe_meta_drops_the_selection(self):
        out = self.merged(dict(subprocess_ids=[0], seed=1),
                          dict(subprocess_ids=[1], seed=2, device="cuda"))
        with open(os.path.join(out, "lhe_meta.json")) as f:
            meta = json.load(f)
        self.assertEqual([h["name"] for h in meta["headers"]],
                         ["MG5ProcCard", "MG7RunCard"])
        self.assertNotIn("subprocess_selection", meta["headers"][1]["content"])

    def test_overlapping_selections(self):
        with self.assertRaisesRegex(merge.MergeError, "overlap"):
            self.merged(dict(subprocess_ids=[0, 1]), dict(subprocess_ids=[1]))

    def test_different_madspace(self):
        with self.assertRaisesRegex(merge.MergeError, "source_hash"):
            self.merged(dict(subprocess_ids=[0]),
                        dict(subprocess_ids=[1], source_hash="other"))

    def test_mixed_run_modes(self):
        with self.assertRaisesRegex(merge.MergeError, "max weights"):
            self.merged(dict(subprocess_ids=[0]),
                        dict(subprocess_ids=[1], fix_max_weight=False))

    def test_different_shared_global(self):
        with self.assertRaisesRegex(merge.MergeError, "pdf"):
            self.merged(dict(subprocess_ids=[0]),
                        dict(subprocess_ids=[1], shared=b"another pdf"))

    def test_different_process_description(self):
        paths = [self.path("a"), self.path("b")]
        write_data(paths[0], [0])
        write_data(paths[1], [1])
        with open(os.path.join(paths[1], "lhe.json"), "w") as f:
            f.write('{"same": false}')
        with self.assertRaisesRegex(merge.MergeError, "lhe.json"):
            merge.merge_gridpack_data(paths, self.path("out"))

    def test_different_card_outside_the_selection(self):
        paths = [self.path("a"), self.path("b")]
        write_data(paths[0], [0])
        write_data(paths[1], [1])
        meta_path = os.path.join(paths[1], "lhe_meta.json")
        with open(meta_path) as f:
            meta = json.load(f)
        meta["headers"][1]["content"] += "\n[beam]\nebeam1 = 1.0\n"
        with open(meta_path, "w") as f:
            json.dump(meta, f)
        with self.assertRaisesRegex(merge.MergeError, "run card"):
            merge.merge_gridpack_data(paths, self.path("out"))

    def test_gridpack_without_recorded_indices_holds_every_matrix_element(self):
        paths = [self.path("a"), self.path("b")]
        write_data(paths[0], [0])
        write_data(paths[1], [1])
        data_path = os.path.join(paths[0], "data.json")
        with open(data_path) as f:
            data = json.load(f)
        del data["matrix_element_indices"]
        with open(data_path, "w") as f:
            json.dump(data, f)
        merge.merge_gridpack_data(paths, self.path("out"))
        with open(os.path.join(self.path("out"), "data.json")) as f:
            self.assertEqual(json.load(f)["matrix_element_indices"], [0, 1, 2, 3])
