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
"""Combining the gridpack runs of the generation pass of the cluster mode."""
from __future__ import absolute_import
import gzip
import json
import math
import os
import tempfile
import unittest

import numpy as np

from madgraph.iolibs.template_files.mg7 import combine_cluster as combine

HEADER = """<LesHouchesEvents version="3.0">
<header>
<MG7Seed>%(seed)d</MG7Seed>
</header>
<init>
2212 2212 6.5000000000e+03 6.5000000000e+03 0 0 331900 331900 3 1
%(mean).10e %(error).10e %(mean_abs).10e 1
</init>
"""
EVENT = "<event>\n 1 0 %+.10e 1.0 0.0 0.1\n 21 -1 0 0\n</event>\n"


def status(count, mean, error, mean_abs=None, **extra):
    return dict({
        "count": 10 * count, "count_after_cuts": 9 * count,
        "count_after_cuts_opt": 9 * count, "count_opt": 10 * count,
        "count_target": count, "count_unweighted": float(count),
        "done": True, "error": error, "error_abs": error, "iterations": 0,
        "iters_without_improvement": 0, "mean": mean,
        "mean_abs": mean if mean_abs is None else mean_abs, "name": "",
        "optimized": False, "rel_std_dev": 1.0, "subprocess": 0}, **extra)


def write_job(path, seed, count, mean, error, fmt, mean_abs=None):
    """The output of a gridpack run with ``count`` events of weight mean_abs."""
    os.makedirs(path)
    mean_abs = mean if mean_abs is None else mean_abs
    info = {
        "status": "done", "seed": seed, "histograms": [],
        "process": status(count, mean, error, mean_abs),
        "channels": [status(count // 2, mean / 2, error, mean_abs / 2,
                            name="0.0"),
                     status(count - count // 2, mean / 2, error, mean_abs / 2,
                            name="0.1")],
        "run_times": {"generate": {"cpu_time_sec": 1.0, "wall_time_sec": 0.5}},
        "systematics": {"event_count": count,
                        "nominal": {"cross_section": mean},
                        "scale": {"min": mean * 0.8, "max": mean * 1.2,
                                  "weight_ids": [1, 2]}},
    }
    with open(os.path.join(path, "info.json"), "w") as f:
        json.dump(info, f)
    with open(os.path.join(path, "events.weights.json"), "w") as f:
        json.dump(info["systematics"], f)
    header = HEADER % dict(seed=seed, mean=mean, error=error, mean_abs=mean_abs)
    if fmt == "lhe":
        with gzip.open(os.path.join(path, "events.lhe.gz"), "wt") as f:
            f.write(header)
            for _ in range(count):
                f.write(EVENT % mean_abs)
            f.write("</LesHouchesEvents>\n")
    else:
        events = np.zeros(count, dtype=[("weight", "f8"), ("seed", "i8")])
        events["weight"] = mean_abs
        events["seed"] = seed
        np.save(os.path.join(path, "events.npy"), events)
        with open(os.path.join(path, "header.lhe"), "w") as f:
            f.write(header + "</LesHouchesEvents>\n")


class TestCombineCluster(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def path(self, name):
        return os.path.join(self.tmp.name, name)

    def jobs(self, fmt, specs):
        paths = []
        for i, spec in enumerate(specs):
            paths.append(self.path("job%d" % i))
            write_job(paths[-1], seed=i + 1, fmt=fmt, **spec)
        return paths

    def test_cross_section_is_weighted_with_the_event_counts(self):
        info = combine.combine_runs(
            self.jobs("lhe", [dict(count=100, mean=10., error=1.),
                              dict(count=300, mean=12., error=2.)]),
            self.path("out"))
        process = info["process"]
        self.assertAlmostEqual(process["mean"], (100 * 10. + 300 * 12.) / 400)
        self.assertAlmostEqual(
            process["error"], math.sqrt(100 ** 2 * 1 + 300 ** 2 * 4) / 400)
        self.assertEqual(process["count_unweighted"], 400)
        self.assertEqual(process["count"], 4000)
        self.assertEqual(info["seed"], [1, 2])

    def test_channels_and_systematics(self):
        info = combine.combine_runs(
            self.jobs("lhe", [dict(count=100, mean=10., error=1.),
                              dict(count=300, mean=12., error=2.)]),
            self.path("out"))
        self.assertEqual([c["name"] for c in info["channels"]], ["0.0", "0.1"])
        self.assertAlmostEqual(info["channels"][0]["mean"], 5.75)
        self.assertEqual(info["channels"][0]["count_unweighted"], 50 + 150)
        self.assertAlmostEqual(info["systematics"]["nominal"]["cross_section"], 11.5)
        self.assertEqual(info["systematics"]["event_count"], 400)
        self.assertEqual(info["systematics"]["scale"]["weight_ids"], [1, 2])
        with open(self.path("out/events.weights.json")) as f:
            self.assertAlmostEqual(json.load(f)["scale"]["max"], 11.5 * 1.2)

    def test_lhe_is_concatenated_with_the_combined_init(self):
        combine.combine_runs(
            self.jobs("lhe", [dict(count=2, mean=10., error=1.),
                              dict(count=3, mean=12., error=2.)]),
            self.path("out"))
        with gzip.open(self.path("out/events.lhe.gz"), "rt") as f:
            text = f.read()
        self.assertEqual(text.count("<event>"), 5)
        self.assertEqual(text.count("<init>"), 1)
        self.assertEqual(text.count("</LesHouchesEvents>"), 1)
        # the header of the first run, the numbers of the combination
        self.assertIn("<MG7Seed>1</MG7Seed>", text)
        self.assertNotIn("<MG7Seed>2</MG7Seed>", text)
        init = text.split("<init>\n")[1].split("</init>")[0].splitlines()
        self.assertEqual(init[0].split()[8:], ["3", "1"])
        mean, error, mean_abs, process = init[1].split()
        self.assertAlmostEqual(float(mean), (2 * 10. + 3 * 12.) / 5)
        self.assertAlmostEqual(float(error), math.sqrt(4 * 1 + 9 * 4) / 5)
        self.assertEqual(process, "1")
        # the events keep their own weights
        self.assertIn("+1.0000000000e+01", text)
        self.assertIn("+1.2000000000e+01", text)

    def test_negative_weights_change_the_weight_convention(self):
        combine.combine_runs(
            self.jobs("lhe", [dict(count=2, mean=10., error=1.),
                              dict(count=3, mean=6., error=1., mean_abs=12.)]),
            self.path("out"))
        with gzip.open(self.path("out/events.lhe.gz"), "rt") as f:
            init = f.read().split("<init>\n")[1].split("</init>")[0].splitlines()
        self.assertEqual(init[0].split()[8], "-4")

    def test_npy_events_are_concatenated(self):
        for fmt in ("lhe_npy", "compact_npy"):
            out = self.path("out_" + fmt)
            specs = [dict(count=4, mean=10., error=1.),
                     dict(count=6, mean=12., error=2.)]
            paths = []
            for i, spec in enumerate(specs):
                paths.append(self.path("%s_job%d" % (fmt, i)))
                write_job(paths[-1], seed=i + 1, fmt=fmt, **spec)
            combine.combine_runs(paths, out)
            events = np.load(os.path.join(out, "events.npy"))
            self.assertEqual(list(events["seed"]), [1] * 4 + [2] * 6)
            self.assertAlmostEqual(events["weight"].mean(), (4 * 10. + 6 * 12.) / 10)
            with open(os.path.join(out, "header.lhe")) as f:
                text = f.read()
            self.assertIn("1.1200000000e+01", text)

    def test_unfinished_run(self):
        paths = self.jobs("lhe", [dict(count=2, mean=10., error=1.),
                                  dict(count=2, mean=10., error=1.)])
        with open(os.path.join(paths[1], "info.json")) as f:
            info = json.load(f)
        info["status"] = "running"
        with open(os.path.join(paths[1], "info.json"), "w") as f:
            json.dump(info, f)
        with self.assertRaisesRegex(combine.CombineError, "not finished"):
            combine.combine_runs(paths, self.path("out"))

    def test_runs_in_different_formats(self):
        paths = [self.jobs("lhe", [dict(count=2, mean=10., error=1.)])[0]]
        paths.append(self.path("npy"))
        write_job(paths[1], seed=9, fmt="lhe_npy", count=2, mean=10., error=1.)
        with self.assertRaisesRegex(combine.CombineError, "different event files"):
            combine.combine_runs(paths, self.path("out"))
