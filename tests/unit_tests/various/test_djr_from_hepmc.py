################################################################################
#
# Copyright (c) 2026 The MadGraph5_aMC@NLO Development team and Contributors
#
# This file is a part of the MadGraph5_aMC@NLO project, an application which
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph5_aMC@NLO license which should accompany this
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################
"""Unit tests for madgraph/various/djr_from_hepmc.py

The clustering measures are checked against configurations whose answer can be
written down by hand, so that a change to the algorithm has to be deliberate.
"""

from __future__ import absolute_import

import math
import os
import shutil
import tempfile

import numpy as np

import tests.unit_tests as unittest

import madgraph.various.djr_from_hepmc as djr


def momentum(pt, y, phi, mass=0.0):
    """A four-momentum (px, py, pz, E) from pT, rapidity and azimuth."""
    mt = math.sqrt(pt * pt + mass * mass)
    return [
        pt * math.cos(phi),
        pt * math.sin(phi),
        mt * math.sinh(y),
        mt * math.cosh(y),
    ]


HEPMC2_EVENT = """\
HepMC::Version 2.06.09
HepMC::IO_GenEvent-START_EVENT_LISTING
E 1 -1 -1.0 -1.0 -1.0 0 0 2 1 2 0 1 2.5
N 1 "0"
U GEV MM
C 1.2500000e+02 3.0000000e+00
V -1 0 0 0 0 0 0 2 0
P 10 21 30.0 0.0 0.0 30.0 0.0 1 0 0 0 0
P 11 21 -30.0 0.0 0.0 30.0 0.0 1 0 0 0 0
P 12 12 0.0 40.0 0.0 40.0 0.0 1 0 0 0 0
P 13 21 5.0 0.0 0.0 5.0 0.0 2 0 0 0 0
E 2 -1 -1.0 -1.0 -1.0 0 0 2 1 2 0 1 3.5
N 1 "0"
U GEV MM
C 1.3000000e+02 3.0000000e+00
V -1 0 0 0 0 0 0 1 0
P 20 21 10.0 0.0 0.0 10.0 0.0 1 0 0 0 0
HepMC::IO_GenEvent-END_EVENT_LISTING
"""

HEPMC3_EVENT = """\
HepMC::Version 3.02.05
HepMC::Asciiv3-START_EVENT_LISTING
E 1 2 3
U GEV MM
W 7.5
A 0 alphaQCD 0.118
P 1 0 21 30.0 0.0 0.0 30.0 0.0 1
P 2 0 21 -30.0 0.0 0.0 30.0 0.0 1
P 3 0 21 5.0 0.0 0.0 5.0 0.0 2
HepMC::Asciiv3-END_EVENT_LISTING
"""


class TestRapidity(unittest.TestCase):

    def test_matches_the_definition(self):
        p = np.array([momentum(50.0, 1.3, 0.4, mass=5.0)])
        self.assertAlmostEqual(
            float(djr.rapidity(p[:, 2], p[:, 3])[0]), 1.3, places=10
        )

    def test_beam_direction_does_not_blow_up(self):
        # E == pz exactly: the guard has to keep this finite
        value = djr.rapidity(np.array([100.0]), np.array([100.0]))
        self.assertTrue(np.isfinite(value).all())


class TestClusteringScales(unittest.TestCase):
    """The kT measures, on configurations that can be worked out by hand."""

    def test_single_particle_is_its_own_beam_distance(self):
        """One particle has nothing to cluster with, so the only step is its
        promotion to a jet at d_iB = pT^2."""
        p = np.array([momentum(37.0, 0.5, 0.0)])
        scales = djr.clustering_scales(p, radius=1.0)
        self.assertEqual(len(scales), 1)
        self.assertAlmostEqual(scales[0], 37.0, places=8)

    def test_two_well_separated_particles(self):
        """Beyond dR = R the pair distance exceeds both beam distances, so the
        two are promoted separately, softest first."""
        p = np.array([momentum(50.0, 0.0, 0.0), momentum(30.0, 2.5, 0.0)])
        scales = djr.clustering_scales(p, radius=1.0)
        self.assertEqual(len(scales), 2)
        self.assertAlmostEqual(scales[0], 30.0, places=8)
        self.assertAlmostEqual(scales[1], 50.0, places=8)

    def test_two_close_particles_merge_first(self):
        """Inside R the pair distance is min(pT^2) dR^2 / R^2, which is below
        both beam distances, so the merge comes first and the combined jet is
        promoted at its own pT."""
        delta = 0.3
        radius = 1.0
        p = np.array([momentum(50.0, 0.0, 0.0), momentum(30.0, delta, 0.0)])
        scales = djr.clustering_scales(p, radius=radius)
        self.assertEqual(len(scales), 2)
        expected_merge = math.sqrt(30.0**2 * delta**2 / radius**2)
        self.assertAlmostEqual(scales[0], expected_merge, places=8)
        # the merged jet's pT, from adding the four-momenta
        merged = np.sum(p, axis=0)
        merged_pt = math.hypot(merged[0], merged[1])
        self.assertAlmostEqual(scales[1], merged_pt, places=8)

    def test_radius_scales_the_pair_distance(self):
        """d_ij goes as 1/R^2, so halving R doubles the merge scale, while the
        beam distances are untouched."""
        p = np.array([momentum(50.0, 0.0, 0.0), momentum(30.0, 0.2, 0.0)])
        wide = djr.clustering_scales(p, radius=1.0)
        narrow = djr.clustering_scales(p, radius=0.5)
        self.assertAlmostEqual(narrow[0] / wide[0], 2.0, places=8)
        self.assertAlmostEqual(narrow[1], wide[1], places=8)

    def test_azimuthal_distance_wraps_around(self):
        """dphi is measured the short way round, so a pair straddling pi is
        close, not almost 2 pi apart."""
        p = np.array(
            [
                momentum(40.0, 0.0, math.pi - 0.1),
                momentum(40.0, 0.0, -math.pi + 0.1),
            ]
        )
        scales = djr.clustering_scales(p, radius=1.0)
        self.assertAlmostEqual(scales[0], math.sqrt(40.0**2 * 0.2**2), places=8)

    def test_scales_are_ordered(self):
        """The kT algorithm produces a monotonically increasing sequence."""
        rng = np.random.default_rng(7)
        p = np.array(
            [
                momentum(
                    float(rng.uniform(1.0, 80.0)),
                    float(rng.uniform(-3.0, 3.0)),
                    float(rng.uniform(-math.pi, math.pi)),
                )
                for _ in range(25)
            ]
        )
        scales = djr.clustering_scales(p, radius=1.0)
        self.assertEqual(len(scales), 25)
        self.assertTrue(np.all(np.diff(scales) >= -1e-9))

    def test_every_particle_is_consumed(self):
        rng = np.random.default_rng(11)
        for size in (1, 2, 5, 17):
            p = np.array(
                [
                    momentum(
                        float(rng.uniform(1.0, 80.0)),
                        float(rng.uniform(-3.0, 3.0)),
                        float(rng.uniform(-math.pi, math.pi)),
                    )
                    for _ in range(size)
                ]
            )
            self.assertEqual(len(djr.clustering_scales(p, radius=1.0)), size)

    def test_empty_event(self):
        self.assertEqual(len(djr.clustering_scales(np.zeros((0, 4)))), 0)

    def test_invariance_under_azimuthal_rotation(self):
        rng = np.random.default_rng(3)
        base = [
            (
                float(rng.uniform(1.0, 80.0)),
                float(rng.uniform(-3.0, 3.0)),
                float(rng.uniform(-math.pi, math.pi)),
            )
            for _ in range(12)
        ]
        p = np.array([momentum(*args) for args in base])
        rotated = np.array([momentum(pt, y, phi + 0.7) for pt, y, phi in base])
        np.testing.assert_allclose(
            djr.clustering_scales(p), djr.clustering_scales(rotated), rtol=1e-9
        )

    def test_invariance_under_longitudinal_boost(self):
        """Rapidity differences and pT are boost invariant, and so is every
        measure built from them."""
        rng = np.random.default_rng(5)
        base = [
            (
                float(rng.uniform(1.0, 80.0)),
                float(rng.uniform(-3.0, 3.0)),
                float(rng.uniform(-math.pi, math.pi)),
            )
            for _ in range(12)
        ]
        p = np.array([momentum(*args) for args in base])
        boosted = np.array([momentum(pt, y + 0.8, phi) for pt, y, phi in base])
        np.testing.assert_allclose(
            djr.clustering_scales(p), djr.clustering_scales(boosted), rtol=1e-9
        )


def naive_clustering_scales(momenta, radius):
    """A deliberately dumb reference: recompute every distance from scratch at
    every step, with plain Python loops and no bookkeeping to get wrong.

    The implementation under test keeps a distance matrix and refreshes only
    the row of the merged cluster, which is where an indexing mistake would
    hide. Comparing against this catches that.
    """
    clusters = [list(p) for p in momenta]
    scales = []
    while clusters:
        def kinematics(p):
            pt2 = p[0] ** 2 + p[1] ** 2
            y = 0.5 * math.log(
                max(p[3] + p[2], 1e-12) / max(p[3] - p[2], 1e-12)
            )
            return pt2, y, math.atan2(p[1], p[0])

        info = [kinematics(p) for p in clusters]
        best, best_pair = None, None
        for i, (pt2, _, _) in enumerate(info):
            if best is None or pt2 < best:
                best, best_pair = pt2, (i, None)
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                pt2_i, y_i, phi_i = info[i]
                pt2_j, y_j, phi_j = info[j]
                dphi = abs(phi_i - phi_j)
                dphi = min(dphi, 2.0 * math.pi - dphi)
                d = min(pt2_i, pt2_j) * ((y_i - y_j) ** 2 + dphi**2) / radius**2
                if d < best:
                    best, best_pair = d, (i, j)
        scales.append(best)
        i, j = best_pair
        if j is None:
            clusters.pop(i)
        else:
            clusters[i] = [clusters[i][k] + clusters[j][k] for k in range(4)]
            clusters.pop(j)
    return np.sqrt(np.array(scales))


class TestAgainstNaiveImplementation(unittest.TestCase):

    def test_random_events_agree(self):
        rng = np.random.default_rng(2024)
        for size in (1, 2, 3, 8, 20):
            for _ in range(5):
                p = np.array(
                    [
                        momentum(
                            float(rng.uniform(0.5, 90.0)),
                            float(rng.uniform(-4.0, 4.0)),
                            float(rng.uniform(-math.pi, math.pi)),
                        )
                        for _ in range(size)
                    ]
                )
                np.testing.assert_allclose(
                    djr.clustering_scales(p, radius=1.0),
                    naive_clustering_scales(p, radius=1.0),
                    rtol=1e-9,
                    atol=1e-9,
                    err_msg="size %d" % size,
                )

    def test_collinear_particles_agree(self):
        """Nearly overlapping particles make several distances almost equal,
        which is where a stale entry in the matrix would show up."""
        p = np.array(
            [
                momentum(30.0, 0.0, 0.0),
                momentum(30.0, 1e-6, 1e-6),
                momentum(30.0, 2e-6, 0.0),
                momentum(50.0, 2.0, 1.0),
            ]
        )
        np.testing.assert_allclose(
            djr.clustering_scales(p, radius=1.0),
            naive_clustering_scales(p, radius=1.0),
            rtol=1e-9,
            atol=1e-9,
        )

    def test_narrow_radius_agrees(self):
        """A small R pushes the algorithm towards beam promotions rather than
        merges, exercising the other branch."""
        rng = np.random.default_rng(99)
        p = np.array(
            [
                momentum(
                    float(rng.uniform(0.5, 90.0)),
                    float(rng.uniform(-4.0, 4.0)),
                    float(rng.uniform(-math.pi, math.pi)),
                )
                for _ in range(15)
            ]
        )
        np.testing.assert_allclose(
            djr.clustering_scales(p, radius=0.4),
            naive_clustering_scales(p, radius=0.4),
            rtol=1e-9,
            atol=1e-9,
        )


class TestDifferentialJetRates(unittest.TestCase):

    def test_largest_scale_comes_first(self):
        """d01 is the hardest step, i.e. the pT of the leading jet."""
        p = np.array(
            [
                momentum(80.0, 0.0, 0.0),
                momentum(40.0, 2.5, 0.0),
                momentum(20.0, -2.5, 1.0),
            ]
        )
        rates = djr.differential_jet_rates(p, radius=1.0, count=4)
        self.assertAlmostEqual(rates[0], 80.0, places=8)
        self.assertAlmostEqual(rates[1], 40.0, places=8)
        self.assertAlmostEqual(rates[2], 20.0, places=8)

    def test_missing_rates_are_nan_not_zero(self):
        """An event with fewer steps than requested must not contribute an
        entry at zero, which would pile up in the lowest bin."""
        p = np.array([momentum(37.0, 0.5, 0.0)])
        rates = djr.differential_jet_rates(p, radius=1.0, count=4)
        self.assertAlmostEqual(rates[0], 37.0, places=8)
        self.assertTrue(np.all(np.isnan(rates[1:])))


class TestVisibleFinalState(unittest.TestCase):

    def test_neutrinos_are_dropped(self):
        pdgs = np.array([21, 12, -14, 11, 16])
        p = np.array([momentum(10.0 * (i + 1), 0.0, 0.0) for i in range(5)])
        kept = djr.visible_final_state(pdgs, p, eta_max=1000.0)
        self.assertEqual(len(kept), 2)
        np.testing.assert_allclose(kept[0], p[0])
        np.testing.assert_allclose(kept[1], p[3])

    def test_eta_cut_is_off_by_default(self):
        pdgs = np.array([21, 21])
        p = np.array([momentum(10.0, 0.0, 0.0), momentum(10.0, 8.0, 0.0)])
        self.assertEqual(len(djr.visible_final_state(pdgs, p)), 2)

    def test_eta_cut_applies_when_set(self):
        pdgs = np.array([21, 21])
        p = np.array([momentum(10.0, 0.0, 0.0), momentum(10.0, 4.0, 0.0)])
        kept = djr.visible_final_state(pdgs, p, eta_max=2.5)
        self.assertEqual(len(kept), 1)


class TestReadHepMC(unittest.TestCase):

    def setUp(self):
        self.directory = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def write(self, name, content):
        path = os.path.join(self.directory, name)
        with open(path, "w") as handle:
            handle.write(content)
        return path

    def test_hepmc2_final_state_only(self):
        path = self.write("events.hepmc", HEPMC2_EVENT)
        events = list(djr.read_hepmc(path))
        self.assertEqual(len(events), 2)

        weight, pdgs, momenta = events[0]
        self.assertAlmostEqual(weight, 2.5)
        # the status-2 particle is not part of the final state
        self.assertEqual(list(pdgs), [21, 21, 12])
        self.assertEqual(momenta.shape, (3, 4))
        np.testing.assert_allclose(momenta[0], [30.0, 0.0, 0.0, 30.0])

        weight, pdgs, momenta = events[1]
        self.assertAlmostEqual(weight, 3.5)
        self.assertEqual(list(pdgs), [21])

    def test_hepmc3_final_state_only(self):
        path = self.write("events.hepmc3", HEPMC3_EVENT)
        events = list(djr.read_hepmc(path))
        self.assertEqual(len(events), 1)
        weight, pdgs, momenta = events[0]
        self.assertAlmostEqual(weight, 7.5)
        self.assertEqual(list(pdgs), [21, 21])
        np.testing.assert_allclose(momenta[1], [-30.0, 0.0, 0.0, 30.0])

    def test_max_events(self):
        path = self.write("events.hepmc", HEPMC2_EVENT)
        self.assertEqual(len(list(djr.read_hepmc(path, max_events=1))), 1)

    def test_cross_section_is_the_last_one(self):
        path = self.write("events.hepmc", HEPMC2_EVENT)
        self.assertAlmostEqual(djr.read_cross_section(path), 130.0)

    def test_cross_section_absent(self):
        path = self.write("events.hepmc3", HEPMC3_EVENT)
        self.assertIsNone(djr.read_cross_section(path))


class TestHistograms(unittest.TestCase):

    def test_normalisation_reproduces_the_cross_section(self):
        """Integrating dsigma/dlog10(DJR) over the bins has to give back the
        cross section, otherwise the samples cannot be added."""
        histograms = djr.DJRHistograms("0j", count=1, bins=30, low=0.0, high=3.0)
        rng = np.random.default_rng(1)
        for _ in range(500):
            value = 10.0 ** rng.uniform(0.2, 2.8)
            histograms.fill(np.array([value]), weight=2.0)
        histograms.normalise(cross_section=42.0)
        width = histograms.edges[1] - histograms.edges[0]
        self.assertAlmostEqual(histograms.values[0].sum() * width, 42.0, places=8)

    def test_unit_area_without_a_cross_section(self):
        histograms = djr.DJRHistograms("0j", count=1, bins=20, low=0.0, high=3.0)
        for value in (10.0, 20.0, 100.0):
            histograms.fill(np.array([value]), weight=1.0)
        histograms.normalise(cross_section=None)
        width = histograms.edges[1] - histograms.edges[0]
        self.assertAlmostEqual(histograms.values[0].sum() * width, 1.0, places=8)

    def test_nan_rates_are_not_filled(self):
        histograms = djr.DJRHistograms("0j", count=2, bins=20, low=0.0, high=3.0)
        histograms.fill(np.array([50.0, np.nan]), weight=1.0)
        self.assertAlmostEqual(histograms.values[0].sum(), 1.0)
        self.assertAlmostEqual(histograms.values[1].sum(), 0.0)

    def test_out_of_range_values_are_dropped(self):
        histograms = djr.DJRHistograms("0j", count=1, bins=20, low=1.0, high=2.0)
        histograms.fill(np.array([1.0]), weight=1.0)      # log10 = 0, below
        histograms.fill(np.array([10000.0]), weight=1.0)  # log10 = 4, above
        self.assertAlmostEqual(histograms.values[0].sum(), 0.0)
        # but they still count towards the normalisation, as they must: the
        # cross section covers the whole sample, not just the plotted window
        self.assertEqual(histograms.event_count, 2)


class TestSampleSpec(unittest.TestCase):

    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "events.hepmc")
        with open(self.path, "w") as handle:
            handle.write(HEPMC2_EVENT)

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_label_and_path(self):
        sample = djr.Sample("0j:%s" % self.path)
        self.assertEqual(sample.label, "0j")
        self.assertEqual(sample.path, self.path)
        self.assertIsNone(sample.cross_section)

    def test_cross_section(self):
        sample = djr.Sample("1j:%s:12.5" % self.path)
        self.assertAlmostEqual(sample.cross_section, 12.5)

    def test_missing_file_is_rejected(self):
        self.assertRaises(ValueError, djr.Sample, "0j:/no/such/file.hepmc")

    def test_malformed_spec_is_rejected(self):
        self.assertRaises(ValueError, djr.Sample, "justalabel")


class TestEndToEnd(unittest.TestCase):
    """Run main() on two synthetic samples and check the table it writes."""

    def setUp(self):
        self.directory = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def write_sample(self, name, leading_pt, count):
        """A file of `count` events, each a single jet of the given pT, so the
        DJR is known exactly."""
        lines = [
            "HepMC::Version 2.06.09",
            "HepMC::IO_GenEvent-START_EVENT_LISTING",
        ]
        for index in range(count):
            lines.append(
                "E %d -1 -1.0 -1.0 -1.0 0 0 1 1 2 0 1 1.0" % index
            )
            lines.append("U GEV MM")
            lines.append("V -1 0 0 0 0 0 0 1 0")
            lines.append(
                "P %d 21 %.6f 0.0 0.0 %.6f 0.0 1 0 0 0 0"
                % (10 + index, leading_pt, leading_pt)
            )
        lines.append("HepMC::IO_GenEvent-END_EVENT_LISTING")
        path = os.path.join(self.directory, name)
        with open(path, "w") as handle:
            handle.write("\n".join(lines) + "\n")
        return path

    def test_two_samples_land_in_the_expected_bins(self):
        soft = self.write_sample("soft.hepmc", 10.0, 20)
        hard = self.write_sample("hard.hepmc", 100.0, 20)
        prefix = os.path.join(self.directory, "out")
        code = djr.main(
            [
                "--sample", "0j:%s:10.0" % soft,
                "--sample", "1j:%s:1.0" % hard,
                "--qcut", "30",
                "--output-prefix", prefix,
                "--no-plot",
                "--n-djr", "1",
                "--bins", "30",
            ]
        )
        self.assertEqual(code, 0)

        table = prefix + "_djr.dat"
        self.assertTrue(os.path.exists(table))
        rows = [
            [float(value) for value in line.split()]
            for line in open(table)
            if line.strip() and not line.startswith("#")
        ]
        self.assertTrue(rows)

        width = rows[1][0] - rows[0][0]
        # column 1 is the 0j sample, column 2 the 1j one, column 3 their sum
        soft_integral = sum(row[1] for row in rows) * width
        hard_integral = sum(row[2] for row in rows) * width
        self.assertAlmostEqual(soft_integral, 10.0, places=6)
        self.assertAlmostEqual(hard_integral, 1.0, places=6)

        # each sample sits entirely on its own side of qcut, which is the
        # start/stop the plot is meant to show
        for row in rows:
            if row[1] > 0.0:
                self.assertLess(row[0], math.log10(30.0))
            if row[2] > 0.0:
                self.assertGreater(row[0], math.log10(30.0))


if __name__ == "__main__":
    unittest.unittest.main()
