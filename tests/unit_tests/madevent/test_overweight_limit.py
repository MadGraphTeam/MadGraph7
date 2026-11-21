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
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################
"""Tests of enforce_overweight_limit: the per-channel overweight budget
(combine_runs.overweight_max_weight / pool_weights / write_pool) and the top-up refine jobs
(gen_ximprove_v4.enforce_overweight_limit)."""
from __future__ import absolute_import
import os
import random
import shutil
import tempfile
import unittest

import madgraph.madevent.combine_runs as combine_runs
import madgraph.madevent.gen_ximprove as gen_ximprove
import madgraph.madevent.sum_html as sum_html


def write_events(path, weights):
    """a channel/job events.lhe with these weights (2-particle events)"""
    with open(path, 'w') as fsock:
        for w in weights:
            fsock.write("<event>\n"
                        " 2 1 %s 0.9E+02 0.75E-02 0.12E+00\n"
                        "        11   -1    0    0    0    0  0.0E+00  0.0E+00"
                        "  0.1E+03  0.1E+03  0.0E+00 0. -1.\n"
                        "       -11   -1    0    0    0    0  0.0E+00  0.0E+00"
                        " -0.1E+03  0.1E+03  0.0E+00 0.  1.\n"
                        "</event>\n" % combine_runs.get_fortran_str(w))


def write_results(path, axsec, nunwgt, nevents=1000, err=None):
    r = sum_html.OneResult('G1')
    r.axsec = r.xsec = axsec
    r.xerru = err if err is not None else 0.01 * axsec
    r.nevents, r.nunwgt = nevents, nunwgt
    r.luminosity = nunwgt / axsec
    r.wgt = 0
    sum_html.Combine_results.write_results_dat(r, path)


class TestOverweightMaxWeight(unittest.TestCase):

    def test_no_overweight_within_budget(self):
        # 1000 unit events and one of 3 units: 3 > 0.001*1000*1, so the maximum
        # has to rise until 3 <= 0.001*1000*m, i.e. m = 3 (the event is no longer above)
        w = [1.] * 1000 + [3.]
        self.assertAlmostEqual(combine_runs.overweight_max_weight(w, 1000, 0.001, 1.), 3.)
        # a budget of 0.01*1000 = 10 units keeps it as an overweight event at m = 1
        self.assertAlmostEqual(combine_runs.overweight_max_weight(w, 1000, 0.01, 1.), 1.)

    def test_budget_between_weights(self):
        # two events of 10 units: at m in [1, 10) both are above (20 units of
        # weight); the budget trunc*n*m = 0.002*1000*m reaches 20 at m = 10
        w = [1.] * 1000 + [10., 10.]
        self.assertAlmostEqual(combine_runs.overweight_max_weight(w, 1000, 0.002, 1.), 10.)
        # with one event of 10 and one of 4: m = 5 keeps the 10 above (10 <= 2*5)
        w = [1.] * 1000 + [10., 4.]
        self.assertAlmostEqual(combine_runs.overweight_max_weight(w, 1000, 0.002, 1.), 5.)

    def test_definition_and_minimality(self):
        rng = random.Random(3)
        for _ in range(50):
            w = [1.] * rng.randint(50, 500) + [1 + rng.paretovariate(1.2) for _ in range(rng.randint(1, 30))]
            n, trunc = rng.randint(100, 3000), 0.001
            m = combine_runs.overweight_max_weight(w, n, trunc, 1.)
            self.assertLessEqual(sum(x for x in w if x > m), trunc * n * m * (1 + 1e-12))
            m2 = m * (1 - 1e-6)
            if m2 >= 1.:
                self.assertGreater(sum(x for x in w if x > m2), trunc * n * m2)

    def test_min_weight(self):
        self.assertEqual(combine_runs.overweight_max_weight([1.] * 10, 10, 0.1, 2.), 2.)


class TestPoolChannel(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir)

    def test_pool(self):
        random.seed(5)
        a = os.path.join(self.dir, 'a.lhe')
        b = os.path.join(self.dir, 'b.lhe')
        # job a: unit 1, one event of 20 units; job b: unit 1.2
        write_events(a, [1.] * 500 + [20.])
        write_events(b, [1.2] * 400)
        n, m = pool_channel(self.dir, [(a, .5), (b, .5)], 450, 0.001)
        # the budget (0.45 units) cannot hold the 10-unit pooled event: m = 10
        self.assertAlmostEqual(m, 10.)
        w = combine_runs.read_abs_weights(os.path.join(self.dir, 'events.lhe'))
        self.assertEqual(len(w), n)
        self.assertAlmostEqual(min(w), m, 5)
        self.assertLessEqual(max(w), m * (1 + 1e-6))

    def test_largest_source_unit(self):
        a = os.path.join(self.dir, 'a.lhe')
        b = os.path.join(self.dir, 'b.lhe')
        write_events(a, [1.] * 100)
        write_events(b, [3.] * 100)
        # the events of b were unweighted at 3*0.5: m cannot go below
        n, m = pool_channel(self.dir, [(a, .5), (b, .5)], 1e6, 0.5)
        self.assertAlmostEqual(m, 1.5)


def pool_channel(gdir, sources, n_budget, trunc):
    """pool the sources and unweight them at the channel's budget maximum"""
    weights, min_wgt = combine_runs.pool_weights(sources)
    m = combine_runs.overweight_max_weight(weights, n_budget, trunc, min_wgt)
    return combine_runs.write_pool(gdir, sources, m), m


class TestProcessBudget(unittest.TestCase):

    def test_raises_only_where_needed(self):
        # A: five 2-unit events (10 units above m=1); B: one 50-unit event.
        # budget 15 units: B must be raised (A alone cannot get below), then
        # A's 10 units fit and A is left alone
        a = [1.] * 1000 + [2.] * 5
        b = [1.] * 1000 + [50.]
        total = sum(a) + sum(b)
        pb = combine_runs.ProcessBudget(15. / total)
        pb.add(a, 1., 900, 1000)
        pb.add(b, 1., 900, 1000)
        self.assertEqual(pb.solve(), [1., 50.])

    def test_prefers_cheap_raise(self):
        # A: 30 units above m=1 (10 events of 3), B: 20 (one event); a budget of
        # 25 units needs A raised (B alone is not enough): raising A to 3 costs
        # 1000*(900/(10+1000/3)-1) points, B to 20 costs 1000*(900/(1+1000/20)-1)
        a = [1.] * 1000 + [3.] * 10
        b = [1.] * 1000 + [20.]
        total = sum(a) + sum(b)
        pb = combine_runs.ProcessBudget(25. / total)
        pb.add(a, 1., 900, 1000)
        pb.add(b, 1., 900, 1000)
        m = pb.solve()
        self.assertEqual(m, [3., 1.])
        # the result is within the budget
        ow = sum(w for w in a if w > m[0]) + sum(w for w in b if w > m[1])
        self.assertLessEqual(ow, 25.)

    def test_within_budget(self):
        pb = combine_runs.ProcessBudget(0.1)
        pb.add([1.] * 100 + [2.], 1., 90, 100)
        self.assertEqual(pb.solve(), [1.])


class FakeResults(list):
    axsec = 0


class FakeChannel(dict):
    def __init__(self, P, G, axsec):
        super(FakeChannel, self).__init__(axsec=axsec, nunwgt=0, nevents=0)
        self.parent_name, self.name = P, G

    def get(self, key):
        return self[key]


class TestEnforceOverweightLimit(unittest.TestCase):
    """the top-up loop, with refine jobs simulated in python"""

    def setUp(self):
        self.me_dir = tempfile.mkdtemp()
        self.pdir = os.path.join(self.me_dir, 'SubProcesses', 'P1_qq_ll')
        os.makedirs(os.path.join(self.pdir, 'G1'))
        os.makedirs(os.path.join(self.pdir, 'G2'))
        random.seed(11)

    def tearDown(self):
        shutil.rmtree(self.me_dir)

    def make_ximprove(self, run_card, grid=True):
        x = object.__new__(gen_ximprove.gen_ximprove_v4)
        x.me_dir = self.me_dir
        x.run_card = run_card
        x.gen_events_security = 1.2
        x.goal_lum = 1.2 * 1000 / 2.  # 1000 events, sigma = 2 pb
        x.results = FakeResults()
        x.ow_channels = {}
        channels = [FakeChannel('P1_qq_ll', 'G1', 1.), FakeChannel('P1_qq_ll', 'G2', 1.)]
        for C in channels:
            info = {'P_dir': C.parent_name, 'directory': C.name, 'base_directory': '',
                    'offset': 1, 'nevents': 1000, 'maxiter': 9, 'miniter': 3}
            x.ow_channels[(C.parent_name, C.name)] = x.ow_channel_entry(C, info, 1, 600.)
            if grid:
                open(os.path.join(self.pdir, C.name, 'ftn26'), 'w').close()
        # 10 points per event. G1: clean, 660 unit events; G2: 660 events, one of 30 units
        write_events(os.path.join(self.pdir, 'G1', 'events.lhe'), [1. / 660] * 660)
        write_results(os.path.join(self.pdir, 'G1', 'results.dat'), 1., 660, nevents=6600)
        write_events(os.path.join(self.pdir, 'G2', 'events.lhe'), [1. / 689] * 659 + [30. / 689])
        write_results(os.path.join(self.pdir, 'G2', 'results.dat'), 1., 660, nevents=6600)
        self.jobs = []
        self.units = []
        self.empty_jobs = False

        def run_ow_jobs(jobs):
            # a top-up job: sigma = 1, one clean unit event per 10 points
            # (a multi-iteration job: 660 events of 6600 points)
            self.jobs += jobs
            for j in jobs:
                d = os.path.join(self.pdir, j['directory'])
                if j['maxiter'] == 1:
                    # prepared by ow_topup_jobs: the grid and the unit
                    self.assertTrue(os.path.exists(os.path.join(d, 'ftn25')))
                    with open(os.path.join(d, 'ow_threshold.dat')) as fsock:
                        self.units.append(float(fsock.read()))
                else:
                    os.makedirs(d)
                points = j['nevents'] if j['maxiter'] == 1 else 6600
                n = points // 10
                if self.empty_jobs and j['directory'].endswith('_0'):
                    # no event stored: results.dat only (sigma estimate 0.5)
                    write_results(os.path.join(d, 'results.dat'), .5, 0, nevents=points)
                    continue
                write_events(os.path.join(d, 'events.lhe'), [1. / n] * n)
                write_results(os.path.join(d, 'results.dat'), 1., n, nevents=points)
        x.run_ow_jobs = run_ow_jobs
        return x

    def test_top_up(self):
        self.check_top_up('channel')

    def test_top_up_process(self):
        self.check_top_up('process')

    def test_top_up_without_grid(self):
        self.check_top_up('channel', grid=False)

    def check_top_up(self, mode, grid=True):
        x = self.make_ximprove({'max_overweight_truncation': 0.001,
                                'overweight_limit_min_events': -1,
                                'overweight_limit_mode': mode}, grid)
        x.enforce_overweight_limit()
        # G1 is within its budget (500 events, floor 500): untouched
        self.assertFalse([j for j in self.jobs if j['directory'].startswith('G1')])
        g2 = [j for j in self.jobs if j['directory'].startswith('G2')]
        self.assertTrue(g2)
        # new offsets (seeds), on the channel's grid; single iterations if it has one
        self.assertEqual(sorted(j['offset'] for j in g2), list(range(2, 2 + len(g2))))
        if grid:
            self.assertTrue(all(j['maxiter'] == 1 and j['miniter'] == 1 for j in g2))
            self.assertTrue(all(j['nevents'] <= x.ow_max_job_points for j in g2))
            # first round: unit = m * points / size / margin, m = 30/689 (G2's large event)
            self.assertAlmostEqual(self.units[0] / (30. / 689 * 6600 / g2[0]['nevents'] / 1.1), 1., 6)
        else:
            self.assertTrue(all(j['maxiter'] == 9 for j in g2))
        r = sum_html.OneResult('G2')
        r.read_results(os.path.join(self.pdir, 'G2', 'results.dat'))
        # points-weighted average of estimates of 1 pb
        self.assertAlmostEqual(r.axsec, 1., 5)
        self.assertGreaterEqual(r.nunwgt, 500)
        w = combine_runs.read_abs_weights(os.path.join(self.pdir, 'G2', 'events.lhe'))
        self.assertEqual(len(w), r.nunwgt)
        m = min(w)
        # the 30-unit event is still there (never discarded for its weight) and in budget
        above = sum(x for x in w if x > m * (1 + 1e-6))
        self.assertLessEqual(above, 0.001 * 500 * m * (1 + 1e-6))
        self.assertAlmostEqual(max(w) * r.nevents / 6600 * 689, 30., 3)
        # the event combination keeps the maximum weight
        self.assertTrue(os.path.exists(gen_ximprove.ow_marker_path(self.me_dir)))
        # the job directories are removed
        self.assertFalse([d for d in os.listdir(self.pdir) if '_ow' in d])
        # the raise is kept over the rounds (G2 by its 30-unit event, G1 not)
        entries = dict((k[1], e) for k, e in x.ow_channels.items())
        self.assertTrue(entries['G2']['raised'])
        self.assertAlmostEqual(entries['G2']['raise'], 30., 4) # 30/689 over the unit 1/689
        self.assertFalse(entries['G1']['raised'])

    def test_job_without_events(self):
        # every round's first job stores no event: its points and its estimate
        # (0.5 pb) still enter the pooled cross-section
        x = self.make_ximprove({'max_overweight_truncation': 0.001,
                                'overweight_limit_min_events': -1,
                                'overweight_limit_mode': 'channel'})
        self.empty_jobs = True
        x.enforce_overweight_limit()
        r = sum_html.OneResult('G2')
        r.read_results(os.path.join(self.pdir, 'G2', 'results.dat'))
        g2 = [j for j in self.jobs if j['directory'].startswith('G2')]
        empty = sum(j['nevents'] for j in g2 if j['directory'].endswith('_0'))
        self.assertAlmostEqual(r.axsec, 1. - .5 * empty / r.nevents, 4) # results.dat: 5 digits

    def test_not_event_generation(self):
        # refine for a precision: no channel bookkeeping, nothing to do
        x = self.make_ximprove({'max_overweight_truncation': 0.001,
                                'overweight_limit_min_events': -1,
                                'overweight_limit_mode': 'channel'})
        del x.ow_channels
        x.enforce_overweight_limit()
        self.assertEqual(self.jobs, [])
        self.assertFalse(os.path.exists(gen_ximprove.ow_marker_path(self.me_dir)))

    def test_requires_truncation(self):
        x = self.make_ximprove({'max_overweight_truncation': -1,
                                'overweight_limit_min_events': -1,
                                'overweight_limit_mode': 'channel'})
        x.enforce_overweight_limit()
        self.assertEqual(self.jobs, [])

    def test_raise_stable_when_reapplied(self):
        # process mode re-applies every channel each round: the raise must not drift
        x = self.make_ximprove({'max_overweight_truncation': 0.001,
                                'overweight_limit_min_events': -1,
                                'overweight_limit_mode': 'process'})
        x.enforce_overweight_limit()
        e = dict((k[1], e) for k, e in x.ow_channels.items())['G2']
        raise0 = e['raise']
        for i in range(3):
            x.ow_gather(e)
            x.ow_apply(e)
        self.assertAlmostEqual(e['raise'] / raise0, 1., 6)


if __name__ == '__main__':
    unittest.main()
