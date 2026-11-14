"""Independent checks of the half-weight immediate-epoch replay bank and fit."""
import itertools
import math
import random
import unittest
from tests.input_files import tiled_s_event_reference as ref


def data(n=16, s=1, f=1, seed=191):
    rng = random.Random(seed)
    b = [[[rng.gauss(0., 1.) for _ in range(f)] for _ in range(16)] for _ in range(n)]
    r = [[[[rng.gauss(0., 1.) for _ in range(f)] for _ in range(16)] for _ in range(s)] for _ in range(n)]
    return b, r


def ramp(n, f, offset=0.):
    b = [[[float((o * 16 + node) * f + k) + offset for k in range(f)] for node in range(16)] for o in range(n)]
    return b, [[[[x - 7. for x in node] for node in orbit]] for orbit in b]


def identity(ns=1):
    return [list(range(16)) for _ in range(ns)]


def zeros(n, s, f):
    return [[[0.] * f for _ in range(16)] for _ in range(n)], \
        [[[[0.] * f for _ in range(16)] for _ in range(s)] for _ in range(n)]


class TestEpochReplayReference(unittest.TestCase):
    def test_current_first_weight_and_canonical_flavour_padding(self):
        b, r = ramp(2, 1)
        pb, pr = ramp(4, 3, offset=-13.)
        a = ref.bank(b, r, pb, pr)
        self.assertEqual(a.previous_scale, .25)
        self.assertEqual((len(a.b), len(a.b[0][0])), (6, 3))
        for o in range(2):
            for node in range(16):
                self.assertEqual(a.b[o][node], b[o][node] + [0., 0.])
                self.assertEqual(a.r[o][0][node], r[o][0][node] + [0., 0.])
        for o in range(4):
            for node in range(16):
                self.assertEqual(a.b[2 + o][node], [.25 * x for x in pb[o][node]])
                self.assertEqual(a.r[2 + o][0][node], [.25 * x for x in pr[o][0][node]])
        b[0][0][0] = 999.
        self.assertNotEqual(a.b[0][0][0], 999.)

    def test_mixed_loss_matches_current_mean_plus_half_previous_mean(self):
        b, r = data(n=7, s=2, f=2)
        pb, pr = data(n=11, s=2, f=3, seed=192)
        p = [list(range(16)), [(x - 3) % 16 for x in range(16)]]
        measured = ref.bank_objective(ref.bank(b, r, pb, pr), p) / 7

        def padded(bank_b, bank_r):
            return ([[node + [0.] * (3 - len(node)) for node in orbit] for orbit in bank_b],
                    [[[node + [0.] * (3 - len(node)) for node in sector] for sector in orbit] for orbit in bank_r])
        expected = ref.objective(*padded(b, r), p) / 7 + .5 * ref.objective(pb, pr, p) / 11
        self.assertAlmostEqual(measured, expected, places=12)

    def test_empty_history_is_exact_current_copy_without_flavour_inflation(self):
        b, r = data(n=4, f=1)
        a = ref.bank(b, r, [], [])
        self.assertFalse(a.replayed)
        self.assertEqual(a.previous_scale, 0.)
        self.assertEqual((a.b, a.r), (b, r))

    def test_empty_current_retains_maps_and_does_not_fit_old_bank(self):
        pb, pr = data(n=100, f=2)
        p = [ref.phase_map(9)]
        result = ref.fit_epoch([], [], pb, pr, p, [[1, 1]], (4, 4, 1), 5)
        self.assertIsNone(result['fit'])
        self.assertEqual(result['before'], 0.)
        self.assertEqual(len(result['bank'].b), 0)
        self.assertEqual(result['maps'], p)

    def test_previous_count_never_satisfies_current_support(self):
        b, r = data(n=15)
        pb, pr = data(n=100)
        with self.assertRaises(ValueError):
            ref.fit_epoch(b, r, pb, pr, identity(2), [[0, 0]] * 2, (1, 1, 1), 4)
        result = ref.fit_epoch(b, r, pb, pr, identity(), [[0, 0]], (1, 1, 1), 4)
        self.assertIsNone(result['fit'])
        self.assertEqual(result['geometry'], (1, 1, 1))
        self.assertEqual(len(result['bank'].b), 115)
        self.assertEqual(result['before'], ref.bank_objective(result['bank'], identity()))
        self.assertEqual(ref.current_geometry(31, (2, 2, 1), 3), ((2, 2, 1), True))
        self.assertEqual(ref.current_geometry(63, (4, 2, 1), 3), ((4, 2, 1), True))

    def test_supported_single_sector_reaches_bruteforce_coarse_optimum(self):
        b, r = data(n=16, f=1)
        pb, pr = data(n=9, f=2)
        result = ref.fit_epoch(b, r, pb, pr, identity(), [[0, 0]], (1, 1, 1), 1)
        value = result['bank']
        best = min(ref.bank_objective(value, ref.lift([list(p)], (2, 2, 1)))
                   for p in itertools.permutations(range(4)))
        self.assertEqual(result['geometry'], (2, 2, 1))
        self.assertAlmostEqual(result['after'], best, places=12)
        self.assertLessEqual(result['after'], result['before'])

    def test_eight_orientation_tie_prefers421_then_freezes_orientation(self):
        b, r = zeros(32, 1, 1)
        pb, pr = zeros(10, 1, 2)
        result = ref.fit_epoch(b, r, pb, pr, identity(), [[0, 0]], (2, 2, 1), 2)
        self.assertEqual(result['geometry'], (4, 2, 1))
        self.assertEqual(len(result['alternatives']), 2)
        result = ref.fit_epoch(b, r, pb, pr, identity(), [[0, 0]], (2, 4, 1), 3)
        self.assertEqual(result['geometry'], (2, 4, 1))
        self.assertEqual(len(result['alternatives']), 1)

    def test_immediate_empty_epoch_drops_nonempty_older_history(self):
        records = [{'entries': [{'slot': 1, 'nt': 100}]}, {'entries': [{'slot': 1, 'nt': 0}]},
                   {'entries': [{'slot': 1, 'nt': 32}]}]
        self.assertIsNone(ref.immediate_previous(records, 0, 1))
        self.assertEqual(ref.immediate_previous(records, 2, 1)['nt'], 0)
        self.assertIsNone(ref.immediate_previous(records, 2, 2))
        records[1]['entries'] *= 2
        with self.assertRaises(ValueError):
            ref.immediate_previous(records, 2, 1)

    def test_metadata_compatibility_and_invalid_raw_banks_reject(self):
        e = dict(slot=1, ns=1, nlegs=3, alias_gauge=0, ids=[5], providers=[1], groups=[1], pdgs=[[1, 2, 3]],
                 q=[[0, 0]])
        ref.compatible(e, dict(e))
        for key in e:
            other = dict(e)
            other[key] = e[key] + 1 if isinstance(e[key], int) else [[2]] + e[key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                ref.compatible(e, other)
        b, r = data()
        pb, pr = data(s=2)
        with self.assertRaises(ValueError):
            ref.bank(b, r, pb, pr)
        with self.assertRaises(ValueError):
            ref.bank(b, r, pb, None)
        huge_b = [[[1e308] for _ in range(16)]]
        with self.assertRaises(ValueError):
            ref.bank(*zeros(10, 1, 1), huge_b, zeros(1, 1, 1)[1])
        b[0][0][0] = math.nan
        with self.assertRaises(ValueError):
            ref.bank(b, r)

    def test_losses_add_left_to_right_like_the_fortran(self):
        # 1e16 absorbs each following 1 when added left to right; a
        # compensated sum (Python 3.12 sum()) would return 1e16+2.
        within = [[[-1e16], [-1.], [-1.]] + [[0.]] * 13]
        self.assertEqual(ref.objective(within, [[[[0.]] * 16]], identity()), 1e16)
        across = [[[-x]] + [[0.]] * 15 for x in (1e16, 1., 1.)]
        self.assertEqual(ref.objective(across, [[[[0.]] * 16]] * 3, identity()), 1e16)

    def test_assignment_matches_bruteforce_and_certifies_optimum(self):
        rng = random.Random(7)
        for n in (1, 2, 3, 4, 5):
            for _ in range(20):
                cost = [[rng.choice([0., 1., 2.5, rng.random()]) for _ in range(n)] for _ in range(n)]
                column, value = ref.assignment(cost)
                self.assertEqual(sorted(column), list(range(n)))
                brute = min(sum(cost[i][p[i]] for i in range(n)) for p in itertools.permutations(range(n)))
                self.assertAlmostEqual(value, brute, places=12)


if __name__ == '__main__':
    unittest.main()
