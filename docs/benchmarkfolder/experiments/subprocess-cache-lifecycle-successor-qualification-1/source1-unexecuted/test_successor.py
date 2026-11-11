"""Pure mutation tests of successor evidence gates; no native execution."""
import copy
import importlib.util
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('successor_cache_analysis', HERE/'analyze.py')
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)


def key_statistics(cache):
    counters = [0] * 15
    keys = {}
    for action in A.KEYS:
        precision, key = (action-1000)//100, action%100
        before = counters[:]
        if precision == 2 and (cache == 0 or key == 13):
            counters[14] += 1
        elif cache == -1 and key != 13:
            cold = key in (1, 3, 5, 8, 10)
            if precision == 1:
                counters[int(cold)] += 1
                if cold:
                    counters[6] += 1
                    counters[7] += 64
            elif cold:
                counters[5] += 1
                counters[6] += 1
                counters[7] += 64
                counters[10] += 1
                counters[11] += 1
                counters[12] += 64
            else:
                counters[4] += 1
                counters[9] += 1
        keys[action] = dict(action=action, precision=precision, key=key,
                            before=before, after=counters[:])
    return dict(cache=cache, keys=keys)


class SuccessorLatches(unittest.TestCase):
    def check(self, frame):
        checks = A.Checks()
        A.check_failure_latches(frame, checks, 'synthetic: ')
        return checks.failures

    def test_each_prefix_recovers_while_other_prefix_stays_poisoned(self):
        for process in (1, 2):
            after = ['F', 'F', 'T', 'T'] if process == 1 else ['T', 'T', 'F', 'F']
            frame = dict(action=302, process=process, latches_before=['T']*4,
                         latches_after=after)
            self.assertFalse(self.check(frame))
            for i in range(4):
                changed = copy.deepcopy(frame)
                changed['latches_after'][i] = 'T' if after[i] == 'F' else 'F'
                self.assertTrue(self.check(changed))

    def test_missing_poison_and_contaminated_clean_reference_rejected(self):
        frame = dict(action=301, process=1, latches_before=['F']*4,
                     latches_after=['F']*4)
        self.assertFalse(self.check(frame))
        frame['latches_before'][0] = 'T'
        self.assertTrue(self.check(frame))
        frame.update(action=302, latches_before=['F']*4)
        self.assertTrue(self.check(frame))


class SuccessorStabilityCache(unittest.TestCase):
    def check(self, data):
        checks = A.Checks()
        A.check_key_caches(data, checks, 'synthetic')
        return checks.failures

    def test_independent_checks_bypass_then_primary_repeats_hit(self):
        for cache in (0, -1):
            self.assertFalse(self.check(key_statistics(cache)))

    def test_stability_check_cannot_hit_or_insert_primary_store(self):
        for precision, index in ((1, 0), (1, 6), (2, 9), (2, 11)):
            data = key_statistics(-1)
            data['keys'][1000+100*precision+13]['after'][index] += 1
            self.assertTrue(any('independent stability check' in x for x in self.check(data)))

    def test_primary_after_check_must_still_hit_without_new_entry(self):
        for precision in (1, 2):
            data = key_statistics(-1)
            frame = data['keys'][1000+100*precision+14]
            frame['after'] = frame['before'][:]
            self.assertTrue(any('repeat' in x for x in self.check(data)))

    def test_failed_or_missing_qp_bypass_counter_is_rejected(self):
        data = key_statistics(-1)
        data['keys'][1213]['after'][14] -= 1
        self.assertTrue(any('independent stability check' in x for x in self.check(data)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
