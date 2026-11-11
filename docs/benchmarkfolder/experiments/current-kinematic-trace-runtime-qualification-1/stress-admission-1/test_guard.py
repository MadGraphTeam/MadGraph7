"""Pure bounds and policy checks for the fixed stress successor."""
import copy
import json
from pathlib import Path
import unittest
import tempfile
from types import SimpleNamespace as NS
from unittest.mock import patch
import guard


class StressRequestTests(unittest.TestCase):
    def setUp(self):
        self.all=json.loads((Path(__file__).resolve().parents[1]/'source/inputs/requests.json').read_bytes())['requests']
        self.requests=[copy.deepcopy(r) for r in self.all if r['stage']=='stress']

    def test_exact_saved_requests_and_ordinary_excluded(self):
        actual=guard.select_requests(self.all)
        self.assertEqual(len(actual),18)
        self.assertEqual(sum(r['maximum_outer_amplitude_entries'] for r in actual),36)
        self.assertEqual({(r['case'],r['point']) for r in actual},{('cms',20),('cms',21),('ew',2)})
        self.assertEqual({r['requested_mode'] for r in actual},{1,4})

    def test_missing_or_duplicate_worker_rejected(self):
        for items in (self.requests[:-1],self.requests+[self.requests[0]],self.requests[:-1]+[self.requests[0]]):
            with self.assertRaisesRegex(ValueError,'request set differs'):guard.select_requests(items)

    def test_other_point_or_mode_rejected(self):
        for field,value in [('point',1),('requested_mode',2),('case','other')]:
            items=copy.deepcopy(self.requests);items[0][field]=value
            with self.assertRaisesRegex(ValueError,'fields or bootstrap differ'):guard.select_requests(items)

    def test_extra_bootstrap_rejected(self):
        for field,value in [('bootstrap_outer_entries',5),('bootstrap_point',20),
                            ('bootstrap','cms-point20.dat'),('maximum_outer_amplitude_entries',6)]:
            items=copy.deepcopy(self.requests);items[0][field]=value
            with self.assertRaisesRegex(ValueError,'fields or bootstrap differ'):guard.select_requests(items)

    def test_unrequested_trace_or_measurement_rejected(self):
        for field,value in [('trace_enabled',True),('measurement','cms-point1.dat')]:
            items=copy.deepcopy(self.requests);items[0][field]=value
            with self.assertRaisesRegex(ValueError,'fields or bootstrap differ'):guard.select_requests(items)

    def test_failed_postcheck_retains_original_error_after_module_replacement(self):
        with tempfile.TemporaryDirectory() as temp:
            results=Path(temp);out=results/'guard'
            (results/'stress-parity.json').write_text('{"passed":true}')
            def save(path,data):
                with Path(path).open('x') as f:json.dump(data,f)
            r=NS(setup_owned_helpers=lambda:None)
            r.main=lambda:r.setup_owned_helpers()
            old=NS(G=NS(save=save,SOURCE_SHA='source',DERIVED_SHA='derived'),
                   install_hook=lambda *args:set(range(18)))
            calls=[]
            def prepare(expected):
                calls.append(expected)
                if len(calls)==1:
                    guard.O=old
                    return r,{}, {}, [], {'protected':'original'}
                guard.O=NS(G=None)
                raise ValueError('closed ordinary bytes changed')
            with patch.object(guard,'prepare',prepare),patch.object(guard,'OUT',out), \
                 patch.object(guard,'RESULTS',results),patch.object(guard,'ROOT',Path.cwd()), \
                 patch.object(guard.sys,'argv',['guard.py','--reviewed-guard-index','reviewed','--execute']):
                with self.assertRaisesRegex(ValueError,'closed ordinary bytes changed'):guard.main()
            receipt=json.loads((out/'result.json').read_bytes())
            self.assertFalse(receipt['passed'])
            self.assertIsNone(receipt['failure'])
            self.assertIn('closed ordinary bytes changed',receipt['postcheck_failure'])
            self.assertEqual(len(receipt['attempted_runtime_admissions']),18)


if __name__=='__main__':unittest.main()
