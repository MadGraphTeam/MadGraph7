"""Meaningful portable integrity, scope, effect and relocation tests; no subprocesses."""
import builtins,copy,gzip,hashlib,importlib.util,json,os,shutil,tempfile
from pathlib import Path
import unittest
import replay as R
HERE=Path(__file__).resolve().parent
class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.ix,cls.c,cls.d,cls.p=R.decode(HERE)
    def test_01_full_guarded_replay(self):
        with R.blocked_apis({}) as attempts:result=R.semantic(self.d,self.c,self.p)
        self.assertFalse(attempts);self.assertEqual(result,self.ix['claims']);self.assertFalse(result['mass_fix_resolved_EW_stress'])
    def test_02_relocation(self):
        with tempfile.TemporaryDirectory(prefix='EW-threshold-portable-') as d:
            target=Path(d)/'relocated';shutil.copytree(HERE,target);self.assertEqual(R.verify(target),self.ix['claims'])
    def test_03_physical_tamper(self):
        with tempfile.TemporaryDirectory(prefix='EW-threshold-tamper-') as d:
            target=Path(d)/'copy';shutil.copytree(HERE,target);f=target/'README.md';f.write_bytes(f.read_bytes()+b'changed\n')
            with self.assertRaisesRegex(ValueError,'Changed physical'):R.verify(target)
    def test_04_false_success(self):
        r=json.loads(self.d['run2/candidate/result.json']);r['lane']['observed_statuses'][1]['code']=333
        with self.assertRaisesRegex(ValueError,'False threshold success'):R.validate_observed_scope(r['lane'],r)
    def test_05_false_QP(self):
        r=json.loads(self.d['run2/candidate/result.json']);r['lane']['observed_statuses'][1].update(precision='QP',accepted_public_status=True,comparison_eligible=True)
        with self.assertRaisesRegex(ValueError,'False accepted QP'):R.validate_observed_scope(r['lane'],r)
    def test_06_false_filter_learning(self):
        r=json.loads(self.d['run2/candidate/result.json']);r['stored_filters_learned']=True
        with self.assertRaisesRegex(ValueError,'falsely called learned'):R.validate_observed_scope(r['lane'],r)
    def test_07_false_state_learning(self):
        r=json.loads(self.d['run2/candidate/result.json']);r['lane']['readiness']['states'][0]['stored_INIT_flags_are_learned']=True
        with self.assertRaisesRegex(ValueError,'False filter-learning'):R.validate_observed_scope(r['lane'],r)
    def test_08_failed_buffer_eligibility(self):
        r=json.loads(self.d['run2/candidate/result.json']);r['lane']['values']['targets'][1]['answers'][0]['comparison_eligible']=True
        with self.assertRaisesRegex(ValueError,'Failed buffers falsely'):R.validate_observed_scope(r['lane'],r)
    def test_09_mutated_raw_status_refused(self):
        protocol=R.load_source('test_actual_protocol',self.d['source/run2/target_protocol.py'],'contained:test_protocol')
        raw=self.d['run2/candidate/target-raw.dat'].decode();self.assertEqual([v['answers'][0]['public']['accepted_public_status'] for v in protocol.raw(raw)['targets']],[True,False])
        self.assertFalse(protocol.public_status(420)['accepted_public_status'])
        # Changing accuracy to zero never makes the failure status eligible.
        text='\n'.join(' '.join(row[:-1]+['0000000000000000']) if row[:2]==['ANS','2'] else ' '.join(row) for row in (line.split() for line in raw.splitlines()))
        self.assertTrue(all(not r['comparison_eligible'] for r in protocol.raw(text)['targets'][1]['answers']))
    def test_10_effects_refused(self):
        import ctypes,socket,subprocess
        with R.blocked_apis({}) as attempts:
            for call in (lambda:Path('/not-contained').read_bytes(),lambda:subprocess.run(['true']),lambda:ctypes.CDLL('/unused'),lambda:socket.socket(),lambda:os.kill(1,0),lambda:Path('/unused').write_text('x')):
                with self.assertRaises(RuntimeError):call()
        self.assertEqual(len(attempts),6)
    def test_11_no_binary_payloads(self):
        for name,data in self.d.items():self.assertFalse(data.startswith(b'\x7fELF'),name);data.decode('utf8')
    def test_12_no_failed_numeric_comparison(self):
        result=R.semantic(self.d,self.c,self.p);self.assertIsNone(result['threshold_numerical_comparison']);self.assertEqual(result['failed_buffer_word_differences_diagnostic_only'],4)
    def test_13_exact_inventory_extra_file_refused(self):
        with tempfile.TemporaryDirectory(prefix='EW-threshold-extra-') as d:
            target=Path(d)/'copy';shutil.copytree(HERE,target);(target/'extra').write_text('x')
            with self.assertRaisesRegex(ValueError,'Exact physical inventory'):R.verify(target)
if __name__=='__main__':unittest.main(verbosity=2)
