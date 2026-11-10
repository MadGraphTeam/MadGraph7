"""Pure parser and mocked failure-retention checks; no native process launch."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('deployment_smoke',HERE/'smoke.py')
smoke=importlib.util.module_from_spec(spec);spec.loader.exec_module(smoke)


class SmokeTest(unittest.TestCase):
    def fixture(self):
        path=smoke.ROOT/'docs/benchmarkfolder/qualification/madloop-adapter/release/release-internal-smoke.json'
        return json.loads(path.read_text())

    def test_historical_driver_schema_and_comparison(self):
        rows=self.fixture()
        reference=smoke.parse(rows[0]['result'],1)
        self.assertEqual(len(rows),6)
        for row in rows:
            value=smoke.parse(row['result'],row['mode'])
            comparisons=smoke.compare(reference,value)
            self.assertEqual(len(comparisons),8)
            self.assertTrue(all(item['ordinary_target_passed'] for item in comparisons.values()))

    def test_rejects_nonfinite_duplicate_status_and_changed_point(self):
        original=self.fixture()[0]['result']
        for changed in [original+'\nBORN 1\n',original.replace('RETCODE 110','RETCODE 219'),
                        original.replace('BORN         1.108153858850734E+002','BORN NaN')]:
            with self.assertRaises((RuntimeError,ValueError)):
                smoke.parse(changed,1)
        value=smoke.parse(original,1)
        changed=json.loads(json.dumps(value));changed['point'][0][0]='46'
        with self.assertRaises(RuntimeError):smoke.compare(value,changed)

    def test_refuses_bad_precision_and_numeric_disagreement(self):
        original=self.fixture()[0]['result']
        with self.assertRaises(RuntimeError):smoke.parse(original,4)
        value=smoke.parse(original,1)
        changed=json.loads(json.dumps(value));changed['split']['SO_Loop:FIN']='12'
        self.assertFalse(smoke.compare(value,changed)['split:SO_Loop:FIN']['regression_passed'])

    def test_launch_failure_retains_attempt_and_partial_result(self):
        with tempfile.TemporaryDirectory(dir=smoke.WORK) as tmp:
            work=Path(tmp);(work/'logs').mkdir();(work/'results').mkdir();(work/'worker').mkdir()
            (work/'worker/result.dat').write_bytes(b'partial before failure')
            rows=[]
            with patch.object(smoke,'WORK',work),patch.object(smoke.subprocess,'Popen',side_effect=OSError('mock launch refusal')):
                with self.assertRaises(OSError):
                    smoke.run_worker(work/'worker/check',work/'worker','fixture',120,
                                     smoke.time.monotonic()+900,rows)
            self.assertEqual(len(rows),1)
            saved=json.loads((work/'results/fixture-execution.json').read_text())
            self.assertIsNone(saved['pid']);self.assertIn('mock launch refusal',saved['failure'])
            self.assertEqual((work/saved['raw_result']).read_bytes(),b'partial before failure')

    def test_timeout_retains_pid_cleanup_and_output(self):
        class Child:
            pid=987654321
            returncode=-15
            def __init__(self):self.calls=0
            def wait(self,timeout):
                self.calls+=1
                if self.calls==1:raise subprocess.TimeoutExpired(['check'],timeout)
                return self.returncode
        with tempfile.TemporaryDirectory(dir=smoke.WORK) as tmp:
            work=Path(tmp);(work/'logs').mkdir();(work/'results').mkdir();(work/'worker').mkdir()
            (work/'worker/result.dat').write_bytes(b'partial timeout')
            rows=[]
            with patch.object(smoke,'WORK',work),patch.object(smoke.subprocess,'Popen',return_value=Child()), \
                    patch.object(smoke.os,'killpg',side_effect=ProcessLookupError):
                with self.assertRaises(subprocess.TimeoutExpired):
                    smoke.run_worker(work/'worker/check',work/'worker','fixture',120,
                                     smoke.time.monotonic()+900,rows)
            saved=json.loads((work/'results/fixture-execution.json').read_text())
            self.assertTrue(saved['timeout']);self.assertTrue(saved['cleanup_complete'])
            self.assertEqual(saved['pid'],987654321);self.assertEqual(saved['returncode'],-15)
            self.assertEqual((work/saved['raw_result']).read_bytes(),b'partial timeout')

    def test_loader_rejects_link_back_to_original_prefix(self):
        with tempfile.TemporaryDirectory(dir=smoke.WORK) as tmp:
            work=Path(tmp);(work/'relocated/lib').mkdir(parents=True)
            (work/'original.so').write_bytes(b'not a real native library')
            dso=work/'relocated/lib/libmadloop_tirust.so';dso.symlink_to(work/'original.so')
            with patch.object(smoke,'WORK',work),self.assertRaises(RuntimeError):
                smoke.loader_evidence('fixture',123,dso,{'runtime_providers':{}})

    def test_loader_requires_prebound_observed_provider(self):
        with tempfile.TemporaryDirectory(dir=smoke.WORK) as tmp:
            work=Path(tmp);(work/'logs').mkdir();(work/'relocated/lib').mkdir(parents=True)
            dso=work/'relocated/lib/libmadloop_tirust.so';dso.write_bytes(b'fixture bytes only')
            (work/'logs/loader-fixture.123').write_text('123: calling init: '+str(dso)+'\n')
            binding={'candidate':{'native':{'files':{'lib/libmadloop_tirust.so':smoke.digest(dso)}}}}
            with patch.object(smoke,'WORK',work),patch.object(smoke,'BIND',binding):
                with self.assertRaises(RuntimeError):
                    smoke.loader_evidence('fixture',123,dso,{'runtime_providers':{}})
                result=smoke.loader_evidence('fixture',123,dso,{'runtime_providers':{str(dso):smoke.digest(dso)}})
                self.assertEqual(result['initialized_providers'],{str(dso):smoke.digest(dso)})


if __name__=='__main__':unittest.main()
