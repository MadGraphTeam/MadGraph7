"""Focused retained-byte worker replay tests. No native process or library work."""
import copy
import ctypes
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent

def load(name):
    spec=importlib.util.spec_from_file_location('checked_capsule_test_'+name,HERE/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


class PortableWorkers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.verify=load('verify');cls.capture=load('capture');cls.worker=load('worker_checks')
        if (HERE/'evidence-index.json').is_file():
            index,data,document=cls.verify.packet(HERE)
            cls.raw={name:data(name) for name in index['origins']}
        else:_,cls.raw=cls.capture.collect()
        cls.ancestors={name:cls.verify.packet(HERE.parent/name,sha) for name,sha in cls.capture.DEPENDENCIES.items()}
    def replay(self,data=None,document=None,ancestors=None):
        callback=data or self.raw.__getitem__
        return self.worker.verify_workers(callback,document or (lambda n:json.loads(callback(n))),ancestors or self.ancestors)
    def test_exact_replay_has_no_host_files_processes_or_library_loading(self):
        def blocked(*a,**k):raise AssertionError('Live I/O/native operation reached portable worker')
        with patch('builtins.open',blocked),patch.object(io,'open',blocked),patch.object(os,'open',blocked),\
             patch.object(Path,'read_bytes',blocked),patch.object(subprocess,'Popen',blocked),patch.object(ctypes,'CDLL',blocked):
            result=self.replay()
        self.assertTrue(result['comparison_serialized_bytes_reproduced'])
        self.assertTrue(result['original_admission_failure_reproduced'])
        self.assertTrue(result['retrospective_bootstrap_reproduced']);self.assertTrue(result['live_bootstrap_reproduced'])
        self.assertEqual((result['ordinary_return'],result['CMS20_return']),(219,323))
        self.assertEqual(result['checked_native_calls'],156);self.assertEqual(result['raw_runtime_maps_verified'],3)
        self.assertFalse(result['failed_DP_tensor_outputs_read']);self.assertEqual(result['external_binary_reads'],0)
    def test_changed_scientific_source_rejected_before_compilation(self):
        def changed(name):
            raw=self.raw[name]
            return raw+b'\n# altered\n' if name=='science/checked_trace.py' else raw
        with self.assertRaisesRegex(ValueError,'Changed retained bytes'):
            self.replay(changed)
    def test_exact_accepted_mode4_join_rejects_changed_ancestor_raw_trace(self):
        ancestor=self.ancestors[self.worker.MATCH];index,data,document=ancestor
        target='results/runs/cms-p20-mode4-lib9-matched/trace.dat'
        def changed(name):return data(name)+b'\n' if name==target else data(name)
        ancestors={**self.ancestors,self.worker.MATCH:(index,changed,document)}
        with self.assertRaisesRegex(ValueError,'checked comparison does not reproduce'):
            self.replay(ancestors=ancestors)
    def test_document_callback_cannot_disagree_with_authenticated_bytes(self):
        def wrong(name):
            row=json.loads(self.raw[name])
            if name=='results1/run-closure.json':row['passed']=True
            return row
        with self.assertRaisesRegex(ValueError,'Byte/document callback differs'):
            self.replay(document=wrong)
    def test_raw_map_parser_rejects_unadmitted_path_even_with_matching_map_hash(self):
        prefix='results2/runs/cms-p20-auto-lib9-checked/'
        native=json.loads(self.raw[prefix+'result.json']);spec=json.loads(self.raw[prefix+'specification.json'])
        capture=copy.deepcopy(native['runtime_maps'][0]);raw=self.raw[prefix+capture['file']]
        path=spec['executable'].encode();self.assertIn(path,raw)
        altered=raw.replace(path,b'/unadmitted/native/path')
        capture['sha256']=self.worker.pin(altered)['sha256']
        with self.assertRaisesRegex(ValueError,'Unadmitted/deleted raw mapping'):
            self.worker.verify_maps(altered,capture,spec['expected_runtime'])


if __name__=='__main__':unittest.main()
