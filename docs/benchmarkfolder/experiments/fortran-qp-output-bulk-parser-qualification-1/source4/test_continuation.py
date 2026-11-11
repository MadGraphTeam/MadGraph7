"""Pure retained-output, narrow correction and no-overlap negative tests."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent;ROOT=Path.cwd()
def load(name):
    spec=importlib.util.spec_from_file_location('bulk_continuation_tests_'+name,HERE/(name+'.py'))
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj
continuation=load('continuation');corrected=load('analyze_rejection');original=load('analyze')


class ContinuationTests(unittest.TestCase):
    def test_all_eighty_eight_retained_cases_replay_without_new_native(self):
        built=json.loads((ROOT/'.tirust-build/fortran-qp-output-bulk-unit-work-3/build/result.json').read_text())
        result=continuation.replay(built,continuation.RUN_SHA)
        self.assertEqual(result['replayed_case_count'],88)
        self.assertEqual(result['unchanged_old_comparisons'],87)
        self.assertEqual(len(result['retained_transcripts']),264)
        self.assertEqual(len(result['pending_case_ids']),47)
        self.assertEqual(result['new_native_executions'],0)
        self.assertEqual(result['old_failed_run_status_preserved'],'failed')
    def test_defined_native_error_correction_preserves_original_failure(self):
        case=next(c for c in json.loads((HERE/'corpus.json').read_text())['cases'] if c['id']=='options-f0-p3')
        folder=ROOT/'.tirust-build/fortran-qp-output-bulk-unit-work-3/run'/case['id']
        pair={v:{'text':(folder/(v+'.stdout.txt')).read_text(),
                 'terminal':json.loads((folder/(v+'.terminal.json')).read_text())}
              for v in ['original','candidate','observer']}
        with self.assertRaisesRegex(ValueError,'Public status/round/halting mismatch'):
            original.compare(case,pair)
        self.assertEqual(corrected.compare(case,pair)['status'],'pass')
        self.assertFalse((folder/'comparison.json').exists())
    def test_incomplete_old_output_rejected(self):
        case=next(c for c in json.loads((HERE/'corpus.json').read_text())['cases'] if c['id']=='options-f0-p3')
        folder=ROOT/'.tirust-build/fortran-qp-output-bulk-unit-work-3/run'/case['id']
        pair={v:{'text':(folder/(v+'.stdout.txt')).read_text().replace('DONE\n',''),
                 'terminal':json.loads((folder/(v+'.terminal.json')).read_text())}
              for v in ['original','candidate','observer']}
        with self.assertRaisesRegex(ValueError,'Incomplete successful driver'):
            corrected.compare(case,pair)
    def test_suffix_overlap_or_case_reordering_rejected(self):
        manifest=json.loads((HERE/'continuation-inputs.json').read_text())
        corpus=json.loads((HERE/'corpus.json').read_text())['cases']
        bad=copy.deepcopy(manifest);bad['pending_case_ids'][0]=bad['prefix_case_ids'][0]
        with self.assertRaises(ValueError):continuation.partition(bad,corpus)
        bad=copy.deepcopy(manifest);bad['prefix_case_ids'][0:2]=reversed(bad['prefix_case_ids'][0:2])
        with self.assertRaises(ValueError):continuation.partition(bad,corpus)
    def test_changed_reviewed_run_rejected_before_replay(self):
        with self.assertRaisesRegex(ValueError,'Exact reviewed failed RUN3'):
            continuation.replay({},'0'*64)


if __name__=='__main__':unittest.main(verbosity=2)
