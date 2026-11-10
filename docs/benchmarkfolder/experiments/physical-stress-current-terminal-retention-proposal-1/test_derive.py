"""Mock protocol/accounting tests only; these are not scientific observations."""
from pathlib import Path
import copy
import importlib.util
import json
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('stress_terminal_derive', HERE / 'derive.py')
D = importlib.util.module_from_spec(spec); spec.loader.exec_module(D)
SOURCE = HERE.with_name('physical-stress-current-source-proposal-1')
PLAN = json.loads((SOURCE / 'proposal.json').read_bytes())
PROTOCOL = D.analyzer((SOURCE / 'protocol/tirust_benchmark_worker.py').read_bytes(),
                      (SOURCE / 'protocol/tirust_benchmark.py').read_bytes())


def record(request, *, code=None, warning='', coefficient='2', raw_edit=None):
    code = code or {1:211, 6:216, 7:217, 8:218, 9:219}[request['backend']]
    raw = {'timing.dat':'1 1\n', 'dimensions.dat':'0 0 0\n',
           'values.dat': ''.join(f'{point} 0 {code} 1 {coefficient} -3 4 1e-8\n' for point in (1,2)),
           'timed_status.dat':f'{code} 2\n',
           'warmup.dat':'# call point order status Born finite single double accuracy\n0 1 0 131 1 2 -3 4 -1\n1 1 0 '+str(code)+' 1 2 -3 4 1e-8\n2 2 0 '+str(code)+' 1 2 -3 4 1e-8\n'}
    if raw_edit: raw_edit(raw)
    parsed = None; failure = None
    try: parsed = PROTOCOL[0](raw, request['backend'])
    except Exception as error: failure = {'class':type(error).__name__, 'message':str(error)}
    warnings = [line for line in warning.splitlines() if '##W03' in line]
    good = parsed is not None and failure is None and not warnings and all(r['accuracy'] <= 1e-6 for r in parsed['rows'])
    result = {'request':request, 'observation':parsed, 'failure':failure, 'mapping_errors':[],
              'incomplete_warnings':warnings, 'complete_runtime_map':True, 'numerically_qualified':good}
    completion = {'result':result, 'outer_failure':None, 'fatal_infrastructure_failure':None}
    return {'raw':raw, 'stdout':warning, 'result':result, 'completion':completion,
            'request_completion':{'request':request, 'result':result}, 'runtime_verified':True}


def run(records, aborted=None, status='passed'):
    completed = {key for key, value in records.items() if value.get('request_completion') is not None}
    accounting = {'requested':24, 'completed':len(completed),
                  'not_completed':[r for r in PLAN['requests'] if D.key(r) not in completed]}
    return D.derive(PLAN, records, {'status':status}, accounting, aborted, PROTOCOL)


class DerivationTests(unittest.TestCase):
    def all_records(self):return {D.key(r):record(r) for r in PLAN['requests']}
    def test_complete_six_lane_consistency_is_not_independent_truth(self):
        result=run(self.all_records())
        self.assertEqual(result['host_checked_complete_streams'],24)
        self.assertEqual(len(result['comparisons']),60)
        self.assertTrue(result['six_lane_all_cases_agree'])
        self.assertFalse(result['independent_physical_accuracy_qualified'])
        self.assertFalse(result['EW_analytic_oracle_qualified'])
        self.assertFalse(result['performance_eligible'])
        self.assertEqual(result,json.loads(json.dumps(result,allow_nan=False)))
    def test_all24_unstarted_with_deferred_first_request(self):
        request=PLAN['requests'][0]
        result=run({}, {'request':request,'state':'deferred-before-launch','exception':'TimeoutError(lock)'},'failed')
        self.assertEqual(result['request_rows'][0]['state'],'deferred_before_launch')
        self.assertTrue(all(row['state']=='not_started' for row in result['request_rows'][1:]))
        self.assertEqual(result['completed_orchestration'],0)
    def test_controller_failure_preserves_partial_raw(self):
        request=PLAN['requests'][0];rid=D.key(request)
        records={rid:{'raw':{'values.dat':'1 0 420 nan 2 3 4 -1\n'},'stdout':'native refusal\n'}}
        result=run(records,{'request':request,'state':'aborted-during-request','exception':'RuntimeError(cleanup)'},'failed')
        row=result['request_rows'][0]
        self.assertEqual(row['state'],'controller_failed');self.assertEqual(row['nonfinite_rows'],1)
        self.assertEqual(row['final_trace']['rows'][0]['status'],420)
    def test_incomplete_w03_truncated120_never_compared(self):
        records=self.all_records();request=PLAN['requests'][0]
        records[D.key(request)]=record(request,code=120,warning='##W03 omitted loop contribution\n')
        result=run(records);row=result['request_rows'][0]
        self.assertFalse(row['coefficient_comparison_eligible']);self.assertEqual(row['state'],'physical_rejected')
        self.assertEqual(row['final_trace']['rows'][0]['raw_coefficients'],['1','2','-3','4'])
        self.assertEqual(row['final_trace']['rows'][0]['status'],120)
        self.assertTrue(row['warnings']);self.assertFalse(result['case_coverage'][0]['six_lane_agreement'])
        self.assertEqual(result['case_coverage'][0]['compared_pairs'],10)
    def test_nonfinite_tokens_retained_without_nonfinite_json(self):
        records=self.all_records();request=PLAN['requests'][0]
        records[D.key(request)]=record(request,coefficient='NaN')
        result=run(records);row=result['request_rows'][0]
        self.assertEqual(row['nonfinite_rows'],2)
        self.assertEqual(row['final_trace']['rows'][0]['nonfinite_tokens'],['NaN'])
        json.dumps(result,allow_nan=False)
    def test_duplicate_final_point_fails_complete_stream(self):
        records=self.all_records();request=PLAN['requests'][0]
        records[D.key(request)]=record(request,raw_edit=lambda raw:raw.update({'values.dat':raw['values.dat'].replace('2 0 ','1 0 ')}))
        result=run(records);self.assertFalse(result['request_rows'][0]['host_checked_complete_stream'])
    def test_accepted_qp_precision_is_decoded_from_full_status(self):
        records=self.all_records();request=next(r for r in PLAN['requests'] if r['backend']==9)
        records[D.key(request)]=record(request,code=323)
        result=run(records);row=next(r for r in result['request_rows'] if r['request_id']==D.key(request))
        self.assertEqual(row['accepted_precision_counts'],{'9:QP':2})
        self.assertEqual(row['warmup_calls_observed'],3)
        self.assertEqual(row['qualified_final_points'],[1,2])
    def test_checked_streams_can_disagree(self):
        records=self.all_records();request=PLAN['requests'][1]
        records[D.key(request)]=record(request,coefficient='2.01')
        result=run(records)
        self.assertEqual(result['host_checked_complete_streams'],24)
        self.assertFalse(result['case_coverage'][0]['six_lane_agreement'])
    def test_missing_accounting_does_not_become_passing_subset(self):
        with self.assertRaisesRegex(ValueError,'complete orchestration'):
            run({D.key(PLAN['requests'][0]):record(PLAN['requests'][0])})
    def test_forged_saved_qualification_rejected(self):
        records=self.all_records();request=PLAN['requests'][0]
        records[D.key(request)]['result']['numerically_qualified']=False
        with self.assertRaisesRegex(ValueError,'Saved numerical qualification differs'):run(records)
    def test_warning_scan_not_saved_boolean_controls_admission(self):
        records=self.all_records();records[D.key(PLAN['requests'][0])]['stdout']='##W03 omitted\n'
        with self.assertRaisesRegex(ValueError,'W03 scan differs'):run(records)
    def test_outer_timeout_cannot_preserve_inner_success(self):
        records=self.all_records();request=PLAN['requests'][0];current=records[D.key(request)]
        failed=copy.deepcopy(current['result']);failed['numerically_qualified']=False
        current['completion']={'result':failed,'outer_failure':'TimeoutError(inclusive cap)','fatal_infrastructure_failure':None}
        current['request_completion']={'request':request,'result':failed}
        result=run(records);row=result['request_rows'][0]
        self.assertFalse(row['host_checked_complete_stream']);self.assertEqual(row['qualified_final_points'],[])
        self.assertIn('TimeoutError',row['controller_failure'])
    def test_warmup_only_failure_does_not_count_final_points(self):
        records=self.all_records();request=PLAN['requests'][0];current=records[D.key(request)]
        current['raw']={'warmup.dat':'0 1 0 420 NaN 2 3 4 -1\n'}
        current['result']['observation']=None;current['result']['numerically_qualified']=False
        current['result']['failure']={'class':'MeasurementError','message':'Numerical warmup failure'}
        result=run(records);row=result['request_rows'][0]
        self.assertEqual(row['warmup_calls_observed'],1)
        self.assertEqual(row['raw_final_points_observed'],[])
        self.assertEqual(row['qualified_final_points'],[])


if __name__=='__main__':unittest.main(verbosity=2)
