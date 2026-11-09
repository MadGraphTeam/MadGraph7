"""Pure-Python accounting fixtures; they do not load a reducer or compile."""
import unittest
from analyze import analyze

class Accounting(unittest.TestCase):
    def journal(self, descriptions):
        events=[]
        for i,d in enumerate(descriptions,1):
            if d.get('clear'): events.append({'kind':'lifecycle','context':1,'action':'clear'})
            begin=dict(kind='parent_begin',parent=i,context=1,point=1,group=i,synthetic=False,
                       n=4,rank=3,poles=0,finite=1,check=0,cache=1)
            begin.update({k:v for k,v in d.items() if k in begin})
            events.extend([begin,dict(kind='store_state',parent=i,budget=d.get('budget',100000),
                qp_bytes=0,dp_bytes=0,pole_bytes=0),
                dict(kind='scalar_begin',parent=i,id=i,key='0102',key_bytes=2,key_ns=1,
                     input_finite=True,family=1,partial=False),
                dict(kind='scalar_end',parent=i,id=i,kernel_ns=3,success=d.get('scalar_success',True),
                     output_finite=True,output_payload_estimate=48),
                dict(kind='parent_end',parent=i,status=d.get('status',0))])
        return dict(events=events,uniqueness_complete=True,scalar_requests=len(descriptions),
                    scalar_completed=len(descriptions),dropped_events=0,key_flag_changes=0,output_flag_changes=0)

    def test_parent_failure_excludes_pending(self):
        result=analyze(self.journal([{'status':1},{},{}]))['counts']
        self.assertEqual(result['failed_parent_candidates_discarded'],1)
        self.assertEqual(result['repeat_of_prior_successful_eligible_parent'],1)
        self.assertEqual(result['simulated_committed_unique_keys'],1)

    def test_public_transport_failure_excludes_pending(self):
        data=self.journal([{},{}]);data['public_status_by_parent']={'1':2,'2':0}
        result=analyze(data)['counts']
        self.assertEqual(result['failed_parent_candidates_discarded'],1)
        self.assertNotIn('repeat_of_prior_successful_eligible_parent',result)

    def test_check_and_clear_are_isolated(self):
        result=analyze(self.journal([{}, {'check':1}, {'clear':True}, {}]))['counts']
        self.assertEqual(result['check_or_cacheoff_candidates_excluded'],1)
        self.assertEqual(result['repeat_of_prior_successful_eligible_parent'],1)
        self.assertEqual(result['simulated_committed_unique_keys'],2)

    def test_cacheoff_and_budget_refuse(self):
        result=analyze(self.journal([{'cache':0},{'budget':0},{'budget':0}]))['counts']
        self.assertEqual(result['check_or_cacheoff_candidates_excluded'],1)
        self.assertEqual(result['simulated_budget_refusals'],2)
        self.assertNotIn('repeat_of_prior_successful_eligible_parent',result)

    def test_partial_and_synthetic_have_separate_namespaces(self):
        result=analyze(self.journal([{}, {'finite':0}, {'synthetic':True}, {}]))['counts']
        self.assertEqual(result['simulated_committed_unique_keys'],3)
        self.assertEqual(result['repeat_of_prior_successful_eligible_parent'],1)

    def test_scalar_error_never_commits(self):
        result=analyze(self.journal([{'scalar_success':False},{}]))['counts']
        self.assertEqual(result['scalar_error_or_nonfinite_excluded'],1)
        self.assertNotIn('repeat_of_prior_successful_eligible_parent',result)

    def test_modes_share_one_context_budget(self):
        result=analyze(self.journal([{'budget':500},{'budget':500,'finite':0}]))['counts']
        self.assertEqual(result['simulated_committed_unique_keys'],1)
        self.assertEqual(result['simulated_budget_refusals'],1)

    def test_capped_journal_retains_totals(self):
        data=self.journal([{}]);data.update(uniqueness_complete=False,dropped_events=6,scalar_requests=100)
        data['events'].append(dict(kind='parent_end',parent=99,status=0))
        result=analyze(data)
        self.assertFalse(result['uniqueness_complete'])
        self.assertEqual(result['raw_total_scalar_requests'],100)
        self.assertEqual(result['orphan_events_from_cap'],1)
        self.assertFalse(result['opportunity_assessment_allowed'])
        self.assertIsNone(result['counts']['simulated_committed_unique_keys'])

if __name__=='__main__': unittest.main()
