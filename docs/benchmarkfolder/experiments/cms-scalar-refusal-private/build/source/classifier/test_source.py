#!/usr/bin/env python3
"""Pure-Python checks of actual classifier extraction and mock-source coverage."""
import hashlib
import importlib.util
import json
from decimal import Decimal
from pathlib import Path
import re
import unittest

HERE=Path(__file__).resolve().parent
ROOT=Path.cwd()


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class ClassifierSource(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec=importlib.util.spec_from_file_location('classifier_generator_tests',HERE/'generate.py')
        cls.g=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.g)
        cls.source=cls.g.SOURCE.read_text()
        cls.metadata=json.loads((HERE/'classification.json').read_text())
        cls.patched=(HERE/'generated/avh_olo_checked.f90').read_text()
        cls.mock=(HERE/'fixture/mock_hosts.f90').read_text()

    def test_bound_actual_source(self):
        self.assertEqual(sha(self.g.SOURCE),self.metadata['source_sha256'])
        self.assertEqual(sha(HERE/'cms_scalar_observer.f90'),self.metadata['observer_sha256'])
        for path,digest in self.metadata['generated_sha256'].items():self.assertEqual(sha(HERE/path),digest)

    def test_16_actual_overloads(self):
        self.assertEqual(len(self.metadata['supported_overloads']),16)
        self.assertEqual({(r['precision'],r['name']) for r in self.metadata['supported_overloads']},
                         {(p,n) for p in ('dp','qp') for n in self.g.NAMES})

    def test_original_wrappers_unchanged(self):
        for r in self.metadata['supported_overloads']:
            old,_=self.g.extract(self.source,r['precision'],r['name'])
            new,_=self.g.extract(self.patched,r['precision'],r['name'])
            self.assertEqual(old,new)

    def test_exact_erasure_site_coverage(self):
        count=0
        for r in self.metadata['supported_overloads']:
            old,_=self.g.extract(self.source,r['precision'],r['name'])
            original_sites=[s for _,_,_,s in self.g.logical_lines(old) if s and re.search(r'\.lt\.hh\).*?=\s*0\s*$',s,re.I)]
            self.assertEqual(original_sites,[event['original_statement'] for event in r['events']])
            new=(HERE/f'generated/{r["precision"]}_{r["name"]}_preflight.f90').read_text()
            self.assertEqual(new.count('call cms_record('),len(original_sites))
            for event in r['events']:
                self.assertIn(f'call cms_record(report,{event["line"]},',new)
                self.assertEqual(self.source.splitlines()[event['line']-1].strip(),event['original_statement'])
            count+=len(original_sites)
        self.assertEqual(count,64)

    def test_native_arithmetic_and_permutations_retained(self):
        for r in self.metadata['supported_overloads']:
            generated=self.g.generate_routine(self.source,r['precision'],r['name'],self.g.NAMES.index(r['name'])+1)
            actual=(HERE/f'generated/{r["precision"]}_{r["name"]}_preflight.f90').read_text()
            self.assertEqual(actual,generated['preflight']+'\n')
            for line in generated['original'].splitlines():
                if re.match(r'\s*(?:ss\(\d\)=pp\(|rr\(\d\)=mm\(|s\dr\d\s*=\s*abs)',line):
                    self.assertIn(line,actual)
            self.assertNotRegex(actual,r'\bcall\s+(?:tadp|bub11|tria\w*|trif\w*|box\w*|init|olo_onshell)\s*\(')

    def test_norms_are_not_unified(self):
        by={(r['precision'],r['name']):r for r in self.metadata['supported_overloads']}
        self.assertEqual(by['qp','a0_c']['events'][0]['norm'],'CMS_ABS_COMPLEX')
        self.assertEqual(by['qp','b11rc']['events'][1]['norm'],'CMS_L1_COMPLEX')
        self.assertTrue(all(e['norm']=='CMS_ABS_COMPLEX' for e in by['qp','c0rc']['events'] if e['kind']=='CMS_DIFFERENCE'))
        self.assertTrue(all(e['norm']=='CMS_L1_COMPLEX' for e in by['qp','d0rc']['events'] if e['kind']=='CMS_DIFFERENCE'))
        self.assertEqual(by['qp','a0_r']['events'][0]['norm'],'CMS_ABS_REAL')

    def test_every_checked_wrapper_gates_and_publishes_scratch(self):
        for r in self.metadata['supported_overloads']:
            text=(HERE/f'generated/{r["precision"]}_{r["name"]}_checked.f90').read_text()
            self.assertLess(text.index('if (report%status.ne.CMS_OK) return'),text.index('  call '+r['name']+'('))
            self.assertLess(text.index('CMS_NONFINITE_RESULT'),text.index('report%result_valid=.true.'))
            native=text[text.index('  call '+r['name']+'('):].splitlines()[0]
            self.assertIn('cms_result_1',native)
            self.assertIn('intent(inout)',text)

    def test_bounded_observer_and_underflow(self):
        text=(HERE/'cms_scalar_observer.f90').read_text()
        self.assertNotRegex(text.lower(),r'\b(?:allocate|save)\b')
        self.assertIn('CMS_MAX_EVENTS=32',text)
        self.assertIn('report%status=CMS_EVENT_OVERFLOW',text)
        self.assertIn('native_metric.eq.0.and.real(classified_value,16).eq.0.and.aimag(classified_value).eq.0',text)
        self.assertIn('report%events(i)%metric_underflowed=(native_metric.eq.0)',text)
        self.assertIn('report%events(i)%original_invariant=original_invariant',text)
        self.assertIn('report%events(i)%original_mass=original_mass',text)
        for r in self.metadata['supported_overloads']:
            generated=(HERE/f'generated/{r["precision"]}_{r["name"]}_preflight.f90').read_text()
            for event in r['events']:
                self.assertIn(event['original_invariant']+','+event['original_mass']+')',generated)

    def test_max_dynamic_events(self):
        for r in self.metadata['supported_overloads']:
            self.assertEqual(r['max_events'],{1:1,2:3,3:9,4:13}[r['topology']])
            self.assertLess(r['max_events'],32)

    def test_mock_each_overload_and_critical_cases(self):
        for p in ('dp','qp'):
            for n in self.g.NAMES:self.assertIn('! '+p+' '+n+': accepted callback receives original inputs.',self.mock)
        for token in ('CMS_EVENT_OVERFLOW','CMS_NONFINITE_RESULT','CMS_BAD_INPUT','CMS_UNINITIALIZED',
                      'CMS_DEFAULT_UNSUPPORTED','CMS_ZERO_TOPOLOGY_UNSUPPORTED','metric_underflowed',
                      'ieee_support_denormal','nearest(onshellthrs,-1._kindr2)','nearest(onshellthrs,1._kindr2)',
                      'transfer(p,p_words)','CMS_ABS_COMPLEX','CMS_DIFFERENCE','report%soft_permutation.eq.2',
                      'original_invariant.eq.cmplx(p(1),kind=16)','original_mass.eq.cmplx(cm(2),kind=16)'):
            self.assertIn(token,self.mock)

    def test_retained_physical_literals_bound(self):
        path=HERE.parent/'retained-inputs/point20-inputs.json'
        self.assertEqual(sha(path),'9dce4ad886df5d03363c1a7c602a2cd5d183dc003ce0c4d9c9ea4eb0382c1d6b')
        tokens={Decimal(token) for token in re.findall(r'[+-]?(?:\d*\.\d+)E[+-]\d+',path.read_text().upper())}
        for token in ('-2.499999999999791305007018193257151140E-7','-2.643850145484542582671511495203734939E-12'):
            # Compare exact decimal values across journal exponent normalization.
            self.assertIn(Decimal(token),tokens)
            self.assertIn(token+'_16',self.mock)

    def test_future_compile_plan_only(self):
        plan=json.loads((HERE/'compile-plan.json').read_text())
        self.assertEqual(plan['state'],'not_authorized_or_executed')
        self.assertEqual(plan['global_compile_link_seconds_max'],120)
        self.assertEqual(len(plan['jobs']),6)
        self.assertEqual(sum('-c' in j['argv'] for j in plan['jobs']),5)
        self.assertFalse(plan['actual_AVH_native_execution_allowed'])
        self.assertEqual(sha(plan['compiler']['path']),plan['compiler']['sha256'])
        for target,record in plan['copied_dependencies'].items():
            self.assertEqual(sha(target),record['sha256'])
            self.assertEqual(sha(record['source']),record['sha256'])
        actual=next(job for job in plan['jobs'] if job['name']=='compile-real-AVH-additions')
        self.assertNotIn('-I'+str(HERE/'build'),actual['argv'])
        self.assertIn('-I'+str(HERE/'observer-build'),actual['argv'])
        self.assertIn('-I'+str(HERE/'includes'),actual['argv'])
        self.assertFalse(any(Path(j['argv'][j['argv'].index('-o')+1]).exists() for j in plan['jobs']))


if __name__=='__main__':unittest.main(verbosity=2)
