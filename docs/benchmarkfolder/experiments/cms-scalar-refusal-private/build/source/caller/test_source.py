#!/usr/bin/env python3
"""Pure Python/source tests; synthetic journal text is never a native result."""
import ast
import copy
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT))
import prepare as p
import prepare_fixture as f
import validate_fixture as v
from madgraph.iolibs import file_writers
from madgraph.various import misc

class CheckedSource(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prepared=json.loads((HERE/'prepared.json').read_text())
        cls.plan=json.loads((HERE/'fixture/commands.json').read_text())
        cls.cut=(HERE/'src/cts_checked_reduction.f90').read_text()
        cls.loops=(HERE/'src/cts_checked_loopfunctions.f90').read_text()

    def test_01_exact_source_hashes(self):
        for name,digest in self.prepared['inputs'].items():self.assertEqual(p.sha(ROOT/name),digest,name)
        for name,digest in self.prepared['outputs'].items():self.assertEqual(p.sha(HERE/name),digest,name)
        for name,digest in self.plan['inputs'].items():self.assertEqual(p.sha(ROOT/name),digest,name)
        for path in HERE.glob('*.py'):ast.parse(path.read_text())

    def test_02_actual_complete_compute_and_stability_blocks_unchanged(self):
        original=(p.CTS/'cts_cutroutines.f90').read_text()
        for name in ('computeamp','mpcomputeamp','dptest','mptest','from_dir_dp','from_inv_dp','from_dir_mp','from_inv_mp'):
            self.assertEqual(p.extract_subroutine(original,name),p.extract_subroutine(self.cut,name))

    def test_03_every_scalar_boundary_is_checked_before_consumers(self):
        calls=list(re.finditer(r'    call (getloop_checked|get_mp_loop_checked)\(number_propagators,(-?1),scalar_report\)',self.cut))
        self.assertEqual([(m[1],m[2]) for m in calls],[('getloop_checked','1'),('getloop_checked','-1'),('get_mp_loop_checked','1'),('get_mp_loop_checked','-1')])
        for call in calls:
            following=self.cut[call.end():]
            boundary=following.index('    call get_coefficients')
            guard=following[:boundary]
            self.assertIn('cms_report_accepted(scalar_report)',guard)
            self.assertIn('goto 9000',guard)
            self.assertIn('call note_refusal(scalar_report)',guard)
        self.assertEqual(self.cut.count('goto 2000'),2)
        self.assertLess(self.cut.index('call poison_result()'),self.cut.index('allocate  (bbn1'))
        tail=self.cut[self.cut.index('9000 continue'):self.cut.index('  contains')]
        self.assertIn('ct_status%result_valid=.false.',tail)
        self.assertIn('call poison_result()',tail)
        self.assertIn('call cts_checked_reset()',tail)
        for array in ('bbn1','bbn2','bbn3','bbn4'):self.assertIn('deallocate  ('+array,tail)

    def test_04_checked_scalar_signatures_publish_only_after_success(self):
        for precision,prefix in [(8,''),(16,'get_mp_')]:
            for topo,letter in enumerate('abcd',1):
                name=('get'+letter+'loop' if precision==8 else 'get_mp_'+letter+'loop')+'_checked'
                source=p.extract_subroutine(self.loops,name)
                self.assertNotIn('call olo(',source)
                self.assertNotIn('scaloop.eq.3',source)
                call=re.search(r'call (?:dp|qp)_checked_\w+\(.*report\)',source)
                self.assertIsNotNone(call)
                check=source.index('if (.not.cms_report_accepted(report))')
                publication=re.search(r'(?m)^     (?:mp_)?[abcd]\w*loopfun\(:,ib\)',source)
                if publication is None:publication=re.search(r'(?m)^     (?:mp_)?[abcd]loopfun\(:,ib\)',source)
                self.assertIsNotNone(publication,name)
                self.assertLess(call.start(),check);self.assertLess(check,publication.start())
                self.assertIn('report%direction=direction',source);self.assertIn('report%subtopology=ib',source)
        self.assertIn('report%status.eq.CMS_OK.and.report%result_valid.and.report%native_called',self.loops)

    def test_05_direction_entry_invalidates_and_no_success_cache_added(self):
        for name,valid in [('getloop_checked','checked_dp_valid'),('get_mp_loop_checked','checked_qp_valid')]:
            source=p.extract_subroutine(self.loops,name)
            self.assertLess(source.index(valid+'=.false.'),source.index('call get'))
            self.assertGreater(source.index(valid+'=.true.'),source.rindex('if(.not.cms_report_accepted(report)) return'))
        self.assertNotIn('cache_insert',self.cut.lower()+self.loops.lower())
        self.assertIn('ct_status=cts_checked_report()',self.cut)

    def test_06_both_adapter_formats_and_private_helas_routes_render(self):
        paths=list((HERE/'templates').rglob('*.inc'))
        self.assertEqual(len(paths),3)
        for path in paths:
            source=path.read_text()
            mapping={key:2 if fmt=='d' else '' for key,fmt in re.findall(r'%\((\w+)\)([ds])',source)}
            mapping.update(proc_prefix='FX_',real_dp_format='REAL*8',real_mp_format='REAL*16',complex_dp_format='COMPLEX*16',complex_mp_format='COMPLEX*32',mass_dp_format='COMPLEX*16')
            is_helper='helas' in path.name
            mapping.update(wfsargsdecl='I1,I2',wfsargs='I1,I2,',margs='M1,M2,',margsdecl='M1,M2',nwfsargs_header='',pairingargs='',pairingdecl='')
            mapping['finalize_CT']='\n'.join('RES(%d)=NORMALIZATION*RES(%d)'%(i,i) for i in range(1,4))
            text=misc.apply_template(source,mapping) if is_helper else source%mapping
            for amplitude_reduction in (False,True):
                with tempfile.TemporaryDirectory(dir=HERE,prefix='source-render-') as directory:
                    output=Path(directory)/'checked.f'
                    with file_writers.FortranWriter(str(output)) as stream:stream.writelines(text,context={'AmplitudeReduction':amplitude_reduction,'samurai_available':False,'ninja_available':False,'ninja_supports_quad_prec':False,'ninjars_available':False,'ninjars_supports_quad_prec':False,'tirust_supports_quad_prec':False})
                    rendered=output.read_text()
                    self.assertNotIn('%%',rendered)
                    self.assertTrue(all(len(line)<=132 for line in rendered.splitlines()))
                    if not is_helper:
                        self.assertIn('USE CTS_CHECKED_REDUCTION',rendered)
                        self.assertLess(rendered.index('CT_STATUS%STATUS.NE.CMS_OK'),rendered.index('RES(1)=NORMALIZATION'))
                    else:self.assertIn('_CMSCHECKED(',rendered)

    def test_07_mock_instrumentation_is_exactly_three_additions(self):
        mockcut=(HERE/'fixture/cts_checked_reduction_instrumented.f90').read_text()
        restored=mockcut.replace('  use mock_state,only:mock_count_amp\n','').replace('    call mock_count_amp(8)\n','').replace('    call mock_count_amp(16)\n','')
        self.assertEqual(restored,self.cut)
        self.assertEqual((HERE/'fixture/cms_scalar_observer.f90').read_bytes(),(HERE.parent/'classifier/cms_scalar_observer.f90').read_bytes())

    def test_08_case_matrix_and_mode0_semantics(self):
        cases=f.cases();self.assertEqual(len(cases),91)
        self.assertEqual({c['mode'] for c in cases},set(range(7)))
        self.assertEqual({c['fault_status'] for c in cases},set(range(9)))
        direct=f.expected(0,[[8,1,4,1]])
        self.assertTrue(direct['valid'] and direct['stable'] and direct['rescue'])
        self.assertEqual(direct['coefficient'],[0,1,0,1])
        inverse=f.expected(0,[[8,-1,4,1]])
        self.assertEqual(inverse['coefficient'],[1,1,0,1])
        terminal=f.expected(0,[[8,1,4,1],[16,-1,3,1]])
        self.assertFalse(terminal['valid']);self.assertEqual(terminal['refusals'],2)
        self.assertEqual(terminal['coefficient'],[0,1,0,0])
        malformed=f.expected(4,[[16,1,2,1]],fault_status=0)
        self.assertEqual(malformed['status'],6)

    def test_09_parser_accepts_only_complete_exact_synthetic_journal(self):
        data=json.loads((HERE/'fixture/cases.json').read_text())
        text='\n'.join(data['expected_journal_rows'])+'\n'
        self.assertTrue(v.validate_text(text,data)['passed'])
        broken=[text.rsplit('\n',2)[0]+'\n',text+'EXTRA\n',text.replace('COUNTS 1','COUNTS 9',1),text.replace('ADAPTER 0 1 F F 1','ADAPTER 0 1 T F 1',1)]
        for value in broken:
            self.assertNotEqual(value,text)
            with self.assertRaises(ValueError):v.validate_text(value,data)

    def test_10_compile_proposal_has_mock_only_link_and_no_native_outputs(self):
        self.assertEqual(len(self.plan['build_jobs']),8)
        self.assertEqual(self.plan['cases'],182)
        link=self.plan['build_jobs'][-1]['argv']
        self.assertTrue(all('avh_olo_checked' not in arg and '.a' not in arg for arg in link))
        self.assertFalse(list((HERE/'fixture/build').iterdir()))
        self.assertFalse(list((HERE/'fixture/run').iterdir()))
        plan=json.loads((HERE/'compile-plan.json').read_text())
        self.assertEqual(len(plan['jobs']),11)
        self.assertFalse(plan['actual_objects_linked'])
        self.assertFalse(list((HERE/'actual-build').iterdir()))
        self.assertNotIn(str(HERE/'fixture/build'),plan['module_search_order']['actual'])

def main():
    assert Path.cwd().resolve()==ROOT
    log=io.StringIO()
    with mock.patch.object(subprocess,'Popen',side_effect=AssertionError('Native subprocess forbidden')):
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CheckedSource))
    print(log.getvalue(),end='')
    if not result.wasSuccessful():raise SystemExit(1)
    summary=dict(scope='Source/Python only; synthetic parser journal only',tests=result.testsRun,
                 compile_performed=False,native_performed=False,test_sha256=p.sha(Path(__file__)),
                 prepared_sha256=p.sha(HERE/'prepared.json'),commands_sha256=p.sha(HERE/'fixture/commands.json'))
    p.write(HERE/'source-tests.json',json.dumps(summary,indent=2,sort_keys=True)+'\n')
    print(json.dumps(summary,sort_keys=True))

if __name__=='__main__':
    p.REFRESH='--refresh-results' in sys.argv
    main()
