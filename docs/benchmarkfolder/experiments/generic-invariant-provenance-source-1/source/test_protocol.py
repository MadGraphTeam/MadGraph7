"""Pure graph/source tests; no compiler, loader, reducer or amplitude execution."""
import importlib
import importlib.util
import copy
import json
from pathlib import Path
import re
import sys
import unittest

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
import madgraph.loop
import madgraph.iolibs
madgraph.loop.__path__.insert(0,str(HERE/'src/madgraph/loop'))
madgraph.iolibs.__path__.insert(0,str(HERE/'src/madgraph/iolibs'))
from madgraph.loop import invariant_provenance as p,invariant_render as r,invariant_finalize as f


def fixture(e=4,n=4):
    leaves={i+7:dict(slot=i+7,leg=i+1,mass='ZERO',offshell=False,
         state='initial' if i<2 else 'final',state_id=-1 if i<2 else 1,
         spin=1,is_part=True) for i in range(e)}
    graph=dict(nexternal=e,ninitial=2,has_born=True,leaves=leaves,
               legs={i+1:dict(offshell=False,state=i>=2,pdg=1) for i in range(e)},
               conflicting_slots=[],process='synthetic structure')
    data=dict(graph=graph,routes=[dict(route=1,n=n,we=list(leaves)[:n],
                                      pairing=[1]*n,loopnum=17,rank=6)])
    def block(name,width,output,input='P',internal=False):
        if internal:return {'arguments':['A','B','OUT'], 'statements':[(1,'TYPE(ALOHA)A',[]),(2,'TYPE(ALOHA)B',[]),(3,'TYPE(ALOHA)OUT',[]),(4,'OUT%P(:)=+A%P(:)-B%P(:)',[])]}
        return {'arguments':['P','M','NHEL','NSS','OUT'],'statements':
            [(1,('REAL*16' if width==128 else 'REAL*8')+'P(0:3)',[]),
             (2,'TYPE('+('MP_ALOHA' if width==128 else 'ALOHA')+')OUT',[])]+
             [(3+mu,f'OUT%P({mu})=P({mu})*NSS',[]) for mu in range(4)]}
    routines={};batches=[]
    for width in (64,128):
        name=('MP_' if width==128 else '')+'VXXXXX';routines[name]=block(name,width,'OUT')
        text=''
        for i,slot in enumerate(leaves):
            text+='      CALL '+name+'(P(0,%d),'%(i+1)+('MP__ZERO' if width==128 else 'ZERO')+',1,%d,W(%d))\n'%(-1 if i<2 else 1,slot)
        batches.append([('producer.f',text)])
    return data,batches,routines


class Protocol(unittest.TestCase):
    def test_generic_geometry_all_n_modes_e(self):
        for e in range(3,7):
            c=[-1,-1]+[1]*(e-2)
            for n in range(1,7):
                xs=[p.Expr(tuple(1 if i==j%e else 0 for i in range(e)),tuple(1 if i==j%e else 0 for i in range(e)),0) for j in range(n)]
                for mode in p.MODES:
                    got=p.geometry(xs,mode,c,[True]*e)
                    self.assertEqual(got['chords'][0]['ops'],n)
                    self.assertEqual(sum(got['chords'][0]['mult']),n)
                    self.assertEqual(got['pair_class'],list(map(list,zip(*got['pair_class']))))
                    expected=xs if mode in (1,4) or n<=2 else [x.neg() for x in reversed(xs)]
                    self.assertEqual(got['corners'],[x.record() for x in expected])
    def test_unsigned_cancellation(self):
        x=p.Expr((1,0,0),(1,0,0),0)
        y=p.seeded([x,x.neg()],3)
        self.assertEqual(y.signed,(0,0,0));self.assertEqual(y.mult,(2,0,0));self.assertEqual(y.ops,2)
    def test_protected_unknown_composite(self):
        c=[-1,-1,1,1]
        self.assertEqual(p.classify([0,0,0,0],c,[False]*4),1)
        self.assertEqual(p.classify([0,0,1,0],c,[True]*4),1)
        self.assertEqual(p.classify([0,0,1,0],c,[False]*4),0)
        self.assertEqual(p.classify([0,0,1,1],c,[True]*4),2)
    def test_nonidentity_external_slot_and_mass_policy(self):
        d,b,routines=fixture()
        got=p.derive(d,*b,routines,{i:False for i in range(1,5)},{i:'ZERO' for i in range(1,5)})
        self.assertTrue(all(x['supported'] for x in got['routes'][0]['modes'].values()))
        self.assertEqual(got['routes'][0]['modes']['1']['external_slots'],[7,8,9,10])
        self.assertTrue(all(got['protected']))
        d['graph']['legs'][3]['offshell']=True
        got=p.derive(d,*b,routines,{i:False for i in range(1,5)},{i:'ZERO' for i in range(1,5)})
        self.assertFalse(got['protected'][2])
    def test_unknown_custom_mixed_and_overwrite(self):
        for tamper in ('kind','nonlinear','duplicate','slot','mass'):
            d,b,routines=fixture()
            if tamper=='kind': routines['VXXXXX']['statements'][1]=(2,'TYPE(MP_ALOHA)OUT',[])
            if tamper=='nonlinear':routines['VXXXXX']['statements'][2]=(3,'OUT%P(0)=SQRT(P(0))',[])
            if tamper=='duplicate':routines['VXXXXX']['statements'].append((8,'OUT%P(0)=P(0)*NSS',[]))
            if tamper=='slot':d['graph']['leaves'][7]['leg']=4
            if tamper=='mass':b[0][0]=('producer.f',b[0][0][1].replace('ZERO','MOTHER_MASS'))
            got=p.derive(d,*b,routines,{i:False for i in range(1,5)},{i:'ZERO' for i in range(1,5)})
            self.assertFalse(got['routes'][0]['modes']['1']['supported'],tamper)
    def test_internal_unsigned_dag(self):
        d,b,routines=fixture()
        routines['SUM']={'arguments':['A','B','OUT'],'statements':[(1,'TYPE(ALOHA)A',[]),(2,'TYPE(ALOHA)B',[]),(3,'TYPE(ALOHA)OUT',[]),(4,'OUT%P(:)=+A%P(:)-B%P(:)',[])]}
        b[0][0]=('producer.f',b[0][0][1]+'      CALL SUM(W(7),W(7),W(20))\n')
        d['routes'][0]['we'][0]=20
        got=p.derive(d,*b,routines,{i:False for i in range(1,5)},{i:'ZERO' for i in range(1,5)})
        corner=got['routes'][0]['modes']['1']['corners'][0]
        self.assertEqual(corner['signed'],[0]*4);self.assertEqual(corner['mult'],[2,0,0,0]);self.assertEqual(corner['ops'],2)
    def test_tables_orientation(self):
        d,b,routines=fixture();got=p.derive(d,*b,routines,{i:False for i in range(1,5)},{i:'ZERO' for i in range(1,5)})
        emitted=p.emit_tables(got)
        self.assertIn('route_mult(4,4,1,4)',emitted)
        self.assertIn('route_external(4,1,4)',emitted)
        self.assertNotIn('26',emitted.splitlines()[1])
    def test_state_capture_before_producers_skip_unknown(self):
        t=(HERE/'base/madgraph/iolibs/template_files/loop_optimized/loop_matrix_standalone.inc').read_text();v=r.main_template(t)
        self.assertLess(v.index('GP_FRESH_DP=.FALSE.',v.index('DO H=1,NCOMB')),v.index('IF (SKIP_LOOPNUM_COEFS_CONSTRUCTION)',v.index('DO H=1,NCOMB')))
        cap=v.index('GP_ENV_DP=READ_KINEMATIC_ENVIRONMENT_TOKEN()')
        self.assertLess(cap,v.index('%(born_ct_helas_calls)s'))
        self.assertEqual(v.count('GP_FRESH_DP=.FALSE.\n  CALL %(proc_prefix)sMP_COMPUTE_LOOP_COEFS_PROVENANCE'),2)
        self.assertIn('5001 CONTINUE\nENDDO\n## }\nGP_FRESH_DP=.FALSE.\nGP_FRESH_QP=.FALSE.',v)
    def test_mp_fresh_actual_branch_not_widened_or_skip(self):
        t=(HERE/'base/madgraph/iolibs/template_files/loop_optimized/mp_compute_loop_coefs.inc').read_text();v=r.mp_template(t)
        self.assertEqual(v.count('GP_ENV=READ_KINEMATIC_ENVIRONMENT_TOKEN()'),1)
        self.assertIn('IF (COMPUTE_INTEGRAND_IN_QP) THEN\n    GP_ENV=',v)
        self.assertLess(v.index('GP_FRESH=.FALSE.'),v.index('IF (SKIP_LOOPNUM_COEFS_CONSTRUCTION)'))
        self.assertIn('GP_FRESH=.NOT.FILTER_SO',v)
        self.assertIn('SUBROUTINE %(proc_prefix)sMP_COMPUTE_LOOP_COEFS(PS,ANSDP)',v)
    def test_factory_formula_only_generic_dimensions(self):
        src=(HERE/'src/madgraph/iolibs/template_files/loop_optimized/invariant_bounds.f90').read_text()
        self.assertNotIn('route>26',src);self.assertNotIn('external_bits(2,0:3,4)',src)
        self.assertIn('err(1)=1+clog2(L)-p+total',src)
        self.assertIn('terms=[chord_error(mu,i),chord_error(mu,j),emin-p,zero_bound]',src)
    def test_conditional_alias_or_foreign_call_falls_back(self):
        for tamper in ('conditional','alias','call','input'):
            d,b,routines=fixture();routine=routines['VXXXXX']
            if tamper=='conditional':
                routine['statements'].insert(2,(2,'IF(NSS.EQ.1)THEN',[]))
                routine['statements'].append((9,'ENDIF',[]))
            if tamper=='alias':routine['statements'].insert(0,(0,'EQUIVALENCE(A,B)',[]))
            if tamper=='call':routine['statements'].append((9,'CALLCUSTOM(P)',[]))
            if tamper=='input':routine['statements'].append((9,'OTHER%P(0)=0',[]))
            got=p.derive(d,*b,routines,{i:False for i in range(1,5)},{i:'ZERO' for i in range(1,5)})
            self.assertFalse(got['routes'][0]['modes']['1']['supported'],tamper)
    def test_immutable_table_fortran_column_major_roundtrip(self):
        d,b,routines=fixture();got=p.derive(d,*b,routines,{i:False for i in range(1,5)},{i:'ZERO' for i in range(1,5)})
        emitted=p.emit_tables(got)
        m=re.search(r'route_mult\(([^)]*)\)=reshape\(\[ &(.*?)\],\[([^]]*)\]\)',emitted,re.S)
        shape=list(map(int,m[1].split(',')))
        self.assertEqual(shape,[4,4,1,4])
        vals=list(map(int,m[2].replace('&','').replace('\n','').split(',')))
        for mode_index,mode in enumerate(p.MODES):
            for chord in range(4):
                for leaf in range(4):
                    self.assertEqual(vals[leaf+4*(chord+4*mode_index)],got['routes'][0]['modes'][str(mode)]['chords'][chord]['mult'][leaf])
    def test_register_without_active_registry_leaves_call_exact(self):
        class Empty:pass
        text='CALL %(proc_prefix)sLOOP_3(1,2,3,DCMPLX(M),DCMPLX(M),DCMPLX(M),2,I_SO,7)'
        self.assertEqual(p.register(Empty(),None,text),text)
    def test_missing_custom_or_indirect_environment_path_is_unknown(self):
        root='      SUBROUTINE ROOT()\n      CALL CHILD()\n      END\n'
        child='      SUBROUTINE CHILD()\n      END\n'
        raw={'root.f':root.encode(),'child.f':child.encode()}
        self.assertTrue(f.environment_closure(raw,[('root.f',root)],True)['call_graph_complete'])
        self.assertFalse(f.environment_closure(raw,[('root.f',root)],True)['complete'])
        self.assertFalse(f.environment_closure(raw,[('root.f',root)],False)['complete'])
        self.assertFalse(f.environment_closure({'root.f':root.encode()},[('root.f',root)],True)['complete'])
        raw['child.f']=child.replace('      END','      EXTERNAL FOREIGN\n      END').encode()
        self.assertFalse(f.environment_closure(raw,[('root.f',root)],True)['complete'])
    def test_unknown_route_shape_and_custom_graph_fallback(self):
        for variant in ('pairing','range','custom'):
            d,b,rs=fixture()
            if variant=='pairing':d['routes'][0]['pairing']=[1,1,1,2]
            if variant=='range':d['routes'][0]['n']=7
            if variant=='custom':d['graph']['unsupported_source_features']=['foreign form factor']
            got=p.derive(d,*b,rs,{i:False for i in range(1,5)},{i:'ZERO' for i in range(1,5)})
            self.assertTrue(all(not x['supported'] for x in got['routes'][0]['modes'].values()))
    def test_implicit_expression_function_cannot_enable_environment(self):
        source='      SUBROUTINE ROOT()\n      REAL F,X\n      X=F()\n      END\n'
        function='      REAL FUNCTION F()\n      CALL FOREIGN_FENV_HELPER()\n      F=0\n      END\n'
        info=f.environment_closure({'root.f':source.encode(),'f.f':function.encode()},[('root.f',source)],True)
        # Deliberately demonstrates why CALL-only diagnostics are insufficient.
        self.assertTrue(info['call_graph_complete'])
        self.assertFalse(info['function_reference_closure_qualified'])
        self.assertFalse(info['complete'])
        d,b,rs=fixture();got=p.derive(d,*b,rs,{i:False for i in range(1,5)},{i:'ZERO' for i in range(1,5)})
        got['environment_contract']={'complete':True}
        self.assertIn('source_environment_supported=.false.',p.emit_tables(got))
    def test_unknown_template_rejected(self):
        with self.assertRaises(ValueError):r.batch_template('custom arbitrary routine')

if __name__=='__main__':unittest.main()
