"""Actual MG object/call/template generation. All external execution is blocked."""
from pathlib import Path
import contextlib
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import warnings
from unittest.mock import patch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
import madgraph.loop,madgraph.iolibs
madgraph.loop.__path__.insert(0,str(HERE/'src/madgraph/loop'))
madgraph.iolibs.__path__.insert(0,str(HERE/'src/madgraph/iolibs'))
from madgraph.loop import invariant_provenance as prov,invariant_finalize as final
from madgraph.loop import loop_diagram_generation as ld,loop_helas_objects as lh,loop_exporters as le
from madgraph.iolibs import helas_call_writers as hw,file_writers as fw
from madgraph.core import base_objects as bo
from madgraph.models import import_ufo
import aloha

WORK=HERE/'generation-work'

def deny(*args,**kwargs):raise AssertionError('external process forbidden in source generation')


def write_guard(event,args):
    if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
        mode=args[1];flags=args[2]
        writing=(isinstance(mode,str) and any(x in mode for x in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
        if writing:
            path=Path(os.fsdecode(args[0])).resolve()
            if not path.is_relative_to(HERE):raise AssertionError('write outside private proposal: '+str(path))
    if event in ('subprocess.Popen','os.system','os.exec','os.posix_spawn','ctypes.dlopen'):
        raise AssertionError('native/process forbidden: '+event)


class ActualGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        WORK.mkdir(exist_ok=True)
        model=WORK/'model/loop_sm';model.mkdir(parents=True,exist_ok=True)
        for x in (ROOT/'models/loop_sm').iterdir():
            if x.is_file() and x.suffix in ('.py','.dat'):shutil.copyfile(x,model/x.name)
        cls.model=import_ufo.import_model(str(model))
        cls.receipts=[]

    @classmethod
    def tearDownClass(cls):
        (HERE/'generation-results.json').write_text(json.dumps(cls.receipts,indent=2,sort_keys=True)+'\n')

    def export_case(self,name,pids,has_born,group,permute=False,group_number=None,element=None):
        target=WORK/name;sub=target/'SubProcesses/P0';sub.mkdir(parents=True,exist_ok=True)
        (target/'SubProcesses/MadLoop5_resources').mkdir(exist_ok=True)
        (target/'Source/DHELAS').mkdir(parents=True,exist_ok=True)
        (target/'Source/MODEL').mkdir(exist_ok=True)
        legs=bo.LegList([bo.Leg(dict(id=x,state=i>=2)) for i,x in enumerate(pids)])
        process=bo.Process(dict(legs=legs,model=self.model,orders={},perturbation_couplings=['QCD'],NLO_mode='virt',has_born=has_born))
        amp=ld.LoopAmplitude(process)
        me=lh.LoopHelasMatrixElement(amp,optimized_output=True)
        if permute:
            seen=set()
            def rename(w):
                if id(w) in seen:return
                seen.add(id(w))
                if not w.get('is_loop') and w.get('number') in (1,3):
                    w.set('number',4-w.get('number'))
                    w.set('me_id',w.get('number'))
                for m in w.get('mothers'):rename(m)
            for d in me.get('diagrams'):
                for w in d.get('wavefunctions'):rename(w)
                if 'loop_wavefunctions' in d:
                    for w in d.get('loop_wavefunctions'):rename(w)
        exp=le.LoopProcessOptimizedExporterFortranSA(str(target),dict(clean=False,complex_mass=False,export_format='madloop',mp=True,loop_dir=str(ROOT/'Template/loop_material'),cuttools_dir=str(ROOT/'vendor/CutTools'),fortran_compiler='gfortran',output_dependencies='external',SubProc_prefix='',compute_color_flows=False,invariant_provenance=True))
        exp.template_dir=str(ROOT/'madgraph/iolibs/template_files/loop_optimized')
        exp.tir_available_dict={x:False for x in exp.all_tir}
        get_context=exp.get_context
        def context(me,*args,**kwargs):
            result=get_context(me,*args,**kwargs)
            result.update(tirust_available=True,tirust_supports_quad_prec=True,ninjars_available=name=='born-internal-grouped')
            return result
        exp.get_context=context
        exp.forbid_loop_grouping=not group
        exp.set_group_loops(me)
        me.compute_all_analytic_information(exp.get_aloha_model(self.model))
        me.rep_dict=exp.generate_general_replace_dict(me,group_number=group_number,proc_id=element)
        exp.set_optimized_output_specific_replace_dict_entries(me)
        prov.start(me)
        writer=hw.FortranUFOHelasCallWriterOptimized(self.model,False)
        previous=os.getcwd()
        try:
            os.chdir(sub)
            exp.write_loopmatrix(fw.FortranWriter('loop_matrix.f'),me,writer,write_auxiliary_files=False)
            exp.write_mp_compute_loop_coefs(fw.FortranWriter('mp_compute_loop_coefs.f'),me,writer)
            exp.write_CT_interface(fw.FortranWriter('CT_interface.f'),me)
            exp.write_TIR_interface(fw.FortranWriter('TIR_interface.f'),me)
            exp.write_TIRUST_interface(fw.FortranWriter('TIRUST_interface.f'),me)
            exp.write_pmass_file(fw.FortranWriter('pmass.inc'),me)
            exp.write_improve_ps(fw.FortranWriter('improve_ps.f'),me)
        finally:os.chdir(previous)
        routes=me._invariant_provenance['routes'];self.assertTrue(routes)
        self.assertEqual([r['route'] for r in routes],list(range(1,len(routes)+1)))
        self.assertEqual(me.get('processes')[0].get('has_born'),has_born)
        # Actual ALOHA writer output for every required graph routine, no model
        # conversion/finalize/bootstrap methods (which could call make).
        am=exp.get_aloha_model(self.model)
        oldmp=aloha.mp_precision;aloha.mp_precision=True
        try:
            am.compute_subset(me.get_used_lorentz())
            am.write(str(target/'Source/DHELAS'),'Fortran',options=[])
        finally:aloha.mp_precision=oldmp
        shutil.copyfile(ROOT/'aloha/template_files/aloha_functions_loop.f',target/'Source/DHELAS/aloha_functions.f')
        # The private template additions live in this overlay; other templates
        # were read unchanged from production and are pinned in the receipt.
        exp.template_dir=str(HERE/'src/madgraph/iolibs/template_files/loop_optimized')
        result=final.finalize_one(exp,str(sub),me)
        unsupported={m:sum(not r['modes'][str(m)]['supported'] for r in result['routes']) for m in prov.MODES}
        self.assertEqual(unsupported,{m:0 for m in prov.MODES},result['source_proof'])
        for file in sub.glob('loop_CT_calls_*.f'):
            text=file.read_text();self.assertIn('GP_ENV,GP_FRESH',text.replace(' ',''))
            self.assertIn('_PROVENANCE(',text)
        actual=(sub/'CT_interface.f').read_text()
        self.assertIn('_PROVENANCE_DISPATCH',actual)
        self.assertIn('METADATA-ONLY SEAM',actual)
        self.receipts.append(dict(case=name,has_born=has_born,group_loops=exp.group_loops,
            routes=len(routes),external_slot_permuted=permute,loopnums=[r['loopnum'] for r in routes],
            source_mode_supported={str(m):len(routes)-unsupported[m] for m in prov.MODES},
            subprocess_prefix=me.rep_dict['proc_prefix'],environment_contract=result['environment_contract'],
            generated={str(x.relative_to(target)):dict(sha256=hashlib.sha256(x.read_bytes()).hexdigest(),bytes=x.stat().st_size) for x in sorted(target.rglob('*')) if x.is_file()},
            numerical_execution=False,compiler_execution=False,
            template_capabilities={'tirust_available':True,'tirust_supports_quad_prec':True,'scope':'source-only explicit context, no library discovery/load'}))
        return me

    def test_born_grouped(self):self.export_case('born-grouped',[2,-2,23],True,True)
    def test_born_ungrouped(self):self.export_case('born-ungrouped',[2,-2,23],True,False)
    def test_born_internal_grouped(self):self.export_case('born-internal-grouped',[2,-2,1,-1],True,True)
    def test_born_internal_ungrouped(self):self.export_case('born-internal-ungrouped',[2,-2,1,-1],True,False)
    def test_external_slot_permutation(self):self.export_case('born-permuted',[2,-2,23],True,True,permute=True)
    def test_loop_induced_group_prefix(self):self.export_case('loop-induced-group-prefix',[21,21,25],False,False,group_number=2,element=3)
    def test_loop_induced(self):self.export_case('loop-induced',[21,21,25],False,False)

if __name__=='__main__':
    # SciPy/NumPy imported by MG may load their own ordinary math dependencies
    # before this audit; no executable/subprocess/ctypes load is permitted now.
    (HERE/'tmp').mkdir(exist_ok=True)
    tempfile.tempdir=str(HERE/'tmp')
    sys.addaudithook(write_guard)
    with warnings.catch_warnings(),patch.object(subprocess,'Popen',deny),patch.object(os,'system',deny):
        warnings.simplefilter('ignore',ResourceWarning)
        unittest.main()
