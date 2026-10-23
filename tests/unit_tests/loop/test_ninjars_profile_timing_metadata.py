"""Execute the live timing/profile methods with compiler and engine mocks."""
import ast
import copy
import hashlib
import json
import logging
import os
import glob
import shutil
from pathlib import Path
import re
from types import SimpleNamespace
import unittest
import tempfile

ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT/'tests/input_files/ninjars_profile_timing_legacy.py'
NEW = ROOT/'madgraph/various/process_checks.py'

class Process:
    def get(self, key):
        if key == 'has_born':
            return True
        raise KeyError(key)

    def shell_string(self):
        return 'P0uux_z'

    def nice_string(self):
        return 'Process: u u~ > z'

class Matrix(dict):
    def __init__(self):
        super().__init__(processes=[Process()])

def actual_functions(path):
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='LoopMatrixElementTimer')
    names = ('time_matrix_element','boot_time_setup','skip_loop_evaluation_setup')
    methods = [node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name in names]
    return methods

def exercise(label, source, optimized=True, initialized=None, tool=8, failure=None, real_card=False,
             mutate_runtime=None, base_root=None, physical_layout=None):
    base = base_root/label
    base.mkdir(parents=True,exist_ok=False)
    calls = []
    count = [0]
    archive_reads = []
    def observed_open(path,mode='r',*args,**kwargs):
        if Path(path).name=='libninjars.a' and 'r' in mode:
            archive_reads.append(str(path))
        return open(path,mode,*args,**kwargs)
    card = dict(MLReductionLib=str(tool),CTModeRun=-1,CTModeInit=1,
                COLLIERComputeUVpoles=True,COLLIERComputeIRpoles=True,
                MLStabThres=1e-9,COLLIERRequiredAccuracy=1e-12)
    if physical_layout is not None:
        card['CTModeRun'] = 1
    class Initializer:
        @staticmethod
        def fix_PSPoint_in_check(path,**settings):
            calls.append(('fix',dict(settings)))
            directory = next((Path(path)).glob('P*_*'))
            hel = settings.get('hel_config',-1)
            call = ('CALL ML5_0_SLOOPMATRIXHEL_THRES(P,%s,MATELEM,ACC)'%hel if hel!=-1
                    else 'CALL ML5_0_SLOOPMATRIX_THRES(P,MATELEM,ACC)')
            (directory/'check_sa.f').write_text('      PARAMETER (NPSPOINTS = %s)\n'%settings['npoints']+
                '      PARAMETER (READPS = .FALSE.)\n      SQRTS=1.0D3\n      CALL ML5_0_SET_COUPLINGORDERS_TARGET(-1)\n      '+call+'\n')
        @staticmethod
        def make_and_run(directory,checkRam=False):
            directory = Path(directory)
            calls.append(('make',checkRam,(directory/'check_sa.f').read_text(),(directory/'loop_matrix.f').read_text()))
            count[0] += 1
            (directory/'check').write_text('mock nonexecutable build '+str(count[0]))
            (directory/'result.dat').write_text('mock stage '+str(count[0]))
            if mutate_runtime==count[0]:
                (directory.parent.parent/'lib/libninjars.a').write_text('changed mock runtime')
            if failure==count[0]:
                return None,None,None
            values = [0.6,0.1,2.0,3.0,1.0,0.8] if initialized is None else [0.1,2.,3.,1.,.8]
            return .01,values[count[0]-1],123 if checkRam else None
    class Timer:
        loop_optimized_output = optimized
        output_path = str(base)
        @classmethod
        def get_MadLoop_Params(cls,path):
            if real_card:
                from madgraph.various.banner import MadLoopParam
                return MadLoopParam(path)
            return copy.deepcopy(card)
        def setup_process(self,matrix,export_dir,*args,**kwargs):
            export = Path(export_dir)
            directory = export/'SubProcesses/P0_uux_z'
            resources = directory/'MadLoop5_resources'
            resources.mkdir(parents=True)
            (export/'Cards').mkdir()
            for name in ('param_card.dat','MadLoopParams.dat'):
                (export/'Cards'/name).write_text('mock '+name)
            if real_card:
                from madgraph.various.banner import MadLoopParam
                actual = MadLoopParam()
                for key,value in card.items():
                    actual[key] = value
                actual.write(str(export/'Cards/MadLoopParams.dat'))
            if physical_layout is not None:
                (resources/'param_card.dat').symlink_to('../../../Cards/param_card.dat')
                (resources/'MadLoopParams.dat').symlink_to('../../../Cards/MadLoopParams.dat')
            if physical_layout == 'flat-first':
                (directory/'param_card.dat').write_text('mock param_card.dat')
                (directory/'Cards').mkdir()
                (directory/'Cards/param_card.dat').write_text('lower-priority decoy')
            if physical_layout == 'ancestor-first':
                (export/'SubProcesses/Cards').mkdir()
                (export/'SubProcesses/Cards/param_card.dat').write_text('higher-priority native card')
            (directory/'proc_prefix.txt').write_text('ML5_0_')
            (resources/'ML5_0_HelFilter.dat').write_text('-10000 1 -10000\n' if optimized else 'F T F\n')
            (resources/'ML5_0_HelConfigs.dat').write_text('1 -1 -1\n1 -1 0\n1 -1 1\n')
            (directory/'loop_matrix.f').write_text('      PARAMETER(SKIPLOOPEVAL=.FALSE.)\n      PARAMETER(BOOTANDSTOP=.FALSE.)\n')
            (directory/'check').write_text('mock setup nonexecutable')
            (export/'lib').mkdir()
            (export/'lib/libninjars.a').write_text('mock archive text, not native')
            (export/'lib/ninjars-config.json').write_text('{}')
            (export/'Source').mkdir()
            (export/'Source/make_opts').write_text('mock flags')
            (export/'SubProcesses/MadLoop_makefile_definitions').write_text('mock link flags')
            Initializer.fix_PSPoint_in_check(str(export/'SubProcesses'),read_ps=False,npoints=4)
            return dict(dir_path=str(directory),Initialization=initialized,Process_output=.2,
                HELAS_MODEL_compilation=.3,Process_compilation=.4)
    ns = dict(os=os,re=re,copy=copy,glob=glob,shutil=shutil,open=observed_open,
        MadGraph5Error=ValueError,pjoin=os.path.join,temp_dir_prefix='TMP_CHECK',
        helas_objects=SimpleNamespace(HelasMatrixElement=Matrix),base_objects=SimpleNamespace(Process=Process),
        MadLoopInitializer=Initializer,subprocess=SimpleNamespace(check_output=lambda *a,**k:'1K total\n'),
        logger=logging.getLogger('mock-timing'),bannermod=SimpleNamespace(MadLoopParam=SimpleNamespace(
            _ID_reduction_tool_map={6:'Ninja',8:'NinjaRS'})))
    exec(compile(ast.Module(body=actual_functions(source),type_ignores=[]),str(source),'exec'),ns)
    for name in ('time_matrix_element','boot_time_setup','skip_loop_evaluation_setup'):
        setattr(Timer,name,ns[name])
    result = Timer().time_matrix_element(Matrix(),MLOptions={'MLReductionLib':[tool]})
    (base/'archive-read-count.json').write_text(json.dumps(archive_reads))
    return result,calls,base

class TimingMetadataTests(unittest.TestCase):
    def setUp(self):
        parent = ROOT/'.ninjars-build/tmp'
        parent.mkdir(parents=True,exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=parent,prefix='mg5-profile-metadata-')
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)

    def compare(self,name,**kwargs):
        old,old_calls,_ = exercise(name+'-old',OLD,base_root=self.work,**kwargs)
        new,new_calls,base = exercise(name+'-new',NEW,base_root=self.work,**kwargs)
        self.assertEqual(old_calls,new_calls)
        if old is None:
            self.assertIsNone(new)
        else:
            raw = dict(new)
            ref = raw.pop('timing_metadata')
            old['dir_path'] = raw['dir_path']
            self.assertEqual(old,raw)
            metadata = json.loads(Path(ref['path']).read_text())
            self.assertEqual(metadata['timing_result'],old)
            self.assertEqual(metadata['selected_timing_helicity']['one_based_index'],2)
            self.assertEqual(metadata['selected_timing_helicity']['physical_tuple'],[1,-1,0])
            self.assertEqual(metadata['selected_timing_helicity']['filter_tokens'],
                ['-10000','1','-10000'] if kwargs.get('optimized',True) else ['F','T','F'])
            self.assertEqual(metadata['reduction_backends'],[int(value) for value in str(kwargs.get('tool',8)).split('|')])
            self.assertFalse(metadata['per_evaluation_statuses_available'])
            self.assertEqual(len(json.loads((base/'archive-read-count.json').read_text())),1)
            for stage in metadata['stages']:
                self.assertEqual(metadata['runtime_files']['lib/libninjars.a']['sha256'],
                    hashlib.sha256(b'mock archive text, not native').hexdigest())
                self.assertIn('sha256',stage['check_sa'])
                self.assertIn('sha256',stage['loop_matrix'])
                self.assertIn('sha256',stage['executable'])
                self.assertIn('sha256',stage['raw_outputs']['result.dat'])
                effective = {k.lower():v for k,v in stage['effective_madloop_parameters'].items()}
                self.assertEqual(effective['mlstabthres'],1e-9)
                self.assertEqual(effective['collierrequiredaccuracy'],1e-12)
                self.assertFalse(stage['raw_output_freshness_verified'])
                self.assertEqual(stage['runtime_files_reference'],'runtime_files')
                self.assertTrue(stage['runtime_stat_unchanged'])
                self.assertEqual(stage['energy_assignment_lines'],['SQRTS=1.0D3'])
                self.assertEqual(stage['raw_outputs']['result.dat']['text'],
                    Path(stage['raw_outputs']['result.dat']['path']).read_text() if
                    stage is metadata['stages'][-1] else 'mock stage '+str(metadata['stages'].index(stage)+1))
                for field in ('check_sa.f','loop_matrix.f'):
                    record = stage['source_snapshots'][field]
                    self.assertEqual(hashlib.sha256(Path(record['path']).read_bytes()).hexdigest(),record['sha256'])
            return metadata

    def test_optimized_native8_all_stage_calls_and_raw_floats_unchanged(self):
        metadata = self.compare('optimized8')
        stages = metadata['stages']
        self.assertEqual([s['name'] for s in stages],['estimate','boot','polarized_full','unpolarized_full','unpolarized_skip','polarized_skip'])
        self.assertEqual([s['actual_npoints'] for s in stages],[3,3,302,151,151,302])
        self.assertEqual([s['flags'] for s in stages],[
            dict(SKIPLOOPEVAL=False,BOOTANDSTOP=False),dict(SKIPLOOPEVAL=False,BOOTANDSTOP=True),
            dict(SKIPLOOPEVAL=False,BOOTANDSTOP=False),dict(SKIPLOOPEVAL=False,BOOTANDSTOP=False),
            dict(SKIPLOOPEVAL=True,BOOTANDSTOP=False),dict(SKIPLOOPEVAL=True,BOOTANDSTOP=False)])
        for stage in stages[2:]:
            self.assertEqual(stage['derived_seconds_per_point'],metadata['timing_result'][
                'run_'+dict(polarized_full='polarized_total',unpolarized_full='unpolarized_total',
                unpolarized_skip='unpolarized_coefs',polarized_skip='polarized_coefs')[stage['name']]])

    def test_optimized_ninja6_same_metadata_contract(self):
        self.compare('optimized6',tool=6)

    def test_real_madloopparam_string_selector_normalized_to_int_labels(self):
        metadata = self.compare('realcard8',real_card=True)
        self.assertEqual(metadata['reduction_backends'],[8])
        self.assertEqual(metadata['backend_labels'],['NinjaRS'])

    def test_real_madloopparam_ordered_selector_normalized(self):
        metadata = self.compare('realcard86',real_card=True,tool='8|6')
        self.assertEqual(metadata['reduction_backends'],[8,6])
        self.assertEqual(metadata['backend_labels'],['NinjaRS','Ninja'])

    def test_prior_initialization_omits_estimate_without_changing_calls(self):
        metadata = self.compare('initialized',initialized=8.)
        self.assertEqual(metadata['stages'][0]['name'],'boot')
        self.assertEqual(metadata['stages'][0]['actual_npoints'],4)

    def test_unoptimized_early_return_is_recorded(self):
        metadata = self.compare('unoptimized',optimized=False)
        self.assertEqual([s['name'] for s in metadata['stages']],['estimate','boot','polarized_full','unpolarized_full'])

    def test_compile_failure_return_and_partial_metadata_preserved(self):
        self.compare('failure',failure=3)
        path = next((self.work/'failure-new').glob('*/SavedRunInfos/profile_timing_metadata.json'))
        partial = json.loads(path.read_text())
        self.assertEqual(len(partial['stages']),3)
        self.assertEqual(partial['stages'][-1]['name'],'polarized_full')
        self.assertIsNone(partial['stages'][-1]['compile_seconds'])

    def test_runtime_mutation_rejected_after_existing_run_without_rehash(self):
        with self.assertRaisesRegex(ValueError,'Timing runtime changed during measurement'):
            exercise('runtime-mutated',NEW,mutate_runtime=2,base_root=self.work)

    def test_profile_merged_raw_dictionary_captured_before_stability_mutation(self):
        _,_,base = exercise('profile-seed',NEW,base_root=self.work)
        path = next(base.glob('*/SavedRunInfos/profile_timing_metadata.json'))
        seed = json.loads(path.read_text())['timing_result']
        seed['timing_metadata'] = dict(path=str(path),schema='madloop-profile-timing-metadata-v1')
        class Profiler:
            loop_optimized_output = True
            def __init__(self,**kwargs): pass
            def time_matrix_element(self,*args,**kwargs): return copy.deepcopy(seed)
            def check_matrix_element_stability(self,*args,**kwargs):
                kwargs['infos_IN']['run_unpolarized_total'] = 999
                return {'Stability':{}}
        proc = SimpleNamespace(get=lambda key:object())
        tree = ast.parse(NEW.read_text())
        fn = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='check_profile')
        ns = dict(copy=copy,FakeInterface=lambda:None,base_objects=SimpleNamespace(Process=Process),
            LoopMatrixElementTimer=Profiler,generate_loop_matrix_element=lambda *a,**k:({'Diagrams_generation':.123456789},Matrix()))
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(NEW),'exec'),ns)
        returned,_ = ns['check_profile'](proc,MLOptions={'MLReductionLib':[8]})
        metadata = json.loads(path.read_text())
        self.assertEqual(metadata['profile_timing_result']['run_unpolarized_total'],seed['run_unpolarized_total'])
        self.assertEqual(metadata['profile_timing_result']['Diagrams_generation'],.123456789)
        self.assertEqual(returned['run_unpolarized_total'],999)

    def assert_card_record(self, record, expected):
        contents = expected.read_bytes()
        self.assertEqual(Path(record['path']).resolve(),expected.resolve())
        self.assertEqual(record['realpath'],str(expected.resolve()))
        self.assertNotEqual(record.get('available'),False)
        self.assertEqual(record['bytes'],len(contents))
        self.assertEqual(record['sha256'],hashlib.sha256(contents).hexdigest())
        self.assertEqual(record['text'],contents.decode('utf-8'))

    def test_modern_resource_only_card_is_observed_by_actual_timing_method(self):
        for tool in (6,8):
            result,_,_ = exercise('modern-%d'%tool,NEW,tool=tool,real_card=True,
                                 physical_layout='modern',base_root=self.work)
            metadata = json.loads(Path(result['timing_metadata']['path']).read_text())
            directory = Path(metadata['subprocess_dir'])
            self.assertFalse((directory/'param_card.dat').exists())
            self.assertTrue((directory/'MadLoop5_resources/param_card.dat').is_file())
            self.assertEqual(metadata['reduction_backends'],[tool])
            self.assertEqual(len(metadata['stages']),6)
            expected = Path(metadata['export_dir'])/'Cards/param_card.dat'
            for stage in metadata['stages']:
                self.assert_card_record(stage['parameter_card'],expected)
                self.assert_card_record(stage['subprocess_parameter_card'],expected)

    def test_native_flat_card_priority_beats_lower_cards_decoy(self):
        result,_,_ = exercise('flat',NEW,real_card=True,
                             physical_layout='flat-first',base_root=self.work)
        metadata = json.loads(Path(result['timing_metadata']['path']).read_text())
        expected = Path(metadata['subprocess_dir'])/'param_card.dat'
        decoy = expected.parent/'Cards/param_card.dat'
        self.assertNotEqual(expected.read_bytes(),decoy.read_bytes())
        for stage in metadata['stages']:
            self.assertEqual(Path(stage['subprocess_parameter_card']['path']),expected)
            self.assert_card_record(stage['subprocess_parameter_card'],expected)
            self.assertEqual(stage['subprocess_parameter_card']['sha256'],
                             stage['parameter_card']['sha256'])

    def test_intervening_ancestor_card_records_different_effective_physics(self):
        result,_,_ = exercise('ancestor',NEW,real_card=True,
                             physical_layout='ancestor-first',base_root=self.work)
        metadata = json.loads(Path(result['timing_metadata']['path']).read_text())
        export = Path(metadata['export_dir'])
        expected = export/'SubProcesses/Cards/param_card.dat'
        canonical = export/'Cards/param_card.dat'
        self.assertNotEqual(expected.read_bytes(),canonical.read_bytes())
        for stage in metadata['stages']:
            self.assert_card_record(stage['parameter_card'],canonical)
            self.assert_card_record(stage['subprocess_parameter_card'],expected)
            self.assertNotEqual(stage['subprocess_parameter_card']['sha256'],
                                stage['parameter_card']['sha256'])

if __name__ == '__main__':
    unittest.main()
