"""Actual whole stability method plus real driver path handling; no native calls."""
import ast
import copy
import datetime
import glob
import logging
import math
import os
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[3]
NEW = ROOT/'madgraph/various/process_checks.py'

class BoundaryReached(RuntimeError):
    pass

class Process:
    def shell_string(self):
        return 'P0uux_z'

class Matrix(dict):
    def __init__(self):
        super().__init__(processes=[Process()])

def extract(path, classname, method):
    cls = next(n for n in ast.parse(path.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == classname)
    node = copy.deepcopy(next(n for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == method))
    node.decorator_list = []
    return node

class OutputPath(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='mg7-profile-output-')
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)

    def exercise(self, source, infos=True, same_root=False, reusing=False):
        installation = self.work/'installation'
        output = installation if same_root else self.work/'private-output'
        directory = output/'SAVEDTMP_CHECK_uux_z/SubProcesses/P0_uux_z'
        directory.mkdir(parents=True)
        driver = directory/'check_sa.f'
        driver.write_text('      PARAMETER (READPS = .FALSE.)\n'
                          '      PARAMETER (NPSPOINTS = 9)\n'
                          '      CALL ML5_0_SLOOPMATRIX_THRES(P,MATELEM,ACC)\n')
        library = self.work/'ninja';library.mkdir()
        (library/'libninja.a').write_text('non-native test fixture')
        calls, setups = [], []
        namespace = {'os': os, 're': re, 'pjoin': os.path.join,
            'misc': SimpleNamespace(glob=lambda pattern,path:glob.glob(str(Path(path)/pattern))),
            'MadGraph5Error': ValueError}
        real_node = extract(ROOT/'madgraph/interface/madevent_interface.py',
            'MadLoopInitializer','fix_PSPoint_in_check')
        exec(compile(ast.Module(body=[real_node],type_ignores=[]),'<real-driver-path-handler>','exec'),namespace)
        real_fix = namespace['fix_PSPoint_in_check']
        def fix(path, **kwargs):
            calls.append((path, kwargs))
            real_fix(path, **kwargs)
            raise BoundaryReached('Real check_sa.f updated; stop before native compilation')
        info = dict(Process_output=.1,HELAS_MODEL_compilation=.1,
            Process_compilation=.1,Initialization=.01,dir_path=str(directory))
        class Timer:
            mg_root = str(installation)
            output_path = str(output)
            loop_optimized_output = True
            tir_dir = {'ninja_dir': str(library)}
            def setup_process(self, matrix, export, *args):
                setups.append(export)
                return dict(info)
        def forbidden(*args, **kwargs):
            raise AssertionError('No compiler/native process is allowed in this regression')
        logger = logging.getLogger('private-output-path-regression');logger.setLevel(logging.CRITICAL)
        namespace = dict(os=os,re=re,copy=copy,math=math,datetime=datetime,logging=logging,
            misc=SimpleNamespace(make_unique=lambda values:list(dict.fromkeys(values))),
            pjoin=os.path.join,temp_dir_prefix='TMP_CHECK',logger=logger,
            MadGraph5Error=ValueError,base_objects=SimpleNamespace(Process=Process),
            helas_objects=SimpleNamespace(HelasMatrixElement=Matrix),
            MadLoopInitializer=SimpleNamespace(fix_PSPoint_in_check=fix),
            bannermod=SimpleNamespace(MadLoopParam=SimpleNamespace(_ID_reduction_tool_map={6:'Ninja'})),
            subprocess=SimpleNamespace(Popen=forbidden,call=forbidden,check_call=forbidden))
        node=extract(source,'LoopMatrixElementTimer','check_matrix_element_stability')
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
        Timer.check_matrix_element_stability=namespace['check_matrix_element_stability']
        timer=Timer()
        try:
            timer.check_matrix_element_stability(Process() if reusing else Matrix(),
                options={'reuse':reusing,'npoints':1,'split_orders':-1,'events':False},
                infos_IN=dict(info) if infos else None,keep_folder=True,
                MLOptions={'MLReductionLib':[6]})
        except BoundaryReached:
            self.assertEqual(calls,[(str(output/'SAVEDTMP_CHECK_uux_z/SubProcesses'),
                {'read_ps':True,'npoints':1,'hel_config':-1,'split_orders':-1})])
            self.assertIn('READPS = .TRUE.',driver.read_text())
            self.assertIn('NPSPOINTS = 1',driver.read_text())
            self.assertEqual(setups,[] if infos else [str(output/'SAVEDTMP_CHECK_uux_z')])
            return
        self.fail('Expected driver boundary')

    def test_profile_reuses_existing_process_in_explicit_private_output(self):
        self.exercise(NEW,reusing=True)

    def test_profile_reuses_infos_in_explicit_private_output(self):
        self.exercise(NEW)

    def test_standalone_stability_exports_into_explicit_private_output(self):
        self.exercise(NEW,infos=False)

    def test_default_installation_output_behavior_unchanged(self):
        self.exercise(NEW,same_root=True)

if __name__=='__main__':
    unittest.main()
