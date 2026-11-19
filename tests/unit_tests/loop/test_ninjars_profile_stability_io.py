"""Run the complete stability/report I/O with real cards and a mocked checker."""
import ast
import copy
import datetime
import errno
import glob
import io
import logging
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from madgraph.iolibs import save_load_object
from madgraph.various import banner

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'madgraph/various/process_checks.py'
INITIALIZER = ROOT / 'madgraph/interface/madevent_interface.py'


def method(path, owner, name):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner)
    node = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name))
    node.decorator_list = []
    return node


def add_functions(namespace, nodes, filename):
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(filename), 'exec'), namespace)


class Process(dict):
    def __init__(self):
        super().__init__(legs=[{}, {}, {}])

    def shell_string(self):
        return 'P0uux_z'

    def nice_string(self):
        return 'Process: u u~ > z'


class Matrix(dict):
    def __init__(self):
        super().__init__(processes=[Process()])


class Checker:
    def __init__(self, values):
        self.stdin = io.BytesIO()
        text = ''.join(' ##TAG#RESULT_START#TAG##\nFIN %.17g\n ##TAG#RESULT_STOP#TAG##\n' % value
                       for value in values)
        self.stdout = io.BytesIO(text.encode())
        self.stderr = io.BytesIO()
        self.killed = False

    def poll(self):
        return None

    def kill(self):
        self.killed = True

    def wait(self):
        return 0


class StabilityFilesystem(unittest.TestCase):
    def setUp(self):
        parent = ROOT / '.ninjars-build/tmp'
        parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=parent, prefix='stability-io-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.installation = self.root / 'installation'
        templates = self.installation / 'Template/loop_material/Checks'
        templates.mkdir(parents=True)
        for name in ('StabilityCheckDriver.f', 'StabilityCheckDriver_loop_induced.f'):
            shutil.copyfile(ROOT / 'Template/loop_material/Checks' / name, templates / name)
        self.logger = logging.getLogger('stability-full-io-software')
        self.logger.setLevel(logging.CRITICAL)
        self.namespace = dict(os=os, re=re, copy=copy, math=math, datetime=datetime,
            logging=logging, io=io, file=io.FileIO, shutil=shutil, sys=sys, errno=errno,
            time=SimpleNamespace(sleep=mock.Mock(side_effect=AssertionError('No retries expected'))),
            random=SimpleNamespace(uniform=lambda a,b: .25), pjoin=os.path.join,
            temp_dir_prefix='TMP_CHECK', logger=self.logger, MadGraph5Error=ValueError,
            base_objects=SimpleNamespace(Process=Process), helas_objects=SimpleNamespace(HelasMatrixElement=Matrix),
            bannermod=banner, save_load_object=save_load_object,
            MatrixElementEvaluator=SimpleNamespace(pass_isolation_cuts=lambda p: True))
        self.compile_calls, self.engine_calls, self.checkers, self.setup_calls = [], [], [], []
        self.namespace['misc'] = SimpleNamespace(
            make_unique=lambda values:list(dict.fromkeys(values)),
            glob=lambda pattern,path:glob.glob(str(Path(path)/pattern)), compile=self.mock_compile)
        self.namespace['subprocess'] = SimpleNamespace(call=self.mock_make, Popen=self.mock_popen,
                                                       PIPE=-1)
        add_functions(self.namespace, [method(INITIALIZER, 'MadLoopInitializer', 'fix_PSPoint_in_check')], INITIALIZER)
        self.namespace['MadLoopInitializer'] = SimpleNamespace(fix_PSPoint_in_check=self.namespace['fix_PSPoint_in_check'])
        nodes = [method(SOURCE, 'LoopMatrixElementTimer', 'check_matrix_element_stability'),
                 method(SOURCE, 'LoopMatrixElementEvaluator', 'fix_MadLoopParamCard'),
                 method(SOURCE, 'LoopMatrixElementTimer', 'get_me_value'),
                 method(SOURCE, 'LoopMatrixElementEvaluator', 'parse_check_output')]
        tree = ast.parse(SOURCE.read_text())
        names = {'boost_momenta', 'format_output', 'output_stability', 'output_timings', 'output_profile'}
        nodes += [copy.deepcopy(n) for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        add_functions(self.namespace, nodes, SOURCE)
        owner = self
        class Timer:
            mg_root = str(owner.installation)
            loop_optimized_output = True
            def setup_process(self, matrix, export, *args):
                owner.setup_calls.append(export)
                self.fix_MadLoopParamCard(str(owner.export / 'Cards'), mp=False,
                                         MLOptions={'MLReductionLib': owner.backend})
                return dict(owner.info)
            def get_momenta(self, process, options):
                return [[45.,0.,0.,45.],[45.,0.,0.,-45.],[90.,0.,0.,0.]], 1.
        for name in ('check_matrix_element_stability', 'fix_MadLoopParamCard'):
            setattr(Timer, name, self.namespace[name])
        Timer.get_me_value = classmethod(self.namespace['get_me_value'])
        Timer.parse_check_output = classmethod(self.namespace['parse_check_output'])
        self.timer = Timer()

    def mock_make(self, command, cwd, **kwargs):
        self.assertEqual(command, ['make', 'check'])
        self.assertEqual(Path(cwd), self.directory)
        self.compile_calls.append(('make', str(cwd)))
        (self.directory / 'check').write_text('not executable')
        return 0

    def mock_compile(self, arg, cwd, **kwargs):
        self.assertEqual(arg, ['StabilityCheckDriver'])
        self.assertEqual(Path(cwd), self.directory)
        self.compile_calls.append(('compile', str(cwd)))
        self.assertIn('READPS = .TRUE.', (self.directory / 'check_sa.f').read_text())
        driver = (self.directory / 'StabilityCheckDriver.f').read_text()
        self.assertIn('ML5_0_GET_ANSWER_DIMENSION', driver)
        self.assertNotIn('%(proc_prefix)', driver)
        (self.directory / 'StabilityCheckDriver').write_text('not executable')

    def mock_popen(self, command, cwd, **kwargs):
        self.assertEqual(command, [str(self.directory / 'StabilityCheckDriver')])
        self.assertEqual(Path(cwd), self.directory)
        active = banner.MadLoopParam(str(self.active_card))
        self.assertTrue(active['DoubleCheckHelicityFilter'])
        self.assertFalse(active['UseLoopFilter'])
        self.assertEqual(active['CTModeRun'], 1)
        self.assertEqual(active['CTModeInit'], 1)
        self.assertEqual(active['MLReductionLib'], str(self.backend))
        checker = Checker(self.values)
        self.checkers.append(checker); self.engine_calls.append(command)
        return checker

    def fixture(self, backend=8, layout='modern', default_output=False, keep=True):
        self.backend = backend
        output = self.installation if default_output else self.root / 'private-output'
        self.export = output / (('SAVED' if keep else '') + 'TMP_CHECK_uux_z')
        self.directory = self.export / 'SubProcesses/P0_uux_z'; self.directory.mkdir(parents=True)
        (self.export / 'Cards').mkdir()
        card = banner.MadLoopParam(str(ROOT / 'Template/loop_material/StandAlone/Cards/MadLoopParams.dat'))
        card.set('MLReductionLib', str(backend)); card.set('DoubleCheckHelicityFilter', False)
        card.write(str(self.export / 'Cards/MadLoopParams.dat'))
        parent_card = self.export / 'SubProcesses/MadLoopParams.dat'
        parent_card.symlink_to('../Cards/MadLoopParams.dat')
        if layout == 'modern':
            resources = self.export / 'SubProcesses/MadLoop5_resources'; resources.mkdir()
            (resources / 'MadLoopParams.dat').symlink_to('../MadLoopParams.dat')
            (self.directory / 'MadLoop5_resources').symlink_to('../MadLoop5_resources')
            self.active_card = self.directory / 'MadLoop5_resources/MadLoopParams.dat'
        else:
            self.active_card = self.directory / 'MadLoopParams.dat'
            self.active_card.symlink_to('../MadLoopParams.dat')
        (self.directory / 'proc_prefix.txt').write_text('ML5_0_')
        (self.directory / 'born_matrix.f').write_text('born fixture')
        (self.directory / 'check_sa.f').write_text('      PARAMETER (READPS = .FALSE.)\n'
            '      PARAMETER (NPSPOINTS = 9)\n      CALL ML5_0_SLOOPMATRIX_THRES(P,MATELEM,ACC)\n')
        (self.directory / 'check').write_text('old check')
        (self.directory / 'check_sa.o').write_text('old object')
        library = self.root / 'library'; library.mkdir()
        name = 'ninja' if backend == 6 else 'ninjars'
        (library / ('lib' + name + '.a')).write_text('not a native archive')
        self.timer.tir_dir = {name + '_dir': str(library)}
        self.timer.output_path = str(output)
        self.info = dict(Process_output=.1, HELAS_MODEL_compilation=.1, Process_compilation=.1,
                         Initialization=.01, dir_path=str(self.directory))
        return output

    def call(self, npoints=2, reusing=True, infos=True, keep=True):
        return self.timer.check_matrix_element_stability(Process() if reusing else Matrix(),
            options={'reuse':reusing,'npoints':npoints,'split_orders':-1,'events':False},
            infos_IN=self.info if infos else None, keep_folder=keep,
            MLOptions={'MLReductionLib':[self.backend]})

    def report(self, result, output):
        result['loop_optimized_output'] = True
        timing = {key:.1 for key in ('HELAS_MODEL_compilation','HelasDiagrams_generation','Process_output',
            'Diagrams_generation','Process_compilation','Initialization','Booting_time')}
        timing.update(loop_optimized_output=True,reduction_tool=self.backend,
            run_unpolarized_total=.001,run_unpolarized_coefs=.0003,
            run_polarized_total=.001,run_polarized_coefs=.0003,
            n_contrib_hel=1,n_tot_hel=1,n_loops=1,n_loop_groups=1,n_loop_wfs=1,
            loop_wfs_ranks=[1],ram_usage=1000,du_source='1',du_process='1',du_color='1',du_exe='1')
        handler = logging.FileHandler(self.root / 'logging-only.log')
        logger = logging.getLogger('madgraph'); logger.addHandler(handler)
        try:
            text = self.namespace['output_profile'](Process(), result, timing, str(output), reusing=True)
        finally:
            logger.removeHandler(handler); handler.close()
        self.assertIn(banner.MadLoopParam._ID_reduction_tool_map[self.backend], text)
        self.assertEqual((output / 'profile_optimized_P0uux_z.log').read_text(), text)
        stable = (output / 'stability_optimized_P0uux_z.log').read_text()
        self.assertIn('reduction=' + ('NINJA' if self.backend == 6 else 'NINJARS'), stable)
        return stable

    def exercise(self, backend=8, layout='modern', default_output=False, zero=None,
                 reusing=True, infos=True, keep=True, cache=False, zero_mean=False, values=None):
        output = self.fixture(backend, layout, default_output, keep)
        self.values = ([zero] * 6 if zero is not None else [1.,1.0001,1.0002,1.,1.5,1.,1.1,1.10001,1.10002])
        if zero_mean:
            self.values = [-1.,0.,1.,1.,1.,1.,1.,1.0001,1.0002]
        if values is not None:
            self.values = values
        result = self.call(reusing=reusing, infos=infos, keep=keep)
        name = banner.MadLoopParam._ID_reduction_tool_map[backend]
        saved = self.directory / ('SavedStabilityRun_%s_0.pkl' % name)
        loaded = save_load_object.load_from_file(str(saved))
        self.assertEqual(loaded, result['Stability'][name])
        self.assertEqual(len(loaded['DP_stability']), 2); self.assertEqual(len(loaded['QP_stability']), 2)
        if zero is not None:
            self.assertEqual([row['Accuracy'] for row in loaded['DP_stability']], [0.,0.])
            self.assertEqual(loaded['Unstable_PS_points'], [])
        else:
            self.assertEqual(len(loaded['Unstable_PS_points']), 1)
            index = loaded['Unstable_PS_points'][0][0]
            self.assertTrue(loaded['QP_stability'][index])
            if zero_mean:
                self.assertEqual(loaded['DP_stability'][0]['Accuracy'], float('inf'))
        self.assertEqual(len(self.compile_calls), 2); self.assertEqual(len(self.checkers), 1)
        self.assertTrue(self.checkers[0].stdin.getvalue().endswith(b'y\n'))
        self.assertFalse(self.checkers[0].killed)
        self.assertTrue((self.export / 'SubProcesses/MadLoopParams.dat').is_symlink())
        self.assertTrue(self.active_card.is_symlink())
        self.assertFalse((self.export / 'SubProcesses/SubProcesses').exists())
        self.assertEqual(self.setup_calls, [] if infos else [str(self.export)])
        if cache:
            contents = saved.read_bytes(); calls = (list(self.compile_calls),list(self.engine_calls))
            cached = self.call(npoints=0, reusing=True, keep=keep)
            self.assertEqual(cached['Stability'][name], loaded)
            self.assertEqual(saved.read_bytes(), contents)
            self.assertEqual((self.compile_calls,self.engine_calls), calls)
        stable = self.report(result, output)
        if loaded['Unstable_PS_points']:
            self.assertIn('QP Median accuracy', stable)
            self.assertIn('DP vs QP stab test consistency', stable)
        if zero is not None:
            self.assertIn('Perfect accuracy', (output / 'profile_optimized_P0uux_z.log').read_text())
            self.assertIn('<accuracies reduction=', stable)
            self.assertIn('0.0000e+00', stable)
        if zero_mean:
            self.assertIn('inf', stable)
        return result

    def test_modern_private_ninja_full_checker_pickle_cache_and_profile(self):
        self.exercise(backend=6, cache=True)

    def test_modern_private_ninjars_full_checker_pickle_cache_and_profile(self):
        self.exercise(backend=8, cache=True)

    def test_legacy_flat_alias_ninja_full_checker_and_report(self):
        self.exercise(backend=6, layout='legacy')

    def test_legacy_flat_alias_ninjars_full_checker_and_report(self):
        self.exercise(backend=8, layout='legacy')

    def test_default_output_standalone_identical_nonzero_fin_keeps_exact_zero_accuracy(self):
        self.exercise(backend=6, default_output=True, zero=1., reusing=False, infos=False, keep=False)

    def test_default_output_standalone_zero_fin_keeps_exact_zero_accuracy(self):
        self.exercise(backend=8, default_output=True, zero=0., reusing=False, infos=False, keep=False)

    def test_ninja_zero_mean_disagreement_keeps_infinite_raw_accuracy(self):
        self.exercise(backend=6, zero_mean=True)

    def test_ninjars_zero_mean_disagreement_keeps_infinite_raw_accuracy(self):
        self.exercise(backend=8, zero_mean=True)

    def test_large_finite_checker_values_keep_raw_samples_and_qp_summary(self):
        result = self.exercise(backend=8, values=[1e308,1.1e308,1.2e308,1.1e308,1.1e308,1.1e308,
                                                1.,1.0001,1.0002])
        point = result['Stability']['NinjaRS']['DP_stability'][0]
        self.assertEqual([point[k] for k in ('CTModeA','CTModeB','Rotation1')], [1e308,1.1e308,1.2e308])
        self.assertTrue(math.isfinite(point['Accuracy']))

    def test_large_finite_accuracy_survives_actual_checker_and_plot_bins(self):
        result = self.exercise(backend=6, values=[1.,-1.,6e-308,1.,1.,1.,1.,1.0001,1.0002])
        point = result['Stability']['Ninja']['DP_stability'][0]
        self.assertTrue(.9e308 < point['Accuracy'] < 1.1e308)
        self.assertEqual(point['Rotation1'], 6e-308)

    def test_unstable_dp_with_valid_zero_qp_values_reports_without_fudging(self):
        result = self.exercise(backend=8, values=[1.,1.5,1.,0.,0.,0.,1.,1.0001,1.0002])
        point = result['Stability']['NinjaRS']['QP_stability'][0]
        self.assertEqual(point, {'CTModeA':0.,'CTModeB':0.,'Rotation1':0.,'Accuracy':0.})

    def test_direct_report_tiny_and_huge_accuracy_bins_keep_raw_records(self):
        output = self.fixture(backend=8)
        rows = [dict(CTModeA=1.,CTModeB=1.,Rotation1=1.,Accuracy=value) for value in (1e-320,1e308)]
        result = {'Process':Process(), 'Stability':{'NinjaRS':{
            'DP_stability':rows,'QP_stability':[{},{}],
            'Unstable_PS_points':[],'Exceptional_PS_points':[]}}}
        stable = self.report(result, output)
        self.assertEqual([r['Accuracy'] for r in rows], [1e-320,1e308])
        raw = stable.split('<accuracies reduction=NINJARS>\n',1)[1].split('</accuracies>',1)[0]
        self.assertEqual([float(line.split()[0]) for line in raw.splitlines()], [1e-320,1e308])
        plot = stable.split('<plot_data reduction=NINJARS>\n',1)[1].split('</plot_data>',1)[0]
        self.assertTrue(plot.splitlines())
        for line in plot.splitlines():
            threshold, fraction = map(float,line.split())
            self.assertTrue(math.isfinite(threshold))
            self.assertTrue(0. <= fraction <= 1.)

    def saved_cache(self, dp, qp):
        name = banner.MadLoopParam._ID_reduction_tool_map[self.backend]
        path = self.directory / ('SavedStabilityRun_%s_0.pkl' % name)
        save_load_object.save_to_file(str(path), {'DP_stability':dp,'QP_stability':qp,
            'Unstable_PS_points':[],'Exceptional_PS_points':[]})
        return path

    def test_invalid_historical_pickle_records_are_rejected_before_report_output(self):
        output = self.fixture(backend=8)
        valid = dict(CTModeA=1.,CTModeB=1.,Rotation1=1.,Accuracy=0.)
        trials = []
        for precision in ('DP_stability','QP_stability'):
            for field, value in (('Accuracy',float('nan')), ('Accuracy',-.1),
                                 ('Accuracy',float('-inf')), ('CTModeA',float('nan')),
                                 ('CTModeA',float('inf')), ('Rotation1',float('-inf'))):
                row = dict(valid); row[field] = value
                trials.append((precision, field, row))
        trials.append(('DP_stability','empty',{}))
        missing = dict(valid); missing.pop('Accuracy')
        trials.append(('DP_stability','missingAccuracy',missing))
        for precision, field, invalid in trials:
            with self.subTest(precision=precision, field=field, invalid=invalid):
                dp = [invalid] if precision == 'DP_stability' else [dict(valid)]
                qp = [invalid] if precision == 'QP_stability' else [{}]
                cache = self.saved_cache(dp, qp); original = cache.read_bytes()
                result = self.call(npoints=0)
                self.assertEqual(self.compile_calls, []); self.assertEqual(self.engine_calls, [])
                with self.assertRaisesRegex(ValueError, 'Invalid stability data.*'):
                    self.report(result, output)
                self.assertEqual(cache.read_bytes(), original)
                self.assertFalse((output / 'stability_optimized_P0uux_z.log').exists())
                self.assertFalse((output / 'profile_optimized_P0uux_z.log').exists())

    def test_valid_infinite_cached_accuracy_and_empty_qp_remain_reportable(self):
        output = self.fixture(backend=8)
        row = dict(CTModeA=-1.,CTModeB=0.,Rotation1=1.,Accuracy=float('inf'))
        cache = self.saved_cache([row], [{}]); original = cache.read_bytes()
        result = self.call(npoints=0)
        self.assertEqual(result['Stability']['NinjaRS']['DP_stability'], [row])
        self.assertEqual(result['Stability']['NinjaRS']['QP_stability'], [{}])
        self.assertIn('inf', self.report(result, output))
        self.assertEqual(cache.read_bytes(), original)
        self.assertEqual(self.compile_calls, []); self.assertEqual(self.engine_calls, [])

    def test_valid_zero_fin_cached_record_remains_exactly_zero(self):
        output = self.fixture(backend=6)
        row = dict(CTModeA=0.,CTModeB=0.,Rotation1=0.,Accuracy=0.)
        cache = self.saved_cache([row], [{}]); original = cache.read_bytes()
        result = self.call(npoints=0)
        self.assertEqual(result['Stability']['Ninja']['DP_stability'], [row])
        self.assertIn('0.0000e+00', self.report(result, output))
        self.assertEqual(cache.read_bytes(), original)
        self.assertEqual(self.compile_calls, []); self.assertEqual(self.engine_calls, [])


if __name__ == '__main__':
    unittest.main()
