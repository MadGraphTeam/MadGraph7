"""Flavour-aware model dependencies remain usable by Fortran and loop checks."""

from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from madgraph.core import base_objects
from madgraph.iolibs import born_support, export_v4
from madgraph.loop import loop_exporters
from madgraph.various import process_checks


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestFlavourCouplingVectorScope(unittest.TestCase):
    def check_generated_model(self, vector_size, nb_warp):
        with tempfile.TemporaryDirectory(prefix='mg7_flavour_vector_') as tmp:
            source = Path(tmp) / 'Source'
            model = source / 'MODEL'
            model.mkdir(parents=True)
            converter = object.__new__(export_v4.UFO_model_to_mg4)
            converter.dir_path = str(model)
            converter.model = {'merged_particles': {81: [1, 2]}}
            converter.vector_size = vector_size
            converter.coups_flv_indep = [base_objects.FLV_Coupling(
                'FLV_STATIC', {(0, 2): 'GC_STATIC'})]
            converter.coups_flv_dep = [base_objects.FLV_Coupling(
                'FLV_DEP', {(1, 2): 'GC_DEP1', (2, 1): 'GC_DEP2'})]
            converter.create_couplings_flavor_merged()

            # The normal vector include is owned by the caller, even when
            # MODEL_OBJECT needs its extents to declare public flavour tables.
            exporter = object.__new__(export_v4.ProcessExporterFortran)
            exporter.opt = {'output_options': {
                'vector_size': max(1, vector_size), 'nb_warp': nb_warp}}
            with export_v4.writers.FortranWriter(str(source / 'vector.inc')) as writer:
                exporter.write_vector_size(writer)
            suffix = '(VECSIZE_MEMMAX)' if vector_size else ''
            born_support.write_fortran(model / 'coupl.inc', [
                'double complex,target::GC_STATIC,GC_DEP1%s,GC_DEP2%s' %
                (suffix, suffix),
                'common/test_couplings/GC_STATIC,GC_DEP1,GC_DEP2'])
            table = 'FLV_DEP(lane)' if vector_size else 'FLV_DEP'
            coupling = 'GC_DEP1(lane)' if vector_size else 'GC_DEP1'
            second = 'GC_DEP2(lane)' if vector_size else 'GC_DEP2'
            born_support.write_fortran(model / 'check.f', [
                'program check', 'use model_object', 'implicit none',
                "include '../vector.inc'", "include 'coupl.inc'", 'integer lane',
                'GC_STATIC=(7d0,-3d0)',
                'do lane=1,VECSIZE_MEMMAX',
                coupling+'=dcmplx(lane,1d0)', second+'=dcmplx(-lane,2d0)',
                'enddo', 'call init_flv_couplings()',
                'if(FLV_STATIC%PARTNER(1).ne.2)stop 1',
                'if(FLV_STATIC%PARTNER2(2).ne.1)stop 2',
                'if(.not.associated(FLV_STATIC%VAL(1)%p,GC_STATIC))stop 3',
                'do lane=1,VECSIZE_MEMMAX',
                'if('+table+'%PARTNER(1).ne.2)stop 4',
                'if('+table+'%PARTNER2(2).ne.1)stop 5',
                'if(.not.associated('+table+'%VAL(1)%p,'+coupling+'))stop 6',
                'if(.not.associated('+table+'%VAL(2)%p,'+second+'))stop 7',
                table+'%VAL(1)%p=dcmplx(100+lane,-lane)',
                'if('+coupling+'.ne.dcmplx(100+lane,-lane))stop 8',
                'if('+second+'.ne.dcmplx(-lane,2d0))stop 9',
                'enddo', "print *, 'PASS flavour coupling lane pointers'", 'end'])
            result = subprocess.run([
                shutil.which('gfortran'), '-O2', '-fcheck=all',
                '-ffixed-line-length-none', 'flavor_couplings.f', 'check.f',
                '-o', 'check'], cwd=model, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([str(model / 'check')], cwd=model,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('PASS flavour coupling lane pointers', result.stdout)

    def test_scalar_flavour_couplings(self):
        self.check_generated_model(vector_size=0, nb_warp=1)

    def test_vector_flavour_couplings(self):
        for width, warps in ((1, 1), (3, 2)):
            with self.subTest(width=width, warps=warps):
                self.check_generated_model(vector_size=width, nb_warp=warps)


class ModelConverted(Exception):
    """Stop after the dependency selection under test, before compiling loops."""


class CouplingMatrix(dict):
    def __init__(self, model, couplings):
        super().__init__(processes=[{'model': model, 'has_born': True}])
        self.couplings = couplings

    def get_used_lorentz(self):
        return ['FFV1', 'FFV1']

    def get_used_couplings(self):
        return self.couplings


class TestLoopCheckFlavourDependencies(unittest.TestCase):
    def check_export_dependencies(self, timer):
        first = base_objects.FLV_Coupling(
            'FLV_A', {(1, 2): 'GC_1', (2, 1): '-GC_2'})
        equivalent = base_objects.FLV_Coupling(
            'FLV_A', {(2, 1): '-GC_2', (1, 2): 'GC_1'})
        alias = base_objects.FLV_Coupling('FLV_B', dict(first['flavors']))
        different = base_objects.FLV_Coupling(
            'FLV_A', {(1, 2): 'GC_1', (2, 1): 'GC_2'})
        model = object()
        matrix = CouplingMatrix(model, [
            [first, 'GC_1', equivalent], [alias, '-GC_2', different, 'GC_1']])
        kind = (process_checks.LoopMatrixElementTimer if timer else
                process_checks.LoopMatrixElementEvaluator)
        evaluator = object.__new__(kind)
        evaluator.full_model = model
        evaluator.loop_optimized_output = False
        evaluator.mg_root = '/unused'
        evaluator.cmass_scheme = False
        evaluator.cuttools_dir = None
        evaluator.tir_dir = {}
        evaluator.cmd = SimpleNamespace(options={
            'fortran_compiler': 'gfortran', 'output_dependencies': 'external'})
        evaluator.proliferate = False
        evaluator.auth_skipping = False
        evaluator.stored_quantities = {}
        with tempfile.TemporaryDirectory(prefix='mg7_check_dependencies_') as tmp:
            evaluator.output_path = tmp
            exporter = mock.Mock()
            exporter.convert_model.side_effect = ModelConverted
            with mock.patch.object(loop_exporters, 'LoopProcessExporterFortranSA',
                                   return_value=exporter), \
                 mock.patch.object(process_checks.helas_call_writers,
                                   'FortranUFOHelasCallWriter'):
                with self.assertRaises(ModelConverted):
                    if timer:
                        evaluator.setup_process(matrix, str(Path(tmp) / 'output'))
                    else:
                        evaluator.evaluate_matrix_element(matrix, p=[[1., 0., 0., 1.]])
            actual_model, lorentz, couplings = exporter.convert_model.call_args.args
            self.assertIs(actual_model, model)
            self.assertEqual(lorentz, ['FFV1'])
            self.assertEqual(couplings, [first, 'GC_1', alias, '-GC_2', different])
            self.assertIs(couplings[0], first)

    def test_loop_evaluator_keeps_canonical_flavour_dependencies(self):
        self.check_export_dependencies(timer=False)

    def test_loop_timer_keeps_canonical_flavour_dependencies(self):
        self.check_export_dependencies(timer=True)
