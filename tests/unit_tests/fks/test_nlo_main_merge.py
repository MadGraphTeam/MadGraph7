"""Physical FKS sectors retain distinct upstream MC-history ownership."""
import copy
import os
import shutil
import subprocess
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from madgraph import MadGraph5Error
from madgraph import MG5DIR
from madgraph.core import base_objects
from madgraph.core import helas_objects
from madgraph.loop.loop_helas_objects import LoopHelasMatrixElement
from madgraph.fks import fks_common
from madgraph.iolibs import born_support, model_state
from madgraph.iolibs.export_fks import ProcessExporterFortranFKS


def process(pdgs):
    return base_objects.Process({'legs': base_objects.LegList([
        base_objects.Leg({'id': pdg, 'number': i+1, 'state': i >= 2})
        for i, pdg in enumerate(pdgs)])})


class BornElement(dict):
    def get_nexternal_ninitial(self):
        return 3, 2


class TestPhysicalNativeHistories(unittest.TestCase):
    def test_sudakov_keeps_scalar_and_explicit_onehel_abis_distinct(self):
        source = ('      call sborn_onehel(p,nhel(1,i),i,ans)\n'
                  '      call sborn_onehel(p,nhel(1,i),i,ans,\n'
                  '     $     ret_saveamp)\n')
        actual = born_support.explicit_onehel_calls(source)
        self.assertIn('call sborn_onehel(p,nhel(1,i),i,ans)', actual)
        self.assertIn('call sborn_onehel_amp(p,nhel(1,i),i,ans,', actual)

    @unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
    def test_native_weight_growth_preserves_state_without_quadratic_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'check.f90').write_text('''program check
            use weight_lines
            implicit none
            call allocate_weight_lines(5)
            fks_flavor_class(1)=2
            born_flavor_class(1)=3
            native_ids(:,1)=[4,5,6]
            group_size(1)=1
            group_members(1)=1
            call weight_lines_allocated(5,65,3,2)
            if(allocated(icontr_sum))stop 1
            if(allocated(parton_pdf))stop 2
            if(allocated(icolour_con))stop 3
            if(fks_flavor_class(1).ne.2)stop 4
            if(born_flavor_class(1).ne.3)stop 5
            if(any(native_ids(:,1).ne.[4,5,6]))stop 6
            if(group_size(1).ne.1.or.group_members(1).ne.1)stop 7
            if(max_contr.lt.65.or.max_wgt.ne.3.or.max_iproc.ne.2)stop 8
            call deallocate_weight_lines
            if(allocated(group_members))stop 9
            end program
            ''')
            build = subprocess.run(['gfortran', '-fcheck=all', '-ffixed-line-length-none',
                str(Path(MG5DIR)/'Template/NLO/SubProcesses/mc_context_memory.f90'), str(Path(MG5DIR)/'Template/NLO/SubProcesses/weight_lines.f'),
                'check.f90', '-o', 'check'], cwd=root, capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stdout+build.stderr)
            run = subprocess.run(['./check'], cwd=root, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout+run.stderr)

    def test_virtual_flavour_population_keeps_madloop_numbering(self):
        matrix = LoopHelasMatrixElement()
        def populate(instance, model=None):
            instance._flavor_allow_trimming = True
        with mock.patch.object(helas_objects.HelasMatrixElement,
                               'populate_flavor_validity', populate):
            matrix.populate_flavor_validity()
        self.assertFalse(matrix._flavor_allow_trimming)

    def test_module_array_imports_are_not_unknown_function_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root/'Source/MODEL'
            model.mkdir(parents=True)
            (model/'couplings.f90').write_text('''module couplings
            complex(8), allocatable :: gc_vec(:)
            end module
            module forwarded
            use couplings, only: renamed => gc_vec
            end module
            subroutine born(x)
            use forwarded, only: local => renamed
            real(8) x
            x=real(local(1))
            end
            subroutine unknown(x)
            real(8) x,missing
            x=missing(x)
            end
            ''')
            sources = model_state._Sources(root, [], {'gc_vec'})
            self.assertEqual(sources.readset(sources.by_name['born'])[0], {'gc_vec'})
            with self.assertRaisesRegex(MadGraph5Error, 'unavailable callee missing'):
                sources.readset(sources.by_name['unknown'])
            without_snapshot = model_state._Sources(root, [], set())
            with self.assertRaisesRegex(MadGraph5Error, 'lacks a model snapshot contract'):
                without_snapshot.readset(without_snapshot.by_name['born'])

    def test_flavour_couplings_keep_exact_model_state_keys(self):
        matrix = SimpleNamespace(get_used_couplings=lambda **kw: [
            [{'flavors': {(1, 2): 'GC_1', (3, 4): '-GC_2'}}], ['GC_3']])
        with mock.patch.object(born_support, 'helicity_coupling_degrees',
                               return_value={'gc_1': 1, 'gc_2': 2, 'gc_3': 1}):
            result = born_support.helicity_matrix_metadata(matrix, {})
        self.assertFalse(result['homogeneous'])
        self.assertEqual(result['couplings'], {'gc_1': 1, 'gc_2': 2, 'gc_3': 1})

    def setUp(self):
        self.matrix = {'processes': [process([81, -81, 22, 22])],
                       'identical_particle_factor': 2}
        fks = dict(i=4, j=1, ij=1, ij_id=81, extra_cnt_index=-1,
                   splitting_type=['QED'])
        self.rows = [dict(n_me=1, fks_info=copy.deepcopy(fks), flavor_class={
            'real_pdgs': [q, -q, 22, 22], 'born_pdgs': [q, -q, 22]})
            for q in (2, 1)]
        self.born = SimpleNamespace(
            born_me=BornElement(processes=[process([81, -81, 22])], identical_particle_factor=1),
            real_processes=[SimpleNamespace(matrix_element=self.matrix,
                particle_tags=[False]*4, fks_j_from_i={4: [1]})],
            get_fks_info_list=lambda: self.rows)

    def export(self):
        with tempfile.TemporaryDirectory() as directory:
            previous = os.getcwd()
            try:
                os.chdir(directory)
                object.__new__(ProcessExporterFortranFKS).write_mc_history_files(self.born)
                return (Path(directory)/'mc_histories.inc').read_text()
            finally:
                os.chdir(previous)

    def test_distinct_physical_flavours_have_distinct_history_groups(self):
        text = self.export()
        self.assertRegex(text, r'MC_HIST_COUNT\s*=\s*4')
        self.assertEqual([leg['id'] for leg in self.matrix['processes'][0]['legs']], [81, -81, 22, 22])

    def test_same_physical_history_is_still_rejected(self):
        self.rows[1]['flavor_class'] = copy.deepcopy(self.rows[0]['flavor_class'])
        with self.assertRaisesRegex(MadGraph5Error, 'Overlapping MC history'):
            self.export()

    def test_native_registry_resolves_physical_sectors_without_collision(self):
        sectors = []
        for number, row in enumerate(self.rows, 1):
            physical = fks_common.physical_fks_processes(self.matrix, row)
            sectors.append(dict(sector=number, real=1, fks=row['fks_info'],
                processes=[fks_common.external_process_identity(p, [False]*4) for p in physical],
                allowed=[(4, 1)]))
        record = dict(directory='P1', context=1, provider=1, nexternal=4,
                      nincoming=2, orders=['QED'], amp_orders=[[3]], sectors=sectors)
        histories = born_support.resolve_histories([record])
        self.assertEqual(len(histories['P1']), 2)
        for number, sector in enumerate(histories['P1'], 1):
            self.assertTrue(sector['histories'])
            self.assertTrue(all(h['sector'] == number for h in sector['histories']))


if __name__ == '__main__':
    unittest.main()
