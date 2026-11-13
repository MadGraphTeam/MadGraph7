"""MLM reuses the ME's weighted diagram draw and only retries rejected draws."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

import madspace as ms


_STUB = r'''
#include "madspace/umami.h"
UmamiStatus umami_get_meta(UmamiMetaKey key, void* result) {
    if (key == UMAMI_META_DEVICE) *static_cast<UmamiDevice*>(result) = UMAMI_DEVICE_CPU;
    else if (key == UMAMI_META_PARTICLE_COUNT) *static_cast<int*>(result) = 6;
    else if (key == UMAMI_META_DIAGRAM_COUNT) *static_cast<int*>(result) = 8;
    else if (key == UMAMI_META_HELICITY_COUNT) *static_cast<int*>(result) = 1;
    else return UMAMI_ERROR_UNSUPPORTED_META;
    return UMAMI_SUCCESS;
}
UmamiStatus umami_initialize(UmamiHandle* handle, const char* mode) {
    *handle = new int[2]{0, mode[0] == 'z'}; return UMAMI_SUCCESS;
}
UmamiStatus umami_free(UmamiHandle handle) { delete[] static_cast<int*>(handle); return UMAMI_SUCCESS; }
UmamiStatus umami_matrix_element(UmamiHandle handle, size_t count, size_t stride, size_t offset,
    size_t ni, const UmamiInputKey* ik, const void* const* in,
    size_t no, const UmamiOutputKey* ok, void* const* out) {
    const int* diagrams = nullptr;
    for (size_t k = 0; k < ni; ++k)
        if (ik[k] == UMAMI_IN_DIAGRAM_INDEX) diagrams = static_cast<const int*>(in[k]);
    int call = ++*static_cast<int*>(handle);
    bool zero = static_cast<int*>(handle)[1];
    // A forced diagram without a topology must fail like MadMatrix does.
    if (diagrams) for (size_t i = offset; i < offset + count; ++i)
        if (diagrams[i] != -1 && diagrams[i] != 7) return UMAMI_ERROR;
    for (size_t k = 0; k < no; ++k) for (size_t i = offset; i < offset + count; ++i) {
        if (ok[k] == UMAMI_OUT_MATRIX_ELEMENT) static_cast<double*>(out[k])[i] = zero ? 0. : 1.;
        else if (ok[k] == UMAMI_OUT_DIAGRAM_AMP2)
            for (int d = 0; d < 8; ++d) static_cast<double*>(out[k])[d * stride + i] = zero ? 0. : 1.;
        else if (ok[k] == UMAMI_OUT_HELICITY_INDEX) static_cast<int*>(out[k])[i] = call;
        else {
            // Alternate an eligible draw, a masked diagram and two invalid
            // indices. The latter also check that eligibility never gathers
            // out of bounds. A retry must force the eligible diagram 7.
            const int initial[] = {7, 2, -1, 12};
            int given = diagrams ? diagrams[i] : -1;
            static_cast<int*>(out[k])[i] = given >= 0 ? given : initial[i % 4];
        }
    }
    return UMAMI_SUCCESS;
}
'''


class MLMDiagramTest(unittest.TestCase):
    def test_reuse_distribution(self):
        # Batch-size outputs live on the host even for a CUDA/HIP runtime, so
        # this kernel test needs no device-specific Python tensor dependency.
        device = {'cpu': ms.cpu_device, 'cuda': ms.cuda_device, 'hip': ms.hip_device}[
            os.environ.get('MADSPACE_TEST_DEVICE', 'cpu')]()
        context = ms.Context(device, 1)
        size = ms.Type([ms.batch_size])
        for preferred in (1, 3, 0, -1, 9):
            with self.subTest(preferred=preferred):
                fb = ms.FunctionBuilder(ms.NamedTypes([('batch', size)]),
                    ms.NamedTypes([('one', size), ('three', size), ('reused', size)]))
                batch = fb.input(0)
                probs = fb.gather_vector(fb.full([0, batch]), ms.Value([0., 0.2, 0., 0.8], [1, 4]))
                choice, replaced = fb.sample_discrete_probs_reuse(
                    fb.squeeze(fb.random(batch, 1)), probs, fb.full([preferred, batch]))
                by_choice = fb.batch_split_by_index(choice, 4)
                by_replaced = fb.batch_split_by_index(replaced, 2)
                for index, value in enumerate((by_choice[1], by_choice[3], by_replaced[0])):
                    fb.output(index, fb.batch_size([value]))
                runtime = ms.FunctionRuntime(fb.function(), context)
                one, three, reused = [int(np.from_dlpack(t)[0])
                    for t in runtime.call([np.array([10000], dtype=np.int64)])]
                self.assertEqual(one + three, 10000)
                self.assertEqual(reused, 10000 if preferred in (1, 3) else 0)
                if preferred in (1, 3):
                    self.assertEqual(one, 10000 if preferred == 1 else 0)
                else:
                    self.assertLess(abs(one / 10000 - 0.2), 0.025)

    def test_integrand_passes_clustered_diagram(self):
        self.check_integrand('nonzero')

    def test_zero_weight_uses_eligible_topology(self):
        self.check_integrand('zero')
        self.check_integrand('zero', include_diagram_output=False)

    def check_integrand(self, mode, include_diagram_output=True):
        with open(Path(__file__).parent / 'test_data/ttgg.json') as stream:
            diagrams = json.load(stream)
        topologies = [ms.Topology(ms.Diagram(
            d['incoming_masses'], d['outgoing_masses'],
            [ms.Propagator(*p) for p in d['propagators']], d['vertices']))
            for d in diagrams]
        # A subset of the ME's diagrams, with a nonzero diagram index, catches
        # accidentally passing a topology/config index instead of a diagram.
        topologies = topologies[:1]
        permutations = [[diagrams[0]['permutations'][0]]]
        indices, count, expected = [[7]], 8, 7
        mapping = ms.PhaseSpaceMapping(topologies[0], 13000.,
                                       permutations=permutations[0])
        with tempfile.TemporaryDirectory(prefix='mlm_diagram_') as directory:
            alphas_file = Path(directory) / 'alphas.info'
            alphas_file.write_text('AlphaS_Qs: [1.0, 10.0, 100.0, 1000.0, 100000.0]\nAlphaS_Vals: [0.1, 0.1, 0.1, 0.1, 0.1]\n')
            for history in ('all_diagrams', 'diagram', 'madevent'):
                if not include_diagram_output and history == 'all_diagrams':
                    continue
                for power in (-1, 4):
                    with self.subTest(history=history, power=power):
                        context = ms.Context(ms.cpu_device(), 1)
                        grid = ms.AlphaSGrid(str(alphas_file))
                        grid.initialize_globals(context)
                        source = Path(directory) / 'matrix.cpp'
                        library = Path(directory) / 'matrix.so'
                        if not library.exists():
                            source.write_text(_STUB)
                            subprocess.check_call(['c++', '-shared', '-fPIC', '-std=c++17',
                                '-I' + str(Path(__file__).resolve().parents[1] / 'include'),
                                str(source), '-o', str(library)])
                        api = context.load_matrix_element(str(library), mode)
                        me = ms.MatrixElement(
                            api.index(), 6, [ms.MatrixElement.momenta_in, ms.MatrixElement.diagram_in],
                            [ms.MatrixElement.matrix_element_out, ms.MatrixElement.diagram_amp2_out,
                             ms.MatrixElement.color_index_out, ms.MatrixElement.helicity_index_out,
                             *([ms.MatrixElement.diagram_index_out] if include_diagram_output else [])], count)
                        xs = ms.DifferentialCrossSection(me, 13000., None, ms.CachedScale())
                        clustering = ms.MLMClustering(
                            topologies, permutations, indices,
                            cm_energy=13000., external_pdg_ids=[21, 21, 6, -6, 21, 21],
                            clustering_history=getattr(ms.MLMClustering.ClusteringHistory, history))
                        integrand = ms.Integrand(
                            mapping, [xs], running_coupling=ms.RunningCoupling(grid),
                            energy_scale=ms.EnergyScale(clustering), mlm_history_qcd_power=power,
                            channel_indices=[0])
                        result = ms.FunctionRuntime(integrand.function(), context)(
                            np.array([37], np.int64))
                        color, actual = np.asarray(result[2]), np.asarray(result[4])
                        self.assertTrue(np.all(np.isfinite(result[0])))
                        np.testing.assert_array_equal(color, actual)
                        if history == 'all_diagrams':
                            np.testing.assert_array_equal(actual, np.resize([7, 2, -1, 12], 37))
                            expected_calls = np.ones(37, dtype=int)
                        else:
                            np.testing.assert_array_equal(actual, np.full(37, expected))
                            expected_calls = np.full(37, 2)
                            if power >= 0 and include_diagram_output:
                                expected_calls[::4] = 1
                        np.testing.assert_array_equal(result[3], expected_calls)


if __name__ == '__main__':
    unittest.main()
