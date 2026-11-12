"""The integrand gives the matrix element the terminal diagram of its MLM walk."""

import json
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
UmamiStatus umami_initialize(UmamiHandle* handle, const char*) {
    *handle = new int(0); return UMAMI_SUCCESS;
}
UmamiStatus umami_free(UmamiHandle handle) { delete static_cast<int*>(handle); return UMAMI_SUCCESS; }
UmamiStatus umami_matrix_element(UmamiHandle handle, size_t count, size_t stride, size_t offset,
    size_t ni, const UmamiInputKey* ik, const void* const* in,
    size_t no, const UmamiOutputKey* ok, void* const* out) {
    const int* diagrams = nullptr;
    for (size_t k = 0; k < ni; ++k)
        if (ik[k] == UMAMI_IN_DIAGRAM_INDEX) diagrams = static_cast<const int*>(in[k]);
    int call = ++*static_cast<int*>(handle);
    for (size_t k = 0; k < no; ++k) for (size_t i = offset; i < offset + count; ++i) {
        if (ok[k] == UMAMI_OUT_MATRIX_ELEMENT) static_cast<double*>(out[k])[i] = 1.;
        else if (ok[k] == UMAMI_OUT_DIAGRAM_AMP2)
            for (int d = 0; d < 8; ++d) static_cast<double*>(out[k])[d * stride + i] = 1.;
        else if (ok[k] == UMAMI_OUT_HELICITY_INDEX) static_cast<int*>(out[k])[i] = call;
        else static_cast<int*>(out[k])[i] = diagrams ? diagrams[i] : -1;
    }
    return UMAMI_SUCCESS;
}
'''


class MLMDiagramTest(unittest.TestCase):
    def test_integrand_passes_clustered_diagram(self):
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
                        api = context.load_matrix_element(str(library), '')
                        me = ms.MatrixElement(
                            api.index(), 6, [ms.MatrixElement.momenta_in, ms.MatrixElement.diagram_in],
                            [ms.MatrixElement.matrix_element_out, ms.MatrixElement.diagram_amp2_out,
                             ms.MatrixElement.color_index_out, ms.MatrixElement.helicity_index_out,
                             ms.MatrixElement.diagram_index_out], count)
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
                        np.testing.assert_array_equal(actual, np.full(37, expected))
                        expected_calls = 1 if history == 'all_diagrams' else 2
                        np.testing.assert_array_equal(result[3], np.full(37, expected_calls))


if __name__ == '__main__':
    unittest.main()
