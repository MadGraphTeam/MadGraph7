"""Externally selected MLM diagrams through the generated UMAMI interface.

No MadSpace runtime or PDF is needed. The helper also runs with device buffers
when called by test_madmatrix_diagram_gpu.
"""

import ctypes as ct
import ctypes.util
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

import madgraph.interface.master_interface as Cmd
import madgraph.various.misc as misc


def check_library(library, card, color_data, flavor_count, backend, tolerance):
    lib = ct.CDLL(library)
    gpu = backend if backend in ('cuda', 'hip') else None
    runtime = None
    if gpu:
        name = 'cudart' if gpu == 'cuda' else 'amdhip64'
        runtime = ct.CDLL(ctypes.util.find_library(name) or 'lib%s.so' % name)
        count = ct.c_int()
        assert runtime[gpu + 'GetDeviceCount'](ct.byref(count)) == 0 and count.value

    handle = ct.c_void_p()
    assert lib.umami_initialize(ct.byref(handle), card.encode()) == 0
    ndiagrams = ct.c_int()
    assert lib.umami_get_meta(2, ct.byref(ndiagrams)) == 0
    masks = {}
    for row, diagram in re.findall(r'\{\s*([a-z, ]+)\}, // ICONFIG=\d+\s*<-- DIAGRAM=(\d+)',
                                   open(color_data).read()):
        masks[int(diagram)] = {i for i, flag in enumerate(row.strip().split(', '))
                               if flag == 'true'}
    assert len(masks) > 1

    def evaluate(flavors, diagrams=None, random_diagram=0.01, invalid=False, alpha_s=None):
        n, offset = len(flavors), 2
        stride = n + 5
        mom = (ct.c_double * (16 * stride))()
        for i in range(n):
            e, theta, phi = 500., 0.6 + 0.01 * i, 0.4
            p = [e * math.sin(theta) * math.cos(phi),
                 e * math.sin(theta) * math.sin(phi), e * math.cos(theta)]
            for leg, vec in enumerate(([e, 0, 0, e], [e, 0, 0, -e],
                                       [e] + p, [e] + [-v for v in p])):
                for mu, value in enumerate(vec):
                    mom[stride * (4 * mu + leg) + i + offset] = value
        flav = (ct.c_uint * stride)(*([0] * offset + flavors + [0] * 3))
        rnddiag = (ct.c_double * stride)(*([random_diagram] * stride))
        rndcol = (ct.c_double * stride)(*[(i + 0.5) / stride for i in range(stride)])
        inputs, in_keys = [mom, flav, rndcol, rnddiag], [0, 2, 3, 5]
        if alpha_s is not None:
            inputs.append((ct.c_double * stride)(*([alpha_s] * stride)))
            in_keys.append(1)
        if diagrams is not None:
            given = (ct.c_int * stride)(*([-1] * offset + diagrams + [-1] * 3))
            inputs.append(given)
            in_keys.append(7)
        sizes = [stride, stride * ndiagrams.value, stride, stride]
        outputs = [(kind * size)(*([-99] * size)) for kind, size in
                   zip((ct.c_double, ct.c_double, ct.c_int, ct.c_int), sizes)]
        hosts = inputs + outputs
        pointers = []
        try:
            for host in hosts:
                if gpu:
                    pointer = ct.c_void_p()
                    size = ct.c_size_t(ct.sizeof(host))
                    assert runtime[gpu + 'Malloc'](ct.byref(pointer), size) == 0
                    pointers.append(pointer.value)
                    assert runtime[gpu + 'Memcpy'](pointer, ct.c_void_p(ct.addressof(host)), size, 1) == 0
                else:
                    pointers.append(ct.addressof(host))
            status = lib.umami_matrix_element(
                handle, ct.c_size_t(n), ct.c_size_t(stride), ct.c_size_t(offset),
                ct.c_size_t(len(inputs)), (ct.c_int * len(inputs))(*in_keys),
                (ct.c_void_p * len(inputs))(*pointers[:len(inputs)]),
                ct.c_size_t(4), (ct.c_int * 4)(0, 1, 2, 4),
                (ct.c_void_p * 4)(*pointers[len(inputs):]))
            assert status == (1 if invalid else 0), status
            if gpu:
                assert runtime[gpu + 'DeviceSynchronize']() == 0
                for pointer, host in zip(pointers[len(inputs):], outputs):
                    assert runtime[gpu + 'Memcpy'](ct.c_void_p(ct.addressof(host)), ct.c_void_p(pointer),
                                                   ct.c_size_t(ct.sizeof(host)), 2) == 0
            if invalid:
                assert all(value == -99 for output in outputs for value in output), \
                    'invalid diagram must fail the whole batch without writing outputs'
                return
            for output in outputs:
                assert output[0] == output[1] == output[stride - 1] == -99
            return [[output[i + offset] for i in range(n)] for output in outputs[:1]] + [
                [[outputs[1][d * stride + i + offset] for d in range(ndiagrams.value)]
                 for i in range(n)],
                list(outputs[2])[offset:offset + n], list(outputs[3])[offset:offset + n]]
        finally:
            if gpu:
                for pointer in pointers:
                    assert runtime[gpu + 'Free'](ct.c_void_p(pointer)) == 0

    try:
        flavors = [i % flavor_count for i in range(19)]
        reference = evaluate(flavors)
        chosen = []
        for i, weights in enumerate(reference[1]):
            valid = [d for d in sorted(masks) if weights[d] > 0]
            assert valid
            chosen.append(-1 if i % 5 == 0 else valid[i % len(valid)])
        selected = evaluate(flavors, chosen)
        repeated = evaluate(flavors, chosen)
        other_random = evaluate(flavors, chosen, 0.99)
        def equivalent(left, right):
            assert left[2:] == right[2:], 'discrete choices must be reproducible'
            for a, b in zip(left[0] + sum(left[1], []), right[0] + sum(right[1], [])):
                assert math.isclose(a, b, rel_tol=tolerance, abs_tol=1e-25), (a, b)
        equivalent(selected, repeated)
        for i, diagram in enumerate(chosen):
            for value, ref in zip([selected[0][i]] + selected[1][i],
                                  [reference[0][i]] + reference[1][i]):
                assert math.isclose(value, ref, rel_tol=tolerance, abs_tol=1e-25), (value, ref)
            expected = diagram if diagram >= 0 else reference[3][i]
            assert selected[3][i] == expected, (i, diagram, selected[3][i])
            assert selected[2][i] in masks[expected], (i, expected, selected[2][i])
            if diagram >= 0:
                assert other_random[3][i] == diagram
                assert other_random[2][i] == selected[2][i]
        # A later call without the input must resume sampling, with no retained override.
        equivalent(evaluate(flavors), reference)
        for diagram in [-2, ndiagrams.value] + sorted(set(range(ndiagrams.value)) - masks.keys()):
            evaluate([0, 0], [chosen[1], diagram], invalid=True)
        equivalent(evaluate(flavors), reference)  # the handle survives a rejected batch
        zero = evaluate(flavors, [chosen[1]] * len(flavors), alpha_s=0.)
        assert zero[0] == [0.] * len(flavors) and zero[2] == [-1] * len(flavors), (zero[0], zero[2])
        print('%s: 19 strided events, %s flavors; diagrams, colors and unchanged |M|^2/amp2 passed'
              % (backend, flavor_count))
    finally:
        assert lib.umami_free(handle) == 0


class MadMatrixDiagramTest(unittest.TestCase):
    def setUp(self):
        self.path = tempfile.mkdtemp(prefix='madmatrix_diagram_')
        self.addCleanup(shutil.rmtree, self.path)

    def test_madmatrix_diagram(self):
        self.check_processes([('scalar', 'd', 1e-12), ('simd_128', 'd', 1e-12),
                             ('simd_128', 'm', 1e-6), ('scalar', 'f', 1e-5),
                             ('simd_128', 'f', 1e-5)])

    def test_madmatrix_diagram_gpu(self):
        # A CUDA host can also have hipcc installed without a HIP device (and
        # conversely). Allow a device-specific runner to select its backend.
        selected = os.environ.get('MADMATRIX_DIAGRAM_GPU_BACKEND')
        if selected not in (None, 'cuda', 'hip'):
            self.fail('MADMATRIX_DIAGRAM_GPU_BACKEND must be cuda or hip')
        builds = [(backend, fptype, tolerance)
                  for backend, compiler in [('cuda', 'nvcc'), ('hip', 'hipcc')]
                  if (selected is None or selected == backend) and misc.which(compiler)
                  for fptype, tolerance in [('d', 1e-12), ('m', 1e-6), ('f', 1e-5)]]
        if not builds:
            self.skipTest('neither nvcc nor hipcc available')
        self.check_processes(builds)

    def check_processes(self, builds):
        for process in ('q q > q q QED=0', 'g g > g g'):
            with self.subTest(process=process):
                self.check_builds(builds, process)

    def check_builds(self, builds, process):
        cmd = Cmd.MasterCmd()
        cmd.no_notification()
        cmd.exec_cmd('define q = u d')
        cmd.exec_cmd('generate ' + process)
        cmd.exec_cmd('output standalone %s -f' % self.path)
        with open(os.path.join(self.path, 'SubProcesses', 'madmatrix_processes.json')) as stream:
            sub, = json.load(stream)['subprocesses']
        proc = os.path.join(self.path, 'SubProcesses', sub['path'])
        if process.startswith('q q'):
            self.assertGreater(len(sub['flavors']), 1)
        for backend, fptype, tolerance in builds:
            subprocess.check_call(['make', 'cleanall'], cwd=proc,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            make = subprocess.run(['make', '-j2', 'BACKEND=' + backend, 'FPTYPE=' + fptype],
                                  cwd=proc, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            self.assertEqual(make.returncode, 0, make.stdout.decode(errors='replace')[-4000:])
            library, = glob.glob(os.path.join(self.path, 'lib', '*_P*_%s_%s.so' % (backend, fptype)))
            helper = subprocess.run([
                sys.executable, '-c',
                'from tests.acceptance_tests.test_madmatrix_diagram import check_library; '
                'import sys; check_library(*sys.argv[1:4], int(sys.argv[4]), sys.argv[5], float(sys.argv[6]))',
                library, os.path.join(self.path, 'Cards', 'param_card.dat'),
                os.path.join(proc, 'ColorData.h'), str(len(sub['flavors'])), backend, str(tolerance)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            self.assertEqual(helper.returncode, 0, helper.stdout.decode(errors='replace'))
            print(helper.stdout.decode().strip())
