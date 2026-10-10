"""Calling a madmatrix process library through umami from Python (ctypes), on the CPU or
on the GPU, for the checks of the GPU CI (interference_checks.py, madmatrix_checks.py).
"""

import ctypes
import ctypes.util
import math
import subprocess

GPU_BACKENDS = ('cuda', 'hip')

# umami.h
META_PARTICLE_COUNT, META_MASSES = 1, 5
IN_MOMENTA, IN_ALPHA_S, IN_FLAVOR_INDEX, IN_RANDOM_COLOR, IN_RANDOM_HELICITY = 0, 1, 2, 3, 4
OUT_MATRIX_ELEMENT, OUT_COLOR_INDEX, OUT_HELICITY_INDEX, OUT_GPU_STREAM = 0, 2, 3, 5


class Memory:
    """Buffers for umami: host memory for a CPU library, device memory for a GPU one
    (through the CUDA/HIP runtime the library itself is linked with)."""

    def __init__(self, backend, library):
        self.gpu = backend in GPU_BACKENDS
        self.hosts = []  # the host arrays umami reads on a CPU, alive until release
        if not self.gpu:
            return
        name, prefix = ('cudart', 'cuda') if backend == 'cuda' else ('amdhip64', 'hip')
        path = None
        ldd = subprocess.run(['ldd', library], capture_output=True, text=True).stdout
        for line in ldd.splitlines():
            if 'lib%s' % name in line and '=>' in line:
                path = line.split('=>')[1].split()[0]
        self.rt = ctypes.CDLL(path or ctypes.util.find_library(name) or 'lib%s.so' % name)
        self.malloc = getattr(self.rt, prefix + 'Malloc')
        self.malloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.memcpy = getattr(self.rt, prefix + 'Memcpy')
        self.memcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        self.free = getattr(self.rt, prefix + 'Free')
        self.free.argtypes = [ctypes.c_void_p]
        self.sync = getattr(self.rt, prefix + 'DeviceSynchronize')
        self.stream_create = getattr(self.rt, prefix + 'StreamCreateWithFlags')
        self.stream_create.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
        self.stream_destroy = getattr(self.rt, prefix + 'StreamDestroy')
        self.stream_destroy.argtypes = [ctypes.c_void_p]
        self.pointers = []

    def buffer(self, host):
        """A buffer for umami holding (a copy of) the ctypes array host."""
        if not self.gpu:
            self.hosts.append(host)
            return ctypes.addressof(host)
        pointer = ctypes.c_void_p()
        assert self.malloc(ctypes.byref(pointer), ctypes.sizeof(host)) == 0, 'device malloc'
        # cudaMemcpyHostToDevice == hipMemcpyHostToDevice == 1
        assert self.memcpy(pointer, ctypes.addressof(host), ctypes.sizeof(host), 1) == 0
        self.pointers.append(pointer)
        return pointer.value

    def fetch(self, pointer, host):
        """Copy a umami output back into the ctypes array host."""
        if not self.gpu:
            return
        assert self.sync() == 0, 'device synchronize'
        # cudaMemcpyDeviceToHost == hipMemcpyDeviceToHost == 2
        assert self.memcpy(ctypes.addressof(host), pointer, ctypes.sizeof(host), 2) == 0

    def release(self):
        self.hosts = []
        if self.gpu:
            for pointer in self.pointers:
                self.free(pointer)
            self.pointers = []

    def nonblocking_stream(self):
        """A stream that does not synchronise with the default stream (as a torch side
        stream): cudaStreamNonBlocking == hipStreamNonBlocking == 1."""
        stream = ctypes.c_void_p()
        assert self.stream_create(ctypes.byref(stream), 1) == 0, 'stream create'
        return stream.value


class Umami:
    """One umami instance of a process library."""

    def __init__(self, backend, library, param_card):
        self.lib = ctypes.CDLL(library)
        self.memory = Memory(backend, library)
        self.handle = ctypes.c_void_p()
        status = self.lib.umami_initialize(ctypes.byref(self.handle), param_card.encode())
        assert status == 0, 'umami_initialize status %d' % status
        npar = ctypes.c_int()
        self.lib.umami_get_meta(META_PARTICLE_COUNT, ctypes.byref(npar))
        self.npar = npar.value
        masses = (ctypes.c_double * self.npar)()
        assert self.lib.umami_get_meta(META_MASSES, masses) == 0, 'UMAMI_META_MASSES'
        self.masses = list(masses)
        self.lib.umami_matrix_element.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_size_t, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_void_p)]
        self.lib.umami_set_parameter.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                                 ctypes.c_double, ctypes.c_double]

    def set_parameter(self, name, value):
        return self.lib.umami_set_parameter(self.handle, name.encode(), value, 0.)

    def __call__(self, points, alpha_s=None, flavor=None, random_helicity=None,
                 random_color=None, outputs=(OUT_MATRIX_ELEMENT,), stream=None):
        """umami_matrix_element on these phase-space points (lists of npar 4-momenta)
        and per-event inputs; the outputs as lists, in the order asked for."""
        count = len(points)
        mom = (ctypes.c_double * (4 * self.npar * count))()
        for ievt, point in enumerate(points):  # momenta[count * (npar * mu + ipart) + ievt]
            for ipart in range(self.npar):
                for mu in range(4):
                    mom[count * (self.npar * mu + ipart) + ievt] = point[ipart][mu]
        in_keys, inputs = [IN_MOMENTA], [self.memory.buffer(mom)]
        for key, values, ctype in ((IN_ALPHA_S, alpha_s, ctypes.c_double),
                                   (IN_FLAVOR_INDEX, flavor, ctypes.c_int),
                                   (IN_RANDOM_HELICITY, random_helicity, ctypes.c_double),
                                   (IN_RANDOM_COLOR, random_color, ctypes.c_double)):
            if values is not None:
                in_keys.append(key)
                inputs.append(self.memory.buffer((ctype * count)(*values)))
        hosts = [((ctypes.c_double if key == OUT_MATRIX_ELEMENT else ctypes.c_int) * count)()
                 for key in outputs]
        out_keys = list(outputs)
        out_pointers = [self.memory.buffer(host) for host in hosts]
        if stream is not None:
            out_keys.append(OUT_GPU_STREAM)
            out_pointers.append(stream)
        status = self.lib.umami_matrix_element(
            self.handle, count, count, 0,
            len(in_keys), (ctypes.c_int * len(in_keys))(*in_keys),
            (ctypes.c_void_p * len(inputs))(*inputs),
            len(out_keys), (ctypes.c_int * len(out_keys))(*out_keys),
            (ctypes.c_void_p * len(out_pointers))(*out_pointers))
        assert status == 0, 'umami_matrix_element status %d' % status
        for pointer, host in zip(out_pointers, hosts):
            self.memory.fetch(pointer, host)
        self.memory.release()
        return [list(host) for host in hosts]

    def free(self):
        self.lib.umami_free(self.handle)


def rambo(masses_out, roots, rng, masses_in=(0., 0.)):
    """One phase-space point: two beams along z (masses masses_in) in their centre-of-mass
    frame, then RAMBO with the masses masses_out."""
    n = len(masses_out)
    q = []
    for _ in range(n):
        c, f = 2 * rng.random() - 1, 2 * math.pi * rng.random()
        e = -math.log(rng.random() * rng.random())
        s = math.sqrt(1 - c * c)
        q.append([e, e * s * math.cos(f), e * s * math.sin(f), e * c])
    big_q = [sum(qi[k] for qi in q) for k in range(4)]
    mass = math.sqrt(big_q[0] ** 2 - sum(big_q[k] ** 2 for k in (1, 2, 3)))
    b = [-big_q[k] / mass for k in (1, 2, 3)]
    g, x = big_q[0] / mass, roots / mass
    a = 1 / (1 + g)
    p = []
    for qi in q:
        bq = sum(b[k] * qi[k + 1] for k in range(3))
        p.append([x * (g * qi[0] + bq)] +
                 [x * (qi[k + 1] + b[k] * qi[0] + a * bq * b[k]) for k in range(3)])
    # rescale the three-momenta so that the energies add up to roots with the masses
    xi = 1.
    for _ in range(100):
        energies = [math.sqrt((xi * pi[0]) ** 2 + m * m) for pi, m in zip(p, masses_out)]
        f = sum(energies) - roots
        df = sum(xi * pi[0] ** 2 / e for pi, e in zip(p, energies))
        xi -= f / df
        if abs(f) < 1e-12 * roots:
            break
    out = []
    for pi, m in zip(p, masses_out):
        vec = [xi * v for v in pi[1:]]
        out.append([math.sqrt(sum(v * v for v in vec) + m * m)] + vec)
    m1, m2 = masses_in
    pz = math.sqrt((roots ** 2 - (m1 + m2) ** 2) * (roots ** 2 - (m1 - m2) ** 2)) / (2 * roots)
    return [[math.hypot(pz, m1), 0., 0., pz], [math.hypot(pz, m2), 0., 0., -pz]] + out
