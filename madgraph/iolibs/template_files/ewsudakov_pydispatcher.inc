"""
EW Sudakov Python Dispatcher

This module provides the interface between Python reweighting code and the
Fortran EW Sudakov routines. PDG mappings are filled in at generation time.
"""
import importlib
import itertools
import os
import sys
import numpy as np

# SubProcesses directory of this output, located from __file__

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_RW_ME_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))  # bin/internal -> bin -> rw_me
_SUBPROCESSES_DIR = os.path.join(_RW_ME_DIR, 'SubProcesses')

if _SUBPROCESSES_DIR not in sys.path:
    sys.path.insert(0, _SUBPROCESSES_DIR)


def import_lib(name):
    """Load the process-specific f2py module from this output directory."""
    old_dir = os.getcwd()
    old_path = sys.path[:]
    try:
        process_dir = os.path.join(_SUBPROCESSES_DIR, name)
        sys.path.insert(0, process_dir)
        os.chdir(process_dir)
        return importlib.import_module('ewsudpy_%s' % name)
    finally:
        os.chdir(old_dir)
        sys.path[:] = old_path


# Filled in at export time; PDG keys are ((initial), (sorted final))

pdir_list = [%(pdir_list)s]

pdg2ewsud_dict = {%(pdg2sud)s}

original_pdg_list_dict = {%(pdgsorted)s}


def ewsudakov(pdgs, p, g):
    """
    Compute scalar EW Sudakov correction.

    Args:
        pdgs: ((initial), (sorted_final)) tuple
        p: list of 4-momenta [[E,px,py,pz], ...]
        g: strong coupling constant

    Returns:
        numpy array [born, sud0, sud1, sud2, sud3, sud4]
    """
    if pdgs not in pdg2ewsud_dict:
        raise KeyError(f"No EW Sudakov module for PDGs: {pdgs}. "
                      f"Available: {list(pdg2ewsud_dict.keys())}")

    p_transp = np.array([[pp[i] for pp in p] for i in range(4)], order='F')
    res = np.array([0., 0., 0., 0., 0., 0.], order='F')

    pdg2ewsud_dict[pdgs].ewsudakov_py(p_transp, g, res)
    return res


# Caps of density_interference_py (sub_f2py_ewsudakov.f); MAX_RES is the
# f2py shape of res_indices/res_dims and must match it
MAX_DIM = 36
MAX_RES = 10
DENSITY_VERSION = 4
PROCESS_TENSOR_VERSION = 1
# Kernel variant order, as ewsudakov_py results(2:6)
VARIANTS = ('qed', 'weak', 'sr_off', 'sr_and_floor_off', 'floor_off')
# Beams with two helicity states, as open incoming spins require
TWO_SPIN_BEAMS = frozenset([1, 2, 3, 4, 5, 6, 11, 13, 15, 21, 22])
# HELAS helicity labels of a two- and a three-state particle
HELICITIES = {2: (-1, 1), 3: (-1, 0, 1)}


def density_sudakov(pdgs, p, g, res_indices, res_dims, keep_initial=False,
                    nvariants=5, goldstone_relative=True):
    """
    Compute density matrix EW Sudakov for spin-correlated reweighting.

    Args:
        pdgs: ((initial), (sorted_final)) tuple
        p: list of 4-momenta [[E,px,py,pz], ...]
        g: strong coupling constant
        res_indices: list of resonance leg indices (1-based)
        res_dims: list of spin dimensions (2 for fermion, 3 for vector)
        keep_initial: also retain both incoming spin indices in process_tensor
        nvariants: evaluate the first nvariants of the five VARIANTS
        goldstone_relative: also evaluate the Goldstone-relative comparison
          (density_relative, zero_gold); otherwise they are None and 0

    Returns:
        dict with:
          - density_born: physical Born matrix, full complex array
          - density_correction: full linear interference, shape (D,D,V)
            with V=nvariants
          - density_relative: Goldstone-normalised comparison, shape (D,D,V)
            (None without goldstone_relative)
          - zero_gold: number of nonzero physical Born states with zero
            Goldstone norm (relative prescription undefined there)
          - total_dim, nres: dimensions
    """
    if keep_initial:
        return density_process_tensor(pdgs, p, g, res_indices, res_dims,
                                      nvariants, goldstone_relative)
    if pdgs not in pdg2ewsud_dict:
        raise KeyError(f"No EW Sudakov module for PDGs: {pdgs}. "
                      f"Available: {list(pdg2ewsud_dict.keys())}")

    if not 1 <= nvariants <= 5:
        raise ValueError('Invalid number of density variants')
    nres = len(res_indices)
    if nres != len(res_dims) or not 1 <= nres <= MAX_RES:
        raise ValueError('Invalid density resonance arrays')
    if any(d not in (2, 3) for d in res_dims):
        raise ValueError('Density resonances must have two or three spin states')
    total_dim = int(np.prod(res_dims))
    if total_dim > MAX_DIM:
        raise ValueError('Density helicity space exceeds compiled capacity')
    if (len(set(res_indices)) != nres or
            any(i < 1 or i > len(p) for i in res_indices)):
        raise ValueError('Invalid density resonance positions')
    p_transp = np.asfortranarray(np.asarray(p, dtype=float).T)
    if p_transp.shape != (4, len(p)) or not np.isfinite(p_transp).all():
        raise ValueError('Invalid density momenta')
    res_idx_arr = np.zeros(MAX_RES, dtype=np.int32, order='F')
    res_dim_arr = np.zeros(MAX_RES, dtype=np.int32, order='F')
    for i, (idx, dim) in enumerate(zip(res_indices, res_dims)):
        res_idx_arr[i] = idx
        res_dim_arr[i] = dim

    kernel = getattr(pdg2ewsud_dict[pdgs], 'density_interference_py', None)
    if kernel is None:
        raise NotImplementedError('Regenerate rw_me: this compiled Sudakov '
                                  'library has no full density interface')
    density_born, correction, relative, zero_gold, status = kernel(
        p_transp, g, nres, res_idx_arr, res_dim_arr, nvariants,
        int(bool(goldstone_relative)))
    if status == 3:
        raise NotImplementedError('The Sudakov kernels need YMT = MT and ntadpole = 1')
    if status == 4:
        raise NotImplementedError('The density kernel cannot retain a polarised leg')
    if status:
        raise RuntimeError(f'Full density kernel failed with status {status}')
    density_born = density_born[:total_dim, :total_dim]
    correction = correction[:total_dim, :total_dim, :nvariants]
    relative = (relative[:total_dim, :total_dim, :nvariants]
                if goldstone_relative else None)
    # relative is checked by its users: a tiny Goldstone norm can overflow
    # it while the physical matrices stay finite.
    if not np.isfinite(density_born).all() or not np.isfinite(correction).all():
        raise FloatingPointError('Non-finite full density interference')

    return {
        'version': DENSITY_VERSION,
        'density_born': density_born,
        'density_correction': correction,
        'density_relative': relative,
        'zero_gold': int(zero_gold),
        'total_dim': total_dim,
        'nres': nres
    }


def density_process_tensor(pdgs, p, g, res_indices, res_dims, nvariants=5,
                           goldstone_relative=True):
    """Retain incoming coherences as well as the reduced outgoing state.

    The combined index is i = a + 4*h: incoming leg 1 varies fastest,
    followed by incoming leg 2 and the supplied resonance legs. Thus the
    matrix can be viewed as (h,a,h_prime,a_prime), with a 4x4 block for
    every outgoing-helicity pair. Incoming colour averages and identical
    final-particle factors are retained; the incoming spin average is undone.
    This is a scattering tensor, not a joint state of incoming and outgoing
    particles. The first nvariants linear EW variants are retained without
    normalisation, and the Goldstone-relative 'relative' with
    goldstone_relative.
    """
    initial = tuple(pdgs[0])
    if len(initial) != 2 or any(abs(pid) not in TWO_SPIN_BEAMS for pid in initial):
        raise NotImplementedError('Open initial spins require two quark, charged '
                                  'lepton, photon or gluon beams')
    if not res_indices or any(i <= 2 for i in res_indices):
        raise ValueError('Open-tensor resonance positions must be outgoing legs')
    dimension = int(np.prod(res_dims))
    if 4 * dimension > MAX_DIM:
        raise NotImplementedError('Open initial/outgoing spin space exceeds '
                                  'compiled capacity (36 combined states)')
    legs = [1, 2] + list(res_indices)
    dims = [2, 2] + list(res_dims)
    raw = density_sudakov(pdgs, p, g, legs, dims, nvariants=nvariants,
                          goldstone_relative=goldstone_relative)
    # The kernel divides by the usual initial spin/colour multiplicities.
    # Remove the spin divisor only; contract with rho_in=I_4/4 to recover B.
    tensor = {
        'version': PROCESS_TENSOR_VERSION,
        'born': 4.0 * raw['density_born'],
        'correction': 4.0 * raw['density_correction'],
        'incoming_dim': 4,
        'outgoing_dim': dimension,
        'leg_indices': legs,
        'spin_dims': dims,
        'index_order': 'first listed leg fastest; i=a+incoming_dim*h',
        'helicities': [list(h[::-1]) for h in itertools.product(
            *(HELICITIES[d] for d in reversed(dims)))],
        'normalisation': 'incoming spin unaveraged; native colour and symmetry divisors retained',
        'variants': list(VARIANTS[:nvariants]),
        'pdg_key': [list(part) for part in pdgs],
        'pdgs': [list(part) for part in original_pdg_list_dict.get(pdgs, pdgs)],
        'momenta': np.asarray(p, dtype=float).tolist(),
        'gstr': float(g),
        'zero_gold': raw['zero_gold'],
        'basis': 'native HELAS helicities at the stored momenta',
    }
    if goldstone_relative:
        tensor['relative'] = 4.0 * raw['density_relative']
    result = dict(version=DENSITY_VERSION, total_dim=dimension,
                  nres=len(res_indices), zero_gold=raw['zero_gold'],
                  process_tensor=tensor, density_relative=None)
    # The incoming spin average of the open kernel result (rho_in=I_4/4).
    for field in ('density_born', 'density_correction', 'density_relative'):
        array = raw[field]
        if array is None:
            continue
        result[field] = np.trace(
            array.reshape((dimension, 4, dimension, 4) + array.shape[2:]),
            axis1=1, axis2=3)
    return result
