"""Helpers shared by the FKS tests."""

import glob
import json
import os

import numpy as np
import torch
from madspace_testing import default_context

import madspace as ms

EPS = np.finfo(np.float64).eps
# the reference data of genps_fks.f and fks_singular.f (see
# test_fks_reference.py)
FKS_DATA = sorted(
    glob.glob(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "test_data", "fks", "*.json"
        )
    )
)
FKS_DATA_IDS = [os.path.basename(p)[:-5] for p in FKS_DATA]


def reference(path):
    with open(path) as f:
        return json.load(f)


def reference_born_mapping(data):
    """RAMBO mapping of the Born of reference data, with the masses of the
    Fortran."""
    conf = data["configs"][0]
    masses = np.delete(np.array(conf["masses"]), conf["i_fks"] - 1)
    return ms.PhaseSpaceMapping(
        list(masses),
        2 * np.sqrt(data["ebeam"][0] * data["ebeam"][1]),
        leptonic=data["lpp"] == [0, 0],
        mode=ms.PhaseSpaceMapping.TChannelMode.rambo,
    )


def evaluate(function, *args, fks_index=None):
    """The outputs of a FunctionGenerator on numpy arguments, with fks_index
    as the last argument where it is given, as numpy arrays with the batch
    dimension."""
    device = torch.get_default_device()
    tensors = [torch.as_tensor(np.asarray(a)).to(device) for a in args]
    if fks_index is not None:
        tensors.append(
            torch.as_tensor(np.asarray(fks_index), dtype=torch.int32).to(device)
        )
    runtime = ms.FunctionRuntime(function.function(), default_context())
    batch = len(args[0])
    outputs = []
    for out in runtime.call(tensors):
        out = torch.as_tensor(out).cpu().numpy()
        outputs.append(np.full(batch, out) if out.ndim == 0 else out)
    return outputs


def massive_fortran_precision(real, p_jbar, i_fks, j_fks, m_j):
    """Precision of genps_fks.f for an emission off a massive final-state
    emitter, from the real momenta (batch, n + 1, 4) in the Born c.m. frame and
    the Born emitter.

    genps_fks.f takes the discriminant D = A xi^2 + B xi + C of the momentum of
    the emitter as a sum, which cancels where the two solutions meet (D -> 0),
    and |k_j| = (-c xi y +- (2 - xi) sqrt(D)) / ((2 - xi (1 - y)) (2 - xi (1 + y)))
    as a difference, which cancels where |k_j| vanishes. Its Jacobian goes like
    |k_j|^2 / sqrt(D). Returns the precision of |k_j| in units of sqrt(s) and the
    relative precision of the Jacobian, both without a safety factor. In units
    of sqrt(s) the madspace kernels are as precise or better.
    """
    p_in = real[:, 0] + real[:, 1]
    rs = np.sqrt(p_in[:, 0] ** 2 - (p_in[:, 1:] ** 2).sum(1))
    p_i, p_j = real[:, i_fks], real[:, j_fks]
    kbar = np.linalg.norm(p_jbar[:, 1:], axis=1)
    erec = rs - p_jbar[:, 0]
    mh2 = (erec - kbar) * (erec + kbar) / rs**2
    mj2 = (m_j / rs) ** 2
    lam = (1 - mh2 - mj2) ** 2 - 4 * mh2 * mj2
    xi = 2 * p_i[:, 0] / rs
    kj = np.linalg.norm(p_j[:, 1:], axis=1) / rs
    ej = p_j[:, 0] / rs
    li = np.linalg.norm(p_i[:, 1:], axis=1)
    y = (p_i[:, 1:] * p_j[:, 1:]).sum(1) / (li * kj * rs)
    sqrt_d = np.abs((2 - xi) * kj + xi * y * ej)
    terms = xi**2 * (1 - mj2 * (1 - y * y)) + 2 * xi * (1 - mh2 - mj2) + lam
    e_d = EPS * terms / sqrt_d**2
    c = 1 - xi + mj2 - mh2
    den = (2 - xi * (1 - y)) * (2 - xi * (1 + y))
    e_k = (
        EPS * (np.abs(c * xi * y) + (2 - xi) * sqrt_d) + 0.5 * e_d * (2 - xi) * sqrt_d
    ) / (kj * den)
    return e_k * kj, 2 * e_k + 0.5 * e_d


def massive_xi_max(real, i_fks, j_fks, m_j):
    """Largest xi at the y of the real momenta (batch, n + 1, 4) in the Born c.m.
    frame, for a massive final-state emitter: xi_m = ((1 - m_j)^2 - m_rec^2) /
    (1 - m_j) for y >= 0, else the point where the two solutions for the
    momentum of the emitter meet (masses in units of sqrt(s))."""
    p_in = real[:, 0] + real[:, 1]
    rs = np.sqrt(p_in[:, 0] ** 2 - (p_in[:, 1:] ** 2).sum(1))
    p_i, p_j = real[:, i_fks], real[:, j_fks]
    k = p_i + p_j
    erec = rs - k[:, 0]
    klen = np.linalg.norm(k[:, 1:], axis=1)
    mh2 = (erec - klen) * (erec + klen) / rs**2
    mj = m_j / rs
    lam = (1 - mh2 - mj**2) ** 2 - 4 * mh2 * mj**2
    y = (p_i[:, 1:] * p_j[:, 1:]).sum(1) / (
        np.linalg.norm(p_i[:, 1:], axis=1) * np.linalg.norm(p_j[:, 1:], axis=1)
    )
    sq = mj * np.sqrt(4 * mh2 + (1 - y) * (1 + y) * lam)
    xi_1 = lam / (1 - mh2 - mj**2 + sq)
    xi_m = ((1 - mj) ** 2 - mh2) / (1 - mj)
    return np.where(y >= 0, xi_m, xi_1)


def reference_configurations(data):
    """The FKSConfigurations of reference data, as madspace.fks builds them."""
    from madspace.fks import fks_configurations

    return fks_configurations(reference_born_mapping(data), data["fks_configurations"])


def configuration(
    i_fks, j_fks, pdgs, colors, charges=None, masses=None, fks_j_from_i=None
):
    """An FKSConfiguration for the unit tests: massless and neutral unless
    given, with (i, j) as the only FKS pair unless the pairs are given."""
    n = len(pdgs)
    if fks_j_from_i is None:
        fks_j_from_i = [[] for _ in range(n)]
        fks_j_from_i[i_fks] = [j_fks]
    return ms.FKSConfiguration(
        i_fks=i_fks,
        j_fks=j_fks,
        pdgs=list(pdgs),
        colors=list(colors),
        charges=list(charges) if charges is not None else [0.0] * n,
        masses=list(masses) if masses is not None else [0.0] * n,
        fks_j_from_i=fks_j_from_i,
    )


def with_pair(config, i_fks, j_fks):
    """The configuration of the same real process with another FKS pair."""
    return ms.FKSConfiguration(
        i_fks=i_fks,
        j_fks=j_fks,
        pdgs=config.pdgs,
        colors=config.colors,
        charges=config.charges,
        masses=config.masses,
        fks_j_from_i=config.fks_j_from_i,
    )
