"""Helpers shared by the FKS tests."""

import glob
import json
import math
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
# the kinematics of the asymmetric map of an incoming emitter, from the
# MCcntRefactor_Sfun branch of MadGraph7 (fks_reference/
# make_fks_asymmetric_reference.py)
FKS_ASYMMETRIC_DATA = sorted(
    glob.glob(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "test_data",
            "fks_asymmetric",
            "*.json",
        )
    )
)

# the kinematics of the resonance-local recoil of a final-state emitter, from
# the MCcntRefactor_Sfun branch of MadGraph7, per Born configuration
# (fks_reference/make_fks_resonance_reference.py)
FKS_RESONANCE_DATA = sorted(
    glob.glob(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "test_data",
            "fks_resonance",
            "*.json",
        )
    )
)


def data_id(path):
    """The name of reference data in test ids, with the directory of the
    asymmetric map or of the resonance-local recoil."""
    name = os.path.basename(path)[:-5]
    if "fks_asymmetric" in path:
        return f"asymmetric-{name}"
    if "fks_resonance" in path:
        return f"resonance-{name}"
    return name


def same_massive_xi_map(data, conf, point):
    """Whether a point of reference data has the map of xi of genps_fks.f of
    main, which madspace has. The map of a massive final-state emitter of the
    MCcntRefactor_Sfun branch differs from it for y < 0, where r_xi below
    rat = xi_max / xi_norm gives the first solution: there it takes xi_hat =
    (stiny + (1 - stiny) r_xi^2) / rat in place of rat (stiny + (1 - stiny)
    (r_xi / rat)^2), which overshoots xi_max by stiny (1 / rat - rat), and
    stiny + (1 - stiny) r_xi in place of r_xi for the second solution (reported
    on PR 397). For y >= 0 rat is 1 and the maps are the same."""
    j = conf["j_fks"] - 1
    if not data.get("mg_branch") or j < data["nincoming"] or conf["masses"][j] == 0:
        return True
    return point["y_ev"] >= 0


def resonance_members(conf):
    """The members of the resonance of a configuration of reference data, 0-based
    positions in the real process, empty without the resonance-local recoil
    (and for the other reference data)."""
    resonance = conf.get("resonance")
    if resonance is None or not resonance["resonance_recoil"]:
        return []
    return [k for k, member in enumerate(resonance["resonance_members"]) if member]


def resonance_stretch(points):
    """Per point of reference data, the largest factor (E_K + |K|) / M_K by
    which the boost out of the rest frame of the resonance stretches a
    momentum, 1 without a resonance: the rounding of an emission in that frame
    grows by it, up to 77 for a Higgs boson far off its mass shell."""
    stretch = np.ones(len(points))
    for k, point in enumerate(points):
        mass2 = point.get("resonance_mass2", 0.0)
        if mass2 > 0:
            res = np.array(point["resonance_momentum"])
            stretch[k] = (res[0] + np.linalg.norm(res[1:])) / math.sqrt(mass2)
    return stretch


def forest_topology(data):
    """The Topology of the Born configuration of resonance reference data, from
    its iforest: a vertex per propagator with its two daughters, and the vertex
    of the three lines left (the incoming partons with s_hat, or the end of the
    t-channel)."""
    conf = data["configs"][0]
    masses = list(np.delete(np.array(conf["masses"]), conf["i_fks"] - 1))
    nin = data["nincoming"]

    def line(label):
        if label < 0:
            return f"p{-label - 1}"
        if label <= nin:
            return f"i{label - 1}"
        return f"o{label - nin - 1}"

    vertices, used = [], set()
    for index, d1, d2, _, _ in data["born_forest"]:
        vertices.append([line(index), line(d1), line(d2)])
        used.update([d1, d2])
    labels = list(range(1, len(masses) + 1))
    labels += [-k for k in range(1, len(data["born_forest"]) + 1)]
    vertices.append([line(label) for label in labels if label not in used])
    propagators = [ms.Propagator(mass=m, width=w) for m, w in data["born_props"]]
    return ms.Topology(ms.Diagram(masses[:nin], masses[nin:], propagators, vertices))


def fortran_sampling_units(data, j_fks, second=False):
    """The factors 1 - stiny and 1 - ctiny of the sampling that the Jacobians of
    reference data leave out: genps_fks.f of main for every emitter (only
    1 - ctiny for the second solution of a massive emitter, whose xi is
    linear), the MCcntRefactor_Sfun branch only for a final-state emitter
    (jac_sampling_factors)."""
    stiny, ctiny = 1e-6, 5e-7
    if data.get("jac_sampling_factors") and j_fks < 2:
        return 1.0
    return (1 - np.where(second, 0.0, stiny)) * (1 - ctiny)


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
    i_fks,
    j_fks,
    pdgs,
    colors,
    charges=None,
    masses=None,
    fks_j_from_i=None,
    splitting_types=("QCD",),
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
        splitting_types=list(splitting_types),
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
        splitting_types=config.splitting_types,
        tags=config.tags,
        extra_counterterm=config.extra_counterterm,
    )


# the index of the splitting types in born_orders of the reference data
SPLITTING_TYPES = {"QCD": 0, "QED": 1}


def order_positions(data):
    """The positions of the Born orders and of the NLO orders of the coupling
    orders of reference data (the orders entry of fks_configurations.json) in
    the amp_split arrays of the dump."""
    amp_split = data["split_orders"]["amp_split"]
    orders = data["fks_configurations"]["orders"]
    return (
        [amp_split.index(o) for o in orders["born"]],
        [amp_split.index(o) for o in orders["nlo"]],
    )


def at_nlo_orders(values, data):
    """Values of the dump over the positions of amp_split (the last axis) at
    the NLO orders."""
    return np.asarray(values)[..., order_positions(data)[1]]


def identical_factor(pdgs, nincoming=2):
    """The identical-particle factor of the final state of a process, the
    product of k! over its flavors of k particles."""
    final = list(pdgs[nincoming:])
    return float(np.prod([math.factorial(final.count(p)) for p in set(final)]))


def extra_born_madfks_factor(data, extra):
    """The factor by which the Born of an extra counterterm of MadFKS
    (born_cnt_<n>.f, divided by the identical-particle factor of the Born)
    differs from that of the process (divided by its own)."""
    exported = data["fks_configurations"]
    pdgs = exported["extra_counterterms"][extra]["born_pdgs"]
    return identical_factor(pdgs) / identical_factor(exported["born_pdgs"])


def extra_born_offsets(data):
    """The position of the first order of the Born of each extra counterterm
    among the Born inputs of FKSLimits (the Born orders first), and their
    number."""
    orders = data["fks_configurations"]["orders"]
    offsets, position = [], len(orders["born"])
    for extra in orders.get("extra_born", []):
        offsets.append(position)
        position += len(extra)
    return offsets, position


def has_collinear_born(data):
    """Whether reference data have the Born of the collinear counter-event
    apart (lepton PDFs, where an incoming emitter radiates without the
    projection of the event)."""
    return any(
        "collinear_born_blocks" in point
        for conf in data["configs"]
        for point in conf["points"] + conf.get("limit_points", [])
    )


def born_inputs(data, rows, k_perp, key="born_blocks"):
    """born and spin_correlated per Born input (the Born orders, then the
    orders of the Born of each extra counterterm, see
    FKSCouplingOrders.born_input_count), with the tensors of the
    spin-correlated Borns contracted with k_perp (zero where there are none),
    and color_correlated per Born order, from the Born blocks of pyAmpliCol
    at the points of the rows (configuration, point) of reference data, those
    of key where a point has them (collinear_born_blocks for the Born of the
    collinear counter-event). The Born of an extra counterterm is that of the
    configuration of the point, zero elsewhere."""
    configurations = data["fks_configurations"]["configurations"]
    count = len(data["fks_configurations"]["orders"]["born"])
    offsets, total = extra_born_offsets(data)
    n = len(data["fks_configurations"]["born_pdgs"])
    born = np.zeros((len(rows), total))
    spin = np.zeros((len(rows), total))
    color = np.zeros((len(rows), count, n, n))
    for k, (conf, point) in enumerate(rows):
        blocks = point.get(key, point["born_blocks"])

        def contracted(block):
            tensor = np.array(block.get("spin_correlated", np.zeros((4, 4))))
            return k_perp[k] @ tensor @ k_perp[k]

        for o, block in enumerate(blocks.get("orders", [blocks])):
            born[k, o] = block["born"]
            color[k, o] = block["color_correlated"]
            spin[k, o] = contracted(block)
        extra = configurations[conf["nfks"] - 1].get("extra_cnt_index", -1)
        for o, block in enumerate(blocks.get("extra", [])):
            born[k, offsets[extra] + o] = block["born"]
            spin[k, offsets[extra] + o] = contracted(block)
    return born, color, spin


def alpha(data):
    """The QED coupling of the Fortran, gal(1)^2 / (4 pi)."""
    return data["gal"] ** 2 / (4 * np.pi)


def madfks_pdf_prefactors(f_pdf, xi_max_c, xi_s_cut):
    """The prefactors of a change of the PDF scheme of MadFKS (f_pdfsch_d,
    f_pdfsch_p, f_pdfsch_l, shape (..., 3)) with min(xi_max_c, xiScut) for
    xiScut, as FKSPrefactors: MadFKS spreads delta(xi) over 0 to xiScut and
    adds log(xiScut), which misses part of them where xi_max_c of the
    collinear counter-event is below xiScut. The reference data come from a
    MadFKS without the fix, so this goes once they are made with it."""
    f_pdf = np.asarray(f_pdf, dtype=float)
    lim = np.minimum(np.asarray(xi_max_c, dtype=float), xi_s_cut)[..., None]
    # points without a collinear counter-event have no prefactors
    lim = np.where(lim > 0, lim, xi_s_cut)
    g = f_pdf[..., :1] * xi_s_cut
    log_lim = np.log(lim)
    return np.concatenate([g, g * log_lim, g * log_lim**2 / 2], axis=-1) / lim
