"""Reference values for the Catani-Seymour dipoles of madspace: the real
matrix element and the Born blocks of its dipoles from pyAmpliCol, along the
soft and collinear limits.

For every setup below (Borns and real processes in the built-in Standard
Model of pyAmpliCol, pip install pyamplicol) the real emission approaches the
limits of each pair (i, j) of its dipoles along rays of the CSMapping of the
first dipole of the pair: the collinear limit at two azimuths and, for a
gluon i, the soft limit, with the random numbers of the emission going to
zero as lam = 1e-1 ... 1e-4. Each pair also has a few points away from the
limits, uniform in the random numbers of the emission (limit none, lam
null). At each point it stores the real matrix element
and, for every dipole of the real process, the Born blocks at the Born of the
dipole that CSDipoles projects the real emission onto: the Born, the
color-correlated Born <T_j . T_k> of its emitter j and spectator k
(all-outgoing color charges) and, for a gluon emitter, the tensor S with
<T_j . T_k>(e) = e^mu e^nu S_mu,nu the same with the real vector e in place
of the polarization vector of the emitter (contravariant components of e).
test_cs_dipoles.py reads only the JSON files
(madspace/tests/test_data/cs/<name>.json), so it needs no pyAmpliCol.

Run from anywhere with pyAmpliCol and madspace importable:

    python madspace/tests/cs_reference/make_cs_reference.py [--work DIR]

The JSON layout: setup (name, nincoming, sqrt_s, leptonic, masses {pdg:
mass}, alpha_s, source), borns, reals and cs (madspace.subtraction.cs.cs_subtraction),
points[]: real, ray (dipole: the index in cs of the dipole whose mapping
gives the ray, limit: collinear or soft, lam, random: the random numbers of
the mapping), momenta, x1, x2 (the real emission in the lab frame), real_me,
dipoles[] (per dipole of the real, in the order of cs): index, born_momenta
(in the order of the mapping of the dipole), born, color_correlated,
spin_correlated and born_spin_correlated (null without a gluon emitter).
"""

import argparse
import json
import os
import sys
import zlib

import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
DATA = os.path.join(HERE, "..", "test_data", "cs")

SQRT_S = 1000.0
ALPHA_S = 0.118
TOP_MASS = 173.0
LAMBDAS = (1e-1, 1e-2, 1e-3, 1e-4)
AZIMUTHS = (0.1, 0.35)
# points per pair of dipoles away from the limits
RANDOM_POINTS = 3

SETUPS = {
    "epem_uux": dict(
        borns=[[-11, 11, 2, -2]], reals=[[-11, 11, 2, -2, 21]], leptonic=True
    ),
    "epem_ttx": dict(
        borns=[[-11, 11, 6, -6]], reals=[[-11, 11, 6, -6, 21]], leptonic=True
    ),
    "epem_uuxg": dict(
        borns=[[-11, 11, 2, -2, 21], [-11, 11, 1, -1, 21]],
        reals=[[-11, 11, 2, -2, 21, 21], [-11, 11, 2, -2, 1, -1]],
        leptonic=True,
    ),
    "pp_epem": dict(
        borns=[[2, -2, 11, -11], [-2, 2, 11, -11]],
        reals=[[2, -2, 11, -11, 21], [21, -2, 11, -11, -2], [2, 21, 11, -11, 2]],
        leptonic=False,
    ),
    "pp_ttx": dict(
        borns=[[21, 21, 6, -6], [2, -2, 6, -6], [-2, 2, 6, -6]],
        reals=[
            [21, 21, 6, -6, 21],
            [2, -2, 6, -6, 21],
            [2, 21, 6, -6, 2],
            [21, 2, 6, -6, 2],
        ],
        leptonic=False,
    ),
    # the joint color and spin correlation of g g > t t~ g does not factorize;
    # its blocks cost 2.5 s per dipole and point in pyAmpliCol, so only the
    # first pair of each family and kernel, at one azimuth
    "pp_ttxg": dict(
        borns=[[21, 21, 6, -6, 21]],
        reals=[[21, 21, 6, -6, 21, 21]],
        leptonic=False,
        pairs="first of each family and kernel",
        azimuths=AZIMUTHS[:1],
        random_points=1,
    ),
}

# PDG code -> particle of the built-in Standard Model of pyAmpliCol
PARTICLES = {1: "d", 2: "u", 3: "s", 4: "c", 5: "b", 6: "t", 11: "e-", 13: "mu-"}
PARTICLES.update({21: "g", 22: "a", 23: "z", 25: "h"})


def particle(pdg):
    if pdg in (21, 22, 23, 25):
        return PARTICLES[pdg]
    name = PARTICLES[abs(pdg)]
    if pdg > 0:
        return name
    return name + "~" if abs(pdg) <= 6 else name[:-1] + "+"


def colored(pdg):
    return abs(pdg) <= 6 or pdg == 21


def runtime(pdgs, work, correlated):
    """The pyAmpliCol runtime of a process, with the color correlators of all
    pairs of colored particles and the spin correlators of its gluons where
    correlated is set."""
    import pyamplicol as pa
    from pyamplicol.config import ColorConfig, RunConfig

    path = os.path.join(
        work, "_".join(str(p) for p in pdgs) + ("_correlated" if correlated else "")
    )
    legs = [k + 1 for k, pdg in enumerate(pdgs) if colored(pdg)]
    if not os.path.isdir(path):
        process = " ".join(particle(p) for p in pdgs[:2])
        process += " > " + " ".join(particle(p) for p in pdgs[2:])
        correlators = None
        if correlated:
            correlators = pa.CorrelatorConfig(
                color_correlations=tuple(
                    pa.ColorCorrelator.dipole(f"T{a}_{b}", a, b)
                    for k, a in enumerate(legs)
                    for b in legs[k:]
                ),
                spin_correlations=tuple(
                    (k + 1,) for k, pdg in enumerate(pdgs) if pdg == 21
                ),
            )
        config = RunConfig(action="generate", color=ColorConfig(accuracy="full"))
        pa.Generator(config).generate(
            process,
            path,
            model=pa.ModelSource.built_in_sm(),
            correlators=correlators,
        )
    rt = pa.Runtime.load(path)
    if tuple(rt.external_permutation) != tuple(range(len(pdgs))):
        raise ValueError(f"pyAmpliCol reorders the particles of {pdgs}")
    rt.set_model_parameters(
        {
            "normalization.alpha_s_me_check": ALPHA_S,
            "particle.6.mass": TOP_MASS,
            "particle.6.width": 0.0,
        }
    )
    return rt


def evaluate(rt, momenta, requests):
    import pyamplicol as pa

    values = rt.evaluate_correlated_many(
        [np.asarray(momenta).tolist()],
        {name: pa.CorrelatedRequest(*r) for name, r in requests.items()},
        arithmetic="double-double",
    )
    return {name: complex(v[0]) for name, v in values.items()}


def born_blocks(rt, momenta, j, k, gluon):
    """The Born, <T_j . T_k> and, for a gluon j, its spin tensor, j and k
    1-based legs of the runtime."""
    a, b = sorted((j, k))
    requests = {"born": ("born", {}), "cc": (f"T{a}_{b}", {})}
    if gluon:
        for mu in range(4):
            for nu in range(mu, 4):
                e = [0.0] * 4
                e[mu] = e[nu] = 1.0
                requests[f"S{mu}{nu}"] = (f"T{a}_{b}", {j: tuple(e)})
                requests[f"B{mu}{nu}"] = ("born", {j: tuple(e)})
    value = {name: v.real for name, v in evaluate(rt, momenta, requests).items()}
    blocks = dict(born=value["born"], color_correlated=value["cc"])

    def tensor(key):
        out = [[0.0] * 4 for _ in range(4)]
        for mu in range(4):
            for nu in range(4):
                diagonal = value[f"{key}{mu}{mu}"] + value[f"{key}{nu}{nu}"]
                both = value[f"{key}{min(mu, nu)}{max(mu, nu)}"]
                out[mu][nu] = both if mu == nu else 0.5 * (both - diagonal)
        return out

    blocks["spin_correlated"] = tensor("S") if gluon else None
    blocks["born_spin_correlated"] = tensor("B") if gluon else None
    return blocks


def born_mapping(masses, leptonic):
    import madspace as ms

    return ms.PhaseSpaceMapping(
        list(masses),
        SQRT_S,
        leptonic=leptonic,
        mode=ms.PhaseSpaceMapping.TChannelMode.rambo,
    )


def dipole_mapping(real_masses, leptonic, dipole):
    """The CSMapping of a dipole, its Born the real process without i."""
    import madspace as ms

    born = born_mapping(np.delete(real_masses, dipole["i"]), leptonic)
    return ms.CSMapping(born, dipole["i"], dipole["j"], dipole["k"])


def forward(mapping, r):
    import torch

    momenta, x1, x2, det = mapping.map_forward([torch.as_tensor(np.atleast_2d(r))])
    n = len(np.atleast_2d(r))
    return (
        np.asarray(momenta),
        np.broadcast_to(np.asarray(x1), (n,)).copy(),
        np.broadcast_to(np.asarray(x2), (n,)).copy(),
    )


def projector(mappings, dipoles):
    """The Born of each dipole that CSDipoles projects a real emission
    (momenta, x1, x2) onto."""
    import torch
    from madspace_testing import default_context

    import madspace as ms

    generator = ms.CSDipoles(
        mappings, [d["kernel"] for d in dipoles], [d["symmetry"] for d in dipoles]
    )
    runtime = ms.FunctionRuntime(generator.function(), default_context())
    names = list(generator.return_types().keys())
    mirror = "mirror_index" in generator.arg_types().keys()

    def project(momenta, x1, x2):
        inputs = [torch.as_tensor(np.asarray(a)) for a in (momenta, x1, x2)]
        if mirror:
            inputs.append(torch.zeros(len(momenta), dtype=torch.int32))
        outputs = [torch.from_dlpack(t).cpu().numpy() for t in runtime.call(inputs)]
        return dict(zip(names, outputs))["born_momenta"]

    return project


def make(name, setup, work):
    from madspace.subtraction.cs import cs_subtraction

    masses = {6: TOP_MASS}
    borns = [dict(pdgs=b, flavors=[b]) for b in setup["borns"]]
    reals = [dict(pdgs=r, flavors=[r]) for r in setup["reals"]]
    block = cs_subtraction(reals, borns, masses)
    leptonic = setup["leptonic"]
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    born_runtimes = {}
    points = []
    for r, real in enumerate(block["reals"]):
        pdgs = real["pdgs"]
        real_masses = np.array([masses.get(abs(p), 0.0) for p in pdgs])
        real_rt = runtime(pdgs, work, False)
        dipoles = [(n, d) for n, d in enumerate(block["cs"]) if d["real"] == r]
        mappings = [dipole_mapping(real_masses, leptonic, d) for _, d in dipoles]
        project = projector(mappings, [d for _, d in dipoles])
        pairs = []
        for n, d in dipoles:
            if (d["i"], d["j"]) not in [(p["i"], p["j"]) for _, p in pairs]:
                pairs.append((n, d))
        if setup.get("pairs") == "first of each family and kernel":
            kinds = {}
            for n, d in pairs:
                kinds.setdefault((d["family"], d["kernel"]), (n, d))
            pairs = list(kinds.values())
        azimuths = setup.get("azimuths", AZIMUTHS)
        random_points = setup.get("random_points", RANDOM_POINTS)
        for n, d in pairs:
            mapping = dipole_mapping(real_masses, leptonic, d)
            r_born = rng.uniform(0.15, 0.85, mapping.random_dim() - 3)
            initial = d["j"] < 2
            rays = [("collinear", phi) for phi in azimuths]
            if pdgs[d["i"]] == 21:
                rays.append(("soft", azimuths[0]))
            rays += [("none", None)] * random_points
            for limit, phi in rays:
                for lam in LAMBDAS if limit != "none" else [None]:
                    if limit == "collinear":
                        emission = [0.5, lam, phi] if initial else [lam, 0.5, phi]
                    elif limit == "soft":
                        emission = [lam, 0.5, phi] if initial else [lam, lam, phi]
                    else:
                        emission = rng.uniform(0.05, 0.95, 3)
                    random = np.concatenate([r_born, emission])
                    momenta, x1, x2 = forward(mapping, random)
                    real_me = float(
                        real_rt.evaluate([momenta[0].tolist()], precision=32)[0].real
                    )
                    projected = project(momenta, x1, x2)[0]
                    entries = []
                    for m, (index, dipole) in enumerate(dipoles):
                        born = block["borns"][dipole["born"]]
                        key = tuple(born["pdgs"])
                        if key not in born_runtimes:
                            born_runtimes[key] = runtime(born["pdgs"], work, True)
                        order = dipole["born_order"]
                        i = dipole["i"]
                        bj = dipole["j"] - (dipole["j"] > i)
                        bk = dipole["k"] - (dipole["k"] > i)
                        legs = order.index(bj) + 1, order.index(bk) + 1
                        gluon = born["pdgs"][legs[0] - 1] == 21
                        blocks = born_blocks(
                            born_runtimes[key],
                            projected[m][order],
                            legs[0],
                            legs[1],
                            gluon,
                        )
                        entries.append(
                            dict(
                                index=index,
                                born_momenta=projected[m].tolist(),
                                **blocks,
                            )
                        )
                    points.append(
                        dict(
                            real=r,
                            ray=dict(
                                dipole=n, limit=limit, lam=lam, random=random.tolist()
                            ),
                            momenta=momenta[0].tolist(),
                            x1=float(x1[0]),
                            x2=float(x2[0]),
                            real_me=real_me,
                            dipoles=entries,
                        )
                    )
    import pyamplicol as pa

    data = dict(
        setup=dict(
            name=name,
            nincoming=2,
            sqrt_s=SQRT_S,
            leptonic=leptonic,
            masses={str(k): v for k, v in masses.items()},
            alpha_s=ALPHA_S,
            source=f"pyAmpliCol {pa.package_version()}, built-in Standard Model, "
            "top width 0",
        ),
        **block,
        points=points,
    )
    with open(os.path.join(DATA, f"{name}.json"), "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(name, len(points), "points")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", default=os.path.join(HERE, "work"))
    parser.add_argument("--setup", action="append", help="only these setups")
    args = parser.parse_args()
    sys.path.insert(0, os.path.join(HERE, ".."))
    os.makedirs(args.work, exist_ok=True)
    os.makedirs(DATA, exist_ok=True)
    for name, setup in SETUPS.items():
        if args.setup and name not in args.setup:
            continue
        make(name, setup, args.work)


if __name__ == "__main__":
    main()
