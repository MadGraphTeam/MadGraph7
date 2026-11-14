"""The Catani-Seymour dipoles of the real processes of an NLO output.

The dipoles are combinatorics on the flavors and masses of a real process
(Catani, Seymour, arXiv:hep-ph/9605323; masses: Catani, Dittmaier, Seymour,
Trocsanyi, arXiv:hep-ph/0201036). A dipole has an emitted parton i, an
emitter j and a spectator k, all colored, with the positions of the real
process (0-based). i is massless and emitted in the final state. The family
is ff, fi, if or ii by whether the emitter and the spectator are in the
final (f) or the initial (i) state, and fd for a final-state emitter with
the decaying particle of a decay as the spectator (Basso, Dittmaier, Huss,
Oggero, arXiv:1507.04676), whose soft terms its dipoles take, as the decaying
particle emits no collinear parton. The kernel names the splitting by the
Born parton of the emitter and the emitted parton, q for a quark or an
antiquark and g for a gluon:

    final emitter (ff, fi)         incoming emitter (if, ii)
    qg  q -> q g                   qg  q -> q g, the quark enters the Born
    gg  g -> g g                   gg  g -> g g
    gq  g -> q q~, massless        qq  g -> q~ q, the q~ enters the Born
                                   gq  q -> g q, the gluon enters the Born

Each pair of final partons with a splitting has one dipole per spectator,
as has each incoming parton with each final parton i it can emit. The Born of
a dipole is the real process without i, with the Born parton of the emitter
at the position of j, one of the Borns of the output up to the order of the
final state. A Born that the output does not have has no matrix element at
its order (g g > e+ e- for g d~ > e+ e- d~), so its dipoles are left out.

The subtraction block of an output (cs_subtraction) has the lists of the
Borns and of the reals and the dipoles that refer to them by index, the cs
block of the NLO output that the MG7 exporter is to write:

    borns[]: pdgs (the order of the matrix element), flavors (all flavor
        lines of the matrix element), subprocess (its P* directory)
    reals[]: pdgs, flavors (those of one real matrix element, the first is
        pdgs), subprocess and configuration (0-based) of a configuration of
        the FKS exporter with this real matrix element
    cs[]: real, born (indices), family, i, j, k, kernel, born_order (the Born
        leg n of the matrix element is leg born_order[n] of the Born of the
        dipole, the real process without i), symmetry (S_B / S_R, the
        factors n! of the identical final-state particles of the Born over
        those of the real, which the normalization of the matrix elements of
        MadGraph divides out)

The alpha parameters of the families and kappa of CDST are parameters of
the subtraction, not of the dipoles: they stay in the run card.
"""

import json
import math
import os

FAMILIES = ("ff", "fi", "if", "ii", "fd")
FINAL_KERNELS = ("qg", "gg", "gq")
INITIAL_KERNELS = ("qg", "gg", "qq", "gq")


def is_quark(pdg):
    return 1 <= abs(pdg) <= 6


def is_colored(pdg):
    return is_quark(pdg) or pdg == 21


def final_splitting(pdgs, masses, a, b):
    """(i, j, the Born flavor of i + j, kernel) of the final-state pair a < b
    of a real process, None without a splitting: the gluon of q g is i, the
    second gluon of g g, the antiquark or the second quark of a massless q
    q~."""
    fa, fb = pdgs[a], pdgs[b]
    if fa == 21 and fb == 21:
        return b, a, 21, "gg"
    if fb == 21 and is_quark(fa):
        return b, a, fa, "qg"
    if fa == 21 and is_quark(fb):
        return a, b, fb, "qg"
    if is_quark(fa) and fb == -fa and masses[a] == 0.0:
        return b, a, 21, "gq"
    return None


def initial_splitting(pdgs, masses, a, i):
    """(i, a, the flavor of the incoming parton of the Born, kernel) of an
    incoming parton a emitting the final-state parton i, None without a
    splitting."""
    fa, fi = pdgs[a], pdgs[i]
    if not is_colored(fa) or masses[i] != 0.0:
        return None
    if fi == 21:
        return i, a, fa, "gg" if fa == 21 else "qg"
    if fa == 21 and is_quark(fi):
        return i, a, -fi, "qq"
    if is_quark(fa) and fi == fa:
        return i, a, 21, "gq"
    return None


def symmetry(pdgs, nincoming=2):
    """The product of n! over the flavors of the final state, n the number
    of particles of each."""
    final = list(pdgs[nincoming:])
    return math.prod(math.factorial(final.count(f)) for f in set(final))


def born_order(line, born, nincoming=2):
    """The order of the legs of a Born of the output in a Born line of a
    dipole, order[n] the position in line of leg n of born, None if born is
    not line up to the order of the final state."""
    line, born = list(line), list(born)
    if len(line) != len(born) or line[:nincoming] != born[:nincoming]:
        return None
    order = list(range(nincoming))
    for flavor in born[nincoming:]:
        n = next(
            (
                n
                for n in range(nincoming, len(line))
                if line[n] == flavor and n not in order
            ),
            None,
        )
        if n is None:
            return None
        order.append(n)
    return order


def _find_born(line, borns, nincoming):
    """(index, order) of the Born of the output that has the flavor line of a
    dipole among its flavors."""
    for b, born in enumerate(borns):
        for flavors in born.get("flavors", [born["pdgs"]]):
            order = born_order(line, flavors, nincoming)
            if order is not None:
                return b, order
    return None, None


def mass_list(pdgs, masses):
    """The masses of the particles of a process from {pdg: mass} (by |pdg|,
    zero for the others)."""
    masses = masses or {}
    return [float(masses.get(abs(pdg), masses.get(pdg, 0.0))) for pdg in pdgs]


def cs_dipoles(real_pdgs, borns, masses=None, nincoming=2):
    """The dipoles of a real process, ff, fi, if, ii, in that order of the
    emitters, each with its spectators: dicts with born (an index of borns),
    family, i, j, k, kernel, born_order and symmetry (see the module). borns
    are dicts with pdgs and flavors, the masses {pdg: mass} (zero by default).
    In a decay (nincoming 1) a colored decaying particle is the spectator of
    the final-state emitters (fd), which have to be quarks.

    Raises:
        NotImplementedError: for a gluon emitter with a colored decaying
            particle as the spectator.
    """
    pdgs = list(real_pdgs)
    m = mass_list(pdgs, masses)
    count = len(pdgs)
    splittings = [
        final_splitting(pdgs, m, a, b)
        for a in range(nincoming, count)
        for b in range(a + 1, count)
    ]
    if nincoming == 2:
        splittings += [
            initial_splitting(pdgs, m, a, i)
            for a in range(nincoming)
            for i in range(nincoming, count)
        ]
    dipoles = []
    for i, j, flavor, kernel in filter(None, splittings):
        line = list(pdgs)
        line[j] = flavor
        del line[i]
        b, order = _find_born(line, borns, nincoming)
        if b is None:
            # a Born without matrix element at the order of the output, as
            # g g > e+ e- for the dipole of g d~ > e+ e- d~ with i the d~
            continue
        factor = symmetry(line, nincoming) / symmetry(pdgs, nincoming)
        for k in range(count):
            if k in (i, j) or not is_colored(pdgs[k]):
                continue
            if nincoming == 1 and k == 0:
                if kernel != "qg":
                    raise NotImplementedError(
                        "dipoles of a final-state gluon with a decaying particle "
                        "as the spectator"
                    )
                family = "fd"
            else:
                family = ("i" if j < nincoming else "f") + (
                    "i" if k < nincoming else "f"
                )
            dipoles.append(
                dict(
                    born=b,
                    family=family,
                    i=i,
                    j=j,
                    k=k,
                    kernel=kernel,
                    born_order=order,
                    symmetry=factor,
                )
            )
    return dipoles


def cs_subtraction(reals, borns, masses=None, nincoming=2):
    """The cs block for reals and borns (lists of dicts with pdgs and
    flavors): borns, reals and their dipoles (see the module). Every flavor
    line of a real has to give its dipoles the same Borns in the same order
    as the first."""
    dipoles = []
    for r, real in enumerate(reals):
        first = cs_dipoles(real["pdgs"], borns, masses, nincoming)
        for flavors in real.get("flavors", [])[1:]:
            other = cs_dipoles(flavors, borns, masses, nincoming)
            if [(d["born"], d["born_order"]) for d in other] != [
                (d["born"], d["born_order"]) for d in first
            ]:
                raise ValueError(
                    f"the flavors {flavors} of the real {real['pdgs']} have other "
                    "Borns or orders"
                )
        dipoles += [dict(real=r, **d) for d in first]
    return dict(borns=borns, reals=reals, cs=dipoles)


def _final_key(pdgs, nincoming):
    return tuple(pdgs[:nincoming]), tuple(sorted(pdgs[nincoming:]))


def read_cs_subtraction(output, masses=None):
    """The cs block of an aMC@NLO output (output DIR of a process with
    [QCD]) from the fks_configurations.json of its P* directories: each
    subprocess a Born, each real matrix element of a configuration a real,
    each flavor line once over the subprocesses (the final state in any
    order)."""
    root = os.path.join(output, "SubProcesses")
    names = sorted(d for d in os.listdir(root) if d.startswith("P"))
    borns, reals, seen, nincoming = [], [], set(), None
    for name in names:
        with open(os.path.join(root, name, "fks_configurations.json")) as f:
            data = json.load(f)
        if nincoming not in (None, data["nincoming"]):
            raise ValueError("the subprocesses have different incoming particles")
        nincoming = data["nincoming"]
        borns.append(
            dict(
                pdgs=data["born_pdgs"],
                flavors=data.get("born_flavors", [data["born_pdgs"]]),
                subprocess=name,
            )
        )
        for c, configuration in enumerate(data["configurations"]):
            me = configuration["real_me"]
            flavors = data.get("real_flavors", [])
            flavors = flavors[me - 1] if me <= len(flavors) else []
            flavors = flavors or [configuration["real_pdgs"]]
            # the flavor lines that no other real has yet
            flavors = [
                line for line in flavors if _final_key(line, nincoming) not in seen
            ]
            if not flavors:
                continue
            seen.update(_final_key(line, nincoming) for line in flavors)
            reals.append(
                dict(
                    pdgs=flavors[0],
                    flavors=flavors,
                    subprocess=name,
                    configuration=c,
                )
            )
    return cs_subtraction(reals, borns, masses, nincoming)
