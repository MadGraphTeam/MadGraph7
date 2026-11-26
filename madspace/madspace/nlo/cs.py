"""The Catani-Seymour dipoles of the real processes of an NLO output.

The dipoles are combinatorics on the flavors and masses of a real process
(Catani, Seymour, arXiv:hep-ph/9605323; masses: Catani, Dittmaier, Seymour,
Trocsanyi, arXiv:hep-ph/0201036). A dipole of the QCD corrections has an
emitted parton i, an emitter j and a spectator k, all colored, with the
positions of the real process (0-based). i is massless and emitted in the final state. The family
is ff, fi, if or ii by whether the emitter and the spectator are in the
final (f) or the initial (i) state, and fd for a final-state emitter with
the decaying particle of a decay as the spectator (Basso, Dittmaier, Huss,
Oggero, arXiv:1507.04676, for gluons Bernreuther, Gonzalez, Mellein,
arXiv:1401.5930), whose soft terms its dipoles take, as the decaying
particle emits no collinear parton. The kernel names the splitting by the
Born parton of the emitter and the emitted parton, q for a quark or an
antiquark and g for a gluon:

    final emitter (ff, fi)         incoming emitter (if, ii)
    qg  q -> q g                   qg  q -> q g, the quark enters the Born
    gg  g -> g g                   gg  g -> g g
    gq  g -> q q~, massless        qq  g -> q~ q, the q~ enters the Born
                                   gq  q -> g q, the gluon enters the Born

The dipoles of the QED corrections (Dittmaier, arXiv:hep-ph/9904440) have
the same kernels with the charged particles in place of the colored ones: a
photon from a charged particle (qg, also from a massive W), a massless
charged fermion pair from a photon (gq), and for an incoming particle a
photon splitting into the antifermion of i (qq) and a charged fermion that
gives a photon to the Born (gq). Their Born has two powers of e less in
place of two of g_s, and their spectators k are the charged particles. A
photon that splits has no charge in the Born, its one term (Dittmaier,
Kabelschacht, Kasprzik, arXiv:0802.1405) goes to its spectators with equal
weights, which take the place of -<T_j . T_k> / T_j^2.

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
    cs[]: real, born (indices), group (QCD or QED), family, i, j, k, kernel,
        born_order (the Born leg n of the matrix element is leg
        born_order[n] of the Born of the dipole, the real process without
        i), symmetry (S_B / S_R, the factors n! of the identical final-state
        particles of the Born over those of the real, which the
        normalization of the matrix elements of MadGraph divides out) and
        weight (that of the spectator of a photon that splits, None for the
        correlated Born)

The alpha parameters of the families and kappa of CDST are parameters of
the subtraction, not of the dipoles: they stay in the run card.
"""

import json
import math
import os

FAMILIES = ("ff", "fi", "if", "ii", "fd")
FINAL_KERNELS = ("qg", "gg", "gq")
INITIAL_KERNELS = ("qg", "gg", "qq", "gq")
GROUPS = ("QCD", "QED")
# the electric charges of the charged particles of the SM in units of e, by
# PDG code, those of the antiparticles opposite
CHARGES = {
    1: -1 / 3,
    2: 2 / 3,
    3: -1 / 3,
    4: 2 / 3,
    5: -1 / 3,
    6: 2 / 3,
    11: -1.0,
    13: -1.0,
    15: -1.0,
    24: 1.0,
    37: 1.0,
}
# the colors of the charged fermions, for the splittings of a photon
COLORS = {1: 3, 2: 3, 3: 3, 4: 3, 5: 3, 6: 3, 11: 1, 13: 1, 15: 1}


def is_quark(pdg):
    return 1 <= abs(pdg) <= 6


def is_colored(pdg):
    return is_quark(pdg) or pdg == 21


def charge(pdg):
    """The electric charge of a particle in units of e."""
    value = CHARGES.get(abs(pdg), 0.0)
    return value if pdg > 0 else -value


def is_charged_fermion(pdg):
    return abs(pdg) in COLORS


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


def qed_final_splitting(pdgs, masses, a, b):
    """(i, j, the Born flavor of i + j, kernel) of the QED splitting of the
    final-state pair a < b, None without one: a photon i from a charged
    particle j (qg), or a massless charged fermion pair from a photon (gq,
    the second one is i)."""
    fa, fb = pdgs[a], pdgs[b]
    if fb == 22 and charge(fa) != 0:
        return b, a, fa, "qg"
    if fa == 22 and charge(fb) != 0:
        return a, b, fb, "qg"
    if is_charged_fermion(fa) and fb == -fa and masses[a] == 0.0:
        return b, a, 22, "gq"
    return None


def qed_initial_splitting(pdgs, masses, a, i):
    """(i, a, the flavor of the incoming particle of the Born, kernel) of
    the QED splitting of an incoming particle a emitting the final-state
    particle i, None without one: a photon from a charged particle (qg), a
    photon splitting into the antifermion of i (qq) and a charged fermion
    giving a photon to the Born (gq)."""
    fa, fi = pdgs[a], pdgs[i]
    if masses[i] != 0.0:
        return None
    if fi == 22 and charge(fa) != 0:
        return i, a, fa, "qg"
    if fa == 22 and is_charged_fermion(fi):
        return i, a, -fi, "qq"
    if is_charged_fermion(fa) and fi == fa:
        return i, a, 22, "gq"
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


def cs_dipoles(real_pdgs, borns, masses=None, nincoming=2, groups=("QCD",)):
    """The dipoles of a real process, for each group of corrections (QCD,
    QED) ff, fi, if, ii, in that order of the emitters, each with its
    spectators: dicts with born (an index of borns), group, family, i, j, k,
    kernel, born_order, symmetry and weight (see the module). borns are dicts
    with pdgs and flavors, the masses {pdg: mass} (zero by default). In a
    decay (nincoming 1) a colored decaying particle is the spectator of the
    final-state emitters (fd), and a charged one that of the emitters of
    photons.

    Raises:
        NotImplementedError: for a photon that splits with no charged
            particle as the spectator.
        ValueError: for a group other than QCD and QED.
    """
    unknown = set(groups) - set(GROUPS)
    if unknown:
        raise ValueError(f"cs_dipoles: groups {sorted(unknown)} not in {GROUPS}")
    pdgs = list(real_pdgs)
    m = mass_list(pdgs, masses)
    count = len(pdgs)
    dipoles = []
    for group in GROUPS:
        if group not in groups:
            continue
        final, initial = (
            (final_splitting, initial_splitting)
            if group == "QCD"
            else (qed_final_splitting, qed_initial_splitting)
        )
        splittings = [
            final(pdgs, m, a, b)
            for a in range(nincoming, count)
            for b in range(a + 1, count)
        ]
        if nincoming == 2:
            splittings += [
                initial(pdgs, m, a, i)
                for a in range(nincoming)
                for i in range(nincoming, count)
            ]
        for i, j, flavor, kernel in filter(None, splittings):
            dipoles += _splitting_dipoles(
                pdgs, borns, nincoming, group, i, j, flavor, kernel
            )
    return dipoles


def _spectators(pdgs, nincoming, group, i, j, kernel):
    """The spectators of a splitting with their weights, None for those of
    the correlated Born (the color-correlated Born for QCD, the charges for
    QED). A photon that splits (gq of QED) has no charge in the Born, so its
    one term (Dittmaier, Kabelschacht, Kasprzik, arXiv:0802.1405) goes to
    the charged particles other than a decaying one in equal parts."""
    if group == "QCD":
        return [(k, None) for k in range(len(pdgs)) if is_colored(pdgs[k])]
    charged = [k for k in range(len(pdgs)) if charge(pdgs[k]) != 0]
    if kernel != "gq":
        return [(k, None) for k in charged]
    candidates = [k for k in charged if k not in (i, j) and k >= (nincoming == 1)]
    if not candidates:
        raise NotImplementedError(
            f"cs_dipoles: no charged spectator for the photon of {pdgs}"
        )
    return [(k, 1.0 / len(candidates)) for k in candidates]


def _splitting_dipoles(pdgs, borns, nincoming, group, i, j, flavor, kernel):
    """The dipoles of the splitting of j into i and j, one per spectator."""
    line = list(pdgs)
    line[j] = flavor
    del line[i]
    b, order = _find_born(line, borns, nincoming)
    if b is None:
        # a Born without matrix element at the order of the output, as
        # g g > e+ e- for the dipole of g d~ > e+ e- d~ with i the d~
        return []
    factor = symmetry(line, nincoming) / symmetry(pdgs, nincoming)
    dipoles = []
    for k, weight in _spectators(pdgs, nincoming, group, i, j, kernel):
        if k in (i, j):
            continue
        if nincoming == 1 and k == 0:
            family = "fd"
        else:
            family = ("i" if j < nincoming else "f") + ("i" if k < nincoming else "f")
        dipoles.append(
            dict(
                born=b,
                group=group,
                family=family,
                i=i,
                j=j,
                k=k,
                kernel=kernel,
                born_order=order,
                symmetry=factor,
                weight=weight,
            )
        )
    return dipoles


def cs_subtraction(reals, borns, masses=None, nincoming=2, groups=("QCD",)):
    """The cs block for reals and borns (lists of dicts with pdgs and
    flavors): borns, reals and their dipoles of the groups of corrections
    (see the module). Every flavor line of a real has to give it the same
    dipoles in the same order as the first. Lines whose dipoles go to other
    Borns (a b of its own subprocess beside the d and s of another) make
    reals of their own, with the same matrix element."""
    out, dipoles = [], []

    def structure(entries):
        return [
            (d["group"], d["i"], d["j"], d["k"], d["kernel"], d["born_order"])
            for d in entries
        ]

    for real in reals:
        lines = real.get("flavors") or [real["pdgs"]]
        first = cs_dipoles(lines[0], borns, masses, nincoming, groups)
        by_borns = {}
        for flavors in lines:
            other = cs_dipoles(flavors, borns, masses, nincoming, groups)
            if structure(other) != structure(first):
                raise ValueError(
                    f"the flavors {flavors} of the real {real['pdgs']} have other "
                    "dipoles or orders"
                )
            key = tuple(d["born"] for d in other)
            by_borns.setdefault(key, (other, []))[1].append(flavors)
        for entries, group in by_borns.values():
            dipoles += [dict(real=len(out), **d) for d in entries]
            out.append(dict(real, pdgs=list(group[0]), flavors=group))
    return dict(borns=borns, reals=out, cs=dipoles)


def _final_key(pdgs, nincoming):
    return tuple(pdgs[:nincoming]), tuple(sorted(pdgs[nincoming:]))


def read_cs_subtraction(output, masses=None):
    """The cs block of an aMC@NLO output (output DIR of a process with
    [QCD]) from the fks_configurations.json of its P* directories: each
    subprocess a Born, each real matrix element of a configuration a real,
    each flavor line once over the subprocesses (the final state in any
    order), with the dipoles of the splitting types of the configurations
    (QCD, QED)."""
    root = os.path.join(output, "SubProcesses")
    names = sorted(d for d in os.listdir(root) if d.startswith("P"))
    borns, reals, seen, nincoming = [], [], set(), None
    groups = set()
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
            groups.update(configuration.get("splitting_type", ["QCD"]))
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
    groups = tuple(group for group in GROUPS if group in groups) or ("QCD",)
    return cs_subtraction(reals, borns, masses, nincoming, groups)
