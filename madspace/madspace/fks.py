"""The FKS configurations of a Born, from the metadata of the FKS exporter.

MadGraph's FKS exporter writes fks_configurations.json into each subprocess
directory, next to fks_info.inc (see get_fks_configurations in
madgraph/iolibs/export_fks.py):

    nincoming: number of incoming particles
    born_pdgs: PDG codes of the Born, in the order of its momenta
    configurations[]: in the order of fks_info.inc, each with
        i_fks, j_fks: positions of i and of the emitter j in the real
            process, 1-based
        ij_pdg: PDG code of the parton i + j of the Born
        real_pdgs: PDG codes of the real process
        splitting_type: the types of the splitting, QCD or QED, a second
            one with an extra counterterm
        extra_cnt_index: the extra counterterm of the second type (a
            position in extra_counterterms), -1 for none
        real_tags: whether each particle of the real process is tagged (an
            identified photon, !a!), absent from older exporters
        real_me, need_color_links, need_charge_links
        real_colors, real_charges: color representation (1, 3, -3, 8) and
            electric charge of each particle of the real process
        fks_j_from_i: for each particle i of the real process, the particles
            j it forms an FKS pair with (1-based), the pairs whose sector
            functions partition the phase space of the real process
    extra_counterterms[]: the Borns of the extra counterterms
        (born_cnt_<n>.f), for a quark pair that comes from a gluon or from a
        photon, absent from older exporters
        born_pdgs: PDG codes of the Born of the other mother
    orders: the coupling orders of orders.inc and amp_split_orders.inc
        split_orders: the names of the split orders (QCD, QED, ...)
        born, nlo: the powers of each split order of the Born orders and of
            the NLO orders, the positions of the amp_split arrays
        max_born, max_nlo: the largest powers of each split order at the
            Born and at NLO (born_orders and nlo_orders)
        extra_born: for each extra counterterm, the powers of each split
            order of the orders of its Born
    ewsudakov: with --ewsudakov, the matrix elements of the Sudakov
        approximation of the EW corrections (see get_ewsudakov_info)
        scheme: the scheme of the parameter renormalisation, gmu or alpha_mz
        has_lo1, has_lo2, lo1, lo2: whether the Born has a LO1 and a LO2,
            and the powers of each split order of them (null for none)
        helicities: the helicities of the Born
        goldstone[]: the amplitudes with Goldstone bosons, each with legs
            (the replaced particles, 1-based) and pdgs (its process)
        interference[]: the interferences, each with type (cew, iz1, iz2 or
            ipm2), base (0 the Born, k the Goldstone amplitude k), legs,
            old_pdgs, new_pdgs and pdgs

The functions here turn it into the FKSMappings of a Born mapping, one per
configuration, or into one FKSConfigurationMapping for all of them, with the
0-based positions of madspace. The Born mapping gives the masses, so the
emitter is massive where it gives it a mass. With the cuts on the Born-level
objects, the configurations get the lower bound on the partonic energy that
the cuts imply (see fks_sqrt_s_hat_min): a floor of the Born mapping for a
final-state emitter, s_hat_min for an incoming one. fks_resonances gives the
resonances of the resonance-local recoil of a Born topology, with
fks_minimal_masses for the masses its decay products need under the cuts of
the run_card.

fks_configurations gives the FKSConfigurations that the classes of the FKS
subtraction take and fks_coupling_orders their FKSCouplingOrders, by which
the Born blocks come per Born order (followed by those of the Borns of the
extra counterterms per their orders) and the real emission, its counterterms
and the n-body part per NLO order. With them, fks_sector_function builds the sector functions
S_ij of the configurations, fks_eikonals the eikonal factors of their soft
limits, fks_splitting_kernels the factors of their collinear limits,
fks_prefactors the prefactors of the real emission and its counterterms,
fks_limits the soft and collinear limits of the real matrix element from the
Born and its correlated versions, fks_subtraction the real emission with
its counterterms and fks_soft_virtual the n-body part of the subtraction
without the virtual. ew_sudakov builds the EWSudakov of the Sudakov
approximation of the EW corrections from the ewsudakov entry.
"""

import json
import math
import os

from ._madspace_py import (
    CutItem,
    Cuts,
    EWSudakov,
    EWSudakovAmplitude,
    EWSudakovMode,
    EWSudakovProcess,
    FKSConfiguration,
    FKSConfigurationMapping,
    FKSCouplingOrders,
    FKSEikonals,
    FKSLimits,
    FKSMapping,
    FKSPrefactors,
    FKSSectorFunction,
    FKSSoftVirtual,
    FKSSplittingKernels,
    FKSSubtraction,
    Observable,
    fks_sqrt_s_hat_min,
)

FILE_NAME = "fks_configurations.json"


def read_fks_configurations(path):
    """The FKS configurations of a Born, from fks_configurations.json or the
    subprocess directory that contains it."""
    if os.path.isdir(path):
        path = os.path.join(path, FILE_NAME)
    with open(path) as f:
        return json.load(f)


def born_of_real(configuration):
    """The PDG codes of the Born of a configuration: the real process without
    i, with i + j at the position of j."""
    real = list(configuration["real_pdgs"])
    i_fks, j_fks = configuration["i_fks"] - 1, configuration["j_fks"] - 1
    if not (0 <= i_fks < len(real) and 0 <= j_fks < len(real)) or i_fks == j_fks:
        raise ValueError(
            f"FKS configuration with invalid i_fks, j_fks: {configuration}"
        )
    born = real[:i_fks] + real[i_fks + 1 :]
    born[j_fks if j_fks < i_fks else j_fks - 1] = configuration["ij_pdg"]
    return born


def sqrt_s_floor(pdgs, sqrt_s_min):
    """Cuts that only bound the partonic center-of-mass energy from below."""
    return Cuts(
        [
            CutItem(
                observable=Observable(pdgs, Observable.obs_sqrt_s, []), min=sqrt_s_min
            )
        ]
    )


def checked_configurations(born_pdgs, particle_count, configurations):
    """The configurations of fks_configurations.json, checked against the
    Born: a Born mapping of its particles, and real processes that are the
    Born with i and j in the place of i + j, the order FKSMapping assumes."""
    if configurations["nincoming"] != 2:
        raise ValueError("FKS mappings need two incoming particles")
    if particle_count != len(born_pdgs):
        raise ValueError(
            f"the Born mapping has {particle_count} particles, "
            f"the Born {len(born_pdgs)}"
        )
    for configuration in configurations["configurations"]:
        if born_of_real(configuration) != born_pdgs:
            raise ValueError(
                f"the real process {configuration['real_pdgs']} is not the Born "
                f"{born_pdgs} with i_fks = {configuration['i_fks']} and "
                f"j_fks = {configuration['j_fks']}"
            )
    return configurations["configurations"]


def fks_mappings(
    born,
    configurations,
    s_hat_min=0.0,
    cuts=None,
    initial_map="symmetric",
    resonances=None,
):
    """The FKSMappings of the FKS configurations of a Born, in their order.

    Args:
        born: the PhaseSpaceMapping of the Born process, its momenta in the
            order of born_pdgs. With cuts, a function that returns the Born
            mapping for given cuts (None for none), so that the configurations
            with a final-state emitter get a floor on sqrt(s_hat).
        configurations: the content of fks_configurations.json (see
            read_fks_configurations).
        s_hat_min: lower bound on the partonic energy squared of the real
            emission, see FKSMapping.
        cuts: the cuts on the Born-level objects (indexed like the Born). The
            bound they imply (fks_sqrt_s_hat_min) is a sqrt(s_hat) floor of
            the Born mapping for a final-state emitter, and raises s_hat_min
            for an incoming one. They stay to be applied to the real emission
            and the counter-events.
        initial_map: the map of an emission off an incoming parton,
            "symmetric" (genps_fks.f) or "asymmetric" (FKSISRMapping 2), see
            FKSRadiation.InitialMap.
        resonances: per configuration the resonance of the resonance-local
            recoil, as fks_resonances gives them for a Born mapping with one
            permutation, None for none.

    Raises:
        ValueError: if the Born mapping does not have the particles of the
            Born, or a real process is not its Born with i and j in the place
            of i + j, the order FKSMapping assumes, or resonances has more
            than one permutation.
    """
    born_pdgs = list(configurations["born_pdgs"])
    if cuts is None:
        born_plain = born_final = born
    elif callable(born):
        born_plain = born(None)
        bound = fks_sqrt_s_hat_min(cuts, born_plain.external_masses()[2:])
        born_final = born(sqrt_s_floor(born_pdgs, bound))
        s_hat_min_initial = max(s_hat_min, bound * bound)
    else:
        raise ValueError(
            "with cuts, born has to be a function that returns the Born mapping "
            "for given cuts"
        )
    entries = checked_configurations(
        born_pdgs, born_plain.particle_count(), configurations
    )
    if resonances is None:
        resonances = [[[]] for _ in entries]
    if len(resonances) != len(entries) or any(len(r) != 1 for r in resonances):
        raise ValueError(
            "resonances needs one entry per configuration, each for one "
            "permutation (FKSConfigurationMapping takes several)"
        )
    mappings = []
    for configuration, resonance in zip(entries, resonances):
        i_fks, j_fks = configuration["i_fks"] - 1, configuration["j_fks"] - 1
        resonance = list(resonance[0])
        if cuts is None:
            mapping = FKSMapping(born, i_fks, j_fks, s_hat_min, initial_map, resonance)
        elif j_fks >= 2:
            mapping = FKSMapping(
                born_final, i_fks, j_fks, s_hat_min, initial_map, resonance
            )
        else:
            mapping = FKSMapping(
                born_plain, i_fks, j_fks, s_hat_min_initial, initial_map, resonance
            )
        mappings.append(mapping)
    return mappings


def fks_configuration_mapping(
    born,
    configurations,
    s_hat_min=0.0,
    cuts=None,
    luminosity=None,
    initial_map="symmetric",
    resonances=None,
):
    """The FKSConfigurationMapping of the FKS configurations of a Born, with
    fks_index the position of the configuration (nFKSprocess - 1): all
    configurations in one pass, each event in that of its fks_index, as the
    FKSMapping of fks_mappings for it.

    Args:
        born: the PhaseSpaceMapping of the Born process, without cuts, its
            momenta in the order of born_pdgs. With cuts that bound sqrt(s_hat)
            from below, it needs sqrt_s_hat_min_condition=True: the floor of
            the configurations with a final-state emitter is a condition of
            the Born mapping, read per event.
        configurations: the content of fks_configurations.json (see
            read_fks_configurations).
        s_hat_min: lower bound on the partonic energy squared of the real
            emission, see FKSMapping.
        cuts: the cuts on the Born-level objects (indexed like the Born), whose
            bound (fks_sqrt_s_hat_min) is the floor of the Born for a
            final-state emitter and raises s_hat_min for an incoming one. They
            stay to be applied to the real emission and the counter-events.
        luminosity: for lepton beams, the LeptonLuminosity of the momentum
            fractions, with s_hat_min_condition=True, and born leptonic with
            sqrt_s_hat_condition=True. An incoming emitter then radiates
            without the projection of the event, see
            FKSConfigurationMapping.
        initial_map: the map of an emission off an incoming parton, see
            fks_mappings, also for lepton beams without the projection of the
            event.
        resonances: per configuration and per permutation of the Born
            mapping the resonance of the resonance-local recoil, as
            fks_resonances gives them, None for none.

    Raises:
        ValueError: as fks_mappings, and if a floor needs the
            sqrt_s_hat_min condition the Born mapping does not have.
    """
    born_pdgs = list(configurations["born_pdgs"])
    pairs = [
        (configuration["i_fks"] - 1, configuration["j_fks"] - 1)
        for configuration in checked_configurations(
            born_pdgs, born.particle_count(), configurations
        )
    ]
    floor = (
        0.0 if cuts is None else fks_sqrt_s_hat_min(cuts, born.external_masses()[2:])
    )
    return FKSConfigurationMapping(
        born,
        pairs,
        s_hat_min=s_hat_min,
        sqrt_s_hat_floor=floor,
        luminosity=luminosity,
        initial_map=initial_map,
        resonances=[] if resonances is None else resonances,
    )


def real_masses(born_masses, configuration):
    """The masses of the real process of a configuration: those of the Born,
    with j in the place of i + j (whose mass it has) and a massless i."""
    i_fks = configuration["i_fks"] - 1
    masses = list(born_masses)
    return masses[:i_fks] + [0.0] + masses[i_fks:]


def _is_jet(pdg, maxjetflavor, gamma_is_j):
    return abs(pdg) <= maxjetflavor or abs(pdg) == 21 or (pdg == 22 and gamma_is_j)


def fks_minimal_masses(
    configurations,
    born_masses,
    ptj=0.0,
    ptgmin=0.0,
    ptl=0.0,
    mll=0.0,
    mll_sf=0.0,
    maxjetflavor=4,
    gamma_is_j=False,
    pdg_cuts=(),
):
    """The lowest invariant mass of each final-state particle of the Born that
    the masses and the cuts of the run_card allow, per FKS configuration, as
    set_tau_min of MadFKS gives it to set_granny (mass_min): the mass of a
    particle plus ptj for a jet, ptgmin for an isolated photon and ptl for a
    charged lepton, a lepton with an opposite one at least half of mll (and
    of mll_sf for the same flavor), a particle with a cut by its PDG code
    (pt_min_pdg, mxx_min_pdg) the bound from them.

    set_tau_min classifies the particles of the real process of the
    configuration (IS_A_J, ...) and reads the values at their positions in
    the real process as those of the Born, which agree where i is last. It
    computes them in a loop over all configurations, each time without its
    i_fks, and keeps those of the last one, so the position of the i_fks of
    the last configuration has the value of an earlier configuration with
    another i_fks, else 0. The values here follow it.

    Args:
        configurations: the content of fks_configurations.json (see
            read_fks_configurations).
        born_masses: the masses of the Born, incoming first, in the order of
            born_pdgs.
        ptj, ptgmin, ptl, mll, mll_sf, maxjetflavor, gamma_is_j: the values of
            the run_card.
        pdg_cuts: the cuts by PDG code of the run_card, each a dict with pdg,
            ptmin (pt_min_pdg), mxxmin (mxx_min_pdg) and part_antipart
            (mxx_only_part_antipart).

    Returns:
        Per configuration the lowest invariant masses of the final-state
        particles of the Born, in their order.
    """
    nincoming = configurations["nincoming"]
    entries = configurations["configurations"]
    last = entries[-1]["i_fks"]
    other_i_fks = any(entry["i_fks"] != last for entry in entries)
    result = []
    for entry in entries:
        pdgs = list(entry["real_pdgs"])
        masses = real_masses(born_masses, entry)
        n = len(pdgs)
        # the cuts by PDG code of setcuts, at 1-based positions of the real
        # process except the last
        etmin = {}
        mxxmin = {}
        for cut in pdg_cuts:
            for i in range(nincoming + 1, n):
                if abs(pdgs[i - 1]) != cut["pdg"]:
                    continue
                etmin[i] = cut.get("ptmin", 0.0)
                if cut.get("mxxmin", 0.0) == 0.0:
                    continue
                for k in range(i + 1, n):
                    if (
                        pdgs[k - 1] == -pdgs[i - 1]
                        if cut.get("part_antipart", False)
                        else abs(pdgs[k - 1]) == cut["pdg"]
                    ):
                        mxxmin[i, k] = cut["mxxmin"]

        def lepton(i, sign):
            return pdgs[i - 1] in (sign * 11, sign * 13, sign * 15)

        def xm(i):
            pdg, mass = pdgs[i - 1], masses[i - 1]
            if _is_jet(pdg, maxjetflavor, gamma_is_j):
                return mass + ptj
            if pdg == 22:
                return mass + ptgmin
            for sign in (-1, 1):
                if not lepton(i, sign):
                    continue
                value = mass + ptl
                for j in range(nincoming + 1, n + 1):
                    if not lepton(j, -sign):
                        continue
                    if pdgs[i - 1] == -pdgs[j - 1] and (mll_sf != 0.0 or mll != 0.0):
                        return value - ptl - mass + max(mll / 2, mll_sf / 2, ptl + mass)
                    if mll != 0.0:
                        return value - ptl - mass + max(mll / 2, ptl + mass)
                return value
            if i == n:
                return mass
            count, mxx = 1, 0.0
            for j in range(nincoming + 1, n):
                a, b = mxxmin.get((i, j), 0.0), mxxmin.get((j, i), 0.0)
                if a != 0.0 or b != 0.0:
                    count += 1
                    if mxx == 0.0:
                        mxx = max(a, b)
            smin = count * ((2 - count) * mass**2 + (count - 1) / 2 * mxx**2)
            pt = etmin.get(i, 0.0)
            if smin < count**2 * (pt**2 + mass**2):
                return math.sqrt(pt**2 + mass**2)
            return math.sqrt(smin) / count

        result.append(
            [
                0.0 if b == last and not other_i_fks else xm(b)
                for b in range(nincoming + 1, n)
            ]
        )
    return result


def fks_resonances(topology, configurations, minimal_masses, permutations=None):
    """The resonances of the resonance-local recoil (see FKSRadiation) of the
    FKS configurations of a Born, per configuration and per permutation of the
    Born mapping, as the MCcntRefactor_Sfun branch of MadGraph7 installs them
    by default (select_fks_recoil): with a final-state emitter j, the
    grandmother of i and j, the propagator whose decay products contain the
    Born emitter, where it is a resonance, as set_granny of MadFKS decides: a
    decay of the topology (neither the propagator of s_hat nor one of the
    t-channel, the root of the topology), with a mass and a positive width,
    that the lowest invariant masses of its decay products allow to be on
    shell. Its members are its decay products in the real process, i and j
    included.

    Args:
        topology: the Topology of the Born mapping.
        configurations: the content of fks_configurations.json (see
            read_fks_configurations).
        minimal_masses: per configuration the lowest invariant masses of the
            final-state particles of the Born (see fks_minimal_masses).
        permutations: the permutations of the Born mapping, as
            PhaseSpaceMapping takes them (particle k of the Born is particle
            permutation[k] of the topology, incoming ones included), None for
            none.

    Returns:
        Per configuration and permutation (one without permutations) the
        positions of the members in the real process, 0-based in ascending
        order, empty without a resonance: the resonances argument of
        FKSConfigurationMapping.
    """
    nincoming = configurations["nincoming"]
    entries = configurations["configurations"]
    decays = topology.decays
    outgoing = list(topology.outgoing_indices)
    if permutations is None or len(permutations) == 0:
        permutations = [list(range(nincoming + len(outgoing)))]
    # the outgoing particles of the topology, by node
    leaf = {node: k for k, node in enumerate(outgoing)}

    def products(node):
        children = list(decays[node].child_indices)
        if not children:
            return [leaf[node]]
        return [k for child in children for k in products(child)]

    result = []
    for entry, masses in zip(entries, minimal_masses, strict=True):
        i_fks, j_fks = entry["i_fks"] - 1, entry["j_fks"] - 1
        per_permutation = []
        for permutation in permutations:
            members = []
            n_out = len(permutation) - nincoming
            if j_fks >= nincoming and n_out + 1 >= 3:
                # the Born emitter, its node in the topology and the grandmother
                jb = j_fks if j_fks < i_fks else j_fks - 1
                node = outgoing[permutation[jb] - nincoming]
                granny = decays[node].parent_index
                if granny != 0:
                    decay = decays[granny]
                    position = {t: k for k, t in enumerate(permutation)}
                    born = sorted(position[nincoming + t] for t in products(granny))
                    mass_min = sum(masses[b - nincoming] for b in born)
                    if (
                        decay.mass != 0.0
                        and decay.width > 0.0
                        and decay.mass > mass_min
                    ):
                        members = sorted(
                            [b if b < i_fks else b + 1 for b in born] + [i_fks]
                        )
            per_permutation.append(members)
        result.append(per_permutation)
    return result


def fks_configurations(born, configurations):
    """The FKSConfigurations of the FKS configurations of a Born, in their
    order (that of fks_index), with the masses of the Born mapping.

    Args:
        born: the PhaseSpaceMapping of the Born process, for its masses, its
            momenta in the order of born_pdgs.
        configurations: the content of fks_configurations.json (see
            read_fks_configurations).

    Raises:
        ValueError: as fks_mappings, for a configuration of a real process
            without FKS pairs (a finite one), and for a splitting of i and j
            that does not give the Born, or with the second splitting type
            the Born of its extra counterterm.
    """
    born_pdgs = list(configurations["born_pdgs"])
    born_masses = list(born.external_masses())
    result = []
    for configuration in checked_configurations(
        born_pdgs, born.particle_count(), configurations
    ):
        pairs = configuration["fks_j_from_i"]
        if not any(pairs):
            raise ValueError(
                f"the real process {configuration['real_pdgs']} has no FKS pairs"
            )
        fks_configuration = FKSConfiguration(
            i_fks=configuration["i_fks"] - 1,
            j_fks=configuration["j_fks"] - 1,
            pdgs=list(configuration["real_pdgs"]),
            colors=list(configuration["real_colors"]),
            charges=list(configuration["real_charges"]),
            masses=real_masses(born_masses, configuration),
            fks_j_from_i=[[j - 1 for j in js] for js in pairs],
            splitting_types=list(configuration["splitting_type"]),
            tags=[bool(tag) for tag in configuration.get("real_tags", [])],
            extra_counterterm=configuration.get("extra_cnt_index", -1),
        )
        # the Borns that FKSConfiguration infers from the splitting, which set
        # the identical-particle ratios of the counterterms
        expected = [born_pdgs]
        if fks_configuration.extra_counterterm >= 0:
            extra = configurations["extra_counterterms"][
                fks_configuration.extra_counterterm
            ]
            expected.append(list(extra["born_pdgs"]))
        for splitting, pdgs in enumerate(expected):
            if fks_configuration.born_pdgs(splitting) != pdgs:
                raise ValueError(
                    f"the splitting of i_fks = {configuration['i_fks']} and "
                    f"j_fks = {configuration['j_fks']} in "
                    f"{configuration['real_pdgs']} does not give the Born {pdgs}"
                )
        result.append(fks_configuration)
    return result


def fks_coupling_orders(configurations):
    """The FKSCouplingOrders of the FKS configurations of a Born, from the
    orders entry of fks_configurations.json: the split orders and the powers
    of each of the Born orders and of the NLO orders, in the order of the
    amp_split arrays of MadFKS.

    Args:
        configurations: the content of fks_configurations.json (see
            read_fks_configurations).

    Raises:
        ValueError: without the orders entry (an older FKS exporter) or for
            inconsistent orders.
    """
    if "orders" not in configurations:
        raise ValueError(
            "fks_configurations.json has no coupling orders, it comes from an "
            "older FKS exporter"
        )
    orders = configurations["orders"]
    return FKSCouplingOrders(
        split_orders=list(orders["split_orders"]),
        born=[list(o) for o in orders["born"]],
        nlo=[list(o) for o in orders["nlo"]],
        max_nlo=list(orders["max_nlo"]),
        extra_born=[[list(o) for o in extra] for extra in orders.get("extra_born", [])],
    )


def fks_sector_function(born, configurations, **options):
    """The FKSSectorFunction of the FKS configurations of a Born, with
    fks_index the position of the configuration (nFKSprocess - 1), as
    FKSConfigurationMapping.

    Args:
        born: the PhaseSpaceMapping of the Born process, for its masses, its
            momenta in the order of born_pdgs.
        configurations: the content of fks_configurations.json (see
            read_fks_configurations).
        options: the parameters of fks_powers.inc (fks_a, fks_b, use_energy,
            use_beta, h_damp_one, h_damp_power), see FKSSectorFunction.

    Raises:
        ValueError: as fks_configurations.
    """
    return FKSSectorFunction(fks_configurations(born, configurations), **options)


def fks_eikonals(born, configurations, resonance=False):
    """The FKSEikonals of the FKS configurations of a Born, with fks_index as
    for fks_sector_function, with the momentum of the resonance of the
    resonance-local recoil for resonance.

    Raises:
        ValueError: as fks_configurations.
    """
    return FKSEikonals(fks_configurations(born, configurations), resonance)


def fks_splitting_kernels(born, configurations, resonance=False):
    """The FKSSplittingKernels of the FKS configurations of a Born, with
    fks_index as for fks_sector_function, with the momentum of the resonance
    of the resonance-local recoil for resonance.

    Raises:
        ValueError: as fks_configurations.
    """
    return FKSSplittingKernels(fks_configurations(born, configurations), resonance)


def fks_prefactors(born, configurations, **parameters):
    """The FKSPrefactors of the FKS configurations of a Born, with fks_index
    as for fks_sector_function.

    Args:
        born: the PhaseSpaceMapping of the Born process, for its masses, its
            momenta in the order of born_pdgs.
        configurations: the content of fks_configurations.json (see
            read_fks_configurations).
        parameters: the parameters of the subtraction of fks_powers.inc
            (xicut, xi_s_cut, delta_o, delta_i, delta_s), and resonance for
            the cuts of the resonance-local recoil, see FKSPrefactors.

    Raises:
        ValueError: as fks_configurations.
    """
    return FKSPrefactors(fks_configurations(born, configurations), **parameters)


def fks_limits(born, configurations, collinear_born=False, resonance=False):
    """The FKSLimits of the FKS configurations of a Born, with fks_index as
    for fks_sector_function and the coupling orders of fks_coupling_orders,
    with the Born of the collinear counter-event apart for lepton beams
    (collinear_born) and the momentum of the resonance of the resonance-local
    recoil (resonance).

    Raises:
        ValueError: as fks_configurations and fks_coupling_orders, and for a
            splitting that is not one of QCD or QED or an extra counterterm.
    """
    return FKSLimits(
        fks_configurations(born, configurations),
        fks_coupling_orders(configurations),
        collinear_born,
        resonance,
    )


def fks_subtraction(born, configurations, **parameters):
    """The FKSSubtraction of the FKS configurations of a Born, with fks_index
    as for fks_sector_function.

    Args:
        born: the PhaseSpaceMapping of the Born process, for its masses, its
            momenta in the order of born_pdgs.
        configurations: the content of fks_configurations.json (see
            read_fks_configurations).
        parameters: the parameters of fks_powers.inc, those of the subtraction
            (xicut, xi_s_cut, delta_o, delta_i, delta_s) and of the sector
            functions (fks_a, fks_b, use_energy, use_beta, h_damp_one,
            h_damp_power), tiny of sreal, the PDF scheme (pdf_scheme), the
            number of light flavors (n_f), for the UPC scheme xi_A R_A of the
            nuclei of the beams (upc_radii), for lepton beams the Born of
            the collinear counter-event apart (collinear_born), and the
            momentum of the resonance of the resonance-local recoil
            (resonance), see FKSSubtraction.

    Raises:
        ValueError: as fks_limits.
    """
    return FKSSubtraction(
        fks_configurations(born, configurations),
        fks_coupling_orders(configurations),
        **parameters,
    )


def fks_soft_virtual(born, configurations, **parameters):
    """The FKSSoftVirtual of the FKS configurations of a Born, the n-body part
    of the subtraction without the virtual, on the momenta of the Born.

    Args:
        born: the PhaseSpaceMapping of the Born process, for its masses, its
            momenta in the order of born_pdgs.
        configurations: the content of fks_configurations.json (see
            read_fks_configurations).
        parameters: xicut and delta_o of fks_powers.inc and the numbers of
            light flavors (n_f) and of massless leptons (n_l), see
            FKSSoftVirtual.

    Raises:
        ValueError: as fks_limits, and without a configuration with a gluon or
            photon i.
    """
    return FKSSoftVirtual(
        fks_configurations(born, configurations),
        fks_coupling_orders(configurations),
        **parameters,
    )


def ew_sudakov_process(configurations):
    """The EWSudakovProcess of a Born, from the ewsudakov entry of
    fks_configurations.json, with the 0-based positions of madspace.

    Args:
        configurations: the content of fks_configurations.json (see
            read_fks_configurations).

    Raises:
        ValueError: without the ewsudakov entry (a process without
            --ewsudakov or an older FKS exporter).
    """
    if "ewsudakov" not in configurations:
        raise ValueError(
            "fks_configurations.json has no ewsudakov entry, the process has no "
            "--ewsudakov or comes from an older FKS exporter"
        )
    info = configurations["ewsudakov"]
    born = list(configurations["born_pdgs"])
    goldstone = []
    for amp in info["goldstone"]:
        legs = [leg - 1 for leg in amp["legs"]]
        goldstone.append(
            EWSudakovAmplitude(
                type="goldstone",
                legs=legs,
                old_pdgs=[born[leg] for leg in legs],
                new_pdgs=[amp["pdgs"][leg] for leg in legs],
            )
        )
    interference = [
        EWSudakovAmplitude(
            type=amp["type"],
            base=amp["base"],
            legs=[leg - 1 for leg in amp["legs"]],
            old_pdgs=list(amp["old_pdgs"]),
            new_pdgs=list(amp["new_pdgs"]),
        )
        for amp in info["interference"]
    ]
    split_orders = configurations.get("orders", {}).get("split_orders", ["QCD", "QED"])
    lo2_qcd_power = info["lo2"][split_orders.index("QCD")] if info["has_lo2"] else 0
    return EWSudakovProcess(
        pdgs=born,
        helicities=[list(h) for h in info["helicities"]],
        goldstone=goldstone,
        interference=interference,
        has_lo1=info["has_lo1"],
        has_lo2=info["has_lo2"],
        lo2_qcd_power=lo2_qcd_power,
        incoming=configurations["nincoming"],
    )


def ew_sudakov(configurations, parameters, modes=None):
    """The EWSudakov of a Born, the Sudakov approximation of its EW
    corrections, from the ewsudakov entry of fks_configurations.json.

    Args:
        configurations: the content of fks_configurations.json (see
            read_fks_configurations).
        parameters: the EWSudakovParameters. Their scheme must be that of the
            entry.
        modes: the EWSudakovModes, by default those of compute_ewsudakov
            (sud_mod 0 and 1).

    Raises:
        ValueError: as ew_sudakov_process, and for parameters in another
            scheme than that of the process.
    """
    process = ew_sudakov_process(configurations)
    scheme = configurations["ewsudakov"]["scheme"]
    if parameters.scheme.name != scheme:
        raise ValueError(
            f"the parameters are in the scheme {parameters.scheme.name}, the "
            f"process in {scheme}"
        )
    if modes is None:
        modes = [EWSudakovMode(sud_mod=0), EWSudakovMode(sud_mod=1)]
    return EWSudakov(process, parameters, modes)
