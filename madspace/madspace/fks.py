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
final-state emitter, s_hat_min for an incoming one.

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
    born, configurations, s_hat_min=0.0, cuts=None, initial_map="symmetric"
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

    Raises:
        ValueError: if the Born mapping does not have the particles of the
            Born, or a real process is not its Born with i and j in the place
            of i + j, the order FKSMapping assumes.
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
    mappings = []
    for configuration in checked_configurations(
        born_pdgs, born_plain.particle_count(), configurations
    ):
        i_fks, j_fks = configuration["i_fks"] - 1, configuration["j_fks"] - 1
        if cuts is None:
            mapping = FKSMapping(born, i_fks, j_fks, s_hat_min, initial_map)
        elif j_fks >= 2:
            mapping = FKSMapping(born_final, i_fks, j_fks, s_hat_min, initial_map)
        else:
            mapping = FKSMapping(
                born_plain, i_fks, j_fks, s_hat_min_initial, initial_map
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
    )


def real_masses(born_masses, configuration):
    """The masses of the real process of a configuration: those of the Born,
    with j in the place of i + j (whose mass it has) and a massless i."""
    i_fks = configuration["i_fks"] - 1
    masses = list(born_masses)
    return masses[:i_fks] + [0.0] + masses[i_fks:]


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


def fks_eikonals(born, configurations):
    """The FKSEikonals of the FKS configurations of a Born, with fks_index as
    for fks_sector_function.

    Raises:
        ValueError: as fks_configurations.
    """
    return FKSEikonals(fks_configurations(born, configurations))


def fks_splitting_kernels(born, configurations):
    """The FKSSplittingKernels of the FKS configurations of a Born, with
    fks_index as for fks_sector_function.

    Raises:
        ValueError: as fks_configurations.
    """
    return FKSSplittingKernels(fks_configurations(born, configurations))


def fks_prefactors(born, configurations, **parameters):
    """The FKSPrefactors of the FKS configurations of a Born, with fks_index
    as for fks_sector_function.

    Args:
        born: the PhaseSpaceMapping of the Born process, for its masses, its
            momenta in the order of born_pdgs.
        configurations: the content of fks_configurations.json (see
            read_fks_configurations).
        parameters: the parameters of the subtraction of fks_powers.inc
            (xicut, xi_s_cut, delta_o, delta_i, delta_s), see FKSPrefactors.

    Raises:
        ValueError: as fks_configurations.
    """
    return FKSPrefactors(fks_configurations(born, configurations), **parameters)


def fks_limits(born, configurations, collinear_born=False):
    """The FKSLimits of the FKS configurations of a Born, with fks_index as
    for fks_sector_function and the coupling orders of fks_coupling_orders,
    with the Born of the collinear counter-event apart for lepton beams
    (collinear_born).

    Raises:
        ValueError: as fks_configurations and fks_coupling_orders, and for a
            splitting that is not one of QCD or QED or an extra counterterm.
    """
    return FKSLimits(
        fks_configurations(born, configurations),
        fks_coupling_orders(configurations),
        collinear_born,
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
            nuclei of the beams (upc_radii), and for lepton beams the Born of
            the collinear counter-event apart (collinear_born), see
            FKSSubtraction.

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
