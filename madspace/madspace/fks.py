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
        real_me, need_color_links, need_charge_links, splitting_type,
        extra_cnt_index
        real_colors, real_charges: color representation (1, 3, -3, 8) and
            electric charge of each particle of the real process
        fks_j_from_i: for each particle i of the real process, the particles
            j it forms an FKS pair with (1-based), the pairs whose sector
            functions partition the phase space of the real process

The functions here turn it into the FKSMappings of a Born mapping, one per
configuration, or into one FKSConfigurationMapping for all of them, with the
0-based positions of madspace. The Born mapping gives the masses, so the
emitter is massive where it gives it a mass. With the cuts on the Born-level
objects, the configurations get the lower bound on the partonic energy that
the cuts imply (see fks_sqrt_s_hat_min): a floor of the Born mapping for a
final-state emitter, s_hat_min for an incoming one.

fks_configurations gives the FKSConfigurations that the classes of the FKS
subtraction take. With them, fks_sector_function builds the sector functions
S_ij of the configurations, fks_eikonals the eikonal factors of their soft
limits, fks_splitting_kernels the factors of their collinear limits and
fks_prefactors the prefactors of the real emission and its counterterms.
"""

import json
import os

from ._madspace_py import (
    CutItem,
    Cuts,
    FKSConfiguration,
    FKSConfigurationMapping,
    FKSEikonals,
    FKSMapping,
    FKSPrefactors,
    FKSSectorFunction,
    FKSSplittingKernels,
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


def fks_mappings(born, configurations, s_hat_min=0.0, cuts=None):
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
            mapping = FKSMapping(born, i_fks, j_fks, s_hat_min=s_hat_min)
        elif j_fks >= 2:
            mapping = FKSMapping(born_final, i_fks, j_fks, s_hat_min=s_hat_min)
        else:
            mapping = FKSMapping(born_plain, i_fks, j_fks, s_hat_min=s_hat_min_initial)
        mappings.append(mapping)
    return mappings


def fks_configuration_mapping(born, configurations, s_hat_min=0.0, cuts=None):
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
        born, pairs, s_hat_min=s_hat_min, sqrt_s_hat_floor=floor
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
        ValueError: as fks_mappings, and for a configuration of a real process
            without FKS pairs (a finite one).
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
        result.append(
            FKSConfiguration(
                i_fks=configuration["i_fks"] - 1,
                j_fks=configuration["j_fks"] - 1,
                pdgs=list(configuration["real_pdgs"]),
                colors=list(configuration["real_colors"]),
                charges=list(configuration["real_charges"]),
                masses=real_masses(born_masses, configuration),
                fks_j_from_i=[[j - 1 for j in js] for js in pairs],
            )
        )
    return result


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
