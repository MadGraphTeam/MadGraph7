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

The functions here turn it into the FKSMappings of a Born mapping, with the
0-based positions of madspace. The Born mapping gives the masses, so the
emitter is massive where it gives it a mass.
"""

import json
import os

from ._madspace_py import FKSConfigurationMapping, FKSMapping

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


def fks_mappings(born, configurations, s_hat_min=0.0):
    """The FKSMappings of the FKS configurations of a Born, in their order.

    Args:
        born: the PhaseSpaceMapping of the Born process, its momenta in the
            order of born_pdgs.
        configurations: the content of fks_configurations.json (see
            read_fks_configurations).
        s_hat_min: lower bound on the partonic energy squared of the real
            emission, see FKSMapping.

    Raises:
        ValueError: if the Born mapping does not have the particles of the
            Born, or a real process is not its Born with i and j in the place
            of i + j, the order FKSMapping assumes.
    """
    born_pdgs = list(configurations["born_pdgs"])
    if configurations["nincoming"] != 2:
        raise ValueError("FKS mappings need two incoming particles")
    if born.particle_count() != len(born_pdgs):
        raise ValueError(
            f"the Born mapping has {born.particle_count()} particles, "
            f"the Born {len(born_pdgs)}"
        )
    mappings = []
    for configuration in configurations["configurations"]:
        if born_of_real(configuration) != born_pdgs:
            raise ValueError(
                f"the real process {configuration['real_pdgs']} is not the Born "
                f"{born_pdgs} with i_fks = {configuration['i_fks']} and "
                f"j_fks = {configuration['j_fks']}"
            )
        mappings.append(
            FKSMapping(
                born,
                configuration["i_fks"] - 1,
                configuration["j_fks"] - 1,
                s_hat_min=s_hat_min,
            )
        )
    return mappings


def fks_configuration_mapping(born, configurations, s_hat_min=0.0):
    """The FKSConfigurationMapping of the FKS configurations of a Born, with
    fks_index the position of the configuration (nFKSprocess - 1). See
    fks_mappings."""
    return FKSConfigurationMapping(fks_mappings(born, configurations, s_hat_min))
