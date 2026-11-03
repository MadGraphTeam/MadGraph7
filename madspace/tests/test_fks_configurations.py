"""The FKS configurations of a Born from the metadata of the FKS exporter.

The reference data of genps_fks.f (see test_fks_reference.py) store, for each
subprocess, the fks_configurations.json that MadGraph's FKS exporter writes
next to fks_info.inc. madspace.fks builds the FKSMappings of a Born mapping
from it. Their FKS configurations must be those of the Fortran, and the
pointwise comparison with genps_fks.f (test_fks_counter_events.py) runs on
them. The color types, charges and FKS pairs of the real processes, which the
sector functions need, must be those of the Fortran's common blocks.
"""

import glob
import json
import os

import numpy as np
import pytest

import madspace as ms
from madspace.fks import (
    born_of_real,
    fks_configuration_mapping,
    fks_mappings,
    read_fks_configurations,
)

DATA = sorted(
    glob.glob(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "test_data", "fks", "*.json"
        )
    )
)


def reference(path):
    with open(path) as f:
        return json.load(f)


def born_mapping(data):
    """RAMBO mapping of the Born, with the masses of the Fortran."""
    conf = data["configs"][0]
    masses = np.delete(np.array(conf["masses"]), conf["i_fks"] - 1)
    return ms.PhaseSpaceMapping(
        list(masses),
        2 * np.sqrt(data["ebeam"][0] * data["ebeam"][1]),
        leptonic=data["lpp"] == [0, 0],
        mode=ms.PhaseSpaceMapping.TChannelMode.rambo,
    )


@pytest.mark.parametrize("path", DATA, ids=[os.path.basename(p)[:-5] for p in DATA])
def test_configurations_of_the_fortran(path):
    data = reference(path)
    configurations = data["fks_configurations"]
    born = born_mapping(data)
    mappings = fks_mappings(born, configurations)
    assert len(mappings) == len(configurations["configurations"])
    for conf in data["configs"]:
        mapping = mappings[conf["nfks"] - 1]
        exported = configurations["configurations"][conf["nfks"] - 1]
        assert exported["real_pdgs"] == conf["pdg"]
        assert (mapping.i_fks(), mapping.j_fks()) == (
            conf["i_fks"] - 1,
            conf["j_fks"] - 1,
        )
        # the emitter is massive where the Born mapping gives it a mass
        j_fks = conf["j_fks"] - 1
        assert mapping.radiation().emitter_mass() == (
            conf["masses"][j_fks] if j_fks >= 2 else 0.0
        )
        assert mapping.particle_count() == len(conf["masses"])
        # what the sector functions need: the color types, the charges and
        # the FKS pairs of the real process, as fks_inc_chooser sets them
        assert exported["real_colors"] == conf["particle_type"]
        # (fks_info.inc has the charges to 15 digits)
        np.testing.assert_allclose(
            exported["real_charges"], conf["particle_charge"], rtol=0, atol=1e-14
        )
        assert exported["fks_j_from_i"] == conf["fks_j_from_i"]
    if len(mappings) > 1:
        mapping = fks_configuration_mapping(born, configurations)
        assert isinstance(mapping, ms.FKSConfigurationMapping)
        assert mapping.configurations() == [(m.i_fks(), m.j_fks()) for m in mappings]


def test_born_of_real():
    # u g > u e- e+ u~ g, i the gluon off the incoming gluon
    configuration = {
        "i_fks": 6,
        "j_fks": 1,
        "ij_pdg": 2,
        "real_pdgs": [21, 21, 2, 11, -11, -2],
    }
    assert born_of_real(configuration) == [2, 21, 2, 11, -11]
    configuration.update(i_fks=1, j_fks=3, real_pdgs=[21, 2, 2, 11, -11])
    assert born_of_real(configuration) == [2, 2, 11, -11]


def test_read_file_or_directory(tmp_path):
    data = reference(DATA[0])["fks_configurations"]
    path = tmp_path / "fks_configurations.json"
    path.write_text(json.dumps(data))
    assert read_fks_configurations(str(path)) == data
    assert read_fks_configurations(str(tmp_path)) == data


def test_rejects_another_born():
    data = reference(
        next(p for p in DATA if os.path.basename(p).startswith("pp_epemj_P0_ug"))
    )
    born = born_mapping(data)
    configurations = data["fks_configurations"]
    # the Born in another order
    swapped = dict(configurations, born_pdgs=configurations["born_pdgs"][::-1])
    with pytest.raises(ValueError):
        fks_mappings(born, swapped)
    # a Born mapping with another number of particles
    other = ms.PhaseSpaceMapping(
        [0.0] * 4, 1000.0, mode=ms.PhaseSpaceMapping.TChannelMode.rambo
    )
    with pytest.raises(ValueError):
        fks_mappings(other, configurations)
    # a decay
    with pytest.raises(ValueError):
        fks_mappings(born, dict(configurations, nincoming=1))
