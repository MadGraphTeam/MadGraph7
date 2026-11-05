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
import torch

import madspace as ms
from madspace.fks import (
    born_of_real,
    fks_configuration_mapping,
    fks_configurations,
    fks_mappings,
    read_fks_configurations,
    real_masses,
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


@pytest.mark.parametrize("path", DATA, ids=[os.path.basename(p)[:-5] for p in DATA])
def test_fks_configurations_of_the_exporter(path):
    """The FKSConfigurations that the classes of the FKS subtraction take hold
    the metadata of the exporter with 0-based positions, and the masses of the
    Born mapping with a massless i."""
    data = reference(path)
    born = born_mapping(data)
    exported = data["fks_configurations"]["configurations"]
    configurations = fks_configurations(born, data["fks_configurations"])
    assert len(configurations) == len(exported)
    for config, exp in zip(configurations, exported):
        assert (config.i_fks, config.j_fks) == (exp["i_fks"] - 1, exp["j_fks"] - 1)
        assert config.pdgs == exp["real_pdgs"]
        assert config.colors == exp["real_colors"]
        assert config.charges == exp["real_charges"]
        assert config.fks_j_from_i == [
            [j - 1 for j in js] for js in exp["fks_j_from_i"]
        ]
        assert config.masses == real_masses(born.external_masses(), exp)
        assert config.splitting_types == exp["splitting_type"]
        assert config.tags == exp.get("real_tags", [])
        assert config.qed_splitting() == (exp["splitting_type"][0] == "QED")
        assert config.particle_count() == len(exp["real_pdgs"])
        assert config.initial_state() == (exp["j_fks"] <= 2)
        born_j = config.born_j_fks()
        assert born_j == (
            config.j_fks if config.j_fks < config.i_fks else config.j_fks - 1
        )
        assert data["fks_configurations"]["born_pdgs"][born_j] == exp["ij_pdg"]


@pytest.mark.parametrize("path", DATA, ids=[os.path.basename(p)[:-5] for p in DATA])
def test_symmetry_factors_of_the_fortran(path):
    """The Born of each configuration is that of the exporter, and its
    symmetry factor and identical-particle ratio are fkssymmetryfactor and
    iden_comp of setfksfactor (fks_singular.f) for the real emission."""
    data = reference(path)
    configurations = fks_configurations(born_mapping(data), data["fks_configurations"])
    for config, conf in zip(configurations, data["configs"]):
        assert config.born_pdgs() == data["fks_configurations"]["born_pdgs"]
        assert config.identical_particle_ratio() == conf["iden_comp"]
        factors = {
            p["fks_symmetry_factor"] for p in conf["points"] if "prefactors" in p
        }
        factors |= {
            p["fks_symmetry_factor_deg"] for p in conf["points"] if "prefactors" in p
        }
        assert factors == {config.symmetry_factor()}


def test_born_of_splittings():
    """The mother of i and j: j for a gluon or photon i, a gluon (QCD) or
    photon (QED) for a final-state pair of a particle and its antiparticle or
    for a particle off an incoming one of the same flavor, the antiparticle of
    i off an incoming gluon or photon, and no Born for other pairs."""

    def born(pdgs, i_fks, j_fks, splitting_types=("QCD",)):
        colors = [
            8 if p == 21 else (1 if p in (22, 11, -11) else (3 if p > 0 else -3))
            for p in pdgs
        ]
        n = len(pdgs)
        pairs = [[] for _ in range(n)]
        pairs[i_fks] = [j_fks]
        return ms.FKSConfiguration(
            i_fks=i_fks,
            j_fks=j_fks,
            pdgs=pdgs,
            colors=colors,
            charges=[0.0] * n,
            masses=[0.0] * n,
            fks_j_from_i=pairs,
            splitting_types=list(splitting_types),
        ).born_pdgs()

    assert born([2, -2, 2, -2, 21], 4, 0) == [2, -2, 2, -2]
    assert born([2, -2, 21, 2, -2], 4, 3) == [2, -2, 21, 21]
    assert born([2, -2, 21, 21, 2], 4, 0) == [21, -2, 21, 21]
    assert born([21, -2, 21, 21, 2], 4, 0) == [-2, -2, 21, 21]
    qed = ("QED",)
    assert born([2, -2, 11, -11, 22], 4, 0, qed) == [2, -2, 11, -11]
    assert born([2, -2, 11, -11, 2, -2], 5, 4, qed) == [2, -2, 11, -11, 22]
    assert born([2, -2, 22, 11, -11], 4, 3, qed) == [2, -2, 22, 22]
    assert born([2, 22, 11, -11, 2], 4, 0, qed) == [22, 22, 11, -11]
    assert born([22, -2, 11, -11, -2], 4, 0, qed) == [2, -2, 11, -11]
    # the first type sets the mother, the second is that of an extra
    # counterterm
    assert born([2, -2, 21, 2, -2], 4, 3, ("QED", "QCD")) == [2, -2, 21, 22]
    for pdgs, i_fks, j_fks in [
        ([2, -2, 21, 2, 2], 4, 3),
        ([2, -2, 21, 21, 2], 4, 1),
        ([2, -2, 21, 2, -1], 4, 3),
    ]:
        with pytest.raises(ValueError, match="splitting"):
            born(pdgs, i_fks, j_fks)


def test_configuration_mapping_from_configurations():
    """FKSConfigurationMapping from FKSConfigurations is the one from their
    (i_fks, j_fks), and checks that they have the masses of the Born
    mapping."""
    data = reference(
        next(p for p in DATA if os.path.basename(p).startswith("pp_ttx_P0_gg"))
    )
    born = born_mapping(data)
    configurations = fks_configurations(born, data["fks_configurations"])
    pairs = [(c.i_fks, c.j_fks) for c in configurations]
    from_configurations = ms.FKSConfigurationMapping(born, configurations)
    assert from_configurations.configurations() == pairs
    from_pairs = ms.FKSConfigurationMapping(born, pairs)
    rng = np.random.default_rng(5)
    batch = 200
    r = 0.05 + 0.9 * rng.random((batch, from_pairs.random_dim()))
    index = rng.integers(0, len(pairs), batch).astype(np.int32)
    args = [torch.as_tensor(r).to(torch.get_default_device())]
    conditions = [
        torch.as_tensor(index, dtype=torch.int32).to(torch.get_default_device())
    ]
    for ours, theirs in zip(
        from_configurations.map_forward(args, conditions),
        from_pairs.map_forward(args, conditions),
    ):
        np.testing.assert_array_equal(
            torch.as_tensor(ours).cpu().numpy(), torch.as_tensor(theirs).cpu().numpy()
        )
    # a top mass that is not the one of the Born mapping
    wrong = configurations[0]
    wrong.masses = [m + 1.0 if m > 0 else m for m in wrong.masses]
    with pytest.raises(ValueError, match="masses"):
        ms.FKSConfigurationMapping(born, [wrong])


def test_checks_of_the_configurations():
    pdgs = [2, -2, 11, -11, 21]
    colors = [3, -3, 1, 1, 8]

    def config(**changes):
        fields = dict(
            i_fks=4,
            j_fks=0,
            pdgs=pdgs,
            colors=colors,
            charges=[0.0] * 5,
            masses=[0.0] * 5,
            fks_j_from_i=[[], [], [], [], [0, 1]],
        )
        fields.update(changes)
        return ms.FKSConfiguration(**fields)

    assert ms.FKSEikonals([config()]).configuration_count() == 1
    with pytest.raises(ValueError, match="at least one"):
        ms.FKSEikonals([])
    with pytest.raises(ValueError, match="massless"):
        ms.FKSEikonals([config(masses=[0.0] * 4 + [1.0])])
    with pytest.raises(ValueError, match="one entry per particle"):
        ms.FKSEikonals([config(charges=[0.0] * 4)])
    with pytest.raises(ValueError, match="out of range"):
        ms.FKSEikonals([config(fks_j_from_i=[[], [], [], [], [0, 5]])])
    with pytest.raises(ValueError, match="FKS pairs"):
        ms.FKSEikonals([config(j_fks=2)])
