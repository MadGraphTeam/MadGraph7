"""madspace.nlo.cs: the Catani-Seymour dipoles of real processes, from their
flavors, and the cs block of an aMC@NLO output.

The dipoles of every real process of the provider outputs
(MADSPACE_FKS_OUTPUTS, as for test_fks_fortran_provider.py) also come from
here, each flavor line of a real once over the subprocesses.
"""

import collections
import json
import os

import pytest
from fks_testing import fks_output

from madspace.nlo.cs import (
    born_order,
    charge,
    cs_dipoles,
    cs_subtraction,
    final_splitting,
    initial_splitting,
    qed_final_splitting,
    qed_initial_splitting,
    read_cs_subtraction,
    symmetry,
)

OUTPUTS = os.environ.get("MADSPACE_FKS_OUTPUTS")
TOP = {6: 173.0}


def borns(*lines):
    return [dict(pdgs=line, flavors=[line]) for line in lines]


def families(dipoles):
    return collections.Counter((d["family"], d["kernel"]) for d in dipoles)


def test_splittings():
    masses = [0.0] * 6
    # final q g, g g, q q~ and none for q q or massive Q Q~
    assert final_splitting([11, -11, 2, 21], masses, 2, 3) == (3, 2, 2, "qg")
    assert final_splitting([11, -11, 21, 2], masses, 2, 3) == (2, 3, 2, "qg")
    assert final_splitting([11, -11, 21, 21], masses, 2, 3) == (3, 2, 21, "gg")
    assert final_splitting([11, -11, 2, -2], masses, 2, 3) == (3, 2, 21, "gq")
    assert final_splitting([11, -11, 2, 2], masses, 2, 3) is None
    assert final_splitting([11, -11, 6, -6], [0, 0, 173, 173], 2, 3) is None
    # an incoming parton with a final one: q g, g g, g -> q~ q, q -> g q
    assert initial_splitting([2, -2, 11, -11, 21], masses, 0, 4) == (4, 0, 2, "qg")
    assert initial_splitting([21, -2, 11, -11, 21], masses, 0, 4) == (
        4,
        0,
        21,
        "gg",
    )
    assert initial_splitting([21, -2, 11, -11, -2], masses, 0, 4) == (
        4,
        0,
        2,
        "qq",
    )
    assert initial_splitting([-2, 21, 11, -11, -2], masses, 0, 4) == (
        4,
        0,
        21,
        "gq",
    )
    assert initial_splitting([11, -11, 2, -2, 21], masses, 0, 4) is None


def test_symmetry_and_born_order():
    assert symmetry([-11, 11, 2, -2, 21, 21]) == 2
    assert symmetry([21, 21, 21, 21, 21]) == 6
    assert symmetry([21, 21, 6, -6]) == 1
    assert born_order([2, 21, 6, -6, 2], [2, 21, 2, 6, -6]) == [0, 1, 4, 2, 3]
    assert born_order([2, 21, 6, -6], [21, 2, 6, -6]) is None


def test_epem_dipoles():
    """e+ e- > u u~ g: the quark and the antiquark emit the gluon, each with
    the other as the spectator, with massless and massive quarks."""
    dipoles = cs_dipoles([-11, 11, 2, -2, 21], borns([-11, 11, 2, -2]))
    assert [(d["i"], d["j"], d["k"], d["kernel"]) for d in dipoles] == [
        (4, 2, 3, "qg"),
        (4, 3, 2, "qg"),
    ]
    assert all(d["family"] == "ff" and d["symmetry"] == 1.0 for d in dipoles)
    massive = cs_dipoles([-11, 11, 6, -6, 21], borns([-11, 11, 6, -6]), TOP)
    assert len(massive) == 2
    # two gluons: q g (with both gluons as i), g g, each with three spectators
    # less one, and the symmetry factor 1/2
    dipoles = cs_dipoles([-11, 11, 2, -2, 21, 21], borns([-11, 11, 2, -2, 21]))
    assert families(dipoles) == {("ff", "qg"): 8, ("ff", "gg"): 2}
    assert {d["symmetry"] for d in dipoles} == {0.5}
    # g -> d d~ onto the gluon of the Born, u u~ -> g onto e+ e- > d d~ g
    dipoles = cs_dipoles(
        [-11, 11, 2, -2, 1, -1], borns([-11, 11, 2, -2, 21], [-11, 11, 1, -1, 21])
    )
    assert families(dipoles) == {("ff", "gq"): 4}
    born_of = {(d["i"], d["j"]): (d["born"], d["born_order"]) for d in dipoles}
    assert born_of == {(5, 4): (0, [0, 1, 2, 3, 4]), (3, 2): (1, [0, 1, 3, 4, 2])}


def test_hadronic_dipoles():
    """p p > t t~: the families of the reals of each initial state, and a
    real that projects onto the Borns of other subprocesses."""
    born_list = borns([21, 21, 6, -6], [2, -2, 6, -6], [-2, 2, 6, -6])
    gg = cs_dipoles([21, 21, 6, -6, 21], born_list, TOP)
    assert families(gg) == {
        ("ff", "qg"): 2,
        ("fi", "qg"): 4,
        ("if", "gg"): 4,
        ("ii", "gg"): 2,
    }
    ug = cs_dipoles([2, 21, 6, -6, 2], born_list, TOP)
    assert families(ug) == {
        ("if", "gq"): 2,
        ("ii", "gq"): 1,
        ("if", "qq"): 2,
        ("ii", "qq"): 1,
    }
    # u -> g u onto g g > t t~, g -> u~ u onto u u~ > t t~
    assert {(d["j"], d["born"]) for d in ug} == {(0, 0), (1, 1)}
    # a Born without matrix element at the order of the output is left out
    epem = borns([2, -2, 11, -11], [-2, 2, 11, -11])
    assert families(cs_dipoles([21, -2, 11, -11, -2], epem)) == {("ii", "qq"): 1}


def test_subtraction_block():
    block = cs_subtraction(
        [
            dict(pdgs=[2, -2, 6, -6, 21], flavors=[[2, -2, 6, -6, 21]]),
            dict(pdgs=[21, 21, 6, -6, 21], flavors=[[21, 21, 6, -6, 21]]),
        ],
        borns([21, 21, 6, -6], [2, -2, 6, -6]),
        TOP,
    )
    assert [d["real"] for d in block["cs"]] == [0] * 12 + [1] * 12
    # the lines of a real whose dipoles go to other Borns make reals of their
    # own
    split = cs_subtraction(
        [
            dict(
                pdgs=[-1, 22, 13, -13, -1],
                flavors=[[-1, 22, 13, -13, -1], [-3, 22, 13, -13, -3]],
            )
        ],
        [
            dict(pdgs=[-1, 1, 13, -13], flavors=[[-1, 1, 13, -13]]),
            dict(pdgs=[-3, 3, 13, -13], flavors=[[-3, 3, 13, -13]]),
        ],
        groups=("QED",),
    )
    assert [real["flavors"] for real in split["reals"]] == [
        [[-1, 22, 13, -13, -1]],
        [[-3, 22, 13, -13, -3]],
    ]
    assert {(d["real"], d["born"]) for d in split["cs"]} == {(0, 0), (1, 1)}
    with pytest.raises(ValueError, match="other dipoles"):
        cs_subtraction(
            [
                dict(
                    pdgs=[2, -2, 6, -6, 21],
                    flavors=[[2, -2, 6, -6, 21], [1, -1, 6, -6, 21]],
                )
            ],
            borns([2, -2, 6, -6]),
            TOP,
        )


def test_decays():
    """A colored decaying particle is the spectator of the final-state quarks
    and gluons (fd), a colorless one of none."""
    masses = {6: 173.0, 5: 4.7}
    dipoles = cs_dipoles([6, 5, 24, 21], borns([6, 5, 24]), masses, nincoming=1)
    assert [
        (d["family"], d["i"], d["j"], d["k"], d["kernel"], d["born_order"])
        for d in dipoles
    ] == [("fd", 3, 1, 0, "qg", [0, 1, 2])]
    assert dipoles[0]["symmetry"] == 1.0
    # t > b e+ ve: the leptons are the recoil system
    dipoles = cs_dipoles(
        [6, 5, -11, 12, 21], borns([6, 5, -11, 12]), masses, nincoming=1
    )
    assert [(d["family"], d["j"], d["k"]) for d in dipoles] == [("fd", 1, 0)]
    # t > b u d~: the quarks of the W are each other's spectators besides t
    dipoles = cs_dipoles([6, 5, 2, -1, 21], borns([6, 5, 2, -1]), masses, nincoming=1)
    assert families(dipoles) == {("ff", "qg"): 6, ("fd", "qg"): 3}
    # a colorless decaying particle: H > b b~
    dipoles = cs_dipoles([25, 5, -5, 21], borns([25, 5, -5]), masses, nincoming=1)
    assert families(dipoles) == {("ff", "qg"): 2}
    # t > b w+ g g and t > b w+ u u~: the gluon of the Born splits against
    # the top and the b
    dipoles = cs_dipoles([6, 5, 24, 21, 21], borns([6, 5, 24, 21]), masses, nincoming=1)
    assert families(dipoles) == {
        ("fd", "qg"): 2,
        ("ff", "qg"): 2,
        ("fd", "gg"): 1,
        ("ff", "gg"): 1,
    }
    assert {d["symmetry"] for d in dipoles} == {0.5}
    dipoles = cs_dipoles([6, 5, 24, 2, -2], borns([6, 5, 24, 21]), masses, nincoming=1)
    assert [(d["family"], d["i"], d["j"], d["k"], d["kernel"]) for d in dipoles] == [
        ("fd", 4, 3, 0, "gq"),
        ("ff", 4, 3, 1, "gq"),
    ]


def test_qed_splittings():
    """The QED splittings: a photon from a charged particle (also a massive
    W), a massless charged fermion pair from a photon, and for an incoming
    particle a photon splitting into the antifermion of i and a fermion that
    gives a photon to the Born."""
    masses = [0.0] * 6
    assert charge(2) == pytest.approx(2 / 3) and charge(-11) == 1.0
    assert charge(-24) == -1.0 and charge(22) == 0.0 and charge(21) == 0.0
    assert qed_final_splitting([2, -2, 11, 22], masses, 2, 3) == (3, 2, 11, "qg")
    assert qed_final_splitting([2, -2, 22, -24], [0, 0, 0, 80], 2, 3) == (
        2,
        3,
        -24,
        "qg",
    )
    assert qed_final_splitting([2, -2, 11, -11], masses, 2, 3) == (3, 2, 22, "gq")
    assert qed_final_splitting([2, -2, 6, -6], [0, 0, 173, 173], 2, 3) is None
    assert qed_final_splitting([2, -2, 22, 22], masses, 2, 3) is None
    assert qed_final_splitting([2, -2, 21, 2], masses, 2, 3) is None
    assert qed_initial_splitting([2, -2, 11, -11, 22], masses, 0, 4) == (
        4,
        0,
        2,
        "qg",
    )
    assert qed_initial_splitting([22, 2, 11, -11, 2], masses, 0, 4) == (
        4,
        0,
        -2,
        "qq",
    )
    assert qed_initial_splitting([2, 22, 11, -11, 2], masses, 0, 4) == (
        4,
        0,
        22,
        "gq",
    )
    assert qed_initial_splitting([21, 2, 11, -11, 2], masses, 0, 4) is None


def test_qed_dipoles():
    """The QED dipoles of p p > e+ e- with their groups: a photon from every
    charged particle with every other one as the spectator, the photon of
    the Born of a fermion that gives it up shared by the charged spectators,
    and with QCD the gluon emissions of the same real too."""
    born_list = borns([2, -2, 11, -11], [22, 22, 11, -11], [-2, 2, 11, -11])
    photon = cs_dipoles([2, -2, 11, -11, 22], born_list, groups=("QED",))
    assert families(photon) == {
        ("fi", "qg"): 4,
        ("ff", "qg"): 2,
        ("ii", "qg"): 2,
        ("if", "qg"): 4,
    }
    assert all(d["group"] == "QED" and d["weight"] is None for d in photon)
    assert cs_dipoles([2, -2, 11, -11, 22], born_list) == []
    fermion = cs_dipoles([2, 22, 11, -11, 2], born_list, groups=("QCD", "QED"))
    shared = [d for d in fermion if d["kernel"] == "gq"]
    assert [(d["j"], d["k"], d["weight"], d["born"]) for d in shared] == [
        (0, 2, 0.5, 1),
        (0, 3, 0.5, 1),
    ]
    split = [d for d in fermion if d["kernel"] == "qq"]
    assert [(d["j"], d["k"], d["weight"], d["born"]) for d in split] == [
        (1, 0, None, 0),
        (1, 2, None, 0),
        (1, 3, None, 0),
    ]
    # a final-state pair from a photon of the Born, shared by the quarks
    pair = cs_dipoles(
        [2, -2, 22, 11, -11], borns([2, -2, 22, 22], [2, -2, 11, -11]), groups=("QED",)
    )
    shared = [d for d in pair if d["kernel"] == "gq"]
    assert [(d["i"], d["j"], d["k"], d["weight"]) for d in shared] == [
        (4, 3, 0, 0.5),
        (4, 3, 1, 0.5),
    ]
    mixed = cs_dipoles([2, -2, 11, -11, 21], born_list, groups=("QCD", "QED"))
    assert {d["group"] for d in mixed} == {"QCD"}
    with pytest.raises(ValueError, match="EW"):
        cs_dipoles([2, -2, 11, -11, 22], born_list, groups=("EW",))


def test_read_groups(tmp_path):
    """read_cs_subtraction takes the dipoles of the splitting types of the
    configurations of the output."""
    directory = tmp_path / "SubProcesses" / "P0_uux_epem"
    directory.mkdir(parents=True)
    data = dict(
        nincoming=2,
        born_pdgs=[2, -2, 11, -11],
        configurations=[
            dict(real_me=1, real_pdgs=[2, -2, 11, -11, 22], splitting_type=["QED"]),
            dict(real_me=2, real_pdgs=[2, -2, 11, -11, 21], splitting_type=["QCD"]),
        ],
    )
    (directory / "fks_configurations.json").write_text(json.dumps(data))
    block = read_cs_subtraction(str(tmp_path))
    groups = collections.Counter(d["group"] for d in block["cs"])
    assert groups == {"QED": 12, "QCD": 2}


@pytest.mark.skipif(not OUTPUTS, reason="MADSPACE_FKS_OUTPUTS names no aMC@NLO outputs")
@pytest.mark.parametrize("output", ["epem_jj", "pp_epem", "pp_ttx"])
def test_outputs(output):
    """The reals of the subprocesses of an output, each flavor line once, and
    their dipoles onto the Borns of the output."""
    block = read_cs_subtraction(fks_output(output), TOP)
    lines = [tuple(f) for real in block["reals"] for f in real["flavors"]]
    assert len(lines) == len(set(lines))
    assert {d["real"] for d in block["cs"]} == set(range(len(block["reals"])))
    for d in block["cs"]:
        real = block["reals"][d["real"]]["pdgs"]
        born = block["borns"][d["born"]]
        line = [p for n, p in enumerate(real) if n != d["i"]]
        emitter = d["j"] - (d["j"] > d["i"])
        # the legs of the Born in the order of the dipole are those of the
        # real process without i, but the emitter
        assert any(
            all(
                flavors[n] == line[m]
                for n, m in enumerate(d["born_order"])
                if m != emitter
            )
            for flavors in born["flavors"]
        )
