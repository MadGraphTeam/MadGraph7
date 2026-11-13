"""madspace.cs: the Catani-Seymour dipoles of real processes, from their
flavors, and the cs block of an aMC@NLO output.

The dipoles of every real process of the provider outputs
(MADSPACE_FKS_OUTPUTS, as for test_fks_fortran_provider.py) also come from
here, each flavor line of a real once over the subprocesses.
"""

import collections
import os

import pytest

from madspace.cs import (
    born_order,
    cs_dipoles,
    cs_subtraction,
    final_splitting,
    initial_splitting,
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
    with pytest.raises(ValueError):
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
    with pytest.raises(NotImplementedError):
        cs_dipoles([6, 5, 24, 21], borns([6, 5, 24]), TOP, nincoming=1)


@pytest.mark.skipif(not OUTPUTS, reason="MADSPACE_FKS_OUTPUTS names no aMC@NLO outputs")
@pytest.mark.parametrize("output", ["epem_jj", "pp_epem", "pp_ttx"])
def test_outputs(output):
    """The reals of the subprocesses of an output, each flavor line once, and
    their dipoles onto the Borns of the output."""
    block = read_cs_subtraction(os.path.join(OUTPUTS, output), TOP)
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
