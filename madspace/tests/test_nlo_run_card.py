"""The settings of the run card the NLO drivers refuse (check_run_card of
madgraph/mg7/nlo/core.py) rather than ignore: cuts by PDG code, functions of
the user (custom_fcts), more than one dynamical scale choice or PDF set, and
beams of different energies for scattering. The number of incoming
particles of an output from its proc_characteristics.
"""

import os

import pytest
from madgraph.mg7.nlo.core import check_run_card, incoming_count

BASE = dict(
    ickkw=0,
    lpp1=1,
    lpp2=1,
    pdlabel="nn23nlo",
    pdfscheme=0,
    dynamical_scale_choice=[-1],
    lhaid=[244600],
    pt_min_pdg={},
    pt_max_pdg={},
    mxx_min_pdg={},
    custom_fcts=[],
    ebeam1=6500.0,
    ebeam2=6500.0,
)


# the same settings as the text of the card, as the references keep them
TEXT = dict(
    BASE,
    dynamical_scale_choice="-1",
    lhaid="244600",
    pt_min_pdg="{}",
    pt_max_pdg="{}",
    mxx_min_pdg="{}",
    custom_fcts="",
)


@pytest.mark.parametrize("base", [BASE, TEXT], ids=["values", "text"])
def test_defaults(base):
    """The defaults of the run card pass, also with an empty name in
    custom_fcts and the type entry of a dictionary."""
    check_run_card(base)
    check_run_card({**BASE, "custom_fcts": [""], "pt_min_pdg": {"__type__": 0.0}})


@pytest.mark.parametrize("key", ["pt_min_pdg", "pt_max_pdg", "mxx_min_pdg"])
@pytest.mark.parametrize("value", [{6: 100.0}, "{6: 100.0}", "{'6': 100.0}"])
def test_cuts_by_pdg(key, value):
    with pytest.raises(NotImplementedError, match=key):
        check_run_card({**BASE, key: value})


@pytest.mark.parametrize("value", [["cuts.f"], "cuts.f"])
def test_functions_of_the_user(value):
    with pytest.raises(NotImplementedError, match="custom_fcts"):
        check_run_card({**BASE, "custom_fcts": value})


@pytest.mark.parametrize(
    "key, value",
    [
        ("dynamical_scale_choice", [-1, 1]),
        ("dynamical_scale_choice", "-1, 1"),
        ("lhaid", [244600, 331700]),
        ("lhaid", "244600,331700"),
    ],
)
def test_lists(key, value):
    with pytest.raises(NotImplementedError, match=key):
        check_run_card({**BASE, key: value})


def test_beam_energies():
    """Beams of different energies are refused for scattering, not for a
    decay, whose beams enter nothing."""
    unequal = {**BASE, "ebeam1": 6500.0, "ebeam2": 2560.0}
    with pytest.raises(NotImplementedError, match="different energies"):
        check_run_card(unequal)
    check_run_card(unequal, incoming=1)


def test_incoming_count(tmp_path):
    """ninitial of SubProcesses/proc_characteristics, 2 without the file."""
    assert incoming_count(str(tmp_path)) == 2
    os.makedirs(tmp_path / "SubProcesses")
    with open(tmp_path / "SubProcesses" / "proc_characteristics", "w") as f:
        f.write("  ninitial = 1 \n  nexternal = 3\n")
    assert incoming_count(str(tmp_path)) == 1
