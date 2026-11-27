"""The pole check of the setup of the NLO runs (pole_check of
madgraph/mg7/nlo/core.py): the poles of MadLoop from the Fortran provider
cancel those of the I operator (CSInsertion) at Born points of the first
channel, as BinothLHA.f of MadFKS checks them, for QCD and QED corrections,
massive partons, decays and 2 > 1 Borns; the NLO orders it leaves out
(tagged photons, orders above max_nlo); and the failure of a wrong setup (the
light flavors, the top mass).

The aMC@NLO outputs are those of MADSPACE_FKS_OUTPUTS.
"""

import copy
import json
import os

import pytest
from fks_testing import fks_output
from madgraph.mg7.nlo.core import (
    light_flavors,
    pole_check,
    read_param_card,
    read_run_card,
)
from madgraph.mg7.nlo.provider import FortranProvider

import madspace as ms

OUTPUTS = os.environ.get("MADSPACE_FKS_OUTPUTS")
pytestmark = pytest.mark.skipif(
    not OUTPUTS, reason="MADSPACE_FKS_OUTPUTS names no aMC@NLO outputs"
)
DATA = os.path.join(os.path.dirname(__file__), "test_data", "fks_cross_section")
THRESHOLD = 1e-5


def setups(name):
    """The run card, param card and light flavors of a reference, and the
    providers of its subprocesses."""
    with open(os.path.join(DATA, f"{name}.json")) as f:
        data = json.load(f)
    output = fks_output(data["output"])
    run_card = read_run_card(output, data["run_card"])
    param_card = data["param_card"] or read_param_card(
        os.path.join(output, "Cards", "param_card.dat")
    )
    n_f, n_l = light_flavors(output)
    root = os.path.join(output, "SubProcesses")
    providers = [
        FortranProvider(os.path.join(root, d), rebuild=False)
        for d in sorted(os.listdir(root))
        if d.startswith("P")
    ]
    return run_card, param_card, n_f, n_l, providers


def check(name, threshold=THRESHOLD, n_f=None, param_card=None):
    """The points of the check of every subprocess of a reference."""
    run_card, param, flavors, n_l, providers = setups(name)
    context = ms.Context(ms.cpu_device())
    points = []
    for provider in providers:
        with provider:
            points.append(
                pole_check(
                    provider,
                    run_card,
                    param_card or param,
                    flavors if n_f is None else n_f,
                    n_l,
                    context,
                    threshold,
                )
            )
    return points


@pytest.mark.parametrize(
    "name", ["epem_jj", "pp_ttx", "t_bwg", "pp_epem_qed", "pp_wp", "epem_ttx_isr"]
)
def test_poles_cancel(name):
    """The check compares points of every subprocess and passes: QCD and QED
    corrections, massive partons, a decay with a gluon, a 2 > 1 Born and
    dressed leptons."""
    points = check(name)
    assert all(n > 0 for n in points)


def test_left_out():
    """The QED corrections of tagged photons have subtraction terms that
    CSInsertion does not have, so the check compares nothing, as for a
    negative threshold. The QED corrections of an output of QCD corrections,
    above max_nlo, are left out of the check."""
    assert check("pp_aa_tagged") == [0] * 6
    assert check("pp_ttx", threshold=-1.0) == [0] * 3
    assert all(n > 0 for n in check("epem_ttx_sdk"))


def test_wrong_setup():
    """With five light flavors for the four of the output, the single pole of
    g g > t t~ misses 2/3 of a Born, and with a top mass other than that of
    MadLoop the logarithms of the mass differ: the check stops."""
    with pytest.raises(RuntimeError, match="do not cancel"):
        check("pp_ttx", n_f=5.0)
    run_card, param, *_ = setups("pp_ttx")
    shifted = copy.deepcopy(param)
    shifted["mass"]["6"] = 172.0
    with pytest.raises(RuntimeError, match="do not cancel"):
        check("pp_ttx", param_card=shifted)
