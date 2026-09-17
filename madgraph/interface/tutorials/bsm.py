################################################################################
#
# Copyright (c) 2026 The MadGraph5_aMC@NLO Development team and Contributors
#
# This file is a part of the MadGraph5_aMC@NLO project, an application which
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph5_aMC@NLO license which should accompany this
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################
"""The `bsm` tutorial -- generating and validating physics beyond the SM.

Uses MSSM_SLHA2, which ships with MG7, so the tutorial works without a network
connection.
"""

from __future__ import absolute_import

import madgraph.interface.tutorials as tutorials
from madgraph.interface.tutorials.session import (Step, Tutorial, check_line,
                                                  counts_line, model_line)

P = 'MG7>'


def _orders(interface=None):
    """The coupling orders the loaded model declares, as the reader just saw
    them.  Read back rather than asserted: which orders a model defines is the
    whole subject of the step, so it has to be the model's own answer."""

    model = getattr(interface, '_curr_model', None)
    try:
        orders = sorted(model.get('coupling_orders'))
    except Exception:
        return ''
    if not orders:
        return 'This model declares no coupling order at all.\n'
    return ('This model declares only %s, and nothing else.\n'
            % ' and '.join('**%s**' % order for order in orders))


tutorial = Tutorial(
    name='bsm',
    title='beyond the Standard Model',
    description='new models, new states, their widths, and validating them',
    order='sequence',
    see_also=('model', 'syntax', 'decays', 'checks'),
    steps=[

Step('tutorial', """
Nothing in MG7 knows about the Standard Model specifically. Hand it a different
model and everything -- diagram generation, matrix elements, events -- works
the same way. This tutorial is about the parts that are *not* the same: finding
your way around an unfamiliar model, getting the widths of new states right,
and convincing yourself the model is sound before you trust a number.

Where BSM models come from:
  * **FeynRules** turns a Lagrangian into a UFO model. If your model does not
    exist yet, this is how it starts.
  * **The model database** -- `display modellist` shows what MG7 can download,
    and `import model NAME` fetches it.
  * **A colleague**, as a UFO directory you point `import model` at.

We will use one that ships with MG7:
%(p)s import model MSSM_SLHA2
""" % {'p': P},
     title='welcome',
     solution='import model MSSM_SLHA2'),

Step('import_model', lambda interface: """
%(model)sThat is the MSSM. `display particles` now lists the superpartners
alongside the SM content -- `go` (gluino), `ul`/`ur`/`t1`/`t2` and friends
(squarks),
`n1`..`n4` (neutralinos), `x1+`/`x2+` (charginos), and an extended Higgs sector
`h01`, `h2`, `h3`, `h+`.

Before generating anything, it is worth asking how this model labels its BSM
content:

%(p)s display coupling_order
""" % {'p': P, 'model': model_line(interface)},
     title='load a BSM model',
     hint="`import model NAME` -- MSSM_SLHA2 ships with MG7.",
     solution='display coupling_order'),

Step('display', lambda interface: """
%(orders)sThe orders a model declares are the first thing to look at in one
you did not write: they are how you ask for the physics you want. Models
split into two camps:

  * **No dedicated order**, like this one. New physics is identified by the
    *particles* -- you get BSM by putting `go` or `n1` in the process, and the
    coupling orders behave exactly as in the SM.
  * **A dedicated order**, like `NP` in most EFT and simplified models. There
    the new physics is a coupling, present in the same final states as the SM,
    and you select it with orders:

      generate p p > e+ e- NP==0            SM only
      generate p p > e+ e- NP==1            BSM amplitude only
      generate p p > e+ e- NP^2==1          the SM-BSM INTERFERENCE
      generate p p > e+ e- NP^2==2          the pure BSM squared term

    For a dimension-six operator the interference term is the one that scales
    as 1/Lambda^2, so it is usually the one you want -- and it is the one
    people forget to ask for. `tutorial syntax` covers the `^2` syntax.

With this model the BSM is in the particles, so:
%(p)s generate p p > go go
""" % {'p': P, 'orders': _orders(interface)},
     title='how the model labels new physics',
     hint="`display coupling_order` lists the orders a model defines.",
     solution='generate p p > go go'),

Step('generate', lambda interface: """
%(counts)sGluino pair production, out of the model's own vertices -- you did
not have to tell MG7 anything about SUSY.

New states usually decay, and there are two ways to handle that:
  * a **decay chain** in the process line,
    `generate p p > go go, go > g n1` -- exact spin correlations, but the
    number of diagrams multiplies with each cascade step;
  * **MadSpin** at run time, which keeps spin correlations without touching
    the production diagrams. For long cascades this is the practical choice.

Neither forces the new state on shell -- both keep it on its Breit-Wigner --
but both keep only the diagrams that go through it, and both are only as good
as the widths in the card. Which is the thing that goes wrong most often in
BSM studies, so:

%(p)s compute_widths go --body_decay=2 --output=./mssm_widths.dat
""" % {'p': P, 'counts': counts_line(interface)},
     title='generate a BSM signal',
     hint="Just name the new particles; the model supplies the vertices.",
     solution='compute_widths go --body_decay=2 --output=./mssm_widths.dat'),

Step('compute_widths', """
That computed the gluino width from the model and wrote a param card with it.

Three things about widths in BSM models:

  * **The card's widths are not automatically right.** A UFO model ships with
    some benchmark point. Change a mass -- which is the whole point of a scan
    -- and every width that mass feeds is now wrong. The symptom is a decayed
    cross section larger than the undecayed one -- nothing forms a branching
    ratio, so nothing caps the effective fraction at 1. The fix is
    `DECAY <pdg> Auto` in the param card, which the scan machinery recomputes
    at every point.
  * **`compute_widths` is tree-level and narrow-width.** Honest for a narrow
    state, not for a wide one, and it says so when you run it. For a resonance
    with a width comparable to its mass, a hand-set width and a hard look at
    the propagator treatment beat an automatic number.
  * **`--output=` is not optional in practice.** Without it the result
    overwrites the param card inside the model directory, silently changing
    every later run that uses that model.

Now the step people skip. You have a model you did not write, and you are about
to trust it. Test it:

%(p)s check permutation p p > go go
""" % {'p': P},
     title='widths for new states',
     hint="`compute_widths PARTICLE --body_decay=2 --output=FILE`",
     solution='check permutation p p > go go'),

Step('check', lambda interface: """
%(verdict)sThat regenerated the process with the external legs permuted and
checked the matrix element came out the same. It is a real test of the model
and of the machinery, and it costs a couple of seconds.

The family:
  check permutation   relabelling the legs must not change |M|^2
  check gauge         unitary and Feynman gauges must agree -- the sharpest
                      test that a model's couplings are consistent
  check lorentz       the amplitude must be frame independent
  check brs           Ward identities, for processes with a massless gauge
                      boson leg
  check cms           complex-mass-scheme consistency near a resonance
  check full          permutation, brs, gauge and lorentz together

A new UFO model that fails `check gauge` has a bug in its Lagrangian or its
conversion, and no amount of careful running will fix the answer.

%(p)s history my_bsm_session.dat
""" % {'p': P, 'verdict': check_line(interface)},
     title='validate the model',
     hint="`check permutation PROCESS`",
     solution='history my_bsm_session.dat'),

Step('history', lambda interface: """
That file replays the session -- `import command my_bsm_session.dat`, or
`./bin/madgraph my_bsm_session.dat` from a shell.

Two last things that bite in BSM work.

**Big models are slow.** A full BSM model can have hundreds of particles, and
generating with all of them is painful. `customize_model` opens the switches
the model exposes -- zero masses, diagonal mixing, dropped sectors -- and
`customize_model --save=NAME` keeps the result, and you reload it later with
`import model MODEL-NAME`. Restricting to the sector you actually study is
normal practice, not a shortcut.

**EFTs need care that models do not.**
  * Order counting is the physics. A dimension-six analysis usually wants the
    interference term, `NP^2==1`, not the squared one -- mixing them up changes
    the answer by more than any systematic.
  * EFT amplitudes grow with energy by construction. A cross section dominated
    by events above your cutoff is telling you the expansion has broken down,
    not that you found something.
  * `set complex_mass_scheme True` matters as soon as a wide resonance is
    involved.

%(see_also)s

Leave with `tutorial stop`.
""" % {'see_also': tutorials.where_next()},
     title='big models and EFT pitfalls'),

    ],
)
