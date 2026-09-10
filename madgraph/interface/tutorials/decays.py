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
"""The `decays` tutorial -- decay chains, MadSpin, MadWidth and the shower."""

from __future__ import absolute_import

import madgraph.interface.tutorials as tutorials
from madgraph.interface.tutorials.session import Step, Tutorial

P = 'MG7>'


tutorial = Tutorial(
    name='decays',
    title='decaying unstable particles',
    description='decay chains, MadSpin, widths, and which to use when',
    order='sequence',
    see_also=('syntax', 'madevent', 'model', 'bsm'),
    steps=[

Step('tutorial', """
Almost nothing you produce is stable. There are three places a decay can
happen, they do different things, and choosing wrongly is one of the easier
ways to get a wrong answer that looks right.

  1. **In the process line** -- a decay chain. Exact spin correlations,
     exact matrix element, and the diagram count multiplies with every step.
  2. **MadSpin**, after generation. Spin correlations kept, production
     diagrams untouched, so the cost barely grows with the cascade length.
  3. **The parton shower** (Pythia8). Cheapest, and it throws the spin
     correlations away.

All three assume the decaying particle is on shell, and all three depend on
widths you have to get right. Start with the first:

%(p)s generate p p > t t~, t > w+ b, t~ > w- b~
""" % {'p': P},
     title='welcome',
     solution='generate p p > t t~, t > w+ b, t~ > w- b~'),

Step('generate', """
Look at the diagram count against plain `p p > t t~`. That growth is the whole
argument: a decay chain computes the full matrix element for production and
decay together, so it is exact, and it gets expensive fast. Add
`w+ > l+ vl, w- > l- vl~` and watch it grow again.

Two things about the syntax that catch people (`tutorial syntax` has more):
  * identical particles are **all** decayed by one decay statement -- you do
    not write `t > w+ b` twice;
  * parentheses nest a sub-decay: `(t > w+ b, w+ > l+ vl)`.

And one thing about the physics: the cross section of a decay chain carries a
branching ratio built from the widths in your **param card**. Which is why the
next command matters more than it looks:

%(p)s compute_widths t --body_decay=2 --output=./my_widths.dat
""" % {'p': P},
     title='decay chains',
     hint="A comma opens the decay; each decaying particle gets one statement.",
     solution='compute_widths t --body_decay=2 --output=./my_widths.dat'),

Step('compute_widths', """
That is MadWidth: it finds the decay channels in the model and integrates them,
giving you a param card with widths that match the model rather than whatever
benchmark the card shipped with.

  --body_decay=N   consider up to N-body decays. An integer means "all
                   channels up to N-body"; a value below 1 means "stop when the
                   estimated error is under this"; and N.M combines the two.
  --min_br=X       skip channels estimated below X
  --output=FILE    **use this.** Without it, the result overwrites the param
                   card inside the model directory, silently changing every
                   later run with that model.
  --nlo            NLO widths, if the model supports it

It is tree-level and narrow-width, and it says so when it runs. For a state
whose width is a sizeable fraction of its mass, that approximation is the thing
you should be worrying about, not the last digit.

`decay_diagram PARTICLE` shows which channels exist without integrating them.

%(p)s history my_decays_session.dat
""" % {'p': P},
     title='computing widths',
     hint="`compute_widths PARTICLE --body_decay=2 --output=FILE`",
     solution='history my_decays_session.dat'),

Step('history', lambda interface: """
**MadSpin** is the run-time option, and for most studies it is the right one.
It takes undecayed events and decays them, keeping the spin correlations, by
reweighting against the decay matrix element. Cost is roughly independent of
how long the cascade is, which is exactly where decay chains hurt.

You use it through `Cards/madspin_card.dat` in a madevent output: put your
decays in it, and `launch` offers to edit it. A minimal card is a few
`decay` lines, e.g.

  set spinmode madspin
  decay t > w+ b, w+ > l+ vl
  decay t~ > w- b~, w- > l- vl~

`set spinmode none` turns the correlations off, which is only useful as a
cross-check to see how much they mattered.

**Which to use.**
  * decay chain -- when you need the exact off-shell/interference treatment,
    and the multiplicity stays manageable;
  * MadSpin -- long cascades, or many decay modes you want to vary without
    regenerating the production;
  * shower -- hadron and tau decays, where the correlations do not matter to
    your observable.

**The trap that catches everyone**, whichever you choose: a decayed cross
section is the production cross section times a branching ratio built from the
card's widths. Change a mass and forget the width, and you get a branching
ratio above 1 -- a decayed cross section larger than the undecayed one. Put
`DECAY <pdg> Auto` in the param card and the width is recomputed with the mass.
`tutorial exercises` and `tutorial madevent` both go through this.

%(see_also)s

Leave with `tutorial stop`.
""" % {'see_also': tutorials.where_next()},
     title='MadSpin, and choosing between the three'),

    ],
)
