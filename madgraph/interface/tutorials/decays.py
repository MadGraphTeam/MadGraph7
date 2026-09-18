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
from madgraph.interface.tutorials.session import (Step, Tutorial,
                                                  counts_line)

P = 'MG7>'

tutorial = Tutorial(
    name='decays',
    title='decaying unstable particles',
    description='decay chains, MadSpin, widths, and which to use when',
    order='sequence',
    see_also=('syntax', 'madevent', 'model', 'bsm'),
    steps=[

Step('tutorial', """
Most massive particles are unstable and decay. There are three places where
the code can simulate a decay, they do different things, and choosing wrongly
is one of the easier ways to get a wrong answer that looks right.

  1. **In the process line** -- a decay chain. Exact spin correlations, the
     exact matrix element, and decay products you can put cuts on.
  2. **MadSpin**, after generation. Spin correlations kept, production
     diagrams untouched, so the cost barely grows with the cascade length.
  3. **The parton shower** (Pythia8). Cheapest, and it throws the spin
     correlations away.

All three rely on the narrow-width approximation to be valid -- the width small
against the mass -- and all three depend on widths you have to get right.
Start with the first:

%(p)s generate p p > t t~, t > w+ b, t~ > w- b~
""" % {'p': P},
     title='welcome',
     solution='generate p p > t t~, t > w+ b, t~ > w- b~'),

Step('generate', lambda interface: """
%(counts)sMG7 generates the production and the decays separately and reports
their diagrams added up. `output` stitches them back together -- one full
diagram per production diagram here, since each decay has only one, so as
many as plain `p p > t t~` has. What a decay chain buys is the full matrix
element: production, propagator and decay together, with each top's spin
correlations carried into its decay products.

Two things about the syntax that catch people (`tutorial syntax` has more):
  * identical particles are **all** decayed by one decay statement -- you do
    not write `t > w+ b` twice;
  * parentheses nest a sub-decay: `(t > w+ b, w+ > l+ vl)`.

And one about the physics: this syntax uses no branching ratio. The decay is
inside the matrix element, so the b quarks and the W bosons are genuine
final-state particles you can put cuts on -- which a branching ratio applied
afterwards would not allow.

What it does rely on is the width in your **param card**: it sits in the
resonance propagator and is checked against nothing. Rather than typing a
number, put `DECAY 6 Auto` there. MG7 then computes the top width from the
model when the run starts, and it follows the mass whenever you change it.

%(p)s history my_decays_session.dat
""" % {'p': P, 'counts': counts_line(interface)},
     title='decay chains',
     hint="A comma opens the decay; each decaying particle gets one "
          "statement.",
     solution='history my_decays_session.dat'),

Step('history', lambda interface: """
That file replays the session -- `import command my_decays_session.dat`, or
`./bin/madgraph my_decays_session.dat` from a shell.

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

**The trap that catches everyone**, whichever you choose: the card's width
sets the size of the resonance propagator and nothing normalises it against
the decay the model actually computes. Change a mass and forget the width and
the effective fraction goes above 1 -- a decayed cross section larger than the
undecayed one, which nothing in the machinery is there to prevent. That is what
`DECAY <pdg> Auto` is for.
`tutorial exercises` and `tutorial madevent` both go through this.

%(see_also)s

Leave with `tutorial stop`.
""" % {'see_also': tutorials.where_next()},
     title='MadSpin, and choosing between the three'),

    ],
)
