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
"""The `bsm` tutorial -- physics beyond the SM, through an effective theory.

Uses SMEFTatNLO, the dimension-six Standard Model EFT at NLO in QCD.  It is in
the MG5 model database, so the first `import model` downloads it.  It teaches
the parts of BSM work that are not the SM's: a dedicated coupling order, an
order hierarchy under which the default search keeps the wrong diagrams, a
convention for that order which changes the process line, and the linear and
quadratic pieces of an EFT cross section.
"""

from __future__ import absolute_import

import madgraph.interface.tutorials as tutorials
from madgraph.interface.tutorials.session import (Step, Tutorial,
                                                  applied_orders, counts_line,
                                                  model_line)

P = 'MG7>'
MODEL = 'SMEFTatNLO-NLO'
DEFAULT = 'generate p p > t t~'
ONE_INSERTION = 'generate p p > t t~ NP<=2'
LINEAR = 'generate p p > t t~ NP^2==2'
RESTRICTION = 'top'
CUSTOMIZE = 'customize_model --save=%s' % RESTRICTION
RELOAD = 'import model SMEFTatNLO-%s' % RESTRICTION

# shown at the customize_model question, where the reader types it
CUSTOMIZE_AT_QUESTION = """
Switch off what a top pair never sees -- the operators with leptons in them,
two whole blocks of the param card:

  set DIM64F2L all 0         two quarks, two leptons
  set DIM64F4L all 0         four leptons

then `done`. `set NAME 0` does it for one coefficient, and `set NAME = OTHER`
ties two together. Leave `DIM6` alone as a block: it also holds `Lambda`, and
a zero scale is a division by zero.

The question starts from `-NLO`: the operators it switches off are already
listed at 0, and `set NAME free` switches one of them back on.
"""


def _orders(interface=None):
    """The coupling orders the loaded model declares, and their hierarchy.

    Read back rather than asserted: which orders a model defines, and what
    each weighs, is the whole subject of the step.
    """

    model = getattr(interface, '_curr_model', None)
    try:
        orders = sorted(model.get('coupling_orders'))
        hierarchy = model.get('order_hierarchy') or {}
    except Exception:
        return ''
    if not orders:
        return 'This model declares no coupling order at all.\n'
    names = ['**%s**' % order for order in orders]
    listed = (', '.join(names[:-1]) + ' and ' + names[-1]
              if len(names) > 1 else names[0])
    text = 'This model declares %s' % listed
    if all(order in hierarchy for order in orders):
        text += (',\nweighed %s in its hierarchy'
                 % ', '.join('%s %s' % (order, hierarchy[order])
                             for order in sorted(orders,
                                                 key=lambda o: hierarchy[o])))
    return text + '.\n'


def _saved_restriction(interface=None):
    """Where customize_model wrote the restriction card, if it did."""

    import os

    try:
        path = os.path.join(interface._curr_model.get('modelpath'),
                            'restrict_%s.dat' % RESTRICTION)
    except Exception:
        return ''
    if not os.path.isfile(path):
        return ''
    return 'The restriction is saved as\n`%s`.\n' % path


def _search_result(interface=None):
    """The orders MG5 put on a process given none, as it printed them."""

    found = [orders for orders in applied_orders(interface) if orders]
    if not found:
        return 'no coupling-order constraint at all'
    return ' and '.join(
        '**%s**' % ' '.join('%s<=%s' % (name, value)
                            for name, value in sorted(orders.items()))
        for orders in found[:2])


tutorial = Tutorial(
    name='bsm',
    title='beyond the Standard Model',
    description='an EFT model: its coupling order, conventions, and checks',
    order='sequence',
    see_also=('model', 'syntax', 'decays', 'checks'),
    steps=[

Step('tutorial', """
Nothing in MG7 knows about the Standard Model specifically. Hand it a different
model and everything -- diagram generation, matrix elements, events -- works
the same way. This tutorial is about the parts that are *not* the same, on the
most common kind of BSM model today: an effective field theory.

Where BSM models come from:
  * **FeynRules** turns a Lagrangian into a UFO model. If your model does not
    exist yet, this is how it starts.
  * **The model database** -- `display modellist` shows what MG7 can download,
    and `import model NAME` fetches it.
  * **A colleague**, as a UFO directory you point `import model` at.

We will use SMEFTatNLO, the dimension-six Standard Model EFT at NLO in QCD,
from the database -- the first import downloads it. `-NLO` is the restriction
the model provides for NLO QCD work:
%(p)s import model %(model)s
""" % {'p': P, 'model': MODEL},
     title='welcome',
     hint="`import model NAME` also downloads a model from the database.",
     solution='import model %s' % MODEL),

Step('import_model', lambda interface: """
%(model)sNot one new particle: that is the Standard Model and its ghosts. The
new physics is in the interactions -- dimension-six operators, each entering
through a Wilson coefficient over Lambda^2. The coefficients are parameters of
the model, in the param card's `DIM6`, `DIM62F` and `DIM64F` blocks, and the
new vertices are what they multiply.

Before generating anything, see how the model labels that new physics:
%(p)s display coupling_order
""" % {'p': P, 'model': model_line(interface)},
     title='load an EFT model',
     solution='display coupling_order'),

Step('display', lambda interface: """
%(orders)sThis is the camp of models with a dedicated order: `NP` marks the new
physics, so you ask for it in the process line rather than by naming new
particles. Two things about it are this model's own, and both matter:

  * **NP counts powers of 1/Lambda.** Every operator comes with 1/Lambda^2, so
    one operator insertion is `NP=2`, not 1 -- and `NP<=1` quietly removes
    every EFT vertex. Other models count an insertion as `NP=1`; a model's
    documentation says which.
  * **NP is the cheapest order.** When you give no orders, MG5 keeps the
    cheapest diagrams it can find, weighing each order by that hierarchy.

See what that does to the simplest process there is:
%(p)s %(default)s
""" % {'p': P, 'default': DEFAULT, 'orders': _orders(interface)},
     title='how the model labels new physics',
     hint="`generate p p > t t~`, with no orders on purpose.",
     solution=DEFAULT),

Step('generate', lambda interface: """
%(counts)sMG5 settled on %(search)s -- and not one of those diagrams is the
Standard Model. A top pair from QCD weighs 4, two QCD vertices at 2 each; one
operator insertion weighs 2. The search took the lowest weight that produces
anything, which is the EFT on its own: a pure 1/Lambda^4 piece with nothing to
interfere with, and a prediction of nothing.

That is the trap `tutorial syntax` warns about, and in a model like this one it
is the default. Say the orders yourself -- the Standard Model plus at most one
operator insertion:
%(p)s %(one)s
""" % {'p': P, 'one': ONE_INSERTION, 'counts': counts_line(interface),
       'search': _search_result(interface)},
     title='the default search, in an EFT',
     hint="One operator insertion is `NP=2` in this model.",
     solution=ONE_INSERTION),

Step('generate', lambda interface: """
%(counts)sThe Standard Model diagrams, and every single-insertion one beside
them. Squared, that amplitude has three pieces:

  SM x SM      the Standard Model                     NP^2==0
  SM x EFT     linear in the coefficients, 1/Lambda^2   NP^2==2
  EFT x EFT    quadratic, 1/Lambda^4                  NP^2==4

The linear term is the one a dimension-six analysis is built on. The quadratic
one is formally of the same order as the dimension-eight operators you did not
include, so whether to keep it is a choice to make and to state. Ask for the
linear term on its own:
%(p)s %(linear)s
""" % {'p': P, 'linear': LINEAR, 'counts': counts_line(interface)},
     title='linear and quadratic',
     hint="`^2` constrains the squared matrix element: `NP^2==2` is the "
          "interference.",
     solution=LINEAR),

Step('generate', lambda interface: """
%(counts)sThe same diagrams as a moment ago: as in `tutorial syntax`, a `^2`
constraint picks which products survive the squaring, not which diagrams
exist. It is also where the model's convention bites a second time. In a
model counting an insertion as `NP=1`, this same term is `NP^2==1` -- which
here asks for a term that does not exist, and generates nothing.

A model this size carries far more than any one study needs: most of those
operators never touch a top pair. `customize_model` builds a restriction of
your own -- coefficients fixed to zero, to one, or tied together -- and
`--save` keeps it, so it loads like the ones the model ships:
%(p)s %(customize)s
""" % {'p': P, 'customize': CUSTOMIZE, 'counts': counts_line(interface)},
     title='the interference on its own',
     # current while customize_model asks its question
     question_hint=CUSTOMIZE_AT_QUESTION,
     hint="`customize_model --save=NAME` opens a question; `done` closes it.",
     solution=CUSTOMIZE),

Step('customize_model', lambda interface: """
%(model)s%(saved)sThat is now the model loaded here, and it started from where
you were: the operators `-NLO` switches off were already in the question at 0,
and your two blocks went on top. The lepton operators are gone, and so are
the vertices they multiplied: a smaller model is a faster generation, and a
shorter list of coefficients to keep track of.

In a later session, it loads by its name:
%(p)s %(reload)s
""" % {'p': P, 'reload': RELOAD, 'model': model_line(interface),
       'saved': _saved_restriction(interface)},
     title='a restriction of your own',
     hint="`import model MODEL-NAME` loads restrict_NAME.dat.",
     solution=RELOAD),

Step('import_model', lambda interface: """
%(model)sThe same restriction, loaded by name -- the way you will use it from
now on. `tutorial checks` is the one to take before you trust a model you did
not write.

Three things that bite in EFT work:
  * **The convention is the physics.** Which `NP` value is one insertion, and
    so which squared order is the linear term, is the model's choice. Read its
    documentation before you write a process line.
  * **EFT amplitudes grow with energy**, by construction. A cross section
    dominated by events above your cutoff says the expansion has broken down,
    not that you found something.
  * **Widths move with the coefficients.** `ctW` changes t > W b, so the top
    width depends on it. With `DECAY 6 Auto` in the param card it is computed
    for the coefficients you set; with a number, it is not.

%(see_also)s

Leave with `tutorial stop`.
""" % {'model': model_line(interface), 'see_also': tutorials.where_next()},
     title='your restriction, and EFT pitfalls'),

    ],
)
