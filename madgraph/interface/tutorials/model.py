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
"""The `model` tutorial -- loading, inspecting and shaping a model.

Takes over the material the pre-2026 `lo` tutorial tacked on after `launch`:
importing a model, looking inside it, restrictions, `customize_model` and
`define`.
"""

from __future__ import absolute_import

import madgraph.interface.tutorials as tutorials
from madgraph.interface.tutorials.session import Step, Tutorial

P = 'MG7>'


tutorial = Tutorial(
    name='model',
    title='working with models',
    description='import, inspect, restrict and customise the physics model',
    order='sequence',
    see_also=('bsm', 'syntax', 'lo', 'checks'),
    steps=[

Step('tutorial', """
Every process you generate is read out of a model: the particles, the vertices,
the parameters and the couplings. MG7 ships with the Standard Model loaded, and
most of the time you never think about it -- until you need a different one, or
a different corner of the same one.

This tutorial covers loading a model, looking inside it, cutting it down, and
changing how it is treated.

Start by loading the SM explicitly:
%(p)s import model sm
""" % {'p': P},
     title='welcome',
     solution='import model sm'),

Step('import_model', """
A model is a UFO directory: Python files describing particles, vertices,
parameters and couplings, usually written by FeynRules. `import model NAME`
looks for `NAME` in `models/`, then downloads it if it is in the model
database. You can also give a path directly:

  import model /path/to/my_UFO_model

Two options worth knowing now:
  `--modelname`   keep the model's own particle names instead of translating
                  them to MG5 conventions
  `import model_v4 NAME`   for the old MG4 model format (deprecated)

Look at what you just loaded:
%(p)s display particles
""" % {'p': P},
     title='load a model',
     hint="`import model NAME`, where NAME is a directory under models/.",
     solution='display particles'),

Step('display', """
`display` is how you interrogate a model without generating anything:

  display particles            every particle, with its name and PDG code
  display particles t          everything about one particle: mass, width,
                               spin, colour, its antiparticle
  display interactions         every vertex
  display interactions t t~ g  the vertices involving exactly those particles
  display couplings            the coupling values
  display parameters           the parameters they are built from
  display multiparticles       the labels p, j, l+, l- and any you defined
  display modellist            every model available to download
  display coupling_order       the coupling orders the model defines

`display interactions` on its own is long; the filtered form is what you
usually want.

Now something less obvious. The SM you just loaded is *restricted*: several
parameters are fixed and some interactions removed, because carrying them adds
diagrams that contribute nothing. Load a different restriction and compare:

%(p)s import model sm-no_b_mass
""" % {'p': P},
     title='look inside the model',
     hint="`display particles`, `display interactions`, `display parameters`.",
     solution='import model sm-no_b_mass'),

Step('import_model', """
`MODEL-RESTRICTION` loads `restrict_RESTRICTION.dat` from the model directory.
With no restriction named you get `restrict_default.dat`, which is why the
plain `sm` already has the light-quark masses and most CKM mixing switched off.

The ones the SM ships with:
  sm                  the default restriction
  sm-full             nothing restricted at all
  sm-no_b_mass        massless b, for 5-flavour-scheme calculations
  sm-lepton_masses    keep the charged-lepton masses

A restriction file is just a param card: any parameter set to zero is removed
from the model along with the interactions that need it, and identical values
are merged into one parameter. That is why a restricted model generates fewer
diagrams and runs faster -- and why a result that surprises you is worth
re-checking against `-full`.

Next, how the model is *treated* rather than what is in it:

%(p)s set gauge Feynman
""" % {'p': P},
     title='restrictions',
     hint="Append `-RESTRICTION` to the model name.",
     solution='set gauge Feynman'),

Step('set', """
`set gauge` chooses the gauge for the non-QCD part, and reloads the model:

  unitary   the default: no goldstones, fewer diagrams, bigger cancellations
  Feynman   goldstones present, better numerical behaviour at high energy,
            and the only choice for loop processes
  axial     the parton-shower gauge, massless particles only
  FD        Feynman Diagram gauge, the extension of axial to massive
            particles (arXiv:2203.10440, 2405.01256)

Comparing unitary and Feynman is also the cheapest test that a model is
self-consistent -- that is exactly what `check gauge` does for you.

Two more model-level settings:
  set complex_mass_scheme True   widths in the propagator *and* in the
                                 couplings, for results that stay gauge
                                 invariant near a resonance
  set EWscheme                   which electroweak inputs are taken as
                                 independent

Now something you will use constantly:
%(p)s define v = w+ w- z a
""" % {'p': P},
     title='gauge and scheme',
     hint="`set gauge unitary|Feynman|axial|FD`.",
     solution='define v = w+ w- z a'),

Step('define', """
`define` makes a multiparticle label. `v` now stands for any electroweak vector
boson and works anywhere a particle name does:

  generate p p > v v

`p`, `j`, `l+`, `l-`, `vl` and `vl~` are predefined the same way, which is why
`p p > j j` means what it means. A definition can use `/` to exclude, as in
`define aUPC = a j / g`.

See them all, including the one you just made:
%(p)s display multiparticles
""" % {'p': P},
     title='multiparticle labels',
     hint="`define LABEL = particle particle ...`",
     solution='display multiparticles'),

Step('display', lambda interface: """
Three more things, worth knowing they exist:

  * **`customize_model`** opens an interactive menu of the switches a model
    exposes -- setting masses to zero, making mixing matrices diagonal,
    dropping interactions. `customize_model --save=NAME` keeps the result, and
    you reload it later with `import model MODEL-NAME`. This is the practical
    way to make a large model tractable.
  * **`add model OTHER`** merges a second model into the current one, which is
    how you bolt an extra sector onto the SM. It writes a combined model
    directory (`sm__OTHER`) and keeps your multiparticle definitions.
  * **`compute_widths PARTICLE`** computes widths from the model rather than
    trusting the card. Always pass `--output=FILE`: without it the result
    overwrites the param card inside the model directory. `--body_decay=N`
    controls how many decay channels are considered. It is a tree-level,
    narrow-width estimate -- honest for most things, not for a wide resonance.

And `save model PATH` writes the current model, restrictions and all, so a
collaborator gets exactly what you had.

%(see_also)s

Leave with `tutorial stop`.
""" % {'see_also': tutorials.where_next()},
     title='customise, merge, save'),

    ],
)
