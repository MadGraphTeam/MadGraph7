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
"""The `syntax` tutorial -- everything you can put in a process line.

This is a *sequenced* tutorial: most steps are triggered by `generate`, each
one teaching something different.  That is exactly what the old
command-name -> text lookup could not express.
"""

from __future__ import absolute_import

from madgraph.interface.tutorials.session import Step, Tutorial

P = 'MG7>'


tutorial = Tutorial(
    name='syntax',
    title='the process generation syntax',
    description='orders, interference, decay chains, s-channels, gauge traps, polarisation',
    order='sequence',
    steps=[

Step('tutorial', """
This tutorial is about the process line itself: everything you can put after
`generate`, one idea at a time. It picks up where `tutorial lo` leaves off,
and it stays there -- nothing here is output or run, these are process lines
only.

What that tutorial already showed, in three lines:

  generate p p > t t~     initial state, `>`, final state; the spaces matter
  p  j  l+  l-  vl        multiparticle labels (`display multiparticles`)
  no orders given         MG5 searches for coupling orders and applies its own

The last one is the habit worth breaking, because a search is not a statement
of physics. Say the orders yourself:
%(p)s generate p p > t t~ QED=2

Each step asks you to type a command. Type it and the next lesson appears. If
you are stuck, `hint` and `solution` print what is expected -- they never run
it for you. `skip` moves on, `repeat` prints the step again, and
`tutorial status` shows how far you have got. `tutorial help` lists the lot.
""" % {'p': P},
     title='welcome',
     hint="`tutorial lo` is the one that starts from nothing; this one starts "
          "from the process line.",
     solution='generate p p > t t~ QED=2'),

Step('generate', """
That `QED=2` constrains the *amplitude*: at most two QED vertices per diagram.
The whole family reads

  QED=0    at most 0 QED vertices     ('=' means '<=' -- this trips people up)
  QED==0   exactly 0 QED vertices
  QED<=2   at most 2
  QED>2    more than 2

and every one of them counts vertices in the diagram.

Now the one people get wrong.

A constraint with `^2` applies to the *squared* matrix element, not to the
amplitude, so it selects one term of |M|^2. For a process with both a QCD and
an EW amplitude, |M|^2 has three pieces:

  generate p p > j j QCD^2==4 QED^2==0     the pure QCD term
  generate p p > j j QCD^2==2 QED^2==2     the INTERFERENCE only
  generate p p > j j QCD^2==0 QED^2==4     the pure EW term

and the three add up to `generate p p > j j` with no constraints at all.
There is also a shorthand: a negative value, `COUP^2==-I`, asks for the
N^(-I+1)LO term of that expansion.

Generate the interference term on its own:
%(p)s generate p p > j j QCD^2==2 QED^2==2

Three things to know before you use this in anger:
 * a negative order constraint may be given on ONE coupling only, and either
   on squared orders or on amplitude orders -- never both;
 * interference *with a decay* (a 1 -> N process carrying squared orders) is
   not fully validated; the suggested cross-check is to regenerate it under
   `set group_subprocesses True` and compare;
 * the `check` command does not accept the `^2` syntax, so do not reach for
   `check` to validate an interference process.
""" % {'p': P},
     title='coupling orders: amplitude and squared',
     hint="'^2' makes the constraint apply to the squared matrix element.",
     solution='generate p p > j j QCD^2==2 QED^2==2'),

Step('generate', """
A comma opens a decay chain. Everything after it decays a particle of the
process before it, and parentheses nest.

%(p)s generate p p > t t~, (t > w+ b, w+ > l+ vl), t~ > w- b~

Two things to keep in mind:
 * identical particles are ALL decayed -- you cannot decay one top and leave
   the other alone by writing the decay once;
 * a decay chain is neither on-shell nor a branching ratio. It is the full
   matrix element, production times propagator times decay, with the spin
   correlations kept and the resonance left off shell -- spread over its
   Breit-Wigner out to `bwcutoff` widths from the pole. What it drops is the
   diagrams that do not go through that resonance.

That second point is why the width catches people. The width in the
*param_card* sits in the propagator denominator; the decay rate comes back out
of the matrix element itself, and nothing divides one by the other. There is no
branching ratio to normalise, so nothing keeps the effective fraction under 1:
change a mass, leave the width alone, and the decayed cross section can come
out *larger* than the undecayed one. (The `exercises` tutorial makes that
happen on purpose.)

MadSpin is the run-time alternative: it decays events after generation and
keeps spin correlations, without multiplying the number of diagrams.

This is also the *safe* way to ask for a resonance. Production and decay are
each a complete set of diagrams, so each is gauge invariant on its own, and
what you dropped is stated plainly: the diagrams that do not go through the
resonance. The next two lessons do a similar-looking job by reaching inside a
single amplitude, and that is where it gets delicate.
""" % {'p': P},
     title='decay chains',
     hint="Use ',' to open the decay, and parentheses to nest a second one.",
     solution='generate p p > t t~, (t > w+ b, w+ > l+ vl), t~ > w- b~'),

Step('generate', """
Now the operators that reach inside one amplitude and keep part of it.

A second `>` names a required s-channel: only diagrams going through that
particle are kept.

%(p)s generate p p > w+ > l+ vl

Alternative s-channels are separated by `|`, which is how you keep an
interfering pair together -- the Z/photon pair being the everyday case:

  generate p p > z | a > e+ e-

Several particles on one side of the `|` are required together, so a
two-per-side form asks for one pair or the other. They have to be distinct
names: `> z z >` is refused rather than quietly read as `> z >`.

**Read this before you use it.** A single Feynman diagram is not an
observable. Only the complete set for an amplitude is gauge invariant, and the
cancellations between diagrams can be enormous. Keep some and drop the rest
and the answer may depend on the gauge it was computed in -- which means it is
not a prediction of anything.

Sometimes the subset you kept is a complete gauge-invariant set by itself and
all is well: photon versus Z exchange in `u u~ > e+ e-` is the textbook case.
Sometimes it is not, and nothing in the output tells you which you are in.
`check gauge` does, and the next lesson runs it.
""" % {'p': P},
     title='required s-channels',
     hint="Put the intermediate particle between two '>'.",
     solution='generate p p > w+ > l+ vl'),

Step('generate', """
The mirror image: excluding things. Three operators, and the difference
between them is exactly the gauge question from the last lesson.

  $   exclude the ON-SHELL contribution of an s-channel particle. The diagram
      is KEPT and only the resonance peak is subtracted, so this one is safe
      by construction.
  $$  forbid that s-channel entirely -- the diagram is dropped.
  /   forbid a particle ANYWHERE in the diagram, internal or external.

%(p)s generate p p > e+ e- / a

Then try `generate p p > e+ e- $ a` and `generate p p > e+ e- $$ a` and
compare the diagram counts -- the contrast is the lesson.

`$$` and `/` delete diagrams, so they carry the same warning as `> A >` above.
Worth running once, so that you have watched it happen:

  check gauge e+ e- > w+ w-           passes
  check gauge e+ e- > w+ w- $$ a      FAILS
  check gauge e+ e- > a > w+ w-       FAILS

Dropping the photon from W pair production leaves a matrix element several
times the full one, and destroys the cancellation that keeps it from growing
with energy. `check gauge` computes the same thing in four gauges and compares
them; `tutorial checks` is the one that goes through it. (`check` needs a model
imported, which `generate` does for you but `check` does not.)
""" % {'p': P},
     title='excluding particles and s-channels',
     hint="'/ a' forbids the photon everywhere; '$ a' only removes it on shell.",
     solution='generate p p > e+ e- / a'),

Step('generate', """
`add process` puts a second process into the same output. It takes exactly the
same syntax as `generate`, and it accumulates instead of replacing.

%(p)s add process p p > w+ j, w+ > l+ vl

`display processes` lists everything defined so far, `display diagrams` draws
all of it, and everything you have added goes into the next `output`.
""" % {'p': P},
     title='several processes at once',
     hint="'add process' takes the same syntax as 'generate'.",
     solution='add process p p > w+ j, w+ > l+ vl'),

Step('add', """
Polarisation. `{X}` after a (multi)particle fixes its helicity: `{L}` and
`{R}` for left and right, `{T}` transverse, `{0}` longitudinal, `{A}` auxiliary.
It works on external particles, massless or massive, and on massive internal
particles before a decay chain.

%(p)s generate p p > z{0} z{T}, z > e+ e-

The process line is only half of it. A polarisation is defined in a frame, and
which frame is a run-card setting -- `me_frame` -- so it is chosen at `launch`,
not here. `tutorial madevent` reaches the cards and goes through it.
""" % {'p': P},
     title='polarisation',
     hint="Append '{0}' or '{T}' to a particle name.",
     solution='generate p p > z{0} z{T}, z > e+ e-'),

Step('generate', """
That is the syntax tour. A few things worth remembering:

 * `define` makes your own multiparticle label, e.g.
   `define v = w+ w- z a`, usable anywhere `p` or `j` is;
 * `display diagrams` draws what you generated before you commit to an
   output -- a cheap habit;
 * `help generate` prints the whole grammar in one screen.

Where to go next:
 * `tutorial lo`         take a process all the way to events
 * `tutorial nlo`        the same process line at next-to-leading order
 * `tutorial exercises`  practise, with the answers checked
 * `tutorial list`       everything on offer

Leave tutorial mode with `tutorial stop`.
""",
     title='wrap-up'),

    ],
)
