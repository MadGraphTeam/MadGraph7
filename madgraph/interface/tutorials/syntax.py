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

from madgraph.interface.tutorials.session import (Step, Tutorial,
                                                  describe_applied_orders)

P = 'MG7>'


tutorial = Tutorial(
    name='syntax',
    title='the process generation syntax',
    description='orders, interference, s-channels, decay chains, polarisation, NLO',
    order='sequence',
    steps=[

Step('tutorial', """
This tutorial walks through the syntax of the `generate` command: everything
you can put in a process line, one idea at a time.

Each step asks you to type a command. Type it and the next lesson appears. If
you are stuck, `hint` and `solution` print what is expected -- they never run
it for you. `skip` moves on, `repeat` prints the step again, and
`tutorial status` shows how far you have got. `tutorial help` lists the lot.

Let's start with the simplest possible process:
%(p)s generate p p > t t~

`p` is a multiparticle label -- a shorthand for a set of particles. Type
`display multiparticles` at any point to see what `p`, `j`, `l+` and `l-`
stand for.
""" % {'p': P},
     title='welcome',
     solution='generate p p > t t~'),

Step('generate', lambda interface: """
Read what MG5 printed back. You gave it no coupling orders, so it chose some
for you.

%(orders)s

Either way it is the first thing to make explicit when a result surprises you.

Coupling orders are constraints on the *amplitude*:
  QED=0    at most 0 QED vertices     ('=' means '<=' -- this trips people up)
  QED==0   exactly 0 QED vertices
  QED<=2   at most 2
  QED>2    more than 2

Ask for the electroweak diagrams back:
%(p)s generate p p > t t~ QED=2

Compare the diagram count with what you got a moment ago.
""" % {'p': P, 'orders': describe_applied_orders(interface)},
     title='coupling orders',
     hint="Orders go at the end of the process line, after the final state.",
     solution='generate p p > t t~ QED=2'),

Step('generate', """
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
     title='interference only',
     hint="'^2' makes the constraint apply to the squared matrix element.",
     solution='generate p p > j j QCD^2==2 QED^2==2'),

Step('generate', """
A second `>` names a required s-channel: only diagrams going through that
particle are kept.

%(p)s generate p p > w+ > l+ vl

Alternative s-channels are separated by `|`, which is how you keep an
interfering pair together:

  generate b b~ > W+ W- | H+ H- > ta+ vt ta- vt~

Try the required-s-channel form:
%(p)s generate p p > w+ > l+ vl
""" % {'p': P},
     title='required s-channels',
     hint="Put the intermediate particle between two '>'.",
     solution='generate p p > w+ > l+ vl'),

Step('generate', """
The mirror image: excluding things. Three different operators, and the
difference matters.

  $   exclude the ON-SHELL contribution of an s-channel particle; the diagram
      is kept but the resonance is removed. Gauge-safe.
  $$  forbid that s-channel entirely -- the diagram is dropped. This breaks
      gauge invariance in general, so use it only when you know why.
  /   forbid a particle ANYWHERE in the diagram, internal or external.

%(p)s generate p p > e+ e- / a

Then try `generate p p > e+ e- $ a` and `generate p p > e+ e- $$ a` and
compare the diagram counts -- the contrast is the lesson.
""" % {'p': P},
     title='excluding particles and s-channels',
     hint="'/ a' forbids the photon everywhere; '$ a' only removes it on shell.",
     solution='generate p p > e+ e- / a'),

Step('generate', """
A comma opens a decay chain. Everything after it decays a particle of the
process before it, and parentheses nest.

%(p)s generate p p > t t~, (t > w+ b, w+ > l+ vl), t~ > w- b~

Two things to keep in mind:
 * identical particles are ALL decayed -- you cannot decay one top and leave
   the other alone by writing the decay once;
 * a decay chain is an on-shell approximation. Its cross section carries a
   branching ratio built from the widths in the *param_card*, so changing a
   mass without recomputing the width silently corrupts the answer. (See the
   `exercises` tutorial for what that looks like when it goes wrong.)

MadSpin is the run-time alternative: it decays events after generation and
keeps spin correlations, without multiplying the number of diagrams.
""" % {'p': P},
     title='decay chains',
     hint="Use ',' to open the decay, and parentheses to nest a second one.",
     solution='generate p p > t t~, (t > w+ b, w+ > l+ vl), t~ > w- b~'),

Step('generate', """
`add process` puts a second process in the same output, and `@N` tags it so
you can tell the pieces apart afterwards.

%(p)s add process p p > w+ j, w+ > l+ vl @2

`display processes` lists everything defined so far. The tag ends up in the
output directory names and in the event file, which is how you separate the
contributions of a multi-process run.
""" % {'p': P},
     title='several processes at once',
     hint="'add process' takes the same syntax as 'generate', plus '@N'.",
     solution='add process p p > w+ j, w+ > l+ vl @2'),

Step('add', """
Polarisation. `{X}` after a (multi)particle fixes its helicity: `{L}` and
`{R}` for left and right, `{T}` transverse, `{0}` longitudinal, `{A}` auxiliary.
It works on external particles, massless or massive, and on massive internal
particles before a decay chain.

%(p)s generate p p > z{0} z{T}, z > e+ e-

The process line is only half of it -- the run needs three settings too, and
forgetting them is the usual failure:
 * `set group_subprocesses False` BEFORE generating, or the polarisations get
   grouped away;
 * `nhel = 1` in the run card;
 * `me_frame` in the run card, naming the legs that define the rest frame.

Careful with `me_frame`: it indexes the NORMALISED leg order, not the order
you wrote in the process line. For `p p > w+ z j j, w+ > l+ vl, z > l+ l-`
the WZ rest frame is `me_frame = [3,4,5,6]`.
""" % {'p': P},
     title='polarisation',
     hint="Append '{0}' or '{T}' to a particle name.",
     solution='generate p p > z{0} z{T}, z > e+ e-'),

Step('generate', """
Finally, NLO. Square brackets after the process ask for the loop and
real-emission contributions:

  [QCD]          the full NLO computation, ready for aMC@NLO
  [real=QCD]     real-emission diagrams only
  [virt=QCD]     loop diagrams only, for standalone MadLoop
  [noborn=QCD]   loop-induced, for processes with no Born

%(p)s generate p p > t t~ [QCD]

Watch what happens: MG5 switches to the aMC@NLO interface by itself, because
the process asked for it. This tutorial keeps running across that switch.

Order constraints mean different things either side of the bracket: those
BEFORE `[` restrict the Born amplitude, those AFTER `]` restrict the squared
matrix element. And decay chains are not allowed at NLO -- use MadSpin.
""" % {'p': P},
     title='NLO processes',
     hint="Put '[QCD]' at the end of the process line.",
     solution='generate p p > t t~ [QCD]'),

Step('generate', """
That is the syntax tour. A few things worth remembering:

 * `define` makes your own multiparticle label, e.g.
   `define v = w+ w- z a`, usable anywhere `p` or `j` is;
 * `display diagrams` draws what you generated before you commit to an
   output -- a cheap habit;
 * `help generate` prints the whole grammar in one screen.

Where to go next:
 * `tutorial lo`         take a process all the way to events
 * `tutorial nlo`        run the NLO process you just generated
 * `tutorial exercises`  practise, with the answers checked
 * `tutorial list`       everything on offer

Leave tutorial mode with `tutorial stop`.
""",
     title='wrap-up'),

    ],
)
