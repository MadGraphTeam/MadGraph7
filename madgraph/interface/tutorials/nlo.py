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
"""The `nlo` tutorial -- running aMC@NLO.  Ported from tutorial_text_nlo.py."""

from __future__ import absolute_import

import madgraph.interface.tutorial_text_nlo as legacy
from madgraph.interface.tutorials._port import retarget
from madgraph.interface.tutorials.session import Step, Tutorial


tutorial = Tutorial(
    name='nlo',
    title='NLO computations with aMC@NLO',
    description='generate, output and run a process at next-to-leading order',
    aliases=('aMCatNLO',),
    order='free',
    steps=[
        Step('tutorial', retarget(legacy.tutorial),
             title='welcome',
             solution='generate p p > t t~ [QCD]'),
        Step('generate', retarget(legacy.generate),
             title='generate a process at NLO',
             solution='output MY_FIRST_AMCATNLO_RUN'),
        Step('display_processes', retarget(legacy.display_processes),
             title='list the defined processes',
             solution='output MY_FIRST_AMCATNLO_RUN'),
        Step('add_process', retarget(legacy.add_process),
             title='add a second process',
             solution='output MY_FIRST_AMCATNLO_RUN'),
        Step(('output', 'open_index'), retarget(legacy.output),
             title='produce an output',
             solution='launch MY_FIRST_AMCATNLO_RUN'),
        Step('launch', retarget(legacy.launch),
             title='run it'),
    ],
)
