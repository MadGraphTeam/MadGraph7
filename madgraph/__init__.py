################################################################################
#
# Copyright (c) 2009 The MadGraph5_aMC@NLO Development team and Contributors
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
from __future__ import absolute_import
class MadGraph5Error(Exception):
    """Exception raised if an exception is find 
    Those Types of error will stop nicely in the cmd interface"""

class InvalidCmd(MadGraph5Error):
    """a class for the invalid syntax call"""

class aMCatNLOError(MadGraph5Error):
    """A MC@NLO error"""

import os
import logging
import time
pjoin = os.path.join

#Look for basic file position MG5DIR and MG4DIR
MG5DIR = os.path.realpath(os.path.join(os.path.dirname(__file__),
                                                                os.path.pardir))
if ' ' in MG5DIR:
   logging.critical('''\033[1;31mpath to MG5: "%s" contains space. 
    This is likely to create code unstability. 
    Please consider changing the path location of the code\033[0m''' % MG5DIR)
   time.sleep(1)
MG4DIR = MG5DIR
ReadWrite = os.access(MG5DIR, os.W_OK) # W_OK is for writing

if ReadWrite:
    # Temporary fix for problem with auto-update
    try:
        tmp_path = pjoin(MG5DIR, 'Template','LO','Source','make_opts')
        #1480375724 is 29/11/16
        if os.path.exists(tmp_path) and os.path.getmtime(tmp_path) < 1480375724:
            os.remove(tmp_path)
            shutil.copy(pjoin(MG5DIR, 'Template','LO','Source','.make_opts'),
                    pjoin(MG5DIR, 'Template','LO','Source','make_opts'))
    except Exception as error:
        pass
  
ADMIN_DEBUG = False  
if os.path.exists(os.path.join(MG5DIR,'bin', 'create_release.py')):
    if os.path.exists(os.path.join(MG5DIR,'.bzr')):
        ADMIN_DEBUG = True

if __debug__ or ADMIN_DEBUG:
    ordering = True
else:
    ordering = False

# Sum the quartic gluon contributions into the cubic amplitude carrying the
# same colour factor, see HelasMatrixElement.get_quartic_amplitude_merges.
# Set through the interface, "set merge_quartic_vertices <value>", and read
# here because it is wanted while the diagrams are generated, long before any
# exporter exists. False, or one of:
#   'speed' -- the current sums, and the diagram order which allows them. Wins
#              on cpu, where the amplitude calls dominate.
#   'slots' -- no current sums, and the order which keeps fewest currents
#              alive. Trades 6% more amplitude calls for 23% fewer
#              wavefunction slots at seven gluons, which is the trade a gpu
#              wants when occupancy is the limit.
#   'auto'  -- decide per process. Below merge_quartic_min_legs external legs
#              only the amplitude merges are taken, which shrink the JAMP
#              block for free; the seed rule -- and with it the reordering,
#              the current sums and the slots they cost -- is left off, since
#              below that size it does not pay for itself. At or above it,
#              generate as for 'speed' and let each output pick 'slots' when
#              the matrix elements go to a gpu backend and 'speed' otherwise.
#              The interface resolves the second half before anything reads it
#              again, so only the generation ever sees 'auto'.
# 'speed' and 'slots' are unconditional -- they are the way to ask for the
# merging on a small process anyway.
# Off by default: several consumers read the diagram or amplitude structure
# rather than the result, see "Why this is not the default" in
# docs/gluon-quartic-plan.md.
merge_quartic_vertices = False

# External legs an 'auto' process needs before the seed rule is worth it.
# Measured on g g > N g and g g > t t~ N g, both of which turn over at six:
# five legs and below the full merging is inside the noise or slightly
# negative (-1% at g g > g g g, which also pays 12 -> 19 wavefunction slots),
# six and above it is a steady 5-8% with the source and the object file
# shrinking too. The amplitude merges alone, which is what is left below the
# threshold, are +4% at g g > g g g and neutral at four and five legs
# elsewhere, at the same slot count -- never a loss. See "Full sweep" in
# docs/gluon-quartic-plan.md.
merge_quartic_min_legs = 6
        
