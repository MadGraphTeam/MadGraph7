#from __future__ import absolute_import
from __future__ import absolute_import
from . import particles
from . import couplings
from . import CT_couplings
from . import lorentz
from . import parameters
from . import CT_parameters
from . import vertices
from . import CT_vertices
from . import write_param_card
from . import coupling_orders
from . import object_library
from . import function_library

# model options
gauge = [0, 1]

all_particles = particles.all_particles
all_vertices = vertices.all_vertices
all_CTvertices = vertices.all_CTvertices
all_couplings = couplings.all_couplings
all_lorentz = lorentz.all_lorentz
all_parameters = parameters.all_parameters
all_CTparameters = CT_parameters.all_CTparameters
all_functions = function_library.all_functions
all_orders = coupling_orders.all_orders

__author__ = "V. Hirschi"
__version__ = "1.2"
__email__ = "valentin.hirschi@gmail.com"


# Bound states (quarkonia, leptonia) of boundstates.py, added by
# tools/make_onia_model.py to the model loop_sm.
dual_mass_scheme = True

__header__ = \
"""------------------------------------------------------------------------
By loading this model the calculation can involve quarkonia or leptonia.
Please cite refs. 'arXiv:2510.26773' and 'arXiv:2607.26739' when using
results from this code.
------------------------------------------------------------------------
"""
