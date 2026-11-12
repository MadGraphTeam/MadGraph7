from . import object_library 
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
from . import function_library

gauge = [1]

all_particles = particles.all_particles
all_vertices = vertices.all_vertices
all_CTvertices = CT_vertices.all_CTvertices
all_couplings = couplings.all_couplings
all_lorentz = lorentz.all_lorentz
all_parameters = parameters.all_parameters
all_CTparameters = CT_parameters.all_CTparameters
all_functions = function_library.all_functions
all_orders = coupling_orders.all_orders

__author__ = "Huasheng Shao"
__version__ = "1.0"
__email__ = "huasheng.shao@cern.ch"
__header__ = """By loading this model the calculation can involve ELECTROWEAK corrections.
Please cite ref. 'arXiv:1804.10017' when using results from this code."""
