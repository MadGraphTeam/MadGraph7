"""The Python driver of the fixed-order NLO runs of madgraph/mg7/nlo that
came before the C++ integrands, kept as their reference in the tests: the
integrands of a channel at given random numbers with the matrix elements of
the Fortran provider called from numpy (evaluate of FKSReferenceIntegrand,
CSBornReference and CSRealReference), the discrete choice of the FKS
configurations and the virtual trick (DiscreteChoice, VirtualTrick), as
mint and MC_integer of MadFKS adapt them.

The runs of the reference (FKSReferenceRun, CSReferenceRun) set up the
integrands of the reference in place of those of madgraph/mg7/nlo, and run
as the runs there, through the event generator of madspace. The tests take
the C++ integrands of their channels and compare them with evaluate point
by point. The reference changes only where the C++ integrands change what
they compute.
"""

from .core import DiscreteChoice, VirtualTrick  # noqa: F401
from .cs import (  # noqa: F401
    CSBornReference,
    CSRealReference,
    CSReferenceRun,
    dipole_variables,
)
from .fks import (  # noqa: F401
    FKSReferenceIntegrand,
    FKSReferenceRun,
    ew_sudakov_corrections,
)
