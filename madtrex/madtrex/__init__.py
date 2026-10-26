"""MadtRex: MatrixElementApi <-> REX::tea event reweighting for MadGraph7.

Built on top of madspace and pyrex (the bundled Rex fork); madspace itself
does not depend on this package.
"""

# madspace first: it preloads libmadspace and registers the MatrixElementApi/
# Context/LHEMeta types that the _madtrex_py bindings use.
import madspace  # noqa: F401
import pyrex  # noqa: F401
import pyrex.tea  # noqa: F401

from ._madtrex_py import *  # noqa: F401,F403
