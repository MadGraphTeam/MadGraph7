# SPDX-License-Identifier: 0BSD
"""Dependency-owned exact four-dimensional components for local loop kernels.

The native epsilon convention is the gamma-five/four-gamma trace divided by
four. Its components are constructed with the dependency's existing matrices
and contraction, rather than an independently implemented epsilon algebra.
This does not define a dimensional gamma-five or UFO import prescription.
"""

from functools import cache

from .._internal.physics.symbols import symbols


@cache
def _native_epsilon_components():
    from symbolica import E
    from symbolica.community import tensor as t

    labels = symbols.qualified(
        *(f"loop_epsilon_components::axis_{index}" for index in range(4))
    )
    gamma = t.TensorExpression.dirac_gamma(4)
    trace = t.TensorExpression.gamma5(4)(0, 1)
    for index, label in enumerate(labels):
        trace *= gamma(index + 1, (index + 2) % 5, label)
    # Keep this local trace unsimplified: the existing matrix components must
    # evaluate it without recursively introducing an unresolved epsilon tensor.
    value = (trace / E("4")).to_tensor(t.TensorLibrary.hep_lib_atom())
    slots = [t.Representation.mink(4)(label) for label in labels]
    value = value.permute_axes([list(value.axes).index(slot) for slot in slots])
    # Library lookup uses an unresolved signature, not these explicit labels.
    return t.Tensor.dense(
        t.TensorName.levi_civita()(*[t.Representation.mink(4)] * 4), list(value)
    )


def physical_tensor_library():
    """Fresh exact HEP library, including the dependency's native 4D epsilon.

    Only the constant epsilon tensor is cached. Routed momenta and external
    state components registered by callers remain local to the returned library.
    """
    from symbolica.community import tensor as t

    library = t.TensorLibrary.hep_lib_atom()
    library.register(_native_epsilon_components())
    return library
