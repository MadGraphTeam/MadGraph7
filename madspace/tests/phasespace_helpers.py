"""Sampling and kinematics shared by the phase-space cut tests.

pytest puts this directory on sys.path (rootdir-relative imports, no
__init__.py), so the test modules import it by name.
"""

import numpy as np


def sample(mapping, seed, n, conditions=()):
    """Map n uniform points through a PhaseSpaceMapping.

    Returns the momenta and the weight (Jacobian times cuts) as arrays, as the
    mapping gives them: non-finite values are left in, see finite_weight.
    conditions are passed on unchanged, e.g. a permutation index per point.
    """
    rng = np.random.default_rng(seed)
    p_ext, _x1, _x2, det = mapping.map_forward(
        [rng.random((n, mapping.random_dim()))], list(conditions)
    )
    return np.asarray(p_ext), np.asarray(det)


def finite_weight(p, det):
    """The weight with every point whose weight or momenta are not finite set
    to 0, i.e. what an integral over the batch should sum."""
    finite = np.isfinite(det) & np.all(np.isfinite(p), axis=(1, 2))
    return np.where(finite, det, 0.0)


def invariant_mass(p, *indices):
    """Invariant mass of the sum of the momenta at the given positions."""
    total = sum(p[:, i, :] for i in indices)
    m2 = total[:, 0] ** 2 - np.sum(total[:, 1:] ** 2, axis=1)
    return np.sqrt(np.maximum(m2, 0.0))
