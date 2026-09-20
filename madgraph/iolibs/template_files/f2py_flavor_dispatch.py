"""Flavor dispatch helper for the f2py ``matrix2py`` standalone module.

The compiled module exposes, for each entry point, two flavors:

* a ``...smatrix`` / ``...get_value`` / ``...smatrixhel`` function taking the
  full per-leg ``FLAVOR(NEXTERNAL)`` array, and
* a matching ``..._idx`` function taking a single integer flavor index.

This wrapper lets you call one Python function and pass *either* form: a scalar
integer (or length-1 sequence) is routed to the ``_idx`` entry, a
length-``NEXTERNAL`` sequence to the array entry. The underlying function names
are auto-detected, so this works regardless of the f2py name-mangling/prefix.

Example
-------
>>> import matrix2py
>>> from flavor_dispatch import FlavorDispatch
>>> me = FlavorDispatch(matrix2py)
>>> me.initialisemodel('param_card.dat')
>>> ans = me.get_value(P, alphas, nhel, 3)              # by flavor index
>>> ans = me.get_value(P, alphas, nhel, [1, -1, 2, -2]) # by flavor array

Crossing / PDG matching
-----------------------
When the module was generated with crossing symmetry on, a single flavor index
also carries a *crossing*: the extended ``FLAV_IDX = cross*NFLAV + flav`` makes
the one generated matrix element evaluate any process related to it by moving
legs between the initial and the final state. The caller usually does not want
to think in those indices -- they have a physical process as a list of signed
PDG codes and want the right index. ``find_pdg`` does that lookup and
``matrix_element_pdg`` / ``get_value_pdg`` call straight through:

>>> me.find_pdg([2, 21, 2, 21])            # u g > u g from a u u~ > g g module
4
>>> ans = me.get_value_pdg(P, alphas, nhel, [2, 21, 2, 21])

The PDG list is matched in the leg order the momenta are given in: the index
``find_pdg`` returns is exactly the one to pass to the ``*_idx`` entry points
together with momenta in that same order. A crossed leg is conjugated (an
incoming ``u~`` that a crossing turns into an outgoing ``u`` matches pdg +2),
which is why the match is on signed PDG codes.
"""

import numbers


def _is_index(flavor):
    """A scalar integer (or length-1 sequence) is treated as a flavor index;
    anything longer is treated as a per-leg FLAVOR array."""
    if isinstance(flavor, numbers.Integral):
        return True
    try:
        return len(flavor) == 1
    except TypeError:
        # No __len__ (e.g. a non-Integral scalar such as numpy.int64 that is
        # not registered with numbers.Integral): treat it as a flavor index.
        # We only swallow TypeError here, the specific error raised by len()
        # on a sizeless object, so genuine errors elsewhere are not masked.
        return True


def _as_index(flavor):
    if isinstance(flavor, numbers.Integral):
        return int(flavor)
    return int(flavor[0])


class FlavorDispatch(object):
    """Thin convenience layer over the f2py matrix2py module."""

    #: entry point -> position of the flavor argument in the call
    _flavor_pos = {
        'smatrix': 1,        # (p, flavor)
        'smatrixhel': 2,     # (p, hel, flavor)
        'get_value': 3,      # (p, alphas, nhel, flavor)
    }

    def __init__(self, module):
        self.module = module
        self._cache = {}

    def _resolve(self, base):
        """Return (array_func, idx_func) for *base*, auto-detected from the
        module's attributes (case-insensitive, suffix match)."""
        if base in self._cache:
            return self._cache[base]
        array_f = idx_f = None
        for name in dir(self.module):
            low = name.lower()
            if low.endswith(base + '_idx'):
                idx_f = getattr(self.module, name)
            elif low.endswith(base):
                array_f = getattr(self.module, name)
        self._cache[base] = (array_f, idx_f)
        return array_f, idx_f

    def _call(self, base, args):
        array_f, idx_f = self._resolve(base)
        pos = self._flavor_pos[base]
        flavor = args[pos]
        if _is_index(flavor):
            if idx_f is None:
                raise AttributeError(
                    "No function ending with '%s_idx' found in f2py module "
                    "for index calls" % base)
            args = list(args)
            args[pos] = _as_index(flavor)
            return idx_f(*args)
        if array_f is None:
            raise AttributeError(
                "No function ending with '%s' found in f2py module "
                "for array calls" % base)
        return array_f(*args)

    # -- dispatching entry points (flavor may be an index or an array) --------
    def smatrix(self, p, flavor):
        return self._call('smatrix', [p, flavor])

    def smatrixhel(self, p, hel, flavor):
        return self._call('smatrixhel', [p, hel, flavor])

    def get_value(self, p, alphas, nhel, flavor):
        return self._call('get_value', [p, alphas, nhel, flavor])

    # -- crossing / PDG matching ---------------------------------------------
    def _find_one(self, suffix):
        """Return the single module function whose (lowercased) name ends with
        *suffix*, or None. Cached under a distinct key so it never collides
        with the (array, idx) pairs stored by _resolve."""
        key = ('one', suffix)
        if key in self._cache:
            return self._cache[key]
        found = None
        for name in dir(self.module):
            if name.lower().endswith(suffix):
                found = getattr(self.module, name)
                break
        self._cache[key] = found
        return found

    def flavor_layout(self):
        """Return (nflav, nexternal, ncross) from GET_FLAVOR_LAYOUT.

        ncross = (nexternal+1)**2 is the number of crossing codes, so the
        extended index ranges over 1 .. ncross*nflav. Raises if the module was
        built without the crossing entry points (an old or non-standalone-v4
        output)."""
        func = self._find_one('get_flavor_layout')
        if func is None:
            raise AttributeError(
                "This module exposes no 'get_flavor_layout': it was not built "
                "with the crossing/PDG entry points.")
        nflav, nexternal, ncross = func()
        return int(nflav), int(nexternal), int(ncross)

    def pdg_for_index(self, flav_idx):
        """Signed per-leg PDG codes of the process an extended FLAV_IDX selects,
        or None if the index names no valid flavor/crossing.

        The codes are in the leg order the momenta must be supplied in for that
        index; a leg that the crossing moved between the initial and the final
        state is conjugated."""
        func = self._find_one('get_pdg_for_flavor')
        if func is None:
            raise AttributeError(
                "This module exposes no 'get_pdg_for_flavor': it was not built "
                "with the crossing/PDG entry points.")
        pdgs = tuple(int(x) for x in func(flav_idx))
        # The Fortran routine zero-fills PDGS for an index it cannot resolve.
        if all(code == 0 for code in pdgs):
            return None
        return pdgs

    def _pdg_map(self):
        """{signed-PDG-tuple: extended FLAV_IDX} over every valid index.

        Built once and cached. When two indices give the same PDG signature in
        the same leg order (physically the same process, e.g. a crossing that
        coincides with the identity for a symmetric flavor) the first is kept:
        they evaluate to the same matrix element."""
        if 'pdg_map' in self._cache:
            return self._cache['pdg_map']
        nflav, _nexternal, ncross = self.flavor_layout()
        mapping = {}
        for cross in range(ncross):
            for flav in range(1, nflav + 1):
                flav_idx = cross * nflav + flav
                pdgs = self.pdg_for_index(flav_idx)
                if pdgs is not None:
                    mapping.setdefault(pdgs, flav_idx)
        self._cache['pdg_map'] = mapping
        return mapping

    def find_pdg(self, pdgs):
        """Extended FLAV_IDX whose crossed process is *pdgs* (signed, in the
        given leg order), or None if no crossing of the generated matrix
        element reproduces it."""
        return self._pdg_map().get(tuple(int(code) for code in pdgs))

    def _require_pdg(self, pdgs):
        flav_idx = self.find_pdg(pdgs)
        if flav_idx is None:
            raise ValueError(
                "No crossing of the generated matrix element yields the "
                "process %s" % (tuple(int(code) for code in pdgs),))
        return flav_idx

    def matrix_element_pdg(self, p, pdgs):
        """SMATRIX for the process *pdgs*, reached through crossing. Momenta
        must be given in the same leg order as *pdgs*."""
        return self.smatrix(p, self._require_pdg(pdgs))

    def get_value_pdg(self, p, alphas, nhel, pdgs):
        """get_value for the process *pdgs*, reached through crossing. Momenta
        must be given in the same leg order as *pdgs*."""
        return self.get_value(p, alphas, nhel, self._require_pdg(pdgs))

    # -- pass-through for the model initialiser -------------------------------
    def initialisemodel(self, path):
        for name in dir(self.module):
            if name.lower().endswith('initialisemodel'):
                return getattr(self.module, name)(path)
        raise AttributeError("No 'initialisemodel' entry found")
