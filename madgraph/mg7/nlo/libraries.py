"""The Fortran provider of a subprocess (nlo/provider.py) as UMAMI libraries
of madspace, through madspace.PythonFunction with the provisional NLO keys of
umami.h, until MadMatrix or pyAmpliCol give them. The C++ NLO integrands
take their matrix elements through MatrixElement from these libraries, which
the libraries of MadMatrix or pyAmpliCol then replace.

One library per matrix element, as at LO, with no FKS rows: the scheme stays
on the side of the caller.

- born: the Born of the subprocess per Born order (squared_orders), amp2 of
  the Born diagrams of the channels of MadFKS (diagram_amp2), the
  colour-correlated Born <T_i . T_j> B of every pair i < j of the coloured
  Born legs, flattened as in umami.h (color_correlated), and the
  spin-correlated Born (spin_correlated) with the polarization of the leg of
  spin_legs replaced by its vector. It reports the colour representations of
  the Born legs, all taken as outgoing.
- real r: the real of the real matrix element r (real_me - 1 of the FKS
  configurations) per NLO order (squared_orders).
- loop: the virtual per NLO order (loop), its finite part and the
  coefficients of 1/eps and 1/eps^2 as MadLoop gives them at the
  renormalization scale of the input scale.

The provider gives its blocks at g_s = 1, so the libraries take alpha_s =
1 / (4 pi) and refuse other strong couplings. Its blocks carry e of the model,
from the QED coupling of the counterterms, and the libraries take any g_ew by
multiplying each order with (g_ew / e) to its power of e, so that g_ew = e
gives the blocks as they are and g_ew = 1 gives them without e. They boost
the momenta and the spin vectors to the partonic centre-of-mass frame, as
fks_blocks does, and keep the restrictions of the provider: colour
correlations exist for the pairs MadFKS registers (with two coloured legs at
least), spin correlations for the Born legs that are the mother (a gluon or a
photon) of an FKS configuration, one spin leg per point. The spin-correlated
Born for a transverse vector v (orthogonal to the leg, v^2 = -1 for k_perp of
FKS) is B / 2 + Re(e A) of FortranProvider.spin_correlated, times the square
of the transverse part of v in general, zero for a vector along the leg.
Without a spin leg it is zero. The Borns of the extra counterterms of mixed
QCD-QED corrections are not covered yet.

The pipes of a provider take one request at a time, while the runtime of
madspace calls a library from several threads: a lock serializes the calls.
"""

import math
import threading

import numpy as np

from madgraph.mg7.nlo.provider import boost_z, helicity_phase, partonic_rapidity

# the coupling of the blocks at g_s = 1
ALPHA_S = 1 / (4 * math.pi)


def check_coupling(alpha_s, name):
    if not np.allclose(alpha_s, ALPHA_S, rtol=1e-12, atol=0):
        raise ValueError(
            f"{name}: the Fortran provider gives its blocks at g_s = 1, alpha_s = "
            "1 / (4 pi)"
        )


def partonic_frame(momenta, incoming):
    """The momenta (batch, n, 4) in their partonic centre-of-mass frame and
    the rapidity of the boost."""
    y = -partonic_rapidity(momenta, incoming)
    return boost_z(momenta, y), y


class ProviderLibraries:
    """The UMAMI libraries of a FortranProvider (see the module): born,
    reals (one per real matrix element) and loop as madspace.PythonFunction,
    and add_to adds them to a context."""

    def __init__(self, provider, name=None):
        from madspace import PythonFunction

        self.provider = provider
        self.name = name or provider.directory
        self.lock = threading.Lock()
        configurations = provider.configurations
        if provider.fks_configurations["orders"].get("extra_born"):
            raise NotImplementedError(
                f"{self.name}: no UMAMI libraries for the Borns of extra "
                "counterterms yet"
            )
        n = provider.particle_count - 1
        self.born_orders = len(provider.orders["born"])
        self.nlo_orders = len(provider.orders["nlo"])
        # e of the blocks, and the power of e of each Born and NLO order
        _, alpha, _ = provider.couplings()
        self.e = math.sqrt(4 * math.pi * alpha)
        qed = provider.qed_pos
        self._e_powers = {
            kind: None if qed < 0 else np.array([o[qed] for o in orders])
            for kind, orders in (
                ("born", provider.orders["born"]),
                ("nlo", provider.orders["nlo"]),
            )
        }
        self.color_representations = born_color_representations(provider)
        self._colored = [
            leg for leg, r in enumerate(self.color_representations) if r != 1
        ]
        # colour correlations need two coloured legs
        self.has_color = len(self._colored) >= 2
        born_outputs = ["squared_orders", "diagram_amp2", "spin_correlated"]
        if self.has_color:
            born_outputs.insert(2, "color_correlated")
        # per Born leg the configuration whose mother it is, a gluon or a
        # photon with that correction in the split orders
        self.spin_configuration = {}
        for c, configuration in enumerate(configurations):
            mother = {21: provider.qcd_pos, 22: provider.qed_pos}.get(
                configuration["ij_pdg"]
            )
            if mother is None or mother < 0:
                continue
            i_fks, j_fks = configuration["i_fks"], configuration["j_fks"]
            leg = j_fks - 1 - (j_fks > i_fks)
            self.spin_configuration.setdefault(leg, (c, mother))
        # per real matrix element a configuration of it
        self.real_configuration = {}
        for c, configuration in enumerate(configurations):
            self.real_configuration.setdefault(configuration["real_me"] - 1, c)
        self.real_count = len(provider.fks_configurations["real_flavors"])
        missing = set(range(self.real_count)) - set(self.real_configuration)
        if missing:
            raise ValueError(f"{self.name}: no configuration of the reals {missing}")
        self.born = PythonFunction(
            self._born,
            ["momenta", "alpha_s", "g_ew", "spin_legs", "spin_vectors"],
            born_outputs,
            particle_count=n,
            diagram_count=provider.channel_count,
            squared_order_count=self.born_orders,
            color_representations=self.color_representations,
            name=f"born of {self.name}",
        )
        self.reals = [
            PythonFunction(
                self._real(r),
                ["momenta", "alpha_s", "g_ew"],
                ["squared_orders"],
                particle_count=n + 1,
                squared_order_count=self.nlo_orders,
                name=f"real {r} of {self.name}",
            )
            for r in range(self.real_count)
        ]
        self.loop = PythonFunction(
            self._loop,
            ["momenta", "alpha_s", "g_ew", "scale"],
            ["loop"],
            particle_count=n,
            squared_order_count=self.nlo_orders,
            name=f"loop of {self.name}",
        )

    def real_indices(self):
        """Per FKS configuration its real matrix element, an index of
        reals."""
        return [c["real_me"] - 1 for c in self.provider.configurations]

    def spin_legs(self):
        """Per FKS configuration the Born leg of its mother (from 0) for a
        gluon or photon mother whose correction the split orders have, the
        leg of its spin correlation, -1 for the others."""
        legs = []
        for configuration in self.provider.configurations:
            i_fks, j_fks = configuration["i_fks"], configuration["j_fks"]
            leg = j_fks - 1 - (j_fks > i_fks)
            vector = configuration["ij_pdg"] in (21, 22)
            legs.append(leg if vector and leg in self.spin_configuration else -1)
        return legs

    def add_to(self, context):
        """The MatrixElementApi of the libraries in context: born, reals (a
        list by real matrix element) and loop."""
        return dict(
            born=context.add_matrix_element(self.born),
            reals=[context.add_matrix_element(real) for real in self.reals],
            loop=context.add_matrix_element(self.loop),
        )

    def _e_factors(self, g_ew, kind, name):
        """(g_ew / e) to the power of e of each order (batch, orders) of the
        Born or NLO orders, None for g_ew = e."""
        g_ew = np.asarray(g_ew)
        if np.all(g_ew == self.e):
            return None
        powers = self._e_powers[kind]
        if powers is None:
            raise ValueError(
                f"{name}: no power of e in the split orders, so g_ew must be e of "
                f"the model, {self.e!r}"
            )
        return (g_ew / self.e)[:, None] ** powers[None, :]

    def _born(self, momenta, alpha_s, g_ew, legs, vectors):
        provider = self.provider
        check_coupling(alpha_s, self.born.name)
        factors = self._e_factors(g_ew, "born", self.born.name)
        momenta, y = partonic_frame(np.asarray(momenta), provider.incoming)
        legs = np.asarray(legs)
        if (legs[:, 1] != -1).any():
            raise ValueError(f"{self.born.name}: one spin leg at most")
        leg = legs[:, 0]
        configuration = np.zeros(len(momenta), dtype=int)
        for value in np.unique(leg[leg >= 0]):
            if value not in self.spin_configuration:
                raise ValueError(
                    f"{self.born.name}: no spin correlation of leg {value}, the "
                    "mother of no configuration with a gluon or a photon"
                )
            configuration[leg == value] = self.spin_configuration[value][0]
        vector = boost_z(np.asarray(vectors)[:, 0], y)
        with self.lock:
            per_position, parts, amp2 = provider.born(momenta, configuration)
            full = provider.color_correlated(momenta) if self.has_color else None
        # the pairs i < j of the coloured legs, flattened as in umami.h
        color = (
            np.stack(
                [
                    full[:, :, a, b]
                    for k, a in enumerate(self._colored)
                    for b in self._colored[k + 1 :]
                ],
                axis=-1,
            )
            if self.has_color
            else None
        )
        plain = provider.born_orders_of(per_position)
        spin = np.zeros_like(plain)
        for value in np.unique(leg[leg >= 0]):
            points = np.flatnonzero(leg == value)
            _, mother = self.spin_configuration[value]
            phase, _ = helicity_phase(momenta[points, value], vector[points])
            if value < provider.incoming:
                phase = np.conj(phase)
            transverse = transverse_square(momenta[points, value], vector[points])
            for o, order in enumerate(provider.orders["born"]):
                position = provider.counterterm_position(order, mother)
                azimuthal = parts[points, mother, 1, position]
                spin[points, o] = transverse * (
                    plain[points, o] / 2 + (phase * azimuthal).real
                )
        if factors is not None:
            plain, spin = plain * factors, spin * factors
            if color is not None:
                color = color * factors[:, :, None]
        values = [np.ascontiguousarray(plain), np.ascontiguousarray(amp2)]
        if color is not None:
            values.append(np.ascontiguousarray(color))
        values.append(np.ascontiguousarray(spin))
        return tuple(values)

    def _real(self, r):
        configuration = self.real_configuration[r]

        def real(momenta, alpha_s, g_ew):
            provider = self.provider
            check_coupling(alpha_s, self.reals[r].name)
            factors = self._e_factors(g_ew, "nlo", self.reals[r].name)
            momenta, _ = partonic_frame(np.asarray(momenta), provider.incoming)
            with self.lock:
                orders = provider.real_orders(
                    momenta, np.full(len(momenta), configuration)
                )
            if factors is not None:
                orders = orders * factors
            return np.ascontiguousarray(orders)

        return real

    def _loop(self, momenta, alpha_s, g_ew, scale):
        provider = self.provider
        check_coupling(alpha_s, self.loop.name)
        factors = self._e_factors(g_ew, "nlo", self.loop.name)
        momenta, _ = partonic_frame(np.asarray(momenta), provider.incoming)
        with self.lock:
            values, _ = provider.virtual(momenta, np.asarray(scale))
        positions = [provider.position(o) for o in provider.orders["nlo"]]
        loop = np.stack(
            [values[part][:, positions] for part in ("finite", "single", "double")],
            axis=-1,
        )
        if factors is not None:
            loop = loop * factors[:, :, None]
        return np.ascontiguousarray(loop)


def born_color_representations(provider):
    """The colour representation of each Born leg (1, 3, -3 or 8), all taken
    as outgoing, from the real legs of the first configuration that emits a
    gluon or a photon, which leaves its mother with the representation of
    j_fks. The configurations give the representations of the particles, so
    those of the incoming legs change sign."""
    for configuration in provider.configurations:
        i_fks = configuration["i_fks"]
        if configuration["real_pdgs"][i_fks - 1] in (21, 22):
            colors = list(configuration["real_colors"])
            del colors[i_fks - 1]
            return [
                -c if leg < provider.incoming and abs(c) == 3 else c
                for leg, c in enumerate(colors)
            ]
    raise ValueError("no FKS configuration that emits a gluon or a photon")


def transverse_square(mother, vector):
    """The square of the part of vector (batch, 4) transverse to the
    massless mother (batch, 4), in the frame of both, as helicity_phase takes
    it: the vector without its part along the mother, then its spatial part
    orthogonal to the direction of the mother."""
    v = vector - (vector[:, :1] / mother[:, :1]) * mother
    n = mother[:, 1:] / np.linalg.norm(mother[:, 1:], axis=1, keepdims=True)
    spatial = v[:, 1:] - (v[:, 1:] * n).sum(1, keepdims=True) * n
    return (spatial**2).sum(1)
