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
- extra e: the Born of extra counterterm e of the mixed QCD-QED corrections
  per order of that Born (squared_orders) and its spin correlation at the
  mother of the configurations with the counterterm (spin_correlated), the
  Born of MadFKS times the ratio of the identical-particle factors of the Born
  of the subprocess and of the counterterm, which makes it the Born of the
  counterterm with its own factor. Orders that the NLO orders do not reach
  are zero.
- scale: the reference scale of the user (user_dynamical_scale) at the
  kinematics of the real process, as ren_scale, fact_scale1 and fact_scale2,
  for a CustomFunction of type scale.
- born_plain and born_scale: the Born without the spin inputs and output and
  the reference scale at the Born, for the n-body stream of Catani-Seymour.
- ew_sudakov: with --ewsudakov, the blocks of the Sudakov approximation of
  the EW corrections per helicity combination of the Born (the provisional
  ew_sudakov outputs of umami.h): the base amplitude squared at the LO1 and
  the LO2, its derivatives with respect to the six parameters of its
  parameter renormalization and its interferences with the interference
  amplitudes, as FortranProvider.ew_sudakov gives them. None without
  --ewsudakov. Its blocks carry e of the model, so it takes g_ew = e only.

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
Without a spin leg it is zero. For the Born of an extra counterterm the spin
leg is the mother of any configuration with the counterterm, A its
azimuthal part at the counterterm of the second splitting type (zero for a
quark mother).

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
        extra_born = provider.orders.get("extra_born", [])
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
                *((("extra", e), orders) for e, orders in enumerate(extra_born)),
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
        # per extra counterterm the configurations with it, which take the
        # same columns
        self.extra_configurations = {}
        for c, configuration in enumerate(configurations):
            e = configuration.get("extra_cnt_index", -1)
            if e >= 0:
                self.extra_configurations.setdefault(e, []).append(c)
        self._extra_columns = {}
        for e in range(len(extra_born)):
            if e not in self.extra_configurations:
                raise ValueError(
                    f"{self.name}: no configuration of extra counterterm {e}"
                )
            columns = [provider.extra_columns(c) for c in self.extra_configurations[e]]
            local = [
                [(row, position, factor) for _, row, position, factor in column]
                for column in columns
            ]
            if any(entry != local[0] for entry in local):
                raise ValueError(
                    f"{self.name}: the configurations of extra counterterm {e} take "
                    "different orders of its Born"
                )
            self._extra_columns[e] = local[0]
        self.extras = [
            PythonFunction(
                self._extra(e),
                ["momenta", "alpha_s", "g_ew", "spin_legs", "spin_vectors"],
                ["squared_orders", "spin_correlated"],
                particle_count=n,
                squared_order_count=len(extra_born[e]),
                name=f"extra counterterm {e} of {self.name}",
            )
            for e in range(len(extra_born))
        ]
        # with the inputs that CustomFunction passes
        self.scale = PythonFunction(
            self._scale,
            ["momenta", "particle_count", "pdg_ids"],
            ["ren_scale", "fact_scale1", "fact_scale2"],
            particle_count=n + 1,
            name=f"reference scale of {self.name}",
        )
        # the Born without spin correlations and the reference scale at the
        # Born, for the n-body stream of Catani-Seymour
        self.born_plain = PythonFunction(
            self._born_plain,
            ["momenta", "alpha_s", "g_ew"],
            born_outputs[: 3 if self.has_color else 2],
            particle_count=n,
            diagram_count=provider.channel_count,
            squared_order_count=self.born_orders,
            color_representations=self.color_representations,
            name=f"born of {self.name} without spin",
        )
        self.born_scale = PythonFunction(
            self._scale,
            ["momenta", "particle_count", "pdg_ids"],
            ["ren_scale", "fact_scale1", "fact_scale2"],
            particle_count=n,
            name=f"reference scale at the Born of {self.name}",
        )
        # the blocks of the Sudakov approximation of the EW corrections
        self.ew_sudakov = None
        info = provider.ew_sudakov_info()
        if info is not None:
            self.ew_sudakov = PythonFunction(
                self._ew_sudakov,
                ["momenta", "alpha_s", "g_ew"],
                [
                    "ew_sudakov_diagonal",
                    "ew_sudakov_lo2",
                    "ew_sudakov_derivatives",
                    "ew_sudakov_interference",
                ],
                particle_count=n,
                helicity_count=info["helicity_count"],
                interference_count=max(info["interference_count"], 1),
                name=f"EW Sudakov blocks of {self.name}",
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

    def extra_indices(self):
        """Per FKS configuration its extra counterterm, an index of extras,
        -1 for none."""
        return [c.get("extra_cnt_index", -1) for c in self.provider.configurations]

    def extra_spin_legs(self):
        """Per FKS configuration with an extra counterterm the Born leg of
        its mother (from 0), -1 for the others."""
        legs = []
        for configuration in self.provider.configurations:
            i_fks, j_fks = configuration["i_fks"], configuration["j_fks"]
            extra = configuration.get("extra_cnt_index", -1) >= 0
            legs.append(j_fks - 1 - (j_fks > i_fks) if extra else -1)
        return legs

    def add_to(self, context):
        """The MatrixElementApi of the libraries in context: born, reals (a
        list by real matrix element), loop, extras (a list by extra
        counterterm), scale, born_plain (the Born without the spin inputs and
        output), born_scale (the reference scale at the Born) and ew_sudakov
        (None without --ewsudakov)."""
        return dict(
            born=context.add_matrix_element(self.born),
            reals=[context.add_matrix_element(real) for real in self.reals],
            loop=context.add_matrix_element(self.loop),
            extras=[context.add_matrix_element(extra) for extra in self.extras],
            scale=context.add_matrix_element(self.scale),
            born_plain=context.add_matrix_element(self.born_plain),
            born_scale=context.add_matrix_element(self.born_scale),
            ew_sudakov=(
                None
                if self.ew_sudakov is None
                else context.add_matrix_element(self.ew_sudakov)
            ),
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

    def _born_plain(self, momenta, alpha_s, g_ew):
        count = len(momenta)
        legs = np.full((count, 2), -1, dtype=np.int32)
        vectors = np.zeros((count, 2, 4))
        # the outputs of _born without the spin correlation, the last one
        return self._born(momenta, alpha_s, g_ew, legs, vectors)[:-1]

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

    def _extra(self, e):
        configuration = self.extra_configurations[e][0]
        columns = self._extra_columns[e]

        def extra(momenta, alpha_s, g_ew, legs, vectors):
            provider = self.provider
            name = self.extras[e].name
            check_coupling(alpha_s, name)
            factors = self._e_factors(g_ew, ("extra", e), name)
            momenta, y = partonic_frame(np.asarray(momenta), provider.incoming)
            legs = np.asarray(legs)
            if (legs[:, 1] != -1).any():
                raise ValueError(f"{name}: one spin leg at most")
            leg = legs[:, 0]
            vector = boost_z(np.asarray(vectors)[:, 0], y)
            with self.lock:
                parts = provider.extra(momenta, np.full(len(momenta), configuration))
            plain = np.zeros((len(momenta), len(columns)))
            spin = np.zeros_like(plain)
            for k, (row, position, factor) in enumerate(columns):
                if position is not None:
                    plain[:, k] = factor * parts[:, row, 0, position].real
            for value in np.unique(leg[leg >= 0]):
                points = np.flatnonzero(leg == value)
                phase, _ = helicity_phase(momenta[points, value], vector[points])
                if value < provider.incoming:
                    phase = np.conj(phase)
                transverse = transverse_square(momenta[points, value], vector[points])
                for k, (row, position, factor) in enumerate(columns):
                    if position is None:
                        continue
                    azimuthal = factor * parts[points, row, 1, position]
                    spin[points, k] = transverse * (
                        plain[points, k] / 2 + (phase * azimuthal).real
                    )
            if factors is not None:
                plain, spin = plain * factors, spin * factors
            return np.ascontiguousarray(plain), np.ascontiguousarray(spin)

        return extra

    def _scale(self, momenta, particle_count, pdg_ids):
        with self.lock:
            scale = self.provider.reference_scales(np.asarray(momenta))
        scale = np.ascontiguousarray(scale)
        return scale, scale, scale

    def _ew_sudakov(self, momenta, alpha_s, g_ew):
        provider = self.provider
        name = self.ew_sudakov.name
        check_coupling(alpha_s, name)
        if not np.all(np.asarray(g_ew) == self.e):
            raise ValueError(f"{name}: the blocks take g_ew = e of the model only")
        momenta, _ = partonic_frame(np.asarray(momenta), provider.incoming)
        # the blocks of the Born, at configuration 0 (their identical-particle
        # factor is that of their base amplitude)
        with self.lock:
            blocks = provider.ew_sudakov(momenta, np.zeros(len(momenta), dtype=int))
        count, h = blocks["diagonal"].shape
        interference = np.zeros((count, h, self.ew_sudakov.interference_count, 2))
        t = blocks["interference"].shape[2]
        interference[:, :, :t, 0] = blocks["interference"].real
        interference[:, :, :t, 1] = blocks["interference"].imag
        return (
            np.ascontiguousarray(blocks["diagonal"]),
            np.ascontiguousarray(blocks["lo2"]),
            np.ascontiguousarray(blocks["derivatives"]),
            interference,
        )

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
