"""The weights of the FKS integrand from madspace kinematics and the
matrix-element blocks of MadFKS, point by point against sigint of
driver_mintFO.f (test_data/fks_integrand, see test_fks_integrand_reference.py).

At each point of the reference, the Born of MadFKS is boosted into the lab
and mapped back to the random numbers of a RAMBO Born mapping, which
FKSKinematics of the FKS configurations of the subprocess runs with the three
random numbers of the radiation at the picked configuration (and at that of
the n-body part). The matrix-element blocks of the point, those of MadFKS
itself, go into FKSSubtraction and FKSSoftVirtual with the strong coupling
divided out (g = 1, as the weight lines of MadFKS store them):

- the real per NLO order, smatrix_real over g^n with n its power of g;
- the Born per Born order, sborn at the split-order position of the
  counterterm (the NLO order) over g^b;
- the color-correlated Borns, B_mn = sborn_sf(m, n) / g^(b + 2) for m != n
  and 2 sborn_sf(m, m) / g^(b + 2) for a massive m, the rest from color
  conservation;
- the spin-correlated Born, B_perp = B / 2 + Re(e A) with the azimuthal
  part A of sborn (amp_split_cnt(:, 2)) and the phase e of sborncol_isr and
  sborncol_fsr at y = 1: conj(xij_aor) for an incoming mother, and for an
  outgoing one xij_aor without the azimuth phi of the mother,
  xij_aor exp(-2 i phi) (the factor of getaziangles in sborncol_fsr);
- the virtual, the finite part of MadLoop at the Ellis-Sexton scale over g^n.

The contributions are then the weight lines of sigint. The real emission and
the soft, collinear and soft-collinear counterterms come per unit Jacobian of
MadFKS (xi xinorm times that of the event, xinorm times those of the
counter-events), times the weight of vegas and the multichannel factor of
include_multichannel_enhance (amp2 of the Born diagram of the channel over the
sum of those of the Born, at the Born of the event). The Born, the integrated
counterterms and the virtual come with f_b and f_nb of compute_prefactors_nbody.

Per unit phase space the weights of madspace (its Jacobians and the flux
1/(2 s_hat), 1/(2 M) for a decay) are then those of MadFKS at the event and
at every counter-event up to one factor per point, the ratio of the Jacobians
of the two Born mappings. MadFKS spreads the n-body part over the random numbers of the
radiation with the soft counter-event of the configuration of the n-body part
over its volume, and halves it where the configurations of the channel have
emitters in the initial and in the final state (ini_fin_fks).

With the couplings and luminosities of MadFKS at each kinematics, FKSWeights
then gives the final weights of MadFKS in pb (a width in GeV for a decay),
and FKSCrossSection gives the same from the blocks directly.

For a decay MadFKS also keeps a dependence on mu_F, of the decaying particle
as if it were an incoming parton (the sum over nincoming of the coefficient
of log(mu_F^2 / Q^2) in bornsoftvirtual). It moves between wgt_1 and wgt_3 of
the integrated counterterms, their sum stays that of madspace, which has
none.
"""

import functools
import glob
import math
import os

import numpy as np
import pytest
import torch
from fks_testing import evaluate, reference, reference_born_mapping

import madspace as ms
from madspace.fks import (
    fks_configuration_mapping,
    fks_cross_section,
    fks_cuts,
    fks_soft_virtual,
    fks_subtraction,
    fks_weights,
)

DATA = sorted(
    glob.glob(
        os.path.join(os.path.dirname(__file__), "test_data", "fks_integrand", "*.json")
    )
)
IDS = [os.path.basename(path)[:-5] for path in DATA]
# the contributions of add_wgt (see test_fks_integrand_reference.py) and the
# outputs of FKSSubtraction and the Jacobians of FKSKinematics they go with
COUNTERTERMS = {
    1: ("real", "weight", "momenta"),
    4: ("soft", "soft_det", "soft_momenta"),
    5: ("collinear", "collinear_det", "collinear_momenta"),
    6: ("soft_collinear", "soft_collinear_det", "soft_collinear_momenta"),
}
# the factors 1 - stiny and 1 - ctiny of the radiation of a final-state
# emitter that MadFKS divides f_b by (fsr_born_cutoff_factor)
FSR_BORN_CUTOFF_FACTOR = (1 - 1e-6) * (1 - 5e-7)
# the cuts of FKSCuts, by their names in the run card
CUTS = (
    "jetalgo",
    "jetradius",
    "ptj",
    "etaj",
    "maxjetflavor",
    "ptl",
    "etal",
    "drll",
    "drll_sf",
    "mll",
    "mll_sf",
)
# GeV^-2 to pb, of madspace (INV_GEV2_TO_PB) and of MadFKS (conv of
# include_PDF_and_alphas)
MADSPACE_CONV = 0.38937937217186e9
MADFKS_CONV = 389379660.0
# the outputs of FKSWeights and the contributions of add_wgt they sum
WEIGHTS = {
    "event_weight": {1},
    "soft_weight": {4},
    "collinear_weight": {5},
    "soft_collinear_weight": {6},
    "nbody_weight": {2, 3, 14, 15},
    "lo_weight": {2},
}
# The Born of a hadron collision comes back through the boost into the lab
# and the round trip of the Born mapping, at rapidities up to 6. The largest
# deviation is 3e-12 (1e-14 for lepton beams).
RTOL = 1e-10


def to_torch(x):
    return torch.as_tensor(np.asarray(x)).to(torch.get_default_device())


def boost_z(p, y):
    """Boost along z by the rapidities y."""
    q = p.copy()
    cosh, sinh = np.cosh(y)[:, None], np.sinh(y)[:, None]
    q[..., 0] = cosh * p[..., 0] + sinh * p[..., 3]
    q[..., 3] = sinh * p[..., 0] + cosh * p[..., 3]
    return q


def incoming_sum(p, incoming=2):
    """The sum of the incoming momenta of p (batch, n, 4), the first
    incoming (two, or one for a decay)."""
    return p[:, :incoming].sum(1)


def to_partonic_frame(p, incoming=2):
    """Momenta (batch, n, 4) in the center-of-mass frame of the incoming
    particles, along z."""
    total = incoming_sum(p, incoming)
    y = 0.5 * np.log((total[:, 0] + total[:, 3]) / (total[:, 0] - total[:, 3]))
    return boost_z(p, -y)


def s_hat(p, incoming=2):
    total = incoming_sum(p, incoming)
    return total[:, 0] ** 2 - (total[:, 1:] ** 2).sum(-1)


def flux(p, incoming=2):
    """1 / (2 s_hat) of a collision, 1 / (2 M) of a decay."""
    s = s_hat(p, incoming)
    return 1 / (2 * (s if incoming == 2 else np.sqrt(s)))


def scale_coefficient(c):
    """wgt_1 + wgt_2 log(mu_R^2 / Q^2) + wgt_3 log(mu_F^2 / Q^2) of a
    contribution."""
    q2, mu_r2, mu_f2 = c["scales2"]
    w1, w2, w3 = c["wgt"]
    return w1 + w2 * math.log(mu_r2 / q2) + w3 * math.log(mu_f2 / q2)


class Channel:
    """The points of a channel of the reference with the madspace kinematics
    at them and the blocks of MadFKS as the inputs of the subtraction."""

    def __init__(self, data, channel):
        run = data["run_card"]
        configurations = channel["fks_configurations"]
        orders = configurations["orders"]
        qcd = orders["split_orders"].index("QCD")
        positions = [entry["orders"] for entry in channel["orders"]]
        # the split-order positions of the NLO orders and of the counterterms
        # of the Born orders, with their powers of g
        self.nlo = [positions.index(o) for o in orders["nlo"]]
        self.nlo_power = [o[qcd] for o in orders["nlo"]]
        self.counterterm = [
            positions.index([v + 2 * (k == qcd) for k, v in enumerate(b)])
            for b in orders["born"]
        ]
        self.born_power = [b[qcd] for b in orders["born"]]
        pdgs = configurations["born_pdgs"]
        masses = [data["param_card"]["mass"][str(abs(p))] for p in pdgs]
        # the masses of MadFKS: those of the param card, unless the model
        # computes them (the W of the sm model, from m_Z, G_F and alpha), as
        # the Born of the first point has them
        first = np.array(channel["points"][0]["p_born"])
        for k, mass in enumerate(masses):
            shell = math.sqrt(max(first[k, 0] ** 2 - (first[k, 1:] ** 2).sum(), 0.0))
            if mass > 0 and abs(shell / mass - 1) > 1e-10:
                masses[k] = shell
        self.massive = np.array(masses) > 0
        self.incoming = configurations["nincoming"]
        # MadFKS keeps a decay in GeV
        self.madfks_conv = MADFKS_CONV if self.incoming == 2 else 1.0
        self.madspace_conv = MADSPACE_CONV if self.incoming == 2 else 1.0
        if self.incoming == 1:
            # the decaying particle at rest into the final state at one vertex
            outgoing = masses[1:]
            vertex = ["i0"] + [f"o{k}" for k in range(len(outgoing))]
            diagram = ms.Diagram(masses[:1], outgoing, [], [vertex])
            self.born = ms.PhaseSpaceMapping(
                ms.Topology(diagram), masses[0], leptonic=True
            )
        else:
            self.born = ms.PhaseSpaceMapping(
                masses,
                2 * math.sqrt(float(run["ebeam1"]) * float(run["ebeam2"])),
                leptonic=int(run["lpp1"]) == 0,
                mode=ms.PhaseSpaceMapping.TChannelMode.rambo,
            )
        self.configurations = configurations
        self.points = [p for p in channel["points"] if "blocks" in p]
        assert len(self.points) == len(channel["points"])
        first = self.points[0]["born"]
        mapping = fks_configuration_mapping(
            self.born, configurations, first["tau_lower_bound"] * first["stot"]
        )
        self.kinematics = ms.FKSKinematics(mapping)
        self.names = list(mapping.counter_event_types().keys())
        emitters = [
            c["j_fks"] <= self.incoming for c in configurations["configurations"]
        ]
        # the n-body part of a channel split into its initial-state and
        # final-state emitters (ini_fin_fks)
        self.nbody_jac = 0.5 if len(set(emitters)) == 2 else 1.0
        self.final_state = [not e for e in emitters]
        # whether a configuration has a massless emitter, with collinear
        # counterterms
        self.collinear = any(
            not self.massive[c["j_fks"] - 1 - (c["j_fks"] > c["i_fks"])]
            for c in configurations["configurations"]
        )
        self.n_f = data["n_f"]
        self.run_card = run
        self.alpha = 1 / data["param_card"]["sminputs"]["1"]

    def field(self, *keys):
        values = []
        for point in self.points:
            for key in keys:
                point = point[key]
            values.append(point)
        return np.array(values)

    @functools.cached_property
    def random_numbers(self):
        """The random numbers of the Born of MadFKS for the Born mapping, then
        those of its radiation."""
        p_born = self.field("p_born")
        ycm = self.field("born", "ycm")
        x = self.field("born", "xbjrk")
        r_born, _ = self.born.map_inverse(
            [to_torch(boost_z(p_born, ycm)), to_torch(x[:, 0]), to_torch(x[:, 1])]
        )
        r_born = torch.as_tensor(r_born).cpu().numpy()
        return np.concatenate([r_born, self.field("radiation")], axis=1)

    def run_kinematics(self, key):
        index = self.field(key) - 1
        outputs = evaluate(self.kinematics, self.random_numbers, fks_index=index)
        return dict(zip(self.names, outputs)), index

    @functools.cached_property
    def event(self):
        """FKSKinematics at the picked configuration."""
        return self.run_kinematics("nfks_picked")

    @functools.cached_property
    def nbody(self):
        """FKSKinematics at the configuration of the n-body part."""
        return self.run_kinematics("nfks_born")

    def azimuthal_phase(self, point):
        """The phase e of the azimuthal part of the Born at the collinear
        counter-event of a point: conj(xij_aor) for an incoming mother,
        xij_aor exp(-2 i phi) for an outgoing one of azimuth phi in p_born."""
        phase = complex(*point["blocks"]["azimuthal_phase"])
        configuration = self.configurations["configurations"][point["nfks_picked"] - 1]
        i_fks, j_fks = configuration["i_fks"], configuration["j_fks"]
        leg = j_fks - 1 - (j_fks > i_fks)
        if leg < self.incoming:
            return phase.conjugate()
        mother = point["p_born"][leg]
        transverse = math.hypot(mother[1], mother[2])
        if transverse == 0:
            return phase
        azimuth = complex(mother[1], mother[2]) / transverse
        return phase * azimuth.conjugate() ** 2

    @functools.cached_property
    def blocks(self):
        """The real per NLO order, the Born, color-correlated and
        spin-correlated Borns per Born order, without the coupling."""
        count, n = len(self.points), len(self.configurations["born_pdgs"])
        real = np.zeros((count, len(self.nlo)))
        born = np.zeros((count, len(self.counterterm)))
        color = np.zeros((count, len(self.counterterm), n, n))
        spin = np.zeros_like(born)
        diagonal = np.full_like(color, np.nan)
        for k, point in enumerate(self.points):
            blocks = point["blocks"]
            if "real" in blocks:
                for o, (pos, power) in enumerate(zip(self.nlo, self.nlo_power)):
                    real[k, o] = blocks["real_orders"][pos] / blocks["g_real"] ** power
            phase = self.azimuthal_phase(point)
            g = blocks["g_born"]
            for o, (pos, power) in enumerate(zip(self.counterterm, self.born_power)):
                born[k, o] = blocks["born_orders"][pos] / g**power
                azimuthal = complex(
                    blocks["azimuthal_orders"][0][pos],
                    blocks["azimuthal_orders"][1][pos],
                )
                spin[k, o] = born[k, o] / 2 + (phase * azimuthal).real / g**power
            nbody = blocks["nbody"]
            for link in nbody["links"]:
                # the legs of the real process at the configuration of the
                # n-body part in the Born
                row, column = (
                    v - 1 - (v > nbody["i_fks"]) for v in (link["m"], link["n"])
                )
                for o, (pos, power) in enumerate(
                    zip(self.counterterm, self.born_power)
                ):
                    value = link["orders"][pos] / link["g"] ** (power + 2)
                    if row == column:
                        diagonal[k, o, row, row] = 2 * value
                    else:
                        color[k, o, row, column] = value
                        color[k, o, column, row] = value
        for m in range(n):
            color[:, :, m, m] = -color[:, :, m].sum(-1)
        return real, born, color, spin, diagonal

    def scales(self):
        """The Ellis-Sexton scale Q of the n-body part (mu_R of the
        references), its mu_F and mu_F of the collinear counterterms (that of
        the n-body part where there are none), from the scales of the
        contributions (those of the real emission are at its own
        kinematics)."""
        q = np.ones(len(self.points))
        mu_f = np.ones(len(self.points))
        collinear = np.zeros(len(self.points))
        for k, point in enumerate(self.points):
            for c in point["contributions"]:
                if c["type"] in (2, 3):
                    q[k] = math.sqrt(c["scales2"][0])
                    mu_f[k] = math.sqrt(c["scales2"][2])
                if c["type"] in (5, 6):
                    collinear[k] = math.sqrt(c["scales2"][2])
        return q, mu_f, np.where(collinear > 0, collinear, mu_f)

    @functools.cached_property
    def subtraction(self):
        """The outputs of FKSSubtraction at the event."""
        out, index = self.event
        real, born, color, spin, _ = self.blocks
        q, _, mu_f = self.scales()
        function = fks_subtraction(self.born, self.configurations, n_f=self.n_f)
        names = [
            "momenta",
            "weight",
            "xi",
            "xi_hat",
            "y",
            "xi_max",
            "p_i_hat",
            "soft_momenta",
            "soft_det",
            "soft_p_i_hat",
            "soft_xi_max",
            "collinear_momenta",
            "collinear_det",
            "collinear_p_i_hat",
            "collinear_xi",
            "collinear_xi_max",
            "soft_collinear_momenta",
            "soft_collinear_det",
            "soft_collinear_p_i_hat",
        ]
        count = len(self.points)
        outputs = evaluate(
            function,
            *[out[name] for name in names],
            real,
            born,
            color,
            spin,
            np.full(count, 1 / (4 * math.pi)),
            np.full(count, self.alpha),
            q,
            mu_f,
            mu_f,
            fks_index=index,
        )
        return dict(zip(["real", "soft", "collinear", "soft_collinear"], outputs))

    @functools.cached_property
    def soft_virtual(self):
        """The outputs of FKSSoftVirtual at the Born, at mu_R = mu_F = Q."""
        out, _ = self.event
        _, born, color, _, _ = self.blocks
        q, _, _ = self.scales()
        count = len(self.points)
        function = fks_soft_virtual(self.born, self.configurations, n_f=self.n_f)
        return evaluate(
            function,
            out["born_momenta"],
            born,
            color,
            np.full(count, 1 / (4 * math.pi)),
            np.full(count, self.alpha),
            q,
            q,
            q,
        )

    def madfks_norm(self, k, kind):
        """The Jacobian of MadFKS per unit output of FKSSubtraction, with the
        weight of vegas and the multichannel factor."""
        point = self.points[k]
        norm = point["xi_norm"]
        event, cnt = point["event"], point["counter_events"]
        jacobian = {
            1: event["xi"] * norm["xinorm_ev"] * event["jacobian"],
            4: norm["xinorm_ev"] * cnt[0]["jacobian"],
            5: norm["xinorm_cnt"][1] * cnt[1]["jacobian"],
            6: norm["xinorm_cnt"][1] * cnt[2]["jacobian"],
        }[kind]
        return jacobian * point["vegas_wgt"] * point["blocks"]["channel"]["enhance"]

    def born_jacobian_ratio(self, k):
        """madspace's Born weight over the Born Jacobian of MadFKS, xjac xpswgt
        (2 pi)^(4 - 3 n) with n final-state particles."""
        point = self.points[k]
        out, _ = self.event
        final = len(self.configurations["born_pdgs"]) - self.incoming
        fortran = point["born"]["xjac"] * point["born"]["xpswgt"]
        return out["born_weight"][k] / (fortran * (2 * math.pi) ** (4 - 3 * final))

    def nbody_spread(self, k):
        """The factor by which MadFKS spreads its n-body part over the random
        numbers of the radiation: the soft counter-event of the configuration
        of the n-body part over its volume xi_max s_hat / (16 pi^2), over
        fsr_born_cutoff_factor for a final-state emitter, times the factor of
        ini_fin_fks."""
        out, _ = self.event
        nbody, index = self.nbody
        radiation = nbody["soft_det"][k] / nbody["born_weight"][k]
        born = out["born_momenta"][k : k + 1]
        volume = nbody["soft_xi_max"][k] * s_hat(born, self.incoming)[0]
        spread = radiation / (volume / (16 * math.pi**2)) * self.nbody_jac
        if self.final_state[index[k]]:
            spread /= FSR_BORN_CUTOFF_FACTOR
        return spread

    def couplings_and_luminosities(self):
        """The strong coupling at the event, the Born and the collinear
        counter-event and the luminosity of each kinematics, from the
        contributions of MadFKS (g_strong and the weight over its value
        without PDFs), one where a point has no contribution there."""
        count = len(self.points)
        g = {name: np.ones(count) for name in ("event", "born", "collinear")}
        luminosity = {
            name: np.ones(count)
            for name in ("event", "soft", "collinear", "soft_collinear", "born")
        }
        coupling_of = {1: "event", 2: "born", 3: "born", 4: "born", 5: "collinear"}
        coupling_of[6] = "born"
        luminosity_of = {1: "event", 4: "soft", 5: "collinear", 6: "soft_collinear"}
        luminosity_of[2] = "born"
        for k, point in enumerate(self.points):
            for c in point["contributions"]:
                if c["type"] in coupling_of:
                    g[coupling_of[c["type"]]][k] = c["g_strong"]
                bare = scale_coefficient(c) * c["g_strong"] ** c["qcd_power"]
                if c["type"] in luminosity_of and bare != 0:
                    value = c["weight"] / (self.madfks_conv * bare)
                    luminosity[luminosity_of[c["type"]]][k] = value
        alpha_s = {name: value**2 / (4 * math.pi) for name, value in g.items()}
        return alpha_s, luminosity

    @functools.cached_property
    def inputs(self):
        """The inputs of FKSWeights and FKSCrossSection by name: the
        kinematics, the outputs of the subtraction, the blocks, the n-body part
        with the scale logs of MadFKS and its virtual (both terms of the
        virtual trick), the couplings and luminosities of MadFKS, the
        multichannel factor and one over the probability of the picked
        configuration."""
        out, _ = self.event
        real, born, color, spin, _ = self.blocks
        soft_virtual, ren, fact = self.soft_virtual
        q, born_mu_f, mu_f = self.scales()
        nbody = soft_virtual.copy()
        virtual = np.zeros_like(soft_virtual)
        for k, point in enumerate(self.points):
            f_nb = point["prefactors"]["f_nb"]
            for c in point["contributions"]:
                if c["type"] == 3:
                    order = self.nlo.index(c["amp_pos"] - 1)
                    q2, mu_r2, mu_f2 = c["scales2"]
                    nbody[k, order] += ren[k, order] * math.log(mu_r2 / q2)
                    nbody[k, order] += fact[k, order].sum() * math.log(mu_f2 / q2)
                elif c["type"] in (14, 15):
                    order = self.nlo.index(c["amp_pos"] - 1)
                    virtual[k, order] += scale_coefficient(c) / f_nb
        alpha_s, luminosity = self.couplings_and_luminosities()
        inputs = dict(out, **self.subtraction)
        inputs.update(
            matrix_element=real,
            born=born,
            color_correlated=color,
            spin_correlated=spin,
            soft_virtual=nbody,
            virtual=virtual,
            alpha=np.full(len(q), self.alpha),
            ren_scale=q,
            fact_scale1=born_mu_f,
            fact_scale2=born_mu_f,
            collinear_fact_scale1=mu_f,
            collinear_fact_scale2=mu_f,
            channel_weight=self.field("blocks", "channel", "enhance"),
            configuration_weight=1 / self.field("vol"),
        )
        inputs.update({f"{k}_alpha_s": v for k, v in alpha_s.items()})
        inputs.update({f"{k}_luminosity": v for k, v in luminosity.items()})
        return inputs

    def cuts(self):
        """The cuts of the run, by the names of FKSCuts."""
        cuts = {name: float(self.run_card[name]) for name in CUTS}
        cuts["maxjetflavor"] = int(cuts["maxjetflavor"])
        return cuts

    @functools.cached_property
    def masks(self):
        """The cuts of each luminosity: those of FKSCuts at the event and,
        for the counterterms and the n-body part, at the soft counter-event,
        as sigint with event projection."""
        out, index = self.event
        function = fks_cuts(self.born, self.configurations, **self.cuts())
        (event,) = evaluate(function, out["momenta"], fks_index=index)
        (soft,) = evaluate(function, out["soft_momenta"], fks_index=index)
        return dict(
            event_luminosity=event,
            soft_luminosity=soft,
            collinear_luminosity=soft,
            soft_collinear_luminosity=soft,
            born_luminosity=soft,
        )

    def run(self, function, masked=False):
        """The outputs of a generator on the inputs of its arguments, by
        name, with the luminosities times the cuts if masked."""
        _, index = self.event
        inputs = dict(self.inputs)
        if masked:
            inputs.update({k: inputs[k] * v for k, v in self.masks.items()})
        names = list(function.arg_types().keys())
        args = [inputs[name] for name in names if name != "fks_index"]
        outputs = evaluate(
            function, *args, fks_index=index if "fks_index" in names else None
        )
        return dict(zip(function.return_types().keys(), outputs))

    @functools.cached_property
    def weights(self):
        """The outputs of FKSWeights, with the cuts."""
        return self.run(fks_weights(self.born, self.configurations), masked=True)


@functools.cache
def channels(path):
    data = reference(path)
    return [Channel(data, channel) for channel in data["channels"]]


@pytest.mark.parametrize("path", DATA, ids=IDS)
def test_kinematics(path):
    """The event and the Born of madspace are those of the contributions of
    MadFKS (in their partonic frames), and its event has zero weight where
    MadFKS rejects it (an empty range of xi)."""
    for channel in channels(path):
        out, _ = channel.event
        event_valid = channel.field("event", "valid")
        assert ((out["weight"] > 0) == event_valid).all()
        for k, point in enumerate(channel.points):
            i_fks = point["blocks"]["i_fks"] - 1
            for c in point["contributions"]:
                theirs = np.array([c["momenta"]])
                if c["type"] == 1:
                    ours = out["momenta"][k : k + 1]
                elif c["type"] == 5:
                    ours = out["collinear_momenta"][k : k + 1]
                elif c["type"] in (2, 3, 4, 6, 14, 15):
                    ours = out["born_momenta"][k : k + 1]
                    theirs = np.delete(theirs, i_fks, axis=1)
                else:
                    continue
                ours = to_partonic_frame(ours, channel.incoming)
                theirs = to_partonic_frame(theirs, channel.incoming)
                scale = theirs[..., 0].max()
                np.testing.assert_allclose(ours / scale, theirs / scale, atol=1e-12)
        # xi and y to the rounding of the Born, absolute: xi is an energy in
        # units of sqrt(s_hat) (up to 1e-14 for a Born at rapidity 6)
        valid = event_valid.astype(bool)
        np.testing.assert_allclose(
            out["xi"][valid], channel.field("event", "xi")[valid], rtol=0, atol=1e-13
        )
        np.testing.assert_allclose(
            out["y"][valid], channel.field("event", "y")[valid], rtol=0, atol=1e-13
        )


@pytest.mark.parametrize("path", DATA, ids=IDS)
def test_real_emission_and_counterterms(path):
    """The real emission and the soft, collinear and soft-collinear
    counterterms of FKSSubtraction with the blocks of MadFKS, per unit
    Jacobian of MadFKS, are its weight lines, with their logs of mu_F. A
    channel with massive emitters alone has no collinear counterterms."""
    seen, expected = set(), set()
    for channel in channels(path):
        ours = channel.subtraction
        for k, point in enumerate(channel.points):
            for c in point["contributions"]:
                if c["type"] not in COUNTERTERMS:
                    continue
                seen.add(c["type"])
                name = COUNTERTERMS[c["type"]][0]
                order = channel.nlo.index(c["amp_pos"] - 1)
                value = ours[name][k, order] * channel.madfks_norm(k, c["type"])
                np.testing.assert_allclose(value, scale_coefficient(c), rtol=RTOL)
        expected |= set(COUNTERTERMS) if channel.collinear else {1, 4}
    assert seen == expected


@pytest.mark.parametrize("path", DATA, ids=IDS)
def test_born_and_integrated_counterterms(path):
    """The Born and FKSSoftVirtual with the blocks of MadFKS, with the
    coefficients of log(mu_R^2 / Q^2) and log(mu_F^2 / Q^2), times f_b and
    f_nb are its weight lines of the Born and of the integrated
    counterterms. For a decay madspace has no dependence on mu_F, and the sum
    at the scales of the line is that of MadFKS."""
    for channel in channels(path):
        _, born, _, _, _ = channel.blocks
        soft_virtual, ren, fact = channel.soft_virtual
        for k, point in enumerate(channel.points):
            f = point["prefactors"]
            for c in point["contributions"]:
                if c["type"] == 2:
                    order = channel.born_power.index(c["qcd_power"])
                    np.testing.assert_allclose(
                        born[k, order] * f["f_b"], c["wgt"][0], rtol=RTOL
                    )
                elif c["type"] == 3:
                    order = channel.nlo.index(c["amp_pos"] - 1)
                    ours = [soft_virtual[k, order], ren[k, order], fact[k, order].sum()]
                    if channel.incoming == 2:
                        np.testing.assert_allclose(
                            np.array(ours) * f["f_nb"], c["wgt"], rtol=RTOL, atol=0
                        )
                        continue
                    assert fact[k, order].sum() == 0
                    np.testing.assert_allclose(
                        ren[k, order] * f["f_nb"], c["wgt"][1], rtol=RTOL, atol=0
                    )
                    q2, mu_r2, _ = c["scales2"]
                    total = ours[0] + ours[1] * math.log(mu_r2 / q2)
                    np.testing.assert_allclose(
                        total * f["f_nb"], scale_coefficient(c), rtol=RTOL
                    )


@pytest.mark.parametrize("path", DATA, ids=IDS)
def test_virtual(path):
    """The finite part of MadLoop over g^n times f_nb is the virtual of the
    virtual trick of MadFKS: the residual (type 14) times the virtual fraction
    plus the average (type 15)."""
    count = 0
    for channel in channels(path):
        for point in channel.points:
            lines = {c["type"]: c for c in point["contributions"]}
            if 14 not in lines:
                assert "virtual" not in point["blocks"]["nbody"]
                continue
            count += 1
            virtual = point["blocks"]["nbody"]["virtual"]
            residual, average = lines[14], lines[15]
            assert residual["amp_pos"] == average["amp_pos"]
            finite = virtual["orders"][residual["amp_pos"] - 1]
            ours = finite / residual["g_strong"] ** residual["qcd_power"]
            theirs = residual["wgt"][0] * virtual["fraction"] + average["wgt"][0]
            np.testing.assert_allclose(
                ours * point["prefactors"]["f_nb"], theirs, rtol=RTOL
            )
    assert count > 0


@pytest.mark.parametrize("path", DATA, ids=IDS)
def test_color_links(path):
    """The color-correlated Borns of sborn_sf conserve color, and those of a
    massive leg with itself are C_m B / 2 times g^2 (the colored ones, here
    quarks, also the decaying top)."""
    for channel in channels(path):
        _, born, color, _, diagonal = channel.blocks
        np.testing.assert_allclose(color.sum(-1), 0.0, atol=1e-13 * born.max())
        given = ~np.isnan(diagonal)
        assert given.any() == channel.massive.any()
        np.testing.assert_allclose(diagonal[given], color[given], rtol=1e-12)
        linked = np.flatnonzero(given.any(axis=(0, 1, 3)))
        assert (channel.massive[linked]).all()
        for m in linked:
            np.testing.assert_allclose(color[:, :, m, m], 4 / 3 * born, rtol=1e-12)


@pytest.mark.parametrize("path", DATA, ids=IDS)
def test_multichannel_factor(path):
    """The multichannel factor of MadFKS is amp2 of the Born diagram of the
    channel over the sum over the Born diagrams, the same at both
    configurations of a point."""
    for channel in channels(path):
        for point in channel.points:
            blocks = point["blocks"]
            for entry in (blocks["channel"], blocks["nbody"]["channel"]):
                amp2 = np.array(entry["amp2"])
                expected = amp2[entry["config"] - 1] / amp2.sum()
                expected *= entry["symmetry_factor"]
                np.testing.assert_allclose(entry["enhance"], expected, rtol=1e-14)
            assert blocks["channel"]["enhance"] == pytest.approx(
                blocks["nbody"]["channel"]["enhance"], rel=1e-14
            )


@pytest.mark.parametrize("path", DATA, ids=IDS)
def test_densities(path):
    """Per unit phase space, the weights of madspace (its Jacobians and the
    flux) over those of MadFKS (per unit weight of vegas, multichannel factor
    and pick of the configuration) are the ratio of the Jacobians of the two
    Born mappings at every contribution of a point: madspace's Born weight
    over xjac xpswgt (2 pi)^(4 - 3 n) of MadFKS, with n final-state particles.
    For the n-body part of MadFKS that ratio carries the soft counter-event of
    the configuration of the n-body part over its volume xi_max s_hat / (16
    pi^2) (and fsr_born_cutoff_factor for a final-state emitter) and the
    factor of ini_fin_fks."""
    for channel in channels(path):
        out, _ = channel.event
        _, born, _, _, _ = channel.blocks
        ours = channel.subtraction
        for k, point in enumerate(channel.points):
            expected = channel.born_jacobian_ratio(k)
            per_unit = point["vegas_wgt"] * point["blocks"]["channel"]["enhance"]
            for c in point["contributions"]:
                if c["type"] in COUNTERTERMS:
                    name, det, momenta = COUNTERTERMS[c["type"]]
                    order = channel.nlo.index(c["amp_pos"] - 1)
                    factor = flux(out[momenta][k : k + 1], channel.incoming)[0]
                    madspace = ours[name][k, order] * out[det][k] * factor
                    madfks = scale_coefficient(c) * point["vol"] / per_unit
                    np.testing.assert_allclose(madspace / madfks, expected, rtol=RTOL)
                elif c["type"] == 2:
                    order = channel.born_power.index(c["qcd_power"])
                    factor = flux(out["born_momenta"][k : k + 1], channel.incoming)[0]
                    madspace = born[k, order] * out["born_weight"][k] * factor
                    madfks = c["wgt"][0] / per_unit / channel.nbody_spread(k)
                    np.testing.assert_allclose(madspace / madfks, expected, rtol=RTOL)


@pytest.mark.parametrize("path", DATA, ids=IDS)
def test_weights(path):
    """FKSWeights gives the weights of MadFKS in pb (wgts(1) of the weight
    lines, summed over the orders and the contributions at each kinematics),
    per unit weight of vegas, up to the ratio of the Jacobians of the two Born
    mappings, MadFKS's spread of the n-body part and the two conversions to
    pb. With the cuts of FKSCuts, a kinematics without contributions of
    MadFKS has zero weight, and the cross section is the sum."""
    seen, expected = set(), set()
    for channel in channels(path):
        ours = channel.weights
        # to the rounding of the parts, which cancel
        parts = [ours[name] for name in WEIGHTS if name != "lo_weight"]
        difference = np.abs(ours["cross_section"] - sum(parts))
        assert (difference <= 1e-14 * np.abs(parts).sum(0)).all()
        for k, point in enumerate(channel.points):
            ratio = channel.born_jacobian_ratio(k)
            ratio *= channel.madspace_conv / channel.madfks_conv
            for name, types in WEIGHTS.items():
                lines = [c for c in point["contributions"] if c["type"] in types]
                if not lines:
                    assert ours[name][k] == 0, (name, k)
                    continue
                seen.add(name)
                madfks = sum(c["weight"] for c in lines) / point["vegas_wgt"]
                if name in ("nbody_weight", "lo_weight"):
                    madfks /= channel.nbody_spread(k)
                np.testing.assert_allclose(ours[name][k] / madfks, ratio, rtol=RTOL)
        collinear = {"collinear_weight", "soft_collinear_weight"}
        expected |= set(WEIGHTS) - (set() if channel.collinear else collinear)
    assert seen == expected


@pytest.mark.parametrize("path", DATA, ids=IDS)
def test_cross_section(path):
    """FKSCrossSection from the blocks with the cuts of the run is
    FKSSubtraction, FKSSoftVirtual and FKSWeights in turn with the
    luminosities times the cuts of FKSCuts, so it gives the weights of
    test_weights."""
    for channel in channels(path):
        function = fks_cross_section(
            channel.born, channel.configurations, cuts=channel.cuts(), n_f=channel.n_f
        )
        ours = channel.run(function)
        for name, value in channel.weights.items():
            np.testing.assert_allclose(ours[name], value, rtol=1e-13, atol=0)


def test_cross_section_with_extra_counterterms():
    """Where the subtraction also takes the Borns of extra counterterms (a u >
    u a with a gluon and a photon as mothers of the quark), the n-body part
    of FKSCrossSection takes the Born orders alone: its Born weight is that of
    the first Born at the Born."""
    path = os.path.join(
        os.path.dirname(__file__), "test_data", "fks", "pp_aj_mix_P0_gu_ua.json"
    )
    data = reference(path)
    born = reference_born_mapping(data)
    configurations = data["fks_configurations"]
    function = fks_cross_section(born, configurations)
    types = function.arg_types()
    weights = fks_weights(born, configurations)
    assert types["born"].shape[0] == weights.born_order_count() + 1
    mapping = fks_configuration_mapping(born, configurations, 0.0)
    kinematics = ms.FKSKinematics(mapping)
    count = 64
    rng = np.random.default_rng(4711)
    r = rng.random((count, born.random_dim() + 3))
    index = rng.integers(len(configurations["configurations"]), size=count)
    names = list(mapping.counter_event_types().keys())
    inputs = dict(zip(names, evaluate(kinematics, r, fks_index=index)))
    args = []
    for name in types.keys():
        if name == "fks_index":
            continue
        if name in inputs:
            args.append(inputs[name])
        else:
            shape = [count, *types[name].shape]
            args.append(rng.random(shape) + 0.5)
    outputs = dict(
        zip(
            function.return_types().keys(),
            evaluate(function, *args, fks_index=index),
        )
    )
    values = dict(zip([n for n in types.keys() if n != "fks_index"], args))
    orders = configurations["orders"]
    power = orders["born"][0][orders["split_orders"].index("QCD")]
    g = np.sqrt(4 * math.pi * values["born_alpha_s"])
    expected = MADSPACE_CONV * values["born_luminosity"] * values["channel_weight"]
    expected *= values["born_weight"] / (2 * s_hat(values["born_momenta"]))
    expected *= g**power * values["born"][:, 0]
    # s_hat of the momenta in the lab rounds to exp(2 |y|) eps at the
    # rapidities y of the random Borns
    np.testing.assert_allclose(outputs["lo_weight"], expected, rtol=1e-9)
    assert np.isfinite(outputs["cross_section"]).all()


def test_decay_round_trip():
    """The FKS mapping of a decay maps the random numbers to the real emission
    of the decaying particle at rest, with momentum fractions one, and back,
    with the inverse weight. A decay radiates off final-state particles only
    and needs the Born mapping of a decay."""
    channel = channels(DATA[IDS.index("t_bw")])[0]
    mapping = fks_configuration_mapping(channel.born, channel.configurations)
    count = 256
    r = np.random.default_rng(5).random((count, channel.born.random_dim() + 3))
    index = torch.zeros(count, dtype=torch.int32).to(torch.get_default_device())
    momenta, x1, x2, det = mapping.map_forward([to_torch(r)], [index])
    p = torch.as_tensor(momenta).cpu().numpy()
    mass = channel.born.external_masses()[0]
    np.testing.assert_allclose(p[:, 0], np.tile([mass, 0, 0, 0], (count, 1)))
    np.testing.assert_allclose(p[:, 1:].sum(1) - p[:, 0], 0.0, atol=1e-12 * mass)
    np.testing.assert_array_equal(torch.as_tensor(x1).cpu().numpy(), 1.0)
    np.testing.assert_array_equal(torch.as_tensor(x2).cpu().numpy(), 1.0)
    r_inv, det_inv = mapping.map_inverse([momenta, x1, x2], [index])
    np.testing.assert_allclose(torch.as_tensor(r_inv).cpu().numpy(), r, atol=1e-9)
    product = (
        torch.as_tensor(det).cpu().numpy() * torch.as_tensor(det_inv).cpu().numpy()
    )
    np.testing.assert_allclose(product, 1.0, rtol=1e-9)
    with pytest.raises(ValueError, match="final-state particles only"):
        ms.FKSRadiation(3, 3, 0, leptonic=True, incoming=1)
    with pytest.raises(ValueError, match="final-state particles only"):
        ms.FKSRadiation(3, 3, 1, leptonic=False, incoming=1)
    collision = ms.PhaseSpaceMapping([0.0, 0.0, 4.7, 80.0], 1000.0, leptonic=True)
    with pytest.raises(ValueError, match="decay topology"):
        fks_configuration_mapping(collision, channel.configurations)
