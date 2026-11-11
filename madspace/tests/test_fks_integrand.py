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
- the spin-correlated Born, B_perp = B / 2 + Re(conj(xij_aor) A) with the
  azimuthal part A of sborn (amp_split_cnt(:, 2)) and its phase xij_aor,
  which sborncol_isr and sborncol_fsr take at y = 1;
- the virtual, the finite part of MadLoop at the Ellis-Sexton scale over g^n.

The contributions are then the weight lines of sigint. The real emission and
the soft, collinear and soft-collinear counterterms come per unit Jacobian of
MadFKS (xi xinorm times that of the event, xinorm times those of the
counter-events), times the weight of vegas and the multichannel factor of
include_multichannel_enhance (amp2 of the Born diagram of the channel over the
sum of those of the Born, at the Born of the event). The Born, the integrated
counterterms and the virtual come with f_b and f_nb of compute_prefactors_nbody.

Per unit phase space the weights of madspace (its Jacobians and the flux
1/(2 s_hat)) are then those of MadFKS at the event and at every counter-event
up to one factor per point, the ratio of the Jacobians of the two Born
mappings. MadFKS spreads the n-body part over the random numbers of the
radiation with the soft counter-event of the configuration of the n-body part
over its volume, and halves it where the configurations of the channel have
emitters in the initial and in the final state (ini_fin_fks).
"""

import functools
import glob
import math
import os

import numpy as np
import pytest
import torch
from fks_testing import evaluate, reference

import madspace as ms
from madspace.fks import fks_configuration_mapping, fks_soft_virtual, fks_subtraction

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


def to_partonic_frame(p):
    """Momenta (batch, n, 4) in the center-of-mass frame of the incoming two,
    along z."""
    total = p[:, 0] + p[:, 1]
    y = 0.5 * np.log((total[:, 0] + total[:, 3]) / (total[:, 0] - total[:, 3]))
    return boost_z(p, -y)


def s_hat(p):
    total = p[:, 0] + p[:, 1]
    return total[:, 0] ** 2 - (total[:, 1:] ** 2).sum(-1)


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
        self.massive = np.array(masses) > 0
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
        emitters = [c["j_fks"] <= 2 for c in configurations["configurations"]]
        # the n-body part of a channel split into its initial-state and
        # final-state emitters (ini_fin_fks)
        self.nbody_jac = 0.5 if len(set(emitters)) == 2 else 1.0
        self.final_state = [not e for e in emitters]
        self.n_f = data["n_f"]
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
            phase = complex(*blocks["azimuthal_phase"])
            g = blocks["g_born"]
            for o, (pos, power) in enumerate(zip(self.counterterm, self.born_power)):
                born[k, o] = blocks["born_orders"][pos] / g**power
                azimuthal = complex(
                    blocks["azimuthal_orders"][0][pos],
                    blocks["azimuthal_orders"][1][pos],
                )
                spin[k, o] = (
                    born[k, o] / 2 + (phase.conjugate() * azimuthal).real / g**power
                )
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
        """The Ellis-Sexton scale Q of the n-body part and mu_F of the
        collinear counterterms (Q where there are none), from the scales of
        the contributions (Q of the real emission is at its own kinematics)."""
        q = np.ones(len(self.points))
        mu_f = np.zeros(len(self.points))
        for k, point in enumerate(self.points):
            for c in point["contributions"]:
                if c["type"] in (2, 3):
                    q[k] = math.sqrt(c["scales2"][0])
                if c["type"] in (5, 6):
                    mu_f[k] = math.sqrt(c["scales2"][2])
        return q, np.where(mu_f > 0, mu_f, q)

    @functools.cached_property
    def subtraction(self):
        """The outputs of FKSSubtraction at the event."""
        out, index = self.event
        real, born, color, spin, _ = self.blocks
        q, mu_f = self.scales()
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
        q, _ = self.scales()
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


@functools.lru_cache(maxsize=None)
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
                ours, theirs = to_partonic_frame(ours), to_partonic_frame(theirs)
                scale = theirs[..., 0].max()
                np.testing.assert_allclose(ours / scale, theirs / scale, atol=1e-12)
        valid = event_valid.astype(bool)
        np.testing.assert_allclose(
            out["xi"][valid], channel.field("event", "xi")[valid], rtol=1e-12
        )
        np.testing.assert_allclose(
            out["y"][valid], channel.field("event", "y")[valid], rtol=0, atol=1e-13
        )


@pytest.mark.parametrize("path", DATA, ids=IDS)
def test_real_emission_and_counterterms(path):
    """The real emission and the soft, collinear and soft-collinear
    counterterms of FKSSubtraction with the blocks of MadFKS, per unit
    Jacobian of MadFKS, are its weight lines, with their logs of mu_F."""
    seen = set()
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
    assert seen == set(COUNTERTERMS)


@pytest.mark.parametrize("path", DATA, ids=IDS)
def test_born_and_integrated_counterterms(path):
    """The Born and FKSSoftVirtual with the blocks of MadFKS, with the
    coefficients of log(mu_R^2 / Q^2) and log(mu_F^2 / Q^2), times f_b and
    f_nb are its weight lines of the Born and of the integrated
    counterterms."""
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
                    np.testing.assert_allclose(
                        np.array(ours) * f["f_nb"], c["wgt"], rtol=RTOL, atol=0
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
    massive leg with itself are C_m B / 2 times g^2."""
    for channel in channels(path):
        _, born, color, _, diagonal = channel.blocks
        np.testing.assert_allclose(color.sum(-1), 0.0, atol=1e-13 * born.max())
        given = ~np.isnan(diagonal)
        assert given.any() == channel.massive.any()
        np.testing.assert_allclose(diagonal[given], color[given], rtol=1e-12)
        for m in np.flatnonzero(channel.massive):
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
        nbody, nbody_index = channel.nbody
        _, born, _, _, _ = channel.blocks
        ours = channel.subtraction
        final = len(channel.configurations["born_pdgs"]) - 2
        for k, point in enumerate(channel.points):
            fortran_born = point["born"]["xjac"] * point["born"]["xpswgt"]
            fortran_born *= (2 * math.pi) ** (4 - 3 * final)
            expected = out["born_weight"][k] / fortran_born
            per_unit = point["vegas_wgt"] * point["blocks"]["channel"]["enhance"]
            for c in point["contributions"]:
                if c["type"] in COUNTERTERMS:
                    name, det, momenta = COUNTERTERMS[c["type"]]
                    order = channel.nlo.index(c["amp_pos"] - 1)
                    flux = 1 / (2 * s_hat(out[momenta][k : k + 1])[0])
                    madspace = ours[name][k, order] * out[det][k] * flux
                    madfks = scale_coefficient(c) * point["vol"] / per_unit
                    np.testing.assert_allclose(madspace / madfks, expected, rtol=RTOL)
                elif c["type"] == 2:
                    order = channel.born_power.index(c["qcd_power"])
                    sqrt_flux = 1 / (2 * s_hat(out["born_momenta"][k : k + 1])[0])
                    madspace = born[k, order] * out["born_weight"][k] * sqrt_flux
                    madfks = c["wgt"][0] / per_unit
                    # MadFKS's n-body part per unit Born phase space
                    radiation = nbody["soft_det"][k] / nbody["born_weight"][k]
                    volume = (
                        nbody["soft_xi_max"][k]
                        * s_hat(out["born_momenta"][k : k + 1])[0]
                    )
                    spread = radiation / (volume / (16 * math.pi**2))
                    if channel.final_state[nbody_index[k]]:
                        spread /= FSR_BORN_CUTOFF_FACTOR
                    spread *= channel.nbody_jac
                    np.testing.assert_allclose(
                        madspace / madfks * spread, expected, rtol=RTOL
                    )
