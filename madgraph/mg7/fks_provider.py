"""The Fortran provider of the FKS integrand of madspace: the matrix-element
blocks that FKSCrossSection takes, from the Fortran of an aMC@NLO output
(output DIR of a process with [QCD]), until the matrix elements of MadMatrix
and pyAmpliCol give them through UMAMI. The fixed-order driver of madspace
(fks_driver.py) and the tests of madspace (madspace/tests) take them from
here.

fks_provider.f (template_files/mg7) is linked in a P* directory of the
output as check_poles (make -f makefile_provider fks_provider) and answers
requests over two pipes. FortranProvider builds and starts it for one
subprocess and evaluates batches of numpy arrays, every block at the strong
coupling g = 1 (see fks_provider.f):
- real: the real at events of one FKS configuration per split-order
  position (smatrix_real);
- born: the Born per position, its counterterm parts amp_split_cnt (the
  Born and its azimuthal part A at the positions of the counterterms, per
  correction) and amp2 of the Born diagrams of the channels (sborn);
- links: the color-linked Borns of pairs of legs of the real process of
  one configuration (sborn_sf), only for the pairs MadFKS registered, those
  of fks_j_from_i of i_fks;
- virtual: the finite part, the poles and the accuracy of MadLoop at a
  renormalization scale per point, summed over helicities;
- channels: the Born configurations of MadFKS, the diagrams of its
  integration channels.

From these it gives the blocks of madspace: the real per NLO order, the
Born per Born order, the color-correlated Borns of all pairs of Born legs
(those of the registered pairs, the diagonal from color conservation), the
spin-correlated Born B_perp = B / 2 + Re(conj(phase) A) of a gluon or photon
emitter with k_perp, and the virtual per NLO order. The phase is that of
k_perp about the emitter in the helicity basis of MadGraph, -exp(2 i (psi +
phi)) with psi the azimuth of k_perp in that basis and phi that of the
emitter, which is xij_aor of genps_fks.f for xij_kperp. fks_blocks gives
them at the kinematics of FKSKinematics, by the names of FKSCrossSection.

Extra counterterms (mixed QCD-EW corrections with a second mother) and the
conversion of tagged photons in the virtual are not covered.
"""

import contextlib
import json
import os
import shutil
import subprocess
import time

import numpy as np

TEMPLATES = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
    "iolibs",
    "template_files",
    "mg7",
)
# the sources and their names in the P* directory
SOURCES = {
    "fks_provider.f": os.path.join(TEMPLATES, "fks_provider.f"),
    "makefile_provider": os.path.join(TEMPLATES, "makefile_fks_provider"),
}

QUIT, INFO, REAL, BORN, LINKS, VIRTUAL, CHANNELS = range(7)
INT = np.dtype(np.int32)
FLOAT = np.dtype(np.float64)


def build(directory, log=None):
    """Compiles fks_provider in a P* directory of an aMC@NLO output whose
    libraries are built (as a fixed-order run leaves them), returns its
    path."""
    for name, path in SOURCES.items():
        target = os.path.join(directory, name)
        with open(path) as f:
            text = f.read()
        if os.path.isfile(target):
            with open(target) as f:
                if f.read() == text:
                    continue
        with open(target, "w") as f:
            f.write(text)
    with open(log or os.devnull, "a") as out:
        subprocess.run(
            ["make", "-f", "makefile_provider", "fks_provider"],
            cwd=directory,
            stdout=out,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return os.path.join(directory, "fks_provider")


def boost_z(p, y):
    """The momenta p (..., 4) boosted along z by the rapidity y (per point)."""
    y = np.asarray(y, dtype=float).reshape(np.shape(p)[:1] + (1,) * (p.ndim - 2))
    out = np.array(p, dtype=float)
    out[..., 0] = np.cosh(y) * p[..., 0] + np.sinh(y) * p[..., 3]
    out[..., 3] = np.sinh(y) * p[..., 0] + np.cosh(y) * p[..., 3]
    return out


def partonic_rapidity(p, incoming):
    """The rapidity of the sum of the incoming momenta of p (batch, n, 4),
    zero for a decay."""
    if incoming == 1:
        return np.zeros(len(p))
    total = p[:, 0] + p[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        y = 0.5 * np.log((total[:, 0] + total[:, 3]) / (total[:, 0] - total[:, 3]))
    return np.where(np.isfinite(y), y, 0.0)


def azimuthal_phase(mother, k_perp):
    """-exp(2 i (psi + phi)) of k_perp (batch, 4) about the massless mother
    (batch, 4): psi is the azimuth of k_perp, without its part along the
    mother, in the helicity basis e1 = (cos th cos ph, cos th sin ph, -sin th), e2 =
    (-sin ph, cos ph, 0) of the mother direction (th, ph), phi = 0 on the
    beam axis. This is xij_aor of genps_fks.f for k_perp = xij_kperp, and
    zero where k_perp has no transverse part."""
    mother = np.asarray(mother, dtype=float)
    v = np.asarray(k_perp, dtype=float)
    v = v - (v[:, :1] / mother[:, :1]) * mother
    n = mother[:, 1:] / np.linalg.norm(mother[:, 1:], axis=1, keepdims=True)
    cth = n[:, 2]
    sth = np.sqrt(np.maximum(1 - cth**2, 0.0))
    on_axis = sth == 0
    safe = np.where(on_axis, 1.0, sth)
    cph = np.where(on_axis, 1.0, n[:, 0] / safe)
    sph = np.where(on_axis, 0.0, n[:, 1] / safe)
    e1 = np.stack([cth * cph, cth * sph, -sth], axis=1)
    e2 = np.stack([-sph, cph, np.zeros_like(cph)], axis=1)
    a = (v[:, 1:] * e1).sum(1)
    b = (v[:, 1:] * e2).sum(1)
    norm = np.hypot(a, b)
    z = (a + 1j * b) / np.where(norm > 0, norm, 1.0)
    return np.where(norm > 0, -(z**2) * (cph + 1j * sph) ** 2, 0.0)


class FortranProvider:
    """The matrix-element blocks of one subprocess (a P* directory of an
    aMC@NLO output) from its Fortran, at g = 1. Use it as a context manager
    or call close. statistics counts per block (real, born, links, virtual)
    the requests, the points, the evaluations (points times pairs for the
    links) and the seconds they took, the time of the pipes included."""

    def __init__(self, directory, rebuild=True, log=None):
        self.directory = os.path.realpath(directory)
        self.log = log or os.path.join(self.directory, "fks_provider.log")
        executable = os.path.join(self.directory, "fks_provider")
        if rebuild or not os.path.isfile(executable):
            executable = build(self.directory, self.log)
        with open(os.path.join(self.directory, "fks_configurations.json")) as f:
            self.fks_configurations = json.load(f)
        self.configurations = self.fks_configurations["configurations"]
        if self.fks_configurations["extra_counterterms"]:
            raise NotImplementedError("extra counterterms are not covered")
        requests, child_in = os.pipe()
        child_out, answers = os.pipe()
        with open(self.log, "a") as out:
            self._process = subprocess.Popen(
                [executable, str(requests), str(answers)],
                cwd=self.directory,
                pass_fds=(requests, answers),
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=subprocess.STDOUT,
            )
        os.close(requests)
        os.close(answers)
        self._send = os.fdopen(child_in, "wb", buffering=0)
        self._receive = os.fdopen(child_out, "rb", buffering=0)
        self.statistics = {
            name: dict(requests=0, points=0, evaluations=0, seconds=0.0)
            for name in ("real", "born", "links", "virtual")
        }
        self._info()

    @contextlib.contextmanager
    def _timed(self, name, points, evaluations=None):
        start = time.perf_counter()
        yield
        entry = self.statistics[name]
        entry["requests"] += 1
        entry["points"] += points
        entry["evaluations"] += points if evaluations is None else evaluations
        entry["seconds"] += time.perf_counter() - start

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self._process.poll() is None:
            try:
                self._send.write(np.array([QUIT, 0, 0], INT).tobytes())
            except BrokenPipeError:
                pass
            self._process.wait(timeout=60)
        self._send.close()
        self._receive.close()

    def _read(self, dtype, count):
        size = dtype.itemsize * count
        data = bytearray()
        while len(data) < size:
            chunk = self._receive.read(size - len(data))
            if not chunk:
                raise RuntimeError(
                    f"fks_provider in {self.directory} stopped, see {self.log}"
                )
            data += chunk
        return np.frombuffer(bytes(data), dtype)

    def _request(self, command, count, argument, *payload):
        message = np.array([command, count, argument], INT).tobytes()
        message += b"".join(np.ascontiguousarray(a).tobytes() for a in payload)
        self._send.write(message)
        (status,) = self._read(INT, 1)
        if status != 0:
            raise RuntimeError(f"fks_provider: status {status} for {command}")

    def _info(self):
        self._request(INFO, 0, 0)
        sizes = self._read(INT, 9)
        (
            self.particle_count,
            self.incoming,
            self.size,
            self.born_size,
            self.split_order_count,
            qcd_pos,
            qed_pos,
            self.configuration_count,
            self.channel_count,
        ) = (int(v) for v in sizes)
        self.qcd_pos, self.qed_pos = qcd_pos - 1, qed_pos - 1
        k = self.split_order_count
        table = self._read(INT, (self.size + 2) * k).reshape(-1, k)
        self.positions = [tuple(int(v) for v in row) for row in table[: self.size]]
        self.born_orders = tuple(int(v) for v in table[self.size])
        self.nlo_orders = tuple(int(v) for v in table[self.size + 1])
        if self.configuration_count != len(self.configurations):
            raise ValueError("fks_configurations.json and the Fortran differ")

    def _momenta(self, momenta, legs):
        momenta = np.ascontiguousarray(momenta, dtype=float)
        if momenta.ndim != 3 or momenta.shape[1:] != (legs, 4):
            raise ValueError(f"momenta of shape (batch, {legs}, 4) expected")
        return momenta

    def _by_configuration(self, fks_index, evaluate):
        """evaluate(index of the points, configuration) per configuration,
        the results scattered back."""
        fks_index = np.asarray(fks_index, dtype=int)
        out = None
        for c in np.unique(fks_index):
            if not 0 <= c < self.configuration_count:
                raise ValueError(f"no FKS configuration {c}")
            points = np.flatnonzero(fks_index == c)
            value = evaluate(points, int(c))
            if out is None:
                out = np.zeros((len(fks_index),) + value.shape[1:], value.dtype)
            out[points] = value
        return out

    def real(self, momenta, fks_index):
        """The real (batch, positions) at events (batch, n + 1, 4) of the
        configurations fks_index (from 0)."""
        momenta = self._momenta(momenta, self.particle_count)

        def evaluate(points, c):
            with self._timed("real", len(points)):
                self._request(REAL, len(points), c + 1, momenta[points])
                out = self._read(FLOAT, len(points) * self.size)
            return out.reshape(-1, self.size)

        return self._by_configuration(fks_index, evaluate)

    def born(self, momenta, fks_index):
        """The Born at Born kinematics (batch, n, 4) with the azimuthal part
        of the configurations fks_index: the Born per position (batch,
        positions), the counterterm parts (batch, split orders, 2, positions)
        (the Born and its azimuthal part per correction, complex) and amp2
        of the Born diagrams of the channels (batch, channels)."""
        momenta = self._momenta(momenta, self.particle_count - 1)
        k = self.split_order_count
        counterterm = 4 * self.size * k
        width = self.size + counterterm + self.channel_count

        def evaluate(points, c):
            with self._timed("born", len(points)):
                self._request(BORN, len(points), c + 1, momenta[points])
                out = self._read(FLOAT, len(points) * width)
            return out.reshape(-1, width)

        out = self._by_configuration(fks_index, evaluate)
        parts = out[:, self.size : self.size + counterterm].copy()
        parts = parts.view(np.complex128).reshape(-1, k, 2, self.size)
        return out[:, : self.size], parts, out[:, self.size + counterterm :]

    def links(self, momenta, configuration, pairs):
        """sborn_sf (batch, pairs, positions) at Born kinematics for pairs
        (m, n) of legs of the real process of configuration (from 0), from 1
        as in MadFKS."""
        momenta = self._momenta(momenta, self.particle_count - 1)
        pairs = np.asarray(pairs, dtype=INT).reshape(-1, 2)
        allowed = self.link_pairs(configuration)
        for m, n in pairs:
            if (int(m), int(n)) not in allowed and (int(n), int(m)) not in allowed:
                raise ValueError(f"MadFKS registered no link ({m}, {n})")
        count = len(momenta)
        with self._timed("links", count, count * len(pairs)):
            self._request(
                LINKS,
                count,
                len(pairs),
                np.array([configuration + 1], INT),
                pairs,
                momenta,
            )
            out = self._read(FLOAT, count * len(pairs) * self.size)
        return out.reshape(count, len(pairs), self.size)

    def virtual(self, momenta, ren_scale):
        """MadLoop at Born kinematics (batch, n, 4) and the renormalization
        scale per point: the finite part, the single and double poles and
        the accuracy per position (each (batch, positions)) and the return
        code of MadLoop (batch,)."""
        momenta = self._momenta(momenta, self.particle_count - 1)
        scale = np.ascontiguousarray(np.broadcast_to(ren_scale, len(momenta)), float)
        count = len(momenta)
        with self._timed("virtual", count):
            self._request(VIRTUAL, count, 0, scale, momenta)
            codes = self._read(INT, count)
            out = self._read(FLOAT, count * self.size * 4)
        out = out.reshape(count, self.size, 4)
        parts = ("finite", "single", "double", "accuracy")
        return {name: out[..., k] for k, name in enumerate(parts)}, codes

    def channels(self):
        """The Born configurations of MadFKS (born_conf.inc), per
        configuration its Born diagram, its forest (per propagator, from -1,
        the propagator, its two daughters, sprop and tprid) and the masses
        and widths of the propagators."""
        self._request(CHANNELS, 0, 0)
        count, branches = (int(v) for v in self._read(INT, 2))
        rows = self._read(INT, count * (1 + 4 * branches)).reshape(count, -1)
        props = self._read(FLOAT, 2 * branches * count).reshape(count, branches, 2)
        return [
            dict(
                diagram=int(row[0]),
                forest=[
                    [-(b + 1)] + [int(v) for v in row[1 + 4 * b : 5 + 4 * b]]
                    for b in range(branches)
                ],
                props=props[c].tolist(),
            )
            for c, row in enumerate(rows)
        ]

    # the layout of madspace

    def position(self, orders):
        """The split-order position (from 0) of orders."""
        return self.positions.index(tuple(orders))

    @property
    def orders(self):
        return self.fks_configurations["orders"]

    def counterterm_position(self, born_order, correction):
        """The position of a Born order raised by the coupling of a
        correction (an index of the split orders)."""
        raised = list(born_order)
        raised[correction] += 2
        return self.position(raised)

    def link_pairs(self, configuration):
        """The pairs (m, n) of legs of the real process (from 1) that
        sbornsoft takes at a configuration (from 0): those of fks_j_from_i
        of i_fks without i_fks, each pair once, none for a configuration
        without links."""
        c = self.configurations[configuration]
        if not (c["need_color_links"] or c["need_charge_links"]):
            return []
        legs = c["fks_j_from_i"][c["i_fks"] - 1]
        return [
            (m, n)
            for k, m in enumerate(legs)
            for n in legs[: k + 1]
            if c["i_fks"] not in (m, n)
        ]

    def real_orders(self, momenta, fks_index):
        """The real per NLO order (batch, o)."""
        real = self.real(momenta, fks_index)
        return real[:, [self.position(o) for o in self.orders["nlo"]]]

    def born_orders_of(self, born):
        """The Born per Born order (batch, b) from the Born per position."""
        return born[:, [self.position(o) for o in self.orders["born"]]]

    def color_correlated(self, momenta):
        """<T_m . T_n> B per Born order (batch, b, n, n) at Born kinematics:
        sborn_sf over the pairs of Born legs that the configurations with
        color links register (each from the first of them), the diagonal
        from color conservation. Every pair of colored Born legs must be
        among them."""
        n = self.particle_count - 1
        qcd = self.qcd_pos
        if qcd < 0:
            raise ValueError("no QCD in the split orders")
        positions = [self.counterterm_position(o, qcd) for o in self.orders["born"]]
        out = np.zeros((len(momenta), len(positions), n, n))
        seen = set()
        for c, configuration in enumerate(self.configurations):
            if not configuration["need_color_links"]:
                continue
            i_fks = configuration["i_fks"]
            pairs, born_pairs = [], []
            for m, k in self.link_pairs(c):
                pair = tuple(sorted(v - 1 - (v > i_fks) for v in (m, k)))
                if m == k or pair in seen:
                    continue
                seen.add(pair)
                pairs.append((m, k))
                born_pairs.append(pair)
            if not pairs:
                continue
            links = self.links(momenta, c, pairs)[:, :, positions]
            for l, (a, b) in enumerate(born_pairs):
                out[:, :, a, b] = out[:, :, b, a] = links[:, l]
        colors = np.array(self.configurations[0]["real_colors"])
        colored = np.flatnonzero(
            np.delete(colors, self.configurations[0]["i_fks"] - 1) != 1
        )
        for k, a in enumerate(colored):
            for b in colored[:k]:
                if (b, a) not in seen:
                    raise ValueError(f"no color link of the Born legs {b}, {a}")
        for m in range(n):
            out[:, :, m, m] = -out[:, :, m].sum(-1)
        return out

    def spin_correlated(self, momenta, k_perp, fks_index, born=None):
        """B_perp per Born order (batch, b) at Born kinematics with k_perp
        (batch, 4) in the same frame, for the emitter of the configurations
        fks_index, a gluon or photon: B / 2 + Re(conj(phase) A) with A the
        azimuthal part of sborn at the counterterm of the correction of the
        emitter. B / 2 for other emitters and where k_perp is zero."""
        b_orders = self.orders["born"]
        momenta = self._momenta(momenta, self.particle_count - 1)
        k_perp = np.asarray(k_perp, dtype=float)
        if born is None:
            born = self.born(momenta, fks_index)
        per_position, parts, _ = born
        plain = self.born_orders_of(per_position)
        out = plain / 2
        fks_index = np.asarray(fks_index, dtype=int)
        for c in np.unique(fks_index):
            configuration = self.configurations[c]
            mother = {21: self.qcd_pos, 22: self.qed_pos}.get(configuration["ij_pdg"])
            if mother is None or mother < 0:
                continue
            points = np.flatnonzero(fks_index == c)
            i_fks, j_fks = configuration["i_fks"], configuration["j_fks"]
            leg = j_fks - 1 - (j_fks > i_fks)
            phase = azimuthal_phase(momenta[points, leg], k_perp[points])
            for o, order in enumerate(b_orders):
                pos = self.counterterm_position(order, mother)
                a = parts[points, mother, 1, pos]
                out[points, o] = plain[points, o] / 2 + (np.conj(phase) * a).real
        return out

    def virtual_orders(self, momenta, ren_scale):
        """The finite part of the virtual per NLO order (batch, o) at the
        renormalization scale."""
        values, _ = self.virtual(momenta, ren_scale)
        return values["finite"][:, [self.position(o) for o in self.orders["nlo"]]]


def fks_blocks(
    provider,
    kinematics,
    fks_index,
    k_perp,
    ren_scale=None,
    real_mask=None,
    born_mask=None,
    virtual_mask=None,
):
    """The inputs of FKSCrossSection from matrix elements at the kinematics
    of FKSKinematics (its outputs by name, in the lab frame) with k_perp of
    FKSSplittingKernels at the collinear counter-event: matrix_element
    (the real at the event, where its weight is not zero), born,
    color_correlated and spin_correlated (at the Born, where the Born
    weight is not zero) and, with ren_scale, virtual, and amp2 of the Born
    diagrams of the channels (zero where the Born weight is). real_mask and
    born_mask leave out the points where they are false, for example those
    that the cuts reject, and virtual_mask the points of the Born without the
    virtual (zero there), for the virtual on a fraction of the points. The
    momenta go to the partonic center-of-mass frame of their kinematics
    first, k_perp with the Born."""
    incoming = provider.incoming
    fks_index = np.asarray(fks_index, dtype=int)
    count = len(fks_index)
    o, b = len(provider.orders["nlo"]), len(provider.orders["born"])
    n = provider.particle_count - 1
    blocks = dict(
        matrix_element=np.zeros((count, o)),
        born=np.zeros((count, b)),
        color_correlated=np.zeros((count, b, n, n)),
        spin_correlated=np.zeros((count, b)),
        amp2=np.zeros((count, provider.channel_count)),
    )
    if ren_scale is not None:
        blocks["virtual"] = np.zeros((count, o))
    event = np.asarray(kinematics["weight"]) != 0
    if real_mask is not None:
        event &= np.asarray(real_mask, dtype=bool)
    event = np.flatnonzero(event)
    if len(event):
        real = np.asarray(kinematics["momenta"])[event]
        real = boost_z(real, -partonic_rapidity(real, incoming))
        blocks["matrix_element"][event] = provider.real_orders(real, fks_index[event])
    valid = np.asarray(kinematics["born_weight"]) != 0
    if born_mask is not None:
        valid &= np.asarray(born_mask, dtype=bool)
    valid = np.flatnonzero(valid)
    if len(valid):
        born = np.asarray(kinematics["born_momenta"])[valid]
        y = -partonic_rapidity(born, incoming)
        born = boost_z(born, y)
        direction = boost_z(np.asarray(k_perp)[valid][:, None], y)[:, 0]
        evaluated = provider.born(born, fks_index[valid])
        blocks["born"][valid] = provider.born_orders_of(evaluated[0])
        blocks["amp2"][valid] = evaluated[2]
        blocks["color_correlated"][valid] = provider.color_correlated(born)
        blocks["spin_correlated"][valid] = provider.spin_correlated(
            born, direction, fks_index[valid], born=evaluated
        )
        if ren_scale is not None:
            take = np.ones(len(valid), dtype=bool)
            if virtual_mask is not None:
                take = np.asarray(virtual_mask, dtype=bool)[valid]
            if take.any():
                scale = np.broadcast_to(ren_scale, count)[valid][take]
                virtual = provider.virtual_orders(born[take], scale)
                blocks["virtual"][valid[take]] = virtual
    return blocks


def providers(output, rebuild=True):
    """A FortranProvider per subprocess of an aMC@NLO output, by the name of
    its P* directory."""
    if shutil.which("make") is None:
        raise RuntimeError("make is needed to build fks_provider")
    root = os.path.join(output, "SubProcesses")
    names = sorted(d for d in os.listdir(root) if d.startswith("P"))
    return {name: FortranProvider(os.path.join(root, name), rebuild) for name in names}
