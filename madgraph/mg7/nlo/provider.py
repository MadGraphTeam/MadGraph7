"""The Fortran provider of the FKS integrand of madspace: the matrix-element
blocks that FKSCrossSection takes, from the Fortran of an aMC@NLO output
(output DIR of a process with [QCD]), until the matrix elements of MadMatrix
and pyAmpliCol give them through UMAMI. The fixed-order drivers of madspace
(nlo/fks.py, nlo/cs.py) take them as UMAMI libraries (nlo/libraries.py),
the tests of madspace (madspace/tests, with the Python reference of the
drivers in nlo_reference) also from here directly.

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
  renormalization scale per point, summed over helicities, with the change
  of scheme of the couplings of tagged photons as BinothLHA adds it;
- channels: the Born configurations of MadFKS, the diagrams of its
  integration channels;
- extra: the counterterm parts of the Born of the extra counterterm of a
  configuration (extra_cnt), the Born of the other mother (a gluon or a
  photon) of a quark pair;
- couplings: the number of tagged photons, the QED coupling of the
  counterterms (gal(1)^2 / (4 pi)) and the factors by which add_wgt
  multiplies the contributions of tagged photons per power of e;
- reference_scales: the reference scale of the user (user_dynamical_scale of
  dummy_fct.f, dynamical_scale_choice 0 or 10) at kinematics of the real
  process, which madspace takes as an argument;
- ew_sudakov: with --ewsudakov, the matrix elements of the Sudakov
  approximation of the EW corrections per helicity combination of the Born,
  those that sudakov_wrapper of ewsudakov_wrapper.f takes, as the inputs of
  EWSudakov of madspace (fks_provider_ewsudakov.f, with the calls to the
  amplitudes of the process that build writes from fks_configurations.json),
  and for the tests the weights of compute_ewsudakov of fks_singular.f.

From these it gives the blocks of madspace: the real per NLO order, the
Born per Born order, the color-correlated Borns of all pairs of Born legs
(those of the registered pairs, the diagonal from color conservation), the
spin-correlated Born B_perp = B / 2 + Re(conj(phase) A) of a gluon or photon
emitter with k_perp, both also for the Born of the extra counterterm of the
configuration of a point (born_inputs), and the virtual per NLO order. The phase is that of
k_perp about the emitter in the helicity basis of MadGraph, -exp(2 i (psi +
phi)) with psi the azimuth of k_perp in that basis and phi that of the
emitter, which is xij_aor of genps_fks.f for xij_kperp. fks_blocks gives
them at the kinematics of FKSKinematics, by the names of FKSCrossSection.

born_cnt_<n>.f of MadFKS divides the Born of an extra counterterm by the
identical-particle factor of the Born of the subprocess, born_inputs gives it
with its own, as FKSLimits takes it.

With tagged photons and QED corrections MadFKS takes the couplings of the
tagged photons in the alpha(0) scheme and the others in the G_mu scheme: the
virtual changes the scheme of the Born (get_virtual_a0Gmu_conv), and every
contribution takes a factor of its power of alpha (get_rescale_alpha_factor),
a power of the ratio of the two alphas. tagged_factors gives them as factors
of the blocks per order and of the QED coupling of the counterterms, which
the counterterms of a QED splitting take with one power of alpha more than
their Born.
"""

import contextlib
import json
import math
import os
import shutil
import subprocess
import time

import numpy as np

# madgraph/iolibs/template_files/mg7, from madgraph/mg7/nlo
TEMPLATES = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))),
    "iolibs",
    "template_files",
    "mg7",
)
# the sources and their names in the P* directory
SOURCES = {
    "fks_provider.f": os.path.join(TEMPLATES, "fks_provider.f"),
    "makefile_provider": os.path.join(TEMPLATES, "makefile_fks_provider"),
}
# the Sudakov approximation of the EW corrections, with and without
# --ewsudakov
EWSUDAKOV_SOURCES = {
    True: os.path.join(TEMPLATES, "fks_provider_ewsudakov.f"),
    False: os.path.join(TEMPLATES, "fks_provider_ewsudakov_dummy.f"),
}

(
    QUIT,
    INFO,
    REAL,
    BORN,
    LINKS,
    VIRTUAL,
    CHANNELS,
    EXTRA,
    COUPLINGS,
    SCALE,
    EWSUDAKOV_INFO,
    EWSUDAKOV,
    EWSUDAKOV_WEIGHTS,
) = range(13)
INT = np.dtype(np.int32)
FLOAT = np.dtype(np.float64)


def ewsudakov_calls(info):
    """fks_provider_ewsudakov_calls.f for the ewsudakov entry info of
    fks_configurations.json (see get_ewsudakov_info of export_fks.py): the
    calls to the base amplitudes (the Born per helicity, sborn_onehel, and
    the Goldstone amplitudes ewsdk_gold_me_<k>), to their parameter
    renormalisation and to the interferences ewsdk_me_<t>, by number, the
    base amplitude of a helicity combination (the first Goldstone amplitude
    whose replaced particles all have helicity 0, the Born where none has)
    and of each interference, and comp_idfac of a base amplitude, as
    sudakov_wrapper takes them. A comment alone without info."""
    lines = [
        "c The amplitudes of the Sudakov approximation of the EW corrections",
        "c of the process for fks_provider_ewsudakov.f, which",
        "c madgraph/mg7/nlo/provider.py writes from fks_configurations.json.",
    ]
    if info is None:
        return "\n".join(lines) + "\n"
    goldstone = info["goldstone"]
    interference = info["interference"]

    def continued(head, items, tail=""):
        """head, the items and tail over continuation lines of fixed form."""
        out, line = [], head
        for item in [*items, tail]:
            if len(line) + len(item) > 72:
                out.append(line)
                line = "     $     "
            line += item
        out.append(line)
        return out

    lines += [
        "      subroutine provider_sdk_base_me(base,p,hels,ihel)",
        "      implicit none",
        "      include 'nexternal.inc'",
        "      integer base,hels(nexternal-1),ihel",
        "      double precision p(0:3,nexternal-1),ans",
        "      if (base.eq.0) call sborn_onehel(p,hels,ihel,ans)",
    ]
    for k in range(1, len(goldstone) + 1):
        lines.append(f"      if (base.eq.{k}) call ewsdk_gold_me_{k}(p,hels,ans)")
    lines += [
        "      end",
        "",
        "",
        "      subroutine provider_sdk_par_ren(base,p,hels,ihel,invariants)",
        "      implicit none",
        "      include 'nexternal.inc'",
        "      integer base,hels(nexternal-1),ihel",
        "      double precision p(0:3,nexternal-1)",
        "      double precision invariants(nexternal-1,nexternal-1)",
        "      if (base.eq.0) call par_ren_sborn_onehel(p,hels,ihel,",
        "     $     invariants)",
    ]
    for k in range(1, len(goldstone) + 1):
        lines += [
            f"      if (base.eq.{k}) call par_ren_ewsdk_gold_me_{k}(p,hels,",
            "     $     ihel,invariants)",
        ]
    lines += [
        "      end",
        "",
        "",
        "      subroutine provider_sdk_interference(t,p,hels)",
        "      implicit none",
        "      include 'nexternal.inc'",
        "      integer t,hels(nexternal-1)",
        "      double precision p(0:3,nexternal-1)",
        "      double complex ans",
    ]
    for t in range(1, len(interference) + 1):
        lines.append(f"      if (t.eq.{t}) call ewsdk_me_{t}(p,hels,ans)")
    lines += [
        "      end",
        "",
        "",
        "      integer function provider_sdk_interference_count()",
        "      implicit none",
        f"      provider_sdk_interference_count={len(interference)}",
        "      end",
        "",
        "",
        "      integer function provider_sdk_interference_base(t)",
        "      implicit none",
        "      integer t",
        f"      integer bases({max(len(interference), 1)})",
    ]
    bases = [amp["base"] for amp in interference] or [0]
    items = [f"{b}," for b in bases[:-1]] + [str(bases[-1])]
    lines += continued("      data bases /", items, "/")
    lines += [
        "      provider_sdk_interference_base=bases(t)",
        "      end",
        "",
        "",
        "      integer function provider_sdk_base(hels)",
        "      implicit none",
        "      include 'nexternal.inc'",
        "      integer hels(nexternal-1)",
        "      provider_sdk_base=0",
    ]
    for k, amp in enumerate(goldstone, start=1):
        items = [f".and.hels({leg}).eq.0" for leg in amp["legs"]]
        tail = f") provider_sdk_base={k}"
        lines += continued("      if (provider_sdk_base.eq.0", items, tail)
    lines += [
        "      end",
        "",
        "",
        "      double precision function provider_sdk_idfac(base)",
        "      implicit none",
        "      integer base",
        "      double precision compensate_identical_factor",
        "      external compensate_identical_factor",
        "      provider_sdk_idfac=1d0",
        "      if (base.gt.0) provider_sdk_idfac=",
        "     $     compensate_identical_factor(base)",
        "      end",
    ]
    return "\n".join(lines) + "\n"


def write_if_changed(path, text):
    if os.path.isfile(path):
        with open(path) as f:
            if f.read() == text:
                return
    with open(path, "w") as f:
        f.write(text)


def build(directory, log=None):
    """Compiles fks_provider in a P* directory of an aMC@NLO output whose
    libraries are built (as a fixed-order run leaves them), returns its
    path."""
    for name, path in SOURCES.items():
        with open(path) as f:
            write_if_changed(os.path.join(directory, name), f.read())
    with open(os.path.join(directory, "fks_configurations.json")) as f:
        info = json.load(f).get("ewsudakov")
    with open(EWSUDAKOV_SOURCES[info is not None]) as f:
        write_if_changed(os.path.join(directory, "fks_provider_ewsudakov.f"), f.read())
    write_if_changed(
        os.path.join(directory, "fks_provider_ewsudakov_calls.f"),
        ewsudakov_calls(info),
    )
    with open(log or os.devnull, "a") as out:
        subprocess.run(
            ["make", "-f", "makefile_provider", "fks_provider"],
            cwd=directory,
            stdout=out,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return os.path.join(directory, "fks_provider")


def momenta_at_rest(p):
    """The momenta (batch, 3, 4) of a 2 > 1 process in its partonic frame
    exactly: the incoming along the beam with the energies of their masses
    at s = (p_1 + p_2)^2, the final-state particle at rest."""
    p = np.asarray(p, dtype=float)

    def square(q):
        return q[:, 0] ** 2 - (q[:, 1:] ** 2).sum(-1)

    s = np.maximum(square(p[:, 0] + p[:, 1]), 0.0)
    masses = [np.maximum(square(p[:, k]), 0.0) for k in range(2)]
    rs = np.sqrt(s)
    safe = np.where(rs > 0, rs, 1.0)
    e1 = (s + masses[0] - masses[1]) / (2 * safe)
    e2 = (s - masses[0] + masses[1]) / (2 * safe)
    k = np.sqrt(np.maximum(e1 * e1 - masses[0], 0.0))
    out = np.zeros_like(p)
    out[:, 0, 0], out[:, 0, 3] = e1, k
    out[:, 1, 0], out[:, 1, 3] = e2, -k
    out[:, 2, 0] = rs
    return out


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


def helicity_phase(mother, k_perp):
    """-exp(2 i psi) of k_perp (batch, 4) about the massless mother (batch,
    4): psi is the azimuth of k_perp, without its part along the mother, in
    the helicity basis e1 = (cos th cos ph, cos th sin ph, -sin th), e2 =
    (-sin ph, cos ph, 0) of the mother direction (th, ph), phi = 0 on the
    beam axis. This is azifact_from_kperp of boost_to_frame.f, and zero where
    k_perp has no transverse part. It comes with exp(i ph), the azimuth of
    the mother (one on the beam axis)."""
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
    return np.where(norm > 0, -(z**2), 0.0), cph + 1j * sph


def azimuthal_phase(mother, k_perp):
    """-exp(2 i (psi + phi)) of k_perp (batch, 4) about the massless mother
    (batch, 4), with psi of helicity_phase and phi the azimuth of the
    mother. This is xij_aor of genps_fks.f for k_perp = xij_kperp, and zero
    where k_perp has no transverse part."""
    phase, azimuth = helicity_phase(mother, k_perp)
    return phase * azimuth**2


class FortranProvider:
    """The matrix-element blocks of one subprocess (a P* directory of an
    aMC@NLO output) from its Fortran, at g = 1. Use it as a context manager
    or call close. statistics counts per block (real, born, links, virtual,
    ew_sudakov)
    the requests, the points, the evaluations (points times pairs for the
    links) and the seconds they took, the time of the pipes included, the
    Borns of the extra counterterms with those of the Born."""

    def __init__(self, directory, rebuild=True, log=None):
        self.directory = os.path.realpath(directory)
        self.log = log or os.path.join(self.directory, "fks_provider.log")
        executable = os.path.join(self.directory, "fks_provider")
        if rebuild or not os.path.isfile(executable):
            executable = build(self.directory, self.log)
        with open(os.path.join(self.directory, "fks_configurations.json")) as f:
            self.fks_configurations = json.load(f)
        self.configurations = self.fks_configurations["configurations"]
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
            for name in ("real", "born", "links", "virtual", "ew_sudakov")
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
        if legs == 3 and self.incoming == 2:
            # one final-state particle, at rest in the partonic frame: the
            # helicity vectors of MadGraph follow the direction of its
            # three-momentum, which rounding would choose, and the helicities
            # that the filter of the matrix element dropped at the first
            # points would then be missing (a W of u d~ > w+ loses 3/4 of its
            # Born for a direction across the beam, all of it for the
            # opposite one). The points take the partonic frame exactly, the
            # particle at rest and the incoming ones along the beam.
            momenta = momenta_at_rest(momenta)
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

    def extra(self, momenta, fks_index):
        """The counterterm parts of the Born of the extra counterterm of the
        configurations fks_index at Born kinematics (batch, n, 4), as those
        of born (batch, split orders, 2, positions), zero for the points of a
        configuration without one."""
        momenta = self._momenta(momenta, self.particle_count - 1)
        k = self.split_order_count
        width = 4 * self.size * k

        def evaluate(points, c):
            if self.configurations[c].get("extra_cnt_index", -1) < 0:
                return np.zeros((len(points), width))
            with self._timed("born", len(points)):
                self._request(EXTRA, len(points), c + 1, momenta[points])
                out = self._read(FLOAT, len(points) * width)
            return out.reshape(-1, width)

        out = self._by_configuration(fks_index, evaluate)
        return (
            np.ascontiguousarray(out).view(np.complex128).reshape(-1, k, 2, self.size)
        )

    def reference_scales(self, momenta, valid=None):
        """The reference scale of the user (user_dynamical_scale) at
        kinematics (batch, n + 1, 4) of the real process or (batch, n, 4) of
        the Born, which gets a last particle without momentum (as the soft
        counter-event), boosted to their partonic center-of-mass frame, zero
        where valid is false."""
        momenta = np.asarray(momenta, dtype=float)
        if momenta.ndim == 3 and momenta.shape[1] == self.particle_count - 1:
            momenta = np.concatenate([momenta, np.zeros_like(momenta[:, :1])], 1)
        momenta = self._momenta(momenta, self.particle_count)
        out = np.zeros(len(momenta))
        points = np.arange(len(momenta))
        if valid is not None:
            points = np.flatnonzero(np.asarray(valid, dtype=bool))
        if len(points):
            p = momenta[points]
            p = boost_z(p, -partonic_rapidity(p, self.incoming))
            self._request(SCALE, len(points), 0, p)
            out[points] = self._read(FLOAT, len(points))
        return out

    def couplings(self, powers=()):
        """The number of tagged photons, the QED coupling of the counterterms
        (gal(1)^2 / (4 pi) of the model) and the factors of the
        contributions of the given powers of e (get_rescale_alpha_factor), 1
        without tagged photons."""
        powers = np.ascontiguousarray(powers, dtype=INT).reshape(-1)
        self._request(COUPLINGS, len(powers), 0, powers)
        (tagged,) = self._read(INT, 1)
        out = self._read(FLOAT, len(powers) + 1)
        return int(tagged), float(out[0]), out[1:]

    def tagged_factors(self):
        """With tagged photons and QED corrections, the factors of the
        contributions of MadFKS as factors of the blocks: per NLO order (the
        real and the virtual), per Born input (the Born and its correlated
        versions) and of the QED coupling of the counterterms, the ratio of
        the factors of two powers of alpha. A QED counterterm then takes the
        factor of its NLO order, its Born that of one power of alpha less.
        None without tagged photons or QED corrections."""
        qed = self.qed_pos
        splittings = {t for c in self.configurations for t in c["splitting_type"]}
        if qed < 0 or "QED" not in splittings:
            return None
        nlo = [o[qed] for o in self.orders["nlo"]]
        inputs = [o[qed] for o in self.orders["born"]]
        inputs += [o[qed] for e in self.orders.get("extra_born", []) for o in e]
        tagged, _, factors = self.couplings(nlo + inputs + [0, 2])
        if tagged == 0:
            return None
        n = len(nlo)
        return dict(
            nlo=factors[:n],
            born=factors[n:-2],
            alpha=factors[-1] / factors[-2],
        )

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

    def ew_sudakov_info(self):
        """With --ewsudakov, the Sudakov approximation of the EW corrections
        of the Fortran: the numbers of helicity combinations of the Born and
        of interferences, has_lo1 and has_lo2, the base amplitude of each
        helicity combination and of each interference (0 for the Born, k for
        the Goldstone amplitude k) and the parameters of the logarithms (mw,
        mz, mh, mt, mb, gal, aewm1, ee, gf), None without."""
        if not hasattr(self, "_ew_sudakov_info"):
            self._request(EWSUDAKOV_INFO, 0, 0)
            present, h, t, lo1, lo2 = (int(v) for v in self._read(INT, 5))
            info = None
            if present:
                bases = self._read(INT, h + t)
                values = self._read(FLOAT, 9)
                names = ("mw", "mz", "mh", "mt", "mb", "gal", "aewm1", "ee", "gf")
                info = dict(
                    helicity_count=h,
                    interference_count=t,
                    has_lo1=bool(lo1),
                    has_lo2=bool(lo2),
                    bases=[int(b) for b in bases[:h]],
                    interference_bases=[int(b) for b in bases[h:]],
                    parameters=dict(zip(names, (float(v) for v in values))),
                )
            self._ew_sudakov_info = info
        return self._ew_sudakov_info

    def _ew_sudakov_momenta(self, momenta):
        """Born kinematics (batch, n, 4) in their partonic center-of-mass
        frame, where compute_ewsudakov takes p_born."""
        momenta = self._momenta(momenta, self.particle_count - 1)
        return np.ascontiguousarray(
            boost_z(momenta, -partonic_rapidity(momenta, self.incoming))
        )

    def ew_sudakov(self, momenta, fks_index):
        """The matrix elements of the Sudakov approximation that
        sudakov_wrapper takes at Born kinematics (batch, n, 4) of the
        configurations fks_index, per helicity combination of the Born and
        with comp_idfac of the configuration, at g = 1: diagonal (batch, h),
        the base amplitude at the LO1, lo2 (batch, h), at the LO2,
        derivatives (batch, h, 6), its derivatives with respect to the
        parameters of numder_*.f, and interference (batch, h, t), complex,
        the interferences at the LO1, zero for those of another base
        amplitude.

        Raises:
            ValueError: for an output without --ewsudakov.
        """
        info = self.ew_sudakov_info()
        if info is None:
            raise ValueError(f"{self.directory} has no --ewsudakov")
        h, t = info["helicity_count"], info["interference_count"]
        width = 8 * h + 2 * t * h
        momenta = self._ew_sudakov_momenta(momenta)

        def evaluate(points, c):
            with self._timed("ew_sudakov", len(points)):
                self._request(EWSUDAKOV, len(points), c + 1, momenta[points])
                out = self._read(FLOAT, len(points) * width)
            return out.reshape(-1, width)

        out = self._by_configuration(fks_index, evaluate)
        helicities = out[:, : 8 * h].reshape(-1, h, 8)
        interference = out[:, 8 * h :].reshape(-1, h, t, 2)
        return dict(
            diagonal=helicities[..., 0],
            lo2=helicities[..., 1],
            derivatives=helicities[..., 2:],
            interference=interference[..., 0] + 1j * interference[..., 1],
        )

    def ew_sudakov_weights(self, momenta, fks_index, ren_scale):
        """The weights of compute_ewsudakov at Born kinematics (batch, n, 4)
        of the configurations fks_index and the renormalization scale of the
        LO2 part per point, without f_b / g^qcd_power, at g = 1: (batch, 2,
        2), per sud_mod the weight at the NLO order of the LO1 (the LO1 at two
        powers of QED more and the LO2 at two of QCD more) and the sum of
        those of the other split-order positions.

        Raises:
            ValueError: for an output without --ewsudakov.
        """
        if self.ew_sudakov_info() is None:
            raise ValueError(f"{self.directory} has no --ewsudakov")
        momenta = self._ew_sudakov_momenta(momenta)
        scale = np.ascontiguousarray(np.broadcast_to(ren_scale, len(momenta)), float)

        def evaluate(points, c):
            self._request(
                EWSUDAKOV_WEIGHTS, len(points), c + 1, scale[points], momenta[points]
            )
            return self._read(FLOAT, len(points) * 4).reshape(-1, 4)

        return self._by_configuration(fks_index, evaluate).reshape(-1, 2, 2)

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

    def born_input_count(self):
        """The number of Born inputs of FKSLimits: the Born orders, then the
        orders of the Born of each extra counterterm."""
        extra = self.orders.get("extra_born", [])
        return len(self.orders["born"]) + sum(len(e) for e in extra)

    def extra_columns(self, configuration):
        """The Born inputs of the extra counterterm of a configuration (from
        0): per order of its Born the column among the Born inputs, the split
        order of the second splitting type (the row of the counterterm
        parts), the position of the counterterm (None where the NLO orders do
        not reach it) and the ratio of the identical-particle factors of the
        Born of the subprocess and of the extra counterterm. None without an
        extra counterterm."""
        c = self.configurations[configuration]
        extra = c.get("extra_cnt_index", -1)
        if extra < 0:
            return None
        orders = self.orders.get("extra_born", [])
        column = len(self.orders["born"]) + sum(len(e) for e in orders[:extra])
        row = self.orders["split_orders"].index(c["splitting_type"][1])
        pdgs = self.fks_configurations["extra_counterterms"][extra]["born_pdgs"]
        factor = identical_factor(
            self.fks_configurations["born_pdgs"], self.incoming
        ) / identical_factor(pdgs, self.incoming)
        columns = []
        for k, order in enumerate(orders[extra]):
            try:
                position = self.counterterm_position(order, row)
            except ValueError:
                position = None
            columns.append((column + k, row, position, factor))
        return columns

    def color_correlated(self, momenta):
        """<T_m . T_n> B per Born order (batch, b, n, n) at Born kinematics:
        sborn_sf over the pairs of Born legs that the configurations with
        color links register (each from the first of them), the diagonal
        from color conservation. Every pair of colored Born legs must be
        among them. Without configurations with color links (the QED
        corrections alone) they are zero, as are those of the Born orders
        without a QCD correction."""
        n = self.particle_count - 1
        qcd = self.qcd_pos
        born_orders = self.orders["born"]
        out = np.zeros((len(momenta), len(born_orders), n, n))
        if not any(c["need_color_links"] for c in self.configurations):
            return out
        if qcd < 0:
            raise ValueError("no QCD in the split orders")
        positions, taken = [], []
        for b, order in enumerate(born_orders):
            try:
                positions.append(self.counterterm_position(order, qcd))
                taken.append(b)
            except ValueError:
                pass
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
                out[:, taken, a, b] = out[:, taken, b, a] = links[:, l]
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
        fks_index, a gluon or photon: B / 2 + Re(e A) with A the azimuthal
        part of sborn at the counterterm of the correction of the emitter and
        e = -exp(2 i psi) of helicity_phase for an outgoing emitter, its
        conjugate for an incoming one (sborncol_fsr and sborncol_isr of
        fks_singular.f, the conjugation that distinguishes a timelike from a
        spacelike splitting). That is k_t B k_t / (-k_t^2), the spin tensor of
        the Born contracted with k_t transverse to the emitter. B / 2 for
        other emitters and where k_perp is zero."""
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
            phase, _ = helicity_phase(momenta[points, leg], k_perp[points])
            if leg < self.incoming:
                phase = np.conj(phase)
            for o, order in enumerate(b_orders):
                pos = self.counterterm_position(order, mother)
                a = parts[points, mother, 1, pos]
                out[points, o] = plain[points, o] / 2 + (phase * a).real
        return out

    def born_inputs(self, momenta, k_perp, fks_index, born=None):
        """The Born and B_perp per Born input (each (batch, B), see
        born_input_count) at Born kinematics with k_perp (batch, 4) in the
        same frame: per Born order those of born_orders_of and
        spin_correlated, then per order of the Born of each extra
        counterterm that of the configuration of the point, zero for the
        others. B_perp of an extra counterterm is that of spin_correlated
        with the azimuthal part of its Born at the counterterm of its
        splitting type, the mother of the other type at the same leg."""
        momenta = self._momenta(momenta, self.particle_count - 1)
        k_perp = np.asarray(k_perp, dtype=float)
        fks_index = np.asarray(fks_index, dtype=int)
        if born is None:
            born = self.born(momenta, fks_index)
        count, b = len(momenta), len(self.orders["born"])
        plain = np.zeros((count, self.born_input_count()))
        perp = np.zeros_like(plain)
        plain[:, :b] = self.born_orders_of(born[0])
        perp[:, :b] = self.spin_correlated(momenta, k_perp, fks_index, born=born)
        extra = [self.extra_columns(int(c)) for c in range(self.configuration_count)]
        if not any(extra[c] for c in np.unique(fks_index)):
            return plain, perp
        parts = self.extra(momenta, fks_index)
        for c in np.unique(fks_index):
            if not extra[c]:
                continue
            points = np.flatnonzero(fks_index == c)
            configuration = self.configurations[c]
            i_fks, j_fks = configuration["i_fks"], configuration["j_fks"]
            leg = j_fks - 1 - (j_fks > i_fks)
            phase, _ = helicity_phase(momenta[points, leg], k_perp[points])
            if leg < self.incoming:
                phase = np.conj(phase)
            for column, row, position, factor in extra[c]:
                if position is None:
                    continue
                value = factor * parts[points, row, 0, position].real
                a = factor * parts[points, row, 1, position]
                plain[points, column] = value
                perp[points, column] = value / 2 + (phase * a).real
        return plain, perp

    def virtual_orders(self, momenta, ren_scale):
        """The finite part of the virtual per NLO order (batch, o) at the
        renormalization scale."""
        values, _ = self.virtual(momenta, ren_scale)
        return values["finite"][:, [self.position(o) for o in self.orders["nlo"]]]


def ew_sudakov_inputs(provider, sudakov, momenta, fks_index):
    """The matrix-element inputs of the EWSudakov sudakov of madspace at
    Born kinematics (batch, n, 4) of the configurations fks_index from the
    provider, at g = 1: diagonal, lo2, derivatives and interference, the
    interferences of its interference_terms() (one entry of zeros without
    any)."""
    blocks = provider.ew_sudakov(momenta, fks_index)
    terms = list(sudakov.interference_terms())
    interference = np.zeros((len(momenta), max(len(terms), 1), 2))
    for k, (h, t) in enumerate(terms):
        value = blocks["interference"][:, h, t]
        interference[:, k, 0] = value.real
        interference[:, k, 1] = value.imag
    return dict(
        diagonal=blocks["diagonal"],
        lo2=blocks["lo2"],
        derivatives=np.ascontiguousarray(blocks["derivatives"]),
        interference=interference,
    )


def identical_factor(pdgs, incoming=2):
    """The identical-particle factor of the final state of a process, the
    product of k! over its flavors of k particles."""
    final = list(pdgs[incoming:])
    return float(np.prod([math.factorial(final.count(p)) for p in set(final)]))


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
    weight is not zero, per Born input, those of the extra counterterms
    included, see born_inputs) and, with ren_scale, virtual, and amp2 of
    the Born diagrams of the channels (zero where the Born weight is). real_mask and
    born_mask leave out the points where they are false, for example those
    that the cuts reject, and virtual_mask the points of the Born without the
    virtual (zero there), for the virtual on a fraction of the points. The
    momenta go to the partonic center-of-mass frame of their kinematics
    first, k_perp with the Born."""
    incoming = provider.incoming
    fks_index = np.asarray(fks_index, dtype=int)
    count = len(fks_index)
    o, b = len(provider.orders["nlo"]), len(provider.orders["born"])
    inputs = provider.born_input_count()
    n = provider.particle_count - 1
    blocks = dict(
        matrix_element=np.zeros((count, o)),
        born=np.zeros((count, inputs)),
        color_correlated=np.zeros((count, b, n, n)),
        spin_correlated=np.zeros((count, inputs)),
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
        blocks["born"][valid], blocks["spin_correlated"][valid] = provider.born_inputs(
            born, direction, fks_index[valid], born=evaluated
        )
        blocks["amp2"][valid] = evaluated[2]
        blocks["color_correlated"][valid] = provider.color_correlated(born)
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
