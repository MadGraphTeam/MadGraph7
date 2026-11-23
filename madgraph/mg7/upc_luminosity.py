"""The gamma-UPC two-photon luminosity of an mg7 run (beam.upc_flux).

madevent multiplies a photon-photon matrix element by
``photonpdfsquare(x1, x2)`` (Source/PDF/gammaUPC) when both beams are |lpp| = 2
with a gamma-UPC pdlabel. That luminosity includes the probability that the two
protons or ions do not interact hadronically, so it is a function of both
photon energies at once: it does not factorise into two beam densities.

Rather than porting gamma-UPC (and following it upstream), the Fortran is run
once, by Source/PDF/gammaUPC/upc_luminosity_table.f, to tabulate the
luminosity on a grid of both momentum fractions. madspace reads the table with
its PDF interpolation (``ms.TabulatedLuminosity``), on any device. The table is
cached next to the sources, keyed on everything it depends on.

Momentum fractions are per nucleon: an ion of energy E and mass number A is a
beam of nucleons of energy E/A, so e_cm is sqrt(s_NN) and x = A x_madevent.
"""

import contextlib
import fcntl
import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
import tempfile

logger = logging.getLogger('madevent.upc_luminosity')

# gamma-UPC's own internal grid starts at x = 1e-8 (ElasticPhotonPhotonFlux.f90,
# extended by decades on demand); the table never starts above it.
DEFAULT_XMIN = 1e-8
# knots per decade of each momentum fraction: twice gamma-UPC's nseg = 10.
# The bicubic interpolation of madspace then follows the luminosity to a few
# 1e-6 for protons. For ions it is ~1e-5 up to W_gamma-gamma ~ 100 GeV, but
# 0.1-1% for 200-500 GeV and a few % above (PbPb at 5 TeV), where the dips
# of the nuclear form factor at x > ~0.1 per nucleon are not resolved; there
# the luminosity is below 1e-8 of its total, and doubling the knots only
# halves the error.
KNOTS_PER_DECADE = 20

# compiled in this order: each module before its users
_SOURCES = [
    'tbessk.f90', 'tbessj.f90', 'elliptic_integral.f90',
    'nielsen_generalized_polylog.f90', 'photoabsorption.f90', 'nintlib.f90',
    'interpolation.f90', 'fmin_module.f90', 'OpticalGlauber_Geometry.f90',
    'ElasticPhotonPhotonFlux.f90', 'photonpdfsquare.f',
    'upc_luminosity_table.f',
]
_EXECUTABLE = 'upc_luminosity_table'


def mass_number(ion):
    """A of a (Z, N) beam."""
    return ion[0] + ion[1]


def nucleon_energies(beam_energies, ions):
    """Energy per nucleon of each beam: what the phase space sees."""
    return tuple(float(e) / mass_number(ion) for e, ion in zip(beam_energies, ions))


def lhe_beam_id(ion):
    """PDG id of a beam in the LHE <init> block, as madevent writes it
    (setrun.f): 2212 for a proton, 100ZZZAAA0 for an ion."""
    Z, N = ion
    if (Z, N) == (1, 0):
        return 2212
    if (Z, N) == (0, 1):
        return 2112
    return 1000000000 + Z * 10000 + (Z + N) * 10


def table_xmin(tau_min):
    """First knot of the table: gamma-UPC's own default, or the decade below
    the smallest tau = s_hat / s_NN the phase space can reach, which every
    momentum fraction is above (x1 x2 >= tau, x <= 1)."""
    if not tau_min > 0:
        logger.warning(
            'nothing bounds s_hat from below in this gamma-UPC run: the '
            'luminosity is tabulated down to x = 1e-10 and extrapolated below')
        tau_min = 1e-10
    return min(DEFAULT_XMIN, 10. ** math.floor(math.log10(tau_min)))


def _dependencies(gamma_dir):
    """Every file the tabulation executable is built from: the gamma-UPC
    sources, the driver, and the pdf.inc whose commons the driver fills."""
    return ([os.path.join(gamma_dir, name) for name in _SOURCES]
            + [os.path.join(os.path.dirname(gamma_dir), 'pdf.inc')])


@contextlib.contextmanager
def _locked(gamma_dir):
    """Hold the lock of this gamma-UPC directory: runs or cluster jobs
    starting together compile its objects and write its tables one at a
    time (the second then finds the first's table)."""
    os.makedirs(os.path.join(gamma_dir, 'tables'), exist_ok=True)
    with open(os.path.join(gamma_dir, 'tables', '.lock'), 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _compile(gamma_dir, fortran_compiler):
    """The tabulation executable, (re)built if a dependency or the compiler
    changed. Call it with the directory locked."""
    exe = os.path.join(gamma_dir, _EXECUTABLE)
    compiler = shutil.which(fortran_compiler or 'gfortran')
    if compiler is None:
        raise RuntimeError(
            'beam.upc_flux needs a Fortran compiler to tabulate the gamma-UPC '
            'luminosity once: %s was not found (fortran_compiler in '
            'input/mg5_configuration.txt)' % (fortran_compiler or 'gfortran'))
    stamp = exe + '.compiler'
    built_with = None
    if os.path.exists(stamp):
        with open(stamp) as fsock:
            built_with = fsock.read().strip()
    if os.path.exists(exe) and built_with == compiler and all(
            os.path.getmtime(exe) >= os.path.getmtime(dep)
            for dep in _dependencies(gamma_dir)):
        return exe
    logger.info('compiling the gamma-UPC luminosity tabulation (%s)', compiler)
    objects = []
    for name in _SOURCES:
        flags = ['-O2', '-w']
        if name.endswith('.f'):
            flags.append('-ffixed-line-length-132')
        result = subprocess.run([compiler] + flags + ['-c', name], cwd=gamma_dir,
                                capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError('compiling %s failed:\n%s' % (name, result.stderr))
        objects.append(os.path.splitext(name)[0] + '.o')
    result = subprocess.run([compiler, '-o', _EXECUTABLE] + objects, cwd=gamma_dir,
                            capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError('linking %s failed:\n%s' % (_EXECUTABLE, result.stderr))
    with open(stamp, 'w') as fsock:
        fsock.write(compiler + '\n')
    return exe


def table_path(gamma_dir, label, ions, beam_energies, xmin,
               knots=KNOTS_PER_DECADE):
    """The cached table of these settings (it need not exist yet)."""
    key = json.dumps({'label': label, 'ions': [list(i) for i in ions],
                      'ebeam': [repr(float(e)) for e in beam_energies],
                      'xmin': repr(float(xmin)), 'knots': knots,
                      'driver': _driver_hash(gamma_dir)}, sort_keys=True)
    digest = hashlib.sha1(key.encode()).hexdigest()[:16]
    return os.path.join(gamma_dir, 'tables', '%s_%s.dat' % (label, digest))


def _driver_hash(gamma_dir):
    """The sources: a new gamma-UPC version gives new tables."""
    sha = hashlib.sha1()
    for path in _dependencies(gamma_dir):
        with open(path, 'rb') as fsock:
            sha.update(fsock.read())
    return sha.hexdigest()[:16]


def tabulate(gamma_dir, label, ions, beam_energies, xmin, fortran_compiler=None,
             knots=KNOTS_PER_DECADE):
    """Path of the table, generated first if it is not cached."""
    path = table_path(gamma_dir, label, ions, beam_energies, xmin, knots)
    if os.path.exists(path):
        return path
    with _locked(gamma_dir):
        # another job may have written it while this one waited
        if os.path.exists(path):
            return path
        exe = _compile(gamma_dir, fortran_compiler)
        fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + '.',
                                   suffix='.tmp', dir=os.path.dirname(path))
        os.close(fd)
        stdin = '%s\n%d %d %d %d\n%.17g %.17g\n%.17g %d\n%s\n' % (
            label, ions[0][0], ions[0][1], ions[1][0], ions[1][1],
            float(beam_energies[0]), float(beam_energies[1]), xmin, knots, tmp)
        logger.info(
            'tabulating the gamma-UPC %s luminosity (Z,N = %s, %s) down to x = %g; '
            'with ions this takes a few minutes, once', label, ions[0], ions[1], xmin)
        log_path = path + '.log'
        try:
            with open(log_path, 'w') as log:
                result = subprocess.run([exe], input=stdin, text=True, cwd=gamma_dir,
                                        stdout=log, stderr=subprocess.STDOUT)
            if result.returncode or not os.path.getsize(tmp):
                raise RuntimeError('the gamma-UPC luminosity tabulation failed '
                                   '(exit %s), see %s' % (result.returncode, log_path))
            # mkstemp makes it private: a table is shared like the sources
            os.chmod(tmp, 0o644)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    return path


class UPCLuminosity:
    """A tabulated gamma-UPC luminosity, uploaded to the contexts of a run."""

    def __init__(self, madspace, contexts, gamma_dir, label, ions, beam_energies,
                 xmin, fortran_compiler=None):
        self.label = label
        self.ions = tuple(tuple(ion) for ion in ions)
        self.xmin = xmin
        self.path = tabulate(gamma_dir, label, self.ions, beam_energies, xmin,
                             fortran_compiler)
        # the globals of the table, named by its range
        self.prefix = 'upc%d_' % round(-math.log10(xmin))
        self.grid = madspace.PdfGrid(self.path)
        for context in contexts:
            self.grid.initialize_globals(context, self.prefix)
        self.luminosity = madspace.TabulatedLuminosity(self.grid, prefix=self.prefix)
