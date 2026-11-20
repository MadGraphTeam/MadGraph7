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

import hashlib
import json
import logging
import math
import os
import shutil
import subprocess

logger = logging.getLogger('madevent.upc_luminosity')

# gamma-UPC's own internal grid starts at x = 1e-8 (ElasticPhotonPhotonFlux.f90,
# extended by decades on demand); the table never starts above it.
DEFAULT_XMIN = 1e-8
# knots per decade of each momentum fraction: twice gamma-UPC's nseg = 10,
# which the bicubic interpolation of madspace then follows to ~1e-6
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


def _compile(gamma_dir, fortran_compiler):
    exe = os.path.join(gamma_dir, _EXECUTABLE)
    sources = [os.path.join(gamma_dir, name) for name in _SOURCES]
    if os.path.exists(exe) and all(
            os.path.getmtime(exe) >= os.path.getmtime(src) for src in sources):
        return exe
    compiler = shutil.which(fortran_compiler or 'gfortran')
    if compiler is None:
        raise RuntimeError(
            'beam.upc_flux needs a Fortran compiler to tabulate the gamma-UPC '
            'luminosity once: %s was not found (fortran_compiler in '
            'input/mg5_configuration.txt)' % (fortran_compiler or 'gfortran'))
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
    for name in _SOURCES:
        with open(os.path.join(gamma_dir, name), 'rb') as fsock:
            sha.update(fsock.read())
    return sha.hexdigest()[:16]


def tabulate(gamma_dir, label, ions, beam_energies, xmin, fortran_compiler=None,
             knots=KNOTS_PER_DECADE):
    """Path of the table, generated first if it is not cached."""
    path = table_path(gamma_dir, label, ions, beam_energies, xmin, knots)
    if os.path.exists(path):
        return path
    exe = _compile(gamma_dir, fortran_compiler)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    stdin = '%s\n%d %d %d %d\n%.17g %.17g\n%.17g %d\n%s\n' % (
        label, ions[0][0], ions[0][1], ions[1][0], ions[1][1],
        float(beam_energies[0]), float(beam_energies[1]), xmin, knots, tmp)
    logger.info(
        'tabulating the gamma-UPC %s luminosity (Z,N = %s, %s) down to x = %g; '
        'with ions this takes a few minutes, once', label, ions[0], ions[1], xmin)
    log_path = path + '.log'
    with open(log_path, 'w') as log:
        result = subprocess.run([exe], input=stdin, text=True, cwd=gamma_dir,
                                stdout=log, stderr=subprocess.STDOUT)
    if result.returncode or not os.path.exists(tmp):
        raise RuntimeError('the gamma-UPC luminosity tabulation failed (exit %s), '
                           'see %s' % (result.returncode, log_path))
    os.replace(tmp, path)
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
        # one prefix per table: a later subprocess may need a lower xmin
        self.prefix = 'upc%d_' % round(-math.log10(xmin))
        self.grid = madspace.PdfGrid(self.path)
        for context in contexts:
            self.grid.initialize_globals(context, self.prefix)
        self.luminosity = madspace.TabulatedLuminosity(self.grid, prefix=self.prefix)
