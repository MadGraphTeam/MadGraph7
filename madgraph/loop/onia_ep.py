################################################################################
#
# Copyright (c) 2026 The MadGraph7 Development team and Contributors
#
# This file is a part of the MadGraph7 project, an application which
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph7 license which should accompany this
# distribution.
#
################################################################################
"""The extended-precision (EP) passes of MadLoop outputs with a P-wave bound
state (#PWaveExtendedPrecision in MadLoopParams.dat, off by default).

The virtual of a P-wave bound state is a derivative in the relative momentum
of the constituents. At high energy its double-precision value loses up to
ten digits to cancellations, both in the loop coefficients and in the tensor
integrals (docs/benchmarkfolder/research/pwave_stability). With the switch on,
the double-precision passes of the stability test (CTMODE 1 and 2) run in
extended precision instead: the phase-space point, the dual wavefunctions,
the loop coefficients, the projections and the contractions in the kind
EPK = SELECTED_REAL_KIND(18) (module EP_PRECISION: the x87 80-bit format on
x86, binary128 where there is no x87 format), and the tensor integrals from
TIRust's double-precision entry with inputs as binary64 hi + lo words
(tirust_tensor_evaluate_powers_ext, installations with
raised_power_extended_inputs). The quadruple-precision rescue (CTMODE 4, 5)
is unchanged.

The EP code is MadLoop's quadruple-precision (MP) code with its names renamed
(MP_ -> EP_) and its kind changed (REAL*16 -> REAL(EPK)). It is derived here
from the generated MP files when the output is written, so that it follows
every change of the MP code. Only the dispatch (loop_matrix_standalone.inc,
helas_loop_amplitude.inc), the dual objects (the PW_DUAL module) and the
TIRust routines (TIRUST_interface.inc) are written in the templates."""

import glob
import os
import re

import madgraph.various.tirust_install as tirust_install
from madgraph import MadGraph5Error

KIND = 'EPK'
MODULE = 'EP_PRECISION'

# The module of the kind, compiled with the dual numbers in Source/DHELAS
KIND_MODULE = """      module ep_precision
c     The kind of the extended-precision passes of MadLoop with a P-wave
c     bound state: the x87 80-bit format (18 digits) on x86, binary128 where
c     there is no x87 format.
      implicit none
      integer, parameter :: epk = selected_real_kind(18)
      end module ep_precision

"""


class OniaEPError(MadGraph5Error):
    """The extended-precision code of a P-wave output cannot be derived."""


def tirust_supports_extended_inputs(libdir):
    """True if the TIRust installation in libdir has the raised-power entry
    with inputs as hi + lo words (tirust_tensor_evaluate_powers_ext,
    metadata raised_power_extended_inputs)."""
    if not libdir or not tirust_install.is_available(libdir):
        return False
    try:
        return tirust_install.read_config(libdir).get(
            'raised_power_extended_inputs') is True
    except MadGraph5Error:
        return False


#===============================================================================
# The renaming of the MP code
#===============================================================================
_KIND_SUBS = [
    (re.compile(r'REAL\*16', re.I), 'REAL(EPK)'),
    (re.compile(r'COMPLEX\*32', re.I), 'COMPLEX(EPK)'),
    (re.compile(r'KIND\s*=\s*16\b', re.I), 'KIND=EPK')]
# numeric literals of kind 16 (1.0E0_16, 0E0_16), one split across a
# continuation line (1.0E / $ -8_16) and gfortran's Q exponent (1Q0)
_LITERAL = re.compile(r'(?<![A-Za-z0-9_])((?:\d+\.\d*|\.\d+)(?:[EeDdQq][+-]?\d+)?|'
                      r'\d+[EeDdQq][+-]?\d+)_16\b')
_SPLIT_LITERAL = re.compile(r'([EeDdQq])(\n {5}\S\s*[+-]?\d+)_16\b')
_Q_LITERAL = re.compile(r'(?<![A-Za-z0-9_.])(\d+\.?\d*|\.\d+)[Qq]([+-]?\d+)(?![A-Za-z0-9_])')
# MP names: MP_X, ML5_0_MP_X, SET_MP_PS, MPPWW (the dual objects of the PW_DUAL
# module) and the constructors qcmplx and qreal of the dual numbers
_NAMES = [re.compile(r'(?<![A-Za-z0-9_])(M)(?=P_)', re.I),
          re.compile(r'(?<=[A-Za-z0-9]_)(M)(?=P_)', re.I),
          re.compile(r'(?<![A-Za-z0-9_])(M)(?=PPW)', re.I),
          re.compile(r'(?<![A-Za-z0-9_])(q)(?=cmplx|real)', re.I)]
_LEFT = re.compile(r'(?i)real\*16|complex\*32|kind\s*=\s*16\b|\d_16\b|(?<![A-Za-z0-9_])MP_'
                   r'|(?<=[A-Za-z0-9]_)MP_|(?<![A-Za-z0-9_])MPPW')


def _rename(match):
    return 'e' if match.group(1).islower() else 'E'


def to_ep(text, proc_prefix=None):
    """The EP version of MP Fortran code: MP names renamed EP, kind 16 replaced
    by EPK. MP_UPDATE_AS_PARAM, which sets the MP couplings, becomes the
    routine <proc_prefix>EP_UPDATE_AS_PARAM (write_ep_couplings)."""
    for regex, new in _KIND_SUBS:
        text = regex.sub(new, text)
    text = _SPLIT_LITERAL.sub(r'\1\2_EPK', text)
    text = _LITERAL.sub(r'\1_EPK', text)
    text = _Q_LITERAL.sub(r'\1E\2_EPK', text)
    for regex in _NAMES:
        text = regex.sub(_rename, text)
    if proc_prefix is not None:
        text = re.sub(r'(?i)(?<![A-Za-z0-9_])EP_UPDATE_AS_PARAM\b',
                      proc_prefix + 'EP_UPDATE_AS_PARAM', text)
    return text


def check_ep(text, name):
    """Refuse EP code that still has an MP name or a kind 16 (outside
    comments): the renaming rules miss a construct of the MP code."""
    for number, line in enumerate(text.split('\n'), 1):
        if line[:1] in ('c', 'C', '*', '!'):
            continue
        code = re.sub(r"'[^']*'", "''", line.split('!')[0])
        if _LEFT.search(code):
            raise OniaEPError('Cannot derive the extended-precision code: %s, line %d: %s'
                              % (name, number, line.strip()))
        if len(line) > 132:
            raise OniaEPError('Extended-precision code longer than 132 columns: %s, line %d'
                              % (name, number))


#===============================================================================
# Program units of fixed-form source
#===============================================================================
# a SUBROUTINE or FUNCTION statement, with its prefixes (type, ELEMENTAL, ...)
_HEADER = re.compile(r'^ {6}\s*(?!END\b)(?:[A-Za-z]+(?:\*\d+|\([^)]*\))?\s+)*?'
                     r'(?:SUBROUTINE|FUNCTION)\s+(\w+)', re.I)
_END = re.compile(r'^ {6}\s*END(?:\s+(?:SUBROUTINE|FUNCTION)(?:\s+\w+)?)?\s*$', re.I)
_MODULE = re.compile(r'^ {6}\s*MODULE\s+(\w+)\s*$', re.I)
_END_MODULE = re.compile(r'^ {6}\s*END\s+MODULE\b', re.I)
_BLOCK_DATA = re.compile(r'^ {6}\s*BLOCK\s*DATA\b', re.I)


def program_units(text):
    """The top-level subroutines and functions of fixed-form source, as
    (name, text) with the comment lines just before them; modules and block
    data are skipped."""
    lines = text.split('\n')
    units = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if _MODULE.match(line):
            while i < len(lines) and not _END_MODULE.match(lines[i]):
                i += 1
            i += 1
            continue
        if _BLOCK_DATA.match(line):
            while i < len(lines) and not _END.match(lines[i]):
                i += 1
            i += 1
            continue
        header = _HEADER.match(line)
        if header:
            start = i
            while start > 0 and lines[start - 1][:1] in ('c', 'C') and lines[start - 1].strip():
                start -= 1
            j = i + 1
            depth = 1
            while j < len(lines):
                if _HEADER.match(lines[j]):
                    depth += 1
                elif _END.match(lines[j]):
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            if j == len(lines):
                raise OniaEPError('Unterminated program unit %s' % header.group(1))
            units.append((header.group(1), '\n'.join(lines[start:j + 1])))
            i = j + 1
            continue
        i += 1
    return units


def add_kind_module(text):
    """Make the kind EPK visible in every top-level program unit: USE
    EP_PRECISION after its SUBROUTINE or FUNCTION statement."""
    lines = text.split('\n')
    out = []
    modules = units = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        if _MODULE.match(line):
            modules += 1
        elif _END_MODULE.match(line):
            modules -= 1
        elif _HEADER.match(line):
            units += 1
            if modules == 0 and units == 1:
                while i + 1 < len(lines) and re.match(r'^ {5}[^ 0]', lines[i + 1]):
                    i += 1
                    out.append(lines[i])
                out.append('      USE %s' % MODULE)
        elif _END.match(line):
            units -= 1
        i += 1
    return '\n'.join(out)


# The finiteness test of the MP coefficient updaters (WL_IS_FINITE of
# q_polynomial.py) reads the binary128 bits and says not finite for any other
# format, so the EP updaters take their dense path (no skipping of zero
# coefficients). That is kept: with an IEEE_IS_FINITE test instead, the EP
# updaters take the skipping path and the evaluation is ten times slower
# (u u~ > chi_c0 g at 1 TeV: 5.5 s instead of 0.5 s per point).


def ep_source(text, name, proc_prefix=None):
    """The EP version of whole MP source (one or several program units)."""
    text = add_kind_module(to_ep(text, proc_prefix))
    check_ep(text, name)
    return text


HEADER = """C     Extended-precision (EP) passes of a P-wave bound state
C     (#PWaveExtendedPrecision): derived from %s by
C     madgraph/loop/onia_ep.py (MP names renamed EP, kind EPK).
"""


#===============================================================================
# The subprocess directory
#===============================================================================
# Files that are entirely MP code
WHOLE_FILES = ['mp_compute_loop_coefs.f', 'mp_coef_construction_*.f', 'mp_helas_calls_*.f']
# Files with MP routines among others: the EP routines go to ep_<file>
MIXED_FILES = ['polynomial.f', 'onia_loop.f', 'improve_ps.f', 'loop_matrix.f',
               'CT_interface.f', 'TIR_interface.f']
# MP routines that have no EP version: the TIRust ones are in the template
# (TIRust's double-precision entry with hi + lo inputs), the other reduction
# libraries are not used for P-wave bound states
_SKIP = re.compile(r'(?i)TIRUST|NINJA|SAMURAI|COLLIER|GOLEM|MPLOOPNUM')
_MP_UNIT = re.compile(r'(?i)(?:^|_)MP_')


def mp_units(text):
    """The MP program units of a mixed file that have an EP version."""
    return [(name, unit) for name, unit in program_units(text)
            if _MP_UNIT.search(name) and not _SKIP.search(name)]


def write_ep_couplings(pdir, proc_prefix):
    """ep_coupl.inc and ep_coupl_same_name.inc, the EP versions of the MP
    model includes, and <proc_prefix>EP_UPDATE_AS_PARAM, which sets their
    values from the MP ones (MP_UPDATE_AS_PARAM, rounded to EPK)."""
    names = []
    for inc in ('mp_coupl.inc', 'mp_coupl_same_name.inc'):
        path = os.path.join(pdir, inc)
        if not os.path.exists(path):
            raise OniaEPError('Missing %s in %s' % (inc, pdir))
        with open(path) as stream:
            text = stream.read()
        if inc == 'mp_coupl.inc':
            # the variables, from the declarations (with their continuations)
            joined = re.sub(r'\n {5}\S', ' ', text)
            for line in joined.split('\n'):
                decl = re.match(r'^\s+(?:REAL\*16|COMPLEX\*32)\s+(.*)$', line, re.I)
                if decl:
                    body = re.sub(r'\([^)]*\)', '', decl.group(1))
                    names.extend(n.strip() for n in body.split(',') if n.strip())
        ep = to_ep(text)
        check_ep(ep, inc)
        with open(os.path.join(pdir, 'ep' + inc[2:]), 'w') as stream:
            stream.write(ep)
    if not names or any(not n.upper().startswith('MP__') for n in names):
        raise OniaEPError('Unexpected mp_coupl.inc in %s' % pdir)
    lines = [HEADER % 'the MP couplings (mp_coupl.inc)',
             '      SUBROUTINE %sEP_UPDATE_AS_PARAM()' % proc_prefix,
             'C     The couplings and parameters of the EP passes: those of the',
             'C     quadruple-precision passes, rounded to EPK',
             '      USE %s' % MODULE,
             '      IMPLICIT NONE',
             "      INCLUDE 'mp_coupl.inc'",
             "      INCLUDE 'ep_coupl.inc'",
             '      CALL MP_UPDATE_AS_PARAM()']
    for name in names:
        lines.append('      %s=%s' % ('EP' + name[2:], name))
    lines.append('      END')
    with open(os.path.join(pdir, 'ep_couplings.f'), 'w') as stream:
        stream.write('\n'.join(lines) + '\n')


def derive_subprocess(pdir, proc_prefix):
    """Write the EP files of the subprocess directory pdir (a P-wave output
    generated with OniaPwaveEP): ep_<file> for each MP file and each file
    with MP routines, and the EP couplings."""
    written = []
    for pattern in WHOLE_FILES:
        for path in sorted(glob.glob(os.path.join(pdir, pattern))):
            name = os.path.basename(path)
            with open(path) as stream:
                text = stream.read()
            out = 'ep' + name[2:]
            with open(os.path.join(pdir, out), 'w') as stream:
                stream.write(HEADER % name + ep_source(text, name, proc_prefix))
            written.append(out)
    for name in MIXED_FILES:
        path = os.path.join(pdir, name)
        if not os.path.exists(path):
            continue
        with open(path) as stream:
            units = mp_units(stream.read())
        if not units:
            continue
        text = '\n\n'.join(unit for _, unit in units) + '\n'
        out = 'ep_' + name
        with open(os.path.join(pdir, out), 'w') as stream:
            stream.write(HEADER % ('the MP routines of ' + name) +
                         ep_source(text, name, proc_prefix))
        written.append(out)
    write_ep_couplings(pdir, proc_prefix)
    written.append('ep_couplings.f')
    for needed in ('ep_compute_loop_coefs.f', 'ep_polynomial.f', 'ep_onia_loop.f'):
        if needed not in written:
            raise OniaEPError('No MP source for %s in %s' % (needed, pdir))
    return written


# The EP files in the build of the subprocesses (SubProcesses/makefile): ep_*.f
# next to the MP files, ep_polynomial.f after the modules of polynomial.f
_MAKEFILE_EDITS = [
    (' $(patsubst $(DOTF),$(DOTO),$(wildcard onia_loop.f)) \\\n',
     ' $(patsubst $(DOTF),$(DOTO),$(wildcard onia_loop.f)) \\\n'
     ' $(patsubst $(DOTF),$(DOTO),$(wildcard ep_*.f)) \\\n'),
    (' $(patsubst $(DOTF),$(DOTO),$(wildcard $(LOOP_PREFIX)*/onia_loop.f))\n',
     ' $(patsubst $(DOTF),$(DOTO),$(wildcard $(LOOP_PREFIX)*/onia_loop.f)) \\\n'
     ' $(patsubst $(DOTF),$(DOTO),$(wildcard $(LOOP_PREFIX)*/ep_*.f))\n'),
    ('             */mp_born_amps_and_wfs.f */jamp?_calls_*.f)\n',
     '             */mp_born_amps_and_wfs.f */jamp?_calls_*.f */ep_*.f)\n'),
    ('%olynomial.o : %olynomial.f\n',
     '# The extended-precision polynomial routines (onia_ep.py) use the modules\n'
     '# of polynomial.f\n'
     'ep_polynomial.o : polynomial.o\n'
     '%/ep_polynomial.o : %/polynomial.o\n'
     '%olynomial.o : %olynomial.f\n')]


def patch_subprocesses_makefile(path):
    """Add the EP files to the build of the subprocesses."""
    with open(path) as stream:
        text = stream.read()
    if 'wildcard ep_*.f' in text:
        return
    for old, new in _MAKEFILE_EDITS:
        if text.count(old) != 1:
            raise OniaEPError('Unexpected SubProcesses makefile: %r' % old.strip())
        text = text.replace(old, new)
    with open(path, 'w') as stream:
        stream.write(text)


#===============================================================================
# The HELAS library (Source/DHELAS)
#===============================================================================
def ep_aloha_object(text):
    """The ALOHA_OBJECT module of a loop dual library with the EP types
    EP_ALOHA and EP_ALOHA2D (from the MP ones) and the EP dual numbers."""
    old = '      use mp_dual_variables\n'
    if text.count(old) != 1:
        raise OniaEPError('Unexpected ALOHA_OBJECT module')
    text = text.replace(old, old + '      use ep_dual_variables\n')
    types = re.search(r'(?ms)^ +TYPE MP_ALOHA\n.*?^ +END TYPE MP_ALOHA2D\n', text)
    if not types:
        raise OniaEPError('Unexpected ALOHA_OBJECT module')
    ep_types = to_ep(types.group(0))
    return text[:types.end()] + ep_types + text[types.end():]


def ep_dual_variables(text):
    """ep_dual_variables.f: the module ep_precision (the kind EPK) and the EP
    version of the module mp_dual_variables (one-direction dual numbers)."""
    old = '      integer, parameter :: qp = 16\n'
    module = '      module mp_dual_variables\n'
    if text.count(old) != 1 or text.count(module) != 1:
        raise OniaEPError('Unexpected mp_dual_variables.f')
    text = text.replace(old, '      integer, parameter :: qp = epk\n')
    text = to_ep(text).replace('      module ep_dual_variables\n',
                               '      module ep_dual_variables\n'
                               '      use ep_precision, only: epk\n')
    check_ep(text, 'mp_dual_variables.f')
    return KIND_MODULE + text


def derive_dhelas(write_dir, mp_dual_functions):
    """The EP part of the HELAS library of a loop dual output: EP_<routine>.f
    for each MP ALOHA routine, ep_dual_variables.f and ep_aloha_functions.f
    (from mp_dual_functions, the text of aloha_functions_mp_dual.f). The
    caller adds the EP types to ALOHA_OBJECT (ep_aloha_object) and the
    objects to the makefile (DHELAS_MAKEFILE_EDITS)."""
    for path in sorted(glob.glob(os.path.join(write_dir, 'MP_*.f'))):
        name = os.path.basename(path)
        with open(path) as stream:
            text = stream.read()
        with open(os.path.join(write_dir, 'EP' + name[2:]), 'w') as stream:
            stream.write(ep_source(text, name))
    with open(os.path.join(write_dir, 'mp_dual_variables.f')) as stream:
        dual = stream.read()
    with open(os.path.join(write_dir, 'ep_dual_variables.f'), 'w') as stream:
        stream.write(ep_dual_variables(dual))
    with open(os.path.join(write_dir, 'ep_aloha_functions.f'), 'w') as stream:
        stream.write('c' + HEADER[1:] % 'aloha_functions_mp_dual.f' +
                     ep_source(mp_dual_functions, 'aloha_functions_mp_dual.f'))


DHELAS_MAKEFILE_EDITS = [
    ('BASIC_OBJS = dual_variables.o mp_dual_variables.o aloha_functions.o',
     'BASIC_OBJS = dual_variables.o mp_dual_variables.o ep_dual_variables.o '
     'aloha_functions.o ep_aloha_functions.o'),
    ('aloha_functions.o: dual_variables.o mp_dual_variables.o\n',
     'aloha_functions.o: dual_variables.o mp_dual_variables.o ep_dual_variables.o\n'
     '# the extended-precision external wavefunctions (onia_ep.py)\n'
     'ep_aloha_functions.o: aloha_functions.o\n')]
