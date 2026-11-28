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
"""MadLoop's P-wave virtuals in an FKS output (fks_onia, export_fks).

The bound-state layer of MadLoop (loop_onia.py) computes the virtual of a
Born with P-wave bound states with dual numbers: its HELAS library, tree and
loop routines in double and quadruple precision, is the dual-number library of
export_v4.write_aloha_library(loop_dual=True), with the names of the plain one
(FFV1_0, VXXXXX, the module ALOHA_OBJECT, ...). In an FKS P directory the Born
and the real matrix elements, which take the P-wave derivative numerically
(onia_amp_kernel_fks_pwave.inc), and libMadLoop are linked into one
executable, with the plain library. The dual library of n P-wave states is
therefore written a second time in Source/DHELAS_LD<n> (libdhelas_ld<n>) with
every external procedure and every module it defines prefixed LD<n>_, and the
same names are prefixed in the sources of the V* directories of the Borns with
n P-wave states, which link it. Types and module procedures keep their names
(they are scoped by their renamed module).

A V* directory of a P-wave Born carries the marker MARKER (its number of P-wave
states), written with the directory, so that the library and the renaming
follow from the output alone (also with low_mem_multicore_nlo_generation)."""

import glob
import os
import re

MARKER = 'onia_loop_dual.dat'
NAMES = 'renamed_names.dat'
MAX_COLUMNS = 132


def prefix(npwave):
    """The prefix of the names of the dual library of npwave P-wave states."""
    return 'LD%d_' % npwave


def library_dir(npwave):
    return 'DHELAS_LD%d' % npwave


def library_name(npwave):
    return 'dhelas_ld%d' % npwave


def write_marker(vdir, npwave):
    with open(os.path.join(vdir, MARKER), 'w') as stream:
        stream.write('%d\n' % npwave)


def marked_vdirs(subprocesses):
    """{V* directory: number of P-wave states} of the FKS output."""
    result = {}
    for marker in sorted(glob.glob(os.path.join(subprocesses, 'P*', 'V*', MARKER))):
        with open(marker) as stream:
            result[os.path.dirname(marker)] = int(stream.read().split()[0])
    return result


def _is_comment(line):
    return line[:1] in ('c', 'C', '*', '!') or line.lstrip().startswith('!')


_MODULE = re.compile(r'module\s+(\w+)\s*$')
_END_MODULE = re.compile(r'end\s*module\b|endmodule\b')
_PROCEDURE = re.compile(
    r'(?:(?:recursive|pure|elemental|impure)\s+)*'
    r'(?:(?:double\s+precision|double\s+complex|complex\s*\*\s*\d+|real\s*\*\s*\d+|integer\s*\*\s*\d+|'
    r'real|complex|integer|logical|character(?:\s*\*\s*\d+)?|'
    r'(?:type|real|complex|integer)\s*\(\s*[\w=]+\s*\))\s+)?'
    r'(subroutine|function)\s+(\w+)')


def defined_names(directory):
    """The external procedures and the modules defined by the Fortran
    sources of a directory (lower case): what the renaming prefixes."""
    names = set()
    sources = sorted(glob.glob(os.path.join(directory, '*.f')) +
                     glob.glob(os.path.join(directory, '*.f90')))
    for path in sources:
        in_module = False
        with open(path, errors='replace') as stream:
            for line in stream:
                if _is_comment(line):
                    continue
                text = line.split('!')[0].strip().lower()
                if not text or text.startswith('end ') or text == 'end' or \
                        text.startswith('endmodule'):
                    if _END_MODULE.match(text):
                        in_module = False
                    continue
                match = _MODULE.match(text)
                if match and match.group(1) != 'procedure':
                    names.add(match.group(1))
                    in_module = True
                    continue
                if in_module:
                    continue
                match = _PROCEDURE.match(text)
                if match:
                    names.add(match.group(2))
    return names


def _pattern(names):
    alternatives = '|'.join(re.escape(n) for n in sorted(names, key=len, reverse=True))
    # not after a letter, digit, '_' or '%' (a component); a fixed-form
    # continuation mark '$' may stand right before a name
    return re.compile(r'(?<![\w%])(' + alternatives + r')(?![\w$])', re.IGNORECASE)


def _split_fixed(line, limit=MAX_COLUMNS):
    """A fixed-form statement line longer than limit as continuation lines,
    split after a comma outside strings."""
    body = line.rstrip('\n')
    pieces = []
    while len(body) > limit:
        quote = None
        cut = -1
        for i, char in enumerate(body[:limit]):
            if quote:
                if char == quote:
                    quote = None
            elif char in "'\"":
                quote = char
            elif char == ',' and i > 6:
                cut = i + 1
        if cut < 0:
            raise ValueError('Cannot split the line:\n' + body)
        pieces.append(body[:cut])
        body = '     $ ' + body[cut:]
    pieces.append(body)
    return '\n'.join(pieces) + ('\n' if line.endswith('\n') else '')


def rename_text(text, names, pre, free_form=False):
    """text with every identifier of names prefixed pre (case kept), fixed
    form lines rewrapped to MAX_COLUMNS columns."""
    if not names:
        return text
    pattern = _pattern(names)
    out = []
    for line in text.splitlines(True):
        new = pattern.sub(lambda m: pre + m.group(1), line)
        if not free_form and new != line and not _is_comment(new) and \
                len(new.rstrip('\n')) > MAX_COLUMNS:
            new = _split_fixed(new)
        out.append(new)
    return ''.join(out)


# makefile includes (object lists), not Fortran
NOT_FORTRAN = ('aloha_file.inc',)


def rename_directory(directory, names, pre, extensions=('.f', '.f90', '.inc')):
    """Prefix names in the Fortran sources of a directory (the file names
    stay). A symbolic link whose target would change becomes a renamed copy
    (the target is shared with the directories of the other Borns). Returns
    the files changed."""
    changed = []
    for entry in sorted(os.listdir(directory)):
        path = os.path.join(directory, entry)
        if not entry.endswith(extensions) or entry in NOT_FORTRAN or not os.path.isfile(path):
            continue
        with open(path, errors='replace') as stream:
            text = stream.read()
        new = rename_text(text, names, pre, free_form=entry.endswith('.f90'))
        if new == text:
            continue
        if os.path.islink(path):
            os.remove(path)
        with open(path, 'w') as stream:
            stream.write(new)
        changed.append(entry)
    return changed


def rename_library(directory, npwave):
    """Prefix the names a dual library defines, in its sources and in its
    makefiles (module order), and record them in NAMES. Returns them."""
    names = defined_names(directory)
    pre = prefix(npwave)
    rename_directory(directory, names, pre, extensions=('.f', '.f90'))
    with open(os.path.join(directory, NAMES), 'w') as stream:
        stream.write('\n'.join(sorted(names)) + '\n')
    makefile = os.path.join(directory, 'makefile')
    with open(makefile) as stream:
        text = stream.read()
    text = text.replace('libdhelas.', 'lib%s.' % library_name(npwave))
    with open(makefile, 'w') as stream:
        stream.write(text)
    return names


def library_names(source_dir, npwave):
    with open(os.path.join(source_dir, library_dir(npwave), NAMES)) as stream:
        return set(line.strip() for line in stream if line.strip())
