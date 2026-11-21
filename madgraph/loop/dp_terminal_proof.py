"""Fail-closed consumer proof for the private binary64 terminal updater.

This deliberately recognizes a small part of the generated Fortran language,
rather than attempting an arbitrary-Fortran alias analysis.  Call it on the
completed export *after* proposing the private calls; False means that all
proposed calls must be restored to their ordinary dense spelling.  The private
updater's arithmetic/IEEE admission proof is separate from this storage and
consumer proof.  In particular, externally linked reduction libraries and the
compiler are trusted, as they are for the dense program.

Two layouts are known.  In a standalone export the loop directory is a
SubProcesses/P* directory and every unit of SubProcesses, Source/MODEL and
Source/DHELAS is checked in full.  In an MC@NLO (FKS) export the loop directory
is the single SubProcesses/P*/V* directory of its process, built into the
libMadLoop.a that only this P* directory links.  The loop directory, MODEL and
DHELAS are then checked in full, and every other source that can be compiled
or included into that program (the P* directory, the shared SubProcesses
files, Source and FixedOrderAnalysis) must not name the loop storage or the
protected routines at all.
"""
from collections import namedtuple
from hashlib import sha256
import functools
from pathlib import Path
import logging
import os
import re

logger = logging.getLogger('madgraph.loop_exporter')


class _Unproved(ValueError):
    pass


def _need(condition, message):
    if not condition:
        raise _Unproved(message)


def _compact(text):
    return re.sub(r'\s+', '', text.upper())


def _mentions(text, name):
    # Blanks inside identifiers are insignificant in fixed-form Fortran.
    # A digit may precede a name in e.g. COMPLEX*16WL; do not mistake the
    # containing identifier DP_WL or WL_ENTRY_INEXACT for WL itself.
    return bool(re.search(r'(?<![A-Z_])' + re.escape(name) +
                          r'(?![A-Z0-9_])', _compact(text)))


def _inside(path, root):
    return path.resolve().is_relative_to(root.resolve())


# Includes written only when the process is built or launched, from the run
# and parameter cards: the run-card parameter declarations (run interface)
# and the parameter values (MODEL makefile, treatcards param).  The links and
# INCLUDE statements naming them exist from the output on.
_LAUNCH_GENERATED = ('Source/run_card.inc', 'Source/param_card.inc')


def _launch_generated(path, export):
    return not path.exists() and path.resolve() in [
        (export / item).resolve() for item in _LAUNCH_GENERATED]


_Row = namedtuple('_Row', 'origin number text label')
_STANDARD = {
    'born_matrix.f', 'loop_matrix.f', 'loop_num.f', 'polynomial.f',
    'mp_compute_loop_coefs.f', 'mp_born_amps_and_wfs.f',
    'MadLoopCommons.f', 'MadLoopParamReader.f', 'TIR_interface.f',
    'COLLIER_interface.f', 'TIRUST_interface.f', 'CT_interface.f',
    'GOLEM_interface.f', 'improve_ps.f', 'mg5_citation.f', 'check_sa.f',
    'check_sa_born_splitOrders.f', 'f2py_wrapper.f', 'all_matrix.f',
    'StabilityCheckDriver.f', 'compute_color_flows.f',
}
_BATCH = re.compile(r'(?:mp_)?(?:coef_construction|helas_calls_(?:ampb|uvct)|loop_CT_calls)_\d+\.f$')
_SOURCE_SUFFIXES = {'.f', '.for', '.f90', '.f95', '.f03', '.f08', '.c', '.cc', '.cpp', '.cxx'}


def _layout(directory):
    """Return the export root and its layout ('standalone' or 'fks')."""
    export = directory
    for parent in directory.parents:
        if (parent / 'SubProcesses').is_dir() and (parent / 'Source').is_dir():
            export = parent
            break
    subprocesses = export / 'SubProcesses'
    if (export != directory and directory.parent.parent == subprocesses and
            (subprocesses / 'makefile_fks_dir').is_file() and
            (subprocesses / 'makefile_loop').is_file()):
        _need(re.fullmatch(r'P[A-Za-z0-9_]+', directory.parent.name) and
              re.fullmatch(r'V[A-Za-z0-9_]+', directory.name), 'Not an FKS loop directory')
        # makefile_fks_dir builds libMadLoop.a from every V* directory of the
        # process directory, each moving its library to the same name.
        loops = [path for path in directory.parent.iterdir()
                 if path.name.startswith('V') and path.is_dir()]
        _need(loops == [directory] and not directory.is_symlink(),
              'Several or aliased FKS loop directories')
        return export, 'fks'
    return export, 'standalone'


def _inventory(directory, source_manifest, include_roots):
    export, layout = _layout(directory)
    scopes = [directory]
    if layout == 'fks':
        scopes += [export / 'Source' / name for name in ('MODEL', 'DHELAS')
                   if (export / 'Source' / name).is_dir()]
    elif export != directory:
        _need(_inside(directory, export / 'SubProcesses'), 'Not a standalone subprocess')
        scopes = [export / 'SubProcesses']
        scopes += [export / 'Source' / name for name in ('MODEL', 'DHELAS')
                   if (export / 'Source' / name).is_dir()]
    roots = []
    for item in include_roots:
        path = Path(item)
        if not path.is_absolute():
            path = directory / path
        _need(path.is_dir() and _inside(path, export), 'External include root')
        roots.append(path)
    # The include search path of the loop compilations: the standalone
    # makefile adds MODEL and DHELAS, makefile_loop adds only lib.
    if layout == 'fks':
        roots += [export / 'lib'] if (export / 'lib').is_dir() else []
    else:
        roots += [export / 'Source' / name for name in ('MODEL', 'DHELAS')
                  if (export / 'Source' / name).is_dir()]
    units = {}
    for scope in scopes:
        for path in sorted(scope.rglob('*')):
            if path.is_symlink() and path.is_dir():
                # Standard resource links contain no compiled source.  Other
                # directory links can hide compile units from rglob.
                _need(path.name == 'MadLoop5_resources' and _inside(path, export),
                      'Aliased source directory')
            if not path.is_file():
                continue
            _need(path.name.lower() != 'gnumakefile' and path.name != 'Makefile',
                  'Alternate make entry point')
            _need(path.suffix.lower() not in ('.mk', '.cmake'), 'Custom build input')
            if path.suffix.lower() not in _SOURCE_SUFFIXES:
                continue
            # all_matrix is generated C++ glue for Python, outside the private
            # Fortran call path; initially do not optimize those exports.
            # The model writer uses a .f90 suffix for this fixed-form module;
            # the checked MODEL recipe compiles it with -ffixed-form.  Other
            # .f90 units still require a free-form parser and stay outside
            # this proof.
            fixed_module = (path == export / 'Source/MODEL/coupl.f90' and
                            (path.parent / 'makefile').is_file())
            # Standalone exports also link it as SubProcesses/coupl.f90.  The
            # link adds no text of its own: the MODEL unit it resolves to is
            # in scope and is checked in full.
            if (path.name == 'coupl.f90' and path.is_symlink() and
                    export / 'Source/MODEL' in scopes and
                    path.resolve() == (export / 'Source/MODEL/coupl.f90').resolve()):
                continue
            _need(path.suffix == '.f' or fixed_module, 'Unsupported compilation language')
            _need(_inside(path, export), 'External compilation unit')
            if scope.name == 'SubProcesses' or scope == directory:
                _need(path.name in _STANDARD or _BATCH.fullmatch(path.name),
                      'Unknown subprocess compilation unit')
            units[path.absolute()] = path
    _need(directory / 'polynomial.f' in units.values() and units, 'Missing polynomial source')
    if source_manifest is not None:
        given = []
        for item in source_manifest:
            path = Path(item)
            if not path.is_absolute():
                path = directory / path
            _need(path.is_file() and _inside(path, export), 'External manifest unit')
            given.append(Path(os.path.abspath(str(path))))
        _need(len(given) == len(set(given)) and set(given) == set(units),
              'Incomplete or duplicate source manifest')
    return export, layout, list(units.values()), roots


# Sources that the exporter copies into Source for output_dependencies=internal.
# Like the linked reducer libraries in the other modes, they are trusted.
_TRUSTED_SOURCE_TREES = ('CutTools', 'IREGI', 'StdHEP')
_FOREIGN_ROOTS = ('SubProcesses', 'Source', 'FixedOrderAnalysis')
_FOREIGN_SUFFIXES = {'.f', '.for', '.ftn', '.f77', '.f90', '.f95', '.f03', '.f08',
                     '.fpp', '.inc', '.h', '.hh', '.hpp', '.hxx', '.c', '.cc',
                     '.cpp', '.cxx', '.c++'}
# Compilation units whose text a preprocessor can rewrite.
_PREPROCESSED_SUFFIXES = ('.c', '.cc', '.cpp', '.cxx', '.c++', '.C', '.F', '.FOR',
                          '.FTN', '.FPP', '.F90', '.F95', '.F03', '.F08', '.fpp')


@functools.lru_cache(maxsize=None)
def _template_digests():
    repository = Path(__file__).resolve().parents[2]
    digests = set()
    for path in (repository / 'Template').rglob('*'):
        if path.suffix in _PREPROCESSED_SUFFIXES and path.is_file() and not path.is_symlink():
            digests.add(sha256(path.read_bytes()).hexdigest())
    return digests


# The other loop directories of an export share most foreign sources: keep
# the INCLUDE names of each file whose text passed, keyed by its identity and
# the process prefix (where the includes resolve depends on the reference).
_FOREIGN_CLEAN = {}


def _fixed_statements(text):
    """Join fixed-form continuation lines (columns 1-6 of a continuation line
    are not part of the statement)."""
    joined = []
    for raw in text.splitlines():
        raw = _fixed(raw)
        if (len(raw) > 6 and joined and raw[:1] not in 'C*!' and
                not raw[:5].strip() and raw[5] not in ' 0'):
            joined[-1] += raw[6:]
        else:
            joined.append(raw)
    return '\n'.join(joined)


def _foreign_scan(export, directory, prefix):
    """Every other source of the FKS program must not name the loop storage.

    A unit outside the loop directory can reach the DP COMMON, the protected
    routines or the private updaters only by naming them: in a COMMON, CALL,
    USE or BIND(C) statement, or as a C symbol.  The names are searched in
    every Fortran and C source and include file (also in comments and string
    literals, across fixed- and free-form continuations), following INCLUDE
    and #include into any file of the export.  C and C++ (and preprocessed
    Fortran) units, whose macros could assemble a name, must in addition be
    unmodified copies of generator templates.
    """
    subprocesses = export / 'SubProcesses'
    excluded = [export / 'Source' / name for name in _TRUSTED_SOURCE_TREES]
    loops = [path for path in subprocesses.glob('P*/V*') if path.is_dir()]
    hidden = [path.resolve() for path in excluded + loops]
    roots = [(export / root).resolve() for root in _FOREIGN_ROOTS]
    names = [prefix + name for name in ('CREATE_LOOP_COEFS', 'MERGE_WL', 'UPDATE_WL_',
                                        'DP_TERMINAL_WL_', 'POLYNOMIAL_CONSTANTS')]
    common = '/' + prefix + 'WL/'
    symbol = re.compile(r'(?<![A-Z0-9_$])_*' + re.escape(prefix) + r'WL_*(?![A-Z0-9_$])')
    pending, seen = [], set()
    for top in _FOREIGN_ROOTS:
        if not (export / top).is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(export / top):
            here = Path(dirpath)
            for name in list(dirnames):
                path = here / name
                if path in excluded or path in loops:
                    dirnames.remove(name)
                elif path.is_symlink():
                    # os.walk does not descend into it: its target must be
                    # scanned through one of the roots.
                    target = path.resolve()
                    _need(any(target.is_relative_to(root) for root in roots)
                          and not any(target.is_relative_to(tree) for tree in hidden),
                          'Aliased foreign source directory')
            for name in filenames:
                if Path(name).suffix.lower() in _FOREIGN_SUFFIXES:
                    pending.append(here / name)
    while pending:
        path = pending.pop()
        target = path.resolve()
        _need(target.is_relative_to(export) and
              not any(target.is_relative_to(tree) for tree in hidden),
              'Foreign source outside the scanned program')
        # Includes resolve relative to the referencing path as well.
        if (target, path.parent) in seen:
            continue
        seen.add((target, path.parent))
        if _launch_generated(target, export):
            continue
        _need(target.is_file(), 'Missing foreign source')
        status = target.stat()
        key = (str(target), status.st_size, status.st_mtime_ns, prefix,
               path.suffix in _PREPROCESSED_SUFFIXES)
        if key not in _FOREIGN_CLEAN:
            _FOREIGN_CLEAN[key] = _foreign_text(target, path, names, common, symbol)
        # Search every directory the FKS recipes pass with -I (the P*
        # directory, lib, FixedOrderAnalysis) besides the including file's
        # own; an include found in none of them is external to the export.
        folders = {path.parent, target.parent, directory.parent, export / 'lib',
                   export / 'FixedOrderAnalysis', export / 'Source', subprocesses}
        pending += [folder / item for item in _FOREIGN_CLEAN[key]
                    for folder in folders if (folder / item).is_file()]


def _foreign_text(target, path, names, common, symbol):
    """Check one foreign file's text; return the names it includes."""
    original = target.read_bytes().decode('latin-1')
    text = original.upper()
    spliced = re.sub(r'\\\r?\n|##', '', text)
    compact = [re.sub(r'[\s&]+', '', spliced),
               re.sub(r'\s+', '', _fixed_statements(text))]
    _need(not any(name in variant for name in names for variant in compact) and
          not any(common in variant for variant in compact) and
          not symbol.search(spliced), 'Foreign reference to the loop storage or routines')
    if path.suffix in _PREPROCESSED_SUFFIXES or target.suffix in _PREPROCESSED_SUFFIXES:
        _need(sha256(target.read_bytes()).hexdigest() in _template_digests(),
              'Foreign preprocessed unit is not a generator template')
    # Included files can have any name and live anywhere in the export.
    fortran = re.sub(r'[ \t]+', '', _fixed_statements(original))
    includes = re.findall(r'(?im)^(?:\d*)include[\'"]([^\'"\n]+)[\'"]', fortran)
    includes += re.findall(r'(?im)^[ \t]*#[ \t]*include[ \t]*"([^"\n]+)"', original)
    for item in includes:
        _need(not Path(item).is_absolute() and '\\' not in item, 'Unsafe foreign INCLUDE')
    return includes


def _make_value(value):
    """A generator-formatted value of a make variable: no shell, eval or
    call, and no other source or object (only the canonical references)."""
    rest = re.sub(r'\$\((?:LIBDIR|libext|dylibext)\)', '', value)
    return ('$' not in rest and ';' not in rest and '`' not in rest and
            not re.search(r'(?:^|\s)[^ ]+\.(?:o|f|f90|c|cc|cpp)(?:\s|$)', rest))


def _include_flags(value):
    """Reducer include directories are passed on the compilation command
    line: only -I options are within the proof there."""
    tokens = value.split()
    while tokens:
        token = tokens.pop(0)
        if token == '-I' and tokens:
            token += tokens.pop(0)
        if not re.fullmatch(r'-I(?:\$\(LIBDIR\)|[A-Za-z0-9_./+-]+)', token):
            return False
    return True


def _rendered(template, actual, values):
    """Match actual against template % values, where each value given as None
    is free (any one-line text, returned for its own check)."""
    marks = {key: '\x00%s\x00' % key for key, value in values.items() if value is None}
    text = template % dict(values, **marks)
    pattern, offset = '', 0
    for token in re.finditer('\x00([a-z_]+)\x00', text):
        pattern += re.escape(text[offset:token.start()]) + '([^\n]*)'
        offset = token.end()
    match = re.fullmatch(pattern + re.escape(text[offset:]), actual)
    _need(match is not None, 'Modified build recipe')
    found = {}
    for key, value in zip(re.findall('\x00([a-z_]+)\x00', text), match.groups()):
        _need(found.setdefault(key, value) == value, 'Inconsistent build recipe')
    return found


def _make_opts_variables(text, allowed):
    variables = {}
    for line in text:
        match = re.fullmatch(r'([A-Z0-9_]+)\s*=\s*([A-Za-z0-9_./:+ =-]*)', line)
        _need(match and match[1] not in variables, 'Unknown compiler option assignment')
        variables[match[1]] = match[2].strip()
    _need(set(variables) == allowed, 'Unknown compiler option variable')
    _need(re.fullmatch(r'[A-Za-z0-9_./+-]+', variables['DEFAULT_F_COMPILER']),
          'Fortran compiler wrapper/options outside proof')
    return variables


def _normalize_make(text):
    return [line.strip() for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith('#')]


def _object_lists(export):
    for name, variable, basic in (('MODEL', 'MODEL', set()),
                                  ('DHELAS', 'ALOHARoutine', {'aloha_functions'})):
        root = export / 'Source' / name
        path = root / ('makeinc.inc' if name == 'MODEL' else 'aloha_file.inc')
        if path.exists():
            _need(_inside(path, export), 'External object list')
            lines = [line.split('#', 1)[0].strip() for line in path.read_text().splitlines()]
            text = ' '.join(line for line in lines if line).replace('\\', ' ')
            match = re.fullmatch(re.escape(variable) + r'\s*=\s*((?:[A-Za-z0-9_]+\.o\s*)*)', text)
            _need(match is not None, 'Unknown object-list build input')
            objects = match[1].split()
            _need(len(objects) == len(set(objects)), 'Repeated compiled object')
            _need(all((root / (obj[:-2] + '.f')).is_file() or
                      (name == 'MODEL' and obj == 'coupl.o' and
                       (root / 'coupl.f90').is_file()) for obj in objects),
                  'Missing compiled source')
            # Source enumeration is deliberately a superset (e.g. testprog).
            _need(all((root / (stem + '.f')).is_file() for stem in basic), 'Missing core source')


@functools.lru_cache(maxsize=None)
def _recipe_templates():
    """Digests of every file the generator copies verbatim."""
    repository = Path(__file__).resolve().parents[2]
    digests = set()
    for folder in ('Template', 'aloha/template_files', 'madgraph/models/template_files'):
        for item in (repository / folder).rglob('*'):
            if item.is_file() and not item.is_symlink():
                digests.add(sha256(item.read_bytes()).hexdigest())
    return digests


def _fks_build_inputs(export, directory):
    """The MC@NLO recipes: the generator's templates, verbatim or rendered.

    Every make input of the program's build tree must be a repository
    template, except the files the exporter renders: makefile_loop (which
    builds the loop directory), make_opts, the MODEL makefile and the object
    lists.  These are compared with their templates, only the generator's
    placeholders being free.
    """
    repository = Path(__file__).resolve().parents[2]
    subprocesses = export / 'SubProcesses'
    loop = subprocesses / 'makefile_loop'
    _need((directory / 'makefile').resolve() == loop.resolve() and
          (directory.parent / 'makefile').resolve() == (subprocesses / 'makefile_fks_dir').resolve(),
          'Unknown FKS build entry point')
    found = _rendered((repository / 'Template/NLO/SubProcesses/makefile_loop.inc').read_text(),
                      loop.read_text(), dict(link_tir_libs=None, tir_libs=None, tir_include=None,
                                             dotf='%.f', doto='%.o'))
    _need(_make_value(found['link_tir_libs']) and _make_value(found['tir_libs']) and
          _include_flags(found['tir_include']), 'Custom reducer build input')
    model = (repository / 'madgraph/models/template_files/fortran/makefile_madevent').read_text()
    rendered = {loop.resolve(), (export / 'Source/make_opts').resolve(),
                (export / 'Source/MODEL/makefile').resolve()}
    _need(all(path.is_relative_to(export) for path in rendered), 'External build recipe')
    _need((export / 'Source/MODEL/makefile').read_text() ==
          model.replace('madevent', 'aMCatNLO').replace('../vector.inc', ''),
          'Modified model build recipe')
    # The Born support library (born_support.write_build): its objects, and
    # the renamed copies of DHELAS and MODEL sources in deps, which are
    # scanned with the other sources.
    providers = export / 'Source/BornSupport/providers.mk'
    if providers.exists():
        objects = r'(?:[a-z][a-z0-9]*/)?[A-Za-z0-9_]+\.o'
        for line in providers.read_text().splitlines():
            _need(re.fullmatch(r'(?:PROVIDER|DEPENDENCY)_OBJECTS =(?: %s)*' % objects, line) or
                  re.fullmatch(r'deps/(?:helas|model)_[A-Za-z0-9_]+\.f: refresh_dependency\.py'
                               r'(?: \.\./(?:DHELAS|MODEL)/[A-Za-z0-9_]+\.(?:f|inc))+', line) or
                  re.fullmatch(r'\t\$\(PYTHON\) refresh_dependency\.py \.\./(?:DHELAS|MODEL)/'
                               r'[A-Za-z0-9_]+\.f \$@', line) or
                  re.fullmatch(r'%s:(?: %s)+' % (objects, objects), line),
                  'Unknown Born support build input')
        rendered.add(providers.resolve())
    excluded = [export / 'Source' / name for name in _TRUSTED_SOURCE_TREES]
    for top in _FOREIGN_ROOTS:
        for path in sorted((export / top).rglob('*')) if (export / top).is_dir() else ():
            if any(_inside(path, tree) for tree in excluded) or not path.is_file():
                continue
            name = path.name.lower()
            if not (name.startswith(('makefile', 'gnumakefile')) or name == 'make_opts' or
                    path.suffix.lower() in ('.mk', '.cmake') or name == 'cmakelists.txt'):
                continue
            if path.resolve() in rendered:
                continue
            _need(_inside(path, export) and
                  sha256(path.read_bytes()).hexdigest() in _recipe_templates(),
                  'Modified build recipe')
    # The P* makefile includes proc_characteristics before make_opts; its
    # generated settings must not assign any variable that the recipes read.
    recipes = '\n'.join(text.read_text() for text in (subprocesses / 'makefile_fks_dir',
                                                     export / 'Source/make_opts', loop))
    read = set(re.findall(r'\$[({]([A-Za-z_][A-Za-z0-9_]*)[)}]', recipes))
    read |= set(re.findall(r'(?m)^\s*(?:ifdef|ifndef)\s+([A-Za-z_][A-Za-z0-9_]*)', recipes))
    characteristics = subprocesses / 'proc_characteristics'
    for line in characteristics.read_text().splitlines() if characteristics.exists() else ():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        match = re.fullmatch(r'\s*([a-z][a-z0-9_]*)\s*=([^$`;\\]*)', line)
        _need(match and match[1] not in read, 'Process characteristics outside proof')
    path = export / 'Source/make_opts'
    marker = '#end_of_make_opts_variables'
    actual = path.read_text().split(marker)
    template = (repository / 'Template/NLO/Source/make_opts.inc').read_text()
    standard = (template % dict(link_tir_libs='\x00', tir_libs='', dotf='%.f', doto='%.o')).split(marker)
    _need(len(actual) == len(standard) == 2, 'Unknown make_opts structure')
    _make_opts_variables(_normalize_make(actual[0]),
                         {'DEFAULT_CPP_COMPILER', 'MACFLAG', 'STDLIB', 'STDLIB_FLAG',
                          'DEFAULT_F_COMPILER', 'DEFAULT_F2PY_COMPILER'})
    lines, wanted = _normalize_make(actual[1]), _normalize_make(standard[1])
    _need(len(lines) == len(wanted), 'Modified compiler recipes')
    for line, expected in zip(lines, wanted):
        if '\x00' in expected:
            head, tail = expected.split('\x00')
            head = head.rstrip()
            _need(line.startswith(head) and line.endswith(tail) and
                  _make_value(line[len(head):len(line) - len(tail)]), 'Custom reducer link input')
        else:
            _need(line == expected, 'Modified compiler recipes')
    _object_lists(export)



def _build_inputs(export, directory, layout):
    """Accept the generator's standard recipes, not arbitrary make programs.

    Comparing current generator templates is a grammar boundary, not a frozen
    export hash. Compiler selection and linked reducer libraries remain trusted.
    """
    if layout == 'fks':
        return _fks_build_inputs(export, directory)
    repository = Path(__file__).resolve().parents[2]
    templates = {
        'SubProcesses': 'Template/loop_material/StandAlone/SubProcesses/makefile',
        'MODEL': 'madgraph/models/template_files/fortran/makefile_standalone',
        'DHELAS': 'aloha/template_files/Makefile_F',
    }
    locations = [(directory / 'makefile', 'SubProcesses')]
    if export != directory:
        locations += [(export / 'SubProcesses/makefile', 'SubProcesses'),
                      (export / 'Source/MODEL/makefile', 'MODEL'),
                      (export / 'Source/DHELAS/makefile', 'DHELAS')]
    checked = set()
    for path, kind in locations:
        if not path.exists() or path.resolve() in checked:
            continue
        checked.add(path.resolve())
        _need(_inside(path, export), 'External build recipe')
        template = (repository / templates[kind]).read_text()
        actual = path.read_text()
        _need(actual.startswith(template), 'Modified build recipe')
        tail = actual[len(template):]
        if kind != 'SubProcesses':
            _need(not tail.strip(), 'Appended build recipe')
            continue
        # export_v4 appends only these per-process f2py targets. They compile
        # units already in the closure and cannot introduce another source.
        for line in tail.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            _need(stripped in ('ifeq ($(origin MENUM),undefined)', 'MENUM=2',
                              'endif', "echo 'done'") or
                  re.fullmatch(r'all:(?: +[A-Za-z0-9_]+/matrix\$\(MENUM\)py\.so)* *', stripped) or
                  re.fullmatch(r'[A-Za-z0-9_]+/matrix\$\(MENUM\)py\.so:', stripped) or
                  re.fullmatch(r'cd [A-Za-z0-9_]+;make matrix\$\(MENUM\)py\.so', stripped),
                  'Unknown appended build input')
    _object_lists(export)
    path = export / 'Source/make_opts'
    if path.exists():
        _need(_inside(path, export), 'External compiler options')
        marker = '#end_of_make_opts_variables'
        actual = path.read_text().split(marker)
        standard = (repository / 'Template/LO/Source/.make_opts').read_text().split(marker)
        _need(len(actual) == len(standard) == 2, 'Unknown make_opts structure')
        _need(_normalize_make(actual[1]) == _normalize_make(standard[1]), 'Modified compiler recipes')
        variables = _make_opts_variables(_normalize_make(actual[0]),
            {'DEFAULT_F2PY_COMPILER', 'DEFAULT_F_COMPILER', 'DEFAULT_CPP_COMPILER',
             'MACFLAG', 'MG5AMC_VERSION', 'STDLIB', 'PYTHIA8_PATH', 'STDLIB_FLAG', 'AMP_FLAG'})
        # Optimisation flag of the tree-level and MadEvent amplitude sources;
        # no loop recipe reads it. Only its default (empty) is within the proof.
        _need(variables['AMP_FLAG'] == '', 'Amplitude optimisation flag outside proof')
    path = export / 'SubProcesses/MadLoop_makefile_definitions'
    if path.exists():
        _need(_inside(path, export), 'External reducer definitions')
        actual = path.read_text()
        template = (repository / 'Template/loop_material/StandAlone/SubProcesses/MadLoop_makefile_definitions.inc').read_text()
        pattern, offset = '', 0
        for token in re.finditer(r'%\([a-z_]+\)s', template):
            pattern += re.escape(template[offset:token.start()]) + r'[^\n]*'
            offset = token.end()
        pattern += re.escape(template[offset:])
        _need(re.fullmatch(pattern, actual), 'Modified reducer build definitions')
        for line in actual.splitlines()[:10]:
            if line.startswith(('DOTO =', 'DOTF =', 'LOOP_PREFIX =')):
                _need(line in ('DOTO = %.o', 'DOTF = %.f', 'LOOP_PREFIX = P'), 'Custom compilation suffix/prefix')
                continue
            # LOOP_INCLUDE is on the compilation command line of every loop
            # unit: it may only add include directories.
            if line.startswith('LOOP_INCLUDE ='):
                _need(_include_flags(line[len('LOOP_INCLUDE ='):]), 'Custom reducer include input')
                continue
            # Only the canonical make references are permitted in formatted
            # values; no shell/eval/call or injected object/source arguments.
            _need(_make_value(line), 'Custom reducer build input')



def _uncomment(text):
    quote = None
    index = 0
    while index < len(text):
        char = text[index]
        if quote:
            if char == quote:
                if index + 1 < len(text) and text[index + 1] == quote:
                    index += 1
                else:
                    quote = None
        elif char in "'\"":
            quote = char
        elif char == '!':
            return text[:index]
        index += 1
    return text


def _fixed(raw):
    if raw.startswith('\t'):
        return ('     $' + raw[2:] if raw[1:2] in '123456789' and raw[1:2]
                else '      ' + raw[1:])
    return raw


def _expanded(unit, export, roots):
    def physical(path, active):
        _need(path.resolve() not in active, 'Cyclic INCLUDE')
        _need(_inside(path, export), 'External INCLUDE')
        active = active + [path.resolve()]
        for number, raw in enumerate(path.read_text().splitlines(), 1):
            directive = re.match(r'(?:[cC*!]\s*\$(?:OMP|ACC)|[!cC]\s*(?:GCC|DEC|DIR)\$)', raw, re.I)
            _need(not directive or (unit.name == 'mg5_citation.f' and
                  _compact(raw) in ('C$OMPCRITICAL(MG5_CITE)', 'C$OMPENDCRITICAL(MG5_CITE)')),
                  'Compiler directive outside proof')
            if not raw.strip() or raw[:1] in 'cC*!':
                continue
            raw = _fixed(raw)
            _need(not raw.lstrip().startswith('#'), 'Preprocessor source')
            _need(len(raw) >= 6 and '\t' not in raw[:6], 'Non fixed-form source')
            _need(not raw[132:].strip(), 'Unproved fixed line length')
            text = _uncomment(raw[6:132]).strip()
            outside_literals = re.sub(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"", '', text)
            _need(';' not in outside_literals, 'Multiple Fortran statements on one line')
            match = re.fullmatch(r'I\s*N\s*C\s*L\s*U\s*D\s*E\s*[\'\"]([^\'\"]+)[\'\"]', text, re.I)
            if match:
                _need(raw[5] in ' 0' and not raw[:5].strip(), 'Continued/labeled INCLUDE')
                target = Path(match[1])
                _need(not target.is_absolute() and '\\' not in match[1], 'Unsafe INCLUDE')
                choices = [folder / target for folder in [path.parent, unit.parent] + roots]
                found = next((choice for choice in choices if choice.is_file()), None)
                if found is None and _launch_generated(choices[0], export):
                    continue
                _need(found is not None and _inside(found, export), 'Missing/external INCLUDE')
                yield from physical(found, active)
            else:
                _need(not _compact(text).startswith('INCLUDE'), 'Unknown INCLUDE form')
                if text:
                    yield _Row(path, number, text, raw[:5].strip()), raw[5] not in ' 0'
    rows = []
    for row, continuation in physical(unit, []):
        if continuation:
            _need(rows and not row.label, 'Orphan continuation')
            old = rows[-1]
            # An included type declaration can precede a source continuation
            # (cts_mpc.h).  Keep included provenance for that joined statement.
            rows[-1] = old._replace(text=old.text + row.text)
        else:
            rows.append(row)
    _need(not any(_compact(row.text).startswith('INCLUDE') for row in rows), 'Unexpanded INCLUDE')
    return rows


def _arguments(text):
    result, depth, start = [], 0, 0
    for index, char in enumerate(text):
        if char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
        elif char == ',' and depth == 0:
            result.append(text[start:index])
            start = index + 1
        _need(depth >= 0, 'Unbalanced arguments')
    _need(depth == 0, 'Unbalanced arguments')
    return result + [text[start:]]


def _subroutine(rows, name):
    starts = [index for index, row in enumerate(rows)
              if _compact(row.text).startswith('SUBROUTINE' + name + '(')]
    _need(len(starts) == 1, 'Missing/duplicate consumer routine')
    start = starts[0]
    end = next((index for index in range(start + 1, len(rows))
                if _compact(rows[index].text) in ('END', 'ENDSUBROUTINE', 'ENDSUBROUTINE' + name)), None)
    _need(end is not None, 'Unterminated consumer routine')
    return rows[start:end + 1]



def _scope(rows, index):
    starts = [offset for offset in range(index + 1)
              if _compact(rows[offset].text).startswith('SUBROUTINE')]
    _need(starts, 'COMMON outside a known subroutine')
    start = starts[-1]
    end = next((offset for offset in range(index + 1, len(rows))
                if _compact(rows[offset].text) == 'END' or
                _compact(rows[offset].text).startswith('ENDSUBROUTINE')), None)
    _need(end is not None, 'Unterminated COMMON scope')
    return rows[start:end + 1]


def _scope_bounds(rows, prefix):
    """All storage dimensions must bind the one certified constant module."""
    texts = [_compact(row.text) for row in rows]
    _need(texts.count('USE' + prefix + 'POLYNOMIAL_CONSTANTS') == 1,
          'Unbound polynomial constant module')
    constants = ('MAXLWFSIZE', 'LOOPMAXCOEFS', 'VERTEXMAXCOEFS')
    allowed_use = {'USE' + prefix + 'POLYNOMIAL_CONSTANTS', 'USEALOHA_OBJECT',
                   'USE,INTRINSIC::IEEE_ARITHMETIC,ONLY:IEEE_IS_FINITE',
                   'USE,INTRINSIC::IEEE_EXCEPTIONS',
                   'USE,INTRINSIC::IEEE_EXCEPTIONS,ONLY:IEEE_GET_FLAG,IEEE_INEXACT',
                   'USE,INTRINSIC::IEEE_EXCEPTIONS,ONLY:IEEE_INEXACT,IEEE_GET_FLAG'}
    for text in texts:
        _need(not text.startswith('USE') or re.match(r'USE[A-Z0-9_]+=(?!=)', text) or text in allowed_use, 'Unknown imported consumer binding')
        _need(not text.startswith('EXTERNAL'), 'Local external consumer binding')
        if text.startswith('SUBROUTINE') and '(' in text:
            arguments = text.split('(', 1)[1]
            _need(not any(prefix + name in arguments for name in
                          ('CREATE_LOOP_COEFS', 'MERGE_WL', 'UPDATE_WL_', 'DP_TERMINAL_WL_')),
                  'Procedure dummy can redirect consumer')
        _need(not re.match(r'(?:CONTAINS|INTERFACE|ABSTRACTINTERFACE|ENTRY|PROCEDURE)', text)
              and '=>' not in text, 'Shadowed/renamed consumer binding')
        _need(not any(re.search(r'(?<![A-Z0-9_])' + name + r'=', text) for name in constants),
              'Local polynomial bound override')
        # A local declaration can shadow a module PARAMETER without a numeric
        # assignment in this file (e.g. COMMON or a dummy argument).
        declaration = re.match(r'(?:INTEGER|REAL|COMPLEX|LOGICAL)(?:\*\d+|\([^)]*\))?(?:::)?(.*)', text)
        if declaration:
            _need(not any(re.match(r'(?:' + '|'.join(constants) + r')(?:$|[=(])', item)
                          for item in _arguments(declaration[1])), 'Local polynomial bound declaration')
        _need(not (text.startswith('SUBROUTINE') and any(name in text for name in constants)),
              'Polynomial bound dummy argument')



def _canonical_create(rows, prefix):
    """Match the generated Born consumer, including its unchanged scalar bounds.

    Only literal process dimensions vary. Loop-induced and custom forms are
    deliberately outside this initial optimization and retain dense updates.
    """
    repository = Path(__file__).resolve().parents[2]
    template = (repository / 'madgraph/iolibs/template_files/loop_optimized/polynomial.inc').read_text()
    lines, active = [], False
    for raw in template.splitlines():
        if raw.strip().startswith('SUBROUTINE '):
            _need(not active, 'Unknown canonical consumer template')
            active = True
        if not active or not raw.strip() or raw[:1] in 'cC*!':
            continue
        if raw.startswith('##'):
            _need(raw.strip() in ('## if (not LoopInduced) {', '## if(not LoopInduced) {', '## }'),
                  'Unknown canonical consumer condition')
            continue
        lines.append(raw.strip())
        if raw.strip().upper() == 'END':
            break
    _need(active and lines[-1].upper() == 'END', 'Incomplete canonical consumer')
    names = {'nbornamps': 'NBORNAMPS', 'nloopamps': 'NCOLORROWS',
             'nloop_groups': 'NLOOPGROUPS', 'ncomb': 'NCOMB', 'nSO': 'NSO',
             'nSquaredSO': 'NSQUAREDSO', 'nAmpSO': 'NAMPSO'}
    dimensions = {}
    for row in rows:
        text = _compact(row.text)
        if text.startswith('PARAMETER('):
            for argument in _arguments(text[10:-1]):
                name, separator, value = argument.partition('=')
                if name in names.values():
                    _need(separator and value.isdigit() and int(value) > 0 and name not in dimensions,
                          'Nonliteral/duplicate consumer dimension')
                    dimensions[name] = int(value)
    _need(set(dimensions) == set(names.values()), 'Missing consumer dimension')
    substitutions = {key: dimensions[name] for key, name in names.items()}
    substitutions.update(mp_prefix='', proc_prefix=prefix, real_format='REAL*8',
                         complex_format='COMPLEX*16', zero_def='0.0D0', one_def='1.0D0',
                         kind=8, color_matrix_size='NBORNAMPS')
    expected = [_compact(line % substitutions) for line in lines]
    _need(not any(row.label for row in rows) and
          [_compact(row.text) for row in rows] == expected, 'Modified canonical CREATE consumer')
    wavefunction_uses = [_compact(row.text) for row in rows if _mentions(row.text, 'LOOP_WF')]
    _need(len(wavefunction_uses) == 3 and
          wavefunction_uses[0].startswith('SUBROUTINE' + prefix + 'CREATE_LOOP_COEFS(LOOP_WF,RANK,LCUT_SIZE,') and
          wavefunction_uses[1] == 'COMPLEX*16LOOP_WF(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE)' and
          wavefunction_uses[2].startswith('CALL' + prefix + 'MERGE_WL(LOOP_WF,RANK,LCUT_SIZE,') and
          sum(text.count('LOOP_WF') for text in wavefunction_uses) == 3,
          'Unknown or aliased CREATE wavefunction consumer')
    # This invariant is independent of template equality: a future template
    # edit must not silently invalidate the already-proved MERGE loop bounds.
    for bound in ('RANK', 'LCUT_SIZE'):
        uses = []
        for row in rows:
            text = _compact(row.text)
            identifiers = text[7:] if text.startswith('INTEGER') else text
            if _mentions(identifiers, bound):
                uses.append(text)
        _need(len(uses) == 3 and uses[0] == expected[0] and uses[1].startswith('INTEGER') and
              uses[2].startswith('CALL' + prefix + 'MERGE_WL(LOOP_WF,RANK,LCUT_SIZE,'),
              'Mutable or escaped CREATE bound')
        declared = _arguments(uses[1][7:])
        _need(bound in declared and all(re.fullmatch(r'[A-Z][A-Z0-9_]*', item) for item in declared),
              'Nonscalar CREATE bound')
        _need(len(re.findall(r'(?<![A-Z_])' + bound + r'(?![A-Z0-9_])', uses[2])) == 1,
              'Repeated CREATE bound argument')


def _readonly_ncoef(expanded, certified_rows):
    """The initialized count table must not later be written or escape by alias."""
    for rows in expanded.values():
        for row in rows:
            text = _compact(row.text)
            if 'NCOEF_R' not in text or row in certified_rows:
                continue
            imported = re.fullmatch(r'USE[A-Z0-9_]*POLYNOMIAL_CONSTANTS,ONLY:NCOEF_R(?:,COEFTORANK_MAP)?', text)
            loop = re.fullmatch(r'DO[A-Z][A-Z0-9_]*=0,NCOEF_R\([A-Z][A-Z0-9_]*\)-1', text)
            scalar = re.fullmatch(r'([A-Z][A-Z0-9_]*)=NCOEF_R\([A-Z][A-Z0-9_]*\)', text)
            _need(imported or loop or (scalar and scalar[1] != 'NCOEF_R'),
                  'Mutable, escaped or shadowed coefficient-count table')


def _polynomial(rows, prefix):
    module = 'MODULE' + prefix + 'POLYNOMIAL_CONSTANTS'
    starts = [index for index, row in enumerate(rows) if _compact(row.text) == module]
    _need(len(starts) == 1, 'Missing/duplicate polynomial module')
    start = starts[0]
    end = next((index for index in range(start + 1, len(rows))
                if _compact(rows[index].text) in ('ENDMODULE', 'END' + module)), None)
    _need(end is not None, 'Unterminated polynomial module')
    values = {}
    for row in rows[start:end + 1]:
        normalized = _compact(row.text)
        _need(normalized == module or normalized in ('IMPLICITNONE', 'ENDMODULE', 'END' + module) or
              normalized.startswith(('INTEGER', 'PARAMETER(', 'DATA')), 'Nonconstant polynomial module')
        _need('EXTERNAL' not in normalized and '=>' not in normalized, 'Aliased polynomial module')
        if normalized.startswith('PARAMETER('):
            for argument in _arguments(normalized[10:-1]):
                if any(argument.startswith(name + '=') for name in ('MAXLWFSIZE', 'LOOPMAXCOEFS', 'VERTEXMAXCOEFS')):
                    _need(re.fullmatch(r'[A-Z]+=[0-9]+', argument), 'Nonliteral polynomial bound')
                match = re.fullmatch(r'(MAXLWFSIZE|LOOPMAXCOEFS|VERTEXMAXCOEFS)=(\d+)', argument)
                if match:
                    key, value = match[1], int(match[2])
                    _need(key not in values or values[key] == value, 'Inconsistent polynomial bounds')
                    values[key] = value
    _need(values.get('MAXLWFSIZE') == 4 and values.get('LOOPMAXCOEFS', 0) >= 15
          and values.get('VERTEXMAXCOEFS', 0) >= 5, 'Unsupported polynomial bounds')
    counts = [row for row in rows[start:end + 1] if 'NCOEF_R' in _compact(row.text)]
    _need(len(counts) == 2, 'Missing/duplicate coefficient-count table')
    declaration = re.fullmatch(r'INTEGERNCOEF_R\(0:([0-9]+)\)', _compact(counts[0].text))
    _need(declaration is not None and int(declaration[1]) >= 2, 'Unproved coefficient-count extent')
    rank = int(declaration[1])
    # Bound the proof work by the declared storage even for malformed input.
    _need(rank <= values['LOOPMAXCOEFS'], 'Excessive coefficient-count rank')
    canonical = [(r + 1) * (r + 2) * (r + 3) * (r + 4) // 24 for r in range(rank + 1)]
    _need(canonical[2] == 15 and canonical[-1] <= values['LOOPMAXCOEFS'] and
          _compact(counts[1].text) == 'DATANCOEF_R/' + ','.join(map(str, canonical)) + '/',
          'Noncanonical/out-of-storage coefficient count')
    merger = _subroutine(rows, prefix + 'MERGE_WL')
    _need(not any(row.label for row in merger), 'Labeled merger')
    _need([_compact(row.text) for row in merger] == [
        'SUBROUTINE' + prefix + 'MERGE_WL(WL,R,LCUT_SIZE,CONST,OUT)',
        'USE' + prefix + 'POLYNOMIAL_CONSTANTS', 'INTEGERI,J',
        'COMPLEX*16WL(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE)',
        'INTEGERR,LCUT_SIZE', 'COMPLEX*16CONST',
        'COMPLEX*16OUT(0:LOOPMAXCOEFS-1)', 'DOI=1,LCUT_SIZE',
        'DOJ=0,NCOEF_R(R)-1', 'OUT(J)=OUT(J)+WL(I,J,I)*CONST',
        'ENDDO', 'ENDDO', 'END'], 'Unproved diagonal merger')
    create = _subroutine(rows, prefix + 'CREATE_LOOP_COEFS')
    _canonical_create(create, prefix)
    return set(counts)


def _canonical_comb(rows, prefix, mother_ranks):
    """Certify the higher-rank updater's literal coefficient-product table.

    This additional gate is deliberately absent from the existing 1x1 path.
    Columns 0..4 multiply a monomial by 1,q0,q1,q2,q3.  The generator also
    initializes unused final rows whose products exceed output storage; only
    rows below NCOEF_R(mother_rank) are reachable by the supported helpers.
    """
    from itertools import combinations_with_replacement

    module = 'MODULE' + prefix + 'POLYNOMIAL_CONSTANTS'
    start = next(index for index, row in enumerate(rows)
                 if _compact(row.text) == module)
    end = next(index for index in range(start + 1, len(rows))
               if _compact(rows[index].text) in ('ENDMODULE', 'END' + module))
    body = rows[start:end + 1]
    texts = [_compact(row.text) for row in body]
    declaration = next(re.fullmatch(r'INTEGERNCOEF_R\(0:([0-9]+)\)', text)
                       for text in texts if text.startswith('INTEGERNCOEF_R('))
    rank = int(declaration[1])
    _need(rank in (3, 4, 5, 6), 'Unsupported higher-terminal process rank')
    counts = [(r + 1) * (r + 2) * (r + 3) * (r + 4) // 24
              for r in range(rank + 2)]
    parameters = {}
    for text in texts:
        if text.startswith('PARAMETER('):
            for argument in _arguments(text[10:-1]):
                name, _, value = argument.partition('=')
                parameters[name] = value
    _need(parameters.get('LOOPMAXCOEFS') == str(counts[rank]) and
          parameters.get('VERTEXMAXCOEFS') == '5',
          'Unsupported higher-terminal coefficient storage')
    _need(all(left in (2, 3) and left + 1 <= rank for left in mother_ranks),
          'Higher terminal rank exceeds coefficient storage')
    table = [row for row in body if 'COMB_COEF_POS' in _compact(row.text)]
    _need(len(table) == counts[rank] + 1 and _compact(table[0].text) ==
          'INTEGERCOMB_COEF_POS(0:LOOPMAXCOEFS-1,0:4)',
          'Combination table declaration/extent')
    monomials = []
    for degree in range(rank + 2):
        monomials.extend(sorted(combinations_with_replacement(range(4), degree),
                                key=lambda indices: indices[::-1]))
    positions = {indices: index for index, indices in enumerate(monomials)}
    reachable = max(counts[left] for left in mother_ranks)
    for index, row in enumerate(table[1:]):
        base = monomials[index]
        values = [index] + [positions[tuple(sorted(base + (axis,)))]
                            for axis in range(4)]
        expected = 'DATACOMB_COEF_POS(%d,0:4)/%s/' % (
            index, ','.join(map(str, values)))
        _need(_compact(row.text) == expected, 'Noncanonical combination table entry')
        if index < reachable:
            _need(all(0 <= value < counts[rank] for value in values),
                  'Consumed combination entry exceeds output storage')
    return set(table)


def _readonly_comb(expanded, certified_rows, prefix):
    """Reject table writes, shadows, renamed imports and argument escapes.

    Generated readers import the certified module directly.  Restrict their
    other imports to intrinsic IEEE modules so an added module or interface
    cannot change the binding of a textually unchanged table lookup.
    """
    for rows in expanded.values():
        checked_scopes = set()
        for index, row in enumerate(rows):
            text = _compact(row.text)
            if 'COMB_COEF_POS' not in text or row in certified_rows:
                continue
            _need(re.fullmatch(r'NEW_POSITION=COMB_COEF_POS\(L,(?:M|[1-4])\)', text),
                  'Mutable, escaped or shadowed combination table')
            scope = _scope(rows, index)
            if scope[0] in checked_scopes:
                continue
            checked_scopes.add(scope[0])
            statements = [_compact(item.text) for item in scope]
            _need(statements.count('USE' + prefix + 'POLYNOMIAL_CONSTANTS') == 1,
                  'Unbound combination table reader')
            allowed_imports = {'USE' + prefix + 'POLYNOMIAL_CONSTANTS',
                               'USE,INTRINSIC::IEEE_ARITHMETIC',
                               'USE,INTRINSIC::IEEE_EXCEPTIONS'}
            _need(all(not text.startswith('USE') or text in allowed_imports
                      for text in statements), 'Unknown combination reader import')
            # Existing IEEE guards contain local bit-classification functions
            # and ASSOCIATE views of TRANSFER results. They do not alias the
            # table: every occurrence of its name is checked above, including
            # declarations and aliases in contained procedures.
            _need(not any(text.startswith(('INTERFACE', 'ABSTRACTINTERFACE',
                                           'ENTRY', 'PROCEDURE'))
                          for text in statements), 'Aliased combination reader binding')


def _private_references(expanded, directory, prefix, certified_calls):
    """Every private invocation must belong to the certified producer graph.

    Checking only COMMON holders misses a caller that supplies some other
    array and manufactures the entry flag. Reject all other references too:
    function syntax, procedure arguments/aliases, and unknown rank spellings.
    """
    stem = prefix + 'DP_TERMINAL_WL_'
    names = {stem + pair for pair in ('1_1', '2_1', '3_1')}
    definitions = set()
    for unit, rows in expanded.items():
        for row in rows:
            text = _compact(row.text)
            if stem not in text:
                continue
            if (unit, row) in certified_calls:
                _need(row.origin == unit and text.count(stem) == 1,
                      'Aliased certified private invocation')
                continue
            _need(unit == directory / 'polynomial.f' and row.origin == unit,
                  'Private reference outside certified coefficient graph')
            definition = re.fullmatch(r'SUBROUTINE(' + re.escape(stem) + r'[123]_1)\(.*\)', text)
            if definition and text.count(stem) == 1:
                _need(definition[1] not in definitions, 'Duplicate private helper definition')
                definitions.add(definition[1])
            else:
                _need(text in {'ENDSUBROUTINE' + name for name in names},
                      'Uncertified private helper reference')


def _module_bodies(expanded, name):
    bodies = []
    for rows in expanded.values():
        for start, row in enumerate(rows):
            if _compact(row.text) != 'MODULE' + name:
                continue
            end = next((index for index in range(start + 1, len(rows))
                        if _compact(rows[index].text) in ('ENDMODULE', 'ENDMODULE' + name)), None)
            _need(end is not None, 'Unterminated ' + name + ' module')
            bodies.append([_compact(item.text) for item in rows[start + 1:end]])
    return bodies


@functools.lru_cache(maxsize=None)
def _canonical_aloha_module():
    repository = Path(__file__).resolve().parents[2]
    path = repository / 'aloha/template_files/aloha_functions_loop.f'
    return _module_bodies({path: _expanded(path, repository, ())}, 'ALOHA_OBJECT')[0]


def _aloha_modules(expanded, prefix):
    """Bind ALOHA records and the generated scalar flavour lookup helpers.

    The helpers import MODEL_OBJECT.  Its only exports may be the passive
    coupling records and their instances: an interface or procedure there
    could otherwise redirect an unchanged call through ALOHA_OBJECT.
    """
    modules = _module_bodies(expanded, 'ALOHA_OBJECT')
    _need(len(modules) <= 1, 'Repeated ALOHA module')
    allowed = {'SEQUENCE', 'DOUBLECOMPLEX::W(4)', 'DOUBLECOMPLEX::W(16)',
               'DOUBLECOMPLEX::P(0:3)', 'INTEGER::FLV_INDEX',
               'COMPLEX*32::W(4)', 'COMPLEX*32::W(16)', 'COMPLEX*32::P(0:3)'}
    allowed |= {marker + name for marker in ('TYPE', 'ENDTYPE')
                for name in ('ALOHA', 'ALOHA2D', 'MP_ALOHA', 'MP_ALOHA2D')}
    for body in modules:
        if all(text in allowed for text in body):
            continue
        _need(body == _canonical_aloha_module(), 'Active/aliased ALOHA module')
        models = _module_bodies(expanded, 'MODEL_OBJECT')
        _need(len(models) == 1, 'Missing/repeated model object module')
        model = models[0]
        _need(len(model) >= 10 and model[:6] == [
            'TYPECOUPPTR', 'SEQUENCE', 'DOUBLECOMPLEX,POINTER::P', 'ENDTYPECOUPPTR',
            'TYPEFLV_COUPLING', 'SEQUENCE'] and model[9] == 'ENDTYPEFLV_COUPLING',
            'Active/aliased model object module')
        extents = [re.fullmatch(pattern, text) for pattern, text in zip((
            r'INTEGER::PARTNER\(([0-9]+)\)', r'INTEGER::PARTNER2\(([0-9]+)\)',
            r'TYPE\(COUPPTR\)::VAL\(([0-9]+)\)'), model[6:9])]
        _need(all(extents) and len({match[1] for match in extents}) == 1,
              'Unknown model flavour record')
        protected = ('MAXLWFSIZE', 'LOOPMAXCOEFS', 'VERTEXMAXCOEFS') + tuple(
            prefix + name for name in ('CREATE_LOOP_COEFS', 'MERGE_WL',
                                      'UPDATE_WL_', 'DP_TERMINAL_WL_'))
        for text in model[10:]:
            match = re.fullmatch(r'TYPE\(FLV_COUPLING\)::([A-Z][A-Z0-9_]*(?:,[A-Z][A-Z0-9_]*)*)', text)
            _need(match and not any(name in match[1].split(',') or
                  (name.endswith('_') and any(item.startswith(name) for item in match[1].split(',')))
                  for name in protected), 'Active/aliased model object export')


def _proof(directory, prefix, expected, source_manifest, include_roots):
    directory = Path(directory).resolve()
    _need(directory.is_dir() and re.fullmatch(r'(?:[A-Za-z][A-Za-z0-9_]*)?', prefix), 'Invalid proof scope')
    prefix = prefix.upper()
    _need(isinstance(expected, dict) and expected, 'No terminal candidates')
    for number, item in expected.items():
        _need(isinstance(number, int) and number > 0 and isinstance(item, dict), 'Invalid terminal map')
        _need(item.get('out') == number and item.get('loop_mother_rank') in (1, 2, 3)
              and item.get('vertex_rank') == 1
              and all(item.get(key) == 4 for key in ('lcut_size', 'in_size', 'out_size')),
              'Unsupported terminal shape')
    export, layout, units, roots = _inventory(directory, source_manifest, include_roots)
    _build_inputs(export, directory, layout)
    if layout == 'fks':
        _foreign_scan(export, directory, prefix)
    expanded = {unit: _expanded(unit, export, roots) for unit in units}
    polynomial = next((rows for unit, rows in expanded.items()
                       if unit.resolve() == (directory / 'polynomial.f').resolve()), None)
    _need(polynomial is not None, 'Missing polynomial')
    certified_counts = _polynomial(polynomial, prefix)
    _readonly_ncoef(expanded, certified_counts)
    higher_ranks = {item['loop_mother_rank'] for item in expected.values()
                    if item['loop_mother_rank'] > 1}
    if higher_ranks:
        certified_comb = _canonical_comb(polynomial, prefix, higher_ranks)
        _readonly_comb(expanded, certified_comb, prefix)
    protected = (prefix + 'CREATE_LOOP_COEFS', prefix + 'MERGE_WL',
                 prefix + 'DP_TERMINAL_WL_1_1')
    if higher_ranks:
        protected += (prefix + 'DP_TERMINAL_WL_2_1', prefix + 'DP_TERMINAL_WL_3_1')
    for unit, rows in expanded.items():
        for row in rows:
            text = _compact(row.text)
            _need('BIND(C' not in text, 'External binding can alias DP storage')
            if any(text.startswith('SUBROUTINE' + name + '(') for name in protected) or text == 'MODULE' + prefix + 'POLYNOMIAL_CONSTANTS':
                _need(unit.resolve() == (directory / 'polynomial.f').resolve(), 'Shadowed protected routine/module')
    # ALOHA_OBJECT is the only other generated module imported by the COMMON
    # holders; bind its flavour helpers as well as the wavefunction records.
    _aloha_modules(expanded, prefix)
    blocks, batch_rows, wavefunction_bounds = {}, [], set()
    common_token = '/' + prefix + 'WL/'
    for unit, rows in expanded.items():
        matches = [row for row in rows if common_token in _compact(row.text)]
        if not matches:
            continue
        _need(unit.parent == directory and len(matches) == 1, 'Unknown or duplicate DP COMMON holder')
        holder = _scope(rows, rows.index(matches[0]))
        _scope_bounds(holder, prefix)
        sizes = []
        for row in holder:
            normalized = _compact(row.text)
            if normalized.startswith('PARAMETER('):
                for item in _arguments(normalized[10:-1]):
                    if item.startswith('NLOOPWAVEFUNCS='):
                        _need(re.fullmatch(r'NLOOPWAVEFUNCS=[0-9]+', item), 'Unresolved COMMON extent')
                        sizes.append(int(item.split('=')[1]))
        _need(len(sizes) == 1 and sizes[0] > 0, 'Missing COMMON extent')
        wavefunction_bounds.add(sizes[0])
        name = unit.name
        is_batch = bool(re.fullmatch(r'coef_construction_\d+\.f', name))
        seed = name in ('loop_matrix.f', 'mp_compute_loop_coefs.f')
        _need(is_batch or seed or re.fullmatch(r'(?:helas_calls_(?:ampb|uvct)|loop_CT_calls)_\d+\.f', name),
              'Unknown DP COMMON holder')
        alias = 'DP_WL' if name == 'mp_compute_loop_coefs.f' else 'WL'
        momentum = 'DP_PL' if alias == 'DP_WL' else 'PL'
        common = 'COMMON' + common_token + alias + ',' + momentum
        declaration = 'COMPLEX*16' + alias + '(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE,-1:NLOOPWAVEFUNCS)'
        momentum_declaration = 'COMPLEX*16' + momentum + '(0:3,-1:NLOOPWAVEFUNCS)'
        _need(_compact(matches[0].text) == common and matches[0].origin == unit, 'DP COMMON alias/layout')
        for target in (declaration, momentum_declaration):
            _need(sum(_compact(row.text) == target and row.origin == unit for row in rows) == 1,
                  'DP COMMON storage declaration')
        _need(not any(re.search(r'^(?:EQUIVALENCE|POINTER|TARGET)|(?:^|[=(,])(?:LOC|C_LOC)\(', _compact(row.text))
                      for row in rows), 'Potential storage alias')
        uses = []
        for row in rows:
            if not _mentions(row.text, alias):
                continue
            _need(row.origin == unit, 'Included DP COMMON use')
            if _compact(row.text) not in (common, declaration):
                uses.append(row)
        blocks[name] = alias
        if is_batch:
            batch_rows.append((int(name[len('coef_construction_'):-2]), unit, rows, uses))
        elif seed:
            allowed = {alias + suffix for suffix in (
                '(I,J,K,-1)=(0.0D0,0.0D0)', '(I,J,K,0)=(1.0D0,0.0D0)',
                '(I,J,K,0)=(0.0D0,0.0D0)', '(1,0,2,-1)=DCMPLX(-1.0D0,0.0D0)',
                '(2,0,1,-1)=DCMPLX(1.0D0,0.0D0)', '(3,0,4,-1)=DCMPLX(1.0D0,0.0D0)',
                '(4,0,3,-1)=DCMPLX(-1.0D0,0.0D0)')}
            _need(len(uses) == 7 and {_compact(row.text) for row in uses} == allowed,
                  'Non-seed DP COMMON use')
        else:
            _need(not uses, 'Unknown DP consumer')
    _need('loop_matrix.f' in blocks and 'mp_compute_loop_coefs.f' in blocks and batch_rows,
          'Incomplete DP COMMON graph')
    # A batch without the expected block must not silently drop out of the proof.
    _need({path.name for path in directory.glob('coef_construction_*.f')} ==
          {unit.name for _, unit, _, _ in batch_rows}, 'Unbound coefficient batch')
    produced, consumed, roots_seen, private = {}, set(), set(), {}
    certified_private_calls = set()
    updater = re.compile(r'CALL' + re.escape(prefix) + r'(UPDATE_WL|DP_TERMINAL_WL)_(\d+)_(\d+)\((.*)\)$')
    wavefunction = re.compile(r'WL\(1,0,1,(-?\d+)\)$')
    for _, unit, rows, uses in sorted(batch_rows):
        normalized = [_compact(row.text) for row in rows]
        private_here = any('DP_TERMINAL_WL_' in text for text in normalized)
        if private_here:
            _need(normalized.count('LOGICALWL_ENTRY_INEXACT') == 1 and
                  any(text in ('USE,INTRINSIC::IEEE_EXCEPTIONS',
                               'USE,INTRINSIC::IEEE_EXCEPTIONS,ONLY:IEEE_GET_FLAG,IEEE_INEXACT',
                               'USE,INTRINSIC::IEEE_EXCEPTIONS,ONLY:IEEE_INEXACT,IEEE_GET_FLAG')
                      for text in normalized), 'Unbound IEEE capture')
            for row in rows:
                text = _compact(row.text)
                _need(not re.search(r'CALLIEEE_SET_', text), 'Manufactured IEEE admission')
                if _mentions(row.text, 'WL_ENTRY_INEXACT'):
                    _need(text == 'LOGICALWL_ENTRY_INEXACT' or
                          text == 'CALLIEEE_GET_FLAG(IEEE_INEXACT,WL_ENTRY_INEXACT)' or
                          any(text.startswith('CALL' + prefix + 'DP_TERMINAL_WL_' + pair + '(')
                              for pair in ('1_1', '2_1', '3_1')),
                          'Aliased or modified capture flag')
        for row in uses:
            text = _compact(row.text)
            match = updater.fullmatch(text)
            if match:
                arguments = _arguments(match[4])
                is_private = match[1] == 'DP_TERMINAL_WL'
                _need(len(arguments) == (7 if is_private else 6), 'Updater argument count')
                source, output = wavefunction.fullmatch(arguments[0]), wavefunction.fullmatch(arguments[5])
                _need(source and output and arguments[2] == 'COEFS' and
                      all(arguments[index].isdigit() for index in (1, 3, 4)), 'Unknown updater view')
                source, output = int(source[1]), int(output[1])
                left, right = int(match[2]), int(match[3])
                _need(output > 0 and output not in produced and source != output and
                      (source in (-1, 0) or source in produced), 'Broken producer order')
                _need(all(0 < int(arguments[index]) <= 4 for index in (1, 3, 4)), 'Updater shape exceeds storage')
                if source in produced:
                    mother = produced[source]
                    _need(left == mother['loop_mother_rank'] + mother['vertex_rank'] and
                          int(arguments[1]) == mother['lcut_size'] and
                          int(arguments[3]) == mother['out_size'], 'Producer rank/shape differs')
                else:
                    _need(left == 0, 'Seed has nonzero rank')
                produced[output] = dict(loop_mother_rank=left, vertex_rank=right,
                    loop_mother_number=source, lcut_size=int(arguments[1]),
                    in_size=int(arguments[3]), out_size=int(arguments[4]), out=output)
                consumed.add(source)
                if is_private:
                    index = rows.index(row)
                    _need((left, right) in ((1, 1), (2, 1), (3, 1)) and arguments[6] == 'WL_ENTRY_INEXACT' and
                          index > 0 and normalized[index - 1] == 'CALLIEEE_GET_FLAG(IEEE_INEXACT,WL_ENTRY_INEXACT)'
                          and not row.label and not rows[index - 1].label and rows[index - 1].origin == unit,
                          'Terminal call without immediate capture')
                    private[output] = produced[output]
                    certified_private_calls.add((unit, row))
            elif text.startswith('CALL' + prefix + 'CREATE_LOOP_COEFS(') and text.endswith(')'):
                arguments = _arguments(text[len('CALL' + prefix + 'CREATE_LOOP_COEFS('):-1])
                root = wavefunction.fullmatch(arguments[0]) if arguments else None
                _need(root and len(arguments) == 8 and text.count('WL(') == 1,
                      'Unknown terminal consumer arguments')
                number = int(root[1])
                _need(number in produced and arguments[1].isdigit() and arguments[2].isdigit(),
                      'Consumer precedes producer')
                _need(int(arguments[1]) == produced[number]['loop_mother_rank'] + produced[number]['vertex_rank']
                      and int(arguments[2]) == produced[number]['lcut_size'], 'Consumer shape differs')
                roots_seen.add(number)
            else:
                raise _Unproved('Unknown coefficient-batch WL use')
    _need(len(wavefunction_bounds) == 1 and max(produced) <= next(iter(wavefunction_bounds)), 'Inconsistent COMMON extents')
    _need(private == expected and set(private) <= roots_seen - consumed, 'Nonterminal or missing private updater')
    _private_references(expanded, directory, prefix, certified_private_calls)
    return True


def validate_dp_terminal_wavefunctions(directory, prefix, expected,
                                      source_manifest=None, include_roots=()):
    """Return whether a proposed DP rewrite has only diagonal consumers.

    ``directory`` is a standalone SubProcesses/P* directory or an MC@NLO
    SubProcesses/P*/V* loop directory (see the module documentation).
    ``expected`` is the terminal map returned by the HELAS graph inspection,
    narrowed to ranks 1/2/3 x 1 and shape (4,4,4). Higher ranks additionally
    require a canonical, read-only coefficient-product table in process ranks
    3..6 with a rank-one vertex. The optional ``source_manifest`` is an
    exact iterable of generated Fortran units (paths relative to ``directory``
    or absolute): it must include the units checked in full (standalone:
    SubProcesses, MODEL and DHELAS; MC@NLO: the loop directory, MODEL and
    DHELAS), not merely coefficient batches.  ``include_roots`` may add
    generated include search directories inside the same export.  Unknown
    layouts, source kinds, consumers, aliases and incomplete exports return
    False.
    """
    try:
        return _proof(directory, prefix, expected, source_manifest, include_roots)
    except _Unproved as error:
        logger.debug('DP terminal proof refused for %s: %s', directory, error)
        return False
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, RecursionError) as error:
        logger.debug('DP terminal proof failed for %s: %r', directory, error)
        return False
