"""Fail-closed consumer proof for the private binary64 terminal updater.

This deliberately recognizes a small part of the standalone generated Fortran
language, rather than attempting an arbitrary-Fortran alias analysis.  Call it
on the completed export *after* proposing the private calls; False means that
all proposed calls must be restored to their ordinary dense spelling.  The
private updater's arithmetic/IEEE admission proof is separate from this
storage and consumer proof.  In particular, externally linked reduction
libraries and the compiler are trusted, as they are for the dense program.
"""
from collections import namedtuple
from pathlib import Path
import os
import re


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


def _inventory(directory, source_manifest, include_roots):
    export = directory
    for parent in directory.parents:
        if (parent / 'SubProcesses').is_dir() and (parent / 'Source').is_dir():
            export = parent
            break
    scopes = [directory]
    if export != directory:
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
            _need(path.suffix == '.f', 'Unsupported compilation language')
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
    return export, list(units.values()), roots



def _build_inputs(export, directory):
    """Accept the generator's standard recipes, not arbitrary make programs.

    Comparing current generator templates is a grammar boundary, not a frozen
    export hash. Compiler selection and linked reducer libraries remain trusted.
    """
    repository = next(p for p in Path(__file__).resolve().parents if (p / 'madgraph/loop/dp_terminal_proof.py').is_file())
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
            _need(all((root / (obj[:-2] + '.f')).is_file() for obj in objects), 'Missing compiled source')
            # Source enumeration is deliberately a superset (e.g. testprog).
            _need(all((root / (stem + '.f')).is_file() for stem in basic), 'Missing core source')
    path = export / 'Source/make_opts'
    if path.exists():
        _need(_inside(path, export), 'External compiler options')
        marker = '#end_of_make_opts_variables'
        actual = path.read_text().split(marker)
        standard = (repository / 'Template/LO/Source/.make_opts').read_text().split(marker)
        _need(len(actual) == len(standard) == 2, 'Unknown make_opts structure')
        normalize = lambda text: [line.strip() for line in text.splitlines()
                                  if line.strip() and not line.lstrip().startswith('#')]
        _need(normalize(actual[1]) == normalize(standard[1]), 'Modified compiler recipes')
        variables = {}
        for line in normalize(actual[0]):
            match = re.fullmatch(r'([A-Z0-9_]+)\s*=\s*([A-Za-z0-9_./:+ =-]*)', line)
            _need(match and match[1] not in variables, 'Unknown compiler option assignment')
            variables[match[1]] = match[2]
        _need(set(variables) == {'DEFAULT_F2PY_COMPILER', 'DEFAULT_F_COMPILER',
                                'DEFAULT_CPP_COMPILER', 'MACFLAG', 'MG5AMC_VERSION',
                                'STDLIB', 'PYTHIA8_PATH', 'STDLIB_FLAG'},
              'Unknown compiler option variable')
        _need(re.fullmatch(r'[A-Za-z0-9_./+-]+', variables['DEFAULT_F_COMPILER']),
              'Fortran compiler wrapper/options outside proof')
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
            # Only the canonical make references are permitted in formatted
            # values; no shell/eval/call or injected object/source arguments.
            rest = re.sub(r'\$\((?:LIBDIR|libext|dylibext)\)', '', line)
            _need('$' not in rest and ';' not in rest and '`' not in rest and
                  not re.search(r'(?:^|\s)[^ ]+\.(?:o|f|f90|c|cpp)(?:\s|$)', rest),
                  'Custom reducer build input')



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
    repository = next(p for p in Path(__file__).resolve().parents if (p / 'madgraph/loop/dp_terminal_proof.py').is_file())
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



def _canonical_comb(rows, prefix):
    """Certify the complete rank-four export table independently of DATA text.

    Monomials use MadLoop's symmetric colexicographic order. Row L times
    column zero preserves the monomial; columns 1..4 multiply q0..q3.
    The final rows legitimately map beyond storage and are never read by
    supported rank2/3 x rank1 kernels (their maximum rows are 14/34).
    """
    from itertools import combinations_with_replacement
    module = 'MODULE' + prefix + 'POLYNOMIAL_CONSTANTS'
    start = next(i for i, row in enumerate(rows) if _compact(row.text) == module)
    end = next(i for i in range(start + 1, len(rows))
               if _compact(rows[i].text) in ('ENDMODULE', 'END' + module))
    body = rows[start:end + 1]
    text = [_compact(row.text) for row in body]
    _need('PARAMETER(LOOPMAXCOEFS=70)' in text or any(
          re.search(r'(?:\(|,)LOOPMAXCOEFS=70(?:,|\))', t) for t in text),
          'Private experiment requires rank-four storage')
    _need('INTEGERNCOEF_R(0:4)' in text and 'DATANCOEF_R/1,5,15,35,70/' in text,
          'Private experiment requires exact rank-four counts')
    relevant = [row for row in body if 'COMB_COEF_POS' in _compact(row.text)]
    _need(len(relevant) == 71 and _compact(relevant[0].text) ==
          'INTEGERCOMB_COEF_POS(0:LOOPMAXCOEFS-1,0:4)', 'Combination table shape/declaration')
    monomials = []
    for rank in range(6):
        monomials.extend(sorted(combinations_with_replacement(range(4), rank), key=lambda t: t[::-1]))
    positions = {value: index for index, value in enumerate(monomials)}
    for index, row in enumerate(relevant[1:]):
        base = monomials[index]
        values = [index] + [positions[tuple(sorted(base + (axis,)))] for axis in range(4)]
        expected = 'DATACOMB_COEF_POS(%d,0:4)/%s/' % (index, ','.join(map(str, values)))
        _need(_compact(row.text) == expected, 'Noncanonical combination table entry')
        if index < 35:
            _need(all(0 <= value < 70 for value in values), 'Consumed map entry exceeds output storage')
    return set(relevant)


def _readonly_comb(expanded, certified_rows):
    """Reject every write, alias, declaration shadow, import rename or escape."""
    for rows in expanded.values():
        for row in rows:
            text = _compact(row.text)
            if 'COMB_COEF_POS' not in text or row in certified_rows:
                continue
            _need(re.fullmatch(r'NEW_POSITION=COMB_COEF_POS\(L,(?:M|[1-4])\)', text),
                  'Mutable, escaped or shadowed combination table')


def collect_expected(directory, prefix='ML5_0_', include_dense_higher=True):
    """Select terminal roots, retaining every existing private 1x1 producer.

    This is an inventory convenience, not a substitute for the full validator.
    The resulting map must be revalidated after proposing the private calls.
    """
    directory = Path(directory).resolve()
    export, units, roots = _inventory(directory, None, ())
    produced, consumed, consumers, private = {}, set(), set(), set()
    pattern = re.compile(r'CALL' + re.escape(prefix.upper()) + r'(UPDATE_WL|DP_TERMINAL_WL)_(\d+)_(\d+)\((.*)\)')
    for unit in sorted(directory.glob('coef_construction_*.f')):
        for row in _expanded(unit, export, roots):
            text = _compact(row.text)
            match = pattern.fullmatch(text)
            if match:
                args = _arguments(match[4])
                mother = int(re.fullmatch(r'WL\(1,0,1,(-?\d+)\)', args[0])[1])
                out = int(re.fullmatch(r'WL\(1,0,1,(-?\d+)\)', args[5])[1])
                _need(out not in produced, 'Duplicate producer')
                produced[out] = dict(loop_mother_rank=int(match[2]), vertex_rank=int(match[3]),
                                     loop_mother_number=mother, lcut_size=int(args[1]),
                                     in_size=int(args[3]), out_size=int(args[4]), out=out)
                consumed.add(mother)
                if match[1] == 'DP_TERMINAL_WL': private.add(out)
            elif text.startswith('CALL' + prefix.upper() + 'CREATE_LOOP_COEFS('):
                args = _arguments(text[text.index('(')+1:-1])
                consumers.add(int(re.fullmatch(r'WL\(1,0,1,(\d+)\)', args[0])[1]))
    selected = {}
    for out, item in produced.items():
        if out in private or (include_dense_higher and out in consumers - consumed and
                (item['loop_mother_rank'], item['vertex_rank']) in ((2,1),(3,1)) and
                all(item[key] == 4 for key in ('lcut_size','in_size','out_size'))):
            selected[out] = item
    return selected


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
    export, units, roots = _inventory(directory, source_manifest, include_roots)
    _build_inputs(export, directory)
    expanded = {unit: _expanded(unit, export, roots) for unit in units}
    polynomial = next((rows for unit, rows in expanded.items()
                       if unit.resolve() == (directory / 'polynomial.f').resolve()), None)
    _need(polynomial is not None, 'Missing polynomial')
    certified_counts = _polynomial(polynomial, prefix)
    certified_comb = _canonical_comb(polynomial, prefix)
    _readonly_comb(expanded, certified_comb)
    _readonly_ncoef(expanded, certified_counts)
    protected = (prefix + 'CREATE_LOOP_COEFS', prefix + 'MERGE_WL',
                 prefix + 'DP_TERMINAL_WL_1_1', prefix + 'DP_TERMINAL_WL_2_1',
                 prefix + 'DP_TERMINAL_WL_3_1')
    for unit, rows in expanded.items():
        for row in rows:
            text = _compact(row.text)
            _need('BIND(C' not in text, 'External binding can alias DP storage')
            if any(text.startswith('SUBROUTINE' + name + '(') for name in protected) or text == 'MODULE' + prefix + 'POLYNOMIAL_CONSTANTS':
                _need(unit.resolve() == (directory / 'polynomial.f').resolve(), 'Shadowed protected routine/module')
    # ALOHA_OBJECT is the only other generated module imported by the
    # COMMON holders.  It must remain a passive collection of the four
    # wavefunction records, without re-exported generic procedures.
    aloha_modules = {}
    for unit, rows in expanded.items():
        starts = [index for index, row in enumerate(rows)
                  if _compact(row.text) == 'MODULEALOHA_OBJECT']
        for start in starts:
            end = next((index for index in range(start + 1, len(rows))
                        if _compact(rows[index].text) in ('ENDMODULE', 'ENDMODULEALOHA_OBJECT')), None)
            _need(end is not None, 'Unterminated ALOHA module')
            body = [_compact(row.text) for row in rows[start + 1:end]]
            allowed = {'SEQUENCE', 'DOUBLECOMPLEX::W(4)', 'DOUBLECOMPLEX::W(16)',
                       'DOUBLECOMPLEX::P(0:3)', 'INTEGER::FLV_INDEX',
                       'COMPLEX*32::W(4)', 'COMPLEX*32::W(16)', 'COMPLEX*32::P(0:3)'}
            allowed |= {marker + name for marker in ('TYPE', 'ENDTYPE')
                        for name in ('ALOHA', 'ALOHA2D', 'MP_ALOHA', 'MP_ALOHA2D')}
            _need(all(text in allowed for text in body), 'Active/aliased ALOHA module')
            aloha_modules[(unit.resolve(), start)] = body
    _need(len(aloha_modules) <= 1, 'Repeated ALOHA module')
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
                          any(text.startswith('CALL' + prefix + 'DP_TERMINAL_WL_' + pair + '(') for pair in ('1_1', '2_1', '3_1')),
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
    return True


def validate_dp_terminal_wavefunctions(directory, prefix, expected,
                                      source_manifest=None, include_roots=()):
    """Return whether a proposed standalone DP rewrite has only diagonal consumers.

    ``expected`` is the terminal map returned by the HELAS graph inspection,
    narrowed to ranks 1/2/3 x 1 and shape (4,4,4).  The optional ``source_manifest`` is an
    exact iterable of generated Fortran units (paths relative to ``directory``
    or absolute): it must include the source closure in SubProcesses, MODEL and
    DHELAS, not merely coefficient batches.  ``include_roots`` may add generated
    include search directories inside the same export.  Unknown layouts, source
    kinds, consumers, aliases and incomplete exports return False.
    """
    try:
        return _proof(directory, prefix, expected, source_manifest, include_roots)
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, RecursionError):
        return False
