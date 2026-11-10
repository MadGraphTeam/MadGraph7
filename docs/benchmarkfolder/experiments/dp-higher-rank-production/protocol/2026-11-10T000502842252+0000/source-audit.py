#!/usr/bin/env python3
"""Fail-closed source scope audit for the narrow DP 2x1/3x1 extension.

This compares source bytes; it does not prove the new helpers' arithmetic or
consumer safety. Those separate qualifications must bind the hashes reported
here. No source, build product, or numerical input is modified.
"""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
import sys


SUFFIXES = {'.f', '.f90', '.f95', '.f03', '.f08', '.inc', '.h', '.hpp',
            '.hxx', '.c', '.cc', '.cpp', '.cxx', '.s', '.mk'}
BUILD_NAMES = {'makefile', 'gnumakefile', 'makefilep', 'make_opts',
               'madloop_makefile_definitions'}
DECL = re.compile(rb"      (?:(?:REAL\*(?:8|16)|COMPLEX\*(?:16|32)|"
                  rb"TYPE\((?:MP_)?ALOHA\)) [A-Z][A-Z0-9_]*(?:\([A-Z0-9_,:*+\-]+\))?"
                  rb"|INCLUDE 'coef_specs\.inc')\n")
GET_FLAG = b'      CALL IEEE_GET_FLAG(IEEE_INEXACT,WL_ENTRY_INEXACT)\n'
DHELAS_SPEC = (b'      INTEGER MAXLWFSIZE\n      PARAMETER (MAXLWFSIZE=4)\n'
               b'      INTEGER VERTEXMAXCOEFS\n      PARAMETER (VERTEXMAXCOEFS=5)\n')


class Refusal(ValueError):
    pass


def require(condition, reason):
    if not condition:
        raise Refusal(reason)


def digest(data):
    return sha256(data).hexdigest()


def selected(path):
    return path.suffix.lower() in SUFFIXES or path.name.lower() in BUILD_NAMES


def inventory(root):
    result = {}
    excluded = Counter()
    directory_links = {}
    for path in sorted(root.rglob('*')):
        if path.is_dir():
            if path.is_symlink():
                rel = str(path.relative_to(root))
                require(re.fullmatch(r'lib/(?:collier|ninja|ninjars|tirust)_(?:include|lib)|'
                                     r'SubProcesses/P[^/]+/MadLoop5_resources', rel) is not None,
                        'unexpected directory symlink: '+str(path))
                directory_links[rel] = str(path.readlink())
            continue
        if not selected(path):
            excluded[path.suffix.lower() or '<no suffix>'] += 1
            continue
        require(path.is_file(), 'unreadable or dangling source path: '+str(path))
        data = path.read_bytes()
        require(b'\0' not in data, 'NUL byte in selected source: '+str(path))
        rel = str(path.relative_to(root))
        result[rel] = dict(sha256=digest(data), bytes=len(data),
                          symlink_target=str(path.readlink()) if path.is_symlink() else None)
    require(result, 'empty selected source inventory: '+str(root))
    return result, dict(sorted(excluded.items())), directory_links


def declaration_parts(data):
    """Recognize only the generated ALOHA header observed in this change."""
    require(b'\r' not in data and b'\t' not in data, 'unsupported source whitespace')
    require(len(re.findall(rb'^      SUBROUTINE ', data, re.M)) == 1,
            'expected exactly one DHELAS subroutine')
    require(len(re.findall(rb'^      END\n', data, re.M)) == 1,
            'expected exactly one DHELAS routine terminator')
    header = re.search(rb'^      IMPLICIT NONE\n      COMPLEX\*(?:16|32) CI\n'
                       rb'      PARAMETER \(CI=\(0[DQ]0,1[DQ]0\)\)\n', data, re.M)
    require(header is not None, 'unrecognized fixed DHELAS declaration header')
    start = header.end()
    end = start
    declarations = []
    for line in data[start:].splitlines(keepends=True):
        if not DECL.fullmatch(line):
            break
        declarations.append(line)
        end += len(line)
    require(declarations, 'empty recognized declaration region')
    require(re.match(rb'      [A-Z][A-Z0-9_]*(?:\([^\n]*\))?\s*=', data[end:]) is not None,
            'declaration region does not end at a recognized assignment')
    require(declarations.count(b"      INCLUDE 'coef_specs.inc'\n") == 1,
            'expected exactly one coefficient specification include')
    include_index = declarations.index(b"      INCLUDE 'coef_specs.inc'\n")
    require(include_index+1 < len(declarations) and re.fullmatch(
        rb'      COMPLEX\*(?:16|32) COEFF\(MAXLWFSIZE,0:VERTEXMAXCOEFS-1,MAXLWFSIZE\)\n',
        declarations[include_index+1]) is not None,
        'coefficient declaration must immediately follow its dimension include')
    require(all(i > include_index for i, line in enumerate(declarations)
                if b'MAXLWFSIZE' in line or b'VERTEXMAXCOEFS' in line),
            'array specification precedes its dimension include')
    return data[:start], declarations, data[end:], start, end


def compare_dhelas(control, candidate):
    a, da, body_a, start_a, end_a = declaration_parts(control)
    b, db, body_b, start_b, end_b = declaration_parts(candidate)
    require(a == b, 'DHELAS bytes before declaration region differ')
    require(Counter(da) == Counter(db), 'DHELAS raw declaration-line multisets differ')
    require(body_a == body_b, 'DHELAS executable body bytes differ')
    return dict(classification='declaration_line_permutation_only',
                header_sha256=digest(a), executable_sha256=digest(body_a),
                declaration_multiset_sha256=digest(b''.join(sorted(da))),
                declaration_lines=len(da),
                control_region_bytes=[start_a, end_a], candidate_region_bytes=[start_b, end_b])


def routine(data, name):
    pattern = rb'^      SUBROUTINE '+re.escape(name)+rb'\(.*?^      END\n'
    matches = list(re.finditer(pattern, data, re.M | re.S))
    require(len(matches) == 1, 'missing or duplicate routine: '+name.decode())
    return matches[0]


def compare_polynomial(control, candidate):
    names = re.findall(rb'^      SUBROUTINE ([A-Z0-9_]+)DP_TERMINAL_WL_1_1\(', control, re.M)
    require(len(names) == 1, 'expected one retained private 1x1 helper')
    prefix = names[0]
    unchanged = {}
    for suffix in (b'DP_TERMINAL_WL_1_1', b'UPDATE_WL_2_1', b'UPDATE_WL_3_1'):
        name = prefix+suffix
        old, new = routine(control, name).group(), routine(candidate, name).group()
        require(old == new, 'retained/public polynomial routine changed: '+name.decode())
        unchanged[name.decode()] = digest(old)
    added = {}
    residual = candidate
    for rank in (2, 3):
        name = prefix+('DP_TERMINAL_WL_%d_1' % rank).encode()
        require(re.search(rb'^      SUBROUTINE '+re.escape(name)+rb'\(', control, re.M) is None,
                'control already contains the new private helper')
        match = routine(residual, name)
        require(residual[match.end():match.end()+1] == b'\n',
                'new helper must have the canonical trailing blank line')
        added[name.decode()] = digest(match.group())
        residual = residual[:match.start()] + residual[match.end()+1:]
    require(residual == control, 'polynomial differs beyond the two new private helper blocks')
    return dict(classification='two_private_helpers_added_only', prefix=prefix.decode(),
                unchanged_routine_sha256=unchanged, added_routine_sha256=added,
                residual_sha256=digest(residual),
                new_helper_correctness='requires separately bound compiled/proof evidence')


def fixed_blocks(data):
    """Group continuations while retaining every physical byte and comment."""
    blocks = []
    for line in data.splitlines(keepends=True):
        require(line.endswith(b'\n'), 'source must end each physical line with newline')
        require(b'\r' not in line and b'\t' not in line, 'unsupported fixed-form whitespace')
        if len(line) >= 6 and line[:1] not in (b'C', b'c', b'*', b'!') and line[5:6] not in (b' ', b'0'):
            require(blocks and blocks[-1][:6] == b'      ', 'orphan fixed-form continuation')
            require(line[:5] == b'     ' and line[5:6] == b'$', 'unknown continuation spelling')
            blocks[-1] += line
        else:
            blocks.append(line)
    return blocks


def call_text(block):
    lines = block.splitlines()
    require(lines and lines[0][:6] == b'      ', 'unexpected statement label/prefix')
    payload = b''.join(line[6:] for line in lines)
    require(not any(char in payload for char in (b"'", b'"', b';', b'!')),
            'unsupported token in updater call')
    return re.sub(rb'\s+', b'', payload)


def compare_caller(control, candidate):
    old, new = fixed_blocks(control), fixed_blocks(candidate)
    i = j = 0
    changes = []
    while i < len(old) and j < len(new):
        if old[i] == new[j]:
            i += 1
            j += 1
            continue
        require(new[j] == GET_FLAG and j+1 < len(new),
                'caller differs outside an immediate IEEE query/new private-call pair')
        match = re.fullmatch(rb'CALL([A-Z0-9_]+)UPDATE_WL_([23])_1\((.+)\)', call_text(old[i]))
        require(match is not None, 'changed control statement is not a supported public call')
        prefix, rank, arguments = match.groups()
        expected = b'CALL'+prefix+b'DP_TERMINAL_WL_'+rank+b'_1('+arguments+b',WL_ENTRY_INEXACT)'
        require(call_text(new[j+1]) == expected, 'new private call changed original arguments')
        target = re.search(rb',WL\(1,0,1,([0-9]+)\)$', arguments)
        require(target is not None, 'unrecognized output-wavefunction argument')
        changes.append(dict(rank_pair=[int(rank), 1], target=int(target.group(1)),
                            helper=(prefix+b'DP_TERMINAL_WL_'+rank+b'_1').decode(),
                            original_call_sha256=digest(old[i]),
                            replacement_sha256=digest(new[j]+new[j+1])))
        i += 1
        j += 2
    require(i == len(old) and j == len(new), 'caller has unmatched trailing source blocks')
    require(changes, 'changed caller has no recognized private-call additions')
    return dict(classification='immediate_query_and_private_call_substitutions_only',
                changes=changes, rank_pair_counts=dict(Counter(str(x['rank_pair']) for x in changes)))


def audit(control, candidate):
    old, old_excluded, old_links = inventory(control)
    new, new_excluded, new_links = inventory(candidate)
    report = dict(schema=1, control=str(control), candidate=str(candidate),
                  script_sha256=digest(Path(__file__).read_bytes()),
                  selected_suffixes=sorted(SUFFIXES), selected_build_names=sorted(BUILD_NAMES),
                  control_inventory=old, candidate_inventory=new,
                  nontraversed_directory_links=dict(control=old_links, candidate=new_links),
                  excluded_suffix_counts=dict(control=old_excluded, candidate=new_excluded),
                  missing=sorted(old.keys()-new.keys()), added=sorted(new.keys()-old.keys()),
                  changed=[], errors=[],
                  scope='Selected native sources, includes and build configuration bytes only; '
                        'cards, generated metadata, Python drivers and build products are excluded. '
                        'Known resource/library directory symlinks must have identical targets; '
                        'installed library contents require separate dependency provenance.')
    if report['missing'] or report['added']:
        report['errors'].append('selected source/build file inventories differ')
    if old_links != new_links:
        report['errors'].append('resource/library directory symlink targets differ')
    for root in (control, candidate):
        require((root/'Source/DHELAS/coef_specs.inc').read_bytes() == DHELAS_SPEC,
                'DHELAS dimensions are not the qualified four-by-five specification')
    unchanged = 0
    for path in sorted(old.keys() & new.keys()):
        if old[path]['sha256'] == new[path]['sha256']:
            unchanged += 1
            continue
        a, b = (control/path).read_bytes(), (candidate/path).read_bytes()
        item = dict(path=path, control_sha256=digest(a), candidate_sha256=digest(b))
        try:
            if re.fullmatch(r'Source/DHELAS/[^/]+\.f', path):
                item.update(compare_dhelas(a, b))
            elif re.fullmatch(r'SubProcesses/P[^/]+/polynomial\.f', path):
                item.update(compare_polynomial(a, b))
            elif re.fullmatch(r'SubProcesses/P[^/]+/coef_construction_[0-9]+\.f', path):
                item.update(compare_caller(a, b))
            else:
                raise Refusal('unexpected changed selected source/build file')
        except Refusal as error:
            item.update(classification='REFUSED', reason=str(error))
            report['errors'].append(path+': '+str(error))
        report['changed'].append(item)
    added_helpers = {name for item in report['changed'] for name in item.get('added_routine_sha256', {})}
    called_helpers = {change['helper'] for item in report['changed'] for change in item.get('changes', [])}
    if not added_helpers or added_helpers != called_helpers:
        report['errors'].append('new helper definitions and changed caller references must match and be nonempty')
    after_old, _, after_old_links = inventory(control)
    after_new, _, after_new_links = inventory(candidate)
    stable = old == after_old and new == after_new and old_links == after_old_links and new_links == after_new_links
    if not stable:
        report['errors'].append('selected sources or directory links changed during the audit')
    report.update(unchanged_selected_files=unchanged,
                  source_snapshots_stable=stable,
                  control_manifest_sha256=digest(json.dumps(old, sort_keys=True).encode()),
                  candidate_manifest_sha256=digest(json.dumps(new, sort_keys=True).encode()),
                  classification_counts=dict(Counter(item['classification'] for item in report['changed'])),
                  passed=not report['errors'])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    control, candidate = args.control.resolve(), args.candidate.resolve()
    try:
        require(control.is_dir() and candidate.is_dir(), 'both source roots must exist')
        require(control != candidate, 'control and candidate must be distinct roots')
        result = audit(control, candidate)
    except (Refusal, OSError) as error:
        result = dict(passed=False, control=str(control), candidate=str(candidate),
                      errors=[str(error)], script_sha256=digest(Path(__file__).read_bytes()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps({key: result[key] for key in
        ('passed', 'unchanged_selected_files', 'classification_counts', 'errors') if key in result}))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
