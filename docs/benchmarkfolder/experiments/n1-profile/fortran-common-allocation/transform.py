#!/usr/bin/env python3
"""Isolated allocation experiments on a generated common TIR interface.

The numerical bodies remain unchanged.  This does not edit exporter templates.
"""
import argparse
from pathlib import Path
import re


VARIANTS = ('tir-alloc', 'switch-guard', 'combined')


def _routine(source, suffix):
    pattern = (r'^      SUBROUTINE (?P<name>[A-Z0-9_]*' + suffix +
               r')\([^\n]*\n(?:     \$[^\n]*\n)*')
    matches = list(re.finditer(pattern, source, re.M | re.I))
    if len(matches) != 1:
        raise ValueError('Expected one generated routine ending ' + suffix)
    match = matches[0]
    end = re.search(r'^      END\s*$', source[match.end():], re.M | re.I)
    if not end:
        raise ValueError('Missing routine END: ' + match['name'])
    stop = match.end() + end.end()
    return match.start(), stop, match['name'], source[match.start():stop]


def _replace_once(source, pattern, replacement):
    result, count = re.subn(pattern, replacement, source, flags=re.M | re.I)
    if count != 1:
        raise ValueError('Expected exactly one match: ' + pattern)
    return result


def _tir_alloc(source):
    start, stop, _, routine = _routine(source, 'TIRLOOP')
    routine = _replace_once(
        routine,
        r'^      (REAL\*8|DOUBLE PRECISION) PCT\(0:3,0:NLOOPLINE-1\),ABSPCT\(0:3\)\s*$',
        r'      \1, ALLOCATABLE :: PCT(:,:)\n      \1 ABSPCT(0:3)')
    routine = _replace_once(
        routine, r'^      (REAL\*8|DOUBLE PRECISION) PDEN\(0:3,NLOOPLINE-1\)\s*$',
        r'      \1, ALLOCATABLE :: PDEN(:,:)')
    routine = _replace_once(
        routine, r'^      (COMPLEX\*16|DOUBLE COMPLEX) M2LCT\(0:NLOOPLINE-1\)\s*$',
        r'      \1, ALLOCATABLE :: M2LCT(:)')
    marker = re.search(r'^      END SELECT\s*$', routine, re.M | re.I)
    if not marker:
        raise ValueError('Missing initial backend rejection SELECT')
    allocate = ("\nC     Legacy tensor workspaces are unused by backend 9.\n"
                "      IF(MLReductionLib(I_LIB).NE.9) THEN\n"
                "        ALLOCATE(PCT(0:3,0:NLOOPLINE-1),\n"
                "     $    PDEN(0:3,NLOOPLINE-1),M2LCT(0:NLOOPLINE-1))\n"
                "      ENDIF\n")
    routine = routine[:marker.end()] + allocate + routine[marker.end():]
    # Explicit cleanup also supports the existing -fno-automatic build mode.
    # Keep cleanup before every RETURN, including an unallocated backend-9 refusal.
    cleanup = '      IF(ALLOCATED(PCT)) DEALLOCATE(PCT,PDEN,M2LCT)\n'
    routine, returns = re.subn(r'^( +RETURN\s*)$',
                              lambda match: cleanup + match[0], routine,
                              flags=re.M | re.I)
    if returns < 1:
        raise ValueError('Expected at least the backend-9 refusal RETURN')
    routine = _replace_once(routine, r'^      END\s*$', cleanup + '      END')
    return source[:start] + routine + source[stop:]


def _switch_guard(source, suffix):
    start, stop, name, routine = _routine(source, suffix)
    # The DP suffix also matches MP; select exact precision before this helper.
    real = re.search(r'^      (REAL\*\d+|DOUBLE PRECISION) PL\(', routine, re.M | re.I)
    complex_ = re.search(r'^      (COMPLEX\*\d+|DOUBLE COMPLEX) M2L\(', routine, re.M | re.I)
    if not real or not complex_:
        raise ValueError('Missing explicit-shape switch arguments')
    first_if = re.search(r'^      IF\s*\(', routine, re.M | re.I)
    first_do = re.search(r'^      DO I=1,NLOOPLINE-1\s*$', routine, re.M | re.I)
    if not first_if or not first_do or first_if.start() >= first_do.start():
        raise ValueError('Unexpected switch guard/body structure')
    guards = routine[first_if.start():first_do.start()]
    if len(re.findall(r'\bRETURN\b', guards, re.I)) != 2:
        raise ValueError('Expected CTMODE and NLOOPLINE early returns')
    worker_name = name + '_WORK'
    worker = routine[:first_if.start()] + routine[first_do.start():]
    worker = worker.replace('SUBROUTINE ' + name + '(', 'SUBROUTINE ' + worker_name + '(', 1)
    public = (
        f'      SUBROUTINE {name}(CTMODE,NLOOPLINE,PL,PDEN,M2L)\n'
        '      IMPLICIT NONE\n'
        '      INTEGER CTMODE,NLOOPLINE\n'
        f'      {real[1]} PL(0:3,NLOOPLINE)\n'
        f'      {real[1]} PDEN(0:3,NLOOPLINE-1)\n'
        f'      {complex_[1]} M2L(NLOOPLINE)\n'
        'C     Enter the original array-owning worker only for a reversal.\n' +
        guards +
        f'      CALL {worker_name}(CTMODE,NLOOPLINE,PL,PDEN,M2L)\n'
        '      END\n\n')
    return source[:start] + public + worker + source[stop:]


def transform(source, variant):
    if variant not in VARIANTS:
        raise ValueError('Unknown variant: ' + variant)
    if variant in ('tir-alloc', 'combined'):
        source = _tir_alloc(source)
    if variant in ('switch-guard', 'combined'):
        # Prefix restrictions distinguish DP from its MP counterpart.
        dp_names = re.findall(r'^      SUBROUTINE ([A-Z0-9_]+_SWITCH_ORDER)\(', source, re.M | re.I)
        dp_names = [name for name in dp_names if not name.upper().endswith('_MP_SWITCH_ORDER')]
        if len(dp_names) != 1:
            raise ValueError('Expected one DP SWITCH_ORDER')
        source = _switch_guard(source, re.escape(dp_names[0]))
        source = _switch_guard(source, 'MP_SWITCH_ORDER')
    return source


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    args = parser.parse_args()
    args.destination.write_text(transform(args.source.read_text(), args.variant))
