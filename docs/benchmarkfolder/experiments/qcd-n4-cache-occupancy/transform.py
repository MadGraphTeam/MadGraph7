#!/usr/bin/env python3
"""Insertion-only private adapter observer; strip markers to recover exact input."""
import re


def transform(text):
    counter = 0

    def insert(body):
        nonlocal counter
        counter += 1
        return f'C OCC_BEGIN {counter}\n{body}C OCC_END {counter}\n'

    source = text
    use = '      USE MTIRUST\n'
    initializer = '      SUBROUTINE ML5_0_INITIALIZE_TIRUST(STATUS)\n'
    if (source.count(initializer) != 1 or source.count(use) != 3
            or source[:source.index(initializer)].count(use) != 2):
        raise ValueError('Unexpected DP/QP diagnostic imports')
    source = re.sub(re.escape(use), lambda match: match[0] +
                    insert('      USE CACHE_OCCUPANCY_OBSERVER, ONLY: OCC_SAMPLE\n'), source, count=2)
    # Process-specific exact native-call text appears once per precision.
    call = ('      CALL TIRUST_TENSOR_EVALUATE(NLOOPLINE,RANK,SHIFTS,INVARIANTS\n'
            '     $ ,MASSES,SCALE,RAW(1:NC,1:4),STATUS,MAX(0,LOOPPOLEMODE)\n'
            '     $ ,TIRUSTCOMPUTEFINITE,CHECK_PASS,CACHE_ENABLED)\n')
    if source.count(call) != 2:
        raise ValueError('Unexpected native call shape')
    for precision in (53, 113):
        marker = insert(f'      CALL OCC_SAMPLE(10,{precision},CTMODE,NLOOPLINE,RANK,\n'
                        '     $ STATUS,MAX(0,LOOPPOLEMODE),TIRUSTCOMPUTEFINITE,\n'
                        '     $ CHECK_PASS,CACHE_ENABLED)\n')
        start = source.index(call, 0 if precision == 53 else source.index('SUBROUTINE ML5_0_MP_'))
        end = start + len(call)
        source = source[:end] + marker + source[end:]
    clear = ('      SUBROUTINE ML5_0_CLEAR_TIRUST_CACHE()\n'
             '      USE MTIRUST, ONLY: TIRUST_CLEAR_CACHE\n')
    if source.count(clear) != 1 or source.count('      CALL TIRUST_CLEAR_CACHE()\n') != 1:
        raise ValueError('Unexpected original clear wrapper')
    source = source.replace(clear, clear + insert('      USE CACHE_OCCUPANCY_OBSERVER, ONLY: OCC_SAMPLE\n'))
    old = '      CALL TIRUST_CLEAR_CACHE()\n'
    def boundary(tag):
        return insert(f'      CALL OCC_SAMPLE({tag},0,0,0,0,-999,-999,\n'
                      '     $ .FALSE.,.FALSE.,.FALSE.)\n')
    source = source.replace(old, boundary(20) + old + boundary(21))
    stripped = re.sub(r'^C OCC_BEGIN ([0-9]+)\n.*?^C OCC_END \1\n', '', source,
                      flags=re.M | re.S)
    if stripped != text:
        raise ValueError('Insertion removal did not recover original bytes')
    if any(len(line) > 132 for line in source.splitlines()):
        raise ValueError('Fixed-form line exceeds original compiler limit')
    return source
