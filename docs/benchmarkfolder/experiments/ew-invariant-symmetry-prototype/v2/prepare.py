#!/usr/bin/env python3
"""Derive the additive integer-classified private adapter; no compiler calls."""
from pathlib import Path
import argparse
import difflib
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
ORIGINAL = ROOT / 'docs/benchmarkfolder/experiments/dp-rescue-cause-observer/prepared/ew_n3_epem_ttxa/TIRUST_interface.f.original'
ORIGINAL_SHA = 'e60570ec599263807165f2f289add2076a4c5b71b9e09fe0de83306e9ab2f7ff'
V1 = HERE.parent / 'TIRUST_interface.f'
V1_SHA = 'a30b838fc2839034a2599d2bf3c04df35ec674847afeadaf111127142dfccbb3'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def declarations(precision):
    if precision == 'dp':
        kind, real_kind, width, digits, maxexp, minexp = 'C_INT64_T', 'KI_TIR', 64, 53, 1024, -1021
        mask = "INT(Z'7FF0000000000000',TIRUST_INVARIANT_KIND)"
    else:
        kind, real_kind, width, digits, maxexp, minexp = 'SELECTED_INT_KIND(33)', 'KI_QTIR', 128, 113, 16384, -16381
        mask = "INT(Z'7FFF0000000000000000000000000000',\n     $ TIRUST_INVARIANT_KIND)"
    return (
        'C     Private v2: integer exponent classification does not quiet sNaN.\n'
        f'      INTEGER,PARAMETER :: TIRUST_INVARIANT_KIND={kind}\n'
        '      INTEGER(TIRUST_INVARIANT_KIND),PARAMETER ::\n'
        f'     $ TIRUST_EXPONENT_MASK={mask}\n'
        'C     Restrict this private source to the qualified GNU IEEE formats.\n'
        '      INTEGER,PARAMETER :: TIRUST_FORMAT_ASSERT=1/MERGE(1,0,\n'
        f'     $ STORAGE_SIZE(REAL(0,KIND={real_kind})).EQ.{width}.AND.\n'
        f'     $ STORAGE_SIZE(INT(0,KIND=TIRUST_INVARIANT_KIND)).EQ.{width}.AND.\n'
        f'     $ BIT_SIZE(INT(0,KIND=TIRUST_INVARIANT_KIND)).EQ.{width}.AND.\n'
        f'     $ RADIX(REAL(0,KIND={real_kind})).EQ.2.AND.\n'
        f'     $ DIGITS(REAL(0,KIND={real_kind})).EQ.{digits}.AND.\n'
        f'     $ MAXEXPONENT(REAL(0,KIND={real_kind})).EQ.{maxexp}.AND.\n'
        f'     $ MINEXPONENT(REAL(0,KIND={real_kind})).EQ.{minexp})\n'
        '      INTEGER(TIRUST_INVARIANT_KIND)\n'
        '     $ TIRUST_INVARIANT_BITS(NLOOPLINE*NLOOPLINE)\n')


INSERTION = '''C     Private v2: the ABI consumes the upper triangle (I<J).
C     Retain all original arithmetic and each already-equal pair's bits.
C     Classify every entry through integer bits before any FP comparison.
      TIRUST_INVARIANT_BITS=TRANSFER(INVARIANTS,
     $ TIRUST_INVARIANT_BITS,NLOOPLINE*NLOOPLINE)
      IF(ALL(IAND(TIRUST_INVARIANT_BITS,TIRUST_EXPONENT_MASK)
     $ .NE.TIRUST_EXPONENT_MASK)) THEN
        DO J=2,NLOOPLINE
          DO I=1,J-1
            IF(INVARIANTS(J,I).NE.INVARIANTS(I,J)) THEN
              INVARIANTS(J,I)=INVARIANTS(I,J)
            ENDIF
          ENDDO
        ENDDO
      ENDIF
'''


def derive():
    original, v1 = ORIGINAL.read_bytes(), V1.read_bytes()
    if sha(original) != ORIGINAL_SHA or sha(v1) != V1_SHA:
        raise ValueError('Frozen original or rejected v1 changed')
    text = original.decode('ascii')
    added = []
    for precision, kind, size, scale in (
            ('dp', 'KI_TIR', 8, 'MU_R**2'),
            ('qp', 'KI_QTIR', 16, 'REAL(MU_R,KIND=KI_QTIR)**2')):
        declaration_anchor = ('      INTEGER CTMODE,NLOOPLINE,RANK,I,J,K,NC,STATUS\n'
                              f'      REAL*{size} PL(0:3,NLOOPLINE)\n')
        decl = declarations(precision)
        if text.count(declaration_anchor) != 1:
            raise ValueError('Ambiguous declaration anchor')
        text = text.replace(declaration_anchor,
                            declaration_anchor.replace(f'      REAL*{size} PL', decl + f'      REAL*{size} PL'))
        added.append(decl)
        anchor = ('          INVARIANTS(I,J)=REAL(S_MAT(I,J)+MASSES(I)+MASSES(J)\n'
                  f'     $     ,KIND={kind})\n'
                  f'        ENDDO\n      ENDDO\n      SCALE={scale}\n')
        if text.count(anchor) != 1:
            raise ValueError('Ambiguous reconstruction anchor')
        text = text.replace(anchor, anchor.replace('      SCALE=', INSERTION + '      SCALE='))
    restored = text.replace(INSERTION, '')
    for decl in added:
        if restored.count(decl) != 1:
            raise ValueError('Ambiguous inserted declarations')
        restored = restored.replace(decl, '')
    if text.count(INSERTION) != 2 or restored.encode('ascii') != original:
        raise ValueError('Original adapter bytes were changed')
    if any(len(line) > 132 for line in text.splitlines()):
        raise ValueError('Fixed-form width exceeds original allowance')
    candidate = text.encode('ascii')
    patch = ''.join(difflib.unified_diff(original.decode().splitlines(True), text.splitlines(True),
                    fromfile='frozen/TIRUST_interface.f', tofile='private-v2/TIRUST_interface.f')).encode('ascii')
    delta = ''.join(difflib.unified_diff(v1.decode().splitlines(True), text.splitlines(True),
                    fromfile='rejected-v1/TIRUST_interface.f', tofile='private-v2/TIRUST_interface.f')).encode('ascii')
    return candidate, patch, delta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise ValueError('Run from MadGraph repository')
    candidate, patch, delta = derive()
    records = {'TIRUST_interface.f': candidate, 'derivation.patch': patch, 'v1-to-v2.patch': delta}
    receipt = {
        'schema': 'ew-invariant-integer-classification-source-v2',
        'original_path': str(ORIGINAL.relative_to(ROOT)), 'original_sha256': ORIGINAL_SHA,
        'rejected_v1_path': str(V1.relative_to(ROOT)), 'rejected_v1_sha256': V1_SHA,
        'candidate_sha256': sha(candidate), 'derivation_patch_sha256': sha(patch),
        'v1_to_v2_patch_sha256': sha(delta),
        'original_body_recovered_by_removing_exact_insertions': True,
        'finite_classification': 'TRANSFER to same-width integer, then integer exponent-mask test',
        'dp_integer_kind': 'C_INT64_T', 'qp_integer_kind': 'SELECTED_INT_KIND(33)',
        'dp_format': 'IEEE binary64', 'qp_format': 'IEEE binary128, 113 significand bits',
        'whole_matrix_finite_before_floating_comparison': True,
        'single_integer_per_real_no_multiword_endian_selection': True,
        'compiled': False, 'native_executed': False, 'numerical_parity_qualified': False,
        'performance_eligible': False,
    }
    records['preparation.json'] = (json.dumps(receipt, indent=2, sort_keys=True) + '\n').encode()
    for name, data in records.items():
        path = HERE / name
        if args.prepare:
            with path.open('xb') as output:
                output.write(data)
        elif path.read_bytes() != data:
            raise ValueError('Derived source changed: ' + name)
    print(json.dumps(receipt, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
