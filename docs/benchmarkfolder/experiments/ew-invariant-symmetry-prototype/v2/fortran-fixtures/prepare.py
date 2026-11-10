#!/usr/bin/env python3
"""Pure source derivation. Default verifies; --write creates wrappers once."""
from pathlib import Path
import argparse
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
PARENT = HERE.parent
ORIGINAL = ROOT / 'docs/benchmarkfolder/experiments/dp-rescue-cause-observer/prepared/ew_n3_epem_ttxa/TIRUST_interface.f.original'
ORIGINAL_SHA = 'e60570ec599263807165f2f289add2076a4c5b71b9e09fe0de83306e9ab2f7ff'
CANDIDATE_SHA = '177a7973ea9182978097d5256fab0f0c93c1d4c799919203f7855631bee294ad'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def extract(source, start, end):
    assert source.count(start) == 1
    position = source.index(start)
    finish = source.index(end, position)
    return source[position:finish]


def derive():
    original = ORIGINAL.read_bytes()
    candidate = (PARENT / 'TIRUST_interface.f').read_bytes()
    assert sha(original) == ORIGINAL_SHA and sha(candidate) == CANDIDATE_SHA
    original, candidate = original.decode('ascii'), candidate.decode('ascii')
    wrappers, canonical = [], []
    for tag, kind, prefix, scale in [
            ('dp', 8, '', 'MU_R**2'),
            ('qp', 16, 'MP_', 'REAL(MU_R,KIND=KI_QTIR)**2')]:
        start = '      CALL ML5_0_' + prefix + 'BUILD_KINEMATIC_MATRIX('
        end = '      SCALE=' + scale + '\n'
        old, new = extract(original, start, end), extract(candidate, start, end)
        assert new.startswith(old)
        insertion = new[len(old):]
        assert insertion.count('INVARIANTS(J,I)=INVARIANTS(I,J)') == 1
        canonical.append(insertion)
        declaration_start = 'C     Private v2: integer exponent classification does not quiet sNaN.\n'
        function_start = candidate.index('      SUBROUTINE ML5_0_' + prefix + 'TIRUST_TENSOR_DIAGNOSTIC(')
        function_end = candidate.index('\n      END\n', function_start)
        function = candidate[function_start:function_end]
        declaration = extract(function, declaration_start, f'      REAL*{kind} PL(')
        iso_use = '      USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_INT64_T\n'
        assert iso_use in function
        for variant, body in [('original', old), ('candidate', new)]:
            wrappers.append(
                f'      SUBROUTINE AUDIT_{tag.upper()}_{variant.upper()}(NLOOPLINE,P_S_MAT,\n'
                '     $ MASSES,INVARIANTS)\n'
                '      USE, INTRINSIC :: IEEE_ARITHMETIC, ONLY: IEEE_IS_FINITE\n'
                + iso_use + '      IMPLICIT NONE\n'
                f'      INTEGER,PARAMETER :: KI_TIR=8,KI_QTIR=16\n'
                '      INTEGER NLOOPLINE,I,J\n'
                + (declaration if variant == 'candidate' else '')
                + f'      REAL*{kind} P_S_MAT(NLOOPLINE,0:3)\n'
                f'      REAL*{kind} INVARIANTS(NLOOPLINE,NLOOPLINE)\n'
                f'      COMPLEX*{kind*2} MASSES(NLOOPLINE),S_MAT(NLOOPLINE,NLOOPLINE)\n'
                + body + '      END\n\n')
        wrappers.append(
            f'      SUBROUTINE AUDIT_{tag.upper()}_CANONICAL(NLOOPLINE,INVARIANTS)\n'
            '      USE, INTRINSIC :: IEEE_ARITHMETIC, ONLY: IEEE_IS_FINITE\n'
            + iso_use + '      IMPLICIT NONE\n'
            '      INTEGER,PARAMETER :: KI_TIR=8,KI_QTIR=16\n'
            '      INTEGER NLOOPLINE,I,J\n'
            + declaration
            + f'      REAL*{kind} INVARIANTS(NLOOPLINE,NLOOPLINE)\n'
            + insertion + '      END\n\n')
    assert canonical[0] == canonical[1]
    wrappers.append("      SUBROUTINE AUDIT_SET_OSTHRES(VALUE)\n"
                    "      IMPLICIT NONE\n      REAL*8 VALUE\n"
                    "      INCLUDE 'MadLoopParams.inc'\n"
                    "      OSTHRES=VALUE\n      END\n")
    text = ''.join(wrappers)
    assert all(len(line) <= 132 for line in text.splitlines())
    outputs = {'wrappers.f': text.encode('ascii')}
    template = (HERE / 'driver.f90.in').read_text()
    for tag, kind, words in [('dp', 8, 1), ('qp', 16, 2)]:
        driver = template.replace('@PREFIX@', tag).replace('@KIND@', str(kind)).replace('@WORDS@', str(words))
        assert '@' not in driver
        outputs['driver_' + tag + '.f90'] = driver.encode('ascii')
    # Each negative control changes only the extracted private block.
    block = canonical[0]
    no_finite = block.replace('      IF(ALL(IAND(TIRUST_INVARIANT_BITS,TIRUST_EXPONENT_MASK)\n     $ .NE.TIRUST_EXPONENT_MASK)) THEN\n', '')
    assert no_finite.endswith('      ENDIF\n')
    no_finite = no_finite[:-len('      ENDIF\n')]
    no_equal = block.replace('            IF(INVARIANTS(J,I).NE.INVARIANTS(I,J)) THEN\n', '')
    no_equal = no_equal.replace('            ENDIF\n', '')
    outputs['mutation-no-finite.f'] = text.replace(block, no_finite).encode('ascii')
    outputs['mutation-no-equality.f'] = text.replace(block, no_equal).encode('ascii')
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    outputs = derive()
    for name, data in outputs.items():
        path = HERE / name
        if args.write:
            with path.open('xb') as stream:
                stream.write(data)
        else:
            assert path.read_bytes() == data, name
    print(json.dumps({name: {'sha256': sha(data), 'size': len(data)}
                      for name, data in outputs.items()}, sort_keys=True))


if __name__ == '__main__':
    main()
