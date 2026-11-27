#!/usr/bin/env python3
# Usage: tools/make_mp_dual_aloha_functions.py aloha/template_files/aloha_functions_dual.f aloha/template_files/aloha_functions_mp_dual.f
"""Translate the dual double-precision external wavefunctions of
aloha_functions_dual.f (ixxxxx, oxxxxx, vxxxxx, sxxxxx) to quadruple
precision (mp_*, type(MP_Dual), mp_dual_variables)."""
import re, sys
src = open(sys.argv[1]).read().splitlines(True)
names = ['ixxxxx', 'oxxxxx', 'vxxxxx', 'sxxxxx']
out = []
for name in names:
    start = [i for i, l in enumerate(src) if re.match(r'(?i)\s+subroutine\s+%s\s*\(' % name, l)]
    end = [i for i, l in enumerate(src) if re.match(r'(?i)\s+end\s+subroutine\s+%s\s*$' % name, l)]
    assert len(start) == 1 and len(end) == 1, name
    out.extend(src[start[0]:end[0] + 1])
    out.append('\n')
text = ''.join(out)
lines = []
for line in text.splitlines(True):
    if line[:1] in 'cC*!':
        lines.append(line)
        continue
    l = line
    for name in names:
        l = re.sub(r'(?i)\b(subroutine\s+)%s\b' % name, r'\1mp_%s' % name, l)
    l = re.sub(r'(?i)type\(aloha\)', 'type(mp_aloha)', l)
    l = re.sub(r'(?i)type\(dual\)', 'type(MP_Dual)', l)
    l = re.sub(r'(?i)double\s+precision', 'real*16', l)
    l = re.sub(r'(?i)double\s+complex', 'complex*32', l)
    l = re.sub(r'(?i)\bdsqrt\(', 'sqrt(', l)
    l = re.sub(r'(?i)\bdabs\(', 'abs(', l)
    l = re.sub(r'(?i)\bdcmplx\(', 'qcmplx(', l)
    l = re.sub(r'(?i)\bdble\(', 'qreal(', l)
    l = re.sub(r'(?<![\w.])(\d+\.\d*|\.\d+|\d+)[dD]([+-]?\d+)\b', r'\1E\2_16', l)
    l = re.sub(r'(?i)(parameter\(\s*ci\s*=\s*)qcmplx\(([^()]*)\)', r'\1cmplx(\2,kind=16)', l)
    lines.append(l)
header = '''c ======================================================================
c  Quadruple-precision dual external wavefunctions (MadLoop with one P-wave
c  bound state), generated from those of aloha_functions_dual.f by
c  tools/make_mp_dual_aloha_functions.py: mp_ names, type(MP_Dual) and
c  quadruple-precision types, constants and constructors.
c ======================================================================

'''
open(sys.argv[2], 'w').write(header + ''.join(lines))
print('written', sys.argv[2])
