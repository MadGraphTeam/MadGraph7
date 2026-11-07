#!/usr/bin/env python3
"""Convert mint_grids from MG5 3.8.1 (nintegrals=26=6+2*10) to MadGraph7 (38=6+2*16) in place:
the SUM and QSM lines get 6 zeros after each block of n_ave_virt values. Usage: convert.py <files>"""
import sys
for f in sys.argv[1:]:
    out = []
    for line in open(f):
        w = line.split()
        if w and w[0] in ('SUM', 'QSM'):
            if len(w) != 27:
                sys.exit('%s: %s line with %d fields' % (f, w[0], len(w)))
            z = ['0.0000000000000000'] * 6
            line = ' ' + ' '.join([w[0]] + w[1:17] + z + w[17:27] + z) + '\n'
        out.append(line)
    open(f, 'w').writelines(out)
