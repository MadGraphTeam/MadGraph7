#! /usr/bin/env python3
"""Compare two Sudakov providers: one-line no-emission probabilities.

Compiles both Fortran providers (with the CERNLIB interpolation) and plots
the value returned by pysudakov, one emitter and one directed colour line,
as a function of the scale for a few dipole masses, per radiator and type.

Usage: compare_tables.py <old.f> <new.f> <output.pdf> [label_old label_new]
"""
import ctypes
import math
import os
import subprocess
import sys
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE = os.path.dirname(os.path.abspath(__file__))
CERNLIB = os.path.join(HERE, '..', '..', 'Template', 'NLO', 'Source', 'CERNLIB')
MASSES = (.33, .33, .5, 1.5, 4.8, 0, 0, 0, 0, 0, .000510998928, 0, .1056583715, 0,
          1.77682, 0, 0, 0, 0, 0, 0)
TYPES = {1: 'II', 2: 'FF', 3: 'IF', 4: 'FI'}
PARTONS = {1: 'd', 2: 'u', 3: 's', 4: 'c', 5: 'b', 6: 't', 21: 'g'}


def load(source, work, name):
    library = os.path.join(work, name + '.so')
    subprocess.check_call(['gfortran', '-O2', '-std=legacy', '-shared', '-fPIC', source] +
                          [os.path.join(CERNLIB, f) for f in ('dfint.f', 'kerset.f', 'abend.f')] +
                          ['-o', library])
    lib = ctypes.CDLL(library)
    f = lib.pysudakov_
    f.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                  ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
                  ctypes.POINTER(ctypes.c_double)]
    f.restype = ctypes.c_double
    masses = (ctypes.c_double * 21)(*MASSES)
    def value(st, xm, pid, itype):
        a = [ctypes.c_double(st), ctypes.c_double(xm), ctypes.c_int(pid), ctypes.c_int(itype)]
        return f(*(ctypes.byref(x) for x in a), masses)
    return value


def main():
    old_src, new_src, output = sys.argv[1:4]
    labels = sys.argv[4:6] if len(sys.argv) > 5 else ('old', 'new')
    work = tempfile.mkdtemp(prefix='sudcompare_')
    old, new = load(old_src, work, 'old'), load(new_src, work, 'new')
    scales = [10 ** (i / 40.) for i in range(0, 161)]       # 1 GeV - 10 TeV
    masses = (20., 91.188, 500., 2000.)
    colours = ('tab:blue', 'tab:orange', 'tab:green', 'tab:red')
    with PdfPages(output) as pdf:
        for itype in (1, 2, 4):
            for pid in (2, 5, 6, 21):
                fig, (ax, axr) = plt.subplots(2, 1, figsize=(7, 6.2), sharex=True,
                                              gridspec_kw={'height_ratios': [3, 1.3], 'hspace': .05})
                for xm, colour in zip(masses, colours):
                    xs = [s for s in scales if s <= 7000.]
                    vo = [old(s, xm, pid, itype) for s in xs]
                    vn = [new(s, xm, pid, itype) for s in xs]
                    ax.plot(xs, vo, ls='--', color=colour, lw=1.2)
                    ax.plot(xs, vn, ls='-', color=colour, lw=1.4, label='m = %g GeV' % xm)
                    ratio = [n / o if o > 0 else float('nan') for n, o in zip(vn, vo)]
                    axr.plot(xs, ratio, color=colour, lw=1.2)
                ax.set_xscale('log')
                ax.set_ylim(0, 1.02)
                ax.set_title('%s radiator %s (pysudakov: one emitter, one colour line)'
                             % (TYPES[itype], PARTONS[pid]), fontsize=10)
                ax.set_ylabel('no-emission probability down to scale')
                ax.plot([], [], 'k--', label=labels[0])
                ax.plot([], [], 'k-', label=labels[1])
                ax.legend(fontsize=8, loc='lower right')
                ax.grid(alpha=.3)
                axr.set_ylim(0.5, 1.5)
                axr.axhline(1, color='k', lw=.8)
                axr.set_ylabel('%s / %s' % (labels[1], labels[0]), fontsize=8)
                axr.set_xlabel('scale [GeV]')
                axr.grid(alpha=.3)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
    print('wrote', output)


if __name__ == '__main__':
    main()
