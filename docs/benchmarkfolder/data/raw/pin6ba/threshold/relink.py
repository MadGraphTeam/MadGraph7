"""Relink the g g > Jpsi g output (generated code identical for every TIRust
build) against another TIRust install and run points A, B, C.

usage: relink.py NAME LIBDIR
"""
import os
import shutil
import sys

sys.path.insert(0, '/shared/external/madloop_onia/madgraph7_madloop_onia/tools')
import onia_loop_check as olc  # noqa: E402

SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP)
import run as R  # noqa: E402

name, lib = sys.argv[1], os.path.abspath(sys.argv[2])
src = R.outdir('v16')
out = R.outdir(name)
if not os.path.isdir(out):
    shutil.copytree(src, out, symlinks=True)
    mk = os.path.join(out, 'SubProcesses', 'MadLoop_makefile_definitions')
    text = open(mk).read().replace(R.PW73, lib)
    open(mk, 'w').write(text)
    for link in ('tirust_lib', 'tirust_include'):
        p = os.path.join(out, 'lib', link)
        os.remove(p)
        os.symlink(lib, p)
    pdir = olc.subprocess_dir(out)
    for f in os.listdir(pdir):
        if f.endswith('.o') or f in ('ml_points_driver', 'check'):
            os.remove(os.path.join(pdir, f))
    olc.build_driver(pdir)
pdir = olc.subprocess_dir(out)
olc.set_reduction(out, [9], dict(R.PARAMS, CTModeRun=-1))
for k, x in zip('ABC', olc.run_points(pdir, [R.POINTS[k] for k in 'ABC'], mu=3.0, warmup=0)):
    print('%-6s %s fin %.10e c1 %.10e pred %.10e acc %.2e ret %d' % (
        name, k, x['finite_norm'], x['single_norm'], x['predicted_single_norm'],
        x['accuracy'], x['ret_code']))
