import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/shared/external/madloop_onia/madgraph7_madloop_onia/tools')
import ml_side, onia_loop_check as olc
name = sys.argv[1]
model, process, masses = ml_side.CASES[name]
out = olc.output_process(model, process, os.path.join(ml_side.SP, 'swj', 'ml', name), regenerate=True)
pdir = olc.subprocess_dir(out)
if masses:
    olc.set_masses(out, masses)
olc.build_driver(pdir)
print('generated', name)
