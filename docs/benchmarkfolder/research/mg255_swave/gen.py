import os, sys
sys.path.insert(0, '/tmp/claude-1134/-shared-external-madloop-onia/6136fd6c-8e41-47c7-a479-0995fb3044a9/scratchpad/swj/tools')
sys.path.insert(0, '/shared/external/madloop_onia/madgraph7_madloop_onia/tools')
import ml_side, onia_loop_check as olc
name = sys.argv[1]
model, process, masses = ml_side.CASES[name]
out = olc.output_process(model, process, os.path.join(ml_side.SP, 'swj', 'ml', name), regenerate=True)
pdir = olc.subprocess_dir(out)
olc.set_masses(out, masses)
olc.build_driver(pdir)
print('generated', name)
