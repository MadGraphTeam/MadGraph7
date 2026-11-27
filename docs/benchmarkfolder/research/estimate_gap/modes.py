"""u u~ > etac g / t t~ b b~ at PS.input of V0_uux_ccxg (sqrt(s) = 3.1015 GeV): the
CTModeRun modes, rotations, and the sensitivity to 1-ulp changes of the input momenta."""
import json, random, sys
sys.path.insert(0, '/shared/external/madloop_onia/madgraph7_madloop_onia/tools')
import onia_loop_check as olc
OUT = '/tmp/claude-1134/-shared-external-madloop-onia/6136fd6c-8e41-47c7-a479-0995fb3044a9/scratchpad/trace/uux'
PS = '/shared/external/madloop_onia/references/NLO_Onia_Nov2026/2.5.5_Onia1S01/pp2etacj_Grid_FKS_MultiChan/SubProcesses/P0_uux_ccxg/V0_uux_ccxg/PS.input'
rows = [[float(x) for x in l.split()] for l in open(PS) if l.strip()]
pt = [rows[0], rows[1], [a + b for a, b in zip(rows[2], rows[3])], rows[4]]
pdir = olc.subprocess_dir(OUT)
REF = -528.791926209  # 2.5.5 direct: -528.791926209 (QP of the v16 run: -528.791926209526)
res = {}

def run(label, pts, extra):
    olc.set_reduction(OUT, ['9'], extra=extra)
    r = olc.run_points(pdir, pts, 11.0, timeout=3600)
    res[label] = r
    return r

base = {'MLStabThres': '1.0d99', 'ImprovePSPoint': '2', 'NRotations_DP': '0', 'NRotations_QP': '0'}
for mode in ('1', '2', '4', '5', '-1'):
    r = run('mode%s' % mode, [pt], dict(base, CTModeRun=mode))[0]
    print('CTModeRun %2s: fin %.12f c1 %.10f pred %.10f acc %.1e code %s' % (
        mode, r['finite_norm'], r['single_norm'], r['predicted_single_norm'], r['accuracy'], r['ret_code']), flush=True)
for nrot in ('1', '2'):
    r = run('rot%s' % nrot, [pt], dict(base, CTModeRun='-1', NRotations_DP=nrot))[0]
    print('CTModeRun -1, NRotations_DP %s: fin %.12f acc %.1e code %s' % (nrot, r['finite_norm'], r['accuracy'], r['ret_code']), flush=True)
for imp in ('-1', '1'):
    for mode in ('1', '4'):
        r = run('imp%s_mode%s' % (imp, mode), [pt], dict(base, CTModeRun=mode, ImprovePSPoint=imp))[0]
        print('ImprovePSPoint %2s CTModeRun %s: fin %.12f c1 %.10f code %s' % (imp, mode, r['finite_norm'], r['single_norm'], r['ret_code']), flush=True)
# 1-ulp perturbations of every input component (then MadLoop restores conservation and the masses)
random.seed(7)
pert = []
for k in range(12):
    pert.append([[c * (1 + random.choice((-1, 1)) * 2.0 ** -52) if c else c for c in p] for p in pt])
for mode in ('4', '1'):
    r = run('pert_mode%s' % mode, pert, dict(base, CTModeRun=mode))
    f = [x['finite_norm'] for x in r]; c = [x['single_norm'] for x in r]
    print('perturbed inputs, CTModeRun %s: fin %.12f .. %.12f (spread %.1e rel), c1 spread %.1e abs' % (
        mode, min(f), max(f), (max(f) - min(f)) / abs(REF), max(c) - min(c)), flush=True)
olc.set_reduction(OUT, ['9'], extra={'CTModeRun': '-1', 'MLStabThres': '1.0d-3', 'ImprovePSPoint': '2'})
json.dump(dict(point=pt, perturbed=pert, results=res), open(OUT + '_modes.json', 'w'), indent=1)
