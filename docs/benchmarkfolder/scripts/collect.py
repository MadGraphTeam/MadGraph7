#!/usr/bin/env python3
"""Collect the campaign results into data/*.json for benchmark.typ.

    python3 scripts/collect.py --validation VALIDATE.json \\
        --timing DIR_WITH_TIMING_JSONS --reducers REDUCERS.json \\
        --git REVISION

validation: the output of tools/onia_loop_validate.py (all cases);
timing: the *.json reports of tools/onia_loop_timing.py;
reducers: the output of tools/onia_loop_validate.py run with several
          reduction libraries (--libs 9 3 8 6 7 1).
"""

import argparse
import glob
import json
import math
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), 'data')

# Typst math labels of the cases (spectroscopic notation ^(2S+1)L_J^([c]))
LABELS = {
    'aa_etac': r'gamma gamma -> eta_c thin [""^1 S_0^((1))]',
    'ee_jpsi': r'e^+ e^- -> J\/psi thin [""^3 S_1^((1))]',
    'gg_etac1': r'g g -> eta_c thin [""^1 S_0^((1))]',
    'gg_etac8': r'g g -> c overline(c) thin [""^1 S_0^((8))]',
    'uu_psi8': r'u overline(u) -> c overline(c) thin [""^3 S_1^((8))]',
    'uu_ps': r'u overline(u) -> "Ps" thin [""^3 S_1^((1))]',
    'ee_etac_a': r'e^+ e^- -> eta_c gamma',
    'ee_psi_etac': r'e^+ e^- -> J\/psi thin eta_c',
    'gg_psi_g': r'g g -> J\/psi thin g',
    'gg_etac8_g': r'g g -> c overline(c) thin [""^1 S_0^((8))] thin g',
    'gg_psi8_g': r'g g -> c overline(c) thin [""^3 S_1^((8))] thin g',
}


REFSHORT = {
    'aa_etac': 'Barbieri et al.',
    'ee_jpsi': 'Barbieri et al., Celmaster',
    'gg_etac1': 'Petrelli et al. (125)',
    'gg_etac8': 'Petrelli et al. (128)',
    'uu_psi8': 'Petrelli et al. (133) + heavy-quark bubbles',
    'uu_ps': 'Drell–Yan vertex',
    'ee_etac_a': 'Sang, Chen (C.1)',
    'ee_psi_etac': 'Gong, Wang (C.4), μ = 2m_c',
}

STATUS_LINES = [
    '*S-wave, one bound state* (singlet or octet, any $J$) with elementary particles: the '
    'analytic $2 -> 1$ and $e^+ e^-$ results reproduced to $10^(-8)$, the published precision '
    'or MadLoop\'s accuracy estimate (@analytic); $2 -> 2$ poles as predicted, finite parts '
    'confirmed by the $v -> 0$ extrapolation of ordinary integrals (@twotwo).',
    '*TIRust in double precision*, stable points in every frame. The other libraries fail '
    'at threshold, not always detected by MadLoop\'s tests: their results are now reported '
    'unstable, and TIRust is the default chain (@reducers).',
    '*Two bound states* ($e^+ e^- -> J\\/psi thin eta_c$): TIRust refuses lanes, the '
    'points are reported exceptional (open TIRust item).',
    '*No QP rescue yet:* TIRust\'s binary128 path refuses the threshold lanes (open TIRust '
    'item); such points are reported unstable, not wrong.',
    '*P-wave* is refused at generation until TIRust has the first-derivative threshold '
    'integrals.',
    '*MadLoop speed* (@perf): `-O2` builds, tabulated colour contraction and hoisted IEEE '
    'checks, bit-identical, faster on every benchmark process.',
]
OPEN_ITEMS = [
    '*P-wave states:* first-derivative (dual) tensor integrals at threshold in TIRust '
    '(phase 1 non-Coulomb lanes, phase 2 Coulomb lanes), dual loop numerators in MadLoop.',
    '*Two S-wave bound states* in one process: threshold bubble pinches and two Coulomb '
    'relations in one loop, in TIRust.',
    '*Quadruple-precision rescue:* the degenerate lanes in TIRust\'s binary128 path.',
    '*Loop-induced processes* with bound states; the FKS integration on `onia_fks` '
    '(its `BinothLHA` can call this virtual).',
    '*MadLoop speed, not yet applied:* rank-limited clearing of the DP updaters\' output '
    '(it breaks tests that pin full clearing) and the opt-in `-march` builds of the audit.',
]
BASELINE = '5c5b573794'
DECISIONS = [
    '*Base:* `onia_loop` from `main` (frozen performance baseline `5c5b573794`); '
    '`onia_fks` is read for its constants and pole formulas only.',
    '*Scheme:* MadLoop-native: \'t Hooft–Veltman, 4-dimensional external projectors, '
    'Coulomb term dropped at $v = 0$ (part of the LDME), on-shell $m_Q$ and $Z_2$, '
    '$overline("MS")$ $alpha_s$ with $n_l$ light flavours.',
    '*Timing:* on this host, paired alternating runs; accepted when the upper 95% bound of '
    'candidate ÷ baseline is at most 1.05 for every benchmark process, results identical.',
    '*TIRust:* extensions and the pin belong to the TIRust development session; none is '
    'made here (requests and reproducers in `research/`).',
]


def commits(base):
    """[sha, subject] of the commits of the branch since `base`, oldest first."""
    import subprocess
    root = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
    out = subprocess.run(['git', 'log', '--reverse', '--format=%h|%s', '%s..HEAD' % base],
                         cwd=root, capture_output=True, text=True).stdout
    return [line.split('|', 1) for line in out.splitlines() if '|' in line]


PERF_NOTE = ('*Where the time goes* (`research/perf_audit.md`, $g g -> t overline(t) g$ with Ninja, '
             'before these changes): the loop-numerator updaters (`UPDATE_WL` and the DP-terminal '
             'updaters) 45% of the time, the counterterm colour loop 16%, the HELAS routines 8%, '
             'the IEEE mode queries of the private updaters 3%. In the audit\'s full stack (these '
             'five changes and C) the colour loop takes 2% and the IEEE queries 0.2%. Not applied: '
             'C, the rank-limited clearing of the updaters\' output (identical results, but it '
             'breaks tests that pin full clearing), and the opt-in `-march` builds.')


def summarize_validation(cases):
    out = []
    for c in cases:
        entry = dict(name=c['name'], label=LABELS.get(c['name'], '"%s"' % c['name']),
                     process=c['process'], source=c['source'], reference=c['reference'],
                     ref_short=REFSHORT.get(c['name'], ''),
                     energy=c.get('energy'), mu=c.get('mu'))
        # the production reducer (TIRust); the others are in reducers.json
        lib = '9' if '9' in c['libs'] else sorted(c['libs'])[0]
        r = c['libs'][lib]
        if r.get('status') != 'ok':
            entry['status'] = r.get('status')
        else:
            pts = r['points']
            entry.update(lib=lib, npoints=len(pts), status='ok',
                         finite=statistics.median(p['finite_norm'] for p in pts),
                         single=statistics.median(p['single_norm'] for p in pts),
                         double=statistics.median(p['double_norm'] for p in pts),
                         max_deviation=r.get('max_deviation'),
                         max_pole_deviation=r.get('max_pole_deviation'),
                         stable=r.get('stable_points'),
                         time_ms=1e3 * statistics.median(p['time'] for p in pts),
                         accuracy=max(p['accuracy'] for p in pts),
                         return_codes=sorted(set(p['ret_code'] for p in pts)))
        if 'relmom' in c:
            rm = c['relmom']
            entry['relmom'] = dict((lib, dict(intercept=x['intercept'], v0=x['v0'],
                                              relative=x['relative_difference'],
                                              coulomb=x['free_coulomb'],
                                              expected=x['expected_coulomb']))
                                   for lib, x in rm['libs'].items())
            entry['velocities'] = rm['velocities']
            if '9' in rm['libs'] and '7' in rm['libs']:
                # TIRust and COLLIER on the ordinary integrals at v > 0
                entry['relmom_libs_agree'] = max(
                    abs(a - b) / max(abs(a), 1.) for a, b in
                    zip(rm['libs']['9']['finite'], rm['libs']['7']['finite']))
        out.append(entry)
    return out


AGREE = 1e-6
MLSTABTHRES = 1e-3  # MadLoopParams.dat default


def point_deviation(p, ref):
    """Largest deviation of a point from the reference finite part (a number,
    or the TIRust point) and from the predicted poles."""
    fin = ref if not isinstance(ref, dict) else ref['finite_norm']
    return max(abs(p['finite_norm'] - fin) / max(abs(fin), 1.),
               abs(p['single_norm'] - p['predicted_single_norm']),
               abs(p['double_norm'] - p['predicted_double_norm']))


def summarize_reducers(cases):
    """Each library against the reference: the analytic c0 when there is one,
    else TIRust's value at the same point (validated in section 3). A library
    is 'ok' when every point agrees within AGREE; otherwise 'silent' when a
    wrong point is reported stable, 'guarded' when a wrong point passed
    MadLoop's own stability tests (accuracy below MLStabThres) and is reported
    unstable only by the bound-state guard of onia_loop.f (results of
    libraries other than TIRust), and 'flagged' when MadLoop's own tests
    caught every wrong point."""
    out = []
    for c in cases:
        entry = dict(name=c['name'], label=LABELS.get(c['name'], '"%s"' % c['name']),
                     reference=c['reference'], libs={})
        tirust = c['libs'].get('9', {})
        if c['reference']:
            refs, against = None, 'analytic'
        elif tirust.get('status') == 'ok' and tirust['stable_points'] == len(tirust['points']):
            refs, against = tirust['points'], 'TIRust'
        else:
            refs, against = None, None
        entry['against'] = against
        for lib, r in c['libs'].items():
            if r.get('status') != 'ok':
                status = r.get('status', 'failed')
                entry['libs'][lib] = dict(status='timeout' if 'timed out' in status else 'failed')
                continue
            pts = r['points']
            if against is None:
                devs = [None for p in pts]
            elif refs is None:
                devs = [point_deviation(p, c['reference'][0]) for p in pts]
            else:
                devs = [point_deviation(p, q) for p, q in zip(pts, refs)]
            stable = [p['ret_code'] // 100 in (2, 3) for p in pts]
            passed = [0 <= p['accuracy'] < MLSTABTHRES for p in pts]
            wrong = [d is not None and d > AGREE for d in devs]
            state = ('unknown' if against is None else
                     'ok' if not any(wrong) else
                     'silent' if any(w and s for w, s in zip(wrong, stable)) else
                     'guarded' if any(w and q for w, q in zip(wrong, passed)) else 'flagged')
            entry['libs'][lib] = dict(
                status='ok', state=state,
                max_deviation=max(devs) if against else None,
                max_pole_deviation=r.get('max_pole_deviation'),
                stable=sum(stable), npoints=len(pts),
                return_codes=sorted(set(p['ret_code'] for p in pts)),
                time_ms=1e3 * statistics.median(p['time'] for p in pts))
        out.append(entry)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validation')
    parser.add_argument('--timing')
    parser.add_argument('--reducers')
    parser.add_argument('--git', default='')
    parser.add_argument('--date', default='')
    args = parser.parse_args()
    os.makedirs(DATA, exist_ok=True)
    if args.validation:
        with open(os.path.join(DATA, 'validation.json'), 'w') as f:
            json.dump(summarize_validation(json.load(open(args.validation))), f, indent=1)
    if args.reducers:
        with open(os.path.join(DATA, 'reducers.json'), 'w') as f:
            json.dump(summarize_reducers(json.load(open(args.reducers))), f, indent=1)
    if args.timing:
        rows = []
        order = ['uux_z', 'gg_ttx', 'uux_zgg', 'gg_h', 'gg_etac8', 'gg_psig']
        paths = sorted(glob.glob(os.path.join(args.timing, '*.json')),
                       key=lambda q: (order.index(os.path.basename(q)[:-5])
                                      if os.path.basename(q)[:-5] in order else 99, q))
        for path in paths:
            r = json.load(open(path))
            if 'pairs' not in r:
                continue
            rows.append(dict(name=os.path.splitext(os.path.basename(path))[0],
                             onia='(' in r['process'],
                             process=r['process'], model=r['model'], libs=r['libs'],
                             npoints=r['npoints'], pairs=len(r['pairs']),
                             base_ms=r['base_ms'], cand_ms=r['cand_ms'],
                             median_ratio=r['median_ratio'], geomean_ratio=r['geomean_ratio'],
                             upper95=r['upper95'], identical=r['identical'],
                             max_relative_difference=r['max_relative_difference']))
        with open(os.path.join(DATA, 'timing.json'), 'w') as f:
            json.dump(rows, f, indent=1)
    meta = dict(
        revision=args.git, date=args.date, host='AMD EPYC 9754 (shared host)',
        baseline=BASELINE, commits=commits(BASELINE), decisions=DECISIONS,
        libnames={'9': 'TIRust (9)', '3': 'IREGI (3)', '8': 'NinjaRS (8)', '6': 'Ninja (6)',
                  '7': 'COLLIER (7)', '1': 'CutTools (1)'},
        status_lines=STATUS_LINES, open_items=OPEN_ITEMS, perf_note=PERF_NOTE)
    with open(os.path.join(DATA, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=1)


if __name__ == '__main__':
    main()
