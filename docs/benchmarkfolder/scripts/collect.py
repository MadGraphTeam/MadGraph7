#!/usr/bin/env python3
"""Collect the campaign results into data/*.json for benchmark.typ.

    python3 scripts/collect.py --validation VALIDATE.json \\
        --timing DIR_WITH_TIMING_JSONS --reducers REDUCERS.json \\
        --git REVISION

validation: the output of tools/onia_loop_validate.py (all cases);
validation-extra: runs of tools/onia_loop_validate.py with an unpinned TIRust
          build (--tirust, --tirust-label), listed after the pinned row of the
          same case;
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
    '*Two bound states* ($e^+ e^- -> J\\/psi thin eta_c$): the pinned TIRust refuses lanes '
    'and the points are reported exceptional; TIRust 370b632 (PR \\#56, not yet pinned) serves '
    'them, stable, Gong–Wang reproduced to $3 dot 10^(-7)$ (@analytic).',
    '*No QP rescue yet:* TIRust\'s binary128 path refuses the threshold lanes (open TIRust '
    'item); such points are reported unstable, not wrong.',
    '*P-wave, one bound state* (@pwave): with TIRust 48958d1 (not yet pinned) the $2 -> 1$ '
    'literature values reproduced to $10^(-12)$, the poles as predicted with the $P -> S$ term '
    'of the soft current, also for $u overline(u) -> chi_(c 0) g$ and $g u -> chi_(c 0) u$.',
    '*S-wave onium + jet against MG5 2.5.5* (@mg255): finite parts within $3 dot 10^(-6)$ '
    '($J\\/psi$: $10^(-8)$) at the points MadLoop reports stable.',
    '*Stability profiles* (@stability, after appendix C of arXiv:1604.01363), charmonium and '
    'bottomonium: S waves steep, apart from points the pinned TIRust refuses (3% at 1 TeV, '
    'served by 08edee3); the P-wave singlets lose digits in double precision at 1 TeV, not at 20 GeV, '
    'through TIRust\'s double-precision pentagons with a doubled line; TIRust 5025e12 '
    '(PR \\#56: relation tiers in double-double) fixes this: no P-wave point above $10^(-3)$, no rescue at 1 TeV, '
    '18–49% more time at 20 GeV (@stability); '
    'the P-wave octets $e^+ e^- -> Q overline(Q) thin [n^((8))] thin g$ stay below $10^(-4)$.',
    '*MadLoop speed* (@perf): `-O2` builds, tabulated colour contraction and hoisted IEEE '
    'checks, bit-identical, faster on every benchmark process.',
]
OPEN_ITEMS = [
    '*P-wave in TIRust:* a pinned revision with the raised powers at threshold (PRs \\#47, '
    '\\#55 and \\#56; until then a P-wave point meets a refused integral and is reported unstable). '
    '370b632 (in \\#56) also serves $g g -> chi_(c 0) g$ (@pwave). For double precision '
    'at high energy, the relation tiers of \\#56 (5025e12) with their bridge patch, and the pole '
    'kernels of the bridge on the `oneloop` of that revision '
    '(@stability).',
    '*Two S-wave bound states* in one process: a pinned TIRust with PR \\#56 '
    '(threshold bubble pinches, two Coulomb relations in one loop; served by 370b632).',
    '*Quadruple-precision rescue:* the degenerate lanes in TIRust\'s binary128 path.',
    '*Soft jets:* TIRust refuses the Coulomb triangle and boxes and pentagons with a jet of '
    '$p_T$ below about 0.05 GeV (charmonium) to 0.9 GeV ($g g -> eta_b\\/Upsilon thin g$); such '
    'points are reported unstable. TIRust 08edee3 serves them, and MadLoop\'s double precision '
    'is then silently wrong at the softest: the R2 counterterms of heavy quarks ($M^2$-enhanced; '
    '22% at a 1.5 MeV gluon with the $10^4$ GeV $b$, $t$ of the comparison, $1.7 dot 10^(-4)$ '
    'with physical masses), which MadLoop\'s loop-direction test does not vary. Fixed: bound-state '
    'outputs now estimate the rounding of the counterterm interference and rescue such points in '
    'quadruple precision (every point right).',
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
        if c.get('tirust'):
            entry['tirust'] = c['tirust']
            entry['ref_short'] += '; TIRust %s' % c['tirust']
        out.append(entry)
    return out


def merge_extra(rows, extra):
    """Insert the rows of unpinned TIRust builds after the pinned row of the
    same case."""
    out = list(rows)
    for e in extra:
        at = max([i for i, r in enumerate(out) if r['name'] == e['name']], default=len(out) - 1)
        out.insert(at + 1, e)
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


#------------------------------------------------------------------------------
# S-wave onium + jet against the MG5 2.5.5 onia outputs (research/mg255_swave)
#------------------------------------------------------------------------------
MG255 = [
    # case, label, m_Q, n_lq
    ('gg_etacg', r'g g -> eta_c thin g', 1.55, 3),
    ('gu_etacu', r'g u -> eta_c thin u', 1.55, 3),
    ('ug_etacu', r'u g -> eta_c thin u', 1.55, 3),
    ('uux_etacg', r'u overline(u) -> eta_c thin g', 1.55, 3),
    ('gg_jpsig', r'g g -> J\/psi thin g', 1.55, 3),
    ('gg_etabg', r'g g -> eta_b thin g', 4.75, 4),
    ('gu_etabu', r'g u -> eta_b thin u', 4.75, 4),
    ('ug_etabu', r'u g -> eta_b thin u', 4.75, 4),
    ('uux_etabg', r'u overline(u) -> eta_b thin g', 4.75, 4),
    ('gg_upsg', r'g g -> Upsilon thin g', 4.75, 4),
]


def summarize_mg255(folder):
    """Per case: the points MadLoop reports stable (return codes 1xx-3xx)
    against the 2.5.5 V0 (its direct evaluation, calcoption=3, when there is
    one; else its grid), the unstable points, the grid against the direct
    evaluation, and the lowest- and highest-energy stable points."""
    out = []
    for name, label, mq, nl in MG255:
        rows = json.load(open(os.path.join(folder, 'cmp_%s.json' % name)))
        opt = '3' if '3' in rows[0]['ref'] else sorted(rows[0]['ref'])[-1]
        stable = [r for r in rows if int(r['ml']['code']) // 100 in (1, 2, 3)]
        unstable = [r for r in rows if r not in stable]

        def rel(r, key='fin', o=opt):
            return abs(r['ml'][key] - r['ref'][o][key]) / abs(r['ref'][o][key])

        both = [r for r in rows if '1' in r['ref'] and '3' in r['ref']]
        stable.sort(key=lambda r: r['sqrts'])
        samples = []
        for r in (stable[0], stable[-1]):
            samples.append(dict(sqrts=r['sqrts'], ml_fin=r['ml']['fin'],
                                ref_fin=r['ref'][opt]['fin'], rel=rel(r),
                                ml_c1=r['ml']['c1'], ref_c1=r['ref'][opt]['c1'],
                                code=int(r['ml']['code'])))
        out.append(dict(
            name=name, label=label, mq=mq, nl=nl,
            reference='direct' if opt == '3' else 'grid',
            npoints=len(rows), stable=len(stable),
            sqrts_min=stable[0]['sqrts'], sqrts_max=stable[-1]['sqrts'],
            max_fin=max(rel(r) for r in stable),
            max_c1=max(abs(r['ml']['c1'] - r['ref'][opt]['c1']) for r in stable),
            max_c2=max(abs(r['ml']['c2'] - r['ref'][opt]['c2']) for r in stable),
            born_ratio=[min(r['ml']['born'] / r['ref'][opt]['born'] for r in stable),
                        max(r['ml']['born'] / r['ref'][opt]['born'] for r in stable)],
            unstable=len(unstable),
            unstable_sqrts_max=max(r['sqrts'] for r in unstable) if unstable else None,
            grid_vs_direct=max(abs(r['ref']['1']['fin'] - r['ref']['3']['fin'])
                               / abs(r['ref']['3']['fin']) for r in both) if both else None,
            samples=samples))
    return out


def summarize_mg255_pol(folder):
    """g g > Jpsi g per polarisation (pol_gg_jpsig.json, pol_jpsi.py): for
    T (J_z = +-1), L (J_z = 0) and their sum, the Born ratio, MadLoop with
    the pinned TIRust (CTModeRun -1) at the points it reports stable, and
    with TIRust 08edee3 in forced quadruple precision (CTModeRun 4) and in
    double precision (CTModeRun -1) at every point, against the direct
    evaluation of Jpsi_double_cs_NEW."""
    path = os.path.join(folder, 'pol_gg_jpsig.json')
    if not os.path.isfile(path):
        return None
    rows = json.load(open(path))
    out = []
    for pol in ('T', 'L', 'all'):
        def dev(key, pts, what='fin'):
            return max((abs(r['ml'][key][what] - r['ref'][pol][what])
                        / (abs(r['ref'][pol][what]) if what == 'fin' else 1.)) for r in pts)
        stable = [r for r in rows if r['ml']['%s_pin_-1' % pol]['code'] // 100 in (1, 2, 3)]
        out.append(dict(
            pol=pol, npoints=len(rows), stable=len(stable),
            born_ratio=[min(r['ml']['%s_pin_-1' % pol]['born'] / r['ref'][pol]['born'] for r in rows),
                        max(r['ml']['%s_pin_-1' % pol]['born'] / r['ref'][pol]['born'] for r in rows)],
            pin_fin=dev('%s_pin_-1' % pol, stable), pin_c1=dev('%s_pin_-1' % pol, stable, 'c1'),
            qp_fin=dev('%s_4_08edee3' % pol, rows), qp_c1=dev('%s_4_08edee3' % pol, rows, 'c1'),
            dp_fin=dev('%s_-1_08edee3' % pol, rows),
            dp_codes=sorted(set(r['ml']['%s_-1_08edee3' % pol]['code'] for r in rows)),
            ct_fin=dev('%s_-1_08edee3_ctacc' % pol, rows),
            ct_codes=sorted(set(r['ml']['%s_-1_08edee3_ctacc' % pol]['code'] for r in rows))))
    return out


#------------------------------------------------------------------------------
# P-wave bound states (research/pwave_fd: pw_check.py, ps_term.py)
#------------------------------------------------------------------------------
_CF, _CA = 4. / 3., 3.
_PI2, _LOG2 = math.pi ** 2, math.log(2.)


def _gg(b, c1):
    """c0, c1, c2 of g g -> onium at mu = M from Petrelli's B (B.5 of
    research/literature.md)."""
    return (2 * b - _CA * _PI2 / 3, c1, -2 * _CA)


PWAVE_21 = [
    # name, label, reference (c0, c1, c2), source, target of the finite part
    ('aa_chic0', r'gamma gamma -> chi_(c 0) thin [""^3 P_0^((1))]',
     (2 * _CF * (_PI2 / 4 - 7. / 3), 0., 0.), '1511.06288 (7): $2 C_F (pi^2\\/4 - 7\\/3)$', 1e-8),
    ('aa_chic2', r'gamma gamma -> chi_(c 2) thin [""^3 P_2^((1))]',
     (-8 * _CF, 0., 0.), '1511.06288 (8b): $-8 C_F$', 1e-8),
    ('gg_chic0', r'g g -> chi_(c 0) thin [""^3 P_0^((1))]',
     _gg(_CF * (-7. / 3 + _PI2 / 4) + _CA * (1. / 3 + 5 * _PI2 / 12), -9.),
     'Petrelli et al. (126)', 1e-8),
    ('gg_chic2', r'g g -> chi_(c 2) thin [""^3 P_2^((1))]',
     _gg(-4 * _CF + _CA * (1. / 3 + 5. / 3 * _LOG2 + _PI2 / 6), -9.),
     'Petrelli et al. (127)', 1e-8),
    ('gg_3p08', r'g g -> c overline(c) thin [""^3 P_0^((8))]',
     _gg(_CF * (-7. / 3 + _PI2 / 4) + _CA * (2. / 3 + 7 * _PI2 / 24), -12.),
     'recomputed (`onia_fks`); Petrelli (130) differs', 1e-8),
    ('gg_3p28', r'g g -> c overline(c) thin [""^3 P_2^((8))]',
     _gg(-4 * _CF + _CA * (11. / 12 + 4. / 3 * _LOG2 + 5 * _PI2 / 12), -12.),
     'recomputed (`onia_fks`); Petrelli (131) differs', 1e-8),
    ('ag_3p08', r'gamma g -> c overline(c) thin [""^3 P_0^((8))]',
     (3.3575140, -7.5, -3.), '2501.04188 $H^((1))$ (8 digits)', 1e-7),
    ('ag_3p28', r'gamma g -> c overline(c) thin [""^3 P_2^((8))]',
     (10.7171815, -7.5, -3.), '2501.04188 $H^((1))$ (8 digits)', 1e-7),
    ('gg_1p18', r'g g -> c overline(c) thin [""^1 P_1^((8))]',
     _gg(_CF * (_PI2 / 2 - 5 - 4 * _LOG2) + _CA * (1 + _LOG2 + _PI2 / 6), -18.),
     '`onia_fks` (CDR, Larin $gamma_5$)', 1e-8),
]
PWAVE_22 = [
    ('ee_3p08g', r'e^+ e^- -> c overline(c) thin [""^3 P_0^((8))] thin g'),
    ('ee_3p18g', r'e^+ e^- -> c overline(c) thin [""^3 P_1^((8))] thin g'),
    ('ee_3p28g', r'e^+ e^- -> c overline(c) thin [""^3 P_2^((8))] thin g'),
    ('ee_1p18g', r'e^+ e^- -> c overline(c) thin [""^1 P_1^((8))] thin g'),
    ('uu_chic0g', r'u overline(u) -> chi_(c 0) thin g'),
    ('gu_chic0u', r'g u -> chi_(c 0) thin u'),
    # TIRust 370b632 (PR #56): the doubled-gluon pentagons that 48958d1 refuses
    ('gg_chic0g', r'g g -> chi_(c 0) thin g'),
]


def summarize_pwave(folder):
    """The threshold values of the 2 -> 1 processes against the literature,
    and for every process the P -> S term of the pole prediction and the
    poles against the prediction (folder: pw_check_<case>.json and
    ps_term.json)."""
    terms = json.load(open(os.path.join(folder, 'ps_term.json')))

    def load(name):
        return json.load(open(os.path.join(folder, 'pw_check_%s.json' % name)))[name]

    def common(name, pts):
        t = terms[name]['ps_term']
        return dict(
            npoints=len(pts), stable=sum(1 for p in pts if p['ret_code'] // 100 in (1, 2, 3)),
            return_codes=sorted(set(p['ret_code'] for p in pts)),
            finite=statistics.median(p['finite_norm'] for p in pts),
            single=statistics.median(p['single_norm'] for p in pts),
            double=statistics.median(p['double_norm'] for p in pts),
            single_range=[min(p['single_norm'] for p in pts), max(p['single_norm'] for p in pts)],
            max_pole_deviation=max(max(abs(p['single_norm'] - p['predicted_single_norm']),
                                       abs(p['double_norm'] - p['predicted_double_norm']))
                                   for p in pts),
            ps_term=[min(t), max(t)], energy=terms[name]['energy'],
            accuracy=max(p['accuracy'] for p in pts))

    two_one = []
    for name, label, ref, source, target in PWAVE_21:
        pts = load(name)
        entry = dict(name=name, label=label, reference=list(ref), source=source,
                     target=target, **common(name, pts))
        entry['max_deviation'] = max(
            max(abs(p['finite_norm'] - ref[0]) / max(abs(ref[0]), 1.),
                abs(p['single_norm'] - ref[1]), abs(p['double_norm'] - ref[2])) for p in pts)
        two_one.append(entry)
    two_two = [dict(name=name, label=label, **common(name, load(name)))
               for name, label in PWAVE_22]
    return dict(two_one=two_one, two_two=two_two)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validation')
    parser.add_argument('--validation-extra', nargs='*', default=[])
    parser.add_argument('--timing')
    parser.add_argument('--reducers')
    parser.add_argument('--mg255', help='research/mg255_swave/data')
    parser.add_argument('--pwave', help='data/raw/pwave')
    parser.add_argument('--git', default='')
    parser.add_argument('--date', default='')
    args = parser.parse_args()
    os.makedirs(DATA, exist_ok=True)
    if args.validation:
        with open(os.path.join(DATA, 'validation.json'), 'w') as f:
            extra = [e for path in args.validation_extra
                     for e in summarize_validation(json.load(open(path)))]
            json.dump(merge_extra(summarize_validation(json.load(open(args.validation))), extra),
                      f, indent=1)
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
    if args.mg255:
        with open(os.path.join(DATA, 'mg255.json'), 'w') as f:
            json.dump(summarize_mg255(args.mg255), f, indent=1)
        pol = summarize_mg255_pol(args.mg255)
        if pol is not None:
            with open(os.path.join(DATA, 'mg255_pol.json'), 'w') as f:
                json.dump(pol, f, indent=1)
    if args.pwave:
        with open(os.path.join(DATA, 'pwave.json'), 'w') as f:
            json.dump(summarize_pwave(args.pwave), f, indent=1)
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
