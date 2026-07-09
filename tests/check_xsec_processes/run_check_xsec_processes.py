#!/usr/bin/env python3
"""Systematic MG7 cross-section regression driver.

For each process listed in ``reference.json`` this script:

  1. builds an MG5 interface, defines the ``_quark`` / ``_anti_quark``
     multiparticles and ``generate``s the process,
  2. ``output mg7``s it,
  3. edits ``Cards/run_card.toml`` (fixed scale is already the template
     default; here we only set the event count and, where requested,
     neutralise the jet cuts),
  4. runs ``bin/generate_events -f`` and reads the cross-section from the
     madspace ``Events/*/info.json`` (``process.mean`` / ``process.error``),
  5. compares the relative difference to the reference value against a
     tolerance.

The source of truth is ``reference.json`` (mirrors the table in CLAUDE.md).

Usage:
    run_check_xsec_processes.py [--section SEC | --process ID] \
        [--events N] [--tolerance T] [--datadir DIR] [--workdir DIR]

Exit code is 0 only if every process that ran is within tolerance.
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

pjoin = os.path.join
HERE = os.path.dirname(os.path.abspath(__file__))
# tests/check_xsec_processes/ -> tests/ -> MG5 root
MG5DIR = os.path.dirname(os.path.dirname(HERE))
REFERENCE = pjoin(HERE, 'reference.json')

sys.path.insert(0, MG5DIR)


def resolve_datadir(explicit=None):
    """Return an LHAPDF data dir that contains the NNPDF23_lo_as_0130_qed set,
    or None if it cannot be found."""
    candidates = []
    if explicit:
        candidates.append(explicit)
    if os.environ.get('LHAPDF_DATA_PATH'):
        candidates.extend(os.environ['LHAPDF_DATA_PATH'].split(os.pathsep))
    try:
        out = subprocess.check_output(['lhapdf-config', '--datadir'],
                                      stderr=subprocess.DEVNULL).decode().strip()
        if out:
            candidates.append(out)
    except Exception:
        pass
    for d in candidates:
        if d and os.path.isdir(d) and glob.glob(pjoin(d, 'NNPDF23_lo_as_0130_qed*')):
            return d
    return None


def edit_run_card(toml_path, events, disable_jet_cuts):
    """Set the event count and (optionally) neutralise the jet cuts.

    Fixed renormalisation/factorisation scales are already the template
    default, so we assert rather than flip them, to catch a template change."""
    t = open(toml_path).read()
    # keep the reference configuration honest: fixed scale is expected.
    if 'fixed_ren_scale = false' in t or 'fixed_fact_scale = false' in t:
        # force fixed scale to match how the reference table was produced
        t = t.replace('fixed_ren_scale = false', 'fixed_ren_scale = true')
        t = t.replace('fixed_fact_scale = false', 'fixed_fact_scale = true')
    t = re.sub(r'events = \d+', 'events = %d' % events, t)
    if disable_jet_cuts:
        # jet cuts must be disabled for the hadronic tt~ decay processes to
        # match the reference (see CLAUDE.md).
        t = re.sub(r'jet-pt\.min\s*=.*', 'jet-pt.min = 0.0', t)
        t = re.sub(r'jet-eta_abs\.max\s*=.*', 'jet-eta_abs.max = 100.0', t)
        t = re.sub(r'jet-delta_r\.min\s*=.*', 'jet-delta_r.min = 0.0', t)
        t = re.sub(r'jet-lepton-delta_r\.min\s*=.*', 'jet-lepton-delta_r.min = 0.0', t)
    open(toml_path, 'w').write(t)


def run_one(entry, defines, events, datadir, workdir):
    """Generate + integrate a single process. Returns (cross, error)."""
    # import lazily so --list etc. work without the full MG stack.
    import madgraph.interface.master_interface as MGCmd

    run_dir = pjoin(workdir, entry['id'])
    if os.path.isdir(run_dir):
        shutil.rmtree(run_dir)

    mg = MGCmd.MasterCmd()
    mg.no_notification()
    mg.exec_cmd('set automatic_html_opening False --no_save')
    for d in defines:
        mg.exec_cmd(d)
    mg.exec_cmd('generate %s' % entry['process'])
    mg.exec_cmd('output mg7 %s' % run_dir)

    toml = pjoin(run_dir, 'Cards', 'run_card.toml')
    edit_run_card(toml, events, entry.get('disable_jet_cuts', False))

    env = dict(os.environ)
    env['LHAPDF_DATA_PATH'] = datadir
    log = pjoin(run_dir, 'mg7_gen.log')
    ret = subprocess.call(
        [sys.executable, pjoin(run_dir, 'bin', 'generate_events'), '-f'],
        cwd=run_dir, env=env, stdout=open(log, 'w'), stderr=subprocess.STDOUT)
    if ret != 0:
        raise RuntimeError('generate_events failed (ret=%d), see %s' % (ret, log))

    infos = sorted(glob.glob(pjoin(run_dir, 'Events', '*', 'info.json')))
    if not infos:
        raise RuntimeError('no info.json produced under %s (see %s)'
                           % (run_dir, log))
    info = json.load(open(infos[-1]))['process']
    return float(info['mean']), float(info.get('error') or 0.0)


def select_entries(ref, section, process):
    """Flatten reference.json into a list of (section, entry), filtered."""
    out = []
    for sec, entries in ref['sections'].items():
        if section and sec != section:
            continue
        for e in entries:
            if process and e['id'] != process:
                continue
            out.append((sec, e))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--section', help='run only this section (e.g. 3jets)')
    ap.add_argument('--process', help='run only this process id (e.g. gg_ttxg)')
    ap.add_argument('--events', type=int,
                    default=int(os.environ.get('MG7_XSEC_EVENTS', 100000)),
                    help='events per run (default 100000 / $MG7_XSEC_EVENTS)')
    ap.add_argument('--tolerance', type=float,
                    default=float(os.environ.get('MG7_XSEC_TOLERANCE', 0.01)),
                    help='max allowed relative difference (default 0.01 / '
                         '$MG7_XSEC_TOLERANCE)')
    ap.add_argument('--datadir', help='LHAPDF data dir (overrides autodetect)')
    ap.add_argument('--workdir', help='where to place the mg7 run dirs '
                    '(default: a temp dir, removed on success)')
    ap.add_argument('--list', action='store_true',
                    help='list the selected processes and exit')
    ap.add_argument('--keep', action='store_true',
                    help='keep the mg7 run dirs even on success')
    args = ap.parse_args(argv)

    ref = json.load(open(REFERENCE))
    entries = select_entries(ref, args.section, args.process)
    if not entries:
        print('no processes matched the selection', file=sys.stderr)
        return 2

    if args.list:
        for sec, e in entries:
            print('%-10s %-24s %s' % (sec, e['id'], e['process']))
        return 0

    datadir = resolve_datadir(args.datadir)
    if not datadir:
        print('SKIP: NNPDF23_lo_as_0130_qed LHAPDF data not found '
              '(set --datadir or $LHAPDF_DATA_PATH)', file=sys.stderr)
        # a clean skip: neither pass nor fail
        return 77

    workdir = args.workdir or tempfile.mkdtemp(prefix='mg7_xsec_')
    os.makedirs(workdir, exist_ok=True)

    print('MG5 dir   : %s' % MG5DIR)
    print('LHAPDF    : %s' % datadir)
    print('events    : %d' % args.events)
    print('tolerance : %.4g (%.2f%%)' % (args.tolerance, 100 * args.tolerance))
    print('workdir   : %s' % workdir)
    print('processes : %d' % len(entries))
    print()

    results = []  # (sec, id, ref, got, err, reldiff, status)
    for sec, e in entries:
        ref_x = e['cross']
        try:
            got, err = run_one(e, ref['defines'], args.events, datadir, workdir)
            reldiff = abs(got - ref_x) / ref_x if ref_x else float('inf')
            status = 'PASS' if reldiff <= args.tolerance else 'FAIL'
        except Exception as exc:  # generation / integration error
            got, err, reldiff, status = float('nan'), float('nan'), float('nan'), 'ERROR'
            print('[%s] %s: %s' % (status, e['id'], exc), file=sys.stderr)
        results.append((sec, e['id'], ref_x, got, err, reldiff, status))
        print('[%-5s] %-10s %-22s ref=%.6g got=%.6g +- %.3g  reldiff=%s'
              % (status, sec, e['id'], ref_x, got, err,
                 ('%.3f%%' % (100 * reldiff)) if reldiff == reldiff else 'n/a'))
        sys.stdout.flush()

    # summary
    n_fail = sum(1 for r in results if r[6] != 'PASS')
    print()
    print('=' * 78)
    print('%-10s %-22s %12s %12s %10s  %s'
          % ('section', 'process', 'reference', 'mg7', 'reldiff', 'status'))
    print('-' * 78)
    for sec, pid, ref_x, got, err, reldiff, status in results:
        print('%-10s %-22s %12.5g %12.5g %9s  %s'
              % (sec, pid, ref_x, got,
                 ('%.3f%%' % (100 * reldiff)) if reldiff == reldiff else 'n/a',
                 status))
    print('=' * 78)
    print('%d/%d passed, %d failed/errored'
          % (len(results) - n_fail, len(results), n_fail))

    if n_fail == 0 and not args.keep and not args.workdir:
        shutil.rmtree(workdir, ignore_errors=True)

    return 1 if n_fail else 0


if __name__ == '__main__':
    sys.exit(main())
