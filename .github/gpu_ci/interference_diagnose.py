#!/usr/bin/env python3
"""TEMPORARY diagnosis of the HIP crash of the mg7 interference run (to be reverted).

On lemaitre4 (MI300A) mg7 p p > u u~ QCD^2==2 aborts in the survey with
HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION, while p p > t t~ and every split-order
standalone/umami check pass on HIP, and the same mg7 run passes on CUDA. Run a few
variants on the same node and report which crash:

  serial          the failing run, kernels serialized and HIP logging on (the tail of
                  the log names the last kernel launched before the fault)
  split_positive  p p > u u~ QED^2<=4: split orders, positive weights
  one_flavour     u u~ > u u~ QCD^2==2: one flavor combination, negative weights
  summed_dynamic  the failing process, interference_helicity = "summed", dynamical scale
  cpu             the failing run on the CPU of the node (control)

Appends diag_<variant>=... lines to summary.txt and leaves diag_<variant>.log (the
tail of each log) in the run directory. Never fails.
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import interference_checks as ic  # noqa: E402

VARIANTS = (
    # name, process, device, fixed scale, interference_helicity, extra environment
    ('serial', 'p p > u u~ QCD^2==2', None, True, None,
     {'AMD_SERIALIZE_KERNEL': '3', 'AMD_SERIALIZE_COPY': '3', 'AMD_LOG_LEVEL': '3'}),
    ('split_positive', 'p p > u u~ QED^2<=4', None, True, None, {}),
    ('one_flavour', 'u u~ > u u~ QCD^2==2', None, True, None, {}),
    ('summed_dynamic', 'p p > u u~ QCD^2==2', None, False, 'summed', {}),
    ('cpu', 'p p > u u~ QCD^2==2', 'cpu', True, None, {}),
)


def run_variant(repo, backend, name, process, device, fixed, helicity, env_extra, events):
    proc_dir = 'DIAG_%s' % name
    ic.madgraph(repo, 'diag_%s_generate' % name,
                ['generate %s' % process, 'output mg7 %s' % proc_dir])
    card_path = os.path.join(proc_dir, 'Cards', 'run_card.toml')
    card = open(card_path).read()
    edits = [(r'(?m)^device = .*', 'device = ["%s"]' % (device or backend)),
             (r'(?m)^seed = .*', 'seed = 31'),
             (r'(?m)^events = \d+', 'events = %d' % events),
             (r'(?m)^output_format = \S+', 'output_format = "lhe"'),
             (r'(?m)^pdf = ".*"$', 'pdf = "%s"' % ic.UUX_MG7_PDF),
             (r'(?m)^(\[systematics\]\n(?:#.*\n)*)enable = \S+', r'\1enable = false')]
    if fixed:
        edits += [(r'(?m)^fixed_ren_scale = false', 'fixed_ren_scale = true'),
                  (r'(?m)^fixed_fact_scale = false', 'fixed_fact_scale = true')]
    if helicity:
        edits.append((r'(?m)^interference_helicity = \S+',
                      'interference_helicity = "%s"' % helicity))
    for pattern, value in edits:
        card = re.sub(pattern, value, card, count=1)
    open(card_path, 'w').write(card)
    env = dict(os.environ, **env_extra)
    full_log = 'diag_%s_full.txt' % name
    with open(full_log, 'w') as out:
        ret = subprocess.call([sys.executable, 'bin/generate_events', '-f'], cwd=proc_dir,
                              stdout=out, stderr=subprocess.STDOUT, env=env)
    with open(full_log, errors='replace') as f:
        lines = f.readlines()
    # the tail of the log (the AMD log of 'serial' is long), and the error lines
    errors = [l for l in lines if re.search(r'error|violation|abort|Traceback', l, re.I)
              and 'rel. error' not in l]
    with open('diag_%s.log' % name, 'w') as f:
        f.write('exit status %d, %d lines\n--- error lines (first 40)\n' % (ret, len(lines)))
        f.writelines(errors[:40])
        f.write('--- tail (400 lines)\n')
        f.writelines(lines[-400:])
    os.remove(full_log)
    infos = sorted(glob.glob(os.path.join(proc_dir, 'Events', '*', 'info.json')))
    xsec = ''
    if infos:
        try:
            p = json.load(open(infos[-1]))['process']
            xsec = ', %.6g +- %.3g pb' % (p['mean'], p['error'])
        except Exception:
            pass
    first_error = errors[0].strip()[:160] if errors else ''
    return ('ok' if ret == 0 else 'CRASH (exit %d)' % ret) + xsec + \
        ('' if ret == 0 else ': ' + first_error)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', required=True)
    parser.add_argument('--backend', required=True)
    parser.add_argument('--events', type=int, default=2000)
    args = parser.parse_args()
    results = []
    for name, process, device, fixed, helicity, env_extra in VARIANTS:
        ic.section('diagnosis %s: %s' % (name, process))
        try:
            result = run_variant(args.repo, args.backend, name, process, device, fixed,
                                 helicity, env_extra, args.events)
        except Exception as error:
            result = 'ERROR %s' % error
        print('diag_%s=%s' % (name, result), flush=True)
        results.append((name, result))
    with open('summary.txt', 'a') as f:
        for name, result in results:
            f.write('diag_%s=%s\n' % (name, result.replace('|', '/')))
    return 0


if __name__ == '__main__':
    sys.exit(main())
