"""Reproduce onia_fks's near-threshold g g > Jpsi g points (p p > Jpsi j at
20 GeV, m_c = 1.5, mu_R = 3) with the pinned TIRust and da7902b + v16.

usage: run.py setup | run [tag ...]
"""
import json
import os
import sys

ROOT = '/shared/external/madloop_onia/madgraph7_madloop_onia'
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import onia_loop_check as olc  # noqa: E402

SP = os.path.dirname(os.path.abspath(__file__))
PW73 = os.path.join(os.path.dirname(SP), 'pw73', 'install', 'lib')
PROCESS = 'g g > Jpsi(1|3S11) g / b b~ t t~ [virt=QCD]'
BUILDS = {'pin': [], 'v16': ['set tirust %s --no_save' % PW73]}

# constituent momenta (E, px, py, pz) of legs g g c c~ g as sent by onia_fks
RAW = {
    'A': [[1.8025259761626071, 0, 0, 1.8025259761626071],
          [1.8025259761626071, 0, 0, -1.8025259761626071],
          [1.5253871421170804, -0.072965596470594890, 0.034793240511262626,
           -0.26508750533084308],
          [1.5253871421170804, -0.072965596470594890, 0.034793240511262626,
           -0.26508750533084308],
          [0.55427766809105350, 0.14593119294118978, -0.069586481022525251,
           0.53017501066168615]],
    'B': [[1.6286553635472774, 0, 0, 1.6286553635472774],
          [1.6286553635472774, 0, 0, -1.6286553635472774],
          [1.5050815546800003, -0.040124781760275716, -0.091292897056366409,
           0.072980100532352585],
          [1.5050815546800003, -0.040124781760275716, -0.091292897056366409,
           0.072980100532352585],
          [0.24714761773455443, 0.080249563520551431, 0.18258579411273282,
           -0.14596020106470517]],
    'C': [[1.6188418657243304, 0, 0, 1.6188418657243304],
          [1.6188418657243304, 0, 0, -1.6188418657243304],
          [1.5043621892131263, -0.11240417914533139, -0.018925135484390117,
           -0.010617725352996958]],
}
c = RAW['C']
c.append(list(c[2]))
c.append([c[0][i] + c[1][i] - 2 * c[2][i] for i in range(4)])


def physical(raw):
    """g g Jpsi g: the J/psi carries both constituent momenta."""
    return [raw[0], raw[1], [raw[2][i] + raw[3][i] for i in range(4)], raw[4]]


POINTS = {k: physical(v) for k, v in RAW.items()}
# MadLoopParams of the onia_fks run
PARAMS = {'CTModeRun': -1, 'MLStabThres': '1.0d-3', 'NRotations_DP': 0,
          'NRotations_QP': 0, 'UseQPIntegrandForTIRust': '.TRUE.'}


def outdir(build):
    return os.path.join(SP, 'gg_psig_' + build)


def setup():
    for build, extra in BUILDS.items():
        out = olc.output_process('loop_sm_onia-c_mass', PROCESS, outdir(build),
                                 extra=extra, regenerate=True)
        olc.set_masses(out, {4: 1.5})
        olc.build_driver(olc.subprocess_dir(out))


def run(tags):
    res = {}
    for build in BUILDS:
        out = outdir(build)
        pdir = olc.subprocess_dir(out)
        for tag, ctrun in (('default', -1), ('dp', 1), ('qp', 4)):
            if tags and tag not in tags:
                continue
            params = dict(PARAMS, CTModeRun=ctrun)
            olc.set_reduction(out, [9], params)
            r = olc.run_points(pdir, [POINTS[k] for k in 'ABC'], mu=3.0, warmup=0)
            res['%s_%s' % (build, tag)] = dict(zip('ABC', r))
            for k, x in zip('ABC', r):
                print('%-4s %-8s %s born %.10e fin %.10e c1 %.10e pred %.10e c2 %.10e '
                      'acc %.2e ret %d' % (build, tag, k, x['born'], x['finite_norm'],
                                           x['single_norm'], x['predicted_single_norm'],
                                           x['double_norm'], x['accuracy'], x['ret_code']))
    with open(os.path.join(SP, 'results.json'), 'w') as f:
        json.dump(res, f, indent=1)


if __name__ == '__main__':
    if sys.argv[1] == 'setup':
        setup()
    else:
        run(sys.argv[2:])
