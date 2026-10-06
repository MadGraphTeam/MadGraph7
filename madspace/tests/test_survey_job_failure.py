"""A job that throws on a worker thread must fail EventGenerator.survey() instead
of hanging it.

survey() and generate() count the jobs they have in flight and block on the
result queue until each has posted. A job that threw used to post nothing: the
thread pool caught the exception and kept it for a ThreadPool.wait() that this
loop never calls, so the main thread blocked forever at 0% CPU right after
"survey started". Seen in mg7 on a $-excluded channel whose points all fail the
veto, where ChannelEventGenerator::start_job throws "not enough points passing
cuts".

The survey runs in a child process: with the bug it never returns, and holding
the GIL it can't be timed out from a thread.
"""

import os
import subprocess
import sys
import textwrap

import madspace as ms

TIMEOUT = 120

SCRIPT = textwrap.dedent(
    """
    import os
    import sys

    import madspace as ms

    CM_ENERGY = 1000.0
    run_dir = sys.argv[1]
    dead_pt_min = float(sys.argv[2])

    context = ms.Context(ms.cpu_device(), 2)
    # RunningCoupling only reads these two keys of an LHAPDF .info file
    alphas_file = os.path.join(run_dir, "alphas.info")
    with open(alphas_file, "w") as f:
        f.write("AlphaS_Qs: [1.0, 10.0, 100.0, 1000.0, 10000.0]\\n")
        f.write("AlphaS_Vals: [0.3, 0.2, 0.12, 0.1, 0.08]\\n")
    alphas_grid = ms.AlphaSGrid(alphas_file)
    alphas_grid.initialize_globals(context)

    config = ms.GeneratorConfig()
    config.max_cut_repetitions = 5
    config.start_batch_size = 100
    config.cpu_batch_size = 100


    def channel(name, pt_min):
        pids = [11, -11, 1, -1]
        obs = ms.Observable(pids, ms.Observable.obs_pt, [[1, -1]])
        mapping = ms.PhaseSpaceMapping(
            [0.0] * 4,
            CM_ENERGY,
            leptonic=True,
            cuts=ms.Cuts([ms.CutItem(obs, min=pt_min)]),
        )
        # 0xBADCAFE: the built-in flat test matrix element, no library needed
        matrix_element = ms.MatrixElement(
            0xBADCAFE,
            4,
            ms.Integrand.matrix_element_inputs,
            ms.Integrand.matrix_element_outputs,
            1,
            True,
        )
        diff_xs = ms.DifferentialCrossSection(
            matrix_element=matrix_element,
            cm_energy=CM_ENERGY,
            running_coupling=None,
            energy_scale=ms.CachedScale(),
        )
        integrand = ms.Integrand(
            mapping,
            [diff_xs],
            running_coupling=ms.RunningCoupling(alphas_grid),
            energy_scale=ms.EnergyScale(4, 91.188),
            channel_indices=[0],
        )
        return ms.ChannelEventGenerator(
            contexts=[context],
            integrand=integrand,
            event_file=os.path.join(run_dir, f"events.{name}.npy"),
            weight_file=os.path.join(run_dir, f"weights.{name}.npy"),
            config=config,
            subprocess_index=0,
            name=name,
            histograms=None,
        )


    generator = ms.EventGenerator(
        [context],
        [channel("alive", 10.0), channel("dead", dead_pt_min)],
        seed=1,
        config=config,
    )
    generator.survey()
    print("survey done")
    """
)


def run_survey(tmp_path, dead_pt_min):
    env = dict(os.environ)
    # the madspace this test imported, not whatever the child's cwd shadows
    madspace_root = os.path.dirname(os.path.dirname(ms.__file__))
    env["PYTHONPATH"] = os.pathsep.join(
        [madspace_root] + [p for p in [env.get("PYTHONPATH")] if p]
    )
    return subprocess.run(
        [sys.executable, "-c", SCRIPT, str(tmp_path), str(dead_pt_min)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
    )


def test_survey_completes_when_every_channel_has_points(tmp_path):
    # Control: the same two channels with a cut both pass.
    result = run_survey(tmp_path, dead_pt_min=10.0)
    assert result.returncode == 0, result.stderr
    assert "survey done" in result.stdout


def test_survey_raises_when_a_job_throws(tmp_path):
    # pt > sqrt(s)/2 is kinematically impossible: every batch of the "dead"
    # channel fails the cuts, and its job throws after max_cut_repetitions.
    # Before the fix this timed out.
    result = run_survey(tmp_path, dead_pt_min=1000.0)
    assert result.returncode != 0
    assert "survey done" not in result.stdout
    assert "not enough points passing cuts" in result.stderr
