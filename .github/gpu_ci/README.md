# GPU CI on the CECI clusters

Two workflows run MadGraph7 on real GPUs:

| workflow | cluster | partition | backend |
|---|---|---|---|
| `gpu_manneback_cuda.yml` | manneback | `gpu` (NVidia) | CUDA |
| `gpu_lemaitre4_hip.yml` | lemaitre4 | `apu` (AMD MI300A) | HIP |

They run on every push to `main` and from the Actions tab ("Run workflow"). There you can
choose the modules to load, which set the CUDA/ROCm, GCC and Python versions. You can also
choose the GPU model (`gres`), the GPU architecture, the number of events and the Slurm time limit.

## How it works

Both call `gpu_runner_ci.yml`. A GitHub self-hosted runner is started inside **one** Slurm
allocation per workflow run, and every CI job of that run executes in it:

| job | runs on | what it does |
|---|---|---|
| `start_runner` | GitHub | ssh to the cluster, `sbatch` `runner_batch.sh` on the GPU partition, wait until the runner is online (`remote_start_runner.sh`) |
| `build_madspace` | cluster | build madspace with `ENABLE_CUDA`/`ENABLE_HIP` for the GPU of the node (`build_madspace_gpu.sh`). The build is kept in `$GLOBALSCRATCH/mg7-gpu-ci/cache` and redone only when the madspace sources, the modules or the GPU architecture change |
| `pp_ttx` | cluster | `generate p p > t t~`, `output mg7`, `device = ["cuda"]` or `["hip"]` in `run_card.toml`, `bin/generate_events -f`, check the cross section in `info.json` (`pp_ttx_mg7.sh`). The logs and cards are uploaded as an artifact, and the cross section is shown in the run summary |
| `stop_runner` | cluster | clean up, then create the stop file: `runner_batch.sh` stops the runner and the allocation ends |

To add a CI job, give it `runs-on: [self-hosted, "${{ inputs.runner_label }}"]` and add it
to the `needs` of `stop_runner`. It runs in the same allocation, after the others, so
`time_limit` must cover all the jobs together. The first madspace build (OpenBLAS + GPU code)
is the slow part.

Other safeguards:
* **One run at a time per cluster.** The runner jobs of a cluster share the name
  `mg7-gh-runner-<label>` and use `--dependency=singleton`, so a second runner waits until the
  first one has stopped. The workflows also use a concurrency group per cluster.
* **Cancelled runs.** If a run is cancelled before its runner starts, the runner stops by
  itself after 10 minutes without a job.

## Security model

The runner runs under a personal CECI account (`omatt`), and 14 people can push to this public
repository. Hence:

* **The ssh key is in the `ceci-gpu` environment, not in the repository secrets.** Only
  `start_runner` (and the module-listing workflow) use that environment. Restricted to `main`,
  a branch cannot get the key, not even by editing a workflow.
  The calling workflows use `secrets: inherit`: without it, GitHub does not pass an
  environment secret to a job of the reusable workflow. The repository has no other secrets.
* **The runner only executes the jobs of the run that started it.** Its job-started hook,
  `job_started_hook.sh`, compares `GITHUB_RUN_ID` with the run the allocation was started for.
  Any other job fails before its first step: another run, a branch, or a fork PR that
  targets the runner label while it is online.
* **Cluster files come from `main`.** `runner_batch.sh` and `job_started_hook.sh` are copied to
  the cluster from `main` at every start, so changing them goes through a reviewed PR.

## One-time setup (repository admin)

1. **Environment and secret.** Create the `ceci-gpu` environment with `main` as the only
   deployment branch, then store the key (the private CECI ssh key, without a passphrase):

   ```bash
   gh api -X PUT repos/MadGraphTeam/MadGraph7/environments/ceci-gpu \
     -F 'deployment_branch_policy[protected_branches]=false' \
     -F 'deployment_branch_policy[custom_branch_policies]=true'
   gh api -X POST repos/MadGraphTeam/MadGraph7/environments/ceci-gpu/deployment-branch-policies \
     -f name=main -f type=branch
   gh secret set CECI_KEY --env ceci-gpu --repo MadGraphTeam/MadGraph7 < /path/to/ceci_private_key
   ```

   Optionally, add yourself as *required reviewer* of the environment (Settings → Environments →
   ceci-gpu). Nothing then touches the cluster without your approval, which is asked once per
   run, for `start_runner`. To use another CECI account, set the environment variable `CECI_USER`.

2. **Runners**, once per cluster, from a clone of the repository:

   ```bash
   { echo "TOKEN=$(gh api -X POST repos/MadGraphTeam/MadGraph7/actions/runners/registration-token --jq .token)"
     cat .github/gpu_ci/install_runner.sh; } | ssh manneback bash -l -s -- manneback-gpu
   { echo "TOKEN=$(gh api -X POST repos/MadGraphTeam/MadGraph7/actions/runners/registration-token --jq .token)"
     cat .github/gpu_ci/install_runner.sh; } | ssh lemaitre4 bash -l -s -- lemaitre4-apu
   ```

   The runners are installed in `~/mg7-gpu-ci/runner-<label>`, and their jobs work in
   `$GLOBALSCRATCH/mg7-gpu-ci/work-<label>`. Do not start them by hand.

3. **Checks.**
   * The GPU nodes must reach github.com. Try e.g.
     `ssh manneback srun -p gpu --gres=gpu:1 -t 2 curl -sI https://github.com`.
   * Run *GPU CI - list cluster modules* with `partition` set to `gpu` or `apu` and `gres` to
     `gpu:1`. Then adapt the default module lists in the two workflow files. The lists need
     python >= 3.11, a C++ compiler and `nvcc`, or `hipcc` with rocBLAS/rocThrust/hipCUB.
   * Recommended: under Settings → Actions → General, require approval for workflows from all
     outside contributors.

## Troubleshooting

* **`start_runner` fails with "Runner registration has been deleted".** GitHub removes a
  self-hosted runner that has not connected for 14 days. Run the install command of step 2 again.
* **A runner job is stuck on the cluster.** Look for it with
  `squeue -u $USER -n mg7-gh-runner-<label>`, and remove it with `scancel`.
* **Runner logs** are in `~/mg7-gpu-ci/runner-<label>/slurm-<jobid>.out`.
