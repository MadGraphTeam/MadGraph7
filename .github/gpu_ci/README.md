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
| `start_runner` | vmadgraph | ask `ceci-gate` on the VM to `sbatch` `runner_batch.sh` on the GPU partition, wait until the runner is online (`remote_start_runner.sh`) |
| `build_madspace` | cluster | build madspace with `ENABLE_CUDA`/`ENABLE_HIP` for the GPU of the node (`build_madspace_gpu.sh`). The build is kept in `$GLOBALSCRATCH/mg7-gpu-ci/cache` and redone only when the madspace sources, the modules or the GPU architecture change |
| `pp_ttx` | cluster | `generate p p > t t~`, `output mg7`, `device = ["cuda"]` or `["hip"]` in `run_card.toml`, `bin/generate_events -f`, check the cross section in `info.json` (`pp_ttx_mg7.sh`). The logs and cards are uploaded as an artifact, and the cross section is shown in the run summary |
| `pp_jj` | cluster | the same with `generate p p > j j` (`PROCESS`/`TAG` of `pp_ttx_mg7.sh`): its jet cuts act on several objects, which `p p > t t~` has none of |
| `madspace_ops` | cluster | madspace GPU runtime operations against the CPU runtime on the same inputs (`madspace_ops_gpu.sh`, `madspace_checks.py`; torch puts the inputs on the GPU): a histogram with underflow and overflow bins, an elementwise kernel on a rank-4 tensor |
| `acceptance` | cluster | the acceptance tests that need a GPU, through `tests/test_manager.py` (`acceptance_gpu.sh`): they skip without `nvcc`/`hipcc` or a device, so a skip fails the job. Now `test_standalone_umami_first_call_gpu` (the good helicities chosen at `umami_initialize`, with device buffers). To add a test, append it to `DEFAULT_TESTS` in `acceptance_gpu.sh` |
| `interference` | cluster | squared split orders on the GPU backend of madmatrix (`interference_gpu.sh`, `interference_checks.py`): standalone `u u~ > u u~` constraints, the helicity choice of an interference matrix element through umami (exact and summed), and `p p > u u~ QCD^2==2` with mg7 against its CPU reference. Input `interference`, on by default and on push |
| `stop_runner` | cluster | clean up, then create the stop file: `runner_batch.sh` stops the runner and the allocation ends |

To add a CI job, give it `runs-on: [self-hosted, "${{ inputs.runner_label }}"]` and add it
to the `needs` of `stop_runner`. It runs in the same allocation, after the others, so
`time_limit` must cover all the jobs together. The first madspace build (OpenBLAS + GPU code)
is the slow part. The default is kept short so that Slurm can backfill the allocation:
20 minutes on lemaitre4 (successful runs take 1-12 minutes) and 30 minutes on manneback
(usually 2-17 minutes, but the runner set-up and the checkout are sometimes slow there).
Raise it from the Actions tab for a run that needs more, or in the caller workflow when a
new job makes every run longer.

Other safeguards:
* **One run at a time per cluster.** The runner jobs of a cluster share the name
  `mg7-gh-runner-<label>` and use `--dependency=singleton`, so a second runner waits until the
  first one has stopped. The workflows also use a concurrency group per cluster.
* **Cancelled runs.** If a run is cancelled before its runner starts, the runner stops by
  itself after 10 minutes without a job.
* **One vmadgraph runner per cluster.** `start_runner` holds its runner while the allocation
  waits in the Slurm queue (up to 5 hours). It runs on `[self-hosted, vmadgraph,
  vmadgraph-<cluster>]`, and each runner on the vmadgraph VM carries the label of one cluster
  (`vmadgraph` → `vmadgraph-lemaitre4`, `vmadgraph-2` → `vmadgraph-manneback`), so a busy
  cluster does not block the other one. The two runners run under their own Unix users,
  `github-runner` and `ghrunner2`.

## Security model

The runner runs under a personal CECI account (`omatt`), 14 people can push to this public
repository, and 3 organisation owners can change any repository setting. Hence:

* **The ssh key is not on GitHub.** It is only on the vmadgraph VM, readable only by the Unix
  user `ceci` (`/var/lib/ceci/.ssh/id_ceci`). The runner users may run one program as `ceci`,
  `/usr/local/sbin/ceci-gate` ([vm_gate/ceci-gate](vm_gate/ceci-gate)), and nothing else.
  Before it touches a cluster, the gate asks the GitHub API whether the run it is called for
  is in progress, comes from `gpu_lemaitre4_hip.yml`, `gpu_manneback_cuda.yml` or
  `gpu_cluster_modules.yml` of this repository (not a fork), was started or re-run by a login
  listed in `/etc/ceci-gate/allowed_actors` on the VM **or** is on a branch listed in
  `/etc/ceci-gate/allowed_branches` (e.g. `main`, so that every commit landing there is tested,
  whoever pushed or merged it), and has its `start_runner` (or `modules`) job running right now
  on the calling runner. It also bounds the Slurm options, starts each
  run attempt once, and logs every decision in `/var/lib/ceci/gate/gate.log`. Who may use the
  cluster is thus decided on the VM: no repository setting, branch or workflow edit changes it.
* **The runner only executes the jobs of the run that started it.** Its job-started hook,
  `job_started_hook.sh`, compares `GITHUB_RUN_ID` with the run the allocation was started for.
  Any other job fails before its first step: another run, a branch, or a fork PR that
  targets the runner label while it is online.
* **Cluster files come from the VM.** `ceci-gate` copies `runner_batch.sh` and
  `job_started_hook.sh` to the cluster at every start, and pipes `remote_start_runner.sh` and
  `remote_modules.sh`, all from `/usr/local/share/ceci-gate`. After a change to one of them,
  install it there again (step 1).

What an allowed run executes on the cluster is still the code of its commit: dispatch only
branches you trust. A listed branch is only as safe as GitHub keeps it: whoever can push to it
gets their code run on the cluster. List only branches that a ruleset restricts (pull requests
for `main`, updates by named people for the others), knowing that the organisation owners can
change rulesets.

## One-time setup (VM administrator)

1. **The gate, on the vmadgraph VM** (Rocky Linux 8, SELinux enforcing), as root:

   ```bash
   # a locked system user that holds the key; a real shell, which ssh needs for ProxyJump
   useradd --system --create-home --home-dir /var/lib/ceci --shell /bin/bash ceci
   chmod 700 /var/lib/ceci
   install -d -m 700 -o ceci -g ceci /var/lib/ceci/.ssh
   # the private CECI key without passphrase as id_ceci, and known_hosts + config with the
   # pinned host keys and the gwceci ProxyJump (User omatt, IdentityFile ~/.ssh/id_ceci)
   chown -R ceci:ceci /var/lib/ceci/.ssh && chmod 600 /var/lib/ceci/.ssh/*
   # outside /home, SELinux only lets ssh read the key with this label
   semanage fcontext -a -t ssh_home_t '/var/lib/ceci/\.ssh(/.*)?' && restorecon -Rv /var/lib/ceci/.ssh
   # the gate and the cluster-side scripts, from a reviewed checkout, root-owned
   install -o root -g root -m 755 vm_gate/ceci-gate /usr/local/sbin/ceci-gate
   install -d -o root -g root -m 755 /usr/local/share/ceci-gate /etc/ceci-gate
   install -o root -g root -m 755 remote_start_runner.sh runner_batch.sh job_started_hook.sh \
     remote_modules.sh /usr/local/share/ceci-gate/
   # the GitHub logins whose runs may use the cluster, and the branches whose runs may use
   # it whoever started them, one per line
   echo oliviermattelaer > /etc/ceci-gate/allowed_actors
   echo main > /etc/ceci-gate/allowed_branches
   echo 'github-runner, ghrunner2 ALL=(ceci) NOPASSWD: /usr/local/sbin/ceci-gate' > /tmp/g \
     && visudo -cf /tmp/g && install -m 440 /tmp/g /etc/sudoers.d/ceci-gate; rm -f /tmp/g
   ```

   Check it as a runner user: `sudo -u github-runner sudo -n -u ceci /usr/local/sbin/ceci-gate
   ping lemaitre4` must print the login node, while `sudo -u github-runner cat
   /var/lib/ceci/.ssh/id_ceci` must be denied. The runner names and Unix users (`RUNNERS`), the
   clusters, their runner labels and partitions (`CLUSTERS`) and the Slurm bounds are set at the
   top of `ceci-gate`. The gate makes 2-5 unauthenticated GitHub API calls per start (60 per hour
   per IP); a read-only token in `/var/lib/ceci/.github_token` (mode 600, owner `ceci`) lifts
   that limit.

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
