"""The cluster mode of the MG7 launcher ([cluster] in the run card).

The subprocesses of a process are trained one by one in a normal run. Here the
run is split in two passes of batch jobs, built on gridpacks:

1. training: every job runs the survey and the training for some of the
   subprocesses (merge groups) and saves the data of its gridpack;
2. the gridpack data of all jobs is merged into one gridpack;
3. generation: every job runs this gridpack with another seed, for a part of
   the events (not with cluster.mode = "gridpack_only", where the merged
   gridpack is the result);
4. the events of the jobs are combined.

The steps that are done are recorded in cluster/state.json of the run
directory: a run that is launched again with the same cards continues where
the earlier one stopped, and only repeats the jobs that did not finish.
"""

import copy
import glob
import hashlib
import json
import logging
import math
import os
import shlex
import shutil
import sys
import time

from madgraph.iolibs.template_files.mg7 import combine_cluster
from madgraph.mg7 import launch
from madgraph.various.banner import RunCardMG7
from madgraph.iolibs.template_files.mg7 import merge_gridpacks
from madgraph.various import cluster as cluster_module

logger = logging.getLogger("madgraph7")

STATE_FILE = os.path.join("cluster", "state.json")


class ClusterRunError(Exception):
    """The cluster run can not go on."""


def derived_seed(master_seed: int, *names) -> int:
    """A reproducible 63 bit seed for the job identified by ``names``."""
    text = ":".join(str(name) for name in (master_seed,) + names)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") >> 1


def card_hash() -> str:
    """Identifies the cards a run was started with."""
    digest = hashlib.sha256()
    for name in ("run_card.toml", "param_card.dat"):
        with open(os.path.join("Cards", name), "rb") as f:
            digest.update(f.read())
    return digest.hexdigest()


def split_events(total: int, events_per_job: int) -> list:
    """The events of each generation job: as equal as possible, none above
    events_per_job."""
    jobs = max(1, math.ceil(total / events_per_job))
    return [total // jobs + (1 if i < total % jobs else 0) for i in range(jobs)]


def split_subprocesses(unit_count: int, per_job: int) -> list:
    """The subprocess (merge group) indices of each training job."""
    return [list(range(start, min(start + per_job, unit_count)))
            for start in range(0, unit_count, per_job)]


def write_json(path: str, data) -> None:
    """Atomic, so that an interrupted run never leaves a half written file."""
    with open(path + ".tmp", "w") as f:
        json.dump(data, f, indent=1)
    os.replace(path + ".tmp", path)


def read_json(path: str):
    with open(path) as f:
        return json.load(f)


def make_cluster(card, options: dict, queue: str, walltime: str, extra_args: str):
    """The cluster object of one pass: [cluster] settings, where empty those of
    the MadGraph configuration (``options``, see launch.load_mg5_options)."""
    section = card["cluster"]
    cluster_type = section["type"]
    if cluster_type == "auto":
        cluster_type = options.get("cluster_type")
        if cluster_type in (None, "None", ""):
            raise ClusterRunError(
                "no batch system: set [cluster] type, or cluster_type in the "
                "MadGraph configuration")
    if cluster_type == "local_parallel":
        parallel_jobs = section["parallel_jobs"]
        if parallel_jobs is None or parallel_jobs < 1:
            parallel_jobs = os.cpu_count() or 1
        return cluster_module.MultiCore(nb_core=parallel_jobs), cluster_type
    if cluster_type not in cluster_module.from_name:
        raise ClusterRunError(f"unknown batch system {cluster_type!r}")

    def pick(value, key):
        return value if value else options.get(key)

    config = {
        "cluster_type": cluster_type,
        "cluster_queue": pick(queue, "cluster_queue"),
        "cluster_walltime": pick(walltime, "cluster_walltime"),
        "cluster_extra_args": extra_args,
        "cluster_nb_retry": section["nb_retry"],
        "cluster_retry_wait": section["retry_wait"],
        "cluster_status_update": tuple(section["status_update"]),
        "cluster_temp_path": pick(section["temp_path"], "cluster_temp_path"),
        "cluster_requirement": options.get("cluster_requirement"),
    }
    return cluster_module.from_name[cluster_type](**config), cluster_type


class ClusterRun:
    """One run of the cluster mode."""

    def __init__(self, process, options: dict):
        self.process = process
        self.card = process.run_card
        self.section = self.card["cluster"]
        self.options = options
        self.process_dir = os.getcwd()
        self.executable = os.path.join(self.process_dir, "bin", "generate_events")
        # the run card of each pass, with its [cluster_<pass>] overrides
        self.pass_cards = {name: self.card.cluster_pass_card(name)
                           for name in ("train", "deploy")}
        self.devices = {name: list(card["run"]["device"])
                        for name, card in self.pass_cards.items()}
        self.state = None

    # ------------------------------------------------------------------
    # run directory and state
    # ------------------------------------------------------------------
    def find_unfinished_run(self, digest: str):
        """The newest run directory of this run name, when it was left
        unfinished by a run with the same cards."""
        prefix = os.path.join("Events", self.card["run"]["run_name"] + "_")
        runs = sorted((d for d in glob.glob(prefix + "*") if d[len(prefix):].isnumeric()),
                      key=lambda d: int(d[len(prefix):]))
        if not runs:
            return None
        state_path = os.path.join(runs[-1], STATE_FILE)
        if not os.path.exists(state_path):
            return None
        state = read_json(state_path)
        if state["stage"] == "done" or state["hash"] != digest:
            return None
        return runs[-1]

    def set_stage(self, stage: str) -> None:
        self.state["stage"] = stage
        write_json(self.state_path, self.state)

    def start(self) -> None:
        digest = card_hash()
        run_path = self.find_unfinished_run(digest)
        if run_path is None:
            run_path = launch.new_run_path(self.card["run"]["run_name"])
            os.makedirs(os.path.join(run_path, "cluster"))
            seed = launch.resolve_seed(self.card["run"]["seed"])
            unit_count = len(self.process.merged_subprocess_data
                             if self.process.merged_subprocess_data is not None
                             else self.process.subprocess_data)
            train_jobs = [
                {"index": i, "selection": selection,
                 "seed": derived_seed(seed, "train", i)}
                for i, selection in enumerate(split_subprocesses(
                    unit_count, self.section["subprocesses_per_job"]))
            ]
            deploy_jobs = [
                {"index": i, "events": events,
                 "seed": derived_seed(seed, "deploy", i)}
                for i, events in enumerate(split_events(
                    self.card["generation"]["events"], self.section["events_per_job"]))
            ]
            self.state = {"hash": digest, "seed": seed, "stage": "train",
                          "train_jobs": train_jobs, "deploy_jobs": deploy_jobs}
            self.state_path = os.path.join(run_path, STATE_FILE)
            write_json(self.state_path, self.state)
        else:
            self.state_path = os.path.join(run_path, STATE_FILE)
            self.state = read_json(self.state_path)
            logger.info("continuing the unfinished run %s (stage %s)",
                        run_path, self.state["stage"])
        self.run_path = os.path.abspath(run_path)
        self.process.run_path = self.run_path
        self.cluster_path = os.path.join(self.run_path, "cluster")

    # ------------------------------------------------------------------
    # preparation
    # ------------------------------------------------------------------
    def thread_count(self, name: str, cluster_type: str) -> int:
        """Threads of one job of the pass ``name``: the cores of the machine
        are shared by the jobs of a local_parallel run."""
        threads = self.pass_cards[name]["run"]["cpu_thread_pool_size"]
        if cluster_type == "local_parallel" and threads <= 0:
            jobs = self.cluster_objects["train"][0].nb_core
            return max(1, (os.cpu_count() or 1) // max(1, jobs))
        return threads

    def build_matrix_elements(self) -> None:
        """Compile the matrix elements for the devices (and cpu_mode) of both
        passes, so that the jobs find them. The backend 'auto' stands for is
        resolved here too: every job then uses the one built."""
        self.backends = {}
        saved = self.process.run_card
        try:
            for name in ("train", "deploy"):
                self.process.run_card = self.pass_cards[name]
                self.backends[name] = self.process.compile_matrix_elements()
        finally:
            self.process.run_card = saved
        self.process.init_custom_function_paths()

    def cpu_mode_of(self, name: str):
        """The resolved cpu_mode of the cpu device of a pass, if it has one."""
        for device, backend in zip(self.devices[name], self.backends[name]):
            if device.split(":")[0] == "cpu":
                return backend
        return None

    def make_clusters(self) -> None:
        self.cluster_objects = {}
        for name in ("train", "deploy"):
            self.cluster_objects[name] = make_cluster(
                self.card, self.options, self.section[name + "_queue"],
                self.section[name + "_walltime"], self.section[name + "_extra_args"])
        self.cluster_type = self.cluster_objects["train"][1]

    # ------------------------------------------------------------------
    # submitting and waiting
    # ------------------------------------------------------------------
    def run_jobs(self, name: str, jobs: list) -> None:
        """Submit the jobs of a pass, ``jobs`` being (directory, arguments,
        required output) and wait until they are done."""
        cluster = self.cluster_objects[name][0]
        # The batch jobs are found in the queue by a name derived from the path
        # that wait() is given. Name them all like that: otherwise a job is not
        # found, taken for finished, and resubmitted once cluster_retry_wait has
        # passed without its output.
        identifier = cluster.get_jobs_identifier(self.run_path)
        cluster.get_jobs_identifier = lambda path, second_path=None: identifier
        for job_dir, arguments, required_output, program in jobs:
            # Batch systems run a copy of the submitted script (slurm: in its
            # spool directory), but bin/generate_events finds the directory it
            # works in from its own location. A launcher script that runs the
            # real file keeps that location.
            launcher = os.path.join(job_dir, "launch.sh")
            with open(launcher, "w") as f:
                # MG7_CLUSTER_JOB_ENV: an optional shell file, read on the node
                # before the job starts (modules, a node-local TMPDIR, ...)
                f.write('#!/bin/bash\n[ -n "${MG7_CLUSTER_JOB_ENV:-}" ] && '
                        '. "$MG7_CLUSTER_JOB_ENV"\nexec %s "$@"\n'
                        % shlex.quote(program))
            os.chmod(launcher, 0o755)
            cluster.submit2(
                launcher, arguments, cwd=job_dir,
                stdout=os.path.join(job_dir, "log"), stderr=-2,
                required_output=required_output)
        self.record_job_ids(name, cluster)
        state = []

        def update(idle, running, done):
            # resubmitted jobs get new ids
            self.record_job_ids(name, cluster)
            if (idle, running, done) != tuple(state):
                state[:] = [idle, running, done]
                logger.info("%s jobs: %d waiting, %d running, %d done",
                            name, idle, running, done)

        try:
            cluster.wait(self.run_path, update)
        except BaseException:
            try:
                cluster.remove()
            except Exception:
                pass
            raise
        self.clear_job_ids(name)

    def record_job_ids(self, name: str, cluster) -> None:
        """Keep the batch ids of the jobs of a pass in the state, for a rerun to
        cancel the ones a killed steering process left behind."""
        if self.cluster_type == "local_parallel":
            return
        ids = self.state.setdefault("job_ids", {}).setdefault(name, [])
        new = [str(i) for i in cluster.submitted_ids if str(i) not in ids]
        if new:
            ids.extend(new)
            write_json(self.state_path, self.state)

    def clear_job_ids(self, name: str) -> None:
        if self.state.get("job_ids", {}).pop(name, None) is not None:
            write_json(self.state_path, self.state)

    def cancel_earlier_jobs(self, name: str) -> None:
        """Cancel the jobs of the pass that an earlier, interrupted steering
        process submitted: they would write into the directories of the jobs
        submitted again."""
        ids = self.state.get("job_ids", {}).get(name)
        if not ids:
            return
        logger.info("cancelling the %d %s job(s) of the interrupted run that may "
                    "still be queued or running", len(ids), name)
        cluster = self.cluster_objects[name][0]
        cluster.submitted_ids = list(ids)
        cluster.remove()
        cluster.submitted_ids = []
        self.clear_job_ids(name)

    def job_dir(self, name: str, index: int) -> str:
        return os.path.join(self.cluster_path, name, "job_%03d" % index)

    # ------------------------------------------------------------------
    # training pass
    # ------------------------------------------------------------------
    def train_job_done(self, job_dir: str) -> bool:
        return os.path.exists(os.path.join(job_dir, "done"))

    def write_train_card(self, job: dict, job_dir: str) -> None:
        threads = self.thread_count("train", self.cluster_type)
        fix_max_weight = self.section["fix_max_weight"]
        # the [cluster_train] overrides first: the settings of the cluster mode
        # below take the overridden devices and threads into account
        settings = self.card.cluster_overrides("train") + [
            ("cluster.enable", False),
            ("phasespace.subprocess_selection", job["selection"]),
            ("run.seed", job["seed"]),
            ("run.device", self.devices["train"]),
            ("run.verbosity", "log"),
            ("run.make_plots", False),
            ("run.postprocessing_histograms", False),
            ("run.cpu_thread_pool_size", threads),
            ("gridpack.save_gridpack", True),
            ("gridpack.run_mode", "fix_max_weight" if fix_max_weight else "minimal"),
            ("gridpack.compress", False),
            ("gridpack.output_dir", job_dir),
            ("gridpack.temp_output_dir", "auto"),
        ]
        cpu_mode = self.cpu_mode_of("train")
        if cpu_mode is not None:
            settings.append(("run.cpu_mode", cpu_mode))
        card = RunCardMG7(os.path.join("Cards", "run_card.toml"))
        for key, value in settings:
            card.set(key, value, user=True)
        card.write(os.path.join(job_dir, "run_card.toml"))

    def train(self) -> None:
        self.cancel_earlier_jobs("train")
        jobs = []
        for job in self.state["train_jobs"]:
            job_dir = self.job_dir("train", job["index"])
            if self.train_job_done(job_dir):
                continue
            shutil.rmtree(job_dir, ignore_errors=True)
            os.makedirs(job_dir)
            self.write_train_card(job, job_dir)
            jobs.append((job_dir, ["--cluster-job", job_dir], ["done"], self.executable))
        total = len(self.state["train_jobs"])
        logger.info("training pass: %d job(s) for %d subprocess group(s)%s",
                    total, sum(len(j["selection"]) for j in self.state["train_jobs"]),
                    "" if len(jobs) == total else
                    f", {total - len(jobs)} of them done already")
        if jobs:
            self.run_jobs("train", jobs)
        failed = [j["index"] for j in self.state["train_jobs"]
                  if not self.train_job_done(self.job_dir("train", j["index"]))]
        if failed:
            raise ClusterRunError(
                f"the training jobs {failed} did not finish, see the log files in "
                f"{os.path.join(self.cluster_path, 'train')}")
        self.log_training_result()

    def log_training_result(self) -> None:
        """The cross section the training jobs found, as far as they say."""
        mean = error2 = 0.
        try:
            for job in self.state["train_jobs"]:
                process = read_json(os.path.join(
                    self.job_dir("train", job["index"]), "info.json"))["process"]
                mean += process["mean_abs"]
                error2 += process["error_abs"] ** 2
        except (OSError, KeyError, ValueError):
            return
        if mean:
            logger.info("training done, absolute cross section of the training "
                        "jobs: %.6g +- %.2g pb", mean, error2 ** 0.5)

    # ------------------------------------------------------------------
    # merging
    # ------------------------------------------------------------------
    def merge(self) -> str:
        """Merge the gridpacks of the training jobs and complete the result."""
        gridpack_path = os.path.join(self.cluster_path, "gridpack")
        shutil.rmtree(gridpack_path, ignore_errors=True)
        os.makedirs(gridpack_path)
        inputs = [os.path.join(self.job_dir("train", job["index"]), "gridpack", "data")
                  for job in self.state["train_jobs"]]
        try:
            merge_gridpacks.merge_gridpack_data(inputs, os.path.join(gridpack_path, "data"))
        except merge_gridpacks.MergeError as error:
            raise ClusterRunError(f"merging the gridpacks failed: {error}") from error
        # the shell of the gridpack is the one for the generation pass: its
        # devices, and the resolved backend of its cpu
        card = RunCardMG7(os.path.join("Cards", "run_card.toml")).cluster_pass_card("deploy")
        card.set("cluster.enable", False, user=True)
        # the mode of the training jobs (see write_train_card), not of the main card
        run_mode = "fix_max_weight" if self.section["fix_max_weight"] else "minimal"
        launch.MadgraphProcess.write_gridpack_shell(
            gridpack_path, card, self.backends["deploy"],
            self.process.custom_function_paths, self.process.subprocess_data,
            run_mode)
        return gridpack_path

    def clean_training_data(self) -> None:
        """Drop the gridpacks of the training jobs (network weights and grids)
        once merged, keeping their logs and info.json."""
        if self.section["keep_job_files"]:
            return
        for job in self.state["train_jobs"]:
            shutil.rmtree(os.path.join(self.job_dir("train", job["index"]), "gridpack"),
                          ignore_errors=True)

    # ------------------------------------------------------------------
    # generation pass
    # ------------------------------------------------------------------
    def deploy_job_done(self, job_dir: str) -> bool:
        # the file of the job itself: info.json and the events are there before
        # a job that is killed while it compresses them
        return os.path.exists(os.path.join(job_dir, "done"))

    def deploy(self, gridpack_path: str) -> None:
        program = os.path.join(gridpack_path, "bin", "generate_events")
        threads = self.thread_count("deploy", self.cluster_type)
        self.cancel_earlier_jobs("deploy")
        jobs = []
        for job in self.state["deploy_jobs"]:
            job_dir = self.job_dir("deploy", job["index"])
            if self.deploy_job_done(job_dir):
                continue
            shutil.rmtree(job_dir, ignore_errors=True)
            os.makedirs(job_dir)
            arguments = [
                "--seed", str(job["seed"]), "--events", str(job["events"]),
                "--output_dir", job_dir, "--done_file", os.path.join(job_dir, "done"),
                "--verbosity", "log", "--temp_output_dir", "auto",
                "--output_format", self.card["run"]["output_format"],
                "--cpu_thread_pool_size", str(threads),
                "--device", *self.devices["deploy"],
            ]
            jobs.append((job_dir, arguments, ["done"], program))
        total = len(self.state["deploy_jobs"])
        logger.info("generation pass: %d job(s) for %d events%s", total,
                    self.card["generation"]["events"],
                    "" if len(jobs) == total else
                    f", {total - len(jobs)} of them done already")
        if jobs:
            self.run_jobs("deploy", jobs)
        failed = [j["index"] for j in self.state["deploy_jobs"]
                  if not self.deploy_job_done(self.job_dir("deploy", j["index"]))]
        if failed:
            raise ClusterRunError(
                f"the generation jobs {failed} did not finish, see the log files in "
                f"{os.path.join(self.cluster_path, 'deploy')}")

    def combine(self) -> dict:
        job_dirs = [self.job_dir("deploy", job["index"])
                    for job in self.state["deploy_jobs"]]
        try:
            info = combine_cluster.combine_runs(job_dirs, self.run_path)
        except combine_cluster.CombineError as error:
            raise ClusterRunError(f"combining the events failed: {error}") from error
        return info

    def clean_deploy_data(self) -> None:
        """Drop the events of the generation jobs once the run is done: a rerun
        of an interrupted combine stage still needs them."""
        if self.section["keep_job_files"]:
            return
        for job in self.state["deploy_jobs"]:
            job_dir = self.job_dir("deploy", job["index"])
            for path in glob.glob(os.path.join(job_dir, "events*")) \
                    + glob.glob(os.path.join(job_dir, "weights*")):
                os.remove(path)

    # ------------------------------------------------------------------
    def execute(self) -> None:
        self.start()
        stage = self.state["stage"]
        self.make_clusters()
        self.build_matrix_elements()
        gridpack_path = os.path.join(self.cluster_path, "gridpack")
        # a stage that was left is not trusted: its jobs are checked again, and
        # only the ones without their done file are repeated
        if stage == "train":
            self.train()
            self.set_stage("merge")
            stage = "merge"
        if stage == "merge":
            gridpack_path = self.merge()
            if self.section["mode"] == "gridpack_only":
                self.finish_gridpack_only(gridpack_path)
                self.clean_training_data()
                return
            self.set_stage("deploy")
            # only after the stage moved on: a rerun no longer needs the inputs
            self.clean_training_data()
        self.deploy(gridpack_path)
        self.set_stage("combine")
        info = self.combine()
        launch.run_lhe_postprocessing(self.process)
        self.set_stage("done")
        self.clean_deploy_data()
        process = info["process"]
        logger.info(
            "cluster run done: %d events, cross section %.6g +- %.2g pb, in %s",
            process["count_unweighted"], process["mean"], process["error"], self.run_path)

    def finish_gridpack_only(self, gridpack_path: str) -> None:
        final_path = os.path.join(self.run_path, "gridpack")
        if os.path.exists(final_path) or os.path.exists(final_path + ".tar.gz"):
            shutil.rmtree(final_path, ignore_errors=True)
            if os.path.exists(final_path + ".tar.gz"):
                os.remove(final_path + ".tar.gz")
        shutil.move(gridpack_path, final_path)
        launch.MadgraphProcess.compress_gridpack(final_path, self.card)
        self.set_stage("done")
        logger.info("cluster run done: the merged gridpack is in %s", final_path)


def check_supported(switch, reweight) -> None:
    """The cluster mode only generates the events (and the post-processing of
    the run card): the tools run on them afterwards."""
    tools = launch.selected_tools(switch or {})
    if tools:
        raise ClusterRunError(
            "cluster mode does not run " + ", ".join(tools) + " yet: switch "
            "them off and run them on the events of the cluster run")
    if reweight is not None:
        raise ClusterRunError(
            "cluster mode does not reweight the events yet: switch the "
            "reweighting off and reweight the events of the cluster run")


def run_cluster(switch=None, reweight=None) -> None:
    """Run the generation of this process directory on a batch system."""
    check_supported(switch, reweight)
    launch.compute_auto_widths()
    process = launch.MadgraphProcess.for_steering()
    if process.run_card["cluster"]["mode"] == "full" and \
            process.run_card["generation"]["events"] <= 0:
        raise ClusterRunError("[generation] events must be positive in cluster mode")
    options = launch.load_mg5_options(with_cluster=True)
    ClusterRun(process, options).execute()
