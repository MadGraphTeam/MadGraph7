"""Portable numerical report validation; synthetic records, no native jobs."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path



HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("campaign_report_numerical", HERE / "campaign_report_numerical.py")
COLLECTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COLLECTOR)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = data if isinstance(data, bytes) else data.encode()
    path.write_bytes(raw)
    return sha(raw)


def dump(path, value):
    return write(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


class Evidence:
    def __init__(self, root, frozen=None):
        self.root, self.files, self.frozen = root, {}, frozen

    def path(self, name):
        path = self.root
        for part in Path(COLLECTOR._relative(name)).parts:
            path /= part
            if path.is_symlink():
                raise ValueError("Evidence symlink")
        return path

    def exists(self, name):
        if self.frozen is not None:
            return name in self.frozen or any(key.startswith(name.rstrip("/") + "/") for key in self.frozen)
        return self.path(name).exists()

    def add(self, name, expected=None):
        if self.frozen is not None and name not in self.frozen:
            raise ValueError("Unfrozen report input")
        raw = self.path(name).read_bytes()
        if expected is not None and sha(raw) != expected:
            raise ValueError("Evidence hash mismatch")
        self.files[name] = {"sha256": sha(raw), "bytes": len(raw)}
        return raw

    def json(self, name, expected=None):
        return json.loads(self.add(name, expected), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))

    def tree(self, name):
        names = (sorted(key for key in self.frozen if key.startswith(name + "/")) if self.frozen is not None
                 else [path.relative_to(self.root).as_posix() for path in sorted(self.path(name).rglob("*"))
                       if path.is_file() or path.is_symlink()])
        for key in names:
            self.add(key)


class Fixture:
    def __init__(self, root):
        self.root = root
        self.relative = "campaigns/final"
        self.campaign = root / self.relative
        self.config = json.loads((HERE / "fixture-cases.json").read_text())
        self.baseline = {"revision": COLLECTOR.BASELINE, "archive_sha256": "1" * 64,
                         "files": {"source.f": "2" * 64}}
        shared = {"cases.json": dump(self.campaign / "cases.json", self.config),
                  "inputs/baseline.json": dump(self.campaign / "inputs/baseline.json", self.baseline)}
        for case in self.config["cases"]:
            n = case["nfinal"] + 2
            raw = ("100\n" + "1 0 0 0\n" * (100 * n)).encode()
            name = "points/" + case["id"]
            shared[name + ".dat"] = write(self.campaign / (name + ".dat"), raw)
            metadata = {"inputs": {"count": 100, "incoming_masses": [0., 0.], "masses": [0.] * case["nfinal"]},
                        "points": [{"momenta": [[1., 0., 0., 0.]] * n}] * 100, "sha256": sha(raw)}
            shared[name + ".json"] = dump(self.campaign / (name + ".json"), metadata)
        source = b"retained exporter source\n"
        identity = {"revision": "candidate-commit", "files": {"source.py": sha(source)}, "dirty": False}
        identity["files_sha256"] = COLLECTOR._canonical(identity["files"])
        self.native = {"lib/libmadloop_tirust.so": "3" * 64, "lib/libtirust.a": "4" * 64,
                       "lib/tirust-config.json": "5" * 64}
        self.manifest = {"schema": "madloop-tirust-campaign-v1", "id": "final",
                         "baseline": {"revision": COLLECTOR.BASELINE, "record_sha256": shared["inputs/baseline.json"]},
                         "candidate": {"exporter_identity": identity, "source_snapshot": "inputs/numerical-sources",
                                       "numerical_sources": {"source.py": {"sha256": sha(source)}},
                                       "native": {"files": self.native}},
                         "shared_inputs": {"files": shared, "policy_sha256": COLLECTOR._canonical(self.config)},
                         "export_environment_binding_version": 1, "export_environment": {"PYTHONHASHSEED": "0"},
                         "administrative_worker_timeout_seconds": {case["id"]: 3600. for case in self.config["cases"]}}
        write(self.campaign / "inputs/numerical-sources/source.py", source)
        dump(self.campaign / ("inputs/source-identities/" + COLLECTOR._canonical(identity) + ".json"), identity)
        manifest_hash = dump(self.campaign / "manifest.json", self.manifest)
        self.binding = {"id": "final", "manifest_sha256": manifest_hash}
        self.protocols = {name: (HERE / ("protocol-" + name)).read_bytes() for name in COLLECTOR.PROTOCOLS}
        self.compare = COLLECTOR._parsers(self.protocols, "fixture")[1]

    def folder(self, case=None):
        case = case or self.config["cases"][0]
        return self.campaign / ("experiments/numerical/full100-warmup-" + case["id"] + "-1")

    def complete_case(self, case=None, offset=0.):
        case = case or self.config["cases"][0]
        folder = self.folder(case)
        for name, raw in self.protocols.items():
            write(folder / name, raw)
        config_hash = dump(folder / "cases.json", self.config)
        dump(folder / "campaign-manifest.json", self.manifest)
        dump(folder / "specification.json", {"case": case, "campaign_binding": self.binding,
             "performance_eligible": False, "measurement_phase": "steady-warmup", "worker_timeout_seconds": 3600.,
             "configuration_sha256": config_hash, "protocols": {name: sha(raw) for name, raw in self.protocols.items()}})
        published, reference = [], None
        for source, backend in COLLECTOR.LANES:
            lane = folder / f"{source}-{backend}"
            card_hash = write(self.campaign / f"inputs/{source}/{case['id']}/param_card.dat", "physical card\n")
            command_hash = write(self.campaign / f"commands/{source}-{case['id']}.mg5", "generate " + case["process"] + "\n")
            identity = self.manifest["candidate"]["exporter_identity"] if source == "candidate" else {
                "revision": COLLECTOR.BASELINE, "archive_sha256": self.baseline["archive_sha256"],
                "files_sha256": COLLECTOR._canonical(self.baseline["files"]), "dirty": False}
            generation = {"campaign_binding": self.binding, "case": case, "source": source,
                          "source_identity": identity, "parameter_card_sha256": card_hash,
                          "command_sha256": command_hash, "export_environment_binding_version": 1,
                          "export_environment": {"PYTHONHASHSEED": "0"}}
            params = {"prefix": "TEST_", "nexternal": case["nfinal"] + 2, "nborn": 1}
            rendered = self.protocols["tirust_benchmark_worker.f"].decode()
            for key, token in (("prefix", "PREFIX"), ("nexternal", "NEXTERNAL"), ("nborn", "NBORN")):
                rendered = rendered.replace("@" + token + "@", str(params[key]))
            runtime = {"libraries": {"/external/libmadloop_tirust.so": self.native["lib/libmadloop_tirust.so"]},
                       "archives": {"/external/libtirust.a": self.native["lib/libtirust.a"]},
                       "metadata": {"/external/tirust-config.json": self.native["lib/tirust-config.json"]}}
            recipe = {"compiler": [{"sha256": "6" * 64, "version": "GNU Fortran test"}], "flags": ["-O2"]}
            flags = {"schema": "madloop-effective-compiler-flags-v1", "environment": {},
                     "scopes": {name: {"recipes": {"file.f": recipe}} for name in ("process", "MODEL", "DHELAS")}}
            compilation = {"campaign_binding": self.binding, "masses": [0.] * (case["nfinal"] + 2),
                           "numerical_build_from_sources": True, "full_numerical_rebuild_this_invocation": True,
                           "worker_render_parameters": params, "driver_sha256": sha(rendered.encode()),
                           "worker_template_sha256": sha(self.protocols["tirust_benchmark_worker.f"]),
                           "sources": {"SubProcesses/P0_test/check_sa.f": sha(rendered.encode()),
                                       "Source/make_opts": sha(b"-O2\n")}, "settings": {"Source/make_opts": "-O2\n"},
                           "compiler": "GNU Fortran test", "effective_compiler_flags": flags,
                           "executable_sha256": "7" * 64,
                           "link_inputs_before": {key: runtime[key] for key in ("archives", "metadata")},
                           "link_inputs_after": {key: runtime[key] for key in ("archives", "metadata")}}
            for name, value in (("generation.json", generation), ("compilation.json", compilation)):
                dump(self.campaign / f"inputs/{source}/{case['id']}" / name, value)
            settings = {name.lower(): value for name, value in self.config["madloop_parameters"].items()}
            if source == "candidate":
                settings.update({name.lower(): value for name, value in self.config["candidate_parameters"].items()})
            settings["mlreductionlib"] = str(backend)
            point = "points/" + case["id"]
            inputs = {"campaign_binding": self.binding, "generation": generation, "compilation": compilation,
                      "settings": settings, "param_card_sha256": card_hash, "runtime": runtime,
                      "points_sha256": self.manifest["shared_inputs"]["files"][point + ".dat"],
                      "points_metadata_sha256": self.manifest["shared_inputs"]["files"][point + ".json"],
                      "worker_timeout_seconds": 3600., "numerical_cards": {"Cards/MadLoopParams.dat": "8" * 64}}
            dump(lane / (source + "-inputs.json"), inputs)
            dump(lane / "inputs-after.json", {"runtime": runtime, "executable_sha256": compilation["executable_sha256"],
                 "points_sha256": inputs["points_sha256"], "numerical_cards": inputs["numerical_cards"]})
            code = 210 + backend
            delta = offset if backend == 9 else 0.
            raw = "# retained warmup\n"
            rows = []
            for point_id in range(1, 101):
                for order in (0, 1):
                    numbers = [1., 2. + delta, -3., -4., 1e-10]
                    raw += f"{point_id} {point_id} {order} {code} " + " ".join(map(repr, numbers)) + "\n"
                    rows.append({"point": point_id, "order": order, "status": code,
                                 "coefficients": numbers[:4], "accuracy": numbers[4],
                                 "actual_backend": backend, "precision": "DP"})
            write(lane / "raw/warmup.dat", raw)
            write(lane / "raw/dimensions.dat", "1 1 1\n")
            dump(lane / "raw/ready.json", {"points": 100, "ndim": 1, "nso": 1, "nborn": 1, "cold_seconds": .5})
            observation = {"cold_seconds": .5, "seconds": None, "seconds_per_point": None,
                "rows": rows, "status_counts": {str(code): 100}, "timed_status_counts": None,
                "backend_precision_counts": {f"{backend}:DP": 100}, "native_dp_only": True,
                "answer_dimensions": {"allocated": 1, "loop_orders": 1, "born_orders": 1},
                "measurement_phase": "steady-warmup", "performance_eligible": False,
                "warmup_first_call": 1, "warmup_last_call": 100,
                "warmup_call_point_order": [{"call": p, "point": p} for p in range(1, 101)],
                "raw_outputs": {"warmup.dat": raw, "dimensions.dat": "1 1 1\n"}}
            dump(lane / "raw/observation.json", observation)
            if source == "baseline":
                reference = observation
            result = {"source": source, "backend": backend, "status": "complete", "performance_eligible": False,
                      "observation": observation, "comparison": self.compare(reference["rows"], rows)}
            dump(lane / "result.json", result)
            published.append(result)
        dump(folder / "results.json", {"case": case, "performance_eligible": False, "results": published})

    def collect(self, evidence=None):
        return COLLECTOR.collect_numerical(evidence or Evidence(self.root), self.relative, self.manifest, self.config)


