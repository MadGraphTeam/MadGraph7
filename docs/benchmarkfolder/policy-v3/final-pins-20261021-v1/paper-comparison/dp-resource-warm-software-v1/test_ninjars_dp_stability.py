"""Source/reader controls; no compiler, numerical library, or benchmark engine."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("dp_stability",ROOT/"tools/ninjars_dp_stability.py")
dp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dp)


def interface(amplitude=False):
    result = "AMPL(1,NCTAMPS+LOOPNUM)" if amplitude else "LOOPRES(1,SQUAREDSOINDEX,LOOPNUM)"
    stable = "S(1,LOOPNUM)" if amplitude else "S(SQUAREDSOINDEX,LOOPNUM)"
    text = "      SUBROUTINE ML5_0_LOOP_4(A)\n"
    text += "      CALL ML5_0_CHOOSE_LOOPLIB(A)\n"
    for name in ["CTLOOP","NINJA_LOOP","MP_NINJA_LOOP","NINJARS_LOOP","MP_NINJARS_LOOP","TIRLOOP"]:
        args = "I_SO,LOOPNUM,I_LIB,NLOOPLINE,PL,M2L,RANK" if name=="TIRLOOP" else "NLOOPLINE,PL,M2L,RANK"
        text += f"      CALL ML5_0_{name}({args},\n     $ {result},{stable})\n"
    return text + "      END\n"


def observation(tool=8, value=0., rejected=0, fpdp=0, qp=0, transform=1):
    counts = [1,rejected,0,qp]+[int(i==tool) for i in range(1,9)]+[0]*6
    return {"point":1,"transform":transform,"complete":True,
            "metadata":[2 if transform==2 else 1,118,fpdp,0,1,*counts],
            "rows":{0:{"values":[1.,value,0.,0.,-1.],"raw":[]}}}


def raw_text(tool=8, n=1, values=None, rejected=0):
    lines=[f"DPSD_HEADER 1 {n} 1 1 -1 {tool} 1","WARM 20 1.0"]
    values=values or [0.,0.,0.]
    for point in range(1,n+1):
        for trans,value in enumerate(values,1):
            counts=[1,rejected,0,0]+[int(i==tool) for i in range(1,9)]+[0]*6
            lines.extend([f"BEGIN {point} {trans}",f"OBS {point} {trans} {2 if trans==2 else 1} 118 0 0 1 "+" ".join(map(str,counts))])
            for order in (0,1):
                lines.append(f"ROW {point} {trans} {order} 1.0 {value} 0.0 0.0 -1.0")
            lines.append(f"ENDOBS {point} {trans}")
    return "\n".join(lines+[f"DONE {3*n} 2.0"])+"\n"


class ObserverSources(unittest.TestCase):
    def test_born_and_loop_induced_restore_original_bytes(self):
        for amplitude in (False,True):
            original=interface(amplitude)
            transformed,receipt=dp.instrument_interface(original)
            self.assertEqual(dp.strip_observer(transformed),original)
            self.assertEqual(len(receipt["dispatches"]),6)
            self.assertEqual(transformed.count("CALL ML5_0_DPSTAB_CAPTURE"),6)

    def test_observer_after_call_before_next_overwrite(self):
        transformed,_=dp.instrument_interface(interface())
        self.assertLess(transformed.index("CALL ML5_0_CTLOOP"),transformed.index("CALL ML5_0_DPSTAB_CAPTURE"))
        self.assertLess(transformed.index("CALL ML5_0_DPSTAB_CAPTURE"),transformed.index("CALL ML5_0_NINJA_LOOP"))

    def test_idempotence_and_unknown_dispatch_fail_closed(self):
        transformed,_=dp.instrument_interface(interface())
        for text in (transformed,interface().replace("CTLOOP(","FUTURELOOP("),"      END\n"):
            with self.assertRaises(ValueError):dp.instrument_interface(text)

    def test_mismatched_result_stable_indices_rejected(self):
        with self.assertRaises(ValueError):
            dp.instrument_interface(interface().replace("S(SQUAREDSOINDEX,LOOPNUM)","S(1,LOOPNUM)"))

    def test_signature_and_orphan_continuation_rejected(self):
        with self.assertRaises(ValueError):dp.instrument_interface(interface().replace("NLOOPLINE,PL,M2L,RANK,","NLOOPLINE,PL,RANK,"))
        with self.assertRaises(ValueError):list(dp.fixed_statements("     $ ORPHAN\n"))

    def test_driver_uses_explicit_dp_and_preserves_complete_coefficients(self):
        text=dp.render_driver("ML5_0_",6,3)
        self.assertNotIn("@",text)
        self.assertNotIn("SLOOPMATRIX_THRES",text)
        self.assertIn("CTMODERUN=MODE",text)
        self.assertIn("NRS_RETRY_MODE=.FALSE.",text)
        self.assertIn("ANS(:,J),INTERNAL_ACC(J)",text)
        self.assertIn("POINTS(2,I,K)",text)
        with self.assertRaises(ValueError):dp.render_driver("x();",6,1)

    def test_compiled_control_shares_exact_observer_and_is_pending(self):
        text=dp.recorder_control_source()
        helper=dp.render_driver("DPT_",3,1).split("C     Observer only:",1)[1]
        self.assertTrue(text.endswith("C     Observer only:"+helper))
        self.assertIn("STABLE=.FALSE.",text)
        self.assertIn("IF(COUNTS(2).NE.1) STOP 1",text)
        self.assertIn("IF(STABLE.OR.ANY(RES.NE.SAVED)) STOP 2",text)
        self.assertIn("NRS_RETRY_MODE=.TRUE.",text)


class Classification(unittest.TestCase):
    def test_failed_three_zero_results_never_stable_for_all_four_tools(self):
        for tool in dp.TOOLS:
            obs={i:observation(tool,rejected=1,transform=i) for i in range(1,4)}
            score=dp.estimator(obs,tool,(1,2,3))
            self.assertEqual(score["classification"],"reduction_rejected")
            self.assertTrue(score["unbounded"])
            self.assertIsNone(score["delta"])
            self.assertEqual(dp.tail([score],[1e-3])[0]["k"],1)

    def test_true_zero_results_preserved_and_expected_untested_accuracy_accepted(self):
        obs={i:observation(transform=i) for i in range(1,4)}
        score=dp.estimator(obs,8,(1,2,3))
        self.assertEqual(score["classification"],"all_zero")
        self.assertEqual(score["delta"],0.)
        self.assertEqual(dp.tail([score],[1e-16])[0]["k"],0)

    def test_zero_mean_nonzero_and_large_finite_values(self):
        obs={1:observation(value=1e308),2:observation(value=-1e308,transform=2)}
        self.assertTrue(dp.estimator(obs,8,(1,2))["unbounded"])
        obs[2]["rows"][0]["values"][1]=1e308*(1-1e-12)
        self.assertTrue(math.isfinite(dp.estimator(obs,8,(1,2))["delta"]))

    def test_fpe_and_nonfinite_and_higher_precision_never_hidden(self):
        controls=[("reduction_fpe",lambda x:x["metadata"].__setitem__(2,1)),
                  ("nonfinite_result",lambda x:x["rows"][0]["values"].__setitem__(1,None)),
                  ("unexpected_higher_precision",lambda x:x["metadata"].__setitem__(8,1)),
                  ("changed_rounding_mode",lambda x:x["metadata"].__setitem__(4,0)),
                  ("unexpected_compensated_dp",lambda x:x["metadata"].__setitem__(19,1)),
                  ("reduction_fpe",lambda x:x["metadata"].__setitem__(17,1))]
        for reason,mutate in controls:
            x=observation();mutate(x)
            self.assertEqual(dp.observation_failure(x,8),reason)

    def test_hardware_flags_retained_separately_from_numerical_rejection(self):
        x=observation()
        x["metadata"][-3:]=[1,1,1]
        self.assertIsNone(dp.observation_failure(x,8))

    def test_other_backend_and_initialization_rejected(self):
        self.assertEqual(dp.observation_failure(observation(tool=6),8),"unexpected_backend")
        x=observation();x["metadata"][1]=138
        self.assertEqual(dp.observation_failure(x,8),"initialization_or_invalid_status")

    def test_direction_and_rotation_use_distinct_values_and_failure_scope(self):
        obs={i:observation(value=float(i),transform=i) for i in range(1,4)}
        direction=dp.estimator(obs,8,(1,2));rotation=dp.estimator(obs,8,(1,3))
        self.assertNotEqual(direction["delta"],rotation["delta"])
        obs[3]["complete"]=False
        self.assertFalse(dp.estimator(obs,8,(1,2))["unbounded"])
        self.assertTrue(dp.estimator(obs,8,(1,3))["unbounded"])

    def test_binomial_resolution_no_pseudocount_and_strict_threshold(self):
        scores=[{"delta":0.,"unbounded":False}]*100
        row=dp.tail(scores,[1e-3])[0]
        self.assertEqual((row["k"],row["N"],row["fraction"]),(0,100,0.))
        self.assertAlmostEqual(row["zero_event_upper95"],1-.05**.01)
        scores[0]={"delta":1e-3,"unbounded":False}
        self.assertEqual(dp.tail(scores,[1e-3])[0]["k"],0)
        scores[0]={"delta":None,"unbounded":True}
        self.assertEqual(dp.tail(scores,[1e-3])[0]["fraction"],.01)


class RawReader(unittest.TestCase):
    def parse(self,text):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"raw.dat";path.write_text(text)
            return dp.parse_raw(path,1,8,1)

    def test_complete_raw_retains_all_physical_orders_and_zeros(self):
        parsed=self.parse(raw_text())
        self.assertTrue(parsed["complete"])
        self.assertEqual(len(parsed["observations"]),3)
        self.assertEqual(parsed["observations"][(1,1)]["rows"][1]["values"], [1.,0.,0.,0.,-1.])

    def test_truncation_retains_missing_slots_as_failures(self):
        parsed=self.parse(raw_text().split("BEGIN 1 2")[0])
        self.assertFalse(parsed["complete"])
        obs={i:parsed["observations"].get((1,i)) for i in range(1,4)}
        self.assertEqual(dp.estimator(obs,8,(1,2,3))["classification"],"missing_or_incomplete_output")

    def test_duplicate_wrong_header_nonfinite_and_missing_row(self):
        text=raw_text()
        with self.assertRaises(ValueError):self.parse(text.replace("DPSD_HEADER 1 1 1 1 -1 8 1","DPSD_HEADER 1 1 1 1 -1 6 1"))
        with self.assertRaises(ValueError):self.parse(text.replace("ENDOBS 1 1","ROW 1 1 0 1 0 0 0 -1\nENDOBS 1 1"))
        parsed=self.parse(text.replace("ROW 1 1 0 1.0 0.0","ROW 1 1 0 1.0 NaN"))
        self.assertEqual(dp.observation_failure(parsed["observations"][(1,1)],8),"nonfinite_result")
        parsed=self.parse(text.replace("ROW 1 1 0 1.0 0.0 0.0 0.0 -1.0\n",""))
        self.assertFalse(parsed["complete"])

    def test_unbound_or_boolean_manifest_never_promotes_a_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"manifest.json"
            dp.dump(path,{"schema":dp.RAW_SCHEMA,"passed":True,"runs":[]})
            with self.assertRaises(ValueError):dp.report_snapshot(path)

    def test_report_revalidates_raw_sources_and_keeps_matching_four_tool_denominators(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)
            def artifact(name,text,binary=False):
                f=p/name
                f.write_bytes(text) if binary else f.write_text(text)
                return {"path":str(f),"sha256":dp.digest(f)}
            original=interface();instrumented,receipt=dp.instrument_interface(original)
            artifacts={"original_interface":artifact("original.f",original),
                       "instrumented_interface":artifact("observed.f",instrumented),
                       "driver":artifact("driver.f",dp.render_driver("ML5_0_",3,1)),
                       "points":artifact("points.dat","1\n"+"0 0 0 0\n"*3),
                       "parameter_card":artifact("param.dat","software fixture\n"),
                       "madloop_card":artifact("ml.dat","software fixture\n"),
                       "executable":artifact("exe",b"\x7fELF software fixture",True),
                       "stdout":artifact("out.txt",""),"stderr":artifact("err.txt","")}
            build={key:artifacts[name]["sha256"] for key,name in [
                ("driver_sha256","driver"),("instrumented_interface_sha256","instrumented_interface"),("executable_sha256","executable")]}
            artifacts["build"]=artifact("build.json",json.dumps(build))
            runs=[]
            for tool in dp.TOOLS:
                a=dict(artifacts)
                a["execution"]=artifact(f"execute-{tool}.json",json.dumps({
                    "requested_tool":tool,"helicity":-1,"exit_code":0,"cpu":4,
                    "thread_environment":{x:"1" for x in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","RAYON_NUM_THREADS","MKL_NUM_THREADS"]}}))
                runs.append({"case":"software","tool":tool,"N":1,"nso":1,"nexternal":3,"helicity":-1,
                    "artifacts":a,"raw":artifact(f"raw-{tool}.dat",raw_text(tool,rejected=int(tool==7)))})
            manifest={"schema":dp.RAW_SCHEMA,"stream":"forced_dp_self_consistency","rescue_enabled":False,
                "bindings":[artifacts["driver"]],"instrumentation":"per-dispatch-observer-v1",
                "cohort":{x:"software-fixture" for x in ["numerical_revision","generation_revision","generator_identity_sha256","runtime_identity_sha256","archive_sha256"]},
                "thresholds":[1e-9,1e-3],"runs":runs}
            path=p/"manifest.json";dp.dump(path,manifest)
            snap=dp.report_snapshot(path)
            self.assertTrue(snap["observations_complete"])
            self.assertFalse(snap["complete"])
            self.assertFalse(snap["physical_qualified"])
            with self.assertRaisesRegex(ValueError,"bound explicit cohort"):
                dp.report_snapshot(path,expected={"cohort":manifest["cohort"]})
            self.assertEqual(snap["cases"]["software"]["tools"]["7"]["thresholds"][0]["k"],1)
            self.assertEqual(snap["cases"]["software"]["tools"]["8"]["thresholds"][0]["k"],0)
            self.assertEqual(snap["requested_tools"],[1,6,7,8])
            dp.dump(p/"summary.json",snap)
            self.assertEqual(dp.report_snapshot(p/"summary.json"),snap)
            Path(runs[0]["raw"]["path"]).write_text("tampered\n")
            with self.assertRaises(ValueError):dp.report_snapshot(p/"summary.json")
            manifest["requested_tools"]=[6,8]
            manifest["runs"]=[run for run in runs if run["tool"] in (6,8)]
            dp.dump(path,manifest);pair=dp.report_snapshot(path)
            self.assertTrue(pair["observations_complete"])
            self.assertEqual(pair["requested_tools"],[6,8])
            self.assertEqual(pair["cases"]["software"]["requested_tools"],[6,8])
            self.assertEqual(set(pair["cases"]["software"]["tools"]),{"6","8"})
            manifest["runs"]=[manifest["runs"][0]];dp.dump(path,manifest)
            self.assertFalse(dp.report_snapshot(path)["observations_complete"])
            manifest["runs"].append(runs[0]);dp.dump(path,manifest)
            with self.assertRaisesRegex(ValueError,"Invalid or duplicate DP run"):dp.report_snapshot(path)
            manifest["runs"]=[runs[1]];manifest["requested_tools"]=[8,6];dp.dump(path,manifest)
            with self.assertRaisesRegex(ValueError,"requested tools"):dp.report_snapshot(path)
            manifest["requested_tools"]=[True,6,7,8];dp.dump(path,manifest)
            with self.assertRaisesRegex(ValueError,"requested tools"):dp.report_snapshot(path)
            manifest["requested_tools"]=[6.0,8.0];dp.dump(path,manifest)
            with self.assertRaisesRegex(ValueError,"requested tools"):dp.report_snapshot(path)

    def test_rescued_manifest_cannot_be_relabelled_dp(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"manifest.json"
            dp.dump(path,{"schema":dp.RAW_SCHEMA,"stream":"normal_controller","rescue_enabled":True})
            with self.assertRaises(ValueError):dp.report_snapshot(path)

    def test_empty_runs_cannot_be_reported_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);bound=p/"bound.txt";bound.write_text("bound artifact\n")
            path=p/"manifest.json"
            dp.dump(path,{"schema":dp.RAW_SCHEMA,"stream":"forced_dp_self_consistency",
                "rescue_enabled":False,"bindings":[{"path":str(bound),"sha256":dp.digest(bound)}],
                "instrumentation":"per-dispatch-observer-v1","cohort":{key:"fixture" for key in [
                "numerical_revision","generation_revision","generator_identity_sha256","runtime_identity_sha256","archive_sha256"]},"runs":[]})
            with self.assertRaisesRegex(ValueError,"no observed runs"):dp.report_snapshot(path)


class SemanticQualification(unittest.TestCase):
    """Small filesystem controls; mocked external authorities are not physics."""
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        self.patch=mock.patch.object(dp,"ROOT",self.root);self.patch.start();self.addCleanup(self.patch.stop)
        (self.root/"tools").mkdir()
        (self.root/"tools/ninjars_dp_stability_driver.f").write_bytes((ROOT/"tools/ninjars_dp_stability_driver.f").read_bytes())
        def put(path,value):
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(value if isinstance(value,str) else json.dumps(value))
            return {"path":str(path),"sha256":dp.digest(path)}
        self.put=put
        def manifest(output):
            files={str(f.relative_to(output)):{"sha256":dp.digest(f),"bytes":f.stat().st_size}
                   for name in ("Source","SubProcesses") for f in sorted((output/name).rglob("*"))
                   if f.is_file() and f.suffix in (".f",".inc")}
            return {"files":files,"files_sha256":__import__("hashlib").sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()}
        self.manifest=manifest
        self.defaults={"MLReductionLib":"6|7|1","CTModeRun":-1,"CTModeInit":1,
            "MLStabThres":.001,"COLLIERRequiredAccuracy":1e-8,"COLLIERComputeUVpoles":True,
            "COLLIERComputeIRpoles":True,"COLLIERGlobalCache":-1,"COLLIERUseInternalStabilityTest":True,
            "CTStabThres":1e-2,"UseQPIntegrandForNinjaRS":True}
        target=dict(self.defaults,MLReductionLib="1",CTModeRun=1,MLStabThres=1e-9,
            COLLIERRequiredAccuracy=1e-12,COLLIERGlobalCache=0,COLLIERUseInternalStabilityTest=False)
        origin=self.root/"origin";self.origin=origin
        source=origin/"SubProcesses/P0_fixture";self.source=source
        original=interface();observed,receipt=dp.instrument_interface(original)
        for name,text in [("CT_interface.f",original),("check_sa.f","original driver\n"),
                ("nsquaredSO.inc","      PARAMETER (NSQUAREDSO=1)\n"),("check","original binary")]:put(source/name,text)
        put(origin/"Source/dummy.f","      END\n")
        put(origin/"Cards/MadLoopParams.dat",self.defaults)
        physical=put(origin/"Cards/param_card.dat","physical software fixture\n")
        release={"generation_root":str(self.root/"generation"),"generation_revision":"revision",
            "generator_identity_sha256":"generator","runtime_identity_sha256":"runtime","archive_sha256":"archive"}
        self.release=release
        cohort=put(self.root/"cohort.json",release)
        self.points=self.root/"generation/points/fixture.dat"
        pointrecord=put(self.points,"1\n"+"0 0 0 0\n"*3)
        put(self.root/"generation/status.json",{"cases":{"fixture":{"status":"complete","output":str(origin),
            "case":{"process":"software fixture","nfinal":1},"physical_inputs":{"points":pointrecord}}}})
        retained=self.root/"generation/inputs/candidate/fixture"
        identity={"revision":"revision","tree_state_sha256":"generator"}
        put(retained/"generation.json",{"source_identity":identity,"parameter_card_sha256":physical["sha256"]})
        put(retained/"compilation.json",{})
        put(retained/"compiled-sources.json",manifest(origin))
        put(retained/"param_card.dat","physical software fixture\n")
        libs={}
        for name in ("libmodel.a","libdhelas.a","libcts.a"):
            libs[name]=put(origin/"lib"/name,name)["sha256"]
        runtime={"identity_sha256":"runtime","libraries":{"libninjars.a":"archive"},"metadata":{}}
        self.runtime=runtime
        self.library_maps={origin:dict(libs)}
        self.runtime_maps={}
        self.results={};self.raws={};self.outputs={};self.proof={}
        threads={k:"1" for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","RAYON_NUM_THREADS","MKL_NUM_THREADS")}
        self.proof["start"]=put(self.root/"start.json",{"cohort":cohort,"cpu":[4],"thread_environment":threads})
        for variant,text in (("plain",original),("observed",observed)):
            output=self.root/variant;directory=output/"SubProcesses/P0_fixture";self.outputs[variant]=output
            for relative in ("Source/dummy.f","SubProcesses/P0_fixture/nsquaredSO.inc"):
                put(output/relative,(origin/relative).read_text())
            put(directory/"CT_interface.f",text)
            put(directory/"check_sa.f",dp.render_driver("ML5_0_",3,1))
            executable=directory/"check";executable.write_bytes(b"\x7fELF software-only fixture")
            put(output/"Cards/param_card.dat","physical software fixture\n")
            put(output/"Cards/MadLoopParams.dat",target)
            for name in libs:put(output/"lib"/name,name)
            self.library_maps[output]=dict(libs);self.runtime_maps[output]=runtime
            raw=raw_text(tool=1)
            if variant=="plain":
                raw="\n".join(" ".join(line.split()[:8]+["0"]*15+line.split()[-3:]) if line.startswith("OBS ") else line for line in raw.splitlines())+"\n"
            rawrecord=put(self.root/(variant+".raw"),raw)
            self.raws[variant]=dp.parse_raw(Path(rawrecord["path"]),1,1,1)
            logs={name:put(self.root/(variant+name+".txt"),"") for name in ("stdout","stderr")}
            build={"command":["make","-j1","check"],"cwd":str(directory),"exit_code":0,"reaped":True,**logs}
            execution={"command":[str(executable),str(self.points),"1","-1",rawrecord["path"]],
                "cwd":str(directory),"exit_code":0,"reaped":True,**logs}
            result={"variant":variant,"tool":1,"build":build,"execution":execution,"raw":rawrecord,
                "compiled_manifest":manifest(output),"runtime":runtime,
                "driver_sha256":dp.digest(directory/"check_sa.f"),"interface_sha256":dp.digest(directory/"CT_interface.f"),
                "executable_sha256":dp.digest(executable),"physical_card_sha256":physical["sha256"],
                "madloop_card_sha256":dp.digest(output/"Cards/MadLoopParams.dat")}
            self.results[variant]=result
            self.proof[variant+"_result"]=put(self.root/(variant+".json"),result)
        self.proof["parity_receipt"]=put(self.root/"parity.json",dp.validate_recorder_parity(self.raws["plain"],self.raws["observed"],1))
        output=self.outputs["observed"];directory=output/"SubProcesses/P0_fixture"
        artifacts={"original_interface":{"path":str(source/"CT_interface.f"),"sha256":dp.digest(source/"CT_interface.f")},
            "instrumented_interface":{"path":str(directory/"CT_interface.f"),"sha256":dp.digest(directory/"CT_interface.f")},
            "driver":{"path":str(directory/"check_sa.f"),"sha256":dp.digest(directory/"check_sa.f")},
            "executable":{"path":str(directory/"check"),"sha256":dp.digest(directory/"check")},"points":pointrecord,
            "parameter_card":{"path":str(output/"Cards/param_card.dat"),"sha256":physical["sha256"]},
            "madloop_card":{"path":str(output/"Cards/MadLoopParams.dat"),"sha256":dp.digest(output/"Cards/MadLoopParams.dat")},
            **self.results["observed"]["execution"]}
        artifacts={key:value for key,value in artifacts.items() if isinstance(value,dict)}
        execution={"requested_tool":1,"helicity":-1,"exit_code":0,"cpu":4,"thread_environment":threads,"source_record":self.proof["observed_result"]}
        artifacts["execution"]=put(self.root/"normalized-execution.json",execution)
        artifacts["build"]=put(self.root/"normalized-build.json",dict(source_record=self.proof["observed_result"],
            driver_sha256=artifacts["driver"]["sha256"],instrumented_interface_sha256=artifacts["instrumented_interface"]["sha256"],executable_sha256=artifacts["executable"]["sha256"]))
        self.run={"case":"fixture","process":"software fixture","tool":1,"N":1,"nexternal":3,"nso":1,"helicity":-1,
            "raw":self.results["observed"]["raw"],"qualification":self.proof,"artifacts":artifacts}
        modules={"ninjars_benchmark":SimpleNamespace(generated_source_manifest=manifest),
            "ninjars_benchmark_pair_v3":SimpleNamespace(THREADS=tuple(threads),validate_origin_records=lambda *args:identity,
                actual_runtime=lambda output,*args:self.runtime_maps[output]),
            "ninjars_benchmark_policy_v2":SimpleNamespace(library_hashes=lambda output:self.library_maps[output],
                runtime_metadata_files=lambda output:{},delta=lambda a,b:{k for k in set(a)|set(b) if a.get(k)!=b.get(k)},
                git_bytes=lambda *args:json.dumps(self.defaults).encode()),
            "madgraph.various.banner":SimpleNamespace(MadLoopParam=lambda value:json.loads(value if value.startswith("{") else Path(value).read_text()))}
        self.modules=mock.patch.dict(__import__("sys").modules,modules);self.modules.start();self.addCleanup(self.modules.stop)

    def qualify(self):
        _,paths,_=dp.validate_run_sources(self.run)
        return dp.validate_target_qualification(self.run,self.release,paths)

    def rewrite_result(self,variant):
        self.proof[variant+"_result"]=self.put(self.root/(variant+".json"),self.results[variant])

    def test_actual_target_semantics_are_required_not_a_passed_flag(self):
        result=self.qualify()
        self.assertTrue(result["source_runtime_qualified"])
        self.assertEqual(result["parity"]["physical_coefficients_compared"],24)
        self.assertEqual(result["settings"]["MLStabThres"],1e-9)

    def test_numerical_source_changes_and_restamped_manifests_rejected(self):
        path=self.outputs["observed"]/"Source/dummy.f";path.write_text("      STOP\n      END\n")
        self.results["observed"]["compiled_manifest"]=self.manifest(self.outputs["observed"])
        self.rewrite_result("observed")
        with self.assertRaisesRegex(ValueError,"Unreviewed numerical"):self.qualify()

    def test_changed_native_runtime_and_model_libraries_rejected(self):
        self.runtime_maps[self.outputs["observed"]]={**self.runtime,"identity_sha256":"other"}
        with self.assertRaisesRegex(ValueError,"runtime differs"):self.qualify()
        self.runtime_maps[self.outputs["observed"]]=self.runtime
        self.library_maps[self.outputs["observed"]]["libmodel.a"]="other"
        with self.assertRaisesRegex(ValueError,"model/DHELAS"):self.qualify()

    def test_control_settings_001_are_not_publication_settings(self):
        target=self.outputs["plain"]/"Cards/MadLoopParams.dat"
        card=json.loads(target.read_text());card["MLStabThres"]=.001
        self.put(target,card);self.results["plain"]["madloop_card_sha256"]=dp.digest(target);self.rewrite_result("plain")
        with self.assertRaisesRegex(ValueError,"publication settings"):self.qualify()

    def test_changed_selector_and_grafted_cohort_rejected(self):
        self.results["plain"]["execution"]["command"][2]="8";self.rewrite_result("plain")
        with self.assertRaisesRegex(ValueError,"selector command"):self.qualify()
        self.results["plain"]["execution"]["command"][2]="1";self.rewrite_result("plain")
        with self.assertRaisesRegex(ValueError,"another explicit cohort"):
            dp.validate_target_qualification(self.run,{**self.release,"archive_sha256":"grafted"},dp.validate_run_sources(self.run)[1])

    def test_reserialized_raw_disagreement_cannot_use_retained_parity_boolean(self):
        path=Path(self.results["plain"]["raw"]["path"])
        path.write_text(path.read_text().replace("ROW 1 1 0 1.0 0.0","ROW 1 1 0 1.0 0.1"))
        self.results["plain"]["raw"]["sha256"]=dp.digest(path);self.rewrite_result("plain")
        with self.assertRaisesRegex(ValueError,"physical coefficient"):self.qualify()

    def test_zero_parity_preserves_sign_and_failed_zeros_stay_numerical_failures(self):
        plain=copy.deepcopy(self.raws["plain"]);observed=copy.deepcopy(self.raws["observed"])
        observed["observations"][(1,1)]["rows"][0]["values"][1]=-0.
        with self.assertRaisesRegex(ValueError,"physical coefficient"):dp.validate_recorder_parity(plain,observed,1)
        observed=copy.deepcopy(self.raws["observed"])
        observed["observations"][(1,1)]["metadata"][6]=1
        parity=dp.validate_recorder_parity(plain,observed,1)
        self.assertEqual(parity["numerical_failure_observations"],{"reduction_rejected":1})
        self.assertTrue(dp.estimator({i:observed["observations"][(1,i)] for i in (1,2,3)},1,(1,2,3))["unbounded"])

    def test_nonfinite_parity_is_token_equality_not_finite_bit_accuracy(self):
        plain=copy.deepcopy(self.raws["plain"]);observed=copy.deepcopy(self.raws["observed"])
        for sample in (plain,observed):
            row=sample["observations"][(1,1)]["rows"][0]
            row["values"][1]=None;row["raw"][1]="NaN"
        parity=dp.validate_recorder_parity(plain,observed,1)
        self.assertFalse(parity["f64_bitexact"])
        self.assertEqual(parity["matching_nonfinite_tokens"],1)
        self.assertEqual(parity["numerical_failure_observations"],{"nonfinite_result":1})
        observed["observations"][(1,1)]["rows"][0]["raw"][1]="Inf"
        with self.assertRaisesRegex(ValueError,"nonfinite output token"):dp.validate_recorder_parity(plain,observed,1)


class RandomRunnerQualification(unittest.TestCase):
    """Whole-method prefix/science controls with explicitly mocked authorities."""
    def setUp(self):
        f=SemanticQualification();f.setUp();self.addCleanup(f.doCleanups);self.f=f
        f.points.write_text("4\n"+"0 0 0 0\n"*12)
        parent={"path":str(f.points),"sha256":dp.digest(f.points)}
        prefix=f.root/"prefix.dat";prefix.write_bytes(dp.point_prefix_bytes(f.points,3,3))
        prefix_record={"path":str(prefix),"sha256":dp.digest(prefix)}
        status=json.loads((f.root/"generation/status.json").read_text())
        status["cases"]["fixture"]["physical_inputs"]["points"]=parent
        f.put(f.root/"generation/status.json",status)
        resolved=dp.resolve_origin(f.run,f.release)
        self.resolve_patch=mock.patch.object(dp,"resolve_origin",return_value={**resolved,
            "measurement_authority":{"kind":"random-dp-v1"},"sample_kind":"preregistered-random-SM-DP"})
        self.resolve_patch.start();self.addCleanup(self.resolve_patch.stop)
        guard=f.root/"guard.c";guard.write_text("audited guard software fixture")
        old_digest=dp.digest
        patch=mock.patch.object(dp,"digest",side_effect=lambda p:(
            "13f5702a530ede39a74333d266bb3078799e7b3f9889a29c9c2fdd2683176477" if Path(p)==guard else old_digest(p)))
        patch.start();self.addCleanup(patch.stop)
        obj=f.root/"guard.o";obj.write_bytes(b"\x7fELF software fixture")
        helper=f.root/"checkpoint.o";helper.write_bytes(b"\x7fELF checkpoint fixture")
        def bound(p):return {"path":str(p),"sha256":dp.digest(p)}
        guard_source,guard_object=bound(guard),bound(obj)
        table=f.origin/"SubProcesses/MadLoop5_resources/ML5_0_HelConfigs.dat"
        f.put(table,"-1 -1 0\n-1 1 0\n1 -1 0\n1 1 0\n")
        start=json.loads(Path(f.proof["start"]["path"]).read_text())
        start["guards"]={"6":{"source":guard_source,"object":guard_object}}
        start["cases"]={"fixture":{"points":parent,"parity_points":prefix_record,"N":4,"parity_N":3,
            "helicity":{"argument":-1,"one_based_index":None,"physical_tuple":None,"table":bound(table),"policy":"full-helicity-color-sum"}}}
        f.proof["start"]=f.put(f.root/"start.json",start)
        symbols=("ninjars_context_create","ninjars_evaluate_dp","ninjars_evaluate_dp_compensated","ninjars_evaluate_mp","ninjars_context_destroy")
        flags=str(helper)+" "+str(obj)+" -L$(LIBDIR) -ldhelas -lmodel $(LINK_LOOP_LIBS) $(LDFLAGS) "+" ".join("-Wl,--wrap="+s for s in symbols)
        for variant,result in f.results.items():
            directory=Path(result["execution"]["cwd"]);output=f.outputs[variant]
            f.put(directory/"check_sa.f",dp.render_custody_driver("ML5_0_",3,1))
            card=json.loads((output/"Cards/MadLoopParams.dat").read_text());card["MLReductionLib"]="6"
            f.put(output/"Cards/MadLoopParams.dat",card)
            raw=raw_text(tool=6,n=3)
            warm_counts=[0]*15 if variant=="plain" else [20,0,0,0]+[20*int(i==6) for i in range(1,9)]+[0,0,0]
            warm_record="WARMDIAG "+" ".join(map(str,[20,*warm_counts,0,0,0,0]))+"\n"
            raw=raw.replace("WARM 20 1.0\n","WARM 20 1.0\n"+warm_record)
            if variant=="plain":raw="\n".join(" ".join(line.split()[:8]+["0"]*15+line.split()[-3:]) if line.startswith("OBS ") else line for line in raw.splitlines())+"\n"
            result.update(tool=6,N=3,points=prefix_record,raw=f.put(f.root/(variant+".raw"),raw),
                compiled_manifest=f.manifest(output),driver_sha256=dp.digest(directory/"check_sa.f"),
                madloop_card_sha256=dp.digest(output/"Cards/MadLoopParams.dat"),
                guard_object=guard_object,additional_link_flags=flags,custody={"helper_object":bound(helper)})
            expected_guard="GUARD_FINAL create=0 dp=0 compensated=0 mp=0 destroy=1 nonnull=0"
            result["guard"]=f.put(f.root/(variant+".guard"),expected_guard.replace("GUARD_FINAL","NULL_TEARDOWN_FORWARDED")+"\n"+expected_guard+"\n")
            result["execution"]["command"]=[str(directory/"check"),str(prefix),"6","-1",result["raw"]["path"]]
            result["build"]["command"]=["make","-j1","check","LINKLIBS="+flags]
            f.rewrite_result(variant)
        science_raw=raw_text(tool=6,n=4).replace("WARM 20 1.0\n","WARM 20 1.0\n"+warm_record)
        science=copy.deepcopy(f.results["observed"]);science.update(N=4,points=parent,raw=f.put(f.root/"science.raw",science_raw))
        science["execution"]["command"][1]=str(f.points);science["execution"]["command"][-1]=science["raw"]["path"]
        f.results["science"]=science;f.proof["science_result"]=f.put(f.root/"science.json",science)
        f.proof["parity_receipt"]=f.put(f.root/"parity.json",dp.validate_recorder_parity(
            dp.parse_raw(Path(f.results["plain"]["raw"]["path"]),3,6,1),dp.parse_raw(Path(f.results["observed"]["raw"]["path"]),3,6,1),6))
        f.run.update(tool=6,N=4,raw=science["raw"],resource_protocol="dp-resource-custody-v1",measurement_authority={"kind":"random-dp-v1"})
        artifacts=f.run["artifacts"];artifacts["points"]=parent
        directory=Path(science["execution"]["cwd"])
        for name,path in (("driver",directory/"check_sa.f"),("madloop_card",f.outputs["observed"]/"Cards/MadLoopParams.dat")):
            artifacts[name]=bound(path)
        normalized=json.loads(Path(artifacts["execution"]["path"]).read_text());normalized.update(requested_tool=6,source_record=f.proof["science_result"])
        artifacts["execution"]=f.put(f.root/"normalized-execution.json",normalized)
        artifacts["build"]=f.put(f.root/"normalized-build.json",dict(source_record=f.proof["science_result"],
            driver_sha256=artifacts["driver"]["sha256"],instrumented_interface_sha256=artifacts["instrumented_interface"]["sha256"],executable_sha256=artifacts["executable"]["sha256"]))
        self.resources={"filters":{"initial":{},"postwarm":{},"final":{}},"helper_object":bound(helper)}
        patch=mock.patch.object(dp,"validate_resource_custody",return_value=self.resources)
        patch.start();self.addCleanup(patch.stop)
        patch=mock.patch.object(dp,"validate_warm_records",return_value={"software_only":True})
        patch.start();self.addCleanup(patch.stop)

    def test_prefix_parity_and_full_science_are_distinct_bound_streams(self):
        qualified=self.f.qualify()
        self.assertEqual(qualified["parity_N"],3)
        self.assertEqual(qualified["parity"]["physical_coefficients_compared"],72)
        self.assertTrue(qualified["initialization_safe"])

    def test_failed_science_warmup_is_separate_and_never_qualified_safe(self):
        f=self.f;path=Path(f.results["science"]["raw"]["path"])
        rows=path.read_text().splitlines()
        fields=next(row.split() for row in rows if row.startswith("WARMDIAG "));fields[3]="1"
        path.write_text("\n".join(" ".join(fields) if row.startswith("WARMDIAG ") else row for row in rows)+"\n")
        f.results["science"]["raw"]={"path":str(path),"sha256":dp.digest(path)}
        f.run["raw"]=f.results["science"]["raw"]
        f.proof["science_result"]=f.put(f.root/"science.json",f.results["science"])
        for key in ("execution","build"):
            record=f.run["artifacts"][key];value=json.loads(Path(record["path"]).read_text())
            value["source_record"]=f.proof["science_result"]
            f.run["artifacts"][key]=f.put(Path(record["path"]),value)
        qualified=f.qualify()
        self.assertTrue(qualified["source_runtime_qualified"])
        self.assertFalse(qualified["initialization_safe"])
        self.assertEqual(qualified["warm_diagnostics"]["science"]["failure"],"warm_reduction_rejected")
        self.assertEqual(len(dp.parse_raw(path,4,6,1)["observations"]),12)

    def test_grafted_prefix_and_missing_science_rejected(self):
        f=self.f;start=json.loads(Path(f.proof["start"]["path"]).read_text())
        path=Path(start["cases"]["fixture"]["parity_points"]["path"]);path.write_text(path.read_text().replace("0 0 0 0","1 0 0 0",1))
        start["cases"]["fixture"]["parity_points"]["sha256"]=dp.digest(path);f.proof["start"]=f.put(f.root/"start.json",start)
        with self.assertRaises(ValueError):f.qualify()
        path.write_bytes(dp.point_prefix_bytes(f.points,3,3));start["cases"]["fixture"]["parity_points"]["sha256"]=dp.digest(path)
        f.proof["start"]=f.put(f.root/"start.json",start);del f.proof["science_result"]
        with self.assertRaises(ValueError):f.qualify()

    def test_science_cannot_swap_its_qualified_executable_or_initial_filters(self):
        f=self.f;f.results["science"]["executable_sha256"]="restamped-other-executable"
        f.proof["science_result"]=f.put(f.root/"science.json",f.results["science"])
        with self.assertRaises(ValueError):f.qualify()


class OriginResolution(unittest.TestCase):
    def setUp(self):
        SemanticQualification.setUp(self)
        self.core_path=self.root/"generation/status.json"
        core=json.loads(self.core_path.read_text());core["generator_tree_state"]={"tracked":{"fixture":"hash"},"untracked":{}}
        self.put(self.core_path,core)
        self.core=core

    def controls(self):
        import sys,shutil
        root=self.root/"controls";alias="control-fixture";cid="supplement-fixture"
        row=copy.deepcopy(self.core["cases"]["fixture"]);row["case"]["id"]=alias
        old=self.root/"generation/inputs/candidate/fixture";new=root/"inputs/candidate"/alias
        shutil.copytree(old,new)
        row["generation"]=json.loads((new/"generation.json").read_text())
        row["compilation"]=json.loads((new/"compilation.json").read_text())
        status={"schema":"final-release-control-generation-v1","git_revision":"revision",
            "generator_tree_state":self.core["generator_tree_state"],"generator_tree_state_sha256":"generator",
            "cases":{alias:row},**{k:True for k in ("complete","preserved_existing_inputs","dependencies_unchanged","original_archive_unchanged")}}
        status_record=self.put(root/"status.json",status)
        points=self.put(self.root/"loop-measurements.dat","100\n"+"0 0 0 0\n"*300)
        origins={key:{"path":str(new/name),"sha256":dp.digest(new/name)} for key,name in
            (("generation","generation.json"),("compilation","compilation.json"),("compiled_manifest","compiled-sources.json"),("physical_card","param_card.dat"))}
        origins["reducer_runtime"]={"identity_sha256":"runtime"}
        spec={"candidate_cohort":{"numerical_revision":"revision",**{k:self.release[k] for k in ("generation_revision","generator_identity_sha256","runtime_identity_sha256","archive_sha256")}},"controls_generation":status_record,"cases":{cid:{
            "definition":{**row["case"],"id":cid},"actual_generation_case":alias,"output":str(self.origin),
            "origin":origins,"points":points}}}
        spec_record=self.put(root/"measurement.json",spec)
        sys.modules["ninjars_paper_timing_v3"]=SimpleNamespace(validate_spec=lambda value:value)
        self.addCleanup(sys.modules.pop,"ninjars_paper_timing_v3",None)
        self.release["numerical_revision"]="revision"
        run={**self.run,"case":cid,"N":100,"origin_authority":{"kind":"controls-v3",
            "generation_case":alias,"generation":status_record,"measurement":spec_record}}
        return run,status,spec

    def stress(self):
        import sys,shutil
        root=self.root/"stress";cid="fixture";case={"id":cid,"process":"software fixture","nfinal":1}
        retained=root/"inputs/candidate"/cid;shutil.copytree(self.root/"generation/inputs/candidate/fixture",retained)
        source={"schema":"stress-generation-source-v1","case":case,
            **{key:json.loads((retained/name).read_text()) for key,name in
               (("generation","generation.json"),("compilation","compilation.json"),("compiled_manifest","compiled-sources.json"))},
            "reducer_runtime":{"identity_sha256":"runtime"},
            "physical_card":self.put(root/"card","physical software fixture\n"),
            "cli_card":self.put(root/"command.mg5","generate software fixture\n"),
            "cli_log":self.put(root/"log","export completed\n")}
        proof=self.put(root/"source.json",source)
        status=self.put(root/"status.json",{"status":"complete","amplitude_evaluations":0,"cases":{
            cid:{"output":str(self.origin),"source_proof":proof}}})
        point={"path":str(self.points),"sha256":dp.digest(self.points)}
        config={"points_per_case":1,"cases":[case]}
        current=self.put(root/"ordinary.json",config);original=self.put(root/"original.json",config)
        early=self.put(root/"early.json",{"schema":"stress-early-build-result-v1","status":status,
            "source_proofs":{cid:proof},"point_files":{cid:point}})
        final={"schema":"final-pin-stress-generation-v1","status":"complete","all_registered_children_reaped":True,
            "amplitude_evaluations":0,"initialized_symbolica_workers":0,"early_build_result":early,
            **{key:self.release[key] for key in ("generation_revision","generator_identity_sha256","runtime_identity_sha256","archive_sha256")},
            "fixed_inputs":{"configurations/ordinary":{"current":current,"original":original},
                            "point_files/fixture":{"current":point,"original":point}}}
        authority=self.put(root/"final.json",final)
        policy=sys.modules["ninjars_benchmark_policy_v2"]
        def verified(value,schema):
            if value.get("schema")!=schema:raise ValueError("Invalid software fixture schema")
            return value
        policy.verified=verified
        sys.modules["ninjars_stress_v2"]=SimpleNamespace(configuration=lambda variant:config)
        self.addCleanup(sys.modules.pop,"ninjars_stress_v2",None)
        run={**self.run,"origin_authority":{"kind":"stress-v3","generation_case":cid,"generation":authority}}
        return run,final,source

    def test_core_resolution_preserves_original_authority(self):
        result=dp.resolve_origin(self.run,self.release)
        self.assertEqual(result["origin"],self.origin)
        self.assertEqual(result["generation_case"],"fixture")

    def test_loop_measurement_count_is_separate_from_export_smoke(self):
        run,status,spec=self.controls();result=dp.resolve_origin(run,self.release)
        self.assertEqual(result["generation_case"],"control-fixture")
        self.assertEqual(result["case"]["case"]["id"],"control-fixture")
        self.assertEqual(result["points"].read_text().split()[0],"100")
        self.assertEqual(result["case"]["physical_inputs"]["points"]["path"],str(self.points))

    def test_loop_alias_graft_and_wrong_actual_cohort_fail(self):
        run,status,spec=self.controls();spec["cases"][run["case"]]["actual_generation_case"]="wrong"
        run["origin_authority"]["measurement"]=self.put(self.root/"controls/measurement.json",spec)
        with self.assertRaisesRegex(ValueError,"alias/process/cohort"):dp.resolve_origin(run,self.release)
        spec["cases"][run["case"]]["actual_generation_case"]="control-fixture"
        run["origin_authority"]["measurement"]=self.put(self.root/"controls/measurement.json",spec)
        status["git_revision"]="another-release"
        run["origin_authority"]["generation"]=self.put(self.root/"controls/status.json",status)
        with self.assertRaisesRegex(ValueError,"actual cohort"):dp.resolve_origin(run,self.release)

    def test_loop_empty_cohort_and_short_point_override_fail(self):
        run,status,spec=self.controls();cohort=spec["candidate_cohort"];spec["candidate_cohort"]={}
        run["origin_authority"]["measurement"]=self.put(self.root/"controls/measurement.json",spec)
        with self.assertRaisesRegex(ValueError,"alias/process/cohort"):dp.resolve_origin(run,self.release)
        spec["candidate_cohort"]=cohort
        spec["cases"][run["case"]]["points"]=self.put(self.root/"short.dat","1\n"+"0 0 0 0\n"*3)
        run["origin_authority"]["measurement"]=self.put(self.root/"controls/measurement.json",spec)
        with self.assertRaisesRegex(ValueError,"complete100-point"):dp.resolve_origin(run,self.release)

    def test_stress_resolution_keeps_declared_small_denominator(self):
        run,final,source=self.stress();result=dp.resolve_origin(run,self.release)
        self.assertEqual(result["generation_case"],"fixture")
        self.assertEqual(result["points"].read_text().split()[0],"1")

    def test_stress_changed_fixed_stream_or_export_process_rejected(self):
        run,final,source=self.stress()
        other=self.put(self.root/"stress/wrong.dat","1\n"+"1 1 1 1\n"*3)
        final["fixed_inputs"]["point_files/fixture"]["original"]=other
        run["origin_authority"]["generation"]=self.put(self.root/"stress/final.json",final)
        with self.assertRaisesRegex(ValueError,"original fixed stream"):dp.resolve_origin(run,self.release)

    def test_unknown_origin_authority_does_not_fallback_to_core(self):
        with self.assertRaisesRegex(ValueError,"Unknown DP origin"):
            dp.resolve_origin({**self.run,"origin_authority":{"kind":"arbitrary"}},self.release)


class NativeGuardReader(unittest.TestCase):
    def parse(self,lines):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/"native.log";p.write_text("\n".join(lines)+"\n")
            return dp.parse_native_guard(p)

    def native(self,status=0):
        return ["GUARD_BEGIN create=0 dp=0 compensated=0 mp=0 destroy=0 nonnull=0",
                "CREATE_RETURN status=0 create=1 dp=0 compensated=0 mp=0 destroy=0 nonnull=0",
                f"NATIVE_DP_RETURN status={status} create=1 dp=1 compensated=0 mp=0 destroy=0 nonnull=0",
                "DESTROY_RETURN status=0 create=1 dp=1 compensated=0 mp=0 destroy=1 nonnull=1",
                "GUARD_FINAL create=1 dp=1 compensated=0 mp=0 destroy=1 nonnull=1"]

    def test_native_error_return_is_retained_without_success_relabelling(self):
        result=self.parse(self.native(2))
        self.assertEqual(result["native_per_call_status_counts"],{"2":1})
        self.assertEqual(result["native_error_returns_retained"],1)
        self.assertTrue(result["no_compensated_or_mp_entry"])

    def test_forbidden_precision_missing_boundary_and_counter_drift_fail(self):
        original=self.native()
        for lines in [original[:-1], [*original[:2],original[2].replace("NATIVE_DP_RETURN","FORBIDDEN_MP_ENTRY"),*original[3:]],
                      [line.replace("dp=1","dp=2") for line in original]]:
            with self.assertRaises(ValueError):self.parse(lines)

    def test_lifecycle_errors_and_missing_native_status_fail(self):
        original=self.native()
        for lines in [[original[0],original[1].replace("status=0","status=1"),*original[2:]],
                      [*original[:2],original[2].replace("status=0 ",""),*original[3:]]]:
            with self.assertRaises(ValueError):self.parse(lines)


class RecorderParityPrefix(unittest.TestCase):
    def test_checkpoint_driver_strips_to_exact_numerical_driver(self):
        import re
        for prefix,nx,nso in (("ML5_0_",3,1),("ML5_12_",6,3)):
            original=dp.render_driver(prefix,nx,nso)
            augmented=dp.render_custody_driver(prefix,nx,nso)
            restored=re.sub(r"C DPSTAB_CUSTODY_BEGIN\n.*?C DPSTAB_CUSTODY_END\n","",augmented,flags=re.S)
            self.assertEqual(restored,original)
            self.assertEqual(augmented.count("C DPSTAB_CUSTODY_BEGIN\n"),5)
            self.assertIn("INTEGER(C_INT), VALUE :: PHASE",augmented)
            self.assertIn("CALL "+prefix+"DPSTAB_CUSTODY_CHECKPOINT(1)\n",augmented)
        for prefix in ("1bad_","x"*64,"bad-name"):
            with self.assertRaises(ValueError):dp.render_custody_driver(prefix,3,1)

    def test_exact_serialized_first_events_without_numeric_rounding(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"parent.dat"
            event=b"1.00000000000000001E+00 0.0 -0.0 1.0\n"*3
            path.write_bytes(b"2\n"+event+event)
            self.assertEqual(dp.point_prefix_bytes(path,3,1),b"1\n"+event)
            self.assertEqual(dp.point_prefix_bytes(path,3,2),path.read_bytes())

    def test_short_trailing_nonfinite_and_boolean_dimensions_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"parent.dat"
            event=b"1.0 0.0 0.0 1.0\n"*3
            for data in (b"2\n"+event,b"1\n"+event+b"extra\n",b"1\n"+event.replace(b"1.0",b"NaN",1)):
                path.write_bytes(data)
                with self.assertRaises(ValueError):dp.point_prefix_bytes(path,3,1)
            path.write_bytes(b"1\n"+event)
            for nx,n in ((3,True),(3,1.0),(True,1),(3,2),(2,1)):
                with self.assertRaises(ValueError):dp.point_prefix_bytes(path,nx,n)


class WarmInitialization(unittest.TestCase):
    def raw(self,tool=6,plain=False,rejected=0,flags=None):
        counts=[20,rejected,0,0]+[20*int(i==tool) for i in range(1,9)]+[0,0,0]
        if plain:counts=[0]*15
        record="WARMDIAG "+" ".join(map(str,[20,*counts,*(flags or [0]*4)]))+"\n"
        text=raw_text(tool=tool).replace("WARM 20 1.0\n","WARM 20 1.0\n"+record)
        if plain:text="\n".join(" ".join(line.split()[:8]+["0"]*15+line.split()[-3:]) if line.startswith("OBS ") else line for line in text.splitlines())+"\n"
        return text

    def parse(self,text,tool=6):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"raw";path.write_text(text)
            return dp.parse_raw(path,1,tool,1)

    def test_every_warm_iteration_and_failure_retained_outside_point_scores(self):
        parsed=self.parse(self.raw(rejected=1))
        self.assertEqual(parsed["warm"]["diagnostics"]["iterations"],20)
        self.assertEqual(dp.warm_failure(parsed["warm"],6,required=True),"warm_reduction_rejected")
        obs={t:parsed["observations"][(1,t)] for t in (1,2,3)}
        self.assertEqual(dp.estimator(obs,6,(1,2,3))["classification"],"all_zero")
        self.assertEqual(len(parsed["observations"]),3)

    def test_warm_parity_keeps_rejections_without_success_conditioning(self):
        left=self.parse(self.raw(plain=True));right=self.parse(self.raw(rejected=1))
        receipt=dp.validate_recorder_parity(left,right,6)
        self.assertEqual(receipt["warm_failure"],"warm_reduction_rejected")
        self.assertTrue(receipt["f64_bitexact"])
        right["warm"]["diagnostics"]["common_flags"][0]=1
        with self.assertRaises(ValueError):dp.validate_recorder_parity(left,right,6)

    def test_duplicate_negative_mismatched_and_misplaced_warm_records_rejected(self):
        text=self.raw();record=next(l for l in text.splitlines() if l.startswith("WARMDIAG "))
        for changed in (text.replace(record,record+"\n"+record),text.replace("WARMDIAG 20","WARMDIAG -1"),
                        text.replace("WARMDIAG 20","WARMDIAG 21"),text.replace(record+"\n","")+record+"\n"):
            with self.assertRaises(ValueError):self.parse(changed)

    def test_missing_historical_record_never_claims_new_initializer_safe(self):
        parsed=self.parse(raw_text(tool=6))
        self.assertIsNone(dp.warm_failure(parsed["warm"],6))
        self.assertEqual(dp.warm_failure(parsed["warm"],6,required=True),"missing_warm_diagnostics")
        for tool in (6,8):
            parsed=self.parse(self.raw(tool=tool,flags=[0,1,0,0]),tool)
            self.assertEqual(dp.warm_failure(parsed["warm"],tool,required=True),"warm_unexpected_higher_precision")


class WarmMetadataCustody(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.raw=self.root/"full.raw"
        self.raw.write_text(WarmInitialization().raw())
        lines=self.raw.read_bytes().splitlines(keepends=True);index=next(i for i,line in enumerate(lines) if line.startswith(b"WARMDIAG "))
        self.view=self.root/"view.raw";self.view.write_bytes(b"".join(lines[:index]+lines[index+1:]))
        counts={"reductions":20,"rejected":0,"nonfinite_reductions":0,"qp_reductions":0,
            **{"backend_"+str(i):20*int(i==6) for i in range(1,9)},
            "observer_fpe_dp":0,"observer_fpe_qp":0,"compensated_dp":0}
        self.result={"raw":self.bound(self.raw),"parser_view":self.bound(self.view),
            "warm_diagnostics":{"schema":"dp-warm-diagnostics-v1","iterations":20,"counts":counts,
            "flags":{"fpe_dp":0,"fpe_qp":0,"nrs_retry_mode":0,"nrs_retry_used":0},
            "initialization_safe":True,"unsafe_reasons":[],"score_denominator_contribution":0,
            "limitations":"No warm ANS read; observer counters unavailable in plain variant"},
            "parser_view_derivation":{"schema":"dp-warmdiag-only-parser-view-v1","removed_record":"WARMDIAG",
                "removed_line_number":index+1,"removed_line_sha256":__import__("hashlib").sha256(lines[index]).hexdigest(),
                "original_sha256":dp.digest(self.raw),"parser_view_sha256":dp.digest(self.view)}}
    def bound(self,path):return {"path":str(path),"sha256":dp.digest(path)}
    def validate(self):return dp.validate_warm_records(self.result,dp.parse_raw(self.raw,1,6,1))
    def test_full_raw_drives_retained_warm_metadata_and_exact_parser_view(self):
        self.assertTrue(self.validate()["initialization_safe"])
        self.assertNotIn(b"WARMDIAG",self.view.read_bytes())
        self.assertIn(b"WARMDIAG",self.raw.read_bytes())
    def test_restamped_extra_point_line_removal_is_not_a_warm_only_view(self):
        self.view.write_bytes(self.view.read_bytes().replace(b"BEGIN 1 1\n",b"",1))
        self.result["parser_view"]=self.bound(self.view)
        self.result["parser_view_derivation"]["parser_view_sha256"]=dp.digest(self.view)
        with self.assertRaises(ValueError):self.validate()
    def test_retained_safety_boolean_is_not_authority(self):
        self.result["warm_diagnostics"]["initialization_safe"]=False
        self.result["warm_diagnostics"]["score_denominator_contribution"]=100
        with self.assertRaises(ValueError):self.validate()


class ResourceCustody(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.base=Path(self.tmp.name);self.origin=self.base/"origin";self.output=self.base/"output"
        self.resources=self.output/"SubProcesses/MadLoop5_resources"
        self.original=self.origin/"SubProcesses/MadLoop5_resources"
        self.directory=self.output/"SubProcesses/P0_fixture"
        self.root=self.base/"custody"
        for path in (self.resources,self.original,self.directory,self.root/"initial",self.root/"postwarm",self.root/"final"):
            path.mkdir(parents=True,exist_ok=True)
        (self.directory/"MadLoop5_resources").symlink_to("../MadLoop5_resources")
        self.helper=self.base/"checkpoint.c";self.helper.write_text("software fixture only")
        old_digest=dp.digest
        self.patch=mock.patch.object(dp,"digest",side_effect=lambda p:(
            "766d47e5ad6d092bbd159651f739c838c18829b57d85a1167416fa4c382e9a10"
            if Path(p)==self.helper else old_digest(p)))
        self.patch.start();self.addCleanup(self.patch.stop)
        obj=self.base/"checkpoint.o";obj.write_bytes(b"\x7fELFsoftware-only fixture")
        static={}
        for name in ("HelConfigs.dat","ColorNumFactors.dat","ColorDenomFactors.dat"):
            paths={"origin":self.original/("ML5_0_"+name),"live":self.resources/("ML5_0_"+name),
                   "initial":self.root/"initial"/("ML5_0_"+name),"final":self.root/"final"/("ML5_0_"+name)}
            for p in paths.values():p.write_bytes(name.encode())
            static[name]={key:self.bound(p) for key,p in paths.items()}
        states={}
        for phase in ("initial","postwarm","final"):
            states[phase]={}
            for name in ("HelFilter.dat","LoopFilter.dat"):
                available=phase!="initial" and name=="HelFilter.dat"
                path=self.root/phase/("ML5_0_"+name)
                if available:path.write_bytes(b"1 2\n")
                states[phase][name]={"available":available,"snapshot":self.bound(path) if available else None,
                                     "bytes":4 if available else 0}
        self.log=self.base/"checkpoint.log"
        self.log.write_text("".join(f"SNAPSHOT phase={i} stage={phase} file={name} available={int(state['available'])} bytes={state['bytes']}\n"
            for i,phase in ((1,"postwarm"),(2,"final")) for name,state in states[phase].items()))
        self.result={"execution":{"cwd":str(self.directory),"custody_environment":{
            "DPSTAB_CUSTODY_DIR":str(self.root),"DPSTAB_PROCESS_PREFIX":"ML5_0_","DPSTAB_CUSTODY_LOG":str(self.log)}},
            "custody":{"schema":"dp-resource-custody-v1","resource_directory":str(self.resources),
            "root":str(self.root),"prefix":"ML5_0_","static":static,"filters":states,
            "helper_source":self.bound(self.helper),"helper_object":self.bound(obj),"checkpoint_log":self.bound(self.log)}}

    def bound(self,path):return {"path":str(path),"sha256":dp.digest(path)}
    def validate(self):return dp.validate_resource_custody(self.result,self.origin,self.output,"ML5_0_")

    def test_static_origin_and_present_absent_filter_snapshots_qualified(self):
        value=self.validate()
        self.assertFalse(value["filters"]["initial"]["HelFilter.dat"]["available"])
        self.assertEqual(value["filters"]["postwarm"]["HelFilter.dat"]["bytes"],4)

    def test_restamped_static_change_and_grafted_snapshot_rejected(self):
        path=self.resources/"ML5_0_ColorNumFactors.dat";path.write_bytes(b"changed")
        self.result["custody"]["static"]["ColorNumFactors.dat"]["live"]=self.bound(path)
        with self.assertRaises(ValueError):self.validate()

    def test_filter_log_order_absence_and_changed_environment_rejected(self):
        rows=self.log.read_text().splitlines();self.log.write_text("\n".join(reversed(rows))+"\n")
        self.result["custody"]["checkpoint_log"]=self.bound(self.log)
        with self.assertRaises(ValueError):self.validate()
        self.log.write_text("\n".join(rows)+"\n");self.result["custody"]["checkpoint_log"]=self.bound(self.log)
        self.result["custody"]["filters"]["final"]["LoopFilter.dat"]["bytes"]=1
        with self.assertRaises(ValueError):self.validate()
        self.result["custody"]["filters"]["final"]["LoopFilter.dat"]["bytes"]=0
        self.result["execution"]["custody_environment"]["DPSTAB_PROCESS_PREFIX"]="OTHER_"
        with self.assertRaises(ValueError):self.validate()

    def test_wrong_initial_filter_and_boolean_lengths_rejected(self):
        (self.original/"ML5_0_LoopFilter.dat").write_bytes(b"initial origin state")
        with self.assertRaises(ValueError):self.validate()
        (self.original/"ML5_0_LoopFilter.dat").unlink()
        self.result["custody"]["filters"]["final"]["HelFilter.dat"]["bytes"]=True
        with self.assertRaises(ValueError):self.validate()


class RandomMeasurementAuthority(unittest.TestCase):
    def setUp(self):
        SemanticQualification.setUp(self)
        self.release["numerical_revision"]="revision"
        self.resolved=dp.resolve_origin(self.run,self.release)
        self.resolved["case"]["masses"]=[0.,0.,1.]
        self.configuration={"points_per_case":1,"energy_gev":1000.,"cases":{"fixture":{"definition":{
            **self.resolved["case"]["case"],"id":"fixture"},"actual_generation_case":"fixture",
            "masses":[0.,0.,1.],"physical_card":{"path":str(self.resolved["retained"]/"param_card.dat"),
                "sha256":dp.digest(self.resolved["retained"]/"param_card.dat")},
            "origin_authority":None,"seed":20261021,"cuts":{"pt_min_gev":None,"delta_r_min":None}}}}
        meta=self.put(self.root/"random-metadata.json",{"software_fixture":True})
        self.manifest={"cases":{"fixture":{"points":{"path":str(self.points),"sha256":dp.digest(self.points)},
            "metadata":meta,"points_count":1,"seed":20261021}}}
        manifest_record=self.put(self.root/"random-manifest.json",self.manifest)
        self.run["measurement_authority"]={"kind":"random-dp-v1","manifest":manifest_record,"case":"fixture"}
        def validate(path,expected_release=None,inputs=None):
            self.assertEqual(set(expected_release),{"numerical_revision","generation_revision","generator_identity_sha256","runtime_identity_sha256","archive_sha256"})
            if inputs is not None:inputs[str(path)]=dp.digest(path)
            return self.manifest,self.configuration
        self.mock_random=mock.patch.dict(__import__("sys").modules,{"ninjars_dp_random_points":SimpleNamespace(validate_manifest=validate)})
        self.mock_random.start();self.addCleanup(self.mock_random.stop)

    def resolve(self):return dp.resolve_measurement(self.run,self.release,self.resolved)

    def test_random_measurement_overrides_only_stream_and_labels_n1_descriptive(self):
        result=self.resolve()
        self.assertEqual(result["origin"],self.resolved["origin"])
        self.assertEqual(result["generation_case"],"fixture")
        self.assertEqual(result["sample_kind"],"random-boosted-on-shell-descriptive")
        self.assertEqual(result["seed"],20261021)
        self.assertEqual(result["center_of_mass_energy_gev"],1.)

    def test_grafted_case_and_unknown_override_rejected(self):
        self.run["measurement_authority"]["case"]="other"
        with self.assertRaisesRegex(ValueError,"grafted"):self.resolve()
        self.run["measurement_authority"]["case"]="fixture"
        self.run["measurement_authority"]["kind"]="arbitrary-points"
        with self.assertRaisesRegex(ValueError,"Unknown"):self.resolve()

    def test_model_or_mass_or_source_authority_drift_rejected(self):
        case=self.configuration["cases"]["fixture"]
        case["masses"]=[0.,0.,2.]
        with self.assertRaisesRegex(ValueError,"process/masses/origin"):self.resolve()
        case["masses"]=[0.,0.,1.];case["definition"]["process"]="another process"
        with self.assertRaisesRegex(ValueError,"process/masses/origin"):self.resolve()

    def test_card_drift_and_denominator_drift_rejected(self):
        case=self.configuration["cases"]["fixture"];saved=case["physical_card"]
        case["physical_card"]=self.put(self.root/"other-card.dat","another physical card\n")
        with self.assertRaisesRegex(ValueError,"physical card"):self.resolve()
        case["physical_card"]=saved;self.manifest["cases"]["fixture"]["points_count"]=2
        with self.assertRaisesRegex(ValueError,"denominator/seed"):self.resolve()


if __name__=="__main__":unittest.main()
