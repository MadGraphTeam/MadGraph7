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
            dp.dump(p/"summary.json",snap)
            self.assertEqual(dp.report_snapshot(p/"summary.json"),snap)
            Path(runs[0]["raw"]["path"]).write_text("tampered\n")
            with self.assertRaises(ValueError):dp.report_snapshot(p/"summary.json")

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


if __name__=="__main__":unittest.main()
