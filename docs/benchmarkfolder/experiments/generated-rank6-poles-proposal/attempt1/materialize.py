#!/usr/bin/env python3
"""Source-only append-only materializer. No subprocesses, compiler or ctypes.

Run from the repository root with python -B after sourcing tools/tirust_env.sh.
Outputs retained materialized Fortran/includes and build-plan.json. Commands in
that plan are inert data; this script has no execution mode.
"""
import hashlib
import json
from pathlib import Path
import re
import shlex
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT))
from madgraph.loop.loop_exporters import LoopProcessOptimizedExporterFortranSA as Exporter
from madgraph.various.q_polynomial import FortranPolynomialRoutines
from madgraph.iolibs.file_writers import FortranWriter


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(text)


def routine(source, name):
    match = re.search(r"(?mi)^      SUBROUTINE " + re.escape(name) + r"\(", source)
    if match is None:
        raise ValueError("Missing routine " + name)
    end = re.search(r"(?mi)^      END\s*$", source[match.start():])
    if end is None:
        raise ValueError("Missing END " + name)
    return source[match.start():match.start() + end.end()] + "\n"


def render(path, text):
    if path.exists():
        raise FileExistsError(path)
    with FortranWriter(str(path)) as writer:
        writer.writelines(text, context={"tirust_available": True,
            "tirust_supports_quad_prec": True, "AmplitudeReduction": False})


def main():
    frozen = json.loads((HERE.parent / "index.json").read_text())
    for name, sha in frozen["files"].items():
        if digest(HERE.parent / name) != sha:
            raise ValueError("Frozen proposal changed: " + name)
    inventory = json.loads((HERE.parent / "inventory.json").read_text())
    for item in inventory["shared_bindings"]:
        if digest(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("Shared binding changed: " + item["path"])
    dest = HERE / "materialized"
    if dest.exists():
        raise FileExistsError(dest)
    dest.mkdir()
    prefix = ROOT / ".tirust-build/qp-transport-install/mg/HEPTools/tirust/lib"
    template = ROOT / "madgraph/iolibs/template_files/loop_optimized"
    dependencies = {}
    builds, schedule = {}, []
    next_id = 1
    for model, entry in enumerate(inventory["processes"]):
        label = "ew" if model == 0 else "qcd"
        sub = ROOT / entry["subprocess"]
        output = sub.parents[1]
        local = dest / label
        local.mkdir()
        for item in entry["bindings"]:
            if digest(ROOT / item["path"]) != item["sha256"]:
                raise ValueError("Process binding changed: " + item["path"])
        capacity = entry["coefficient_capacity"]
        source = (sub / "CT_interface.f").read_text()
        group_routine = routine(source, "ML5_0_LOOP_6")
        integers = {name: int(re.search(name+r"\s*=\s*(\d+)", group_routine)[1])
                    for name in ("NWAVEFUNCS", "NLOOPGROUPS", "NCOMB")}
        groups = [269, 270] if model == 0 else [399]
        calls = [next(row for row in entry["calls"] if row["group_id"] == group) for group in groups]
        exterior = (sub / "helas_calls_ampb_1.f").read_text().splitlines()
        exterior = [line for line in exterior
                    if re.match(r"      CALL [IOV]XXXXX\(P\(0,[1-6]\)", line)]
        if len(exterior) != 6:
            raise ValueError("External HELAS declarations changed")
        capture = []
        for k, call in enumerate(calls, 1):
            capture += [f"      IF(K.EQ.{k}) THEN", f"      LOOPCOEFS(0,1,{call['group_id']})=1D0",
                        "      " + call["statement"].replace(",I_SO,", ",1,"), "      ENDIF"]
        driver = (HERE / "driver.f.in").read_text()
        replacements = {"NC": capacity, "NCM1": capacity-1, "MODEL": model,
                        "NCAPTURE": len(groups), "CAPTURE_BASE": 900 if model == 0 else 902,
                        "NWAVE": integers["NWAVEFUNCS"], "NGROUP": integers["NLOOPGROUPS"],
                        "NCOMB": integers["NCOMB"], "EXTERNAL_CALLS": "\n".join(exterior),
                        "CAPTURE_CALLS": "\n".join(capture)}
        for key, value in replacements.items():
            driver = driver.replace("@" + key + "@", str(value))
        if "@" in driver:
            raise ValueError("Unexpanded driver token")
        render(local / "driver.f", driver)
        for filename in ("coupl.inc", "MadLoopParams.inc", "unique_id.inc", "coef_specs.inc"):
            candidate = sub / filename
            if not candidate.exists():
                candidate = output / "Source/DHELAS" / filename
            save(local / filename, candidate.read_text())
            dependencies[str(candidate.relative_to(ROOT))] = digest(candidate)
        save(local / "loop_max_coefs.inc", "      INTEGER LOOPMAXCOEFS\n      PARAMETER(LOOPMAXCOEFS=210)\n")
        global_source = (output / "SubProcesses/global_specs.inc").read_text()
        global_source, changes = re.subn(r"OVERALLMAXRANK=\d+", "OVERALLMAXRANK=6", global_source)
        if changes != 1:
            raise ValueError("Global rank declaration changed")
        save(local / "global_specs.inc", global_source)
        fpr = FortranPolynomialRoutines(6, updater_max_rank=1, sub_prefix="CAP6_")
        render(local / "cap_polynomial.f", fpr.write_polynomial_constant_module())
        save(local / "monomials.json", json.dumps(fpr.get_COLLIER_mapping()) + "\n")
        rep = {"proc_prefix": "CAP6_", "maxrank": 6, "nloop_groups": 1,
               "nSquaredSO": 1, "real_dp_format": "REAL*8", "real_mp_format": "REAL*16",
               "complex_dp_format": "complex*16", "complex_mp_format": "complex*32",
               "mass_dp_format": "complex*16", "info_lines": "C Private capacity adapter",
               "process_lines": "C Synthetic rank-six tensor extension, not an amplitude"}
        # Deliberately synthetic tensor-plan declarations, not a physical HELAS object.
        amps = [SimpleNamespace(get_call_key=lambda n=n: (None,n),
                                get_analytic_info=lambda key,r=r: r)
                for n,r in ((2,6),(6,4),(6,5),(6,6))]
        matrix = SimpleNamespace(rep_dict=rep,
            get_loop_diagrams=lambda: [SimpleNamespace(get_loop_amplitudes=lambda: amps)])
        exporter = SimpleNamespace(opt={"vector_size": 0}, template_dir=str(template))
        tirust = Exporter.write_TIRUST_interface(exporter, None, matrix)
        render(local / "cap_tirust.f", tirust)
        collier = Exporter.write_COLLIER_interface(exporter, None, matrix)
        collier = collier.replace("CALL INITCOLLIER()", "CALL CAP6_INITCOLLIER()")
        marker = "C     We can now call COLLIER"
        if collier.count(marker) != 1:
            raise ValueError("COLLIER observation anchor changed")
        collier = collier.replace(marker,
            "      CALL RP_CLL_INPUT(NLOOPLINE,RANK,MOMVEC,MOMINV,M2LCOLLIER)\n"+marker)
        render(local / "cap_collier.f", collier)
        helpers = ""
        for file,names in [(sub/"CT_interface.f", ["BUILD_KINEMATIC_MATRIX","MP_BUILD_KINEMATIC_MATRIX"]),
                           (sub/"TIR_interface.f", ["SWITCH_ORDER","MP_SWITCH_ORDER"])]:
            for name in names:
                helpers += routine(file.read_text(), "ML5_0_" + name).replace("ML5_0_", "CAP6_")
        helpers += routine((output/"SubProcesses/MadLoopCommons.f").read_text(), "INITCOLLIER").replace(
            "SUBROUTINE INITCOLLIER", "SUBROUTINE CAP6_INITCOLLIER")
        save(local / "cap_helpers.f", helpers)
        point = ROOT / "docs/benchmarkfolder/campaigns/split-dp-terminal-1/points" / (entry["process"]+".dat")
        save(local / "point.dat", "\n".join(point.read_text().splitlines()[:7])+"\n")
        for name in ("param_card.dat", "ident_card.dat"):
            save(local / "nominal" / name, (output/"Cards"/name).read_text())
        fixtures = [101,103,105,106,107] if model == 0 else [102,104]
        rows = []
        for fixture in fixtures:
            n = 2 if fixture == 106 else 6
            rank = {101:4,102:5}.get(fixture,6)
            for ct in (1,4):
                for mode in range(3):
                    for finite in (1,0):
                        row=[next_id,fixture,9,ct,mode,finite,0,268435456,1,n,rank]
                        next_id+=1;rows.append(row)
            for mode in range(3):
                row=[next_id,fixture,7,1,mode,1,0,268435456,1,n,rank]
                next_id+=1;rows.append(row)
        if model == 1:
            for ct in (1,4):
                for j,(finite,mode,cache,clear,budget,direction) in enumerate([
                    (1,0,-1,1,268435456,ct),(1,0,-1,0,268435456,ct),
                    (0,0,-1,0,268435456,ct),(0,0,-1,0,268435456,ct),
                    (0,1,-1,0,268435456,ct),(0,2,-1,0,268435456,ct),
                    (0,0,-1,0,268435456,ct+1),(0,0,0,0,268435456,ct+1),
                    (0,0,-1,0,0,ct)]):
                    rows.append([2000+ct*10+j,104,9,direction,mode,finite,cache,budget,clear,6,6])
        save(local/"schedule.dat",str(len(rows))+"\n"+"\n".join(" ".join(map(str,row)) for row in rows)+"\n")
        schedule += [{"model":label,"action":row[0],"fixture":row[1],"backend":row[2],"ct":row[3],
                      "mode":row[4],"finite":row[5],"cache":row[6],"budget":row[7],
                      "clear":row[8],"n":row[9],"rank":row[10]} for row in rows]
        work = ROOT / ".tirust-build/generated-rank6-poles-1" / label
        includes = [local,work,sub,output/"Source/MODEL",output/"Source/DHELAS",prefix,
                    ROOT/"HEPTools/collier/include",ROOT/"HEPTools/include"]
        flags=["-O2","-fPIC","-ffixed-line-length-132"]+["-I"+str(p) for p in includes]+["-J"+str(work)]
        commands=[]
        sources=["cap_polynomial.f","cap_tirust.f","cap_collier.f","cap_helpers.f","driver.f"]
        for name in sources:
            commands.append(["gfortran",*flags,"-c",str(local/name),"-o",str(work/Path(name).with_suffix(".o"))])
        commands.append(["gcc","-std=c99","-O2","-fPIC","-DORIGINAL_NC="+str(capacity),
                         "-I"+str(ROOT/"vendor/tirust"),"-c",str(HERE/"observer.c"),"-o",str(work/"observer.o")])
        objects=sorted(p for p in sub.glob("*.o") if p.name!="check_sa.o")
        expected={p.with_suffix(".f").name for p in objects}
        if not objects or any(not (sub/name).is_file() for name in expected):
            raise ValueError("Object provenance missing")
        for obj in objects:
            dependencies[str(obj.relative_to(ROOT))]=digest(obj)
            dependencies[str(obj.with_suffix(".f").relative_to(ROOT))]=digest(obj.with_suffix(".f"))
        definitions=(output/"SubProcesses/MadLoop_makefile_definitions").read_text()
        libs=re.search(r"(?m)^LINK_LOOP_LIBS = (.*)$",definitions)[1]
        libs=shlex.split(libs.replace("$(LIBDIR)",str(output/"lib")+"/"))
        link=["gfortran","-o",str(work/"rank6-poles"),*[str(work/Path(s).with_suffix(".o")) for s in sources],
              str(work/"observer.o"),*map(str,objects),"-L"+str(output/"lib"),"-ldhelas","-lmodel",*libs,
              "-lstdc++","-Wl,--wrap=ml5_0_tirustloop_","-Wl,--wrap=ml_tirust_evaluate_dp",
              "-Wl,--wrap=ml_tirust_evaluate_mp","-Wl,-Map,"+str(work/"link.map")]
        commands.append(link)
        for lib in (output/"lib").glob("*.a"):
            dependencies[str(lib.relative_to(ROOT))]=digest(lib)
        for file in ("libmadloop_tirust.so","libtirust.a","mtirust.mod","tirust-config.json"):
            dependencies[str((prefix/file).relative_to(ROOT))]=digest(prefix/file)
        builds[label]={"cwd":str(work),"commands":commands,"runtime_inputs":["point.dat","schedule.dat","nominal/param_card.dat","nominal/ident_card.dat"],
                       "runtime_outputs":["records.txt","observer.txt","param.log"],"captures":groups,
                       "original_nc":capacity,"entry":entry["process"],"reuse_objects":list(map(str,objects))}
    if sum(row["backend"]==9 for row in schedule)+3!=105 or sum(row["backend"]==7 for row in schedule)!=21:
        raise ValueError("Call budget changed")
    save(HERE/"schedule.json",json.dumps(schedule,indent=2,sort_keys=True)+"\n")
    save(HERE/"build-plan.json",json.dumps({"schema":1,"status":"source-only; commands not executed",
        "models":builds,"dependencies":dependencies,"limits":{"compile_seconds":240,"compile_jobs":2,
        "worker_seconds":900,"worker_address_space_bytes":4294967296,"tirust_calls":105,"collier_adapter_calls":21}},indent=2,sort_keys=True)+"\n")
    files={str(path.relative_to(HERE)):digest(path) for path in sorted(dest.rglob("*")) if path.is_file()}
    save(HERE/"materialized-index.json",json.dumps({"schema":1,"files":files,
        "proposal_index_sha256":digest(HERE.parent/"index.json"),"materializer_sha256":digest(Path(__file__))},indent=2,sort_keys=True)+"\n")
    print("Prepared two model source sets and inert compile/link commands; no compiler/native execution")


if __name__=="__main__":
    main()
