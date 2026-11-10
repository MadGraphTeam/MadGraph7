def resume_host_build(build, snapshot, observed_dso, plan):
    cases = {}
    installed = Path(snapshot["installed_native"]["lib/libmadloop_tirust.so"]["path"])
    for case in CASE_NAMES:
        saved = snapshot["cases"][case]
        original_root = Path(saved["generated_root"])
        sub_relative = Path(saved["focus_sources"]["check_sa.f"]["path"]).parent.relative_to(original_root)
        original_sub = original_root/sub_relative
        scope = saved["effective_compiler_flags"]["scopes"]["process"]
        original_lines = scope["expanded_make_output"].splitlines()
        logfile = ROOT/"docs/benchmarkfolder/campaigns/split-dp-terminal-1/inputs/candidate"/case/"compilation-process.log"
        compilation = read(saved["compilation"]["path"])
        if digest(logfile) != compilation["build_logs"]["process"]:
            raise ValueError("Historical process command log changed")
        links = [shlex.split(line) for line in logfile.read_text().splitlines() if " -o check " in line]
        if len(links) != 1:
            raise ValueError("Ambiguous historical link command")
        case_result = {"historical_link_log": pin(logfile), "arms": {}}
        for arm in ("original", "observed"):
            clone = build.work/"cases"/case/arm
            if case == "qcd_n3_uux_zgg" and arm == "original":
                # Already compiled/linked; prior artifact pins are checked by the
                # continuation before this exact object/executable reuse.
                sub = clone/sub_relative
                changed_names = ("check_sa.f",)
                fc = scope["recipes"]["check_sa.f"]["compiler"][0]
                check_pin(fc)
                extras = {name: pin(sub/name) for name in ("host_observer.o", "observer_noop.o")}
                selected = installed
                link_map = sub/"private-link.map"
            else:
                for collection in (saved["all_sources_and_includes"], saved["existing_numerical_objects"]):
                    for rel, record in collection.items():
                        build.copy_file(check_pin(record), clone/rel)
                # MODEL/DHELAS and original exported library aliases are materialized,
                # not symlinked back to mutable source trees.
                for record in saved["link_inputs"]["archives"].values():
                    p = Path(record["path"])
                    if p.is_relative_to(original_root):
                        build.copy_file(check_pin(record), clone/p.relative_to(original_root))
                # Export lib aliases may resolve outside the generated root.
                for p in (original_root/"lib").iterdir():
                    if p.is_file():
                        resolved = str(p.resolve())
                        known = saved["link_inputs"]["archives"]
                        if resolved in known:
                            build.copy_file(check_pin(known[resolved]), clone/"lib"/p.name)
                sub = clone/sub_relative
                changed_names = CHANGED if arm == "observed" else ("check_sa.f",)
                for name in changed_names:
                    (sub/name).write_bytes((PROPOSAL/"prepared"/case/name).read_bytes())
                (sub/"host_observer.f90").write_bytes((PROPOSAL/"host_observer.f90").read_bytes())
                fc = scope["recipes"]["check_sa.f"]["compiler"][0]
                check_pin(fc)
                module = sub/"observer-modules"
                module.mkdir()
                build.command([fc["path"], "-w", "-fPIC", "-J"+str(module), "-I"+str(module),
                               "-c", "host_observer.f90", "-o", "host_observer.o"], sub,
                              case+"-"+arm+"-host-module")
                for name in changed_names:
                    commands = [shlex.split(line) for line in original_lines if " -c "+name+" " in line]
                    if len(commands) != 1:
                        raise ValueError("Ambiguous saved compile recipe: "+name)
                    argv = [x.replace(str(original_root), str(clone)) for x in commands[0]]
                    argv += ["-I"+str(module), "-J"+str(module)]
                    build.command(argv, sub, case+"-"+arm+"-"+Path(name).stem)
                extras = {"host_observer.o": pin(sub/"host_observer.o")}
                if arm == "original":
                    build.copy_file(PROPOSAL/"observer_noop.c", sub/"observer_noop.c")
                    build.command([build.tool("cc"), "-O0", "-fPIC", "-c", "observer_noop.c", "-o",
                                   "observer_noop.o"], sub, case+"-noop")
                    extras["observer_noop.o"] = pin(sub/"observer_noop.o")
                argv = [x.replace(str(original_root), str(clone)) for x in links[0]]
                selected = observed_dso if arm == "observed" else installed
                if arm == "observed":
                    if argv.count("-lmadloop_tirust") != 1:
                        raise ValueError("Unexpected historical TIRust link")
                    argv[argv.index("-lmadloop_tirust")] = str(selected)
                    argv = [x.replace("-Wl,-rpath,"+str(installed.parent),
                                      "-Wl,-rpath,"+str(selected.parent)) for x in argv]
                insert = argv.index("-L"+str(clone/"lib")+"/")
                argv[insert:insert] = list(extras)
                link_map = sub/"private-link.map"
                argv += ["-Wl,-Map,"+str(link_map)]
                build.command(argv, sub, case+"-"+arm+"-link")
            runtime_env = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                           "MKL_NUM_THREADS": "1", "RAYON_NUM_THREADS": "1",
                           "LD_LIBRARY_PATH": os.pathsep.join(map(str, [selected.parent,
                               ROOT/"HEPTools/ninja/lib", ROOT/"HEPTools/collier", ROOT/"HEPTools/ninjars/lib"]))}
            env = {"PATH": os.defpath, "LANG": "C", "LC_ALL": "C", "TZ": "UTC"}
            env.update(runtime_env)
            closure, interpreter = runtime(build, sub/"check", selected, env, case+"-"+arm)
            historical_runtime = {value["resolved_path"]: value["sha256"]
                                  for value in saved["runtime"]["libraries"].values()
                                  if "ld-linux" not in Path(value["resolved_path"]).name
                                  and Path(value["resolved_path"]).name != "libmadloop_tirust.so"}
            actual_runtime = {path: expected for path, expected in closure.items()
                              if "ld-linux" not in Path(path).name and Path(path).name != "libmadloop_tirust.so"}
            if actual_runtime != historical_runtime:
                raise ValueError("Non-TIRust runtime closure changed from the frozen snapshot")
            actual_objects = {rel: pin(clone/rel) for rel in saved["existing_numerical_objects"]}
            allowed = {str(sub_relative/Path(x).with_suffix('.o')) for x in changed_names}
            for rel, record in actual_objects.items():
                if rel not in allowed and record["sha256"] != saved["existing_numerical_objects"][rel]["sha256"]:
                    raise ValueError("Unexpected numerical object change: "+rel)
            loads = [line[5:].strip() for line in link_map.read_text().splitlines() if line.startswith("LOAD ")]
            link_inputs = {}
            allowed_loads = {str(Path(v["path"]).resolve()): v["sha256"]
                             for v in plan["toolchain_link_inputs"].values()}
            for group in (saved["link_inputs"]["archives"], saved["runtime"]["libraries"]):
                allowed_loads.update({str(Path(v["path"]).resolve()): v["sha256"] for v in group.values()})
            allowed_loads.update({str(Path(v["path"]).resolve()): v["sha256"] for v in actual_objects.values()})
            allowed_loads.update({str(Path(v["path"]).resolve()): v["sha256"] for v in extras.values()})
            allowed_loads[str(selected.resolve())] = digest(selected)
            for p in (clone/"lib").iterdir():
                if p.is_file():
                    if p.relative_to(clone).as_posix() == "lib/mpmodule.mod":
                        module_input = saved["existing_numerical_objects"].get("lib/mpmodule.mod")
                        if (module_input is None
                                or module_input["sha256"] != "b8a883c9a27983dc9a57fdc1f064980886b246772ee359c9edecf91ce7eaa14f"
                                or module_input["size_bytes"] != 24356
                                or Path(module_input["resolved_path"]) != ROOT/"vendor/CutTools/includects/mpmodule.mod"):
                            raise ValueError("Unrecognized compiler module provenance")
                        check_pin(module_input)
                        if digest(p) != module_input["sha256"] or p.stat().st_size != module_input["size_bytes"]:
                            raise ValueError("Private compiler module changed")
                        continue
                    expected = {v["sha256"] for v in saved["link_inputs"]["archives"].values()}
                    if digest(p) not in expected:
                        raise ValueError("Private library copy is not a frozen archive")
                    allowed_loads[str(p.resolve())] = digest(p)
            for value in loads:
                p = Path(value)
                p = p if p.is_absolute() else sub/p
                if not p.is_file():
                    raise ValueError("Unresolved real linker input: "+value)
                if allowed_loads.get(str(p.resolve())) != digest(p):
                    raise ValueError("Unbound actual linker input: "+str(p))
                link_inputs[str(p.resolve())] = pin(p)
            case_result["arms"][arm] = {
                "executable": pin(sub/"check"), "runtime": closure, "interpreter": interpreter,
                "launch_prefix": [interpreter["path"]], "runtime_environment": runtime_env,
                "objects": actual_objects, "extra_objects": extras, "link_inputs": link_inputs,
                "link_map": pin(link_map), "sources": {rel: pin(clone/rel)
                    for rel in saved["all_sources_and_includes"]},
                "extra_sources": {"host_observer.f90": pin(sub/"host_observer.f90")},
                "compiler_check": {"status": "complete", "compiler": fc},
                "observer_enabled_at_build": False}
            if arm == "original":
                case_result["arms"][arm]["extra_sources"]["observer_noop.c"] = pin(sub/"observer_noop.c")
        cases[case] = case_result
    return cases

