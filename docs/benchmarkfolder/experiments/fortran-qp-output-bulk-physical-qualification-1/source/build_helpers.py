"""Read-only build-selection helpers derived from closed profile-controller3.

Only expanded argv for mtirust.f90, check_sa.f and one final check link are
returned. All other dry-run recipes remain text. Callers own command execution,
input/recipe pinning, immutable numerical objects and runtime/linker closure.
"""
from collections import Counter
from pathlib import Path
import re
import shlex
import shutil

FORTRAN_SUFFIXES = {".f", ".for", ".f90", ".f95", ".f03", ".f08"}
SOURCE_SUFFIXES = FORTRAN_SUFFIXES | {".c", ".cc", ".cpp", ".cxx", ".s", ".asm"}
LINK_MAP_OPTION = "-Wl,-Map,private-link.map,--cref"
WRAP_OPTION = "-Wl,--wrap=ml_tirust_evaluate_mp"
WRAPPER_SYMBOL = "__wrap_ml_tirust_evaluate_mp"


def require(value, message):
    if not value:
        raise ValueError(message)


def private_makefile(text, wire=False):
    """Add private objects without changing a closed numerical source recipe."""
    old_link = "$(FC) $(FFLAGS) -o $(PROG) check_sa.o $(PROCESS) $(LINKLIBS)"
    old_dependencies = "$(PROG): check_sa.o $(PROCESS)"
    require(text.count(old_link) == 1, "Ambiguous original check link recipe")
    require(text.count(old_dependencies) == 1, "Ambiguous original check dependencies")
    require("mtirust.o" not in text and "wire_capture.o" not in text and
            "--wrap=ml_tirust_evaluate_mp" not in text,
            "Makefile already contains a private adapter or wire observer")
    extra_objects = "mtirust.o" + (" wire_capture.o" if wire else "")
    options = LINK_MAP_OPTION + (" " + WRAP_OPTION if wire else "")
    new_link = ("$(FC) $(FFLAGS) " + options + " -o $(PROG) check_sa.o " +
                extra_objects + " $(PROCESS) $(LINKLIBS)")
    result = text.replace(old_link, new_link, 1).replace(
        old_dependencies, "$(PROG): check_sa.o " + extra_objects + " $(PROCESS)", 1)
    return result + """
# Exact private adapter; closed TIRUST/TIR/model/DHELAS objects are reused.
mtirust.o: mtirust.f90
\t$(FC) -O3 -fPIC -c $< -o $@
"""


def _tokens(line):
    lexer = shlex.shlex(line, posix=True, punctuation_chars=";&|<>")
    lexer.whitespace_split = True
    lexer.commenters = ""
    return list(lexer)


def _sources(tokens):
    return [t for t in tokens if not t.startswith("-") and
            Path(t).suffix.lower() in SOURCE_SUFFIXES]


def select_commands(output, expected_compiler, environment, wire=False):
    """Return exactly two strict single compiler recipes and one check link.

    The fixed module recipe is checked here. The unchanged worker recipe and
    every reused numerical object still require the caller's frozen-input
    comparison; this helper never executes the other expanded make recipes.
    """
    needed = ["mtirust.f90", "check_sa.f"]
    selected, links = {}, []
    for line in output.replace("\\\n", " ").splitlines():
        tokens = _tokens(line)
        if not tokens:
            continue
        sources = _sources(tokens)
        wanted = [s for s in sources if Path(s).name in needed]
        outputs = [tokens[i + 1] for i, token in enumerate(tokens[:-1]) if token == "-o"]
        final_link = "-c" not in tokens and "check" in outputs
        if not wanted and not final_link:
            continue
        require(not re.search(r"[;&|<>`$\n\r]", line), "Compound/shell selected build recipe")
        require(not any("@" in token or token.startswith("#") for token in tokens),
                "Selected response file/comment syntax")
        resolved = shutil.which(tokens[0], path=environment.get("PATH"))
        require(resolved is not None and Path(resolved).resolve() == Path(expected_compiler).resolve(),
                "Unknown selected compiler prefix")
        require(tokens.count("-o") == 1 and len(outputs) == 1 and
                not any(t.startswith("-o") and t != "-o" for t in tokens),
                "Ambiguous selected output")
        if final_link:
            require(not sources and tokens.count("mtirust.o") == 1 and
                    tokens.count("check_sa.o") == 1, "Unexpected source/private object in check link")
            require(all(t == Path(t).name for t in tokens if
                        Path(t).name in {"mtirust.o", "check_sa.o", "wire_capture.o"}),
                    "Ambiguous private object spelling")
            require(tokens.count("-ltirust") == 1 and
                    tokens.index("mtirust.o") < tokens.index("-ltirust"),
                    "Private MTIRUST link precedence differs")
            # Every other explicit process object must follow the adapter.
            position = tokens.index("mtirust.o")
            process_objects = [i for i, t in enumerate(tokens) if t.endswith(".o") and
                               t not in {"check_sa.o", "mtirust.o", "wire_capture.o"}]
            require(all(position < i for i in process_objects), "Private adapter follows process objects")
            require(tokens.count(LINK_MAP_OPTION) == 1, "Missing/duplicate private link map")
            require(not any("-Map" in t and t != LINK_MAP_OPTION for t in tokens),
                    "Unexpected additional link map")
            require(tokens.count("wire_capture.o") == int(bool(wire)) and
                    tokens.count(WRAP_OPTION) == int(bool(wire)), "Wire observer selection differs")
            require(not any("--wrap" in t and t != WRAP_OPTION for t in tokens),
                    "Unexpected linker wrapper")
            links.append(tokens)
        else:
            require(tokens.count("-c") == 1 and len(sources) == len(wanted) == 1 and
                    sources[0] == Path(sources[0]).name,
                    "Unexpected/ambiguous source in selected compile")
            name = sources[0]
            require(outputs == [Path(name).stem + ".o"], "Selected object target differs")
            require(name not in selected, "Duplicate selected compile recipe")
            if name == "mtirust.f90":
                require(tokens[1:] == ["-O3", "-fPIC", "-c", "mtirust.f90", "-o", "mtirust.o"],
                        "Unexpected explicit MTIRUST module recipe")
            selected[name] = tokens
    require(set(selected) == set(needed), "Private build lacks exactly two source recipes")
    require(len(links) == 1, "Expected exactly one selected check link")
    return [(name, selected[name]) for name in needed], links[0]


def _defined(text):
    symbols = Counter()
    for line in text.splitlines():
        match = re.fullmatch(r"\s*([0-9a-fA-F]+)\s+([A-Za-z?])\s+(\S+)\s*", line)
        if match:
            require(match[2] not in {"U", "?"}, "Undefined/unknown entry in defined-symbol listing")
            symbols[match[3]] += 1
    return symbols


def verify_symbols(module_nm, exe_nm, map_text, wire=False):
    """Require every compared MTIRUST definition once and one private owner."""
    module_all, executable_all = _defined(module_nm), _defined(exe_nm)
    own = {name: count for name, count in module_all.items() if name.startswith("__mtirust_MOD_")}
    linked = {name: count for name, count in executable_all.items() if name.startswith("__mtirust_MOD_")}
    require(own and all(v == 1 for v in own.values()) and own == linked,
            "MTIRUST symbols are missing/duplicated")
    require(not re.search(r"libtirust\.a\([^)]*mtirust[^)]*\)", map_text, re.I),
            "Original MTIRUST archive member was pulled")
    loads = re.findall(r"^LOAD\s+(.*?)\s*$", map_text, re.M)
    require(loads.count("mtirust.o") == 1, "Private MTIRUST object absent/duplicated in link map")
    require(loads.count("wire_capture.o") == int(bool(wire)), "Wire capture object selection differs")
    require(executable_all[WRAPPER_SYMBOL] == int(bool(wire)), "Wire wrapper definition differs")
    require(executable_all["__real_ml_tirust_evaluate_mp"] == 0,
            "Unexpected defined real-provider wrapper alias")
    return {"definitions": own, "original_archive_member_loaded": False,
            "private_module_object_loads": 1,
            "wire_wrapper_definitions": executable_all[WRAPPER_SYMBOL],
            "wire_capture_object_loads": loads.count("wire_capture.o")}
