"""Inspect effective numerical compilation recipes without compiling objects.

GNU make resolves environment overrides, target-specific assignments, and the
special polynomial recipe. File text alone is insufficient evidence of flags.
Only include/module-output paths and source/object names are removed from the
comparison; optimization, arithmetic, architecture, bounds and macro flags keep
their exact order. The supported environment is GNU Fortran on Linux.
"""
import hashlib
import os
from pathlib import Path
import shlex
import shutil
import subprocess


SCHEMA = "madloop-effective-compiler-flags-v1"
FORTRAN_SUFFIXES = {".f", ".for", ".f90", ".f95", ".f03", ".f08"}
# Compiler wrappers/search mechanisms can add behavior outside make's argv.
COMPILER_ENVIRONMENT = (
    "NIX_CFLAGS_COMPILE", "NIX_CFLAGS_COMPILE_FOR_TARGET", "NIX_CFLAGS_LINK",
    "NIX_LDFLAGS", "NIX_LDFLAGS_FOR_TARGET", "NIX_ENFORCE_NO_NATIVE", "NIX_HARDENING_ENABLE",
    "GCC_EXEC_PREFIX", "COMPILER_PATH", "LIBRARY_PATH", "CPATH",
    "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "GFORTRAN_INCLUDE_PATH",
    "GFORTRAN_CONVERT_UNIT", "SOURCE_DATE_EPOCH",
)


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _compile_command(line, directory, env, compiler_cache):
    """Parse one expanded make recipe, or return None for non-compilation."""
    tokens = shlex.split(line)
    if "-c" not in tokens:
        return None
    sources = [token for token in tokens if not token.startswith("-")
               and Path(token).suffix.lower() in FORTRAN_SUFFIXES]
    if not sources:
        return None
    if len(sources) != 1:
        raise ValueError("Expected one Fortran source per compilation recipe")
    if any(token in {";", "&&", "||", "|", ">", ">>", "<"} for token in tokens):
        raise ValueError("Compound compilation recipes require explicit audit")
    if any(token.startswith("@") for token in tokens):
        raise ValueError("Compiler response files require explicit flag expansion")
    source = sources[0]
    prefix = []
    position = 0
    while position < len(tokens) and not tokens[position].startswith("-") and tokens[position] != source:
        prefix.append(tokens[position])
        position += 1
    if not prefix:
        raise ValueError("Missing compiler command")
    identities = []
    for command in prefix:
        executable = shutil.which(command, path=env.get("PATH"))
        if not executable:
            raise ValueError("Cannot resolve compiler/wrapper: " + command)
        executable = str(Path(executable).resolve())
        if executable not in compiler_cache:
            compiler_cache[executable] = {
                "path": executable, "sha256": _digest(executable),
                "version": subprocess.check_output([executable, "--version"],
                    cwd=directory, env=env, text=True, stderr=subprocess.STDOUT).strip()}
        identities.append(compiler_cache[executable])
    flags = []
    while position < len(tokens):
        token = tokens[position]
        position += 1
        if token == source or token == "-c":
            continue
        if token in {"-o", "-I", "-J", "-isystem", "-iquote"}:
            if position >= len(tokens):
                raise ValueError("Missing compiler option argument: " + token)
            position += 1
            continue
        if token.startswith(("-I", "-J")) and len(token) > 2:
            continue
        if token.startswith(("-isystem=", "-iquote=")):
            continue
        # A joined output option is accepted, but -Ofast/-O3 remain untouched.
        if token.startswith("-o") and len(token) > 2:
            continue
        flags.append(token)
    return Path(source).name, {"compiler": identities, "flags": flags}


def capture_effective_flags(directory, env=None):
    """Return expanded recipe/driver evidence for process, MODEL, and DHELAS.

    ``directory`` is the exported P*_... subprocess directory. All existing
    Fortran sources are inspected, including helpers not linked into ``check``;
    no recipe is executed by this routine's ordinary make dry runs. Call before
    and after the actual build with exactly the same environment.
    """
    directory = Path(directory).resolve()
    env = dict(os.environ if env is None else env)
    output = directory.parents[1]
    scopes = {"process": directory, "MODEL": output / "Source/MODEL",
              "DHELAS": output / "Source/DHELAS"}
    injected_names = set(COMPILER_ENVIRONMENT) | {
        name for name in env if name.startswith(("NIX_CFLAGS_", "NIX_LDFLAGS_"))}
    result = {"schema": SCHEMA, "environment": {name: env[name] for name in sorted(injected_names) if name in env},
              "scopes": {}}
    compiler_cache = {}
    for name, location in scopes.items():
        if not location.is_dir():
            raise ValueError("Missing numerical compilation directory: " + str(location))
        sources = sorted(path for path in location.iterdir()
                         if path.is_file() and path.suffix.lower() in FORTRAN_SUFFIXES)
        if not sources:
            raise ValueError("No Fortran compilation inputs: " + str(location))
        targets = sorted({path.with_suffix(".o").name for path in sources})
        command = ["make", "--no-print-directory", "--dry-run", "--always-make", *targets]
        local_env = dict(env, PWD=str(location))
        completed = subprocess.run(command, cwd=location, env=local_env, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if completed.returncode:
            raise RuntimeError("Cannot resolve compilation recipes in " + str(location) + "\n" + completed.stderr)
        recipes = {}
        # GNU make may preserve explicit backslash continuations in recipes.
        for line in completed.stdout.replace("\\\n", " ").splitlines():
            parsed = _compile_command(line, location, local_env, compiler_cache)
            if parsed is None:
                continue
            source, value = parsed
            if source in recipes and recipes[source] != value:
                raise ValueError("Conflicting effective recipes for " + source)
            recipes[source] = value
        missing = {path.name for path in sources} - set(recipes)
        if missing:
            raise ValueError("Unresolved numerical compilation recipes: " + repr(sorted(missing)))
        result["scopes"][name] = {"recipes": recipes, "expanded_make_output": completed.stdout,
                                 "make_command": command}
    return result


def _signature(recipe):
    return (tuple((item["sha256"], item["version"]) for item in recipe["compiler"]),
            tuple(recipe["flags"]))


def validate_compiler_equality(baseline, candidate):
    """Reject changed effective flags, drivers, or injected compiler settings.

    Extra generated helpers may be present, but must use a compilation policy
    already present in that scope. Every shared source must retain its own
    policy, so moving a special polynomial -O3 rule to ordinary code fails.
    """
    if baseline.get("schema") != SCHEMA or candidate.get("schema") != SCHEMA:
        raise ValueError("Missing or incompatible effective compiler evidence")
    if baseline["environment"] != candidate["environment"]:
        raise ValueError("Baseline/candidate compiler-injected environment differs")
    if set(baseline["scopes"]) != set(candidate["scopes"]) or not baseline["scopes"]:
        raise ValueError("Baseline/candidate numerical compiler scopes differ")
    for scope in baseline["scopes"]:
        left = baseline["scopes"][scope]["recipes"]
        right = candidate["scopes"][scope]["recipes"]
        if not left or not right:
            raise ValueError("Missing effective numerical compiler recipes")
        if {_signature(value) for value in left.values()} != {_signature(value) for value in right.values()}:
            raise ValueError("Baseline/candidate effective compiler flags differ in " + scope)
        for source in left.keys() & right.keys():
            if _signature(left[source]) != _signature(right[source]):
                raise ValueError("Baseline/candidate effective compiler flags differ for " + scope + "/" + source)
