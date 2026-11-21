"""Acceptance compares expanded numerical recipes, including environment flags."""
import copy
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
from tirust_benchmark_compiler import capture_effective_flags, validate_compiler_equality


@unittest.skipUnless(shutil.which("make"), "GNU make is required")
class TestEffectiveCompilerFlags(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.compiler = self.root / "fake-gfortran"
        self.compiler.write_text('#!/bin/sh\nprintf "fixture compiler 1.0\\n"\n')
        self.compiler.chmod(0o755)
        self.env = dict(os.environ)
        for key in ("FFLAGS", "GLOBAL_FLAG", "ALOHA_FLAG", "MAKEFLAGS", "MFLAGS", "MAKEOVERRIDES"):
            self.env.pop(key, None)

    def output(self, name):
        output = self.root / name
        for scope, names in (("SubProcesses/P0_test", ("loop_matrix", "polynomial", "check_sa")),
                             ("Source/MODEL", ("couplings",)),
                             ("Source/DHELAS", ("aloha_functions",))):
            directory = output / scope
            directory.mkdir(parents=True)
            makefile = f"""FC = {self.compiler}
ifeq ($(origin FFLAGS),undefined)
FFLAGS = -O0 -fPIC
endif
FFLAGS += $(GLOBAL_FLAG)
POLYNOMIAL_OPTIMIZATION = -O3
INCLUDE_DIR = {directory}/includes
%.o: %.f
\t$(FC) $(FFLAGS) -I$(INCLUDE_DIR) -c $< -o $@
\ttouch forbidden-build-side-effect
polynomial.o: polynomial.f
\t$(FC) $(filter-out -O%,$(FFLAGS)) $(POLYNOMIAL_OPTIMIZATION) -c $< -o $@
"""
            if scope.endswith("DHELAS"):
                makefile += "FFLAGS += $(ALOHA_FLAG)\n"
            (directory / "makefile").write_text(makefile)
            for stem in names:
                (directory / (stem + ".f")).write_text("      END\n")
        return output / "SubProcesses/P0_test"

    def capture(self, directory, **changes):
        return capture_effective_flags(directory, env=dict(self.env, **changes))

    def test_normalizes_include_locations_and_never_compiles(self):
        baseline = self.output("baseline")
        candidate = self.output("candidate")
        left, right = self.capture(baseline), self.capture(candidate)
        validate_compiler_equality(left, right)
        self.assertFalse(list(self.root.rglob("*.o")))
        self.assertFalse(list(self.root.rglob("forbidden-build-side-effect")))
        process = left["scopes"]["process"]["recipes"]
        self.assertEqual(process["loop_matrix.f"]["flags"], ["-O0", "-fPIC"])
        self.assertEqual(process["polynomial.f"]["flags"], ["-fPIC", "-O3"])
        self.assertIn("-I", left["scopes"]["process"]["expanded_make_output"])

    def test_environment_override_and_arithmetic_flags_are_not_hidden(self):
        directory = self.output("output")
        left = self.capture(directory)
        right = self.capture(directory, FFLAGS="-O3 -ffast-math -march=native")
        self.assertEqual(right["scopes"]["process"]["recipes"]["loop_matrix.f"]["flags"],
                         ["-O3", "-ffast-math", "-march=native"])
        with self.assertRaisesRegex(ValueError, "effective compiler flags"):
            validate_compiler_equality(left, right)
        with self.assertRaisesRegex(ValueError, "effective compiler flags"):
            validate_compiler_equality(left, self.capture(directory, GLOBAL_FLAG="-fcheck=bounds"))

    def test_target_specific_and_dhelas_flags_are_expanded(self):
        directory = self.output("output")
        left = self.capture(directory)
        path = directory / "makefile"
        path.write_text(path.read_text() + "loop_matrix.o: FFLAGS += -frounding-math\n")
        right = self.capture(directory, ALOHA_FLAG="-fno-unsafe-math-optimizations")
        self.assertIn("-frounding-math", right["scopes"]["process"]["recipes"]["loop_matrix.f"]["flags"])
        self.assertIn("-fno-unsafe-math-optimizations", right["scopes"]["DHELAS"]["recipes"]["aloha_functions.f"]["flags"])
        with self.assertRaises(ValueError):
            validate_compiler_equality(left, right)

    def test_shared_source_policy_cannot_be_swapped_with_polynomial(self):
        left = self.capture(self.output("output"))
        right = copy.deepcopy(left)
        recipes = right["scopes"]["process"]["recipes"]
        recipes["loop_matrix.f"], recipes["polynomial.f"] = recipes["polynomial.f"], recipes["loop_matrix.f"]
        with self.assertRaisesRegex(ValueError, "for process/"):
            validate_compiler_equality(left, right)

    def test_extra_helper_with_same_policy_is_allowed_but_new_policy_is_not(self):
        left = self.capture(self.output("output"))
        right = copy.deepcopy(left)
        recipes = right["scopes"]["process"]["recipes"]
        recipes["tirust_helper.f"] = copy.deepcopy(recipes["loop_matrix.f"])
        validate_compiler_equality(left, right)
        recipes["tirust_helper.f"]["flags"].append("-Ofast")
        with self.assertRaises(ValueError):
            validate_compiler_equality(left, right)

    def test_compiler_wrapper_bytes_and_injected_flags_are_bound(self):
        directory = self.output("output")
        left = self.capture(directory)
        with self.assertRaisesRegex(ValueError, "environment"):
            validate_compiler_equality(left, self.capture(directory, NIX_CFLAGS_COMPILE="-Ofast"))
        with self.assertRaisesRegex(ValueError, "environment"):
            validate_compiler_equality(left, self.capture(directory, NIX_CFLAGS_COMPILE_x86_64_linux="-Ofast"))
        self.compiler.write_text(self.compiler.read_text() + "# wrapper changed\n")
        with self.assertRaisesRegex(ValueError, "effective compiler flags"):
            validate_compiler_equality(left, self.capture(directory))

    def test_unsupported_response_file_and_missing_recipe_fail_closed(self):
        directory = self.output("output")
        with self.assertRaisesRegex(ValueError, "response files"):
            self.capture(directory, GLOBAL_FLAG="@hidden-flags.rsp")
        path = directory / "makefile"
        path.write_text(path.read_text() + "check_sa.o:\n\ttrue\n")
        with self.assertRaisesRegex(ValueError, "Unresolved"):
            self.capture(directory)

    def test_missing_legacy_evidence_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing or incompatible"):
            validate_compiler_equality({}, {})


if __name__ == "__main__":
    unittest.main()
