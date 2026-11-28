"""Reject the colliding experimental ABI before a provider can write outputs."""

import ctypes
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import madspace as ms
import numpy as np
import pytest


pytestmark = pytest.mark.cpu_only


@unittest.skipUnless(shutil.which("c++"), "requires a C++ compiler")
class UmamiAbiCompatibilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="umami-abi-")
        cls.root = Path(cls.directory.name)
        cls.source = cls.root / "provider.cc"
        cls.source.write_text(r'''
#include "madspace/umami.h"
static int initialize_calls = 0;
static int evaluate_calls = 0;
extern "C" int test_initialize_calls() { return initialize_calls; }
extern "C" int test_evaluate_calls() { return evaluate_calls; }
UmamiStatus umami_get_meta(UmamiMetaKey key, void* result) {
    switch (key) {
    case UMAMI_META_DEVICE:
        *static_cast<UmamiDevice*>(result) = UMAMI_DEVICE_CPU;
        return UMAMI_SUCCESS;
    case UMAMI_META_PARTICLE_COUNT:
    case UMAMI_META_DIAGRAM_COUNT:
        *static_cast<int*>(result) = 1;
        return UMAMI_SUCCESS;
#if REPORTED_MINOR >= 0
    case UMAMI_META_ABI_MAJOR_VERSION:
        *static_cast<int*>(result) = 1;
        return UMAMI_SUCCESS;
    case UMAMI_META_ABI_MINOR_VERSION:
        *static_cast<int*>(result) = REPORTED_MINOR;
        return UMAMI_SUCCESS;
#endif
    default: return UMAMI_ERROR_UNSUPPORTED_META;
    }
}
UmamiStatus umami_supported_inputs(const bool** supported, int* count) {
    static bool keys[UMAMI_INPUT_KEY_COUNT] = {};
    keys[15] = true;
    keys[16] = true;
    *supported = keys;
    *count = REPORTED_MINOR <= 4 ? 17 : UMAMI_INPUT_KEY_COUNT;
    return UMAMI_SUCCESS;
}
UmamiStatus umami_supported_outputs(const bool** supported, int* count) {
    static bool keys[UMAMI_OUTPUT_KEY_COUNT] = {};
    keys[13] = true;
    *supported = keys;
    *count = REPORTED_MINOR <= 4 ? 14 : UMAMI_OUTPUT_KEY_COUNT;
    return UMAMI_SUCCESS;
}
UmamiStatus umami_initialize(UmamiHandle* handle, const char*) {
    ++initialize_calls;
    *handle = nullptr;
    return UMAMI_SUCCESS;
}
UmamiStatus umami_free(UmamiHandle) { return UMAMI_SUCCESS; }
UmamiStatus umami_matrix_element(UmamiHandle, size_t count, size_t, size_t offset,
    size_t ni, const UmamiInputKey* ik, const void* const* in,
    size_t no, const UmamiOutputKey* ok, void* const* out) {
    ++evaluate_calls;
    const double* random_flavor = nullptr;
    for (size_t k = 0; k < ni; ++k) {
        if (ik[k] == 15) random_flavor = static_cast<const double*>(in[k]);
        else if (ik[k] != 16) return UMAMI_ERROR_UNSUPPORTED_INPUT;
    }
    if (!random_flavor) return UMAMI_ERROR_MISSING_INPUT;
    if (no != 1 || ok[0] != 13) return UMAMI_ERROR_UNSUPPORTED_OUTPUT;
    for (size_t i = offset; i < offset + count; ++i)
        static_cast<int*>(out[0])[i] = random_flavor[i] > .5;
    return UMAMI_SUCCESS;
}
''')
        include = Path(__file__).resolve().parents[1] / "include"
        cls.libraries = {}
        for minor in (-1, 3, 4, 5):
            library = cls.root / ("provider_%s.so" % minor)
            subprocess.run([
                "c++", "-shared", "-fPIC", "-I", str(include),
                "-DREPORTED_MINOR=%s" % minor, str(cls.source), "-o", str(library),
            ], check=True)
            cls.libraries[minor] = library

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_experimental_1_4_is_rejected_before_initialization(self):
        library = ctypes.CDLL(str(self.libraries[4]))
        context = ms.Context(ms.cpu_device(), 1)
        with self.assertRaisesRegex(RuntimeError, "experimental UMAMI 1.4.*regenerate"):
            context.load_matrix_element(str(self.libraries[4]), "")
        self.assertEqual(library.test_initialize_calls(), 0)
        self.assertEqual(library.test_evaluate_calls(), 0)
        with self.assertRaisesRegex(RuntimeError, "out of bounds"):
            context.matrix_element(0)

    def test_public_1_3_and_current_1_5_remain_loadable(self):
        for minor in (3, 5):
            with self.subTest(minor=minor):
                library = ctypes.CDLL(str(self.libraries[minor]))
                before = library.test_initialize_calls()
                context = ms.Context(ms.cpu_device(), 1)
                api = context.load_matrix_element(str(self.libraries[minor]), "")
                self.assertEqual(api.particle_count(), 1)
                self.assertTrue(api.supported_inputs()[15])
                self.assertTrue(api.supported_inputs()[16])
                self.assertTrue(api.supported_outputs()[13])
                self.assertFalse(api.supported_inputs()[17])
                self.assertFalse(api.supported_outputs()[14])
                self.assertGreater(library.test_initialize_calls(), before)
                self.assertEqual(library.test_evaluate_calls(), 0)
                element = ms.MatrixElement(
                    api, [ms.MatrixElement.random_flavor_in],
                    [ms.MatrixElement.flavor_index_out])
                result = ms.FunctionRuntime(element.function(), context)(
                    np.array([.1, .9, .5, .8]))
                np.testing.assert_array_equal(result, np.array([0, 1, 0, 1]))
                self.assertEqual(result.dtype, np.dtype("int32"))

    def test_public_provider_without_abi_metadata_remains_loadable(self):
        context = ms.Context(ms.cpu_device(), 1)
        api = context.load_matrix_element(str(self.libraries[-1]), "")
        self.assertEqual(api.particle_count(), 1)


if __name__ == "__main__":
    unittest.main()
