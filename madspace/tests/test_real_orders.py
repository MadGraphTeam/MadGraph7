"""Check the local-order axis and per-event couplings through the CPU graph."""
from pathlib import Path
import subprocess
import tempfile
import unittest

import madspace as ms
import numpy as np


class RealOrdersTest(unittest.TestCase):
    def test_provider_order_axis(self):
        source = r'''
#include "madspace/umami.h"
UmamiStatus umami_get_meta(UmamiMetaKey key, void* out) {
    if (key == UMAMI_META_DEVICE) *static_cast<UmamiDevice*>(out) = UMAMI_DEVICE_CPU;
    else if (key == UMAMI_META_SQUARED_ORDER_COUNT) *static_cast<int*>(out) = 2;
    else if (key == UMAMI_META_PARTICLE_COUNT || key == UMAMI_META_DIAGRAM_COUNT)
        *static_cast<int*>(out) = 1;
    else return UMAMI_ERROR_UNSUPPORTED_META;
    return UMAMI_SUCCESS;
}
UmamiStatus umami_initialize(UmamiHandle* handle, const char*) {
    *handle = nullptr;
    return UMAMI_SUCCESS;
}
UmamiStatus umami_free(UmamiHandle) { return UMAMI_SUCCESS; }
UmamiStatus umami_matrix_element(UmamiHandle, size_t count, size_t stride, size_t offset,
    size_t ni, const UmamiInputKey* ik, const void* const* in,
    size_t no, const UmamiOutputKey* ok, void* const* out) {
    if (ni != 2 || ik[0] != UMAMI_IN_G_STRONG || ik[1] != UMAMI_IN_G_EW ||
        no != 1 || ok[0] != UMAMI_OUT_SQUARED_ORDERS) return UMAMI_ERROR;
    for (size_t j=0; j<2; ++j)
        for (size_t i=offset; i<offset+count; ++i)
            static_cast<double*>(out[0])[stride*j+i] = static_cast<const double*>(in[j])[i];
    return UMAMI_SUCCESS;
}
'''
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cpp = root/'provider.cc'
            cpp.write_text(source)
            library = root/'provider.so'
            subprocess.run(['c++', '-shared','-fPIC','-I',str(Path(__file__).resolve().parents[1]/'include'),
                            str(cpp),'-o',str(library)], check=True)
            ctx = ms.Context(ms.cpu_device(), 2)
            api = ctx.load_matrix_element(str(library), '')
            self.assertEqual(api.squared_order_count(), 2)
            inputs = [ms.MatrixElement.g_strong_in, ms.MatrixElement.g_ew_in]
            outputs = [ms.MatrixElement.squared_orders_out]
            elem = ms.MatrixElement(api, inputs, outputs)
            strong = np.arange(257, dtype=np.float64)
            ew = strong + 0.75
            result = ms.FunctionRuntime(elem.function(), ctx)(strong, ew)
            np.testing.assert_array_equal(result, np.stack((strong, ew), axis=1))
            wrong = ms.MatrixElement(api.index(), 1, inputs, outputs, squared_order_count=1)
            with self.assertRaisesRegex((RuntimeError, ValueError), 'extent'):
                ms.FunctionRuntime(wrong.function(), ctx)(strong, ew)


if __name__ == '__main__':
    unittest.main()
