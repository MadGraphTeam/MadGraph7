#pragma once

// GPU gzip compression library behind gzip.cu. CMake defines MADSPACE_GPUCOMP when
// the backend has one; without it the LHE output is gzipped on the CPU.

#if defined(__CUDACC__)

#define GPUCOMP_FALLBACK_REASON "madspace built without nvCOMP"

#ifdef MADSPACE_GPUCOMP
#include <nvcomp/gzip.h>

#define gpucompStatus_t nvcompStatus_t
#define gpucompSuccess nvcompSuccess
#define gpucompAlignmentRequirements_t nvcompAlignmentRequirements_t
#define gpucompBatchedGzipCompressOpts_t nvcompBatchedGzipCompressOpts_t
#define gpucompBatchedGzipCompressDefaultOpts nvcompBatchedGzipCompressDefaultOpts
#define gpucompBatchedGzipCompressGetRequiredAlignments                                \
    nvcompBatchedGzipCompressGetRequiredAlignments
#define gpucompBatchedGzipCompressGetMaxOutputChunkSize                                \
    nvcompBatchedGzipCompressGetMaxOutputChunkSize
#define gpucompBatchedGzipCompressGetTempSizeAsync                                     \
    nvcompBatchedGzipCompressGetTempSizeAsync
#define gpucompBatchedGzipCompressAsync nvcompBatchedGzipCompressAsync
#endif

#elif defined(__HIPCC__)

// hipCOMP only decompresses gzip. Once it compresses, map its names here like nvCOMP
// and define MADSPACE_GPUCOMP for madspace_hip in CMakeLists.txt.
#define GPUCOMP_FALLBACK_REASON "hipCOMP gzip compression not supported"

#endif
