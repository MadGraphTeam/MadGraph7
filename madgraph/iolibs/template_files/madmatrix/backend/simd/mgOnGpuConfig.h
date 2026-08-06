// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created originally by: A. Valassi (Jul 2020) for the MG5aMC CUDACPP plugin.
// Further modified by: S. Hageboeck, O. Mattelaer, S. Roiser, J. Teig, A. Valassi (2020-2024).
// Integrated with the MadGraph7 project in Feb 2026.

#ifndef MGONGPUCONFIG_H
#define MGONGPUCONFIG_H 1

// simd backend: always built with a plain host compiler, never nvcc/hipcc, so the
// GPU-backend selection macro (see gpu/mgOnGpuConfig.h) is deliberately never
// defined here - single-file, all-backend headers rely on that to pick branches.

// Choose floating point precision (for everything but color algebra #537)
// If set from outside with e.g. -DMGONGPU_FPTYPE_FLOAT, nothing happens (#167)
#if not defined MGONGPU_FPTYPE_DOUBLE and not defined MGONGPU_FPTYPE_FLOAT
#define MGONGPU_FPTYPE_DOUBLE 1 // default
//#define MGONGPU_FPTYPE_FLOAT 1 // 2x faster
#endif

// Choose floating point precision (for color algebra alone #537)
#if not defined MGONGPU_FPTYPE2_DOUBLE and not defined MGONGPU_FPTYPE2_FLOAT
#define MGONGPU_FPTYPE2_DOUBLE 1 // default
//#define MGONGPU_FPTYPE2_FLOAT 1 // 2x faster
#endif

// Choose whether to inline all HelAmps functions (can gain ~4x, issue #229)
// By default off; set from outside with -DMGONGPU_INLINE_HELAMPS
//#define MGONGPU_INLINE_HELAMPS 1

// Choose whether to hardcode cIPD physics parameters instead of reading user cards
// By default off; set from outside with -DMGONGPU_HARDCODE_PARAM
//#define MGONGPU_HARDCODE_PARAM 1

// Complex type in C++: cxsmpl by default, or std::complex (CHOOSE ONLY ONE)
//#define MGONGPU_CPPCXTYPE_STDCOMPLEX 1 // ~8% slower on float, same on double
#define MGONGPU_CPPCXTYPE_CXSMPL 1 // default

// No BLAS on the simd backend (cuBLAS/hipBLAS are GPU-only)
#define MGONGPU_HAS_NO_BLAS 1

// nsight compute (ncu) debugging is CUDA-only; always off here
#undef MGONGPU_NSIGHT_DEBUG

// SANITY CHECKS
#if defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE_FLOAT
#error You must CHOOSE (ONE AND) ONLY ONE of MGONGPU_FPTYPE_DOUBLE or MGONGPU_FPTYPE_FLOAT
#endif
#if defined MGONGPU_FPTYPE2_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
#error You must CHOOSE (ONE AND) ONLY ONE of MGONGPU_FPTYPE2_DOUBLE or MGONGPU_FPTYPE2_FLOAT
#endif
#if defined MGONGPU_FPTYPE2_DOUBLE and defined MGONGPU_FPTYPE_FLOAT
#error You cannot use double precision for color algebra and single precision elsewhere
#endif
#if defined MGONGPU_CPPCXTYPE_STDCOMPLEX and defined MGONGPU_CPPCXTYPE_CXSMPL
#error You must CHOOSE (ONE AND) ONLY ONE of MGONGPU_CPPCXTYPE_STDCOMPLEX or MGONGPU_CPPCXTYPE_CXSMPL for C++
#endif

// NB: namespace mgOnGpu includes types which are defined in exactly the same way for CPU and GPU builds (see #318 and #725)
namespace mgOnGpu
{
  // Floating point type (for everything but color algebra #537): fptype
#if defined MGONGPU_FPTYPE_DOUBLE
  typedef double fptype; // double precision (8 bytes, fp64)
#elif defined MGONGPU_FPTYPE_FLOAT
  typedef float fptype;  // single precision (4 bytes, fp32)
#endif

  // Floating point type (for color algebra alone #537): fptype2
#if defined MGONGPU_FPTYPE2_DOUBLE
  typedef double fptype2; // double precision (8 bytes, fp64)
#elif defined MGONGPU_FPTYPE2_FLOAT
  typedef float fptype2; // single precision (4 bytes, fp32)
#endif

  // Maximum number of threads per block
  const int ntpbMAX = 1024; // NB: 512 is ok, but 1024 does fail with "too many resources requested for launch"

  // Alignment requirement for using reinterpret_cast with SIMD vectorized code
  constexpr int cppAlign = 64; // 64-byte i.e. 512-bit
}

// Expose typedefs outside the namespace
using mgOnGpu::fptype;
using mgOnGpu::fptype2;

// Undefine ARM_NEON (hack for cppnone on Apple silicon ARM)
#ifdef MGONGPU_NOARMNEON
#undef __ARM_NEON
#endif

// C++ SIMD vectorization width (this will be used to set neppV)
#if defined __AVX512VL__ && defined MGONGPU_PVW512 // "512z" AVX512 512-bit: 8 (DOUBLE) or 16 (FLOAT)
#ifdef MGONGPU_FPTYPE_DOUBLE
#define MGONGPU_CPPSIMD 8
#else
#define MGONGPU_CPPSIMD 16
#endif
#elif defined __AVX512VL__ // "512y" AVX512 256-bit: 4 (DOUBLE) or 8 (FLOAT) [gcc default]
#ifdef MGONGPU_FPTYPE_DOUBLE
#define MGONGPU_CPPSIMD 4
#else
#define MGONGPU_CPPSIMD 8
#endif
#elif defined __AVX2__ // "avx2" 256-bit: 4 (DOUBLE) or 8 (FLOAT) [clang default]
#ifdef MGONGPU_FPTYPE_DOUBLE
#define MGONGPU_CPPSIMD 4
#else
#define MGONGPU_CPPSIMD 8
#endif
#elif defined __SSE4_2__ // "sse4" SSE4.2 128-bit: 2 (DOUBLE) or 4 (FLOAT) [Power9 default]
#ifdef MGONGPU_FPTYPE_DOUBLE
#define MGONGPU_CPPSIMD 2
#else
#define MGONGPU_CPPSIMD 4
#endif
#elif defined __ARM_NEON // ARM NEON 128-bit: 2 (DOUBLE) or 4 (FLOAT) [ARM default]
#ifdef MGONGPU_FPTYPE_DOUBLE
#define MGONGPU_CPPSIMD 2
#else
#define MGONGPU_CPPSIMD 4
#endif
#else // "none" i.e. no SIMD
#undef MGONGPU_CPPSIMD
#endif

// No-op debug macros (nsight-based debugging is CUDA-only, unused here)
#define mgDebugDeclare() /*noop*/
#define mgDebugInitialise() /*noop*/
#define mgDebug( code, text ) /*noop*/
#define mgDebugFinalise() /*noop*/

// Define empty CUDA/HIP declaration specifiers for C++
#define __global__
#define __host__
#define __device__

// For SANITY CHECKS: check that neppR, neppM, neppV... are powers of two
inline constexpr bool
ispoweroftwo( int n )
{
  return ( n > 0 ) && !( n & ( n - 1 ) );
}

// Compiler version support (#96)
#if defined __clang__
#if( __clang_major__ < 11 )
#error Unsupported clang version: please use clang >= 11
#endif
#elif defined __GNUC__
#if( __GNUC__ < 9 ) || ( __GNUC__ == 9 && __GNUC_MINOR__ < 3 )
#error Unsupported gcc version: please gcc >= 9.3
#endif
#endif

#endif // MGONGPUCONFIG_H
