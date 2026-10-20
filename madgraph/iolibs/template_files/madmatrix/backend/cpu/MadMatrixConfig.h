// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created originally by: A. Valassi (Jul 2020) for the MadGraph7 CUDACPP plugin.
// Further modified by: S. Hageboeck, O. Mattelaer, S. Roiser, J. Teig, A. Valassi (2020-2024).
// Integrated with the MadGraph7 project in Feb 2026.

#ifndef MADMATRIXCONFIG_H
#define MADMATRIXCONFIG_H 1

#include <sstream>
#include <string>

// cpu backend: always built with a plain host compiler, never nvcc/hipcc, so the
// GPU-backend selection macro (see gpu/MadMatrixConfig.h) is deliberately never
// defined here - single-file, all-backend headers rely on that to pick branches.

// Choose floating point precision (for everything but color algebra #537)
// If set from outside with e.g. -DMADMATRIX_FPTYPE_FLOAT, nothing happens (#167)
#if not defined MADMATRIX_FPTYPE_DOUBLE and not defined MADMATRIX_FPTYPE_FLOAT
#define MADMATRIX_FPTYPE_DOUBLE 1 // default
//#define MADMATRIX_FPTYPE_FLOAT 1 // 2x faster
#endif

// Choose floating point precision (for color algebra alone #537)
#if not defined MADMATRIX_FPTYPE2_DOUBLE and not defined MADMATRIX_FPTYPE2_FLOAT
#define MADMATRIX_FPTYPE2_DOUBLE 1 // default
//#define MADMATRIX_FPTYPE2_FLOAT 1 // 2x faster
#endif

// Choose whether to inline all HelAmps functions (can gain ~4x, issue #229)
// By default off; set from outside with -DMADMATRIX_INLINE_HELAMPS
//#define MADMATRIX_INLINE_HELAMPS 1

// Choose whether to hardcode cIPD physics parameters instead of reading user cards
// By default off; set from outside with -DMADMATRIX_HARDCODE_PARAM
//#define MADMATRIX_HARDCODE_PARAM 1

// Complex type in C++: cxsmpl by default, or std::complex (CHOOSE ONLY ONE)
//#define MADMATRIX_CPPCXTYPE_STDCOMPLEX 1 // ~8% slower on float, same on double
#define MADMATRIX_CPPCXTYPE_CXSMPL 1 // default

// No BLAS on the cpu backend (cuBLAS/hipBLAS are GPU-only)
#define MADMATRIX_HAS_NO_BLAS 1

// SANITY CHECKS
#if defined MADMATRIX_FPTYPE_DOUBLE and defined MADMATRIX_FPTYPE_FLOAT
#error You must CHOOSE (ONE AND) ONLY ONE of MADMATRIX_FPTYPE_DOUBLE or MADMATRIX_FPTYPE_FLOAT
#endif
#if defined MADMATRIX_FPTYPE2_DOUBLE and defined MADMATRIX_FPTYPE2_FLOAT
#error You must CHOOSE (ONE AND) ONLY ONE of MADMATRIX_FPTYPE2_DOUBLE or MADMATRIX_FPTYPE2_FLOAT
#endif
#if defined MADMATRIX_FPTYPE2_DOUBLE and defined MADMATRIX_FPTYPE_FLOAT
#error You cannot use double precision for color algebra and single precision elsewhere
#endif
#if defined MADMATRIX_CPPCXTYPE_STDCOMPLEX and defined MADMATRIX_CPPCXTYPE_CXSMPL
#error You must CHOOSE (ONE AND) ONLY ONE of MADMATRIX_CPPCXTYPE_STDCOMPLEX or MADMATRIX_CPPCXTYPE_CXSMPL for C++
#endif

// NB: namespace madmatrix includes types which are defined in exactly the same way for CPU and GPU builds (see #318 and #725)
namespace madmatrix
{
  // Floating point type (for everything but color algebra #537): fptype
#if defined MADMATRIX_FPTYPE_DOUBLE
  typedef double fptype; // double precision (8 bytes, fp64)
#elif defined MADMATRIX_FPTYPE_FLOAT
  typedef float fptype;  // single precision (4 bytes, fp32)
#endif

  // Floating point type (for color algebra alone #537): fptype2
#if defined MADMATRIX_FPTYPE2_DOUBLE
  typedef double fptype2; // double precision (8 bytes, fp64)
#elif defined MADMATRIX_FPTYPE2_FLOAT
  typedef float fptype2; // single precision (4 bytes, fp32)
#endif

  // --- Mixed-precision stage types (3 independent precisions) ---
  // fptype       (== fptype_amp): wavefunctions, helicity amplitudes, ME, vertex
  //                               and polarization computation (MADMATRIX_FPTYPE_*)
  // fptype_momenta (== fptype_denom): momenta storage and in-vertex denominators
  //                               (MADMATRIX_FPTYPE_MOMENTA_*, default fptype)
  // fptype2      (== fptype_colour): color algebra alone (MADMATRIX_FPTYPE2_*)

#if defined MADMATRIX_FPTYPE_MOMENTA_DOUBLE
  typedef double fptype_momenta;
#elif defined MADMATRIX_FPTYPE_MOMENTA_FLOAT
  typedef float fptype_momenta;
#else
  typedef fptype fptype_momenta;
#endif
  typedef fptype_momenta fptype_denom; // denominator precision == momenta precision
  typedef fptype fptype_amp;           // amplitudes/wavefunctions == fptype
  typedef fptype2 fptype_colour;       // color algebra == fptype2

  // Valid precision ordering: colour <= amp <= momenta (4=fp32, 8=fp64)
  static_assert( sizeof( fptype_colour ) <= sizeof( fptype_amp ), "colour precision must not exceed amp precision" );
  static_assert( sizeof( fptype_amp ) <= sizeof( fptype_momenta ), "amp precision must not exceed momenta precision" );

  // Maximum number of threads per block
  const int ntpbMAX = 1024; // NB: 512 is ok, but 1024 does fail with "too many resources requested for launch"

  // Alignment requirement for using reinterpret_cast with SIMD vectorized code
  constexpr int cppAlign = 64; // 64-byte i.e. 512-bit

  // Retrieve the compiler that was used to build this module
  inline std::string
  getCompiler()
  {
    std::stringstream out;
    // ICX version
#if defined __INTEL_COMPILER
#error "icc is no longer supported: please use icx"
#elif defined __INTEL_LLVM_COMPILER // alternative: __INTEL_CLANG_COMPILER
    out << "icx " << __INTEL_LLVM_COMPILER;
    out << " (";
#endif
    // CLANG version (either as CXX or as host compiler inside ICX)
#if defined __clang__
#if defined __clang_major__ && defined __clang_minor__ && defined __clang_patchlevel__
#ifdef __APPLE__
    out << "Apple clang " << __clang_major__ << "." << __clang_minor__ << "." << __clang_patchlevel__;
#else
    out << "clang " << __clang_major__ << "." << __clang_minor__ << "." << __clang_patchlevel__;
#endif
#else
    out << "clang UNKNOWKN";
#endif
#else
    // GCC version
#if defined __GNUC__ && defined __GNUC_MINOR__ && defined __GNUC_PATCHLEVEL__
    out << "gcc " << __GNUC__ << "." << __GNUC_MINOR__ << "." << __GNUC_PATCHLEVEL__;
#else
    out << "gcc UNKNOWKN";
#endif
#endif
#if defined __INTEL_LLVM_COMPILER
    out << ")";
#endif
    return out.str();
  }
}

// Expose typedefs outside the namespace
using madmatrix::fptype;
using madmatrix::fptype2;
using madmatrix::fptype_momenta;
using madmatrix::fptype_denom;
using madmatrix::fptype_amp;
using madmatrix::fptype_colour;

// Undefine ARM_NEON (hack for the 'scalar' backend on Apple silicon ARM)
#ifdef MADMATRIX_NOARMNEON
#undef __ARM_NEON
#endif
#undef MADMATRIX_SIMD


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

#endif // MADMATRIXCONFIG_H
