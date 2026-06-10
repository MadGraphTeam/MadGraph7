// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created originally by: A. Valassi (Jul 2020) for the MG5aMC CUDACPP plugin.
// Further modified by: S. Hageboeck, O. Mattelaer, S. Roiser, J. Teig, A. Valassi (2020-2024).
// Integrated with the MadGraph7 project in Feb 2026.

#ifndef MGONGPUCONFIG_H
#define MGONGPUCONFIG_H 1

// Is this a GPU (CUDA, HIP) or CPU implementation?
#ifdef __CUDACC__ // this must be __CUDACC__ (not MGONGPUCPP_GPUIMPL)
#define MGONGPUCPP_GPUIMPL cuda
#elif defined __HIPCC__
#define MGONGPUCPP_GPUIMPL hip
#include "hip/hip_runtime.h" // needed for blockDim, blockIdx, threadIdx: better in mgOnGpuConfig.h than in GpuAbstraction.h
#else
#undef MGONGPUCPP_GPUIMPL
#endif

// Make sure that __HIP_PLATFORM_NVIDIA__ is undefined
// (__HIP_PLATFORM_AMD__ is defined by hipcc or in HiprandRandomNumberKernel.cc)
#undef __HIP_PLATFORM_NVIDIA__ // disable hiprand for NVidia (curand)

// ** NB1 Throughputs (e.g. 6.8E8) are events/sec for "./gcheck.exe -p 65536 128 12"
// ** NB2 Baseline on b7g47n0004 fluctuates (probably depends on load on other VMs)

// Choose if curand is supported for generating random numbers
// For HIP, by default, do not allow curand to be used (hiprand or common random numbers will be used instead)
// For both CUDA and C++, by default, do not inline, but allow this macro to be set from outside with e.g. -DMGONGPU_HAS_NO_CURAND
// (there exist CUDA installations, e.g. using the HPC package, which do not include curand - see PR #784 and #785)
#if defined __HIPCC__
#define MGONGPU_HAS_NO_CURAND 1
#else
//#ifdef __CUDACC__ // this must be __CUDACC__ (not MGONGPUCPP_GPUIMPL)
//#undef MGONGPU_HAS_NO_CURAND // default
////#define MGONGPU_HAS_NO_CURAND 1
//#else
//#undef MGONGPU_HAS_NO_CURAND // default
////#define MGONGPU_HAS_NO_CURAND 1
//#endif
#endif

// Choose if hiprand is supported for generating random numbers
// For CUDA, by default, do not allow hiprand to be used (curand or common random numbers will be used instead)
// For both HIP and C++, by default, do not inline, but allow this macro to be set from outside with e.g. -DMGONGPU_HAS_NO_HIPRAND
// (there may exist HIP installations which do not include hiprand?)
#if defined __CUDACC__ // this must be __CUDACC__ (not MGONGPUCPP_GPUIMPL)
#define MGONGPU_HAS_NO_HIPRAND 1
#else
//#ifdef __HIPCC__
//#undef MGONGPU_HAS_NO_HIPRAND // default
////#define MGONGPU_HAS_NO_HIPRAND 1
//#else
//#undef MGONGPU_HAS_NO_HIPRAND // default
////#define MGONGPU_HAS_NO_HIPRAND 1
//#endif
#endif

// Choose floating point precision (for everything but color algebra #537)
// If one of these macros has been set from outside with e.g. -DMGONGPU_FPTYPE_FLOAT, nothing happens (issue #167)
#if not defined MGONGPU_FPTYPE_DOUBLE and not defined MGONGPU_FPTYPE_FLOAT
// Floating point precision (CHOOSE ONLY ONE)
#define MGONGPU_FPTYPE_DOUBLE 1 // default
//#define MGONGPU_FPTYPE_FLOAT 1 // 2x faster
#endif

// Choose floating point precision (for color algebra alone #537)
// If one of these macros has been set from outside with e.g. -DMGONGPU_FPTYPE2_FLOAT, nothing happens (issue #167)
#if not defined MGONGPU_FPTYPE2_DOUBLE and not defined MGONGPU_FPTYPE2_FLOAT
// Floating point precision (CHOOSE ONLY ONE)
#define MGONGPU_FPTYPE2_DOUBLE 1 // default
//#define MGONGPU_FPTYPE2_FLOAT 1 // 2x faster
#endif

// Choose whether to inline all HelAmps functions
// This optimization can gain almost a factor 4 in C++, similar to -flto (issue #229)
// By default, do not inline, but allow this macro to be set from outside with e.g. -DMGONGPU_INLINE_HELAMPS
//#undef MGONGPU_INLINE_HELAMPS // default
////#define MGONGPU_INLINE_HELAMPS 1

// Choose whether to hardcode the cIPD physics parameters rather than reading them from user cards
// This optimization can gain 20%% in CUDA in eemumu (issue #39)
// By default, do not hardcode, but allow this macro to be set from outside with e.g. -DMGONGPU_HARDCODE_PARAM
// ** NB: The option to use hardcoded cIPD physics parameters is supported again even now when alphas is running (#373)
// ** NB: Note however that it now only refers to cIPD parameters (cIPC parameters are always accessed through global memory)
//#undef MGONGPU_HARDCODE_PARAM // default
////#define MGONGPU_HARDCODE_PARAM 1

/* clang-format off */
// Complex type in CUDA: thrust or cucomplex or cxsmpl (CHOOSE ONLY ONE)
// (NB THIS IS MGONGPU_*CU*CXTYPE_xxx)
#ifdef __CUDACC__ // this must be __CUDACC__ (not MGONGPUCPP_GPUIMPL)
#define MGONGPU_CUCXTYPE_THRUST 1 // default (~1.15E9/double, ~3.2E9/float)
//#define MGONGPU_CUCXTYPE_CUCOMPLEX 1 // ~10 percent slower (1.03E9/double, ~2.8E9/float)
//#define MGONGPU_CUCXTYPE_CXSMPL 1 // ~10 percent slower (1.00E9/double, ~2.9E9/float)

// Complex type in HIP: cxsmpl (ONLY ONE OPTION POSSIBLE? #810)
// (NB THIS IS MGONGPU_*HIP*CXTYPE_xxx)
#elif defined __HIPCC__
#define MGONGPU_HIPCXTYPE_CXSMPL 1 // default for HIP

// Complex type in C++: std::complex or cxsmpl (CHOOSE ONLY ONE)
// (NB THIS IS MGONGPU_*CPP*CXTYPE_xxx)
#else
//#define MGONGPU_CPPCXTYPE_STDCOMPLEX 1 // ~8 percent slower on float, same on double (5.1E6/double, 9.4E6/float)
#define MGONGPU_CPPCXTYPE_CXSMPL 1 // new default (5.1E6/double, 10.2E6/float)
#endif

// Choose if cuBLAS and hipBLAS are supported for generating random numbers
// For both CUDA and HIP, by default, do not inline, but allow this macro to be set from outside with e.g. -DMGONGPU_HAS_NO_BLAS
// (there may exist CUDA/HIP installations, e.g. using the HPC package, which do not include cuBLAS/hipBLAS?)
#ifdef __CUDACC__ // this must be __CUDACC__ (not MGONGPUCPP_GPUIMPL)
//#undef MGONGPU_HAS_NO_BLAS // default
////#define MGONGPU_HAS_NO_BLAS 1
#elif defined __HIPCC__
//#undef MGONGPU_HAS_NO_BLAS // default
////#define MGONGPU_HAS_NO_BLAS 1
#else
#define MGONGPU_HAS_NO_BLAS 1
#endif

// CUDA nsight compute (ncu) debug: add dummy lines to ease SASS program flow navigation
#ifdef __CUDACC__ // this must be __CUDACC__ (not MGONGPUCPP_GPUIMPL)
#undef MGONGPU_NSIGHT_DEBUG // default in CUDA
//#define MGONGPU_NSIGHT_DEBUG 1 // CURRENTLY NO LONGER SUPPORTED!
#else
#undef MGONGPU_NSIGHT_DEBUG // only option in HIP or C++
#endif /* clang-format on */

// SANITY CHECKS (floating point precision for everything but color algebra #537)
#if defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE_FLOAT
#error You must CHOOSE (ONE AND) ONLY ONE of MGONGPU_FPTYPE_DOUBLE or defined MGONGPU_FPTYPE_FLOAT
#endif

// SANITY CHECKS (floating point precision for color algebra alone #537)
#if defined MGONGPU_FPTYPE2_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
#error You must CHOOSE (ONE AND) ONLY ONE of MGONGPU_FPTYPE2_DOUBLE or defined MGONGPU_FPTYPE2_FLOAT
#endif
#if defined MGONGPU_FPTYPE2_DOUBLE and defined MGONGPU_FPTYPE_FLOAT
#error You cannot use double precision for color algebra and single precision elsewhere
#endif

// SANITY CHECKS (CUDA complex number implementation)
#ifdef __CUDACC__ // this must be __CUDACC__ (not MGONGPUCPP_GPUIMPL)
#if defined MGONGPU_CUCXTYPE_THRUST and defined MGONGPU_CUCXTYPE_CUCOMPLEX
#error You must CHOOSE (ONE AND) ONLY ONE of MGONGPU_CUCXTYPE_THRUST or MGONGPU_CUCXTYPE_CUCOMPLEX for CUDA
#elif defined MGONGPU_CUCXTYPE_THRUST and defined MGONGPU_CUCXTYPE_CXSMPL
#error You must CHOOSE (ONE AND) ONLY ONE of MGONGPU_CUCXTYPE_THRUST or MGONGPU_CUCXTYPE_CXSMPL for CUDA
#elif defined MGONGPU_CUCXTYPE_CUCOMPLEX and defined MGONGPU_CUCXTYPE_CXSMPL
#error You must CHOOSE (ONE AND) ONLY ONE OF MGONGPU_CUCXTYPE_CUCOMPLEX or MGONGPU_CUCXTYPE_CXSMPL for CUDA
#endif
#endif

// SANITY CHECKS (C++ complex number implementation)
#ifndef MGONGPUCPP_GPUIMPL
#if defined MGONGPU_CPPCXTYPE_STDCOMPLEX and defined MGONGPU_CPPCXTYPE_CXSMPL
#error You must CHOOSE (ONE AND) ONLY ONE of MGONGPU_CPPCXTYPE_STDCOMPLEX or MGONGPU_CPPCXTYPE_CXSMPL for C++
#endif
#endif

// NB: namespace mgOnGpu includes types which are defined in exactly the same way for CPU and GPU builds (see #318 and #725)
namespace mgOnGpu
{

  // --- Type definitions

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

  // --- Multi-precision stage-specific types (madmatrix precision experiment) ---
  // Each stage of the calculation can run in a different precision than the global
  // fptype. fptype_polarization and fptype_invmass DEFAULT TO double so that a build
  // with FPTYPE=f keeps the ixxxxx/vxxxxx polarization and the offshell-propagator
  // virtuality (m_ij/FIXP2) in double while everything else is float. The remaining
  // stage types default to fptype/fptype2 (backward compatible). Each is overridable
  // via its own macro.

  // fptype_momenta: for momenta storage and polarization vector inputs
#if defined MGONGPU_FPTYPE_MOMENTA_DOUBLE
  typedef double fptype_momenta;
#elif defined MGONGPU_FPTYPE_MOMENTA_FLOAT
  typedef float fptype_momenta;
#else
  typedef fptype fptype_momenta;
#endif

  // fptype_polarization: precision of the ixxxxx/vxxxxx polarization (wavefunction) computation
#if defined MGONGPU_FPTYPE_POLARIZATION_DOUBLE
  typedef double fptype_polarization;
#elif defined MGONGPU_FPTYPE_POLARIZATION_FLOAT
  typedef float fptype_polarization;
#else
  typedef double fptype_polarization; // default: double (madmatrix precision experiment)
#endif

  // fptype_invmass: precision of the offshell-propagator virtuality (m_ij / FIXP2) and of the
  // propagator denominator computed from it. Default double. NB: unlike the other stage
  // types this one is only wired for scalar builds (cppnone / CUDA); in a SIMD build it must
  // equal fptype (a static_assert at the end of this file enforces it).
#if defined MGONGPU_FPTYPE_INVMASS_DOUBLE
  typedef double fptype_invmass;
#elif defined MGONGPU_FPTYPE_INVMASS_FLOAT
  typedef float fptype_invmass;
#else
  typedef double fptype_invmass; // default: double
#endif

  // fptype_vertex: for VVF vertex function (FFV1_0, FFV1P0_3) internal computation
#if defined MGONGPU_FPTYPE_VERTEX_DOUBLE
  typedef double fptype_vertex;
#elif defined MGONGPU_FPTYPE_VERTEX_FLOAT
  typedef float fptype_vertex;
#else
  typedef fptype2 fptype_vertex;
#endif

  // fptype_denom: for denominator variables in vertex functions
#if defined MGONGPU_FPTYPE_DENOM_DOUBLE
  typedef double fptype_denom;
#elif defined MGONGPU_FPTYPE_DENOM_FLOAT
  typedef float fptype_denom;
#else
  typedef fptype fptype_denom;
#endif

  // fptype_amp: for amplitude/jamp variables
#if defined MGONGPU_FPTYPE_AMP_DOUBLE
  typedef double fptype_amp;
#elif defined MGONGPU_FPTYPE_AMP_FLOAT
  typedef float fptype_amp;
#else
  typedef fptype2 fptype_amp;
#endif

  // fptype_colour: for color algebra (alias for fptype2)
  typedef fptype2 fptype_colour;

  // --- Platform-specific software implementation details

  // Maximum number of blocks per grid
  // ** NB Some arrays of pointers will be allocated statically to fit all these blocks
  // ** (the actual memory for each block will then be allocated dynamically only for existing blocks)
  //const int nbpgMAX = 2048;

  // Maximum number of threads per block
  //const int ntpbMAX = 256; // AV Apr2021: why had I set this to 256?
  const int ntpbMAX = 1024; // NB: 512 is ok, but 1024 does fail with "too many resources requested for launch"

  // Alignment requirement for using reinterpret_cast with SIMD vectorized code
  // (using reinterpret_cast with non aligned memory may lead to segmentation faults!)
  // Only needed for C++ code but can be enforced also in NVCC builds of C++ code using CUDA>=11.2 and C++17 (#318, #319, #333)
#ifndef MGONGPUCPP_GPUIMPL
  constexpr int cppAlign = 64; // alignment requirement for SIMD vectorization (64-byte i.e. 512-bit)
#endif

}

// Expose typedefs and operators outside the namespace
using mgOnGpu::fptype;
using mgOnGpu::fptype2;
using mgOnGpu::fptype_momenta;
using mgOnGpu::fptype_polarization;
using mgOnGpu::fptype_invmass;
using mgOnGpu::fptype_vertex;
using mgOnGpu::fptype_denom;
using mgOnGpu::fptype_amp;
using mgOnGpu::fptype_colour;

// Undefine ARM_NEON (hack for cppnone on Apple silicon ARM)
#ifdef MGONGPU_NOARMNEON
#undef __ARM_NEON
#endif

// C++ SIMD vectorization width (this will be used to set neppV)
#ifdef MGONGPUCPP_GPUIMPL // CUDA and HIP implementations have no SIMD
#undef MGONGPU_CPPSIMD
#elif defined __AVX512VL__ && defined MGONGPU_PVW512 // C++ "512z" AVX512 with 512 width (512-bit ie 64-byte): 8 (DOUBLE) or 16 (FLOAT)
#ifdef MGONGPU_FPTYPE_DOUBLE
#define MGONGPU_CPPSIMD 8
#else
#define MGONGPU_CPPSIMD 16
#endif
#elif defined __AVX512VL__ // C++ "512y" AVX512 with 256 width (256-bit ie 32-byte): 4 (DOUBLE) or 8 (FLOAT) [gcc DEFAULT]
#ifdef MGONGPU_FPTYPE_DOUBLE
#define MGONGPU_CPPSIMD 4
#else
#define MGONGPU_CPPSIMD 8
#endif
#elif defined __AVX2__ // C++ "avx2" AVX2 (256-bit ie 32-byte): 4 (DOUBLE) or 8 (FLOAT) [clang DEFAULT]
#ifdef MGONGPU_FPTYPE_DOUBLE
#define MGONGPU_CPPSIMD 4
#else
#define MGONGPU_CPPSIMD 8
#endif
#elif defined __SSE4_2__ // C++ "sse4" SSE4.2 (128-bit ie 16-byte): 2 (DOUBLE) or 4 (FLOAT) [Power9 default]
#ifdef MGONGPU_FPTYPE_DOUBLE
#define MGONGPU_CPPSIMD 2
#else
#define MGONGPU_CPPSIMD 4
#endif
#elif defined __ARM_NEON // C++ "sse4" ARM NEON (128-bit ie 16-byte): 2 (DOUBLE) or 4 (FLOAT) [ARM default]
#ifdef MGONGPU_FPTYPE_DOUBLE
#define MGONGPU_CPPSIMD 2
#else
#define MGONGPU_CPPSIMD 4
#endif
#else // C++ "none" i.e. no SIMD
#undef MGONGPU_CPPSIMD
#endif

// fptype_invmass (the m_ij / FIXP2 offshell-propagator virtuality) is only wired for scalar
// builds (cppnone or CUDA/HIP). In a SIMD build it would need a double-width SIMD lane
// alongside the float momenta lanes, which is not implemented for this stage. Reject it
// loudly. (sizeof equality <=> same float/double type.) NB: the other stage types
// (polarization, vertex, amp, ...) DO support SIMD via their _v vector types.
#ifdef MGONGPU_CPPSIMD
static_assert( sizeof( fptype_invmass ) == sizeof( fptype ),
               "fptype_invmass must equal fptype in SIMD builds: use BACKEND=cppnone (or CUDA), or define MGONGPU_FPTYPE_INVMASS_FLOAT" );
// fptype_polarization_v aliases fptype_v in SIMD, so the polarization stage must share
// fptype's width here (a double-precision polarization in a float SIMD build would need
// mixed-width lanes). The double-precision polarization experiment runs in cppnone/CUDA.
static_assert( sizeof( fptype_polarization ) == sizeof( fptype ),
               "fptype_polarization must equal fptype in SIMD builds: use BACKEND=cppnone (or CUDA), or define MGONGPU_FPTYPE_POLARIZATION_FLOAT" );
#endif

/* clang-format off */
// CUDA nsight compute (ncu) debug: add dummy lines to ease SASS program flow navigation [NB: CURRENTLY NO LONGER SUPPORTED!]
// Arguments (not used so far): text is __FUNCTION__, code is 0 (start) or 1 (end)
//#if defined __CUDACC__ && defined MGONGPU_NSIGHT_DEBUG // this must be __CUDACC__ (not MGONGPUCPP_GPUIMPL)
//#define mgDebugDeclare() __shared__ float mgDebugCounter[mgOnGpu::ntpbMAX];
//#define mgDebugInitialise() { mgDebugCounter[threadIdx.x] = 0; }
//#define mgDebug( code, text ) { mgDebugCounter[threadIdx.x] += 1; }
//#define mgDebugFinalise() { if ( blockIdx.x == 0 && threadIdx.x == 0 ) printf( "MGDEBUG: counter=%%f\n", mgDebugCounter[threadIdx.x] ); }
//#else
#define mgDebugDeclare() /*noop*/
#define mgDebugInitialise() /*noop*/
#define mgDebug( code, text ) /*noop*/
#define mgDebugFinalise() /*noop*/
//#endif /* clang-format on */

// Define empty CUDA/HIP declaration specifiers for C++
#ifndef MGONGPUCPP_GPUIMPL
#define __global__
#define __host__
#define __device__
#endif

// For SANITY CHECKS: check that neppR, neppM, neppV... are powers of two (https://stackoverflow.com/a/108360)
inline constexpr bool
ispoweroftwo( int n )
{
  return ( n > 0 ) && !( n & ( n - 1 ) );
}

// Compiler version support (#96): require nvcc from CUDA >= 11.2, e.g. to use C++17 (see #333)
#ifdef __NVCC__
#if( __CUDACC_VER_MAJOR__ < 11 ) || ( __CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ < 2 )
#error Unsupported CUDA version: please use CUDA >= 11.2
#endif
#endif

// Compiler version support (#96): require clang >= 11
#if defined __clang__
#if( __clang_major__ < 11 )
#error Unsupported clang version: please use clang >= 11
#endif
// Compiler version support (#96): require gcc >= 9.3, e.g. for some OMP issues (see #269)
// [NB skip this check for the gcc toolchain below clang or icx (TEMPORARY? #355)]
#elif defined __GNUC__
#if( __GNUC__ < 9 ) || ( __GNUC__ == 9 && __GNUC_MINOR__ < 3 )
#error Unsupported gcc version: please gcc >= 9.3
#endif
#endif

#endif // MGONGPUCONFIG_H
