// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created originally by: A. Valassi (Sep 2025) for the MG5aMC CUDACPP plugin.
// Further modified by: A. Valassi (2025).
// Integrated with the MadGraph7 project in Feb 2026.

#include "color_sum.h"

#include "mgOnGpuConfig.h"

#include "MemoryAccessMatrixElements.h"

#ifdef MGONGPUCPP_GPUIMPL
namespace mg5amcGpu
#else
namespace mg5amcCpu
#endif
{
  constexpr int ncolor = CPPProcess::ncolor; // the number of leading colors

  //--------------------------------------------------------------------------

  // *** COLOR MATRIX BELOW ***
%(color_matrix_lines)s

#ifdef MGONGPUCPP_GPUIMPL
  // The normalized color matrix (divide each column by denom)
  template<typename T>
  struct NormalizedColorMatrix
  {
    constexpr __host__ __device__ NormalizedColorMatrix()
      : value()
    {
      for( int icol = 0; icol < ncolor; icol++ )
        for( int jcol = 0; jcol < ncolor; jcol++ )
          value[icol * ncolor + jcol] = colorMatrix[icol][jcol] / colorDenom[icol];
    }
    T value[ncolor * ncolor];
  };
  // The fptype2 version is the default used by kernels (supporting mixed floating point mode)
  static __device__ fptype2 s_pNormalizedColorMatrix2[ncolor * ncolor];
#endif

  //--------------------------------------------------------------------------

#ifdef MGONGPUCPP_GPUIMPL
  void createNormalizedColorMatrix()
  {
    static bool first = true;
    if( first )
    {
      first = false;
      constexpr NormalizedColorMatrix<fptype2> normalizedColorMatrix2;
      gpuMemcpyToSymbol( s_pNormalizedColorMatrix2, normalizedColorMatrix2.value, ncolor * ncolor * sizeof( fptype2 ) );
    }
  }
#endif

  //--------------------------------------------------------------------------

#ifndef MGONGPUCPP_GPUIMPL
  void
  color_sum_cpu( fptype* allMEs,              // output: allMEs[nevt], add |M|^2 for one specific helicity
                 const cxtype_sv* allJamp_sv, // input: jamp_sv[ncolor] (float/double) or jamp_sv[2*ncolor] (mixed) for one specific helicity
                 const int ievt0 )            // input: first event number in current C++ event page (for CUDA, ievt depends on threadid)
  {
    // Pre-compute a constexpr triangular color matrix properly normalized #475
    struct TriangularNormalizedColorMatrix
    {
      // See https://stackoverflow.com/a/34465458
      __host__ __device__ constexpr TriangularNormalizedColorMatrix()
        : value()
      {
        for( int icol = 0; icol < ncolor; icol++ )
        {
          // Diagonal terms
          value[icol][icol] = colorMatrix[icol][icol] / colorDenom[icol];
          // Off-diagonal terms
          for( int jcol = icol + 1; jcol < ncolor; jcol++ )
            value[icol][jcol] = 2 * colorMatrix[icol][jcol] / colorDenom[icol];
        }
      }
      fptype2 value[ncolor][ncolor];
    };
    static constexpr auto cf2 = TriangularNormalizedColorMatrix();
    // Use the property that M is a real matrix (see #475):
    // we can rewrite the quadratic form (A-iB)(M)(A+iB) as AMA - iBMA + iBMA + BMB = AMA + BMB
    // In addition, on C++ use the property that M is symmetric (see #475),
    // and also use constexpr to compute "2*" and "/colorDenom[icol]" once and for all at compile time:
    // we gain (not a factor 2...) in speed here as we only loop over the up diagonal part of the matrix.
    // Strangely, CUDA is slower instead, so keep the old implementation for the moment.
    fptype_sv deltaMEs = { 0 };
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
    fptype_sv deltaMEs_next = { 0 };
#endif
    // NB in mixed mode the two neppV vectors of allJamp_sv, at icol and at ncolor+icol, are
    // two halves of the event page and not two colors: it is the color index inside each of
    // them which is gathered, and the two are merged into one neppV2 vector.
    fptype2_sv jampR_sv[ncolor];
    fptype2_sv jampI_sv[ncolor];
    for( int icol = 0; icol < ncolor; icol++ )
    {
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
      jampR_sv[icol] = fpvmerge( cxreal( allJamp_sv[icol] ), cxreal( allJamp_sv[ncolor + icol] ) );
      jampI_sv[icol] = fpvmerge( cximag( allJamp_sv[icol] ), cximag( allJamp_sv[ncolor + icol] ) );
#else
      jampR_sv[icol] = (fptype2_sv)( cxreal( allJamp_sv[icol] ) );
      jampI_sv[icol] = (fptype2_sv)( cximag( allJamp_sv[icol] ) );
#endif
    }
    for( int icol = 0; icol < ncolor; icol++ )
    {
      // Diagonal terms
      fptype2_sv& jampRi_sv = jampR_sv[icol];
      fptype2_sv& jampIi_sv = jampI_sv[icol];
      fptype2_sv ztempR_sv = cf2.value[icol][icol] * jampRi_sv;
      fptype2_sv ztempI_sv = cf2.value[icol][icol] * jampIi_sv;
      for( int jcol = icol + 1; jcol < ncolor; jcol++ )
      {
        // Off-diagonal terms
        fptype2_sv& jampRj_sv = jampR_sv[jcol];
        fptype2_sv& jampIj_sv = jampI_sv[jcol];
        ztempR_sv += cf2.value[icol][jcol] * jampRj_sv;
        ztempI_sv += cf2.value[icol][jcol] * jampIj_sv;
      }
      fptype2_sv deltaMEs2 = ( jampRi_sv * ztempR_sv + jampIi_sv * ztempI_sv ); // may underflow #831
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
      deltaMEs += fpvsplit0( deltaMEs2 );
      deltaMEs_next += fpvsplit1( deltaMEs2 );
#else
      deltaMEs += deltaMEs2;
#endif
    }
    // *** STORE THE RESULTS ***
    using E_ACCESS = HostAccessMatrixElements; // non-trivial access: buffer includes all events
    fptype* MEs = E_ACCESS::ieventAccessRecord( allMEs, ievt0 );
    // NB: color_sum ADDS |M|^2 for one helicity to the running sum of |M|^2 over helicities for the given event(s)
    fptype_sv& MEs_sv = E_ACCESS::kernelAccess( MEs );
    MEs_sv += deltaMEs; // fix #435
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
    fptype* MEs_next = E_ACCESS::ieventAccessRecord( allMEs, ievt0 + neppV );
    fptype_sv& MEs_sv_next = E_ACCESS::kernelAccess( MEs_next );
    MEs_sv_next += deltaMEs_next;
#endif
  }
#endif
%(cpp_blas_color_sum)s
  //--------------------------------------------------------------------------

#ifdef MGONGPUCPP_GPUIMPL
  __global__ void
  color_sum_kernel( fptype* allMEs,                 // output: allMEs[nevt], add |M|^2 for one specific helicity
                    const fptype* allJamps,         // input: jamp[ncolor*2*nevt] for one specific helicity
                    const int nGoodHel,             // input: number of good helicities
                    const int nevtIfAllHelicities ) // input: zero in single-helicity mode, number of events in multi-helicity mode
  {
    if (nevtIfAllHelicities) {
      int ighel = blockIdx.y;
      allMEs = allMEs + ighel * nevtIfAllHelicities; // MEs for one specific helicity ighel
      allJamps = allJamps + ighel * nevtIfAllHelicities; // Jamps for one specific helicity ighel
    }
    using J_ACCESS = DeviceAccessJamp;
    fptype jampR[ncolor];
    fptype jampI[ncolor];
    for( int icol = 0; icol < ncolor; icol++ )
    {
      constexpr int ihel0 = 0; // the input buffer allJamps already points to a specific helicity
      cxtype jamp = J_ACCESS::kernelAccessIcolIhelNhelConst( allJamps, icol, ihel0, nGoodHel );
      jampR[icol] = jamp.real();
      jampI[icol] = jamp.imag();
    }
    fptype deltaMEs = { 0 };
    for( int icol = 0; icol < ncolor; icol++ )
    {
      fptype2 ztempR = { 0 };
      fptype2 ztempI = { 0 };
      fptype2 jampRi = jampR[icol];
      fptype2 jampIi = jampI[icol];
      // OLD IMPLEMENTATION (ihel3: symmetric square matrix) - Loop over all jcol
      //for( int jcol = 0; jcol < ncolor; jcol++ )
      //{
      //  fptype2 jampRj = jampR[jcol];
      //  fptype2 jampIj = jampI[jcol];
      //  ztempR += s_pNormalizedColorMatrix2[icol * ncolor + jcol] * jampRj; // use fptype2 version of color matrix
      //  ztempI += s_pNormalizedColorMatrix2[icol * ncolor + jcol] * jampIj; // use fptype2 version of color matrix
      //}
      // NEW IMPLEMENTATION #475 (ihel3p1: triangular lower diagonal matrix) - Loop over jcol < icol
      ztempR += s_pNormalizedColorMatrix2[icol * ncolor + icol] * jampRi; // use fptype2 version of color matrix
      ztempI += s_pNormalizedColorMatrix2[icol * ncolor + icol] * jampIi; // use fptype2 version of color matrix
      for( int jcol = 0; jcol < icol; jcol++ )
      {
        fptype2 jampRj = jampR[jcol];
        fptype2 jampIj = jampI[jcol];
        ztempR += 2 * s_pNormalizedColorMatrix2[icol * ncolor + jcol] * jampRj; // use fptype2 version of color matrix
        ztempI += 2 * s_pNormalizedColorMatrix2[icol * ncolor + jcol] * jampIj; // use fptype2 version of color matrix
      }
      deltaMEs += ztempR * jampRi;
      deltaMEs += ztempI * jampIi;
    }
    // *** STORE THE RESULTS ***
    using E_ACCESS = DeviceAccessMatrixElements; // non-trivial access: buffer includes all events
    // NB: color_sum ADDS |M|^2 for one helicity to the running sum of |M|^2 over helicities for the given event(s)
    E_ACCESS::kernelAccess( allMEs ) += deltaMEs; // fix #435
  }
#endif

  //--------------------------------------------------------------------------

#ifdef MGONGPUCPP_GPUIMPL
#ifndef MGONGPU_HAS_NO_BLAS
  // Convert the jamps from fptype to fptype2, which is only ever double to float in mixed
  // floating point mode: the color sum reads them where compute_jamps wrote them otherwise.
  __global__ void
  convertD2F_Jamps( fptype2* allJamps2,     // output: jamp[2][ncolor][ihel][nevt] for one specific helicity ihel
                    const fptype* allJamps, // input: jamp[2][ncolor][ihel][nevt] for one specific helicity ihel
                    const int nhel )        // input: number of good helicities nGoodHel
  {
    const int nevt = gridDim.x * blockDim.x;
    const int ievt = blockDim.x * blockIdx.x + threadIdx.x;
    constexpr int ihel = 0; // the input buffer allJamps already points to a specific helicity
    // NB! The same striding as in compute_jamps and cuBLAS is used here
    for( int ix2 = 0; ix2 < mgOnGpu::nx2; ix2++ )
      for( int icol = 0; icol < ncolor; icol++ )
        allJamps2[ix2 * ncolor * nhel * nevt + icol * nhel * nevt + ihel * nevt + ievt] =
          allJamps[ix2 * ncolor * nhel * nevt + icol * nhel * nevt + ihel * nevt + ievt];
  }

  // Convert the jamps of every good helicity into ghelAllJampsBuf, and return it
  fptype2*
  convertD2F_AllJamps( fptype2* ghelAllJampsBuf,    // output: allJamps super-buffer[2][ncolor][nhel][nevt]
                       const fptype* ghelAllJamps,  // input: allJamps super-buffer[2][ncolor][nhel][nevt]
                       gpuStream_t* ghelStreams,    // input: cuda streams (index is ighel)
                       const int nhel,              // input: number of good helicities
                       const int gpublocks,         // input: cuda gpublocks
                       const int gputhreads )       // input: cuda gputhreads
  {
    const int nevt = gpublocks * gputhreads;
    for( int ighel = 0; ighel < nhel; ighel++ )
    {
      const fptype* hAllJamps = ghelAllJamps + ighel * nevt;         // jamps for a single helicity ihel
      fptype2* hAllJamps2 = ghelAllJampsBuf + ighel * nevt;          // converted jamps for a single helicity ihel
      gpuLaunchKernelStream( convertD2F_Jamps, gpublocks, gputhreads, ghelStreams[ighel], hAllJamps2, hAllJamps, nhel );
    }
    return ghelAllJampsBuf;
  }
#endif
#endif

  //--------------------------------------------------------------------------

#ifdef MGONGPUCPP_GPUIMPL
#ifndef MGONGPU_HAS_NO_BLAS
#if defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
  __global__ void
  convertF2D_MEs( fptype* allMEs,             // output: allMEs[nevt] for one specific helicity
                  const fptype2* allMEsFpt2 ) // input: allMEs[nevt] for one specific helicity
  {
    const int ievt = blockDim.x * blockIdx.x + threadIdx.x;
    allMEs[ievt] = allMEsFpt2[ievt];
  }
#endif
#endif
#endif

  //--------------------------------------------------------------------------

#ifdef MGONGPUCPP_GPUIMPL /* clang-format off */
#ifndef MGONGPU_HAS_NO_BLAS
  void
  color_sum_blas( fptype* ghelAllMEs,           // output: allMEs super-buffer[nhel][nevt], add |M|^2 separately for each helicity
                  const fptype* ghelAllJamps,   // input: allJamps super-buffer[2][ncol][nhel][nevt] for nhel good helicities
                  fptype2* ghelAllBlasTmp,      // tmp: allBlasTmp super-buffer for nhel good helicities
                  gpuBlasHandle_t* pBlasHandle, // input: cuBLAS/hipBLAS handle
                  gpuStream_t* ghelStreams,     // input: cuda streams (index is ighel: only the first nhel <= ncomb are non-null)
                  const int nhel,               // input: number of good helicities (nhel == nGoodHel)
                  const int gpublocks,          // input: cuda gpublocks
                  const int gputhreads )        // input: cuda gputhreads
  {
    const int nevt = gpublocks * gputhreads;

    // Get the address associated with the normalized color matrix in device memory
    static fptype2* devNormColMat = nullptr;
    if( !devNormColMat ) gpuGetSymbolAddress( (void**)&devNormColMat, s_pNormalizedColorMatrix2 );

    // The scratch buffer holds the BLAS intermediate results and, in mixed precision mode,
    // the converted jamps and the fptype2 MEs: see the layout in blasColorSumTmpSize, which
    // is what MatrixElementKernels.cc allocates.
    fptype2* ghelAllZtempBoth = ghelAllBlasTmp;                                        // start of the fptype2[ncolor*2*nhel*nevt] buffer
#if defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
    // Mixed precision mode: the jamps go through a buffer of their own, as they must be
    // converted from double to float on the way
    fptype2* ghelAllJampsBuf = ghelAllBlasTmp + ncolor * mgOnGpu::nx2 * nhel * nevt;    // start of the second one
    fptype2* ghelAllMEsFpt2 = ghelAllBlasTmp + 2 * ncolor * mgOnGpu::nx2 * nhel * nevt; // start of the fptype2[nhel*nevt] buffer
    const fptype2* ghelAllJamps2 = convertD2F_AllJamps( ghelAllJampsBuf, ghelAllJamps, ghelStreams, nhel, gpublocks, gputhreads );
#else
    static_assert( std::is_same<fptype2, fptype>::value );
    fptype2* ghelAllMEsFpt2 = ghelAllMEs;
    // Nothing to convert: read the jamps where compute_jamps wrote them (this is not a
    // cast, the two types are identical)
    const fptype2* ghelAllJamps2 = ghelAllJamps;
#endif
    // Real and imaginary components
    const fptype2* ghelAllJampsReal = ghelAllJamps2;
    const fptype2* ghelAllJampsImag = ghelAllJamps2 + ncolor * nhel * nevt;
    fptype2* ghelAllZtempReal = ghelAllZtempBoth;
    fptype2* ghelAllZtempImag = ghelAllZtempBoth + ncolor * nhel * nevt;

    // Note: striding for cuBLAS (that of DeviceAccessJamp, over ncolor):
    // - ghelAllJamps2(icol,ihel,ievt).real is ghelAllJamps2[0 * ncolor * nhel * nevt + icol * nhel * nevt + ihel * nevt + ievt]
    // - ghelAllJamps2(icol,ihel,ievt).imag is ghelAllJamps2[1 * ncolor * nhel * nevt + icol * nhel * nevt + ihel * nevt + ievt]

    // Step 1: Compute Ztemp[ncolor][nhel*nevt] = ColorMatrix[ncolor][ncolor] * JampsVector[ncolor][nhel*nevt] for both real and imag
    // In this case alpha=1 and beta=0: the operation is Ztemp = alpha * ColorMatrix * JampsVector + beta * Ztemp
    fptype2 alpha1 = 1;
    fptype2 beta1 = 0;
    const int ncolorM = ncolor;
    const int nevtN = nhel*nevt;
    const int ncolorK = ncolor;
    checkGpuBlas( gpuBlasTgemm( *pBlasHandle,
                                GPUBLAS_OP_N,                  // do not transpose ColMat
                                GPUBLAS_OP_T,                  // transpose JampsV (new1)
                                ncolorM, nevtN, ncolorK,
                                &alpha1,
                                devNormColMat, ncolorM,        // ColMat is ncolorM x ncolorK
                                ghelAllJampsReal, nevtN,       // JampsV is nevtN x ncolorK
                                &beta1,
                                ghelAllZtempReal, ncolorM ) ); // Ztemp is ncolorM x nevtN
    checkGpuBlas( gpuBlasTgemm( *pBlasHandle,
                                GPUBLAS_OP_N,                  // do not transpose ColMat
                                GPUBLAS_OP_T,                  // transpose JampsV (new1)
                                ncolorM, nevtN, ncolorK,
                                &alpha1,
                                devNormColMat, ncolorM,        // ColMat is ncolorM x ncolorK
                                ghelAllJampsImag, nevtN,       // JampsV is nevtN x ncolorK (new1)
                                &beta1,
                                ghelAllZtempImag, ncolorM ) ); // Ztemp is ncolorM x nevtN

    // Step 2: For each ievt, compute the dot product of JampsVector[ncolor][ievt] dot tmp[ncolor][ievt]
    // In this case alpha=1 and beta=1: the operation is ME = alpha * ( Tmp dot JampsVector ) + beta * ME
    // Use cublasSgemmStridedBatched to perform these batched dot products in one call
    fptype2 alpha2 = 1;
    fptype2 beta2 = 1;
    checkGpuBlas( gpuBlasTgemmStridedBatched( *pBlasHandle,
                                              GPUBLAS_OP_N,                             // do not transpose JampsV (new1)
                                              GPUBLAS_OP_N,                             // do not transpose Tmp
                                              1, 1, ncolor,                         // result is 1x1 (dot product)
                                              &alpha2,
                                              ghelAllJampsReal, nevtN, 1,               // allJamps is nevtN x ncolor, stride 1 for each ievt column
                                              ghelAllZtempReal, ncolor, ncolor, // allZtemp is ncolor x nevtN, with stride ncolor for each ievt column
                                              &beta2,
                                              ghelAllMEsFpt2, 1, 1,                     // output is a 1x1 result for each "batch" (i.e. for each ievt)
                                              nevtN ) );                                // there are nevtN (nhel*nevt) "batches"
    checkGpuBlas( gpuBlasTgemmStridedBatched( *pBlasHandle,
                                              GPUBLAS_OP_N,                             // do not transpose JampsV (new1)
                                              GPUBLAS_OP_N,                             // do not transpose Tmp
                                              1, 1, ncolor,                         // result is 1x1 (dot product)
                                              &alpha2,
                                              ghelAllJampsImag, nevtN, 1,               // allJamps is nevtN x ncolor, stride 1 for each ievt column (new1)
                                              ghelAllZtempImag, ncolor, ncolor, // allZtemp is ncolor x nevtN, with stride ncolor for each ievt column
                                              &beta2,
                                              ghelAllMEsFpt2, 1, 1,                     // output is a 1x1 result for each "batch" (i.e. for each ievt)
                                              nevtN ) );                                // there are nevt (nhel*nevt) "batches"

#if defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
    // Convert MEs from float to double
    for( int ighel = 0; ighel < nhel; ighel++ )
    {
      fptype* hAllMEs = ghelAllMEs + ighel * nevt;          // MEs for a single helicity ihel
      fptype2* hAllMEsFpt2 = ghelAllMEsFpt2 + ighel * nevt; // MEs for a single helicity ihel      
      gpuLaunchKernelStream( convertF2D_MEs, gpublocks, gputhreads, ghelStreams[ighel], hAllMEs, hAllMEsFpt2 );
    }
#endif
  }
#endif /* clang-format on */
#endif

  //--------------------------------------------------------------------------

#ifdef MGONGPUCPP_GPUIMPL
  void
  color_sum_gpu( fptype* ghelAllMEs,               // output: allMEs super-buffer for nGoodHel <= ncomb individual helicities (index is ighel)
                 const fptype* ghelAllJamps,       // input: allJamps super-buffer[2][ncol][nGoodHel][nevt] for nGoodHel <= ncomb individual helicities
                 fptype2* ghelAllBlasTmp,          // tmp: allBlasTmp super-buffer for nGoodHel <= ncomb individual helicities
                 gpuBlasHandle_t* pBlasHandle,     // input: cuBLAS/hipBLAS handle
                 gpuStream_t* ghelStreams,         // input: cuda streams (index is ighel: only the first nGoodHel <= ncomb are non-null)
                 const int nGoodHel,               // input: number of good helicities
                 const int gpublocks,              // input: cuda gpublocks
                 const int gputhreads,             // input: cuda gputhreads
                 const bool processAllHelicities ) // input: if true, use blockIdx.y to index helicities
  {
    const int nevt = gpublocks * gputhreads;
    // CASE 1: KERNEL
    if( !pBlasHandle )
    {
      assert( ghelAllBlasTmp == nullptr );  // sanity check for HASBLAS=hasNoBlas or CUDACPP_RUNTIME_BLASCOLORSUM not set
      if (processAllHelicities) {
        gpuLaunchKernel2D( color_sum_kernel, gpublocks, nGoodHel, gputhreads, ghelStreams[0], ghelAllMEs, ghelAllJamps, nGoodHel, nevt );
      } else {
        // Loop over helicities
        for( int ighel = 0; ighel < nGoodHel; ighel++ )
        {
          fptype* hAllMEs = ghelAllMEs + ighel * nevt;           // MEs for one specific helicity ighel
          const fptype* hAllJamps = ghelAllJamps + ighel * nevt; // Jamps for one specific helicity ighel
          gpuStream_t hStream = ghelStreams[ighel];
          gpuLaunchKernelStream( color_sum_kernel, gpublocks, gputhreads, hStream, hAllMEs, hAllJamps, nGoodHel, 0 );
        }
      }
    }
    // CASE 2: BLAS
    else
    {
#ifdef MGONGPU_HAS_NO_BLAS
      assert( false ); // sanity check: no path to this statement for HASBLAS=hasNoBlas
#else
      if (processAllHelicities) {
        assert( false ); // BLAS in async mode not supported for now
      } else {
        checkGpu( gpuDeviceSynchronize() ); // do not start the BLAS color sum for all helicities until the loop over helicities has completed
        // Reset the tmp buffer (same size as the one MatrixElementKernelDevice allocated)
        gpuMemset( ghelAllBlasTmp, 0, blasColorSumTmpSize( nGoodHel, nevt ) * sizeof( fptype2 ) );
        // Delegate the color sum to BLAS for 
        color_sum_blas( ghelAllMEs, ghelAllJamps, ghelAllBlasTmp, pBlasHandle, ghelStreams, nGoodHel, gpublocks, gputhreads );
      }
#endif
    }
  }
#endif

  //--------------------------------------------------------------------------

} // end namespace
