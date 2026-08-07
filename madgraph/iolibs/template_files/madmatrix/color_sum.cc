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
  // The normalized folded color matrix (divide each column by denom)
  template<typename T>
  struct NormalizedColorMatrix
  {
    constexpr __host__ __device__ NormalizedColorMatrix()
      : value()
    {
      for( int ifold = 0; ifold < ncolorfold; ifold++ )
        for( int jfold = 0; jfold < ncolorfold; jfold++ )
          value[ifold * ncolorfold + jfold] = colorMatrix[ifold][jfold] / colorDenom[ifold];
    }
    T value[ncolorfold * ncolorfold];
  };
  // The fptype2 version is the default used by kernels (supporting mixed floating point mode)
  static __device__ fptype2 s_pNormalizedColorMatrixFold2[ncolorfold * ncolorfold];
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
      gpuMemcpyToSymbol( s_pNormalizedColorMatrixFold2, normalizedColorMatrix2.value, ncolorfold * ncolorfold * sizeof( fptype2 ) );
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
        for( int ifold = 0; ifold < ncolorfold; ifold++ )
        {
          // Diagonal terms
          value[ifold][ifold] = colorMatrix[ifold][ifold] / colorDenom[ifold];
          // Off-diagonal terms
          for( int jfold = ifold + 1; jfold < ncolorfold; jfold++ )
            value[ifold][jfold] = 2 * colorMatrix[ifold][jfold] / colorDenom[ifold];
        }
      }
      fptype2 value[ncolorfold][ncolorfold];
    };
    static constexpr auto cf2 = TriangularNormalizedColorMatrix();
    // Use the property that M is a real matrix (see #475):
    // we can rewrite the quadratic form (A-iB)(M)(A+iB) as AMA - iBMA + iBMA + BMB = AMA + BMB
    // In addition, on C++ use the property that M is symmetric (see #475),
    // and also use constexpr to compute "2*" and "/colorDenom[ifold]" once and for all at compile time:
    // we gain (not a factor 2...) in speed here as we only loop over the up diagonal part of the matrix.
    // Strangely, CUDA is slower instead, so keep the old implementation for the moment.
    fptype_sv deltaMEs = { 0 };
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
    fptype_sv deltaMEs_next = { 0 };
#endif
    // Gather the color flows the sum runs over: one per reversal pair when the color basis
    // folds (ncolorfold < ncolor), every flow otherwise (ncolorfold == ncolor, identity map).
    // NB in mixed mode the two neppV vectors of allJamp_sv, at icol and at ncolor+icol, are
    // two halves of the event page and not two colors: it is the color index inside each of
    // them which is gathered, and the two are merged into one neppV2 vector.
    fptype2_sv jampR_sv[ncolorfold];
    fptype2_sv jampI_sv[ncolorfold];
    for( int ifold = 0; ifold < ncolorfold; ifold++ )
    {
      const int icol = colorFoldRep[ifold];
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
      jampR_sv[ifold] = fpvmerge( cxreal( allJamp_sv[icol] ), cxreal( allJamp_sv[ncolor + icol] ) );
      jampI_sv[ifold] = fpvmerge( cximag( allJamp_sv[icol] ), cximag( allJamp_sv[ncolor + icol] ) );
#else
      jampR_sv[ifold] = (fptype2_sv)( cxreal( allJamp_sv[icol] ) );
      jampI_sv[ifold] = (fptype2_sv)( cximag( allJamp_sv[icol] ) );
#endif
    }
    // Loop over ifold
    for( int ifold = 0; ifold < ncolorfold; ifold++ )
    {
      // Diagonal terms
      fptype2_sv& jampRi_sv = jampR_sv[ifold];
      fptype2_sv& jampIi_sv = jampI_sv[ifold];
      fptype2_sv ztempR_sv = cf2.value[ifold][ifold] * jampRi_sv;
      fptype2_sv ztempI_sv = cf2.value[ifold][ifold] * jampIi_sv;
      // Loop over jfold
      for( int jfold = ifold + 1; jfold < ncolorfold; jfold++ )
      {
        // Off-diagonal terms
        fptype2_sv& jampRj_sv = jampR_sv[jfold];
        fptype2_sv& jampIj_sv = jampI_sv[jfold];
        ztempR_sv += cf2.value[ifold][jfold] * jampRj_sv;
        ztempI_sv += cf2.value[ifold][jfold] * jampIj_sv;
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
    // Gather the color flows the sum runs over: one per reversal pair when the color basis
    // folds (ncolorfold < ncolor), every flow otherwise (ncolorfold == ncolor, identity map)
    fptype jampR[ncolorfold];
    fptype jampI[ncolorfold];
    for( int ifold = 0; ifold < ncolorfold; ifold++ )
    {
      constexpr int ihel0 = 0; // the input buffer allJamps already points to a specific helicity
      cxtype jamp = J_ACCESS::kernelAccessIcolIhelNhelConst( allJamps, colorFoldRep[ifold], ihel0, nGoodHel );
      jampR[ifold] = jamp.real();
      jampI[ifold] = jamp.imag();
    }
    // Loop over ifold
    fptype deltaMEs = { 0 };
    for( int ifold = 0; ifold < ncolorfold; ifold++ )
    {
      fptype2 ztempR = { 0 };
      fptype2 ztempI = { 0 };
      fptype2 jampRi = jampR[ifold];
      fptype2 jampIi = jampI[ifold];
      // OLD IMPLEMENTATION (ihel3: symmetric square matrix) - Loop over all jfold
      //for( int jfold = 0; jfold < ncolorfold; jfold++ )
      //{
      //  fptype2 jampRj = jampR[jfold];
      //  fptype2 jampIj = jampI[jfold];
      //  ztempR += s_pNormalizedColorMatrixFold2[ifold * ncolorfold + jfold] * jampRj; // use fptype2 version of color matrix
      //  ztempI += s_pNormalizedColorMatrixFold2[ifold * ncolorfold + jfold] * jampIj; // use fptype2 version of color matrix
      //}
      // NEW IMPLEMENTATION #475 (ihel3p1: triangular lower diagonal matrix) - Loop over jfold < ifold
      ztempR += s_pNormalizedColorMatrixFold2[ifold * ncolorfold + ifold] * jampRi; // use fptype2 version of color matrix
      ztempI += s_pNormalizedColorMatrixFold2[ifold * ncolorfold + ifold] * jampIi; // use fptype2 version of color matrix
      for( int jfold = 0; jfold < ifold; jfold++ )
      {
        fptype2 jampRj = jampR[jfold];
        fptype2 jampIj = jampI[jfold];
        ztempR += 2 * s_pNormalizedColorMatrixFold2[ifold * ncolorfold + jfold] * jampRj; // use fptype2 version of color matrix
        ztempI += 2 * s_pNormalizedColorMatrixFold2[ifold * ncolorfold + jfold] * jampIj; // use fptype2 version of color matrix
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
  // Compact the jamps onto the ncolorfold color flows the sum runs over, one per reversal
  // pair when the color basis folds (see color_sum_blas): the representative flows are not
  // contiguous, so this is a gather through colorFoldRep and not a copy. Without a folding
  // ncolorfold == ncolor and colorFoldRep is the identity, so this only converts. The
  // conversion to fptype2 is a no-op except in mixed floating point mode (double to float).
  __global__ void
  gatherFold_Jamps( fptype2* allJampsFold2, // output: jamp[2][ncolorfold][ihel][nevt] for one specific helicity ihel
                    const fptype* allJamps, // input: jamp[2][ncolor][ihel][nevt] for one specific helicity ihel
                    const int nhel )        // input: number of good helicities nGoodHel
  {
    const int nevt = gridDim.x * blockDim.x;
    const int ievt = blockDim.x * blockIdx.x + threadIdx.x;
    constexpr int ihel = 0; // the input buffer allJamps already points to a specific helicity
    // NB! From a functional point of view, any striding will be ok here as long as ncolorfold*2*nevt elements are all correctly gathered!
    // NB! Just in case this may be better for performance reasons, however, the same striding as in compute_jamps and cuBLAS is used here
    for( int ix2 = 0; ix2 < mgOnGpu::nx2; ix2++ )
      for( int ifold = 0; ifold < ncolorfold; ifold++ )
        allJampsFold2[ix2 * ncolorfold * nhel * nevt + ifold * nhel * nevt + ihel * nevt + ievt] =
          allJamps[ix2 * ncolor * nhel * nevt + colorFoldRep[ifold] * nhel * nevt + ihel * nevt + ievt];
  }

  // Gather the jamps of every good helicity into ghelAllJampsBuf, and return it
  fptype2*
  gatherFold_AllJamps( fptype2* ghelAllJampsBuf,    // output: allJamps super-buffer[2][ncolorfold][nhel][nevt]
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
      fptype2* hAllJampsFold2 = ghelAllJampsBuf + ighel * nevt;      // folded jamps for a single helicity ihel
      gpuLaunchKernelStream( gatherFold_Jamps, gpublocks, gputhreads, ghelStreams[ighel], hAllJampsFold2, hAllJamps, nhel );
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

    // As in color_sum_cpu and color_sum_kernel, the sum is folded onto one color flow per
    // reversal pair: the jamps are first gathered from the ncolor flows they are written in
    // down to the ncolorfold flows which are kept (see gatherFold_Jamps), and it is those
    // which are multiplied by the folded color matrix. Without a folding ncolorfold == ncolor,
    // the gather is the identity and this is the plain ncolor x ncolor color sum.

    // Get the address associated with the normalized folded color matrix in device memory
    static fptype2* devNormColMat = nullptr;
    if( !devNormColMat ) gpuGetSymbolAddress( (void**)&devNormColMat, s_pNormalizedColorMatrixFold2 );

    // The scratch buffer holds the BLAS intermediate results, the gathered jamps if they need
    // a buffer of their own, and in mixed precision mode the fptype2 MEs: see the layout in
    // blasColorSumTmpSize, which is what MatrixElementKernels.cc allocates.
    fptype2* ghelAllZtempBoth = ghelAllBlasTmp;                                          // start of the fptype2[ncolorfold*2*nhel*nevt] buffer
    fptype2* ghelAllJampsBuf = ghelAllBlasTmp + ncolorfold * mgOnGpu::nx2 * nhel * nevt;  // start of the second one, if there is one
#if defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
    // Mixed precision mode: the jamps are gathered into a buffer of their own in any case,
    // as they must be converted from double to float on the way
    static_assert( blasColorSumNeedsJampBuffer() );
    fptype2* ghelAllMEsFpt2 = ghelAllBlasTmp + 2 * ncolorfold * mgOnGpu::nx2 * nhel * nevt; // start of the fptype2[nhel*nevt] buffer
    const fptype2* ghelAllJampsFold2 = gatherFold_AllJamps( ghelAllJampsBuf, ghelAllJamps, ghelStreams, nhel, gpublocks, gputhreads );
#else
    static_assert( std::is_same<fptype2, fptype>::value );
    fptype2* ghelAllMEsFpt2 = ghelAllMEs;
    // Without a folding there is nothing to gather and nothing to convert: read the jamps
    // where compute_jamps wrote them (this is not a cast, the two types are identical)
    const fptype2* ghelAllJampsFold2 =
      ( blasColorSumNeedsJampBuffer()
          ? gatherFold_AllJamps( ghelAllJampsBuf, ghelAllJamps, ghelStreams, nhel, gpublocks, gputhreads )
          : ghelAllJamps );
#endif
    // Real and imaginary components
    const fptype2* ghelAllJampsReal = ghelAllJampsFold2;
    const fptype2* ghelAllJampsImag = ghelAllJampsFold2 + ncolorfold * nhel * nevt;
    fptype2* ghelAllZtempReal = ghelAllZtempBoth;
    fptype2* ghelAllZtempImag = ghelAllZtempBoth + ncolorfold * nhel * nevt;

    // Note: striding for cuBLAS from gatherFold_Jamps (that of DeviceAccessJamp, over ncolorfold):
    // - ghelAllJampsFold2(ifold,ihel,ievt).real is ghelAllJampsFold2[0 * ncolorfold * nhel * nevt + ifold * nhel * nevt + ihel * nevt + ievt]
    // - ghelAllJampsFold2(ifold,ihel,ievt).imag is ghelAllJampsFold2[1 * ncolorfold * nhel * nevt + ifold * nhel * nevt + ihel * nevt + ievt]

    // Step 1: Compute Ztemp[ncolorfold][nhel*nevt] = ColorMatrix[ncolorfold][ncolorfold] * JampsVector[ncolorfold][nhel*nevt] for both real and imag
    // In this case alpha=1 and beta=0: the operation is Ztemp = alpha * ColorMatrix * JampsVector + beta * Ztemp
    fptype2 alpha1 = 1;
    fptype2 beta1 = 0;
    const int ncolorM = ncolorfold;
    const int nevtN = nhel*nevt;
    const int ncolorK = ncolorfold;
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

    // Step 2: For each ievt, compute the dot product of JampsVector[ncolorfold][ievt] dot tmp[ncolorfold][ievt]
    // In this case alpha=1 and beta=1: the operation is ME = alpha * ( Tmp dot JampsVector ) + beta * ME
    // Use cublasSgemmStridedBatched to perform these batched dot products in one call
    fptype2 alpha2 = 1;
    fptype2 beta2 = 1;
    checkGpuBlas( gpuBlasTgemmStridedBatched( *pBlasHandle,
                                              GPUBLAS_OP_N,                             // do not transpose JampsV (new1)
                                              GPUBLAS_OP_N,                             // do not transpose Tmp
                                              1, 1, ncolorfold,                         // result is 1x1 (dot product)
                                              &alpha2,
                                              ghelAllJampsReal, nevtN, 1,               // allJamps is nevtN x ncolorfold, stride 1 for each ievt column
                                              ghelAllZtempReal, ncolorfold, ncolorfold, // allZtemp is ncolorfold x nevtN, with stride ncolorfold for each ievt column
                                              &beta2,
                                              ghelAllMEsFpt2, 1, 1,                     // output is a 1x1 result for each "batch" (i.e. for each ievt)
                                              nevtN ) );                                // there are nevtN (nhel*nevt) "batches"
    checkGpuBlas( gpuBlasTgemmStridedBatched( *pBlasHandle,
                                              GPUBLAS_OP_N,                             // do not transpose JampsV (new1)
                                              GPUBLAS_OP_N,                             // do not transpose Tmp
                                              1, 1, ncolorfold,                         // result is 1x1 (dot product)
                                              &alpha2,
                                              ghelAllJampsImag, nevtN, 1,               // allJamps is nevtN x ncolorfold, stride 1 for each ievt column (new1)
                                              ghelAllZtempImag, ncolorfold, ncolorfold, // allZtemp is ncolorfold x nevtN, with stride ncolorfold for each ievt column
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
