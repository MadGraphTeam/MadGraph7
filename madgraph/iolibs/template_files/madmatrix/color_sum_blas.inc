// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Integrated with the MadGraph7 project in Feb 2026.
//
// The C++ color sum for every good helicity at once, through BLAS. This is
// only written out for processes whose color matrix is large enough that the
// call is worth setting up (see blas_wanted); everywhere else color_sum.cc is
// character for character the file written before any of this existed.

  //--------------------------------------------------------------------------

#ifndef MGONGPUCPP_GPUIMPL
#ifdef MGONGPU_CPP_HAS_BLAS

#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
  constexpr int nParityCS = 2; // mixed mode merges two neppV pages into one call
#else
  constexpr int nParityCS = 1;
#endif

  // The color matrix does not depend on the helicity, so the jamps of every
  // good helicity (and of every event in the SIMD page) are columns of one
  // right hand side and the whole color sum becomes two SYMM calls. SYMM is
  // real, so the real and the imaginary part of JAMP go through separately,
  // which is the same property of M being real that the scalar sum uses to
  // rewrite (A-iB)M(A+iB) as AMA + BMB (see #475).
  //
  // The scalar sum walks the upper triangle with its off diagonal doubled;
  // SYMM wants the whole symmetric matrix with each entry counted once. What
  // is written out is normalized row by row by colorDenom[icol], which is not
  // symmetric when the denominators differ; only the symmetric part of a
  // matrix contributes to a quadratic form, so take it explicitly. When all
  // the denominators agree (the usual case) this is exactly colorMatrix/denom.
  struct SymmetricNormalizedColorMatrix
  {
    constexpr SymmetricNormalizedColorMatrix()
      : value()
    {
      for( int icol = 0; icol < ncolor; icol++ )
        for( int jcol = 0; jcol < ncolor; jcol++ )
          value[icol * ncolor + jcol] =
            ( colorMatrix[icol][jcol] / colorDenom[icol] + colorMatrix[jcol][icol] / colorDenom[jcol] ) / 2;
    }
    fptype2 value[ncolor * ncolor];
  };

  // The Fortran BLAS interface, which every implementation exports (the
  // reference BLAS ships no CBLAS of its own). Column major, as in the
  // Fortran color sum.
  extern "C"
  {
    void dsymm_( const char* side, const char* uplo, const int* m, const int* n,
                 const double* alpha, const double* a, const int* lda,
                 const double* b, const int* ldb,
                 const double* beta, double* c, const int* ldc );
    void ssymm_( const char* side, const char* uplo, const int* m, const int* n,
                 const float* alpha, const float* a, const int* lda,
                 const float* b, const int* ldb,
                 const float* beta, float* c, const int* ldc );
  }

  static inline void
  blas_symm( const int m, const int n, const double* a, const double* b, double* c )
  {
    const char side = 'L', uplo = 'U';
    const double alpha = 1, beta = 0;
    dsymm_( &side, &uplo, &m, &n, &alpha, a, &m, b, &m, &beta, c, &m );
  }

  static inline void
  blas_symm( const int m, const int n, const float* a, const float* b, float* c )
  {
    const char side = 'L', uplo = 'U';
    const float alpha = 1, beta = 0;
    ssymm_( &side, &uplo, &m, &n, &alpha, a, &m, b, &m, &beta, c, &m );
  }

  void
  color_sum_cpu_blas( fptype* allMEs,                  // input/output: allMEs[nevt], add |M|^2 summed over all good helicities
                      fptype_sv* MEs_ighel,            // output: [ncomb] running sum of |M|^2 up to ighel (first - and/or only - neppV page)
                      fptype_sv* MEs_ighel2,           // output: [ncomb] the same for the second neppV page (mixed mode only)
                      const cxtype_sv* ghelAllJamp_sv, // input: jamp_sv[nGoodHel][nParity*ncolor] for all good helicities
                      const int nGoodHel,              // input: number of good helicities
                      const int ievt0 )                // input: first event number in current C++ event page
  {
    static constexpr auto cfsym = SymmetricNormalizedColorMatrix();
    constexpr int nevtB = nParityCS * neppV; // events covered by one call
    const int ncol = nGoodHel * nevtB;       // number of BLAS right hand side columns
    // Column major scratch: JR/JI hold the ncolor x ncol jamps, ZR/ZI take
    // the SYMM results and MEcol one |M|^2 per column. Kept on the heap and grown
    // once per thread: for ncolor=60 and ncomb=128 this is a few hundred kB.
    static thread_local std::vector<fptype2> scratch;
    const size_t need = 4 * (size_t)ncolor * ncol + ncol;
    if( scratch.size() < need ) scratch.resize( need );
    fptype2* JR = scratch.data();
    fptype2* JI = JR + (size_t)ncolor * ncol;
    fptype2* ZR = JI + (size_t)ncolor * ncol;
    fptype2* ZI = ZR + (size_t)ncolor * ncol;
    fptype2* MEcol = ZI + (size_t)ncolor * ncol;
    // Transpose the jamps into the column major right hand side: colour is the
    // fast index, (helicity, event) the slow one.
    for( int ighel = 0; ighel < nGoodHel; ighel++ )
    {
      const cxtype_sv* jamp_sv = ghelAllJamp_sv + (size_t)ighel * nParityCS * ncolor;
      for( int ip = 0; ip < nParityCS; ip++ )
        for( int ieppV = 0; ieppV < neppV; ieppV++ )
        {
          const size_t off = (size_t)( ighel * nevtB + ip * neppV + ieppV ) * ncolor;
          for( int icol = 0; icol < ncolor; icol++ )
          {
#ifdef MGONGPU_CPPSIMD
            JR[off + icol] = cxreal( jamp_sv[ip * ncolor + icol] )[ieppV];
            JI[off + icol] = cximag( jamp_sv[ip * ncolor + icol] )[ieppV];
#else
            JR[off + icol] = cxreal( jamp_sv[ip * ncolor + icol] );
            JI[off + icol] = cximag( jamp_sv[ip * ncolor + icol] );
#endif
          }
        }
    }
    // Ztemp[ncolor][ncol] = ColorMatrix[ncolor][ncolor] * Jamps[ncolor][ncol], real and imaginary parts apart
    blas_symm( ncolor, ncol, cfsym.value, JR, ZR );
    blas_symm( ncolor, ncol, cfsym.value, JI, ZI );
    // |M|^2 for one (helicity, event) is the dot product of one column of Jamps with one column of Ztemp
    for( int j = 0; j < ncol; j++ )
    {
      const size_t off = (size_t)j * ncolor;
      fptype2 me = 0;
      for( int icol = 0; icol < ncolor; icol++ )
        me += JR[off + icol] * ZR[off + icol] + JI[off + icol] * ZI[off + icol];
      MEcol[j] = me; // may underflow #831
    }
    // *** STORE THE RESULTS ***
    // NB: MEs_ighel carries the running sum over helicities of |M|^2, which the
    // event by event choice of helicity needs. The color sum is no longer added
    // to allMEs one helicity at a time, so build those running sums here,
    // starting from whatever allMEs already held (fix #435).
    using E_ACCESS = HostAccessMatrixElements; // non-trivial access: buffer includes all events
    for( int ip = 0; ip < nParityCS; ip++ )
    {
      fptype_sv* running = ( ip == 0 ? MEs_ighel : MEs_ighel2 );
      fptype* MEsp = E_ACCESS::ieventAccessRecord( allMEs, ievt0 + ip * neppV );
      fptype_sv& MEsp_sv = E_ACCESS::kernelAccess( MEsp );
      for( int ieppV = 0; ieppV < neppV; ieppV++ )
      {
#ifdef MGONGPU_CPPSIMD
        fptype sum = MEsp_sv[ieppV];
        for( int ighel = 0; ighel < nGoodHel; ighel++ )
        {
          sum += MEcol[ighel * nevtB + ip * neppV + ieppV];
          running[ighel][ieppV] = sum;
        }
#else
        fptype sum = MEsp_sv;
        for( int ighel = 0; ighel < nGoodHel; ighel++ )
        {
          sum += MEcol[ighel * nevtB + ip * neppV + ieppV];
          running[ighel] = sum;
        }
#endif
      }
      MEsp_sv = running[nGoodHel - 1];
    }
  }
#endif
#endif
