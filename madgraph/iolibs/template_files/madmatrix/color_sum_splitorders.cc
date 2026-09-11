// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created originally by: A. Valassi (Sep 2025) for the MG5aMC CUDACPP plugin.
// Further modified by: A. Valassi (2025).
// Integrated with the MadGraph7 project in Feb 2026.
//
// SQUARED SPLIT ORDERS. This is color_sum.cc for a process whose '^2'
// constraint leaves more than one amplitude split order, and it is a separate
// file because the sum itself changes shape rather than gaining a hole: the
// jamps arrive as nampso vectors end to end (njampso = nampso*ncolor) and the
// color sum pairs them, which is the Fortran GET_MATRIX contract
// JAMP(NCOLOR,NAMPSO) -> RES(NSQAMPSO).
//
// THE PAIR LOOP RUNS OVER ALL nampso*nampso ORDERED PAIRS, never a triangle.
// The color contraction below keeps the triangular matrix whose off-diagonal
// is doubled, which for a single jamp vector is just the statement that the
// (i,j) and (j,i) terms are conjugates. For two DIFFERENT jamp vectors they
// are not, so the value computed for one ordered pair (m,n) is NOT that pair's
// contribution -- only (m,n) and (n,m) together are, and they are equal to the
// true sum of the two. That is why both orderings must be summed, and why it
// is safe to mask on the squared order: sqSoIndex is symmetric (a squared
// order is the SUM of the two amplitude orders), so a pair and its transpose
// are always kept or dropped together and the masked total stays real.

#include "color_sum.h"

#include "mgOnGpuConfig.h"

#include "MemoryAccessMatrixElements.h"

#ifdef MGONGPUCPP_GPUIMPL
#error The squared split-order color sum is implemented for the CPU (SIMD) backend only. \
The GPU jamp buffers are sized for a single jamp vector per helicity (ncolor, not njampso), \
so a GPU build of this process would silently sum the wrong thing. Use a CPU backend, or \
generate the process with a constraint that leaves a single amplitude split order.
#endif

namespace mg5amcCpu
{
  constexpr int ncolor = CPPProcess::ncolor;     // the number of leading colors
  constexpr int nampso = CPPProcess::nampso;     // the amplitude split orders
  constexpr int njampso = CPPProcess::njampso;   // ncolor * nampso: the jamps of every order, end to end
  constexpr int nsqampso = CPPProcess::nsqampso; // the squared orders their pairs produce

  //--------------------------------------------------------------------------

  // *** COLOR MATRIX BELOW ***
%(color_matrix_lines)s

  //--------------------------------------------------------------------------

  // *** SQUARED SPLIT ORDERS BELOW ***
%(sqso_tables)s

  //--------------------------------------------------------------------------

  void
  color_sum_cpu( fptype* allMEs,              // output: allMEs[nevt], add |M|^2 for one specific helicity
                 const cxtype_sv* allJamp_sv, // input: jamp_sv[njampso] (float/double) or jamp_sv[2*njampso] (mixed) for one specific helicity
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
    fptype_sv deltaMEs = { 0 };
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
    fptype_sv deltaMEs_next = { 0 };
#endif
    // Gather the color flows the sum runs over, for each amplitude split order.
    // The order index strides by ncolor, exactly as calculate_jamps wrote them.
    // NB in mixed mode the two neppV vectors of allJamp_sv, at ijamp and at
    // njampso+ijamp, are two halves of the event page and not two colors.
    fptype2_sv jampR_sv[nampso][ncolor];
    fptype2_sv jampI_sv[nampso][ncolor];
    for( int iao = 0; iao < nampso; iao++ )
    {
      for( int icol = 0; icol < ncolor; icol++ )
      {
        const int ijamp = iao * ncolor + icol;
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
        jampR_sv[iao][icol] = fpvmerge( cxreal( allJamp_sv[ijamp] ), cxreal( allJamp_sv[njampso + ijamp] ) );
        jampI_sv[iao][icol] = fpvmerge( cximag( allJamp_sv[ijamp] ), cximag( allJamp_sv[njampso + ijamp] ) );
#else
        jampR_sv[iao][icol] = (fptype2_sv)( cxreal( allJamp_sv[ijamp] ) );
        jampI_sv[iao][icol] = (fptype2_sv)( cximag( allJamp_sv[ijamp] ) );
#endif
      }
    }
    // Loop over ORDERED pairs of amplitude split orders (see the note at the top:
    // this must not be turned into a triangle over iao/jao)
    for( int iao = 0; iao < nampso; iao++ )
    {
      for( int jao = 0; jao < nampso; jao++ )
      {
        // The squared order this pair contributes to, dropped if the process
        // asked for a contribution that does not include it
        if( !chosenSqso[sqSoIndex[iao][jao]] ) continue;
        // Loop over icol
        for( int icol = 0; icol < ncolor; icol++ )
        {
          // Diagonal terms (the color contraction is still triangular)
          fptype2_sv ztempR_sv = cf2.value[icol][icol] * jampR_sv[iao][icol];
          fptype2_sv ztempI_sv = cf2.value[icol][icol] * jampI_sv[iao][icol];
          // Loop over jcol
          for( int jcol = icol + 1; jcol < ncolor; jcol++ )
          {
            // Off-diagonal terms
            ztempR_sv += cf2.value[icol][jcol] * jampR_sv[iao][jcol];
            ztempI_sv += cf2.value[icol][jcol] * jampI_sv[iao][jcol];
          }
          // ztemp comes from the jamps of order iao, and is contracted with those
          // of order jao (Fortran: ZTEMP from JAMP(:,M), times DCONJG(JAMP(I,N)))
          fptype2_sv deltaMEs2 = ( jampR_sv[jao][icol] * ztempR_sv + jampI_sv[jao][icol] * ztempI_sv ); // may underflow #831
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
          deltaMEs += fpvsplit0( deltaMEs2 );
          deltaMEs_next += fpvsplit1( deltaMEs2 );
#else
          deltaMEs += deltaMEs2;
#endif
        }
      }
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

  //--------------------------------------------------------------------------
}
