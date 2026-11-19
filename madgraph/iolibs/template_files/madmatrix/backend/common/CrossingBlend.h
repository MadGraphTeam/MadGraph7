// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
//==========================================================================
// Crossing symmetry (CPU/SIMD): the per-lane NSF blend of a crossed external
// leg, sp * a + sm * b, with sp (sm) 1 on the lanes whose leg kept (changed)
// side and 0 elsewhere. Used by the external calls of EvaluateDiagrams.inc
// and of the --hel_recycling block (HelRecyclingCalls.inc, and the chunks of
// HelRecyclingChunks_<k>.cc). The momenta of FPTYPE=v are double where sp is
// float. Include it after CPPProcess.h, which brings the vector types.
//==========================================================================
#ifndef CROSSINGBLEND_H
#define CROSSINGBLEND_H 1

namespace madmatrix
{
  template<typename T>
  inline T
  xblend( const fptype_sv& sp, const fptype_sv& sm, const T& a, const T& b )
  {
    return sp * a + sm * b;
  }
#ifdef MGONGPU_SIMD_DENOM64
  inline fptype_momenta_sv
  xblend( const fptype_sv& sp, const fptype_sv& sm, const fptype_momenta_sv& a, const fptype_momenta_sv& b )
  {
    return fpdenom_widen( sp ) * a + fpdenom_widen( sm ) * b;
  }
#endif
}

#endif // CROSSINGBLEND_H
