// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Integrated with the MadGraph7 project in Feb 2026.
//
// Process-specific compile-time data tables, generated once per subprocess,
// for backend-owned code (backend/{cpu,simd,gpu}/SigmaKin.cc) that can't take
// this data as a runtime parameter without losing constexpr-ness. Unlike
// ProcessData.h these are arrays, not scalars, and unlike ColorData.h
// there's no backend-conditional algorithm consuming them directly - it's
// pulled in via ProcessTables::name from backend-owned function bodies.
//
// Namespace-wrapped (unlike ProcessData.h) because it needs FLV_COUPLING,
// which is itself namespace-wrapped too (madmatrix::, see Parameters.h).

#ifndef PROCESSTABLES_H
#define PROCESSTABLES_H 1

#include "mgOnGpuConfig.h" // for __device__
#include "ProcessData.h"
#include "Parameters.h" // for FLV_COUPLING::max_flavor

namespace madmatrix
{
  namespace ProcessTables
  {
    using ProcessData::nDPF;
    constexpr int nMF = FLV_COUPLING::max_flavor; // max #merged flavors for any merged particle in the model

    // calculate_jamps' shared sub-expressions (see MadMatrixUFOHelasCallWriter.
    // build_jamp_plan): partial amplitude sums that several color flows share,
    // computed once and reused. 0 for a process with no such sharing.
    constexpr int nb_tmp_jamp = %(nb_tmp_jamp)d;

    // Dependent (event-by-event, running-alphas) flavor couplings: partner
    // indices and the per-flavor idcoup are pure compile-time constants (the
    // complex values are gathered per event page in calculate_jamps).
%(cdpfdecl)s

    // Decay-aware identical-particle (broken-)symmetry factor data, shared with
    // the Fortran / standalone_cpp exporters (_get_broken_symmetry_data).
    constexpr int broken_sym_ncomponents = %(broken_sym_ncomponents)d;
    constexpr int broken_sym_nentries = %(broken_sym_nentries)d;
    __device__ constexpr int broken_sym_component_starts[broken_sym_ncomponents] = { %(broken_sym_component_starts)s };
    __device__ constexpr int broken_sym_component_ends[broken_sym_ncomponents] = { %(broken_sym_component_ends)s };
    __device__ constexpr int broken_sym_component_old_factors[broken_sym_ncomponents] = { %(broken_sym_component_old_factors)s };
    __device__ constexpr int broken_sym_pid_list[broken_sym_nentries] = { %(broken_sym_pid_list)s };
    __device__ constexpr int broken_sym_block_starts[broken_sym_nentries] = { %(broken_sym_block_starts)s };
    __device__ constexpr int broken_sym_block_lengths[broken_sym_nentries] = { %(broken_sym_block_lengths)s };

%(crossing_tables)s
    // Slot relabelling of crossing code `cross`: perm[k] is the input slot
    // landing in crossed slot k and ic[k] its NSF sign flip. Left a valid
    // permutation (the identity for an inapplicable code) so a momentum gather
    // never reads out of range; returns whether the code is applicable.
    __host__ __device__ inline bool cross_perm_ic( int cross, int* perm, int* ic )
    {
      constexpr int npar = ProcessData::npar;
      for( int k = 0; k < npar; k++ ) { perm[k] = k; ic[k] = 1; }
      if( cross < 0 || cross >= ncross ) return false;
      const int xi = cross / ( npar + 1 );
      const int xj = cross %% ( npar + 1 );
      // Overlapping-swap codes compose into a 3-cycle the consumers read
      // with opposite orientation: pure redundancy, invalid.
      if( xi != 0 && xi != 1 && xj != 0 && xj != 2 &&
          ( xi == 2 || xj == 1 || xi == xj ) ) return false;
      if( xi != 0 && xi != 1 )
      { int t = perm[0]; perm[0] = perm[xi - 1]; perm[xi - 1] = t; ic[0] = -ic[0]; ic[xi - 1] = -ic[xi - 1]; }
      if( xj != 0 && xj != 2 )
      { int t = perm[1]; perm[1] = perm[xj - 1]; perm[xj - 1] = t; ic[1] = -ic[1]; ic[xj - 1] = -ic[xj - 1]; }
      return true;
    }
  }
}

#endif // PROCESSTABLES_H
