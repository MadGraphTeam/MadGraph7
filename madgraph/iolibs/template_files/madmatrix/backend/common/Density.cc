// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Integrated with the MadGraph7 project in Feb 2026.
//
// Spin-density matrices for MadSpin (see Density.h for the interface and its
// conventions). Shared by the C++ backends: the jamps come from the backend's
// calculate_jamps, one event page at a time, and the color contraction between
// two helicity combinations is done here, per event, in double precision.

#include "Density.h"

#include "mgOnGpuConfig.h"

#include "ColorData.h" // colorMatrix/colorDenom, sqSoIndex/chosenSqso
#include "MemoryAccessMomenta.h"
#include "Parameters.h" // ndcoup
#include "ProcessData.h"
#include "ProcessTables.h" // the crossing table (use_crossing, ncross, cross_pinv)
#include "SigmaKin.h"      // calculate_jamps, computeDependentCouplings, getHelicities

#include <algorithm>
#include <cmath>
#include <complex>
#include <map>
#include <mutex>
#include <new>
#include <type_traits>
#include <vector>

namespace madmatrix
{
  namespace
  {
    using namespace ProcessData;
    using cplx = std::complex<double>;
    using ProcessTables::ncross;
    using ProcessTables::use_crossing;

    // Events evaluated by one call of calculate_jamps (two SIMD pages in mixed precision)
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
    constexpr int nParity = 2;
#else
    constexpr int nParity = 1;
#endif
    constexpr int npage = nParity * neppV;
    static_assert( npage % MemoryAccessMomentaBase::neppM == 0, "an event page must hold whole momenta pages" );

    // A helicity row is evaluated on every event until nscan events have been
    // seen, and afterwards only if one of its jamps was ever non-zero: the
    // Fortran SMATRIX filter (NTRY < 20). Only real events count: the copies
    // padding a page to npage lanes are not new points.
    constexpr int nscan = 20;

    struct HelicityFilter
    {
      int ntry = 0;
      std::vector<char> good = std::vector<char>( ncomb, 0 );
      std::vector<int> base; // the rows of the spectator sum (see density_kernel)
      bool hasBase = false;
    };

    // One filter per flavor and helicity structure (the open legs and their
    // combinations decide which rows contribute, and are part of the key)
    std::map<std::vector<int>, HelicityFilter> g_filters;

    // A growable buffer with the alignment the SIMD kernels need
    template<typename T>
    struct AlignedBuffer
    {
      T* data = nullptr;
      size_t capacity = 0;
      T* get( const size_t n )
      {
        if( n > capacity )
        {
          release();
          data = new( std::align_val_t( mgOnGpu::cppAlign ) ) T[n]();
          capacity = n;
        }
        return data;
      }
      void release()
      {
        if( data ) ::operator delete[]( data, std::align_val_t( mgOnGpu::cppAlign ) );
        data = nullptr;
        capacity = 0;
      }
      ~AlignedBuffer() { release(); }
    };

    // Everything a call needs, kept from one call to the next: MadSpin evaluates
    // one point per call, and allocating all of it again every time costs more
    // than the jamps of a small process. Used under g_mutex.
    struct Scratch
    {
      AlignedBuffer<fptype_momenta> momenta;
      AlignedBuffer<fptype> gs;
      AlignedBuffer<fptype> couplings;
      AlignedBuffer<unsigned int> flavors;
      AlignedBuffer<fptype_amp> numerators;
      AlignedBuffer<fptype_amp> denominators;
      std::vector<cxtype_amp_sv> jamps;
      std::vector<short> rows;
      std::vector<cplx> lj, mj, inter;
      std::vector<size_t> events;
      std::vector<int> key, pos0;
    };
    Scratch g_scratch;
    std::mutex g_mutex;

    double g_beampol[2] = { 0., 0. };

    // The color matrix normalised by its denominators, in double precision,
    // full (not triangular): the two jamp vectors contracted are different.
    struct NormalizedColorMatrix
    {
      constexpr NormalizedColorMatrix()
        : value()
      {
        for( int a = 0; a < ncolor; a++ )
          for( int b = 0; b < ncolor; b++ )
            value[a][b] = double( ColorMatrixData::colorMatrix[a][b] ) / double( ColorMatrixData::colorDenom[a] );
      }
      double value[ncolor][ncolor];
    };
    constexpr NormalizedColorMatrix cf{};

    // One SIMD lane of a jamp component (the value itself without SIMD)
    template<typename V>
    inline double lane( const V& v, const int i )
    {
      if constexpr( std::is_arithmetic_v<V> )
      {
        (void)i;
        return v;
      }
      else
        return v[i];
    }

    // The crossing an extended flavor id carries (ProcessTables::use_crossing):
    // the row cross of the crossing table in the input-slot view, input slot k
    // (the caller's leg k) being fed to base slot pinv[k], with the side flip
    // sgn[k]. The identity for cross 0 (and without crossing).
    //
    // The caller gives the momenta, the open legs and their helicities in its
    // own leg order and the helicities as physical ones, as the Fortran
    // GET_DENSITY_IDX takes them (its APPLY_CROSSING_TABLE view). calculate_jamps
    // gathers the momenta itself, from the flavor id, but evaluates the base
    // helicity row of the base slot s with the opposite helicity where the leg
    // changes side (the (-nsf, -h) call of a crossed external, see
    // _crossing_external_block): the physical helicity h of input slot k is
    // therefore row[pinv[k]] = sgn[k] * h. The spectator sum runs over every row,
    // which a side flip only reorders.
    struct CrossView
    {
      int cross = 0;
      int pinv[npar];
      int sgn[npar];
      explicit CrossView( const int xcross )
        : cross( xcross )
      {
        if constexpr( use_crossing )
        {
          ProcessTables::cross_pinv( xcross, pinv, sgn );
        }
        else
        {
          for( int k = 0; k < npar; k++ )
          {
            pinv[k] = k;
            sgn[k] = 1;
          }
        }
      }
      // the physical helicity of input slot k in the base helicity row hel
      inline int physical( const short* hel, const int k ) const { return sgn[k] * hel[pinv[k]]; }
    };

    // Beam-polarisation weight of helicity row hel, as the Fortran density: the
    // helicities of the incoming legs of the process the flavor id names (a
    // crossing may have moved them into other base slots)
    inline double beam_polarisation_factor( const short* hel, const CrossView& xv )
    {
      double factor = 1.;
      if( npari != 2 ) return factor; // a 1 -> N process: leg 1 is a resonance, not a beam
      for( int jj = 0; jj < npari; jj++ )
      {
        const double pol = g_beampol[jj];
        if( std::abs( pol ) <= 1. ) continue;
        factor *= ( xv.physical( hel, jj ) == ( pol > 0 ? 1 : -1 ) ) ? std::abs( pol ) : 2. - std::abs( pol );
      }
      return factor;
    }

    // What the density and |M|^2 of an extended flavor id are multiplied by, as
    // the Fortran GET_INTER_RESCALE/SMATRIX and sigmaKin: 1/iden times the
    // identical-particle factor of the actual flavors (broken_symmetry_factor)
    // for the base process, and the crossed denominator (the initial-state
    // spin*color average of the crossed process times its identical-final-state
    // factor) for a crossing; 0 for a row out of the table.
    inline double normalisation( const unsigned int flavorId )
    {
      const int xcross = (int)( flavorId / (unsigned int)nmaxflavor );
      const int flav = (int)( flavorId % (unsigned int)nmaxflavor );
      if( xcross == 0 ) return broken_symmetry_factor( flav ) / double( helcolDenominators[0] );
      if( xcross >= ncross ) return 0.;
      const double denominator = double( ProcessTables::xspincol_tab[xcross] ) * double( ident_cross( xcross, flav ) );
      return denominator > 0. ? 1. / denominator : 0.;
    }

    // Adds the density of nevt events (whole pages of one flavor id each) to inter[nevt][ninter],
    // the first nreal of them real and the others padding
    void
    density_kernel( const fptype_momenta* allmomenta,
                    const fptype* allcouplings,
                    const unsigned int* allflavors,
                    const int nevt,
                    const int nreal,
                    const int nChanging,
                    const int* pos0, // 0-based
                    const int nComb,
                    const int* allowHel,
                    const CrossView& xv,
                    HelicityFilter& filter,
                    Scratch& s,
                    cplx* inter )
    {
      const int ninter = nComb * ( nComb + 1 ) / 2;
      const short* hels = getHelicities();
      if( !filter.hasBase )
      {
        // The spectator sum runs over the rows whose open legs carry allowHel[0]
        for( int ihel = 0; ihel < ncomb; ihel++ )
        {
          bool match = true;
          for( int n = 0; n < nChanging && match; n++ )
            match = hels[ihel * npar + pos0[n]] == allowHel[n];
          if( match ) filter.base.push_back( ihel );
        }
        filter.hasBase = true;
      }
      s.rows.resize( (size_t)nComb * npar );
      s.jamps.resize( (size_t)nComb * nParity * njampso );
      s.lj.resize( (size_t)nComb * njampso );
      s.mj.resize( (size_t)nComb * njampso );
      fptype_amp* numerators = s.numerators.get( (size_t)nevt * ndiagrams );
      fptype_amp* denominators = s.denominators.get( nevt );
      for( int ievt00 = 0; ievt00 < nevt; ievt00 += npage )
      {
        const bool scanning = filter.ntry < nscan;
        for( const int ihel : filter.base )
        {
          if( !scanning && !filter.good[ihel] ) continue;
          const double polfact = beam_polarisation_factor( hels + ihel * npar, xv );
          for( int icomb = 0; icomb < nComb; icomb++ )
          {
            short* row = &s.rows[(size_t)icomb * npar];
            for( int ipar = 0; ipar < npar; ipar++ ) row[ipar] = hels[ihel * npar + ipar];
            for( int n = 0; n < nChanging; n++ ) row[pos0[n]] = allowHel[icomb * nChanging + n];
            cxtype_amp_sv* jamp_sv = &s.jamps[(size_t)icomb * nParity * njampso];
            for( int i = 0; i < nParity * njampso; i++ ) jamp_sv[i] = cxzero_sv<cxtype_amp_sv>();
            calculate_jamps( row, allmomenta, allcouplings, allflavors, jamp_sv, false, numerators, denominators, nullptr, ievt00 );
          }
          bool nonzero = false;
          for( int iParity = 0; iParity < nParity; iParity++ )
          {
            for( int ieppV = 0; ieppV < neppV; ieppV++ )
            {
              for( int icomb = 0; icomb < nComb; icomb++ )
              {
                const cxtype_amp_sv* jamp_sv = &s.jamps[( (size_t)icomb * nParity + iParity ) * njampso];
                for( int ijamp = 0; ijamp < njampso; ijamp++ )
                {
                  const cplx z( lane( cxreal( jamp_sv[ijamp] ), ieppV ), lane( cximag( jamp_sv[ijamp] ), ieppV ) );
                  nonzero = nonzero || z != 0.;
                  s.lj[(size_t)icomb * njampso + ijamp] = z;
                }
                // C JAMP_I, for every amplitude order
                for( int iao = 0; iao < nampso; iao++ )
                {
                  const cplx* j = &s.lj[(size_t)icomb * njampso + iao * ncolor];
                  cplx* m = &s.mj[(size_t)icomb * njampso + iao * ncolor];
                  for( int a = 0; a < ncolor; a++ )
                  {
                    cplx sum = 0.;
                    for( int b = 0; b < ncolor; b++ ) sum += cf.value[a][b] * j[b];
                    m[a] = sum;
                  }
                }
              }
              // rho(I,J) = JAMP_J^dagger C JAMP_I, summed over the ordered pairs of
              // amplitude orders the process keeps (see color_sum_cpu_splitorders)
              cplx* out = inter + (size_t)( ievt00 + iParity * neppV + ieppV ) * ninter;
              int sol = 0;
              for( int I = 0; I < nComb; I++ )
              {
                for( int J = I; J < nComb; J++ )
                {
                  cplx z = 0.;
                  for( int iao = 0; iao < nampso; iao++ )
                  {
                    for( int jao = 0; jao < nampso; jao++ )
                    {
                      if( !ColorMatrixData::chosenSqso[ColorMatrixData::sqSoIndex[iao][jao]] ) continue;
                      const cplx* jj = &s.lj[(size_t)J * njampso + jao * ncolor];
                      const cplx* mi = &s.mj[(size_t)I * njampso + iao * ncolor];
                      for( int a = 0; a < ncolor; a++ ) z += std::conj( jj[a] ) * mi[a];
                    }
                  }
                  out[sol++] += polfact * z;
                }
              }
            }
          }
          if( nonzero ) filter.good[ihel] = 1;
        }
        filter.ntry += std::max( 0, std::min( npage, nreal - ievt00 ) );
      }
    }

    // One momentum given in the rest frame of b, the momentum of that frame in
    // the lab, taken to the lab: HELAS boostx, with the arithmetic of MadSpin's
    // lhe_parser.Event.boost (MadSpin decays are generated at rest).
    inline void boost_from_rest_frame( double p[4], const double b[4] )
    {
      const double pnorm = b[1] * b[1] + b[2] * b[2] + b[3] * b[3];
      if( pnorm == 0. ) return; // b at rest: nothing to do
      const double mass = std::sqrt( std::max( b[0] * b[0] - pnorm, 0. ) );
      const double invMass = 1. / mass;
      const double k = ( b[0] - mass ) / pnorm;
      const double s3 = p[1] * b[1] + p[2] * b[2] + p[3] * b[3];
      const double lf = ( p[0] + k * s3 ) * invMass;
      p[0] = ( p[0] * b[0] + s3 ) * invMass;
      p[1] += b[1] * lf;
      p[2] += b[2] * lf;
      p[3] += b[3] * lf;
    }

    // The density (or, with no open leg, the helicity sum) of count events, each
    // normalised as the Fortran density and SMATRIX (normalisation, above):
    // out[count][ninter]. flavor holds extended ids (cross * nmaxflavor + flavor)
    // with crossing, pos and allowHel follow the legs of the process the id
    // names (see CrossView). Groups the events by flavor id, since all the events
    // of a page must share it, padding each group to whole pages with copies of
    // its first event (zero momenta would give nan, see umami.cc). boosts, if not
    // null, holds per event the lab momentum of the frame its momenta are given
    // in (boost_from_rest_frame).
    int
    evaluate( const size_t count,
              const double* momenta,
              const double* boosts,
              const double* alpha_s,
              const unsigned int* flavor,
              const int nChanging,
              const int* pos,
              const int nComb,
              const int* allowHel,
              cplx* out )
    {
      if( nComb < 1 || nChanging < 0 || ( nChanging == 0 && nComb != 1 ) ) return 1;
      for( int n = 0; n < nChanging; n++ )
        if( pos[n] < 1 || pos[n] > npar ) return 2;
      const unsigned int nflavorId = (unsigned int)( ncross * nmaxflavor );
      for( size_t i = 0; i < count; i++ )
        if( flavor[i] >= nflavorId ) return 3;
      std::lock_guard<std::mutex> lock( g_mutex );
      Scratch& s = g_scratch;
      const int ninter = nComb * ( nComb + 1 ) / 2;
      constexpr size_t neppM = MemoryAccessMomentaBase::neppM;
      std::vector<unsigned int> ids( flavor, flavor + count );
      std::sort( ids.begin(), ids.end() );
      ids.erase( std::unique( ids.begin(), ids.end() ), ids.end() );
      std::vector<int> allowHelBase( (size_t)nComb * nChanging );
      for( const unsigned int flavorId : ids )
      {
        s.events.clear();
        for( size_t i = 0; i < count; i++ )
          if( flavor[i] == flavorId ) s.events.push_back( i );
        const size_t nreal = s.events.size();
        const double norm = normalisation( flavorId );
        if( norm == 0. )
        {
          // a crossing out of the table (or one that cannot be applied): no
          // process, every entry 0, as the Fortran GET_DENSITY_IDX
          for( size_t k = 0; k < nreal; k++ )
            for( int sol = 0; sol < ninter; sol++ )
              out[s.events[k] * ninter + sol] = 0.;
          continue;
        }
        // the open legs and their helicities in the base slots and the base
        // row convention (CrossView)
        const CrossView xv( (int)( flavorId / (unsigned int)nmaxflavor ) );
        s.pos0.resize( nChanging );
        for( int n = 0; n < nChanging; n++ ) s.pos0[n] = xv.pinv[pos[n] - 1];
        for( int icomb = 0; icomb < nComb; icomb++ )
          for( int n = 0; n < nChanging; n++ )
            allowHelBase[(size_t)icomb * nChanging + n] = xv.sgn[pos[n] - 1] * allowHel[(size_t)icomb * nChanging + n];
        const size_t nevt = ( nreal + npage - 1 ) / npage * npage;
        fptype_momenta* allmomenta = s.momenta.get( nevt * npar * 4 );
        fptype* gs = s.gs.get( nevt );
        fptype* couplings = s.couplings.get( nevt * Parameters_dependentCouplings::ndcoup * 2 );
        unsigned int* flavors = s.flavors.get( nevt );
        for( size_t k = 0; k < nevt; k++ )
        {
          const size_t i = s.events[k < nreal ? k : 0];
          const size_t ipage = k / neppM, ivec = k % neppM;
          for( int ipar = 0; ipar < npar; ipar++ )
          {
            double p[4] = { momenta[( i * npar + ipar ) * 4], momenta[( i * npar + ipar ) * 4 + 1],
                            momenta[( i * npar + ipar ) * 4 + 2], momenta[( i * npar + ipar ) * 4 + 3] };
            if( boosts ) boost_from_rest_frame( p, boosts + i * 4 );
            for( int imom = 0; imom < 4; imom++ )
              allmomenta[ipage * npar * 4 * neppM + ipar * 4 * neppM + imom * neppM + ivec] = p[imom];
          }
          gs[k] = std::sqrt( 4. * M_PI * alpha_s[i] );
          // the extended id: calculate_jamps gathers the momenta of a crossing
          flavors[k] = flavorId;
        }
        computeDependentCouplings( gs, couplings, nevt );
        s.key.assign( { (int)flavorId, nChanging } );
        s.key.insert( s.key.end(), s.pos0.begin(), s.pos0.end() );
        s.key.push_back( nComb );
        s.key.insert( s.key.end(), allowHelBase.begin(), allowHelBase.end() );
        s.inter.assign( nevt * ninter, 0. );
        density_kernel( allmomenta, couplings, flavors, (int)nevt, (int)nreal, nChanging, s.pos0.data(), nComb, allowHelBase.data(), xv, g_filters[s.key], s, s.inter.data() );
        for( size_t k = 0; k < nreal; k++ )
          for( int sol = 0; sol < ninter; sol++ )
            out[s.events[k] * ninter + sol] = s.inter[k * ninter + sol] * norm;
      }
      return 0;
    }
  }
}

extern "C"
{
  int madmatrix_density( size_t count,
                         const double* momenta,
                         const double* alpha_s,
                         const unsigned int* flavor,
                         int n_changing,
                         const int* pos,
                         int n_comb,
                         const int* allow_hel,
                         double* inter )
  {
    if( n_changing < 1 ) return 1;
    return madmatrix::evaluate( count, momenta, nullptr, alpha_s, flavor, n_changing, pos, n_comb, allow_hel,
                                reinterpret_cast<std::complex<double>*>( inter ) );
  }

  int madmatrix_density_boosted( size_t count,
                                 const double* momenta,
                                 const double* boosts,
                                 const double* alpha_s,
                                 const unsigned int* flavor,
                                 int n_changing,
                                 const int* pos,
                                 int n_comb,
                                 const int* allow_hel,
                                 double* inter )
  {
    if( n_changing < 1 || !boosts ) return 1;
    return madmatrix::evaluate( count, momenta, boosts, alpha_s, flavor, n_changing, pos, n_comb, allow_hel,
                                reinterpret_cast<std::complex<double>*>( inter ) );
  }

  int madmatrix_matrix_element( size_t count,
                                const double* momenta,
                                const double* alpha_s,
                                const unsigned int* flavor,
                                double* me )
  {
    std::vector<std::complex<double>> sum( count );
    const int status = madmatrix::evaluate( count, momenta, nullptr, alpha_s, flavor, 0, nullptr, 1, nullptr, sum.data() );
    if( status != 0 ) return status;
    for( size_t i = 0; i < count; i++ )
      me[i] = sum[i].real();
    return 0;
  }

  void madmatrix_set_beampol( double pol1, double pol2 )
  {
    madmatrix::g_beampol[0] = pol1;
    madmatrix::g_beampol[1] = pol2;
  }

  void madmatrix_process_info( int* info )
  {
    info[0] = ProcessData::npar;
    info[1] = ProcessData::npari;
    info[2] = ProcessData::ncomb;
    info[3] = ProcessData::nmaxflavor;
    info[4] = ProcessData::ncolor;
    info[5] = ProcessData::helcolDenominators[0];
    info[6] = madmatrix::ncross;
  }

  double madmatrix_iden( unsigned int flavor_id )
  {
    const int xcross = (int)( flavor_id / (unsigned int)ProcessData::nmaxflavor );
    if( xcross == 0 ) return ProcessData::helcolDenominators[0];
    if( xcross >= madmatrix::ncross ) return 0.;
    return double( madmatrix::ProcessTables::xspincol_tab[xcross] ) *
           double( madmatrix::ident_cross( xcross, (int)( flavor_id % (unsigned int)ProcessData::nmaxflavor ) ) );
  }
}
