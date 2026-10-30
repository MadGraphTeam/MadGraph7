// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
//==========================================================================
// --hel_recycling (CPU only): the tabulated amplitude stage of the recycled
// build. A fragment of namespace madmatrix, not a standalone header: it is
// #included inside the namespace, after nParity and the process constants
// (njampso, ndiagrams), by SigmaKin.cc and by HelRecyclingChunks.h, which also
// include <immintrin.h> (x86) and <cstring> for it at file scope.
//
// A P1N group of the recycled block is the vertex evaluated once with one leg
// left off, then, for each helicity row that shares it (a member), the dot
// product with that row's own wavefunction of the peeled leg and the fold of
// the result into the row's tables. Written out member by member that is most
// of the recycled code -- 90% of the 1.29M statements of g g > t t~ g g g --
// and code that size is fetched from memory on every event page. The
// recycler (madgraph/madevent/hel_recycle.py, CppDialect._p1n_table) writes
// the members and their folds as static tables instead, which hr_p1n_fold
// reads in order. The arithmetic is the one the straight-line code does, in
// the same order.
//==========================================================================

// The rows mode stores every amplitude of every row into one table
// (ampAll_sv[nHrRows][nParity][hrNamp]) and reads it back only once the whole
// page is built, row by row in hr_row. A big table is long gone from the cache
// by then: kept there it only pushes out the wavefunctions and the fold tables
// that are read again, and every line of it is read from memory before it is
// written over. A non-temporal store writes it around the cache instead. That
// pays only when
// - each store fills whole cache lines (cxtype_amp_sv a multiple of 64 bytes:
//   simd_256 and simd_512). Half lines, filled one at a time far apart, are
//   flushed to memory half written, which is much slower;
// - the table of one call is big (MGONGPU_HR_STREAM_MIN_MB, 10 by default): a
//   smaller one is still in the caches when hr_row reads it.
// g g > t t~ g g g (30 MB per call at simd_256 in mixed precision), recycled
// over plain with all cores busy: 0.84 -> 1.23 on 2x EPYC 7452 (128 threads),
// 0.65 -> 0.94 on 2x EPYC 9655 (192 cores); at simd_128 (32 bytes) 1.15 -> 0.49.
// Over 21 processes (the helicity-recycling paper's among them) on the 2x EPYC
// 9655, at 1, 64 and 192 cores, the non-temporal stores lose to the normal
// ones at every table of up to 6.2 MB per call (g g > g g g g at simd_512:
// 0.86x on one core, 0.65x on all), and win at every one of 14.8 MB and more
// once many cores are busy (1.34x-1.57x at 64 and 192); 10 MB is between.
// MGONGPU_HR_TABLE_ENTRIES (nHrRows * hrNamp) comes from HelRecycling.inc, or
// from the chunk translation unit itself, which does not see HelRecycling.inc.
// x86 only (MGONGPU_HR_NO_STREAM, make HRNOSTREAM=1, turns it off).
#ifndef MGONGPU_HR_TABLE_ENTRIES
#define MGONGPU_HR_TABLE_ENTRIES 0 // not the rows mode: there is no table to store
#endif
#ifndef MGONGPU_HR_STREAM_MIN_MB
#define MGONGPU_HR_STREAM_MIN_MB 10
#endif
#if defined __x86_64__ and defined MGONGPU_CPPSIMD and !defined MGONGPU_HR_NO_STREAM
#if defined __AVX512F__
constexpr bool hrHasStream = sizeof( fptype_amp_sv ) == 32 || sizeof( fptype_amp_sv ) == 64; // _mm{256,512}_stream_si{256,512}
#elif defined __AVX__
constexpr bool hrHasStream = sizeof( fptype_amp_sv ) == 32;
#else
constexpr bool hrHasStream = false;
#endif
#else
constexpr bool hrHasStream = false;
#endif
constexpr bool hrStreamStores = hrHasStream && sizeof( cxtype_amp_sv ) % 64 == 0 &&
                                (size_t)( MGONGPU_HR_TABLE_ENTRIES ) * nParity * sizeof( cxtype_amp_sv ) > (size_t)MGONGPU_HR_STREAM_MIN_MB * 1024 * 1024;

// One store into the amplitude table (see hrStreamStores).
inline void
hr_store( cxtype_amp_sv& dst, const cxtype_amp_sv& amp )
{
#if defined __x86_64__ and defined MGONGPU_CPPSIMD and !defined MGONGPU_HR_NO_STREAM
  if constexpr( hrStreamStores )
  {
    // the real and the imaginary vector, each aligned to its own size
    fptype_amp_sv* p = reinterpret_cast<fptype_amp_sv*>( &dst );
    const fptype_amp_sv parts[2] = { amp.real(), amp.imag() };
    for( int k = 0; k < 2; k++ )
    {
#if defined __AVX512F__
      if constexpr( sizeof( fptype_amp_sv ) == 64 )
      {
        __m512i x;
        std::memcpy( &x, &parts[k], 64 );
        _mm512_stream_si512( reinterpret_cast<__m512i*>( p + k ), x );
        continue;
      }
#endif
      __m256i x;
      std::memcpy( &x, &parts[k], 32 );
      _mm256_stream_si256( reinterpret_cast<__m256i*>( p + k ), x );
    }
    return;
  }
#endif
  dst = amp;
}

// After the last store of a page and before hr_row reads the table: the
// non-temporal stores are not ordered with the others.
inline void
hr_store_fence()
{
#if defined __x86_64__ and defined MGONGPU_CPPSIMD and !defined MGONGPU_HR_NO_STREAM
  if constexpr( hrStreamStores ) _mm_sfence();
#endif
}

// One member of a P1N group: the wavefunction slot of its peeled leg, its
// helicity row in the recycled tables, and how many fold entries follow.
struct HrMember
{
  unsigned int slot;
  unsigned short row;
  unsigned short nfold;
};

// Add one amplitude of row `row` into that row's tables, entry by entry. An
// entry is 32 bits: kind (2: jampAll_sv, the second table tabAll_sv, numAll_sv,
// a store into tabAll_sv), how it adds (2: +=, -=, += real *, += complex *),
// its coefficient in c[] as (re, im) pairs (8), the index in the row (20).
// The second table, of `ntab` entries per row, is jampTmpAll_sv (the shared
// sub-expressions, fold mode) or ampAll_sv (the amplitudes, rows mode).
// Returns where the next amplitude's entries start.
inline const unsigned int*
hr_fold_amp( const cxtype_amp_sv& amp,
             const int row,
             const unsigned int* f,
             const int nfold,
             const double* c,
             const int ntab,
             cxtype_amp_sv* jampAll_sv,
             cxtype_amp_sv* tabAll_sv,
             fptype_amp_sv* numAll_sv,
             const int iParity,
             const bool storeChannelWeights )
{
  const size_t slot = (size_t)row * nParity + iParity;
  cxtype_amp_sv* jamp = jampAll_sv + slot * njampso;
  cxtype_amp_sv* tab = tabAll_sv + slot * ntab;
  for( int k = 0; k < nfold; k++ )
  {
    const unsigned int e = f[k];
    const unsigned int index = e & 0xfffffu;
    const unsigned int kind = e >> 30;
    if( kind == 3 )
    {
      hr_store( tab[index], amp );
      continue;
    }
    if( kind == 2 )
    {
      if( storeChannelWeights ) numAll_sv[slot * ndiagrams + index] += cxabs2( amp );
      continue;
    }
    cxtype_amp_sv& target = ( kind == 0 ? jamp[index] : tab[index] );
    const double* cc = c + 2 * ( ( e >> 20 ) & 0xffu );
    switch( ( e >> 28 ) & 0x3u )
    {
      case 0: target += amp; break;
      case 1: target -= amp; break;
      case 2: target += cc[0] * amp; break;
      default: target += cxtype( cc[0], cc[1] ) * amp; break;
    }
  }
  return f + nfold;
}

// The members of one P1N group: p1n holds the vertex with the peeled leg (of
// NCOMP components) left off; each member is its dot product with the member's
// own wavefunction of that leg.
// A fermion peel makes the flavour test here (see CppDialect._p1n_group).
template<class W_ACCESS, int NCOMP, bool FERMION>
inline void
hr_p1n_fold( const ALOHAOBJ& p1n,
             const ALOHAOBJ* aloha_obj,
             const HrMember* m,
             const int nm,
             const unsigned int* f,
             const double* c,
             const int ntab,
             cxtype_amp_sv* jampAll_sv,
             cxtype_amp_sv* tabAll_sv,
             fptype_amp_sv* numAll_sv,
             const int iParity,
             const bool storeChannelWeights )
{
  const cxtype_sv* _pt = W_ACCESS::kernelAccessConst( p1n.w );
  for( int i = 0; i < nm; i++ )
  {
    const ALOHAOBJ& w = aloha_obj[m[i].slot];
    const cxtype_sv* _pw = W_ACCESS::kernelAccessConst( w.w );
    cxtype_amp_sv amp;
    if constexpr( NCOMP == 1 )
      amp = _pt[0] * _pw[0];
    else
      amp = _pt[0] * _pw[0] + _pt[1] * _pw[1] + _pt[2] * _pw[2] + _pt[3] * _pw[3];
    if constexpr( FERMION )
    {
      if( p1n.flv_index != w.flv_index || p1n.flv_index == -1 ) amp = cxzero_sv();
    }
    f = hr_fold_amp( amp, m[i].row, f, m[i].nfold, c, ntab, jampAll_sv, tabAll_sv, numAll_sv, iParity, storeChannelWeights );
  }
}
