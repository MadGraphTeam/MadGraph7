#ifdef MGONGPU_CPP_HAS_BLAS
      // The color matrix does not depend on the helicity, so keep the jamps of
      // every good helicity and hand them all to BLAS in a single call after
      // the loop (see color_sum_cpu_blas). Everything else the loop was doing -
      // the jamp2 sums for the color selection, the numerators and the
      // denominators - stays inside calculate_jamps and still happens once per
      // helicity, exactly as before.
      static thread_local std::vector<cxtype_sv> ghelJamp_sv( (size_t)ncomb * nParity * ncolor );
      for( int ighel = 0; ighel < %(sigmakin_hel_bound)s; ighel++ )
      {
        const int ihel = %(sigmakin_ihel_expr)s;
        cxtype_sv* jamp_sv = ghelJamp_sv.data() + (size_t)ighel * nParity * ncolor;
        for( int i = 0; i < nParity * ncolor; i++ ) jamp_sv[i] = cxzero_sv(); // calculate_jamps accumulates into jamp_sv
        // **NB! in "mixed" precision, using SIMD, calculate_jamps computes MEs for TWO neppV pages with a single channelId! #924
        bool storeChannelWeights = allChannelIds != nullptr || allrnddiagram != nullptr;
        calculate_jamps( ihel, allmomenta, allcouplings, iflavorVec, jamp_sv, storeChannelWeights, allNumerators, allDenominators, jamp2_sv, ievt00%(calc_jamps_ihlane_arg)s );
      }
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
      color_sum_cpu_blas( allMEs, MEs_ighel, MEs_ighel2, ghelJamp_sv.data(), %(sigmakin_hel_bound)s, ievt00 );
#else
      color_sum_cpu_blas( allMEs, MEs_ighel, nullptr, ghelJamp_sv.data(), %(sigmakin_hel_bound)s, ievt00 );
#endif
#else
