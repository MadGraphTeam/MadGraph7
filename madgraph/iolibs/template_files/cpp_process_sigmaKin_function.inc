// Local variables and constants
std::complex<double> **wfs;
// Helicities for the process
%(helicity_matrix)s
// Denominators: spins, colors and identical particles
const int denominator = %(den_factors)s;
// Flavor lookup table
%(flavor_table)s
%(cross_tables_decode)s
const int* flavor = &flavor_table[%(fidx)s][0];

ntry[%(fidx)s]++;

// Define permutation
%(cross_perm_block)s

double matrix_element = 0.;

// C-parity helicity de-duplication (mirror of the fortran SMATRIX): flip[ihel]
// is the helicity row with every helicity negated, an involution built once
// from the helicity table. dedup_ok gates the reuse to the UNCROSSED process:
// a crossing permutes/sign-flips the helicities, so a base-row flip is no
// longer the crossed C-parity partner and the crossed flavors keep the full sum.
if (!flip_ready){
    for(int i = 0; i < ncomb; i++){
        flip[i] = i;
        for(int j = 0; j < ncomb; j++){
            bool same = true;
            for(int k = 0; k < nexternal; k++){
                if (helicities[j][k] != -helicities[i][k]){
                    same = false;
                }
            }
            if (same){
                flip[i] = j;
                break;
            }
        }
    }
    flip_ready = true;
}
const bool dedup_ok = (%(csym_dedup_ok)s);
// The C-parity verdict is validated on the first nscan_csym-1 calls the reuse
// may apply to (ntry_csym counts only those, like the fortran NTRY_CSYM, and
// stops at nscan_csym): while csym_scan every row is compared with its flip.
// From then on dedup computes only the lower-index row of each C-parity pair
// and counts it twice -- unless the scan refused the pairing (csym_bad).
const int nscan_csym = 20;
if (dedup_ok && ntry_csym[%(fidx)s] < nscan_csym){
    ntry_csym[%(fidx)s]++;
}
const bool csym_scan = dedup_ok && ntry_csym[%(fidx)s] < nscan_csym;
const bool dedup = dedup_ok && !csym_scan && !csym_bad[%(fidx)s];

// Full sum over the good helicities: always in the default sum_hel == 0 mode,
// and in the helicity-sampling mode (sum_hel > 0) while the good-helicity
// filter trains (ntry < 10) or the C-parity verdict is validated. A crossed
// call (!dedup_ok) always takes the full sum: igood/igoodrep hold rows of the
// UNCROSSED process, which are not the rows a crossing evaluates.
if (sum_hel[%(fidx)s] == 0 || ntry[%(fidx)s] < 10 || !dedup_ok || csym_scan){
    double tstore[ncomb] = {};
    for(int ihel = 0; ihel < ncomb; ihel ++){
        // Settled C-parity pairing: the higher-index row of every pair is
        // skipped, its lower-index partner below carries both contributions.
        if (dedup && ihel > flip[ihel]){
            continue;
        }
        %(cross_ghidx_setup)sif (%(cross_goodhel_gate)s){
            calculate_wavefunctions(perm, helicities[ihel], flavor%(cross_cw_args)s);
            %(get_matrix_t_lines)s
            tstore[ihel] = t;
            // The representative counts for its skipped partner too. amp2 and
            // jamp2, accumulated per computed row, then hold half their full
            // helicity sum: a uniform factor, which leaves their ratios (all a
            // channel or colour weight uses) unchanged, as in madevent.
            if (dedup){
                t = t + t;
            }
            matrix_element += t;
            // Store which helicities give non-zero result
            %(cross_goodhel_train)s
        }
    }
    if (csym_scan){
        // Drop the C-parity pairing of any row whose flipped partner gave a
        // different |M|^2 (parity/C violation). One mismatch at any scan point
        // permanently invalidates the pair (robust, like the zero-filter).
        // All-or-nothing per flavor: a self-paired row (flip[ihel]==ihel) has no
        // distinct partner, and one mismatching pair is enough to give up, so
        // that when the reuse does run it halves the loop uniformly.
        // Two rows that are both numerically zero differ only by roundoff, on
        // which a purely relative test fails at random (and one such pair
        // vetoes every real one): the difference must also be significant
        // against tsmax, the largest |M|^2 of this scan point (as in fortran).
        double tsmax = 0.;
        for(int ihel = 0; ihel < ncomb; ihel++){
            double tt = tstore[ihel];
            if (tt < 0){
                tt = -tt;
            }
            if (tt > tsmax){
                tsmax = tt;
            }
        }
        for(int ihel = 0; ihel < ncomb; ihel++){
            if (flip[ihel] == ihel){
                csym_bad[%(fidx)s] = true;
            } else if (flip[ihel] > ihel){
                double a = tstore[ihel];
                double b = tstore[flip[ihel]];
                double diff = a - b;
                if (diff < 0){
                    diff = -diff;
                }
                double aa = a;
                if (aa < 0){
                    aa = -aa;
                }
                double bb = b;
                if (bb < 0){
                    bb = -bb;
                }
                if (diff > 1e-6 * (aa + bb) && diff > 1e-12 * tsmax){
                    csym_bad[%(fidx)s] = true;
                }
            }
        }
        // The reuse skips a good helicity only if its representative is itself
        // a good helicity: a pair split across the good/dropped boundary would
        // lose the skipped row entirely. Fold that into the flavor verdict.
        for(int g = 0; g < ngood[%(fidx)s]; g++){
            int ihel = igood[%(fidx)s][g];
            if (flip[ihel] == ihel || !goodhel[%(fidx)s][flip[ihel]]){
                csym_bad[%(fidx)s] = true;
            }
        }
        // Reduce the good helicities to the lower-index representative of every
        // surviving C-parity pair, carrying a doubled weight (all of them with
        // weight 1 once the pairing is refused): the list the sampling mode
        // below draws from.
        nrep[%(fidx)s] = 0;
        for(int g = 0; g < ngood[%(fidx)s]; g++){
            int ihel = igood[%(fidx)s][g];
            bool paired = !csym_bad[%(fidx)s];
            if (paired && ihel > flip[ihel]){
                continue;
            }
            igoodrep[%(fidx)s][nrep[%(fidx)s]] = ihel;
            if (paired){
                repwgt[%(fidx)s][nrep[%(fidx)s]] = 2;
            } else {
                repwgt[%(fidx)s][nrep[%(fidx)s]] = 1;
            }
            nrep[%(fidx)s]++;
        }
    }
    jhel[%(fidx)s] = 0;
    if (dedup_ok){
        sum_hel[%(fidx)s] = min(sum_hel[%(fidx)s], nrep[%(fidx)s]);
    } else {
        sum_hel[%(fidx)s] = min(sum_hel[%(fidx)s], ngood[%(fidx)s]);
    }
} else {
    // Helicity sampling (uncrossed, verdict settled): draw sum_hel of the nrep
    // C-parity representatives, each carrying the weight of its skipped
    // partner (repwgt, 1 when the pairing was refused).
    for(int j=0; j < sum_hel[%(fidx)s]; j++){
        jhel[%(fidx)s]++;
        if (jhel[%(fidx)s] >= nrep[%(fidx)s]){
            jhel[%(fidx)s]=0;
        }
        double hwgt = double(nrep[%(fidx)s])/double(sum_hel[%(fidx)s]);
        int ihel = igoodrep[%(fidx)s][jhel[%(fidx)s]];
        double cwgt = double(repwgt[%(fidx)s][jhel[%(fidx)s]]);
        calculate_wavefunctions(perm, helicities[ihel], flavor%(cross_cw_args)s);
        %(get_matrix_t_lines)s
        matrix_element += t*hwgt*cwgt;
    }
}

%(cross_return)s
