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

if (sum_hel[%(fidx)s] == 0 || ntry[%(fidx)s] < 10){
    // Scan phase: calculate the matrix element for all helicities (full sum).
    double tstore[ncomb] = {};
    for(int ihel = 0; ihel < ncomb; ihel ++){
        %(cross_ghidx_setup)sif (%(cross_goodhel_gate)s){
            calculate_wavefunctions(perm, helicities[ihel], flavor%(cross_cw_args)s);
            %(get_matrix_t_lines)s
            matrix_element += t;
            tstore[ihel] = t;
            // Store which helicities give non-zero result
            %(cross_goodhel_train)s
        }
    }
    if (dedup_ok){
        // Drop the C-parity pairing of any row whose flipped partner gave a
        // different |M|^2 (parity/C violation). One mismatch at any scan point
        // permanently invalidates the pair (robust, like the zero-filter).
        for(int ihel = 0; ihel < ncomb; ihel++){
            if (flip[ihel] > ihel){
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
                if (diff > 1e-6 * (aa + bb)){
                    csym_bad[%(fidx)s][ihel] = true;
                    csym_bad[%(fidx)s][flip[ihel]] = true;
                }
            }
        }
        // Reduce the good helicities to the lower-index representative of every
        // surviving C-parity pair, carrying a doubled weight; the skipped
        // higher-index partner has an identical |M|^2.
        nrep[%(fidx)s] = 0;
        for(int g = 1; g <= ngood[%(fidx)s]; g++){
            int ihel = igood[%(fidx)s][g];
            bool paired = !csym_bad[%(fidx)s][ihel] && flip[ihel] != ihel;
            if (paired && ihel > flip[ihel]){
                continue;
            }
            nrep[%(fidx)s]++;
            igoodrep[%(fidx)s][nrep[%(fidx)s]] = ihel;
            if (paired){
                repwgt[%(fidx)s][nrep[%(fidx)s]] = 2;
            } else {
                repwgt[%(fidx)s][nrep[%(fidx)s]] = 1;
            }
        }
    }
    jhel[%(fidx)s] = 0;
    if (dedup_ok){
        sum_hel[%(fidx)s] = min(sum_hel[%(fidx)s], nrep[%(fidx)s]);
    } else {
        sum_hel[%(fidx)s] = min(sum_hel[%(fidx)s], ngood[%(fidx)s]);
    }
} else {
    // Only use the "good" helicities (C-parity representatives when uncrossed).
    int nsel = ngood[%(fidx)s];
    if (dedup_ok){
        nsel = nrep[%(fidx)s];
    }
    for(int j=0; j < sum_hel[%(fidx)s]; j++){
        jhel[%(fidx)s]++;
        if (jhel[%(fidx)s] >= nsel){
            jhel[%(fidx)s]=0;
        }
        double hwgt = double(nsel)/double(sum_hel[%(fidx)s]);
        int ihel = igood[%(fidx)s][jhel[%(fidx)s]];
        double cwgt = 1.;
        if (dedup_ok){
            ihel = igoodrep[%(fidx)s][jhel[%(fidx)s]];
            cwgt = double(repwgt[%(fidx)s][jhel[%(fidx)s]]);
        }
        calculate_wavefunctions(perm, helicities[ihel], flavor%(cross_cw_args)s);
        %(get_matrix_t_lines)s
        matrix_element += t*hwgt*cwgt;
    }
}

%(cross_return)s
