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

if (sum_hel[%(fidx)s] == 0 || ntry[%(fidx)s] < 10){
    // Calculate the matrix element for all helicities
    for(int ihel = 0; ihel < ncomb; ihel ++){
        %(cross_ghidx_setup)sif (%(cross_goodhel_gate)s){
            calculate_wavefunctions(perm, helicities[ihel], flavor%(cross_cw_args)s);
            %(get_matrix_t_lines)s
            matrix_element += t;
            // Store which helicities give non-zero result
            %(cross_goodhel_train)s
        }
    }
    jhel[%(fidx)s] = 0;
    sum_hel[%(fidx)s]=min(sum_hel[%(fidx)s], ngood[%(fidx)s]);
} else {
    // Only use the "good" helicities
    for(int j=0; j < sum_hel[%(fidx)s]; j++){
        jhel[%(fidx)s]++;
        if (jhel[%(fidx)s] >= ngood[%(fidx)s]) jhel[%(fidx)s]=0;
        double hwgt = double(ngood[%(fidx)s])/double(sum_hel[%(fidx)s]);
        int ihel = igood[%(fidx)s][jhel[%(fidx)s]];
        calculate_wavefunctions(perm, helicities[ihel], flavor%(cross_cw_args)s);
        %(get_matrix_t_lines)s
        matrix_element += t*hwgt;
    }
}

%(cross_return)s
