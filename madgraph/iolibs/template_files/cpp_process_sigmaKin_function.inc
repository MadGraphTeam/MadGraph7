// Local variables and constants
std::complex<double> **wfs;
// Helicities for the process
%(helicity_matrix)s
// Denominators: spins, colors and identical particles
const int denominator = %(den_factors)s;
// Flavor lookup table
%(flavor_table)s

const int* flavor = &flavor_table[mirror][flavor_id][0];
int multiplier = flavor_multipliers[flavor_id];

ntry[mirror]++;

// Define permutation
int perm[nexternal];
for(int i = 0; i < nexternal; i++){
    perm[i]=i;
}
if (mirror) {
    perm[0] = 1;
    perm[1] = 0;
}

double matrix_element = 0.;
int offset = ncomb * mirror;

if (sum_hel[mirror] == 0 || ntry[mirror] < 10){
    // Calculate the matrix element for all helicities
    for(int ihel = 0; ihel < ncomb; ihel ++){
        if (goodhel[offset + ihel] || ntry[mirror] < 2){
            calculate_wavefunctions(perm, helicities[ihel], flavor);
            %(get_matrix_t_lines)s
            matrix_element += t;
            // Store which helicities give non-zero result
            if (t != 0. && !goodhel[offset + ihel]){
                goodhel[offset + ihel]=true;
                ngood[mirror] ++;
                igood[offset + ngood[mirror]] = ihel;
            }
        }
    }
    jhel[mirror] = 0;
    sum_hel[mirror]=min(sum_hel[mirror], ngood[mirror]);
} else {
    // Only use the "good" helicities
    for(int j=0; j < sum_hel[mirror]; j++){
        jhel[mirror]++;
        if (jhel[mirror] >= ngood[mirror]) jhel[mirror]=0;
        double hwgt = double(ngood[mirror])/double(sum_hel[mirror]);
        int ihel = igood[offset + jhel[mirror]];
        calculate_wavefunctions(perm, helicities[ihel], flavor);
        %(get_matrix_t_lines)s
        matrix_element += t*hwgt;
    }
}

return multiplier * matrix_element / denominator;
