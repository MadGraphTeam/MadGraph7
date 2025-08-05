// Local variables and constants
std::complex<double> **wfs;
// Helicities for the process
%(helicity_matrix)s
// Denominators: spins, colors and identical particles
const int denominator = %(den_factors)s;
// Flavor lookup table
%(flavor_table)s

const int* flavor = &flavor_table[flavor_id][0];
int multiplier = flavor_multipliers[flavor_id];

ntry++;

// Define permutation
int perm[nexternal];
for(int i = 0; i < nexternal; i++){
    perm[i]=i;
}

double matrix_element = 0.;

if (sum_hel == 0 || ntry < 10){
    // Calculate the matrix element for all helicities
    for(int ihel = 0; ihel < ncomb; ihel ++){
        if (goodhel[ihel] || ntry < 2){
            calculate_wavefunctions(perm, helicities[ihel], flavor);
            %(get_matrix_t_lines)s
            matrix_element += t;
            // Store which helicities give non-zero result
            if (t != 0. && !goodhel[ihel]){
                goodhel[ihel]=true;
                ngood ++;
                igood[ngood] = ihel;
            }
        }
    }
    jhel = 0;
    sum_hel=min(sum_hel, ngood);
} else {
    // Only use the "good" helicities
    for(int j=0; j < sum_hel; j++){
        jhel++;
        if (jhel >= ngood) jhel=0;
        double hwgt = double(ngood)/double(sum_hel);
        int ihel = igood[jhel];
        calculate_wavefunctions(perm, helicities[ihel], flavor);
        %(get_matrix_t_lines)s
        matrix_element += t*hwgt;
    }
}

return multiplier * matrix_element / (denominator * broken_sym(flavor));
