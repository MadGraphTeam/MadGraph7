#include "CPPProcess.h"
#include "api.h"
#include <vector>

extern "C" {

const SubProcessInfo* subprocess_info() {
    static SubProcessInfo info = {
        /* on_gpu          = */ false,
        /* particle_count  = */ CPPProcess::nexternal,
        /* diagram_count   = */ CPPProcess::ndiagrams,
        /* helicity_count  = */ CPPProcess::ncomb
    };
    return &info;
}

void* init_subprocess(const char* param_card_path) {
    CPPProcess* process = new CPPProcess(param_card_path);
    std::vector<double*>& momenta = process->getMomenta();
    for (int i = 0; i < CPPProcess::nexternal; ++i) momenta.push_back(new double[4]());
    return process;
}

void compute_matrix_element(
    void* subprocess,
    size_t count,
    size_t stride,
    const double* momenta_in,
    const int* flavor_in,
    double* m2_out
) {
    CPPProcess* process = static_cast<CPPProcess*>(subprocess);

    std::vector<double*>& process_momenta = process->getMomenta();
    for (size_t i_batch = 0; i_batch < count; ++i_batch) {
        for (size_t i_part = 0; i_part < CPPProcess::nexternal; ++i_part) {
            for(size_t i_mom = 0; i_mom < 4; ++i_mom) {
                process_momenta[i_part][i_mom] =
                    momenta_in[stride * (CPPProcess::nexternal * i_mom + i_part) + i_batch];
            }
        }
        m2_out[i_batch] = process->sigmaKin(flavor_in[i_batch]);
    }
}

void compute_matrix_element_multichannel(
    void* subprocess,
    size_t count,
    size_t stride,
    const double* momenta_in,
    const double* alpha_s_in,
    const double* random_in,
    const int* flavor_in,
    double* m2_out,
    double* amp2_out,
    int* diagram_out,
    int* color_out,
    int* helicity_out
) {
    CPPProcess* process = static_cast<CPPProcess*>(subprocess);

    std::vector<double*>& process_momenta = process->getMomenta();
    for (size_t i_batch = 0; i_batch < count; ++i_batch) {
        for (size_t i_part = 0; i_part < CPPProcess::nexternal; ++i_part) {
            for(size_t i_mom = 0; i_mom < 4; ++i_mom) {
                process_momenta[i_part][i_mom] =
                    momenta_in[stride * (CPPProcess::nexternal * i_mom + i_part) + i_batch];
            }
        }
        process->getParameters().aS = alpha_s_in[i_batch];
        m2_out[i_batch] = process->sigmaKin(flavor_in[i_batch]);
        color_out[i_batch] = 0;
        diagram_out[i_batch] = 0;
        helicity_out[i_batch] = 0;
        double chan_total = 0.;
        const double* amp2 = process->getAmp2();
        for(size_t i_amp = 0; i_amp < CPPProcess::ndiagrams; ++i_amp) {
            double amp2_item = amp2[i_amp];
            amp2_out[i_amp * stride + i_batch] = amp2_item;
            chan_total += amp2_item;
        }
        for(size_t i_amp = 0; i_amp < CPPProcess::ndiagrams; ++i_amp) {
            amp2_out[i_amp * stride + i_batch] /= chan_total;
        }
    }
}

void free_subprocess(void* subprocess) {
    CPPProcess* process = static_cast<CPPProcess*>(subprocess);
    std::vector<double*>& momenta = process->getMomenta();
    for (int i = 0; i < CPPProcess::nexternal; ++i) delete[] momenta[i];
    delete process;
}

}
