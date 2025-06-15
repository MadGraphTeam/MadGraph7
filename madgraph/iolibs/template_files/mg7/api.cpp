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
    uint64_t count,
    uint64_t stride,
    const double* momenta_in,
    const int64_t* flavor_in,
    const int64_t* mirror_in,
    double* m2_out
) {
    CPPProcess* process = reinterpret_cast<CPPProcess*>(subprocess);

    std::vector<double*>& process_momenta = process->getMomenta();
    for (uint64_t i_batch = 0; i_batch < count; ++i_batch) {
        for (uint64_t i_part = 0; i_part < CPPProcess::npar; ++i_part) {
            for(uint64_t i_mom = 0; i_mom < 4; ++i_mom) {
                process_momenta[i_part][i_mom] =
                    momenta_in[stride * (CPPProcess::npar * i_mom + i_part) + i_batch];
            }
        }
        m2_out[i_batch] = process->sigmaKin(flavor_in[i_batch], mirror_in[i_batch]);
    }
}

void compute_matrix_element_multichannel(
    void* subprocess,
    uint64_t count,
    uint64_t stride,
    uint64_t channel_count,
    const double* momenta_in,
    const double* alpha_s_in,
    const double* random_in,
    const int64_t* flavor_in,
    const int64_t* mirror_in,
    const int64_t* amp2_remap_in,
    double* m2_out,
    double* channel_weights_out,
    int64_t* color_out,
    int64_t* diagram_out
) {
    CPPProcess* process = reinterpret_cast<CPPProcess*>(subprocess);

    std::vector<double*>& process_momenta = process->getMomenta();
    for (uint64_t i_batch = 0; i_batch < count; ++i_batch) {
        for (uint64_t i_part = 0; i_part < CPPProcess::nexternal; ++i_part) {
            for(uint64_t i_mom = 0; i_mom < 4; ++i_mom) {
                process_momenta[i_part][i_mom] =
                    momenta_in[stride * (CPPProcess::nexternal * i_mom + i_part) + i_batch];
            }
        }
        process->getParameters().aS = alpha_s_in[i_batch];
        m2_out[i_batch] = process->sigmaKin(flavor_in[i_batch], mirror_in[i_batch]);
        color_out[i_batch] = 0;
        diagram_out[i_batch] = 0;
        for(uint64_t i_chan = 0; i_chan < channel_count; ++i_chan) {
            channel_weights_out[i_chan * stride + i_batch] = 0;
        }
        double chan_total = 0.;
        const double* amp2 = process->getAmp2();
        for(uint64_t i_amp = 0; i_amp < CPPProcess::ndiagrams; ++i_amp) {
            double amp2_item = amp2[i_amp];
            int64_t target_chan = amp2_remap_in[i_amp];
            if (target_chan == -1) continue;
            channel_weights_out[target_chan * stride + i_batch] += amp2_item;
            chan_total += amp2_item;
        }
        for(uint64_t i_chan = 0; i_chan < channel_count; ++i_chan) {
            channel_weights_out[i_chan * stride + i_batch] /= chan_total;
        }
    }
}

void free_subprocess(void* subprocess) {
    CPPProcess* process = reinterpret_cast<CPPProcess*>(subprocess);
    std::vector<double*>& momenta = process->getMomenta();
    for (int i = 0; i < CPPProcess::nexternal; ++i) delete[] momenta[i];
    delete process;
}

}
