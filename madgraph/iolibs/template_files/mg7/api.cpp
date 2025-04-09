#include "CPPProcess.h"
#include "api.h"
#include <vector>

extern "C" {

std::vector<uint64_t> diagram_counts(CPPProcess::nprocesses, CPPProcess::ndiagrams);
SubProcessInfo info = {
    /* matrix_element_count = */ CPPProcess::nprocesses,
    /* on_gpu               = */ false,
    /* particle_count       = */ CPPProcess::nexternal,
    /* diagram_counts       = */ diagram_counts.data()
};

const SubProcessInfo* subprocess_info() {
    return &info;
}

void* init_subprocess(uint64_t matrix_element_index, const char* param_card_path, double alpha_s) {
    CPPProcess* process = new CPPProcess();
    process->initProc(param_card_path);
    process->curr_process = matrix_element_index;
    process->pars->aS = alpha_s;
    return process;
}

void compute_matrix_element(
    void* subprocess,
    uint64_t count,
    uint64_t stride,
    const double* momenta_in,
    double* m2_out
) {
    CPPProcess* process = reinterpret_cast<CPPProcess*>(subprocess);

    for (uint64_t i_batch = 0; i_batch < count; ++i_batch) {
        for (uint64_t i_part = 0; i_part < CPPProcess::nexternal; ++i_part) {
            for(uint64_t i_mom = 0; i_mom < 4; ++i_mom) {
                process->p[i_part][i_mom] =
                    momenta_in[stride * (CPPProcess::nexternal * i_mom + i_part) + i_batch];
            }
        }
        process->sigmaKin();
        m2_out[i_batch] = process->matrix_element[process->curr_process];
    }
}

void compute_matrix_element_multichannel(
    void* subprocess,
    uint64_t count,
    uint64_t stride,
    uint64_t channel_count,
    const double* momenta_in,
    const int64_t* amp2_remap_in,
    double* m2_out,
    double* channel_weights_out
) {
    CPPProcess* process = reinterpret_cast<CPPProcess*>(subprocess);

    for (uint64_t i_batch = 0; i_batch < count; ++i_batch) {
        for (uint64_t i_part = 0; i_part < CPPProcess::nexternal; ++i_part) {
            for(uint64_t i_mom = 0; i_mom < 4; ++i_mom) {
                process->p[i_part][i_mom] =
                    momenta_in[stride * (CPPProcess::nexternal * i_mom + i_part) + i_batch];
            }
        }
        process->sigmaKin();
        m2_out[i_batch] = process->matrix_element[process->curr_process];
        for(uint64_t i_chan = 0; i_chan < channel_count; ++i_chan) {
            channel_weights_out[i_chan * stride + i_batch] = 0;
        }
        double chan_total = 0.;
        for(uint64_t i_amp = 0; i_amp < CPPProcess::ndiagrams; ++i_amp) {
            double amp2 = process->amp2[i_amp];
            int64_t target_chan = amp2_remap_in[i_amp];
            if (target_chan == -1) continue;
            channel_weights_out[target_chan * stride + i_batch] += amp2;
            chan_total += amp2;
        }
        for(uint64_t i_chan = 0; i_chan < channel_count; ++i_chan) {
            channel_weights_out[i_chan * stride + i_batch] /= chan_total;
        }
    }
}

void free_subprocess(void* subprocess) {
    CPPProcess* process = reinterpret_cast<CPPProcess*>(subprocess);
    delete process;
}

}
