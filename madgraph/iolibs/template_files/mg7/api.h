#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Contains information on the matrix elements contained in this subprocess */
struct SubProcessInfo {
    /** boolean indicating whether this code runs on the GPU */
    uint8_t on_gpu;

    /** number of particles (incoming and outgoing) */
    uint64_t particle_count;

    /** number of diagrams */
    uint64_t diagram_count;

    /** number of amplitudes */
    uint64_t amplitude_count;

    /** number of helicity configurations */
    uint64_t helicity_count;
};

/**
 * Returns information about the subprocess
 * @return
 *     pointer to a SubProcessInfo object
 */
const SubProcessInfo* subprocess_info();

/** 
 * Initializes a subprocess object 
 *
 * @param matrix_element_index
 *     index of the matrix element, between 0 and matrix_element_count
 * @param param_card_path
 *     path to the parameter file
 * @return 
 *     pointer to an instance of the subprocess. Has to be cleaned up by
 *     the caller with `free_subprocess`.
 */
void* init_subprocess(const char* param_card_path);

/**
 * Computes the squared matrix elements for a batch of events.
 * All pointers (except for the first argument) can either point to CPU or GPU memory,
 * depending on the value of `on_gpu`.
 * 
 * @param subprocess
 *     pointer to the subprocess object
 * @param count
 *     size of the batch
 * @param stride
 *     step size between events in memory. Can be larger than count.
 * @param momenta_in
 *     pointer to a batch of four-momenta of the incoming and outgoing particles.
 *     The i-th component of the j-th particle of the k-th event is accessed as
 *     `momenta_in[stride * particle_count * i + stride * j + k]`
 *     with `i` between `0` and `3`, `j` between 0 and `particle_count - 1`,
 *     `k` between `0` and `count - 1`
 * @param flavor_in
 *     pointer to a batch of flavor configuration indices
 * @param mirror_in
 *     pointer to a batch of integers indicating whether to flip the initial state momenta (0 or 1)
 * @param m2_out
 *     pointer to the batch of the computed squared matrix elements
 */
void compute_matrix_element(
    void* subprocess,
    uint64_t count,
    uint64_t stride,
    const double* momenta_in,
    const int64_t* flavor_in,
    const int64_t* mirror_in,
    double* m2_out
);

/**
 * Computes the squared matrix elements and channel weights for a batch of events.
 * All pointers (except for the first argument) can either point to CPU or GPU memory,
 * depending on the value of `on_gpu`.
 * 
 * @param subprocess
 *     pointer to the subprocess object
 * @param count
 *     size of the batch
 * @param stride
 *     step size between events in memory. Can be larger than count.
 * @param channel_count
 *     number of channels
 * @param momenta_in
 *     pointer to a batch of four-momenta of the incoming and outgoing particles.
 *     The i-th component of the j-th particle of the k-th event is accessed as
 *     `momenta_in[stride * particle_count * i + stride * j + k]`
 *     with `i` between `0` and `3`, `j` between 0 and `particle_count - 1`,
 *     `k` between `0` and `count - 1`
 * @param alpha_s_in
 *     pointer to a batch of strong couplings
 * @param random_in
 *     pointer to a batch of random numbers from [0,1) to determine color and diagram
 *     The random number for color of the k-th event is accessed as `random_in[k]`,
 *     the random number for the diagram is accessed as `random_in[stride + k]`
 *     with `k` between `0` and `count - 1`
 * @param flavor_in
 *     pointer to a batch of flavor configuration indices
 * @param mirror_in
 *     pointer to a batch of integers indicating whether to flip the initial state momenta (0 or 1)
 * @param amp2_remap_in
 *     defines the mapping from squared amplitudes to channel weights
 * @param m2_out
 *     pointer to the batch of the computed squared matrix elements
 * @param channel_weights_out
 *     pointer to the batch of the computed channel weights.
 *     The i-th channel weight of the k-th event,
 *     with `i` between `0` and `channel_count - 1`
 *     and `k` between `0` and `count - 1`
 *     is accessed as `channel_weights_out[stride * i + k]`
 * @param m2_out
 *     pointer to the batch of the computed squared matrix elements
 * @param color_out
 *     pointer to the batch of picked color configurations
 * @param diagram_out
 *     pointer to the batch of picked diagrams
 */
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
);

/**
 * Frees subprocess object
 *
 * @param subprocess
 *     pointer to the subprocess object
 */
void free_subprocess(void* subprocess);

#ifdef __cplusplus
}
#endif
