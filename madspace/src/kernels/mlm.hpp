#pragma once

#include "definitions.hpp"

namespace madspace {
namespace kernels {

constexpr int N_EXT_MAX = 12;

KERNELSPEC void compute_scale(
    double* momentum1,
    double* momentum2,
    bool resonant,
    bool massive_in,
    bool massive_out1,
    bool massive_out2
) {
    /*
    initial state:
         cluster_scale=sqrt(djb_clus(pi))
         ! prefer clustering when outgoing is in the direction of incoming
         if(sign(1d0,pi(3)).ne.sign(1d0,pj(3)))
     $        cluster_scale=cluster_scale*one_plus_tiny

    final state:
    resonant:
      cluster_scale=sqrt(max(sumdot(pi,pj,1d0),0d0))

    2 : 1 0 1
      dj_clus = DJB_clus(p1)*(1d0+1d-6)

    3 : 1 1 0
      dj_clus = DJB_clus(p2)*(1d0+1d-6)

    1 : 0 0 0
    6 : 0 1 1
    7 : 1 1 1
           pt1 = p1(1)**2+p1(2)**2
           pt2 = p2(1)**2+p2(2)**2
           if (pt1.eq.0d0 .or. pt2.eq.0d0) then
              dj_clus=0d0
              return
           endif
           p1a = dsqrt(pt1+p1(3)**2)
           p2a = dsqrt(pt2+p2(3)**2)
           eta1 = 0.5d0*log((p1a+p1(3))/(p1a-p1(3)))
           eta2 = 0.5d0*log((p2a+p2(3))/(p2a-p2(3)))
           dj_clus = max(m1,m2)**2+min(pt1,pt2)*2d0*(cosh(eta1-eta2)-
     &          (p1(1)*p2(1)+p1(2)*p2(2))/dsqrt(pt1*pt2))/D**2
           if (dj_clus.lt.0d0 .or. dj_clus.ne.dj_clus)
     &                 dj_clus=0d0     ! prevent numerical inaccuracies

    4 : 0 1 0
     cluster_scale=sqrt(abs(dot(pj,(pi+pj))))/2d0

    5 : 0 0 1
     cluster_scale=sqrt(abs(dot(pi,(pi+pj))))/2d0

    8 : 1 0 0
    resonant:
     cluster_scale=sqrt(max(sumdot(pi,pj,1d0),0d0))
    */
}

template <typename T>
KERNELSPEC void mlm_clustering_hadronic(
    FIn<T, 2> momenta,
    FIn<T, 0> random,
    IIn<T, 1> state_machine,
    FIn<T, 1> masses,
    FIn<T, 1> widths,
    FOut<T, 0> ren_scale,
    FOut<T, 0> fact_scale1,
    FOut<T, 0> fact_scale2,
    FOut<T, 1> outgoing_scales,
    IOut<T, 0> diagram_index
) {
    static_assert(std::is_same_v<IVal<T>, int>);
    int state = 0, cluster_count = 0;
    int cluster_max = momenta.size() - 3;
    double momenta_tmp[N_EXT_MAX][4];
    int cluster_history[N_EXT_MAX - 3];
    int cluster_scales[N_EXT_MAX - 3];

    for (int i = 0; i < momenta.size(); ++i) {
        for (int j = 0; j < 4; ++j) {
            momenta_tmp[i][j] = momenta[i][j];
        }
    }

    // Iterate over the possible clustering states. This loop iterates over the possible
    // clusterings at a given points as well as the full sequence of clusterings.
    // Compared to two nested loops, this means that no thread has to wait on the GPU
    int win_next_state = -1, win_data = 0;
    double win_scale = 1e308;
    while (cluster_count < cluster_max) {
        int data = state_machine[state];
        int next_state = state_machine[state + 1];
        int particle1 = data & 0xFF;
        int particle2 = (data >> 8) & 0xFF;
        int mass_index = (data >> 16) & 0xFF;
        bool massive_in = (data >> 24) & 1;
        bool massive_out1 = (data >> 25) & 1;
        bool massive_out2 = (data >> 26) & 1;
        bool is_last = (data >> 28) & 1;

        double scale = compute_scale_hadronic(
            momenta_tmp[particle1],
            momenta_tmp[particle2],
            massive_in,
            massive_out1,
            massive_out2
        );

        if (scale < win_scale) {
            win_next_state = next_state;
            win_scale = scale;
            win_data = data;
        }
        if (is_last) {
            state = win_next_state;
            cluster_history[cluster_count] = win_data;
            cluster_scales[cluster_count] = win_scale;
            ++cluster_count;
        } else {
            ++state;
        }
    }

    // now determine the scales:
    //   renormalization scale: geometric mean of clustering scales, replacing those
    //     with is_qcd==false with the maximal scale
    //   factorization scale: minimal scale with is_qcd==true
    double fac_scale = 1e308, max_scale = 0;
    for (int i = 0; i < cluster_max; ++i) {
        double scale = cluster_scales[i];
        bool is_qcd = (cluster_history[i] >> 27) & 1;
        if (is_qcd && scale < fac_scale) {
            fac_scale = scale;
        }
        if (scale > max_scale) {
            max_scale = scale;
        }
    }

    double ren_scale = 1.;
    for (int i = 0; i < cluster_max; ++i) {
        double scale = cluster_scales[i];
        bool is_qcd = (cluster_history[i] >> 27) & 1;
        ren_scale *= is_qcd ? scale : max_scale;
    }
    ren_scale = pow(ren_scale, 1. / cluster_max);
    if (fac_scale > ren_scale) {
        fac_scale = ren_scale;
    }

    // if multiple diagrams exist, pick one randomly
    int diag_count = state_machine[state];
    int rand_index = static_cast<int>(static_cast<double>(random) * diag_count);
    if (rand_index >= diag_count) {
        rand_index = diag_count - 1;
    }
    diagram_index = state_machine[state + rand_index + 1];
}

} // namespace kernels
} // namespace madspace
