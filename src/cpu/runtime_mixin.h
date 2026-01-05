// This file was automatically generated based on instruction_set.yaml
// Do not modify its content directly

case 0:
    op_stack(instr, locals, device);
    break;
case 1:
    op_unstack(instr, locals, device);
    break;
case 2:
    op_unstack_sizes(instr, locals, device);
    break;
case 3:
    op_pop(instr, locals, device);
    break;
case 4:
    op_batch_cat(instr, locals, device);
    break;
case 5:
    op_batch_split(instr, locals, device);
    break;
case 6:
    op_cat(instr, locals, device);
    break;
case 7:
    op_batch_size(instr, locals, device);
    break;
case 8:
    op_offset_indices(instr, locals, device);
    break;
case 9:
    op_full(instr, locals, device);
    break;
case 10:
    op_squeeze(instr, locals, device);
    break;
case 11:
    op_unsqueeze(instr, locals, device);
    break;
case 12:
    batch_foreach<tensor_foreach_dynamic<kernel_add<CpuTypes>, kernel_add<SimdTypes>, 2, 1, DeviceType>, 2, 1>(instr, locals, device);
    break;
case 13:
    batch_foreach<tensor_foreach_dynamic<kernel_add_int<CpuTypes>, kernel_add_int<SimdTypes>, 2, 1, DeviceType>, 2, 1>(instr, locals, device);
    break;
case 14:
    batch_foreach<tensor_foreach_dynamic<kernel_sub<CpuTypes>, kernel_sub<SimdTypes>, 2, 1, DeviceType>, 2, 1>(instr, locals, device);
    break;
case 15:
    batch_foreach<tensor_foreach_dynamic<kernel_mul<CpuTypes>, kernel_mul<SimdTypes>, 2, 1, DeviceType>, 2, 1>(instr, locals, device);
    break;
case 16:
    batch_foreach<tensor_foreach_dynamic<kernel_reduce_sum<CpuTypes>, kernel_reduce_sum<SimdTypes>, 1, 1, DeviceType>, 1, 1>(instr, locals, device);
    break;
case 17:
    batch_foreach<tensor_foreach_dynamic<kernel_reduce_sum_vector<CpuTypes>, kernel_reduce_sum_vector<SimdTypes>, 1, 1, DeviceType>, 1, 1>(instr, locals, device);
    break;
case 18:
    batch_foreach<tensor_foreach<kernel_reduce_product<CpuTypes>, kernel_reduce_product<SimdTypes>, 1, 1, 1, DeviceType>, 1, 1>(instr, locals, device);
    break;
case 19:
    batch_foreach<tensor_foreach<kernel_sqrt<CpuTypes>, kernel_sqrt<SimdTypes>, 1, 1, 1, DeviceType>, 1, 1>(instr, locals, device);
    break;
case 20:
    batch_foreach<tensor_foreach<kernel_square<CpuTypes>, kernel_square<SimdTypes>, 1, 1, 1, DeviceType>, 1, 1>(instr, locals, device);
    break;
case 21:
    batch_foreach<tensor_foreach_dynamic<kernel_min<CpuTypes>, kernel_min<SimdTypes>, 2, 1, DeviceType>, 2, 1>(instr, locals, device);
    break;
case 22:
    batch_foreach<tensor_foreach_dynamic<kernel_max<CpuTypes>, kernel_max<SimdTypes>, 2, 1, DeviceType>, 2, 1>(instr, locals, device);
    break;
case 23:
    batch_foreach<tensor_foreach_dynamic<kernel_obs_e<CpuTypes>, kernel_obs_e<SimdTypes>, 1, 1, DeviceType>, 1, 1>(instr, locals, device);
    break;
case 24:
    batch_foreach<tensor_foreach_dynamic<kernel_obs_px<CpuTypes>, kernel_obs_px<SimdTypes>, 1, 1, DeviceType>, 1, 1>(instr, locals, device);
    break;
case 25:
    batch_foreach<tensor_foreach_dynamic<kernel_obs_py<CpuTypes>, kernel_obs_py<SimdTypes>, 1, 1, DeviceType>, 1, 1>(instr, locals, device);
    break;
case 26:
    batch_foreach<tensor_foreach_dynamic<kernel_obs_pz<CpuTypes>, kernel_obs_pz<SimdTypes>, 1, 1, DeviceType>, 1, 1>(instr, locals, device);
    break;
case 27:
    batch_foreach<tensor_foreach_dynamic<kernel_obs_mass<CpuTypes>, kernel_obs_mass<SimdTypes>, 1, 1, DeviceType>, 1, 1>(instr, locals, device);
    break;
case 28:
    batch_foreach<tensor_foreach_dynamic<kernel_obs_pt<CpuTypes>, kernel_obs_pt<SimdTypes>, 1, 1, DeviceType>, 1, 1>(instr, locals, device);
    break;
case 29:
    batch_foreach<tensor_foreach_dynamic<kernel_obs_p_mag<CpuTypes>, kernel_obs_p_mag<SimdTypes>, 1, 1, DeviceType>, 1, 1>(instr, locals, device);
    break;
case 30:
    batch_foreach<tensor_foreach_dynamic<kernel_obs_phi<CpuTypes>, kernel_obs_phi<SimdTypes>, 1, 1, DeviceType>, 1, 1>(instr, locals, device);
    break;
case 31:
    batch_foreach<tensor_foreach_dynamic<kernel_obs_theta<CpuTypes>, kernel_obs_theta<SimdTypes>, 1, 1, DeviceType>, 1, 1>(instr, locals, device);
    break;
case 32:
    batch_foreach<tensor_foreach_dynamic<kernel_obs_y<CpuTypes>, kernel_obs_y<SimdTypes>, 1, 1, DeviceType>, 1, 1>(instr, locals, device);
    break;
case 33:
    batch_foreach<tensor_foreach_dynamic<kernel_obs_y_abs<CpuTypes>, kernel_obs_y_abs<SimdTypes>, 1, 1, DeviceType>, 1, 1>(instr, locals, device);
    break;
case 34:
    batch_foreach<tensor_foreach_dynamic<kernel_obs_eta<CpuTypes>, kernel_obs_eta<SimdTypes>, 1, 1, DeviceType>, 1, 1>(instr, locals, device);
    break;
case 35:
    batch_foreach<tensor_foreach_dynamic<kernel_obs_eta_abs<CpuTypes>, kernel_obs_eta_abs<SimdTypes>, 1, 1, DeviceType>, 1, 1>(instr, locals, device);
    break;
case 36:
    batch_foreach<tensor_foreach_dynamic<kernel_obs_delta_eta<CpuTypes>, kernel_obs_delta_eta<SimdTypes>, 2, 1, DeviceType>, 2, 1>(instr, locals, device);
    break;
case 37:
    batch_foreach<tensor_foreach_dynamic<kernel_obs_delta_phi<CpuTypes>, kernel_obs_delta_phi<SimdTypes>, 2, 1, DeviceType>, 2, 1>(instr, locals, device);
    break;
case 38:
    batch_foreach<tensor_foreach_dynamic<kernel_obs_delta_r<CpuTypes>, kernel_obs_delta_r<SimdTypes>, 2, 1, DeviceType>, 2, 1>(instr, locals, device);
    break;
case 39:
    batch_foreach<tensor_foreach<kernel_boost_beam<CpuTypes>, kernel_boost_beam<SimdTypes>, 3, 1, 1, DeviceType>, 3, 1>(instr, locals, device);
    break;
case 40:
    batch_foreach<tensor_foreach<kernel_boost_beam_inverse<CpuTypes>, kernel_boost_beam_inverse<SimdTypes>, 3, 1, 1, DeviceType>, 3, 1>(instr, locals, device);
    break;
case 41:
    batch_foreach<tensor_foreach<kernel_com_p_in<CpuTypes>, kernel_com_p_in<SimdTypes>, 1, 2, 1, DeviceType>, 1, 2>(instr, locals, device);
    break;
case 42:
    batch_foreach<tensor_foreach<kernel_r_to_x1x2<CpuTypes>, kernel_r_to_x1x2<SimdTypes>, 3, 3, 1, DeviceType>, 3, 3>(instr, locals, device);
    break;
case 43:
    batch_foreach<tensor_foreach<kernel_x1x2_to_r<CpuTypes>, kernel_x1x2_to_r<SimdTypes>, 3, 2, 1, DeviceType>, 3, 2>(instr, locals, device);
    break;
case 44:
    batch_foreach<tensor_foreach<kernel_diff_cross_section<CpuTypes>, kernel_diff_cross_section<SimdTypes>, 6, 1, 1, DeviceType>, 6, 1>(instr, locals, device);
    break;
case 45:
    batch_foreach<tensor_foreach<kernel_two_body_decay_com<CpuTypes>, kernel_two_body_decay_com<SimdTypes>, 5, 3, 1, DeviceType>, 5, 3>(instr, locals, device);
    break;
case 46:
    batch_foreach<tensor_foreach<kernel_two_body_decay<CpuTypes>, kernel_two_body_decay<SimdTypes>, 6, 3, 1, DeviceType>, 6, 3>(instr, locals, device);
    break;
case 47:
    batch_foreach<tensor_foreach<kernel_two_to_two_particle_scattering_com<CpuTypes>, kernel_two_to_two_particle_scattering_com<SimdTypes>, 6, 3, 1, DeviceType>, 6, 3>(instr, locals, device);
    break;
case 48:
    batch_foreach<tensor_foreach<kernel_two_to_two_particle_scattering<CpuTypes>, kernel_two_to_two_particle_scattering<SimdTypes>, 6, 3, 1, DeviceType>, 6, 3>(instr, locals, device);
    break;
case 49:
    batch_foreach<tensor_foreach<kernel_two_to_three_particle_scattering<CpuTypes>, kernel_two_to_three_particle_scattering<SimdTypes>, 8, 3, 1, DeviceType>, 8, 3>(instr, locals, device);
    break;
case 50:
    batch_foreach<tensor_foreach<kernel_three_body_decay_com<CpuTypes>, kernel_three_body_decay_com<SimdTypes>, 9, 4, 1, DeviceType>, 9, 4>(instr, locals, device);
    break;
case 51:
    batch_foreach<tensor_foreach<kernel_three_body_decay<CpuTypes>, kernel_three_body_decay<SimdTypes>, 10, 4, 1, DeviceType>, 10, 4>(instr, locals, device);
    break;
case 52:
    batch_foreach<tensor_foreach<kernel_t_inv_min_max<CpuTypes>, kernel_t_inv_min_max<SimdTypes>, 4, 2, 1, DeviceType>, 4, 2>(instr, locals, device);
    break;
case 53:
    batch_foreach<tensor_foreach<kernel_s_inv_min_max<CpuTypes>, kernel_s_inv_min_max<SimdTypes>, 6, 2, 1, DeviceType>, 6, 2>(instr, locals, device);
    break;
case 54:
    batch_foreach<tensor_foreach<kernel_invariants_from_momenta<CpuTypes>, kernel_invariants_from_momenta<SimdTypes>, 2, 1, 1, DeviceType>, 2, 1>(instr, locals, device);
    break;
case 55:
    batch_foreach<tensor_foreach<kernel_sde2_channel_weights<CpuTypes>, kernel_sde2_channel_weights<SimdTypes>, 4, 1, 1, DeviceType>, 4, 1>(instr, locals, device);
    break;
case 56:
    batch_foreach<tensor_foreach<kernel_subchannel_weights<CpuTypes>, kernel_subchannel_weights<SimdTypes>, 6, 1, 1, DeviceType>, 6, 1>(instr, locals, device);
    break;
case 57:
    batch_foreach<tensor_foreach<kernel_apply_subchannel_weights<CpuTypes>, kernel_apply_subchannel_weights<SimdTypes>, 4, 1, 1, DeviceType>, 4, 1>(instr, locals, device);
    break;
case 58:
    batch_foreach<tensor_foreach<kernel_pt_eta_phi_x<CpuTypes>, kernel_pt_eta_phi_x<SimdTypes>, 3, 1, 1, DeviceType>, 3, 1>(instr, locals, device);
    break;
case 59:
    batch_foreach<tensor_foreach<kernel_mirror_momenta<CpuTypes>, kernel_mirror_momenta<SimdTypes>, 2, 1, 1, DeviceType>, 2, 1>(instr, locals, device);
    break;
case 60:
    batch_foreach<tensor_foreach<kernel_momenta_to_x1x2<CpuTypes>, kernel_momenta_to_x1x2<SimdTypes>, 2, 2, 1, DeviceType>, 2, 2>(instr, locals, device);
    break;
case 61:
    batch_foreach<tensor_foreach<kernel_uniform_invariant<CpuTypes>, kernel_uniform_invariant<SimdTypes>, 3, 2, 1, DeviceType>, 3, 2>(instr, locals, device);
    break;
case 62:
    batch_foreach<tensor_foreach<kernel_uniform_invariant_inverse<CpuTypes>, kernel_uniform_invariant_inverse<SimdTypes>, 3, 2, 1, DeviceType>, 3, 2>(instr, locals, device);
    break;
case 63:
    batch_foreach<tensor_foreach<kernel_breit_wigner_invariant<CpuTypes>, kernel_breit_wigner_invariant<SimdTypes>, 5, 2, 1, DeviceType>, 5, 2>(instr, locals, device);
    break;
case 64:
    batch_foreach<tensor_foreach<kernel_breit_wigner_invariant_inverse<CpuTypes>, kernel_breit_wigner_invariant_inverse<SimdTypes>, 5, 2, 1, DeviceType>, 5, 2>(instr, locals, device);
    break;
case 65:
    batch_foreach<tensor_foreach<kernel_stable_invariant<CpuTypes>, kernel_stable_invariant<SimdTypes>, 4, 2, 1, DeviceType>, 4, 2>(instr, locals, device);
    break;
case 66:
    batch_foreach<tensor_foreach<kernel_stable_invariant_inverse<CpuTypes>, kernel_stable_invariant_inverse<SimdTypes>, 4, 2, 1, DeviceType>, 4, 2>(instr, locals, device);
    break;
case 67:
    batch_foreach<tensor_foreach<kernel_stable_invariant_nu<CpuTypes>, kernel_stable_invariant_nu<SimdTypes>, 5, 2, 1, DeviceType>, 5, 2>(instr, locals, device);
    break;
case 68:
    batch_foreach<tensor_foreach<kernel_stable_invariant_nu_inverse<CpuTypes>, kernel_stable_invariant_nu_inverse<SimdTypes>, 5, 2, 1, DeviceType>, 5, 2>(instr, locals, device);
    break;
case 69:
    batch_foreach<tensor_foreach<kernel_fast_rambo_massless<CpuTypes>, kernel_fast_rambo_massless<SimdTypes>, 3, 2, 1, DeviceType>, 3, 2>(instr, locals, device);
    break;
case 70:
    batch_foreach<tensor_foreach<kernel_fast_rambo_massless_com<CpuTypes>, kernel_fast_rambo_massless_com<SimdTypes>, 2, 2, 1, DeviceType>, 2, 2>(instr, locals, device);
    break;
case 71:
    batch_foreach<tensor_foreach<kernel_fast_rambo_massive<CpuTypes>, kernel_fast_rambo_massive<SimdTypes>, 4, 2, 1, DeviceType>, 4, 2>(instr, locals, device);
    break;
case 72:
    batch_foreach<tensor_foreach<kernel_fast_rambo_massive_com<CpuTypes>, kernel_fast_rambo_massive_com<SimdTypes>, 3, 2, 1, DeviceType>, 3, 2>(instr, locals, device);
    break;
case 73:
    batch_foreach<tensor_foreach<kernel_cut_unphysical<CpuTypes>, kernel_cut_unphysical<SimdTypes>, 4, 1, 1, DeviceType>, 4, 1>(instr, locals, device);
    break;
case 74:
    batch_foreach<tensor_foreach<kernel_cut_one<CpuTypes>, kernel_cut_one<SimdTypes>, 3, 1, 1, DeviceType>, 3, 1>(instr, locals, device);
    break;
case 75:
    batch_foreach<tensor_foreach<kernel_cut_all<CpuTypes>, kernel_cut_all<SimdTypes>, 3, 1, 1, DeviceType>, 3, 1>(instr, locals, device);
    break;
case 76:
    batch_foreach<tensor_foreach<kernel_cut_any<CpuTypes>, kernel_cut_any<SimdTypes>, 3, 1, 1, DeviceType>, 3, 1>(instr, locals, device);
    break;
case 77:
    batch_foreach<tensor_foreach<kernel_scale_transverse_energy<CpuTypes>, kernel_scale_transverse_energy<SimdTypes>, 1, 1, 1, DeviceType>, 1, 1>(instr, locals, device);
    break;
case 78:
    batch_foreach<tensor_foreach<kernel_scale_transverse_mass<CpuTypes>, kernel_scale_transverse_mass<SimdTypes>, 1, 1, 1, DeviceType>, 1, 1>(instr, locals, device);
    break;
case 79:
    batch_foreach<tensor_foreach<kernel_scale_half_transverse_mass<CpuTypes>, kernel_scale_half_transverse_mass<SimdTypes>, 1, 1, 1, DeviceType>, 1, 1>(instr, locals, device);
    break;
case 80:
    batch_foreach<tensor_foreach<kernel_scale_partonic_energy<CpuTypes>, kernel_scale_partonic_energy<SimdTypes>, 1, 1, 1, DeviceType>, 1, 1>(instr, locals, device);
    break;
case 81:
    batch_foreach<tensor_foreach<kernel_chili_forward<CpuTypes>, kernel_chili_forward<SimdTypes>, 5, 4, 1, DeviceType>, 5, 4>(instr, locals, device);
    break;
case 82:
    op_matrix_element(instr, locals, device);
    break;
case 83:
    batch_foreach<tensor_foreach<kernel_collect_channel_weights<CpuTypes>, kernel_collect_channel_weights<SimdTypes>, 3, 1, 1, DeviceType>, 3, 1>(instr, locals, device);
    break;
case 84:
    batch_foreach<tensor_foreach<kernel_interpolate_pdf<CpuTypes>, kernel_interpolate_pdf<CpuTypes>, 6, 1, 1, DeviceType>, 6, 1>(instr, locals, device);
    break;
case 85:
    batch_foreach<tensor_foreach<kernel_interpolate_alpha_s<CpuTypes>, kernel_interpolate_alpha_s<CpuTypes>, 3, 1, 1, DeviceType>, 3, 1>(instr, locals, device);
    break;
case 86:
    op_matmul(instr, locals, device);
    break;
case 87:
    batch_foreach<tensor_foreach_dynamic<kernel_relu<CpuTypes>, kernel_relu<SimdTypes>, 1, 1, DeviceType>, 1, 1>(instr, locals, device);
    break;
case 88:
    batch_foreach<tensor_foreach_dynamic<kernel_leaky_relu<CpuTypes>, kernel_leaky_relu<SimdTypes>, 1, 1, DeviceType>, 1, 1>(instr, locals, device);
    break;
case 89:
    batch_foreach<tensor_foreach_dynamic<kernel_elu<CpuTypes>, kernel_elu<SimdTypes>, 1, 1, DeviceType>, 1, 1>(instr, locals, device);
    break;
case 90:
    batch_foreach<tensor_foreach_dynamic<kernel_gelu<CpuTypes>, kernel_gelu<SimdTypes>, 1, 1, DeviceType>, 1, 1>(instr, locals, device);
    break;
case 91:
    batch_foreach<tensor_foreach_dynamic<kernel_sigmoid<CpuTypes>, kernel_sigmoid<SimdTypes>, 1, 1, DeviceType>, 1, 1>(instr, locals, device);
    break;
case 92:
    batch_foreach<tensor_foreach_dynamic<kernel_softplus<CpuTypes>, kernel_softplus<SimdTypes>, 1, 1, DeviceType>, 1, 1>(instr, locals, device);
    break;
case 93:
    op_rqs_reshape(instr, locals, device);
    break;
case 94:
    batch_foreach<tensor_foreach<kernel_rqs_find_bin<CpuTypes>, kernel_rqs_find_bin<SimdTypes>, 4, 1, 2, DeviceType>, 4, 1>(instr, locals, device);
    break;
case 95:
    batch_foreach<tensor_foreach<kernel_rqs_forward<CpuTypes>, kernel_rqs_forward<SimdTypes>, 2, 2, 2, DeviceType>, 2, 2>(instr, locals, device);
    break;
case 96:
    batch_foreach<tensor_foreach<kernel_rqs_inverse<CpuTypes>, kernel_rqs_inverse<SimdTypes>, 2, 2, 2, DeviceType>, 2, 2>(instr, locals, device);
    break;
case 97:
    batch_foreach<tensor_foreach_dynamic<kernel_softmax<CpuTypes>, kernel_softmax<SimdTypes>, 1, 1, DeviceType>, 1, 1>(instr, locals, device);
    break;
case 98:
    batch_foreach<tensor_foreach<kernel_softmax_prior<CpuTypes>, kernel_softmax_prior<SimdTypes>, 2, 1, 1, DeviceType>, 2, 1>(instr, locals, device);
    break;
case 99:
    batch_foreach<tensor_foreach<kernel_sample_discrete<CpuTypes>, kernel_sample_discrete<SimdTypes>, 2, 2, 1, DeviceType>, 2, 2>(instr, locals, device);
    break;
case 100:
    batch_foreach<tensor_foreach<kernel_sample_discrete_inverse<CpuTypes>, kernel_sample_discrete_inverse<SimdTypes>, 2, 2, 1, DeviceType>, 2, 2>(instr, locals, device);
    break;
case 101:
    batch_foreach<tensor_foreach<kernel_sample_discrete_probs<CpuTypes>, kernel_sample_discrete_probs<SimdTypes>, 2, 2, 1, DeviceType>, 2, 2>(instr, locals, device);
    break;
case 102:
    batch_foreach<tensor_foreach<kernel_sample_discrete_probs_inverse<CpuTypes>, kernel_sample_discrete_probs_inverse<SimdTypes>, 2, 2, 1, DeviceType>, 2, 2>(instr, locals, device);
    break;
case 103:
    op_discrete_histogram(instr, locals, device);
    break;
case 104:
    batch_foreach<tensor_foreach<kernel_permute_momenta<CpuTypes>, kernel_permute_momenta<SimdTypes>, 3, 1, 1, DeviceType>, 3, 1>(instr, locals, device);
    break;
case 105:
    batch_foreach<tensor_foreach<kernel_gather<CpuTypes>, kernel_gather<SimdTypes>, 2, 1, 1, DeviceType>, 2, 1>(instr, locals, device);
    break;
case 106:
    batch_foreach<tensor_foreach<kernel_gather_int<CpuTypes>, kernel_gather_int<SimdTypes>, 2, 1, 1, DeviceType>, 2, 1>(instr, locals, device);
    break;
case 107:
    batch_foreach<tensor_foreach<kernel_select_int<CpuTypes>, kernel_select_int<SimdTypes>, 2, 1, 1, DeviceType>, 2, 1>(instr, locals, device);
    break;
case 108:
    batch_foreach<tensor_foreach<kernel_select<CpuTypes>, kernel_select<SimdTypes>, 2, 1, 1, DeviceType>, 2, 1>(instr, locals, device);
    break;
case 109:
    batch_foreach<tensor_foreach<kernel_select_vector<CpuTypes>, kernel_select_vector<CpuTypes>, 2, 1, 1, DeviceType>, 2, 1>(instr, locals, device);
    break;
case 110:
    batch_foreach<tensor_foreach<kernel_argsort<CpuTypes>, kernel_argsort<CpuTypes>, 1, 1, 1, DeviceType>, 1, 1>(instr, locals, device);
    break;
case 111:
    batch_foreach<tensor_foreach<kernel_one_hot<CpuTypes>, kernel_one_hot<SimdTypes>, 2, 1, 1, DeviceType>, 2, 1>(instr, locals, device);
    break;
case 112:
    op_nonzero(instr, locals, device);
    break;
case 113:
    op_batch_gather(instr, locals, device);
    break;
case 114:
    op_batch_scatter(instr, locals, device);
    break;
case 115:
    op_random(instr, locals, device);
    break;
case 116:
    op_unweight(instr, locals, device);
    break;
case 117:
    batch_foreach<tensor_foreach<kernel_vegas_forward<CpuTypes>, kernel_vegas_forward<SimdTypes>, 2, 2, 2, DeviceType>, 2, 2>(instr, locals, device);
    break;
case 118:
    batch_foreach<tensor_foreach<kernel_vegas_inverse<CpuTypes>, kernel_vegas_inverse<SimdTypes>, 2, 2, 2, DeviceType>, 2, 2>(instr, locals, device);
    break;
case 119:
    op_vegas_histogram(instr, locals, device);
    break;
case 120:
    op_histogram(instr, locals, device);
    break;
